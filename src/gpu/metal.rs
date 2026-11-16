//! Metal backend (macOS). The mapped model files become GPU buffers without a copy (unified memory);
//! inputs, unpacked rows and results live in shared scratch buffers that grow as needed.
use super::{describe, dequant_threads, locate, plan, Backend, DqP, GemmP, KERNELS};
use crate::tensor::Mat;
use metal::objc::runtime::{Object, NO, YES};
use metal::objc::{class, msg_send, sel, sel_impl};
use metal::{Buffer, BufferRef, CommandBufferRef, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::sync::Mutex;

#[link(name = "MetalPerformanceShaders", kind = "framework")]
extern "C" {}

/// MPSDataTypeFloat32
const MPS_F32: u32 = 0x1000_0020;

pub struct MetalBackend {
    device: Device,
    queue: metal::CommandQueue,
    dequant: ComputePipelineState,
    gemm: ComputePipelineState,
    inner: Mutex<Inner>,
}

/// an MPSMatrixMultiplication kernel (retained)
struct Mm(*mut Object);

impl Drop for Mm {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.0, release];
        }
    }
}

#[derive(Default)]
struct Inner {
    /// MPSMatrixMultiplication per (rows, columns, interior)
    mps: HashMap<(usize, usize, usize), Mm>,
    /// page-aligned window (start, end) of a mapped file -> buffer over it (no copy); other slices
    /// (start, length) -> a copy
    regions: HashMap<(usize, usize), Buffer>,
    x: Option<Buffer>,
    tmp: Option<Buffer>,
    out: Option<Buffer>,
}

// the Metal objects are reference-counted Objective-C objects, safe to share; all encoding happens
// under the mutex
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

fn page() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

impl MetalBackend {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device")?;
        let opts = CompileOptions::new();
        opts.set_fast_math_enabled(false);
        let lib = device.new_library_with_source(KERNELS, &opts).map_err(|e| format!("Metal kernels: {e}"))?;
        let f = |n: &str| -> Result<ComputePipelineState, String> {
            let func = lib.get_function(n, None).map_err(|e| format!("Metal kernel {n}: {e}"))?;
            device.new_compute_pipeline_state_with_function(&func).map_err(|e| format!("Metal pipeline {n}: {e}"))
        };
        let dequant = f("dequant")?;
        let gemm = f("gemm_sg")?;
        let queue = device.new_command_queue();
        Ok(MetalBackend { device, queue, dequant, gemm, inner: Mutex::new(Inner::default()) })
    }

    /// A buffer holding a slice, and the slice's offset in it. Slices of a mapped file get a buffer over
    /// the pages they lie in, without a copy (one per tensor: a whole file can exceed the largest buffer
    /// Metal allows); anything else is copied.
    fn region(&self, inner: &mut Inner, s: &[u8]) -> Option<(Buffer, u64)> {
        let start = s.as_ptr() as usize;
        if locate(s).is_some() {
            let pg = page();
            let (a, e) = (start / pg * pg, (start + s.len()).div_ceil(pg) * pg);
            if (e - a) as u64 > self.device.max_buffer_length() {
                return None;
            }
            let b = inner.regions.entry((a, e)).or_insert_with(|| {
                self.device.new_buffer_with_bytes_no_copy(a as *const std::ffi::c_void, (e - a) as u64, MTLResourceOptions::StorageModeShared, None)
            });
            Some((b.clone(), (start - a) as u64))
        } else {
            let b = inner
                .regions
                .entry((start, usize::MAX - s.len()))
                .or_insert_with(|| self.device.new_buffer_with_data(s.as_ptr() as *const std::ffi::c_void, s.len().max(4) as u64, MTLResourceOptions::StorageModeShared));
            Some((b.clone(), 0))
        }
    }

    fn scratch(&self, slot: &mut Option<Buffer>, bytes: usize) -> Buffer {
        let bytes = bytes.max(4) as u64;
        if slot.as_ref().is_none_or(|b| b.length() < bytes) {
            *slot = Some(self.device.new_buffer(bytes.next_power_of_two(), MTLResourceOptions::StorageModeShared));
        }
        slot.as_ref().unwrap().clone()
    }

    /// C = A·Bᵀ with Apple's tuned GEMM (MPSMatrixMultiplication), encoded into `cb`
    #[allow(clippy::too_many_arguments)]
    fn mps_gemm(&self, inner: &mut Inner, cb: &CommandBufferRef, a: &BufferRef, b: &BufferRef, c: &BufferRef, g: &GemmP) {
        unsafe {
            let key = (g.m as usize, g.n as usize, g.k as usize);
            let device: &metal::DeviceRef = &self.device;
            let mm = inner.mps.entry(key).or_insert_with(|| {
                let o: *mut Object = msg_send![class!(MPSMatrixMultiplication), alloc];
                let o: *mut Object = msg_send![o, initWithDevice: device transposeLeft: NO transposeRight: YES resultRows: g.m as u64 resultColumns: g.n as u64 interiorColumns: g.k as u64 alpha: 1.0f64 beta: 0.0f64];
                Mm(o)
            });
            let mat = |buf: &BufferRef, off_floats: u64, rows: u32, cols: u32, ld: u32| -> *mut Object {
                let d: *mut Object = msg_send![class!(MPSMatrixDescriptor), matrixDescriptorWithRows: rows as u64 columns: cols as u64 rowBytes: (ld as u64) * 4 dataType: MPS_F32];
                let m: *mut Object = msg_send![class!(MPSMatrix), alloc];
                msg_send![m, initWithBuffer: buf offset: off_floats * 4 descriptor: d]
            };
            let am = mat(a, g.a_off, g.m, g.k, g.lda);
            let bm = mat(b, g.b_off, g.n, g.k, g.ldb);
            let cm = mat(c, g.c_off, g.m, g.n, g.ldc);
            let _: () = msg_send![mm.0, encodeToCommandBuffer: cb leftMatrix: am rightMatrix: bm resultMatrix: cm];
            for o in [am, bm, cm] {
                let _: () = msg_send![o, release];
            }
        }
    }

    /// Unpack rows `row0..row0+nrows` into `tmp` and run the given GEMMs on it (one command buffer)
    #[allow(clippy::too_many_arguments)]
    fn run(&self, inner: &mut Inner, m: &Mat, x: Option<&[f32]>, blocks: &[(usize, usize, Vec<(GemmP, u32, u32)>)], out_len: usize, keep_tmp: bool) -> Option<Vec<f32>> {
        let d = describe(m);
        let (wb, wo) = self.region(inner, d.w)?;
        let (sb, so) = match d.s {
            Some(s) => self.region(inner, s)?,
            None => (wb.clone(), 0),
        };
        let (bb, bo) = match d.b {
            Some(s) => self.region(inner, s)?,
            None => (wb.clone(), 0),
        };
        let max_nb = blocks.iter().map(|b| b.1).max().unwrap_or(0);
        let mut slot = inner.tmp.take();
        let tmp = self.scratch(&mut slot, max_nb * m.cols * 4);
        inner.tmp = slot;
        let xbuf = x.map(|x| {
            let mut slot = inner.x.take();
            let b = self.scratch(&mut slot, x.len() * 4);
            inner.x = slot;
            unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), b.contents() as *mut f32, x.len()) };
            b
        });
        let mut slot = inner.out.take();
        let out = self.scratch(&mut slot, out_len * 4);
        inner.out = slot;
        let cb = self.queue.new_command_buffer();
        let mps = std::env::var_os("APOLLON_METAL_OWN_GEMM").is_none();
        let mut enc = cb.new_compute_command_encoder();
        for (row0, nb, gemms) in blocks {
            let p = DqP { off_w: wo, off_s: so, off_b: bo, row0: *row0 as u32, nrows: *nb as u32, ..d.p };
            enc.set_compute_pipeline_state(&self.dequant);
            enc.set_buffer(0, Some(&wb), 0);
            enc.set_buffer(1, Some(&sb), 0);
            enc.set_buffer(2, Some(&bb), 0);
            enc.set_buffer(3, Some(if keep_tmp { &out } else { &tmp }), 0);
            enc.set_bytes(4, std::mem::size_of::<DqP>() as u64, &p as *const DqP as *const std::ffi::c_void);
            let n = dequant_threads(&p);
            enc.dispatch_thread_groups(MTLSize::new(n.div_ceil(256), 1, 1), MTLSize::new(256, 1, 1));
            if let (Some(xb), true) = (&xbuf, mps) {
                enc.end_encoding();
                for (g, _, _) in gemms {
                    self.mps_gemm(inner, cb, xb, &tmp, &out, g);
                }
                enc = cb.new_compute_command_encoder();
            } else if let Some(xb) = &xbuf {
                enc.set_compute_pipeline_state(&self.gemm);
                for (g, gx, gy) in gemms {
                    enc.set_buffer(0, Some(xb), 0);
                    enc.set_buffer(1, Some(&tmp), 0);
                    enc.set_buffer(2, Some(&out), 0);
                    enc.set_bytes(3, std::mem::size_of::<GemmP>() as u64, g as *const GemmP as *const std::ffi::c_void);
                    enc.dispatch_thread_groups(MTLSize::new(*gx as u64, *gy as u64, 1), MTLSize::new(128, 1, 1));
                }
            }
        }
        enc.end_encoding();
        let tw = std::time::Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let gpu_ms = tw.elapsed().as_secs_f32() * 1e3;
        if cb.status() != metal::MTLCommandBufferStatus::Completed {
            return None;
        }
        let tc = std::time::Instant::now();
        let mut v: Vec<f32> = Vec::with_capacity(out_len);
        unsafe {
            std::ptr::copy_nonoverlapping(out.contents() as *const f32, v.as_mut_ptr(), out_len);
            v.set_len(out_len);
        }
        if std::env::var_os("APOLLON_GPU_TIMING").is_some() {
            eprintln!("metal: {} rows × {}×{}: GPU {:.2} ms, copy out {:.2} ms", x.map_or(0, |x| x.len() / m.cols), m.rows, m.cols, gpu_ms, tc.elapsed().as_secs_f32() * 1e3);
        }
        Some(v)
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> String {
        format!("{} (Metal)", self.device.name())
    }

    fn matmul(&self, m: &Mat, x: &[f32], t: usize, parts: usize, r0: usize, nr: usize) -> Option<Vec<f32>> {
        if t == 0 || nr == 0 {
            return Some(Vec::new());
        }
        let blocks = plan(m, t, parts, r0, nr);
        let mut inner = self.inner.lock().ok()?;
        // command buffers and encoders are autoreleased: drain them after every product
        metal::objc::rc::autoreleasepool(|| self.run(&mut inner, m, Some(x), &blocks, t * parts * nr, false))
    }

    fn dequant(&self, m: &Mat, r0: usize, nr: usize) -> Option<Vec<f32>> {
        let blocks = vec![(r0, nr, Vec::new())];
        let mut inner = self.inner.lock().ok()?;
        metal::objc::rc::autoreleasepool(|| self.run(&mut inner, m, None, &blocks, nr * m.cols, true))
    }
}
