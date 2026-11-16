//! CUDA backend (NVIDIA). The driver API and NVRTC are loaded at runtime (libcuda / nvcuda.dll, libnvrtc /
//! nvrtc64_*.dll), so apollon builds without CUDA and uses it where it is installed. The kernels are the
//! same source as for Metal, compiled by NVRTC for the GPU found. Every weight tensor is copied to the GPU
//! once, the first time it is used.
use super::{describe, dequant_threads, plan, Backend, DqP, GemmP, KERNELS};
use crate::tensor::Mat;
use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::sync::Mutex;

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuDevPtr = u64;
type NvrtcProgram = *mut c_void;

const ATTR_CC_MAJOR: c_int = 75;
const ATTR_CC_MINOR: c_int = 76;

struct Api {
    _cuda: Library,
    _nvrtc: Library,
    init: unsafe extern "C" fn(c_uint) -> CuResult,
    device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    device_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    device_attr: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    ctx_create: unsafe extern "C" fn(*mut CuContext, c_uint, CuDevice) -> CuResult,
    ctx_set: unsafe extern "C" fn(CuContext) -> CuResult,
    ctx_sync: unsafe extern "C" fn() -> CuResult,
    module_load: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    module_fn: unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut CuDevPtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(CuDevPtr) -> CuResult,
    htod: unsafe extern "C" fn(CuDevPtr, *const c_void, usize) -> CuResult,
    dtoh: unsafe extern "C" fn(*mut c_void, CuDevPtr, usize) -> CuResult,
    #[allow(clippy::type_complexity)]
    launch: unsafe extern "C" fn(CuFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> CuResult,
    nvrtc_create: unsafe extern "C" fn(*mut NvrtcProgram, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int,
    nvrtc_compile: unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> c_int,
    nvrtc_ptx_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int,
    nvrtc_ptx: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int,
    nvrtc_log_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int,
    nvrtc_log: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int,
    nvrtc_destroy: unsafe extern "C" fn(*mut NvrtcProgram) -> c_int,
}

fn open(names: &[&str]) -> Result<Library, String> {
    for n in names {
        if let Ok(l) = unsafe { Library::new(n) } {
            return Ok(l);
        }
    }
    Err(format!("none of {} found", names.join(", ")))
}

fn sym<T: Copy>(lib: &Library, name: &str) -> Result<T, String> {
    unsafe {
        let s: Symbol<T> = lib.get(name.as_bytes()).map_err(|e| format!("{name}: {e}"))?;
        Ok(*s)
    }
}

fn nvrtc_names() -> Vec<String> {
    let mut v = vec!["libnvrtc.so".to_string()];
    for major in (11..=13).rev() {
        v.push(format!("libnvrtc.so.{major}"));
        for minor in (0..=9).rev() {
            v.push(format!("libnvrtc.so.{major}.{minor}"));
            v.push(format!("nvrtc64_{major}{minor}_0.dll"));
        }
        v.push(format!("nvrtc64_{major}0_0.dll"));
    }
    v
}

impl Api {
    fn load() -> Result<Self, String> {
        let cuda = open(&["libcuda.so.1", "libcuda.so", "nvcuda.dll"]).map_err(|e| format!("CUDA driver: {e}"))?;
        let names = nvrtc_names();
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        let nvrtc = open(&refs).map_err(|e| format!("NVRTC: {e}"))?;
        Ok(Api {
            init: sym(&cuda, "cuInit")?,
            device_get: sym(&cuda, "cuDeviceGet")?,
            device_name: sym(&cuda, "cuDeviceGetName")?,
            device_attr: sym(&cuda, "cuDeviceGetAttribute")?,
            ctx_create: sym(&cuda, "cuCtxCreate_v2")?,
            ctx_set: sym(&cuda, "cuCtxSetCurrent")?,
            ctx_sync: sym(&cuda, "cuCtxSynchronize")?,
            module_load: sym(&cuda, "cuModuleLoadData")?,
            module_fn: sym(&cuda, "cuModuleGetFunction")?,
            mem_alloc: sym(&cuda, "cuMemAlloc_v2")?,
            mem_free: sym(&cuda, "cuMemFree_v2")?,
            htod: sym(&cuda, "cuMemcpyHtoD_v2")?,
            dtoh: sym(&cuda, "cuMemcpyDtoH_v2")?,
            launch: sym(&cuda, "cuLaunchKernel")?,
            nvrtc_create: sym(&nvrtc, "nvrtcCreateProgram")?,
            nvrtc_compile: sym(&nvrtc, "nvrtcCompileProgram")?,
            nvrtc_ptx_size: sym(&nvrtc, "nvrtcGetPTXSize")?,
            nvrtc_ptx: sym(&nvrtc, "nvrtcGetPTX")?,
            nvrtc_log_size: sym(&nvrtc, "nvrtcGetProgramLogSize")?,
            nvrtc_log: sym(&nvrtc, "nvrtcGetProgramLog")?,
            nvrtc_destroy: sym(&nvrtc, "nvrtcDestroyProgram")?,
            _cuda: cuda,
            _nvrtc: nvrtc,
        })
    }
}

fn ck(r: CuResult, what: &str) -> Result<(), String> {
    if r == 0 { Ok(()) } else { Err(format!("CUDA {what}: error {r}")) }
}

pub struct Cuda {
    api: Api,
    ctx: CuContext,
    dequant: CuFunction,
    gemm: CuFunction,
    name: String,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    /// host slice (start, length) -> device copy
    tensors: HashMap<(usize, usize), CuDevPtr>,
    x: (CuDevPtr, usize),
    tmp: (CuDevPtr, usize),
    out: (CuDevPtr, usize),
}

// the context is made current on every call; all GPU work happens under the mutex
unsafe impl Send for Cuda {}
unsafe impl Sync for Cuda {}

impl Cuda {
    pub fn new() -> Result<Self, String> {
        let api = Api::load()?;
        unsafe {
            ck((api.init)(0), "cuInit")?;
            let mut dev: CuDevice = 0;
            ck((api.device_get)(&mut dev, 0), "cuDeviceGet")?;
            let mut name = [0 as c_char; 256];
            ck((api.device_name)(name.as_mut_ptr(), 256, dev), "cuDeviceGetName")?;
            let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().to_string();
            let (mut major, mut minor) = (0, 0);
            ck((api.device_attr)(&mut major, ATTR_CC_MAJOR, dev), "compute capability")?;
            ck((api.device_attr)(&mut minor, ATTR_CC_MINOR, dev), "compute capability")?;
            let mut ctx: CuContext = std::ptr::null_mut();
            ck((api.ctx_create)(&mut ctx, 0, dev), "cuCtxCreate")?;
            // compile the kernels for this GPU
            let src = CString::new(KERNELS).unwrap();
            let pname = CString::new("apollon_kernels.cu").unwrap();
            let mut prog: NvrtcProgram = std::ptr::null_mut();
            if (api.nvrtc_create)(&mut prog, src.as_ptr(), pname.as_ptr(), 0, std::ptr::null(), std::ptr::null()) != 0 {
                return Err("nvrtcCreateProgram failed".into());
            }
            let arch = CString::new(format!("--gpu-architecture=compute_{major}{minor}")).unwrap();
            let std = CString::new("--std=c++14").unwrap();
            let opts = [arch.as_ptr(), std.as_ptr()];
            let res = (api.nvrtc_compile)(prog, opts.len() as c_int, opts.as_ptr());
            if res != 0 {
                let mut n = 0usize;
                (api.nvrtc_log_size)(prog, &mut n);
                let mut log = vec![0u8; n.max(1)];
                (api.nvrtc_log)(prog, log.as_mut_ptr() as *mut c_char);
                (api.nvrtc_destroy)(&mut prog);
                return Err(format!("NVRTC: {}", String::from_utf8_lossy(&log)));
            }
            let mut n = 0usize;
            (api.nvrtc_ptx_size)(prog, &mut n);
            let mut ptx = vec![0u8; n];
            (api.nvrtc_ptx)(prog, ptx.as_mut_ptr() as *mut c_char);
            (api.nvrtc_destroy)(&mut prog);
            let mut module: CuModule = std::ptr::null_mut();
            ck((api.module_load)(&mut module, ptx.as_ptr() as *const c_void), "cuModuleLoadData")?;
            let f = |n: &str| -> Result<CuFunction, String> {
                let c = CString::new(n).unwrap();
                let mut f: CuFunction = std::ptr::null_mut();
                ck((api.module_fn)(&mut f, module, c.as_ptr()), n)?;
                Ok(f)
            };
            let dequant = f("dequant")?;
            let gemm = f("gemm_nt")?;
            Ok(Cuda { api, ctx, dequant, gemm, name: format!("{name} (CUDA, compute {major}.{minor})"), inner: Mutex::new(Inner::default()) })
        }
    }

    /// device copy of a host slice (made once)
    fn tensor(&self, inner: &mut Inner, s: &[u8]) -> Result<CuDevPtr, String> {
        let key = (s.as_ptr() as usize, s.len());
        if let Some(&p) = inner.tensors.get(&key) {
            return Ok(p);
        }
        let mut p: CuDevPtr = 0;
        unsafe {
            ck((self.api.mem_alloc)(&mut p, s.len().max(4)), "cuMemAlloc (weights)")?;
            ck((self.api.htod)(p, s.as_ptr() as *const c_void, s.len()), "cuMemcpyHtoD (weights)")?;
        }
        inner.tensors.insert(key, p);
        Ok(p)
    }

    fn scratch(&self, slot: &mut (CuDevPtr, usize), bytes: usize) -> Result<CuDevPtr, String> {
        if slot.1 < bytes {
            unsafe {
                if slot.0 != 0 {
                    (self.api.mem_free)(slot.0);
                }
                let n = bytes.next_power_of_two();
                let mut p: CuDevPtr = 0;
                ck((self.api.mem_alloc)(&mut p, n), "cuMemAlloc (scratch)")?;
                *slot = (p, n);
            }
        }
        Ok(slot.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(&self, m: &Mat, x: Option<&[f32]>, blocks: &[(usize, usize, Vec<(GemmP, u32, u32)>)], out_len: usize, into_out: bool) -> Result<Vec<f32>, String> {
        let mut inner = self.inner.lock().map_err(|_| "CUDA state poisoned")?;
        unsafe { ck((self.api.ctx_set)(self.ctx), "cuCtxSetCurrent")? };
        let d = describe(m);
        let w = self.tensor(&mut inner, d.w)?;
        let s = match d.s {
            Some(s) => self.tensor(&mut inner, s)?,
            None => w,
        };
        let b = match d.b {
            Some(s) => self.tensor(&mut inner, s)?,
            None => w,
        };
        let max_nb = blocks.iter().map(|b| b.1).max().unwrap_or(0);
        let mut slot = inner.tmp;
        let tmp = self.scratch(&mut slot, max_nb * m.cols * 4)?;
        inner.tmp = slot;
        let mut slot = inner.out;
        let out = self.scratch(&mut slot, out_len.max(1) * 4)?;
        inner.out = slot;
        let xd = match x {
            Some(x) => {
                let mut slot = inner.x;
                let p = self.scratch(&mut slot, x.len() * 4)?;
                inner.x = slot;
                unsafe { ck((self.api.htod)(p, x.as_ptr() as *const c_void, x.len() * 4), "cuMemcpyHtoD (input)")? };
                Some(p)
            }
            None => None,
        };
        for (row0, nb, gemms) in blocks {
            let mut p = DqP { row0: *row0 as u32, nrows: *nb as u32, ..d.p };
            let target = if into_out { out } else { tmp };
            let (mut w_, mut s_, mut b_, mut t_) = (w, s, b, target);
            let mut args: [*mut c_void; 5] = [
                &mut w_ as *mut _ as *mut c_void,
                &mut s_ as *mut _ as *mut c_void,
                &mut b_ as *mut _ as *mut c_void,
                &mut t_ as *mut _ as *mut c_void,
                &mut p as *mut _ as *mut c_void,
            ];
            let n = dequant_threads(&p);
            unsafe {
                ck((self.api.launch)(self.dequant, n.div_ceil(256) as c_uint, 1, 1, 256, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut()), "dequant")?;
            }
            if let Some(xd) = xd {
                for (g, gx, gy) in gemms {
                    let (mut a_, mut b2, mut c_, mut g_) = (xd, tmp, out, *g);
                    let mut args: [*mut c_void; 4] = [
                        &mut a_ as *mut _ as *mut c_void,
                        &mut b2 as *mut _ as *mut c_void,
                        &mut c_ as *mut _ as *mut c_void,
                        &mut g_ as *mut _ as *mut c_void,
                    ];
                    unsafe {
                        ck((self.api.launch)(self.gemm, *gx, *gy, 1, 16, 16, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut()), "gemm")?;
                    }
                }
            }
        }
        let mut v = vec![0f32; out_len];
        unsafe {
            ck((self.api.ctx_sync)(), "cuCtxSynchronize")?;
            ck((self.api.dtoh)(v.as_mut_ptr() as *mut c_void, out, out_len * 4), "cuMemcpyDtoH")?;
        }
        Ok(v)
    }
}

impl Backend for Cuda {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn matmul(&self, m: &Mat, x: &[f32], t: usize, parts: usize, r0: usize, nr: usize) -> Option<Vec<f32>> {
        if t == 0 || nr == 0 {
            return Some(Vec::new());
        }
        let blocks = plan(m, t, parts, r0, nr);
        match self.run(m, Some(x), &blocks, t * parts * nr, false) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("CUDA: {e} – this product runs on the CPU");
                None
            }
        }
    }

    fn dequant(&self, m: &Mat, r0: usize, nr: usize) -> Option<Vec<f32>> {
        self.run(m, None, &[(r0, nr, Vec::new())], nr * m.cols, true).ok()
    }
}
