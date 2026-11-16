//! GPU backends for the matrix products: Metal (macOS) and CUDA (NVIDIA, loaded at runtime – apollon
//! builds and runs without CUDA installed). The weights stay quantized as in the file; a kernel unpacks
//! a block of rows exactly as the CPU does (src/kernels.h ↔ quant.rs, tensor.rs), a second kernel
//! multiplies. Everything else of the forward pass stays on the CPU.
//!
//! Every output of the GPU product is summed over k in the same fixed order, whatever the batch size:
//! a row gives bit for bit the same result alone or among other rows. Against the CPU (Accelerate /
//! plain Rust), the summation order differs, so results agree to float rounding (≈ 1e-6 relative).
use crate::quant::DType;
use crate::tensor::{Mat, View};
use std::sync::{Mutex, OnceLock};

#[cfg(target_os = "macos")]
mod metal;
mod cuda;

/// The kernels, one source for both backends
pub const KERNELS: &str = include_str!("kernels.h");

// formats, as in kernels.h
pub const FMT_MLX: u32 = 100;

pub fn fmt_id(dt: DType) -> u32 {
    match dt {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::Q4_0 => 3,
        DType::Q4_1 => 4,
        DType::Q5_0 => 5,
        DType::Q5_1 => 6,
        DType::Q8_0 => 7,
        DType::Q2K => 8,
        DType::Q3K => 9,
        DType::Q4K => 10,
        DType::Q5K => 11,
        DType::Q6K => 12,
        DType::IQ4NL => 13,
        DType::IQ4XS => 14,
        DType::MXFP4 => 15,
    }
}

fn dt_id(dt: DType) -> u32 {
    match dt {
        DType::F32 => 0,
        DType::F16 => 1,
        _ => 2,
    }
}

/// Parameters of the dequant kernel (layout = struct DqP in kernels.h)
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct DqP {
    pub off_w: u64,
    pub off_s: u64,
    pub off_b: u64,
    pub fmt: u32,
    pub cols: u32,
    pub row_bytes: u32,
    pub row0: u32,
    pub nrows: u32,
    pub chunk: u32,
    pub blk_bytes: u32,
    pub wpr: u32,
    pub ng: u32,
    pub bits: u32,
    pub gs: u32,
    pub sdt: u32,
    pub bdt: u32,
    pub has_b: u32,
}

/// Parameters of the GEMM kernel (layout = struct GemmP in kernels.h); offsets in floats
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct GemmP {
    pub a_off: u64,
    pub b_off: u64,
    pub c_off: u64,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
}

/// The up to three byte ranges a matrix lives in (weights, MLX scales, MLX biases)
pub struct Desc {
    pub p: DqP,
    pub w: &'static [u8],
    pub s: Option<&'static [u8]>,
    pub b: Option<&'static [u8]>,
}

/// What the kernels need to unpack rows of `m` (byte offsets are relative to each slice: the backend adds
/// the slice's position in its device buffer)
pub fn describe(m: &Mat) -> Desc {
    let mut p = DqP { cols: m.cols as u32, ..Default::default() };
    match m.view() {
        View::Plain { dt, data, row_bytes } => {
            let (be, bb) = dt.block();
            p.fmt = fmt_id(dt);
            p.row_bytes = row_bytes as u32;
            p.chunk = if be == 1 { 32 } else { be as u32 };
            p.blk_bytes = bb as u32;
            Desc { p, w: data, s: None, b: None }
        }
        View::Mlx { packed, wpr, scales, biases, ng, bits, gs } => {
            p.fmt = FMT_MLX;
            p.wpr = wpr as u32;
            p.ng = ng as u32;
            p.bits = bits as u32;
            p.gs = gs as u32;
            p.chunk = gs as u32;
            p.sdt = dt_id(scales.0);
            p.bdt = biases.map_or(0, |b| dt_id(b.0));
            p.has_b = biases.is_some() as u32;
            Desc { p, w: packed, s: Some(scales.1), b: biases.map(|b| b.1) }
        }
    }
}

/// Threads of one dequant dispatch (rows × chunks per row)
pub fn dequant_threads(p: &DqP) -> u64 {
    let per_row = if p.fmt == FMT_MLX { p.ng } else { p.cols.div_ceil(p.chunk) };
    p.nrows as u64 * per_row as u64
}

pub trait Backend: Send + Sync {
    fn name(&self) -> String;
    /// x [t][cols] times rows r0..r0+nr of m, split into `parts` column ranges like Mat::matmul_split:
    /// result [t][parts][nr]. None: this matrix cannot be done here (the CPU takes over).
    fn matmul(&self, m: &Mat, x: &[f32], t: usize, parts: usize, r0: usize, nr: usize) -> Option<Vec<f32>>;
    /// rows r0..r0+nr unpacked [nr][cols] (for the self-test)
    fn dequant(&self, m: &Mat, r0: usize, nr: usize) -> Option<Vec<f32>>;
}

/// Rows unpacked per dispatch: at most 64 MB of floats
pub fn row_block(cols: usize, nr: usize) -> usize {
    ((16usize << 20) / cols.max(1)).clamp(64, 4096).min(nr.max(1))
}

/// The GEMM dispatches of one block of rows: (params, groups in x, groups in y)
pub fn gemm_plan(t: usize, cols: usize, parts: usize, nr: usize, blk: usize, nb: usize) -> Vec<(GemmP, u32, u32)> {
    let seg = cols / parts;
    (0..parts)
        .map(|p| {
            let g = GemmP {
                a_off: (p * seg) as u64,
                b_off: (p * seg) as u64,
                c_off: (p * nr + blk) as u64,
                m: t as u32,
                n: nb as u32,
                k: seg as u32,
                lda: cols as u32,
                ldb: cols as u32,
                ldc: (parts * nr) as u32,
            };
            (g, nb.div_ceil(64) as u32, t.div_ceil(64) as u32)
        })
        .collect()
}

// ---------------------------------------------------------------------- mapped files

/// Memory-mapped files (start, length): the weights live in them
static REGIONS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

pub fn register_region(data: &'static [u8]) {
    REGIONS.lock().unwrap().push((data.as_ptr() as usize, data.len()));
}

/// The registered region a slice lies in: (start, length, offset of the slice)
pub fn locate(s: &[u8]) -> Option<(usize, usize, usize)> {
    let p = s.as_ptr() as usize;
    REGIONS.lock().unwrap().iter().find(|(a, l)| p >= *a && p + s.len() <= a + l).map(|&(a, l)| (a, l, p - a))
}

// ---------------------------------------------------------------------- choosing the backend

static BACKEND: OnceLock<Option<Box<dyn Backend>>> = OnceLock::new();

/// The active backend, if any
pub fn backend() -> Option<&'static dyn Backend> {
    BACKEND.get().and_then(|b| b.as_deref())
}

/// Choose the backend: "auto" (Metal on macOS, else CUDA if a driver is found, else CPU), "metal",
/// "cuda", "off". Returns a one-line description.
pub fn init(choice: &str) -> Result<String, String> {
    let b: Option<Box<dyn Backend>> = match choice {
        "off" | "cpu" => None,
        "metal" => Some(metal_backend()?),
        "cuda" => Some(Box::new(cuda::Cuda::new()?)),
        "auto" => match metal_backend() {
            Ok(b) => Some(b),
            Err(_) => cuda::Cuda::new().ok().map(|c| Box::new(c) as Box<dyn Backend>),
        },
        o => return Err(format!("--gpu {o}: expected auto, metal, cuda or off")),
    };
    let name = b.as_ref().map_or("CPU".to_string(), |b| b.name());
    let _ = BACKEND.set(b);
    Ok(name)
}

#[cfg(target_os = "macos")]
fn metal_backend() -> Result<Box<dyn Backend>, String> {
    Ok(Box::new(metal::MetalBackend::new()?))
}

#[cfg(not(target_os = "macos"))]
fn metal_backend() -> Result<Box<dyn Backend>, String> {
    Err("Metal is only available on macOS".into())
}

// ---------------------------------------------------------------------- self-test

/// GPU against CPU: every weight format with random blocks (unpacked values), the products of every
/// matrix given (random inputs, a few batch sizes). Returns a report and whether everything passed.
pub fn selftest(mats: &[(String, &Mat)]) -> (String, bool) {
    let Some(g) = backend() else { return ("no GPU backend active".into(), false) };
    let mut out = format!("GPU self-test on {}\n", g.name());
    let mut ok = true;
    // 1. every format with random blocks: unpacked values against the CPU
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let formats = [
        DType::F32, DType::F16, DType::BF16, DType::Q4_0, DType::Q4_1, DType::Q5_0, DType::Q5_1, DType::Q8_0, DType::Q2K, DType::Q3K, DType::Q4K, DType::Q5K,
        DType::Q6K, DType::IQ4NL, DType::IQ4XS, DType::MXFP4,
    ];
    for dt in formats {
        let (be, bb) = dt.block();
        let (rows, cols) = (37usize, 512usize);
        let row_bytes = cols / be * bb;
        let mut data = vec![0u8; rows * row_bytes];
        for v in data.iter_mut() {
            *v = rnd() as u8;
        }
        sanitize(dt, &mut data, &mut rnd);
        let data: &'static [u8] = Box::leak(data.into_boxed_slice());
        register_region(data);
        let m = Mat::new(rows, cols, dt, data, "selftest").unwrap();
        let (dev, prod) = compare(g, &m, &mut rnd);
        let pass = dev == 0.0 && prod < 1e-5;
        ok &= pass;
        out += &format!("  {:<8} unpacked: max |Δ| {:.1e}   product: max rel. dev. {:.1e}   {}\n", dt.name(), dev, prod, if pass { "OK" } else { "FAILED" });
    }
    // MLX affine, every bit width, group sizes, scale types, with and without biases
    for (bits, gs, sdt, with_b) in [(4, 64, DType::F16, true), (4, 32, DType::BF16, true), (8, 64, DType::F16, true), (2, 128, DType::F32, false), (3, 64, DType::F16, true), (5, 32, DType::BF16, false), (6, 64, DType::F16, true)] {
        let (rows, cols) = (29usize, 768usize);
        let ng = cols / gs;
        let wpr = cols * bits / 32;
        let mut packed = vec![0u8; rows * wpr * 4];
        for v in packed.iter_mut() {
            *v = rnd() as u8;
        }
        let ssz = if sdt == DType::F32 { 4 } else { 2 };
        let mut sc = vec![0u8; rows * ng * ssz];
        let mut bi = vec![0u8; rows * ng * ssz];
        for buf in [&mut sc, &mut bi] {
            for c in buf.chunks_mut(ssz) {
                let v = (rnd() % 2000) as f32 / 1000.0 - 1.0;
                match sdt {
                    DType::F32 => c.copy_from_slice(&v.to_le_bytes()),
                    DType::BF16 => c.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes()),
                    _ => c.copy_from_slice(&f32_to_f16(v).to_le_bytes()),
                }
            }
        }
        let (packed, sc, bi): (&'static [u8], &'static [u8], &'static [u8]) = (Box::leak(packed.into_boxed_slice()), Box::leak(sc.into_boxed_slice()), Box::leak(bi.into_boxed_slice()));
        for r in [packed, sc, bi] {
            register_region(r);
        }
        let m = Mat::mlx(rows, wpr, packed, (sdt, sc), with_b.then_some((sdt, bi)), ng, gs, "selftest").unwrap();
        let (dev, prod) = compare(g, &m, &mut rnd);
        let pass = dev == 0.0 && prod < 1e-5;
        ok &= pass;
        out += &format!("  MLX-Q{bits}/{gs} {:<4} unpacked: max |Δ| {:.1e}   product: max rel. dev. {:.1e}   {}\n", sdt.name(), dev, prod, if pass { "OK" } else { "FAILED" });
    }
    // 2. the matrices of a model (and the time of one product with 512 rows, GPU against CPU)
    for (name, m) in mats.iter().take(8) {
        if m.rows > 20000 {
            continue;
        }
        for t in [64usize, 2048] {
            let x: Vec<f32> = (0..t * m.cols).map(|_| (rnd() % 2001) as f32 / 1000.0 - 1.0).collect();
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                let _ = g.matmul(m, &x, t, 1, 0, m.rows);
            }
            out += &format!("  time, {t} rows × {name}: GPU {:.1} ms\n", t0.elapsed().as_secs_f32() / 3.0 * 1e3);
        }
        let t = 512;
        let x: Vec<f32> = (0..t * m.cols).map(|_| (rnd() % 2001) as f32 / 1000.0 - 1.0).collect();
        let best = |f: &mut dyn FnMut()| -> f32 {
            (0..3)
                .map(|_| {
                    let t0 = std::time::Instant::now();
                    f();
                    t0.elapsed().as_secs_f32()
                })
                .fold(f32::INFINITY, f32::min)
        };
        let tg = best(&mut || {
            let _ = g.matmul(m, &x, t, 1, 0, m.rows);
        });
        let td = best(&mut || {
            let _ = g.dequant(m, 0, m.rows);
        });
        let tc = best(&mut || {
            let _ = m.matmul_split_cpu(&x, t, 1);
        });
        out += &format!("  time, 512 rows × {name} ({}×{}): GPU {:.1} ms (unpacking alone {:.1} ms), CPU {:.1} ms\n", m.rows, m.cols, tg * 1e3, td * 1e3, tc * 1e3);
    }
    for (name, m) in mats {
        let (dev, prod) = compare(g, m, &mut rnd);
        let pass = dev == 0.0 && prod < 1e-5;
        ok &= pass;
        out += &format!("  {:<28} {:>6}×{:<6} {:<10} unpacked: max |Δ| {:.1e}   product: max rel. dev. {:.1e}   {}\n", name, m.rows, m.cols, m.type_name(), dev, prod, if pass { "OK" } else { "FAILED" });
    }
    (out, ok)
}

/// (max |Δ| of the unpacked rows, max relative deviation of the products over t = 1, 5, 70)
fn compare(g: &dyn Backend, m: &Mat, rnd: &mut impl FnMut() -> u64) -> (f32, f32) {
    let nr = m.rows.min(300);
    let mut dev = 0f32;
    if let Some(gpu) = g.dequant(m, 0, nr) {
        let mut row = vec![0f32; m.cols];
        for r in 0..nr {
            m.row(r, &mut row);
            for (a, b) in row.iter().zip(&gpu[r * m.cols..(r + 1) * m.cols]) {
                if !(a.is_nan() && b.is_nan()) && a.to_bits() != b.to_bits() {
                    dev = dev.max((a - b).abs().max(f32::MIN_POSITIVE));
                }
            }
        }
    } else {
        dev = f32::INFINITY;
    }
    let mut prod = 0f32;
    for t in [1usize, 5, 70] {
        let x: Vec<f32> = (0..t * m.cols).map(|_| (rnd() % 2001) as f32 / 1000.0 - 1.0).collect();
        // CPU reference in f64, rows 0..nr
        let mut row = vec![0f32; m.cols];
        let mut refv = vec![0f64; t * nr];
        let mut scale = 0f64;
        for r in 0..nr {
            m.row(r, &mut row);
            for k in 0..t {
                let mut s = 0f64;
                let mut a = 0f64;
                for c in 0..m.cols {
                    s += row[c] as f64 * x[k * m.cols + c] as f64;
                    a += (row[c] as f64 * x[k * m.cols + c] as f64).abs();
                }
                refv[k * nr + r] = s;
                scale = scale.max(a);
            }
        }
        match g.matmul(m, &x, t, 1, 0, nr) {
            Some(y) => {
                for (a, b) in y.iter().zip(&refv) {
                    let d = ((*a as f64 - b).abs() / scale.max(1e-30)) as f32;
                    if d.is_nan() {
                        if !(a.is_nan() || b.is_nan()) {
                            prod = f32::INFINITY;
                        }
                    } else {
                        prod = prod.max(d);
                    }
                }
            }
            None => prod = f32::INFINITY,
        }
    }
    (dev, prod)
}

/// Random bytes, but the f16/e8 scale fields of every block finite and moderate
fn sanitize(dt: DType, data: &mut [u8], rnd: &mut impl FnMut() -> u64) {
    let mut half = |b: &mut [u8], i: usize| {
        let v = (rnd() % 2000) as f32 / 1000.0 - 1.0;
        b[i..i + 2].copy_from_slice(&f32_to_f16(v).to_le_bytes());
    };
    let (_, bb) = dt.block();
    match dt {
        DType::F32 => {
            for c in data.chunks_mut(4) {
                let v = (rnd() % 2000) as f32 / 1000.0 - 1.0;
                c.copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::F16 => data.chunks_mut(2).for_each(|c| half(c, 0)),
        DType::BF16 => {
            for c in data.chunks_mut(2) {
                let v = (rnd() % 2000) as f32 / 1000.0 - 1.0;
                c.copy_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        DType::MXFP4 => data.chunks_mut(bb).for_each(|c| c[0] = 110 + (rnd() % 30) as u8),
        _ => {
            let fields: &[usize] = match dt {
                DType::Q4_0 | DType::Q5_0 | DType::Q8_0 | DType::IQ4NL | DType::IQ4XS => &[0],
                DType::Q4_1 | DType::Q5_1 | DType::Q4K | DType::Q5K => &[0, 2],
                DType::Q2K => &[80, 82],
                DType::Q3K => &[108],
                DType::Q6K => &[208],
                _ => &[],
            };
            for c in data.chunks_mut(bb) {
                for &f in fields {
                    half(c, f);
                }
            }
        }
    }
}

/// f32 -> f16 (round to nearest, normal range only – enough for test values)
fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    if e <= 0 {
        return sign;
    }
    let m = (b >> 13) & 0x3ff;
    sign | ((e as u16) << 10) | m as u16
}

/// One dequant + GEMM schedule, shared by both backends: blocks of rows, per block the parts
pub fn plan(m: &Mat, t: usize, parts: usize, r0: usize, nr: usize) -> Vec<(usize, usize, Vec<(GemmP, u32, u32)>)> {
    let rb = row_block(m.cols, nr);
    let mut v = Vec::new();
    let mut b = 0;
    while b < nr {
        let nb = rb.min(nr - b);
        v.push((r0 + b, nb, gemm_plan(t, m.cols, parts, nr, b, nb)));
        b += nb;
    }
    v
}
