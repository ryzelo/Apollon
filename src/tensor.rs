//! Weight matrix directly on the memory-mapped file. Rows are dequantized only when
//! they are used (one row per thread in a buffer) - nothing is inflated in RAM.
use crate::quant::{dequant, DType};
use rayon::prelude::*;

/// Storage format of a matrix in the file
enum Store {
    /// GGML block format or F32/F16/BF16, row by row
    Plain { dt: DType, data: &'static [u8], row_bytes: usize },
    /// MLX "affine": w = scale * q + bias, q as a bit stream in uint32, groups of gs values
    Mlx { packed: &'static [u8], wpr: usize, scales: (DType, &'static [u8]), biases: Option<(DType, &'static [u8])>, ng: usize, bits: usize, gs: usize },
}

pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    store: Store,
}

#[inline]
fn scalar(dt: DType, d: &[u8], i: usize) -> f32 {
    match dt {
        DType::F32 => f32::from_le_bytes([d[4 * i], d[4 * i + 1], d[4 * i + 2], d[4 * i + 3]]),
        DType::F16 => crate::quant::f16(u16::from_le_bytes([d[2 * i], d[2 * i + 1]])),
        _ => f32::from_bits((u16::from_le_bytes([d[2 * i], d[2 * i + 1]]) as u32) << 16),
    }
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let ca = a.chunks_exact(16);
    let cb = b.chunks_exact(16);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for j in 0..16 {
            acc[j] += x[j] * y[j];
        }
    }
    let mut s = 0f32;
    for v in acc {
        s += v;
    }
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

impl Mat {
    pub fn new(rows: usize, cols: usize, dt: DType, data: &'static [u8], name: &str) -> Result<Self, String> {
        let (be, bb) = dt.block();
        if cols % be != 0 {
            return Err(format!("{name}: {cols} columns do not fit {} blocks", dt.name()));
        }
        let row_bytes = cols / be * bb;
        if data.len() < rows * row_bytes {
            return Err(format!("{name}: file too short for {rows}x{cols} {}", dt.name()));
        }
        Ok(Mat { rows, cols, store: Store::Plain { dt, data: &data[..rows * row_bytes], row_bytes } })
    }

    /// MLX-quantized matrix: packed [rows][wpr] u32, scales/biases [rows][ng]
    #[allow(clippy::too_many_arguments)]
    pub fn mlx(
        rows: usize, wpr: usize, packed: &'static [u8], scales: (DType, &'static [u8]), biases: Option<(DType, &'static [u8])>,
        ng: usize, gs: usize, name: &str,
    ) -> Result<Self, String> {
        let cols = ng * gs;
        if (32 * wpr) % cols != 0 {
            return Err(format!("{name}: MLX packing does not fit ({wpr} words, {ng} groups of {gs})"));
        }
        let bits = 32 * wpr / cols;
        if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) || packed.len() < rows * wpr * 4 {
            return Err(format!("{name}: MLX bit width {bits} not plausible"));
        }
        Ok(Mat { rows, cols, store: Store::Mlx { packed, wpr, scales, biases, ng, bits, gs } })
    }

    pub fn type_name(&self) -> String {
        match &self.store {
            Store::Plain { dt, .. } => dt.name().to_string(),
            Store::Mlx { bits, gs, .. } => format!("MLX-Q{bits}/{gs}"),
        }
    }

    #[inline]
    pub fn row(&self, r: usize, out: &mut [f32]) {
        match &self.store {
            Store::Plain { dt, data, row_bytes } => dequant(*dt, &data[r * row_bytes..(r + 1) * row_bytes], &mut out[..self.cols]),
            Store::Mlx { packed, wpr, scales, biases, ng, bits, gs } => {
                let w = &packed[r * wpr * 4..(r + 1) * wpr * 4];
                let word = |k: usize| u32::from_le_bytes([w[4 * k], w[4 * k + 1], w[4 * k + 2], w[4 * k + 3]]) as u64;
                let mask = (1u64 << bits) - 1;
                for g in 0..*ng {
                    let s = scalar(scales.0, scales.1, r * ng + g);
                    let b = biases.map_or(0.0, |(dt, d)| scalar(dt, d, r * ng + g));
                    for c in g * gs..(g + 1) * gs {
                        let bit = c * bits;
                        let (k, sh) = (bit / 32, bit % 32);
                        let mut v = word(k) >> sh;
                        if sh + bits > 32 {
                            v |= word(k + 1) << (32 - sh);
                        }
                        out[c] = s * (v & mask) as f32 + b;
                    }
                }
            }
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = vec![0f32; self.rows * self.cols];
        for r in 0..self.rows {
            self.row(r, &mut v[r * self.cols..(r + 1) * self.cols]);
        }
        v
    }

    /// x: [t][cols] -> [t][rows]
    pub fn matmul(&self, x: &[f32], t: usize) -> Vec<f32> {
        self.matmul_split(x, t, 1)
    }

    /// Like matmul, but the dot product of every row is split into `parts` column ranges of
    /// equal length (e.g. per attention head). Result: [t][parts][rows].
    pub fn matmul_split(&self, x: &[f32], t: usize, parts: usize) -> Vec<f32> {
        let (rows, cols) = (self.rows, self.cols);
        assert_eq!(x.len(), t * cols);
        assert_eq!(cols % parts, 0);
        #[cfg(target_os = "macos")]
        if t >= 64 {
            return self.matmul_gemm(x, t, parts);
        }
        let seg = cols / parts;
        const BR: usize = 8;
        let stride = t * parts;
        let mut tmp = vec![0f32; rows * stride]; // [rows][t][parts]
        tmp.par_chunks_mut(BR * stride).enumerate().for_each_init(
            || vec![0f32; cols],
            |buf, (ci, chunk)| {
                for (i, orow) in chunk.chunks_mut(stride).enumerate() {
                    self.row(ci * BR + i, buf);
                    for k in 0..t {
                        let xk = &x[k * cols..(k + 1) * cols];
                        for p in 0..parts {
                            orow[k * parts + p] = dot(&buf[p * seg..(p + 1) * seg], &xk[p * seg..(p + 1) * seg]);
                        }
                    }
                }
            },
        );
        let mut out = vec![0f32; rows * stride];
        out.par_chunks_mut(rows).enumerate().for_each(|(kp, o)| {
            for r in 0..rows {
                o[r] = tmp[r * stride + kp];
            }
        });
        out
    }

    /// Many rows at once (64 or more): unpack weights block by block, matrix multiplication via
    /// Accelerate. Same result as matmul_split (up to rounding of the summation order).
    #[cfg(target_os = "macos")]
    fn matmul_gemm(&self, x: &[f32], t: usize, parts: usize) -> Vec<f32> {
        let (rows, cols) = (self.rows, self.cols);
        let seg = cols / parts;
        const BR: usize = 512;
        let mut out = vec![0f32; t * parts * rows];
        let mut wb = vec![0f32; BR.min(rows) * cols];
        let mut r0 = 0;
        while r0 < rows {
            let nb = BR.min(rows - r0);
            wb[..nb * cols].par_chunks_mut(cols).enumerate().for_each(|(i, o)| self.row(r0 + i, o));
            for p in 0..parts {
                crate::blas::gemm_nt_ld(&x[p * seg..], cols, &wb[p * seg..], cols, t, nb, seg, &mut out[p * rows + r0..], parts * rows);
            }
            r0 += nb;
        }
        out
    }

    /// W^T u  (length cols)
    pub fn t_matvec(&self, u: &[f32]) -> Vec<f32> {
        assert_eq!(u.len(), self.rows);
        let cols = self.cols;
        (0..self.rows)
            .into_par_iter()
            .fold(
                || (vec![0f32; cols], vec![0f32; cols]),
                |(mut acc, mut buf), r| {
                    if u[r] != 0.0 {
                        self.row(r, &mut buf);
                        let ur = u[r];
                        for (a, b) in acc.iter_mut().zip(&buf) {
                            *a += ur * b;
                        }
                    }
                    (acc, buf)
                },
            )
            .map(|(a, _)| a)
            .reduce(|| vec![0f32; cols], |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            })
    }

    /// Wᵀ u for t vectors at once: u [t][rows] -> [t][cols]. Every row is unpacked once.
    pub fn t_matmul(&self, u: &[f32], t: usize) -> Vec<f32> {
        let (rows, cols) = (self.rows, self.cols);
        assert_eq!(u.len(), t * rows);
        const BR: usize = 64;
        (0..rows.div_ceil(BR))
            .into_par_iter()
            .fold(
                || (vec![0f32; t * cols], vec![0f32; cols]),
                |(mut acc, mut buf), blk| {
                    for r in blk * BR..((blk + 1) * BR).min(rows) {
                        if (0..t).all(|k| u[k * rows + r] == 0.0) {
                            continue;
                        }
                        self.row(r, &mut buf);
                        for k in 0..t {
                            let c = u[k * rows + r];
                            if c != 0.0 {
                                for (a, b) in acc[k * cols..(k + 1) * cols].iter_mut().zip(&buf) {
                                    *a += c * b;
                                }
                            }
                        }
                    }
                    (acc, buf)
                },
            )
            .map(|(a, _)| a)
            .reduce(|| vec![0f32; t * cols], |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            })
    }

    /// Euclidean norm of every column
    pub fn col_norms(&self) -> Vec<f32> {
        let cols = self.cols;
        let sq = (0..self.rows)
            .into_par_iter()
            .fold(
                || (vec![0f64; cols], vec![0f32; cols]),
                |(mut acc, mut buf), r| {
                    self.row(r, &mut buf);
                    for (a, b) in acc.iter_mut().zip(&buf) {
                        *a += (*b as f64) * (*b as f64);
                    }
                    (acc, buf)
                },
            )
            .map(|(a, _)| a)
            .reduce(|| vec![0f64; cols], |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            });
        sq.into_iter().map(|v| v.sqrt() as f32).collect()
    }
}
