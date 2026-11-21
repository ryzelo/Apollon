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

/// The storage of a matrix as the GPU kernels need it
pub enum View {
    Plain { dt: DType, data: &'static [u8], row_bytes: usize },
    Mlx { packed: &'static [u8], wpr: usize, scales: (DType, &'static [u8]), biases: Option<(DType, &'static [u8])>, ng: usize, bits: usize, gs: usize },
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

/// Rows per dot_rows call in the CPU matmul
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const DOT_R: usize = 4;

/// dot(w[j], x) for R rows at once – the same lanes, the same order, so every result is bit-identical to
/// dot(); the R sums are independent, so they run side by side instead of waiting on each other
#[inline(always)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn dot_rows(w: [&[f32]; DOT_R], x: &[f32]) -> [f32; DOT_R] {
    let n = x.len();
    let (w0, w1, w2, w3) = (&w[0][..n], &w[1][..n], &w[2][..n], &w[3][..n]);
    let (mut a0, mut a1, mut a2, mut a3) = ([0f32; 16], [0f32; 16], [0f32; 16], [0f32; 16]);
    for ((((xc, c0), c1), c2), c3) in x.chunks_exact(16).zip(w0.chunks_exact(16)).zip(w1.chunks_exact(16)).zip(w2.chunks_exact(16)).zip(w3.chunks_exact(16)) {
        for j in 0..16 {
            a0[j] += c0[j] * xc[j];
            a1[j] += c1[j] * xc[j];
            a2[j] += c2[j] * xc[j];
            a3[j] += c3[j] * xc[j];
        }
    }
    let n16 = n / 16 * 16;
    let fin = |acc: [f32; 16], wr: &[f32]| {
        let mut s = 0f32;
        for v in acc {
            s += v;
        }
        for (a, b) in wr[n16..].iter().zip(&x[n16..]) {
            s += a * b;
        }
        s
    };
    [fin(a0, w0), fin(a1, w1), fin(a2, w2), fin(a3, w3)]
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

    pub fn view(&self) -> View {
        match &self.store {
            Store::Plain { dt, data, row_bytes } => View::Plain { dt: *dt, data, row_bytes: *row_bytes },
            Store::Mlx { packed, wpr, scales, biases, ng, bits, gs } => View::Mlx { packed, wpr: *wpr, scales: *scales, biases: *biases, ng: *ng, bits: *bits, gs: *gs },
        }
    }

    /// x [t][cols] times rows r0..r0+nb: [t][nb] (on the GPU if one is active)
    pub fn rows_times(&self, r0: usize, nb: usize, x: &[f32], t: usize) -> Vec<f32> {
        if let Some(y) = crate::gpu::backend().and_then(|g| g.matmul(self, x, t, 1, r0, nb)) {
            return y;
        }
        let cols = self.cols;
        let mut wb = vec![0f32; nb * cols];
        wb.par_chunks_mut(cols).enumerate().for_each(|(i, r)| self.row(r0 + i, r));
        let mut c = vec![0f32; t * nb];
        crate::blas::gemm_nt(x, &wb, t, nb, cols, &mut c);
        c
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
            // fast path: 4 bits, groups of whole words – 8 values per u32, same arithmetic as below
            Store::Mlx { packed, wpr, scales, biases, ng, bits: 4, gs } if gs % 8 == 0 => {
                let w = &packed[r * wpr * 4..(r + 1) * wpr * 4];
                for g in 0..*ng {
                    let s = scalar(scales.0, scales.1, r * ng + g);
                    let b = biases.map_or(0.0, |(dt, d)| scalar(dt, d, r * ng + g));
                    let c0 = g * gs;
                    for (k, o) in out[c0..c0 + gs].chunks_exact_mut(8).enumerate() {
                        let i = (c0 / 8 + k) * 4;
                        let v = u32::from_le_bytes([w[i], w[i + 1], w[i + 2], w[i + 3]]);
                        for (j, x) in o.iter_mut().enumerate() {
                            *x = s * ((v >> (4 * j)) & 0xF) as f32 + b;
                        }
                    }
                }
            }
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
        if let Some(y) = crate::gpu::backend().and_then(|g| g.matmul(self, x, t, parts, 0, rows)) {
            return y;
        }
        self.matmul_split_cpu(x, t, parts)
    }

    /// matmul_split on the CPU only. Outside macOS always the blocked GEMM (FMA): every result depends only on
    /// its own row of x and of W, never on how many rows are in the batch or where – a batch of variants and a
    /// standalone rerun stay bit-identical.
    pub fn matmul_split_cpu(&self, x: &[f32], t: usize, parts: usize) -> Vec<f32> {
        let _pg = crate::prof::time(crate::prof::MATMUL);
        #[cfg(target_os = "macos")]
        if t < 64 {
            return self.matmul_split_dot(x, t, parts);
        }
        #[cfg(target_os = "macos")]
        return self.matmul_gemm(x, t, parts);
        #[cfg(not(target_os = "macos"))]
        self.matmul_tiles(x, t, parts)
    }

    /// Outside macOS: every thread takes a strip of weight rows, unpacks it into its own buffer (it stays in
    /// the cache) and multiplies it with all of x via matrixmultiply (FMA), part by part.
    #[cfg(not(target_os = "macos"))]
    fn matmul_tiles(&self, x: &[f32], t: usize, parts: usize) -> Vec<f32> {
        #[derive(Clone, Copy)]
        struct OutPtr(*mut f32);
        unsafe impl Send for OutPtr {}
        unsafe impl Sync for OutPtr {}
        let (rows, cols) = (self.rows, self.cols);
        assert_eq!(x.len(), t * cols);
        let seg = cols / parts;
        let mut out = vec![0f32; t * parts * rows];
        if t == 0 || rows == 0 {
            return out;
        }
        // about two strips per thread, a multiple of 16 rows, at most 256 rows (unpacked ≤ 256·cols floats)
        let tn = rows.div_ceil(2 * rayon::current_num_threads()).next_multiple_of(16).clamp(16, 256);
        let op = OutPtr(out.as_mut_ptr());
        let strips: Vec<usize> = (0..rows).step_by(tn).collect();
        strips.into_par_iter().for_each_init(Vec::new, |buf: &mut Vec<f32>, r0| {
            let op = op;
            let nb = tn.min(rows - r0);
            buf.resize(nb * cols, 0.0);
            for i in 0..nb {
                self.row(r0 + i, &mut buf[i * cols..(i + 1) * cols]);
            }
            for p in 0..parts {
                // C[k][r] at out[(k·parts + p)·rows + r0 + r] = Σ_c x[k·cols + p·seg + c] · W[r0 + r][p·seg + c]
                // SAFETY: strips cover disjoint columns r0..r0+nb of every output row; x and buf hold the ranges read
                unsafe {
                    matrixmultiply::sgemm(
                        t, seg, nb, 1.0, x.as_ptr().add(p * seg), cols as isize, 1, buf.as_ptr().add(p * seg), 1, cols as isize, 0.0,
                        op.0.add(p * rows + r0), (parts * rows) as isize, 1,
                    );
                }
            }
        });
        out
    }

    /// matmul_split row by row with dot() (the reference; on macOS for few rows)
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn matmul_split_dot(&self, x: &[f32], t: usize, parts: usize) -> Vec<f32> {
        let (rows, cols) = (self.rows, self.cols);
        let seg = cols / parts;
        const BR: usize = 8;
        let stride = t * parts;
        let mut tmp = vec![0f32; rows * stride]; // [rows][t][parts]
        tmp.par_chunks_mut(BR * stride).enumerate().for_each_init(
            || vec![0f32; BR * cols],
            |buf, (ci, chunk)| {
                let nr = chunk.len() / stride;
                for i in 0..nr {
                    self.row(ci * BR + i, &mut buf[i * cols..(i + 1) * cols]);
                }
                // DOT_R weight rows against one x row at a time: independent sums for the CPU to overlap,
                // every single one in exactly the order of dot() (bit-identical)
                for k in 0..t {
                    let xk = &x[k * cols..(k + 1) * cols];
                    for p in 0..parts {
                        let xs = &xk[p * seg..(p + 1) * seg];
                        let mut i = 0;
                        while i + DOT_R <= nr {
                            let ws: [&[f32]; DOT_R] = std::array::from_fn(|j| &buf[(i + j) * cols + p * seg..][..seg]);
                            let r = dot_rows(ws, xs);
                            for j in 0..DOT_R {
                                chunk[(i + j) * stride + k * parts + p] = r[j];
                            }
                            i += DOT_R;
                        }
                        for i in i..nr {
                            chunk[i * stride + k * parts + p] = dot(&buf[i * cols + p * seg..][..seg], xs);
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

    /// Unpack the weights block by block, then a matrix multiplication per block (Accelerate on macOS,
    /// matrixmultiply elsewhere). Same result as matmul_split_dot up to rounding of the summation order.
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
#[cfg(test)]
mod tests {
    use super::*;

    fn vals(n: usize, seed: u32) -> Vec<f32> {
        let mut z = seed.wrapping_mul(2654435761).wrapping_add(12345);
        (0..n)
            .map(|_| {
                z ^= z << 13;
                z ^= z >> 17;
                z ^= z << 5;
                (z as f32 / u32::MAX as f32 - 0.5) * 3.0
            })
            .collect()
    }

    #[test]
    #[ignore]
    fn bench_matmul() {
        let leak = |v: Vec<f32>| -> &'static [u8] { Box::leak(v.into_iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>().into_boxed_slice()) };
        for (rows, cols, t, parts) in [(1536usize, 576usize, 300usize, 1usize), (576, 1536, 300, 1), (576, 576, 300, 9), (1536, 576, 770, 1), (576, 1536, 770, 1), (576, 576, 770, 9), (1536, 576, 5, 1)] {
            let m = Mat::new(rows, cols, DType::F32, leak(vals(rows * cols, 1)), "w").unwrap();
            let x = vals(t * cols, 2);
            let _ = m.matmul_split_cpu(&x, t, parts);
            let s = std::time::Instant::now();
            for _ in 0..10 {
                std::hint::black_box(m.matmul_split_cpu(&x, t, parts));
            }
            println!("{rows}x{cols} t={t} parts={parts}: {:.2} ms", s.elapsed().as_secs_f64() * 100.0);
            let s = std::time::Instant::now();
            for _ in 0..10 {
                std::hint::black_box(m.matmul_split_dot(&x, t, parts));
            }
            println!("    dot path: {:.2} ms", s.elapsed().as_secs_f64() * 100.0);
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    #[ignore]
    fn bench_sgemm_raw() {
        let (m, k, n) = (300usize, 576usize, 1536usize);
        let a = vals(m * k, 1);
        let b = vals(n * k, 2);
        let mut c = vec![0f32; m * n];
        for (tm, tn) in [(300usize, 1536usize), (300, 128), (64, 64), (128, 256), (300, 64)] {
            let s = std::time::Instant::now();
            for _ in 0..5 {
                // one thread, tiles one after another
                for i0 in (0..m).step_by(tm) {
                    for j0 in (0..n).step_by(tn) {
                        let (mm, nn) = (tm.min(m - i0), tn.min(n - j0));
                        unsafe { matrixmultiply::sgemm(mm, k, nn, 1.0, a.as_ptr().add(i0 * k), k as isize, 1, b.as_ptr().add(j0 * k), 1, k as isize, 0.0, c.as_mut_ptr().add(i0 * n + j0), n as isize, 1) };
                    }
                }
            }
            let secs = s.elapsed().as_secs_f64() / 5.0;
            println!("1 thread, tiles {tm}x{tn}: {:.2} ms, {:.1} GFLOP/s", secs * 1e3, 2.0 * (m * k * n) as f64 / secs / 1e9);
        }
        let s = std::time::Instant::now();
        for _ in 0..5 {
            crate::blas::gemm_nt(&a, &b, m, n, k, &mut c);
        }
        let secs = s.elapsed().as_secs_f64() / 5.0;
        println!("gemm_nt parallel: {:.2} ms, {:.1} GFLOP/s", secs * 1e3, 2.0 * (m * k * n) as f64 / secs / 1e9);
    }

    #[test]
    fn dot_rows_is_bit_identical_to_dot() {
        for n in [1usize, 15, 16, 17, 64, 100, 576, 1536] {
            let w: Vec<Vec<f32>> = (0..4).map(|r| vals(n, r as u32 + 7)).collect();
            let x = vals(n, 99);
            let got = dot_rows([&w[0][..], &w[1][..], &w[2][..], &w[3][..]], &x);
            for r in 0..4 {
                assert_eq!(got[r].to_bits(), dot(&w[r], &x).to_bits(), "n={n} r={r}");
            }
        }
    }

    /// the property the circuit checks rely on: a row's result does not depend on the other rows of the
    /// batch (how many, or where the row sits), bit for bit (outside macOS, where matrixmultiply does every size)
    #[test]
    #[cfg(not(target_os = "macos"))]
    fn matmul_rows_are_independent_of_the_batch() {
        let leak = |v: Vec<f32>| -> &'static [u8] { Box::leak(v.into_iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>().into_boxed_slice()) };
        for (rows, cols, parts) in [(1536usize, 576usize, 1usize), (576, 1536, 1), (576, 576, 9), (37, 80, 5), (700, 64, 1)] {
            let w = vals(rows * cols, 3);
            let m = Mat::new(rows, cols, DType::F32, leak(w.clone()), "w").unwrap();
            let big = 300;
            let xs = vals(big * cols, 4);
            let all = m.matmul_split_cpu(&xs, big, parts);
            let stride = parts * rows;
            // the same rows alone, and in batches of other sizes at other offsets
            for (t, off) in [(1usize, 0usize), (1, 137), (1, 299), (5, 0), (5, 211), (64, 17), (130, 101)] {
                let y = m.matmul_split_cpu(&xs[off * cols..(off + t) * cols], t, parts);
                for k in 0..t {
                    for (a, b) in y[k * stride..(k + 1) * stride].iter().zip(&all[(off + k) * stride..(off + k + 1) * stride]) {
                        assert_eq!(a.to_bits(), b.to_bits(), "{rows}x{cols} parts={parts} t={t} off={off} row {k}");
                    }
                }
            }
            // and close to the plain dot products
            let seg = cols / parts;
            for k in [0usize, 150, 299] {
                for p in 0..parts {
                    for r in (0..rows).step_by(7) {
                        let want = dot(&w[r * cols + p * seg..][..seg], &xs[k * cols + p * seg..][..seg]);
                        let got = all[(k * parts + p) * rows + r];
                        assert!((got - want).abs() <= 1e-4 * (1.0 + want.abs()), "{got} vs {want}");
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn gemm_nt_rows_are_independent_of_the_batch() {
        let (n, k) = (1000usize, 576usize);
        let b = vals(n * k, 5);
        let a = vals(200 * k, 6);
        let mut all = vec![0f32; 200 * n];
        crate::blas::gemm_nt(&a, &b, 200, n, k, &mut all);
        for (t, off) in [(1usize, 0usize), (1, 199), (3, 50), (129, 70)] {
            let mut c = vec![0f32; t * n];
            crate::blas::gemm_nt(&a[off * k..(off + t) * k], &b, t, n, k, &mut c);
            for (x, y) in c.iter().zip(&all[off * n..(off + t) * n]) {
                assert_eq!(x.to_bits(), y.to_bits(), "t={t} off={off}");
            }
        }
    }

    #[test]
    fn matmul_split_dot_is_bit_identical_to_rowwise_dot() {
        let leak = |v: Vec<f32>| -> &'static [u8] { Box::leak(v.into_iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>().into_boxed_slice()) };
        for (rows, cols, t, parts) in [(13usize, 64usize, 5usize, 1usize), (32, 576, 7, 1), (19, 576, 3, 9), (8, 48, 1, 3)] {
            let w = vals(rows * cols, 1);
            let x = vals(t * cols, 2);
            let m = Mat::new(rows, cols, DType::F32, leak(w.clone()), "w").unwrap();
            let y = m.matmul_split_dot(&x, t, parts);
            let seg = cols / parts;
            for k in 0..t {
                for p in 0..parts {
                    for r in 0..rows {
                        let want = dot(&w[r * cols + p * seg..][..seg], &x[k * cols + p * seg..][..seg]);
                        assert_eq!(y[(k * parts + p) * rows + r].to_bits(), want.to_bits(), "{rows}x{cols} t={t} parts={parts}");
                    }
                }
            }
        }
    }
}
