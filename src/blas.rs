//! Fast primitives for streaming over the vocabulary. On macOS via Apple's Accelerate
//! (matrix multiplication on the AMX unit, vectorized exp/tanh), elsewhere matrixmultiply (AVX2/FMA
//! kernels chosen at run time) in parallel tiles, and plain Rust.
#[cfg(not(target_os = "macos"))]
use rayon::prelude::*;

#[cfg(target_os = "macos")]
mod ffi {
    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub fn cblas_sgemm(
            order: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32, alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: f32,
            c: *mut f32, ldc: i32,
        );
        pub fn vvexpf(y: *mut f32, x: *const f32, n: *const i32);
        pub fn vvtanhf(y: *mut f32, x: *const f32, n: *const i32);
    }
}

/// C[m][n] = A[m][k] · B[n][k]ᵀ  (all row-major, f32)
pub fn gemm_nt(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    assert!(a.len() >= m * k && b.len() >= n * k && c.len() >= m * n);
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(target_os = "macos")]
    unsafe {
        // 101 = row-major, 111 = no transpose, 112 = transpose
        ffi::cblas_sgemm(101, 111, 112, m as i32, n as i32, k as i32, 1.0, a.as_ptr(), k as i32, b.as_ptr(), k as i32, 0.0, c.as_mut_ptr(), n as i32);
    }
    #[cfg(not(target_os = "macos"))]
    gemm_nt_ld(a, k, b, k, m, n, k, c, n);
}

/// C[m][n] = A[m][k] · B[k][n]  (all row-major, f32)
pub fn gemm_nn(a: &[f32], b: &[f32], m: usize, n: usize, k: usize, c: &mut [f32]) {
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n);
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(target_os = "macos")]
    unsafe {
        ffi::cblas_sgemm(101, 111, 111, m as i32, n as i32, k as i32, 1.0, a.as_ptr(), k as i32, b.as_ptr(), n as i32, 0.0, c.as_mut_ptr(), n as i32);
    }
    #[cfg(not(target_os = "macos"))]
    tiled(m, n, c.as_mut_ptr(), n, |i0, j0, mm, nn, cp| unsafe {
        matrixmultiply::sgemm(mm, k, nn, 1.0, a.as_ptr().add(i0 * k), k as isize, 1, b.as_ptr().add(j0), n as isize, 1, 0.0, cp, n as isize, 1);
    });
}

/// a raw output pointer that the tiles share (each tile writes only its own elements)
#[cfg(not(target_os = "macos"))]
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
#[cfg(not(target_os = "macos"))]
unsafe impl Send for OutPtr {}
#[cfg(not(target_os = "macos"))]
unsafe impl Sync for OutPtr {}

/// Split an m×n output into tiles and run `f(i0, j0, rows, cols, &C[i0][j0])` for every tile in parallel.
/// The tiling only decides who computes an element, not how: matrixmultiply sums every element over k
/// in the same fixed order, whatever the size of the matrix or the place of the element in it.
#[cfg(not(target_os = "macos"))]
fn tiled(m: usize, n: usize, c: *mut f32, ldc: usize, f: impl Fn(usize, usize, usize, usize, *mut f32) + Sync) {
    // large tiles (matrixmultiply is ~2x faster on 256x128 than on 64x64), about two per thread
    let tm = m.min(256).max(1);
    let want = 2 * rayon::current_num_threads();
    let per_col = want.div_ceil(m.div_ceil(tm));
    let tn = n.div_ceil(per_col).next_multiple_of(16).clamp(32, 512);
    let cp = OutPtr(c);
    let tiles: Vec<(usize, usize)> = (0..m).step_by(tm).flat_map(|i| (0..n).step_by(tn).map(move |j| (i, j))).collect();
    tiles.into_par_iter().for_each(|(i0, j0)| {
        let cp = cp;
        // SAFETY: the caller checked that C holds (m-1)·ldc + n elements; tiles do not overlap
        f(i0, j0, tm.min(m - i0), tn.min(n - j0), unsafe { cp.0.add(i0 * ldc + j0) });
    });
}

/// Like gemm_nt, with row strides: C[i*ldc + j] = Σ_c A[i*lda + c] · B[j*ldb + c]
#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
pub fn gemm_nt_ld(a: &[f32], lda: usize, b: &[f32], ldb: usize, m: usize, n: usize, k: usize, c: &mut [f32], ldc: usize) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(a.len() >= (m - 1) * lda + k && b.len() >= (n - 1) * ldb + k && c.len() >= (m - 1) * ldc + n);
    tiled(m, n, c.as_mut_ptr(), ldc, |i0, j0, mm, nn, cp| unsafe {
        matrixmultiply::sgemm(mm, k, nn, 1.0, a.as_ptr().add(i0 * lda), lda as isize, 1, b.as_ptr().add(j0 * ldb), 1, ldb as isize, 0.0, cp, ldc as isize, 1);
    });
}

/// Like gemm_nt, with row strides: C[i*ldc + j] = Σ_c A[i*lda + c] · B[j*ldb + c]  (macOS only)
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
pub fn gemm_nt_ld(a: &[f32], lda: usize, b: &[f32], ldb: usize, m: usize, n: usize, k: usize, c: &mut [f32], ldc: usize) {
    if m == 0 || n == 0 {
        return;
    }
    assert!(a.len() >= (m - 1) * lda + k && b.len() >= (n - 1) * ldb + k && c.len() >= (m - 1) * ldc + n);
    unsafe {
        ffi::cblas_sgemm(101, 111, 112, m as i32, n as i32, k as i32, 1.0, a.as_ptr(), lda as i32, b.as_ptr(), ldb as i32, 0.0, c.as_mut_ptr(), ldc as i32);
    }
}

/// y = exp(x)
pub fn exp(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_os = "macos")]
    unsafe {
        let n = x.len() as i32;
        ffi::vvexpf(y.as_mut_ptr(), x.as_ptr(), &n);
    }
    #[cfg(not(target_os = "macos"))]
    for (o, v) in y.iter_mut().zip(x) {
        *o = v.exp();
    }
}

/// y = tanh(x)
pub fn tanh(y: &mut [f32], x: &[f32]) {
    assert_eq!(y.len(), x.len());
    #[cfg(target_os = "macos")]
    unsafe {
        let n = x.len() as i32;
        ffi::vvtanhf(y.as_mut_ptr(), x.as_ptr(), &n);
    }
    #[cfg(not(target_os = "macos"))]
    for (o, v) in y.iter_mut().zip(x) {
        *o = v.tanh();
    }
}
