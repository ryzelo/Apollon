//! Fast primitives for streaming over the vocabulary. On macOS via Apple's Accelerate
//! (matrix multiplication on the AMX unit, vectorized exp/tanh), elsewhere in plain Rust.
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
    c[..m * n].par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let ai = &a[i * k..(i + 1) * k];
        for (j, o) in row.iter_mut().enumerate() {
            *o = crate::tensor::dot(ai, &b[j * k..(j + 1) * k]);
        }
    });
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
    c[..m * n].par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        row.iter_mut().for_each(|x| *x = 0.0);
        for kk in 0..k {
            let aik = a[i * k + kk];
            if aik != 0.0 {
                for (o, bv) in row.iter_mut().zip(&b[kk * n..(kk + 1) * n]) {
                    *o += aik * bv;
                }
            }
        }
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
