#![allow(non_snake_case)]

use std::arch::x86_64::*;
use std::slice;

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(x: &[f32], y: &[f32]) -> f64 {
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let xc = x.chunks_exact(8);
    let yc = y.chunks_exact(8);
    let (xr, yr) = (xc.remainder(), yc.remainder());
    for (a, b) in xc.zip(yc) {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let lo = _mm256_mul_pd(
            _mm256_cvtps_pd(_mm256_castps256_ps128(va)),
            _mm256_cvtps_pd(_mm256_castps256_ps128(vb)),
        );
        let hi = _mm256_mul_pd(
            _mm256_cvtps_pd(_mm256_extractf128_ps(va, 1)),
            _mm256_cvtps_pd(_mm256_extractf128_ps(vb, 1)),
        );
        acc0 = _mm256_add_pd(acc0, lo);
        acc1 = _mm256_add_pd(acc1, hi);
    }
    let mut buf = [0.0f64; 4];
    _mm256_storeu_pd(buf.as_mut_ptr(), _mm256_add_pd(acc0, acc1));
    let mut total = buf[0] + buf[1] + buf[2] + buf[3];
    for (p, q) in xr.iter().zip(yr) {
        total += *p as f64 * *q as f64;
    }
    total
}

#[no_mangle]
pub unsafe extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {
    let len = n as usize;
    let (xs, ys) = unsafe { (slice::from_raw_parts(x, len), slice::from_raw_parts(y, len)) };
    let total = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { dot_avx2(xs, ys) }
    } else {
        xs.iter()
            .zip(ys)
            .map(|(&p, &q)| p as f64 * q as f64)
            .sum::<f64>()
    };
    unsafe { *out = total as f32 };
}
