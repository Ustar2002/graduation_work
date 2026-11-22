#![allow(non_snake_case)]

use std::slice;

const BLOCK: usize = 48;

#[no_mangle]
pub unsafe extern "C" fn gemm(A: *const f32, B: *const f32, C: *mut f32, M: i32, N: i32, K: i32) {
    let (m, n, k) = (M as usize, N as usize, K as usize);
    let a = slice::from_raw_parts(A, m * k);
    let b = slice::from_raw_parts(B, k * n);
    let c = slice::from_raw_parts_mut(C, m * n);
    c.fill(0.0);
    for ii in (0..m).step_by(BLOCK) {
        for pp in (0..k).step_by(BLOCK) {
            for i in ii..(ii + BLOCK).min(m) {
                for p in pp..(pp + BLOCK).min(k) {
                    let a_ip = a[i * k + p];
                    for j in 0..n {
                        c[i * n + j] += a_ip * b[p * n + j];
                    }
                }
            }
        }
    }
}
