#![allow(non_snake_case)]

use rayon::prelude::*;
use std::slice;

#[no_mangle]
pub unsafe extern "C" fn gemm(A: *const f32, B: *const f32, C: *mut f32, M: i32, N: i32, K: i32) {
    let (m, n, k) = (M as usize, N as usize, K as usize);
    let a = slice::from_raw_parts(A, m * k);
    let b = slice::from_raw_parts(B, k * n);
    let c = slice::from_raw_parts_mut(C, m * n);

    // Rows of C are independent: one task per row, i-k-j inside so the inner loop is contiguous.
    c.par_chunks_mut(n)
        .zip(a.par_chunks(k))
        .for_each(|(c_row, a_row)| {
            c_row.fill(0.0);
            for (p, &a_ip) in a_row.iter().enumerate() {
                let b_row = &b[p * n..(p + 1) * n];
                for (c_ij, &b_pj) in c_row.iter_mut().zip(b_row) {
                    *c_ij += a_ip * b_pj;
                }
            }
        });
}
