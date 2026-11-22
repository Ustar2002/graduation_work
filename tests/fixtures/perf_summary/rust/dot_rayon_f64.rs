#![allow(non_snake_case)]

use rayon::prelude::*;
use std::slice;

const CHUNK: usize = 1 << 14;

#[no_mangle]
pub unsafe extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {
    let len = n as usize;
    let xs = slice::from_raw_parts(x, len);
    let ys = slice::from_raw_parts(y, len);
    let total: f64 = xs
        .par_chunks(CHUNK)
        .zip(ys.par_chunks(CHUNK))
        .map(|(a, b)| {
            a.iter()
                .zip(b)
                .map(|(&p, &q)| p as f64 * q as f64)
                .sum::<f64>()
        })
        .sum();
    *out = total as f32;
}
