These notes describe the parts of Rust and of the `rayon` crate that concern a function exported through the C ABI. They are the same for every task.

**Exported function and raw pointers**

```rust
#![allow(non_snake_case)]

#[no_mangle]
pub unsafe extern "C" fn scale(src: *const f32, dst: *mut f32, n: i32) {
    let len = n as usize;
    let input: &[f32] = std::slice::from_raw_parts(src, len);
    let output: &mut [f32] = std::slice::from_raw_parts_mut(dst, len);
    for i in 0..len {
        output[i] = 2.0 * input[i];
    }
}
```

- `#[no_mangle]` exports the function under its plain name (`#[unsafe(no_mangle)]` is an accepted spelling of the same attribute); without it the symbol is not found under that name. `extern "C"` selects the C calling convention.
- Inner attributes such as `#![allow(non_snake_case)]` are placed at the top of the file, before any item.
- In edition 2021 the body of an `unsafe fn` may dereference raw pointers and call `unsafe` functions directly; in a safe function these operations need an `unsafe { ... }` block.
- `std::slice::from_raw_parts(ptr, len)` gives `&[T]` and `std::slice::from_raw_parts_mut(ptr, len)` gives `&mut [T]`; `len` is a number of elements of type `usize`.
- A raw pointer can also be used without a slice: `*ptr.add(i)` reads or writes element `i`, without a bounds check.
- Slice indexing `s[i]` is bounds-checked, and an index out of range panics.
- A structure received by pointer (`args: *const T`) is read through a reference: `let a = &*args;`, then `a.field`.

**Integer and floating-point conversions**

- There are no implicit conversions between numeric types; `as` converts explicitly: `n as usize`, `i as i32`, `x as f64`, `v as f32`. Slice lengths and indices are `usize`.
- `as` from a negative `i32` to `usize` wraps around to a very large value. `as` from a floating-point value to an integer truncates toward zero and saturates at the bounds of the integer type.
- Literal suffixes fix a type: `0.0f32`, `1.0_f64`, `0usize`, `1i64`. A literal without a suffix takes the type required by its context.
- In the release profile, integer overflow in `+`, `-` and `*` wraps around silently; `wrapping_add`, `checked_mul` and similar methods state the behaviour explicitly.

**Rayon parallel iterators**

```rust
use rayon::prelude::*;

pub fn examples(input: &[f32], output: &mut [f32], width: usize) -> usize {
    output.par_iter_mut().zip(input.par_iter()).for_each(|(o, &x)| *o = 2.0 * x);
    output.par_chunks_mut(width).enumerate().for_each(|(row, chunk)| {
        for (col, value) in chunk.iter_mut().enumerate() {
            *value += (row * width + col) as f32;
        }
    });
    let positives = input.par_iter().filter(|&&x| x > 0.0).count();
    let largest = (0..input.len()).into_par_iter().map(|i| input[i]).reduce(|| f32::MIN, f32::max);
    let squares: usize = (0..width).into_par_iter().map(|i| i * i).sum();
    positives + squares + largest as usize
}
```

- `use rayon::prelude::*;` brings the parallel iterator methods into scope.
- Sources: `slice.par_iter()`, `slice.par_iter_mut()`, `slice.par_chunks(n)`, `slice.par_chunks_mut(n)` and `(a..b).into_par_iter()`.
- Adaptors: `.enumerate()`, `.zip(other)`, `.map(f)`, `.filter(f)`. Consumers: `.for_each(f)`, `.count()`, `.sum::<T>()`, `.reduce(identity, op)`, where `identity` is a closure that returns the neutral element.
- `rayon::join(|| a, || b)` runs two closures, possibly in parallel, and returns both results. `rayon::scope(|s| { s.spawn(|_| ...); })` runs spawned closures to completion before it returns.
- A closure given to a parallel iterator may run on several threads at once. It can read shared data through `&` references and it owns the item it receives (for example one `&mut` chunk); it cannot mutate a captured local variable. Shared mutable state uses `std::sync::atomic` types or `std::sync::Mutex`.
- Raw pointers are neither `Send` nor `Sync`, so a closure that captures one is rejected by parallel iterators, `rayon::join` and `rayon::scope`; slices made from the pointers can be captured.
