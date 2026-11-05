# Task: 2-D Jacobi relaxation (5-point stencil, T steps)

## Definition

Implement a function named `jacobi2d` that applies T steps of a 5-point Jacobi iteration to a 2-D grid with H rows and W columns.

Let u(0) be the input grid `src`. For each step t = 1, ..., T the grid u(t) is defined from u(t-1) by

    u(t)[y][x] = 0.25 * ( u(t-1)[y-1][x] + u(t-1)[y+1][x] + u(t-1)[y][x-1] + u(t-1)[y][x+1] )    for 1 <= y <= H-2 and 1 <= x <= W-2

    u(t)[y][x] = u(t-1)[y][x]    for boundary cells (y = 0, y = H-1, x = 0 or x = W-1)

Every step reads only values of the previous step (Jacobi, not Gauss-Seidel): a cell updated in step t must not be used by another cell in the same step. The boundary cells therefore keep their input values through all steps. The output `dst` must contain u(T).

## Data layout

- Both grids are arrays of IEEE-754 single-precision (32-bit) floating-point numbers (float32).
- Storage is row-major and contiguous, with no padding between rows: the cell in row y and column x is the element at linear index y*W + x.
- `src` and `dst` each contain exactly H*W elements. Do not read or write outside these ranges.
- The start addresses of `src` and `dst` are 64-byte aligned. The two buffers do not overlap.
- H, W and T are positive integers with H >= 3, W >= 3 and T >= 1, and H and W always agree with the actual sizes of the buffers.

## Contract

- The function has exactly five parameters, in this order: src, dst, H, W, T. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- dst is allocated by the caller. Its initial contents are unspecified (arbitrary finite values) and must not be used: the function must overwrite every element of dst with the correct result. It must not add to, or otherwise combine with, whatever dst contained before the call.
- The result must be written into dst. The return value of the function is ignored: results are read only from the buffer dst, so returning a new array instead of filling dst counts as not writing dst.
- `src` is read-only. The function must not modify it, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce results that agree within the tolerance below.
- The function must be correct for every H >= 3, W >= 3 and T >= 1, including non-square grids, very small grids, odd and even T, and dimensions that are not multiples of 2, 4, 8, 16 or any other block size.
- Input values are finite float32 numbers (in the evaluation they are drawn uniformly from the interval [0, 1)). No special handling of NaN, infinity or subnormal values is required, but the output must not contain NaN or infinity.

## Accuracy

The output is compared element by element against a reference computed from the same input with 64-bit (double-precision) arithmetic. The check passes if, for every element,

    |dst[y][x] - ref[y][x]| <= atol + rtol * |ref[y][x]|,    with rtol = 1e-5 and atol = 1e-6.

Any NaN or infinity in `dst` fails the check. Within this tolerance, the order in which the four neighbours are added and the precision used for intermediate values are free choices.

## Evaluation

- Correctness shapes (H, W, T): (5, 7, 3), (130, 200, 4) and (512, 512, 8).
- Performance shape (H, W, T): (4096, 4096, 20).
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with dst pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
