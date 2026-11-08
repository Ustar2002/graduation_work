# Task: N-body gravitational acceleration by direct summation (single precision)

## Definition

Implement a function named `nbody` that computes, for each of n bodies, the softened gravitational acceleration exerted on it by all bodies, by direct summation over all pairs. Body i has position p_i = (pos[i][0], pos[i][1], pos[i][2]) and mass mass[i]. For every i from 0 to n-1 and every coordinate c (0 = x, 1 = y, 2 = z):

    acc[i][c] = sum over j from 0 to n-1 of mass[j] * (pos[j][c] - pos[i][c]) / (r2(i, j) + EPS2)^(3/2)

    where r2(i, j) = (pos[j][0] - pos[i][0])^2 + (pos[j][1] - pos[i][1])^2 + (pos[j][2] - pos[i][2])^2

with the softening constant EPS2 = 1e-4 (that is, 0.0001) and the gravitational constant G = 1 (no further scaling). The term with j = i is exactly zero (its numerator is zero while its denominator is EPS2^(3/2) > 0), so the sum may either include it or skip it; both give the same result.

Because EPS2 > 0 the denominator is never zero, so the result is finite for every input, even when two bodies coincide.

## Data layout

- `pos` holds the positions of the n bodies as n * 3 IEEE-754 single-precision (32-bit) floating-point numbers (float32), contiguous and ordered body by body: coordinate c of body i is the element at linear index 3*i + c (x, y, z of body 0, then x, y, z of body 1, and so on).
- `mass` holds the n masses as n contiguous float32 values; mass[i] is the mass of body i.
- `acc` holds the output accelerations as n * 3 contiguous float32 values in the same layout as `pos`: component c of the acceleration of body i is the element at linear index 3*i + c.
- Do not read or write outside these ranges.
- The start addresses of `pos`, `mass` and `acc` are 64-byte aligned. The three buffers do not overlap.
- n is a positive integer and always agrees with the actual sizes of the buffers.

## Contract

- The function has exactly four parameters, in this order: pos, mass, acc, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- acc is allocated by the caller. Its initial contents are unspecified (arbitrary finite values) and must not be used: the function must overwrite every element of acc with the correct result. It must not add to, or otherwise combine with, whatever acc contained before the call.
- The result must be written into acc. The return value of the function is ignored: results are read only from the buffer acc, so returning a new array instead of filling acc counts as not writing acc.
- `pos` and `mass` are read-only. The function must not modify them, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce results that agree within the tolerance below.
- The function must be correct for every positive n, including n = 1 (all accelerations are zero), very small n, and n that is not a multiple of 2, 4, 8, 16 or any other block size.
- The sum must be taken over all n bodies for every body (direct summation). Approximation schemes that skip, truncate or merge the contributions of distant bodies are not permitted: every pair must contribute to the result.
- Input values are finite float32 numbers. In the evaluation the positions are drawn uniformly from the unit cube [0, 1)^3 and the masses uniformly from the interval [0.5, 1.5). With these inputs the individual terms of the sum range from below 1 to several thousand, the terms partly cancel, and the resulting acceleration components have magnitudes up to a few tens of thousands. No special handling of NaN, infinity or subnormal values is required, but the output must not contain NaN or infinity.

## Accuracy

The output is compared element by element against a reference computed from the same inputs in 64-bit (double-precision) arithmetic by direct summation over all pairs. The check passes if, for every element,

    |acc[i][c] - ref[i][c]| <= atol + rtol * |ref[i][c]|,    with rtol = 1e-3 and atol = 1e-3 * RMS(ref),

where RMS(ref) is the root mean square of all reference acceleration components of the shape being checked (so atol scales with the size of the problem). Any NaN or infinity in `acc` fails the check. Within this tolerance, the order of summation, the precision used for intermediate values and the way the power (r2 + EPS2)^(-3/2) is evaluated are free choices.

## Evaluation

- Correctness shapes (n): 5, 301 and 2,048.
- Performance shape (n): 16,384. The output for the performance shape must also pass the accuracy check.
- Memory: the process that calls the function is limited to 8 GiB of address space, inputs and outputs included. An allocation beyond that limit fails the call and counts as a failure. A full n-by-n table of pair vectors does not fit at n = 16,384.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with acc pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
