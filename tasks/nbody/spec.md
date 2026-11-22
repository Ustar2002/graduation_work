# Task: gravitational N-body accelerations (all pairs, softened)

## Definition

Implement a function named `nbody` that computes, for each of n particles, the acceleration caused by the gravitational attraction of all n particles. Particle i has the position (pos[i][0], pos[i][1], pos[i][2]) and the mass mass[i]. For 0 <= i < n and each coordinate d in {0, 1, 2}:

    acc[i][d] = sum over j from 0 to n-1 of  mass[j] * (pos[j][d] - pos[i][d]) / (r2(i, j) + eps^2)^(3/2)

    r2(i, j) = (pos[j][0] - pos[i][0])^2 + (pos[j][1] - pos[i][1])^2 + (pos[j][2] - pos[i][2])^2

The softening length eps is a fixed constant of the task, not a parameter: eps = 0.01 exactly, so eps^2 = 0.0001. The gravitational constant is 1. The term j = i is exactly zero (its numerator is zero and its denominator is eps^3, which is positive), so it may be included in the sum or skipped. The output is the acceleration, not the force: acc[i] is not multiplied by mass[i].

## Data layout

- `pos`: an array of shape (n, 3) of IEEE-754 single-precision (32-bit) floating-point numbers (float32). Storage is row-major and contiguous, with no padding between rows: the coordinate d of particle i is the element at linear index 3*i + d, so the three coordinates of one particle are adjacent.
- `mass`: n float32 values; mass[j] is the mass of particle j.
- `acc`: an array of shape (n, 3) of float32 values with the same layout as `pos`: the component d of the acceleration of particle i is the element at linear index 3*i + d.
- `pos` and `acc` each contain exactly 3*n elements and `mass` contains exactly n elements. Do not read or write outside these ranges.
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
- The function must be correct for every n >= 1, including very small n and n that is not a multiple of 2, 4, 8, 16 or any other block size.
- Input values are finite float32 numbers (in the evaluation every coordinate is drawn uniformly from the interval [0, 1) and every mass uniformly from the interval between 0.5/n and 1.5/n, so the total mass is close to 1 and the accelerations are of order 1 for every n). No minimum distance between particles is guaranteed: two different particles may be closer to each other than eps. No special handling of NaN, infinity or subnormal values is required, but the output must not contain NaN or infinity.

## Accuracy

The output is compared element by element against a reference computed from the same inputs with 64-bit (double-precision) arithmetic. The check passes if, for every element,

    |acc[i][d] - ref[i][d]| <= atol + rtol * |ref[i][d]|,    with rtol = 1e-4 and atol = 1e-4.

Any NaN or infinity in `acc` fails the check. Within this tolerance, the order of accumulation and the precision used for intermediate values are free choices.

## Evaluation

- Correctness shapes (n): 5, 317 and 1,024.
- Performance shape (n): 8,192. The output for the performance shape must also pass the accuracy check.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with acc pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
