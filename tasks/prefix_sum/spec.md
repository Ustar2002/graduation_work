# Task: inclusive prefix sum of an integer array

## Definition

Implement a function named `prefix_sum` that computes the running totals (inclusive prefix sums) of an array x of n integers:

    out[i] = x[0] + x[1] + ... + x[i],    for 0 <= i < n

The sum for index i includes x[i] itself: out[0] = x[0], and out[n-1] is the sum of all n elements.

## Data layout

- `x`: n signed 32-bit integers (int32), each in the range 0 <= x[i] < 16.
- `out`: n signed 32-bit integers (int32).
- Both arrays are contiguous. Do not read or write outside the stated ranges.
- The start addresses of `x` and `out` are 64-byte aligned. The two buffers do not overlap.
- n is a positive integer, at most 67,108,864 (2^26), and always agrees with the actual sizes of the buffers.
- Every partial sum fits in a signed 32-bit integer: the largest possible value is 15 * 67,108,864 = 1,006,632,960, which is smaller than 2^31 = 2,147,483,648.

## Contract

- The function has exactly three parameters, in this order: x, out, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- out is allocated by the caller. Its initial contents are unspecified (arbitrary values) and must not be used: the function must overwrite every element of out with the correct result. It must not add to, or otherwise combine with, whatever out contained before the call.
- The result must be written into out. The return value of the function is ignored: results are read only from the buffer out, so returning a new array instead of filling out counts as not writing out.
- `x` is read-only. The function must not modify it, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce identical results.
- The function must be correct for every n from 1 to 67,108,864, including very small n and n that is not a multiple of 2, 4, 8, 16 or any other block size.
- Input values are integers in the range 0 <= x[i] < 16 (in the evaluation they are drawn independently and uniformly at random from 0, 1, ..., 15). Negative values and values of 16 or more do not occur, and no range check is required.

## Accuracy

The output is compared element by element against a reference computed from the same input with exact integer arithmetic. The comparison is exact; there is no tolerance. The check passes only if

    out[i] == ref[i]    for every i with 0 <= i < n.

## Evaluation

- Correctness shapes (n): 7, 100,003 and 1,048,576.
- Performance shape (n): 67,108,864. The output for the performance shape must also pass the accuracy check.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with out pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must be identical. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
