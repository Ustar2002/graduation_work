# Task: inclusive prefix sum (scan) of 32-bit integers

## Definition

Implement a function named `scan` that computes the inclusive prefix sum of an integer array x of length n and stores it in the output array `out` of the same length:

    out[i] = sum over k from 0 to i of x[k],    for 0 <= i < n

So out[0] = x[0], out[1] = x[0] + x[1], and so on; out[n-1] is the sum of all elements of x.

## Data layout

- `x` and `out` are arrays of n signed 32-bit integers (two's complement, int32), contiguous, with no padding between elements.
- `x` has exactly n elements and `out` has exactly n elements. Do not read or write outside these ranges.
- The start addresses of `x` and `out` are 64-byte aligned. The two buffers do not overlap.
- n is a positive integer and always agrees with the actual sizes of `x` and `out`.

## Contract

- The function has exactly three parameters, in this order: x, out, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- out is allocated by the caller. Its initial contents are unspecified (arbitrary values, not zero) and must not be used: the function must overwrite every element of out with the correct result. It must not add to, or otherwise combine with, whatever out contained before the call, and it must not assume that out starts zeroed.
- The result must be written into out. The return value of the function is ignored: results are read only from the buffer out, so returning a new array instead of filling out counts as not writing out.
- `x` is read-only. The function must not modify it, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce identical results.
- The function must be correct for every positive n, including n = 1, very small n, and n that is not a multiple of 2, 4, 8, 16, the number of threads or any other block size.
- Input values are non-negative and small: every element of x is an integer in the range 0 to 31 inclusive (in the evaluation they are drawn uniformly from that range). Every prefix sum is therefore at most 31 * n, which fits in a signed 32-bit integer for all evaluation shapes (31 * 33,554,432 < 2^31). The function may rely on this: no overflow handling is needed and 32-bit accumulation is sufficient; using wider intermediate accumulators is a free choice.

## Accuracy

The output is compared element by element against a reference computed from the same inputs with 64-bit integer arithmetic. The check requires exact equality:

    out[i] == ref[i]    for every 0 <= i < n

There is no tolerance: an element that differs from the reference by any amount fails the check. Within this requirement, the algorithm, the partitioning of the work and the order in which elements are added are free choices.

## Evaluation

- Correctness shapes (n): 7, 100,003 and 1,048,576.
- Performance shape (n): 33,554,432. The output for the performance shape must also pass the exactness check.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with out pre-filled with arbitrary values before each call. Each output must pass the exactness check, and the two outputs must be identical. Writes outside the buffers and modifications of the input are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
