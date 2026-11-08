# Task: sorting 32-bit integers

## Definition

Implement a function named `sort` that writes the n integers of the input array `x` into the output array `out` in ascending (non-decreasing) order:

    out[0] <= out[1] <= ... <= out[n-1],

where `out` is a rearrangement of `x`: every value occurs in `out` exactly as many times as it occurs in `x`.

## Data layout

- `x` is an array of n signed 32-bit integers (int32), contiguous.
- `out` is an array of n int32 values, contiguous.
- Do not read or write outside these ranges.
- The start addresses of `x` and `out` are 64-byte aligned. The two buffers do not overlap.
- n is a positive integer and always agrees with the actual sizes of `x` and `out`.

## Contract

- The function has exactly three parameters, in this order: x, out, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- out is allocated by the caller. Its initial contents are unspecified (arbitrary values, not zero) and must not be used: the function must overwrite every element of out with the correct result. It must not assume that out is zero-initialized, and it must not combine the result with whatever out contained before the call.
- The result must be written into out. The return value of the function is ignored: results are read only from the buffer out, so returning a new array instead of filling out counts as not writing out.
- `x` is read-only. The function must not modify it, not even temporarily: in particular, it must not sort x in place, not even if the original order would be restored afterwards. Sort a copy, or write the result directly into out.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce identical results.
- The function must be correct for every positive n, including very small n (such as 1), n that is not a power of two or a multiple of 2, 4, 8, 16 or any other block size, and inputs that contain repeated values.
- Input values are non-negative: every element satisfies 0 <= x[i] <= 2^31 - 1, so the whole non-negative range of int32 can occur, but negative values never occur. In the evaluation the values are drawn independently and uniformly from this range, so repeated values are possible and do occur in the larger shapes.

## Accuracy

The output is compared element by element against a reference: the same input sorted in ascending order. The check passes only if every element matches exactly,

    out[i] == ref[i],    for all 0 <= i < n.

There is no tolerance: a single differing element fails the check. The ascending arrangement of a multiset of integers is unique, so every correct algorithm, sequential or parallel, stable or not, produces the same output.

## Evaluation

- Correctness shapes (n): 7, 100,003 and 1,048,576.
- Performance shape (n): 10,000,000. The output for the performance shape must also pass the check.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with out pre-filled with arbitrary values before each call. Each output must pass the check, and the two outputs must be identical. Writes outside the buffers and modifications of x are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The input is already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
