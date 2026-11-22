# Task: sorting an array of 32-bit integers

## Definition

Implement a function named `sort` that arranges n signed 32-bit integers in ascending order. The output `out` must contain exactly the same values as the input `keys`, each value as many times as it occurs in `keys`, in non-decreasing order:

    out[0] <= out[1] <= ... <= out[n-1]

The order is the ordinary order of signed integers, so every negative value comes before every non-negative value. Values may repeat, and values may be negative. The keys carry no other data, so equal keys are indistinguishable and the result is uniquely determined by the input.

## Data layout

- `keys`: n signed 32-bit integers (int32) in arbitrary order. Any int32 value may occur, from -2,147,483,648 to 2,147,483,647.
- `out`: n signed 32-bit integers (int32).
- Both arrays are contiguous. Do not read or write outside the stated ranges.
- The start addresses of `keys` and `out` are 64-byte aligned. The two buffers do not overlap.
- n is a positive integer and always agrees with the actual sizes of the buffers.

## Contract

- The function has exactly three parameters, in this order: keys, out, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- out is allocated by the caller. Its initial contents are unspecified (arbitrary values) and must not be used: the function must overwrite every element of out with the correct result. It must not add to, or otherwise combine with, whatever out contained before the call.
- The result must be written into out. The return value of the function is ignored: results are read only from the buffer out, so returning a new array instead of filling out counts as not writing out.
- `keys` is read-only. The function must not modify it, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce identical results.
- The function must be correct for every n >= 1 and every array of n int32 values, including very small n, n that is not a multiple of 2, 4, 8, 16 or any other block size, negative values, values that occur many times, and the smallest and the largest int32 value.

## Accuracy

The output is compared element by element against a reference computed from the same input (the values of `keys` in ascending order). The comparison is exact; there is no tolerance. The check passes only if

    out[i] == ref[i]    for every i with 0 <= i < n.

## Evaluation

- Correctness shapes (n): 9, 100,003 and 65,536.
- Performance shape (n): 16,777,216. The output for the performance shape must also pass the accuracy check.
- Key values: the smallest correctness shape contains negative values, repeated values and both the smallest and the largest int32 value. In the correctness shape with n = 100,003 every key is drawn independently and uniformly at random from the 2,000 integers from -1,000 to 999, so each value occurs about 50 times. In the correctness shape with n = 65,536 and in the performance shape the keys are drawn independently and uniformly at random from the whole int32 range.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with out pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must be identical. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
