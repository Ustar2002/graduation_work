# Task: histogram of integer keys

## Definition

Implement a function named `histogram` that counts, for each of n_bins bins, how many of the n input keys refer to it. Every key is a bin index in the range 0 <= keys[i] < n_bins, and the output `hist` receives one count per bin:

    hist[b] = number of indices i with 0 <= i < n and keys[i] == b,    for 0 <= b < n_bins

A bin that no key refers to has the count 0. The n_bins counts add up to n.

## Data layout

- `keys`: n signed 32-bit integers (int32), each in the range 0 <= keys[i] < n_bins.
- `hist`: n_bins signed 32-bit integers (int32). A count never exceeds n, so every count fits in this type.
- Both arrays are contiguous. Do not read or write outside the stated ranges.
- The start addresses of `keys` and `hist` are 64-byte aligned. The two buffers do not overlap.
- n and n_bins are positive integers (n >= 1, n_bins >= 1) and always agree with the actual sizes of the buffers.

## Contract

- The function has exactly four parameters, in this order: keys, hist, n, n_bins. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- hist is allocated by the caller. Its initial contents are unspecified (arbitrary values) and must not be used: the function must overwrite every element of hist with the correct result. It must not add to, or otherwise combine with, whatever hist contained before the call.
- The result must be written into hist. The return value of the function is ignored: results are read only from the buffer hist, so returning a new array instead of filling hist counts as not writing hist.
- `keys` is read-only. The function must not modify it, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce identical results.
- The function must be correct for every n >= 1, every n_bins >= 1 and every valid array of keys (any values in the stated range, in any order), including very small inputs, bins that receive no key, values of n_bins that are not powers of two, and values of n that are not multiples of 2, 4, 8, 16 or any other block size.
- Every key is a valid bin index: negative keys and keys greater than or equal to n_bins do not occur, and no range check is required.

## Accuracy

The output is compared element by element against a reference computed from the same input with exact integer arithmetic. The comparison is exact; there is no tolerance. The check passes only if

    hist[b] == ref[b]    for every b with 0 <= b < n_bins.

## Evaluation

- Correctness shapes (n, n_bins): (10, 4), (100003, 1000) and (1048576, 256).
- Performance shape (n, n_bins): (67108864, 256). The output for the performance shape must also pass the accuracy check.
- Except in the smallest correctness shape, where at least one bin receives no key, the keys are uniformly distributed over the bins: each key is drawn independently and uniformly at random from 0, 1, ..., n_bins-1.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with hist pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must be identical. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
