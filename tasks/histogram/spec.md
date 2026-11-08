# Task: integer histogram (exact bin counts)

## Definition

Implement a function named `histogram` that counts, for every bin b, how many elements of the integer array `data` are equal to b, and stores the counts in the output array `hist`:

    hist[b] = number of indices i with 0 <= i < n and data[i] == b,    for 0 <= b < nbins

Every element of `data` falls into exactly one bin, so the counts sum to n. A bin that no element falls into has the count 0.

## Data layout

- `data` is an array of n signed 32-bit integers (int32), contiguous. Every value satisfies 0 <= data[i] < nbins. This is guaranteed by the caller: no element lies outside the bin range, so no range check is required.
- `hist` is an array of exactly nbins signed 32-bit integers (int32), contiguous.
- Do not read or write outside these ranges.
- The start addresses of `data` and `hist` are 64-byte aligned. The two buffers do not overlap.
- n and nbins are positive integers and always agree with the actual sizes of `data` and `hist`. n fits in a signed 32-bit integer, so every count fits in a signed 32-bit integer as well.

## Contract

- The function has exactly four parameters, in this order: data, hist, n, nbins. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- hist is allocated by the caller. Its initial contents are arbitrary (unspecified finite values; they are not zero) and must not be used: the function must write every element of hist, including the bins whose count is 0. Setting the bins to zero before counting is the function's job. The function must not add to, or otherwise combine with, whatever hist contained before the call.
- The result must be written into hist. The return value of the function is ignored: results are read only from the buffer hist, so returning a new array instead of filling hist counts as not writing hist.
- `data` is read-only. The function must not modify it, not even temporarily (for example, it must not sort `data` in place).
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce identical results.
- The function must be correct for every positive n and nbins, including nbins = 1, n smaller than nbins, and values of n or nbins that are not multiples of 2, 4, 8, 16 or any other block size.
- In the evaluation the values of `data` are drawn independently and uniformly from the integers 0, 1, ..., nbins - 1, so every bin receives about n / nbins elements and consecutive elements are unrelated.

## Accuracy

Counting is exact integer arithmetic, so there is no tolerance. The output is compared element by element against reference counts computed from the same input, and the check passes only if

    hist[b] == ref[b]    for every 0 <= b < nbins.

A bin that differs from the reference by any amount, including a count that is off by one, fails the check.

## Evaluation

- Correctness shapes (n, nbins): (13, 4), (100,003, 37) and (1,048,576, 256).
- Performance shape (n, nbins): (33,554,432, 1024). The output for the performance shape must also pass the exactness check.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with hist pre-filled with arbitrary non-zero values before each call. Each output must equal the reference exactly, and the two outputs must be identical to each other. Writes outside the buffers and modifications of the input are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
