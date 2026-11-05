# Task: single-precision dot product

## Definition

Implement a function named `dot` that computes the dot product of two vectors x and y of length n and stores it in the first (and only) element of the output buffer `out`:

    out[0] = sum over i from 0 to n-1 of x[i] * y[i]

## Data layout

- `x` and `y` are arrays of n IEEE-754 single-precision (32-bit) floating-point numbers (float32), contiguous.
- `out` is an array of exactly one float32 value.
- Do not read or write outside these ranges.
- The start addresses of `x`, `y` and `out` are 64-byte aligned. The three buffers do not overlap.
- n is a positive integer and always agrees with the actual sizes of `x` and `y`.

## Contract

- The function has exactly four parameters, in this order: x, y, out, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- out is allocated by the caller. Its initial contents are unspecified (arbitrary finite values) and must not be used: the function must overwrite every element of out with the correct result. It must not add to, or otherwise combine with, whatever out contained before the call.
- The result must be written into out. The return value of the function is ignored: results are read only from the buffer out, so returning a new array instead of filling out counts as not writing out.
- `x` and `y` are read-only. The function must not modify them, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce results that agree within the tolerance below.
- The function must be correct for every positive n, including very small n and n that is not a multiple of 2, 4, 8, 16 or any other block size.
- Input values are finite float32 numbers (in the evaluation they are drawn uniformly from the interval [0, 1), so every product is non-negative and the sum grows to about n/4). No special handling of NaN, infinity or subnormal values is required, but the output must not be NaN or infinity.

## Accuracy

The output is compared against a reference computed from the same inputs with 64-bit (double-precision) accumulation. The check passes if

    |out[0] - ref| <= atol + rtol * |ref|,    with rtol = 1e-3 and atol = 1e-3.

NaN or infinity in `out` fails the check. Within this tolerance, the order of accumulation and the precision used for intermediate values are free choices.

## Evaluation

- Correctness shapes (n): 7, 100,003 and 1,048,576.
- Performance shape (n): 33,554,432. The output for the performance shape must also pass the accuracy check.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with out pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
