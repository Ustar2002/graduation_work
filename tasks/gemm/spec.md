# Task: single-precision general matrix multiplication (GEMM)

## Definition

Implement a function named `gemm` that computes the matrix product C = A · B:

    C[i][j] = sum over k from 0 to K-1 of A[i][k] * B[k][j],    for 0 <= i < M and 0 <= j < N

A has M rows and K columns, B has K rows and N columns, and C has M rows and N columns.

## Data layout

- All three matrices are arrays of IEEE-754 single-precision (32-bit) floating-point numbers (float32).
- Storage is row-major and contiguous, with no padding between rows: A[i][k] is the element at linear index i*K + k of A, B[k][j] is at linear index k*N + j of B, and C[i][j] is at linear index i*N + j of C.
- A contains exactly M*K elements, B exactly K*N elements, and C exactly M*N elements. Do not read or write outside these ranges.
- The start addresses of A, B and C are 64-byte aligned. The three buffers do not overlap.
- M, N and K are positive integers and always agree with the actual sizes of the buffers.

## Contract

- The function has exactly six parameters, in this order: A, B, C, M, N, K. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- C is allocated by the caller. Its initial contents are unspecified (arbitrary finite values) and must not be used: the function must overwrite every element of C with the correct result. It must not add to, or otherwise combine with, whatever C contained before the call.
- The result must be written into C. The return value of the function is ignored: results are read only from the buffer C, so returning a new array instead of filling C counts as not writing C.
- A and B are read-only. The function must not modify them, not even temporarily.
- The function must be correct for every positive M, N and K, including non-square shapes, very small shapes, and dimensions that are not multiples of 2, 4, 8, 16 or any other block size.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce results that agree within the tolerance below.
- Input values are finite float32 numbers of moderate magnitude (in the evaluation they are drawn uniformly from the interval [-1, 1)). No special handling of NaN, infinity or subnormal values is required, but the output must not contain NaN or infinity.

## Accuracy

The output is compared element by element against a reference computed from the same inputs with 64-bit (double-precision) accumulation. The check passes if, for every element,

    |C[i][j] - ref[i][j]| <= atol + rtol * |ref[i][j]|,    with rtol = 1e-3 and atol = 1e-3.

Any NaN or infinity in C fails the check. Within this tolerance, the order of accumulation, any blocking or tiling, and the precision used for intermediate values are free choices.

## Evaluation

- Correctness shapes (M, N, K): (7, 5, 3), (200, 130, 312) and (256, 256, 256).
- Performance shape (M, N, K): (1024, 1024, 1024).
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with C pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of A or B are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
