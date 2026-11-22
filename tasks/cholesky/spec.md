# Task: Cholesky factorization of a symmetric positive definite matrix

## Definition

Implement a function named `cholesky` that computes the Cholesky factor of a symmetric positive definite matrix A with n rows and n columns: the lower-triangular matrix L with positive diagonal elements such that A = L · L^T, where L^T is the transpose of L. Element by element, L satisfies

    A[i][j] = sum over k from 0 to min(i, j) of L[i][k] * L[j][k],    for 0 <= i < n and 0 <= j < n

    L[i][i] > 0    for 0 <= i < n

    L[i][j] = 0    for 0 <= i < j < n (the elements above the diagonal)

For a symmetric positive definite matrix this factor exists and is unique, so these conditions determine every element of L. The output `L` must contain the complete matrix L: the elements on and below the diagonal, and the zeros above the diagonal.

## Data layout

- Both matrices are arrays of IEEE-754 single-precision (32-bit) floating-point numbers (float32).
- Storage is row-major and contiguous, with no padding between rows: the element in row i and column j is the element at linear index i*n + j.
- `A` and `L` each contain exactly n*n elements. Do not read or write outside these ranges.
- `A` is stored in full: all n*n elements are present and valid, and A[i][j] and A[j][i] hold exactly the same value for every i and j. Every element of `A` may be read; whether an implementation reads the lower triangle, the upper triangle or both is a free choice.
- The start addresses of `A` and `L` are 64-byte aligned. The two buffers do not overlap.
- n is a positive integer and always agrees with the actual sizes of the buffers.

## Contract

- The function has exactly three parameters, in this order: A, L, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- L is allocated by the caller. Its initial contents are unspecified (arbitrary finite values) and must not be used: the function must overwrite every element of L with the correct result. It must not add to, or otherwise combine with, whatever L contained before the call. This includes the elements above the diagonal: they do not contain 0 when the function is called, and the function must write 0 into each of them.
- The result must be written into L. The return value of the function is ignored: results are read only from the buffer L, so returning a new array instead of filling L counts as not writing L.
- `A` is read-only. The function must not modify it, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce results that agree within the tolerance below.
- The function must be correct for every positive n, including very small matrices and n that is not a multiple of 2, 4, 8, 16 or any other block size.
- Input values are finite float32 numbers of moderate magnitude, and `A` is always symmetric and positive definite. In the evaluation A = G · G^T / n + I, where G is an n-by-n matrix whose elements are drawn uniformly from the interval [-1, 1) and I is the identity matrix; A is computed in double precision and then rounded to float32. Its diagonal elements lie between 1 and 2, and the ratio of its largest to its smallest eigenvalue is below 3. No special handling of NaN, infinity or subnormal values is required, but the output must not contain NaN or infinity.

## Accuracy

The output is compared element by element against a reference computed from the same input with 64-bit (double-precision) arithmetic. The check covers all n*n elements, including the elements above the diagonal, whose reference value is 0. It passes if, for every element,

    |L[i][j] - ref[i][j]| <= atol + rtol * |ref[i][j]|,    with rtol = 1e-4 and atol = 1e-5.

Any NaN or infinity in `L` fails the check. Within this tolerance, the order of the arithmetic operations and the precision used for intermediate values are free choices.

## Evaluation

- Correctness shapes (n): 4, 97 and 256.
- Performance shape (n): 2048. The output for the performance shape must also pass the accuracy check.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with L pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
