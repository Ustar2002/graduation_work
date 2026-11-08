# Task: Cholesky factorization of a symmetric positive definite matrix (single precision)

## Definition

Implement a function named `cholesky` that computes the Cholesky factor of a symmetric positive definite n-by-n matrix A: the unique lower-triangular matrix L with strictly positive diagonal entries such that

    L · Lᵀ = A,    that is,    A[i][j] = sum over k from 0 to min(i, j) of L[i][k] * L[j][k]    for all 0 <= i, j < n.

The result is written into the n-by-n output matrix `L`:

- for j <= i (diagonal and strictly lower triangle): L[i][j] is the Cholesky factor entry, with L[i][i] > 0;
- for j > i (strictly upper triangle): L[i][j] must be 0.

Equivalently, in terms of the entries, for every column j from 0 to n-1:

    L[j][j] = sqrt( A[j][j] - sum over k < j of L[j][k]^2 )
    L[i][j] = ( A[i][j] - sum over k < j of L[i][k] * L[j][k] ) / L[j][j]    for i > j

The order in which the entries are computed is a free choice, as long as the result is the factor defined above.

## Data layout

- `A` holds the input matrix as n * n IEEE-754 single-precision (32-bit) floating-point numbers (float32), contiguous, in row-major order: A[i][j] is the element at linear index i*n + j. Both triangles of A are stored and are equal (A is exactly symmetric: the stored value at i*n + j is bit-for-bit the stored value at j*n + i).
- `L` holds the output matrix as n * n contiguous float32 values in the same row-major layout: L[i][j] is the element at linear index i*n + j. All n * n elements belong to the output, including the strictly upper triangle.
- Do not read or write outside these ranges.
- The start addresses of `A` and `L` are 64-byte aligned. The two buffers do not overlap.
- n is a positive integer and always agrees with the actual sizes of the buffers.

## Guarantees on the input

- A is symmetric and positive definite, so the factorization exists and every square root above has a strictly positive argument and every division has a non-zero divisor, in exact arithmetic and with a wide margin in single precision.
- A is well conditioned. In the evaluation, A is generated as B·Bᵀ + n·I where B is an n-by-n matrix with entries drawn uniformly from [-1, 1), computed in double precision and then rounded to float32. Consequently every eigenvalue of A lies between about n and 2.4·n (condition number about 2.3), each diagonal entry is close to 4n/3, the off-diagonal entries have magnitudes of the order of sqrt(n/3), and the diagonal entries of L are close to sqrt(4n/3) (about 52 for n = 2,048). No pivoting, scaling or regularization is needed.
- Input values are finite float32 numbers. No special handling of NaN, infinity or subnormal values is required, but the output must not contain NaN or infinity.

## Contract

- The function has exactly three parameters, in this order: A, L, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- L is allocated by the caller. Its initial contents are unspecified (arbitrary finite values) and must not be used: the function must overwrite every element of L with the correct result, including the strictly upper triangle, which must be set to 0. It must not add to, or otherwise combine with, whatever L contained before the call.
- The result must be written into L. The return value of the function is ignored: results are read only from the buffer L, so returning a new array instead of filling L counts as not writing L.
- `A` is read-only. The function must not modify it, not even temporarily (for example, the factorization must not be performed in place in A).
- Temporary memory (working buffers, including a working copy of A) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce results that agree within the tolerance below.
- The function must be correct for every positive n, including n = 1 (L[0][0] = sqrt(A[0][0])), very small n, and n that is not a multiple of 2, 4, 8, 16 or any other block size.

## Accuracy

The output is compared element by element against a reference factor computed from the same float32 input matrix in 64-bit (double-precision) arithmetic. The check passes if, for every element of the n-by-n matrix (strictly upper triangle included),

    |L[i][j] - ref[i][j]| <= atol + rtol * |ref[i][j]|,    with rtol = 1e-3 and atol = 1e-3 * RMS(ref),

where RMS(ref) is the root mean square of all n * n entries of the reference factor of the shape being checked (zeros of the upper triangle included; for the matrices described above RMS(ref) is about 1.15 for every n, so atol is about 1.15e-3). Since the reference is exactly 0 in the strictly upper triangle, an entry there passes only if its magnitude is at most atol. Any NaN or infinity in `L` fails the check. Within this tolerance, the order of operations, the precision used for intermediate values and the way square roots and divisions are evaluated are free choices.

## Evaluation

- Correctness shapes (n): 6, 131 and 512.
- Performance shape (n): 2,048. The output for the performance shape must also pass the accuracy check.
- Memory: the process that calls the function is limited to 8 GiB of address space, inputs and outputs included. An allocation beyond that limit fails the call and counts as a failure. A few n-by-n working matrices fit comfortably at n = 2,048 (16 MiB each in float32).
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with L pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of the input are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
