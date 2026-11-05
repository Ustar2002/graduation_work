# Task: sparse matrix-vector product in CSR format (SpMV)

## Definition

Implement a function named `spmv_csr` that computes y = A · x, where A is a sparse matrix with n_rows rows and n_cols columns stored in compressed sparse row (CSR) format, x is a dense vector of length n_cols and y is a dense vector of length n_rows:

    y[i] = sum over p from row_ptr[i] to row_ptr[i+1]-1 of vals[p] * x[col_idx[p]],    for 0 <= i < n_rows

A row with row_ptr[i] == row_ptr[i+1] has no stored entries, and its result is y[i] = 0.

## Data layout

- `row_ptr`: n_rows + 1 signed 32-bit integers, non-decreasing, with row_ptr[0] = 0 and row_ptr[n_rows] = nnz.
- `col_idx`: nnz signed 32-bit integers, each in the range 0 <= col_idx[p] < n_cols. Within each row the column indices are strictly increasing (no duplicates).
- `vals`: nnz IEEE-754 single-precision (32-bit) floating-point numbers (float32); vals[p] is the matrix entry in the column col_idx[p].
- `x`: n_cols float32 values. `y`: n_rows float32 values.
- All arrays are contiguous. Do not read or write outside the stated ranges.
- The start addresses of all five arrays are 64-byte aligned. The buffers do not overlap.
- n_rows, n_cols and nnz are integers with n_rows >= 1, n_cols >= 1 and nnz >= 0, and always agree with the actual sizes of the buffers. Rows may have different numbers of entries, including zero.

## Contract

- The function has exactly eight parameters, in this order: row_ptr, col_idx, vals, x, y, n_rows, n_cols, nnz. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- y is allocated by the caller. Its initial contents are unspecified (arbitrary finite values) and must not be used: the function must overwrite every element of y with the correct result. It must not add to, or otherwise combine with, whatever y contained before the call.
- The result must be written into y. The return value of the function is ignored: results are read only from the buffer y, so returning a new array instead of filling y counts as not writing y.
- `row_ptr`, `col_idx`, `vals` and `x` are read-only. The function must not modify them, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce results that agree within the tolerance below.
- The function must be correct for every valid CSR input, including non-square matrices, very small matrices, empty rows, and sizes that are not multiples of 2, 4, 8, 16 or any other block size.
- Input values are finite float32 numbers of moderate magnitude (in the evaluation `vals` and `x` are drawn uniformly from the interval [-1, 1)). No special handling of NaN, infinity or subnormal values is required, but the output must not contain NaN or infinity.

## Accuracy

The output is compared element by element against a reference computed from the same inputs with 64-bit (double-precision) accumulation. The check passes if, for every element,

    |y[i] - ref[i]| <= atol + rtol * |ref[i]|,    with rtol = 1e-4 and atol = 1e-4.

Any NaN or infinity in `y` fails the check. Within this tolerance, the order of accumulation and the precision used for intermediate values are free choices.

## Evaluation

- Correctness shapes (n_rows, n_cols, nnz): (5, 7, about 10), (300, 517, about 1,800) and (10,000, 10,000, about 100,000).
- Performance shape (n_rows, n_cols, nnz): (1,000,000, 1,000,000, about 16,000,000). The number of entries per row varies between 0 and 32, and the column indices of a row are spread over the whole column range.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with y pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
