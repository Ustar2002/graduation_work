# Task: all-pairs shortest path lengths in a dense directed graph

## Definition

Implement a function named `floyd_warshall` that computes the length of a shortest path between every ordered pair of vertices of a weighted directed graph with n vertices, numbered 0 to n-1.

The graph is complete: for every ordered pair (i, j) with i != j there is an edge from vertex i to vertex j, and its weight W[i][j] is positive. The graph is directed: the edge from i to j and the edge from j to i are different edges and in general have different weights.

A path from i to j is a sequence of vertices v(0) = i, v(1), ..., v(m) = j with m >= 1, and its length is the sum of the weights of its m edges, W[v(0)][v(1)] + W[v(1)][v(2)] + ... + W[v(m-1)][v(m)]. The output `D` is defined by

    D[i][j] = minimum, over all paths from i to j, of the length of the path,    for 0 <= i < n, 0 <= j < n and i != j

    D[i][i] = 0    for 0 <= i < n

A path may pass through any number of intermediate vertices, in any order. The single edge from i to j is one of these paths, so D[i][j] <= W[i][j]. Because all weights are positive, the minimum exists and is attained by a path that visits no vertex twice.

## Data layout

- Both matrices are arrays of IEEE-754 single-precision (32-bit) floating-point numbers (float32).
- Storage is row-major and contiguous, with no padding between rows: the element in row i and column j is the element at linear index i*n + j. Row i of `W` holds the weights of the edges that leave vertex i, and row i of `D` holds the lengths of the shortest paths that start at vertex i.
- `W` and `D` each contain exactly n*n elements. Do not read or write outside these ranges.
- The diagonal elements W[i][i] are 0. All other elements of `W` are positive and finite; there are no missing edges and no special values that stand for a missing edge. `W` is in general not symmetric.
- The start addresses of `W` and `D` are 64-byte aligned. The two buffers do not overlap.
- n is a positive integer and always agrees with the actual sizes of the buffers.

## Contract

- The function has exactly three parameters, in this order: W, D, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- D is allocated by the caller. Its initial contents are unspecified (arbitrary finite values) and must not be used: the function must overwrite every element of D with the correct result. It must not add to, or otherwise combine with, whatever D contained before the call.
- The result must be written into D. The return value of the function is ignored: results are read only from the buffer D, so returning a new array instead of filling D counts as not writing D.
- `W` is read-only. The function must not modify it, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce results that agree within the tolerance below.
- The function must be correct for every positive n, including very small graphs and n that is not a multiple of 2, 4, 8, 16 or any other block size.
- Input values are finite float32 numbers (in the evaluation the weights W[i][j] with i != j are drawn uniformly from the interval [1, 100), so every D[i][j] with i != j is at least 1 and less than 100). No special handling of NaN, infinity or subnormal values is required, but the output must not contain NaN or infinity.

## Accuracy

The output is compared element by element against a reference computed from the same input with 64-bit (double-precision) arithmetic. The check passes if, for every element,

    |D[i][j] - ref[i][j]| <= atol + rtol * |ref[i][j]|,    with rtol = 1e-4 and atol = 1e-4.

Any NaN or infinity in `D` fails the check. Within this tolerance, the order in which the edge weights of a path are added and the precision used for intermediate values are free choices.

## Evaluation

- Correctness shapes (n): 5, 97 and 256.
- Performance shape (n): 1024. The output for the performance shape must also pass the accuracy check.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with D pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
