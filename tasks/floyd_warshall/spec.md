# Task: all-pairs shortest paths (Floyd-Warshall) on a dense directed graph

## Definition

Implement a function named `floyd_warshall` that computes the length of the shortest path between every ordered pair of vertices of a directed graph with n vertices, given as a dense n-by-n distance matrix.

The input matrix `dist_in` describes the edges: dist_in[i][j] is the weight of the directed edge from vertex i to vertex j, dist_in[i][i] = 0 for every i, and the special constant

    INF = 1.0e30

in place of a weight means that there is no edge from i to j. The output matrix `dist_out` must satisfy, for every pair (i, j),

    dist_out[i][j] = the minimum, over all directed paths from i to j, of the sum of the edge weights along the path

with dist_out[i][i] = 0 (the empty path) and dist_out[i][j] = INF, exactly the constant 1.0e30, whenever no directed path from i to j exists. A path may use any number of intermediate vertices. All edge weights are strictly positive, so there are no negative cycles and every shortest path is simple.

Equivalently, dist_out is the result of the following relaxation applied to a copy d of dist_in, in this order of k:

    for k = 0, ..., n-1:   d[i][j] = min( d[i][j], d[i][k] + d[k][j] )   for all i, j

Any algorithm that produces the same matrix is acceptable; the recurrence above is the definition, not a required implementation.

## Data layout

- Both matrices are arrays of IEEE-754 single-precision (32-bit) floating-point numbers (float32).
- Storage is row-major and contiguous, with no padding between rows: the element in row i and column j is at linear index i*n + j.
- `dist_in` and `dist_out` each contain exactly n*n elements. Do not read or write outside these ranges.
- The start addresses of `dist_in` and `dist_out` are 64-byte aligned. The two buffers do not overlap.
- n is a positive integer and always agrees with the actual dimensions of the buffers.

## Contract

- The function has exactly three parameters, in this order: dist_in, dist_out, n. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- dist_out is allocated by the caller. Its initial contents are unspecified (arbitrary finite values) and must not be used: the function must overwrite every element of dist_out with the correct result, including the diagonal and the INF entries of unreachable pairs. It must not add to, or otherwise combine with, whatever dist_out contained before the call.
- The result must be written into dist_out. The return value of the function is ignored: results are read only from the buffer dist_out, so returning a new array instead of filling dist_out counts as not writing dist_out.
- `dist_in` is read-only. The function must not modify it, not even temporarily. Copying dist_in into dist_out first and then relaxing dist_out in place is permitted.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce results that agree within the tolerance below.
- The function must be correct for every positive n, including n = 1, very small n, and n that is not a multiple of 2, 4, 8, 16, the thread count or any other block size.
- Input values: the diagonal is exactly 0; every off-diagonal element is either a finite edge weight in the interval [1, 10) or exactly INF = 1.0e30. In the evaluation each off-diagonal element is an edge with probability 0.3, independently, with its weight drawn uniformly from [1, 10). All finite shortest-path lengths are therefore positive and small (at most a few tens), and the only large value that can appear is INF itself. Note that INF + INF = 2.0e30 is a finite float32 value and INF + w = INF for every weight w in the graph, so a plain minimum over candidate sums keeps unreachable pairs at exactly INF without any special handling; implementations that treat INF differently must still produce exactly 1.0e30 for unreachable pairs. The output must not contain NaN or infinity.

## Accuracy

The output is compared element by element against a reference computed from the same input with 64-bit (double-precision) arithmetic. The check passes if, for every element,

    |dist_out[i][j] - ref[i][j]| <= atol + rtol * |ref[i][j]|,    with rtol = 1e-4 and atol = 1e-3.

Any NaN or infinity in `dist_out` fails the check. Within this tolerance, the order in which candidate paths are compared and the precision used for intermediate sums are free choices; when two different paths have lengths that agree within the tolerance, either length is accepted. For unreachable pairs the reference holds exactly INF, so the output must hold INF (a value such as 2.0e30 fails).

## Evaluation

- Correctness shapes (n): 8, 37 and 256.
- Performance shape (n): 1,024. The output for the performance shape must also pass the accuracy check.
- Memory: the process that calls the function is limited to 8 GiB of address space, inputs and outputs included. An allocation beyond that limit fails the call and counts as a failure.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with dist_out pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must agree with each other within the tolerance. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
