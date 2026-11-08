# Task: breadth-first search distances from vertex 0 (directed graph in CSR form)

## Definition

Implement a function named `bfs` that computes, for every vertex v of a directed graph with n vertices, the length in edges of a shortest directed path from vertex 0 to v, and stores it in the output array `dist`:

    dist[0] = 0
    dist[v] = smallest number of edges on a directed path from vertex 0 to v,    if such a path exists
    dist[v] = -1,                                                                 if no such path exists

Edges are directed: an edge (u, v) can be followed from u to v only. Edges have no weights; the length of a path is the number of edges on it. Since all edges have the same length, the distances are the levels of a breadth-first traversal starting at vertex 0: vertex 0 is at level 0, its out-neighbours at level 1, their not yet reached out-neighbours at level 2, and so on.

## Data layout

The graph is given in compressed sparse row (CSR) form, one row per vertex:

- `row_ptr`: n + 1 signed 32-bit integers (int32), non-decreasing, with row_ptr[0] = 0 and row_ptr[n] = nnz.
- `col_idx`: nnz signed 32-bit integers (int32), each in the range 0 <= col_idx[p] < n. The out-neighbours of vertex i are col_idx[row_ptr[i]], col_idx[row_ptr[i] + 1], ..., col_idx[row_ptr[i+1] - 1], that is, the edges (i, col_idx[p]) for row_ptr[i] <= p < row_ptr[i+1]. Within one row the entries are strictly increasing, so a row contains no duplicate edge. A row may be empty (row_ptr[i] == row_ptr[i+1]): the vertex has no out-edges.
- A row may contain its own index (a self loop i -> i). Self loops are valid input; they never change any distance and need no special handling.
- The graph is in general not symmetric: an edge (i, j) does not imply an edge (j, i).
- `dist`: n signed 32-bit integers (int32).
- All arrays are contiguous. Do not read or write outside the stated ranges.
- The start addresses of all three arrays are 64-byte aligned. The buffers do not overlap.
- n and nnz are integers with n >= 1 and nnz >= 0, and always agree with the actual sizes of the buffers. Every distance is smaller than n, so every value of `dist` fits in a signed 32-bit integer.

## Contract

- The function has exactly five parameters, in this order: row_ptr, col_idx, dist, n, nnz. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- dist is allocated by the caller. Its initial contents are arbitrary (unspecified finite values; they are neither zero nor -1) and must not be used: the function must write every element of dist, including dist[0] = 0 and the value -1 for every unreachable vertex. The function must not add to, or otherwise combine with, whatever dist contained before the call.
- The result must be written into dist. The return value of the function is ignored: results are read only from the buffer dist, so returning a new array instead of filling dist counts as not writing dist.
- `row_ptr` and `col_idx` are read-only. The function must not modify them, not even temporarily.
- Temporary memory (working buffers such as queues, frontiers or visited marks) may be allocated inside the function, provided it is released before the function returns.
- Memory: the process that calls the function is limited to 8 GiB of address space, inputs and outputs included. An allocation beyond that limit fails the call and counts as a failure. A dense n-by-n adjacency matrix does not fit at n = 2,097,152.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce identical results.
- The function must be correct for every valid CSR input, including n = 1, graphs with no edges at all (nnz = 0), empty rows, vertex 0 having no out-edges, vertices that cannot be reached from vertex 0, self loops, and sizes that are not multiples of 2, 4, 8, 16 or any other block size.

## Accuracy

Distances are exact integers, so there is no tolerance. The output is compared element by element against reference distances computed from the same input, and the check passes only if

    dist[v] == ref[v]    for every 0 <= v < n,

where ref[v] is -1 for every vertex that is not reachable from vertex 0. A single element that differs, including an unreachable vertex that is not set to -1, fails the check.

## Evaluation

- Correctness shapes (n, nnz): (7, 13), (1,003, 15,833) and (65,536, 1,045,504). The first graph has out-degrees between 0 and 2 and contains vertices that are not reachable from vertex 0; in the other two every vertex has between 0 and 32 out-edges (about 16 on average), whose targets are spread over the whole vertex range.
- Performance shape (n, nnz): (2,097,152, 33,541,983). Out-degrees are between 0 and 32 (about 16 on average), targets are spread over the whole vertex range, and the traversal from vertex 0 reaches almost every vertex within a few levels. The output for the performance shape must also pass the exactness check.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with dist pre-filled with arbitrary values before each call. Each output must equal the reference exactly, and the two outputs must be identical to each other. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
