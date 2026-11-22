# Task: single-source distances (edge counts) in a directed graph in CSR format

## Definition

Implement a function named `bfs` that computes, for every vertex of a directed graph, the smallest number of edges on a path from a given start vertex `source` to that vertex. The graph has n_vertices vertices, numbered 0 to n_vertices-1, and n_edges edges.

The graph is stored in compressed sparse row (CSR) format as one list of out-neighbours per vertex. The edges that leave vertex u lead to the vertices

    col_idx[p]    for row_ptr[u] <= p < row_ptr[u+1]

so vertex u has row_ptr[u+1] - row_ptr[u] outgoing edges. A vertex with row_ptr[u] == row_ptr[u+1] has no outgoing edges. The graph is directed: an edge from u to v can be followed only from u to v, and it does not imply an edge from v to u.

A path from source to a vertex v is a sequence of vertices w(0) = source, w(1), ..., w(m) = v with m >= 0 such that there is an edge from w(k-1) to w(k) for every k from 1 to m. The number of edges of this path is m. The output `dist` is defined, for 0 <= v < n_vertices, by

    dist[v] = minimum, over all paths from source to v, of the number of edges of the path,    if there is at least one path from source to v

    dist[v] = -1    if there is no path from source to v

The path that consists of the vertex source alone has no edges, so dist[source] = 0 for every graph, whatever edges enter or leave source. A vertex v with dist[v] = -1 is called not reachable from source.

## Data layout

- `row_ptr`: n_vertices + 1 signed 32-bit integers (int32), non-decreasing, with row_ptr[0] = 0 and row_ptr[n_vertices] = n_edges.
- `col_idx`: n_edges signed 32-bit integers (int32), each a vertex number in the range 0 <= col_idx[p] < n_vertices.
- The out-neighbours of a vertex are listed in no particular order. The list of vertex u may contain u itself (a self-loop), and it may contain the same vertex more than once (duplicate edges). A vertex may have no outgoing edges, and a vertex may have no incoming edges.
- `dist`: n_vertices signed 32-bit integers (int32). A distance is at most n_vertices - 1, so every result fits in this type.
- All three arrays are contiguous. Do not read or write outside the stated ranges.
- The start addresses of all three arrays are 64-byte aligned. The buffers do not overlap.
- n_vertices and n_edges are integers with n_vertices >= 1 and n_edges >= 0, and always agree with the actual sizes of the buffers. `source` is an integer in the range 0 <= source < n_vertices; it is a vertex number, not a size.

## Contract

- The function has exactly six parameters, in this order: row_ptr, col_idx, dist, n_vertices, n_edges, source. No parameter may be added, removed, renamed or reordered; this includes parameters with default values. The exact declaration is given in the language-specific instructions that accompany this specification.
- dist is allocated by the caller. Its initial contents are unspecified (arbitrary values) and must not be used: the function must overwrite every element of dist with the correct result. It must not add to, or otherwise combine with, whatever dist contained before the call. This includes the vertices that are not reachable from source: their elements do not contain -1 when the function is called, and the function must write -1 into each of them.
- The result must be written into dist. The return value of the function is ignored: results are read only from the buffer dist, so returning a new array instead of filling dist counts as not writing dist.
- `row_ptr` and `col_idx` are read-only. The function must not modify them, not even temporarily.
- Temporary memory (working buffers) may be allocated inside the function, provided it is released before the function returns.
- The function must not read or write files, use the network, print or write anything to standard output or standard error, read or modify environment variables, or rely on or modify global or static mutable state. It must not keep state between calls.
- The function will be called many times in the same process, on the same or on different buffers. Every call must produce the complete, correct result on its own, and two calls with equal inputs must produce identical results.
- The function must be correct for every valid input: every graph in the layout above and every source in the stated range. This includes very small graphs, graphs with self-loops, duplicate edges and cycles (also cycles through source), vertices without outgoing edges, vertices that are not reachable from source, and sizes that are not multiples of 2, 4, 8, 16 or any other block size.
- The inputs always describe a valid graph: `row_ptr` and `col_idx` satisfy the conditions of the data layout, and source is a valid vertex number. No validation of the inputs is required.

## Accuracy

The output is compared element by element against a reference computed from the same inputs with exact integer arithmetic. The comparison is exact; there is no tolerance. The check passes only if

    dist[v] == ref[v]    for every v with 0 <= v < n_vertices.

## Evaluation

- Correctness shapes (n_vertices, n_edges): (6, 10), (1,000, about 4,000) and (65,536, about 520,000).
- Performance shape (n_vertices, n_edges): (4,194,304, about 33,500,000). The output for the performance shape must also pass the accuracy check.
- Graphs: the smallest correctness shape is a fixed graph. In the other shapes the number of outgoing edges of each vertex is drawn independently and uniformly at random from 0, 1, ..., 16 (from 0, 1, ..., 8 in the shape with 1,000 vertices), and the vertex that each edge leads to is drawn independently and uniformly at random from all n_vertices vertices. Every graph of the evaluation contains self-loops, duplicate edges, vertices without outgoing edges and vertices that are not reachable from source, and source always has at least one outgoing edge. In the performance shape about 1,400 of the 4,194,304 vertices (0.03%) are not reachable from source, and the largest distance is 10.
- Procedure: for each correctness shape, the function is called twice on fresh copies of the inputs, with dist pre-filled with arbitrary values before each call. Each output must pass the accuracy check, and the two outputs must be identical. Writes outside the buffers and modifications of the inputs are detected and count as failures.
- If all correctness checks pass, the function is timed on the performance shape: 3 warm-up calls followed by 10 timed calls, and the median of the timed calls is the reported execution time. The inputs are already in memory when the function is called; the measured time is the duration of the call itself.
- The hardware, thread count and build settings used for the evaluation are stated in the language-specific instructions.

## Goal

Provide the fastest correct implementation you can for the performance shape, subject to all the rules above. Correctness on every shape takes precedence over speed: an implementation that fails any correctness check receives no performance result.
