/* bfs 성능 기준선: 손으로 쓴 순차 C, 배열 큐 (설계 v3 §4; signatures.yaml 의 openmp: false).
 * 정점은 거리를 받는 순간 큐에 한 번만 들어가므로 큐 길이는 n_vertices 로 충분하다. */
#include <stdlib.h>

void bfs(const int* row_ptr, const int* col_idx, int* dist, int n_vertices, int n_edges, int source) {
    (void)n_edges;
    for (int v = 0; v < n_vertices; v++) dist[v] = -1; /* 도달 불가 정점은 이 값으로 남는다 */
    int *queue = malloc((size_t)n_vertices * sizeof(int));
    if (!queue) return;
    int head = 0, tail = 0;
    dist[source] = 0;
    queue[tail++] = source;
    while (head < tail) {
        int u = queue[head++];
        int d = dist[u] + 1;
        for (int p = row_ptr[u]; p < row_ptr[u + 1]; p++) {
            int v = col_idx[p];
            if (dist[v] < 0) {
                dist[v] = d;
                queue[tail++] = v;
            }
        }
    }
    free(queue);
}
