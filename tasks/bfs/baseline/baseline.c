/* bfs 성능 기준선: 손으로 쓴 C + OpenMP (설계 §18). 레벨 동기 top-down BFS.
 * - dist 를 −1 로 채우고 dist[0] = 0, 프론티어 = {0}.
 * - 레벨마다 `#pragma omp parallel for` 로 프론티어를 나눠 이웃을 훑고, 미방문 정점은 dist 에 대한 원자적 CAS
 *   (−1 → level+1)로 정확히 한 스레드만 차지한다. 차지한 정점은 스레드별 지역 버퍼에 모았다가 공유 다음-프론티어 배열에
 *   원자적 fetch-add 로 구간을 예약해 복사한다(한 정점은 정확히 한 번 들어간다).
 * - 프론티어가 비면 끝. 도달 불가 정점은 −1 로 남는다(모든 원소가 기록됨). 할당 실패 시 메모리 없는 순차 폴백. */
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define LOCAL_CAP 2048   /* 스레드별 지역 버퍼(정점 수); 가득 차면 공유 배열로 내보낸다 */

static void bfs_fallback(const int* row_ptr, const int* col_idx, int* dist, int n) {
    /* 큐 없이 레벨마다 전 정점을 훑는다: O(n·levels), 추가 메모리 0. */
    for (int level = 0;; level++) {
        int found = 0;
        for (int u = 0; u < n; u++) {
            if (dist[u] != level) continue;
            for (int p = row_ptr[u]; p < row_ptr[u + 1]; p++) {
                int v = col_idx[p];
                if (dist[v] == -1) { dist[v] = level + 1; found = 1; }
            }
        }
        if (!found) return;
    }
}

void bfs(const int* row_ptr, const int* col_idx, int* dist, int n, int nnz) {
    (void)nnz;
    if (n <= 0) return;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) dist[i] = -1;
    dist[0] = 0;

    int *cur = malloc((size_t)n * sizeof(int));
    int *next = malloc((size_t)n * sizeof(int));
    if (!cur || !next) {
        free(cur); free(next);
        bfs_fallback(row_ptr, col_idx, dist, n);
        return;
    }
    cur[0] = 0;
    int cur_size = 1;
    for (int level = 0; cur_size > 0; level++) {
        int next_size = 0;
        const int next_level = level + 1;
#pragma omp parallel
        {
            int local[LOCAL_CAP];
            int cnt = 0;
#pragma omp for schedule(dynamic, 64) nowait
            for (int f = 0; f < cur_size; f++) {
                const int u = cur[f];
                const int end = row_ptr[u + 1];
                for (int p = row_ptr[u]; p < end; p++) {
                    const int v = col_idx[p];
                    int expected = -1;
                    if (__atomic_load_n(&dist[v], __ATOMIC_RELAXED) == -1 &&
                        __atomic_compare_exchange_n(&dist[v], &expected, next_level, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
                        local[cnt++] = v;
                        if (cnt == LOCAL_CAP) {
                            int base = __atomic_fetch_add(&next_size, cnt, __ATOMIC_RELAXED);
                            memcpy(next + base, local, (size_t)cnt * sizeof(int));
                            cnt = 0;
                        }
                    }
                }
            }
            if (cnt > 0) {
                int base = __atomic_fetch_add(&next_size, cnt, __ATOMIC_RELAXED);
                memcpy(next + base, local, (size_t)cnt * sizeof(int));
            }
        }   /* 암묵적 장벽: next·next_size 가 모든 스레드에서 보인다 */
        int *tmp = cur; cur = next; next = tmp;
        cur_size = next_size;
    }
    free(cur);
    free(next);
}
