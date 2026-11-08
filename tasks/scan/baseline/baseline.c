/* scan 성능 기준선: 손으로 쓴 C + OpenMP (설계 §17). 2-pass 블록 스캔.
 *   1) 스레드마다 연속 블록 [lo, hi) 의 합을 구한다(블록 합).
 *   2) 블록 합의 exclusive 접두합이 각 블록의 시작 오프셋이다(스레드마다 자기 앞 블록들의 합을 더한다; 스레드 수가 작아 비용 무시).
 *   3) 스레드마다 오프셋에서 시작해 자기 블록을 순차 inclusive 스캔한다.
 * 입력을 두 번 읽고 출력을 한 번 쓴다(메모리 바운드). 과제 보장(x ∈ [0,32), 접두합 < 2^31)으로 int32 누적이 정당하다;
 * 블록 합·오프셋은 long long 으로 둔다(비용 없음). 스레드 수는 omp_get_max_threads() 를 상한으로 malloc 한다. */
#include <stdlib.h>
#include <omp.h>

void scan(const int* x, int* out, int n) {
    if (n <= 0) return;
    int max_threads = omp_get_max_threads();
    if (max_threads < 1) max_threads = 1;
    long long* block_sum = (long long*)malloc((size_t)max_threads * sizeof(long long));
    if (block_sum == NULL) { /* 할당 실패: 순차 스캔으로 대체 */
        int acc = 0;
        for (int i = 0; i < n; i++) { acc += x[i]; out[i] = acc; }
        return;
    }
#pragma omp parallel
    {
        const int nt = omp_get_num_threads();  /* ≤ max_threads (num_threads 절 없음) */
        const int tid = omp_get_thread_num();
        const long long lo = (long long)n * tid / nt;
        const long long hi = (long long)n * (tid + 1) / nt;

        long long s = 0;  /* 1) 블록 합 — 리덕션이라 컴파일러가 벡터화한다 */
        for (long long i = lo; i < hi; i++) s += x[i];
        block_sum[tid] = s;
#pragma omp barrier

        long long offset = 0;  /* 2) exclusive 오프셋 */
        for (int t = 0; t < tid; t++) offset += block_sum[t];

        int acc = (int)offset;  /* 3) 블록 내 순차 스캔 */
        for (long long i = lo; i < hi; i++) {
            acc += x[i];
            out[i] = acc;
        }
    }
    free(block_sum);
}
