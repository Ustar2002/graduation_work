/* histogram 성능 기준선: 손으로 쓴 C + OpenMP (CLAUDE.md §4). 스레드별 사본에 센 뒤 합친다.
 *
 * 병렬 구문은 블록 번호 t 에 대한 `parallel for` 하나뿐이다.
 * - 블록 t 는 keys[lo_t, hi_t) 만 읽고 자기 사본 local[t*stride, t*stride + n_bins) 에만 쓴다. 키 구간은 [0, n) 을 겹치지
 *   않게 나누고(hi_t == lo_{t+1}; 같은 식 n*t/nt) stride >= n_bins 라 사본도 겹치지 않는다 → 공유 쓰기가 없으므로
 *   atomic·critical·reduction 이 필요 없다.
 * - cnt, lo, hi, i 는 루프 본문 안에서 선언해 반복마다 따로 있다(private). keys, local, n, nt, n_bins, stride 는 읽기만 한다.
 * - hist 는 병렬 루프 밖에서만 쓴다(0 채움, 순차 합산). 합산은 nt * n_bins 번의 덧셈이라 병렬화하지 않는다.
 * - 블록 수는 omp_get_max_threads() 로 정한다. 실제 팀이 더 작아도 모든 블록이 실행된다(한 스레드가 여러 블록을 맡는다).
 */
#include <omp.h>
#include <stdlib.h>
#include <string.h>

void histogram(const int* keys, int* hist, int n, int n_bins) {
    memset(hist, 0, (size_t)n_bins * sizeof(int)); /* hist 의 초기 내용은 미정의다 */
    int nt = omp_get_max_threads();
    if (nt > n) nt = n;
    /* 사본 간격을 캐시 라인(64 B = int 16개)의 배수로 올려 서로 다른 스레드의 사본이 한 캐시 라인을 나눠 쓰지 않게 한다 */
    size_t stride = ((size_t)n_bins + 15) / 16 * 16;
    int *local = nt > 1 ? aligned_alloc(64, (size_t)nt * stride * sizeof(int)) : NULL;
    if (!local) { /* 스레드가 하나이거나 할당 실패: hist 에 직접 센다 */
        for (int i = 0; i < n; i++) hist[keys[i]]++;
        return;
    }
#pragma omp parallel for schedule(static)
    for (int t = 0; t < nt; t++) {
        int *cnt = local + (size_t)t * stride;
        size_t lo = (size_t)n * t / nt, hi = (size_t)n * (t + 1) / nt;
        memset(cnt, 0, (size_t)n_bins * sizeof(int));
        for (size_t i = lo; i < hi; i++) cnt[keys[i]]++;
    }
    for (int t = 0; t < nt; t++) {
        const int *cnt = local + (size_t)t * stride;
        for (int b = 0; b < n_bins; b++) hist[b] += cnt[b];
    }
    free(local);
}
