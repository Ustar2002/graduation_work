/* histogram 성능 기준선: 손으로 쓴 C + OpenMP (설계 §17). 스레드마다 private 히스토그램에 자기 블록을 세고,
 * 마지막에 빈을 스레드별로 나눠 합산해 hist 에 쓴다(각 빈은 한 스레드만 쓰므로 동기화 없음). 모든 빈을 덮어쓴다. */
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void histogram(const int* data, int* hist, int n, int nbins) {
    if (nbins <= 0) return;
    int max_t = omp_get_max_threads();
    int *priv = calloc((size_t)max_t * (size_t)nbins, sizeof(int));
    if (!priv) { /* 할당 실패: 순차 폴백 */
        memset(hist, 0, (size_t)nbins * sizeof(int));
        for (int i = 0; i < n; i++) hist[data[i]]++;
        return;
    }
#pragma omp parallel
    {
        int nt = omp_get_num_threads(), tid = omp_get_thread_num();
        int *local = priv + (size_t)tid * (size_t)nbins;
        long long lo = (long long)n * tid / nt, hi = (long long)n * (tid + 1) / nt;
        for (long long i = lo; i < hi; i++) local[data[i]]++;
#pragma omp barrier
#pragma omp for schedule(static)
        for (int b = 0; b < nbins; b++) {
            int s = 0;
            for (int t = 0; t < nt; t++) s += priv[(size_t)t * (size_t)nbins + b];
            hist[b] = s;
        }
    }
    free(priv);
}
