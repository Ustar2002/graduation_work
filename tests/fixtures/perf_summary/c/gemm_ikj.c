#include <stddef.h>

void gemm(const float *restrict A, const float *restrict B, float *restrict C, int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        float *Ci = C + (size_t)i * N;
        for (int j = 0; j < N; j++)
            Ci[j] = 0.0f;
        for (int k = 0; k < K; k++) {
            const float a = A[(size_t)i * K + k];
            const float *Bk = B + (size_t)k * N;
            for (int j = 0; j < N; j++)
                Ci[j] += a * Bk[j];
        }
    }
}
