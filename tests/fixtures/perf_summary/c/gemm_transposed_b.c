#include <stdlib.h>

/* B is copied into a transposed buffer so that the dot product walks both operands contiguously. */
void gemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    float *Bt = malloc((size_t)N * K * sizeof(float));
    if (!Bt) return;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            Bt[(size_t)j * K + k] = B[(size_t)k * N + j];
    for (int i = 0; i < M; i++) {
        const float *a = A + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const float *b = Bt + (size_t)j * K;
            double sum = 0.0;
            for (int k = 0; k < K; k++)
                sum += (double)a[k] * b[k];
            C[(size_t)i * N + j] = (float)sum;
        }
    }
    free(Bt);
}
