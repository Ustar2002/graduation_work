#include <stddef.h>
#include <string.h>

#define BLOCK 64
#define MIN(a, b) ((a) < (b) ? (a) : (b))

void gemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    memset(C, 0, (size_t)M * N * sizeof(float));
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < M; ii += BLOCK) {
        for (int jj = 0; jj < N; jj += BLOCK) {
            for (int kk = 0; kk < K; kk += BLOCK) {
                for (int i = ii; i < MIN(ii + BLOCK, M); i++) {
                    for (int k = kk; k < MIN(kk + BLOCK, K); k++) {
                        const float a = A[(size_t)i * K + k];
                        #pragma omp simd
                        for (int j = jj; j < MIN(jj + BLOCK, N); j++)
                            C[(size_t)i * N + j] += a * B[(size_t)k * N + j];
                    }
                }
            }
        }
    }
}
