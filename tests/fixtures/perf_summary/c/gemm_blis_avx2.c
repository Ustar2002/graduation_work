#include <immintrin.h>
#include <stdlib.h>
#include <string.h>

/* Register block (micro-kernel) and cache blocks. */
#define MR 6
#define NR 16
#define MC 96
#define KC 256
#define NC 4080

static inline int imin(int a, int b) { return a < b ? a : b; }

/* Pack an mc x kc block of A into row panels of height MR (zero padded). */
static void pack_a(const float *A, int lda, int mc, int kc, float *restrict ap)
{
    for (int i = 0; i < mc; i += MR) {
        int rows = imin(MR, mc - i);
        for (int p = 0; p < kc; p++) {
            for (int r = 0; r < rows; r++)
                ap[r] = A[(size_t)(i + r) * lda + p];
            for (int r = rows; r < MR; r++)
                ap[r] = 0.0f;
            ap += MR;
        }
    }
}

/* Pack a kc x nc block of B into column panels of width NR (zero padded). */
static void pack_b(const float *B, int ldb, int kc, int nc, float *restrict bp)
{
    for (int j = 0; j < nc; j += NR) {
        int cols = imin(NR, nc - j);
        for (int p = 0; p < kc; p++) {
            const float *row = B + (size_t)p * ldb + j;
            for (int c = 0; c < cols; c++)
                bp[c] = row[c];
            for (int c = cols; c < NR; c++)
                bp[c] = 0.0f;
            bp += NR;
        }
    }
}

/* out[0:6, 0:16] += Ap (6 x kc) * Bp (kc x 16); the twelve accumulators stay in registers. */
static void micro_kernel(int kc, const float *restrict ap, const float *restrict bp, float *restrict out, int ldo)
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
    for (int p = 0; p < kc; p++) {
        const __m256 b0 = _mm256_load_ps(bp);
        const __m256 b1 = _mm256_load_ps(bp + 8);
        __m256 a;
        a = _mm256_broadcast_ss(ap + 0);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(ap + 1);
        c10 = _mm256_fmadd_ps(a, b0, c10);
        c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(ap + 2);
        c20 = _mm256_fmadd_ps(a, b0, c20);
        c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(ap + 3);
        c30 = _mm256_fmadd_ps(a, b0, c30);
        c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(ap + 4);
        c40 = _mm256_fmadd_ps(a, b0, c40);
        c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(ap + 5);
        c50 = _mm256_fmadd_ps(a, b0, c50);
        c51 = _mm256_fmadd_ps(a, b1, c51);
        ap += MR;
        bp += NR;
    }
    _mm256_storeu_ps(out + 0 * ldo, _mm256_add_ps(_mm256_loadu_ps(out + 0 * ldo), c00));
    _mm256_storeu_ps(out + 0 * ldo + 8, _mm256_add_ps(_mm256_loadu_ps(out + 0 * ldo + 8), c01));
    _mm256_storeu_ps(out + 1 * ldo, _mm256_add_ps(_mm256_loadu_ps(out + 1 * ldo), c10));
    _mm256_storeu_ps(out + 1 * ldo + 8, _mm256_add_ps(_mm256_loadu_ps(out + 1 * ldo + 8), c11));
    _mm256_storeu_ps(out + 2 * ldo, _mm256_add_ps(_mm256_loadu_ps(out + 2 * ldo), c20));
    _mm256_storeu_ps(out + 2 * ldo + 8, _mm256_add_ps(_mm256_loadu_ps(out + 2 * ldo + 8), c21));
    _mm256_storeu_ps(out + 3 * ldo, _mm256_add_ps(_mm256_loadu_ps(out + 3 * ldo), c30));
    _mm256_storeu_ps(out + 3 * ldo + 8, _mm256_add_ps(_mm256_loadu_ps(out + 3 * ldo + 8), c31));
    _mm256_storeu_ps(out + 4 * ldo, _mm256_add_ps(_mm256_loadu_ps(out + 4 * ldo), c40));
    _mm256_storeu_ps(out + 4 * ldo + 8, _mm256_add_ps(_mm256_loadu_ps(out + 4 * ldo + 8), c41));
    _mm256_storeu_ps(out + 5 * ldo, _mm256_add_ps(_mm256_loadu_ps(out + 5 * ldo), c50));
    _mm256_storeu_ps(out + 5 * ldo + 8, _mm256_add_ps(_mm256_loadu_ps(out + 5 * ldo + 8), c51));
}

void gemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    memset(C, 0, (size_t)M * N * sizeof(float));
    float *ap = aligned_alloc(64, (size_t)MC * KC * sizeof(float));
    float *bp = aligned_alloc(64, (size_t)KC * NC * sizeof(float));
    if (!ap || !bp) {
        free(ap);
        free(bp);
        return;
    }
    for (int jc = 0; jc < N; jc += NC) {
        int nc = imin(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            int kc = imin(KC, K - pc);
            pack_b(B + (size_t)pc * N + jc, N, kc, nc, bp);
            for (int ic = 0; ic < M; ic += MC) {
                int mc = imin(MC, M - ic);
                pack_a(A + (size_t)ic * K + pc, K, mc, kc, ap);
                for (int jr = 0; jr < nc; jr += NR) {
                    int cols = imin(NR, nc - jr);
                    const float *b_panel = bp + (size_t)(jr / NR) * NR * kc;
                    for (int ir = 0; ir < mc; ir += MR) {
                        int rows = imin(MR, mc - ir);
                        const float *a_panel = ap + (size_t)(ir / MR) * MR * kc;
                        float *c = C + (size_t)(ic + ir) * N + jc + jr;
                        if (rows == MR && cols == NR) {
                            micro_kernel(kc, a_panel, b_panel, c, N);
                        } else {
                            float edge[MR * NR];
                            memset(edge, 0, sizeof edge);
                            micro_kernel(kc, a_panel, b_panel, edge, NR);
                            for (int r = 0; r < rows; r++)
                                for (int q = 0; q < cols; q++)
                                    c[(size_t)r * N + q] += edge[r * NR + q];
                        }
                    }
                }
            }
        }
    }
    free(ap);
    free(bp);
}
