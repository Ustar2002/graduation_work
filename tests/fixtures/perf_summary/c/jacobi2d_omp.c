/* jacobi2d 성능 기준선: 손으로 쓴 C + OpenMP (CLAUDE.md §4). 버퍼 두 개를 번갈아 쓰고 마지막 스텝이 dst 에 쓰도록 맞춘다. */
#include <stdlib.h>
#include <string.h>

static void step(const float *restrict in, float *restrict out, int H, int W) {
#pragma omp parallel for schedule(static)
    for (int y = 1; y < H - 1; y++) {
        const float *up = in + (size_t)(y - 1) * W, *mid = in + (size_t)y * W, *down = in + (size_t)(y + 1) * W;
        float *o = out + (size_t)y * W;
#pragma omp simd
        for (int x = 1; x < W - 1; x++) o[x] = 0.25f * (up[x] + down[x] + mid[x - 1] + mid[x + 1]);
    }
}

void jacobi2d(const float* src, float* dst, int H, int W, int T) {
    size_t n = (size_t)H * W;
    memcpy(dst, src, n * sizeof(float)); /* 경계 포함 전체 복사: 경계는 이후 바뀌지 않는다 */
    if (T <= 0 || H < 3 || W < 3) return;
    if (T == 1) { step(src, dst, H, W); return; }
    float *tmp = aligned_alloc(64, (n * sizeof(float) + 63) / 64 * 64);
    if (!tmp) return;
    memcpy(tmp, src, n * sizeof(float));
    /* 쓰는 버퍼 순서가 … , tmp, dst 로 끝나도록 첫 스텝의 대상을 고른다 */
    float *out = (T % 2) ? dst : tmp;
    const float *in = src;
    for (int t = 0; t < T; t++) {
        step(in, out, H, W);
        in = out;
        out = (out == dst) ? tmp : dst;
    }
    free(tmp);
}
