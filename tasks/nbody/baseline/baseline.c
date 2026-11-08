/* nbody 성능 기준선: 손으로 쓴 C + OpenMP (docs/harness_design.md §17). i 병렬, j 내부 루프는 스칼라 1.0f/sqrtf, AoS 그대로.
 * acc_i = Σ_j mass_j · (p_j − p_i) / (|p_j − p_i|² + EPS2)^(3/2), EPS2 = 1e-4, G = 1. j = i 항은 d = 0 이라 정확히 0. */
#include <math.h>

void nbody(const float* pos, const float* mass, float* acc, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const float xi = pos[3 * i], yi = pos[3 * i + 1], zi = pos[3 * i + 2];
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        for (int j = 0; j < n; j++) {
            const float dx = pos[3 * j] - xi;
            const float dy = pos[3 * j + 1] - yi;
            const float dz = pos[3 * j + 2] - zi;
            const float r2 = dx * dx + dy * dy + dz * dz + 1e-4f;
            const float inv_r = 1.0f / sqrtf(r2);
            const float w = mass[j] * inv_r * inv_r * inv_r;
            ax += w * dx;
            ay += w * dy;
            az += w * dz;
        }
        acc[3 * i] = ax;
        acc[3 * i + 1] = ay;
        acc[3 * i + 2] = az;
    }
}
