/* nbody 성능 기준선: 손으로 쓴 C + OpenMP. 좌표를 SoA 사본으로 옮기고, i 에 대해 병렬, j 루프는 simd 리덕션.
 *
 * 병렬성 근거:
 * - SoA 사본(px, py, pz)은 병렬 영역 전에 한 스레드가 채우고 그 뒤로는 읽기만 한다. mass 도 읽기 전용이다.
 * - 바깥 루프의 반복 i 는 acc[3i], acc[3i+1], acc[3i+2] 만 쓴다(반복마다 서로 다른 원소) → 쓰기 경쟁 없음.
 * - xi, yi, zi, ax, ay, az 와 j 루프의 변수는 반복 안에서 선언한 지역 변수라 스레드마다 따로 있다.
 * - 반복 사이에 전달되는 값이 없다(각 i 의 합은 독립). 안쪽 루프의 ax, ay, az 는 simd 리덕션 변수다.
 */
#include <math.h>
#include <stdlib.h>

void nbody(const float* pos, const float* mass, float* acc, int n) {
    const float eps2 = 0.01f * 0.01f;
    size_t bytes = ((size_t)n * sizeof(float) + 63) / 64 * 64;
    float *soa = aligned_alloc(64, 3 * bytes);
    if (!soa) return;
    float *restrict px = soa, *restrict py = soa + bytes / sizeof(float), *restrict pz = soa + 2 * (bytes / sizeof(float));
    for (int j = 0; j < n; j++) {
        px[j] = pos[3 * (size_t)j];
        py[j] = pos[3 * (size_t)j + 1];
        pz[j] = pos[3 * (size_t)j + 2];
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        const float xi = px[i], yi = py[i], zi = pz[i];
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
#pragma omp simd reduction(+:ax,ay,az)
        for (int j = 0; j < n; j++) {
            float dx = px[j] - xi, dy = py[j] - yi, dz = pz[j] - zi;
            float r2 = dx * dx + dy * dy + dz * dz + eps2;
            float inv = 1.0f / sqrtf(r2);
            float s = mass[j] * (inv * inv * inv);
            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }
        acc[3 * (size_t)i] = ax;
        acc[3 * (size_t)i + 1] = ay;
        acc[3 * (size_t)i + 2] = az;
    }
    free(soa);
}
