/* floyd_warshall 성능 기준선: 손으로 쓴 C + OpenMP (설계 §18). dist_in 을 dist_out 에 복사한 뒤 k 를 순차로 돌고,
 * k 마다 i 를 스레드에 나누고 j 내부 루프는 gcc 가 벡터화한다(d_ik + d_kj 와 d_ij 의 비교·선택).
 * 도달 불가는 INF = 1e30f 로 남는다: INF + w 는 float32 에서 INF 에 흡수되고 INF + INF = 2e30 은 유한하므로 min 이 INF 를 유지한다.
 * i == k 행은 d_kk = 0 이라 갱신이 항상 무효이므로 건너뛴다(그 덕에 rowi 와 rowk 가 같은 메모리를 가리키는 일도 없다). */
#include <string.h>

void floyd_warshall(const float* dist_in, float* dist_out, int n) {
    if (n <= 0) return;
    size_t nn = (size_t)n * (size_t)n;
    memcpy(dist_out, dist_in, nn * sizeof(float));
    for (int k = 0; k < n; k++) {
        const float *rowk = dist_out + (size_t)k * n;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            if (i == k) continue;
            float *rowi = dist_out + (size_t)i * n;
            const float dik = rowi[k];
            for (int j = 0; j < n; j++) {
                float via = dik + rowk[j];
                rowi[j] = via < rowi[j] ? via : rowi[j];
            }
        }
    }
}
