/* floyd_warshall 성능 기준선: 손으로 쓴 C + OpenMP (설계 v3 §4). 출력 버퍼에서 제자리 갱신한다.
 *
 * 중간 정점 k 의 루프는 순차다(스텝 k 는 스텝 0..k-1 의 결과를 읽는다). 스텝 k 안에서 행 i 들은 서로 독립이라 병렬로 돈다:
 *   - 행 i 를 맡은 스레드는 행 i 만 쓰고, 행 i 와 행 k 만 읽는다.
 *   - 행 k 는 스텝 k 에서 바뀌지 않으므로(D[k][k] = 0) i == k 를 건너뛴다 → 어떤 원소도 한 스레드가 쓰는 동안 다른 스레드가
 *     읽지 않는다.
 *   - D[i][k] 는 안쪽 루프 전에 한 번 읽어 둔다(j == k 에서 같은 값이 다시 쓰일 뿐이다).
 */
#include <stddef.h>
#include <string.h>

void floyd_warshall(const float* W, float* D, int n) {
    memcpy(D, W, (size_t)n * (size_t)n * sizeof(float));
    for (int k = 0; k < n; k++) {
        const float *dk = D + (size_t)k * n;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            if (i == k) continue;
            float *restrict di = D + (size_t)i * n;
            const float dik = di[k];
#pragma omp simd
            for (int j = 0; j < n; j++) {
                const float t = dik + dk[j];
                di[j] = t < di[j] ? t : di[j];
            }
        }
    }
}
