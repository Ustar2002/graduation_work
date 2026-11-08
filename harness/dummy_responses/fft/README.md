# fft 더미 응답 (docs/harness_design.md §18)

`provider: dummy` 일 때 `harness.llm_client` 가 API 대신 읽는 가짜 모델 응답이다. `<language>/<k>.md` 가
`sample_idx mod 파일 수` 번째 샘플의 응답이 된다(세 언어 모두 정답 1개씩). 각 파일은 짧은 설명 + 펜스 코드 블록 하나다.

| language | k | 내용 | 기대 (error_stage / error_type / error_detail) | pass |
|---|---|---|---|---|
| numpy | 0 | (n, 2) float32 버퍼를 `reshape(-1).view(np.complex64)` 로 보고 `np.fft.fft` 결과를 out 의 complex64 뷰에 대입 (기준선 `tasks/fft/baseline/baseline.py` 와 같은 코드) | 없음 — bench 까지 정상, `speedup_vs_baseline` ≈ 1 | 1 |
| c_seq | 0 | 반복(iterative) radix-2 DIT: 비트 반전 복사 → log2 n 단계 제자리 버터플라이. 데이터 float32, 단계마다 트위들 테이블을 double `cos`/`sin` 으로 계산, 버터플라이 산술은 double 로 하고 float32 로 반올림 | 없음 | 1 |
| c_openmp | 0 | 위 c_seq 커널의 비트 반전 루프·트위들 루프·버터플라이 루프(`collapse(2)`)에 `#pragma omp parallel for` | 없음, large 에서 c_seq 의 약 2배 | 1 |

## 변형별 주의

- **허용오차(`tolerance.yaml`)**: `rtol = 1e-3`, `atol_rms_rel = 1e-3`(테스트마다 atol = 1e-3 × 그 형태 참조 출력 성분의 RMS; RMS ≈ sqrt(n/3):
  tiny 1.90, rect 18.4, small 148, large 1182), `numeric_pattern_tol_ratio = 20`, **센티널 `-1e9`**(|X_k| ≤ √2·n ≈ 5.9e6 이라 -2048 은 정당 값일 수 있다).
  참조는 float64 복소수 입력의 `np.fft.fft`(정방향, 스케일 없음)를 실수·허수로 나눠 저장한 값.
- **double 트위들 C 커널(c_seq·c_openmp)의 `max_tol_ratio` 실측(2026-11-09)**: tiny 1.65e-5 / rect 1.9e-4 / small 3.9e-4 / large 5.3e-4 — 허용오차의 1/2000 수준.
  **float32 점화식 트위들(`w *= wm`, 블록마다 w = 1 에서 시작) + float32 버터플라이 변형**: tiny 1.6e-4 / rect 8.2e-3 / small 0.63 / **large 1.43 → `numeric_instability`/`imprecise`**(배율 20 이내).
  점화식 오차가 단계 길이(마지막 단계 n/2 = 2²¹)에 비례해 커지기 때문이다. 더미로 싣지 않았다(측정 코드: 구현자 스크래치).
  즉 트위들 정밀도는 이 과제에서 LLM 이 내려야 하는 실제 결정이다: float32 점화식은 small(2¹⁶)까지는 통과하고 large 에서만 떨어진다.
- **NumPy complex64 FFT(numpy/0 = 기준선)**: NumPy 2.3.5 의 pocketfft 는 complex64 를 단정밀도로 변환해 complex64 를 돌려준다(승격 없음).
  `max_tol_ratio` tiny 1.65e-5 / rect 3.6e-5 / small 4.5e-5 / large 4.6e-5.
- **부호 규약 반대(역변환, 스케일 없음)**: `np.fft.ifft(z) * n` 처럼 지수 부호가 양이면 `wrong_output`/`mismatch`(2-D 출력 (n, 2) 의 공통 전치 패턴
  `out.reshape(2, n).T ≈ ref` 에는 걸리지 않는다). `tests/test_tier3_fft.py` 가 rect 에서 단언한다.
- **실수부만 기록(허수부 센티널 잔류)**: `wrong_output`/`partial_write:n/2n`(쓴 원소는 전부 맞으므로 규칙 4c). 같은 테스트가 단언한다.
- C 변형은 `gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native [-fopenmp] -o kernel.so kernel.c -lm -Wl,--no-undefined`
  로 경고 0 으로 컴파일된다. `M_PI` 는 `-std=gnu11` 에서 `<math.h>` 가 제공한다.

## 실측 (2026-11-09, 구현자 워커 직접 실행, large n = 2²², warmup 3 + timed 10 의 중앙값)

| 커널 | 스레드·코어 | median ms | cv | GFLOPS (5·n·log2 n) | max_tol_ratio (large) |
|---|---|---|---|---|---|
| 기준선 `tasks/fft/baseline/baseline.py` (NumPy complex64) | 4T 환경, cpu 0,2,4,6 | 151.2 | 0.015 | 3.05 | 4.6e-5 |
| 기준선 1스레드 | 1T, cpu 2 | 151.4 | 0.010 | 3.05 | 4.6e-5 |
| c_openmp/0 | 4T, cpu 0,2,4,6 | 98.0 / 100.4 (2회 실행) | 0.19 (1회 158 ms 이상치; 나머지 87–104) / 0.048 | 4.71 / 4.60 | 5.3e-4 |
| c_seq/0 | 1T, cpu 2 | 197.8 | 0.026 | 2.33 | 5.3e-4 |

pocketfft 는 단일 스레드라 기준선은 스레드 수와 무관하다. 속도비(기준선 대비): c_openmp ≈ 1.5, c_seq ≈ 0.76, numpy ≈ 1.
c_openmp 는 단계마다 병렬 영역 2개(트위들·버터플라이)를 22번 열어 작은 n 에서는 오버헤드가 크지만 정확성에는 영향이 없다.
