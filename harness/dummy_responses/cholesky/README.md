# cholesky 더미 응답 (docs/harness_design.md §18)

`provider: dummy` 일 때 `harness.llm_client` 가 API 대신 읽는 가짜 모델 응답이다. `<language>/<k>.md` 가
`sample_idx mod 파일 수` 번째 샘플의 응답이 된다(세 언어 모두 정답 1개씩). 각 파일은 짧은 설명 + 펜스 코드 블록 하나다.

| language | k | 내용 | 기대 (error_stage / error_type / error_detail) | pass |
|---|---|---|---|---|
| numpy | 0 | `L[...] = np.linalg.cholesky(A)` (float32 LAPACK potrf; 반환 L 의 상삼각이 0 이라 한 번의 대입으로 전체 기록) — 기준선 `tasks/cholesky/baseline/baseline.py` 와 같은 코드 | 없음 — bench 까지 정상, `speedup_vs_baseline` ≈ 1 | 1 |
| c_seq | 0 | 행 단위 Cholesky–Banachiewicz(i, j ≤ i, k < j 3중 루프), 내적을 double 로 누적, `sqrt`/나눗셈도 double, 행마다 엄격 상삼각을 0 으로 기록 | 없음 | 1 |
| c_openmp | 0 | 오른쪽 보기(right-looking, 외적) Cholesky: A 의 하삼각을 L 에 복사·상삼각 0(행 병렬) → 열 k 마다 피벗 열을 스케일해 연속 스크래치 `col[]` 에 두고, 후행 갱신 `L[i][j] -= col[i]·col[j]` 를 **행 i 에 대해 `#pragma omp parallel for schedule(dynamic, 16)`**(내부 j 루프가 행 방향 연속이라 벡터화됨) | 없음, `speedup_vs_baseline` ≈ 1.3 | 1 |

## 변형별 주의

- **허용오차(`tolerance.yaml`)**: `rtol = 1e-3`, `atol_rms_rel = 1e-3`(테스트마다 atol = 1e-3 × 그 형태 참조 L 전체의 RMS; 실측 RMS ≈ 1.1547 = sqrt(4/3) 모든 형태 → atol ≈ 1.15e-3), `numeric_pattern_tol_ratio = 20`, 센티널 `-2048`(|L| ≤ 52.5).
  참조는 **float32 로 캐스트한 A** 를 float64 로 올려 계산한 `np.linalg.cholesky` — 커널이 보는 입력과 같은 행렬이다.
- **직진 float32 커널의 계약(§18)**: 순차 Cholesky–Banachiewicz 를 float32 누적·`sqrtf`·나눗셈으로 쓴 커널의 `max_tol_ratio` 실측(2026-11-09)
  tiny 4.7e-5 / rect 5.7e-5 / small 2.2e-4 / large 3.9e-4 — 모두 0.2 이하(여유 500배 이상; A 의 조건수가 약 2.3 이라 float32 로도 충분하다).
  더미 c_openmp/0(float32 right-looking)은 tiny 4.7e-5 / rect 4.0e-4 / small 7.9e-4 / large 2.1e-3, c_seq/0(double 누적)은 2.5e-5 / 3.3e-5 / 3.6e-5 / 4.2e-5.
- **엄격 상삼각 미기록은 `wrong_output`/`partial_write:{n(n−1)/2}/{n²}`**: 출력 버퍼가 센티널로 채워져 오므로 하삼각만 쓰면 상삼각에 센티널이 남는다(쓴 원소는 다 맞으므로 규칙 4c 의 부분 기록).
  `tests/test_tier3_cholesky.py` 가 rect(n=131; 8515/17161)에서 단언한다.
- **Lᵀ(상삼각 인자)를 쓰면 `arg_order`/`transposed_output`**: 2-D 출력 (n, n) 의 공통 전치 패턴 `out.T ≈ ref` 에 걸린다(python·C 커널로 rect 에서 단언).
- **더미 정답과 기준선이 같은 코드**: numpy/0 = `baseline.py` → 이 행의 속도비 ≈ 1 은 하네스 검증용 수치이지 관찰이 아니다(sort numpy/0 과 같은 상황).
- C 변형은 `gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native [-fopenmp] -o kernel.so kernel.c -lm -Wl,--no-undefined`
  로 경고 0 으로 컴파일된다. c_openmp/0 은 호출마다 `malloc(n·4 B)` 스크래치 하나를 쓰고 반환 전에 해제한다(계약 허용).

## 실측 (2026-11-09, 구현자 워커 직접 실행, large n = 2,048, warmup 3 + timed 10 의 중앙값, governor `performance`)

| 커널 | 스레드·코어 | median ms | cv | GFLOPS (n³/3) | max_tol_ratio (large) |
|---|---|---|---|---|---|
| 기준선 `baseline.py` (`np.linalg.cholesky`, OpenBLAS 0.3.30 spotrf) | 4T, cpu 0,2,4,6 | 186.6 | 0.015 | 15.3 | 3.7e-5 |
| 기준선 1스레드 | 1T, cpu 2 | 206.7 | 0.015 | 13.8 | 3.7e-5 |
| c_openmp/0 | 4T | 147.1 | 0.094 | 19.5 | 2.1e-3 |
| c_seq/0 | 1T, cpu 2 | 951.6 | 0.048 | 3.0 | 4.2e-5 |
| numpy/0 (= 기준선) | 4T 환경 | (기준선과 같음) | | | 3.7e-5 |

check(tiny·rect·small)의 max_tol_ratio: numpy 2.5e-5 / 3.3e-5 / 3.6e-5, c_seq 2.5e-5 / 3.3e-5 / 3.6e-5, c_openmp 4.7e-5 / 4.0e-4 / 7.9e-4 — 모두 pass, 실행 간 불일치 0.
속도비(대응 기준선 대비): c_openmp ≈ 1.27, c_seq ≈ 0.22, numpy ≈ 1. OpenBLAS 의 float32 potrf 가 4스레드에서 15 GFLOPS 밖에 내지 못해(n=2048 은 스레드 병렬이 잘 안 먹는 크기;
1스레드 13.8 GFLOPS 와 거의 같다) 단순한 right-looking OpenMP 커널도 기준선을 넘는다 — 실제 결과 해석 때 "기준선이 약한 과제" 로 둔다(nbody 와 비슷한 주의).
