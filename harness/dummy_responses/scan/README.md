# scan 더미 응답 (docs/harness_design.md §17)

`provider: dummy` 일 때 `harness.llm_client` 가 API 대신 읽는 가짜 모델 응답이다.
`<language>/<k>.md` 가 `sample_idx mod 파일 수` 번째 샘플의 응답이 된다(numpy·c_seq 는 1개, c_openmp 는 2개).
각 파일은 실제 응답처럼 짧은 설명 + 펜스 코드 블록 하나다.

파이프라인(extract → build → check → bench)이 아래 기대표대로 분류해야 한다. 분류 규칙은 §10, 정수 출력 규칙은 §17.

| language | k | 내용 | 기대 (error_stage / error_type / error_detail) | pass |
|---|---|---|---|---|
| numpy | 0 | `np.cumsum(x, dtype=np.int32, out=out)` | 없음 — bench 까지 정상 (large 약 60 ms; 단일 스레드 누적) | 1 |
| c_seq | 0 | 순차 누적 루프 `acc += x[i]; out[i] = acc` | 없음 (large 약 13 ms, P-core 1개) | 1 |
| c_openmp | 0 | 2-pass 블록 스캔: 스레드별 블록 합 → exclusive 오프셋 → 블록 내 순차 스캔 (스택 배열 `MAX_THREADS=256`, `num_threads` 절) | 없음 (large 약 10.5 ms ≈ 기준선 `tasks/scan/baseline/baseline.c`) | 1 |
| c_openmp | 1 | `out[0] = x[0]; #pragma omp parallel for schedule(static)` 로 `out[i] = out[i-1] + x[i]` — 순차 의존 무시 | check / race_condition / `nondeterministic:<불일치 테스트 목록>` (실측 3/3; 목록은 타이밍 의존, 아래 주의). 세 형태 모두에서 두 호출이 같게 틀리면 check / wrong_output / `mismatch` (§17 이 둘 다 허용) | 0 |

## 변형별 주의

- **c_openmp/1 의 의존성 위반.** 정적 분할에서 각 스레드의 첫 반복은 이전 블록의 마지막 원소 `out[lo-1]` 을 읽는다. 이전 스레드가 아직 쓰지 않았으면
  센티널(−2048), 이미 썼으면 정답이므로, 첫 블록을 뺀 각 블록이 상수(−2048 − 참 오프셋)만큼 밀린다(블록 안은 순차라 일관됨). 어느 블록이 밀리는지는
  스레드 시작 시점에 달려 호출마다 달라질 수 있다 → 2회 호출 비교에서 `nondeterministic` → 샘플 수준 `race_condition`.
  실측(2026-11-08, `taskset -c 0,2,4,6`, `OMP_NUM_THREADS=4`, `OMP_PROC_BIND=true OMP_PLACES=cores`): 3회 모두 `race_condition`, detail 은
  `nondeterministic:tiny,rect,small` / `nondeterministic:tiny` / `nondeterministic:rect` 로 매번 달랐다. 호출 하나하나는 항상
  `wrong_output/mismatch`(max_abs_err: tiny 2115~2204, rect 1,163,377, small 12,185,827)였다 — 즉 어느 형태도 "항상 불일치"가 아니고,
  두 호출이 세 형태 모두에서 우연히 같게 틀리면 `wrong_output`/`mismatch` 로 떨어진다. 단위 테스트는 두 분류를 모두 허용한다.
- **정수 출력.** `tolerance.yaml` 이 rtol = atol = 0 이라 틀린 원소의 배율(`max_tol_ratio`)이 inf → `numeric_instability`(imprecise)는 나올 수 없다.
  센티널 −2048 은 음이 아닌 접두합이 될 수 없으므로 미기록·부분 기록·누적(`accumulated_into_c`) 판정이 그대로 동작한다.
- **C 변형 컴파일.** 하네스 명령 `gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native [-fopenmp] -o kernel.so kernel.c -lm -Wl,--no-undefined`
  로 경고 0 으로 빌드됨을 확인했다. 벡터화 리마크(`-fopt-info-vec-*`): 블록 합 루프는 32바이트 벡터로 벡터화, 스캔 루프는 `not vectorized: unsupported use in stmt`
  (접두 의존), c_openmp/1 은 `possible dependence between data-refs`.

## 검증

2026-11-08 실측(governor `performance`, large n = 2²⁵, 작업집합 256 MiB — L3 18 MiB 밖이라 메모리 대역폭 바운드):
기준선 10.59 ms(cv 2.3 %, 3.45 GHz) · c_openmp/0 10.46 ms(cv 3.2 %) · c_seq/0 13.42 ms(1스레드, 4.6 GHz) · numpy/0 59.85 ms(cv 1.7 %).
4스레드 2-pass 가 순차보다 1.27배만 빠른 것은 대역폭 한계(입력 2회 읽기 + 출력 1회 쓰기 384 MiB / 10.6 ms ≈ 36 GB/s)다.
`tests/test_tier2_scan.py` 가 더미 4개의 추출(펜스 1개·시그니처 OK), C 빌드, worker check(정답 3개 pass, c_openmp/1 은 pass 아님)를 단언한다.

## 통합 검증 (2026-11-08, `dummy-tier2`, `--n 2`)

6행 모두 기대표와 일치: 정답 5행 pass(large median numpy 95.9·72.0 ms, c_seq 20.0·17.2 ms(1T), c_openmp/0 20.3 ms(cv 0.19); 기준선 run 시작 10.58 ms·1T 24.37 ms —
통합 작업과 동시에 돌아 위 실측보다 느리고 흔들렸다), c_openmp/1 은 `check / race_condition / nondeterministic:tiny,rect`(small 은 두 호출이 같게 틀림 — max_abs_err 12,185,827;
호출 수준은 전부 `wrong_output/mismatch`). 전체 결과는 `harness/dummy_responses/README.md`.
