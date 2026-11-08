# floyd_warshall 더미 응답 (docs/harness_design.md §18)

`provider: dummy` 일 때 `harness.llm_client` 가 API 대신 읽는 가짜 모델 응답이다.
`<language>/<k>.md` 가 `sample_idx mod 파일 수` 번째 샘플의 응답이 된다(numpy·c_seq 는 1개, c_openmp 는 2개).
각 파일은 실제 응답처럼 짧은 설명 + 펜스 코드 블록 하나다.

파이프라인(extract → build → check → bench)이 아래 기대표대로 분류해야 한다. 분류 규칙은 §10, Tier 3 계약은 §18.

| language | k | 내용 | 기대 (error_stage / error_type / error_detail) | pass |
|---|---|---|---|---|
| numpy | 0 | `np.copyto(dist_out, dist_in)` 뒤 k 루프(Python)마다 `np.minimum(dist_out, dist_out[:, k:k+1] + dist_out[k:k+1, :], out=dist_out)` (float32 제자리) | 없음 — bench 까지 정상 (large 약 850 ms; 1024회의 1M 원소 외적합+최소, 임시 배열 4 MiB) | 1 |
| c_seq | 0 | k-i-j 삼중 루프, `if (via < rowi[j]) rowi[j] = via;` — gcc 가 j 루프를 32바이트 벡터로 벡터화 | 없음 (large 약 165 ms, P-core 1개, 13 GFLOP/s) | 1 |
| c_openmp | 0 | = 기준선 `tasks/floyd_warshall/baseline/baseline.c`: k 순차, `#pragma omp parallel for` 로 i 분할, j 는 삼항 선택(벡터화), i == k 행 건너뜀 | 없음 (large 약 46 ms ≈ 기준선 48 ms, 47 GFLOP/s) | 1 |
| c_openmp | 1 | `#pragma omp parallel for` 를 **k 루프**에 걸어 단계 k 의 의존(단계 k−1 결과)을 깬다; i·j 는 c_seq/0 과 같음 | check / race_condition / `nondeterministic:rect,small` (실측 3/3). 두 호출이 같게 틀리면 check / wrong_output / `mismatch` (둘 다 허용) | 0 |

## 변형별 주의

- **c_openmp/1 의 의존성 위반.** Floyd–Warshall 의 k 단계는 k−1 단계가 끝난 행렬을 읽어야 한다. k 를 스레드에 나누면 네 스레드가 서로 다른
  k 로 같은 행렬을 동시에 읽고 쓴다: 어떤 중간 정점을 거친 완화가 다른 스레드의 `rowk`/`dik` 읽기 전에 반영되느냐가 타이밍에 달려
  결과가 호출마다 다르다 → 2회 호출 비교에서 `nondeterministic` → 샘플 수준 `race_condition`.
  실측(2026-11-09, `taskset -c 0,2,4,6`, `OMP_NUM_THREADS=4`, `OMP_PROC_BIND=true OMP_PLACES=cores`): 3회 모두 `race_condition` /
  `nondeterministic:rect,small`. 호출 수준은 rect·small 모두 `wrong_output/mismatch`(max_abs_err 2.6~4.9, max_tol_ratio 1.5e3~3.2e3 —
  허용오차의 수천 배라 `numeric_instability` 가 아니다). **tiny(n=8)는 통과한다**: 기본 시드의 tiny 그래프(간선 1→0, 1→2, 3→2, 4→2)에는
  2홉 경로가 없어 참조가 입력과 같고, 어떤 순서로 완화해도 바뀌는 원소가 없다. 같은 이유로 k 루프를 아예 빼고 복사만 하는 커널도
  tiny 는 통과하고 rect(1369 원소 중 1081 개가 완화로 줄어듦)에서 `wrong_output/mismatch` 로 잡힌다(`tests/test_tier3_floyd_warshall.py`).
- **INF 처리.** INF = 1e30 은 float32 에서 INF + w(w < 10) = INF(흡수), INF + INF = 2e30(유한) 이므로 단순 min-plus 로 도달 불가 쌍이 정확히
  INF 로 남는다. 참조(float64)도 같은 성질(1e30 의 float64 ulp ≈ 1.4e14 ≫ 10)이라 도달 불가 원소는 정확히 float64(float32(1e30)) 다.
  허용오차 비교에서 INF 원소는 rtol·1e30 = 1e26 폭을 가져 float32/float64 표현 차(1.5e22)는 통과하고 2e30 은 배율 1e4 로 `wrong_output` 이다.
  도달 불가 쌍은 tiny 에만 있다(비대각 20쌍 중 16쌍); rect·small·large(밀도 0.3, n ≥ 37)는 모든 쌍이 도달 가능하다(설계 §18: 밀도를 바꾸지 않고 확인만).
- **C 변형 컴파일.** 하네스 명령 `gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native [-fopenmp] -o kernel.so kernel.c -lm -Wl,--no-undefined`
  로 경고 0 으로 빌드됨을 확인했다. 벡터화 리마크(`-fopt-info-vec-*`): 세 C 커널 모두 j 루프 `loop vectorized using 32 byte vectors`
  (기준선·c_openmp/0 은 `loop versioned for vectorization because of possible aliasing` — rowi 와 rowk 가 같은 배열이라 런타임 별칭 검사가 붙는다).
- **정확도.** 정답 커널(float32 누적)의 max_tol_ratio: tiny·rect·small 4.4e-4, large 2.6e-4(max_abs_err 3.6e-7) — 허용오차의 0.05 % 로 여유가 크다.

## 검증

2026-11-09 실측(governor `performance`, load 0.9, large n = 1024, 행렬 4 MiB × 2 — L3 안; compute 바운드 2·n³ = 2.1 GFLOP):
기준선 48.36 ms(cv 9.0 %, 44.4 GFLOP/s; 측정 중 코어 클럭 1.85~2.98 GHz 로 흔들림) · c_openmp/0 45.84 ms(cv 4.5 %, 46.8 GFLOP/s) ·
c_seq/0 164.74 ms(1스레드, cv 5.4 %, 13.0 GFLOP/s, 2.6~4.6 GHz) · numpy/0 852.75 ms(cv 6.1 %, 2.5 GFLOP/s).
4스레드 c_openmp 가 순차의 3.6배인 것은 k 마다 parallel region 1024개(≈ 45 µs/region)의 fork-join 오버헤드와 클럭 하향 때문이다.
bench 의 final_check 는 네 커널 모두 pass(max_tol_ratio 2.6e-4).
`tests/test_tier3_floyd_warshall.py` 가 더미 4개의 추출(펜스 1개·시그니처 OK), C 빌드, worker check(정답 3개 + 기준선 pass, c_openmp/1 은 pass 아님)를 단언한다.
