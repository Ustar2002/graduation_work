# bfs 더미 응답 (docs/harness_design.md §13·§18)

`provider: dummy` 일 때 `harness.llm_client` 가 API 대신 읽는 가짜 모델 응답이다.
`<language>/<k>.md` 가 `sample_idx mod 파일 수` 번째 샘플의 응답이 된다(numpy·c_seq 는 1개, c_openmp 는 2개).
각 파일은 실제 응답처럼 짧은 설명 + 펜스 코드 블록 하나다.

파이프라인(extract → build → check → bench)이 아래 기대표대로 분류해야 한다. 분류 규칙은 §10, 정수 출력 규칙은 §17·§18.

| language | k | 내용 | 기대 (error_stage / error_type / error_detail) | pass |
|---|---|---|---|---|
| numpy | 0 | 레벨 동기 BFS: `dist[...] = -1` 뒤 프론티어의 간선 구간을 `np.repeat`/`np.arange` 로 한꺼번에 모아 미방문 대상에 레벨 부여, `np.unique` 로 다음 프론티어 | 없음 — bench 까지 정상 (large 약 1.25 s; 단일 스레드) | 1 |
| c_seq | 0 | 큐 BFS(`malloc` 한 n 칸 배열, head/tail), `dist` 를 −1 로 채운 뒤 정점 0 부터 완화; 할당 실패 시 큐 없는 레벨 스윕 | 없음 (large 약 358 ms, P-core 1개) | 1 |
| c_openmp | 0 | 레벨 동기 top-down: 프론티어 `omp for schedule(dynamic)`, `__sync_bool_compare_and_swap(&dist[v], -1, level)` 로 정점 차지, 스레드별 지역 버퍼(1024)를 `__sync_fetch_and_add` 로 예약한 구간에 복사 | 없음 (large 약 107 ms ≈ 기준선 `tasks/bfs/baseline/baseline.c`) | 1 |
| c_openmp | 1 | 레벨 동기 BFS 의 프론티어 루프를 `#pragma omp parallel for` 로 돌리되 원자 연산 없음: `if (dist[v] == -1) { dist[v] = level; next[next_size++] = v; }` (방문 표시도 push 도 보호 없음) | check / race_condition / `nondeterministic:rect,small` (실측 6/6; 목록은 타이밍 의존, 아래 주의). 세 형태 모두에서 두 호출이 같게 틀리면 check / wrong_output / `mismatch` (§17 이 둘 다 허용) | 0 |

## 변형별 주의

- **c_openmp/1 의 race.** 공유 카운터 `next_size++` 가 읽기-더하기-쓰기라 4스레드가 서로의 push 를 덮어쓴다(갱신 유실). 다음 프론티어에서
  빠진 정점은 확장되지 않으므로 그 정점만 거쳐 도달하는 정점들이 −1 로 남거나(최대 오차 = 그 레벨 + 1), 더 늦은 레벨에서 다른 경로로
  발견돼 거리가 커진다. 방문 표시 `dist[v] == -1` → `dist[v] = level` 의 경쟁은 같은 값을 쓰므로 거리 자체는 틀리지 않고 중복 push 만
  만든다(이론상 `next` 가 n 을 넘칠 수 있으나 유실이 압도적이라 실측에서 oob_write 는 없었다).
  실측(2026-11-09, `taskset -c 0,2,4,6`, `OMP_NUM_THREADS=4`, `OMP_PROC_BIND=true OMP_PLACES=cores`, tiny·rect·small, 3회 × 2세트):
  6/6 `race_condition/nondeterministic:rect,small`(한 번은 `nondeterministic:small` 로 rect 두 호출이 모두 맞았다). 호출 수준은 rect 가
  `wrong_output/mismatch`(max_abs_err 0~5) 또는 pass, small 은 항상 `wrong_output/mismatch`(max_abs_err 7~9). tiny(n = 7, 프론티어 ≤ 2)는
  충돌이 없어 항상 pass — tiny 의 포함 여부는 검증 기준이 아니다. 두 호출이 세 형태 모두에서 같게 틀리면 `wrong_output` 으로 떨어질 수 있다.
- **도달 불가 정점 −1.** tiny(시드 20261105, n = 7, 행 길이 상한 2)는 dist = `[0, −1, 1, 2, 1, 2, −1]` 로 도달 불가 정점 2개를 담는다.
  rect·small·large 는 평균 바깥 차수 16 이라 진입 차수 0 인 정점의 기대 수가 n·e^{−16} ≈ 0 이고 실제로 도달 불가 정점이 없다
  (`tests/test_tier3_bfs.py` 가 rect·small 의 개수를 보고한다). "도달 불가는 −1" 규칙은 tiny 와 테스트의 손그래프(n = 6, 도달 불가 2개,
  자기 루프·사이클 포함)가 검사한다. 도달 불가에 0 을 쓰는 커널(테스트가 python 으로 심는다)은 tiny 에서 `wrong_output/mismatch`(max_abs_err 1).
- **정수 출력.** `tolerance.yaml` 이 rtol = atol = 0 이라 틀린 원소의 배율(`max_tol_ratio`)이 inf → `numeric_instability` 는 나올 수 없다.
  센티널 −2048 은 홉 수(≥ −1)가 될 수 없으므로 미기록·부분 기록·누적 판정이 그대로 동작한다.
- **C 변형 컴파일.** 하네스 명령 `gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native [-fopenmp] -o kernel.so kernel.c -lm -Wl,--no-undefined`
  로 네 C 파일(기준선 포함) 모두 경고 0 으로 빌드됨을 확인했다.

## 실측 (2026-11-09, large = (n, nnz) = (2²¹, 33,541,983), warmup 3 + timed 10 의 중앙값, load avg 0.6)

| 커널 | 스레드·코어 | median ms | cv |
|---|---|---|---|
| 기준선 `tasks/bfs/baseline/baseline.c` (C+OpenMP, CAS + fetch-add) | 4T, cpu 0,2,4,6 | 109.5 (재측정 103.3) | 0.029 (0.019) |
| c_openmp/0 | 4T, cpu 0,2,4,6 | 107.4 | 0.029 |
| 기준선 1스레드 | 1T, cpu 2 | 413.3 | 0.039 |
| c_seq/0 (큐 BFS) | 1T, cpu 2 | 357.6 | 0.017 |
| numpy/0 (레벨 동기, 단일 스레드 구현) | 4T 환경, cpu 0,2,4,6 | 1250.3 | 0.018 |

모두 check(tiny·rect·small)와 bench 의 final_check 가 `pass`(max_abs_err 0.0, 실행 간 불일치 0). large 의 BFS 는 9 레벨(0~8)이며
레벨 6·7 에 정점의 96 % 가 몰려 있다(프론티어 최대 약 1.09M 정점). 4스레드 기준선은 1스레드 기준선의 약 4.0배, 순차 큐 BFS 의 약 3.4배다
(불규칙 간접 참조 `col_idx → dist` 가 병목; 작업집합 col_idx 134 MB + dist 8 MB 는 L3 18 MiB 밖).

## 테스트 입력 생성 메모

- `tasks/bfs/gen_tests.py` 는 spmv_csr 의 층화 추출을 `n_cols = n` 으로 재사용하되 층 경계를 **정수 산술**(`ceil(j·n/L)`)로 잡는다.
  spmv_csr 처럼 `floor((j+u)·n/L)` 로 자르면 경계가 정수가 아닐 때 확률 ≈ 0.25·(L/n)² 로 이웃 층과 같은 열이 나올 수 있어(테스트의 n = 4096 형태에서
  실제로 발생) 구성적으로 유일성을 보장하도록 바꿨다. 행 길이(= nnz)는 바뀌지 않는다.
- large 생성은 약 3 s, 최대 RSS 약 1.2 GB(행 블록 2¹⁷ 단위 추출).
