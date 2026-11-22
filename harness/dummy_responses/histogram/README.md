# histogram 더미 응답 (docs/harness_design.md §13·§17)

`provider: dummy` 일 때 `harness.llm_client` 가 API 대신 읽는 가짜 모델 응답이다. `<language>/<k>.md` 가
`sample_idx mod 파일 수` 번째 샘플의 응답이 된다(c_openmp 는 2개, 그 밖의 언어는 1개). 각 파일은 짧은 설명 + 펜스 코드 블록 하나다.

| language | k | 내용 | 기대 (error_stage / error_type / error_detail) | pass |
|---|---|---|---|---|
| numpy | 0 | `hist[...] = np.bincount(data, minlength=nbins)` | 없음 — bench 까지 정상 | 1 |
| c_seq | 0 | `memset` 으로 0 초기화 뒤 `hist[data[i]]++` | 없음 | 1 |
| c_openmp | 0 | 스레드별 private 히스토그램(`calloc` 한 스크래치의 슬라이스) → 빈 루프를 나눠 합산 | 없음 | 1 |
| c_openmp | 1 | 0 초기화 뒤 보호 없는 `#pragma omp parallel for` 로 공유 `hist[data[i]]++` (race) | check / race_condition / `nondeterministic:rect,small` (tiny 는 아래 주의) | 0 |

### 새 언어 (numba · halide_auto · halide_manual · rust)

모두 정답 `0.md` 하나이고 기대는 오류 없음 — bench 까지 정상, pass = 1 이다(공통 규칙은 `harness/dummy_responses/README.md` "새 언어 더미"). 넷 다 모든 빈을 덮어쓰므로 센티널이 남지 않는다.

| language | 내용 |
|---|---|
| numba | `prange` 반복마다 `data` 의 연속 청크 하나를 `(nchunks, nbins)` int32 스크래치의 자기 행에 세고, 두 번째 `prange`(빈 단위)가 행들을 합쳐 `hist[b]` 를 덮어쓴다(numba 에는 원자 증가가 없다) |
| halide_auto | 알고리즘만: 최대 64 청크의 부분 히스토그램 `partial[b, c]` 를 RDom 산포로 세고(빈 번호는 `unsafe_promise_clamped`, 꼬리는 `select` 로 가림) `hist[b] = 0; hist[b] += partial[b, rc.x]`. 스케줄 호출 없음 |
| halide_manual | 같은 알고리즘 + `partial.compute_root()`, `partial.update(0).parallel(c)`(청크 행마다 스레드 하나), 청크 합산은 `reorder(b, rc.x).vectorize(b, 8, GuardWithIf)` |
| rust | rayon `par_chunks` 로 청크마다 비공개 `Vec<u32>` 에 세고 `reduce` 로 빈별 합산, 마지막 루프가 모든 `hist[b]` 를 덮어쓴다(크기 계산은 `usize`) |

## 변형별 주의

- **c_openmp/1 의 race.** 4스레드가 같은 빈을 읽기-더하기-쓰기 하므로 갱신이 유실돼 개수가 작게 나오고 호출마다 달라진다
  (정확 일치 요구라 어떤 유실도 `wrong_output/mismatch`; 두 호출이 다르면 `nondeterministic` → 샘플 합산 `race_condition`).
  실측(2026-11-08, `taskset -c 0,2,4,6`, `OMP_NUM_THREADS=4`, rect·small 3회): 3/3 모두 `race_condition/nondeterministic:rect,small`,
  rect 의 max_abs_err 711·1965·1354, small 의 788·1256·590(유실된 개수). tiny(n=13)는 원소가 13개뿐이라 충돌이 거의 없어
  `pass` 로 나왔다(1/1) — tiny 의 포함 여부는 검증 기준이 아니다. 틀린 값이 우연히 두 번 일치하면 `wrong_output` 으로 떨어질 수 있다.
- **0 초기화 누락**(더미에는 없음; `tests/test_tier2_histogram.py` 가 python 커널로 심는다): 센티널(-2048) 위에 더하면
  `(out − sentinel) == ref` 가 정확히 성립해 `wrong_output/accumulated_into_c` 다(설계 §10 규칙 4b). tiny 의 참조는
  `[4, 0, 3, 6]`(시드 20261105) 로 빈 1 이 비어 있어, 더하기만 한 출력은 그 원소가 정확히 센티널로 남는다(`n_unwritten=1`) —
  규칙 4b 가 `partial_write` 보다 먼저여야 하는 이유를 이 과제에서도 실제로 밟는다. tiny 는 "개수 0 인 빈도 써야 한다"는 규칙을
  직접 검사하는 유일한 형태다(rect·small·large 는 모든 빈이 비어 있지 않다).
- C 변형은 `gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native [-fopenmp] -o kernel.so kernel.c -lm -Wl,--no-undefined`
  로 경고 0 으로 컴파일됨을 확인했다.

## 실측 (2026-11-08, large = (2²⁵, 1024), warmup 3 + timed 10 의 중앙값)

| 커널 | 스레드·코어 | median ms | cv |
|---|---|---|---|
| 기준선 `tasks/histogram/baseline/baseline.c` (C+OpenMP) | 4T, cpu 0,2,4,6 | 7.52 (재측정 7.45) | 0.055 |
| c_openmp/0 | 4T, cpu 0,2,4,6 | 7.68 | 0.057 |
| 기준선 1스레드 | 1T, cpu 2 | 14.95 | 0.035 |
| c_seq/0 | 1T, cpu 2 | 13.52 | 0.065 |
| numpy/0 (`np.bincount`) | 4T 환경(단일 스레드 구현), cpu 0,2,4,6 | 72.6 | 0.040 |

모두 check(tiny·rect·small)와 bench 의 final_check 가 `pass`(max_abs_err 0.0, 실행 간 불일치 0).

## 통합 검증 (2026-11-08, `dummy-tier2`, `--n 2`)

6행 모두 기대표와 일치: numpy/0·c_seq/0·c_openmp/0 의 5행 pass(large median numpy 86.1·89.8 ms, c_seq 17.8·17.7 ms(1T), c_openmp/0 8.73 ms;
기준선 run 시작 8.34 ms·1T 19.45 ms), c_openmp/1 은 `check / race_condition / nondeterministic:rect,small`(tiny 두 호출 pass; rect 는 호출 1 pass·호출 2 유실 660;
small 유실 1607·1793). 이 run 은 통합 작업과 동시에 돌아 위 실측표보다 느리다(절대 시간은 참고용). 전체 결과는 `harness/dummy_responses/README.md`.
