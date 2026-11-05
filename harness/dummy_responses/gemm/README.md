# gemm 더미 응답 (docs/harness_design.md §13)

`provider: dummy` 일 때 `harness.llm_client` 가 API 대신 읽는 가짜 모델 응답이다.
`<language>/<k>.md` 가 `sample_idx mod 파일 수` 번째 샘플의 응답이 된다(numpy는 5개, C는 3개).
각 파일은 실제 응답처럼 짧은 설명 + 펜스 코드 블록 하나다.

파이프라인(extract → build → check → bench)이 아래 기대표대로 분류해야 한다. 분류 규칙은 §10, 열 의미는 §11.

| language | k | 내용 | 기대 (error_stage / error_type / error_detail) | pass |
|---|---|---|---|---|
| numpy | 0 | `np.matmul(A, B, out=C)` | 없음 — bench까지 정상 | 1 |
| numpy | 1 | `def gemm(A, B, M, N, K): return A @ B` (매개변수 5개) | extract / signature_mismatch / `param_mismatch:...` | 0 |
| numpy | 2 | `C[:] = (A @ B) * 1.01` | check / numeric_instability / `imprecise` (max_tol_ratio ≈ 10, 20 이하) | 0 |
| numpy | 3 | 순수 Python 3중 루프(정확; 256³ 검사에 수십 초) | check 통과; bench / timeout (1024³ 첫 호출이 예산 초과) — pass=1, 시간 NA. n ≥ 4 일 때만 쓰인다 | 1 |
| numpy | 4 | 시그니처는 정상이나 `return A @ B` (C 미기록) | check / wrong_output / `returned_array_not_written` — n ≥ 5 일 때만 쓰인다 | 0 |
| c_seq | 0 | i-k-j 루프, `restrict` (시그니처 한정자 차이 → signature_exact=0, signature_ok=1) | 없음 | 1 |
| c_seq | 1 | 세미콜론 누락 (`float sum = 0.0f`) | build / compile_error / `error: expected ',' or ';' before 'for'` | 0 |
| c_seq | 2 | `C[j*M + i] = sum` (전치 저장; j<N, i<M 이므로 모든 형태에서 범위 안, 센티널 안 남음) | check / arg_order / `transposed_output` | 0 |
| c_openmp | 0 | `#pragma omp parallel for schedule(static)` (i 루프), i-k-j, `restrict` | 없음 | 1 |
| c_openmp | 1 | NULL 포인터 **저장** `float * volatile scratch = NULL; ... *scratch = 1.0f;` | check / runtime_crash / `signal SIGSEGV during tiny call 0` | 0 |
| c_openmp | 2 | k 루프를 병렬화하고 공유 누산기 `sum` 에 리덕션 없이 누적 | check / race_condition / `nondeterministic:<불일치 테스트 목록>` — rect·small 은 항상 포함, tiny(K=3)는 타이밍에 따라 포함되기도 함(`nondeterministic:rect,small` 과 `nondeterministic:tiny,rect,small` 둘 다 실측됨; 아래 주의) | 0 |

## 변형별 주의

- **c_openmp/1 의 `volatile`.** `float *p = NULL; *p = 1.0f;` 처럼 상수 NULL 저장은 gcc `-O3` 의
  `-fisolate-erroneous-paths-dereference` 가 `ud2` 트랩으로 바꿔 **SIGILL** 이 난다(실측, gcc 15.2).
  포인터 변수를 `volatile` 로 두면 컴파일러가 값을 메모리에서 읽어야 하므로 실제 주소 0 저장이 남고 **SIGSEGV** 가 난다.
  읽기만 하는 `float x = *p;` 는 `-O3` 가 제거해 크래시가 나지 않는다.
- **c_openmp/2 의 race.** `sum` 은 병렬 영역 밖에서 선언돼 공유 변수다. 스레드마다 갱신이 유실돼 결과가 크게 틀리고 호출마다 달라진다
  (2회 호출 비교가 허용오차 밖 → `nondeterministic` → `race_condition`). 틀린 값이 우연히 두 번 일치하면 `wrong_output` 으로 떨어질 수 있다.
  실측(2026-11-04, `make dummy RUN_ID=dummy-m1`): rect·small 은 두 호출 모두 `wrong_output/mismatch` 이면서 서로 달라 항상 `nondeterministic`.
  tiny(K=3)는 타이밍에 따라 갈린다 — 11:03 실행에서는 두 호출 모두 pass(`nondeterministic:rect,small`)였지만, 13:26 재검증과 이어진
  `--force` 재실행 7회는 모두 두 호출이 틀리고 서로 달랐다(`nondeterministic:tiny,rect,small`; max_abs_err 0.12~0.51). 스레드당 반복이 1회여도
  공유 `sum` 의 읽기-더하기-쓰기가 겹치면 갱신이 유실되므로 "반복 1회 = 안전" 은 성립하지 않는다. 따라서 이 변형의 기대는
  `check / race_condition` 과 detail 에 rect·small 이 포함되는 것이며, tiny 의 포함 여부는 검증 기준이 아니다. 샘플 합산은 §10 우선순위대로 `race_condition`.
- **numpy/2 의 배율 1.01.** 오차 = 0.01·|ref|, 허용오차 = 1e-3 + 1e-3·|ref| 이므로 비율 ≈ 10·|ref|/(1+|ref|) < 10 → `numeric_pattern_tol_ratio`(20) 이내 → `imprecise`.
- **numpy/3 의 bench.** 256³ 2회 + rect 2회 ≈ 5천만 회 내부 반복으로 check 예산(180 s) 안이지만 1024³ 는 10억 회라
  첫 warmup 호출이 bench 예산(300 s)을 넘겨 `timeout` 이 된다. 성능 실패이지 정확성 실패가 아니므로 pass=1.
- C 변형은 `gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native [-fopenmp] -o k.so k.c -lm -Wl,--no-undefined`
  로 컴파일됨을 확인했다(c_seq/1 은 의도된 컴파일 오류).

## 검증

`make dummy`(또는 `generate --provider dummy` → `run` → `aggregate --csv results/runs_dummy.csv`)의 행이 위 표와 같으면 파이프라인이 정상이다.
`tests/test_extract.py` 는 모든 더미 파일에 대해 extract 단계의 기대(numpy/1 만 `param_mismatch`)를 검사한다.

위 기대표 11건은 분류 규칙 순서 v2.0(센티널 규칙 먼저)과 v2.1(arg_order 패턴 먼저; 설계 §10, 2026-11-04)에서 **같은 결과**를 내므로 `make dummy` 는 규칙 순서의 검증 근거가 아니다 — v2.1 고유 동작(`swapped_MN_dims` 도달, 정사각 `swapped_operands`, 센티널+NaN → `nan_inf`)은 `tests/test_classify.py` 가 단언한다. 2026-11-04 11:18 통합 검증(`make dummy RUN_ID=dummy-m1`; numpy/3·4 는 `dummy-m1x` 로 n≥5 보강)에서 11건 모두 위 표와 일치했다(`results/runs_dummy.csv`).
리뷰 반영(classify·worker·procutil·run·llm_client·extract·docs) 뒤 2026-11-04 13:26 재검증(`results/` 의 dummy-m1·dummy-m1x 와 CSV 를 지우고 `make dummy RUN_ID=dummy-m1` 9행 + `dummy-m1x` numpy/4 1행; numpy/3 은 느려서 생략, 단위 테스트 153개 통과)에서도 10건의 stage/error_type/pass 와 테스트별 pattern 이 전부 일치했다. 유일한 차이는 c_openmp/2 의 detail 에 tiny 가 포함된 것이었고(7/7 재현), 위 표의 그 행을 이에 맞게 고쳤다.
