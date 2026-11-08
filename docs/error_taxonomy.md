# 오류 분류표 (docs/error_taxonomy.md)

`CLAUDE.md` §1의 분류표는 실험 전에 고정됐고 **변경 금지**다. 이 문서는 그 아홉 범주 각각이 코드(`harness/classify.py`, `harness/worker.py`, `harness/run.py`, `harness/extract.py`, `harness/build/*.py`)에서 **정확히 어떤 규칙으로** 결정되는지를 적는다(설계 계약 `docs/harness_design.md` §10). 규칙을 바꾸면 이 문서를 같이 고친다.

## 1. 고정 분류표 (아홉 범주)

| `error_type` | 정의(CLAUDE.md) | 결정되는 단계 |
|---|---|---|
| `compile_error` | 빌드 실패(컴파일·링크 오류, 컴파일 타임아웃), 또는 워커(import 모드)가 산출물을 로드하지 못함 | build (워커 import 모드의 로드 실패도 `error_stage=build`) |
| `runtime_crash` | 실행 중 시그널 종료·예외·비정상 종료 | build(import 워커의 시그널·비정상 종료), check, bench |
| `timeout` | 단계 시간 상한 초과(import 워커 180 s = `check_timeout_s`, check 180 s, bench 300 s) | build(import 워커), check, bench |
| `wrong_output` | 허용오차 밖 결과 중 패턴이 수치적이지도 인자 순서도 아닌 것(미기록·부분 기록·누적·범위 밖 쓰기·입력 변조 포함) | check, bench |
| `numeric_instability` | NaN/Inf, 또는 허용오차 초과이면서 패턴이 수치적(모든 원소가 허용오차의 20배 이내) | check, bench |
| `arg_order` | 인자 순서·차원 뒤바뀜으로 판정되는 wrong_output(전치·열우선·피연산자 교환·M/N 교환 시 일치) | check, bench |
| `memory_error` | ASan 탐지 — **M5에서만**. M1에서는 쓰지 않는다(OOB 쓰기는 가드 영역으로 잡아 `wrong_output`) | (M5) |
| `race_condition` | 반복 실행 간 결과 불일치(허용오차 초과) — TSan 탐지는 M5 | check, bench |
| `signature_mismatch` | 요구한 함수 시그니처를 지키지 않음(추출 단계 시그니처 검사 실패, 심벌 미노출, 함수 없음·호출 불가) | extract, build(심벌 미노출; import 워커의 `gemm` 없음·호출 불가도 `error_stage=build`) |

행에서 `error_type`은 위 아홉 값 또는 빈칸이고, `error_stage ∈ {"", generate, extract, build, check, bench}`, `error_detail`은 아래 규칙의 패턴·사유 문자열이다. `error_stage`에 `import`는 **없다**: 워커 import 모드(로드만 하는 첫 워커 실행)의 실패 — 로드 실패든 워커 프로세스의 timeout·시그널이든 — 는 모두 `error_stage=build`로 기록한다(§5; 설계 v2.1, decisions.md 2026-11-04). `pass=1`인 행은 `error_type`이 빈칸이거나 `timeout`(bench 타임아웃)뿐이다.

## 2. 출력 분류 규칙 `classify_output` (호출 1회의 출력; 설계 §10 **v2.1**)

```python
compare(out, ref64, rtol, atol) -> {"ok", "nan_inf", "max_abs_err", "max_tol_ratio", "n_fail", "max_abs_ref"}
classify_output(out, ref64, A0, B0, M, N, K, tol, *, oob_write=False, return_type=None, out_full=None) -> (status, pattern_or_detail, metrics)
```

- 허용오차: `|out − ref| ≤ atol + rtol·|ref|`, `rtol = atol = 1e-3`(`tasks/gemm/tolerance.yaml`), `ref`는 float64 누적. `max_tol_ratio = max |out − ref| / (atol + rtol·|ref|)`(유한 원소 기준; 1 이하면 통과).
- 센티널 `sentinel = -2048.0`(C 사전 채움값). `numeric_pattern_tol_ratio = 20`.
- `out_full`은 가드 포함 전체 C 버퍼(워커가 넘긴다; 규칙 3d `swapped_MN_dims` 판정에만 쓰고 없으면 그 패턴은 생략). `oob_write`는 워커가 가드 영역에서 본 플래그로, 규칙 4d에서만 판정을 바꾸고 그 밖에는 `metrics`에만 남는다.

**순서대로 먼저 맞는 규칙이 결정한다.** v2.1(2026-11-04): arg_order 패턴(규칙 3)을 센티널·계약 규칙(규칙 4)보다 **먼저** 본다. 예전 순서(센티널 규칙 0a–0d가 맨 앞)에서는 `C[i*M + j]`(M/N 교환) 커널이 항상 `partial_write`/`out_of_bounds_write`에 먼저 걸려 `swapped_MN_dims`가 도달 불가였다(실측: (7,5,3)에서 `partial_write:10/35`, (200,130,312)에서 `partial_write:9100/26000`). 패턴이 버퍼 전체를 설명하면 구조적 오류(`arg_order`)가 더 정확한 분류다.

| 순서 | 조건 | status / pattern |
|---|---|---|
| 1 | NaN/Inf 존재(센티널은 유한값이라 제외) | `numeric_instability` / `nan_inf` |
| 2 | `compare(out, ref)` 통과 | `pass` (`oob_write`가 있어도 여기서는 `pass`; 호출 수준 덮어쓰기는 아래 주) |
| 3 | arg_order 패턴(같은 rtol/atol로 일치; **3a → 3b → 3c → 3d 순서**로 검사, 단순한 가설부터) | `arg_order` / 패턴명 — 센티널·OOB가 남아 있어도 패턴이 맞으면 `arg_order`로 보고하고 `n_unwritten`·`oob_write`는 `metrics`에 그대로 둔다 |
|   | 3a `out.reshape(N, M).T ≈ ref` (전치 저장 `C[j*M + i]`) | `transposed_output` |
|   | 3b M == N == K 이고 `out ≈ B0 @ A0` (피연산자 교환; 정사각에서만 검사) | `swapped_operands` |
|   | 3c `A_cm = A0.reshape(K, M).T`, `B_cm = B0.reshape(N, K).T`, `out.reshape(N, M).T ≈ A_cm @ B_cm` (입력·출력 모두 열우선으로 해석) | `column_major_all` |
|   | 3d M ≠ N 이고, 가드 포함 전체 C 버퍼 `out_full`(길이 ≥ max(M,N)²)에서 `idx = i·M + j`(i<M, j<N) 위치가 `ref[i,j]`와 허용오차 내로 맞는 비율 ≥ 99 % — `C[i*M + j]`로 쓴 M/N 교환(`out_full`이 없으면 생략; 정사각은 `C[i*N + j]`와 같아 검사하지 않음) | `swapped_MN_dims` |
| 4 | 센티널·계약 검사(**4a → 4b → 4c → 4d 순서**) | `wrong_output` / 아래 |
|   | 4a `n_unwritten = count(out == sentinel)` 가 M·N 전부 | `output_not_written` — python 백엔드이고 `return_type`이 ndarray면 `returned_array_not_written`(C를 채우지 않고 새 배열을 반환) |
|   | 4b `compare(out − sentinel, ref)` 통과 (C에 더함) | `accumulated_into_c` |
|   | 4c 일부 원소만 센티널 | `partial_write:<n_unwritten>/<M·N>` |
|   | 4d `oob_write`(가드 영역에 센티널이 아닌 값) | `out_of_bounds_write` |
| 5 | `max_tol_ratio ≤ 20`(모든 원소가 허용오차의 20배 이내) | `numeric_instability` / `imprecise` |
| 6 | 그 밖 | `wrong_output` / `mismatch` |

주:

- **정사각 모호성(3b를 3c보다 먼저 두는 이유).** M == N == K이면 `A_cm = A0ᵀ`, `B_cm = B0ᵀ`이므로 3c의 조건은 `out.reshape(N, M).T ≈ (B0 @ A0)ᵀ`, 즉 `out ≈ B0 @ A0`로 3b와 **대수적으로 동일**하다. 예전 순서(3c 먼저)에서는 `swapped_operands`가 행에 절대 나타날 수 없었으므로 v2.1은 3b를 먼저 검사해 그 이름으로 보고한다. 비정사각에서는 3b를 검사하지 않는다(`B0 @ A0`의 형태가 맞지 않음). **샘플 수준 부작용**: 정확성 테스트가 `tiny`·`rect`(비정사각)·`small`(정사각)이므로 열우선 커널은 `column_major_all`, `column_major_all`, `swapped_operands`로 패턴이 갈려 §4 규칙 3("전부 같은 패턴")을 못 채우고 `wrong_output`/`mixed:arg_order/column_major_all,arg_order/swapped_operands`가 된다(실측 2026-11-04). 피연산자를 바꾼 numpy 커널(`np.matmul(B, A, out=C)`)은 비정사각에서 `ValueError`로 `runtime_crash`다. 이 부작용의 처리(두 패턴을 같은 족으로 합산할지)는 `docs/decisions.md`에 미결로 올라가 있다.
- **4b가 4c보다 먼저인 이유.** float32에서 `sentinel + ref`는 |ref| < 2⁻¹³이면 정확히 `-2048.0`으로 되돌아가 `n_unwritten`에 세어진다(실측: `C += A@B`가 small(256³)에서 `partial_write:1/65536`, large에서 `partial_write:10/1048576`로 잘못 기록됐다). 진짜 부분 기록은 모든 원소가 `sentinel + ref`와 일치할 수 없으므로 누적 검사가 부분 기록을 가리지 않는다. 누적 + OOB는 `accumulated_into_c`다(샘플 수준 detail은 §4가 `oob_write` 플래그를 먼저 보므로 행 결과는 `out_of_bounds_write`).
- **호출 수준 덮어쓰기(워커 `_apply_contract_flags`, 설계 §8).** 규칙 2가 규칙 4보다 먼저라 출력이 맞으면 `classify_output`은 OOB가 있어도 `pass`를 돌려준다. 워커는 그 호출을 `wrong_output`/`out_of_bounds_write`로 기록하고(`pass`만 덮음; `arg_order`는 유지), `inputs_mutated`면 `wrong_output`이 아닌 모든 판정을 `wrong_output`/`inputs_mutated`로 덮는다. bench의 `final_check`도 같은 규칙이다.
- **v2.1의 다른 부수 효과.** 일부 센티널 + NaN은 규칙 1이 먼저라 `nan_inf`다(예전: `partial_write`). 전부 센티널인 출력에는 NaN이 있을 수 없으므로 "C를 전혀 안 쓴" 커널은 그대로 4a `wrong_output`이다.

`metrics`에는 `max_abs_err`, `max_tol_ratio`, `n_fail`, `nan_inf`, `n_unwritten`, `n_total`, `max_abs_ref`, `oob_write`가 판정과 무관하게 항상 들어가고, check 단계의 테스트별 결과는 행의 `check_tests` JSON(`{name, classification, pattern, max_tol_ratio, elapsed_ms, nondeterministic}`)과 `check_max_abs_err`, `check_max_tol_ratio`로 남는다.

예: 더미 `numpy/2`(`C[:] = (A @ B) * 1.01`)는 `max_tol_ratio ≈ 10` → 규칙 5 `numeric_instability`/`imprecise`; 더미 `c_seq/2`(`C[j*M + i] = …`)는 규칙 3a `arg_order`/`transposed_output`; 더미 `numpy/4`(`return A @ B`)는 규칙 4a `wrong_output`/`returned_array_not_written`. 더미 기대표 11건(§8)은 v2.0·v2.1 어느 순서에서도 같은 결과를 내므로 통합 검증(2026-11-04 11:18)은 순서 변경의 검증 근거가 아니다 — v2.1 고유 동작(`swapped_MN_dims` 도달, 정사각 `swapped_operands`, 센티널 + NaN → `nan_inf`, 4b/4c 순서)은 `tests/test_classify.py`가 단언한다.

## 3. 호출 간 불일치 (`nondeterministic`)

- check: 테스트마다 `check_repeats = 2`회 호출. `nondeterministic = not np.allclose(out1, out2, rtol, atol, equal_nan=True) or (isfinite 마스크 불일치)`. 두 출력이 비트만 다르고 허용오차 안이면 불일치가 **아니다**(누적 순서 차이는 허용).
- bench: 첫 timed 결과 `C_first`와 이후 timed 호출을 비교해 `nondeterministic_runs`(허용오차 초과 수), `nondeterministic_bitwise_runs`(비트 불일치 수; 정보용)를 센다.

## 4. 샘플 수준 합산 (check 테스트들 + bench `final_check`)

우선순위 순서(먼저 맞는 것이 `error_type`; `classify.aggregate_sample`):

1. 어떤 테스트든 예외/크래시 → `runtime_crash` (detail `<test>: <ExcType>: <message>`)
2. 어떤 테스트든 `nondeterministic`(허용오차 초과 불일치) → `race_condition` (detail `nondeterministic:<test,…>`)
3. 실패한 테스트 **전부**가 `arg_order`이고 **같은 패턴** → `arg_order` / 그 패턴
4. `inputs_mutated` → `wrong_output`/`inputs_mutated`; `oob_write` → `wrong_output`/`out_of_bounds_write`; 어떤 테스트든 `wrong_output` → `wrong_output` / 첫 `wrong_output` 테스트의 패턴
5. 실패가 **전부** `numeric_instability` → `numeric_instability` / 첫 실패의 패턴
6. 그 밖 — 패턴이 섞인 경우(`arg_order` 패턴이 둘 이상이거나 `arg_order` + `numeric_instability`; 설계가 상정하지 않은 조합) → 보수적으로 `wrong_output` / `mixed:<classification/pattern,…>`(정렬·쉼표 구분. 예 `mixed:arg_order/column_major_all,arg_order/transposed_output`)
7. 전부 pass → `pass`

`error_detail`은 결정한 테스트의 패턴(예 `transposed_output`, `partial_write:3/35`, `imprecise`)이나 사유다.

## 5. 프로세스 수준 매핑 (`run.py`)

| 단계 | 조건 | `error_stage` | `error_type` | `error_detail` | `pass` | `valid_sample` |
|---|---|---|---|---|---|---|
| generate | raw의 `error`가 non-null(API 오류: 인증·404·400·연결·5xx·재시도 소진) | `generate` | (빈칸) | `api_error` (오류 문자열 자체는 `api_error` 열에 `fatal:…`/`transient:…`) | 0 | **0** |
| generate | `stop_reason == max_tokens`(출력 절단) | `generate` | (빈칸) | `truncated` | 0 | **0** |
| generate | `stop_reason == refusal` | `generate` | (빈칸) | `refusal:<stop_details.category>` | 0 | 1 |
| extract | `extract().ok == False` | `extract` | `signature_mismatch` | `no_code_block` / `function_not_found` / `param_mismatch:<설명>` / `static_linkage` (정보 노트 `unterminated_fence` 등이 `;`로 병기될 수 있음 — 그 자체는 실패 사유가 아니다, 아래 주) | 0 | 1 |
| build (C) | gcc rc ≠ 0 | `build` | `compile_error` | `error:` 포함 첫 3줄 | 0 | 1 |
| build (C) | 컴파일 타임아웃(60 s) | `build` | `compile_error` | `compile_timeout` | 0 | 1 |
| build (C) | `nm -D`에 `T`/`W gemm` 없음 | `build` | `signature_mismatch` | `symbol gemm not exported` | 0 | 1 |
| build (py) | `compile(code)` 실패 | `build` | `compile_error` | `SyntaxError: …` | 0 | 1 |
| import (워커 `--mode import`, 로드만) | `ctypes.CDLL` `OSError`(미정의 심벌 등) / python `exec_module` 예외 | **`build`** | `compile_error` | 예외 한 줄(`load_error`) | 0 | 1 |
| import (워커 `--mode import`) | `AttributeError`(gemm 없음) / `gemm` 없음·호출 불가·코루틴(`async def`) | **`build`** | `signature_mismatch` | `symbol gemm not found: …` / `module has no attribute gemm` / `gemm is not callable (…)` / `gemm is a coroutine function (async def)` | 0 | 1 |
| import (워커 `--mode import`) | 워커 프로세스 `timed_out`(상한 `check_timeout_s` = 180 s; 로드 시 무한 루프 등) | **`build`** | `timeout` | `worker timeout after 180s during ? call ?`(import 모드에는 진행 중 테스트가 없다) | 0 | 1 |
| import (워커 `--mode import`) | 워커 rc < 0(로드 시 시그널) / rc == 0인데 `complete`가 아님 / 그 밖의 rc > 0 | **`build`** | `runtime_crash` | check/bench 행과 같은 형식(`signal <name> during ? call ?` 등) | 0 | 1 |
| check/bench | 워커 `timed_out`(check 180 s, bench 300 s) | 단계 | `timeout` | `worker timeout after <t>s during <test> call <idx>`(결과 파일의 `in_progress`에서; 없으면 `?`) | check: 0, **bench: 1** | 1 |
| check/bench | 워커 rc < 0(시그널) | 단계 | `runtime_crash` | `signal <SIGSEGV 등> during <test> call <idx>` | 0 | 1 |
| check/bench | Python 예외/`SystemExit`/`KeyboardInterrupt`(`BaseException`으로 잡음) | 단계 | `runtime_crash` | `exception{type, message}` | 0 | 1 |
| check/bench | rc == 0 인데 결과가 `complete`가 아님 | 단계 | `runtime_crash` | `worker exited without complete result` | 0 | 1 |
| check/bench | 그 밖의 rc > 0 | 단계 | `runtime_crash` | `rc=<n>: <stderr 꼬리>` | 0 | 1 |
| check/bench | rc == 3 또는 결과에 `worker_error`(하네스 자체 버그) | — | **harness_error: 행을 쓰지 않고** traceback 출력 후 run 중단(exit 5) | | | |
| check | §4 합산 결과 | `check` | 합산값 | 패턴 | 0 | 1 |
| bench | `final_check` 실패 / 크래시 / `nondeterministic_runs > 0` | `bench` | 합산값 | 패턴 | 0 | 1 |
| bench | 적응 규칙의 즉시 timeout(첫 호출이 너무 느림) | `bench` | `timeout` | `first_call_<t1>ms_projected_<…>s` | **1** | 1 |

- 주: `error_stage`의 허용값은 `"", generate, extract, build, check, bench`뿐이고 `import`는 없다(설계 §11). 워커 import 모드(로드만)는 extract·build 뒤, check 앞의 첫 워커 실행이며 그 실패는 **모두 `error_stage=build`**로 기록한다(설계 v2.1; 로드 실패는 링크·적재 문제이므로 빌드 범주): 로드 실패는 `compile_error`/`signature_mismatch`, 워커 프로세스 자체의 timeout(`check_timeout_s` 적용)·시그널·비정상 종료는 §8-6 종료 규칙대로 `timeout`/`runtime_crash`다. 따라서 `error_stage=build`인 행의 `error_type`은 `compile_error`·`signature_mismatch`·`timeout`·`runtime_crash` 네 가지가 가능하다.
- **생성 단계 실패는 모델의 코드 오류로 세지 않는다**: API 오류·절단 행은 `valid_sample=0`으로 pass@k 분모에서 빠지고, 거절은 `valid_sample=1`·`pass=0`·`error_type` 빈칸으로 두어 분류표를 바꾸지 않고 수만 따로 센다(집계의 `n_generate_failed`, `n_truncated`, `n_refusal`). 이 행들은 extract/build/실행을 하지 않는다. `generate.py`는 이런 실패 샘플만 재호출한다(`--no-retry-failed`로 끌 수 있음).
- `signature_exact=0`(한정자 `const`/`restrict` 차이, 데코레이터 등)은 **실패가 아니다**. `extract`의 `unterminated_fence`(닫히지 않은 펜스 — 그 뒤 텍스트 전체를 블록으로 보고 계속한다; 실측: 닫히지 않은 ```` ```python ```` 블록 + 올바른 시그니처 → `ok=True`, `signature_ok=True`, `detail=unterminated_fence`), `bare_code`, `multiple_blocks_chose:<n>`, `syntax_error_deferred`(python 구문 오류는 build 단계로 미룸), `preprocess_failed`도 정보일 뿐이며 `detail`에 `;`로 병기된다. 실패 사유(`ok=False`)는 `no_code_block`·`function_not_found`·`param_mismatch:…`·`static_linkage`만이다.
- **`pass` 정의**: `passed = (check 모든 테스트 pass) ∧ ¬inputs_mutated ∧ ¬oob_write ∧ (bench 미실행 ∨ (bench final_check pass ∧ 크래시 없음 ∧ nondeterministic_runs == 0))`. **bench 타임아웃만** `pass=1`·시간 NA·`error_stage=bench, error_type=timeout`이다(성능 실패이지 정확성 실패가 아님). 그 밖의 bench 실패는 `pass=0`.
- `turns_to_pass`: pass면 0(M1은 수정 루프 없음), 아니면 NA.

## 6. 관찰 열 `contract_violation` (분류표 밖)

분류에 영향을 주지 않는 계약 위반 관찰을 JSON 리스트로 기록한다:

- `threads_in_c_seq` — `c_seq` 산출물의 미정의 심벌에 `build.thread_symbols_prefix`(`pthread_create`, `GOMP_`, `omp_`, `__kmpc_`)가 있음(단일 코어에 고정되므로 성능 이점은 없지만 계약 위반).
- `stdout_output` — 커널이 stdout에 무언가 썼음(`stdout_bytes > 0`).
- `forbidden_symbol:<s>` — `build.forbidden_symbols`(`pthread_create`, `clone`, `fork`, `vfork`, `system`, `popen`, `execve`, `execv`, `execl`) 중 하나가 미정의 심벌에 있음.

## 7. M1에서 비어 있는 범주·열

- `memory_error`는 **M5(ASan 재빌드)** 에서만 기록한다. M1의 범위 밖 쓰기는 가드 영역으로 잡아 `wrong_output`/`out_of_bounds_write`가 된다.
- `race_condition`은 M1에서 실행 간 불일치로만 판정하고 TSan은 M5다.
- `asan_findings`, `ubsan_findings`, `lint_*`, `perf_summary`, `roofline_pct` 열은 M1에서 NA다.

## 8. 더미 응답 기대표 (검증 기준, 설계 §13)

### 8.1 gemm (M1)

| language | k | 내용 | 기대 (stage / error_type / detail) |
|---|---|---|---|
| numpy | 0 | `np.matmul(A, B, out=C)` | pass, bench ok |
| numpy | 1 | `def gemm(A, B, M, N, K): return A @ B` | extract / signature_mismatch / param_mismatch |
| numpy | 2 | `C[:] = (A @ B) * 1.01` | check / numeric_instability / imprecise (max_tol_ratio ≈ 10) |
| numpy | 3 | 순수 Python 3중 루프(정확) | pass; bench / timeout (n ≥ 4일 때만) |
| numpy | 4 | `return A @ B` (C 미기록, 시그니처 정상) | check / wrong_output / returned_array_not_written (n ≥ 5) |
| c_seq | 0 | i-k-j 루프, `restrict` | pass |
| c_seq | 1 | 세미콜론 누락 | build / compile_error |
| c_seq | 2 | `C[j*M + i] = …`(전치 저장) | check / arg_order / transposed_output |
| c_openmp | 0 | `#pragma omp parallel for` (i 루프) | pass |
| c_openmp | 1 | NULL 저장 `float *p = NULL; *p = 1.0f;` | check / runtime_crash / signal SIGSEGV |
| c_openmp | 2 | k 루프 병렬화 + 공유 누산 | check / race_condition |

### 8.2 Tier 1 (M2a) · Tier 2 (M2b)

Tier 1(`jacobi2d`, `spmv_csr`, `dot`)은 세 언어의 정답(k=0)만 있다(`dummy-m2`: 9행 모두 pass). Tier 2(설계 §17; `harness/dummy_responses/README.md` 와 과제별 README 에 실측 포함)는 정답 12개 + 경쟁 조건 변형 2개다. `k = sample_idx mod 파일 수` 이므로 `--n 2` 로 돌리면 sample 1 이 변형을 밟는다.

| task | language | k | 내용 | 기대 (stage / error_type / detail) |
|---|---|---|---|---|
| histogram | numpy | 0 | `hist[...] = np.bincount(data, minlength=nbins)` | pass |
| histogram | c_seq | 0 | `memset` 0 초기화 + `hist[data[i]]++` | pass |
| histogram | c_openmp | 0 | 스레드별 private 히스토그램(calloc 스크래치) → `omp for` 빈 합산 | pass |
| histogram | c_openmp | 1 | 0 초기화 뒤 보호 없는 `#pragma omp parallel for` 공유 `hist[data[i]]++` | check / race_condition / `nondeterministic:<rect,small…>` (유실된 갱신이 두 호출에서 우연히 같으면 wrong_output / mismatch; tiny(n=13)는 충돌이 거의 없어 pass 일 수 있다) |
| scan | numpy | 0 | `np.cumsum(x, dtype=np.int32, out=out)` | pass |
| scan | c_seq | 0 | 순차 누적 `acc += x[i]; out[i] = acc` | pass |
| scan | c_openmp | 0 | 2-pass 블록 스캔(블록 합 → exclusive 오프셋 → 블록 내 스캔) | pass |
| scan | c_openmp | 1 | `out[i] = out[i-1] + x[i]` 를 `parallel for` 로 — 블록 경계에서 미기록 센티널을 읽음 | check / race_condition / `nondeterministic:<목록>` (어느 형태가 불일치인지는 타이밍 의존; 세 형태 모두 같게 틀리면 wrong_output / mismatch) |
| nbody | numpy | 0 | float64 로 256행씩 브로드캐스트 직접 합산 → `acc[i0:i1] = …` | pass |
| nbody | c_seq | 0 | i·j 2중 루프, `1.0f/sqrtf`, AoS | pass (max_tol_ratio ≤ 0.2 계약) |
| nbody | c_openmp | 0 | 위 커널에 `#pragma omp parallel for schedule(static)` (= 기준선과 같은 코드) | pass, speedup ≈ 1 |
| sort | numpy | 0 | `out[...] = np.sort(x)` (= 기준선) | pass, speedup ≈ 1 |
| sort | c_seq | 0 | `memcpy` + `qsort` | pass (large 약 1.3 s/호출, 1스레드) |
| sort | c_openmp | 0 | 8비트 4패스 병렬 LSD 기수 정렬(스레드별 히스토그램 → 안정 분산) | pass, speedup > 1 |

## 9. 린트 범주 (M5, 안전성 요약 — CLAUDE.md §6)

하네스는 린트의 내부를 모르고 계약만 안다. 호출과 출력 형식:

```
plugins/lint/run_lint.py --language <lang> --file <path> --task <task> --signature <json> --out <json>
```

```json
{
  "tool": "mllint-v0",
  "findings": [
    {"rule": "numeric/accumulate-in-float32", "severity": "warning", "line": 12,
     "message": "float32 누적으로 큰 K에서 오차 증가", "category": "numeric"}
  ],
  "summary": {"memory": 0, "numeric": 1, "concurrency": 0, "shape_dtype": 0, "api_misuse": 0, "performance": 2}
}
```

범주 정의(변경 시 이 문서 갱신):

| category | 정의 | 행 열 |
|---|---|---|
| `memory` | 경계·초기화·별칭(aliasing) | `lint_memory` |
| `numeric` | NaN/Inf 위험, 정수 오버플로, 누적 정밀도, 부동소수 비교 | `lint_numeric` |
| `concurrency` | 병렬 영역 내 공유 쓰기, 리덕션 누락, 비결정성 | `lint_concurrency` |
| `shape_dtype` | 차원·브로드캐스트·dtype 암묵 변환 | `lint_shape_dtype` |
| `api_misuse` | 시그니처 위반, 입력 변형, 전역 상태 | `lint_api_misuse` |
| `performance` | 배열 원소 Python 루프, 불필요한 복사, 비연속 접근(안전성은 아니지만 같은 도구로 뽑는 성능 냄새) | `lint_performance` |

`lint_findings_total`은 findings 수. 쓰임: (1) `lint_*` 열, (2) `feedback=compiler+lint` 조건에서 수정 프롬프트에 `findings[].message`를 넣음(M4). 플러그인이 없으면 빈 결과로 진행하고 경고만 남긴다(실험이 막히면 안 된다). 린트 범주는 **오류 분류표(§1)와 별개**이며 `error_type`에 영향을 주지 않는다.

## 부록: 과제 일반화 뒤의 적용 범위 (2026-11-05, M2)

- 규칙 1–6과 샘플 합산은 모든 과제에 그대로 적용한다(`classify_generic`).
- arg_order 패턴: `gemm`은 §2의 4개 패턴, 그 밖의 과제는 2-D 출력의 `transposed_output`만 본다. 과제별 패턴을 추가하면 이 문서를 갱신한다.
- `dot`은 출력이 스칼라 하나라 `numeric_pattern_tol_ratio`가 500이다: 허용오차는 넘지만 상대오차 약 50 % 이내면 `numeric_instability`/`imprecise`(float32 누적 정밀도 부족이 대표 사례), 그 밖은 `wrong_output`.
- 입력 변조와 범위 밖 쓰기 검사는 정수 입력 배열(`row_ptr`, `col_idx`)에도 적용한다.

## 부록 2: Tier 2 — 정수 출력·nbody 센티널·새 패턴 (2026-11-08, M2b; 설계 §17)

**정수 출력(`histogram`, `scan`, `sort`; 출력 dtype int32).**

- `tolerance.yaml` 은 `rtol: 0.0, atol: 0.0, numeric_pattern_tol_ratio: 0.0, sentinel: -2048.0` 이고 spec 은 "exact equality" 를 요구한다. `compare` 는 허용오차 0 에서 틀린 원소의 배율(`max_tol_ratio`)을 **inf** 로 두므로(`err > 0 → err / 0`), 규칙 5(`imprecise`)는 절대 성립하지 않는다 — 정수 출력은 `pass` 아니면 `wrong_output`(또는 아래 `race_condition`)이다. 집계에서 정수 과제의 `numeric_instability` 는 0 이어야 하며, 0 이 아니면 하네스 버그다.
- 출력 버퍼는 int32 배열이고 센티널은 정수 `-2048`(`alloc_aligned` 가 `dtype.type(sentinel)` 로 채운다). 세 과제의 정당한 출력은 모두 음이 아니므로(개수·음이 아닌 입력의 접두합·[0, 2³¹−1] 의 정렬) 센티널이 정당한 값일 수 없고, 규칙 4a–4c(`output_not_written`·`accumulated_into_c`·`partial_write`)는 float32 와 똑같이 작동한다. 누적 검사는 `out − (−2048) == ref` 의 정확 일치다.
- **4b 가 4c 보다 먼저인 이유가 정수에서도 생긴다**: `hist += counts` 처럼 센티널 위에 더한 히스토그램에서 개수 0 인 빈은 정확히 `-2048` 로 남아 `n_unwritten` 에 세어진다(histogram tiny 의 빈 1: `n_unwritten=1`). 순서가 반대였다면 이 커널은 `partial_write:1/4` 로 잘못 기록됐을 것이다(`tests/test_tier2_histogram.py`).
- 1-D 출력에는 arg_order 패턴이 없다(`transposed_output` 은 2-D 출력에만). 따라서 세 과제의 `error_type` 은 `pass`·`wrong_output`·`race_condition`·`runtime_crash`·`timeout`·`compile_error`·`signature_mismatch` 만 가능하다.
- 호출 간 불일치(§3)도 허용오차 0 으로 비교하므로 두 호출의 출력이 **비트 단위로 같아야** 한다. 보호 없는 병렬 `hist[data[i]]++`(histogram c_openmp/1)나 순차 의존을 무시한 `parallel for` 스캔(scan c_openmp/1)은 유실·미기록 값이 호출마다 달라 `race_condition`/`nondeterministic:<테스트…>` 가 되고, 우연히 같게 틀리면 `wrong_output`/`mismatch` 다 — 두 결과 모두 "틀린 병렬화" 이며 더미 README 가 둘 다 허용한다. `race_condition` 은 샘플 합산 규칙 2 라 `wrong_output` 보다 먼저다.
- `gflops` 는 세 과제 모두 NA(`work` 없음); `nbody` 만 `work: 20*n*n`.

**`nbody`(float32 (n, 3) 출력).**

- 센티널은 `-1.0e6`(2026-11-08 변경; 처음 `-1.0e30` 은 float32 덧셈을 흡수해 누적 커널이 '미기록'으로 보였다). 성능 형태(n=16384)의 참조 가속도 성분은 최대 |acc| = 45,663 이라 `-2048` 은 정당한 값일 수 있어 쓰지 않는다. atol 은 형태마다 `1e-3 × RMS(참조 성분)`(large 18.1, tiny 0.0032; 부록 3).
- 2-D 출력이므로 공통 `transposed_output` 패턴이 적용된다: SoA 로 쓴 출력(`acc[c*n + i]`)은 `out.reshape(3, n).T ≈ ref` 라 `arg_order`/`transposed_output` 이다(의도된 분류; `tests/test_tier2_nbody.py` 가 python·C 커널로 단언). `n = 5` 에서는 전치 뷰 (3, 5)ᵀ 와의 일치가 우연히 성립할 가능성이 있어 단위 테스트는 rect(n=301)에서 검사한다.
- 입력이 `pos`(n×3)·`mass`(n) 두 개라 GEMM 의 `swapped_operands`·`column_major_all`·`swapped_MN_dims` 는 해당 없음.

**`sort`.** 기준선이 python(`np.sort`, 단일 스레드)이고 `gate_blas: false` 라 기준선 게이트는 완료·`final_check` pass 만 본다(BLAS 정체 무관). 병렬 정렬 커널의 `speedup_vs_baseline > 1` 은 정상이다(dot 과 같은 해석, decisions.md 2026-11-08).

## 부록 3: 2026-11-08 보강 (Tier 2 리뷰 반영)

- **규칙 4c `partial_write` 조건**: 센티널로 남은 원소가 있고 **쓴 원소가 전부 참조와 맞을 때만** `partial_write:<k>/<n>`. 쓴 원소도 틀리면 OOB → `out_of_bounds_write`, 아니면 규칙 5·6(`imprecise`/`mismatch`)으로 간다. 틀린 값이 우연히 센티널과 같아도 더 이상 '부분 기록'으로 라벨되지 않는다.
- **가드**: 모든 배열에 앞쪽 4096원소·뒤쪽 1024원소(출력은 과제별 추가분 포함) 가드가 있다. 입력 가드 값은 출력 센티널과 다르다(정수 -1073741824, 실수 -3e9). 음수·원거리 인덱스 쓰기가 가드를 벗어나 힙을 깨뜨리면 워커가 종료 중 죽으며, run.py 는 `runtime_crash`/`signal … after complete result (heap corruption suspected)` 로 적는다.
- **형태별 허용오차**: `tolerance.yaml` 에 `atol_rms_rel` 이 있으면 테스트마다 `atol = atol_rms_rel × RMS(참조)` 를 쓴다(nbody). 적용값은 check/bench 문서의 `atol_effective`.
- **nbody 센티널은 -1e6**(-1e30 은 덧셈을 흡수해 누적 커널이 '미기록'으로 보였다).
- **JSON**: 비유한 실수(정수 과제의 배율 inf 등)는 결과 JSON 에 null 로 기록한다. CSV 의 `check_max_tol_ratio` 는 `inf` 문자열.

## 부록 4: Tier 3 (2026-11-09)

- `cholesky`·`fft` 는 실수 출력(형태별 atol), `floyd_warshall` 은 고정 atol 1e-3(거리 1~수십), `bfs` 는 정수 정확 일치. 2-D 출력(cholesky, floyd_warshall)에는 공통 `transposed_output` 패턴이 적용된다(Lᵀ 를 쓴 Cholesky 는 `arg_order`).
- `floyd_warshall` 의 INF(1e30)는 허용오차 안에서 비교되므로 도달 불가 쌍에 2·INF 를 쓰면 `wrong_output/mismatch`.
- `fft` 센티널은 -1e9(출력 크기가 n 까지 커질 수 있어 -2048 을 피함).
