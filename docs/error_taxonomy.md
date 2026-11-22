# 오류 분류표 (docs/error_taxonomy.md)

`CLAUDE.md` §1의 분류표는 실험 전에 고정됐고 **변경 금지**다. 이 문서는 그 아홉 범주 각각이 코드(`harness/classify.py`, `harness/worker.py`, `harness/kernels/*.py`, `harness/run.py`, `harness/safety.py`, `harness/extract.py`, `harness/build/*.py`)에서 **정확히 어떤 규칙으로** 결정되는지를 적는다(설계 계약 `docs/harness_design.md` §10, `docs/harness_design_v3.md` §8.2–§8.5·§9.2). 규칙을 바꾸면 이 문서를 같이 고친다.

새 언어(`numba`, `halide_auto`, `halide_manual`, `rust`), ASan/UBSan, 린트는 아홉 범주를 바꾸지 않는다. 더하는 것은 `error_detail` 의 새 값(§5), sanitizer 의 `memory_error` 재분류(§5a), 관찰 열(§6, §7, §9, §10)과 적용 범위(부록 5)다(docs/decisions.md 2026-11-12·2026-11-23). sanitizer 는 Linux 에서만 돈다. sanitizer 와 C 린트 도구(cppcheck·clang-tidy)에 관한 부분은 코드와 손으로 만든 로그로 단위 테스트한 규칙이고, 실제 gcc·cppcheck·clang-tidy 출력으로는 아직 확인하지 않았다.

## 1. 고정 분류표 (아홉 범주)

| `error_type` | 정의(CLAUDE.md) | 결정되는 단계 |
|---|---|---|
| `compile_error` | 빌드 실패(컴파일·링크 오류, 컴파일 타임아웃; Rust 는 커널 크레이트의 컴파일 오류만), 또는 워커(import 모드)가 산출물을 로드하지 못함(Python 계열은 모듈 임포트 예외, Halide 는 첫 정확성 형태에서 빌드 함수 호출·파이프라인 구성·오토스케줄·JIT 실패) | build (워커 import 모드의 로드 실패도 `error_stage=build`) |
| `runtime_crash` | 실행 중 시그널 종료·예외·비정상 종료(Rust 커널의 panic 은 상세 `rust_panic:<메시지>`) | build(import 워커의 시그널·비정상 종료), check, bench |
| `timeout` | 단계 시간 상한 초과(import 워커 180 s = `check_timeout_s`, check 180 s, bench 300 s) | build(import 워커), check, bench |
| `wrong_output` | 허용오차 밖 결과 중 패턴이 수치적이지도 인자 순서도 아닌 것(미기록·부분 기록·누적·범위 밖 쓰기·입력 변조 포함) | check, bench |
| `numeric_instability` | NaN/Inf, 또는 허용오차 초과이면서 패턴이 수치적(모든 원소가 허용오차의 20배 이내) | check, bench |
| `arg_order` | 인자 순서·차원 뒤바뀜으로 판정되는 wrong_output(전치·열우선·피연산자 교환·M/N 교환 시 일치) | check, bench |
| `memory_error` | ASan 탐지(M5): `safety.enabled` 이고 C 커널(`c_seq`, `c_openmp`)인 샘플에서, check 또는 bench 라운드의 판정이 `runtime_crash` 또는 `wrong_output` 인데 sanitizer 실행의 ASan 보고가 1건 이상이면 재분류(§5a). 통과한 샘플은 바꾸지 않는다. 가드 영역으로 잡는 OOB 쓰기는 sanitizer 가 꺼져 있으면 여전히 `wrong_output`/`out_of_bounds_write` | check, bench (재분류 전 판정의 단계 그대로) |
| `race_condition` | 반복 실행 간 결과 불일치(허용오차 초과; 허용오차가 0 인 int32 출력 과제는 한 원소라도 다름). TSan 은 쓰지 않는다(libgomp 가 계측되지 않아 거짓 양성) | check, bench |
| `signature_mismatch` | 요구한 함수 시그니처를 지키지 않음(추출 단계 시그니처 검사 실패, 심벌 미노출, 함수 없음·호출 불가, Numba 의 명시 서명·`cache` 규칙 위반, Numba 디스패처가 하네스의 인자 타입을 컴파일 없이 받지 않음, Halide 빌드 함수의 반환값이 출력 계약과 다름) | extract, build(심벌 미노출·Numba 데코레이터; import 워커의 함수 없음·호출 불가·서명 없는 디스패처·인자 타입 검사·Halide 반환값 검사도 `error_stage=build`) |

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

- check: 테스트마다 `check_repeats = 2`회 호출. `nondeterministic = not np.allclose(out1, out2, rtol, atol, equal_nan=True) or (isfinite 마스크 불일치)`. 두 출력이 비트만 다르고 허용오차 안이면 불일치가 **아니다**(누적 순서 차이는 허용). int32 출력 과제는 `rtol = atol = 0` 이라 두 출력이 정확히 같아야 한다(부록 2).
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
| extract | `extract().ok == False` | `extract` | `signature_mismatch` | `no_code_block` / `function_not_found` / `param_mismatch:<설명>`(python `async def` 는 `param_mismatch:async_def`) / `static_linkage`(C) / `no_mangle_missing`(Rust) (정보 노트 `unterminated_fence` 등이 `;`로 병기될 수 있음 — 그 자체는 실패 사유가 아니다, 아래 주) | 0 | 1 |
| build (C) | gcc rc ≠ 0 | `build` | `compile_error` | `error:` 포함 첫 3줄을 ` \| ` 로 이음(없으면 마지막 3줄; 컴파일러가 시그널로 죽었으면 `compiler died with <SIG>: …`) | 0 | 1 |
| build (C·Rust) | 컴파일 타임아웃(C 60 s, Rust `build.rust.build_timeout_s` = 600 s; Rust 는 `Compiling kernel` 뒤의 시간 초과만) | `build` | `compile_error` | `compile_timeout` | 0 | 1 |
| build (C·Rust) | `nm -D --defined-only`에 `T`/`W` `<fn>` 없음(Rust 는 산출물이 ELF 이고 `nm` 이 있을 때만 검사 — 개발용 Windows 의 DLL 은 건너뛴다) | `build` | `signature_mismatch` | `symbol <fn> not exported` | 0 | 1 |
| build (Rust) | 로그에 ``error: could not compile `kernel` `` | `build` | `compile_error` | rustc 오류 첫 3개(`error[E0425]: … --> src/lib.rs:LINE:COL`)를 ` \| ` 로 이음 | 0 | 1 |
| build (py·numba·halide) | `compile(code)` 실패 | `build` | `compile_error` | `SyntaxError: …` | 0 | 1 |
| build (numba·halide) | 최상위 `def <fn>` 없음(extract 를 거쳤으면 나오지 않는다) | `build` | `signature_mismatch` | `function_not_found` | 0 | 1 |
| build (numba) | 과제 함수에 Numba 의 `njit`/`jit` 데코레이터가 없음, 또는 데코레이터가 `jit` 인데 `nopython=True` 리터럴이 없음 | `build` | `signature_mismatch` | `numba_signature_missing` | 0 | 1 |
| build (numba) | 모듈 어디서든(최상위·중첩 함수·클래스 본문, 데코레이터와 `njit(f)` 호출 꼴 모두) `njit`/`jit` 로 컴파일되는 함수가 명시 서명을 첫 위치 인자(또는 `signature_or_function=`)로 받지 않음. `signature=` 키워드는 서명으로 치지 않는다(Numba 0.68 은 이 키워드를 쓰지 않고 지연 컴파일한다) | `build` | `signature_mismatch` | `numba_unsigned_jit:<함수 이름>`(소스 순서로 첫 번째) | 0 | 1 |
| build (numba) | 어느 `njit`/`jit` 에든 `cache=` 가 리터럴 `False`/`None` 이 아님(`cache=True`, 식) | `build` | `signature_mismatch` | `numba_cache_true` | 0 | 1 |
| import (워커 `--mode import`, 로드만) | `ctypes.CDLL` `OSError`(미정의 심벌 등) / python·numba·halide `exec_module` 예외 | **`build`** | `compile_error` | 예외 한 줄(`load_error`) | 0 | 1 |
| import (워커 `--mode import`) | `AttributeError`(함수 없음) / 함수 없음·호출 불가·코루틴(`async def`) | **`build`** | `signature_mismatch` | `symbol <fn> not found: …` / `module has no attribute <fn>` / `<fn> is not callable (…)` / `<fn> is a coroutine function (async def)` | 0 | 1 |
| import (numba) | 로드한 함수가 Numba 디스패처가 아니거나 `signatures` 가 비어 있음(지연 컴파일; Numba 가 무시하는 `signature=` 키워드 포함) | **`build`** | `signature_mismatch` | `numba_signature_missing` | 0 | 1 |
| import (numba) | 디스패처의 새 컴파일을 끈 뒤, 하네스가 넘길 인자 타입을 컴파일 없이 받는 서명이 없거나 둘 이상이라 모호함(예: float64 서명) | **`build`** | `signature_mismatch` | `numba_signature_rejects_args: no compiled signature of <fn> accepts (<타입…>)` / `numba_signature_rejects_args: ambiguous compiled signatures for (<타입…>)` | 0 | 1 |
| import (halide, 첫 정확성 형태) | 빌드 함수가 예외를 올림 / 파이프라인 구성·오토스케줄·JIT 실패 | **`build`** | `compile_error` | 예외 한 줄 / `pipeline construction failed: …` / `autoscheduler <이름> failed: …` / `JIT compilation failed: …` | 0 | 1 |
| import (halide, 첫 정확성 형태) | 반환값이 `hl.Func` 가 아님·정의 없음·Tuple·차원 수 또는 원소 타입이 출력과 다름 | **`build`** | `signature_mismatch` | `<fn> returned <type>; expected hl.Func` / `<fn> returned an hl.Func that has no definition` / `… with <k> outputs (a Tuple); expected one` / `… with <d> dimension(s); the output <out> has <n>` / `… of type <t>; the output <out> is <dtype>` | 0 | 1 |
| check/bench (halide) | 그 테스트 형태의 파이프라인 빌드 실패(위 두 줄과 같은 원인; 성능 형태 등 import 에서 빌드해 보지 않은 크기에서 주로 생긴다) | 단계 | `runtime_crash` | check 는 `<test>: <위 두 줄의 상세>`(예: `rect: pipeline construction failed: …`), bench 는 `<위 두 줄의 상세>`. 반환값 검사 실패는 결과 문서의 예외 기록 `exception.type` 이 `signature_mismatch` 다 | 0 | 1 |
| import (워커 `--mode import`) | 워커 프로세스 `timed_out`(상한 `check_timeout_s` = 180 s; 로드 시 무한 루프 등) | **`build`** | `timeout` | `worker timeout after 180s during ? call ?`(import 모드에는 진행 중 호출이 없다. halide 는 첫 정확성 형태를 빌드하는 동안 `in_progress.test` 가 그 형태라 `during tiny call ?`) | 0 | 1 |
| import (워커 `--mode import`) | 워커 rc < 0(로드 시 시그널) / rc == 0인데 `complete`가 아님 / 그 밖의 rc > 0 | **`build`** | `runtime_crash` | check/bench 행과 같은 형식(`signal <name> during ? call ?` 등) | 0 | 1 |
| check/bench | 워커 `timed_out`(check 180 s, bench 300 s) | 단계 | `timeout` | `worker timeout after <t>s during <test> call <idx>`(결과 파일의 `in_progress`에서; 없으면 `?`) | check: 0, **bench: 1** | 1 |
| check/bench | 워커 rc < 0(시그널) | 단계 | `runtime_crash` | `signal <SIGSEGV 등> during <test> call <idx>` | 0 | 1 |
| check/bench | Python 예외/`SystemExit`/`KeyboardInterrupt`(`BaseException`으로 잡음) | 단계 | `runtime_crash` | `exception{type, message}` | 0 | 1 |
| check/bench | rc == 0 인데 결과가 `complete`가 아님 | 단계 | `runtime_crash` | `worker exited without complete result` | 0 | 1 |
| check/bench | 그 밖의 rc > 0 (결과 문서가 있고 `worker_error` 가 없으면 rc == 3 도 여기: 커널이 `exit(3)` 을 부른 것) | 단계 | `runtime_crash` | `rc=<n>: <stderr 꼬리>` | 0 | 1 |
| import·check·bench (rust) | 위 규칙의 `runtime_crash` 이고 워커 stderr(`<단계>.stderr.log` 전문, 없으면 꼬리)에 Rust panic 이 있음(`panic = "abort"` 라 워커는 abort 로 끝난다 — Linux 에서는 SIGABRT) | 단계(import 는 **`build`**) | `runtime_crash` | `rust_panic:<메시지 첫 줄>`(300자까지; 위치 `<파일>:<줄>:<열>` 은 `result.json` 의 `stages.<단계>.rust_panic`). panic 이 없으면 위 규칙의 상세 그대로 | 0 (bench 도 0) | 1 |
| 모든 워커 단계 | 결과에 `worker_error`(하네스 자체 버그, 또는 백엔드 준비 중의 종료) / 결과 문서를 한 번도 쓰지 못하고 끝남 | — | **harness_error: 행을 쓰지 않고** traceback 출력 후 run 중단(exit 5) | | | |
| setup·build·import | 도구가 없거나 쓸 수 없음(준비 단계의 도구 점검·시험 실행 실패 — numba 워커에서 SciPy 가 임포트되는 경우 포함 —, 또는 빌드·import 중의 `ToolchainMissing`: 컴파일러·cargo 없음, Rust 의존성 캐시·링커·잠금 문제, Halide 오토스케줄러 로드 실패 `halide_autoscheduler_unavailable:<name>`) | — | **행을 쓰지 않고** run 중단(exit 7) | | | |
| check | §4 합산 결과 | `check` | 합산값 | 패턴 | 0 | 1 |
| bench | `final_check` 실패 / 크래시 / `nondeterministic_runs > 0` | `bench` | 합산값 | 패턴 | 0 | 1 |
| bench | 적응 규칙의 즉시 timeout(첫 호출이 너무 느림) | `bench` | `timeout` | `first_call_<t1>ms_projected_<…>s` | **1** | 1 |

- 주: `error_stage`의 허용값은 `"", generate, extract, build, check, bench`뿐이고 `import`는 없다(설계 §11). 워커 import 모드(로드만)는 extract·build 뒤, check 앞의 첫 워커 실행이며 그 실패는 **모두 `error_stage=build`**로 기록한다(설계 v2.1; 로드 실패는 링크·적재 문제이므로 빌드 범주): 로드 실패는 `compile_error`/`signature_mismatch`, 워커 프로세스 자체의 timeout(`check_timeout_s` 적용)·시그널·비정상 종료는 §8-6 종료 규칙대로 `timeout`/`runtime_crash`다. 따라서 `error_stage=build`인 행의 `error_type`은 `compile_error`·`signature_mismatch`·`timeout`·`runtime_crash` 네 가지가 가능하다.
- **생성 단계 실패는 모델의 코드 오류로 세지 않는다**: API 오류·절단 행은 `valid_sample=0`으로 pass@k 분모에서 빠지고, 거절은 `valid_sample=1`·`pass=0`·`error_type` 빈칸으로 두어 분류표를 바꾸지 않고 수만 따로 센다(집계의 `n_generate_failed`, `n_truncated`, `n_refusal`). 이 행들은 extract/build/실행을 하지 않는다. `generate.py`는 이런 실패 샘플만 재호출한다(`--no-retry-failed`로 끌 수 있음).
- `signature_exact=0`(한정자 `const`/`restrict` 차이, typedef, 데코레이터, Rust 의 FFI 별칭 등)은 **실패가 아니다**. C 의 기본 타입은 한정자를 뺀 단어열 그대로 비교한다 — `int32_t`·`signed int` 같은 다른 철자는 `int` 와 달라 `param_mismatch` 다(decisions.md 2026-11-23). Rust 는 ABI 가 같은 FFI 별칭 `c_int`(= `i32`)·`c_float`(= `f32`; 접두 없음 또는 `std::ffi::`·`core::ffi::`·`std::os::raw::`·`libc::`)만 같게 보고, 목록에 없는 철자(`::std::ffi::c_int` 등)는 `param_mismatch` 다. `extract`의 `unterminated_fence`(닫히지 않은 펜스 — 그 뒤 텍스트 전체를 블록으로 보고 계속한다; 실측: 닫히지 않은 ```` ```python ```` 블록 + 올바른 시그니처 → `ok=True`, `signature_ok=True`, `detail=unterminated_fence`), `bare_code`, `multiple_blocks_chose:<n>`, `syntax_error_deferred`(python 구문 오류는 build 단계로 미룸), `preprocess_failed`도 정보일 뿐이며 `detail`에 `;`로 병기된다. 실패 사유(`ok=False`)는 `no_code_block`·`function_not_found`·`param_mismatch:…`·`static_linkage`(C)·`no_mangle_missing`(Rust)뿐이다.
- **`pass` 정의**: `passed = (check 모든 테스트 pass) ∧ ¬inputs_mutated ∧ ¬oob_write ∧ (bench 미실행 ∨ (bench final_check pass ∧ 크래시 없음 ∧ nondeterministic_runs == 0))`. **bench 타임아웃만** `pass=1`·시간 NA·`error_stage=bench, error_type=timeout`이다(성능 실패이지 정확성 실패가 아님). 그 밖의 bench 실패는 `pass=0`.
- `turns_to_pass`: 통과한 행이면 그 행의 턴 번호(턴 0 = 0, 수정 턴 t = t; 최대 3), 아니면 NA. 수정 턴 행(`turn ≥ 1`)의 분류 규칙은 턴 0 과 같다.
- import 워커가 `load_ok=true` 를 쓴 뒤 관찰값(`--inspect`) 수집 중에 끝나면(성능 형태 빌드의 크래시·시간 초과) **판정이 아니다**: 경고하고 관찰값 없이 check·bench 로 간다. 성능 형태에서 정말 죽는 커널은 bench 가 같은 자리에서 `runtime_crash`/`timeout` 으로 잡는다.

## 5a. `memory_error` 재분류 (M5, `harness/run.py`)

- 조건: `safety.enabled` 이고 백엔드가 C(`c_seq`, `c_openmp`)이며 빌드가 성공한 샘플. check 워커가 끝난 뒤(통과 여부와 무관) `safety.run_sanitizers` 가 ASan/UBSan 빌드로 check 를 한 번 더 돌린다. check 판정이 `timeout` 인 샘플은 건너뛴다(계측 빌드는 더 느려 같은 상한에 또 걸린다). Python·Numba·Halide·Rust 는 sanitizer 를 돌리지 않는다.
- 규칙: 그 턴의 판정(check 단계, 또는 check 를 통과한 뒤 bench 라운드 — 라운드 2 이후 포함 — 의 판정)이 `runtime_crash` 또는 `wrong_output` 이고 `asan_findings > 0` 이면 `error_type = memory_error`, `error_detail = "asan:<첫 ASan 보고의 종류>; was <이전 error_type>/<이전 error_detail>"`(예: `asan:heap-buffer-overflow; was wrong_output/out_of_bounds_write`). `error_stage` 는 이전 판정의 것 그대로다.
- 바꾸지 않는 것: 통과한 샘플의 `pass`(잠재 결함은 `asan_findings` 열로만 남는다), `numeric_instability`·`arg_order`·`race_condition`·`timeout`·`compile_error`·`signature_mismatch` 판정, UBSan 보고(`ubsan_findings` 열에만 남고 분류를 바꾸지 않는다).
- 수정 루프의 요청은 재분류 전 판정으로 쓴다(`; was …` 에서 되돌린다) — sanitizer 는 측정이지 피드백이 아니다.

## 5b. 행이 되지 않는 실패

다음은 생성된 코드의 잘못이 아니므로 **행을 쓰지 않고** run 을 멈춘다. 처리 중이던 샘플의 `result.json`·CSV 행도 남지 않는다. 고친 뒤 같은 `--run-id` 로 다시 돌리면 이어서 처리한다.

| 종료 코드 | 원인 |
|---|---|
| 5 (`EXIT_HARNESS`) | 워커가 `worker_error` 를 남김(하네스 버그; 백엔드 준비 구간 — 도구 임포트, Numba 스레딩 계층 시작, 오토스케줄러 시험 — 의 프로세스 종료 포함) 또는 결과 문서를 한 번도 쓰지 못하고 끝남(반환 코드와 무관). 기준선 워커의 같은 경우도 5. 샘플 처리 중의 예기치 않은 예외(하네스 버그)도 5 |
| 7 (`EXIT_TOOLCHAIN`) | 처리할 언어에 필요한 도구가 없거나 쓸 수 없음: `build.cc`(bench 를 하는 run 에 C 기준선 과제가 있을 때 포함), cargo·rustc, numba·halide 임포트, 오토스케줄러 로드, 실행 접두(taskset 코어) 시험, 언어 런타임 시험(halide_auto 오토스케줄러·매개변수, `NUMBA_THREADING_LAYER`), numba 워커 인터프리터에서 `scipy` 가 임포트됨, Rust 자리표시 크레이트 빌드, 빌드 도중의 `ToolchainMissing`, import 워커의 `halide_autoscheduler_unavailable:<name>` |

그 밖에 run 을 멈추는 코드는 1(입력 없음 등), 2(설정 오류), 4(`--strict-protocol` 위반·BLAS 가 OpenBLAS 아님), 6(기준선 게이트 실패), 130(중단)이고, 모두 행을 쓰기 전이거나 처리 중이던 샘플의 행을 쓰지 않는다. 생성 단계 실패(API 오류·절단·거절)는 이 표가 아니라 §5 의 generate 행으로 **기록된다**.

## 6. 관찰 열 `contract_violation` (분류표 밖)

분류에 영향을 주지 않는 계약 위반 관찰을 JSON 리스트로 기록한다:

- `threads_in_c_seq` — `c_seq` 산출물의 미정의 심벌에 `build.thread_symbols_prefix`(`pthread_create`, `GOMP_`, `omp_`, `__kmpc_`)가 있음(단일 코어에 고정되므로 성능 이점은 없지만 계약 위반).
- `stdout_output` — 커널이 stdout에 무언가 썼음(`stdout_bytes > 0`).
- `forbidden_symbol:<s>` — `build.forbidden_symbols`(`pthread_create`, `clone`, `fork`, `vfork`, `system`, `popen`, `execve`, `execv`, `execl`) 중 하나가 미정의 심벌에 있음.
- `schedule_in_auto` — `halide_auto`(알고리즘만 쓰고 하네스가 오토스케줄러를 적용하는 조건)에서 모델이 스케줄을 썼음: 소스에 Halide 스케줄 호출(`harness/build/halide.py` 의 `SCHEDULE_METHODS` 60개 메서드 이름 중 하나)이 있거나, 빌드 함수가 도는 동안 워커가 무동작으로 바꿔 둔 `hl.Func`·`hl.Stage` 의 스케줄 메서드나 `Pipeline.apply_autoscheduler` 가 불렸다(`import.json` 의 `ignored_schedule_calls`). 그 지시는 **무시하고 기록할 뿐** 오류가 아니다 — 빌드·판정(pass/fail)은 그대로이고, 측정되는 스케줄은 오토스케줄러의 것이다(decisions.md 2026-11-12·2026-11-23).

값은 이 네 종류뿐이다(`forbidden_symbol:<s>` 는 심벌마다 하나씩). numba 빌드가 `njit` 또는 `jit(nopython=True)` 만 통과시키므로 object mode 는 관찰 대상이 아니다.

## 7. 비어 있을 수 있는 열과 NA 의 뜻

아래 열에서 **NA 는 "검사하지 않음 / 판정 불가", 0 은 "검사했고 없음"** 이다. 0 은 검사가 실제로 끝까지 돈 경우에만 나온다.

| 열 | 값이 있는 경우 | NA 인 경우 |
|---|---|---|
| `asan_findings`, `ubsan_findings` | `safety.enabled` 이고 C 커널(`c_seq`, `c_openmp`)이며 sanitizer 빌드의 check 가 커널을 로드해 끝까지 돌았다(보고가 있으면 끝까지 못 갔어도 그 수) | sanitizer 를 돌리지 않음(`safety.enabled: false` — 기본값 —, C 가 아닌 언어, check timeout, check 전에 끝난 샘플: `stages.sanitize` 가 없다; 비-Linux 는 `detail = unsupported_platform:<p>`), sanitizer 빌드 실패(`build_failed:<kind>: <detail>`), 보고 없이 로드 실패(`load_failed:…`)나 미완료(`incomplete:…`). `detail` 은 `result.json` 의 `stages.sanitize.detail` |
| `lint_findings_total`, `lint_<범주>` | 린트 플러그인이 돌았고 그 범주를 볼 수 있는 도구가 끝까지 돌았다(§9) | 린트 꺼짐(`lint.enabled: false`), 빌드 전에 끝난 행(생성 단계·extract 실패), 플러그인 실패, `tool == "none"`(Rust 는 늘; 도구 미설치; 구문·컴파일 오류로 분석 못 함), 그 범주를 보는 도구가 돌지 않음(그 열만) |
| `perf_summary` | `perf_summary.enabled` 이고 빌드·로드(import)가 성공했다. 정할 수 없는 값은 JSON 안에서 `"unknown"`/`null`/`[]` | 꺼짐, 그 전 단계에서 실패 |
| `roofline_pct` | — | 아직 채우지 않는다(`perf stat` 을 쓸 수 없는 머신; M6 이후) |

- `race_condition` 은 실행 간 불일치로만 판정한다. TSan 은 쓰지 않는다.
- `memory_error` 는 §5a 의 재분류로만 생긴다. sanitizer 가 꺼진 run(기본값)과 C 가 아닌 언어에는 나오지 않는다.

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

### 8.3 새 언어 · 수정 턴

- `numba`, `halide_auto`, `halide_manual`, `rust` 는 과제마다 정답 하나(`0.md`)씩 44개다(Halide 두 언어는 `sort`·`bfs` 가 없다 — 부록 5). 기대는 모두 pass 이고, `sort`·`bfs` × Halide 두 언어 4쌍은 행 없이 지원하지 않는 조합으로 처리된다.
- gemm 의 수정 턴 더미: `c_seq/1.turn1.md`(턴 0 build / compile_error → 턴 1 pass), `numpy/2.turn1.md`(턴 0 check / numeric_instability → 턴 1 pass). 턴 파일이 없는 실패 변형은 턴마다 턴 0 과 같은 응답을 받아 같은 판정이 반복된다(`harness/dummy_responses/gemm/README.md`).

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

**NA 는 "검사하지 않음"이다.** 0 이 "검사했고 없음"과 "검사하지 못함"을 함께 뜻하지 않게, 구현은 CLAUDE.md §6 의 예시(여섯 범주 모두 정수)를 넓혀 `summary` 의 값으로 `null` 을 허용한다. 이 한 문장을 CLAUDE.md §6 에 더할지는 사용자 확인 사항이다(decisions.md 2026-11-23).

- 플러그인 출력: 도구가 하나도 끝까지 돌지 않으면 `{"tool": "none", "findings": [], "summary": {여섯 범주: null}}`. 도구가 파일을 분석하지 못하면(시간 초과, 비정상 종료, 읽을 수 없는 출력, 구문·컴파일 오류) 그 도구는 돌지 않은 것으로 친다. 범주 값은 그 범주를 볼 수 있는 도구(`plugins/lint/rules.yaml` 의 그 도구 표에 그 범주가 있다)가 하나라도 끝까지 돌았을 때만 정수이고 아니면 `null` 이다.
- 행의 열(`safety.lint_columns`): 플러그인이 실패했거나(`ok=false`; 파일 없음, 0 이 아닌 종료, 시간 초과, JSON 아님, 계약 위반) `tool == "none"` 이면 `lint_findings_total` 과 여섯 `lint_<범주>` 가 전부 NA. 돌았으면 `null` 범주의 열만 NA. `lint_findings_total` = 검사한 범주의 합(= findings 수)이고 검사한 범주가 없으면 NA.
- 베이스라인 도구: `c_seq`·`c_openmp` → cppcheck, clang-tidy / `numpy`·`numba`·`halide_auto`·`halide_manual` → ruff, pyflakes / `rust` → 없음(늘 `tool == "none"` → 열 NA). 설치되지 않은 도구는 건너뛴다(C 도구가 없는 머신에서는 C 행의 린트 열이 NA). 표에 없는 규칙의 보고는 버리고 출력의 `unmapped` 에 수만 센다.
- 베이스라인의 한계: 기성 린터는 배열 원소 Python 루프(`performance`), 입력 배열 변형·시그니처 위반(`api_misuse`), float32 누적 정밀도(`numeric`) 같은 것을 거의 보지 못하고 `concurrency` 도 좁다. 이 범주의 열이 거의 상수일 수 있으므로 조건 효과를 해석할 때 주의한다. 인지프 ML lint 는 같은 CLI 로 교체한다.

## 10. sanitizer 열 (M5)

- `asan_findings`: ASan 보고 수(`==<pid>==ERROR: AddressSanitizer:` 하나가 1건; `abort_on_error=1` 이라 첫 오류에서 워커가 SIGABRT 로 끝나므로 보통 0 또는 1). `ubsan_findings`: UBSan `runtime error` 의 서로 다른 위치 수(`halt_on_error=0` 이라 끝까지 돈다). 보고는 `san/asan.<pid>`, `san/ubsan.<pid>` 와 `<sample>/safety.stderr.log` 전체에서 센다. 종류 목록(`kinds`, ASan 종류가 먼저)은 `stages.sanitize.kinds` 에 있고 첫 ASan 종류가 §5a 의 detail 에 들어간다.
- sanitizer 실행은 정상 check 와 따로 한 번(`--check-repeats 1 --sanitize`) 돈다. `--sanitize` 모드의 워커는 배열을 가드 없이 정확한 크기로 할당해 하네스 배열을 조금 넘는 접근도 ASan 에 보이게 하고, 그 실행의 판정(`classification`)은 쓰지 않는다 — 행의 판정은 정상 check 의 것이다.
- NA 규칙은 §7 의 표.

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

## 부록 5: 새 언어의 적용 범위 (numba, halide_auto, halide_manual, rust)

- 새 언어는 분류 규칙을 바꾸지 않는다. 비교 방식(허용오차·int32 출력의 rtol = atol = 0)·센티널·arg_order 패턴은 과제의 것이고(§2, 부록·부록 2–4) 언어와 무관하다.
- 판정 경로: numba 는 python 과 같은 로드·호출 경로(모듈의 최상위 함수)이고, halide 는 워커가 형태마다 만든 파이프라인을 하네스의 출력 버퍼에 직접 실현한 결과를, rust 는 C ABI 공유 라이브러리를 c 백엔드(`--backend c`)로 부른 결과를 같은 규칙으로 판정한다.
- 커널 호출 예외의 `error_detail` 은 c·python 이 `Type: message` 그대로이고, numba·halide 는 여러 줄 메시지를 한 줄로 잇는다.
- 새 언어가 더한 detail 은 §5 표에 있다: extract 의 `no_mangle_missing`(Rust), numba 의 `numba_signature_missing`·`numba_unsigned_jit:<이름>`·`numba_cache_true`(build)와 `numba_signature_rejects_args:…`(import), halide 의 반환값 검사·파이프라인 구성·오토스케줄·JIT 문구, Rust 의 rustc 오류와 `rust_panic:<메시지>`. 모두 아홉 범주 가운데 기존 값(`signature_mismatch`, `compile_error`, `runtime_crash`)이다. halide_auto 에서 모델이 쓴 스케줄은 오류가 아니라 계약 위반 관찰 `schedule_in_auto` 다(§6). 도구 문제(Rust 의존성 캐시·링커, 오토스케줄러 로드, SciPy)는 §5b 대로 행이 되지 않는다.
- Halide 는 `sort`·`bfs` 를 지원하지 않는다(그 과제의 `signatures.yaml` 의 `unsupported`). 그 조합의 raw 는 run 이 경고하고 처리하지 않으므로 행이 없다 — 실패로 세지 않는다.
- sanitizer(§5a, §10)는 C 언어(`c_seq`, `c_openmp`)에만 돌므로 새 언어의 행에서 `asan_findings`·`ubsan_findings` 는 늘 NA 이고 `memory_error` 는 나오지 않는다. rust 의 린트 열도 늘 NA 다(§9).
