# 결정 기록 (docs/decisions.md)

형식: 날짜 · 결정 · 근거 · 영향. `CLAUDE.md`와 다르게 정한 것은 **[CLAUDE.md 이탈]**로 표시한다. "사용자 확인 필요" 항목은 실제 API 호출 전에 답을 받는다.

## 2026-11-03

- **저장소 위치·이름.** 사용자가 제공한 디렉터리 `/home/ustar/dev/graduate_project`를 그대로 쓴다(`CLAUDE.md` §2의 `llm-lang-hpc/`로 이름을 바꾸지 않음). git 초기화와 GitHub 원격은 사용자 답을 기다린다(§7 질문 4). → **사용자 확인 필요**. git이 없는 동안 `harness_version`은 `nogit-<소스 해시>`로 기록한다.
- **CLAUDE.md §7 질문에 대한 잠정 답(사용자 미확인).** (1) 제공자 `anthropic`, 키 변수 `ANTHROPIC_API_KEY`(현재 환경에 없음), 모델 `claude-opus-5-5`(env `LLM_MODEL`로 교체 가능). (2) 머신 정보는 `config/machine.yaml`로 자동 기록; `taskset` 가능, `cpupower`는 sudo 비밀번호 필요. (3) M1 언어는 `numpy`, `c_seq`, `c_openmp`. (4) 저장소 이름/원격 미정. → **사용자 확인 필요**.
- **파이프라인 검증은 `provider: dummy`.** 키가 없으므로 더미 응답(`harness/dummy_responses/`)으로 전체 경로를 검증한다. 더미 행은 `results/runs_dummy.csv`에 분리 저장하고 provider 열이 `dummy`다. 실제 제공자에서 키가 없을 때 dummy로 자동 폴백하지 **않는다**(섞이면 안 됨).
- **Python 시그니처에도 출력 버퍼 `C`를 둔다** `def gemm(A, B, C, M, N, K)` **[CLAUDE.md 이탈]**: §4 키워드 예시 `gemm(*, A, B, M, N, K)`에는 `C`가 없지만, 공통 규칙 "출력 버퍼는 호출자가 할당"을 우선하고 C·Numba(M3)와 계약을 맞춘다. 키워드 버전(M4)도 `gemm(*, A, B, C, M, N, K)`.
- **정확성 테스트 형태 추가**: `tiny`(7,5,3), `rect`(200,130,312)를 `small`(256³) 앞에 둔다. 정사각 입력만으로는 M/N 교환·전치·꼬리 처리 오류(`arg_order`, 8/16 배수 가정)가 보이지 않는다. 성능 측정은 `large`=1024³ 유지.
- **허용오차에 `atol=1e-3` 추가** **[CLAUDE.md 이탈: rtol만 명시]**: 입력 uniform[-1,1)에서 0 근처 원소의 상대오차가 발산하므로 절대항이 필요하다. 참조는 float64 누적.
- **컴파일러는 gcc 15 고정**: clang 21에 `omp.h`(libomp)가 없다. 벡터화 리마크는 clang `-Rpass` 대신 gcc `-fopt-info-vec-*`로 수집(M5 입력).
- **측정 코어 고정**: P-core 4개, 코어당 1스레드(`taskset -c 0,2,4,6`, `OMP_NUM_THREADS=4`). 하이브리드 CPU에서 E-core/SMT 형제를 섞으면 OpenMP 정적 스케줄 불균형과 SMT 간섭(실측 +36%)으로 분산이 커진다. `c_seq`(1스레드)는 단일 P-core(`taskset -c 2`)에 더 좁게 고정해 스레드 사용 금지를 코드로 강제한다.
- **NumPy는 pip 휠(OpenBLAS 내장)만 사용.** 시스템 `python3-numpy`는 참조 BLAS(netlib)라 1024³가 ~250 ms이고 스레드 변수가 무효다. 격리 venv + `numpy==2.3.5` 휠(scipy-openblas 0.3.30)로 14.7 ms(4스레드). `CLAUDE.md`의 기준선 "NumPy(OpenBLAS)"는 이 구성에서만 성립하므로 워커가 BLAS 정체를 기록하고 기준선 단계가 OpenBLAS가 아니면 중단한다.
- **`large`=1024³ 유지(기준선 ≈15 ms, 순진한 C ≈9.5 s).** 2048³는 순진한 C가 호출당 ~80 s라 bench 예산 안에서 측정이 불가능해진다. 1024³의 작업집합 12 MiB는 L3(18 MiB)에 상주하므로 측정은 warm-cache이며 DRAM 영역을 보지 않는다(`docs/protocol.md`에 명시).

## 2026-11-04 (설계 리뷰 반영)

- **모델 샘플링 제어** **[CLAUDE.md 이탈: "temperature 고정 0.2"]**: Claude 4.7 이후·5.x(`claude-opus-5-5` 포함)는 `temperature/top_p/top_k`를 400으로 거부하고 Messages API에 `seed`가 없으며 thinking을 끌 수 없다. M1은 **제공자 기본 샘플링**, `thinking={"type":"adaptive"}` 명시, `effort=medium` 고정, 분산은 n 샘플로 다룬다. 행에 `sampling` JSON(`temperature_effective: provider_default` 등)을 기록한다. 2차 오픈 모델(openai_compat)은 `temperature=0.2`·thinking 없음이라 **다른 조건**이며 같은 그룹으로 집계하지 않는다. → 모델·effort 선택은 **사용자 확인 필요**.
- **`tokens_out`은 thinking 포함 총 출력 토큰(과금 기준).** `tokens_thinking`·`tokens_text`를 분리 기록하고 README에 의미를 적는다.
- **프롬프트에 하드웨어 사실 공개**(CPU 모델, 고정 코어 4개, AVX2/FMA, 캐시, 스레드 수, 전체 컴파일 명령): "성능 결정을 누가 내리는가"를 보려면 모델이 결정을 내릴 정보가 있어야 한다. 모든 언어에 같은 사실을 준다(정보 균형). 라이브러리 지시는 없다. → **사용자 확인 필요**(프롬프트가 바뀌면 `prompt_hash`가 바뀌어 실행 간 비교가 끊김).
- **spec에 계약 추가**: C 초기값 미정의·전부 덮어쓰기, 반환값 무시, 64바이트 정렬 보장, 매개변수 정확히 6개, 평가 형태·절차 공개, "fastest correct implementation" 요구.
- **빌드 플래그**: `-std=gnu11`(ISO `-std=c11`은 gcc 15에서 `posix_memalign` 등 POSIX 선언을 숨겨 하네스가 만든 `compile_error`가 생김 — 실측), `-fno-semantic-interposition`(gcc 공유 라이브러리의 비-static 헬퍼 PLT 호출 제거), `-Wl,--no-undefined`(미정의 심벌을 빌드 단계 오류로). 전체 명령을 프롬프트와 행(`cflags`)에 그대로 기록.
- **출력 버퍼 센티널은 유한값 `-2048.0`**(NaN 아님): NaN 센티널은 "안 썼음/누적/반환만 함"을 `numeric_instability`로 오분류한다. 센티널 규칙(미기록·부분 기록·누적·OOB)을 `wrong_output`으로 먼저 판정한다. → **2026-11-04 설계 v2.1로 대체**: 센티널 규칙은 NaN·pass·arg_order 패턴 **뒤**의 규칙 4가 됐다(아래 "통합 검증 후 설계 v2.1", "리뷰 반영: classify·worker"). 센티널이 유한값이라는 결정 자체는 그대로다.
- **numeric_instability 판정은 허용오차 배율**(`max_tol_ratio ≤ 20`)로: 절대오차/최대값 5% 규칙은 off-by-one 논리 오류를 수치 문제로 오분류한다. 샘플 합산 우선순위 `runtime_crash > race_condition > arg_order(실패 테스트 전부 같은 패턴) > wrong_output > numeric_instability`.
- **`pass` 정의**: check 전부 통과 ∧ 입력 미변조 ∧ OOB 없음 ∧ (bench 미실행 ∨ bench 정확성·크래시·race 없음). **bench timeout만** pass=1·시간 NA(성능 실패이지 정확성 실패가 아님).
- **생성 단계 실패는 모델 오류로 세지 않는다**: API 오류·`max_tokens` 절단 행은 `valid_sample=0`(pass@k 분모 제외), `refusal`은 `valid_sample=1`·pass=0·`error_type` 빈칸(분류표 변경 금지)으로 두고 수를 따로 보고한다. generate는 실패 샘플만 재호출한다(정상 샘플 재과금 없음). `max_tokens=32000`, 스트리밍 전송.
- **측정 규약 추가 강제**: (1) 프로토콜 동적 검사(부하·고정/형제 코어 사용률·거버너)를 **bench 직전마다** 수행해 행에 기록, (2) 기준선을 run 시작·1스레드·**샘플별 대응**(bench 직전)으로 측정해 `speedup_vs_baseline`은 대응 기준선 사용, drift 10% 초과 플래그, (3) EPP·turbo·bench 중 코어 주파수(`scaling_cur_freq`)·스로틀 횟수·패키지 온도 기록(powersave/EPP=power에서 P-core가 1.3–1.7 GHz에 머무는 것을 실측), (4) 워커 환경은 화이트리스트(OMP_*/OPENBLAS_* 등 상속 제거), OpenMP 워커에 `OMP_PROC_BIND=true OMP_PLACES=cores`, (5) 워커는 코어 덤프를 끄고 `RLIMIT_AS=8 GiB`(RLIMIT_NPROC은 사용자 전체 기준이라 걸지 않음), (6) 모든 서브프로세스는 프로세스 그룹 단위로 타임아웃·정리(`subprocess.run(timeout)`은 `cc1`·OpenMP 자식을 남김 — 실측).
- **느린 커널의 반복 축소 규칙** **[측정 규약 변경 — 사용자 확인 필요]**: 첫 호출 시간으로 13회가 예산(300 s)의 80%를 넘으면 `warmup=1`, `repeats`를 3~10으로 줄이고 `repeats_reduced=1` 플래그; 그래도 안 되면 즉시 `timeout`으로 종료하고 첫 호출 시간을 기록한다. 대안은 "축소 없이 timeout"(데이터 손실). 집계는 축소 행 수를 표시한다.
- **`-std=gnu11`·정렬·센티널 등은 모델이 아니라 하네스가 만든 실패를 없애기 위한 것**이며, 모델의 선택(라이브러리 사용, 병렬화, 타일링)은 건드리지 않는다.
- **M1 저장소 구조 추가**(CLAUDE.md §2에 없음): `harness/paths.py`, `config.py`, `procutil.py`, `classify.py`, `worker.py`, `dummy_responses/`, `tests/`. 이유: 실행 격리·분류 규칙 고정·검증.
- **refusal fallback(서버 측 모델 교체) 미사용**: 거절 시 다른 모델로 자동 재실행되면 행의 모델이 바뀐다. 거절은 `stop_reason=refusal`로 기록만 한다.
- **M1에서 하지 않는 것**: 수정 루프, 린트 플러그인, ASan/UBSan, perf_summary, 그래프, 대시보드, 다른 과제·언어. 부트스트랩 CI는 M6.

## 2026-11-04 (통합 검증: `make dummy RUN_ID=dummy-m1`)

- **워커의 주파수·스로틀 샘플링 CPU 집합은 커널 로드 전에 읽는다** (`harness/worker.py`, 설계 §8 bench 문구 보강). 실측: `OMP_PROC_BIND=true OMP_PLACES=cores` 아래에서 `ctypes.CDLL(kernel.so)` 가 libgomp를 올리는 순간 주 스레드 친화도가 `{0,2,4,6}` → `{0}` 으로 좁아진다(워커 스레드는 2/4/6에 묶임). 커널은 4코어에서 돌지만 `os.sched_getaffinity(0)` 을 로드 뒤에 읽던 워커는 `pinned_cpus=[0]` 으로 기록해 `cpu_mhz_*`·`throttle_events` 가 CPU 0 하나만 반영했다(dummy-m1 의 c_openmp/0 행). 측정값(시간)에는 영향 없음. 결과 문서에 최상위 `pinned_cpus` 를 모든 모드에서 기록한다(추가 키).
- **더미 기대표 보정**: c_openmp/2(race)는 실측에서 rect·small 만 실행 간 불일치이고 tiny(K=3)는 두 호출 모두 맞는다(설계 §13 과 일치; README 의 "tiny·rect·small 모두" 문구를 고침). 샘플 합산은 `race_condition`/`nondeterministic:rect,small`.
- **재실행 시 run 시작 기준선은 다시 잰다**(설계 §8-3 그대로): 모든 샘플이 `raw_sha256` 일치로 건너뛰어져도 `--no-baseline` 이 없으면 `bench_run.json`/`bench_1t.json` 을 새로 측정한다(약 1 s). 건너뛴 행의 기준선 열은 바뀌지 않는다. 의도된 동작이며 바꾸지 않는다.

## 2026-11-04 (통합 검증 후 설계 v2.1)

- **분류 규칙 순서 변경**: arg_order 패턴 검사를 센티널·계약 규칙(미기록/부분 기록/누적/OOB)보다 **먼저** 둔다. 구현 검토에서 `C[i*M+j]`(M/N 교환) 커널은 항상 partial_write 또는 out_of_bounds_write에 먼저 걸려 `swapped_MN_dims`가 도달 불가임이 드러났다. 패턴이 전체 버퍼를 설명하면 구조적 오류(`arg_order`)가 더 정확한 분류다. `swapped_operands`는 정사각 입력에서 `column_major_all`과 대수적으로 동일하므로 먼저 검사해 그 이름으로 보고한다(모호성은 `docs/error_taxonomy.md`에 명시). **적용 상태**: 이 결정 시점(통합 검증 직후)에는 설계 문서만 바뀌고 코드(`harness/classify.py`)·테스트·`docs/error_taxonomy.md`는 옛 순서였다. 코드·테스트 반영은 같은 날 아래 "리뷰 반영: classify·worker" 절, 분류표 문서 반영은 "리뷰 반영: docs-consistency" 절.
- **워커 import 모드 실패의 `error_stage`는 `build`**(열 enum에 `import`가 없음; 로드 실패는 링크/적재 문제이므로 빌드 범주).
- **더미 파이프라인 검증 완료(2026-11-04 11:18)**: 11개 더미 변형 모두 기대표와 일치, 단위 테스트 120개 통과, 멱등 재실행·강제 재실행·프로세스 정리 확인(**v2.0 순서의 코드 기준** — 더미 기대표 11건은 v2.0·v2.1 어느 순서에서도 같은 결과를 내므로 이 검증은 위 순서 변경의 근거가 아니며, v2.1은 `tests/test_classify.py`로 검증한다). 거버너가 `powersave`라 모든 행에 `protocol_ok=0`(`governor=powersave`)가 기록됨 — 실제 실험 전 사용자가 `performance`로 바꿔야 한다.

## 2026-11-04 (리뷰 반영: classify·worker)

- **§10 v2.1 순서를 코드에 반영**(`harness/classify.py`): NaN/Inf → pass → arg_order(`transposed_output` → `swapped_operands`(M==N==K) → `column_major_all` → `swapped_MN_dims`) → 센티널·계약 → imprecise → mismatch. 결과: 정사각 `B@A`는 `column_major_all`이 아니라 `swapped_operands`로 보고되고, `C[i*M+j]` 커널은 실제 rect(200,130,312)에서 센티널·OOB가 남아도 `arg_order`/`swapped_MN_dims`다(`n_unwritten`·`oob_write`는 metrics에 유지; metrics에 `oob_write` 키 추가 — 추가 키라 소비자 영향 없음). 부수 효과: 센티널 일부 + NaN은 이제 `nan_inf`(규칙 1이 먼저).
- **규칙 4 내부 순서 고정: 전부 센티널 → `accumulated_into_c` → `partial_write` → `out_of_bounds_write`**. 근거: float32에서 `-2048 + ref`는 |ref| < 2⁻¹³이면 정확히 `-2048.0`으로 되돌아가 `n_unwritten`에 세어진다. 실측: `C += A@B`가 small(256³)에서 `partial_write:1/65536`, large(1024³)에서 `partial_write:10/1048576`으로 잘못 기록됐다(tiny·rect는 0개). 진짜 부분 기록은 모든 원소가 `sentinel + ref`와 일치할 수 없으므로 누적 검사를 앞에 둬도 부분 기록을 가리지 않는다. `n_unwritten` 지표의 정의(`count(out == sentinel)`)는 바꾸지 않았다. 이에 따라 누적 + OOB는 `accumulated_into_c`(예전: `out_of_bounds_write`) — 샘플 수준 detail은 `aggregate_sample`이 `oob_write` 플래그를 먼저 보므로 행 결과는 그대로다.
- **출력이 맞고 OOB만 있는 호출**: 규칙 2(pass)가 계약 규칙보다 먼저이므로 `classify_output`은 `pass`를 돌려준다. 워커(`_apply_contract_flags`)가 그 호출을 `wrong_output`/`out_of_bounds_write`로 기록해 bench `error_detail`과 `check_tests`가 비지 않게 한다(`inputs_mutated` 덮어쓰기와 같은 자리; `arg_order` 판정은 OOB가 있어도 유지). 샘플·행 수준은 변화 없음(`aggregate_sample`, run.py pass 정의).
- **워커 `BaseException` 핸들러의 `str(e)` 보호**: `__str__`이 예외를 던지는 예외 객체를 커널이 올리면 핸들러가 죽어 `worker_error`(rc=3) → run.py가 harness_error로 run 전체를 중단했다(실측). `message`/로드 실패 detail은 `<unprintable Type>`으로 대체하고 `traceback.format_exc()`도 같은 방식으로 감싼다.
- **테스트 방법 보정**(`tests/test_worker.py`): (1) `pinned_cpus` 회귀 테스트는 SMT 형제 쌍(0,1)이 아니라 **서로 다른 물리 코어의 CPU 두 개**(`thread_siblings_list` 기준 → 이 머신에서 0,2)를 고정하고, 같은 .so·환경으로 '로드 뒤 친화도가 실제로 좁아지는가'를 먼저 확인해 아니면 건너뛴다 — 형제 쌍에서는 `OMP_PLACES=cores`의 place가 하나뿐이라 축소가 일어나지 않아 단언이 공허했다(실측: 0,1 → 변화 없음, 0,2 → {0}). (2) 세그폴트 테스트는 부분 결과의 `limits.dumpable == 0`을 단언한다; `core_pattern`이 파이프(apport)면 cwd의 `core*` 검사는 아무것도 검증하지 않으므로 파이프가 아닐 때만 본다.
- **미반영(파일 소유 밖)**: `docs/error_taxonomy.md` §2 표는 아직 예전 순서(0a–0d 센티널 규칙이 먼저, `column_major_all`이 `swapped_operands`보다 먼저)다 — §10 v2.1과 위 규칙 4 순서로 갱신해야 한다. → **2026-11-04 반영 완료**(아래 "리뷰 반영: docs-consistency").

## 2026-11-04 (리뷰 반영: procutil·run — 오케스트레이터 중단 시 워커 정리)

- **`run_isolated`는 어떤 예외로 빠져나가든 자식 프로세스 그룹을 죽인다**(`harness/procutil.py`, 설계 §7·§8-11 보강). 실측: `run_isolated(["bash","-c","… exec sleep 300"], timeout_s=100)`을 돌리는 부모에 SIGINT를 보내면 부모는 `KeyboardInterrupt`(rc=-2)로 끝나고 자식 `sleep 300`은 PPID=1·자기 세션으로 살아남았다(SIGTERM도 같음; 예전 코드는 `TimeoutExpired`에서만 `killpg`). 워커는 `start_new_session=True`라 터미널의 Ctrl-C를 받지 않으므로 멈춘 LLM 커널이 고정 코어(0,2,4,6 / 2)를 계속 점유해 다음 run의 측정을 오염시킨다(동적 검사는 플래그만 남긴다). 고침: `Popen` 뒤 `except BaseException: killpg → wait → raise`(Popen과 `communicate` 사이의 틈도 포함). 추가 방어: (1) 자식에 `prctl(PR_SET_PDEATHSIG, SIGKILL)`(`preexec_fn`; exec를 거쳐 `taskset` → 워커까지 유지, 손자는 제외) — 부모가 SIGKILL/OOM처럼 핸들러가 돌 수 없는 방식으로 죽어도 직접 자식은 커널이 죽인다. `preexec_fn`은 vfork 경로를 끄지만(fork 수 ms) 측정은 서브프로세스 안에서 하므로 영향 없음. (2) `install_terminate_handlers()`가 SIGTERM·SIGHUP을 `KeyboardInterrupt`로 바꾸고 run.py `main`이 exit 130으로 끝난다(`SIG_IGN`이면 유지 → `nohup` 동작 보존). 워커(`harness/worker.py`)는 바꾸지 않았다. 테스트: `tests/test_procutil.py` `ParentDeathTest`(부모 SIGINT/SIGTERM/SIGKILL 뒤 `pgrep -g` 비어 있음), `TerminateHandlersTest`.

## 2026-11-04 (리뷰 반영: llm_client·extract·prompts)

- **스트림 도중 SSE `error` 이벤트는 body `error.type`으로 분류**(`harness/llm_client.py`, 설계 §5 보강). anthropic 2.4.0 `_streaming.py`는 in-stream error를 `_make_status_error(..., response=self.response)`로 던지는데 응답 상태가 200이라 평범한 `APIStatusError(status_code=200)`가 되고, 예전 `_classify_error`는 fatal/transient 타입 목록 어디에도 못 넣고 `status >= 500`도 거짓이라 **fatal**로 낙착시켰다 — 32k 토큰 thinking 스트림에서 가장 흔한 일시 오류(`overloaded_error`)가 generate 배치를 즉시 중단(exit 3)시킨다. 재현(`httpx2.MockTransport`, 네트워크 없음): HTTP 200 + `message_start`/`text_delta` 뒤 `event: error` overloaded → `fatal:APIStatusError:200:...`; 같은 body를 HTTP 529로 주면 `transient:OverloadedError:529`. SDK `max_retries`는 스트림 시작 후 오류에 적용되지 않는다. 고침: 상태가 4xx/5xx가 아닌 `APIStatusError`는 `error.type`으로 가른다(`overloaded_error`/`api_error`/`rate_limit_error`/`timeout_error`/모르는 타입 → transient, `invalid_request_error`/`authentication_error`/`permission_error`/`not_found_error`/`billing_error` → fatal). 오류 문자열 형식 `kind:Type:status:msg`는 유지(generate의 `startswith("fatal:")`·README 그대로)하고 msg 앞에 `in-stream error event (<type>): `를 붙인다. 호출 안 재시도는 두지 않는다(설계대로 transient는 다음 generate 실행에서 재호출). 테스트: `tests/test_llm_client.py`(**신규** — 이 그룹의 지정 경로 밖이지만 llm_client 테스트가 전혀 없었고 새 파일이라 다른 그룹과 충돌하지 않는다): `MockTransport`로 SDK 스트리밍 경로를 통과 — in-stream overloaded/invalid_request, HTTP 529/401, 정상 스트림의 텍스트 결합·thinking 토큰·stop_details·request_id, 요청 JSON의 `thinking`/`output_config`/`stream`/`extra_body` temperature, 분류 표.
- **C 매개변수의 `__attribute__((aligned(64)))`와 단순 typedef를 추출 단계에서 해소**(`harness/extract.py`, 설계 §6 보강). `_c_def_re`의 매개변수 그룹이 괄호 1단계만 허용해 매개변수 속성(3단계)이 있으면 정의를 못 찾아 `function_not_found` → 행이 `signature_mismatch`로 오분류됐고(같은 모듈의 `parse_c_param`은 속성을 지우도록 돼 있어 모순), `typedef float real; ... const real* A`는 `param_mismatch:... real* expected float*`였다. 두 소스 모두 실제 빌드 명령으로 컴파일되고 `nm -D`에 `T gemm`이 있으므로 모델 오류가 아닌 하네스 오분류다. 고침: (1) 매개변수 그룹 괄호 3단계 중첩 허용(`_C_PAREN3`; 대안의 첫 글자가 `(`/그 외로 갈려 역추적 폭발 없음); (2) 검사 텍스트에서 `typedef <타입 토큰열> <이름>;`/`typedef <타입> *<이름>;`만 수집(`collect_typedefs`; 구조체·열거형·함수 포인터·배열·다중 선언자 제외)해 기본 타입·깊이 비교 전에 치환(체인 8회 상한, 순환 안전). `CParam.tokens`는 원문 그대로라 typedef 차이는 `signature_exact=False`; `typedef double real;`은 여전히 `param_mismatch:... double* expected float*`(해소된 타입으로 보고). 매개변수 속성은 `)` 뒤 속성과 같이 exact 판정에 영향 없음. `detail` enum 변경 없음. 테스트: `tests/test_extract.py`에 속성 매개변수·typedef(단순/포인터/체인/속성 붙은 typedef/틀린 타입)·전처리 경로 사례 추가.
- **numpy 프롬프트의 환경 문장을 규범문으로**(`prompts/lang/numpy.md` 6행, 설계 §4 보강). "Only the Python standard library and NumPy are installed. No other third-party package is available, and nothing can be installed."는 거짓이었다 — 워커 인터프리터 `.venv`에 pandas 3.1.0·PyYAML·pydantic·anthropic·openai·httpx2가 있고 `launch_env`가 `PYTHONPATH=ROOT`를 주어 `harness`·`tasks`(참조 구현)도 임포트되며, `build/python.py`는 `compile()` 구문 검사만 하고 임포트 제한이 없다. 설계 §4는 환경 서술을 사실 진술로 못박았으므로 규칙 문장 "Use only the Python standard library and NumPy: do not import any other package, and do not install anything."으로 바꿨다(결과를 단언하는 "counts as a failure"류 문구는 하네스가 강제하지 않으므로 쓰지 않음). 영향: numpy `prompt_hash` 변경 — 실험 시작 전 반영이며 기존 더미 raw와의 비교 대상이 아니다. 하네스 측 강제(워커 import 모드 `sys.meta_path` 화이트리스트 → `contract_violation`)는 `harness/worker.py` 소유라 이 그룹에서 하지 않았다 → **미반영(파일 소유 밖)**; GEMM에선 다른 패키지의 실익이 없어 영향은 작다.

## 2026-11-04 (리뷰 반영: docs-consistency — 문서·계약 정합; 코드 변경 없음)

- **`docs/error_taxonomy.md` §2를 §10 v2.1 순서로 갱신**: NaN → pass → arg_order 3a–3d(`transposed_output` → `swapped_operands`(M==N==K) → `column_major_all` → `swapped_MN_dims`) → 센티널·계약 4a–4d(전부 미기록 → 누적 → 부분 기록 → OOB) → imprecise → mismatch. `swapped_MN_dims`의 의미(M≠N, 가드 포함 `out_full`의 `i·M+j` 위치 99 % 일치, 센티널·OOB가 남아도 `arg_order`로 보고하고 `n_unwritten`·`oob_write`는 metrics에 유지), 정사각 모호성, 호출 수준 덮어쓰기(`_apply_contract_flags`), import 모드 실패의 `error_stage=build`를 명시. 코드는 이미 v2.1이라 바꾸지 않았다.
- **정사각 모호성의 샘플 수준 부작용 — 미결(classify 소유자·사용자 확인 필요)**: 정확성 테스트가 tiny·rect(비정사각)·small(정사각)이라 열우선 커널은 `column_major_all`×2 + `swapped_operands`×1로 패턴이 갈려 §10 합산 규칙 3("전부 같은 패턴")을 못 채우고 `wrong_output`/`mixed:arg_order/column_major_all,arg_order/swapped_operands`가 된다(실측 2026-11-04). 즉 "정사각에서 `swapped_operands`로 보고"는 호출 수준에서 그 이름을 살리지만 샘플 수준에서는 열우선 커널을 `arg_order`에서 밀어낸다. 후보: (a) `aggregate_sample`에서 두 패턴을 같은 족으로 보아 `arg_order`/`column_major_all|swapped_operands`로 합산, (b) 정사각 3b 판정명을 `swapped_operands|column_major_all`로. 피연산자를 바꾼 numpy 커널은 비정사각에서 `ValueError` → `runtime_crash`라 `swapped_operands`가 샘플 수준에 남는 경우는 사실상 없다. 이 그룹은 문서만 소유하므로 코드는 건드리지 않았다.
- **`--strict-protocol`의 범위는 정적 검사만**(설계 §8-1·§8-7, `docs/protocol.md` §8–§9·§13, README §4 명료화; 코드 그대로): run 시작의 거버너·EPP·taskset 플래그만 exit 4로 중단하고, bench 직전 동적 플래그(`load_1m`, `pinned_busy`, `sibling_busy`, 거버너 재확인)·`baseline_drift`·`paired_baseline_failed`·`model_served_mixed`는 strict여도 행에 기록만 한다(`run.py` 826–833행; `check_dynamic` 결과는 `add_flag`만). 근거: 측정 중 일시적 부하로 run 전체가 죽는 것보다 행 단위 `protocol_ok=0`으로 분석에서 가려내는 쪽이 데이터 손실이 적다. README·protocol.md가 "STRICT=1이면 중단"으로 뭉뚱그려 적어 "플래그 없는 행만 남는다"로 읽힐 수 있었다. `Makefile` 24행 주석("규약 플래그가 있으면 중단")은 이 그룹 소유 밖 — "정적"을 붙여야 한다.
- **`protocol_ok` 정의 통일**: `protocol_ok = flags_ok(protocol_flags)` = 정보용(`turbo=`, `env_overrides_dropped`) 외 플래그가 하나도 없음(`finalize_row`). `baseline_drift=…%`·`paired_baseline_failed`(대응 기준선 bench 실패 → `baseline_time_ms`·`speedup_vs_baseline` NA; 어느 문서에도 없었음)·`model_served_mixed`도 0을 만든다(실측 `flags_ok([...]) → False`). 설계 §8-7·§9, protocol.md §9, README §6 열 표에 두 플래그를 추가.
- **`extract` 실패 사유에서 `unterminated_fence` 제외**(설계 §10, error_taxonomy §5): 닫히지 않은 펜스는 그 뒤 전체를 블록으로 보고 계속하므로 `ok=True`인 정보 노트다(실측: 닫히지 않은 ```` ```python ```` + 올바른 시그니처 → `ok=True`, `detail=unterminated_fence`). 실패 사유는 `no_code_block`·`function_not_found`·`param_mismatch:…`·`static_linkage`만.
- **샘플 합산의 `mixed:<…>` 규칙 문서화**(설계 §10 합산 6, error_taxonomy §4): 패턴이 섞이면(arg_order 패턴 둘 이상, arg_order + numeric) `wrong_output`/`mixed:<classification/pattern,…>`. 구현(`aggregate_sample`)은 처음부터 이렇게 동작했고 문서만 없었다.
- **import 단계 워커의 프로세스 실패도 `error_stage=build`**(설계 §8-6·§10, error_taxonomy §1·§5, protocol.md §6): `--mode import` 워커의 timeout(`check_timeout_s`=180 s)·시그널·비정상 종료는 §8-6 종료 규칙대로 `timeout`/`runtime_crash`이되 `error_stage=build`(`run.py` 601–605행; 실측 `WorkerOutcome(stage="import", timed_out=True).verdict()` → `('timeout', 'worker timeout after 180s during ? call ?')`). 따라서 `timeout`은 build 단계에도 나올 수 있다. error_taxonomy §5의 `api_error`·`worker timeout after …` detail 문자열도 코드와 맞췄다.
- **행의 `warmup`·`repeats`는 계획값**(설계 §8 bench, protocol.md §5, README §6): 축소 행에서는 축소된 값(= 실제 횟수)이지만 적응 규칙 1의 즉시 timeout 행과 bench 크래시 행은 3/10이 남는다(실측 `results/runs_dummy.csv` dummy-m1x/numpy/sample_3: `timeout`, `warmup=3`, `repeats=10`, `time_samples_ms={"warmup":[186929],"timed":[]}`). 실제 호출 수는 `time_samples_ms` 길이. 대안(`worker.py` 조기 종료 시 `len(warmup_ms)`/`len(timed_ms)`로 덮어쓰기)은 워커 소유라 하지 않았다 — "계획값"으로 고정하면 기존 CSV 행과도 일치한다.
- **설계 §0 보정표에 주파수 상태 주석 추가**(기존 값 유지): 통합 검증(governor `powersave`, EPP `performance`, `cpu_mhz_median` 2.6–2.9 GHz)에서 NumPy 기준선 run 시작 4스레드 7.7–8.3 ms(대응 7.7–8.9 ms), 1스레드 19.5–21.7 ms, c_seq/0 77 ms, c_openmp/0 29 ms — 09시 보정값(14.7 ms, EPP `power`)의 약 절반. 절대 시간은 전원 상태에 따라 2배까지 달라지므로 비교는 대응 기준선 대비 속도비로 한다. protocol.md §1·README §2에도 같은 주석.
- **README §9 마일스톤 상태 갱신**: 더미 파이프라인 끝까지 검증(2026-11-04; 11개 변형 일치, 단위 테스트 120개(당시)/153개(현재), 멱등 재실행·강제 재실행), 남은 M1 단계는 사용자 답(§8 미결) 뒤 실제 9회 호출.
- **`harness/dummy_responses/gemm/README.md`**: 기대표가 v2.0/v2.1 순서를 구분하지 못한다는 주석과 통합 검증 결과를 추가.
- **`config/experiment.yaml`**: `baseline_drift_pct_max` 주석에 "기록만; --strict-protocol 무관"을 붙였다(주석만; 설계 §2 YAML 사본도 같이). `harness_version`(nogit 해시)이 바뀌지만 실험 전이라 영향 없음.

## 2026-11-04 (재검증: 리뷰 반영 뒤 `make dummy RUN_ID=dummy-m1`)

- **절차**: `.venv/bin/python -m unittest discover -s tests -v` → 153개 통과(수정 없음). `results/raw|built/{dummy-m1,dummy-m1x}`와 `results/runs_dummy.csv`를 지우고(세션 스크래치에 백업) `make dummy RUN_ID=dummy-m1`(9행, 14 s, rc 0) → `harness.generate --provider dummy --run-id dummy-m1x --languages numpy --n 5` + `harness.run --run-id dummy-m1x --languages numpy --samples 4`(1행; numpy/3은 느려서 생략) → `analysis.aggregate --csv results/runs_dummy.csv --run-id dummy-m1`(rc 0; 언어마다 n=3·pass_at_1 0.333·benchmarked 1, paired_mean 7.81 ms, max|drift| 1.5 %) → `harness.run --run-id dummy-m1` 재실행(9개 전부 "result.json이 최신이라 건너뜀", CSV sha256 동일·result.json 9개 동일, 키 중복 0, CSV 총 10행) → `pgrep -af "harness[.]worker|[c]c1"` 비어 있음.
- **결과**: 10건 모두 stage/error_type/pass와 테스트별 pattern이 더미 README 기대표와 일치. 유일한 차이는 c_openmp/2의 detail이 `nondeterministic:tiny,rect,small`(README: `rect,small`)인 것. check.json상 tiny 두 호출이 모두 `wrong_output/mismatch`(max_abs_err 0.51/0.12)이고 서로 달라 분류 자체는 옳다. `--force --no-baseline`으로 그 샘플만 6회 더 돌려 7/7 재현. 11:03 실행(백업 check.json)은 tiny 두 호출 모두 max_abs_err 0.0이었다 — 공유 `sum`의 읽기-더하기-쓰기 겹침은 스레드당 반복 1회여도 일어날 수 있으므로 "K=3이라 안전"은 틀린 근거였고 detail의 테스트 목록은 타이밍 의존이다. 워커의 친화도 처리(읽기만, `sched_setaffinity` 없음)·`start_new_session`(기존)·`PR_SET_PDEATHSIG`(신규)는 스레드 배치를 바꾸지 않고 c_openmp/0 bench도 28.6 ms(11:03: 28.7 ms)로 같다 — 하네스 변경이 원인이라는 증거는 없다. **고침(문서만)**: 더미 README의 기대표 행·"변형별 주의"·검증 문단, 설계 §13 행 — "rect·small은 항상 포함, tiny는 타이밍에 따라 포함되기도". `docs/error_taxonomy.md` §8은 detail을 적지 않아 그대로. 위 "통합 검증" 절의 "tiny는 두 호출 모두 맞는다"는 당시 1회 실측 기록으로 남긴다.
- **관찰(미수정, run.py 소유자 판단)**: 멱등 재실행도 `prepare_baselines`가 run 시작·1스레드 기준선을 다시 재어 `results/built/dummy-m1/gemm/baseline/bench_run.json`·`bench_1t.json`을 덮어쓴다(7.76→7.80 ms, 19.92→19.09 ms). 행의 `baseline_run_time_ms`/`baseline_time_1t_ms`는 원래 값이고 CSV는 바이트 동일하므로 결과는 안 바뀌지만, 디스크의 기준선 파일이 행의 증빙과 어긋난다. 처리할 샘플이 없으면 기준선을 건너뛰거나 재실행에 `--no-baseline`을 쓰는 것이 맞다. 거버너는 여전히 `powersave`(모든 행 `protocol_ok=0`, `governor=powersave`·`turbo=on`).

## 2026-11-04 (마무리: run.py 두 가지 보정)

- **워커 rc=3 의 해석**: `harness_error`(행 없이 run 중단)는 워커가 스스로 `worker_error` 를 남겼거나 결과 문서를 하나도 쓰기 전에 rc=3 으로 죽은 경우에만. 결과 문서(`in_progress` 등)가 있는데 `worker_error` 가 없는 rc=3 은 LLM 커널이 `exit(3)` 을 부른 것으로 보아 `runtime_crash`(rc>0 규칙)로 행을 남긴다(리뷰에서 제기, 다수 반박됐으나 실험 데이터 손실 경로라 보정). 테스트 `tests/test_run.py::VerdictTest`.
- **처리할 샘플이 없는 과제는 run 시작·1스레드 기준선을 다시 재지 않는다**(`needs_processing` 사전 평가; `--force` 면 항상 측정). 재검증에서 지적된 "멱등 재실행이 디스크의 기준선 파일을 덮어쓴다" 를 없앤다. 테스트 `tests/test_run.py::NeedsProcessingTest`. Makefile 의 STRICT 주석도 "정적 플래그만" 으로 고침. 단위 테스트 164개 통과.
- 리뷰 중 생성된 잡동사니 `extra.yaml`, `l16.yaml`, `m16.yaml`(experiment.yaml 사본) 삭제.

## 2026-11-05 (사용자 확인 — "사용자 확인 필요" 항목 전부 확정)

- **모델: `claude-sonnet-5-5`, `effort=medium`, `thinking=adaptive`** (잠정값 `claude-opus-5-5`에서 변경). 사용자가 모델 선택을 위임하면서 Max 구독의 API 크레딧만 쓰겠다고 했다. 문서 기준 Max는 매달 100달러의 Claude Platform API 크레딧을 준다(Console 조직 연결 필요, 이월 없음, Claude Code 사용량과는 별개). 호출당 입력 3k·출력 5k 토큰을 가정하면 Opus 5.5는 약 0.11달러, Sonnet 5.5는 약 0.06달러이고, 12개 커널 × 6개 언어 × 5샘플에 조건·수정 턴을 더하면 1,500~3,000회가 된다. 실험 도중 모델을 바꾸면 비교가 끊기므로 크레딧 안에서 끝까지 유지할 수 있는 Sonnet 5.5로 정했다. Sonnet 5.5도 temperature를 받지 않으므로 샘플링 결정은 그대로다. M1 9회 실행에서 실제 토큰 사용량을 재고, M2 시작 전에 이 선택을 한 번 재확인한다.
- **구독 로그인으로 `claude -p`를 돌리는 경로는 쓰지 않는다.** Claude Code는 시스템 프롬프트 밖에서도 환경 정보를 주입하고, 구독 한도는 대량 자동 호출을 전제로 하지 않는다. 재현성 때문에 Messages API 직접 호출을 유지한다.
- **거버너**: 사용자가 `sudo cpupower frequency-set -g performance` 실행, `harness.protocol --check`로 governor·EPP 모두 `performance` 확인. 재부팅 뒤에는 다시 설정해야 한다.
- **M1 언어 세 개 확정**, **프롬프트 하드웨어 공개 승인**, **반복 축소 규칙 승인**, **Python 시그니처의 `C`·`atol`·추가 정확성 형태 승인**.
- **저장소**: `https://github.com/Ustar2002/graduation_work` (비공개)를 원격으로 연결하고 첫 커밋을 올림. 디렉터리 이름은 그대로 둔다.
- **다른 자리에서 논의된 계획(커널 12개 3단계, 린트 플러그인 타임라인)은 아직 `CLAUDE.md`에 없다.** M2 시작 전에 반영하되 충돌 항목(Jacobi 1회 반복 대 T스텝, 실험 서버, 입력 크기 기준, Halide로 표현하기 어려운 커널)을 먼저 정한다.

## 2026-11-05 (M1 완료 — 첫 실제 실행 `20261105-real`)

- **결과**: `claude-sonnet-5-5`(effort medium), GEMM × 3언어 × 3샘플 = 9회 모두 pass. 속도비(대응 NumPy/OpenBLAS 4스레드 기준선 대비) 중앙값: numpy 1.01, c_openmp 0.97, c_seq 0.33(1스레드 기준선 대비 0.90). 하네스 버전 `e86a39b`, governor `performance`.
- **관찰 1 — 성능 결정의 내용**: numpy 3개는 모두 `np.matmul(A, B, out=C)` 한 줄(위임). C 6개는 모두 BLIS 방식의 블록화 + 패킹 + AVX2/FMA 6×16 마이크로커널을 직접 작성했고 OpenBLAS와 비슷한 속도를 냈다. GEMM은 교과서 커널이라 모델이 성능 결정을 직접 내려도 천장에 닿는다 → GEMM 하나로는 가설을 가를 수 없고, 덜 알려진 커널(Tier 2·3)에서 언어 간 차이가 드러날지가 관건이다.
- **관찰 2 — 토큰·비용**: 입력 25,332 + 출력 15,881 토큰, thinking 0(adaptive가 생각하지 않음). 공시 단가 기준 약 0.21달러(호출당 약 0.023달러). 사전 가정(호출당 0.06달러)의 40 % 수준이다. numpy 출력은 평균 136 토큰, C는 2,300~2,800 토큰.
- **관찰 3 — drift 플래그 9행 중 7행**: run 시작 기준선이 10.21 ms로 느린 상태에서 재졌고 대응 기준선은 대부분 8.5~8.9 ms였다(numpy/0만 10.30 ms). 기준선이 약 8.5 ms와 10.3 ms 두 상태를 오간다. 속도비는 대응 기준선을 쓰므로 영향이 작지만, drift 기준점이 run 시작 1회 측정이라 플래그가 과하게 붙는다. **제안(측정 규약 변경 — 사용자 확인 필요)**: drift 기준을 run 안 대응 기준선들의 중앙값으로 바꾼다.
- **관찰 4 — 터보 교란**: 1스레드 c_seq는 3.4~4.0 GHz, 4스레드 실행은 2.6~2.7 GHz에서 돌았다. c_seq 실행 중 패키지 온도 81~84 °C. 노트북에서는 스레드 수가 다른 조건 사이의 비교에 주파수 차이가 섞인다 → 실험 서버로 옮길 근거.
- **관찰 5 — 첫 실제 API 경로 확인**: preflight, 스트리밍, `request_id`, `effort`·`thinking` 수용, `model_served` 단일값 모두 정상.

## 2026-11-05 (M2a — Tier 1 커널과 하네스 일반화; 사용자 승인 반영)

- **drift 기준 변경(사용자 승인)**: `baseline_drift_pct`와 `baseline_drift=…%` 플래그의 기준을 run 시작 1회 측정에서 **그 run의 과제별 대응 기준선 중앙값**으로 바꿨다(`run.py` `patch_baseline_drift`, run 끝에서 일괄 계산). `baseline_run_time_ms` 열은 그대로 기록한다.
- **모델은 `claude-sonnet-5-5` 유지(사용자 확인).** 실험은 노트북에서 계속한다. 측정 시간이 문제가 되면 데스크탑으로 옮기되, 옮기면 `make probe`와 `measurement.cores`를 다시 맞추고 기존 행과 섞어 비교하지 않는다.
- **커널 12개 3단계(사용자 확정)**: M2a = Tier 1 네 개. `CLAUDE.md` §3·§4를 갱신했다. Tier 2·3은 Tier 1 실제 결과를 본 뒤 넣는다.
- **하네스 일반화**: 과제별 계약을 `tasks/<task>/signatures.yaml`의 `args`(인자 종류·dtype·형태식)·`work`·`baseline`으로 옮기고(`harness/taskspec.py`), 워커·분류기·run·프롬프트가 이를 읽는다. 입력은 float32·int32 배열, 출력은 float32 배열 1개, 스칼라는 `int`만 지원한다(**정수 출력은 Tier 2에서 추가**). GEMM 프롬프트는 M1 실행과 바이트 단위로 같다(`prompt_hash` 동일 확인). GEMM 밖의 과제는 arg_order 패턴으로 2-D 출력의 전치(`transposed_output`)만 본다.
- **`jacobi2d`** **[CLAUDE.md 이탈: 원문은 1회 반복]**: `jacobi2d(src, dst, H, W, T)`, T 스텝, 경계 고정. 버퍼 교환을 모델의 결정으로 남기려고 작업 버퍼를 주지 않는다. 형태 (5,7,3)·(130,200,4)·(512,512,8) / 성능 (4096,4096,20). rtol 1e-5, atol 1e-6. 기준선은 손으로 쓴 C+OpenMP(`baseline/baseline.c`, 샘플과 같은 플래그로 run마다 빌드).
- **`spmv_csr`**: `spmv_csr(row_ptr, col_idx, vals, x, y, n_rows, n_cols, nnz)`, int32 인덱스. 행 길이 0~2·avg 균등(빈 행 포함), 열 인덱스는 행 안에서 오름차순·유일. small 10⁴×10⁴ 밀도 1e-3, large 10⁶행·행당 평균 16개(nnz 약 1,600만; 원문 "밀도 1e-3"을 10⁶행에 그대로 쓰면 10⁹개라 불가). rtol 1e-4, atol 1e-4. 기준선 C+OpenMP.
- **`dot`(STREAM triad 대신)**: reduction 병렬화가 표의 결정 사항이라 dot을 골랐다. `dot(x, y, out, n)`, 결과는 `out[0]`. 입력 uniform[0,1)이라 합이 n/4까지 커진다 — n=2²⁵에서 NumPy float32 내적은 상대오차 1.5e-4로 통과, 단순 float32 순차 누적은 7e-2, 4분할 float32 누적은 6.6e-3으로 실패(실측). **누적 정밀도를 모델의 결정 사항으로 만들기 위한 의도된 설계**다(CLAUDE.md §6의 린트 예시 `numeric/accumulate-in-float32`와 같은 현상). rtol·atol 1e-3. 출력이 스칼라 하나라 `numeric_pattern_tol_ratio`를 500으로 넓혔다(단순 누적 커널이 `numeric_instability`로 분류되는 것 확인). 기준선은 NumPy `np.dot`.
- **입력 크기 기준(순차 C 0.3~3초)과의 차이**: 이 머신의 기준선 중앙값은 gemm 약 9 ms, jacobi2d 113 ms(1스레드 230 ms), spmv_csr 13 ms(1스레드 43 ms), dot 12 ms다. spmv·dot은 기준보다 짧다. 더 키우면 테스트 파일이 GB 단위가 되고 14 GiB 노트북의 워커 메모리 한도(8 GiB)에 걸린다. ms 단위 측정은 M1에서 변동계수 1~10 %로 가능함을 확인했다. → 데스크탑으로 옮기면 다시 정한다.
- **Halide로 표현하기 어려운 커널(sort, BFS, prefix scan)** 은 M3에서 Halide를 넣을 때 정한다(해당 조합 제외 또는 Halide 대상 축소).
- **린트**: 마감과 무관하게 필요해질 때 구현한다(사용자). M1·M2는 빈 결과로 진행.
- **검증**: 단위 테스트 170개 통과. GEMM 더미 회귀 9행이 M1과 같은 분류. 새 과제 3개 × 언어 3개의 정답 더미 9행 모두 pass(기준선 포함 69초).

## 2026-11-20 (M2b–M6 코드 선행 구현 — Windows 에서 작성, Linux 일괄 테스트 대기)

- **작업 방식(사용자 지시)** **[CLAUDE.md 이탈: "각 단계 끝에 결과를 보여주고 멈춘다", "앞 단계 결과를 보고 다음 단계를 넣는다"]**: 데스크탑의 Linux 환경은 나중에 준비하고, 그 전에 구현할 수 있는 코드를 Windows 에서 최대한 써 둔 뒤 Linux 에서 한 번에 테스트한다. 범위는 M2b·M2c(커널 8개), M3(numba·halide_auto·halide_manual·rust), M4(키워드 인자·문법 안내·수정 루프), M5(성능 결정 요약·ASan/UBSan·린트 베이스라인), M6(부트스트랩 CI·그래프)의 **코드**다. 하지 않는 것: 실제 API 호출(M2a 60회 포함 — 여전히 단계마다 확인받는다), 대시보드(CLAUDE.md 가 마지막으로 둔 항목), M7. 인터페이스는 `docs/harness_design_v3.md`(설계 계약 v3)에 먼저 정하고 그룹별로 나눠 구현했다. Linux 에서만 확인되는 항목은 `docs/linux_test_plan.md` 에 모았다.
- **로컬 개발 환경**: `.venv-win`(Python 3.14.0, numpy 2.3.5, pandas 3.1.1, matplotlib, numba 0.68.1, halide 22.1.0, ruff, pyflakes, ziglang; git 에는 올리지 않는다), Rust 1.92(MSVC). gcc·taskset·sysfs 는 없다. C 커널은 `python -m ziglang cc` 로 **직렬** 컴파일해 논리만 확인했다(OpenMP pragma 무시, `omp.h` 스텁) — 병렬 실행·gcc 고유 동작·sanitizer·cppcheck/clang-tidy 는 Linux 에서 처음 돈다.
- **Windows 지원 범위는 "임포트가 되고 python 계열 백엔드의 check 까지 돈다"가 전부다.** `procutil` 은 POSIX 가 아니면 프로세스 그룹 정리 없이 `proc.kill()` 만 하고, 워커는 `resource`/`prctl` 없이 돈다. 비-Linux 에서 만든 행에는 `platform=<sys.platform>` 플래그(정보용 아님)가 붙어 `protocol_ok=0` 이다. 측정은 Linux 에서만 한다.
- **정수 출력과 정확 비교**(CLAUDE.md §4 "정수 출력의 처리는 해당 단계 시작 때 정한다"): 출력 dtype 에 int32 를 허용하고 `tolerance.yaml` 의 `compare: exact` 로 원소가 전부 같아야 통과한다. 분류 규칙은 같은 순서에서 수치 규칙(1·5)만 빠진다: 같음 → pass, 전부 센티널 → `output_not_written`, `out − sentinel == ref` → `accumulated_into_c`(0 으로 시작한다고 가정한 히스토그램이 여기 걸린다), 일부 센티널 → `partial_write`, 그 밖 → `mismatch`. 정확 비교 과제에는 `numeric_instability` 가 없다. 참조는 int32 로 저장한다(파일 크기). 정수 출력 과제의 참조에는 센티널 값(−2048)이 나오지 않게 생성기가 보장한다.
- **Tier 2·3 커널 계약**(§4 표의 여덟 개; 세부는 `tasks/<task>/`):
  - `histogram(keys, hist, n, n_bins)` — int32 키, int32 개수. 실수 값을 구간으로 나누는 형태는 경계 반올림이 모호해 쓰지 않았다.
  - `prefix_sum(x, out, n)` — **int32 포함형 누적 합** **[CLAUDE.md §4 는 정수 출력을 histogram·sort·BFS 로만 적었다]**: float32 누적 합은 n = 2²⁵ 에서 합이 2²⁴ 를 넘으면 더하는 값이 반올림에 묻혀 순차 float32 구현이 구조적으로 틀리고, 2-pass 병렬 구현은 누적 순서가 달라 허용오차를 과제 크기에 맞춰 넓혀야 한다. 이 커널의 결정 사항은 정밀도가 아니라 알고리즘 재구성(2-pass)이므로 순서와 무관하게 정확한 정수 합으로 정했다. 누적 정밀도는 `dot` 이 이미 본다.
  - `nbody(pos, mass, acc, n)` — 가속도, 연화 길이 ε = 0.01 고정, 입력은 AoS (n,3)(SoA 로 바꿀지는 모델의 결정).
  - `sort(keys, out, n)` — int32 키만(음수 포함), 오름차순.
  - `cholesky(A, L, n)` — **LU 대신 Cholesky**: 양의 정부호 입력에서 분해가 유일하고 피벗이 없어 정답 검증이 단순하다(LU 는 피벗 여부를 명세로 금지해야 한다).
  - `floyd_warshall(W, D, n)` — 완전 그래프·양의 유한 가중치(출력에 inf 가 없어야 분류기의 NaN/Inf 규칙과 충돌하지 않는다).
  - `bfs(row_ptr, col_idx, dist, n_vertices, n_edges, source)` — 출력은 레벨(간선 수), 도달 불가 −1. 부모 배열은 탐색 순서에 따라 달라 자동 검증이 안 되므로 쓰지 않았다.
  - `fft(x, out, n)` — 복소수는 실수·허수 교차 float32 배열 하나(출력 1개 제약 유지), 순방향·정규화 없음, n 은 2의 거듭제곱, 입력 진폭 1/√n(출력 크기가 형태와 무관해져 과제당 상수인 atol 이 성립한다).
  - 큰 형태는 잠정값이다(데스크탑에서 `make probe` 뒤 다시 정한다 — 2026-11-05 결정과 같다).
- **(과제, 언어) 조합은 검증된 정답 더미(`harness/dummy_responses/<task>/<language>/0.md`)가 있어야 실험 대상이다.** 하네스 경로가 그 조합에서 실제로 도는지와 그 언어로 표현 가능한지를 한 번에 보장한다. 제외는 `tasks/<task>/signatures.yaml` 의 `unsupported: {language: 이유}` 에 적고 분석 표에 `n/a` 로 나온다. Halide 는 `sort`·`bfs`(데이터 의존 제어 흐름)를 제외한다(2026-11-05 미결 항목의 답). 나머지 조합의 통과 여부는 아래 "구현 보고"에 적는다.
- **시그니처·매개변수 설명은 `harness/sigs.py` 가 `args` 에서 만든다.** 과제 12개 × 언어 7개를 YAML 에 손으로 쓰면 어긋나기 쉽다. 기존 4개 과제의 numpy/c_seq/c_openmp 항목은 YAML 것이 우선해 프롬프트가 바이트 단위로 그대로다(`prompt_hash` 불변을 테스트로 고정).
- **Halide 계약**(pip `halide` 22.1, Python 바인딩): LLM 은 `def <fn>(<입력 ImageParam…>, <크기 int…>)` 가 출력 `hl.Func` 를 돌려주는 **파이프라인 빌드 함수**를 쓴다. 크기는 `Param` 이 아니라 그 호출의 실제 정수로 주고 형태마다 새 파이프라인을 만든다(T 스텝 반복·FFT 단계처럼 크기에 따라 파이프라인 구조가 달라지는 커널을 쓸 수 있게). 좌표는 Halide 기본(`hl.Buffer(ndarray)` 의 `reverse_axes=True`: 마지막 축이 첫 좌표)을 그대로 두고 프롬프트에 사실로 적는다 — 행·열 뒤바뀜은 관찰 대상이다. `halide_auto` 는 하네스가 실제 크기로 estimate 를 넣고 **Mullapudi2016**(결정적, 가중치 파일 없음; Adams2019 는 탐색에 난수가 있다)을 적용한다. 실측(2026-11-20, 512³ GEMM): 스케줄 없음 105 ms, Mullapudi2016 2.0 ms, Adams2019 3.8 ms. 오토스케줄러 플러그인은 이름으로는 로드되지 않고 패키지 디렉터리의 절대 경로로만 로드된다(실측). 파이프라인 구성·스케줄·JIT 시간(약 0.1–1 s)은 측정에 넣지 않는다. 스레드 수는 `HL_NUM_THREADS`. → **사용자 확인 필요**(측정 규약의 확장): 오토스케줄러 선택, JIT 제외.
- **Numba 계약**: `@njit("<서명 문자열>")` 로 **명시 서명·즉시 컴파일**을 요구한다(CLAUDE.md "타입 전부 명시"). 실측: 명시 서명이면 임포트 때 0.83 s 에 컴파일되고 첫 호출이 5.9 ms, 서명이 없으면 `signatures` 가 비어 첫 호출에 JIT 가 섞인다 → 워커가 로드 뒤 `signatures` 가 비어 있으면 `signature_mismatch`/`numba_signature_missing` 으로 판정한다(빌드 단계의 AST 검사와 이중). `parallel`·`fastmath` 등 옵션은 모델의 결정으로 남긴다. 스레드는 `NUMBA_NUM_THREADS`, 스레딩 계층은 `omp` 로 고정한다. 읽기 전용 배열은 쓰기 가능 서명과 맞지 않으므로(실측 TypeError) 하네스는 지금처럼 쓰기 가능한 뷰를 넘기고 호출 뒤 변조를 검사한다. → **사용자 확인 필요**: 스레딩 계층.
- **Rust 계약**: `cdylib` + `#[no_mangle] pub unsafe extern "C" fn`, 워커는 C 와 같은 ctypes 경로로 부른다. 의존성은 `rayon` 하나(버전 고정, `Cargo.lock` 저장소 포함), `opt-level=3, codegen-units=1, lto=fat, panic=abort`, `RUSTFLAGS=-C target-cpu=native`, 빌드는 `--offline --locked`(`make rust-setup` 이 1회 내려받는다). 패닉은 abort → `runtime_crash`. Rust 에는 sanitizer 를 돌리지 않는다(nightly 전용).
- **`args=keyword`**: Python 계열(numpy·numba·halide)은 키워드 전용 시그니처. C 는 하네스가 제공하는 `kernel_args.h` 의 구조체를 `#include` 해 필드 이름으로 접근한다(모델이 구조체를 다시 선언하게 하면 필드 순서 실수가 ABI 불일치로 나타나 "이름으로 전달"이라는 조건의 뜻이 사라진다). Rust 는 같은 구조체를 `src/kernel_args.rs` 로 준다.
- **`grammar_hint=on`** 은 `prompts/hints/<language>.md`(100줄 이하, 과제와 무관한 문법·API 사실만; 성능 조언 없음)를 붙인다. 힌트 파일이 없는 언어는 이 조건에서 건너뛴다(on 과 off 가 같은 프롬프트가 되는 것을 막는다).
- **수정 루프**(`harness/repair.py`): 턴마다 행 하나(`turn` 이 행 키에 있다), `turns_to_pass` = 통과한 턴. 대화 형식은 다중 턴 메시지(원래 프롬프트 → 응답 → 수정 요청 → …)다. 수정 요청에 주는 정보는 **컴파일러·테스트 러너가 줄 법한 것까지만**: 컴파일러 출력, 예외·시그널, 형태별 통과/실패와 최대 오차·실패 원소 수, 타임아웃. 하네스 내부 패턴 이름(`transposed_output` 등)과 참조값은 주지 않는다(진단을 대신해 주면 수정 능력이 아니라 하네스의 분류기를 재는 것이 된다). `feedback=compiler+lint` 는 여기에 린트 메시지 절만 더한다. 두 피드백 조건은 같은 turn 0 응답에서 갈라지게 한다(`--seed-from-feedback`; 대응 비교). bench 타임아웃(pass=1)은 수정 대상이 아니다. → **사용자 확인 필요**: 피드백에 넣는 정보의 범위.
- **`memory_error` 규칙**: 그 턴의 판정이 `runtime_crash` 또는 `wrong_output` 이고 ASan 이 오류를 보고하면 `memory_error` 로 바꾼다(이전 판정은 `error_detail` 에 남긴다). **통과한 샘플의 `pass` 는 바꾸지 않는다** — `pass` 는 정확성 정의(2026-11-04)이고 잠재 결함은 `asan_findings`/`ubsan_findings` 열로 본다. TSan 은 하지 않는다(libgomp 가 계측되지 않아 거짓 양성; `race_condition` 은 지금처럼 반복 실행 간 불일치로만 판정). sanitizer 는 C 백엔드에서만, 정확성 형태로만 돈다. → **사용자 확인 필요**(분류표의 적용 규칙).
- **성능 결정 요약**: 벡터화 리마크는 gcc `-fopt-info-vec`(2026-11-03 결정과 같다). CLAUDE.md §5 스키마에 `decisions_by_library` 키를 **추가**했다 **[스키마 확장]** — `np.matmul` 한 줄처럼 결정을 라이브러리에 넘긴 경우를 LLM·컴파일러 어느 쪽에도 넣을 수 없다(가설의 세 번째 답).
- **린트**: `plugins/lint/run_lint.py` 는 CLAUDE.md §6 계약의 기성 린터 베이스라인(Python 계열 ruff·pyflakes, C cppcheck·clang-tidy)이다. 하네스를 임포트하지 않는 독립 CLI 라 인지프 ML lint 로 바꿔 끼울 수 있다.
- **결과 스키마는 열을 추가하지 않았다.** 기존 `results/runs.csv` 와 헤더가 같다. `backend` 값에 numba·halide·rust, `turn ≥ 1` 행이 생길 뿐이다.
- **없는 도구는 모델 오류가 아니다**: 처리할 언어에 필요한 도구(gcc, cargo, numba, halide, 오토스케줄러 플러그인)가 없으면 행을 쓰기 전에 멈춘다(exit 7). 컴파일러가 없어서 생긴 실패가 `compile_error` 행으로 남으면 pass 율이 오염된다.

## 2026-11-22 (M2b–M6 구현 보고 — Windows 검증 결과, 리뷰 반영, 확인이 필요한 결정)

2026-11-20 절은 구현 전에 정한 것이고, 이 절은 구현하는 동안 정하고 잰 것이다. 시간과 오차 비율은 따로 적지 않으면 Windows 개발 머신(i7-1260P, 다른 에이전트가 함께 돌던 공유 상태)에서 잰 값이다. 시간은 참고값이고 Linux 에서 다시 잰다. 실제 API 호출은 하지 않았고 커밋도 하지 않았다. 마지막 항목(§6)은 사용자가 확정할 목록이다.

### 1. 작업 방식과 검증 범위

- **작업 분할**: 설계 계약 v3(`docs/harness_design_v3.md`)에 파일 소유를 나눠 적고 그룹별로 구현했다. G1 기반(procutil·worker 의 비-Linux 경로, taskspec, classify 정확 비교, testgen, sigs, paths, devcheck), G2a 워커·커널 계층, G2b 추출·빌드, G3 설정·생성·수정 루프·프롬프트, G4 run·schema·protocol, G5 perf_summary, G6 safety·린트, G7 분석, T1–T4 새 커널 8개, L 새 언어 더미다. 2026-11-21 사용량 한도로 에이전트 8개가 중단됐고, 남은 작업은 2026-11-22 작업 트리에 남은 상태에서 이어서 했다.
- **블라인드 검증**: 새 커널마다 다른 에이전트가 `spec.md` 와 렌더된 시그니처만 보고 numpy·c_seq·c_openmp 를 구현했다. 24개 모두 통과했고, 그 과정에서 드러난 명세 결함은 §3 에 적었다.
- **리뷰**: 8개 영역 리뷰가 27건을 냈다. 별도 검증자가 각 건을 반박하려 했고 27건 모두 확정됐다(§5). 그룹끼리 서로의 모듈에 대해 보고한 문제(K 번호)도 같은 단계에서 고쳤다.
- **에이전트 모델**: 핵심 모듈 구현은 세션 모델, 블라인드 검증·더미 작성·리뷰 일부·수정·문서는 Opus 로 돌렸다(주간 사용량 한도). 결과가 실행으로 판정되는 작업에 한했고 오케스트레이터가 통합 실행으로 다시 확인했다.
- **도구**: `.venv-win` 의 numba 0.68.1·halide 22.1.0, Rust 1.92(cargo 1.92, MSVC), C 는 `python -m ziglang cc`. numba·halide·cargo 는 실제 도구로 돌았다.
- **C 빌드 두 가지(개발용, `tools/devcheck.py --cc`)**: `zig` 는 OpenMP pragma 를 무시하는 직렬 빌드다(`omp.h` 스텁, 판정 줄에 `toolchain=zig-serial`). 논리만 보고 경쟁 상태는 못 본다(gemm 의 경쟁 더미 `c_openmp/2.md` 가 PASS). `zig-omp` 는 직접 줄 때만 고르며, `-fopenmp -c` 로 컴파일해 Visual C++ 재배포 패키지의 LLVM OpenMP 런타임 `libomp140.x86_64.dll`(`DEVCHECK_LIBOMP` 로 바꾼다)에 링크하고 실제 스레드로 돌린다(`toolchain=zig-omp`). 컴파일러(clang)도 런타임(libomp)도 측정 경로의 gcc + libgomp 가 아니므로 Linux 확인을 대신하지 않는다.
- **오케스트레이터가 직접 확인한 것**: 전부 dummy provider 였고, 결과 경로를 스크래치로 바꾼 설정 사본으로 돌려 저장소 `results/` 는 건드리지 않았다.
  - `tools/devcheck.py dummies --languages numba,halide_auto,halide_manual,rust`(12과제): 44 pass, 0 fail, 0 missing, 0 problem, 4 unsupported(sort·bfs × Halide 두 언어).
  - `tools/devcheck.py dummies --cc zig-omp --languages c_openmp,c_seq`: 24 pass. c_openmp 12개는 4스레드 실제 실행이고 `DEVCHECK_OMP_THREADS=8` 에서도 prefix_sum·sort·histogram·bfs·fft 가 통과했다. gemm 경쟁 더미는 3회 모두 `race_condition`(rect·small 비결정)으로 분류됐다. C+OpenMP 기준선 다섯 개(jacobi2d, spmv_csr, histogram, nbody, floyd_warshall)도 통과했다(dot·bfs 기준선은 OpenMP 빌드가 아니다).
  - run itA: `harness.generate --provider dummy` 로 12과제 × numpy·numba·rust, n=1(36회) → `harness.run --skip-bench` 36행 모두 pass=1. `omp_env` 에 `NUMBA_NUM_THREADS`·`NUMBA_THREADING_LAYER=omp`·`RAYON_NUM_THREADS`, rust 행의 `cflags` 에 release 프로필, lint 열은 rust NA·numba 0, protocol 플래그 `platform=win32`·`governor=unknown`·`taskset_missing`(비-Linux 이므로 설계대로 `protocol_ok=0`).
  - run itB: 9과제 × halide_auto·halide_manual 18행 모두 pass=1(spmv_csr 의 Halide 더미는 그 뒤에 완성돼 devcheck 로만 확인했다).
  - run itC(bench 경로, 시간은 의미 없음): gemm·dot·fft × numpy·numba·rust·halide_auto·halide_manual 15행 pass=1, `time_median_ms`·`time_cv`·대응 기준선·`speedup_vs_baseline` 이 채워졌다.
  - run itD(수정 루프): gemm × numpy, n=5 더미 변형 → `LLM_PROVIDER=dummy python -m harness.repair --max-turns 2`. 샘플 2(부정확)는 턴 1 에 통과(`turns_to_pass=1`), 샘플 1(`signature_mismatch`)·4(`returned_array_not_written`)는 턴 더미가 없어 턴 2 까지 같은 실패를 반복했다. `--seed-from-feedback compiler --feedback compiler+lint` 로 만든 두 번째 조건도 같은 결과였다.
  - `analysis.aggregate --ci --coverage`, `--ci --by-turn --conditions`(itD: 두 피드백 수준 모두 pass_within_0/1/2 = 0.4/0.6/0.6, 대응 n = 5/5, 차 0), `analysis.plots --all-runs`(그림 4개), `harness.run --rebuild-csv`(result.json 89개 → 89행).
- **여기서 확인할 수 없는 것**: C 언어의 `harness.run` 경로(gcc 가 없어 설계대로 exit 7), taskset, CPU 거버너, ASan/UBSan, cppcheck·clang-tidy, 모든 시간 측정. Linux 데스크탑 한 세션에서 확인한다. 그곳에서 `python tools/devcheck.py dummies` 를 옵션 없이 돌리면 80 pass, 0 fail, 0 missing, 0 problem, 4 unsupported 이고 어느 행에도 `toolchain=` 표시가 없어야 한다(12과제 × 7언어 − 4). 그 뒤 위 네 run 을 일곱 언어 전부로 다시 돈다.

### 2. 모듈별 결정 (동작을 정하거나 v3 계약과 다른 것)

**기반·devcheck(G1)**

- **비-Linux 축소 경로**: `procutil` 은 POSIX 가 아니면 새 세션·pdeathsig 없이 띄우고 타임아웃 때 직접 자식만 `kill()` 한다(rc 는 운영체제 값이고 `signal_name` 은 None 이라 호출자는 `timed_out` 을 먼저 본다). 타임아웃 뒤에는 파이프를 닫지 않으므로(읽기 스레드가 손자 프로세스가 끝날 때까지 막힌다) devcheck 는 컴파일러·워커 출력을 파일로 받는다. 워커는 `resource` 가 없으면 `limits.unsupported`, `--max-mem-gb <= 0` 이면 `disabled` 를 기록하고 setrlimit 을 부르지 않는다. POSIX 경로는 바뀌지 않았다.
- **정확 비교의 세부**: `compare: exact` 인데 rtol·atol·`numeric_pattern_tol_ratio` 에 0 이 아닌 값이 있으면 모순된 파일로 보고 `load_tolerance` 가 거부한다. 차·누적 검사는 int64 로 한다(sort 의 전체 int32 범위에서도 넘치지 않게). 정수 출력 과제의 참조가 int32 로 저장돼 있지 않으면 조용히 자르지 않고 `worker_error`(rc=3)로 멈춘다(테스트 파일의 버그이지 커널 오류가 아니다).
- **testgen**: 기존 과제의 npz 는 변경 전 코드로 만든 것과 바이트 단위로 같고, 네 과제의 `test_hash` 가 Linux 에서 만든 기록과 같아 입력 생성이 플랫폼과 무관함을 확인했다. `manifest.json` 은 LF 로 쓴다.
- **[v3 §3 이탈] 스칼라 매개변수의 기본 설명**: 'always consistent with the array shapes above' 는 배열 형태 식에 나오는 dim 에만 쓴다. 어떤 형태 식에도 나오지 않는 스칼라(jacobi2d 의 `T`, bfs 의 `source`)는 'a scalar parameter described in the task specification' 이다(형태와 무관한 인자에 틀린 사실을 말하지 않게). 그런 인자에는 `signatures.yaml` 에 `doc` 을 주는 것이 원칙이다.
- **`signature_bundle`**: C·Rust 의 keyword 모드는 YAML `param_docs` 앞에도 구조체 안내 문단을 붙여 12과제가 같은 틀이 된다. Rust 의 안내 문단은 `mod kernel_args;` 와 `use kernel_args::DotArgs;` 꼴의 선언이 필요하다는 것을 적는다(없으면 rustc E0412). 매개변수 설명에는 수치 라이브러리 이름이 없다(테스트로 강제).
- **devcheck**: 종료 코드는 0 통과 / 1 커널이 틀림 / 2 사용법·도구 문제다. `dummies` 는 지원하는 (과제, 언어) 에 `0.md` 가 없으면 missing 으로 실패한다(검증된 더미가 있어야 실험 대상). 코어 고정과 측정 규약 검사는 하지 않고 `--bench` 는 warmup 0·repeats 1 의 참고값이다. Windows 의 zig 빌드는 `tools/devcheck_shim/shim.h` 로 malloc 계열을 16바이트 정렬로 다시 세운다(glibc x86-64 와 같게; 과정렬하면 Linux 에서 터질 정렬 가정 버그가 가려진다). Windows 에서는 `long` 이 32비트이고, C 커널의 접근 위반은 ctypes 가 OSError 로 바꿔 예외 경로의 `runtime_crash` 가 된다(Linux 는 SIGSEGV 로 워커가 죽는 경로). 판정의 기준은 Linux 실행이다.
- **BOM**: 소스 파일과 raw JSON 은 BOM 을 떼고 읽지만 응답 문서(.md)는 떼지 않는다. dummy provider 가 BOM 째 응답으로 쓰므로 devcheck 만 떼면, 실제 파이프라인에서는 추출이 실패하는 더미를 devcheck 가 통과시킨다.
- **테스트 표시**: `tests/_support.py` 의 `requires_modules` 는 비-Linux 에서 패키지가 없으면 skip, Linux 에서 없으면 실패다(예: Linux 에 Anthropic SDK 가 없으면 `tests/test_llm_client.py` 가 실패한다).

**워커·커널 계층(G2a)**

- **커널 계층 분리**: `harness/kernels`(cabi·pyfunc·halide_pipe)가 로드·호출을 맡고 워커는 제한·버퍼·결과 문서·측정 루프만 한다. `open_backend` 는 하네스가 고른 도구(임포트, Numba 스레딩 계층 시작, 오토스케줄러 로드와 시험 적용)만 건드리고 LLM 코드는 `load` 부터 돈다. 그 전에 결과 문서에 `worker_error` 를 써 둔다. Halide 22.1 은 틀린 오토스케줄러 매개변수(없는 키, 해석되지 않는 값)에 예외 대신 프로세스를 중단하기 때문이다(실측: Windows 종료 코드 0xC0000409).
- **Numba 스레딩 계층은 커널보다 먼저 띄운다**: 설정한 계층을 로드할 수 없으면 `parallel=True` 커널의 임포트가 `ValueError('No threading layer could be loaded')` 로 실패해(실측: tbb 가 없는 머신에 tbb 지정) 도구 문제가 `compile_error` 행이 된다. 먼저 띄우면 같은 실패가 `worker_error`(행 없음)다. `omp` 계층은 Windows 에서도 동작했다.
- **Halide 어댑터**: estimate 는 빌드 함수가 돌아온 뒤 실제 크기로 준다(빌드 함수가 준 값을 덮는다; 512³ GEMM 에서 순서를 바꿔도 Mullapudi2016 의 스케줄이 같았다). 하네스가 부르는 컴파일 단계(파이프라인 구성·오토스케줄·JIT·`print_loop_nest`)의 출력은 stderr 로 돌린다. Halide 는 컴파일 경고를 Python `print` 로 내므로 그대로 두면 `contract_violation` `stdout_output` 이 된다. 빌드 함수(LLM 코드)가 호출되는 동안의 출력과 `hl.print` 는 커널의 출력이다.
- **halide_manual 의 인라인 폭발(실측)**: 스케줄이 없는 T 스텝 Jacobi(스텝마다 Func 하나)는 전부 인라인돼 식이 5^T 로 커지고, 512²·T=8 에서 JIT 가 9분을 넘겼다. 실험에서 이런 응답은 빌드·check 단계의 `timeout` 으로 남는다.
- **관찰 단계는 판정이 아니다**: `--inspect` 수집 전에 문서에 `inspect_error` 와 `in_progress.test='inspect'` 를 써 둔다. 성능 형태의 파이프라인 구성이 워커를 죽여도 `load_ok=true` 가 남고, run.py 는 경고 뒤 check·bench 로 계속한다(§6 B5).
- **키워드 모드 C 구조체**: 과제 `args` 순서로 만든 `ctypes.Structure` 의 포인터다. 스칼라와 포인터가 번갈아 나오는 과제로 실제 컴파일러 배치와 맞음을 확인했고(`_pack_=4` 로 바꾸면 틀린 출력), rustc 로 만든 cdylib(`#[repr(C)]`)도 같은 경로로 통과한다.
- **`--sanitize`(리뷰 worker#1)**: sanitizer 실행에서만 배열마다 정확한 크기의 C 할당(POSIX `posix_memalign`, 64바이트 정렬, 가드 없음)을 쓴다. 평소 배치는 가드가 같은 블록 안에 있어 가드 1024원소 이내의 넘침이 ASan 에 보이지 않는다. 부수 효과로 gemm 의 전치 여유(`guard_extra_of`)도 없어서, M·N 밖에 쓰는 커널은 safety 실행에서 heap-buffer-overflow 로 보고된다.
- **check 문서**: 검사한 호출이 하나도 없으면 `inputs_mutated`·`oob_write` 는 None(행에서 NA)이다(리뷰 verdicts#5). 호출 기록에 `n_fail`·`n_total` 을 싣는다(K6; 수정 요청의 "k of n elements outside tolerance" 용).

**추출·빌드(G2b)**

- **cargo 의 신선도 판단(실측, cargo 1.92)**: run 당 `CARGO_TARGET_DIR` 하나를 샘플끼리 나눠 쓰면 패키지 이름이 같아 fingerprint 가 겹친다. cargo 는 소스 mtime 만 보므로, mtime 이 오래된 소스를 빌드하면 컴파일 오류가 있어도 rc=0 이고 이전 샘플의 라이브러리가 남는다. 그래서 `build_rust` 는 빌드마다 먼저 `cargo clean --release -p kernel` 로 커널 패키지만 지운다(rayon 빌드는 남는다).
- **Rust 동시 빌드 잠금(K1)**: 같은 target 을 쓰는 동시 빌드가 서로의 라이브러리를 복사하는 것을 실측했다(병렬 devcheck 에서 올바른 커널이 `symbol <fn> not found` 로 실패). 정리 → 빌드 → 라이브러리 읽기 전체를 `<target>/.harness-build.lock`(Windows `msvcrt.locking`, POSIX `fcntl.flock`) 아래에서 한다. 대기 상한은 2 × (정리 120 s + `build_timeout_s`)이고 넘기면 `ToolchainMissing` 이다. 측정은 바뀌지 않고 빌드만 직렬화된다.
- **Rust 템플릿**: `rayon = "=1.12.0"`, `Cargo.lock`(형식 v4 — cargo 1.78 이상 필요)을 저장소에 둔다. 두 파일은 `.gitattributes` 로 LF 고정이고(샘플 디렉터리로 바이트 그대로 복사된다) `harness_version` 의 소스 해시에 들어간다(K15).
- **cargo 실패의 분류**: run 시작 전에 템플릿의 자리표시 크레이트를 샘플과 같은 명령·환경·잠금으로 한 번 빌드한다(`probe_toolchain`, K4·K14). 실패하면 exit 7 이다(환경 탓의 링크 실패가 모든 rust 샘플의 `compile_error` 가 되는 것을 막는다). 샘플 빌드에서는 로그에 ``error: could not compile `kernel` `` 이 있을 때만 `compile_error`, `Compiling kernel` 뒤의 시간 초과만 `timeout` 이다. 그 앞의 실패(의존성 캐시 없음, 잠금 불일치, 링커 없음)와 rc=0 인데 산출물이 없는 경우는 `ToolchainMissing` 이다.
- **cargo·rustc 의 환경**: POSIX 는 최소 환경에 `CARGO_HOME`·`RUSTUP_HOME`·`RUSTUP_TOOLCHAIN` 만 통과시키고(`protocol.toolchain_env`), 도구 점검·환경 지문·generate 의 rustc 버전도 같은 환경으로 읽는다(리뷰 linux_paths#3). `RUSTFLAGS`·`CARGO_TARGET_DIR`·`CARGO_TERM_COLOR=never`·`CARGO_TERM_QUIET=false` 는 덮어쓴다(사용자 cargo 설정이 분류를 바꾸면 안 된다).
- **nm**: 실패(rc≠0·시간 초과)는 빈 심벌 목록이 아니라 `ToolchainMissing` 이다(Linux 동작 변경). 내보낸 함수로 `T`·`W`·`i`(GNU ifunc)를 받는다(리뷰 extract_build#2). Rust 는 ELF 일 때만 같은 검사를 하고, undefined·forbidden·thread 심벌 관찰은 하지 않는다(std 가 pthread 등을 늘 참조한다).
- **Numba 빌드 규칙과 실측(0.68.1)**: 과제 함수의 데코레이터는 `njit`(별칭 무관) 또는 `jit(nopython=True)` 호출이고, 서명은 첫 위치 인자나 `signature=`·`signature_or_function=` 키워드로 준다. 실측: 문자열·리스트·타입 식·`signature_or_function=` 은 임포트 때 컴파일되지만 `signature=` 키워드는 Numba 가 서명으로 쓰지 않고 조용히 지연 컴파일하며(`signatures` 0개), 문자열 튜플은 임포트 때 TypeError 다. 그래서 빌드의 AST 검사와 워커의 `signatures` 검사를 둘 다 둔다. 기대 서명과 글자만 다른 것(`f4[::1]` 등)은 `signature_matches_expected=False` 로만 남긴다.
- **Halide 빌드**: `schedule_calls` 는 계약의 스케줄 메서드 이름 50개의 속성 호출이다(리터럴과 임포트한 모듈의 호출 제외; AST 로는 수신 객체의 타입을 몰라 문자열 변수의 `split` 도 센다). 실측: halide_auto 에서 소스에 스케줄 호출이 있으면 Mullapudi2016 이 `cannot auto-schedule function … partially specified schedules` 로 실패한다. 그 샘플은 워커 단계에서 실패하고 `schedule_in_halide_auto` 가 원인을 설명하는 관찰값이 된다.
- **Rust 추출**: 주석·문자열을 지운 텍스트에서 본문이 있는 `pub [unsafe] extern "C" fn {fn}(` 를 찾고, 바로 앞에 `#[no_mangle]`/`#[unsafe(no_mangle)]` 이 없으면 `no_mangle_missing` 이다. extract 의 halide·rust 는 시그니처 묶음(bundle)이 없으면 ValueError(하네스 오류)다.
- **ABI 가 같은 다른 표기**: C 는 `signed`·`signed int`·`int signed`·`int32_t` 를 `int` 로(리뷰 extract_build#1), Rust 는 `c_int`·`c_float`(접두 없음 또는 `std::ffi::`·`core::ffi::`·`std::os::raw::`·`libc::`)를 `i32`·`f32` 로 받고 `signature_exact=False` 로 남긴다(K2). 크기·부호가 다른 타입은 그대로 `param_mismatch` 다(§6 B4).
- **C 전처리 컴파일러**: extract 는 설정의 `build.cc` 로 전처리한다(K3·K17). 예전에는 gcc 고정이라 gcc 가 없으면 원문 검사(preprocess_failed)로 조용히 떨어졌다.

**설정·생성·수정 루프·프롬프트(G3)**

- **스레드 변수 강제**: numba·halide·rust 언어는 `languages.<lang>.env` 에 `NUMBA_NUM_THREADS`·`HL_NUM_THREADS`·`RAYON_NUM_THREADS` = `"${threads}"` 가 있어야 하고 아니면 `load_config` 가 거부한다. 이 런타임들은 `OMP_NUM_THREADS` 를 읽지 않아 빠뜨리면 논리 코어 수만큼 스레드를 띄우고 프롬프트의 문장이 거짓이 된다. env 값의 자리표시자는 `${threads}` 만 허용하고, `halide.autoscheduler_params` 의 `parallelism` 은 예약어다(하네스가 스레드 수로 넣는다).
- **프롬프트 템플릿 변수 추가 [v3 §9.1 확장]**: `numba_threading_layer`, `rayon_version`(템플릿 `Cargo.toml` 에서 읽음), `halide_coord_example`. positional·hint off 에서는 `args_note`·`grammar_hint` 가 빈 문자열이라 기존 12개 프롬프트(4과제 × 3언어)의 `prompt_hash` 가 그대로다(`PromptHashTest`).
- **numba 프롬프트의 환경 사실**: "SciPy 가 없어 컴파일된 함수 안에서 `np.dot`·`np.vdot`·`@`·`np.linalg` 를 쓸 수 없다"를 적었다(실측: Numba 0.68.1 은 SciPy 없이 이 호출을 컴파일하지 못한다). `NumbaEnvironmentClaimTest` 가 확인하므로 requirements 에 SciPy 가 들어오면 테스트가 깨져 프롬프트를 고치게 된다.
- **Halide 프롬프트**: 명세가 출력 버퍼를 매개변수로 나열해도 아래 시그니처가 우선하고, 명세의 '호출'은 반환된 Func 를 출력 전체에 한 번 realize 하는 것이라는 문장을 넣었다. halide_auto 에는 스케줄 호출 금지와 하네스가 하는 일(estimate = 실제 크기, 오토스케줄러 이름, parallelism = 스레드 수)만, halide_manual 에는 오토스케줄러가 없다는 사실만 적었다. 스케줄·알고리즘·라이브러리를 권하는 문장은 어느 템플릿·힌트에도 없다.
- **힌트 4개**(numba 57줄, halide_auto 57줄, halide_manual 76줄, rust 58줄): 로컬 도구로 확인한 문법·API 사실만 담고, 코드 조각은 `HintFragmentsTest` 가 매번 실행·컴파일한다. halide_auto 힌트에는 스케줄 메서드 이름이 없다. 초안의 틀린 서술은 실측으로 고쳤다(예: `hl.Buffer(array)` 는 NumPy 배열의 수명을 잡아 두지 않아 빌드 함수가 끝난 뒤 깨진 값·크래시가 난다; Rust 에서 심벌을 내보내는 것은 `pub` 이 아니라 `#[no_mangle]` 이다).
- **수정 요청의 내용**: 컴파일러·cargo 출력은 `error` 또는 `undefined reference` 가 처음 나온 줄부터(없으면 꼬리 30줄), 절대 경로는 파일 이름만 남긴다. 임포트·로드 실패는 예외 한 줄 + 트레이스백 끝 10줄이고 하네스 패키지의 프레임은 먼저 뺀다(K8). check 는 형태별 한 줄을 지표(`max_tol_ratio`, `nan_inf`, `inputs_mutated`, `oob_write`, `nondeterministic`)로 쓴다. `harness/classify.py` 의 패턴 이름이 진단에 나오지 않음을 `NoHarnessVocabularyTest` 가 확인한다. sanitizer 가 재분류한 `memory_error` 는 재분류 전 판정으로 되돌려 말한다(`safety.enabled` 에 따라 수정 요청이 달라지면 안 된다). 오류 어휘는 §6 B11.
- **수정 루프의 실행 규칙**: `harness.repair` 에는 `--provider/--model` 이 없고, 설정·`LLM_PROVIDER`·`LLM_MODEL` 의 (provider, model) 이 턴 0 raw 와 다르면 호출 없이 exit 1 이다(dummy run 에 실제 호출이 붙지 않게). 앞 턴의 현재 행이 수정 대상이 아니면 사슬을 잇지 않는다(리뷰 generate_repair#1). 직전 응답이 비었거나 공백뿐인 샘플은 보내지 않는다(리뷰 generate_repair#3; 요약 줄 `skipped_empty_reply=N`). `harness.run` 이 exit 4·6·7 로 멈추면 그 코드를 그대로 돌려주고 행을 쓰지 않는다.
- **`--seed-from-feedback`**: 복사만 하고 끝낸다(`conditions.feedback`·`seeded_from`·`seeded_sha256` 만 원본과 다르다). 생성에 실패한 원본은 복사하지 않는다. 원본이 다시 생성됐으면 수정 턴이 없는 복사본만 `.prev<N>.json` 으로 보관하고 다시 복사하며, 수정 턴이 있으면 exit 1 이다. generate 는 시드 복사본을 다시 호출하지 않는다(리뷰 generate_repair#2). 두 피드백 조건이 같은 턴 0 응답에서 갈라져야 대응 비교가 성립하기 때문이다.
- **generate `--tasks`**: `run.tasks` 밖의 과제는 호출 전에 exit 1 이다(리뷰 interfaces#4). `run.tasks` 가 generate·run·repair 의 단일 기준이다.
- **dummy provider**: turn ≥ 1 은 `<k>.turn<t>.md`, 없으면 `<k>.md`(같은 실패 반복). `messages` 는 user/assistant 가 번갈아 나오고 user 로 끝나야 하며, 공백뿐인 내용은 하네스 버그(ValueError)다.

**run·schema·protocol(G4)**

- **재사용 API**: `setup_run` / `process_sample` / `finish_run`. 준비 단계의 실패는 `RunSetupError(exit_code)`(행 없음)이고 종료 코드 `EXIT_TOOLCHAIN = 7` 이 더해졌다.
- **행을 쓰기 전의 도구 점검(exit 7)**: raw 가 있는 언어 전부와, bench 를 하는 run 의 C 기준선이 대상이다. c 는 `build.cc --version`, rust 는 cargo·rustc, numba·halide 는 `PYTHON_EXE` 에서의 실제 임포트, 오토스케줄러는 halide_auto 만 본다. 그 뒤 실제로 한 번씩 돌려 본다: 실행 접두(taskset)마다 `[*접두, PYTHON_EXE, -c, pass]`(K13), `python -m harness.protocol --probe-runtime {halide|numba}` 로 오토스케줄러와 매개변수·`NUMBA_THREADING_LAYER`(K20), rust 가 있으면 `probe_toolchain`(K14). `unsupported` 조합의 raw 는 경고와 함께 빠진다.
- **[v2.1 §8-6 강화] 결과 문서가 없는 워커 종료는 하네스 오류**: rc·시그널·시간 초과와 무관하게 행 없이 exit 5 다. 워커는 커널 로드 전에 첫 문서를 쓰므로 문서가 없으면 LLM 코드는 돈 적이 없다(§6 B5).
- **[v2.1 §8-7 이탈] 대응 기준선의 결과 위치**: 샘플(턴) 디렉터리의 `paired_baseline.json` 이다. 예전 이름 `baseline/paired_sample_<lang>_<i>.json` 은 수정 턴과 두 피드백 조건에서 서로 덮어쓴다. 행의 값은 같다.
- **`memory_error` 의 적용 시점**: 그 턴의 최종 판정에 적용한다. check 는 통과했고 ASan 보고가 있는데 bench 가 `runtime_crash`/`wrong_output` 으로 끝난 경우도 `memory_error`(`asan:<첫 종류>; was <이전 type>/<detail>`)다. 통과 샘플과 bench timeout 은 바꾸지 않는다. check 시간 초과 샘플과 C 가 아닌 백엔드는 sanitizer 를 돌리지 않는다(열 NA).
- **행의 백엔드 열**: `compiler_version` 은 python 인터프리터·numba/halide 패키지·`rustc --version` 이다. `cflags` 는 rust `RUSTFLAGS=<설정>; profile.release: <Cargo.toml 의 프로필>`, halide_auto `autoscheduler=<이름> params=<정규 JSON, 하네스가 더한 parallelism 포함>`, halide_manual `autoscheduler=none`(K16), python 계열 NA 다. 결과 스키마의 열은 바뀌지 않았다.
- **run 끝의 보정(리뷰 verdicts#1)**: drift 플래그와 `model_served_mixed` 는 그 run 의 모든 result.json 행(수정 턴 포함)으로 정한다. `--languages`/`--samples` 로 거른 호출이 다시 재지 않은 행의 `protocol_ok` 를 바꾸던 것을 고쳤다.
- **재처리 규칙(리뷰 verdicts#4·interfaces#2, 오케스트레이터가 수정 단계의 결과를 고침)**: result.json 에 `requested = {bench, safety, lint}` 를 남긴다. 자동 재처리는 bench 가 빠진 경우뿐이다(`run.REPROCESS_STAGES = ("bench",)`). safety·lint 만 빠진 결과는 건너뛰면서 알리고 `--force` 에 맡긴다. 다시 처리하면 빌드·check·bench 를 전부 다시 해 이미 잰 시간이 바뀌기 때문이다(예: 린트가 없던 `20261105-real` 9행). 그래서 sanitizer 는 측정 run 을 시작하기 전에 켠다(§6 B10).
- **행의 `run_id`(리뷰 interfaces#1)**: 처리 중인 run 디렉터리의 이름이다. raw 안의 값이 다르면 run 당 한 번 경고하고 result.json 의 `raw_run_id` 에 남긴다(§6 B9).
- **`--rebuild-csv`(리뷰 verdicts#2)**: result.json 에서 다시 만든 행에, 같은 행 키의 result.json 이 없는 기존 행을 더한다. `results/built` 가 없는 새 clone 에서 `results/runs.csv` 의 실제 행이 헤더만 남고 지워지던 것을 고쳤다(§6 B8).
- **부하 안정 대기(리뷰 verdicts#3)**: `measurement.max_settle_s`(기본 0 = 기다리지 않음, 설정 파일에는 주석 줄만)를 넣었다. 양수면 bench 직전 검사 전에 1분 부하가 `max_load_1m` 이하가 될 때까지 5 s 간격으로 기다리고 기다린 시간을 `stages.dynamic.settle_wait_s` 에 남긴다. 근거(산술): 1분 load average 는 시상수 60 s 의 지수 평균이라 하네스 자신의 앞선 4스레드 bench 가 다음 검사까지 남는다. 배경 1.3 에서 4스레드 15 s 작업 뒤 5 s 면 약 2.1, 유휴 0 에서 60 s 작업 뒤 10 s 면 2.14 로 `max_load_1m`(2.0)을 넘는다. M1 은 bench 가 약 0.25 s 라 드러나지 않았다. 켜는 것은 측정 규약 변경이다(§6 A5). Linux 확인: `stress-ng --cpu 4 --timeout 15s; sleep 5; cat /proc/loadavg`.
- **환경 지문**: `toolchain_identity` 에 `rustc`·`numba_version`·`halide_version`·`lint_tools` 를 더했다(`env_hash` 가 바뀐다). machine.yaml 에 `cargo`·`rustc`·`numba_version`·`halide_version`·`halide_autoscheduler`·`halide_autoscheduler_loadable`·`lint_tools` 키가 생긴다. `source_hash` 에는 `plugins/lint/run_lint.py`·`plugins/lint/rules.yaml` 과 Rust 템플릿 두 파일을 더했다.
- **비-Linux**: `check_static`·`check_dynamic` 이 `platform=<sys.platform>` 플래그를 붙이고, `/proc/stat`·`os.getloadavg` 가 없으면 사용률·부하를 0.0 으로 둔다(잰 값이 아님은 플래그가 알린다). Linux 의 워커 환경과 플래그는 그대로다.

**perf_summary(G5)**

- **스키마 [CLAUDE.md §5 확장]**: 2026-11-20 의 `decisions_by_library` 에 더해 `parallel` 에 `loops`·`placement`(none|outer|inner|mixed|region|unknown)·`constructs`(병렬 구문 원문)를 더했다. `collapse`·`reduction` 은 CLAUDE.md 예시대로 불리언이고 세부는 `constructs` 에 원문으로 남는다.
- **모르는 값의 표기**: 문자열은 "unknown", 수·불리언은 null, 리스트는 빈 리스트다. false·0·none 은 '찾아봤는데 없다'는 뜻이다. 크기 상한은 loops 20개, 그 밖의 리스트 20개, notes 500자.
- **C 벡터화(리뷰 aux_columns#3 반영)**: gcc `-fopt-info-vec` 줄을 루프 조건식 연산자의 줄·열로 대응시킨다. 판정은 근거가 있는 가장 안쪽 루프만으로 한다(벡터화 리마크·인트린식, 루프 머리의 missed 리마크, `not vectorized:` 사유, basic block 벡터화). 리마크가 없는 루프(memset 으로 바뀐 초기화 루프 등)는 notes 에만 남기고, 근거 있는 루프가 없으면 `unknown` 이다. 인트린식·GCC 벡터 확장·효력 있는 `#pragma omp simd` 는 LLM 의 결정, 그 밖에 gcc 가 한 벡터화는 컴파일러의 결정이다. 빌드 로그에 `ignoring '#pragma omp …'` 가 있으면(c_seq) 그 지시문은 `constructs` 에만 남는다.
- **위임**: `library_delegation` 은 `decisions_by_library` 에만 싣는다. 위임으로 보는 것은 Python 의 NumPy·scipy 계산 호출과 배열 산술 식, C 의 `cblas_*`/`LAPACKE_*`/`qsort`, Rust 의 sort 계열이다. memcpy·memset·할당·뷰 함수는 `library_calls` 에만 실린다.
- **타일링·누적 정밀도**: 타일링은 증분이 1보다 큰 루프 가운데 그 변수가 안쪽 루프의 경계나 min 식에 쓰이는 것이다(수동 언롤·SIMD 폭 걸음·FFT 보폭 제외). `accumulation_precision` 은 출력이 float32 인 과제에서만, double·f64·`np.float64` 를 명시한 누적만 LLM 의 결정으로 센다(Python 기본 float 누적은 언어 기본값).
- **Numba·Halide·Rust·스레드**: prange 는 `parallel=True` 일 때만 병렬이고 중첩 prange 는 가장 바깥만 센다(리뷰 aux_columns#5). Numba 의 벡터화는 볼 수 없어 항상 `unknown` 이다. Halide 는 `loop_nest` 에서 읽고, 소스의 스케줄 호출은 LLM, 오토스케줄러가 만든 스케줄은 컴파일러의 결정이다. `split` 은 안쪽 변수가 vectorize·unroll 되지 않을 때만 타일링이다(Mullapudi 의 `split(j, j_vo, j_vi, 8).vectorize(j_vi)` 는 벡터 폭 자르기). 명시적 스레드(pthread, `std::thread`, `ThreadPoolExecutor` 등)는 `kind = threads` 다. while/do 안의 병렬 루프는 outer 가 아니다(리뷰 aux_columns#4). inspect 가 끝나지 않았으면 notes 에 남긴다(K10).
- **gcc 로그 픽스처는 손으로 만든 것이다**: `tests/fixtures/perf_summary/c/*.build.log` 는 gcc 가 없어 줄 모양을 소스의 줄·열에 맞춰 조립했다(파일 첫 줄에 표시). Linux 에서 실제 로그로 바꿔 확인한다. Halide·Numba 의 inspect 픽스처는 실제 라이브러리로 만든다.

**safety·린트(G6)**

- **규칙 표**: `plugins/lint/rules.yaml` 에는 CLAUDE.md §6 의 범주에 분명히 들어맞는 규칙만 넣고 나머지는 `unmapped` 로 센다. 실측: 저장소의 Python 더미 커널 8개 중 4개가 F401(안 쓰는 임포트)에 걸린다 — 넣으면 `lint_findings_total` 이 위생 문제로 부푼다. cppcheck 는 `--enable=warning,portability`, clang-tidy 는 `clang-analyzer-*`·`bugprone-*`·`openmp-*`·`performance-*`·`cert-flp30-c` 를 검사별로 대응한다. 하네스가 준 시그니처에 늘 걸리는 `bugprone-easily-swappable-parameters` 와 memcpy 마다 걸리는 `clang-analyzer-security.insecureAPI.*` 는 끈다. 심각도는 도구가 아니라 규칙 표가 정한다.
- **concurrency 범주는 베이스라인에서 사실상 비어 있다**: 기성 린터에는 OpenMP 경쟁·리덕션 누락 규칙이 없어 `openmp-use-default-none` 만 concurrency/info 로 넣었다. 베이스라인의 `lint_concurrency` 는 'default(none) 이 없는 병렬 지시문의 수'에 가깝고, ML lint 가 채울 자리다.
- **[v3 §12 이탈; CLAUDE.md §6 예시와 다름] 검사하지 않은 범주는 NA**: 플러그인 `summary` 의 값은 수 또는 null(그 범주를 보는 도구가 하나도 끝까지 돌지 않음)이다. 하네스는 null 범주의 `lint_<범주>` 열을 NA 로 쓰고, `lint_findings_total` 은 검사한 범주의 합(검사한 범주가 없으면 NA)이다(리뷰 aux_columns#1). 도구가 파일을 해석하지 못한 경우(시간 초과, 구문 오류, clang-tidy 의 `error:` 진단)는 '보고 0건'이 아니라 그 도구가 돌지 않은 것이다(§6 A9·B7).
- **clang-tidy 의 `omp.h`(리뷰 linux_paths#1)**: M1 실험 머신의 clang(machine.yaml: 21.1.8, `openmp_clang: false`)에는 libomp 의 `omp.h` 가 없어, `#include <omp.h>` 를 쓴 c_openmp 커널에서는 clang-tidy 가 컴파일 오류로 통째로 빠진다(리뷰가 코드와 machine.yaml 로 확인했고 실제 실행은 Linux 에서 본다). 하네스가 `build.cc` 를 환경변수 `CC` 로 넘기고, 플러그인은 `<cc> -print-file-name=include` 의 `omp.h` 한 파일만 임시 디렉터리에 복사해 `-isystem` 으로 준다(디렉터리를 통째로 넣으면 clang 이 gcc 의 인트린식 헤더를 해석하지 못한다).
- **플러그인 출력의 추가 키**: `unmapped`, `versions`(분석을 마친 도구의 버전 — 원칙 4). 하네스는 `tool`·`findings`·`summary` 만 엄격히 본다. 도구 하나의 상한은 20 s(`lint.timeout_s` 60 s 안), 플러그인 stderr 는 `lint.stderr.log` 에 남는다.
- **sanitizer**: `ASAN_OPTIONS` 에 `handle_sigfpe=0` 을 넣는다(정수 0 나눗셈이 ASan 보고가 되어 `memory_error` 규칙에 걸리지 않게). 끝까지 가지 못한 실행에서 보고가 없는 종류는 None 이다(예: 첫 ASan 오류의 SIGABRT → asan=1, ubsan=None, `incomplete:signal SIGABRT`). 커널이 로드되지 않았으면 두 열 NA 와 `load_failed:<사유>` 다(리뷰 verdicts#6·aux_columns#2). gcc 의 libubsan 이 실제 파일이면 `LD_PRELOAD` 에 libasan 다음으로 넣는다. 워커가 커널 로드 전에 `PR_SET_DUMPABLE=0` 을 걸면 libubsan 이 `/proc/self/environ` 을 읽지 못해 `UBSAN_OPTIONS` 가 적용되지 않았다(리뷰 linux_paths#2). 보고는 `san/` 의 로그와 `safety.stderr.log` 전체에서 센다.

**분석(G7)**

- **기본 표는 첫 시도만**: 수정 턴 행이 있으면 기본 그룹 표는 turn 0 만 세고 `주:` 로 알린다. 턴을 합친 결과는 `--by-turn` 에서 본다. 분모는 턴 0 이 유효한 샘플 수이고, 결과를 모르는 실패 샘플(고쳐 보지 않음, API 오류 턴)이 있는 t 부터 `pass_within_t`·`tokens_cum_t` 가 NA 다. 첫 통과 뒤 턴의 낡은 행은 빼고 그 수를 알린다(K11).
- **[v3 §13 이탈] `--conditions` 의 비교 단위**: 같은 provider·model·task·language 와 나머지 두 조건, 선택한 run 전체다. args·grammar_hint 는 턴 0 의 독립 표본 부트스트랩이고 feedback 으로 나누지 않는다(시드 복사본은 한 번만 센다 — 리뷰 analysis#2). feedback 은 같은 run 의 같은 `sample_idx` 짝의 대응 표본 부트스트랩이고, 모든 짝의 두 수준 결과를 아는 가장 큰 수정 턴 H 에서 비교한다(`pass_within_<H>`, H < 1 이면 NA — 리뷰 analysis#1).
- **부트스트랩 고정**: 백분위법, 재추출 2000회, seed 0, 95 %(`analysis/stats.py` 의 기본값, CLI 로 열지 않음). n = 5 에서 통과 0..5개의 구간은 [0,0], [0,0.6], [0,0.8], [0.2,1], [0.4,1], [1,1] 이다. 폭 0 은 '변동을 관측하지 못했다'는 뜻일 뿐이고(5/5 의 Clopper–Pearson 95 % 하한은 0.48), 이 구간은 차이의 방향을 가늠하는 데까지만 쓴다.
- **[v3 §13 이탈] `--coverage`**: `n/a`(signatures.yaml 의 `unsupported`, 이유를 함께 출력)와 `-`(지원하는데 행이 없음)는 기본 표가 아니라 `--coverage` 행렬에 나온다. 기본 표는 바이트 단위로 유지한다.
- **언어 축 순서**: `analysis.aggregate.LANGUAGE_ORDER`(numpy, numba, halide_auto, halide_manual, c_openmp, c_seq, rust, …) 한 곳에 둔다.
- **그림**: 그림 1·2 는 첫 시도와 기준 조건 조합 하나만 그린다. 그림 2 는 15개 범주(pass, pass(bench timeout), 오류 분류 9종, refusal·truncated·API 오류, other)로 모든 행이 정확히 한 범주에 든다. 그림 4 는 수정 행이 있는 run 만 합쳐 그린다(리뷰 analysis#3). 그림의 글자는 영어(matplotlib 기본 글꼴에 한글이 없다), 색은 Okabe–Ito 다.

### 3. 새 커널 8개의 계약과 허용오차

공통: seed 20261105, 형태 순서 tiny / rect / small / large(large 는 잠정값 — 데스크탑에서 다시 정한다). 생성기는 두 번 돌려 npz 의 sha256 이 같았고(결정적), 명세가 말하는 입력 수치를 assert 한다. 정수 출력 과제는 `compare: exact`, 센티널 −2048 이다.

- `histogram(keys, hist, n, n_bins)`: int32, `hist` 는 전부 덮어써야 한다. (10,4) / (100003,1000) / (1048576,256) / (67108864,256). tiny 의 키는 [1, 3) 에서 뽑아 첫 구간과 마지막 구간이 빈다(블라인드 감사 뒤 변경, §6 B3). 기준선 C+OpenMP. npz 273.0 MB.
- `prefix_sum(x, out, n)`: int32 포함형, 0 ≤ x < 16, 최대 부분합 15·2²⁶ = 1,006,632,960 < 2³¹. n = 7 / 100003 / 1048576 / 67108864. 기준선 `np.cumsum(x, dtype=np.int32, out=out)`. npz 546.1 MB(과제당 약 550 MB 한도 바로 아래).
- `nbody(pos, mass, acc, n)`: AoS (n,3), ε = 0.01, 가속도. n = 5 / 317 / 1024 / 8192. pos uniform[0,1)³, mass uniform[0.5,1.5)/n, 참조 float64. 기준선 C+OpenMP(SoA 사본, i 병렬, j simd 리덕션). rtol = atol = 1e-4.
- `sort(keys, out, n)`: int32 전체 범위(키에서 −2048 은 −2047 로 바꿔 뺀다), 제자리 정렬은 입력 변형이다. n = 9(INT32_MIN·INT32_MAX 를 포함해 손으로 고름) / 100003([−1000, 1000), 값마다 약 50번 중복) / 2¹⁶ / 2²⁴. 기준선 `np.sort`. Halide 는 unsupported.
- `cholesky(A, L, n)`: A 는 대칭 양의 정부호(양쪽 저장), L 은 하삼각이고 대각 위 0 도 써야 한다. n = 4 / 97 / 256 / 2048. A = G·Gᵀ/n + I(G 는 2⁻²⁰ 격자라 float64 곱이 정확해 머신이 달라도 입력이 비트 단위로 같다), 고유값 비 < 3. 기준선 `np.linalg.cholesky`(`gate_blas: true`). rtol 1e-4, atol 1e-5.
- `floyd_warshall(W, D, n)`: 완전 방향 그래프, 대각 0, 가중치 [1, 100). n = 5 / 97 / 256 / 1024. 명세는 결과만 정의하고 루프 구조는 정하지 않는다. 기준선 C+OpenMP. rtol = atol = 1e-4.
- `bfs(row_ptr, col_idx, dist, n_vertices, n_edges, source)`: int32, 도달 불가 −1, `dist[source] = 0`, 자기 루프·중복 간선 허용(명세에 사실로 적었다). (n_vertices, n_edges, source) = (6, 10, 3) / (1000, 3927, 287) / (65536, 522726, 38219) / (4194304, 33543940, 3448857). 나가는 차수는 0…2·avg 균등(rect 0…8, small·large 0…16), 도달 불가 정점 1 / 24 / 28 / 1435개, 최대 거리 3 / 9 / 9 / 10. 명세의 수치는 생성기의 `spec_max_dist: 10`, `spec_unreachable_min: 1300`, `spec_unreachable_max: 1500` 이 강제한다(§6 A7). 기준선 순차 C 배열 큐(`openmp: false`). Halide 는 unsupported.
- `fft(x, out, n)`: 실수·허수 교차 float32 배열(형태 식 `2*n`), 순방향·정규화 없음·자연 순서. n = 8 / 1024 / 65536 / 4194304. 입력 성분 uniform[−1, 1)/√n 이라 |X[k]| 의 RMS 가 형태와 무관하게 √(2/3) ≈ 0.82 다(생성기가 n ≥ 1024 에서 ±0.03 확인). `work` = 5·n·log₂n. 기준선 `np.fft.fft(x.view(np.complex64), out=…)`(§6 A2). rtol = atol = 1e-3(§6 A1).

| 과제 | `test_hash` | 과제 | `test_hash` |
|---|---|---|---|
| `gemm` (seed 20261103) | `2077183b4edc9ec8` | `nbody` | `c15bade031489ac4` |
| `jacobi2d` | `2817c6d5544625d8` | `sort` | `1e13ecc4e2a44a13` |
| `spmv_csr` | `fcab7d32e352a1eb` | `cholesky` | `5080269a9d095f90` |
| `dot` | `0ed48e0b583e1ac6` | `floyd_warshall` | `845ae92a4a402949` |
| `histogram` | `467726b382000cdd` (이전 `a380ff26d6f40c46`) | `bfs` | `846392026088677c` |
| `prefix_sum` | `2bb751afd41770a3` | `fft` | `7f78472cc4a65f08` |

허용오차의 여유는 비율 = max|err| / (atol + rtol·|ref|) 로 잰다(형태 순서 tiny / rect / small / large). 새 float32 과제 네 개(nbody·cholesky·floyd_warshall·fft)의 `numeric_pattern_tol_ratio` 는 모두 20 이다(gemm·jacobi2d·spmv_csr 도 20, dot 만 500).

- **nbody rtol = atol = 1e-4** [v3 §4 제안값 rtol 1e-3 이탈]: 정확한 float32 변형(순차 누적, pairwise, 8-레인 블록, 근사 rsqrt + 뉴턴 1회)의 최대 비율 0.038 → 여유 26배 이상. 보정 없는 근사 rsqrt 는 모든 형태에서 3.96 이상으로 실패한다. rtol 1e-3 이면 결정적 근사 rsqrt 가 tiny 에서 0.90 으로 경계에 걸리고, rtol 1e-4·atol 1e-5 이면 근사 + 뉴턴의 small 여유가 9.7배로 10배 기준에 못 미친다.
- **cholesky rtol 1e-4, atol 1e-5** [v3 §4 제안값 1e-3/1e-4 이탈]: float32 흉내 8종 중 최악은 right-looking n = 2048 의 대각, 비율 0.0194 → 여유 52배. 제안값에서는 반정밀도로 반올림한 출력(0.40)과 보정 없는 12비트 근사 역제곱근(0.13)이 통과하고, 확정값에서는 둘 다 실패한다(4.0, 1.28).
- **floyd_warshall rtol = atol = 1e-4**: float32 k 정방향·역방향·무작위 순서·min-plus 거듭제곱의 최대 비율 1.42e-3 → 여유 약 700배. 틀린 구현(k 를 가장 안쪽에 둔 1회 완화)의 오차는 11–21 이다.
- **fft rtol = atol = 1e-3(설계값 유지, 확정은 §6 A1)**: (a) 배정밀도 회전 인자의 float32 radix-2: 4.7e-5 / 2.1e-4 / 3.0e-4 / 4.7e-4 → 여유 2100배. (b) 단계마다 float32 점화식 `w = w·w_step`: 7.8e-5 / 5.3e-3 / 0.31 / 0.88 — 통과하지만 n = 2²² 의 여유가 1.14배다(컴파일 옵션 3종 0.878–0.907, 다른 시드 6개 0.78–0.88). (b′) 회전 인자 표 n/2개 전체를 점화식으로 채우면 large 8.65 로, 정확성 형태는 통과하고 bench 의 최종 검사에서 `numeric_instability` 가 된다. (c) 버터플라이마다 `cosf/sinf`: large 1.9e-3. NumPy 기준선은 4.0e-5. 구조가 틀린 커널(부호·정규화·순열·배치)은 370–2500배로 실패한다.
- **큰 형태의 참고 시간(1회, 공유 머신; C 기준선은 zig 빌드, 나머지는 NumPy)**: 기준선 histogram 78 ms(직렬)·55 ms(zig-omp), prefix_sum 152 ms, nbody 157 ms(직렬)·33 ms(zig-omp), sort 303 ms, cholesky 443 ms, floyd_warshall 1224 ms(직렬)·276 ms(zig-omp), bfs 1.27–1.31 s, fft 190 ms(반복 호출)–470 ms(첫 호출). bfs 는 메모리 지연 시간에 묶인 커널이라 부하에 따라 3배까지 흔들렸다(무작위 gather 원소당 한가할 때 12–28 ns, 부하 중 87–138 ns). 큰 형태를 다시 정할 때 Linux 에서 다시 잰다.
- **명세 수정(블라인드 감사·오케스트레이터)**: histogram 의 형태를 자릿수 구분 쉼표 없이 썼다. sort 의 '가장 큰 정확성 형태' 문장을 n = 65,536 형태로 고쳤다. floyd_warshall `D` 의 설명에 "; the diagonal is 0" 을, cholesky·floyd_warshall 명세에 "The output for the performance shape must also pass the accuracy check." 를 붙였다(두 과제의 `prompt_hash` 가 바뀐다; 실제 실행 기록 없음). gemm·jacobi2d·spmv_csr 명세는 `prompt_hash` 를 지키려고 건드리지 않았다. 기각: histogram 의 `n_bins` 상한을 명세에 적자는 제안(상한을 적으면 사본 크기라는 성능 결정의 힌트가 된다).
- **음성 검사**: 과제마다 대표적 오답이 기대한 분류로 실패했다. 예: histogram 초기화 누락 → `accumulated_into_c`, 배타형 스캔 → `mismatch`, 제자리 변형 → `inputs_mutated`, nbody ε 누락(j = i 포함) → `numeric_instability`/`nan_inf`, cholesky Lᵀ 반환 → `arg_order`/`transposed_output`, floyd_warshall k 가장 안쪽 → tiny 통과·rect·small `mismatch`, bfs 깊이 우선 → `mismatch`, fft 지수 부호·정규화·비트 반전 오류 → `mismatch`(배율 370–2500). `accumulated_into_c` 는 GEMM 에서 온 이름이라 histogram 에서도 출력 이름과 무관하게 이 이름이 나온다(판정은 맞다).

### 4. 새 언어 더미에서 배운 것

더미는 파이프라인 검증용 정답이고 튜닝한 기준선이 아니다. 지금 (과제, 언어) 80조합(12 × 7 − 4)에 모두 `0.md` 가 있다. Halide 는 과제마다 알고리즘 하나를 두 언어에 같이 쓰고 halide_auto 쪽은 스케줄 호출이 0개다. 시간은 공유 Windows 머신의 1회 호출 참고값이다. 기존 C 더미 가운데 기준선 파일을 복사한 흔적이 있던 것(jacobi2d·spmv_csr 의 c_seq·c_openmp 머리 주석, floyd_warshall c_openmp 의 머리 주석과 `omp simd`)은 코드의 정답성은 그대로 두고 중립 주석으로 바꿨다.

- **numba(12개)**: 모두 렌더된 서명 문자열을 그대로 쓴 `@njit(sig, parallel=True)` 이고 fastmath·`np.dot`·`@`·`np.linalg` 는 쓰지 않는다. Numba 에는 원자 연산이 없어 bfs 는 순차 큐 탐색이다(초기화만 prange, large 약 1.5 s). 병렬 numba BFS 를 기준으로 삼으려면 설계 결정(전치 그래프의 bottom-up 등)이 필요하다. dot 은 리덕션 변수 대신 스레드 수와 무관한 고정 1024청크 float64 부분합으로 결과를 비트 단위로 고정했다.
- **rust(12개)**: 포인터를 맨 위에서 한 번 슬라이스로 바꾸고 rayon 은 `par_chunks_mut`·`par_chunks(...).map(...).collect()` 만 쓴다. 블록 크기를 고정해 rayon 의 작업 분할과 무관하게 결정적이다. bfs 는 `AtomicI32` 의 `compare_exchange(-1, level)` 로 레벨 동기 top-down 이다. 이 더미를 병렬 devcheck 로 돌리다 cargo target 공유 레이스를 찾았다(K1).
- **halide_auto·halide_manual(각 10개, sort·bfs 제외)** — halide_auto 결과를 해석할 때 필요한 관찰:
  - **spmv_csr**: 자연스러운 행별 형태(RDom [0, nnz) 에 `where(row_ptr[i] <= p < row_ptr[i+1])`)는 스케줄 없이 10⁶행 25 ms 이지만, Mullapudi2016 이 행 루프를 4폭으로 벡터화하면 p 루프가 nnz 전체를 돌아 O(n_rows·nnz) 가 된다(10⁴행 421 ms, 2-D RDom 판 1510 ms, 10⁶행은 둘 다 250 s 안에 끝나지 않음). 더미는 float64 누적합 S 의 차로 썼다(auto 498 ms, manual 62 ms). LLM 의 halide_auto SpMV 가 자연스러운 형태로 쓰면 틀린 출력이 아니라 bench `timeout`(pass=1, 시간 NA)으로 나올 가능성이 크다.
  - **jacobi2d**: 스텝마다 경계를 `repeat_edge(u, bounds)` 로 클램프하는 자연스러운 형태는 T = 20 에서 Mullapudi 23.7 s + JIT 123 s, `hl.clamp` 판은 590 s 를 넘겼다. 더미는 입력만 `repeat_edge(src)` 하고 `select(interior, …)` 로 경계를 버린다(오토스케줄 2.4 s + JIT 2.0 s). LLM 응답이 그렇게 쓰면 컴파일 비용만으로 check·import 시간 제한에 걸릴 수 있다(§6 A4). 시간은 auto 49 ms(스텝을 타일로 융합) 대 manual 354 ms(스텝마다 compute_root).
  - **histogram**: `hl.clamp(keys, 0, n_bins-1)` 은 n_bins = 1 에서 상수로 접혀 Mullapudi2016 이 내부 오류(`Pipeline_bounds is missing keys`)로 중단한다(Halide 22.1 결함; n_bins = 1 은 테스트 형태가 아니다). 더미는 `hl.unsafe_promise_clamped` 를 쓴다.
  - **prefix_sum**: auto 1.07 s 대 manual 0.16 s(약 7배). 가설의 관찰 대상이라 그대로 둔다.
  - **floyd_warshall·cholesky**: Mullapudi2016 은 순수하지 않은 RVar 갱신을 병렬화하지 않아 halide_auto 는 직렬 갱신을 잰다(auto 861 / 974 ms, manual 211 / 411 ms). manual 의 제자리 갱신 병렬화에는 `allow_race_conditions()` 가 필요하다(Halide 22.1 의 경합 검사가 거부한다; §6 A6).
  - **fft**: Stockham DIF, 단계마다 Tuple Func 하나(n = 2²² 에서 Func 23개). n = 2²² 에서 auto 는 오토스케줄 약 4.1 s + JIT 2.2 s, manual 은 JIT 약 12 s 이고, manual 의 정확성 형태 셋은 합쳐 약 14 s 로 check 제한 180 s 안이다. 나머지 과제의 성능 형태 빌드 + JIT 은 0.25 s 미만(floyd_warshall)에서 4.4 s(jacobi2d) 사이였다.
  - **정렬 가정**: Halide 의 벡터 코드는 출력 버퍼의 32바이트 이상 정렬을 가정한다(16바이트 어긋난 뷰로는 3/3 접근 위반). 하네스는 64바이트 정렬 버퍼를 주므로 문제없지만, 정렬되지 않은 뷰를 넘기는 하네스 경로를 만들면 거짓 `runtime_crash` 가 된다.

### 5. 리뷰 27건과 반영

| id | 등급 | 지적 | 반영 |
|---|---|---|---|
| `verdicts#1` | major | 거른 `harness.run` 이 그 호출의 행만으로 drift·`model_served_mixed` 를 다시 정해 다시 재지 않은 행의 `protocol_ok` 를 바꾼다 | `finish_run` 에 run 의 모든 result.json 행을 넘긴다 |
| `verdicts#2` | major | built 에 실제 provider 의 result.json 이 없으면 `--rebuild-csv` 가 `runs.csv` 를 헤더만 남긴다 | 다시 만들 수 없는 기존 행을 남긴다 |
| `verdicts#3` | major | 하네스 자신의 앞선 다중 스레드 bench 가 다음 샘플의 `load_1m` 플래그를 올린다 | `measurement.max_settle_s`(기본 꺼짐); 켜기는 §6 A5 |
| `verdicts#4` | minor | `--skip-bench` 로 쓴 행이 최신으로 보여 뒤의 측정 run 이 bench 없이 exit 0 | `requested` 기록, bench 누락만 자동 재처리 |
| `verdicts#5` | minor | 워커가 검사 전에 끝나도 `inputs_mutated`·`oob_write` 가 0 | 검사한 호출이 없으면 None(NA) |
| `verdicts#6` | minor | 계측 라이브러리 로드 실패에도 sanitizer 수가 0 | `load_ok` 가 참일 때만 완료, 아니면 NA·`load_failed:` |
| `extract_build#1` | major | `int32_t`·`signed int` 를 `param_mismatch` 로 거부 | ABI 같은 철자를 `int` 로(`signature_exact=False`) |
| `extract_build#2` | minor | nm 이 T/W 만 받아 target_clones(ifunc) 커널이 거짓 `signature_mismatch` | `i` 도 받는다 |
| `worker#1` | major | ASan 실행이 가드 있는 버퍼를 써 작은 넘침이 안 보인다 | 워커 `--sanitize`(정확한 크기 할당) |
| `generate_repair#1` | major | 앞 턴이 이제 통과인데 수정 사슬을 계속 이어 유료 호출 | 수정 대상이 아닌 턴에서 사슬 중단 |
| `generate_repair#2` | major | 원본 턴 0 이 다시 생성돼도 낡은 시드 복사본이 남아 짝이 깨진다 | `seeded_sha256`, 재시드 때 교체(수정 턴 있으면 exit 1) |
| `generate_repair#3` | minor | 빈 assistant 응답을 대화에 그대로 보내 API 가 치명 오류(재실행마다 반복) | 그 샘플은 건너뛰고 `conversation` 이 공백 내용 거부 |
| `aux_columns#1` | critical | 유일한 도구가 돌지 않은 린트 범주가 0 으로 기록된다 | `summary` 의 null = 검사 안 함 → 열 NA |
| `aux_columns#2` | major | verdicts#6 과 같은 결함(다른 경로에서 발견) | verdicts#6 과 함께 |
| `aux_columns#3` | major | 리마크 없는 가장 안쪽 루프를 '벡터화 안 됨'으로 세어 partial/no | 근거 있는 루프만으로 판정, 없으면 `unknown` |
| `aux_columns#4` | minor | while 안의 병렬 for 를 `outer` 로 보고 | while/do 깊이를 센다 |
| `aux_columns#5` | minor | 중첩 prange 를 모든 수준에서 병렬로 센다 | 가장 바깥만 |
| `analysis#1` | major | feedback 비교가 고치지 않은 샘플·API 오류 턴을 실패로 센다 | 공통 지평 H 에서 비교, H < 1 이면 NA |
| `analysis#2` | minor | 그림 3 이 다른 층의 비교를 구분 없이 그리고 시드 복사본이 같은 비교를 두 번 낸다 | args·grammar_hint 는 feedback 무관, 점마다 at 표시 |
| `analysis#3` | minor | `--all-runs` 의 그림 4 가 run 마다 같은 모양의 선, 수정 없는 run 섞임 | 수정 행이 있는 run 만 합쳐 그린다 |
| `linux_paths#1` | major | c_openmp 에서 clang-tidy 가 `<omp.h>` 를 못 찾아 조용히 빠진다 | 설정 컴파일러의 `omp.h` 를 `-isystem` 으로 제공 + aux_columns#1 |
| `linux_paths#2` | minor | `PR_SET_DUMPABLE=0` 뒤 로드된 libubsan 이 `UBSAN_OPTIONS` 를 못 읽는다 | libubsan 을 `LD_PRELOAD` 에 함께 넣는다 |
| `linux_paths#3` | minor | 도구 점검이 빌드와 다른 환경으로 cargo·rustc 를 띄운다 | `protocol.toolchain_env` 를 같이 쓴다 |
| `interfaces#1` | major | 행 `run_id` 를 raw 에서 읽어 복사한 raw 의 재처리가 원래 run 의 행을 덮는다 | 처리 중인 run id, raw 값은 `raw_run_id` |
| `interfaces#2` | major | verdicts#4 와 같은 결함 | verdicts#4 와 함께 |
| `interfaces#3` | minor | devcheck 에 import 단계가 없어 Halide 정의 오류를 파이프라인과 다르게 분류 | Halide 는 import 워커를 먼저, 로드 실패는 `stage=build` |
| `interfaces#4` | minor | generate 가 `run.tasks` 밖의 과제를 호출한다(행이 되지 못하는 유료 호출) | 호출 전 exit 1 |

- **그룹 사이에 넘겨진 문제(K)**: K1 Rust target 잠금, K2 Rust FFI 별칭, K3·K17 전처리 컴파일러 `build.cc`, K4·K14 `probe_toolchain`, K6 `n_fail`·`n_total`, K8 수정 요청에서 하네스 프레임 제거, K10 inspect 미완료 notes, K11 첫 통과 뒤 행 제외, K12·K19 진입점의 `setup_stdio`(Windows cp949 파이프), K13 실행 접두 시험, K15 Rust 템플릿의 소스 해시, K16 Halide `cflags`, K18 테스트 픽스처, K20 `--probe-runtime`. 모두 고쳤다.
- **수정이 바꾸지 않은 것**: 결과 CSV 의 열, 오류 분류표, 측정 절차(대기는 기본 꺼짐), numpy·c_seq·c_openmp 프롬프트 템플릿.

### 6. 사용자 확인이 필요한 결정

- **사용자 확인이 필요한 결정(2026-11-22 기준).** 아래는 구현하면서 값을 정했거나 기본값을 그대로 둔 채 남겨 둔 항목이다. 실제 API 호출 전에 확정한다. 확정하면 이 문서에 날짜와 함께 적는다(2026-11-05 항목과 같은 방식).
  - **A. 값을 바꾸지 않고 남겨 둔 것(권고 있음)**
    1. **fft 허용오차** — 지금은 설계값 `rtol = atol = 1e-3` 그대로다. 회전 인자를 float32 점화식(`w = w·w_step`)으로 구하는 radix-2 구현이 성능 형태(n = 2²²)에서 허용오차의 0.78–0.91(시드 7개·컴파일 옵션 3종)로 **통과와 실패의 경계**에 있다. 정확한 회전 인자를 쓰는 float32 구현은 4.7e-4(여유 2100배)다. 권고: **`1e-4` 로 조인다**(점화식 구현은 n = 65,536 에서 3.2–4.4배 초과로 정확성 단계에서 일관되게 `numeric_instability`, 정확한 구현은 여유 210배, 버터플라이마다 `cosf/sinf` 는 약 50배). `2e-4` 도 같은 판정을 준다(초과 1.6–2.2배, 여유 400배). 바꾸는 곳은 `tasks/fft/tolerance.yaml` 의 두 값과 `tasks/fft/spec.md` 의 Accuracy 한 줄이고 fft 의 `prompt_hash` 가 바뀐다(실제 실행 기록은 아직 없다).
    2. **fft 기준선** — `np.fft.fft(x.view(complex64), out=…)` 그대로다. NumPy 2.3.5 는 이 호출을 내부에서 배정밀도로 계산해(결과가 complex128 계산 후 반올림과 비트 단위로 같다) 같은 라이브러리의 단정밀도 경로보다 약 1.9배 느리다. 권고: **그대로 둔다**(기준선은 "라이브러리를 평범하게 부른 값"이고, numpy 언어의 응답도 같은 호출일 가능성이 높아 속도비가 1 근처가 된다). 다른 언어의 fft 속도비는 단정밀도 기준선일 때보다 약 1.9배 크게 나온다는 점을 결과 해석에 적는다.
    3. **numba 의 스레드 바인딩** — `measurement.omp_env_extra`(`OMP_PROC_BIND=true`, `OMP_PLACES=cores`, `OMP_DYNAMIC=false`)는 지금 C 의 OpenMP 백엔드에만 들어간다. numba 는 스레딩 계층이 omp 인데 바인딩 변수 없이 돈다(코어 집합 고정은 taskset 으로 같다). 권고: **numba 에도 같은 변수를 준다**(같은 OpenMP 런타임 조건). 측정 규약 변경이라 확인 뒤에 바꾼다.
    4. **Halide 의 빌드(JIT) 시간과 bench 제한 시간** — 형태별 빌드(오토스케줄 + JIT)는 측정값에서 빠지지만 run.py 의 bench 제한(300 s)에는 들어간다. 빌드가 길면(약 60 s 이상) 측정 가능한 커널이 bench 의 `timeout`(pass=1, 시간 NA)이 될 수 있다. 권고: **관찰된 빌드 시간(`build_ms`)을 bench 제한에 더해 준다.** 측정 규약 변경이라 확인 뒤에 바꾼다.
    5. **부하 안정 대기** — 리뷰에서 확인: 스레드 4개짜리 bench 가 길면(대략 60 s 이상) 그 직후 샘플의 1분 부하 평균이 `max_load_1m`(2.0)을 넘어 **하네스 자신의 부하 때문에** `protocol_ok=0` 이 된다. 대기 기능은 넣었지만 꺼 두었다(`measurement.max_settle_s: 0`). 권고: **60–90 s 로 켠다**(부하가 내려올 때까지만 기다리고 기다린 시간을 `result.json` 에 남긴다). 측정 규약 변경이라 확인 뒤에 켠다.
    6. **Halide manual 의 `allow_race_conditions()`** — 제자리 갱신을 병렬화하려면 이 호출이 필요하다(더미의 floyd_warshall·cholesky manual 이 쓴다; 경쟁이 없음은 주석으로 논증). 모델이 같은 호출로 실제 경쟁을 가릴 수 있고, 하네스는 그것을 틀린 출력·실행 간 불일치로만 잡는다(OpenMP 와 같은 수준). 권고: **허용한 채 둔다**(금지하면 manual 조건에서 제자리 갱신의 병렬화 자체가 막힌다; 스케줄 문자열은 perf_summary 에 남는다).
    7. **bfs 명세가 공개하는 작업 부하 사실** — 차수 분포(0…16 균등), 성능 형태의 도달 불가 정점 수(약 1,400)와 최대 거리(10)를 명세에 적었고 생성기가 그 수치를 강제한다. 권고: **그대로 둔다**(spmv_csr 의 "행 길이 0–32"와 같은 종류의 사실이고 알고리즘 지시가 아니다). 줄이려면 수치 문장과 생성기의 `spec_*` 키를 같이 지운다.
    8. **수정 루프에서 컴파일 언어의 크래시 정보** — Python·Numba·Halide 의 예외는 메시지와 트레이스백 끝이 수정 요청에 들어가지만, C·Rust 의 크래시는 `crashed: signal …` 한 줄뿐이다(Rust 패닉의 메시지는 stderr 에만 있다). 언어 사이의 `turns_to_pass` 를 비교하기 전에 정한다. 권고: 커널 프로세스의 stderr 끝부분을 걸러서(워커 자신의 줄 제외) 같이 준다 — M4 실행 전에 구현한다.
    9. **CLAUDE.md §6 의 린트 출력 예시** — 예시의 `summary` 는 범주마다 정수인데, 구현한 플러그인 계약은 `null`(그 범주를 보는 도구가 돌지 않았음 → 행에서는 NA)도 허용한다. CLAUDE.md 는 사용자만 고치므로 손대지 않았다. 권고: §6 에 "`summary` 의 값이 `null` 이면 그 범주를 검사하지 못한 것"이라는 한 문장을 더한다.
  - **B. 구현하면서 정한 것(이견이 있으면 되돌린다)**
    1. **nbody 허용오차 `rtol = atol = 1e-4`**(설계 제안값은 rtol 1e-3) — 정확한 float32 구현의 여유는 26배 이상이고, 보정 없는 근사 역제곱근은 모든 형태에서 4배 이상 초과로 분명히 실패한다. rtol 1e-3 이면 가장 작은 형태가 0.90 으로 경계에 걸린다.
    2. **cholesky 허용오차 `rtol 1e-4, atol 1e-5`**(제안값의 10분의 1) — 여유 52배. 제안값에서는 반정밀도로 반올림한 출력과 보정 없는 근사 역제곱근이 통과한다.
    3. **histogram 의 가장 작은 테스트** — 키를 [1, 3) 에서 뽑아 첫 구간과 마지막 구간이 모두 비게 했다(키 범위 밖 구간을 쓰지 않는 구현을 잡는다). `test_hash` 가 `467726b382000cdd` 로 바뀌었다.
    4. **시그니처 표기의 허용 범위** — ABI 가 같은 다른 표기는 시그니처 위반으로 보지 않는다: C 의 `int32_t`·`signed int`(= `int`), Rust 의 `c_int`·`c_float`(= `i32`·`f32`). `signature_exact=False` 로 남는다. 크기·부호가 다른 타입은 그대로 `signature_mismatch` 다.
    5. **판정 규칙의 적용** — (a) 워커가 결과 문서를 한 번도 쓰지 못하고 끝나면 모델 오류가 아니라 하네스 오류다(행 없음, exit 5). (b) 커널 로드가 끝난 뒤 관찰값(`--inspect`)을 모으다 워커가 끝난 것은 빌드 실패가 아니다(경고 후 check·bench 로 계속). (c) 없는 도구·실패한 도구 점검(컴파일러, cargo, 링커, taskset 접두, 오토스케줄러, 스레딩 계층)은 행을 쓰기 전에 exit 7 로 멈춘다.
    6. **sanitizer 실행의 버퍼 배치** — ASan 실행에서는 워커가 가드 영역 없는 정확한 크기의 버퍼를 준다(`--sanitize`). 평소의 가드 영역이 있으면 배열을 한두 칸 넘는 접근이 ASan 에 보이지 않는다.
    7. **린트·sanitizer 열의 NA** — 도구가 돌지 않았거나 그 범주를 보는 도구가 없었으면 0 이 아니라 NA 다(0 은 "검사했고 없음").
    8. **`--rebuild-csv`** — `result.json` 으로 다시 만들 수 없는 기존 행은 지우지 않고 남긴다(`results/built` 가 없는 머신에서 `results/runs.csv` 의 실제 실행 행이 사라지는 것을 막는다).
    9. **행의 `run_id`** 는 처리 중인 run 디렉터리의 이름이다(raw 파일 안의 값이 아니라). 기존 응답을 새 run_id 로 복사해 다시 측정해도 원래 run 의 행을 덮어쓰지 않는다.
    10. **`--skip-bench` 로 만든 행** 은 뒤의 측정 run 에서 최신으로 치지 않는다(bench 를 요청한 run 이 다시 처리한다). 반대로 `safety`·`lint` 를 **나중에 켠** 경우는 자동으로 다시 처리하지 않는다 — 다시 처리하면 빌드·check·bench 를 전부 다시 해 이미 잰 시간이 바뀐다. 건너뛰면서 그 사실을 알리고(그 열은 NA) 채우는 것은 `--force` 에 맡긴다. 그래서 sanitizer 는 측정 run 을 시작하기 전에 켠다.
    11. **수정 요청의 어휘** — `wrong_output`·`arg_order`·`numeric_instability` 는 모두 "incorrect output", `race_condition` 은 "non-deterministic output" 으로 낮춰 준다. 하네스의 패턴 이름·참조값·하네스 자신의 트레이스백 프레임은 넣지 않는다.
    12. **Halide 오토스케줄러 Mullapudi2016, Numba 스레딩 계층 omp**(2026-11-20 항목 그대로) — 이 PC 에서 둘 다 동작을 확인했다. Linux 휠에서 다시 확인한다.
  - **C. 하지 않은 것**
    1. **실제 API 호출은 한 번도 하지 않았다.** 보류 중인 M2a 60회(4과제 × 3언어 × 5샘플)와 그 뒤의 모든 호출은 Linux 검증이 끝난 뒤 확인을 받고 실행한다.
    2. **큰 형태(성능 형태)는 잠정값이다.** 데스크탑에서 `protocol --probe` 와 기준선 시간을 본 뒤 다시 정한다(bfs 는 명세의 수치와 생성기의 `spec_*` 키를 같이 고쳐야 한다).
    3. **커밋하지 않았다.** 작업 트리의 변경 전체가 미커밋 상태다.

## 2026-11-23 (이 브랜치의 기준 — `origin/main` 보다 16커밋 뒤에서 작업했다; 통합 전)

- **발견(커밋 직전)**: 이 PC 의 저장소는 `6109866`(M2a 하네스, 2026-11-05)에 머물러 있었고, `origin/main` 은 `6f02d16`(2026-11-12)으로 그보다 16커밋 앞서 있다(노트북에서 한 작업). 위 2026-11-20·2026-11-22 항목의 작업은 그 16커밋을 모르는 채로 했다 — 세션을 시작할 때 원격과의 차이를 확인하지 않았다.
- **`origin/main` 에 이미 있는 것**: ① M2a·M2b·M2c 완료 — 과제 12개(Tier 2 의 누적 합 과제 이름은 `scan`)와 실제 실행 180회(`20261108-tier1`, `20261108-tier2`, `20261109-tier3`). ② 측정 규약 변경(사용자 승인) — 커널·기준선 교차 측정, 호출 친화도 결함 수정, 구현마다 3라운드 × (워밍업 3 + 10회)의 평균·95% 신뢰구간, `OMP_WAIT_POLICY=passive`, `load_1m` 플래그의 정보용 전환, A/A 점검 도구, 결과 121열(이 브랜치는 115열). ③ 호스트별 머신 프로필(`config/machines/`), 저장소 루트 `.env`, 중간보고서(`report/interim/`). ④ 11/11 면담 결정(M3 대상은 `numba`·`halide_auto`·`halide_manual`·`rust`·`mojo`, 최종 장비에서 전부 다시 생성)과 M3 사전 검증에서 채택한 언어별 통합 결정(Halide 자동 스케줄러 Li2018, Rust 크레이트 설정, Mojo 빌드 방식 등). CLAUDE.md 도 그에 맞춰 갱신돼 있다.
- **그래서 이 브랜치의 문서에서 사실이 아닌 것**: "M2a 실제 호출 60회가 보류 중"이라는 서술 전부(이미 실행됐다), "실제 API 호출은 한 번도 하지 않았다"(이 세션이 하지 않았다는 뜻으로만 맞다), 그리고 새 커널 8개를 "추가했다"는 서술 — 같은 과제가 `origin/main` 에 다른 명세·시그니처·테스트로 이미 있고 실제 결과가 그 명세의 `prompt_hash`·`test_hash` 에 묶여 있으므로 기준은 `origin/main` 쪽이다(이 브랜치의 `prefix_sum` 은 `scan` 과 같은 자리의 중복이다).
- **`origin/main` 의 결정과 어긋나는 것(통합할 때 `origin/main` 을 따른다)**: 측정 규약과 결과 스키마(이 브랜치는 M2a 시점의 10회 중앙값·115열), Halide 자동 스케줄러(이 브랜치는 Mullapudi2016), `mojo` 없음, 머신 설정 파일의 위치. 2026-11-22 §6 의 확인 항목도 다시 본다 — A3(numba 스레드 바인딩)은 `origin/main` 이 numba 를 OpenMP 런타임 언어로 취급하기로 이미 정했고, A5(부하 안정 대기)는 `load_1m` 이 정보용이 되어 전제가 바뀌었으며, 새 커널 8개의 허용오차·테스트에 관한 항목(A1·A2·A7, B1–B3)은 `origin/main` 의 과제 계약이 기준이 되면 대상이 없어진다.
- **이 브랜치에만 있는 것(옮길 대상)**: numba·halide·rust 빌더와 `harness/kernels/`, 그 네 언어의 더미 응답, `args=keyword`·`grammar_hint`·`feedback=compiler+lint` 조건과 수정 루프, 성능 결정 요약, sanitizer 실행, 린트 플러그인, 부트스트랩 통계·그림, Windows 점검 도구(`tools/devcheck.py`), Linux 테스트 계획.
- **처리**: 작업 트리는 Windows 에서 검증한 상태 그대로 `main` 이 아닌 브랜치 `claude/m2b-m6-on-m2a-base` 에 커밋해 보존한다(`main` 과 `origin/main` 은 건드리지 않고, push 하지 않는다). `origin/main` 위로 옮기는 작업은 사용자 확인 뒤에 한다.
