# 하네스 설계 v3 — M3–M6 계층

`docs/harness_design.md` 위에 얹힌 M3–M6 계층의 설계 설명이다. `docs/harness_design.md` 는 M1 계약(§0–§15), M2 일반화(§16), Tier 2·3 과제 계약(§17·§18)을 담고, 측정 규약(교차 측정, 호출 친화도, bench 라운드)은 `docs/protocol.md` 와 `docs/decisions.md` 2026-11-08·2026-11-09·2026-11-11·2026-11-12 항목이 기준이다. 이 문서는 그 위에 더해진 것 — 언어 numba·halide_auto·halide_manual·rust, 마이크로 조건(`args=keyword`·`grammar_hint`·`feedback=compiler+lint`), 수정 루프, 성능 결정 요약, sanitizer, 린트 플러그인, 분석 확장, 개발용 점검 도구 — 이 지금 어떻게 동작하는지를 적는다. 결정의 이유와 사용자 확인이 남은 항목은 `docs/decisions.md` 2026-11-23 이 기준이다. 문서와 코드가 다르면 코드가 맞고, 옵션 이름은 각 모듈의 `--help` 로 확인한다. `mojo` 는 아직 하네스에 없다(분석의 언어 순서에만 자리가 있다).

## 0. 불변 조건

- **기존 세 언어(numpy·c_seq·c_openmp)와 12개 과제의 동작은 2026-11-12 판 그대로다**: 워커 명령과 환경, bench 계획, 호출 친화도, 교차 측정, 라운드, 결과 문서, 커널의 판정. 새 언어의 규칙은 이 경로 위에 더해질 뿐이다. 예외는 하네스 쪽 실패의 처리 몇 가지다(결과 문서 없이 끝난 워커는 반환 코드와 무관하게 exit 5, C 컴파일러가 없으면 행을 쓰기 전에 exit 7 등 — `docs/decisions.md` 2026-11-23 "기존 동작에서 달라진 것" ①~⑦, 사용자 확인 대기; §9.2·§9.5).
- **CSV 는 121열 그대로다**(`harness/schema.py` `COLUMNS`). 새 언어는 `backend` 열에 `numba`·`halide`·`rust` 를 쓰고, 수정 루프는 `turn ≥ 1` 행을 더할 뿐이다. 새 기록은 `result.json` 의 추가 키로만 남는다(§9.7).
- **환경 지문(`env_hash` 의 입력)은 `protocol.environment_fingerprint(cfg, machine, static)` 의 2026-11-12 판 키 그대로다.** 새 언어의 도구 판은 행의 `compiler`·`compiler_version`·`cflags` 와 `result.json` 의 `toolchain` 에 남는다.
- **기존 조건의 프롬프트는 바이트 단위로 같다**: numpy·c_seq·c_openmp × 12 과제 × 기본 조건의 `prompt_hash` 36개가 `tests/fixtures/prompt_hashes_upstream.json` 에 고정돼 있고 `tests/test_generate.py` 가 확인한다.
- 그 밖의 규칙은 `docs/harness_design.md` §15 그대로다: 모든 서브프로세스는 `harness.procutil.run_isolated`, LLM 코드는 `harness.worker` 안에서만 실행, 결과 파일은 원자적 쓰기, 주석·docstring 은 한국어, 식별자는 영어, 프롬프트·명세는 영어.
- **비-Linux(개발용 Windows)**: 모든 모듈이 임포트되고 python 계열(numpy·numba·halide)과 Rust(MSVC cargo)의 check·bench 경로가 돈다 — 경로 확인용이며 측정이 아니다. `procutil` 은 `SIGHUP`·프로세스 그룹이 없으면 그것 없이 띄우고, 워커는 `resource`·`prctl`·`os.getloadavg`·`os.sched_getaffinity` 가 없어도 돈다(`limits.unsupported`). 비-Linux 에서 만든 행은 `protocol_flags` 에 `platform=<sys.platform>` 이 붙어 `protocol_ok=0` 이다. gcc 가 없으면 C 언어와 C 기준선이 필요한 run 은 `harness.run` 이 exit 7 로 멈춘다(개발용 C 점검은 §14 의 devcheck). Linux 전용 테스트는 `tests/_support.py` 의 `requires_linux`·`requires_tool(...)`·`requires_modules(...)` 로 건너뛴다. Linux 에서 확인할 항목은 `docs/linux_test_plan.md`.

## 1. 모듈 지도

| 파일 | 역할 |
|---|---|
| `config/experiment.yaml`, `harness/config.py` | 언어 설정(`backend`·`omp_runtime`·`env`·`halide_mode`), `build.rust`, `build.sanitize_flags`, `halide`, `safety`, `lint`, `perf_summary` (§3) |
| `harness/taskspec.py`, `tasks/*/signatures.yaml` | `args` 의 `doc:`, `unsupported:` (§2) |
| `harness/sigs.py` | `args` 에서 언어별 시그니처·매개변수 설명·구조체 렌더링 (§4) |
| `harness/generate.py`, `prompts/lang/*.md`, `prompts/hints/*.md` | 프롬프트 변수, 조건 덮어쓰기 (§5) |
| `harness/extract.py` | 백엔드별 추출·시그니처 검사 (§6) |
| `harness/build/{c,python,numba,halide,rust}.py`, `harness/build/rust_template/` | 언어별 빌드 (§7) |
| `harness/worker.py`, `harness/kernels/{__init__,cabi,pyfunc,halide_pipe}.py` | 커널 로드·호출(백엔드별), `--inspect`, `--sanitize` (§8) |
| `harness/run.py` | 샘플 처리, 라운드, 준비 점검, `result.json` (§9) |
| `harness/repair.py`, `prompts/repair.md` | 수정 루프 (§10) |
| `harness/perf_summary.py` | 성능 결정 요약 (§11) |
| `harness/safety.py`, `plugins/lint/{run_lint.py,rules.yaml}` | sanitizer, 린트 플러그인 (§12) |
| `analysis/{aggregate,stats,plots}.py` | 집계·구간·그림 (§13) |
| `tools/devcheck.py`, `tools/devcheck_shim/` | 개발용 커널 점검 (§14) |
| `harness/dummy_responses/<task>/<language>/` | 더미 응답 (§14) |

## 2. 과제 명세에 더해진 키

과제 계약(정의·형태·허용오차·기준선·정수 출력 규칙)은 `docs/harness_design.md` §16–§18 과 `tasks/<task>/` 가 기준이다. M3–M6 계층이 `signatures.yaml` 에 더한 것은 둘이다.

```yaml
args:
  - {name: data, kind: in, dtype: int32, shape: [n], doc: "..."}   # doc 는 선택: 새 언어의 매개변수 설명에 붙는 영어 구절
unsupported:                                                       # 선택: 이 과제에서 돌리지 않는 언어 → 이유(영어 한 줄)
  halide_auto: "data-dependent control flow is not expressible as a Halide pipeline"
```

- `doc:` 는 기존 세 언어의 `languages.<language>.param_docs` 가 이미 말하는 사실(값의 범위, 스칼라의 뜻, 배치)을 새 언어의 렌더링에 옮긴 것이다. 기존 세 언어는 YAML 의 원문을 쓰므로 프롬프트가 바뀌지 않는다. Tier 2·3 과제 8개와 jacobi2d 가 `doc:` 를 가진다.
- `unsupported:` 는 `sort`·`bfs` 의 `halide_auto`·`halide_manual` 에만 있다 — Halide 는 10개 과제를 돈다.
- `TaskSpec` 의 필드: `ArgSpec.doc`, `unsupported: dict[str, str]`, `languages_raw` (YAML `languages` 원문), 메서드 `supports(language)`. `generate`·`run`·`repair`·`devcheck` 가 지원하지 않는 조합을 건너뛴다.

## 3. 언어 설정

```yaml
languages:
  numpy:         {backend: python, prompt: prompts/lang/numpy.md,    threads: measurement}
  c_seq:         {backend: c,      prompt: prompts/lang/c_seq.md,    openmp: false, threads: 1}
  c_openmp:      {backend: c,      prompt: prompts/lang/c_openmp.md, openmp: true,  threads: measurement}
  numba:         {backend: numba,  prompt: prompts/lang/numba.md,         threads: measurement, omp_runtime: true, env: {NUMBA_NUM_THREADS: "${threads}", NUMBA_THREADING_LAYER: omp}}
  halide_auto:   {backend: halide, prompt: prompts/lang/halide_auto.md,   threads: measurement, halide_mode: auto,   env: {HL_NUM_THREADS: "${threads}"}}
  halide_manual: {backend: halide, prompt: prompts/lang/halide_manual.md, threads: measurement, halide_mode: manual, env: {HL_NUM_THREADS: "${threads}"}}
  rust:          {backend: rust,   prompt: prompts/lang/rust.md,          threads: measurement, env: {RAYON_NUM_THREADS: "${threads}"}}
```

`run.languages` 의 기본값은 `[numpy, c_seq, c_openmp]` 이고 새 언어는 `--languages` 로 고른다.

- **백엔드 이름**: `harness.config.BACKENDS = ("python", "c", "numba", "halide", "rust")`. 워커의 `--backend` 는 `run.WORKER_BACKEND` 가 정한다: c·rust → `c` (Rust 산출물은 C ABI 공유 객체), python → `python`, numba → `numba`, halide → `halide`. 소스 파일 이름은 `run.SOURCE_FILE` (c `kernel.c`, python·numba·halide `kernel.py`, rust `src/lib.rs`).
- **`omp_runtime`**: 커널이 OpenMP 런타임(libgomp) 위에서 도는가. c 는 `openmp` 와 같아야 하고(없으면 그 값), numba 는 `env.NUMBA_THREADING_LAYER == "omp"` 와 같아야 하며(없으면 그 값), python·halide·rust 는 거짓만 된다 — 어긋나면 `ConfigError`. 저장소 설정에서 참인 언어는 c_openmp 와 numba 다. 이 값이 워커 환경의 `omp_env_extra` 여부와 bench 의 호출 친화도를 정한다(§8.6).
- **`env`**: 워커 환경에 마지막으로 덮어쓰는 변수(`protocol.launch_env(..., extra_env=env)`). 값의 `${threads}` 는 그 언어의 스레드 수로 바뀌고 다른 `${…}` 는 거부된다. numba·halide·rust 백엔드는 `harness.config.THREAD_ENV` 의 변수(`NUMBA_NUM_THREADS`·`HL_NUM_THREADS`·`RAYON_NUM_THREADS`)가 그 언어의 스레드 수와 같아야 한다(이 런타임들은 `OMP_NUM_THREADS` 를 읽지 않는다). 행의 `omp_env` 열은 접두 `run.OMP_ENV_PREFIXES` (`OMP_`, `OPENBLAS_`, `MKL_`, `GOMP_`, `KMP_`, `NUMBA_`, `RAYON_`, `HL_`)의 변수를 남긴다.
- **`halide_mode`**: halide 백엔드에 필수(`auto` | `manual`), 다른 백엔드에 쓰면 거부.
- **`build.rust`**: `{cargo: cargo, rustflags: ["-C", "target-cpu=native"], rustflags_linux: ["-C", "link-arg=-Wl,--no-undefined"], build_timeout_s: 600, offline: true}`. `RUSTFLAGS` 는 `protocol.rust_flags(rcfg)` = `rustflags` + (Linux 에서만) `rustflags_linux` 이고, 빌드·프롬프트의 빌드 명령·행의 `cflags` 가 모두 이 값을 쓴다.
- **`build.sanitize_flags`**: `["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]` (sanitizer 빌드에만 붙는다; §12).
- **`halide`**: `{autoscheduler: Li2018, autoscheduler_params: {}}`. halide_auto 에만 적용된다. `parallelism` 은 하네스가 스레드 수로 넣는 예약 키라 설정에 쓰면 거부된다(`HALIDE_RESERVED_PARAMS`). 로더는 `HALIDE_AUTOSCHEDULER_PARAMS` 로 매개변수를 검사한다: Li2018 없음, Mullapudi2016 `last_level_cache_size`·`balance`, Adams2019 `beam_size`. 표에 없는 이름은 검사하지 않고 run 시작의 런타임 점검(§9.5)에 맡긴다. Adams2019·Mullapudi2016 은 같은 응답을 다시 재는 민감도 분석용이다.
- **`safety`** `{enabled: false, timeout_s: 300}`, **`lint`** `{enabled: true, plugin: plugins/lint/run_lint.py, timeout_s: 60}`, **`perf_summary`** `{enabled: true}`.
- **`run.max_repair_turns`**: 0–3 (`MAX_REPAIR_TURNS = 3`). 저장소 값은 0 이라 `harness.repair` 는 `--max-turns` 없이는 아무것도 하지 않는다.
- **조건 덮어쓰기**: `load_config(path, overrides={"args": …, "grammar_hint": …, "feedback": …})` (`generate` 의 `--args`·`--grammar-hint`·`--feedback`).
- **선택 키의 기본값**은 `harness.protocol.OPTION_DEFAULTS` 에도 있고 `protocol.config_option(obj, "lint.enabled")` 처럼 읽는다 — 이 키들을 모르는 `Config` 객체와 설정 없이 도는 프로브(`cfg=None`)도 같은 값을 얻는다.

## 4. 시그니처 렌더링 (`harness/sigs.py`)

새 언어의 시그니처·매개변수 설명·매개변수 구조체는 `signatures.yaml` 의 `args` 에서 만든다. `signature_bundle(spec, language, backend, args_mode)` 가 generate(프롬프트)·extract(검사)·run(빌드)·devcheck 의 공통 입력이고, YAML `languages.<language>` 의 문자열이 있으면 그것이 우선한다(기존 세 언어는 늘 YAML 이다).

`render_signature` 의 출력(`{P}` = `", ".join(params)`):

| backend | positional | keyword |
|---|---|---|
| python | `def {fn}({P}):` | `def {fn}(*, {P}):` |
| numba | `@njit("{numba_signature}")` + 줄바꿈 + `def {fn}({P}):` | `@njit("{numba_signature}")` + 줄바꿈 + `def {fn}(*, {P}):` |
| halide | `def {fn}({입력 배열과 dim}):` | `def {fn}(*, {입력 배열과 dim}):` |
| c | `void {fn}({c_type name, ...})` | `void {fn}(const {fn}_args* args)` |
| rust | `#[no_mangle]` + 줄바꿈 + `pub unsafe extern "C" fn {fn}({name}: {rust_type}, ...)` | `#[no_mangle]` + 줄바꿈 + `pub unsafe extern "C" fn {fn}(args: *const {Camel}Args)` |

- 타입: `c_type` in `const float*`/`const int*`, out `float*`/`int*`, dim `int`; `rust_type` `*const f32`/`*const i32`/`*mut f32`/`*mut i32`/`i32`; `numba_type` C-연속 배열 `float32[::1]`·`float32[:, ::1]`(int32 도 같은 규칙), dim `int64`; `numba_signature(spec)` = `"void(" + 인자 타입들 + ")"`. `halide_params(spec)` 는 입력 배열과 dim 만이다(출력 인자 없음 — 빌드 함수가 출력 `hl.Func` 를 돌려준다). 매개변수 이름은 모든 언어에서 `args` 의 이름 그대로다.
- `render_param_docs`: 결정적인 영어 문장. 배열마다 dtype·형태·읽기 전용/출력·`doc`, dim 마다 `doc` — 없으면 형태 식에 나오는 dim 은 `DIM_DEFAULT_DOC` ("always consistent with the array shapes above"), 형태와 무관한 스칼라는 `SCALAR_DEFAULT_DOC`. C/Rust 는 다차원 배열이 있으면 행우선 선형 인덱스 규칙 한 줄, halide 는 좌표 규칙(`hl.Buffer(array)` 의 기본 `reverse_axes=True`: 형태 (D0, D1) 의 원소 (i0, i1) 은 `X[i1, i0]`)과 출력 `hl.Func` 의 차원·원소 타입을 적는다. 수치 라이브러리 이름은 쓰지 않는다.
- keyword 모드의 c/rust: 구조체 파일 전문 `render_args_struct` — C `kernel_args.h` (가드 `KERNEL_ARGS_H`, `typedef struct { … } {fn}_args;`), Rust `kernel_args.rs` (`#[repr(C)] #[allow(non_snake_case)] pub struct {Camel}Args { … }`). 매개변수 설명 맨 앞에 `keyword_struct_intro` 문단이 붙는다(Rust 문단은 `mod kernel_args;` 와 `use kernel_args::{Camel}Args;` 를 사실로 적는다). YAML `param_docs` 를 쓰는 언어에도 같은 문단이 앞에 붙는다.
- 묶음의 키(늘 전부 있다): `function`, `params`, `signature`, `param_docs`, `args_struct` (keyword 의 c/rust 에서만 비어 있지 않다), `numba_signature` (numba 에서만 비어 있지 않다), `source` (`yaml` | `rendered`; 시그니처 기준).
- 진단 CLI: `python -m harness.sigs --task gemm --backend c --args-mode keyword [--language L] [--json]`.

## 5. 프롬프트 (`harness/generate.py`)

- 템플릿 변수(모든 언어 템플릿에 제공되고, 템플릿이 쓰는 것만 계산한다): 기존 변수 + `numba_signature`, `numba_version`, `numba_threading_layer` (언어 `env` 의 `NUMBA_THREADING_LAYER`), `halide_version`, `halide_autoscheduler`, `halide_coord_example`, `rustc_version`, `cargo_command`, `rust_profile`, `rayon_version` (`rust_profile`·`rayon_version` 은 템플릿 `Cargo.toml`), `args_note`, `grammar_hint`. 도구 판(`numba_version`·`halide_version`·`rustc_version`)은 머신 프로필(`config/machines/<호스트>.yaml`, 없으면 `config/machine.yaml`)에 값이 있으면 그 값이고, 없으면 `PYTHON_EXE` 서브프로세스(numba·halide)나 `protocol.rust_tool("rustc")` 의 `--version` 에서 읽는다.
- `cargo_command` 는 `harness.build.rust.cargo_command_text(rcfg)` 다: `RUSTFLAGS="<protocol.rust_flags>" cargo build --release --offline --locked`. 도구는 늘 맨 이름 `cargo` 로 보인다(경로가 아니다).
- `args_note` (keyword 모드에서만, 빈 줄 하나로 시작): python 계열은 키워드 전용 호출이라는 문단, c 는 `kernel_args.h` 전문과 `#include "kernel_args.h"`, rust 는 `src/kernel_args.rs` 전문과 `mod kernel_args;`.
- `grammar_hint` (on 일 때): `"## Language notes\n\n" + prompts/hints/<language>.md + "\n\n"`. 힌트 파일은 numba·halide_auto·halide_manual·rust 에만 있다 — 파일이 없는 언어는 그 조합을 건너뛴다(`skip`). 힌트는 문법·API 사실만 담는다(성능 조언 없음).
- 기존 세 템플릿에는 `${param_docs}` 바로 뒤의 `${args_note}` 와 마지막 `---` 줄 앞의 `${grammar_hint}` 가 있고, positional·hint off 에서 둘 다 빈 문자열이다(§0 의 바이트 동일성).
- 새 언어 템플릿이 사실로 밝히는 것: numba — 진입 함수의 서명 문자열을 그대로 쓴다, 모듈의 다른 모든 `njit`/`jit` 함수도 첫 위치 인자로 명시 서명을 가져야 한다, `cache=True` 는 허용되지 않는다(둘 다 임포트 전에 거부된다), OpenMP 환경(`OMP_NUM_THREADS`·`OMP_PROC_BIND=true`·`OMP_PLACES=cores`·`OMP_DYNAMIC=false`)은 c_openmp 와 같은 문장. halide_auto — 알고리즘만 쓰고, `hl.Func` 와 그 update stage 의 스케줄 메서드·`apply_autoscheduler` 호출은 효과가 없으며 하네스가 무시하고 기록한다, 오토스케줄러는 `${halide_autoscheduler}`. halide_manual — 스케줄을 같은 빌드 함수 안에 쓴다. rust — `cargo_command`·release 프로필·쓸 수 있는 크레이트(std, rayon)·panic 은 abort.
- 과제가 언어를 지원하지 않으면 그 조합은 `unsupported` 다. 끝에 과제 × 언어 표(`call`/`skip`/`unsupported`)를 낸다.

## 6. 추출 (`harness/extract.py`)

`extract(response_text, *, backend, signature_spec, args_mode="positional", work_dir=None, bundle=None, cc=None)`. run.py·devcheck 는 언제나 `bundle` 과 `cc = build.cc` 를 넘긴다. halide·rust 는 `bundle` 이 필수다.

- 펜스 태그: python·numba·halide → {python, py, python3}, c → {c}, rust → {rust, rs}. 펜스가 전혀 없고 본문에 함수 정의가 있으며 응답 첫 줄이 코드처럼 시작하면 본문 전체를 코드로 본다(`bare_code`; 아니면 `no_code_block`). 첫 토큰 규칙(`_CODE_START_RE`)은 c·python 에 2026-11-12 판 그대로이고, rust backend 에만 Rust 첫 토큰(`#[`, `#![`, `use`, `pub`, `pub(`, `fn`, `unsafe`, `mod`, `extern`, `const`, `static`, 주석)을 더한다.
- python·numba·halide: `ast.parse` 로 최상위 `FunctionDef` 의 매개변수를 `bundle["params"]` 와 비교한다(halide 는 `halide_params`). keyword 모드는 키워드 전용 매개변수만. numba 의 데코레이터 검사는 빌드 단계의 일이다.
- c: `<cc> -E` 전처리 뒤 kernel.c 유래 줄에서 정의를 찾는다. 기본 타입은 한정자를 뺀 단어열 그대로 비교한다(`int32_t`·`signed int` 는 `int` 와 다른 타입 — `param_mismatch`). keyword 모드는 `void {fn}(const {fn}_args* args)` (매개변수 이름은 자유이고 `args` 가 아니면 `signature_exact=False`), 전처리 전에 `kernel_args.h` 를 `work_dir` 에 둔다.
- rust: 주석·문자열을 지운 텍스트에서 본문이 있는 `pub [unsafe] extern "C" fn {fn}(...)` 를 찾는다. 없으면 `function_not_found`, 바로 앞 속성에 `#[no_mangle]`/`#[unsafe(no_mangle)]` 가 없으면 `no_mangle_missing`, 매개변수 개수·이름·타입·반환형이 다르면 `param_mismatch:…`. `unsafe` 생략, `-> ()`, `mut` 바인딩, ABI 가 같은 FFI 별칭(`c_int` = `i32`, `c_float` = `f32`; `RUST_FFI_ALIASES`), keyword 모드의 경로 표기(`*const kernel_args::DotArgs`)는 `signature_exact=False` 일 뿐이다.
- 실패 사유: `no_code_block`, `function_not_found`, `param_mismatch:…`, `static_linkage` (C), `no_mangle_missing` (Rust). 실패 행은 `extract/signature_mismatch`.

## 7. 빌드

공통 결과형은 `harness.build.c.BuildResult` 이고 백엔드별 관찰값은 `extra` (`build.json` 에 저장)다. 도구가 없거나 쓸 수 없는 것은 `ToolchainMissing` 이며 행이 되지 않는다(run 은 exit 7).

### 7.1 C (`build_c`)

`build_c(code, out_dir, bcfg, *, openmp, launch_prefix, symbol, args_header=None, extra_flags=None, artifact_name="kernel.so", write_json=True)`. 기본 인자에서는 2026-11-12 판과 같은 명령·판정이다. 더해진 것: `args_header` 가 있으면 `kernel_args.h` 를 쓴다(keyword), `extra_flags` 는 `cflags` 뒤에 붙고 `-fsanitize=` 가 있으면 링크 플래그에서 `-Wl,--no-undefined` 를 뺀다(sanitizer 빌드), 컴파일러를 PATH 에서 찾을 수 없거나 `nm` 을 시작할 수 없으면 `ToolchainMissing`. 심벌 검사는 `nm -D --defined-only` 의 타입 `T`/`W` 이고, `nm` 이 시작은 됐는데 실패하면 경고하고 빈 목록으로 본다.

### 7.2 numba (`build_numba`)

`build_numba(code, out_dir, *, function, expected_signature)` — `kernel.py` 저장, `compile()` 구문 검사, AST 검사(코드를 임포트하지 않는다). 판정은 먼저 걸리는 것 하나이고 모두 `signature_mismatch` 다:

1. 최상위 `def {function}` 없음 → `function_not_found`.
2. 과제 함수에 Numba 의 `njit`/`jit` 데코레이터가 없음 → `numba_signature_missing`.
3. 모듈 어디서든(최상위·중첩 함수·클래스 본문, 데코레이터와 호출 꼴 `njit(f)` 모두) 명시 서명을 첫 위치 인자로 받지 않고 컴파일되는 함수 → `numba_unsigned_jit:<이름>` (소스 순서로 첫 번째). `signature=` 키워드는 서명으로 치지 않는다(Numba 0.68 은 쓰지 않고 지연 컴파일한다). Numba 의 실제 첫 매개변수 이름 `signature_or_function=` 은 위치 인자와 같다.
4. 어느 `njit`/`jit` 에든 리터럴 `False`/`None` 이 아닌 `cache=` → `numba_cache_true`.
5. 과제 함수의 데코레이터가 `jit` 인데 리터럴 `nopython=True` 가 없음 → `numba_signature_missing`.

`extra = {decorator, signature_literal, signature_matches_expected, options: {parallel, fastmath, cache, boundscheck, nogil}, uses_prange, unsigned_jit_functions, blas_calls}`. `blas_calls` 는 Numba 가 컴파일하는 본문 안의 `np.dot`·`np.vdot`·`np.matmul`·`np.linalg.*`·`배열.dot(…)`·`@` 이다(`{name, line}`) — 거부하지 않고 성능 결정 요약에 넘긴다. `signature_matches_expected` 는 관찰값이라 글자가 달라도 빌드를 실패시키지 않는다(워커가 인자 타입으로 확인한다; §8.3).

### 7.3 halide (`build_halide`)

`build_halide(code, out_dir, *, function, halide_mode)` — `compile()` 과 AST 만. 최상위 `def {function}` 이 없으면 `function_not_found`. 정적 기록 `extra = {schedule_calls: [메서드 이름…], imports_halide}`: 이름이 `SCHEDULE_METHODS` (Halide 22.1 의 `hl.Func`·`hl.Stage` 스케줄 지시 전부; `harness.perf_summary.DEFAULT_SCHEDULE_METHODS` 와 같은 집합)에 있는 속성 호출이다. `halide_mode` 는 판정을 바꾸지 않는다 — auto 인데 스케줄 호출이 있어도 빌드는 통과하고 run.py 가 계약 위반 태그 `schedule_in_auto` 를 남긴다(§9.2). 파이프라인 구성·오토스케줄·JIT 는 워커가 형태마다 한다.

### 7.4 rust (`build_rust`)

`build_rust(code, out_dir, rcfg, *, launch_prefix, symbol, args_module=None, target_dir=None)`.

- 고정 템플릿 `harness/build/rust_template/{Cargo.toml,Cargo.lock}` (LF 고정, `.gitattributes`): 패키지 `kernel`, edition 2021, `crate-type = ["cdylib"]`, 의존성 `rayon = "=1.12.0"` 하나(잠금 파일 고정), `[profile.release] opt-level = 3, codegen-units = 1, lto = "off", panic = "abort", debug = false`. 샘플 디렉터리에는 `Cargo.toml`·`Cargo.lock`·`src/lib.rs` (+ keyword 모드의 `src/kernel_args.rs`)를 쓴다.
- 명령: `launch_prefix + [cargo, "build", "--release", "--offline", "--locked"]` (`build.rust.offline: false` 면 `--offline` 을 뺀다), `RUSTFLAGS` = `protocol.rust_flags`, `CARGO_TARGET_DIR` = run 당 하나(`<built_dir>/<run_id>/_cargo_target`) — 의존성 빌드를 샘플이 재사용한다. 의존성 캐시는 `make rust-setup` (`cargo fetch --locked --manifest-path harness/build/rust_template/Cargo.toml`, 온라인 1회)로 받는다.
- 도구 찾기: cargo 는 `protocol.rust_tool("cargo", build.rust.cargo)`, rustc 는 `protocol.rust_tool("rustc")` — 설정값이 경로면 그대로, 이름이면 PATH → `$CARGO_HOME/bin` → `~/.cargo/bin`. cargo 서브프로세스의 환경은 `protocol.rust_tool_env(cargo)` (= `toolchain_env()` 의 PATH 앞에 찾은 cargo 의 디렉터리; POSIX 의 `toolchain_env` 는 최소 환경 + `CARGO_HOME`·`RUSTUP_HOME`·`RUSTUP_TOOLCHAIN`, 개발용 Windows 는 현재 환경). 도구 점검(`protocol.toolchain_identity`)도 같은 실행 파일·환경을 쓴다.
- 공유 target 디렉터리에서 cargo 는 소스 mtime 으로만 신선도를 판단하므로, 빌드 전에 늘 `cargo clean --release -p kernel` 로 커널 패키지만 지우고 정리 → 빌드 → 산출물 읽기 전체를 `<target>/.harness-build.lock` 의 프로세스 간 배타 잠금 아래에서 한다.
- 산출물 `<target>/release/libkernel.so` 를 `out_dir/kernel.so` 로 복사한다(Windows `kernel.dll`). ELF 이고 `nm` 이 있으면 C 와 같은 심벌 검사(없으면 `signature_mismatch`/`symbol <fn> not exported`); 할 수 없으면 `extra["symbol_check"]` 에 이유(`skipped:not_elf` | `skipped:nm_not_found`).
- 판정: 로그에 ``error: could not compile `kernel` `` 이 있을 때만 `compile_error` (detail = rustc 오류 첫 3개), `Compiling kernel` 뒤의 시간 초과만 `timeout`. 그 밖의 실패(cargo 를 시작할 수 없음, 의존성 캐시 없음, 잠금 파일 불일치, 의존성 컴파일 실패, 링커 없음, rc=0 인데 라이브러리 없음, `nm` 실패, target 잠금 대기 초과)는 `ToolchainMissing`.
- `BuildResult`: `compiler = "rustc"`, `compiler_version` = `rustc --version` 첫 줄, `extra = {rustflags, cargo (실행한 경로), cargo_version, symbol_check}`.
- 헬퍼: `effective_rustflags(rcfg)`, `cargo_command_text(rcfg)` (프롬프트), `rustflags_text(rcfg)` (행의 `cflags`), `probe_toolchain(rcfg, target_dir, *, launch_prefix)` (템플릿의 자리표시 크레이트를 같은 명령·환경·잠금으로 한 번 빌드; 실패는 모두 `ToolchainMissing`).

## 8. 워커

### 8.1 CLI

```
PYTHON_EXE -m harness.worker --backend {c,python,numba,halide} --artifact PATH --mode {import,check,bench} --out PATH
    [--tests …] [--tolerance …] [--signatures …] [--check-repeats 2] [--warmup 3 --repeats 10 --min-repeats 3 --budget-s 300]
    [--args-mode positional|keyword] [--halide-mode auto|manual] [--halide-autoscheduler Li2018] [--halide-params JSON]
    [--threads N] [--inspect] [--inspect-dims JSON] [--max-mem-gb 8] [--sanitize]
    [--baseline-backend c|python --baseline-artifact PATH] [--kernel-cpus auto|none|LIST] [--baseline-cpus auto|none|LIST]
    [--no-loop-fsync]
```

- 새 옵션: `--halide-mode` (halide 에 필수), `--halide-autoscheduler` (기본 `kernels.DEFAULT_AUTOSCHEDULER` = Li2018), `--halide-params` (JSON 객체; Li2018 은 받지 않는다), `--threads` (halide auto 에 필수 — 오토스케줄러 `parallelism`; 그 밖에는 `backend_detail` 의 기록), `--inspect`/`--inspect-dims`, `--sanitize`, `--max-mem-gb ≤ 0` (RLIMIT_AS 없음; `limits.disabled`). 실제 스레드 수는 언어 `env` 의 변수가 정한다.
- 종료 코드: 0 문서 완성, 3 `worker_error` (하네스 쪽 예외; 문서에 트레이스백), 2 인자 오류(문서 없음), 그 밖은 LLM 코드 안에서 죽거나 끝난 것(부분 문서에 `in_progress`). 백엔드 준비 구간(§8.2)에서 프로세스가 죽으면 반환 코드와 무관하게 문서에 `worker_error` 가 남는다.

### 8.2 커널 백엔드 (`harness/kernels`)

워커는 프로세스 수준의 일(제한, 버퍼, 결과 문서, 측정 루프)만 하고, 커널을 로드하고 부르는 방법은 `harness.kernels` 가 감춘다.

```
load = kernels.open_backend(backend, options)   # 하네스 쪽 준비 — LLM 코드는 아직 실행하지 않는다
handle = load(artifact, spec)                   # LLM 모듈 임포트·심벌 조회 → KernelHandle
build_ms = handle.prepare(dims, arrays)         # 형태 하나의 준비(halide 는 파이프라인 빌드·JIT)
ret = handle.invoke()                           # 커널 호출 한 번 — 재는 구간에는 이것만 들어간다
info = handle.inspect(dims)                     # --inspect 관찰값
```

| backend | 모듈 | 산출물 | positional | keyword |
|---|---|---|---|---|
| c | `cabi` | 공유 객체(C·Rust) | `fn(*c_args)` | `fn(byref(ArgsStruct(...)))` (`args` 순서의 `ctypes.Structure`) |
| python | `pyfunc` | `kernel.py` | `fn(*args)` | `fn(**kwargs)` |
| numba | `pyfunc` | `kernel.py` | python 과 같다 | python 과 같다 |
| halide | `halide_pipe` | `kernel.py` (빌드 함수) | 형태마다 파이프라인 → `pipeline.realize(출력 버퍼)` | 빌드 함수 호출만 키워드 |

- `open_backend` 는 하네스가 고른 도구만 건드린다(numba·halide 임포트, numba 스레딩 계층 시작, 오토스케줄러 플러그인 로드와 하네스의 한 줄짜리 파이프라인에 시험 적용). 이 구간 동안 워커는 결과 문서에 `worker_error` 를 미리 써 두므로(`harness_only`), 여기서 프로세스가 죽어도 모델 오류 행이 되지 않는다 — Halide 22.1 은 모르는 오토스케줄러 매개변수에 예외 대신 프로세스 중단으로 끝난다.
- 워커의 순서: BLAS 스레드 예열(`prewarm_blas_threads`) → 고정 코어 읽기(`pinned_cpus`) → `open_backend` → 커널 로드 → (bench) 기준선 로드(`load_baseline`, 모듈 이름 `baseline_kernel`).
- `KernelError(classification, detail[, exception])` 는 LLM 커널 쪽 원인이다: import 모드에서는 `load_ok=false` 와 그 `classification` (`compile_error` | `signature_mismatch`), check/bench 의 형태별 준비 실패는 그 테스트의 `runtime_crash`. 그 밖의 예외는 `worker_error`.

### 8.3 백엔드별 로드 규칙

- **numba**: 로드 뒤 `fn` 이 Numba 디스패처이고 `fn.signatures` 가 비어 있지 않아야 한다(아니면 `signature_mismatch`/`numba_signature_missing`). 그다음 새 컴파일을 끄고(`disable_compile`) 하네스가 넘길 인자 타입(`Buffers` 의 뷰와 같은 C-연속 배열, dim 은 Python `int`)을 컴파일된 서명이 받는지 타입 컨텍스트로 확인한다(`pyfunc.check_accepts_harness_args`) — 아니면 `signature_mismatch`/`numba_signature_rejects_args: no compiled signature of <fn> accepts (<types>)` (또는 `ambiguous compiled signatures for (…)`). 측정 호출 중에는 아무것도 컴파일되지 않는다. 스레딩 계층은 커널보다 먼저 띄운다.
- **halide**: 빌드 함수를 형태마다 부른다 — 입력마다 `hl.ImageParam`, dim 은 그 형태의 실제 크기(Python `int`). 반환값은 정의된 `hl.Func` 하나이고 차원 수·원소 타입이 출력과 같아야 한다(아니면 `signature_mismatch`). auto 면 빌드 함수가 도는 동안 `hl.Func`·`hl.Stage` 의 스케줄 메서드(`harness.build.halide.SCHEDULE_METHODS` ∪ `EXTRA_SCHEDULE_METHODS`)와 `hl.Pipeline.apply_autoscheduler` 가 기록만 하는 무동작이고(`ignore_schedule_calls`; 기록은 `"Func.parallel"`·`"Stage.reorder"`·`"Stage.rfactor->compute_root"` 꼴), 그 뒤 estimate 를 실제 크기로 주고 오토스케줄러를 적용한다(`parallelism` = `--threads`). manual 은 감싸지 않는다. 오토스케줄러 플러그인은 halide 패키지 디렉터리에서 파일을 찾아 절대 경로로 로드한다 — 못 찾으면 `compile_error`/`halide_autoscheduler_unavailable:<이름>`. 이어서 `hl.Pipeline(func).compile_jit(호스트 타깃)`.
- **c**: 2026-11-12 판과 같은 로드·판정(Rust 산출물도 같은 경로).

### 8.4 형태별 빌드(halide)와 `--inspect`

- halide 의 import 모드는 `--tests` 의 첫 테스트(run.py 는 첫 정확성 형태를 준다) 크기로 빌드·JIT 까지 해 본다(실패 → `load_ok=false`, run.py 에서 `error_stage=build`). check 는 테스트마다, bench 는 루프 전에 한 번 빌드한다. 빌드는 커널의 호출 친화도 아래에서 하고 시간 측정 밖이며 `build_ms` 로 남는다. Halide 스레드 풀은 첫 `realize` 에서 만들어진다. halide_manual 의 빌드 함수도 형태마다 그 형태의 구체적인 크기를 받는다.
- `--inspect` (import 모드)는 `inspect` 를 결과 문서에 넣는다. numba: `{signatures, parallel_diagnostics, threading_layer, num_threads}` (`parallel_diagnostics` 는 `parallel=True` 로 컴파일한 함수만). halide: `{loop_nest, schedule_source, autoscheduler, target, autoschedule_ms, jit_ms, dims, ignored_schedule_calls}`. `--inspect-dims` (run.py 가 manifest 에서 읽은 성능 형태의 dim)가 있으면 halide 는 그 크기로 파이프라인을 새로 만들어 기술한다(빌드·오토스케줄·`print_loop_nest` 만, JIT·realize 없음) — 오토스케줄러의 결정은 크기에 따라 다르다. 수집을 시작하기 전에 `inspect.inspect_error` 와 `in_progress.test == "inspect"` 를 미리 써 두므로, 수집 중에 프로세스가 끝나도 문서에는 이미 `load_ok=true` 가 있다(관찰 실패이지 로드 실패가 아니다; §9.2). c·python 의 `inspect` 는 `{}` 다.

### 8.5 `--sanitize` 와 그 밖의 문서 키

- `--sanitize` (sanitizer 실행 전용): 배열마다 C 할당자로 정확히 n 원소만 잡는다(64바이트 정렬 시작, 앞 여유·뒤 가드 없음) — 가드가 같은 malloc 블록 안이면 하네스 배열을 조금 넘는 접근이 ASan 에게 정상 접근으로 보이기 때문이다. 이 모드의 `oob_write` 는 늘 False 다. 정상 check·bench 의 버퍼는 그대로다.
- 추가 키(모두 기존 키에 더해진다): `backend_detail` (`--threads` 를 받았으면 `threads`; halide 는 그에 더해 `halide_mode`·`autoscheduler` (manual 이면 null)와 auto 의 `autoscheduler_params`. run.py 는 `--threads` 를 halide 에만 주므로 파이프라인에서 c·python·numba(·rust)는 `{}`), `sanitize`, 로드 실패가 예외에서 왔으면 `exception = {type, message, traceback}`, halide 의 `build_ms`·`ignored_schedule_calls` (import·bench 는 최상위, check 는 테스트별; import 는 `test`·`dims` 도), check 의 호출 기록마다 `n_fail`·`n_total`, bench 의 `call_affinity.libgomp_paths` (매핑된 libgomp 경로 전부).
- 호출 예외의 `error_detail`: c·python 은 `Type: message` 그대로(2026-11-12 판 형식), numba·halide 는 여러 줄 메시지를 한 줄로 이어 500자까지(`crash_detail`).

### 8.6 교차 측정과 호출 친화도의 언어별 적용

bench 는 언어와 무관하게 `measurement.cores`·`measurement.threads` 로 뜨고 커널과 과제 기준선을 한 워커에서 번갈아 부른다(2026-11-09). 호출 동안의 주 스레드 CPU 집합은 `protocol.call_cpus(cfg, backend=…, openmp=lang.omp_runtime, threads=…)` 가 정한다: `openmp` 면 None(libgomp 의 고정을 그대로), 그 밖에 threads == 1 이면 고정 코어의 첫 CPU, 아니면 고정 코어 전체(`CALL_AFFINITY_POLICY = "2026-11-11"`; `backend` 는 판정에 쓰지 않는다). 워커의 `--kernel-cpus auto` 는 python·halide 에 고정 코어 전체, c·numba 에 None 이지만(`AUTO_FULL_MASK_BACKENDS`) run.py 는 늘 명시 목록이나 `none` 을 넘긴다. `bench_plan.openmp_env` = 커널의 `omp_runtime` ∨ 기준선이 OpenMP C.

저장소 설정(코어 `0,2,4,6`, threads 4)에서의 값:

| 언어 | `--backend` | 백엔드 인자 (`worker_backend_args`) | 언어 `env` | `omp_runtime` | `--kernel-cpus` | 행 threads / cores |
|---|---|---|---|---|---|---|
| numpy | python | — | — | 거짓 | `0,2,4,6` | 4 / `0,2,4,6` |
| c_seq | c | — | — | 거짓 | `0` | 1 / `0` |
| c_openmp | c | — | — | 참 | `none` | 4 / `0,2,4,6` |
| numba | numba | — | `NUMBA_NUM_THREADS=4`, `NUMBA_THREADING_LAYER=omp` | 참 (`omp_env_extra` 를 받는다) | `none` | 4 / `0,2,4,6` |
| halide_auto | halide | `--halide-mode auto --threads 4 --halide-autoscheduler Li2018 --halide-params {}` | `HL_NUM_THREADS=4` | 거짓 | `0,2,4,6` | 4 / `0,2,4,6` |
| halide_manual | halide | `--halide-mode manual --threads 4` | `HL_NUM_THREADS=4` | 거짓 | `0,2,4,6` | 4 / `0,2,4,6` |
| rust | c | — | `RAYON_NUM_THREADS=4` | 거짓 | `0,2,4,6` | 4 / `0,2,4,6` |

- import·check·sanitizer 워커도 같은 백엔드 인자와 언어 `env` 를 받는다(`--inspect` 는 import 에서 numba·halide 에만).
- 교차 측정이라 기준선은 커널 언어의 환경(NUMBA_/HL_/RAYON_ 변수, numba 면 `omp_env_extra`)을 함께 쓴다. 행의 `omp_env` 는 bench 워커가 받은 변수다.
- numba 의 omp 계층과 C(기준선·커널)가 서로 다른 libgomp 파일을 올리면(`call_affinity.libgomp_paths` 의 서로 다른 경로가 둘 이상) 행에 정보용 플래그 `multiple_libgomp` 가 붙는다(`protocol.INFO_FLAG_PREFIXES`).
- `bench_timeout_s` 는 bench 워커 전체의 상한이다. halide 의 형태별 빌드와 numba 의 임포트 컴파일은 측정 구간 밖이지만 라운드마다 새 워커에서 다시 하므로 이 상한을 쓴다(사용자 확인이 남은 항목; `docs/decisions.md` 2026-11-23).

## 9. `harness/run.py`

### 9.1 재사용 API

```python
ctx = setup_run(cfg, run_id, *, skip_bench=False, force=False, strict_protocol=False, no_baseline=False,
                tasks=None, languages=None, samples=None)          # 실패는 RunSetupError(exit_code); 행을 쓰지 않는다
row, status = process_sample(ctx, ctx.tasks[task], raw_path)      # status ∈ processed | skipped | ignored; bench 는 라운드 1 까지
run_pending_rounds(ctx, entries)                                  # entries = [(row, result_json_path(row)), …]: 라운드 2..R
finish_run(ctx, entries)                                          # 과제별 대응 기준선 중앙값 기준 drift, model_served 혼재 플래그
```

`harness.run` 의 `_main` 과 `harness.repair` 가 같은 API 를 쓴다. CLI: `--run-id`, `--config`, `--tasks`, `--languages`, `--samples`, `--skip-bench`, `--no-baseline`, `--force`, `--strict-protocol`, `--rebuild-csv`. `ctx.raw_files` 는 처리 순서(과제·언어·조건 태그·샘플 번호·턴; 샘플 번호는 수로 정렬)의 raw 이고, 턴 파일 `sample_<i>.turn<t>.json` 도 포함한다(`paths.parse_raw_turn_name`). 과제가 지원하지 않는 언어의 raw 는 경고와 함께 빠진다. 행의 `run_id` 는 처리 중인 run 디렉터리 이름이고, raw 안의 값이 다르면 `result.json` 의 `raw_run_id` 에 남긴다.

### 9.2 `process_sample` 의 단계

| # | 단계 | 기록 |
|---|---|---|
| 0 | raw 를 읽을 수 없음 / 설정에 없는 언어 / 과제가 지원하지 않는 언어 → `(None, "ignored")` | — |
| 1 | `sample_ref` → 턴 디렉터리(`paths.built_turn_dir`: 턴 0 `…/sample_<i>/`, 턴 t `…/sample_<i>/turn_<t>/`), `raw_sha256` | — |
| 2 | 건너뛰기 판정 `is_up_to_date` (§9.3) → 저장된 행, `"skipped"` | — |
| 3 | `openmp = lang.omp_runtime`, `lang_env = env`; `base_row` (`launch_env(cfg, lang.threads, openmp=…, extra_env=lang_env)`) | 행 |
| 4 | 생성 게이트(API 오류·절단·거절) | `error_stage=generate` |
| 5 | extract (`signature_bundle`, `cc = build.cc`) | `stages["extract"]` |
| 6 | build (`build_kernel`: c·python·numba·halide·rust), `cflags` = `build_flags_text`, 계약 위반 `forbidden_symbol:*`·`threads_in_c_seq`·`schedule_in_auto` (halide_auto 인데 `extra["schedule_calls"]` 가 있음) | `stages["build"]`, numba·halide·rust 는 `toolchain` |
| 7 | lint (`lint.enabled`; 소스 파일이 생긴 뒤라 빌드가 실패해도) | `stages["lint"]`, `lint_*` 열 |
| 8 | 빌드 실패 → `build/compile_error` (`compile_timeout`) 또는 `build/signature_mismatch` | — |
| 9 | import 워커(halide 는 첫 정확성 형태; `--inspect [--inspect-dims]` 는 numba·halide 에만) | `stages["import"]` |
| 10 | perf_summary (`perf_summary.enabled`; `inspect` 는 import 문서의 것) | `stages["perf_summary"]`, `perf_summary` 열 |
| 11 | check 워커 → `with_panic_detail` 또는 `check_failure` | `stages["check"]` |
| 12 | sanitizer (`safety.enabled` ∧ backend c ∧ check 가 timeout 이 아님) | `stages["sanitize"]`, `stages["safety"]`, `asan/ubsan_findings` |
| 13 | check 실패 → `check/<유형>` (memory_error 규칙) | — |
| 14 | `--skip-bench` → 여기서 행을 쓴다(`requested.bench=False`) | `result.json` |
| 15 | bench 라운드 1: `baseline_run_time_ms`·`baseline_time_1t_ms`, `run_bench_round(…, 1)`, `apply_rounds` | `stages["dynamic"]`, `["bench"]`, `["call_affinity"]`, `["bench_artifact"]`, `["rounds_planned"]`, `["rounds"]` |

- **import 판정**: `--inspect` 를 받지 않는 백엔드(기존 세 언어와 rust)는 2026-11-12 판 그대로다 — 종료 규칙의 판정이 있으면 `build` 단계의 그 판정, 아니면 `load_ok` 거짓이면 문서의 `classification`. numba·halide 는 커널이 로드된 뒤(`load_ok`) 관찰값을 모으다 끝난 것을 판정으로 보지 않는다: 경고하고 관찰값 없이 계속하며 판정은 check·bench 에 맡긴다. import 가 `halide_autoscheduler_unavailable:` 로 끝나면 `ToolchainMissing` 이다. halide_auto 의 import 문서(최상위 또는 `inspect`)에 `ignored_schedule_calls` 가 있으면 `schedule_in_auto` 를 더한다(한 번만; 판정은 바뀌지 않는다).
- **워커 종료 규칙** (`WorkerOutcome.verdict`): 결과 문서에 `worker_error` 가 있거나 문서를 한 번도 쓰지 못하고 끝나면 반환 코드와 무관하게 `HarnessError` (행 없음, exit 5) — 워커는 LLM 코드를 로드하기 전에 첫 문서를 쓰므로 문서가 없으면 LLM 코드는 실행된 적이 없다. 그 밖은 2026-11-12 판 그대로(timed_out → `timeout`, rc<0 → `runtime_crash` (signal), 문서가 있는 rc>0 → `runtime_crash`, rc==0 인데 미완료 → `runtime_crash`).
- **Rust panic** (`with_panic_detail`): rust 커널의 `runtime_crash` 판정이면 `<stage>.stderr.log` 전문(없으면 꼬리)에서 **첫** panic 을 읽어 상세를 `rust_panic:<메시지>` 로 바꾸고 `stages[<stage>]["rust_panic"] = {message, location}` 을 남긴다(import·check·모든 bench 라운드; Rust 1.73 이후와 그 전의 두 형식). 다른 언어의 판정은 그대로다.
- **memory_error** (`apply_memory_error`): 판정이 `runtime_crash`·`wrong_output` 이고 `asan_findings > 0` 이면 `error_type = memory_error`, `error_detail = "asan:<첫 종류>; was <이전 유형>/<이전 상세>"`. check 와 bench 의 모든 라운드에 같은 규칙이고, 통과한 샘플의 `pass` 는 바꾸지 않는다.
- **행의 도구 열**: `compiler`·`compiler_version` 은 빌드 결과에서(c 는 컴파일러, python·numba·halide 는 패키지와 판, rust 는 `rustc` 와 `rustc --version`). `cflags` (`build_flags_text`): c 는 `base_flags + cflags (+ openmp_flag) + ldflags`, rust 는 `rustflags_text` + `; profile.release: <키=값 …>` (그 빌드의 `Cargo.toml`), halide 는 auto 면 `autoscheduler=<이름> params=<워커가 넘기는 매개변수의 정규 JSON; parallelism 포함>`, manual 이면 `autoscheduler=none`, python·numba 는 NA.
- 그 밖의 계약 위반 태그 `stdout_output` (워커 stdout 이 비어 있지 않음)과 플래그 `call_affinity_failed`, `paired_baseline_failed` (그 라운드의 교차 기준선 실패 — 그 라운드는 속도비에서 빠진다)는 2026-11-12 판 그대로다.

### 9.3 다시 처리할지, 라운드를 이을지

`result.json` 의 `requested = {bench, safety, lint}` 는 그 결과를 쓸 때 요청된 선택 단계다(`requested_stages`: bench = `--skip-bench` 아님, safety·lint = 설정). 규칙마다 함수 하나가 정한다.

1. **다시 처리** = `is_up_to_date` 가 거짓일 때(`needs_processing` 과 `process_sample` 이 같은 함수를 쓴다): `--force` 이거나, `result.json` 이 없거나, `raw_sha256` 이 다르거나, 이 run 이 요청했는데 저장된 결과에 없고 그 행에서 돌았을 단계(`missing_stages`; `stage_applies`)가 `REPROCESS_STAGES = ("bench",)` 에 있을 때. 라운드가 덜 찬 것은 빠진 단계가 아니다. `requested` 가 없는 예전 결과는 `stored_stages` 가 행과 단계 기록에서 추정한다.
2. **다음 라운드** = `next_round` (2026-11-12 판 그대로): `--skip-bench` 아님, `bench_ran`, `passed`, `error_type` 없음, 라운드 1개 이상, `rounds_planned == measurement.bench_rounds`, `row.env_hash == ctx.env_hash`, 지금까지 라운드가 모두 `ok`, 계획보다 적음.
3. **처리하면 라운드는 처음부터**: `stages` 를 새로 만들고 라운드 1 을 다시 돈다.
4. **라운드를 다시 쓸 때 출처는 그대로**: `run_pending_rounds` 는 `row`·`stages`·`written_at` 만 바꾸고 저장된 `requested` (없으면 `stored_stages`)·`raw_run_id`·`toolchain` 을 둔다.

| 저장된 result.json | 호출 | `process_sample` | 라운드 2..R |
|---|---|---|---|
| 없음, 또는 raw 가 바뀜 | 측정 | 전 과정 + 라운드 1 | R 까지 |
| 없음, 또는 raw 가 바뀜 | `--skip-bench` | check 까지; `requested.bench=False` | 없음 |
| 통과, `requested.bench=False` (`--skip-bench` 로 씀) | 측정 | **다시 처리**(extract…check, 라운드 1) | 있음 |
| 같음 | `--skip-bench` | 건너뜀 | 없음 |
| bench 전에 실패 | 측정 | 건너뜀(bench 가 적용되지 않음) | 없음 |
| bench 완료, 라운드 수 = `rounds_planned` | 측정 | 건너뜀 | 필요 없음 |
| bench 일부, `rounds_planned == bench_rounds`, 같은 `env_hash`, 모두 ok (중단된 run, 수정 턴) | 측정 | 건너뜀 | **이어서** `len+1` 부터; 간격은 저장된 `finished_at` 부터 |
| 같음 | `--skip-bench` | 건너뜀(`result.json` 그대로) | 없음 |
| bench 일부인데 `env_hash` 가 다르거나 `rounds_planned ≠ bench_rounds` | 측정 | 건너뜀 | 없음 — 행마다 한 번 경고(`stalled_rounds`), `bench_rounds` = ok 라운드 수로 남는다; `--force` 로 다시 잰다 |
| 라운드가 오류로 끝남(bench timeout 은 pass 유지) | 측정 | 건너뜀 | 없음 |
| 라운드 기록 이전의 결과(`rounds_planned` 없음) | 측정 | 건너뜀 | 없음 |
| 무엇이든 | `--force` | 전부 다시, 라운드 1 부터 | 있음 |
| 무엇이든 | `--force --skip-bench` | check 까지 다시; bench 없이 덮어씀 | 없음 |
| `safety`·`lint` 를 켜기 전에 씀 | 측정 | 건너뜀(그 열은 NA 라고 알린다 — 채우려면 `--force`, 측정도 다시 한다) | 평소대로; `requested.safety/lint` 는 거짓으로 남는다 |

기준선: `setup_run` 은 다시 처리할 샘플이 있는 과제만 run 시작·1스레드 기준선을 재고, 없는 과제는 디스크의 기준선을 읽어 둔다(`adopt_existing_baselines`). 아직 빌드하지 않은 C 기준선은 `run_bench_round` 안에서 빌드된다.

### 9.4 `run_pending_rounds`

`run_pending_rounds(ctx, entries)`: 라운드 2..R (`measurement.bench_rounds`)를 라운드 번호 순으로 한 바퀴씩 돌아 같은 구현의 라운드가 서로 다른 시각에 재지게 한다. `entries` 는 이 호출이 처리했거나 건너뛴 행 전부이고(멈췄던 run 의 라운드를 잇는다) 라운드를 돈 항목은 갱신된 행으로 바뀐다. 같은 행의 라운드 사이 간격 `bench_round_gap_s` 는 저장된 마지막 라운드의 `finished_at` 부터 센다. 과제 컨텍스트나 언어 설정이 없는 행, bench 산출물(`stages["bench_artifact"]`)이 없는 행은 경고하고 건너뛴다. `--skip-bench` 면 아무것도 하지 않는다. `HarnessError` 는 `라운드 r 중단 (<샘플>)` 을 붙여, `ToolchainMissing` 은 그대로 올린다(둘 다 행을 쓰지 않는다). 라운드마다 결과 문서는 `round_file(r)` (1 → `bench.json`, r → `bench_r<r>.json`), 요약은 `stages["bench_r<r>"]` 와 `stages["rounds"][r-1] = {round, status, result_file, finished_at, dynamic_flags}` 이고, `apply_rounds` 가 모든 라운드 문서를 `harness/benchstats.py` 로 모아 시간·기준선·속도비 열(`speedup_vs_baseline`, `speedup_mean`, `speedup_ci_lo/hi`, `speedup_rounds`, `bench_rounds`)을 다시 채운다.

### 9.5 준비 점검과 종료 코드

`setup_run` 은 행을 쓰기 전에 다음을 차례로 확인한다.

1. `protocol.check_static()`: BLAS 가 OpenBLAS 가 아니면 exit 4, 정보용 외 플래그는 경고(`--strict-protocol` 이면 exit 4).
2. raw 탐색(없으면 exit 1)과 과제 명세 읽기.
3. 도구(`missing_tools`, `protocol.toolchain_identity` 의 결과로): c 언어 → `build.cc`, rust → `rust_tool` 이 찾은 cargo·rustc, numba·halide → `PYTHON_EXE` 에서 임포트 가능, halide_auto → 설정의 오토스케줄러 플러그인 로드 가능, bench 를 하고 C 기준선 과제가 있으면 → `build.cc`. 없으면 exit 7.
4. 쓸 수 있는지 한 번씩 돌려 보기(실패는 exit 7): 실행 접두(`probe_launch_prefixes`: run 이 쓸 `protocol.launch_prefix` 마다 `[*접두, PYTHON_EXE, -c, pass]`; 접두가 비면 건너뛴다) → 언어 런타임(`probe_language_runtimes`: halide_auto 의 오토스케줄러와 매개변수, numba 의 스레딩 계층을 그 언어의 실행 접두·환경으로 `python -m harness.protocol --probe-runtime halide|numba` 서브프로세스에서 워커와 같은 코드로) → SciPy(`probe_numba_scipy`: numba 언어가 있으면 그 워커와 같은 인터프리터·환경에서 `scipy` 를 실제로 임포트해 본다 — 임포트되거나 결과 없이 끝나면 문제) → Rust(`probe_rust_toolchain` → `build.rust.probe_toolchain`; run 의 공유 target 디렉터리에서).
5. 과제마다 테스트 입력(`ensure_tests`)과 기준선(`prepare_baselines`; 게이트 실패 exit 6, 기준선 빌드의 도구 없음 exit 7, 기준선 워커의 harness_error exit 5).

환경 지문은 `environment_fingerprint(cfg, machine, static)` 로 계산하고, `toolchain_identity` 는 도구 점검에만 쓴다.

`harness.run` 의 종료 코드(`run.EXIT_*`): 0 정상, 1 입력 없음 등, 2 설정 오류, 4 `--strict-protocol` 위반 또는 BLAS 가 OpenBLAS 가 아님, 5 harness_error(샘플 처리 중의 예기치 않은 예외 포함; 행 없음), 6 기준선 게이트 실패, 7 도구가 없거나 쓸 수 없음(행 없음), 130 중단. 샘플 처리나 라운드 도중의 `ToolchainMissing` 도 exit 7 이다.

### 9.6 `finish_run` 과 `--rebuild-csv`

- `finish_run` 에는 이 run 의 모든 `result.json` 행(`read_result_rows(<built_dir>/<run_id>)`; 수정 턴 포함)을 넘긴다 — `--tasks`·`--languages`·`--samples` 로 좁힌 호출도 같다. 요약은 이번 호출의 행만, 보정된 값으로 낸다.
- `--rebuild-csv` 는 `result.json` 에서 CSV 를 다시 만들되(provider 가 dummy 인 행은 `results_csv_dummy` 로) `result.json` 이 없는 기존 CSV 행을 남긴다. 기존 CSV 의 헤더가 스키마와 다르면 그 행들을 열 이름으로 옮긴다(없는 열은 NA, 스키마에 없는 열은 버린다; `schema.rebuild_csvs`).

### 9.7 샘플 디렉터리와 `result.json`

샘플(턴) 디렉터리의 파일: 소스(`kernel.c` | `kernel.py` | `src/lib.rs` + `Cargo.toml`·`Cargo.lock`), keyword 모드의 `kernel_args.h` | `src/kernel_args.rs`, 산출물 `kernel.so` (개발용 Windows 의 Rust 는 `kernel.dll`), `build.log` (C·Rust), `build.json`, `lint.json`·`lint.stderr.log`, 워커 문서 `import.json`·`check.json`·`bench.json`·`bench_r<r>.json`·`safety.json` 과 단계마다 `<stage>.stdout.log`·`<stage>.stderr.log`, sanitizer 빌드·로그 `san/` (`kernel.c`, `kernel_san.so`, `build.log`, `build.json`, `asan.<pid>`, `ubsan.<pid>`), `result.json`. run 공용: `<built_dir>/<run_id>/_cargo_target/` (`_toolchain_probe/` 포함), `<built_dir>/<run_id>/<task>/baseline/{bench_run.json,bench_1t.json}`.

`result.json` = `{"row", "raw_sha256", "stages", "requested": {"bench", "safety", "lint"}, ["raw_run_id"], ["toolchain"], "written_at"}`. `toolchain` 은 빌드 단계까지 간 numba·halide·rust 행에만 있다: `{compiler, compiler_version}` (+ rust 는 `cargo`, `cargo_version`). `stages` 의 키:

- `extract`: `{ok, signature_ok, signature_exact, detail, n_blocks}`
- `build`: `BuildResult.to_dict()` 에서 `log` 를 뺀 것(+ `extra`)
- `lint`: `{tool, findings, summary, ok, error}` (`summary` 의 `null` = 그 범주를 검사하지 않음)
- `import` / `check` / `bench` / `bench_r<r>` / `safety`: `WorkerOutcome.summary()` = `{rc, timed_out, signal, elapsed_s, timeout_s, stdout_bytes, stdout_tail, stderr_tail, result_file, complete, classification, error_detail}` (+ rust panic 이면 `rust_panic: {message, location}`)
- `perf_summary`: `{ok, notes}` (요약 본문은 행의 `perf_summary` 열)
- `sanitize`: `{ran, build_ok, asan_findings, ubsan_findings, kinds, detail, summary}`
- `dynamic` (라운드 1 의 `DynamicStatus`), `call_affinity` (라운드 1 bench 문서의 것), `bench_artifact`, `rounds_planned`, `rounds`
- 키가 없는 경우: `lint` 는 꺼져 있거나 extract 실패·생성 게이트 행, `perf_summary` 는 빌드나 로드 실패, `sanitize`·`safety` 는 safety 가 꺼져 있거나 C 가 아니거나 check 가 timeout 인 행(`safety` 는 sanitizer 워커가 실제로 돈 경우 — Linux, sanitizer 빌드 성공 — 에만), bench 관련 키는 check 를 통과하지 못했거나 `--skip-bench` 인 행. 읽는 쪽은 `.get` 으로 읽는다.

## 10. 수정 루프 (`harness/repair.py`)

```
PYTHON_EXE -m harness.repair --run-id ID [--config] [--max-turns N] [--tasks …] [--languages …] [--samples …]
    [--feedback compiler|compiler+lint] [--seed-from-feedback compiler|compiler+lint] [--yes] [--skip-bench]
```

- 전제: 같은 run 의 턴 0 이 `harness.generate` + `harness.run` 으로 처리돼 있다. 직전 턴의 `result.json` 이 그 raw 의 것이 아니면 그 샘플은 경고하고 건너뛴다.
- 범위: `conditions.feedback` 이 `--feedback` (없으면 설정값)인 샘플. `args`·`grammar_hint` 는 raw 의 것을 따른다. 제공자·모델은 턴 0 과 같아야 한다(다르면 호출 없이 exit 1; dummy run 은 `LLM_PROVIDER=dummy` 로 돌린다).
- 대상: 직전 턴 행이 `valid_sample=1 ∧ pass=0 ∧ error_stage ≠ generate`. bench timeout(`pass=1`)·거절·API 오류·절단은 대상이 아니다. 직전 턴 응답이 비어 있는 샘플, 앞 턴의 지금 행이 대상이 아닌 샘플은 더 잇지 않는다.
- 턴 t = 1..N (N = `--max-turns`, 없으면 `run.max_repair_turns`; 상한 3): 대상 수를 보여 주고 확인 → 대화 조립 → 호출 → raw 저장(`sample_<i>.turn<t>.json`) → `run.process_sample` (bench 는 라운드 1 까지). 턴 루프가 끝나면 이 호출이 만든 행으로 `run.run_pending_rounds` 를 돌고(확인 거부·치명 API 오류로 멈춘 경우에도) 이 run 의 모든 행으로 `run.finish_run` 을 한다 — 수정 턴의 행은 같은 raw 를 `harness.run` 이 처리한 행과 같은 열·같은 `result.json` 키를 갖는다. 그래서 턴 t+1 의 대상은 턴 t 의 라운드 1 판정으로 정해지고, 라운드 2 이후에만 실패한 턴은 다음 수정 호출에서 이어진다. bench 를 하는 호출이면 수정할 과제에 디스크의 run 시작·1스레드 기준선이 없을 때만 먼저 잰다(`ensure_baselines`). `--samples` 는 `setup_run` 에도 넘긴다.
- 대화: `[user: 턴 0 프롬프트, assistant: 턴 0 응답, user: 수정 요청 1, …, user: 수정 요청 t]`. system 은 턴 0 의 것. raw 의 추가 키: `turn`, `prompt = {system, user, messages}`, `prompt_hash = sha256(system + "\n\x00\n" + json.dumps(messages, ensure_ascii=False))[:16]`, `repair = {parent_raw, feedback, diagnostics, lint_included, n_lint_findings}`. 이미 응답이 있는 턴은 다시 묻지 않고, 생성 실패 턴이나 대화가 이어지지 않는 턴부터 다시 호출한다(이전 파일은 `.prev<N>.json`).
- 수정 요청 = `prompts/repair.md` (`${stage}`, `${error_type}`, `${diagnostics}`, `${lint_section}`). 주는 정보는 컴파일러·테스트 러너가 줄 법한 사실까지다 — 하네스의 패턴·분류 이름과 참조값은 넣지 않는다(`build_diagnostics`, 4000자 상한):
  - extract: 추출 실패 사유를 풀어 쓴 문단.
  - build: 컴파일러·cargo 출력의 첫 오류 줄부터(없으면 꼬리 30줄), 또는 로드 실패. Numba 서명 규칙(`numba_unsigned_jit:<이름>`, `numba_cache_true`, `numba_signature_rejects_args: …`)은 `reason_text` 의 영어 문장으로 바뀐다(하네스의 사유 이름은 나가지 않는다).
  - check/bench: 형태별 `shape (M=…, …): <결과>` (`passed`, `incorrect output (max abs error …, k of n elements outside tolerance)`, `non-finite values in the output`, `results differ between two identical calls`, `modified a read-only input`, `wrote outside the output buffer`, 예외 메시지와 트레이스백 끝, 프로세스의 끝). bench 는 실패한 라운드의 기록(`bench_record_name`: 마지막 라운드의 `result_file` 이름 — `bench` 또는 `bench_r<r>`)을 읽는다. Rust panic 이면 프로세스의 끝(abort) 대신 `panicked at src/lib.rs:<줄>:<열>: <메시지>` (`panic_text`; 절대 경로는 파일 이름만).
  - `${error_type}` 은 테스트 러너 수준의 말로 낮추고(`wrong_output`·`arg_order`·`numeric_instability` → `incorrect output`, `race_condition` → `non-deterministic output`), sanitizer 가 바꾼 `memory_error` 는 상세의 `; was <유형>/` 에서 원래 판정으로 되돌린다(수정 요청이 `safety.enabled` 에 따라 달라지지 않게).
  - `feedback=compiler+lint` 는 `${lint_section}` 만 다르다: `"\n## Static analysis findings\n\n- line <n>: <message>\n…"` (최대 20개; `safety.lint_messages`), 린트가 돌았고 없으면 `None.`, 돌지 않았으면(단계 없음, `ok` 거짓, `tool == "none"`) `Not available.`. Rust 는 베이스라인 린터가 없어 늘 `Not available.` 이다.
- `--seed-from-feedback compiler --feedback compiler+lint`: `fb-compiler` 조건의 턴 0 raw 를 `fb-compiler+lint` 조건 디렉터리로 복사하고 끝낸다(API 호출 없음; `conditions.feedback` 을 바꾸고 `seeded_from`·`seeded_sha256` 기록). 같은 실패 샘플에서 두 피드백 조건이 갈라지는 대응 비교용이고, 복사본은 `harness.run` 으로 처리한 뒤 수정한다. 복사본에 수정 턴이 이미 있는데 원본이 바뀌었으면 거부한다(exit 1). `make repair-dummy` 가 이 순서를 돈다.
- 종료 코드: 0 정상, 1 설정·입력, 2 확인 거부, 3 치명 API 오류, 5 하네스 오류, 130 중단. `harness.run` 의 준비·처리·라운드가 멈추면 그 코드(4, 5, 6, 7)를 그대로 돌려준다.

## 11. 성능 결정 요약 (`harness/perf_summary.py`)

`summarize(*, language, backend, code, build_log="", task="", spec=None, inspect=None, build_extra=None) -> dict` — 순수 함수(파일·서브프로세스·LLM 코드 실행 없음)이고 예외를 올리지 않는다(내부 실패는 전체 키 집합에 `notes = "perf_summary error: <Type>: <msg>"`, `ERROR_NOTE_PREFIX`). run.py 는 import 문서의 `inspect` 와 빌드의 `extra` 를 넘긴다.

- 출력은 CLAUDE.md §5 의 키 전부(`language, loops, parallel, vectorized, tiling, memory, library_calls, decisions_by_llm, decisions_by_compiler, notes`)와 확장 `decisions_by_library`, `parallel.loops`·`parallel.placement` (`none|outer|inner|mixed|region|unknown`)·`parallel.constructs` 를 갖는다. `parallel.kind ∈ {none, openmp, threads, rayon, numba, halide, unknown}`, `vectorized.status ∈ {yes, partial, no, unknown}`, `vectorized.source ∈ {compiler_remark, intrinsics, loop_nest, interpreter, none}`. 결정 어휘는 `DECISIONS` (`loop_order`, `tiling`, `vectorization`, `parallel_region`, `reduction`, `memory_layout`, `library_delegation`, `algorithm_choice`, `accumulation_precision`)로 고정이고 `library_delegation` 은 `decisions_by_library` 에만 싣는다.
- 거짓 확신을 만들지 않는다: 정할 수 없으면 문자열 `"unknown"`, 수·불리언 `null`, 리스트 `[]`. `false`·`0`·`"none"` 은 "찾아봤는데 없다"는 뜻이다. `loops` 는 가장 안쪽 루프마다 하나(최대 20개), `notes` 는 영어로 `; ` 로 잇고 500자까지.
- 백엔드별 소스: **c** — 토큰열의 `for` 중첩, `#pragma omp`, 타일링 단서, 인트린식, 알려진 라이브러리 호출, 할당 수, `build_log` 의 gcc `-fopt-info-vec` 줄(루프 줄 번호에 대응; gcc 가 `ignoring '#pragma omp …'` 라고 경고했으면 지시문은 `constructs` 에만 남는다). **rust** — `for … in` 중첩, rayon 병렬 반복자·`join`·`scope`, `std::arch` 인트린식(리마크가 없어 벡터화는 손으로 쓴 SIMD 가 가장 안쪽 루프를 덮을 때만 `yes`). **python·numba** — `ast` 의 원소 루프, NumPy 호출(`@` 는 `np.matmul`), 임시 배열, 데코레이터 옵션·`prange`·병렬 진단; numba 는 빌드의 `blas_calls` 를 `library_calls` (`np.dot`, `np.matmul`, `np.linalg.*`)와 라이브러리 위임으로, `unsigned_jit_functions` 를 notes 로 싣는다. **halide** — `inspect["loop_nest"]` 에서 루프·병렬·벡터화. halide_manual 은 소스의 스케줄 호출이 LLM 결정이다. halide_auto 는 `inspect["schedule_source"]` (오토스케줄러 출력; Li2018·Mullapudi2016·Adams2019 모두 읽는다)의 결정이 컴파일러 결정이고, 소스의 스케줄 호출과 빌드의 `schedule_calls` 는 결정으로 세지 않고 notes 에만 `schedule in source (ignored by the harness[, N call(s) at build]): …` 로 남긴다. auto 에서 오토스케줄러 출력이 없으면 타일링·전치는 `null` 이다.
- `SCHEDULE_METHODS` (`harness/build/halide.py`)와 `DEFAULT_SCHEDULE_METHODS` 는 같은 집합이다(테스트가 비교한다). 고정 입력은 `tests/fixtures/perf_summary/` (Mullapudi2016·manual 케이스와 Li2018 케이스; `make_inspect_fixtures.py --write` 는 없는 파일만 쓰고 `--overwrite` 가 덮어쓴다).
- 진단 CLI: `python -m harness.perf_summary --language L --backend {python,numba,halide,c,rust} --file SRC [--build-log build.log] [--inspect import.json]` (과제 명세 없이 요약하므로 파이프라인 행과 같지 않을 수 있다).

## 12. 안전성: sanitizer 와 린트 플러그인

### 12.1 sanitizer (`harness/safety.py`)

- `run_sanitizers(cfg, *, code, sample_dir, openmp, launch_prefix, symbol, args_header, tests, worker)`: Linux 의 C 커널만(Python·Numba·Halide·Rust 는 돌리지 않고 열은 NA; TSan 은 쓰지 않는다). `san/` 에 `build_c(extra_flags=build.sanitize_flags, artifact_name="kernel_san.so")` → run.py 의 워커 클로저로 `worker("safety", "check", tests, safety.timeout_s, ["--check-repeats", "1", "--sanitize"], artifact=…, env_extra=sanitizer_env(…), max_mem_gb=0)` → `san/asan.*`·`san/ubsan.*` 와 `<sample>/safety.stderr.log` 전체를 파싱(`parse_sanitizer_logs`: ASan 보고 수, UBSan `runtime error` 의 서로 다른 위치 수, 종류).
- `sanitizer_env`: `LD_PRELOAD` = `<cc> -print-file-name=libasan.so` (gcc 처럼 `libubsan.so` 가 실제 파일이면 그 뒤에 함께 — 워커가 `PR_SET_DUMPABLE=0` 을 건 뒤에는 libubsan 이 `/proc/self/environ` 을 읽지 못한다), `ASAN_OPTIONS` = `detect_leaks=0:abort_on_error=1:allocator_may_return_null=1:handle_sigfpe=0:log_path=<san>/asan`, `UBSAN_OPTIONS` = `print_stacktrace=1:halt_on_error=0:log_path=<san>/ubsan`.
- 값의 뜻: 0 은 커널을 로드해 끝까지 돈 실행에서만 0 이다. 로드하지 못했는데 보고가 없으면 None 과 `detail = "load_failed:<사유>"`, 끝까지 가지 못했는데 보고가 없으면 None 과 `"incomplete:<사유>"`. 꺼져 있으면 `disabled`, Linux 가 아니면 `unsupported_platform:<p>`, sanitizer 빌드 실패는 `build_failed:<kind>: <detail>` (모두 `ran=False` 또는 열 NA). 컴파일러가 없으면 `ToolchainMissing` 을 그대로 올린다. 판정 규칙은 §9.2 의 memory_error.

### 12.2 린트 플러그인 (`plugins/lint/run_lint.py`)

- 호출: `PYTHON_EXE <plugin> --language <lang> --file <path> --task <task> --signature <json> --out <json>` (`safety.run_lint`; `--signature` 는 `{function, params, signature, args_mode}`, 환경에 `CC=<build.cc>`). 플러그인은 하네스를 임포트하지 않는 독립 CLI 이고 같은 계약을 지키는 다른 구현으로 바꿔 끼울 수 있다. 플러그인 문제(없음, 비정상 종료, 시간 초과, 계약 위반)는 빈 결과 `ok=False` 와 경고 한 줄로 끝난다 — 실험은 막히지 않는다.
- 베이스라인: numpy·numba·halide_auto·halide_manual → ruff, pyflakes / c_seq·c_openmp → cppcheck, clang-tidy / rust → 없음(`tool: "none"`). 규칙 ID → 범주(`memory, numeric, concurrency, shape_dtype, api_misuse, performance`)·심각도는 `plugins/lint/rules.yaml` 이고, 표에 없는 규칙은 버리고 출력의 `unmapped` 에 수만 센다. clang-tidy 의 컴파일 인자는 `rules.yaml` 의 `compile_args` 이고, OpenMP 언어는 `CC` 의 `omp.h` 한 파일만 담은 임시 디렉터리를 `-isystem` 으로 준다. 출력에는 계약 키 외에 `unmapped`, `versions` 가 있다. 종료 코드 0 정상(도구가 없거나 실패해도), 2 인자·입력·`rules.yaml` 오류.
- 값의 뜻: 도구가 파일을 분석하지 못하면(시간 초과, 비정상 종료, 읽을 수 없는 출력, 구문·컴파일 오류) 그 도구는 돌지 않은 것이다. 도구가 하나도 돌지 않으면 `{"tool": "none", "findings": [], "summary": {범주: null…}}`. `summary` 의 범주는 그 범주를 볼 수 있는 도구가 하나라도 끝까지 돌았을 때만 수이고 아니면 `null` 이다. 두 도구가 같은 줄·같은 범주를 보고하면 하나만 남긴다.
- 행의 열(`safety.lint_columns`): 플러그인 실패나 `tool == "none"` 이면 `lint_*` 전부 NA, `null` 범주는 그 열만 NA, `lint_findings_total` 은 검사한 범주의 합(검사한 범주가 없으면 NA) — 0 은 "검사했고 없음"이다.
- `schema.source_hash` (행 `harness_version` 의 dirty·nogit 해시)의 대상에는 `plugins/lint/run_lint.py`·`rules.yaml` 과 `harness/build/rust_template/{Cargo.toml,Cargo.lock}` 이 들어 있다.

## 13. 분석

### 13.1 `analysis/aggregate.py`

```
python -m analysis.aggregate [--csv results/runs.csv] [--run-id ID | --latest | --all-runs] [--json]
                             [--ci] [--coverage] [--by-turn] [--conditions] [--halide]
                             [--primary-speedup mean|median] [--config config/experiment.yaml]
```

- **속도비 주 지표** (`choose_speedup`): 설정의 `measurement.primary_speedup` (또는 `--primary-speedup`)이 `mean` 이면 `speedup_mean` (라운드를 모은 기준선 평균 / 커널 평균), `median` 이면 `speedup_vs_baseline` (중앙값 비). `mean` 이어도 선택된 통과·측정 행 가운데 `speedup_mean` 이 빈 행(이전 규약으로 잰 행)이 하나라도 있으면 전부 `speedup_vs_baseline` 으로 비교한다. 선택 전체에서 한 번 정하고, `--ci` 의 속도비 구간·`--halide`·`analysis.plots` 가 같은 열을 쓴다. 그룹 표의 `speedup_median`·`speedup_mean_median` 열은 주 지표와 무관하게 둘 다 나온다.
- 수정 루프 행이 있으면 그룹 표는 첫 시도(`turn == 0`)만 센다. `--by-turn`: 그룹별 `pass_within_<t>` (턴 t 까지 허용한 누적 통과율), `turns_to_pass` 분포, 누적 토큰 — 결과를 모르는 턴은 NA. `--conditions`: 같은 provider·model·과제·언어 안의 조건 수준 쌍의 통과율 차와 구간(`feedback` 은 같은 run 의 같은 `sample_idx` 끼리 대응 비교; `args`·`grammar_hint` 는 턴 0 의 독립 표본). `--coverage`: 과제 × 언어 표(첫 시도 유효 샘플 수 n, unsupported 는 `n/a`, 행 없음은 `-`).
- `--halide`: Halide 두 언어에 대해 과제 묶음(`supported` / `without_awkward` / `awkward`)마다 언어별 첫 시도 통과율과 속도비 기하평균. `HALIDE_UNSUPPORTED_TASKS = (sort, bfs)`, `HALIDE_AWKWARD_TASKS = (jacobi2d, spmv_csr, floyd_warshall, cholesky, fft)` — 언어가 순차 알고리즘을 강제하는 과제(2026-11-12 결정).
- 언어 순서 `LANGUAGE_ORDER` (성능 결정 부담 순): `numpy, numba, halide_auto, halide_manual, mojo, rust, c_openmp, c_seq, opencl, c_neon, llvm_ir, aarch64_asm`. 목록에 없는 언어는 이름순으로 뒤를 잇는다. 과제 순서 `TASK_ORDER` 는 Tier 순. 옵션이 더하는 표는 이 순서이고 그룹 표는 이름순이다.
- 종료 코드: 0 정상(행이 없어도), 1 CSV·run 없음, 2 CSV 열이 스키마와 다름.

### 13.2 `analysis/stats.py`

`bootstrap_ci(values, stat=…, n_boot=2000, alpha=0.05, seed=0)` (백분위 구간, NaN 은 뺀다), `bootstrap_pass_rate`, `paired_bootstrap_diff` (같은 샘플 id 의 짝), `unpaired_bootstrap_diff` (독립 표본). 모두 `n_boot=2000, alpha=0.05, seed=0` 이 기본이다.

### 13.3 `analysis/plots.py`

`python -m analysis.plots [--csv …] [--run-id ID | --latest | --all-runs] [--out results/plots] [--format png|pdf|svg] [--by-task] [--primary-speedup mean|median] [--config …]`. 그림: `fig1_languages` (과제마다 패널 한 쌍; x = 성능 결정 부담 순 언어, 위 pass@1 과 95 % 구간, 아래 통과 샘플의 속도비 로그 축 — 주 지표 열), `fig2_error_types` (언어별 누적 막대: 통과 / 오류 분류 / 생성 단계 실패; `--by-task` 면 `fig2_error_types_by_task`), `fig3_condition_effects`, `fig4_repair_turns`. 그림 1·2 는 턴 0 과 기준 조건 조합 하나만 그린다. 색은 Okabe–Ito 팔레트, 그림 안의 글자는 영어다. 자료가 없는 그림은 건너뛰고 stderr 에 알리며, 쓴 파일 경로는 stdout 에 한 줄씩 나온다. 종료 코드는 aggregate 와 같다.

## 14. 더미 응답·devcheck·테스트

- 더미 `harness/dummy_responses/<task>/<language>/<k>.md`: `0.md` 는 정답(응답처럼 한두 문장 + 펜스 블록 하나). 12개 과제 × 7개 언어에서 `unsupported` 가 아닌 80개 조합 모두에 `0.md` 가 있다(새 언어 44개: numba 12, halide_auto 10, halide_manual 10, rust 12). 수정 루프 검증용으로 gemm 에 `c_seq/1.turn1.md`, `numpy/2.turn1.md` 가 있다. dummy provider 는 턴 ≥ 1 이면 `<k>.turn<t>.md`, 없으면 `<k>.md` 를 준다.
- `make` 타깃: `make rust-setup` (Rust 의존성 캐시; rust 가 들어가는 run 전에 한 번), `make dummy-m3` / `make dummy-all` (새 언어 네 개 / 언어 일곱 개의 더미 파이프라인, n=1, 과제 전부, `--coverage`), `make repair-dummy` (수정 루프와 피드백 조건 대응 비교), `make devcheck`, `make devcheck-baselines`.
- `tools/devcheck.py` (개발용; 측정 아님, 행을 쓰지 않는다):

```
python tools/devcheck.py kernel --task T --language L --file PATH [--args-mode positional|keyword] [--tests tiny,rect,small] [--bench] [--keep DIR]
python tools/devcheck.py baseline --task T [--tests …] [--bench] [--keep DIR]
python tools/devcheck.py dummies [--tasks a,b] [--languages x,y] [--args-mode …] [--dummy-dir DIR] [--keep DIR]
python tools/devcheck.py gen --task T
#   네 명령 공통: [--config] [--tasks-dir]; kernel·baseline·dummies 공통: [--tests …] [--keep DIR] [--cc auto|gcc|zig|zig-omp] [--timeout]
```

  - 단계와 판정은 `harness.run.process_sample` 과 같다: extract → 백엔드별 빌드 → 모든 백엔드의 import 워커(halide 는 첫 정확성 형태; `--inspect` 없음) → check. 추출 실패 `stage=extract`, 빌드·로드 실패 `stage=build`, 정확성 실패 `stage=check`, rust panic 은 `runtime_crash`/`rust_panic:<메시지>`. run.py 가 행에 남길 계약 위반(`schedule_in_auto`, `stdout_output`, `forbidden_symbol:*`, `threads_in_c_seq`)은 판정 줄의 `violations=` 로 보인다.
  - 언어 설정은 `harness.config.load_config` 의 `LanguageCfg`, 워커 환경은 `protocol.launch_env` (Windows 에서는 `SYSTEMROOT` 등 시스템 변수만 더한다), 오토스케줄러 기본값은 설정의 `halide.autoscheduler` (없으면 `kernels.DEFAULT_AUTOSCHEDULER`).
  - C: Linux 에 설정의 컴파일러가 있으면 `build_c` 와 설정의 플래그, 없으면 `python -m ziglang cc` 직렬 빌드(OpenMP pragma 무시, `tools/devcheck_shim/omp.h` 스텁과 `shim.h`; 논리 검증만, 판정 줄에 `toolchain=zig-serial`). `--cc zig-omp` 는 OpenMP 대상(c_openmp, `openmp: true` 인 C 기준선)을 `python -m ziglang cc -fopenmp` (clang)로 컴파일해 이 머신의 LLVM OpenMP 런타임(Windows `libomp140.x86_64.dll`; `DEVCHECK_LIBOMP`)에 링크하고(`<omp.h>` 는 `tools/devcheck_shim/omp_real/omp.h`), 스레드 수는 설정값 또는 `DEVCHECK_OMP_THREADS` 다. 이 모드는 워커 환경에서 `OMP_PROC_BIND`·`OMP_PLACES` 를 뺀다(`ZIG_OMP_DROP_ENV` — libomp140 에서 이 두 변수가 `schedule(dynamic, N)` 루프를 깨뜨린다). clang + libomp 는 측정 경로의 gcc + libgomp 가 아니므로 Linux 확인을 대신하지 않는다.
  - `dummies` 를 `--tasks`/`--languages` 없이 돌리면 과제는 과제 디렉터리 ∪ 더미 디렉터리, 언어는 `0.md` 가 있는 언어 ∪ 설정의 `languages` 이고, `unsupported` 가 아닌데 `0.md` 가 없는 조합은 `missing` 이다.
  - 종료 코드: 0 통과, 1 커널이 틀림, 2 사용법·도구 문제.
- 단위 테스트: `tests/test_<module>.py` (unittest). 전체 실행 `python -m unittest discover -s tests` (`make unittest`). Linux 전용 경로(gcc 빌드, taskset, sanitizer, cppcheck·clang-tidy 등)는 순수 함수로 쪼개 고정 입력으로 시험하고, 실제 실행은 `docs/linux_test_plan.md` 에 있다.

## 15. 열린 항목

사용자 확인이 남은 결정(halide_manual 의 크기 전달, 결과 문서 없는 워커의 exit 5, `bench_timeout_s` 와 빌드 시간, 민감도 분석 절차, C 크래시의 수정 루프 진단 등)과 Linux 에서만 검증할 수 있는 항목은 `docs/decisions.md` 2026-11-23 과 `docs/linux_test_plan.md` 가 기준이다. 이 문서는 그 결정이 바뀌면 해당 절을 고친다.
