# 하네스 설계 계약 v3 (M2b–M6 코드 선행 구현)

v3 2026-11-20, **구현 반영 2026-11-22**. `docs/harness_design.md`(v2.1 + §16)의 **확장**이다. v2 문서의 규칙(파일 소유, `procutil.run_isolated` 단일 진입점, 오케스트레이터에서 LLM 코드 실행 금지, 원자적 쓰기, 한국어 주석·영어 식별자)은 그대로 유효하고, 이 문서는 M2b·M2c(커널 8개), M3(numba·halide·rust), M4(키워드 인자·문법 안내·수정 루프), M5(성능 결정 요약·ASan/UBSan·린트), M6(분석·그래프)에 필요한 인터페이스를 정한다. 근거와 이탈은 `docs/decisions.md` "2026-11-20".

구현자는 자기 그룹의 파일만 고친다. 다른 그룹과 맞닿는 함수 시그니처·JSON 키·CLI 옵션은 이 문서가 기준이며, 바꿔야 하면 이 문서를 먼저 고치고 최종 보고에 적는다.

**이 문서의 상태(2026-11-22).** 이 문서는 구현 **전에** 그룹 사이의 계약으로 썼고, 구현·리뷰·수정이 끝난 뒤 지금의 코드에 맞춰 고쳤다. 각 절은 코드가 지금 하는 일을 적는다. 계획과 달라진 곳은 그 자리에 한 문장으로 이유를 남겼고, 구현 중에 정한 세부는 §16 에 모았다. 이 개발 머신(Windows 11)에서 확인한 것과 Linux 에서만 확인할 수 있는 것은 §16 끝의 "검증 범위"에 나눠 적었다. 결정의 근거와 사용자 확인이 필요한 항목은 `docs/decisions.md` 의 2026-11-20 이후 항목이 기준이다(이 문서는 결정을 따로 기록하지 않는다). 문서와 코드가 다르면 코드가 맞고, 옵션 이름은 각 모듈의 `--help` 로 확인한다.

## 0. 작업 방식 (2026-11-20 사용자 지시)

- **코드를 먼저 쓰고 Linux 테스트는 나중에 한 번에 한다.** 지금 머신은 Windows 11(Python 3.14, gcc·taskset·sysfs 없음)이다. Linux 에서만 확인되는 것은 구현자가 최종 보고의 `linux_only` 목록에 적고, 마지막에 `docs/linux_test_plan.md` 로 모은다(문서 그룹 D 의 몫; 이 문서를 고친 시점에는 아직 없었다).
- **로컬에서 검증할 수 있는 것은 전부 지금 검증한다.** 개발용 venv: `C:\graduation_work\.venv-win\Scripts\python.exe` (Python 3.14.0, numpy 2.3.5(OpenBLAS 휠), pyyaml, pandas 3.1, matplotlib, numba 0.68.1, halide 22.1.0, ruff, pyflakes, ziglang). Rust 는 `cargo`/`rustc` 1.92(MSVC). 모든 로컬 실행은 이 인터프리터로 한다(`.venv/bin/python` 이 없으면 `PYTHON_EXE` 는 `sys.executable` 로 대체된다).
- 하네스는 여전히 **Linux 전용 측정 도구**다. Windows 지원은 "임포트가 되고, python 계열 백엔드의 check 까지 돈다"가 목표였다(§5.6). 구현 뒤에는 python 계열(numpy·numba·halide)과 Rust(MSVC cargo)의 check·bench 경로까지 돈다 — 경로 확인용이며 측정이 아니다. C 언어는 gcc 가 없어 `harness.run` 이 exit 7 로 멈춘다(개발용 C 점검은 §14 의 devcheck). 비-Linux 에서 만든 행은 `protocol_flags` 에 `platform=<sys.platform>`(정보용 아님)이 붙어 `protocol_ok=0` 이다.
- **실제 API 호출·git commit/push 는 하지 않는다.** provider 는 `dummy` 만 쓴다.
- 구현자의 최종 보고(구조화 출력): `files_changed`, `tests_run`(명령과 결과), `decisions`(decisions.md 에 들어갈 문장), `linux_only`(Linux 에서 확인할 항목과 확인 방법), `open_issues`.

## 1. 파일 목록과 소유

```
G1 기반        harness/taskspec.py, classify.py, testgen.py, sigs.py(신규), procutil.py, paths.py,
               harness/worker.py(정수 출력·비-Linux 임포트 안전성까지만; 이후 G2a 로 넘어감),
               tools/devcheck.py(신규), tools/devcheck_shim/{omp.h,shim.h}(신규), tests/_support.py(신규),
               tests/test_{taskspec,classify,sigs,procutil,testgen}.py, 기존 tests/*.py 의 비-Linux skip 표시, .gitignore
G2a 워커       harness/worker.py, harness/kernels/{__init__,cabi,pyfunc,halide_pipe}.py(신규), tests/test_worker.py, tests/test_kernels.py
G2b 추출·빌드  harness/extract.py, harness/build/{c,python,numba,halide,rust}.py, harness/build/rust_template/*, tests/test_extract.py, tests/test_build_*.py
G3 생성·수정   harness/config.py, config/experiment.yaml, harness/generate.py, harness/llm_client.py, harness/repair.py(신규),
               prompts/lang/{numba,halide_auto,halide_manual,rust}.md, prompts/lang/{numpy,c_seq,c_openmp}.md(변수 2개 추가만),
               prompts/hints/*.md, prompts/repair.md, harness/dummy_responses/gemm/** 의 turn 파일, tests/test_{config,generate,llm_client,repair}.py
G4 실행        harness/run.py, harness/schema.py, harness/protocol.py, tests/test_{run,schema,protocol}.py
G5 성능 요약   harness/perf_summary.py, tests/test_perf_summary.py
G6 안전성      harness/safety.py, plugins/lint/**, tests/test_{safety,lint_plugin}.py
G7 분석        analysis/aggregate.py, analysis/plots.py, analysis/stats.py(신규), tests/test_{aggregate,plots,stats}.py
T1..T4 과제    tasks/<task>/**, harness/dummy_responses/<task>/{numpy,c_seq,c_openmp}/0.md  (과제 2개씩)
L1..L3 언어 더미  harness/dummy_responses/<12 과제>/{numba,halide_auto,halide_manual,rust}/0.md, tasks/*/signatures.yaml 의 unsupported 갱신
D 문서         README.md, Makefile, requirements.txt, docs/{decisions,protocol,error_taxonomy,linux_test_plan,harness_design_v3}.md
```

순서: G1 → (G2a, G2b, G3, G4, G5, G6, G7, T1..T4 병렬) → (L1..L3, 통합 검증) → 리뷰·수정 → D.

실제 진행: L 단계는 언어별 세 그룹(numba, halide, rust)이 Tier 별로 나눠 더미를 썼고, 리뷰에서 확인된 결함 27건은 영역별 수정 그룹(run, worker, build, gen, aux, analysis)이 고쳤다. 오케스트레이터가 직접 더한 파일은 `.gitattributes`(Rust 템플릿 `Cargo.toml`·`Cargo.lock` 을 LF 로 고정)와 `tools/devcheck_shim/omp_real/omp.h`(§14 `--cc zig-omp`)다. 테스트 픽스처 `tests/fixtures/prompt_hashes_m2a.json`(§9.1)과 `tests/fixtures/{kernels,perf_summary,safety}/**` 도 새 파일이다(perf_summary·safety 픽스처의 gcc·sanitizer 로그는 손으로 만든 것이라 Linux 에서 실제 로그로 바꿔야 한다).

## 2. 과제 명세 확장 (G1)

### 2.1 `signatures.yaml` 추가 키

```yaml
args:
  - {name: keys, kind: in,  dtype: int32, shape: [n], doc: "each value is in the range 0 <= keys[i] < n_bins"}   # doc 는 선택
  - {name: hist, kind: out, dtype: int32, shape: [n_bins]}
  - {name: n, kind: dim}
  - {name: n_bins, kind: dim, doc: "the number of bins"}
unsupported:                       # 선택. 이 과제에서 돌리지 않는 언어 → 이유(영어 한 줄; 분석 표에 그대로 나온다)
  halide_auto: "data-dependent control flow (frontier expansion) is not expressible as a Halide pipeline"
languages:                         # 선택. 있으면 렌더러(§3)보다 우선한다(기존 4개 과제의 numpy/c_seq/c_openmp 항목은 그대로 둔다)
```

- `kind`: `in` | `out` | `dim`. `dim` 은 "C `int` 스칼라 인자"라는 뜻이며 차원이 아닌 정수(예: BFS 의 `source`)에도 쓴다.
- 출력은 과제당 정확히 1개, dtype 은 `float32` 또는 `int32`(Tier 2 부터). 복수 출력은 없다(FFT 는 실수·허수 교차 배열 하나).
- `TaskSpec` 추가 필드: `unsupported: dict[str, str]`, `languages_raw: dict[str, dict]`, `ArgSpec.doc: str`. 메서드 `supports(language) -> bool`(= unsupported 에 없음).

### 2.2 `tolerance.yaml` — 정확 비교

```yaml
task: histogram
dtype: int32
compare: exact          # tolerance(기본) | exact
sentinel: -2048         # 정수. 참조 출력에 이 값이 나오지 않게 gen_tests 가 보장한다(assert)
```

`Tolerance` 에 `compare: str = "tolerance"` 추가. `exact` 면 `rtol = atol = 0`, `numeric_pattern_tol_ratio = 0` 이 기본이고 YAML 에 없어도 된다.

### 2.3 분류 (`classify.py`)

- `compare_exact(out, ref) -> dict`: `compare` 와 같은 키. `ok` = 전부 같음, `max_abs_err` = 정수 차의 최댓값(float), `max_tol_ratio` = None, `nan_inf` = False.
- `classify_generic` 은 `tol.compare == "exact"` 이면: 규칙 1 생략 → 규칙 2(전부 같음 → `pass`) → 규칙 3(패턴 함수) → 규칙 4(전부 센티널 → `output_not_written`/`returned_array_not_written`; `out − sentinel == ref` → `accumulated_into_c`; 일부 센티널 → `partial_write:k/n`; `oob_write` → `out_of_bounds_write`) → 규칙 5 생략 → 규칙 6 `wrong_output`/`mismatch`. 정확 비교 과제에는 `numeric_instability` 가 없다.
- `classify_task` 시그니처는 그대로다. GEMM 전용 패턴, 그 밖의 2-D 전치 패턴도 그대로다.
- 워커의 실행 간 비교(`outputs_close`)는 exact 면 `array_equal`.
- 패턴 이름 `accumulated_into_c` 는 과제와 무관하게 그대로 쓴다(이름을 바꾸면 기존 행과 어긋난다).
- 구현 결과: 정확 비교 과제 넷(`histogram`, `prefix_sum`, `sort`, `bfs`)은 출력이 모두 1-D 라 패턴 함수(2-D 전치만 본다)가 맞을 일이 없고, 이 과제들에서 `arg_order` 는 나오지 않는다. 누적 검사는 부분 기록 검사보다 먼저다 — 0 으로 지우지 않은 히스토그램의 빈 구간은 센티널이 그대로 남아 `n_unwritten` 에 세어지기 때문이다. 정확 비교의 `metrics` 는 허용오차 경로와 키가 같고 `max_tol_ratio = None`, `nan_inf = False` 다.

### 2.4 테스트 입력 (`testgen.py`)

- 참조 dtype: float 출력은 float64(기존), int32 출력은 **int32**(파일 크기 때문에 int64 로 넓히지 않는다). `generate()` 가 출력 dtype 에 맞는지 검사한다(호출자가 `out_dtype="int32"` 를 넘긴다; 기본 `"float32"`).
- int 출력 과제는 `ref` 에 센티널 값이 없음을 `generate()` 가 확인한다(`sentinel` 인자, 기본 None).
- npz 키·manifest·test_hash 정의는 그대로다. 형태 이름은 모든 과제가 `tiny, rect, small, large`(1-D 과제에서 `rect` 는 "2의 거듭제곱이 아닌 중간 크기"라는 뜻이다).

## 3. 시그니처 렌더러 `harness/sigs.py` (G1)

과제 12개 × 언어 7개의 시그니처·매개변수 설명을 YAML 에 손으로 쓰지 않고 `args` 에서 만든다. YAML `languages.<language>` 에 해당 문자열이 있으면 그것이 우선한다(기존 프롬프트의 바이트 동일성 유지).

```python
BACKENDS = ("python", "numba", "halide", "c", "rust")
def camel(function: str) -> str                      # "spmv_csr" → "SpmvCsr"
def c_type(arg) -> str                               # in float32 "const float*", in int32 "const int*", out "float*"/"int*", dim "int"
def rust_type(arg) -> str                            # "*const f32" | "*const i32" | "*mut f32" | "*mut i32" | "i32"
def numba_type(spec, arg) -> str                     # 1-D "float32[::1]", 2-D "float32[:, ::1]", int32 동일 규칙, dim "int64"
def numba_signature(spec) -> str                     # "void(" + ", ".join(numba_type) + ")"
def halide_params(spec) -> list[str]                 # in 배열과 dim 만(출력 인자 없음), args 순서
def render_signature(spec, backend, args_mode) -> str
def render_param_docs(spec, backend, args_mode) -> str
def render_args_struct(spec, backend) -> str         # c: kernel_args.h 전문, rust: kernel_args.rs 전문, 그 외 ""
def signature_bundle(spec, language, backend, args_mode) -> dict
    # {"function", "params", "signature", "param_docs", "args_struct", "numba_signature", "source": "yaml"|"rendered"}
```

`render_signature` 의 출력(정확한 문자열; `{P}` = `", ".join(params)`):

| backend | positional | keyword |
|---|---|---|
| python | `def {fn}({P}):` | `def {fn}(*, {P}):` |
| numba | `@njit("{numba_signature}")\ndef {fn}({P}):` | `@njit("{numba_signature}")\ndef {fn}(*, {P}):` |
| halide | `def {fn}({halide P}):` | `def {fn}(*, {halide P}):` |
| c | `void {fn}({c_type name, ...})` | `void {fn}(const {fn}_args* args)` |
| rust | `#[no_mangle]\npub unsafe extern "C" fn {fn}({name}: {rust_type}, ...)` | `#[no_mangle]\npub unsafe extern "C" fn {fn}(args: *const {Camel}Args)` |

- C 구조체(`kernel_args.h`): `typedef struct {\n    const float* A;\n    ...\n    int M;\n} {fn}_args;` — 필드는 `args` 순서, 헤더 가드 `KERNEL_ARGS_H`.
- Rust 구조체(`kernel_args.rs`): `#[repr(C)]\n#[allow(non_snake_case)]\npub struct {Camel}Args {\n    pub A: *const f32,\n    ...\n}` — 같은 순서.
- 매개변수 이름은 모든 언어에서 `args` 의 이름 그대로다(Rust 도 `A`, `M`; 템플릿이 `#![allow(non_snake_case)]` 를 안내한다).
- `render_param_docs` 는 영어, 언어 중립 규칙으로 쓴다(라이브러리 언급 금지). 배열마다 dtype·형태·읽기 전용/출력 여부·`doc`, dim 마다 `doc`(없으면 "always consistent with the array shapes above"), 반환 규칙. 다차원 배열이 있으면 C/Rust 는 "row-major: element (i0, i1) of an array of shape (D0, D1) is at linear index i0*D1 + i1" 한 줄을 넣는다. halide 는 §5.3 의 좌표 규칙 문장을 넣는다.
- 테스트: 기존 4개 과제의 numpy/c_seq/c_openmp positional 은 `source == "yaml"` 이고 YAML 문자열과 같다; C positional 렌더링은 기존 YAML 의 C 시그니처와 문자 단위로 같다(gemm, jacobi2d, spmv_csr, dot).

## 4. 새 커널 8개 (T1..T4)

공통: 행우선, 64바이트 정렬, 입력 읽기 전용, 출력은 호출자가 할당(초기 내용 미정의, 전부 덮어써야 함), 반환값 무시. `spec.md` 는 기존 4개와 같은 절 구성(Definition / Data layout / Contract / Accuracy / Evaluation / Goal)·같은 문장 틀을 쓴다. 언어 중립(라이브러리·알고리즘 이름으로 **지시**하지 않는다; 정의에 필요한 수학만).

| task(=function) | 인자(순서) | 형태 tiny / rect / small / large | 판정 | 기준선 |
|---|---|---|---|---|
| `histogram` | `keys` in int32 [n] (0 ≤ 값 < n_bins), `hist` out int32 [n_bins], `n`, `n_bins` | (10,4) / (100003,1000) / (2²⁰,256) / (2²⁶,256) | exact | C+OpenMP (스레드별 사본 후 합산) |
| `prefix_sum` | `x` in int32 [n] (0 ≤ 값 < 16), `out` out int32 [n] (포함형: out[i] = x[0]+…+x[i]), `n` | 7 / 100003 / 2²⁰ / 2²⁶ | exact | NumPy `np.cumsum(x, dtype=np.int32, out=out)` |
| `nbody` | `pos` in f32 [n,3], `mass` in f32 [n], `acc` out f32 [n,3], `n` | 5 / 317 / 1024 / 8192 | rtol 1e-4, atol 1e-4 (2026-11-22 실측으로 확정; 제안값은 rtol 1e-3) | C+OpenMP |
| `sort` | `keys` in int32 [n] (전체 int32 범위, 센티널 제외), `out` out int32 [n] (오름차순), `n` | 9 / 100003 / 2¹⁶ / 2²⁴ | exact | NumPy `np.sort` |
| `cholesky` | `A` in f32 [n,n] (대칭 양의 정부호, 전체 저장), `L` out f32 [n,n] (하삼각, 위쪽은 0), `n` | 4 / 97 / 256 / 2048 | rtol 1e-4, atol 1e-5 (2026-11-22 실측으로 확정; 제안값은 rtol 1e-3, atol 1e-4) | NumPy `np.linalg.cholesky` |
| `floyd_warshall` | `W` in f32 [n,n] (대각 0, 나머지 양의 유한값), `D` out f32 [n,n], `n` | 5 / 97 / 256 / 1024 | rtol 1e-4, atol 1e-4 | C+OpenMP |
| `bfs` | `row_ptr` in int32 [n_vertices+1], `col_idx` in int32 [n_edges], `dist` out int32 [n_vertices] (도달 불가 −1), `n_vertices`, `n_edges`, `source` | (6,~10) / (1000,~4000) / (2¹⁶,~2¹⁹) / (2²²,~2²⁵) | exact | C 순차(큐) — `openmp: false` |
| `fft` | `x` in f32 [2n] (re,im 교차), `out` out f32 [2n], `n` (2의 거듭제곱) | 8 / 1024 / 2¹⁶ / 2²² | rtol 1e-3, atol 1e-3 (입력 진폭 1/√n) | NumPy `np.fft.fft` |

세부(구현자가 spec 에 정확히 쓴다):

- **nbody**: `acc[i] = Σ_j mass[j]·(pos[j] − pos[i]) / (|pos[j] − pos[i]|² + ε²)^{3/2}`, ε = 0.01 고정(명세에 수치로), j = i 항은 0 이라 포함해도 된다. 입력: `pos` uniform[0,1)³, `mass` uniform[0.5,1.5)/n → `acc` 가 O(1). 근사 역제곱근만 쓴 구현(상대오차 ~4e-4 의 세제곱)이 경계에 걸리는지 로컬에서 재서 허용오차를 확정하고 decisions 에 수치를 남긴다 → 확정: rtol = atol = 1e-4(정확한 float32 구현은 여유 26배 이상, 보정 없는 근사 역제곱근은 모든 형태에서 4배 이상 초과; rtol 1e-3 에서는 tiny 가 0.90 으로 경계).
- **cholesky**: `A = G·Gᵀ/n + I`(G uniform[-1,1), float64 로 만들고 float32 로 내림, 대칭 보장). 참조는 float32 입력을 float64 로 올려 `np.linalg.cholesky`. `L` 의 상삼각(대각 위)은 0 이어야 한다(센티널이 남으면 실패).
- **floyd_warshall**: `D[i][j]` = i 에서 j 까지 최단 거리(중간 정점 임의). 완전 그래프, 가중치 uniform[1,100), 대각 0. 참조는 float64 로 표준 k-i-j 갱신.
- **bfs**: 방향 그래프 CSR(spmv_csr 와 같은 배치: 정점 u 의 이웃은 `col_idx[row_ptr[u] .. row_ptr[u+1])`). `dist[v]` = source 에서 v 까지의 최소 간선 수, source 자신은 0, 도달 불가는 −1. 큰 형태는 평균 차수 8 의 무작위 그래프(자기 루프·중복 간선 허용 여부를 명세에 적는다; 고립 정점이 있도록 만든다).
- **fft**: `X[k] = Σ_{j=0}^{n-1} x[j]·exp(−2πi·jk/n)`, 정규화 없음. 입력은 uniform[-1,1)/√n(출력 RMS 가 형태와 무관하게 O(1) 이 되도록; 명세에 사실로 적는다).
- **sort**: 키만 정렬(안정성 무의미). 키에 음수가 있다는 것을 명세의 사실로 적는다.
- **prefix_sum**: 합이 int32 를 넘지 않음(최대 15·2²⁶ < 2³¹)을 명세에 적는다.
- **histogram**: 범위 밖 키는 없다(계약). `hist` 의 초기 내용은 미정의이므로 0 으로 시작한다고 가정하면 안 된다(누적 버그는 `accumulated_into_c` 로 잡힌다). tiny 는 키를 [1, 3) 에서 뽑아 첫 구간과 마지막 구간이 둘 다 빈다(키 범위 밖 구간을 쓰지 않는 구현을 잡는다).
- 큰 형태는 잠정값이다(순차 C 기준 0.05–3 s, 과제당 npz ≤ 약 550 MB). 데스크탑 Linux 에서 `make probe` 뒤 다시 정한다.
- **Halide 지원 범위**: `sort`, `bfs` 는 `unsupported`(halide_auto, halide_manual). 나머지는 L 단계에서 **정확한 더미 구현이 로컬 검증을 통과한 조합만** 켠다. 통과시키지 못한 조합은 이유와 함께 `unsupported` 에 넣는다(규칙: 어떤 (과제, 언어) 조합이든 검증된 더미 `0.md` 가 있어야 실험 대상이다). 결과(2026-11-22): 나머지 10개 과제는 두 Halide 언어 모두 더미가 통과해 `unsupported` 는 `sort`·`bfs` 의 Halide 두 언어뿐이다(`tools/devcheck.py dummies --languages numba,halide_auto,halide_manual,rust`: 44 pass, 4 unsupported).
- 허용오차의 최종값은 각 과제의 `tolerance.yaml` 이 기준이다. 기준선 종류는 `signatures.yaml` 의 `baseline`(`backend: python|c`, `openmp`, python 기준선의 게이트 `gate_blas`·`gate_min_gflops`)이고, 위 표의 "C+OpenMP" 과제(`histogram`, `nbody`, `floyd_warshall`)는 `openmp: true`, `bfs` 는 `openmp: false` 인 C, 나머지 넷은 python 기준선이다(`cholesky` 만 `gate_blas: true`).

과제마다 만드는 파일: `spec.md`, `signatures.yaml`, `tolerance.yaml`, `gen_tests.py`(`harness.testgen` 사용), `baseline/baseline.{py,c}`, `harness/dummy_responses/<task>/{numpy,c_seq,c_openmp}/0.md`(정답; 응답처럼 한두 문장 + 펜스 블록 하나). 검증: `tools/devcheck.py`(§14) 로 기준선과 더미 3개가 tiny·rect·small 에서 pass, large 는 생성 시간·파일 크기·기준선 1회 실행 시간을 재서 보고한다.

## 5. 워커 확장 (G1 → G2a)

### 5.1 CLI

```
PYTHON_EXE -m harness.worker --backend {c,python,numba,halide} --artifact PATH --mode {import,check,bench}
    --tests … --tolerance … --signatures … --out result.json
    [--check-repeats 2] [--warmup 3 --repeats 10 --min-repeats 3 --budget-s 300] [--args-mode positional|keyword]
    [--halide-mode auto|manual] [--halide-autoscheduler Mullapudi2016] [--halide-params JSON] [--threads N]
    [--inspect] [--inspect-dims JSON] [--max-mem-gb 8] [--sanitize]
    # --max-mem-gb 가 0 이하면 RLIMIT_AS 를 걸지 않는다(ASan 실행용; limits.disabled)
    # --sanitize: sanitizer 실행 전용 버퍼 배치(§12, §16)
```

- `--threads` 는 halide auto 의 오토스케줄러 `parallelism` 이자 `backend_detail` 의 기록일 뿐이다. 실제 스레드 수는 언어 설정의 환경변수(`NUMBA_NUM_THREADS`, `HL_NUM_THREADS`, `RAYON_NUM_THREADS`)가 정하고, 그 변수가 빠진 설정은 로더가 거부한다(§8). 워커는 스레드 수를 따로 검사하지 않는다.

- `rust` 산출물은 C ABI 공유 라이브러리이므로 `--backend c` 로 실행한다.
- `--backend numba` 는 python 과 같은 로드·호출 경로다(모듈 임포트 시 명시 서명으로 즉시 컴파일된다). 로드 뒤 `fn` 이 Numba 디스패처이고 `fn.signatures` 가 비어 있지 않아야 한다 — 아니면 `signature_mismatch`/`numba_signature_missing`(지연 컴파일이면 첫 호출 시간에 JIT 가 섞인다).
- `--inspect` (import 모드): 결과 문서에 `inspect` 를 넣는다. `--inspect-dims '{"M": 1024, …}'`(run.py 가 manifest 에서 읽은 **성능 형태**의 dim 값)가 있으면 halide 의 `inspect` 는 그 크기로 만든 파이프라인에서 모은다(빌드·오토스케줄·`print_loop_nest` 만, realize 없음 — 입력 데이터가 필요 없다). 오토스케줄러의 스케줄은 estimate 에 따라 달라지므로(실측: gemm 은 (7,5,3) 에서 벡터화가 없고 1024³ 에서 붙는다) 요약은 측정되는 스케줄을 기술해야 한다. 옵션이 없으면 첫 정확성 테스트 크기를 쓴다. numba: `{"signatures": [str…], "parallel_diagnostics": str(level 4, ≤8000자; stdout 을 가로채 얻는다), "threading_layer": str|None, "num_threads": int}`. halide: `{"loop_nest": str, "schedule_source": str, "autoscheduler": str|None, "target": str, "autoschedule_ms": float|None, "jit_ms": float, "dims": {…}}` — `dims` 는 기술한 파이프라인을 만든 dim 값이고, `--inspect-dims` 경로에서는 JIT 를 하지 않으므로 `jit_ms` 는 첫 정확성 테스트 빌드의 값이다. 수집 실패는 `inspect_error` 문자열로 남기고 워커는 계속한다.
- 구현 세부: 수집을 시작하기 전에 문서에 `inspect.inspect_error = "inspect did not finish: …"` 와 `in_progress = {"test": "inspect", "call_idx": null}` 를 미리 써 둔다. 성능 형태의 빌드가 워커를 죽이거나 시간 상한에 걸려도 문서에는 이미 `load_ok=true` 가 있어, run.py 는 이것을 커널의 빌드 실패가 아니라 관찰 실패로 다룬다(§10.2). Numba 의 스레딩 계층은 커널보다 먼저 띄우므로 로드된 numba 커널의 `threading_layer` 는 직렬 커널에서도 그 프로세스의 계층 이름(문자열)이다.

### 5.2 호출 방식

| backend | positional | keyword |
|---|---|---|
| c | `fn(*c_args)` | `fn(byref(ArgsStruct(...)))` — `ctypes.Structure` 를 `args` 순서로 만든다(포인터 필드는 `POINTER(c_float|c_int32)`, dim 은 `c_int`) |
| python·numba | `fn(*py_args)` | `fn(**dict(zip(names, py_args)))` |
| halide | §5.3 | 빌드 함수 호출만 키워드로 |

### 5.3 Halide 어댑터 (`harness/kernels/halide_pipe.py`)

LLM 모듈(`kernel.py`)은 최상위에 **파이프라인 빌드 함수** `def {fn}({in 배열…, dim…}):` 를 정의한다(출력 인자 없음).

- 배열 인자는 `hl.ImageParam(hl.Float(32) | hl.Int(32), ndim, name)`, dim 인자는 그 호출의 **실제 크기인 Python `int`**. 반환값은 출력과 차원 수·타입이 같은 `hl.Func`.
- **좌표 규칙(프롬프트에 그대로 쓴다)**: 버퍼는 `hl.Buffer(numpy_array)`(기본 `reverse_axes=True`)로 만든다. 형태 (D0, D1) 의 행우선 배열에서 행 i0·열 i1 원소는 Halide 에서 `X[i1, i0]` 이다(마지막 축이 첫 좌표). 1-D 는 `x[i]`. 출력 `Func` 도 같은 규칙으로 정의한다.
- 워커는 테스트(형태)마다 새 파이프라인을 만든다: ImageParam 생성 → 빌드 함수 호출 → 반환 타입·차원 검사(틀리면 `signature_mismatch`) → `halide_mode == "auto"` 면 입력·출력 estimate 를 실제 크기로 설정하고 오토스케줄러 적용 → `hl.Pipeline(f).compile_jit(hl.get_host_target())` → 호출마다 `pipeline.realize(hl.Buffer(out_view_nd))`(출력 버퍼에 직접 쓴다). 파이프라인 구성·스케줄·JIT 시간은 측정에 넣지 않는다(`build_ms` 로 테스트 문서에 기록).
- 오토스케줄러 플러그인은 이름으로 로드되지 않는다(실측). `Path(halide.__file__).parent` 아래 `**/*autoschedule_<name소문자>*.{so,dll,dylib}` 를 찾아 절대 경로로 `hl.load_plugin` 한다(Windows 는 `os.add_dll_directory`). 못 찾거나 로드 실패 → `load_ok=false`, classification `compile_error`, detail `halide_autoscheduler_unavailable:<name>` — run.py 의 도구 점검(§10.6)이 먼저 잡아 모델 오류로 기록되지 않게 한다.
- `AutoschedulerParams(name, {"parallelism": str(threads), **halide_params})`. 스레드 수는 환경변수 `HL_NUM_THREADS`.
- import 모드에는 `--tests <첫 정확성 테스트>` 를 준다: 그 크기로 빌드·JIT 까지 해 본다(실패 → `compile_error`, `error_stage=build`). check/bench 에서 형태별 빌드가 예외를 던지면 그 테스트의 `runtime_crash`.
- `print_loop_nest()` 는 stderr 에 쓴다 → fd 2 를 임시 파일로 돌려 `inspect.loop_nest` 로 가져온다. stdout 에는 아무것도 쓰지 않는다.
- 구현 세부: estimate 는 빌드 함수가 돌아온 뒤에 준다(빌드 함수가 스스로 준 estimate 가 있어도 실제 크기가 남는다). 출력 `hl.Buffer` 는 형태마다 한 번 만들고 호출은 `pipeline.realize(buffer)` 뿐이다. 오토스케줄러·JIT 실패의 detail 은 `autoscheduler <이름> failed: HalideError: …` / `JIT compilation failed: …` 꼴이다. 반환값 검사 실패(타입이 `hl.Func` 가 아님, 정의 없음, Tuple, 차원 수·원소 타입 불일치)는 `signature_mismatch` 이고 detail 은 예: `dot returned an hl.Func with 0 dimension(s); the output out has 1`(출력 형태가 `[1]` 인 과제는 길이 1 의 1-D `Func` 를 돌려줘야 한다). check·bench 의 형태별 빌드 실패는 그 테스트의 `runtime_crash` 이고, 예외 없이 난 실패(반환값 검사)는 예외 기록의 `type` 자리에 `signature_mismatch`, `traceback` 은 빈 문자열이다.
- 하네스가 부른 컴파일 단계(파이프라인 구성·오토스케줄·JIT·`print_loop_nest`)의 Python `print` 출력(Halide 의 컴파일 경고)은 stderr 로 돌린다. 빌드 함수(LLM 코드) 호출 중의 출력과 realize 중의 `hl.print` 는 커널의 출력이라 stdout 으로 나가고 `contract_violation` `stdout_output` 으로 관찰된다.
- 백엔드 준비(`harness.kernels.open_backend`: 도구 임포트, numba 스레딩 계층 시작, 오토스케줄러 플러그인 로드와 시험 적용)는 LLM 코드를 실행하기 전의 일이다. 그 구간에서 프로세스가 죽으면 도구·설정 문제이므로 워커는 그동안 결과 문서에 `worker_error` 를 미리 써 둔다(`harness_only`). Halide 22.1 은 틀린 오토스케줄러 매개변수에 예외 대신 프로세스 중단으로 끝나기 때문이다(실측: Windows 종료 코드 0xC0000409). run.py 는 이것을 행이 아닌 하네스 오류로 보고, 같은 시험을 run 시작 전에 미리 돌려 막는다(§10.6).
- halide_manual 에서 스케줄 없는 다단계 파이프라인은 전부 인라인돼 JIT 가 폭발한다(실측: 스텝마다 `Func` 하나인 jacobi2d 는 식이 5^T 로 커지고 512²·T=8 에서 9분 넘게 끝나지 않았다). 더미·픽스처는 단계마다 `compute_root` 를 쓰고, 실험에서 이런 응답은 build(import) 또는 check 단계의 `timeout` 으로 기록된다.

### 5.4 정수 출력·센티널

`Buffers.ref` 는 출력 dtype 이 int32 면 int32 로 읽는다. 가드·센티널 채움은 dtype 에 맞춘다(기존 `alloc_aligned` 그대로).

### 5.5 그 밖

- 결과 문서의 기존 키는 유지한다. 추가 키: `backend_detail`(받은 백엔드 설정; 예: `{"halide_mode": "auto"}`), `inspect`(§5.1), `build_ms`(halide; check 는 테스트별, import·bench 는 최상위), `sanitize`(`--sanitize` 여부). halide 의 import 문서에는 빌드해 본 형태의 `test`·`dims` 도 있다. 로드 실패가 예외에서 왔으면 `exception = {type, message, traceback}`.
- check 문서의 호출 기록(`runs`)에는 `n_fail`·`n_total` 이 있다(수정 루프의 진단 "k of n elements outside tolerance" 가 이 값을 쓴다). check 문서의 `inputs_mutated`·`oob_write` 는 호출을 하나도 검사하지 못한 동안 `null` 이고, run.py 는 그 경우 행의 열을 NA 로 둔다.
- `runtime_crash` 의 설명(호출 기록의 `pattern` → 행의 `error_detail`)은 한 줄이다: 여러 줄 예외 메시지는 줄을 공백으로 이어 500자까지 쓴다(예외 기록의 `message` 는 원문). Halide 메시지는 줄바꿈으로 끝나고 Numba 메시지는 원인이 뒷줄에 있기 때문이다. python 백엔드의 한 줄 메시지는 전과 같다.
- 워커가 여는 LLM 모듈은 여전히 이 프로세스 하나뿐이다. numba 의 `cache=True` 는 막지 않는다(관찰만).
- 커널을 로드하고 부르는 방법은 `harness/kernels/` 의 백엔드 모듈(`cabi`, `pyfunc`, `halide_pipe`)이 감추고, 워커는 프로세스 수준의 일(제한, 버퍼, 결과 문서, 측정 루프)만 한다. 인터페이스: `open_backend(backend, options)` → `load(artifact, spec)` → `KernelHandle.prepare(dims, arrays)` / `invoke()` / `inspect(dims)`.

### 5.6 비-Linux 임포트 안전성 (개발 편의; G1)

- `procutil`: `SIGHUP` 이 없으면 `TERMINATE_SIGNALS` 에서 뺀다. `run_isolated` 는 `os.name != "posix"` 에서 `preexec_fn`·`start_new_session` 없이 띄우고 타임아웃·예외 시 `proc.kill()` 한다(프로세스 그룹 정리 없음 — 측정용이 아니다).
- `worker`: `resource` 임포트 실패 허용(`apply_memory_limit` 은 `{"rlimit_as_bytes": None, "unsupported": true}`), `prctl` 실패는 경고, `os.getloadavg`/`sched_getaffinity` 없음 허용.
- POSIX 전용 테스트는 `tests/_support.py` 의 `requires_linux`, `requires_tool("gcc")` 로 건너뛴다. Linux 동작은 한 줄도 바꾸지 않는다.

## 6. 빌드 백엔드 (G2b)

공통 결과형 `BuildResult`(기존) + 선택 필드 `extra: dict[str, Any] = {}`(백엔드별 관찰값; `build.json` 에 저장).

```python
class ToolchainMissing(RuntimeError)        # harness.build.c 에 정의. 컴파일러·cargo 실행 파일이 없을 때(OSError) — 모델 오류로 기록하면 안 된다
def build_c(code, out_dir, bcfg, *, openmp, launch_prefix, symbol="gemm", args_header: str | None = None,
            extra_flags: list[str] | None = None, artifact_name: str = "kernel.so", write_json: bool = True) -> BuildResult
    # args_header 가 있으면 out_dir/kernel_args.h 로 쓴다(키워드 모드). extra_flags 는 cflags 뒤에 붙는다(ASan 빌드).
    # extra_flags 에 `-fsanitize=` 가 있으면 링크 플래그에서 `-Wl,--no-undefined` 를 뺀다(sanitizer 런타임 심벌은 로드 시점에
    # LD_PRELOAD 된 libasan 이 채울 수 있다; 미정의 심벌 검사는 일반 빌드가 이미 했다).
def build_python(code, out_dir) -> BuildResult                       # 기존
def build_numba(code, out_dir, *, function: str, expected_signature: str) -> BuildResult
    # compile() 구문 검사 + AST: 최상위 FunctionDef {function} 의 데코레이터가 njit/jit(nopython=True) 호출이고 첫 위치 인자
    # (또는 signature= / Numba 의 실제 매개변수 이름 signature_or_function=)로 서명을 준다.
    # 없으면 error_kind="signature_mismatch", detail "numba_signature_missing". 최상위 def 가 없으면 "function_not_found".
    # extra = {"decorator": "njit", "signature_literal": str|None, "signature_matches_expected": bool|None,
    #          "options": {"parallel": bool|None, "fastmath": …, "cache": …, "boundscheck": …, "nogil": …}, "uses_prange": bool}
def build_halide(code, out_dir, *, function: str, halide_mode: str) -> BuildResult
    # compile() + AST. extra = {"schedule_calls": [메서드 이름…], "imports_halide": bool}. halide_mode == "auto" 인데 스케줄 호출이
    # 있어도 빌드는 ok — run.py 가 contract_violation `schedule_in_halide_auto` 로 관찰만 한다.
def build_rust(code, out_dir, rcfg, *, launch_prefix, symbol, args_module: str | None = None, target_dir: Path | None = None) -> BuildResult
    # out_dir/{Cargo.toml,Cargo.lock,src/lib.rs[,src/kernel_args.rs]} 를 쓰고
    # `cargo build --release --offline --locked`(RUSTFLAGS, CARGO_TARGET_DIR=target_dir) → 산출물을 out_dir/kernel.so 로 복사
    # (Windows 는 kernel.dll) → C 와 같은 `nm` 심벌 검사(nm 이 없으면 건너뛰고 extra 에 기록).
    # extra = {"rustflags": [...], "symbol_check": "nm" | "skipped:not_elf" | "skipped:nm_not_found"}
def probe_toolchain(rcfg, target_dir: Path, *, launch_prefix: list[str]) -> None
    # (구현 때 추가) 템플릿의 자리표시 크레이트를 build_rust 와 같은 명령·환경·잠금으로 한 번 빌드한다. 실패는 전부 ToolchainMissing.
```

- numba 빌드는 소스의 꼴만 본다(코드를 임포트하지 않는다). 실측(Numba 0.68.1): `signature=` 키워드는 Numba 의 매개변수가 아니라 조용히 무시되고 지연 컴파일되며, 문자열의 튜플은 임포트 때 TypeError 다. 이런 커널은 빌드를 통과하고 워커가 판정한다(서명이 빈 디스패처 → `signature_mismatch`/`numba_signature_missing`, 임포트 예외 → `compile_error`). `signature_matches_expected` 는 관찰값이라 글자가 달라도(`f4[::1]`) 빌드를 실패시키지 않는다 — 프롬프트(`prompts/lang/numba.md`)는 보여 준 서명 문자열을 그대로 쓰라고 하므로 프롬프트가 검사보다 엄격하다.
- 빌드 실패의 판정(구현 세부): `nm -D --defined-only` 에서 내보낸 심벌로 인정하는 종류는 `T`/`W`/`i`(`i` = GNU 간접 함수: `target_clones`·ifunc; 대문자 `I` 는 아니다)다. C 와 Rust 가 같은 규칙이고, 없으면 `signature_mismatch`/`symbol <fn> not exported`.
- Rust 빌드가 `ToolchainMissing`(행 없음, exit 7)인 경우: cargo 를 시작할 수 없을 때, cargo 가 커널 크레이트를 컴파일하기 전에 실패하거나 시간을 다 썼을 때(의존성 캐시 없음, 잠금 파일 불일치, 의존성 컴파일 실패, 링커 없음), rc=0 인데 라이브러리가 제자리에 없을 때, `nm` 이 실패했을 때, target 잠금을 제시간에 얻지 못했을 때. 로그에 ``error: could not compile `kernel` `` 이 있을 때만 `compile_error`(detail = rustc 오류 첫 3개)이고, `Compiling kernel` 뒤의 시간 초과만 `BuildResult.error_kind = "timeout"` 이다(C 와 같이 run.py 가 행에 `compile_error`/`compile_timeout` 으로 쓴다).
- 공유 target 디렉터리에서는 cargo 가 소스 mtime 으로만 신선도를 판단해 다른 샘플의 라이브러리를 복사할 수 있다(실측). 그래서 빌드 전에 항상 `cargo clean --release -p kernel` 로 커널 패키지만 지우고, 정리 → 빌드 → 산출물 읽기 전체를 `<target>/.harness-build.lock` 의 프로세스 간 배타 잠금 아래에서 한다(대기 상한 = 2 × (정리 + 빌드 시간 초과)).
- cargo·rustc 의 환경은 `protocol.toolchain_env()` 다: POSIX 는 화이트리스트 환경 + `CARGO_HOME`·`RUSTUP_HOME`·`RUSTUP_TOOLCHAIN`, 개발용 Windows 는 현재 환경. 도구 점검과 환경 지문의 rustc 버전도 같은 환경에서 읽는다. `offline: false` 설정이면 `--offline` 을 뺀다.

- Rust 템플릿 `harness/build/rust_template/{Cargo.toml,Cargo.lock}`: 패키지 `kernel`, edition 2021, `crate-type = ["cdylib"]`, 의존성은 `rayon`(정확한 버전 고정) 하나, `[profile.release] opt-level = 3, codegen-units = 1, lto = "fat", panic = "abort", debug = false`. `Cargo.lock` 을 저장소에 넣는다. 의존성 캐시는 `cargo fetch --locked --manifest-path harness/build/rust_template/Cargo.toml`(온라인 1회)로 받고 이후 빌드는 오프라인이다. `Cargo.lock` 이 v4 형식이라 cargo 1.78 이상이 필요하다. Makefile 의 `rust-setup` 타깃으로 두기로 했다(이 문서를 고친 시점의 Makefile 에는 아직 없다). 캐시가 없으면 `build_rust`·`probe_toolchain` 이 이 명령을 안내하는 `ToolchainMissing` 을 던진다.
- `target_dir` 는 run 당 하나(`results/built/<run_id>/_cargo_target`)를 run.py 가 넘겨 의존성 빌드를 재사용한다.
- `panic = "abort"` 이므로 패닉은 SIGABRT → `runtime_crash`.
- `SCHEDULE_METHODS`(halide 스케줄 호출 이름 집합)는 `harness/build/halide.py` 에 두고 perf_summary 가 임포트해 쓴다. `harness.perf_summary.DEFAULT_SCHEDULE_METHODS`(50개)와 **같은 집합**이어야 한다(perf_summary 의 테스트가 비교한다): align_bounds, align_extent, align_storage, allow_race_conditions, always_partition, always_partition_all, async_, atomic, bound, bound_extent, bound_storage, clone_in, compute_at, compute_inline, compute_root, compute_with, copy_to_device, copy_to_host, fold_storage, fuse, gpu, gpu_blocks, gpu_lanes, gpu_single_thread, gpu_threads, gpu_tile, hexagon, hoist_storage, hoist_storage_root, in_, memoize, never_partition, never_partition_all, parallel, partition, prefetch, rename, reorder, reorder_storage, rfactor, ring_buffer, serial, specialize, split, store_at, store_in, store_root, tile, unroll, vectorize.

## 7. 추출 확장 (G2b)

```python
def extract(response_text, *, backend, signature_spec, args_mode="positional", work_dir=None, bundle: dict | None = None,
            cc: str | None = None) -> ExtractResult
    # cc (구현 때 추가): C 시그니처 검사의 전처리기. run.py·devcheck 는 설정의 build.cc 를 넘긴다. None 이면 gcc.
```

- `backend ∈ {python, numba, halide, c, rust}`. 펜스 태그: python·numba·halide → {python, py, python3}, c → {c}, rust → {rust, rs}.
- `bundle` 은 `sigs.signature_bundle(...)` 결과(run.py 가 넘긴다). 있으면 기대 시그니처·매개변수는 `bundle` 에서 읽고, 없으면 기존 방식(`signature_spec`)으로 동작한다(하위 호환: 기존 테스트 그대로 통과).
- numba: python 검사와 같다(데코레이터 검사는 빌드 단계). halide: 매개변수 = `halide_params`. keyword 모드는 python 계열 모두 kwonly 검사.
- c keyword: 정의가 `void {fn}(const {fn}_args* args)` 꼴(매개변수 1개, 구조체 포인터, 이름 자유 → 이름이 `args` 가 아니면 `signature_exact=False`). 구조체를 스스로 다시 정의했는지는 보지 않는다(헤더와 충돌하면 컴파일 오류).
- rust: 주석 제거 뒤 `pub (unsafe )?extern "C" fn {fn}(...)` 정의를 찾는다. 없으면 `function_not_found`; 직전 속성에 `#[no_mangle]`/`#[unsafe(no_mangle)]` 가 없으면 `no_mangle_missing`(실패); 매개변수 개수·이름·타입(`*const f32` 등, 공백 무시) 불일치는 `param_mismatch:…`; `unsafe` 가 없거나 반환형 `-> ()` 표기는 `signature_exact=False` 일 뿐이다.
- 새 실패 사유 문자열: `no_mangle_missing`. 그 밖의 detail 규칙은 그대로다. 실패 사유(`ok=False`) 전체는 `no_code_block`, `function_not_found`, `param_mismatch:…`(python 의 `async def` 는 `param_mismatch:async_def`), `static_linkage`(C), `no_mangle_missing`(Rust)이다.
- ABI 가 같은 다른 철자는 불일치가 아니고 `signature_exact=False` 다(구현 때 추가): C 의 `int` 자리의 `signed`, `signed int`, `int signed`, `<stdint.h>` 의 `int32_t`(`C_INT_ALIASES`; 크기·부호가 다른 `unsigned`·`long`·`int64_t`·`size_t` 는 그대로 불일치), Rust 의 `c_int`·`c_float`(접두 없음 또는 `std::ffi::`·`core::ffi::`·`std::os::raw::`·`libc::`; `*const`/`*mut` 포인터 포함). `__int32_t` 와 앞에 `::` 가 붙은 절대 경로(`::std::ffi::c_int`)는 목록에 없어 `param_mismatch` 다.

## 8. 설정 확장 (G3)

`config/experiment.yaml` 에 추가(모두 선택 키 — 없으면 아래 기본값; 기존 파일도 그대로 읽힌다):

```yaml
languages:
  numba:         {backend: numba,  prompt: prompts/lang/numba.md,         threads: measurement, env: {NUMBA_NUM_THREADS: "${threads}", NUMBA_THREADING_LAYER: omp}}
  halide_auto:   {backend: halide, prompt: prompts/lang/halide_auto.md,   threads: measurement, halide_mode: auto,   env: {HL_NUM_THREADS: "${threads}"}}
  halide_manual: {backend: halide, prompt: prompts/lang/halide_manual.md, threads: measurement, halide_mode: manual, env: {HL_NUM_THREADS: "${threads}"}}
  rust:          {backend: rust,   prompt: prompts/lang/rust.md,          threads: measurement, env: {RAYON_NUM_THREADS: "${threads}"}}
build:
  rust: {cargo: cargo, rustflags: ["-C", "target-cpu=native"], build_timeout_s: 600, offline: true}
  sanitize_flags: ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]
halide: {autoscheduler: Mullapudi2016, autoscheduler_params: {}}
safety: {enabled: false, timeout_s: 300}          # ASan/UBSan (C 백엔드만). 실험 때 켠다
lint:   {enabled: true, plugin: plugins/lint/run_lint.py, timeout_s: 60}
perf_summary: {enabled: true}
```

- `run.languages` 기본값은 `[numpy, c_seq, c_openmp]` 그대로 둔다(M2a 실제 실행 대기). `run.tasks` 에 새 과제 8개를 더한다(지금 12개). `run.max_repair_turns` 는 아직 M1 값 `0` 이라 `harness.repair` 를 `--max-turns` 없이 부르면 아무것도 하지 않는다(M4 에서 정한다).
- `measurement.max_settle_s`(구현 때 추가, 기본 `0` = 기다리지 않음): 동적 규약 검사 전에 1분 부하가 내려가길 기다리는 상한(초). `docs/protocol.md` §9.
- `LanguageCfg` 추가: `env: dict[str, str]`(값의 `${threads}` 는 로더가 정수로 치환), `halide_mode: str | None`. `BACKENDS = ("python", "c", "numba", "halide", "rust")`. halide 백엔드는 `halide_mode ∈ {auto, manual}` 필수(다른 백엔드에 쓰면 거부).
- 로더 규칙(구현 때 추가; `harness/config.py`): numba·halide·rust 백엔드의 언어는 `env` 에 각각 `NUMBA_NUM_THREADS`·`HL_NUM_THREADS`·`RAYON_NUM_THREADS` 가 그 언어의 스레드 수(`"${threads}"`)로 있어야 하고, 없거나 다르면 `ConfigError` 다(`THREAD_ENV`; 이 런타임들은 `OMP_NUM_THREADS` 를 읽지 않는다). `env` 값의 자리표시는 `${threads}` 만 허용한다. `halide.autoscheduler_params` 의 `parallelism` 은 예약어다(하네스가 스레드 수로 넣는다; 설정에 쓰면 거부).
- 선택 키의 기본값은 `harness.protocol.OPTION_DEFAULTS` 에도 있고 `protocol.config_option(obj, "lint.enabled")` 처럼 읽는다: 이 키들을 모르던 때의 `Config` 객체와 설정 없이 도는 프로브(`cfg=None`)도 같은 값을 얻는다. 두 곳의 기본값이 같은지는 테스트가 확인한다.
- 새 dataclass: `RustCfg`, `HalideCfg`, `SafetyCfg`, `LintCfg`, `PerfSummaryCfg`; `BuildCfg.rust`, `BuildCfg.sanitize_flags`; `Config.halide/safety/lint/perf_summary`.
- 조건 덮어쓰기 CLI(§9.1)는 `load_config(path, overrides={"args": …, "grammar_hint": …, "feedback": …})` 로 들어간다.

## 9. 생성·조건·수정 루프 (G3)

### 9.1 `generate.py`

- 새 옵션: `--args positional|keyword`, `--grammar-hint on|off`, `--feedback compiler|compiler+lint`(설정의 `run.conditions` 를 덮어쓴다; raw 의 `conditions` 와 `cond_tag` 에 반영).
- 과제가 그 언어를 지원하지 않으면(`TaskSpec.supports`) 건너뛰고 한 줄 알린다. 끝에 (과제 × 언어) 표를 낸다(`call`/`skip`/`unsupported`).
- `--tasks` 는 설정의 `run.tasks` 의 부분집합이어야 한다(구현 때 추가). 밖의 과제는 `harness.run`·`harness.repair` 가 처리하지 않으므로 호출 전에 exit 1 로 거절한다.
- `seeded_from` 이 있는 raw(§9.3 의 시드 복사본)는 `generate` 가 다시 호출하지 않는다(경고만).
- 시그니처·설명은 `sigs.signature_bundle`. 템플릿 변수(모든 언어 템플릿에 전부 제공): 기존 것 + `numba_signature`, `numba_version`, `halide_version`, `halide_autoscheduler`, `rustc_version`, `cargo_command`, `rust_profile`, `args_note`, `grammar_hint`. 구현이 더한 셋: `numba_threading_layer`(언어 `env` 의 `NUMBA_THREADING_LAYER`; 없으면 그 템플릿 렌더링이 `ConfigError`), `rayon_version`(Rust 템플릿 `Cargo.toml` 의 고정 버전), `halide_coord_example`(그 과제의 입력 중 차원이 가장 많은 첫 배열로 만든 좌표 예시 한 문장). 값은 그 템플릿이 실제로 쓰는 변수만 구한다. rustc 버전은 machine.yaml 에 없으면 `protocol.toolchain_env()` 환경에서 읽는다.
- **기존 3개 템플릿의 바이트 동일성**: `prompts/lang/{numpy,c_seq,c_openmp}.md` 에는 변수 두 개만 넣는다 — `${param_docs}` 바로 뒤에 `${args_note}`, 마지막 `---` 줄 바로 앞에 `${grammar_hint}`. positional·hint off 에서 두 값은 빈 문자열이어서 렌더링 결과가 지금과 바이트 단위로 같아야 한다(테스트: 기존 4개 과제 × 3개 언어의 `prompt_hash` 가 변경 전과 같다 — 변경 전 해시는 작업 시작 시 계산해 테스트에 상수로 넣는다. machine.yaml 은 저장소의 것을 쓴다).
- `args_note`(keyword 모드에서만, 개행 문자 두 개 = 빈 줄 하나로 시작; "빈 줄 두 개"를 이렇게 해석했다): python 계열 — 키워드 전용 호출이라는 사실 한 문단; c — `kernel_args.h` 전문을 코드 블록으로 보여 주고 `#include "kernel_args.h"` 를 쓰라는 문단; rust — `src/kernel_args.rs` 전문과 `mod kernel_args;` 안내.
- `grammar_hint`(on 일 때): `"## Language notes\n\n" + prompts/hints/<language>.md 내용 + "\n\n"`. 그 언어의 힌트 파일이 없으면 그 조합은 건너뛴다(경고). 힌트는 100줄 이하, 과제와 무관한 문법·API 사실만 쓴다(성능 조언 금지 — 성능 결정은 관찰 대상이다).
- 새 언어 템플릿은 기존 템플릿과 같은 절 구성(Environment / Function to implement / Execution environment / Output format)·같은 정보량으로 쓴다. numba: 데코레이터의 서명 문자열은 그대로 두고 키워드 옵션(예: 병렬·fastmath)은 자유라고 사실대로 적는다. halide_auto: 알고리즘만 쓰고 스케줄 지시는 쓰지 않으며 하네스가 오토스케줄러 `${halide_autoscheduler}` 를 적용한다는 사실. halide_manual: 스케줄을 같은 함수 안에서 직접 쓴다는 사실. rust: `cargo` 명령·프로파일·사용 가능한 크레이트(std, rayon)·패닉은 abort.

### 9.2 `llm_client.py`

```python
def complete(self, system: str, user: str, *, messages: list[dict[str, str]] | None = None, dummy_key: tuple | None = None) -> LLMResponse
```

- `messages`(user/assistant 번갈아, 마지막은 user)가 있으면 그것을 보낸다(`user` 는 무시; anthropic 은 `messages=` 그대로, openai_compat 은 system 뒤에 이어 붙인다). 수정 턴에서 이전 응답의 thinking 블록은 다시 보내지 않는다(텍스트만).
- dummy: `dummy_key = (task, language, sample_idx)` 또는 `(task, language, sample_idx, turn)`. turn ≥ 1 이면 `<k>.turn<t>.md` 를 찾고, 없으면 `<k>.md` 를 준다(= 같은 실패가 반복된다). k 를 고를 때의 파일 수에는 숫자 이름의 `<k>.md` 만 센다. 더미 파일은 utf-8-sig 로 읽는다.
- 구현 세부: `messages` 가 user/assistant 번갈아·user 로 끝나지 않으면 API 오류로 기록하지 않고 `ValueError` 를 올린다(하네스 버그를 모델·API 오류로 남기지 않는다). SDK 가 없으면 호출 전에 `LLMClientError`(설정 오류)다.

### 9.3 `repair.py` (수정 루프)

```
PYTHON_EXE -m harness.repair --run-id ID [--config] [--max-turns N] [--tasks …] [--languages …] [--samples …]
    [--feedback compiler|compiler+lint] [--seed-from-feedback compiler|compiler+lint] [--yes] [--skip-bench]
```

- 전제: 같은 run 의 turn 0 이 `harness.generate` + `harness.run` 으로 처리돼 있다. 직전 턴의 `result.json` 이 그 raw 의 것이 아니면(`raw_sha256` 불일치·없음) 그 샘플은 경고하고 건너뛰고, 처리된 샘플이 하나도 없으면 exit 1 이다.
- `--provider`/`--model` 옵션은 없다(계약대로). 설정·`LLM_PROVIDER`·`LLM_MODEL` 로 정해진 (provider, model) 이 턴 0 raw 와 다르면 호출 없이 exit 1 이다 — dummy run 에 실제 호출이 붙거나 한 대화에 두 모델이 섞이지 않게 한다. dummy run 의 수정은 `LLM_PROVIDER=dummy python -m harness.repair …` 로 돌린다.
- 범위(구현 때 추가): 한 번의 실행은 `conditions.feedback` 이 `--feedback`(없으면 설정값)과 같은 샘플만 다룬다. `args`·`grammar_hint` 는 raw 의 것을 따른다.
- 다시 돌리면 이미 응답이 있는 턴은 다시 묻지 않고 이어서 한다. 그 턴의 raw 가 생성 실패(API 오류·절단·거절)거나 저장된 대화의 앞부분이 지금의 앞 턴과 다르면 그 턴부터 다시 호출하고 이전 파일은 `.prev<N>.json` 으로 보관한다. 앞 턴의 지금 행이 수정 대상이 아니면(`harness.run --force` 의 재분류로 통과 등) 그 샘플은 더 잇지 않는다. 직전 턴의 응답이 비어 있는 샘플도 수정하지 않는다(빈 assistant 턴은 API 가 거부한다; 요약의 `skipped_empty_reply`).
- 대상: 직전 턴 행이 `valid_sample=1 ∧ pass=0 ∧ error_stage ≠ generate` 인 샘플. bench timeout(`pass=1`)과 거절·API 오류·절단은 대상이 아니다.
- 턴 t(1..N, N ≤ `run.max_repair_turns` 또는 `--max-turns`): 대상 수를 보여 주고 확인(`generate` 와 같은 게이트) → 샘플마다 메시지 구성 → 호출 → raw 저장 → `run.process_sample` 로 처리 → 다음 턴. 대상이 없으면 끝.
- 메시지: `[user: turn 0 프롬프트(raw 의 prompt.user), assistant: turn 0 응답, user: 수정 요청 1, assistant: turn 1 응답, …, user: 수정 요청 t]`. system 은 `prompts/system.md` 그대로.
- 수정 요청 = `prompts/repair.md` 렌더링. 변수: `${stage}`, `${error_type}`, `${diagnostics}`, `${lint_section}`. `feedback=compiler` 면 `lint_section` 은 빈 문자열, `compiler+lint` 면 `"\n## Static analysis findings\n\n- line 12: <message>\n…\n"`(린트가 돌았고 결과가 비면 `"\n## Static analysis findings\n\nNone.\n"`; 린트가 돌지 않았으면 — 직전 턴 `stages["lint"]` 의 `ok` 가 거짓이거나 `tool == "none"` — 본문이 `Not available.`). 두 조건의 프롬프트는 이 절만 다르다. 린트 메시지는 직전 턴 `result.json` 의 `stages["lint"]["findings"]` 에서 `line <n>: <message>` 꼴로 최대 20개(`safety.lint_messages` 와 같은 형식).
- `diagnostics`(영어, 사실만, 4000자 상한; `repair.build_diagnostics(row: dict, stages: dict, sample_dir: Path) -> str`):
  - extract: 추출 실패 사유를 풀어 쓴 한 문단(예: 기대 시그니처와 받은 시그니처).
  - build: 컴파일러·cargo 출력에서 `error` 가 처음 나온 줄부터(없으면 꼬리), 또는 임포트 예외 한 줄. 경로는 `kernel.c` 등 파일 이름만 남긴다.
  - check/bench: 테스트별로 `shape (M=…, N=…): <outcome>` — outcome 은 `passed` / `incorrect output (max abs error 3.1e+00, 35 of 35 elements outside tolerance)` / `non-finite values in the output` / `crashed: signal SIGSEGV` / `raised ValueError: …`(트레이스백 끝 10줄) / `results differ between two identical calls` / `timed out after 180 s` / `modified a read-only input` / `wrote outside the output buffer`. **하네스 내부 패턴 이름(`transposed_output`, `accumulated_into_c` 등)과 참조값은 주지 않는다** — 컴파일러·테스트 러너가 줄 법한 정보까지만.
  - 구현 세부: 컴파일러·cargo 출력은 `error` 또는 `undefined reference` 가 처음 나온 줄부터(없으면 꼬리 30줄). 임포트·로드 실패는 예외 한 줄 + 트레이스백 끝 10줄. 워커가 중간에 끝나면 그때 돌던 형태에 `crashed: signal …`/`timed out after … s` 를 붙인다. "k of n elements outside tolerance" 는 워커가 그 수(`n_fail`, `n_total`)를 기록했을 때만 쓴다. 문장은 패턴 이름이 아니라 지표(`max_tol_ratio`, `nan_inf`, `inputs_mutated`, `oob_write`, `nondeterministic`)로 만들고, `harness/classify.py` 의 패턴 이름이 진단에 나오지 않음을 테스트한다.
  - `${error_type}` 은 테스트 러너가 관찰한 수준의 말로 낮춘다: `wrong_output`·`arg_order`·`numeric_instability` → `incorrect output`, `race_condition` → `non-deterministic output`. sanitizer 가 재분류한 `memory_error` 는 `error_detail` 의 `; was <type>/` 에서 재분류 전 판정으로 되돌린다(`safety.enabled` 여부로 수정 요청이 달라지지 않게).
  - 판단이 남은 두 가지(`docs/decisions.md` 참조): 크래시의 진단은 `crashed: signal SIGABRT` 같은 한 구절뿐이라 Rust 패닉이 stderr 에 남기는 이유·위치는 전달되지 않는다(Python 계열 예외는 메시지와 트레이스백 끝이 전달된다). 트레이스백 끝 10줄에는 커널의 트레이스백이 짧으면 하네스 프레임(`worker.py`, `halide_pipe.py` 의 파일 이름)이 섞일 수 있다.
- raw: `…/sample_<i>.turn<t>.json`. 키는 turn 0 과 같고 `turn: t`, `prompt: {system, user(턴 0 원문), messages}`, `prompt_hash = sha256(system + "\n\x00\n" + json.dumps(messages, ensure_ascii=False))[:16]`, `repair: {parent_raw, feedback, diagnostics, lint_included, n_lint_findings}` 추가. `lint_included` 는 요청에 린트 절이 들어갔는가(`compiler+lint` 면 `Not available.` 이어도 true), `n_lint_findings` 는 실은 메시지 수다(린트가 돌았고 없으면 0, 돌지 않았거나 절이 없으면 null).
- `--seed-from-feedback compiler --feedback compiler+lint`: `fb-compiler` 조건의 turn 0 raw 를 `fb-compiler+lint` 조건 디렉터리로 복사하고 끝낸다(`conditions.feedback` 을 바꾸고 `seeded_from`·`seeded_sha256`(복사할 때 원본 파일의 sha256) 기록; API 호출 없음). 같은 실패 샘플에서 두 피드백 조건이 갈라지는 **대응 비교**용이다. 복사한 turn 0 은 `harness.run` 으로 다시 처리한다(행 키가 다르다). 생성에 실패한 원본은 복사하지 않는다. 다시 시드할 때 원본이 바뀌었으면 복사본을 바꾸고(이전 복사본은 `.prev<N>.json`), 복사본에 수정 턴이 이미 있으면 거부하고 exit 1 이다. `seeded_sha256` 이 없는 예전 복사본은 내용으로 비교한다.
- 린트 절의 "Not available." 은 직전 턴 `stages["lint"]` 가 없거나 `ok` 가 거짓이거나 `tool == "none"` 일 때다. Rust 는 베이스라인 린터가 없어 항상 `tool == "none"` 이므로 `compiler+lint` 조건이 사실상 `compiler` 조건 + "Not available." 한 절이다(대응 비교를 해석할 때 주의).
- 종료 코드: 0 정상, 1 설정/입력, 2 확인 거부, 3 치명 API 오류, 5 하네스 오류, 130 중단. `harness.run` 의 준비·처리 단계가 멈추면 그 코드(4 규약, 6 기준선, 7 도구 없음)를 그대로 돌려주고 행을 쓰지 않는다. 끝날 때 `finish_run` 에는 그 run 의 모든 행(턴 0 포함)을 넘긴다.

### 9.4 경로 (`paths.py`, G1)

```python
def raw_turn_path(raw_dir, run_id, task, language, tag, sample_idx, turn) -> Path   # turn 0 → sample_<i>.json, t ≥ 1 → sample_<i>.turn<t>.json
def built_turn_dir(built_dir, run_id, task, language, tag, sample_idx, turn) -> Path # turn 0 → …/sample_<i>/, t ≥ 1 → …/sample_<i>/turn_<t>/
RAW_TURN_RE = r"^sample_(\d+)(?:\.turn(\d+))?\.json$"
```

## 10. `run.py` 통합 (G4)

### 10.1 재사용 API (repair.py 가 쓴다)

```python
class RunSetupError(RuntimeError): exit_code: int
def setup_run(cfg, run_id, *, skip_bench=False, force=False, strict_protocol=False, no_baseline=False,
              tasks: list[str] | None = None, languages: set[str] | None = None,
              samples: set[int] | None = None) -> RunContext            # samples 는 구현 때 추가
    # 정적 규약 → 처리 대상 raw → 도구 정체(toolchain_identity)·지문 → 도구 점검·시험 실행(§10.6) → 과제 컨텍스트·기준선.
    # 실패는 RunSetupError(exit_code).
def process_sample(ctx, tctx, raw_path) -> tuple[RunRow | None, str]   # 상태 ∈ processed | skipped | ignored
def finish_run(ctx, processed: list[tuple[RunRow, Path]]) -> None     # drift·model_served 패치
```

`main` 은 이 셋으로 다시 짠다. turn ≥ 1 raw(`sample_<i>.turn<t>.json`)도 `find_raw_files` 가 찾아 처리한다(턴 순서대로).

- `process_sample` 은 읽을 수 없는 raw, 모르는 언어, 지원하지 않는 (과제, 언어) 조합에 `(None, "ignored")` 를 돌려준다. `HarnessError`(exit 5)와 `ToolchainMissing`(exit 7)는 행을 쓰지 않고 올린다.
- `finish_run` 에는 필터(`--languages`, `--samples`)와 무관하게 **그 run 의 모든 `result.json` 행**을 넘긴다(drift 는 과제별 대응 기준선 중앙값 기준이라 일부 행만 주면 기준이 달라진다; `run.result_json_path(row)`).
- 행의 `run_id` 는 언제나 `--run-id` 다. raw 의 `run_id` 가 다르면(다른 run 의 raw 를 복사한 경우) 경고하고 원래 값을 `result.json` 의 `raw_run_id` 에 남긴다.
- 건너뛰기 규칙(`is_up_to_date`): `--force` 가 아니고, 같은 raw(`raw_sha256`)로 썼고, 이 run 이 요청한 bench 가 빠지지 않았으면 건너뛴다(`REPROCESS_STAGES = ("bench",)`: `--skip-bench` 로 쓴 통과 행은 bench 를 요청한 run 에서 다시 처리된다). safety·lint 만 빠진 결과(설정을 나중에 켠 경우)는 다시 처리하면 측정까지 다시 하게 되므로 알리기만 하고 건너뛴다 — 채우려면 `--force`. 판단 근거는 `result.json` 의 `requested = {bench, safety, lint}` 이고, 이 키가 없는 예전 파일은 행과 단계에서 추론한다.
- `harness.run` 종료 코드(`run.EXIT_*`): 0 정상, 1 입력 없음 등, 2 설정 오류, 4 `--strict-protocol` 위반 또는 BLAS 가 OpenBLAS 가 아님, 5 harness_error(행 없음), 6 기준선 게이트 실패, 7 도구 없음·쓸 수 없음(행 없음; §10.6), 130 중단. CLI 에는 `--tasks` 가 없다(과제 필터는 API 의 `tasks` 인자; `harness.repair` 가 쓴다).
- `--rebuild-csv` 는 `result.json` 에서 CSV 를 다시 만들되 `result.json` 이 없는 기존 CSV 행은 남긴다(쓸 것이 없으면 파일을 건드리지 않는다).

### 10.2 단계 순서

```
생성 게이트 → extract(bundle) → build(백엔드별) → [lint] → import(--inspect) → [perf_summary] → check → [safety] → bench
```

- 백엔드 분기: `c` → `build_c`(keyword 면 `args_header`), `python` → `build_python`, `numba` → `build_numba`, `halide` → `build_halide`, `rust` → `build_rust`. 워커 backend: c·rust → `c`, python → `python`, numba → `numba`, halide → `halide`(+ `--halide-mode` 등).
- 워커 환경: `protocol.launch_env(cfg, threads, openmp=…, extra_env=lang.env)`.
- **lint**(`cfg.lint.enabled`): 소스 파일이 생긴 뒤(빌드 성공 여부와 무관; extract 실패·생성 게이트 행에는 없다) `safety.run_lint` → `lint_*` 열(`safety.lint_columns(res)` — 린트가 돌지 않았으면 전부 None = NA; 0 은 "돌았고 없음"; 범주별 NA 규칙은 §12), `stages["lint"] = dataclasses.asdict(res)`. 플러그인이 없거나 실패하면 빈 결과 + 경고(실험은 계속). 플러그인에 넘기는 `--signature` 는 `{"function", "params", "signature", "args_mode"}` 이다.
- **perf_summary**(`cfg.perf_summary.enabled`): 빌드가 성공하고 import 워커가 커널을 로드하면(`load_ok`) `perf_summary.summarize(language=…, backend=<언어 설정의 백엔드; rust 는 "rust">, code=<소스 원문>, build_log=<build.log 전문>, task=…, spec=…, inspect=import 문서의 inspect, build_extra=br.extra)` → `row.perf_summary`. `summarize` 는 예외를 올리지 않고 실패를 `notes`(`perf_summary.ERROR_NOTE_PREFIX` 로 시작)로 돌려준다 → `stages["perf_summary"] = {"ok": not notes.startswith(ERROR_NOTE_PREFIX), "notes": notes}`.
- **safety**(`cfg.safety.enabled` 이고 backend == c): check 워커가 끝난 뒤(pass 여부와 무관, 빌드 성공 샘플) `safety.run_sanitizers(...)` → `asan_findings`, `ubsan_findings`(돌리지 않았거나 끝까지 못 갔으면 None), `stages["safety"]`(워커 요약), `stages["sanitize"] = dataclasses.asdict(res)`. check 판정이 `timeout` 인 샘플은 건너뛴다(계측 빌드는 더 느려 같은 시간 상한에 또 걸린다).
- `memory_error` 규칙: 그 턴의 판정이 `runtime_crash` 또는 `wrong_output` 이고 `(asan_findings or 0) > 0` 이면 `error_type = memory_error`, `error_detail = "asan:<첫 종류>; was <이전 type>/<이전 detail>"`. **통과한 샘플의 `pass` 는 바꾸지 않는다**(잠재 결함은 `asan_findings` 열로만 남는다).
- `turns_to_pass = row.turn if row.passed else None`.
- 산출물 경로: `paths.built_turn_dir(...)`.
- `contract_violation` 추가 값: `schedule_in_halide_auto`(halide_auto 인데 `build.extra["schedule_calls"]` 가 비어 있지 않음). 계획했던 `numba_object_mode` 는 **넣지 않았다**: 빌드가 `njit` 또는 `jit(nopython=True)` 만 통과시키고 `extra` 에 nopython·forceobj 키가 없어 관찰될 수 없기 때문이다.
- 워커 종료 규칙(구현 때 정함; v2.1 §8-6 의 보충): `worker_error` 가 있거나 결과 문서를 한 번도 쓰지 못하고 끝난 워커는 harness_error(행 없음, exit 5)다 — 워커는 커널을 로드하기 전에 첫 문서를 쓰므로 문서가 없으면 LLM 코드는 실행된 적이 없다. 문서가 있는데 `worker_error` 가 없으면 rc 가 3 이어도 커널이 `exit(3)` 을 부른 것으로 보아 `runtime_crash` 다. import 워커가 `load_ok=true` 를 쓴 뒤 끝까지 가지 못한 것(관찰값 수집 중의 크래시·시간 초과)은 판정이 아니다: 경고하고 `inspect` 없이 계속하며 판정은 check·bench 에 맡긴다. 이 불변식 때문에 워커는 `load_ok` 를 로드(halide 는 첫 형태의 빌드까지)가 끝난 뒤에만 참으로 쓰고, `--inspect` 시작 전에 문서를 내보낸다.
- import 워커가 `halide_autoscheduler_unavailable:` 로 끝나면 `ToolchainMissing`(exit 7)이다.
- 행의 `cflags`(구현 때 확장): c 는 `base_flags + cflags (+ openmp_flag) + ldflags`, rust 는 `RUSTFLAGS` 와 그 빌드의 `Cargo.toml` `[profile.release]`(`키=값`), halide 는 auto 면 `autoscheduler=<이름> params=<워커가 넘기는 매개변수의 정규 JSON; parallelism 포함>`, manual 이면 `autoscheduler=none`, python·numba 는 NA.

### 10.2a 샘플 디렉터리의 파일과 `result.json` 의 `stages` (repair.py·perf_summary 가 읽는다)

샘플(턴) 디렉터리: `kernel.{c,py}` 또는 `src/lib.rs`(+ `Cargo.toml`·`Cargo.lock`), `kernel_args.h`/`src/kernel_args.rs`(키워드 모드), 산출물 `kernel.so`(C·Rust; 개발용 Windows 의 Rust 는 `kernel.dll`), `build.log`(C·Rust 만: 컴파일러·cargo stderr. python·numba·halide 샘플에는 없다), `build.json`(`BuildResult` + `extra`), `lint.json`(플러그인 출력 원문)·`lint.stderr.log`(플러그인 stderr), `import.json`·`check.json`·`bench.json`·`safety.json`(워커 결과 문서; `safety` 는 sanitizer 빌드로 돈 check), `<stage>.stdout.log`·`<stage>.stderr.log`(위 네 단계 각각), `paired_baseline.json`·`paired_baseline.{stdout,stderr}.log`(그 샘플 bench 직전의 대응 기준선; 예전 위치 `baseline/paired_sample_<lang>_<i>.json` 에서 옮겼다), `san/`(sanitizer 빌드·로그: `kernel.c`, `kernel_san.so`, `build.log`, `build.json`, `asan.<pid>`, `ubsan.<pid>` — `ubsan.<pid>` 는 gcc 에서 libubsan 을 선행 로드할 때 생긴다, §12), `result.json`. 수정 턴 t 는 같은 파일을 `…/sample_<i>/turn_<t>/` 에 둔다. run 공용: `results/built/<run_id>/_cargo_target/`(Rust 의존성 빌드 재사용, `_toolchain_probe/` 포함), `results/built/<run_id>/<task>/baseline/{bench_run.json,bench_1t.json}`(run 시작 기준선).

`result.json` = `{"row": {...}, "raw_sha256": str, "stages": {...}, "requested": {"bench": bool, "safety": bool, "lint": bool}, ["raw_run_id": str,] "written_at": str}`. `requested`·`raw_run_id` 는 구현 때 추가했다(§10.1). `stages` 의 키와 형태:

- `extract`: `{ok, signature_ok, signature_exact, detail, n_blocks}` (기존)
- `build`: `BuildResult.to_dict()` 에서 `log` 를 뺀 것 (+ `extra`)
- `lint`: `{"tool": str, "ok": bool, "error": str, "summary": {범주: 수 | null}, "findings": [{rule, severity, line, message, category}]}` (`null` = 그 범주를 검사하지 않음; 플러그인의 `unmapped`·`versions` 는 `lint.json` 에만 있다)
- `import` / `check` / `bench` / `safety`: `WorkerOutcome.summary()` = `{rc, timed_out, signal, elapsed_s, timeout_s, stdout_bytes, stdout_tail, stderr_tail, result_file, complete, classification, error_detail}` — 상세는 `<stage>.json`
- `perf_summary`: `{"ok": bool, "notes": str}` (요약 본문은 행의 `perf_summary` 열)
- `sanitize`: `SanitizeResult` 를 dict 로 = `{ran, build_ok, asan_findings, ubsan_findings, kinds, detail, summary}`
- `dynamic`: `DynamicStatus` 를 dict 로(기존 키 + 구현 때 추가한 `settle_wait_s`), `paired_baseline`: 대응 기준선 워커의 `summary()`
- 키가 없는 경우: `lint` 는 린트가 꺼져 있거나 extract 실패·생성 게이트 행이면 없다. `perf_summary` 는 빌드나 로드(import)가 실패하면, `sanitize`·`safety` 는 safety 가 꺼져 있거나 C 가 아니거나 check 가 timeout 이면(`safety` 는 그 밖에도 sanitizer 워커가 실제로 돈 경우 — Linux, sanitizer 빌드 성공 — 에만 있다), `dynamic`·`paired_baseline`·`bench` 는 check 를 통과하지 못했거나 `--skip-bench` 면 없다. 읽는 쪽(repair.py)은 `.get` 으로 읽는다.

### 10.3 `run_worker` 추가 인자

`env_extra: dict[str, str] | None`(launch_env 뒤에 덮어씀), `backend_args: list[str] | None`, `max_mem_gb: float | None`(구현 때 추가; 없으면 `measurement.max_worker_mem_gb`, sanitizer 실행은 0). 샘플 처리의 클로저 `worker(stage, mode, tests, timeout_s, extra=None, *, artifact=None, env_extra=None, max_mem_gb=None)` 를 `safety.run_sanitizers` 에 넘긴다.

### 10.4 `protocol.py`

- `launch_env(cfg, threads, *, openmp, extra_env=None)`. `extra_env`(언어의 `env`)는 마지막에 덮어쓴다. 개발용 Windows 에서는 `PYTHONIOENCODING=utf-8` 을 더한다.
- `check_static`: 비-Linux 면 플래그 `platform=<sys.platform>`(정보용 아님; `protocol.platform_flag()`). `check_dynamic`: `/proc/stat`·`getloadavg` 가 없으면 사용률·load 는 0 으로 두고 같은 플래그.
- `toolchain_identity(cfg) -> dict`: 키 `cc`, `cc_version`, `cargo`, `rustc`, `numba_version`, `halide_version`, `halide_autoscheduler`, `halide_autoscheduler_loadable`, `lint_tools`(ruff·pyflakes 는 `PYTHON_EXE` 의 패키지, cppcheck·clang-tidy 는 PATH), `errors`. numba·halide·린트 패키지와 오토스케줄러 로드 가능 여부는 `PYTHON_EXE` 서브프로세스(`--python-toolchain NAME`)로 본다. cargo·rustc 는 `toolchain_env()` 환경으로 본다. 쓸 수 없는 도구의 값은 None.
- `environment_fingerprint` 에 `rustc`, `numba_version`, `halide_version`, `lint_tools` 를 넣는다(env_hash 가 바뀐다 — 머신을 옮기므로 어차피 바뀐다). cargo 버전, Halide 오토스케줄러 이름·매개변수는 지문에 없다(오토스케줄러 설정은 halide 행의 `cflags` 에 남는다).
- 행의 `omp_env` 열의 접두 `run.OMP_ENV_PREFIXES` 에 `NUMBA_`, `RAYON_`, `HL_` 를 더했다(이 상수는 계획과 달리 `protocol.py` 가 아니라 `run.py` 에 있다).
- `schema.source_hash` 의 대상에 `plugins/lint/run_lint.py`, `plugins/lint/rules.yaml` 과(구현 때 추가) `harness/build/rust_template/{Cargo.toml,Cargo.lock}` 을 더했다(규칙 표·release 프로필·rayon 버전을 고치면 `harness_version` 의 dirty 해시가 바뀐다).
- 구현 때 추가: `toolchain_env()`·`TOOLCHAIN_ENV_KEYS`(§6), `settle_load()` 와 `measurement.max_settle_s`(기본 0; `docs/protocol.md` §9), `--probe-runtime halide|numba`(언어의 런타임 설정을 워커와 같은 코드로 써 보고 `{"error": null | 사유}` 를 출력; §10.6), `OPTION_DEFAULTS`/`config_option`(§8).

### 10.5 스키마 (`schema.py`)

**열은 추가하지 않는다**(기존 CSV 와 헤더가 같아야 한다). `backend` 값에 numba·halide·rust 가 생기고 `turn ≥ 1` 행이 생길 뿐이다. `read_result_rows` 는 `turn_<t>/result.json` 도 읽는다(rglob 이라 이미 된다 — 테스트 추가).

### 10.6 도구 점검

`setup_run` 은 처리할 raw 의 언어에 필요한 도구가 없으면 **행을 쓰기 전에** 멈춘다(`EXIT_TOOLCHAIN = 7`): c → `build.cc`, rust → cargo, numba·halide → `PYTHON_EXE` 에서 임포트 가능, halide_auto → 오토스케줄러 로드 가능, 그리고 이번에 C 기준선을 빌드해야 하면 `build.cc`. 빌드 도중 `ToolchainMissing` 도 같은 코드로 끝낸다. 없는 도구가 `compile_error` 행이 되는 일은 없어야 한다.

도구가 있어도 쓸 수 있는지 한 번씩 돌려 본다(구현 때 추가; 실패는 모두 exit 7, 행 없음):

1. 실행 접두(`probe_launch_prefixes`): run 이 쓸 `protocol.launch_prefix` 마다 `[*접두, PYTHON_EXE, -c, pass]`(명령당 60 s). `measurement.cores`·`single_core` 가 이 머신에 없는 코어면 taskset 이 모든 명령을 rc 1 로 끝내 빌드는 `compile_error`, 워커는 `runtime_crash` 가 되기 때문이다. 접두가 비면(taskset 없음, 비-Linux) 건너뛴다.
2. 언어 런타임(`probe_language_runtimes`): halide_auto 의 오토스케줄러 매개변수와 numba 의 스레딩 계층을 그 언어의 실행 접두·환경으로 띄운 서브프로세스(`protocol.runtime_probe_subprocess` → `python -m harness.protocol --probe-runtime …`)에서 워커와 같은 코드로 써 본다. 틀린 매개변수와 띄울 수 없는 `NUMBA_THREADING_LAYER` 는 그렇지 않으면 첫 샘플의 `worker_error`(exit 5)로 드러난다.
3. Rust(`probe_rust_toolchain` → `build.rust.probe_toolchain`): 자리표시 크레이트를 run 의 공유 target 디렉터리에서 샘플과 같은 설정·접두로 한 번 빌드한다(의존성 캐시 누락·링커 실패를 잡고, 성공하면 rayon 빌드를 샘플이 재사용한다).

## 11. 성능 결정 요약 `perf_summary.py` (G5)

```python
def summarize(*, language: str, backend: str, code: str, build_log: str = "", task: str = "", spec: TaskSpec | None = None,
              inspect: dict | None = None, build_extra: dict | None = None) -> dict
```

출력은 `CLAUDE.md` §5 스키마의 키를 전부 갖는다(`language, loops, parallel, vectorized, tiling, memory, library_calls, decisions_by_llm, decisions_by_compiler, notes`). 정할 수 없는 값은 `"unknown"`/빈 리스트로 두고 거짓 확신을 만들지 않는다. 순수 함수(파일·서브프로세스 없음, LLM 코드 실행 없음).

- **C**: 주석·문자열 제거 뒤 `for` 루프 중첩(깊이, 유도 변수 순서, 상한 식), `#pragma omp`(parallel/for/simd/collapse/reduction/schedule/critical/atomic), 타일링 단서(`+= 상수|BLOCK`, 내부 루프 상한의 `min(`·`MIN(`), 인트린식(`_mm*`, `__builtin_*`, `<immintrin.h>`), 호출 이름 중 알려진 라이브러리(`cblas_*`, `memcpy`, `memset`, `qsort`, `aligned_alloc`, `malloc`…), 임시 버퍼 할당 수. 벡터화 상태는 gcc `-fopt-info-vec` 줄(`optimized: loop vectorized using N byte vectors`, `missed: couldn't vectorize loop` + 이유)에서: `source: "compiler_remark"`; 인트린식이 있으면 `decisions_by_llm` 에 `vectorization`.
- **Rust**: `par_iter`/`par_chunks*`/`rayon::scope`/`join`, `std::arch` 인트린식, `chunks_exact`, `unsafe` 블록 수(`notes`), 루프 중첩은 `for … in` 기준 휴리스틱. 벡터화 리마크 없음 → `unknown`.
- **Python(numpy)**: `ast` 로 배열 원소 루프(`for` 안의 첨자 대입·읽기) 탐지, NumPy 호출 목록(`np.matmul`, `np.dot`, `@` 연산자는 `matmul`), 임시 배열 생성 추정, `out=` 사용 여부.
- **Numba**: 데코레이터 옵션(`build_extra["options"]`), `prange`, 루프 중첩(AST), `inspect["parallel_diagnostics"]` 가 있으면 융합·병렬 루프 수를 `notes` 에.
- **Halide**: `build_extra["schedule_calls"]`(manual), `inspect["schedule_source"]`(auto), `inspect["loop_nest"]` 를 파싱해 `loops`(순서), `parallel`, `vectorized`, `tiling`(split/tile 인자)을 채운다. auto 면 스케줄 결정은 `decisions_by_compiler`, manual 이면 `decisions_by_llm`.
- 구현이 더한 키(CLAUDE.md §5 예시와 키·타입 호환): `decisions_by_library`(라이브러리 호출에 넘긴 결정 — `library_delegation` 만 싣는다), `parallel.loops`/`parallel.placement`(none|outer|inner|mixed|region|unknown)/`parallel.constructs`(병렬 구문 원문 — OpenMP 지시문 전부, rayon 진입점, prange, Halide `parallel(…)`). 표기: 정할 수 없으면 문자열 `"unknown"`, 수·불리언 `null`, 리스트 `[]`; `false`·`0`·`"none"` 은 "찾아봤는데 없다". `parallel.kind ∈ {none, openmp, threads, rayon, numba, halide, unknown}`, `vectorized.status ∈ {yes, partial, no, unknown}`, `vectorized.source ∈ {compiler_remark, intrinsics, loop_nest, interpreter, none}`. `loops` 는 가장 안쪽 루프마다 하나(최대 20개), `notes` 는 영어로 `; ` 로 잇고 500자까지다. 내부 실패는 전체 키 집합에 `notes = "perf_summary error: <Type>: <msg>"`(`ERROR_NOTE_PREFIX`)를 담는다.
- 판정 규칙(리뷰 수정 뒤): `vectorized.status` 는 판정 근거(리마크·루프 중첩·인트린식)가 있는 가장 안쪽 루프로만 정한다 — 근거 없는 루프는 빼고 `notes` 에 남기며, 근거 있는 루프가 없으면 `unknown`. C·Python 의 `while`/`do` 안의 병렬 루프는 가장 바깥이 아니다(placement 한 단계를 더한다; 반복마다 fork/join). Numba 의 중첩 `prange` 는 가장 바깥 것만 병렬로 센다. 워커의 `inspect_error` 는 `notes` 에 남기고, numba 의 `threading_layer` 는 병렬 커널에서만 `notes` 에 남긴다. gcc 가 `ignoring '#pragma omp …'` 라고 경고했으면(`-fopenmp` 없이 빌드) OpenMP 지시문은 `constructs` 에만 남고 병렬로 세지 않는다 — 이 판정은 `build.base_flags` 의 `-Wall` 에 기댄다.
- halide_auto 의 요약은 측정되는 스케줄을 기술한다: run.py 가 import 워커에 성능 형태의 dim(`--inspect-dims`)을 넘기고 워커가 그 크기로 파이프라인을 만들어 `inspect` 를 낸다(§5.1). 계획은 첫 정확성 테스트 크기였으나, 오토스케줄러의 결과가 크기에 따라 다르다는 것이 실측으로 드러나 바꿨다(gemm 은 (7,5,3) 에서 `parallel` 만, 1024³ 에서 `split(…).vectorize(…)` 가 붙는다).
- 범위 밖으로 둔 것: C 의 libm 호출과 수동 다중 누산기, `while`/`do` 루프와 컴프리헨션 자체(`loops` 는 `for` 만), 함수 경계를 넘는 중첩. `algorithm_choice` 는 Python 정렬 호출의 `kind=` 에서만 나온다.
- 진단용 CLI: `python -m harness.perf_summary --language L --backend {python,numba,halide,c,rust} --file SRC [--build-log build.log] [--inspect import.json]`. `--signatures` 가 없어 과제 명세 없이 요약하므로 파이프라인 행과 똑같지 않을 수 있다.
- `decisions_by_llm` / `decisions_by_compiler` 의 어휘(고정): `loop_order`, `tiling`, `vectorization`, `parallel_region`, `reduction`, `memory_layout`, `library_delegation`, `algorithm_choice`, `accumulation_precision`.

## 12. 안전성 `safety.py` + 린트 플러그인 (G6)

```python
@dataclass class LintResult: tool: str; findings: list[dict]; summary: dict[str, int | None]; ok: bool; error: str = ""
def run_lint(cfg, *, language: str, backend: str, file: Path, task: str, signature: dict, out: Path, cwd: Path) -> LintResult
    # `PYTHON_EXE <plugin> --language … --file … --task … --signature <json> --out <json>` 를 run_isolated 로. 계약(CLAUDE.md §6) 위반·실패 → 빈 결과, ok=False.
def lint_columns(res: LintResult) -> dict[str, int | None]   # lint_findings_total, lint_memory, …, lint_performance (린트가 돌지 않았으면 전부 None)
def lint_messages(res: LintResult, limit: int = 20) -> list[str]     # 수정 프롬프트용 "line N: message"

@dataclass class SanitizeResult: ran: bool; build_ok: bool; asan_findings: int | None; ubsan_findings: int | None
                                 kinds: list[str]; detail: str; summary: dict
def sanitizer_env(cfg, cc: str, log_dir: Path) -> dict[str, str]     # LD_PRELOAD(libasan[:libubsan]), ASAN_OPTIONS, UBSAN_OPTIONS(log_path)
def parse_sanitizer_logs(log_dir: Path, stderr_text: str = "") -> tuple[int, int, list[str]]   # (asan 보고 수, ubsan 'runtime error' 서로 다른 위치 수, 종류)
def run_sanitizers(cfg, *, code: str, sample_dir: Path, openmp: bool, launch_prefix: list[str], symbol: str,
                   args_header: str | None, tests: list[Path], worker: Callable[..., Any]) -> SanitizeResult
    # san/ 하위에 build_c(extra_flags=cfg.build.sanitize_flags, artifact_name="kernel_san.so") → worker("safety", "check", tests, timeout,
    # ["--check-repeats", "1", "--sanitize"], artifact=…, env_extra=sanitizer_env(...), max_mem_gb=0) → 로그 파싱.
```

- Python·Numba·Halide·Rust 는 sanitizer 를 돌리지 않는다(`asan_findings`·`ubsan_findings` NA). TSan 은 하지 않는다(libgomp 가 계측되지 않아 거짓 양성).
- sanitizer 세부(리뷰 수정 뒤): 워커는 `--sanitize` 로 배열마다 C 할당자로 정확히 n 원소만 잡는다(64바이트 정렬 시작, 앞 여유·뒤 가드 없음). 가드가 같은 malloc 블록 안이면 하네스 배열을 조금 넘는 접근이 ASan 에게 정상 접근으로 보이기 때문이다. 이 모드의 `oob_write` 는 늘 False 이고 판정은 쓰지 않는다(보고 수만 센다). `LD_PRELOAD` 는 `<cc> -print-file-name=libasan.so` 이고, `<cc> -print-file-name=libubsan.so` 가 실제 파일이면(gcc) 그 뒤에 함께 선행 로드한다 — 워커가 커널 로드 전에 `PR_SET_DUMPABLE=0` 을 걸면 libubsan 이 `/proc/self/environ` 에서 옵션을 읽지 못하기 때문이다. `ASAN_OPTIONS` = `detect_leaks=0`, `abort_on_error=1`, `allocator_may_return_null=1`, `handle_sigfpe=0`, `log_path=<san>/asan`; `UBSAN_OPTIONS` = `print_stacktrace=1`, `halt_on_error=0`, `log_path=<san>/ubsan`. 보고는 `san/asan.*`, `san/ubsan.*` 와 `<sample>/safety.stderr.log` 전체에서 센다.
- 값의 뜻: 0 은 끝까지 돈 실행에서만 0 이다. 커널을 로드하지 못했는데 보고가 없으면 None 과 `detail = "load_failed:<사유>"`, 끝까지 가지 못했는데 보고가 없으면 None 과 `"incomplete:<사유>"` 다. safety 가 꺼져 있거나(`disabled`) Linux 가 아니면(`unsupported_platform:<p>`) `ran=False`, sanitizer 빌드가 실패하면 `build_failed:<kind>: <detail>` 이고 둘 다 열은 NA 다. check 판정이 `timeout` 인 샘플은 run.py 가 sanitizer 를 건너뛴다(§10.2). 컴파일러가 없으면 `ToolchainMissing` 을 그대로 올린다.
- 플러그인 `plugins/lint/run_lint.py`: CLAUDE.md §6 계약의 **기성 린터 베이스라인**. c_seq·c_openmp → cppcheck, clang-tidy(있을 때), numpy·numba·halide_auto·halide_manual → ruff, pyflakes, rust → 없음(항상 `tool: "none"`). 규칙 ID → 범주(`memory, numeric, concurrency, shape_dtype, api_misuse, performance`) 대응표는 `plugins/lint/rules.yaml`(표에 없는 규칙은 버리고 출력의 `unmapped` 에 수만 센다). 플러그인은 하네스를 임포트하지 않는다(독립 CLI; 인지프 ML lint 로 교체 가능해야 한다).
- 린트 값의 뜻(리뷰 수정 뒤; 계획의 "도구가 없으면 summary 전부 0" 을 바꿨다 — 0 이 "검사했고 없음"과 섞이지 않게): 도구가 하나도 끝까지 돌지 않으면 `{"tool": "none", "findings": [], "summary": {범주: null…}}`. 도구가 파일을 분석하지 못하면(시간 초과, 비정상 종료, 읽을 수 없는 출력, 구문·컴파일 오류) 그 도구는 돌지 않은 것으로 친다. `summary` 의 범주는 그 범주를 볼 수 있는 도구(`rules.yaml` 의 그 도구 표에 그 범주가 있다)가 하나라도 끝까지 돌았을 때만 수이고 아니면 `null` 이다(예: c_openmp 에서 clang-tidy 가 빠지면 cppcheck 가 보지 않는 범주는 `null`). `lint_columns`: 플러그인 실패나 `tool == "none"` 이면 열 전부 NA, `null` 범주는 그 열만 NA, `lint_findings_total` 은 검사한 범주의 합(= findings 수; 검사한 범주가 없으면 NA). 두 도구가 같은 줄·같은 범주를 보고하면 하나만 남긴다(먼저 돈 도구의 심각도).
- 플러그인 환경: `CC=<build.cc>` 를 받는다. OpenMP 언어는 그 컴파일러의 `omp.h` 한 파일만 담은 임시 디렉터리를 clang-tidy 에 `-isystem` 으로 준다(clang 에 libomp 의 `omp.h` 가 없으면 `#include <omp.h>` 가 컴파일 오류가 된다). clang-tidy 의 컴파일 인자는 `rules.yaml` 의 `compile_args` 에 따로 있어 `build.base_flags`/`cflags` 를 바꾸면 손으로 맞춰야 한다. 출력에는 계약의 키 외에 `unmapped`, `versions`(분석을 마친 도구의 버전)가 있다. 종료 코드: 0 정상(도구가 없거나 실패해도 0), 2 인자·입력·rules.yaml 오류.

## 13. 분석·그래프 (G7)

- `analysis/stats.py`: `bootstrap_ci(values, stat=np.median, n_boot=2000, alpha=0.05, seed=0)`(백분위 구간, NaN 은 빼고 센다), `bootstrap_pass_rate`, `paired_bootstrap_diff`(같은 샘플 id 의 짝), `unpaired_bootstrap_diff`(독립 표본; 구현 때 추가). 모두 `n_boot=2000, alpha=0.05, seed=0` 이 기본이다.
- `analysis/aggregate.py`: 기존 출력 유지 + `--ci`(그룹 표에 `pass_at_1_lo/hi`, `speedup_median_lo/hi` 열), `--by-turn`, `--conditions`, `--coverage`(과제 × 언어 표: 칸마다 첫 시도 유효 샘플 수 n, `unsupported` 는 `n/a`, 행 없음은 `-`). 그룹 키에 `turn` 이 이미 있으므로 수정 루프 행이 있으면 기본 표는 `turn == 0` 만 세고 그 사실을 한 줄로 알리며, `--by-turn` 이 턴을 합친다. `n/a` 는 기본 그룹 표가 아니라 `--coverage` 행렬에만 나온다. `--json` 이면 옵션마다 `coverage`·`by_turn`·`conditions` 키가 더해진다. 종료 코드: 0 정상(행이 없어도 0), 1 CSV·run 없음, 2 CSV 열이 스키마와 다름.
  - `--by-turn`: 그룹별 `n`(턴 0 이 유효한 샘플 수), `pass_within_<t>`(턴 t 이하에서 통과; `pass_within_0 == pass_at_1`), `turns_to_pass` 분포, `tokens_cum_<t>`(턴 t 이하 모든 행의 tokens_in + tokens_out 합 — 수정 턴의 입력에는 앞선 대화가 다시 들어 있다). 결과를 모르는 샘플(고쳐 보지 않은 턴, API 오류·절단으로 무효인 턴)이 있으면 그 턴의 값은 NA 다(실패로 세지 않는다). 처음 통과한 턴보다 뒤 턴의 낡은 행(`--force` 재처리로 앞 턴이 통과로 바뀐 경우)은 경고하고 세지 않는다.
  - `--conditions`: 선택된 행에서 provider·model·과제·언어와 나머지 두 조건이 같은 묶음 안의 조건 수준 쌍마다 통과율 차와 구간. `feedback` 은 같은 run 의 같은 `sample_idx` 끼리 짝지은 대응 비교이고, 두 수준 모두 결과를 아는 수정 턴 수 H 까지 비교한다(`quantity = pass_within_<H>`; H < 1 이면 통과율·차·구간이 NA 이고 경고). `args`·`grammar_hint` 는 턴 0 의 독립 표본(비교량은 pass@1)이고 feedback 으로 나누지 않는다(시드 복사본은 한 번만 센다). 두 feedback 조건을 다른 run_id 로 돌리면 짝이 0개가 된다.
- `analysis/plots.py`(matplotlib, Agg): `python -m analysis.plots [--csv …] [--run-id ID | --latest | --all-runs] [--out results/plots] [--format png|pdf|svg] [--by-task]`. 그림 파일: `fig1_languages`(과제마다 패널 한 쌍; x = 성능 결정 부담 순 언어, 위 pass@1 과 구간, 아래 통과 샘플의 속도비 로그 축), `fig2_error_types`(언어별 누적 막대: 통과 / 오류 분류 / 생성 단계 실패; `--by-task` 면 `fig2_error_types_by_task`), `fig3_condition_effects`(조건 수준 쌍의 통과율 차와 구간; 행 = (과제, 언어)), `fig4_repair_turns`(과제마다 패널, 수정 턴별 누적 통과율, 언어마다 선). 그림 1·2 는 턴 0 과 기준 조건 조합 하나만 그린다. run 이 여럿이면 그림 1·2 는 run 을 합쳐 세고, 그림 4 는 수정 루프 행이 있는 run 만 합쳐 그리며 처음 통과한 턴 뒤의 낡은 행은 세지 않는다. 색은 Okabe–Ito 팔레트, 계열마다 빗금·선 모양도 다르다. 그림 안의 글자는 영어다(기본 글꼴에 한글 글리프가 없다). 데이터가 없는 그림은 건너뛰고 stderr 에 알린다. 쓴 파일 경로는 stdout 에 한 줄씩 나온다.
- 언어 순서(`LANGUAGE_ORDER`, 성능 결정 부담 순): `numpy, numba, halide_auto, halide_manual, c_openmp, c_seq, rust`(상수는 `analysis/aggregate.py` 에 있고 plots 가 임포트한다. 뒤에 CLAUDE.md §1 의 후순위 언어 `opencl, c_neon, llvm_ir, aarch64_asm` 이 더 있다; 목록에 없는 언어는 이름순으로 뒤를 잇는다). 모델이 다른 run 을 `--all-runs` 로 함께 고르면 그림 1·2 는 모델을 합쳐 세고 그림 3·4 는 표시 없이 함께 그린다(stderr 경고만).

## 14. 더미 응답·devcheck·테스트

- 더미 `harness/dummy_responses/<task>/<language>/<k>.md`: `0.md` 는 정답. 새 과제·새 언어는 `0.md` 만 필수. 수정 루프 검증용으로 gemm 에 `c_seq/1.turn1.md`(세미콜론 고친 정답), `numpy/2.turn1.md`(정답) 를 둔다(G3). 지금 12개 과제 × 7개 언어에서 `unsupported` 가 아닌 조합(80개)에 모두 `0.md` 가 있다.
- `tools/devcheck.py`(G1; 개발용, 측정 아님):

```
python tools/devcheck.py kernel --task T --language L --file PATH [--args-mode positional|keyword] [--tests tiny,rect,small] [--bench] [--keep DIR]
python tools/devcheck.py baseline --task T [--tests …]
python tools/devcheck.py dummies [--tasks a,b] [--languages x,y] [--args-mode …] [--tests …] [--keep DIR]   # 모든 0.md → 표, 하나라도 실패면 exit 1
python tools/devcheck.py gen --task T                                # tasks/T/gen_tests.py 를 그대로 실행(네 형태 전부)
```

  `.md` 면 `harness.extract` 로 코드를 뽑는다. c: gcc 가 있으면 설정의 실제 플래그, 없으면 `python -m ziglang cc`(직렬; `tools/devcheck_shim/omp.h` 스텁과 `aligned_alloc` 보정 `shim.h` — OpenMP pragma 는 무시된다 = **논리 검증만**). rust: `harness.build.rust.build_rust`. python 계열: 그대로. 그 뒤 `harness.worker --mode check` 를 서브프로세스로 띄워 판정을 출력한다. `--bench` 는 large 로 1회 측정(참고값).
- 구현 때 추가: `--cc auto|gcc|zig|zig-omp`. `auto` 는 Linux 에 설정의 컴파일러가 있으면 gcc, 아니면 zig(직렬)다. `zig-omp` 는 직접 줄 때만 쓰는 개발 머신용 실제 스레드 모드다: OpenMP 로 빌드하는 대상(c_openmp, `openmp: true` 인 C 기준선)을 `python -m ziglang cc -fopenmp -c` 로 컴파일해 이 머신의 LLVM OpenMP 런타임(Windows: Visual C++ 재배포 패키지의 `libomp140.x86_64.dll`; 환경변수 `DEVCHECK_LIBOMP` 로 다른 경로)에 링크하고, `<omp.h>` 는 선언만 있는 `tools/devcheck_shim/omp_real/omp.h` 를 쓴다. 스레드 수는 그 언어의 설정값이고 `DEVCHECK_OMP_THREADS` 를 주면 그 값이다. c_seq 는 이 옵션에서도 직렬 빌드다. 판정 줄의 표시는 `toolchain=zig-serial` / `toolchain=zig-omp` 다. clang + libomp 는 측정 경로의 gcc + libgomp 가 아니므로 `zig-omp` 의 PASS 도 Linux 확인을 대신하지 않는다.
- halide 커널은 check 앞에 import 단계(첫 정확성 형태로 빌드·JIT)를 거치고, 로드 실패는 `harness.run` 의 행과 같이 `stage=build` 로 보고한다. extract 에는 설정의 `build.cc` 를 넘긴다. 여러 devcheck 프로세스가 공유하는 Rust target(`<tempdir>/devcheck_cargo_target`)은 `build_rust` 의 잠금이 직렬화한다.
- 단위 테스트: `tests/test_<module>.py`(unittest). 로컬 실행 `C:\graduation_work\.venv-win\Scripts\python.exe -m unittest tests.test_<module> -v`. 병렬 작업 중에는 **자기 모듈의 테스트만** 돌린다(다른 그룹의 파일이 중간 상태일 수 있다).
- Linux 전용 경로(gcc 빌드, taskset, sanitizer, cppcheck 등)는 순수 함수로 쪼개 고정 입력(로그·명령줄)으로 단위 테스트하고, 실제 실행은 `linux_only` 에 적는다.

## 16. 구현 반영 — 계약 보충 (G1·G5·G6·G7 완료 뒤, 2026-11-21)

이 절은 구현 중 정해진 세부다. 앞 절과 다르면 이 절이 맞다.

- **sigs**: `signature_bundle` 의 `signature`·`param_docs` 는 각각 YAML 에 있으면 그대로(끝 개행은 뗀다), 없으면 렌더링. `source` 는 시그니처 문자열의 출처다. `args_struct` 는 keyword 모드의 c/rust 에서만, `numba_signature` 는 numba 백엔드에서만 채운다(다른 언어 템플릿에 값이 필요하면 `sigs.numba_signature(spec)` 를 직접 부른다). halide 의 `params` 는 `halide_params`. C/Rust keyword 의 `param_docs` 는 구조체 안내 문단(`sigs.keyword_struct_intro`)으로 시작한다 — Rust 는 `mod kernel_args;` 와 `use kernel_args::{Camel}Args;` 가 둘 다 있어야 컴파일된다. dim 설명 기본 문구는 형태 식에 나오는 dim 에만 쓰고, 형태와 무관한 스칼라(`T`, `source`)는 `signatures.yaml` 의 `doc` 으로 설명한다(없으면 "a scalar parameter described in the task specification").
- **구조체 파일**: `kernel_args.h` = `#ifndef KERNEL_ARGS_H` / `#define KERNEL_ARGS_H` / 빈 줄 / `typedef struct { … } {fn}_args;` / 빈 줄 / `#endif /* KERNEL_ARGS_H */`. `kernel_args.rs` = `#[repr(C)]` `#[allow(non_snake_case)]` `pub struct {Camel}Args { pub … }`. 둘 다 개행으로 끝난다.
- **paths**: `parse_raw_turn_name(name) -> (sample_idx, turn) | None`, `rel_to_root` 는 `/` 구분자로 돌려준다(Windows 에서도).
- **procutil**: `setup_stdio()`(파이프·파일로 나가는 출력을 UTF-8 로; 각 모듈의 `__main__` 에서만 부른다). 비-POSIX 에서 타임아웃 뒤의 `rc` 는 운영체제 값이고 `signal_name` 은 None 이므로 호출자는 `timed_out` 을 먼저 본다.
- **worker**: `limits` 에 `unsupported`(resource 없음)·`disabled`(`--max-mem-gb <= 0`) 키. 정수 출력 과제의 참조가 int32 로 저장돼 있지 않으면 `worker_error`(테스트 파일의 버그). 넘겨받은 숙제 둘: 길이 0 배열의 `alloc_aligned`, 상대 경로 `--artifact`(G2a).
- **classify/tolerance**: `compare: exact` 에 0 이 아닌 `rtol`/`atol`/`numeric_pattern_tol_ratio` 가 있으면 `load_tolerance` 가 거부한다. `reference_dtype` 기본값은 dtype 이 int32 면 int32.
- **testgen**: `generate(...)`/`main_for(...)` 의 `out_dtype`, `sentinel` 은 키워드 전용. 정수 출력 과제는 `main_for(..., out_dtype="int32", sentinel=-2048)` 로 부르고 참조를 명시적으로 `np.int32` 로 만든다.
- **tests/_support.py**: `requires_linux`(이유 문자열 선택), `requires_tool(*names)`, `requires_modules(*packages)`(비-Linux 에서 없으면 skip, Linux 에서 없으면 실패).
- **devcheck**: 추가 옵션 `--cc auto|gcc|zig|zig-omp`(`zig-omp` 는 2026-11-22 추가, §14), `--timeout`, `--tasks-dir`, `--dummy-dir`, `--config`, `baseline --bench`. 종료 코드 0 통과 / 1 커널이 틀림 / 2 사용법·도구 문제. zig 로 빌드한 판정 줄에는 `toolchain=zig-serial` 이 붙는다(직렬 = 경쟁 상태를 못 잡는다). `dummies` 를 `--tasks`/`--languages` 없이 돌리면 과제는 과제 디렉터리 ∪ 더미 디렉터리, 언어는 `0.md` 가 있는 언어 ∪ 설정의 `languages` 전부가 대상이고, `unsupported` 가 아닌데 `0.md` 가 없는 조합은 `missing`(exit 1)이다.
- **perf_summary / safety / analysis**: §10.2, §11, §12, §13 에 반영했다.

### 16.1 구현·리뷰 수정 뒤 보충 (2026-11-22)

리뷰에서 확인된 결함 27건을 고친 뒤의 상태다. 대부분은 해당 절에 직접 반영했고, 여기에는 절을 가로지르는 것과 찾기 어려운 것만 모은다.

- **어디에 반영했나**: 워커의 `--sanitize`·문서 키·`n_fail`/`n_total`·`null` 인 `inputs_mutated`/`oob_write` → §5.1, §5.5, §12. Halide 어댑터의 오류 문구·출력 경로·`harness_only` → §5.3. Rust 잠금·`probe_toolchain`·`ToolchainMissing` 조건·`nm` 의 `T`/`W`/`i` → §6. extract 의 `cc`·ABI 동일 철자 → §7. 설정 로더 규칙·`max_settle_s` → §8. `--tasks` 부분집합·시드 복사본·수정 루프의 범위·재개·진단 → §9. `requested`·`raw_run_id`·건너뛰기 규칙·`--rebuild-csv`·`finish_run` 입력 → §10.1. 워커 종료 규칙·`cflags`·`numba_object_mode` 제외 → §10.2. 샘플 디렉터리 파일·`stages` 키 → §10.2a. 지문·`toolchain_env`·소스 해시 → §10.4. 실행 접두·런타임·Rust 시험 실행 → §10.6. perf_summary 판정 규칙 → §11. 린트 `null` 과 sanitizer NA 규칙 → §12. 분석 옵션·그림 이름 → §13. devcheck `--cc zig-omp` → §14.
- **대응 기준선 위치**: 샘플(턴) 디렉터리의 `paired_baseline.json` 이다(`run.PAIRED_BASELINE_JSON`). v2 문서·README 의 `baseline/paired_sample_<lang>_<i>.json` 은 예전 위치다.
- **drift 기준**: `finish_run` 이 과제별 대응 기준선 `baseline_time_ms` 의 중앙값을 기준으로 `baseline_drift_pct` 와 `baseline_drift=…%` 플래그를 다시 계산한다(2026-11-05 변경; `docs/protocol.md` §10).
- **stdio**: `harness.config`·`generate`·`llm_client`·`repair`·`run`·`protocol`·`extract`·`sigs`·`paths`·`procutil`, `harness/build/*`, `analysis.aggregate`·`analysis.plots`, `tools/devcheck.py` 의 `__main__` 이 `procutil.setup_stdio()` 를 부른다(개발용 Windows 파이프에서 한국어·기호 출력이 죽지 않게; Linux 동작은 같다). 워커는 부르지 않는 대신 `launch_env` 가 Windows 에서 `PYTHONIOENCODING=utf-8` 을 준다.
- **halide 행의 cflags 계산**: run.py 가 `harness.kernels` 를 지연 임포트해 워커와 같은 오토스케줄러 매개변수를 만든다(하네스 코드이며 LLM 코드가 아니다).
- **알려진 한계(코드 그대로 둔 것)**: 같은 출력 디렉터리를 다른 인자 방식으로 다시 빌드하면 이전의 `kernel_args.h`·`src/kernel_args.rs` 가 남는다(파이프라인은 `cond_tag` 로 디렉터리가 갈려 해당 없고 devcheck 의 `--keep DIR` 재사용에서만 생긴다). 같은 run id·같은 샘플을 두 프로세스가 동시에 처리하면 `src/lib.rs` 를 서로 덮어쓸 수 있다(잠금은 target 디렉터리만 지킨다). numba 빌드는 `@jit("<서명>")`(`nopython=True` 없음)과 변수에 담은 데코레이터(`fast = numba.njit` 뒤 `@fast("…")`)를 `numba_signature_missing` 으로 거절한다 — Numba 0.59 이후 nopython 이 기본이라 실제로는 즉시 컴파일되는 꼴이다.

### 16.2 검증 범위 (2026-11-22, Windows 개발 머신)

- **여기서 확인한 것**(수정·오케스트레이터 보고 기준): 단위 테스트 전체(리뷰 수정이 끝난 뒤의 전체 실행 `python -m unittest discover -s tests -t .`: 1320 OK, 52 skipped — skip 은 gcc·taskset·sanitizer 등이 필요한 Linux 전용 테스트. 이 문서를 맞춘 2026-11-22 에 다시 돌려 같은 수를 확인했다). 아래의 devcheck·dummy 파이프라인 통합 실행은 오케스트레이터가 리뷰 수정 **전에** 한 것이다. 수정 뒤에는 수정 그룹의 작은 점검과, 이 문서를 맞추면서 돌린 dummy 파이프라인(gemm·histogram × numpy·numba·halide_auto·rust `--skip-bench` 8행 전부 pass, gemm × numpy bench 1행에 대응 기준선·속도비가 채워짐, C 기준선이 필요한 run 은 gcc 가 없어 exit 7)만 있다. `tools/devcheck.py dummies` 로 새 언어 더미 44개 pass·4개 unsupported, `--cc zig-omp` 로 C 더미 24개 pass(c_openmp 12개는 clang + libomp 실제 스레드), gemm 경쟁 상태 더미가 3번 모두 `race_condition`. dummy provider 로 `harness.generate` → `harness.run --skip-bench`(numpy·numba·rust 12과제 36행, Halide 두 언어 9과제 18행 전부 pass), bench 경로(행이 채워지는지만; 시간 값은 의미 없음), `harness.repair --max-turns 2` 와 `--seed-from-feedback`, `analysis.aggregate --ci --coverage --by-turn --conditions`, `analysis.plots --all-runs`(그림 네 장), `harness.run --rebuild-csv`. 이 머신의 행은 `platform=win32`·`governor=unknown`·`taskset_missing` 플래그로 `protocol_ok=0` 이다.
- **Linux 에서만 확인되는 것**: C 언어의 `harness.run` 전 과정(gcc 가 없어 여기서는 설계대로 exit 7), gcc + libgomp 의 OpenMP 실행, taskset 코어 고정과 실행 접두 시험, CPU 거버너·EPP 검사, 실제 시간·대응 기준선·drift, ASan/UBSan(libasan·libubsan 선행 로드, `-shared` 빌드의 링크, 보고 파싱), cppcheck·clang-tidy 의 실제 출력과 `rules.yaml` 대응, gcc 15 의 `-fopt-info-vec` 리마크 위치에 기대는 perf_summary 벡터화 판정, numba `omp` 스레딩 계층, Halide 오토스케줄러 플러그인(.so) 탐색, Rust 의 `nm` 심벌 검사. 구체적인 명령과 기대 결과는 `docs/linux_test_plan.md`(작성 예정)와 각 구현 보고의 `linux_only` 가 기준이다.
- **사용자 확인이 남은 것**(근거·대안은 `docs/decisions.md`): `measurement.max_settle_s` 를 켤지(지금 0), halide 의 형태별 빌드(오토스케줄·JIT) 시간이 run.py 의 bench 시간 상한에는 들어가는 문제, numba 에 `OMP_PROC_BIND` 등 스레드 묶기를 줄지(지금은 c_openmp 에만 `omp_env_extra`), 수정 턴이 절단으로 무효일 때의 누적 통과율 NA 규칙, 수정 루프 진단에 Rust 패닉 메시지를 줄지와 트레이스백의 하네스 프레임, 린트 병합에서 같은 결함의 심각도를 어느 도구 것으로 남길지.
