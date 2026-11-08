# 하네스 설계 계약 v2 (M1 수직 슬라이스)

v1 2026-11-03 → v2 2026-11-04 (5개 관점의 적대적 설계 리뷰 반영) → **v2.1 2026-11-04** (통합 검증 후 §10 분류 순서 변경·import 모드 실패의 `error_stage=build`; 같은 날 리뷰 반영 명료화는 절 안에 "2026-11-04 명료화"로 표시). 반영 내역은 `docs/decisions.md`.
`CLAUDE.md`의 M1을 구현하는 모든 모듈이 지키는 인터페이스·규약이다. 이 문서와 `CLAUDE.md`가 충돌하면 `CLAUDE.md`가 우선하되 충돌은 `docs/decisions.md`에 적는다.
구현자는 자기 파일 밖의 계약(함수 시그니처, JSON 키, CSV 열, CLI 옵션)을 바꾸지 않는다. 바꿔야 하면 이 문서를 먼저 고치고 `docs/decisions.md`에 날짜와 함께 적는다.

## 0. 실험 머신 실측 (2026-11-03/04)

| 항목 | 값 |
|---|---|
| 호스트 | `ustar-Zenbook-UX3402ZA-UX3402ZA`, Linux 7.0.0-38-generic (Ubuntu), glibc 2.43 |
| CPU | 12th Gen Intel Core i7-1260P (Alder Lake 하이브리드) — P-core 4개 = CPU 0-7 (HT 쌍 0/1, 2/3, 4/5, 6/7; 최대 4.7 GHz), E-core 8개 = CPU 8-15 (최대 3.4 GHz). `/sys/devices/cpu_core/cpus`=0-7, `cpu_atom/cpus`=8-15 |
| ISA | AVX2, FMA, SSE4.2 (AVX-512 없음). `gcc -march=native` → `-march=alderlake -mtune=alderlake` (P/E 코어 동일, `--param l1-cache-size`만 48/32로 다름) |
| 캐시 | L1d 48 KiB/P-core, L2 1.25 MiB/P-core, L3 18 MiB (1024³ GEMM 작업집합 12 MiB는 **L3 상주** → warm-cache 측정) |
| 메모리 | 14 GiB |
| 전원 | `intel_pstate` HWP, governor `powersave`, EPP(`energy_performance_preference`)=`power`, turbo 켜짐(`no_turbo=0`). **실측: 이 상태에서 P-core는 부하 중에도 1.3–1.7 GHz에 머물고 샘플 간 400–1800 MHz로 요동한다.** `performance` 거버너는 사용자가 `sudo cpupower frequency-set -g performance`로 켜야 한다(sudo 비밀번호 필요). `perf_event_paranoid=4`라 `perf stat`은 못 쓴다 |
| SMT 간섭 | 고정 코어의 HT 형제(cpu1)에 spin 1개 → 4스레드 OpenBLAS 1024³ 14.6 → 19.9 ms(+36%). E-core(cpu8) spin은 영향 없음 → 형제 코어 부하도 검사한다 |
| 도구 | `taskset`, `cpupower`, `perf`, `numactl`, `nm`, `objdump` 있음. apport 활성(`core_pattern=|/usr/share/apport/apport`) → 세그폴트마다 코어 덤프가 생기므로 워커가 끈다 |
| 컴파일러 | gcc 15.2.0 (`-fopenmp` 동작; gcc 14+는 암묵 함수 선언이 **오류**). clang 21.1.8은 `omp.h` 없음(libomp 미설치) → M1은 gcc 고정 |
| Python | 3.14.4. 저장소 `.venv`는 **격리형**(시스템 site-packages 미사용), `requirements.txt`로 numpy 2.3.5 pip 휠(**scipy-openblas 0.3.30 내장**, pthread 변형, DYNAMIC_ARCH Haswell 커널)·pyyaml·pandas 3.1·anthropic 2.4·openai 3.44 설치. 모든 Python 실행은 `.venv/bin/python` |
| BLAS 주의 | 시스템 `python3-numpy`는 netlib 참조 BLAS(`/usr/lib/x86_64-linux-gnu/blas/libblas.so.3`, alternatives priority 10, OpenBLAS 미설치)에 링크돼 1024³ sgemm이 ~250 ms(스레드 변수 무효). 같은 버전 문자열 2.3.5라 행에서 구분이 안 되므로 **워커가 런타임 BLAS 정체를 기록하고 기준선 단계가 검증한다**(§8) |
| LLM 키 | 환경변수 없음, `ant` CLI 없음 → `provider: dummy`로 파이프라인 검증 |

실측 보정값 (`taskset -c 0,2,4,6`, powersave):

| 커널 | 1024³ 시간 | 비고 |
|---|---|---|
| NumPy `np.matmul(out=)` (.venv OpenBLAS) | 4스레드 14.7 ms, 2스레드 31 ms, 1스레드 51 ms | `OPENBLAS_NUM_THREADS`(우선)/`OMP_NUM_THREADS`(폴백)로 제어 확인 |
| 순진한 C i-j-k, `-O3 -march=native`, 1스레드 | ≈ 9.5 s/호출 | 내부 루프 미벡터화("unsupported control flow"); 13회 ≈ 125 s |
| C i-k-j (`restrict`), 1스레드 | ≈ 240 ms (236–328 ms 요동) | 32바이트 벡터화 |
| 순진한 C 256³ | 25 ms | check 단계 비용 기준 |
| 순수 Python 3중 루프 256³ | 수십 초(추정) | check 예산 안; 1024³는 bench timeout |

**절대 시간은 주파수 상태에 따라 달라진다(2026-11-04 통합 검증 추가; 위 값은 지우지 않는다).** 위 표는 governor `powersave`·EPP `power`(P-core 1.3–1.7 GHz)에서 잰 값이다. `make dummy RUN_ID=dummy-m1`(governor `powersave`, EPP `performance`, bench 중 `cpu_mhz_median` 2.6–2.9 GHz)에서는 같은 NumPy 기준선이 **run 시작 4스레드 7.7–8.3 ms**(대응 기준선 7.7–8.9 ms, drift −2 ~ +7 %), 1스레드 19.5–21.7 ms(`taskset -c 2`, 4.1 GHz)로 위 값의 약 절반이었고, 더미 c_seq/0(i-k-j, `restrict`)은 77 ms(위 240 ms), c_openmp/0(i 루프 병렬)은 29 ms, numpy/0은 9.0 ms였다 — 같은 커널·같은 코어에서 전원 상태만 달라도 2배 차이가 난다. 따라서 결과 비교는 대응 기준선 대비 속도비(`speedup_vs_baseline`)로 하고, 행마다 `governor`·`epp`·`cpu_mhz_*`·`throttle_events`를 기록한다(§8, `docs/protocol.md` §1·§11).

gcc 15에서 `-fopt-info-vec-optimized -fopt-info-vec-missed`는 stderr에 `optimized:`/`missed:` 줄을 낸다. `nm -D --defined-only kernel.so`에 `T gemm` 심벌이 보인다. glibc 2.34+는 `-lpthread` 없이도 `pthread_create`가 링크된다(실측).

핵심 결정: 측정은 **P-core 4개에 코어당 1스레드**(`taskset -c 0,2,4,6`, `OMP_NUM_THREADS=4`)로 고정한다. `threads: 1`인 언어(`c_seq`)는 **단일 P-core**(`taskset -c 2`)로 더 좁게 고정한다. 모든 언어·기준선이 같은 코어 집합·스레드 수·환경변수를 쓴다.

## 1. 파일 목록과 소유 (M1에서 만드는 것만)

```
CLAUDE.md                      (사용자 문서, 수정 금지)
README.md                      실행 방법·결과 열 의미 요약(한국어)
Makefile                       venv / tests / probe / dummy / real / aggregate / unittest 타깃
requirements.txt
.gitignore
config/experiment.yaml         §2
config/machine.yaml            §2 (harness.protocol --probe --write 가 생성; .venv python으로 실행)
tasks/gemm/spec.md             §3 (영어, 프롬프트에 그대로 삽입)
tasks/gemm/spec.ko.md          §3 (한국어 번역, 사람용)
tasks/gemm/signatures.yaml     §3
tasks/gemm/reference.py        §3
tasks/gemm/gen_tests.py        §3  → tasks/gemm/tests/{tiny,rect,small,large}.npz + manifest.json
tasks/gemm/tolerance.yaml      §3
tasks/gemm/baseline/baseline.py §3
prompts/system.md              §4
prompts/lang/{numpy,c_seq,c_openmp}.md   §4
harness/__init__.py
harness/paths.py               ROOT, PYTHON_EXE(= ROOT/.venv/bin/python), 경로 헬퍼
harness/config.py              §2 설정 로더·검증
harness/procutil.py            §7 격리 서브프로세스 실행기 (모든 서브프로세스가 이것을 쓴다)
harness/protocol.py            §9
harness/llm_client.py          §5
harness/generate.py            §5
harness/extract.py             §6
harness/build/__init__.py
harness/build/c.py             §7
harness/build/python.py        §7
harness/classify.py            §10 (오류 분류 규칙, 고정)
harness/worker.py              §8 (서브프로세스 실행기; LLM 코드를 실행하는 유일한 곳)
harness/run.py                 §8 (오케스트레이션)
harness/schema.py              §11
harness/dummy_responses/gemm/<language>/<k>.md + README.md   §13
analysis/__init__.py
analysis/aggregate.py          §12
tests/test_*.py                §15 (unittest; 각 모듈 소유자가 작성)
docs/decisions.md, docs/protocol.md, docs/error_taxonomy.md, docs/harness_design.md(이 문서)
```

M1에서 만들지 않는 것(자리만): `harness/repair.py`, `harness/safety.py`, `harness/perf_summary.py`, `prompts/repair.md`, `prompts/hints/`, `plugins/lint/`, `analysis/plots.py`, `dashboard/`, 다른 과제.

## 2. 설정

### config/experiment.yaml (키 전부; 값 = 이 머신의 M1 기본값)

```yaml
run:
  tasks: [gemm]
  languages: [numpy, c_seq, c_openmp]
  n_samples: 3                 # M1=3, M2부터 5
  max_repair_turns: 0          # M1=0 (설계 상한 3; M4)
  conditions:
    args: positional           # positional | keyword   (keyword는 M4)
    grammar_hint: "off"        # on | off               (M4)
    feedback: compiler         # compiler | compiler+lint (M4)

model:
  provider: anthropic          # anthropic | openai_compat | dummy   (env LLM_PROVIDER 가 덮어씀)
  name: claude-opus-5-5        # env LLM_MODEL 이 덮어씀
  max_tokens: 32000            # thinking 토큰 포함 상한. 스트리밍으로 보내므로 SDK 비스트리밍 가드(≈21k) 무관
  temperature: null            # Claude 4.7 이후·5.x는 temperature/top_p/top_k를 400으로 거부 → null = 보내지 않음(제공자 기본 샘플링).
                               # non-null이면 anthropic은 extra_body로만 전송(4.6 이하 전용), openai_compat은 그대로 전송.
  seed: null                   # Anthropic Messages API에는 seed가 없다. openai_compat에서만 전달(정수).
  request_timeout_s: 600       # 스트리밍에서는 이벤트 간 read timeout. max_retries는 타임아웃에도 적용됨(SDK)
  max_retries: 3
  anthropic:
    api_key_env: ANTHROPIC_API_KEY
    effort: medium             # output_config.effort (low|medium|high|xhigh|max). null 금지(서버 기본값에 조용히 의존하지 않기 위해)
    thinking: adaptive         # 항상 명시 전송 {"type":"adaptive"} (Opus 5.5는 끌 수 없음; 4.7/4.8도 같은 조건이 되게)
  openai_compat:
    api_key_env: OPENAI_API_KEY      # ANTHROPIC_ 접두 금지(키 유출 방지; 로더가 거부)
    base_url_env: LLM_BASE_URL       # 없으면 실패
    max_tokens_param: max_tokens     # max_tokens | max_completion_tokens

languages:
  numpy:    {backend: python, prompt: prompts/lang/numpy.md,    threads: measurement}
  c_seq:    {backend: c,      prompt: prompts/lang/c_seq.md,    openmp: false, threads: 1}
  c_openmp: {backend: c,      prompt: prompts/lang/c_openmp.md, openmp: true,  threads: measurement}

build:
  cc: gcc
  cflags: ["-O3", "-march=native"]                   # 성능 결정 플래그. 프롬프트에 공개.
  base_flags: ["-std=gnu11", "-fPIC", "-shared", "-Wall", "-fno-semantic-interposition"]
  openmp_flag: "-fopenmp"
  ldflags: ["-lm", "-Wl,--no-undefined"]             # 미정의 심벌은 빌드 단계 compile_error
  diag_flags: ["-fopt-info-vec-optimized", "-fopt-info-vec-missed"]   # 코드 생성 무관; build.log에 저장(M5 입력)
  compile_timeout_s: 60
  forbidden_symbols: ["pthread_create", "clone", "fork", "vfork", "system", "popen", "execve", "execv", "execl"]
  thread_symbols_prefix: ["pthread_create", "GOMP_", "omp_", "__kmpc_"]   # c_seq에서 검출 시 contract_violation

measurement:
  cores: "0,2,4,6"             # taskset -c (P-core 4개, SMT 형제 제외)
  single_core: "2"             # threads==1 언어의 taskset (cpu0은 IRQ가 몰려 회피)
  threads: 4                   # OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS
  omp_env_extra: {OMP_PROC_BIND: "true", OMP_PLACES: "cores", OMP_DYNAMIC: "false"}   # OpenMP 백엔드 워커에만
  warmup: 3
  repeats: 10
  min_repeats: 3               # 느린 커널에서 예산 안에 맞추려 줄일 때의 하한 (repeats_reduced=1 플래그)
  check_repeats: 2             # 정확성 테스트마다 호출 횟수 (실행 간 불일치 = race 탐지)
  check_timeout_s: 180
  bench_timeout_s: 300
  max_worker_mem_gb: 8         # RLIMIT_AS
  max_load_1m: 2.0
  max_pinned_busy_pct: 10.0    # 고정 코어·형제 코어의 사전 사용률(1초 샘플)
  require_performance_governor: true   # 아니면 경고 + governor_ok=0 (실험은 계속; --strict-protocol이면 중단)
  baseline_min_gflops: 30      # 기준선이 이보다 느리면 BLAS 정체 의심 → 중단
  baseline_drift_pct_max: 10   # 대응 기준선이 run 시작 기준선과 이만큼 다르면 플래그(기록만; --strict-protocol 무관)
  correctness_tests: [tiny, rect, small]
  performance_test: large

paths:
  results_csv: results/runs.csv
  results_csv_dummy: results/runs_dummy.csv   # provider == dummy 인 run의 행은 여기로
  raw_dir: results/raw
  built_dir: results/built
```

`harness/config.py`:
```python
@dataclass(frozen=True) class Config  # 위 구조를 그대로 담는 중첩 dataclass (RunCfg, ModelCfg, LanguageCfg, BuildCfg, MeasurementCfg, PathsCfg)
def load_config(path: Path | None = None) -> Config
    # 1) YAML 2) env 덮어쓰기: LLM_PROVIDER→model.provider, LLM_MODEL→model.name
    # 3) 검증(실패 시 ConfigError에 키 경로): provider∈{anthropic,openai_compat,dummy}; anthropic이면 seed는 null·effort non-null;
    #    temperature non-null이면 stderr 경고("Claude 4.7+/5.x는 400"); openai_compat이면 api_key_env가 ANTHROPIC_로 시작하면 거부;
    #    languages[*].threads ∈ {"measurement", 양의 정수} → 정수로 치환; cores/single_core 형식
    # 4) 상대 경로는 ROOT 기준 절대화
def load_machine(path=ROOT/"config/machine.yaml") -> dict   # 없으면 {} + 경고
def config_snapshot(cfg) -> dict   # raw JSON에 넣을 직렬화
```

### config/machine.yaml

`.venv/bin/python -m harness.protocol --probe --write` 가 생성·갱신한다. 키:
`probed_at, hostname, os, kernel, glibc, cpu_model, sockets, physical_cores, logical_cpus, smt, p_core_cpus, e_core_cpus, thread_siblings (dict cpu→[siblings]), isa (리스트), caches (dict), memory_gb, governor, governor_available, epp, no_turbo, scaling_driver, taskset_available, cpupower_available, sudo_nopasswd, apport_core_pattern (bool), compilers {gcc, clang, openmp_gcc, openmp_clang}, march_resolved (예 "-march=alderlake -mtune=alderlake"), python_exe, python_version, numpy_version, numpy_blas {name, version, config, loaded_lib}, sdk_versions {anthropic, openai}, bandwidth_gbs: null, peak_gflops: null, notes`.
`numpy_blas`는 반드시 `PYTHON_EXE`(.venv)로 측정한다(§9 `blas_identity()`).

## 3. 과제 gemm

### 수학·계약 (spec.md에 영어로 그대로 서술)
- `C[i][j] = Σ_{k=0}^{K-1} A[i][k] · B[k][j]`, 0 ≤ i < M, 0 ≤ j < N.
- 모두 float32, row-major, C-contiguous. `A` M×K, `B` K×N, `C` M×N(호출자가 할당). **A, B, C의 시작 주소는 64바이트 정렬**(하네스가 보장; 사실로 서술).
- `C`의 초기 내용은 미정의(임의의 유한값)이며 함수는 **모든** 원소를 덮어써야 한다(누적 금지). 결과는 반드시 `C`에 써야 하며 반환값은 무시된다(Python은 None 반환).
- `A`, `B`는 읽기 전용. 임의의 양의 M, N, K에 대해 올바라야 한다(정사각이 아니거나 작거나 8/16의 배수가 아닌 경우 포함). 평가 형태 공개: (M,N,K) = (7,5,3), (200,130,312), (256,256,256) 정확성; (1024,1024,1024) 성능.
- 허용오차: `|out − ref| ≤ atol + rtol·|ref|`, rtol = 1e-3, atol = 1e-3 (ref는 float64 누적). 누적 순서·블록화 자유.
- 매개변수는 **정확히 이 여섯 개**(추가 매개변수 금지, 기본값 있는 것도 금지). 임시 메모리 할당 가능(반환 전 해제). 파일·네트워크·stdout·환경변수·전역 상태 금지. 여러 번 호출해도 같은 결과.
- 평가 절차 공개: 정확성 형태마다 2회 호출(실행 간 일치 확인) → 통과 시 1024³에서 워밍업 3회 + 10회 반복의 중앙값. 머신·스레드·플래그 사실은 언어 파일(§4)이 준다. "fastest correct implementation"을 요구. 라이브러리 언급 금지.

### signatures.yaml
```yaml
task: gemm
function: gemm
params: [A, B, C, M, N, K]
languages:
  numpy:
    positional: "def gemm(A, B, C, M, N, K):"
    keyword:    "def gemm(*, A, B, C, M, N, K):"
    types: {A: "numpy.ndarray float32 shape (M, K), C-contiguous, 64-byte aligned, read-only", B: "numpy.ndarray float32 shape (K, N)", C: "numpy.ndarray float32 shape (M, N), output", M: int, N: int, K: int}
    returns: "None (results are written into C)"
  c_seq:
    positional: "void gemm(const float* A, const float* B, float* C, int M, int N, int K)"
    keyword: null        # M4: 매개변수 구조체
    indexing: "A[i*K + k], B[k*N + j], C[i*N + j]"
  c_openmp: (c_seq와 동일)
```

### reference.py
`reference_gemm(A, B) -> np.ndarray`: float64로 변환해 `A @ B`, float64 반환.

### gen_tests.py
`PYTHON_EXE tasks/gemm/gen_tests.py [--seed 20261103] [--out tasks/gemm/tests]`
- 형태별 독립 시드 `seed + index`(`np.random.default_rng`), uniform[-1, 1) float32. 파일 `tiny`(7,5,3), `rect`(200,130,312), `small`(256,256,256), `large`(1024,1024,1024); 키 `A`, `B`(float32), `C_ref`(float64), `M`, `N`, `K`, `seed`. `np.savez`(비압축; 재생성 시 바이트 동일).
- `manifest.json`: `{"seed", "generated_at", "c_ref_method": "numpy float64 matmul", "c_ref_blas": <blas name>, "files": {name: {"M","N","K","sha256_inputs"}}, "test_hash": sha256(seed ∥ 각 형태의 M,N,K ∥ A.tobytes() ∥ B.tobytes() 를 형태 순서대로)[:16]}`. **test_hash는 입력만으로 정의**(C_ref는 BLAS에 따라 비트가 달라 제외).

### tolerance.yaml
```yaml
task: gemm
dtype: float32
rtol: 1.0e-3
atol: 1.0e-3                  # 입력 uniform[-1,1)에서 |C|~10; 0 근처 원소 보호. CLAUDE.md는 rtol만 명시 → decisions.md
reference_dtype: float64
numeric_pattern_tol_ratio: 20  # §10: 모든 원소가 허용오차의 20배 이내로 틀리면 numeric_instability(imprecise), 아니면 wrong_output
sentinel: -2048.0              # §8: C 사전 채움값. |C| ≤ K ≤ 1024 이므로 정당한 출력이 될 수 없음
```

### baseline/baseline.py
```python
import numpy as np
def gemm(A, B, C, M, N, K):
    np.matmul(A, B, out=C)
```
하네스가 python 백엔드 커널로 똑같이 측정한다(§8).

## 4. 프롬프트

템플릿 치환은 `string.Template`(`${name}`)만. 변수: `${signature}`, `${indexing}`, `${python_version}`, `${numpy_version}`, `${cc}`, `${cc_version}`, `${build_command}`(실제 전체 컴파일 명령, 예 `gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native [-fopenmp] -o kernel.so kernel.c -lm -Wl,--no-undefined`), `${threads}`, `${cores_desc}`, `${cpu_summary}`, `${task_spec}`.
- `${cpu_summary}`: machine.yaml에서 조립하되 **고정 코어 집합만** 서술: "Intel Core i7-1260P; the kernel runs pinned to 4 physical performance cores (Golden Cove), one thread per core, no SMT; AVX2 + FMA (no AVX-512); L1d 48 KiB and L2 1.25 MiB per core, L3 18 MiB shared". c_seq는 `${cores_desc}`="1 physical performance core, 1 thread" 로.
- `prompts/system.md`(영어, 공통): 역할 · **정확히 하나의** 펜스 코드 블록(완전한 소스) · 시그니처 글자 그대로(이름·순서·타입 변경 금지, **추가 매개변수 금지**) · 파일/네트워크/stdout/환경변수 금지 · `main`/테스트 드라이버/임포트 시 실행 코드 금지 · 질문 금지 · 정확성 우선 후 성능(실행 시간 측정됨).
- `prompts/lang/numpy.md`: 환경(Python ${python_version}, NumPy ${numpy_version}; **규범문** "Use only the Python standard library and NumPy: do not import any other package, and do not install anything" — 워커 인터프리터 `.venv`에는 pandas·PyYAML·pydantic·anthropic 등 다른 패키지가 설치돼 있고 `launch_env`의 `PYTHONPATH=ROOT`로 `harness`·`tasks`도 임포트되며 임포트 제한이 없으므로 "다른 패키지가 없다"는 **사실 진술은 쓰지 않는다**; 하네스 측 강제(import 화이트리스트 → `contract_violation`)는 M1 범위 밖), 시그니처·타입, 실행 환경(${cpu_summary}, ${cores_desc}, threads=${threads}), 출력은 ```python 블록 하나, 마지막에 `${task_spec}`.
- `prompts/lang/c_seq.md`: 환경("C11 with GNU extensions, gcc ${cc_version}, compiled as a shared library with: ${build_command}; only libc and libm are linked; **single-threaded: no OpenMP, no threads of any kind**"), 시그니처·인덱싱, 실행 환경(1 core), 출력 ```c 블록 하나, `${task_spec}`.
- `prompts/lang/c_openmp.md`: 위와 같되 `-fopenmp` 포함 명령, OpenMP 사용 가능, `OMP_NUM_THREADS=${threads}` on ${cores_desc}, `OMP_PROC_BIND=true OMP_PLACES=cores`.
- 환경 서술은 사실 진술이며 특정 라이브러리 사용 지시는 절대 없음. 하드웨어 공개 여부는 사용자 확인 대상(decisions.md).

## 5. 생성: llm_client.py / generate.py

### llm_client.py
```python
@dataclass class LLMResponse:
    text: str; model_requested: str; model_served: str; provider: str
    tokens_in: int; tokens_out: int; tokens_thinking: int | None; tokens_text: int | None
    latency_ms: float; stop_reason: str; raw_stop_reason: str; stop_details_category: str | None
    n_thinking_blocks: int; n_text_blocks: int; request_id: str | None; raw_usage: dict
    base_url: str | None; sdk_version: str | None; api_key_present: bool
    error: str | None      # "fatal:<Type>:<status>:<msg>" | "transient:<Type>:<status>:<msg>" | None
class LLMClient:
    def __init__(self, cfg: Config) -> None
    def preflight(self) -> None                 # anthropic: client.models.retrieve(name) (과금 없음) → Authentication/NotFound면 예외
    def complete(self, system: str, user: str, *, dummy_key: tuple[str, str, int] | None = None) -> LLMResponse
```
- **anthropic**: `anthropic.Anthropic(timeout=request_timeout_s, max_retries=max_retries)`(키는 SDK가 `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`/프로필에서 찾음; `api_key_env`가 다른 이름이면 `api_key=os.environ[...]` 명시). 요청은 **스트리밍**: `with client.messages.stream(model=, max_tokens=, system=, messages=[{"role":"user","content":user}], thinking={"type":"adaptive"}, output_config={"effort": effort}, **extra) as stream: msg = stream.get_final_message()`; `extra = {"extra_body": {"temperature": t}}` (temperature non-null일 때만; `temperature=` 키워드는 SDK 2.x에 없어 TypeError). `fallbacks`·베타 헤더는 쓰지 않는다(모델이 바뀌면 실험 오염). 텍스트 = `type=="text"` 블록 결합; `n_thinking_blocks`/`n_text_blocks` 집계; `tokens_in/out = usage.input_tokens/output_tokens`; `tokens_thinking = getattr(getattr(usage,"output_tokens_details",None),"thinking_tokens",None)`; `tokens_text = tokens_out − tokens_thinking`(둘 다 있을 때); `model_served = msg.model`; `stop_reason = msg.stop_reason`(end_turn|max_tokens|stop_sequence|refusal|...); refusal이면 `stop_details_category = msg.stop_details.category`; `request_id = getattr(stream,"request_id",None) or getattr(msg,"_request_id",None)`(**설치된 SDK 소스로 확인할 것**); `base_url = str(client.base_url)`; `sdk_version = anthropic.__version__`. 타임아웃 세분화가 필요하면 `anthropic.Timeout`만 쓴다(`httpx` 객체 금지; SDK 2.x는 httpx2).
- 예외 체인(구체→일반): 치명 `AuthenticationError, PermissionDeniedError, NotFoundError, BadRequestError, UnprocessableEntityError` → `error="fatal:..."`; 일시 `RateLimitError, InternalServerError, APIStatusError(status≥500), APIConnectionError(APITimeoutError 포함)` → `error="transient:..."`. **in-stream error 이벤트(상태 200)**: 스트림 도중 SSE `event: error`로 오는 오류는 SDK(`_streaming.py`)가 응답 상태 그대로 `APIStatusError(status_code=200)`로 던지고 `max_retries`가 적용되지 않는다. 따라서 HTTP 상태가 4xx/5xx가 아닌 `APIStatusError`는 body `error.type`(`types/shared/error_type.py`)으로 분류한다: `overloaded_error`/`api_error`/`rate_limit_error`/`timeout_error`/알 수 없는 타입 → `transient`, `invalid_request_error`/`authentication_error`/`permission_error`/`not_found_error`/`billing_error` → `fatal`; 오류 문자열은 같은 형식에 msg 앞 `in-stream error event (<type>): `를 붙인다. 호출 안 재시도는 두지 않는다(transient는 다음 generate 실행에서 재호출). 어떤 경우에도 예외를 호출자에게 던지지 않는다(단 `preflight`는 던진다).
- **openai_compat**: `openai.OpenAI(base_url=os.environ[base_url_env], api_key=os.environ.get(api_key_env,"EMPTY"), timeout=, max_retries=)`; `chat.completions.create(model, messages=[system,user], temperature(있으면), seed(있으면), **{max_tokens_param: max_tokens})`; usage `prompt_tokens/completion_tokens`, `tokens_thinking = completion_tokens_details.reasoning_tokens`(있으면); `raw_stop_reason = finish_reason`, 정규화 `stop→end_turn, length→max_tokens, content_filter→refusal, 그 외 원문`.
- **dummy**: `harness/dummy_responses/<task>/<language>/<k>.md`(k = sample_idx mod 파일 수); tokens = 단어 수(가짜; provider 열이 dummy), latency 0, model_served="dummy", stop_reason="end_turn".

### generate.py
`PYTHON_EXE -m harness.generate [--config] [--run-id ID] [--provider P] [--model M] [--languages a,b] [--n N] [--tasks t] [--yes] [--force] [--no-retry-failed]`
- run_id 기본 `YYYYmmdd-HHMMSS-<provider>`. 프롬프트: system = `prompts/system.md`; user = 언어 파일 렌더링(그 안 `${task_spec}` = spec.md). `prompt_hash = sha256(system + "\n\x00\n" + user)[:16]`.
- 비-dummy: `preflight()` → 계획 출력(`<n> calls to <model> via <provider>, max_tokens=..., effort=...`) → `--yes` 없으면 `Proceed? [y/N]`; 비대화형(stdin 비tty)이고 `--yes` 없음 → exit 2. `harness_version`이 `nogit-*`이면 경고(원칙 4).
- 저장: `results/raw/<run_id>/<task>/<language>/<cond_tag>/sample_<i>.json`, `cond_tag = "args-{args}_hint-{grammar_hint}_fb-{feedback}"`. **건너뛰기 규칙**: 기존 파일이 있고 `error is None and stop_reason in {"end_turn","stop_sequence"}`이면 건너뜀; 그 외(오류·절단·refusal 아님)는 재호출(`--no-retry-failed`면 건너뜀); `--force`면 모두 재호출. 기존 파일은 `sample_<i>.prev<N>.json`으로 보관.
- 치명 오류(`error`가 `fatal:`)가 나오면 그 raw를 저장한 뒤 **즉시 중단**(exit 3).
- raw JSON 키: `run_id, task, language, backend, sample_idx, turn(0), conditions{args,grammar_hint,feedback}, provider, model_requested, model_served, sampling{temperature_sent, temperature_effective("provider_default"|값), thinking, effort, seed, top_p: null}, max_tokens, prompt{system,user}, prompt_hash, response_text, stop_reason, raw_stop_reason, stop_details_category, n_thinking_blocks, n_text_blocks, tokens_in, tokens_out, tokens_thinking, tokens_text, latency_ms, request_id, base_url, api_key_present, sdk_version, error, created_at(ISO 8601, 시간대 포함), harness_version, config_snapshot`.
- 끝에 요약: 호출/건너뜀/실패 수, tokens 합계.

## 6. 추출: extract.py

```python
@dataclass class ExtractResult:
    ok: bool; code: str | None; signature_ok: bool | None; signature_exact: bool | None
    detail: str   # "" | no_code_block | bare_code | unterminated_fence | function_not_found | param_mismatch:<설명> | static_linkage | syntax_error_deferred | multiple_blocks_chose:<n> | preprocess_failed
    n_blocks: int
def extract(response_text: str, *, backend: str, signature_spec: dict, args_mode: str = "positional", work_dir: Path | None = None) -> ExtractResult
```
1. 펜스 블록(```` ``` ````/`~~~`, CRLF 허용) 수집. 닫히지 않은 펜스는 그 뒤 텍스트 전체를 블록으로 보고 detail `unterminated_fence`.
2. 후보: python → 태그 ∈ {python, py, python3} 또는 무태그; c → 태그 ∈ {c, C} 또는 무태그(`cpp` 제외). 태그 있는 것 우선.
3. `gemm` 정의를 포함하는 후보만; 여러 개면 가장 긴 것(`multiple_blocks_chose:<n>`). 없으면 후보 중 가장 긴 것. 펜스가 없고 본문에 정의가 있으면 본문 전체(`bare_code`). 아니면 `no_code_block`.
4. 시그니처 검사:
   - python: `ast.parse` 실패 → ok=True, signature_ok=None, `syntax_error_deferred`. 최상위 `FunctionDef` `gemm`(데코레이터 허용). `AsyncFunctionDef` → signature_ok=False, `param_mismatch:async_def`. positional: `posonlyargs+args` 이름 == `[A,B,C,M,N,K]`, `kwonlyargs`·`vararg`·`kwarg` 없음, **기본값 있는 추가 매개변수도 불허**. keyword(M4): `kwonlyargs` == 목록, `args` 비어 있음.
   - c: `work_dir`가 주어지면 먼저 `gcc -E -std=gnu11 [-fopenmp] kernel.c`(procutil, 타임아웃 10 s, 코드 실행 없음)로 전처리하고 `# N "kernel.c"` 라인 마커로 kernel.c 유래 줄만 남긴 텍스트를 검사(주석·매크로 해소). 실패 시 원문에 주석 제거 후 검사하고 `preprocess_failed` 병기. 정규식은 `gemm` **정의**(`)` 뒤 공백·개행·`__attribute__((...))` 허용 후 `{`)를 찾는다; 선언만 있으면 `function_not_found`. 매개변수 목록 안의 괄호는 **3단계 중첩**까지 허용한다(`__attribute__((aligned(64)))`가 매개변수에 붙는 경우; 함수 포인터는 1단계). `static` 판정은 그 정의의 지정자 토큰열(직전 `;` 또는 `}` 이후)에서만 → `static_linkage`. 매개변수 6개: `__attribute__((...))` 제거, 검사 텍스트(전처리 결과 또는 주석 제거 원문)의 단순 `typedef <타입 토큰열> <이름>;`/`typedef <타입> *<이름>;`(구조체·열거형·함수 포인터·배열·다중 선언자 제외; 체인 8회 상한) 해소, `const/restrict/__restrict__/__restrict/volatile` 제거 후 — 반환형 `void`; 1·2: `float` 포인터(`*`/`[]`) 이름 `A`,`B`; 3: `float` 포인터 이름 `C`; 4·5·6: `int` 이름 `M`,`N`,`K`. `param_mismatch`는 개수·순서·기본 타입·이름 불일치에만(typedef는 해소된 타입으로 보고, 예 `double* expected float*`); 한정자·typedef 차이는 `signature_exact=False`로만(매개변수 속성은 `)` 뒤 속성과 같이 exact 판정에 영향 없음).
5. `ok = code is not None and signature_ok is not False`.

## 7. 빌드: procutil.py, build/c.py, build/python.py

### procutil.py (모든 서브프로세스의 단일 진입점)
```python
@dataclass class ProcResult: rc: int; timed_out: bool; elapsed_s: float; stdout_tail: str; stderr_tail: str; signal_name: str | None
def run_isolated(cmd: list[str], *, timeout_s: float, cwd: Path, env: dict[str,str] | None = None,
                 stdout_path: Path | None = None, stderr_path: Path | None = None, tail_chars: int = 2000) -> ProcResult
```
`Popen(start_new_session=True, stdout=파일 or PIPE, stderr=파일 or PIPE)` → `communicate(timeout)` → `TimeoutExpired`면 `os.killpg(os.getpgid(p.pid), SIGKILL)`(`ProcessLookupError` 무시) 후 `p.wait()`. rc<0이면 `signal_name = signal.Signals(-rc).name`. 파일로 리다이렉트하면 실행 후 꼬리만 읽는다. 단위 테스트: 타임아웃 뒤 `pgrep -g <pgid>`가 비어 있음(손자 프로세스 포함).

오케스트레이터 중단 시 정리(2026-11-04 보강; 워커는 새 세션이라 터미널의 Ctrl-C를 받지 않으므로 부모가 죽이지 않으면 멈춘 커널이 고정 코어에 남는다 — 실측): `Popen` 뒤 `TimeoutExpired` 외의 어떤 예외(`KeyboardInterrupt`=Ctrl-C, 시그널 핸들러의 예외, `SystemExit` 등 `BaseException`)로 빠져나가든 아직 거두지 않은 그룹을 SIGKILL·`wait()`한 뒤 예외를 그대로 올린다. 자식은 `preexec_fn`에서 `prctl(PR_SET_PDEATHSIG, SIGKILL)`을 건다(`pdeathsig_preexec`; exec를 거쳐 `taskset` → 워커까지 유지, 손자·setuid 바이너리 제외): 부모가 SIGKILL/OOM처럼 핸들러가 돌 수 없는 방식으로 죽어도 직접 자식은 커널이 죽인다. `install_terminate_handlers()`는 SIGTERM·SIGHUP을 `KeyboardInterrupt`로 바꾼다(이미 `SIG_IGN`이면 유지 → `nohup` 동작 보존; 주 스레드에서만). run.py가 `main`에서 호출하고 `KeyboardInterrupt`를 받으면 exit 130. 단위 테스트: 부모를 SIGINT/SIGTERM(핸들러 설치)/SIGKILL로 죽인 뒤 자식 그룹이 비어 있음.

### build/c.py
```python
@dataclass class BuildResult:
    ok: bool; artifact: Path | None; error_kind: str   # "" | compile_error | signature_mismatch | timeout
    error_detail: str; log: str; build_time_ms: float; command: list[str]; compiler: str; compiler_version: str
    n_warnings: int; undefined_symbols: list[str]; forbidden_symbols: list[str]; thread_symbols: list[str]; march_resolved: str
def build_c(code: str, out_dir: Path, bcfg: BuildCfg, *, openmp: bool, launch_prefix: list[str]) -> BuildResult
```
- `out_dir/kernel.c` 저장 → 명령 `launch_prefix + [cc, *base_flags, *cflags, (openmp_flag), *diag_flags, "-o", kernel.so, kernel.c, *ldflags]`(P-core에서 컴파일 → `-march=native` 해석 고정) → `run_isolated(timeout=compile_timeout_s, stderr_path=build.log)`. 실패 시 `error_detail` = `error:` 포함 첫 3줄; 타임아웃 → `timeout`(error_kind) — run.py는 `compile_error`/detail `compile_timeout`으로 기록.
- 성공 시 `nm -D --defined-only kernel.so`를 줄 단위 파싱: 세 번째 토큰이 정확히 `gemm`이고 타입 `T`/`W`인 줄 없음 → `signature_mismatch`/`symbol gemm not exported`. `nm -D --undefined-only`로 `undefined_symbols`; 그중 `forbidden_symbols` 매치 → `forbidden_symbols`, `thread_symbols_prefix` 매치 → `thread_symbols`(`-Wl,--no-undefined`라 libc/libm/libgomp 밖 심벌은 이미 링크 오류).
- `compiler_version` = `cc --version` 첫 줄; `march_resolved` = `cc -march=native -Q --help=target`의 `-march=`/`-mtune=` 값(모듈 캐시). `out_dir/build.json`에 BuildResult 저장.

### build/python.py
`build_python(code, out_dir) -> BuildResult`: `kernel.py` 저장 → `compile(code, "kernel.py", "exec")`만(구문 오류 → `compile_error`, detail `SyntaxError: ...`). **임포트 검사는 하지 않는다**(워커 `--mode import`가 한다, §8). artifact = kernel.py.

## 8. 실행·측정: worker.py, run.py

### worker.py (LLM 코드를 실행하는 유일한 곳)
```
PYTHON_EXE -m harness.worker --backend {c,python} --artifact PATH --mode {import,check,bench}
    --tests tiny.npz,rect.npz,small.npz --tolerance tasks/gemm/tolerance.yaml --out result.json
    [--check-repeats 2] [--warmup 3 --repeats 10 --min-repeats 3 --budget-s 300] [--args-mode positional] [--max-mem-gb 8]
```
- 시작: `prctl(PR_SET_DUMPABLE=0)`(ctypes `libc.prctl(4,0,0,0,0)`; 코어 덤프/apport 차단), `resource.setrlimit(RLIMIT_AS, max_mem_gb)`(RLIMIT_NPROC은 사용자 전체 기준이라 걸지 않음). 적용값을 결과에 `limits`로 기록. `blas = blas_identity()`(§9) 기록.
- 결과 파일은 **매 커널 호출 직전**과 매 테스트/호출 뒤에 원자적으로(tmp → `os.replace`) 갱신: `{"complete": bool, "in_progress": {"test","call_idx"}, ...}`. 워커 자체의 예외(하네스 버그)는 최상위 핸들러가 `worker_error`(traceback)로 기록하고 **rc=3**으로 종료.
- import 모드: 로드만(아래) → `{"mode":"import","load_ok","load_error","classification"}`.
- 로드: c → `ctypes.CDLL(artifact)`; `OSError`(undefined symbol 등) → `load_ok=false`, classification `compile_error`; `AttributeError`(gemm 없음) → `signature_mismatch`; `argtypes=[POINTER(c_float)]*3+[c_int]*3`, `restype=None`. python → `importlib.util.spec_from_file_location("kernel", artifact)` + `exec_module`; 예외 → `compile_error`(detail 예외 한 줄); `fn=module.gemm` 없음/`not callable`/`inspect.iscoroutinefunction` → `signature_mismatch`.
- 버퍼: 모든 배열은 **64바이트 정렬 + 가드 영역**으로 만든다. `alloc_aligned(n, guard) -> (view, base, guard_view)`: `base = np.empty(n + 16 + guard, np.float32)`, `off = (-base.ctypes.data % 64)//4`, `view = base[off:off+n]`, `guard_view = base[off+n:off+n+guard]`(센티널 채움). A: n=M·K, guard 1024; B: n=K·N, guard 1024; C: n=M·N, guard = max(M,N)² − M·N + 1024. python 백엔드에는 `view.reshape(M,K)` 등 2-D 뷰를, c 백엔드에는 `view.ctypes.data_as(POINTER(c_float))`를 넘긴다. `A0=A.copy(), B0=B.copy()` 보관; **매 호출 전** `A[...] = A0; B[...] = B0`로 복원하고 C는 센티널(`tolerance.sentinel`)로 채운다(check 모드 매 호출; bench 모드는 첫 warmup 전 1회).
- 호출 후 공통 검사: `inputs_mutated = not(array_equal(A,A0) and array_equal(B,B0))`; `oob_write = 가드 영역 중 센티널이 아닌 원소 존재`(A/B/C 각각); python 백엔드는 반환값 타입명 `return_type` 기록. Python 예외/`SystemExit`/`KeyboardInterrupt`는 `BaseException`으로 잡아 `exception={type,message,traceback 끝 2000자}` → 그 테스트 `runtime_crash`, 이후 테스트 중단. `message = str(e)`이되 예외 객체의 `__str__` 자체가 예외를 던지면(LLM 코드) `<unprintable Type>`으로 대체한다(로드 실패의 한 줄 detail도 같음) — 핸들러가 다시 죽어 `worker_error`(rc=3, run 중단)가 되면 안 된다.
- check 모드: 테스트마다 `check_repeats`회 호출 → 각 호출 `classify_output(C, C_ref, A0, B0, M, N, K, tol, oob_write, return_type, out_full=C.full)`(§10; `out_full`은 가드 포함 전체 C 버퍼로 `swapped_MN_dims` 판정에 쓴다). 판정이 `pass`인데 `inputs_mutated`/`oob_write`면 그 호출은 `wrong_output`/`inputs_mutated`·`out_of_bounds_write`로 기록한다(`inputs_mutated`는 `wrong_output`이 아닌 모든 판정을 덮고, `oob_write`는 `pass`만 덮는다 — `arg_order`는 §10-3대로 유지). bench의 `final_check`도 같은 규칙이다. 두 호출의 출력 비교는 `nondeterministic = not np.allclose(out1, out2, rtol, atol, equal_nan=True) or (isfinite 마스크 불일치)`. 결과: `{"mode":"check","backend","artifact","load_ok","load_error","blas","limits","classification","error_detail","inputs_mutated","oob_write","nondeterministic","tests":[{"name","M","N","K","runs":[{"classification","pattern","max_abs_err","max_tol_ratio","n_unwritten","nan_inf","elapsed_ms","exception","return_type"}],"classification","nondeterministic"}],"max_abs_err","max_tol_ratio","loadavg_start","loadavg_end","complete"}`.
- bench 모드(테스트 1개=large): 센티널 채움 1회 → warmup 호출 → timed 호출. 각 호출 `perf_counter_ns`. **적응·조기 종료**: 첫 warmup 시간 `t1`에 대해 `t1·(1+min_repeats) > 0.8·budget_s`이면 즉시 `classification="timeout"`, `error_detail=f"first_call_{t1:.0f}ms_projected_{...}s"`로 기록하고 종료; 아니면 `t1·(warmup+repeats) > 0.8·budget_s`일 때 `warmup=1, repeats=clamp(floor(0.8·budget_s/t1)−1, min_repeats, repeats)`, `repeats_reduced=true`. 결과의 `warmup`/`repeats`는 **계획값**이다(축소 시 축소된 값으로 갱신; 즉시 timeout·크래시로 끝나면 그대로 남는다) — 실제 호출 수는 `warmup_ms`/`timed_ms`의 길이다(2026-11-04 명료화). 첫 timed 결과 복사본 `C_first`; 이후 timed 호출마다(계측 밖에서) `bitwise_equal = array_equal(C, C_first)`, 아니면 `compare(C, C_first, rtol, atol)`로 허용오차 초과 여부 → `nondeterministic_runs`(허용오차 초과 수), `nondeterministic_bitwise_runs`(비트 불일치 수). 각 timed 호출 직후 고정 코어들의 `scaling_cur_freq`(kHz→MHz)를 읽어 리스트에 추가(`os.sched_getaffinity(0)` 기준 — **커널 로드 전에** 읽어 둔 값(`pinned_cpus`)을 쓴다. libgomp는 `OMP_PROC_BIND=true`면 로드 시점에 주 스레드를 place 0에 묶어 로드 후의 친화도가 CPU 1개로 좁아진다(실측 2026-11-04)). 시작·끝에 `core_throttle_count` 합, `x86_pkg_temp` 온도, `os.getloadavg()[0]` 기록. 마지막 호출 후 `final_check = classify_output(...)`, `inputs_mutated`, `oob_write`. 통계는 timed만: median, mean, min, max, cv(ddof=1). 결과: `{"mode":"bench","test","M","N","K","load_ok","blas","limits","alignment":64,"warmup","repeats","repeats_reduced","warmup_ms":[...],"timed_ms":[...],"median_ms","mean_ms","min_ms","max_ms","cv","gflops"(2MNK/median_s/1e9),"nondeterministic_runs","nondeterministic_bitwise_runs","cpu_mhz_samples":[...],"cpu_mhz_median","cpu_mhz_min","throttle_events","temp_c_start","temp_c_end","loadavg_start","loadavg_end","final_check":{...},"inputs_mutated","oob_write","classification","error_detail","exception","working_set_mb","complete"}`.
- 워커는 stdout에 아무것도 쓰지 않는다(커널의 stdout은 run.py가 파일로 받는다). 입력 배열은 절대 덤프하지 않는다.

### run.py (오케스트레이션)
`PYTHON_EXE -m harness.run --run-id ID [--config] [--languages a,b] [--samples 0,1] [--skip-bench] [--no-baseline] [--force] [--strict-protocol] [--rebuild-csv]`
1. 설정·machine.yaml 로드. `protocol.check_static()` 1회(거버너·EPP·turbo·taskset·BLAS 정체). 플래그는 stderr 경고(해결 명령 포함)와 함께 계속; `--strict-protocol`이면 **이 정적 검사의** 정보용 외 플래그가 있을 때 중단(exit 4) — strict의 효력은 여기까지이고 §8-7의 동적 플래그·drift·`paired_baseline_failed`, §8-10의 `model_served_mixed`는 strict여도 기록만 한다(2026-11-04 명료화; 일시적 부하로 run 전체가 죽는 대신 행 단위 `protocol_ok=0`으로 가려낸다). BLAS 이름에 `openblas`가 없으면 **항상** 중단.
2. 테스트 npz 없으면 `gen_tests.py`를 `run_isolated`로 실행. manifest에서 `test_seed`, `test_hash`.
3. run 시작 기준선: `baseline.py`를 python 백엔드로 `worker --mode bench`(threads=measurement.threads) → `baseline_run_time_ms`, `baseline_run_gflops`; `final_check`가 pass가 아니거나 `gflops < baseline_min_gflops`면 중단(원인 안내: 참조 BLAS/격리 venv). 1스레드로 1회 더 → `baseline_time_1t_ms`. 저장 `results/built/<run_id>/<task>/baseline/{bench_run.json,bench_1t.json}`. `--no-baseline`이면 기존 파일 재사용(없으면 오류).
4. raw 순회(필터 적용). 샘플 디렉터리 `results/built/<run_id>/<task>/<language>/<cond_tag>/sample_<i>/`. `result.json`이 있고 그 안의 `raw_sha256`이 현재 raw 파일 해시와 같고 `--force`가 아니면 건너뜀.
5. **생성 게이트**(extract 전): raw의 `error`가 non-null → 행 `error_stage=generate, error_type="", passed=0, valid_sample=0, api_error=error`; `stop_reason=="max_tokens"` → `error_stage=generate, error_detail=truncated, valid_sample=0`; `stop_reason=="refusal"` → `error_stage=generate, error_detail=refusal:<category>, valid_sample=1, passed=0, error_type=""`. 이 행들은 extract/build/실행을 하지 않는다.
6. 단계: extract → build → import(워커) → check → bench. 워커 실행: `protocol.launch_prefix(cfg, threads) + [PYTHON_EXE, "-m", "harness.worker", ...]`, `env=protocol.launch_env(cfg, threads, openmp=backend=="c" and lang.openmp)`, stdout/stderr는 샘플 디렉터리의 `<stage>.stdout.log`/`<stage>.stderr.log`로(꼬리 2000자를 result에, `stdout_bytes` 기록). 종료 규칙: timed_out → `timeout`; rc<0 → `runtime_crash`, detail `signal <name> during <in_progress.test> call <call_idx>`; rc==3 또는 결과에 `worker_error` → **harness_error**: 행을 쓰지 않고 stderr에 traceback 출력 후 run 중단(exit 5); rc==0이고 `complete`가 아님 → `runtime_crash`, detail `worker exited without complete result`; 그 밖의 rc>0 → `runtime_crash`, detail `rc=<n>: <stderr 꼬리>`. import 단계(워커 `--mode import`, 상한 `check_timeout_s`)에 이 종료 규칙이 적용되면 그 행의 `error_stage`는 **`build`**다(enum에 `import`가 없음; 로드 실패 `compile_error`/`signature_mismatch`와 같은 자리 — 2026-11-04 명료화).
7. bench 직전(통과 샘플마다): `protocol.check_dynamic()`(1 s; load_1m, pinned_busy_pct, sibling_busy_pct, governor 재확인) → **대응 기준선** `baseline.py` bench(같은 설정, 결과 `baseline/paired_sample_<lang>_<i>.json`) → `baseline_time_ms`(대응값), `baseline_drift_pct = (paired − run_start)/run_start·100`, 초과 시 protocol_flags `baseline_drift=<±x.x>%`; 대응 기준선 bench 자체가 실패(게이트 불통과·크래시)하면 경고 + 플래그 **`paired_baseline_failed`**, `baseline_time_ms`·`speedup_vs_baseline`은 NA(샘플 bench는 계속) → 샘플 bench. 동적 플래그·drift·`paired_baseline_failed`는 `--strict-protocol`이어도 기록만 한다(§8-1; 2026-11-04 명료화).
8. `pass` 정의: `passed = (check 모든 테스트 pass) ∧ ¬inputs_mutated ∧ ¬oob_write ∧ (bench 미실행 ∨ (bench final_check pass ∧ 크래시 없음 ∧ nondeterministic_runs == 0))`. **bench timeout만** `passed=1`·시간 NA·`error_stage=bench, error_type=timeout`. 그 외 bench 실패는 `passed=0`.
9. 행(§11) → `result.json`(행 + 단계별 상세 + `raw_sha256`) 저장 → `schema.upsert_row(csv, row)`. CSV는 raw의 provider가 dummy면 `paths.results_csv_dummy`, 아니면 `paths.results_csv`.
10. 끝에 run 요약(언어별 pass/오류 유형/생성 실패/플래그 수, `model_served`가 단일값이 아니면 경고 + 플래그 `model_served_mixed`). `--rebuild-csv`는 built_dir의 모든 result.json에서 CSV를 다시 쓴다(result.json이 진실의 원천).
11. 중단: `main`이 `procutil.install_terminate_handlers()`로 SIGTERM·SIGHUP을 `KeyboardInterrupt`로 받고(Ctrl-C는 Python 기본), 실행 중이던 워커 그룹은 `run_isolated`가 SIGKILL한다(§7). 처리 중이던 샘플의 행은 쓰지 않고 exit 130; 재실행하면 §8-4의 `raw_sha256` 규칙으로 완료된 샘플만 건너뛴다.

## 9. 측정 규약: protocol.py

```python
@dataclass class StaticStatus:  governor: str; governor_ok: bool; epp: str; epp_ok: bool; no_turbo: int|None; taskset_available: bool; cores: str; single_core: str; threads: int; blas: dict; blas_ok: bool; flags: list[str]; ok: bool; checked_at: str
@dataclass class DynamicStatus: load_1m: float; load_ok: bool; pinned_busy_pct: float; sibling_busy_pct: float; busy_ok: bool; governor: str; governor_ok: bool; flags: list[str]; ok: bool; checked_at: str
def check_static(cfg, machine) -> StaticStatus      # run당 1회
def check_dynamic(cfg) -> DynamicStatus             # 1초(/proc/stat 2회 샘플); bench 직전마다
def launch_prefix(cfg, threads: int) -> list[str]   # threads==1 → ["taskset","-c",single_core], 아니면 cores; taskset 없으면 [] + 플래그
def launch_env(cfg, threads: int, *, openmp: bool) -> dict[str,str]
    # 화이트리스트 환경(PATH, HOME, LANG, LC_ALL=C, TMPDIR, VIRTUAL_ENV, PYTHONHASHSEED=0)에서 시작, 상속하지 않은 OMP_*/GOMP_*/OPENBLAS_*/MKL_*/KMP_*가 사용자 환경에 있었으면 stderr 경고(플래그 env_overrides_dropped).
    # OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=threads; openmp=True면 measurement.omp_env_extra 추가.
def blas_identity() -> dict   # 현재 프로세스에서: numpy.show_config(mode="dicts")의 blas name/version/openblas configuration + /proc/self/maps에서 로드된 *blas*/*openblas* .so 경로. PYTHON_EXE 서브프로세스로 돌리는 헬퍼 blas_identity_subprocess()도 제공.
def probe_machine() -> dict                         # §2 machine.yaml 키 (blas·sdk 버전은 서브프로세스로)
def environment_fingerprint(cfg, machine, static: StaticStatus) -> tuple[dict, str]   # cpu_model, kernel, glibc, cc 버전, march_resolved, python, numpy, numpy_blas, sdk 버전, base_flags+cflags+openmp_flag+ldflags, cores, single_core, threads, omp_env, governor, epp, no_turbo → (dict, sha256[:16])
```
- 거버너·EPP: 고정 코어 각 CPU의 `scaling_governor`, `energy_performance_preference`; 전부 `performance`여야 ok. 아니면 경고 + 해결 명령(`sudo cpupower frequency-set -g performance`; EPP는 `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference`).
- 부하: `os.getloadavg()[0]`; `/proc/stat`의 `cpu<n>` 줄을 1.0 s 간격 2회 읽어 고정 코어 평균 `pinned_busy_pct`와 형제 코어(`topology/thread_siblings_list`) 평균 `sibling_busy_pct`(idle = idle+iowait).
- flags 예: `governor=powersave`, `epp=power`, `turbo=on`(정보용, ok에는 불포함), `env_overrides_dropped`(정보용), `load_1m=2.4>2.0`, `pinned_busy=23.1%>10%`, `sibling_busy=45.0%>10%`, `taskset_missing`, `blas=reference`. `ok = not flags(정보용 제외)` (`flags_ok(flags)`; `is_informational`은 접두 `turbo=`·`env_overrides_dropped`만). run.py가 덧붙이는 `baseline_drift=<±x>%`, `paired_baseline_failed`, `model_served_mixed`(§8-7, §8-10)도 정보용이 아니므로 행의 `protocol_ok = flags_ok(protocol_flags)`를 0으로 만든다(2026-11-04 명료화).
- CLI: `--check`(static+dynamic JSON), `--probe [--write config/machine.yaml]`.

## 10. 오류 분류 규칙 (harness/classify.py; 고정 — 바꾸면 docs/error_taxonomy.md 갱신)

```python
def compare(out, ref64, rtol, atol) -> dict   # {"ok","nan_inf","max_abs_err","max_tol_ratio"(= max |out−ref|/(atol+rtol|ref|), 유한 원소 기준),"n_fail","max_abs_ref"}
def classify_output(out, ref64, A0, B0, M, N, K, tol: Tolerance, *, oob_write: bool = False, return_type: str | None = None, out_full=None) -> tuple[str, str, dict]
    # (status, pattern_or_detail, metrics); status ∈ pass | wrong_output | numeric_instability | arg_order
    # metrics = {"max_abs_err","max_tol_ratio","n_fail","nan_inf","n_unwritten","n_total","max_abs_ref","oob_write"} (판정과 무관하게 항상 기록)
```
순서(먼저 맞는 규칙이 결정; v2.1 2026-11-04: arg_order 패턴을 센티널 규칙보다 먼저 본다 — 그렇지 않으면 `swapped_MN_dims`는 partial_write/out_of_bounds_write에 가려 도달 불가):
1. NaN/Inf(센티널 제외) → `numeric_instability`/`nan_inf`.
2. 허용오차 내 → `pass`.
3. arg_order 패턴(같은 rtol/atol로 매치되면 → `arg_order`/패턴명; 단순한 가설부터):
   - `transposed_output`: `out.reshape(N, M).T ≈ ref`.
   - `swapped_operands`: M==N==K일 때 `out ≈ B0 @ A0`. (정사각에서는 아래 `column_major_all`과 대수적으로 동일하므로 먼저 검사해 이 이름으로 보고한다; `docs/error_taxonomy.md`에 모호성 명시.)
   - `column_major_all`: `A_cm = A0.reshape(K, M).T`, `B_cm = B0.reshape(N, K).T`, `out.reshape(N, M).T ≈ A_cm @ B_cm`.
   - `swapped_MN_dims`: M≠N일 때, `C[i*M + j]` 가정 — 가드 포함 전체 C 버퍼(`out_full`, 길이 ≥ max(M,N)²)에서 i<M, j<N의 idx=i*M+j 위치가 ref[i,j]와 맞는 비율 ≥ 99% (워커가 `out_full`을 kwarg로 넘긴다; 없으면 생략). 매치되면 센티널/OOB가 남아 있어도 `arg_order`로 보고하고 `oob_write`·`n_unwritten`은 metrics에 그대로 기록한다.
4. 센티널·계약 검사(이 순서대로): `n_unwritten = count(out == sentinel)`. (a) 전부 센티널 → `wrong_output`/`output_not_written`(python이고 `return_type`이 ndarray면 `returned_array_not_written`); (b) `compare(out − sentinel, ref)` 통과 → `wrong_output`/`accumulated_into_c` — **부분 기록보다 먼저** 본다: float32에서 `sentinel + ref`는 |ref| < 2⁻¹³이면 정확히 센티널로 되돌아가 `n_unwritten`에 세어지므로(실제 small/large 형태에서 실측) 순서가 반대면 `C += A@B`가 `partial_write:1/65536`이 된다. 진짜 부분 기록은 모든 원소가 `sentinel + ref`와 일치할 수 없어 이 검사에 걸리지 않는다; (c) 일부 센티널 → `wrong_output`/`partial_write:<n_unwritten>/<M·N>`; (d) `oob_write` → `wrong_output`/`out_of_bounds_write`. 출력이 허용오차 내이면 규칙 2가 먼저이므로 `oob_write`만 있는 호출은 `classify_output`에서 `pass`다 — 워커가 호출 수준에서 `wrong_output`/`out_of_bounds_write`로 덮고(§8), 샘플 수준은 `aggregate_sample(oob_write=)`과 run.py의 pass 정의가 플래그를 본다.
5. `max_tol_ratio ≤ numeric_pattern_tol_ratio`(모든 원소가 허용오차의 20배 이내) → `numeric_instability`/`imprecise`.
6. 그 밖 → `wrong_output`/`mismatch`.

샘플 수준 합산(check 테스트들 + bench final_check):
1. 어떤 테스트든 예외/크래시 → `runtime_crash`.
2. 어떤 테스트든 `nondeterministic`(허용오차 초과 불일치; 비트만 다르고 허용오차 안이면 아님) → `race_condition`.
3. 실패한 테스트 **전부**가 arg_order이고 같은 패턴 → `arg_order`.
4. 어떤 테스트든 `wrong_output` → `wrong_output`(inputs_mutated·oob_write 포함).
5. 실패가 전부 numeric → `numeric_instability`.
6. 그 밖(패턴이 섞인 경우: arg_order 패턴이 둘 이상이거나 arg_order + numeric) → 보수적으로 `wrong_output`/`mixed:<classification/pattern,…>`(정렬·쉼표 구분; 2026-11-04 명료화 — 구현은 처음부터 이렇게 동작했고 문서만 없었다). 정사각 모호성(3b/3c) 때문에 열우선 커널은 tiny·rect `column_major_all` + small `swapped_operands`로 이 경로에 떨어진다(`docs/error_taxonomy.md` §2 주; 처리 방안은 decisions.md 미결).
7. 전부 pass → `pass`.

프로세스 수준(run.py): 생성 게이트(§8-5) → `error_stage=generate`(error_type 빈칸; detail `api_error`/`truncated`/`refusal:<category>`); 추출 실패 → `signature_mismatch`(실패 사유 detail: no_code_block/function_not_found/param_mismatch:<…>/static_linkage — `unterminated_fence`·`bare_code`·`multiple_blocks_chose:<n>`·`syntax_error_deferred`·`preprocess_failed`는 `ok`를 바꾸지 않는 정보 노트로 `;` 병기, 2026-11-04 명료화); 빌드 실패 → `compile_error`(심벌 미노출은 `signature_mismatch`); 워커 로드(import 모드) 실패 → `error_stage=build`, `compile_error`/`signature_mismatch`, 같은 워커의 timeout(`check_timeout_s`)·시그널도 `error_stage=build`로 `timeout`/`runtime_crash`(§8-6); 시그널 종료 → `runtime_crash`; 시간 초과 → `timeout`(detail `worker timeout after <t>s during <test> call <idx>`). `memory_error`는 M5(ASan)에서만.
관찰 열(분류표 밖, 분류에 영향 없음): `contract_violation`(JSON 리스트: `threads_in_c_seq`, `stdout_output`, `forbidden_symbol:<s>`).

## 11. 결과 스키마: schema.py

`RunRow` dataclass(필드명 `passed`; CSV 열 이름은 `pass` — `FIELD_TO_COLUMN = {"passed": "pass"}` 와 역매핑). 직렬화: 불리언 `1`/`0`, NA 빈 문자열, 리스트/딕트 JSON 문자열. 행 키 `ROW_KEY = (run_id, task, language, args, grammar_hint, feedback, sample_idx, turn)`. 열 순서(고정):

```
run_id, created_at, task, language, backend, args, grammar_hint, feedback, sample_idx, turn,
provider, model, model_served, sampling, effort, max_tokens, temperature, seed, prompt_hash, stop_reason, stop_details_category,
tokens_in, tokens_out, tokens_thinking, tokens_text, n_thinking_blocks, latency_ms, api_error, valid_sample,
extract_ok, signature_ok, signature_exact, build_ok, build_time_ms, n_warnings, undefined_symbols, contract_violation,
pass, turns_to_pass, error_type, error_stage, error_detail, inputs_mutated, oob_write, nondeterministic_runs, nondeterministic_bitwise_runs,
check_max_abs_err, check_max_tol_ratio, check_tests,
bench_ran, time_median_ms, time_mean_ms, time_min_ms, time_max_ms, time_cv, time_samples_ms, warmup, repeats, repeats_reduced, gflops,
baseline_time_ms, baseline_time_cv, baseline_samples_ms, baseline_run_time_ms, baseline_time_1t_ms, baseline_drift_pct, speedup_vs_baseline, speedup_vs_baseline_1t,
cpu_mhz_median, cpu_mhz_min, throttle_events, temp_c_start, temp_c_end,
compiler, compiler_version, cflags, march_resolved, threads, cores, omp_env,
governor, governor_ok, epp, no_turbo, load_1m, load_ok, pinned_busy_pct, sibling_busy_pct, taskset_used, protocol_ok, protocol_flags, protocol_checked_at,
hostname, cpu_model, python_version, numpy_version, numpy_blas, sdk_version, test_seed, test_hash, env_hash,
asan_findings, ubsan_findings, lint_findings_total, lint_memory, lint_numeric, lint_concurrency, lint_shape_dtype, lint_api_misuse, lint_performance, perf_summary, roofline_pct,
raw_path, built_path, harness_version
```
- `error_stage ∈ {"", generate, extract, build, check, bench}`; `error_type ∈ {"", compile_error, runtime_crash, timeout, wrong_output, numeric_instability, arg_order, memory_error, race_condition, signature_mismatch}`.
- `cflags` = base_flags+cflags(+openmp_flag)+ldflags 전체 문자열. `tokens_out`은 **thinking 포함 총 출력 토큰(과금 기준)**; `tokens_text`는 텍스트 블록 몫(둘 다 있을 때). `speedup_vs_baseline = baseline_time_ms / time_median_ms`(대응 기준선), `speedup_vs_baseline_1t = baseline_time_1t_ms / time_median_ms`. `turns_to_pass`: pass면 0(M1), 아니면 NA. `check_tests`: 테스트별 `{name, classification, pattern, max_tol_ratio, elapsed_ms, nondeterministic}` JSON. `time_samples_ms`: `{"warmup":[...],"timed":[...]}` JSON(timeout 행도 부분값 기록). 안전성·perf_summary·roofline 열은 M1에서 NA.
- API: `COLUMNS`, `row.to_csv_dict()`, `RunRow.from_csv_dict()`, `upsert_row(csv_path, row)`(헤더 없으면 쓰고, 있으면 **열 일치 검사**(불일치면 예외+`--rebuild-csv` 안내); 같은 `ROW_KEY` 행은 교체 — 전체를 tmp에 다시 쓰고 `os.replace`), `rebuild_csv(built_dir, csv_path)`, `read_rows(csv_path) -> list[dict]`(키 중복 시 `created_at` 최신만, 경고). 테스트: `list(to_csv_dict().keys()) == COLUMNS`.
- `harness_version`: `git describe --always --dirty`; git이 없으면 `nogit-<sha256(harness/**/*.py ∥ analysis/*.py ∥ prompts/** ∥ tasks/gemm/{spec.md,signatures.yaml,tolerance.yaml} ∥ config/experiment.yaml)[:12]>`; git이 있어도 `-dirty+<hash12>` 병기.

## 12. 집계: analysis/aggregate.py

`PYTHON_EXE -m analysis.aggregate [--csv results/runs.csv] [--run-id ID | --latest(기본)] [--all-runs] [--json]`
- pandas(`dtype=str, keep_default_na=False` 후 수동 변환). 키 중복 제거(최신 created_at). 그룹 = (run_id, provider, model, task, language, args, grammar_hint, feedback, turn).
- 열: `n`(valid_sample==1), `n_pass`, `pass_at_1`(= n_pass/n), `pass_at_k`(k=1,5,n; `1 − C(n−c,k)/C(n,k)`, k>n이면 NA), `n_generate_failed`, `n_refusal`, `n_truncated`, `benchmarked`, `n_repeats_reduced`, `n_protocol_flagged`, `time_median_ms`(pass 샘플의 중앙값), `speedup_median/min/max`, `speedup_1t_median`, `tokens_out_mean`, `tokens_text_mean`, `tokens_per_pass`(= Σ(tokens_in+tokens_out)/n_pass), `tokens_out_per_pass`, `error_types`("유형:수, …"; passed=0 행만).
- run 단위 요약: 기준선(run 시작/1스레드/대응 평균), `model_served`·`effort`가 단일값인지(아니면 경고). 표는 `to_string()`; `--json`이면 JSON. 부트스트랩 CI는 M6.

## 13. 더미 응답 (harness/dummy_responses/gemm/)

실제 응답처럼 짧은 설명 + 펜스 코드 블록. `README.md`에 기대 결과표(검증 기준):

| language | k | 내용 | 기대 (stage / error_type / detail) |
|---|---|---|---|
| numpy | 0 | `np.matmul(A, B, out=C)` | pass, bench ok |
| numpy | 1 | `def gemm(A, B, M, N, K): return A @ B` | extract / signature_mismatch / param_mismatch |
| numpy | 2 | `C[:] = (A @ B) * 1.01` | check / numeric_instability / imprecise (max_tol_ratio ≈ 10) |
| numpy | 3 | 순수 Python 3중 루프(정확) | pass; bench / timeout (n≥4일 때만) |
| numpy | 4 | `return A @ B` (C 미기록, 시그니처 정상) | check / wrong_output / returned_array_not_written (n≥5) |
| c_seq | 0 | i-k-j 루프, `restrict` | pass |
| c_seq | 1 | 세미콜론 누락 | build / compile_error |
| c_seq | 2 | `C[j*M + i] = ...`(전치 저장; 모든 형태에서 범위 안) | check / arg_order / transposed_output |
| c_openmp | 0 | `#pragma omp parallel for` (i 루프) | pass |
| c_openmp | 1 | NULL **저장** `float *p = NULL; *p = 1.0f;`(읽기만 하면 -O3가 제거함) | check / runtime_crash / signal SIGSEGV |
| c_openmp | 2 | k 루프 병렬화 + 공유 누산 | check / race_condition (실측: rect·small은 항상 실행 간 불일치; tiny(K=3)는 타이밍에 따라 불일치가 나기도 안 나기도 함(2026-11-04 11:03 맞음, 13:26 이후 7/7 불일치) — detail의 테스트 목록은 고정 기대값이 아님) |

## 14. CLI 요약·Makefile

```
make venv        # python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make tests       # gen_tests.py
make probe       # harness.protocol --probe --write
make unittest    # PYTHON_EXE -m unittest discover -s tests
make dummy       # RUN_ID=dummy-<timestamp>: generate --provider dummy → run → aggregate --csv results/runs_dummy.csv
make real RUN_ID=… [STRICT=1]   # generate(확인 프롬프트) → run [--strict-protocol] → aggregate
make aggregate
```
모든 python은 `.venv/bin/python`.

## 15. 코딩 규칙

- Python 3.12+ 문법(설치 3.14). 타입 힌트, `dataclasses`, `pathlib`. 외부 의존: numpy, yaml, pandas(analysis만), anthropic/openai(llm_client에서 지연 임포트).
- 모든 서브프로세스는 `procutil.run_isolated`. **오케스트레이터 프로세스 안에서는 LLM 코드를 절대 임포트·실행하지 않는다**(`ast.parse`/`compile()`/gcc/nm만 허용); 실행은 `harness.worker` 서브프로세스에서만.
- 각 모듈은 `python -m harness.<모듈>`로 실행 가능한 `main()`(argparse). `harness/paths.py`의 `ROOT`, `PYTHON_EXE` 사용.
- 진단은 stderr에 `[모듈명] ...`. 예외를 삼키지 않는다. 결과 파일 쓰기는 원자적(tmp → `os.replace`).
- 테스트: `tests/test_<module>.py`(`unittest`). 최소: classify(더미 기대표의 numpy/2, c_seq/2 패턴, 센티널 규칙, 우선순위), extract(펜스·시그니처 사례), schema(열 일치·upsert 교체), procutil(타임아웃 후, 그리고 부모가 SIGINT/SIGTERM/SIGKILL로 죽은 뒤 프로세스 그룹 비어 있음), protocol(/proc/stat 파서).
- 주석·docstring 한국어, 식별자 영어. 테스트 입력·raw·built는 git에 올리지 않고 `results/runs.csv`(와 `runs_dummy.csv`)만 올린다.

## 16. M2 일반화 (2026-11-05)

§3·§8·§10의 GEMM 전용 서술은 다음으로 일반화됐다. 세부 근거는 `docs/decisions.md` "M2a".

- **과제 명세**: `tasks/<task>/signatures.yaml`에 `args`(`{name, kind: in|out|dim, dtype: float32|int32, shape: [식…]}`), `work`(연산 수 식), `out_guard_extra`, `baseline {backend: python|c, file, openmp, gate_blas, gate_min_gflops}`, 언어별 `param_docs`(프롬프트에 그대로 삽입). 로더는 `harness/taskspec.py`.
- **프롬프트 변수 추가**: `${function}`, `${param_docs}`.
- **테스트 npz**: 인자 이름의 입력 배열, `<out>_ref`(float64), dim 스칼라. 형태 이름은 모든 과제가 `tiny, rect, small, large`. 새 과제는 `harness/testgen.py`를 쓴다.
- **워커**: `--signatures`로 과제 명세를 받아 인자 순서대로 정렬·가드 버퍼를 만들고 호출한다. 입력 변조·가드 검사는 모든 배열(int32 포함)에 적용. 결과 문서에 `task`, `dims`.
- **분류**: `classify_task(task, …)` — GEMM은 기존 4개 패턴, 그 밖은 `classify_generic` + 2-D 전치 패턴. 규칙 1–6과 샘플 합산은 그대로.
- **기준선**: 과제 명세가 정한다. C 기준선은 run마다 샘플과 같은 플래그로 빌드해 워커로 측정한다. BLAS·gflops 게이트는 과제가 켠 경우에만.
- **drift**: run 끝에서 과제별 대응 기준선 중앙값 기준으로 계산한다.
- **제약**: 출력은 float32 배열 1개, 스칼라 인자는 `int`. 정수 출력·복수 출력은 Tier 2에서 추가한다.

## 17. Tier 2 커널 계약 (2026-11-08)

공통: §16의 과제 일반 계약을 그대로 따른다. 각 과제는 `tasks/<task>/{spec.md, signatures.yaml, tolerance.yaml, gen_tests.py, baseline/}` 와 `harness/dummy_responses/<task>/<lang>/0.md`(정답), `tests/test_tier2_<task>.py` 를 갖는다. `gen_tests.py` 는 `harness.testgen` 을 쓰고 형태 이름은 `tiny, rect, small, large`. `spec.md` 는 Tier 1 과 같은 구조(Definition / Data layout / Contract / Accuracy / Evaluation / Goal), 영어, 언어 중립, 라이브러리 무언급, 평가 형태·절차·허용오차 공개. `param_docs` 는 Tier 1 의 문체를 따른다(`tasks/dot/signatures.yaml` 참고).

**정수 출력**(histogram, scan, sort): `tolerance.yaml` 은 `rtol: 0.0, atol: 0.0, numeric_pattern_tol_ratio: 0.0, sentinel: -2048.0`, spec 은 "exact equality" 를 요구한다. 하네스는 int32 출력을 정확 일치로 판정한다(`compare` 는 tol=0 에서 틀린 원소의 배율을 inf 로 둬 `numeric_instability` 가 되지 않는다). 출력 버퍼는 센티널로 채워져 넘어가므로 **0 초기화는 커널의 책임**이다(spec 에 명시).

| 과제 | C 시그니처 | 입력 | 출력 | 형태 tiny / rect / small / large | 기준선 |
|---|---|---|---|---|---|
| `histogram` | `void histogram(const int* data, int* hist, int n, int nbins)` | `data` int32 n개, 값은 `[0, nbins)` 균등 | `hist` int32 nbins개, `hist[b]` = `data == b` 인 원소 수 | (n, nbins) = (13, 4) / (100003, 37) / (2²⁰, 256) / (2²⁵, 1024) | C+OpenMP: 스레드별 private 히스토그램 → 합산 |
| `scan` | `void scan(const int* x, int* out, int n)` | `x` int32 n개, 값은 `[0, 32)` 균등 | `out[i] = Σ_{k≤i} x[k]` (inclusive; 합 < 2³¹) | n = 7 / 100003 / 2²⁰ / 2²⁵ | C+OpenMP: 블록 합 → 오프셋 → 블록 내 스캔(2-pass) |
| `nbody` | `void nbody(const float* pos, const float* mass, float* acc, int n)` | `pos` float32 n×3 AoS(`pos[3*i+c]`), `[0,1)³` 균등; `mass` float32 n개, `[0.5, 1.5)` 균등 | `acc` float32 n×3: `acc_i = Σ_{j=0}^{n-1} mass[j]·(p_j − p_i) / (|p_j − p_i|² + EPS2)^{3/2}`, `EPS2 = 1e-4`, G = 1 (j = i 항은 정확히 0 이라 포함해도 된다) | n = 5 / 301 / 2048 / 16384 | C+OpenMP: i 병렬, j 내부 루프 스칼라 `1.0f/sqrtf`, AoS 그대로 |
| `sort` | `void sort(const int* x, int* out, int n)` | `x` int32 n개, `[0, 2³¹−1]` 균등(중복 가능) | `out` = `x` 의 오름차순 정렬 | n = 7 / 100003 / 2²⁰ / 10⁷ | NumPy `np.sort` (python; `gate_blas: false`) |

Python 시그니처는 같은 이름·순서(`def histogram(data, hist, n, nbins):` 등); 배열은 위 형태의 ndarray(`pos`/`acc` 는 `(n, 3)`), dim 은 `int`. 반환값 None.

세부:
- `histogram`/`scan`/`sort` 는 `work` 를 두지 않는다(gflops NA). `nbody` 는 `work: "20*n*n"`.
- `nbody` 허용오차: `rtol: 1.0e-3`, `atol` = 1e-3 × (large 참조 `acc` 성분의 RMS) 를 유효숫자 1자리로 반올림한 값(주석에 RMS 실측치 기록), `sentinel: -1.0e30`(가속도 값이 수천을 넘을 수 있어 -2048 은 쓰지 않는다). 구현자는 (a) 직진 float32 C 커널(`1.0f/sqrtf`)의 `max_tol_ratio` 가 모든 형태에서 0.2 이하임을, (b) 정제 없는 `_mm256_rsqrt_ps` 커널의 `max_tol_ratio` 를 측정해 `docs/decisions.md` 에 적는다. 참조는 float64 로 i 를 묶어(예: 256행) 정확히 계산한다(n=16384 에서 수십 초 허용).
- `nbody` 의 2-D 출력 `(n, 3)` 에는 공통 `transposed_output` 패턴이 적용된다(SoA 로 쓴 출력이 이 패턴에 걸린다 — 의도된 arg_order).
- 더미 응답: 과제마다 세 언어의 정답(k=0). `histogram/c_openmp/1.md` 는 보호 없는 `hist[data[i]]++` 병렬 루프(기대: `race_condition`; 결정적으로 틀리면 `wrong_output` — README 에 기록). `scan/c_openmp/1.md` 는 순차 의존을 무시한 `#pragma omp parallel for` 누적(기대: `race_condition` 또는 `wrong_output`).
- 테스트(`tests/test_tier2_<task>.py`): gen_tests 2회 바이트 동일, 참조 손계산 대조(tiny), python 정답 커널의 worker check pass(tiny·rect), 심은 버그(예: histogram 0 초기화 누락 → `accumulated_into_c`; scan exclusive 스캔 → `wrong_output`; sort 내림차순 → `wrong_output`; nbody SoA 출력 → `arg_order/transposed_output`) 분류.
- `config/experiment.yaml` 의 `run.tasks` 에 네 과제를 추가하고 `docs/error_taxonomy.md`·`README.md`·더미 README 의 표를 갱신한다.

검증(2026-11-08, 통합): `config/experiment.yaml` `run.tasks` 에 네 과제를 추가했고(Tier 1 유지, 8개), 단위 테스트 214개 통과(`tests/test_taskspec.py` 는 run.tasks 전체를 검사하도록 바꿈), 더미 run `dummy-tier2`(4과제 × 3언어 × 2샘플 = 24행, 354 s)가 기대표와 전부 일치했다 — 정답 22행 pass, 경쟁 변형 2행(`histogram`·`scan` c_openmp/1) `race_condition`; 멱등 재실행 0.26 s(재실행 0, 중복 0). 하네스 코드는 바꾸지 않았다. 기대표·실측은 `harness/dummy_responses/README.md`(과제 전체)·과제별 README·`docs/error_taxonomy.md` §8.2·부록 2, 구현자 실측과 이탈 사항은 `docs/decisions.md` "M2b" 절.

### §17 보강 (2026-11-08 리뷰 반영)
- 워커 버퍼: 앞쪽 가드 4096원소 추가, 입력 가드 값 분리(정수 -1073741824, 실수 -3e9), `AlignedArray.guards_intact()`.
- `Tolerance.atol_rms_rel`(선택) → `classify.effective_tolerance(tol, ref)`; nbody 는 `atol_rms_rel: 1e-3`, `sentinel: -1e6`, spec 에 8 GiB 주소 공간 한도 명시.
- 규칙 4c: 쓴 원소가 전부 맞을 때만 `partial_write`.
- `paths.atomic_write_json` 은 비유한 실수를 null 로 쓴다(`allow_nan=False`).

## 18. Tier 3 커널 계약 (2026-11-09)

공통 규칙은 §17 과 같다(파일 구성, `harness.testgen`, 형태 이름 `tiny/rect/small/large`, spec 구조·언어 중립, 더미 k=0 ×3 언어, `tests/test_tier3_<task>.py`). 실수 출력 과제는 형태별 atol(`atol_rms_rel`)을 쓴다. `work` 식에 `log2` 를 쓸 수 있다.

| 과제 | C 시그니처 | 입력 | 출력 | 형태 tiny / rect / small / large | 기준선 |
|---|---|---|---|---|---|
| `cholesky` | `void cholesky(const float* A, float* L, int n)` | `A` float32 n×n row-major, 대칭 양정치(생성: `B·Bᵀ + n·I`, B 는 uniform[-1,1) n×n, float64 로 만들어 float32 로 캐스트) | `L` float32 n×n 하삼각: `L·Lᵀ = A`, 대각 > 0, **엄격 상삼각은 0 으로 써야 함** | n = 6 / 131 / 512 / 2048 | NumPy `np.linalg.cholesky`(python; LAPACK potrf) |
| `floyd_warshall` | `void floyd_warshall(const float* dist_in, float* dist_out, int n)` | `dist_in` float32 n×n, 대각 0, 간선 가중치 uniform[1,10), 간선 없음 = `INF = 1.0e30`(밀도 0.3) | `dist_out` 모든 쌍 최단거리; 도달 불가는 **정확히 INF** | n = 5 / 37 / 256 / 1024 | C+OpenMP (k 순차, i 병렬, j 벡터화) |
| `bfs` | `void bfs(const int* row_ptr, const int* col_idx, int* dist, int n, int nnz)` | 방향 그래프 CSR(int32), 평균 차수 ≈ 16(행 길이 균등 0..32), 열 오름차순·유일 | `dist` int32: 정점 0 에서의 홉 수, 도달 불가 −1, `dist[0] = 0` | n = 7 / 1003 / 2¹⁶ / 2²¹ (nnz ≈ 16n) | C+OpenMP 레벨 동기 top-down(프론티어 병렬, 원자적 방문 표시) |
| `fft` | `void fft(const float* x, float* out, int n)` | `x` float32 길이 2n, 교차 저장 복소수(x[2j]=실수부, x[2j+1]=허수부), uniform[-1,1), n 은 2 의 거듭제곱 | `out` 길이 2n 교차 저장: `X_k = Σ_j x_j·e^{−2πi jk/n}` (정방향, 스케일 없음) | n = 8 / 1024 / 2¹⁶ / 2²² | NumPy `np.fft.fft`(python, complex64) |

Python 시그니처: `def cholesky(A, L, n):` (A, L 은 (n, n)), `def floyd_warshall(dist_in, dist_out, n):` ((n, n)), `def bfs(row_ptr, col_idx, dist, n, nnz):`, `def fft(x, out, n):` (x, out 은 (n, 2) float32 — 복소수 뷰는 구현의 선택).

세부:
- `cholesky`: tolerance `rtol 1e-3`, `atol_rms_rel 1e-3`, 센티널 -2048(L 원소 크기는 sqrt(n·3) 정도라 정당 범위 밖), `work "n*n*n/3"`, 참조는 float32 로 캐스트한 A 의 float64 Cholesky. spec 에 "A 는 대칭 양정치가 보장되며 대각 우세(조건수 작음)", "L 의 상삼각은 0 으로 채워야 한다(출력 전체 기록 규칙)" 명시. 직진 float32 구현(순차 j-k 루프)의 max_tol_ratio 를 측정해 ≤ 0.2 확인.
- `floyd_warshall`: tolerance `rtol 1e-4`, `atol 1e-3`(고정; 거리 1~수십 범위), 센티널 -2048, `work "2*n*n*n"`. `INF` 는 spec 에 상수로 공개(1.0e30; `INF + INF` 가 float32 에서 유한). 참조는 float64 로 k 루프를 numpy 벡터화. 생성: 모든 정점이 도달 가능하지는 않도록(밀도 0.3 이면 거의 전부 도달) rect 이하에서는 밀도 0.3 그대로 두고 도달 불가 쌍이 있는지 테스트에서 확인만 한다.
- `bfs`: 정수 출력 정확 일치(rtol=atol=0, ratio 0), 센티널 -2048. 참조는 numpy 레벨 동기 BFS(프론티어의 이웃을 `np.repeat`/누적합으로 벡터화). 생성은 spmv_csr 의 층화 추출을 재사용(`n_cols = n`). `work` 없음.
- `fft`: tolerance `rtol 1e-3`, `atol_rms_rel 1e-3`, 센티널 -2048(출력 크기 ≤ n 이지만 n ≤ 2²² 이라 -2048 이 정당 값일 확률은 무시 가능하지 않다 → 센티널 **-1.0e9** 사용), `work "5*n*log2(n)"`. 참조 `np.fft.fft(float64 복소수)` 를 실수·허수 교차로 저장. spec 에 부호 규약·스케일·n 이 2 의 거듭제곱임을 명시, 비트 반전·복소수 표현은 자유.
- 더미(정답)는 세 언어 × 4과제. C 기준선(floyd_warshall, bfs)은 샘플과 같은 플래그로 빌드된다.
