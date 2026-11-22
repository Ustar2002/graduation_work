# LLM 생성 HPC 코드 평가 하네스 (졸업프로젝트)

경희대 컴퓨터공학부 졸업프로젝트 **"프로그래밍 언어 선택과 언어 특성이 LLM 생성 HPC 코드의 정확성·성능·안전성에 미치는 영향 — 실증 분석(예비 연구)"** 의 실험 하네스다. 같은 과제(커널 12개, §1.1)를 여러 언어(§1.2)로 LLM에 생성시켜 **추출 → 빌드 → 린트 → 격리 실행 → 정확성 판정 → (sanitizer) → 시간 측정 → CSV 행** 까지를 자동으로 돌리고, 실패한 샘플에는 진단을 돌려주는 수정 루프를 돌린 뒤, pass@k·속도비·토큰·오류 유형을 부트스트랩 구간과 그래프로 집계한다. 핵심 가설은 "성능 격차는 추상화 수준이 아니라 성능 결정(루프 순서·타일링·벡터화·병렬화·메모리 배치)을 LLM이 직접 내려야 하는가로 설명된다"이다(`CLAUDE.md` §1).

하네스는 **Linux 전용 측정 도구**다. 새 언어·마이크로 조건·수정 루프·perf_summary·sanitizer·린트·통계 그림(M3–M6)은 Windows 개발 머신에서만 검증됐고, Linux 에서 확인할 것은 `docs/linux_test_plan.md` 에 모여 있다(§9).

- 프로젝트 지시서: `CLAUDE.md` (수정 금지)
- 설계 계약: `docs/harness_design.md`(v2.1 + Tier 2·3 과제 §16–§18 — 모듈 인터페이스·JSON 키·CSV 열), `docs/harness_design_v3.md`(M3–M6 확장의 인터페이스)
- 결정 기록: `docs/decisions.md` · 측정 규약: `docs/protocol.md` · 오류 분류표: `docs/error_taxonomy.md`
- 측정 장비 이전: `docs/desktop_setup.md` · Linux 일괄 검증: `docs/linux_test_plan.md` · 1차 실험 결과표: `docs/meeting_2026-11.md`

## 1. 하네스가 하는 일

### 1.1 과제 12개 (계약은 `tasks/<task>/spec.md`·`signatures.yaml`·`tolerance.yaml`, 근거는 `docs/harness_design.md` §3·§16–§18, `docs/decisions.md`)

모두 호출자가 할당한 출력 버퍼 1개에 결과를 쓰고(반환값 무시), 입력은 읽기 전용, 64바이트 정렬, 형태 이름은 `tiny / rect / small`(정확성) · `large`(성능). 정수 출력은 정확 일치(rtol = atol = 0)라 `numeric_instability` 가 나오지 않는다. `spec.md` 는 프롬프트에 그대로 들어간다.

| task | Tier | 시그니처(C; Python 은 같은 이름·순서) | 출력 | 형태 tiny / rect / small / large | 허용오차 | 기준선 |
|---|---|---|---|---|---|---|
| `gemm` | 1 | `void gemm(const float* A, const float* B, float* C, int M, int N, int K)` | float32 M×N | (7,5,3) / (200,130,312) / 256³ / 1024³ | rtol·atol 1e-3 | NumPy `np.matmul(out=)`(OpenBLAS) |
| `jacobi2d` | 1 | `void jacobi2d(const float* src, float* dst, int H, int W, int T)` | float32 H×W (T 스텝) | (5,7,3) / (130,200,4) / (512,512,8) / (4096,4096,20) | rtol 1e-5, atol 1e-6 | C+OpenMP |
| `spmv_csr` | 1 | `void spmv_csr(const int* row_ptr, const int* col_idx, const float* vals, const float* x, float* y, int n_rows, int n_cols, int nnz)` | float32 n_rows | 5 / 300 / 10⁴ / 10⁶ 행 (large nnz ≈ 1.6×10⁷) | rtol·atol 1e-4 | C+OpenMP |
| `dot` | 1 | `void dot(const float* x, const float* y, float* out, int n)` | float32 `out[0]` | 7 / 100,003 / 2²⁰ / 2²⁵ | rtol·atol 1e-3 (배율 500) | NumPy `np.dot` |
| `histogram` | 2 | `void histogram(const int* data, int* hist, int n, int nbins)` | int32 nbins (0 초기화는 커널 책임) | (13,4) / (100,003,37) / (2²⁰,256) / (2²⁵,1024) | 정확 일치 | C+OpenMP(스레드별 private 히스토그램) |
| `scan` | 2 | `void scan(const int* x, int* out, int n)` | int32 n (inclusive; x ∈ [0,32)) | 7 / 100,003 / 2²⁰ / 2²⁵ | 정확 일치 | C+OpenMP(2-pass 블록 스캔) |
| `nbody` | 2 | `void nbody(const float* pos, const float* mass, float* acc, int n)` | float32 n×3 AoS (EPS2 = 1e-4, G = 1) | 5 / 301 / 2,048 / 16,384 | rtol 1e-3, atol = 1e-3 × RMS(참조) (형태별), 센티널 −1e6 | C+OpenMP(i 병렬, `1.0f/sqrtf`) |
| `sort` | 2 | `void sort(const int* x, int* out, int n)` | int32 n 오름차순 (x ∈ [0, 2³¹−1]) | 7 / 100,003 / 2²⁰ / 10⁷ | 정확 일치 | NumPy `np.sort`(단일 스레드) |
| `cholesky` | 3 | `void cholesky(const float* A, float* L, int n)` | float32 n×n 하삼각 L (엄격 상삼각은 0 을 쓴다) | 6 / 131 / 512 / 2,048 | rtol 1e-3, atol = 1e-3 × RMS(참조) (형태별) | NumPy `np.linalg.cholesky` |
| `floyd_warshall` | 3 | `void floyd_warshall(const float* dist_in, float* dist_out, int n)` | float32 n×n (도달 불가 = INF 1.0e30) | 8 / 37 / 256 / 1,024 | rtol 1e-4, atol 1e-3 | C+OpenMP |
| `bfs` | 3 | `void bfs(const int* row_ptr, const int* col_idx, int* dist, int n, int nnz)` | int32 n (정점 0 부터의 홉 수, 도달 불가 −1) | 7 / 1,003 / 2¹⁶ / 2²¹ 정점 (nnz ≈ 16n) | 정확 일치 | C+OpenMP(CAS 로 정점 차지) |
| `fft` | 3 | `void fft(const float* x, float* out, int n)` | float32 n×2 (실수·허수; 정방향, 스케일 없음) | 2³ / 2¹⁰ / 2¹⁶ / 2²² | rtol 1e-3, atol = 1e-3 × RMS(참조) (형태별), 센티널 −1e9 | NumPy `np.fft.fft`(complex64) |

`sort`·`bfs` 는 Halide 로 표현할 수 없어 `halide_auto`·`halide_manual` 에서 돌리지 않는다(`signatures.yaml` 의 `unsupported`; 생성 단계가 건너뛰고 `aggregate --coverage` 표에 `n/a` 로 나온다). `jacobi2d`·`spmv_csr`·`floyd_warshall`·`cholesky`·`fft` 는 Halide 에서 "언어가 순차 알고리즘을 강제하는 어색한 과제"로 따로 묶는다(`aggregate --halide`; 2026-11-12 결정).

### 1.2 언어 7개

| 언어 | 설정 백엔드 (워커에서) | 빌드·실행 | 스레드 |
|---|---|---|---|
| `numpy` | `python` | 소스 그대로, 워커가 임포트 | `OMP_NUM_THREADS`·`OPENBLAS_NUM_THREADS` |
| `numba` | `numba` (모듈 임포트) | 모든 `njit`/`jit` 함수에 명시 시그니처(`@njit("<서명>")`; 임포트 때 컴파일). 시그니처가 없거나 `cache=True` 이면 빌드 단계 `signature_mismatch`, 워커가 디스패처가 하네스의 인자 타입을 컴파일 없이 받는지 확인한다. 워커 인터프리터에서 SciPy 가 임포트되면 run 이 exit 7 | `NUMBA_NUM_THREADS`, `NUMBA_THREADING_LAYER=omp` (OpenMP 런타임 언어) |
| `halide_auto` | `halide` (모듈 임포트, 형태별 빌드·JIT) | 파이프라인 빌드 함수(알고리즘만). 하네스가 오토스케줄러 `Li2018`(`parallelism` = 스레드 수)을 적용한다. 모델이 쓴 스케줄 호출은 무시하고 기록한다(`contract_violation` 의 `schedule_in_auto`; 통과 여부와 무관) | `HL_NUM_THREADS` |
| `halide_manual` | `halide` | 같은 빌드 함수 + LLM 이 쓴 스케줄(형태마다 구체적인 크기를 받는다) | `HL_NUM_THREADS` |
| `c_openmp` | `c` | gcc `-O3 -march=native -fopenmp` | `OMP_NUM_THREADS` (+ `OMP_PROC_BIND`·`OMP_PLACES`·`OMP_DYNAMIC`·`OMP_WAIT_POLICY`) |
| `c_seq` | `c` | gcc `-O3 -march=native` (스레드 심벌이 보이면 `contract_violation` 의 `threads_in_c_seq`) | 1 |
| `rust` | `rust` (C ABI 공유 객체) | 고정 템플릿의 `cdylib`(edition 2021, `rayon = "=1.12.0"` 잠금, opt-level 3, codegen-units 1, LTO 없음, panic = abort), `RUSTFLAGS="-C target-cpu=native"`(Linux 에서는 `-C link-arg=-Wl,--no-undefined` 추가), `cargo build --release --offline --locked`. panic 은 `runtime_crash`, 상세 `rust_panic:<메시지>` | `RAYON_NUM_THREADS` |

- 빌드 플래그·스레드 수·코어 집합은 `config/experiment.yaml` 이 정한다. `run.languages` 기본값은 `numpy, c_seq, c_openmp` 이고, 새 언어는 `--languages`(또는 `make … LANGUAGES=…`)로 고른다.
- 처리할 언어에 필요한 도구(C 언어는 gcc, rust 는 cargo·rustc, numba, halide, 오토스케줄러 플러그인)가 없으면 `harness.run` 은 **행을 쓰기 전에** exit 7 로 멈춘다(없는 도구는 모델의 오류가 아니다). bench 를 요청했고 C 기준선 과제가 있으면 C 컴파일러도 run 시작 때 확인한다. cargo·rustc 는 PATH → `$CARGO_HOME/bin` → `~/.cargo/bin` 순으로 찾는다.
- `halide.autoscheduler` 는 `Li2018`(기본)·`Adams2019`·`Mullapudi2016` 중 하나이고, `halide.autoscheduler_params` 는 설정 로더가 스케줄러별 허용 표(Li2018 없음, Mullapudi2016 `last_level_cache_size`·`balance`, Adams2019 `beam_size`)로 검사한다. Halide 의 빌드·JIT 와 numba 의 임포트 컴파일은 측정 구간 밖이다(라운드마다 bench 워커의 제한 시간 `bench_timeout_s` 안에서 돈다).
- `mojo` 는 2026-11-12 결정으로 정한 언어지만 아직 하네스에 없다(컴파일러가 Linux 에만 있다; `docs/decisions.md` 2026-11-23 "남은 일"). 분석의 언어 축에는 자리가 있다.

### 1.3 마이크로 조건 3개와 수정 루프

| 조건 | 값 | 내용 |
|---|---|---|
| `args` | `positional` · `keyword` | keyword: Python 계열은 키워드 전용 시그니처, C 는 하네스가 주는 `kernel_args.h` 구조체, Rust 는 `src/kernel_args.rs` |
| `grammar_hint` | `off` · `on` | on: `prompts/hints/<language>.md`(100줄 이하 문법 안내)를 붙인다. 힌트 파일은 `numba`, `halide_auto`, `halide_manual`, `rust` 에만 있고, 없는 언어는 이 조건에서 건너뛴다 |
| `feedback` | `compiler` · `compiler+lint` | 수정 요청에 린트 메시지 절을 더하는지 여부. 두 수준은 같은 턴 0 응답에서 갈라지게 한다(`harness.repair --seed-from-feedback`, 대응 비교) |

조건 조합은 디렉터리 이름 `cond_tag = args-<args>_hint-<grammar_hint>_fb-<feedback>` 로 갈린다. 새 언어의 시그니처는 `harness/sigs.py` 가 `signatures.yaml` 의 `args`(와 `doc:`)에서 만들고, 기존 세 언어는 YAML 의 `languages.<언어>` 원문을 그대로 쓴다 — 기존 36개 조건(과제 12 × `numpy`·`c_seq`·`c_openmp`, 기본 조건)의 `prompt_hash` 는 `results/runs.csv` 의 실제 행과 같다(`tests/fixtures/prompt_hashes_upstream.json` 으로 고정).

**수정 루프**(`harness/repair.py`): 직전 턴 행이 `valid_sample=1 ∧ pass=0 ∧ error_stage ≠ generate` 인 샘플에 진단을 돌려주고 다시 묻는다. 대화는 다중 턴 메시지(원래 프롬프트 → 응답 → 수정 요청 → …), 최대 3턴(`--max-turns`; 설정 `run.max_repair_turns` 기본 0). 진단은 컴파일러·테스트 러너가 줄 법한 정보까지만 준다(컴파일러 출력, 예외·시그널, Rust panic 의 메시지와 위치, 형태별 통과/실패와 최대 오차, 타임아웃). 하네스 내부 패턴 이름과 참조값은 주지 않는다. 턴마다 행이 하나 생긴다(`turn` 이 행 키에 있다). 턴 루프가 끝나면 그 호출이 만든 행의 bench 라운드 2..R 을 `harness.run` 과 같은 함수로 마친다 — 다음 턴의 대상은 라운드 1 까지의 판정으로 정해진다.

### 1.4 단계와 보조 산출물

```
생성 게이트 → extract → build(백엔드별) → [lint] → import(numba·halide 는 --inspect) → [perf_summary] → check → [safety] → bench 라운드 1 → 행 → CSV
→ (모든 raw 뒤) bench 라운드 2..R
```

- **린트**(`lint.enabled: true`): `plugins/lint/run_lint.py` 가 `CLAUDE.md` §6 계약의 기성 린터 베이스라인이다 — Python 계열 ruff·pyflakes, C cppcheck·clang-tidy, Rust 없음. 하네스를 임포트하지 않는 독립 CLI 라 인지프 ML lint 로 바꿔 끼울 수 있다(`plugins/lint/README.md`). 플러그인이 없거나 실패하면 경고하고 계속한다.
- **성능 결정 요약**(`perf_summary.enabled: true`): 빌드와 워커의 로드(import)가 성공한 샘플마다 `CLAUDE.md` §5 의 JSON(+ `decisions_by_library` 키)을 만든다. 소스: C 는 OpenMP pragma 와 gcc `-fopt-info-vec` 리마크, Rust 는 rayon 호출, Python·Numba 는 AST 와 Numba 진단, Halide 는 `print_loop_nest()` 와 스케줄 호출(halide_auto 의 스케줄은 오토스케줄러의 결정으로 적는다). 정하지 못하면 `"unknown"` 이다.
- **sanitizer**(`safety.enabled: false` 가 기본): C 백엔드만, 정확성 형태로만 ASan/UBSan 빌드를 따로 돌린다(워커 `--sanitize`). 판정이 `runtime_crash`·`wrong_output` 이고 ASan 보고가 있으면 `error_type=memory_error` 로 바꾼다(check 와 bench 의 모든 라운드). 통과한 샘플의 `pass` 는 바꾸지 않는다. TSan 은 쓰지 않는다(`race_condition` 은 반복 실행 간 불일치로 판정).
- **분석**: `analysis/aggregate.py`(pass@k, 속도비, `--ci` 부트스트랩 95% 구간, `--coverage`, `--by-turn`, `--conditions`, `--halide`), `analysis/stats.py`(고정 seed 부트스트랩), `analysis/plots.py`(그림 4개). 대시보드는 아직 없다(`CLAUDE.md` 가 마지막으로 둔 항목).

## 2. 저장소 구조

```
config/experiment.yaml      실험 설정(과제·언어·조건·모델·빌드 플래그·halide·safety·lint·측정 규약·경로)
config/machines/<호스트>.yaml  머신 프로브 결과(`make probe` 가 생성; 없으면 이전 위치 config/machine.yaml)
tasks/<task>/               spec.md(프롬프트에 그대로 삽입; 일부 과제는 한국어 사본 spec.ko.md), signatures.yaml(기존 세 언어의 시그니처·param_docs 원문,
                            args(+ doc)·work·baseline·unsupported; harness/taskspec.py 가 읽음),
                            tolerance.yaml, gen_tests.py(harness/testgen.py; 시드 고정 npz), baseline/{baseline.py | baseline.c}, tests/(생성됨, git 제외)
                            Tier 1: gemm, jacobi2d, spmv_csr, dot   Tier 2: histogram, scan, nbody, sort   Tier 3: cholesky, floyd_warshall, bfs, fft
prompts/                    system.md, lang/<7개 언어>.md, hints/{numba,halide_auto,halide_manual,rust}.md, repair.md
harness/                    generate(LLM 호출) → extract → build/{c,python,numba,halide,rust}.py → run(오케스트레이션) ← worker(LLM 코드 실행 유일 장소)
                            repair(수정 루프), llm_client, config, protocol(측정 규약), benchstats(라운드 합산), aa_check(A/A 점검),
                            classify(오류 분류), schema(CSV 스키마), safety(sanitizer·린트 호출), perf_summary, sigs(언어별 시그니처 렌더링),
                            taskspec, testgen, procutil(격리 실행), paths
harness/kernels/            워커의 커널 어댑터: cabi(C·Rust ctypes), pyfunc(numpy·numba), halide_pipe
harness/build/rust_template/  Cargo.toml·Cargo.lock(rayon 고정)·자리표시 src/lib.rs
harness/dummy_responses/<task>/<language>/<k>.md   더미 제공자의 고정 응답(0.md = 검증된 정답)
plugins/lint/               run_lint.py, rules.yaml, README.md
analysis/                   aggregate.py, stats.py, plots.py
tools/devcheck.py           개발용 커널 점검(측정 아님), tools/devcheck_shim/(zig 빌드용 omp.h 스텁·shim.h, --cc zig-omp 용 omp_real/omp.h)
report/interim/             중간보고서(build.py → PDF)
tests/                      unittest (tests/fixtures/ 포함)
results/                    raw/, built/(git 제외), runs.csv, runs_dummy.csv, archive/(1차 실험 응답 원문 묶음), plots/(git 제외), aa/(A/A 점검 요약)
docs/                       decisions, protocol, error_taxonomy, harness_design(v2.1), harness_design_v3, linux_test_plan, desktop_setup, meeting_2026-11
```

## 3. Linux 설치와 측정 준비 (새 장비로 옮기는 절차는 `docs/desktop_setup.md`, 통합 하네스의 확인 절차는 `docs/linux_test_plan.md`)

필요: Linux, Python 3.14, gcc(`-fopenmp`; sanitizer 를 켜면 libasan·libubsan), `taskset`(util-linux), `nm`, Rust 툴체인(rustup; `Cargo.lock` 은 v4 형식). 선택: 린트용 `cppcheck`·`clang-tidy`(없으면 C 행의 `lint_*` 열이 NA 이고 run 은 계속된다). sudo 는 거버너 설정에만 쓰고 하네스는 시스템 설정을 바꾸지 않는다.

```sh
make venv                 # python3 -m venv .venv && .venv/bin/pip install -r requirements.txt  (다른 인터프리터: make venv SYSPY=python3.14)
make rust-setup           # cargo fetch --locked (Rust 템플릿의 고정 크레이트; 온라인 1회, 이후 rust 빌드는 --offline)
                          #   cargo 가 PATH 에 없으면 make rust-setup CARGO=~/.cargo/bin/cargo
make probe                # config/machines/<호스트>.yaml 생성 (CPU 토폴로지·거버너·BLAS 정체·컴파일러·cargo·rustc·numba·halide·린트 도구)
make tests                # tasks/*/tests/{tiny,rect,small,large}.npz + manifest.json (과제 12개, 시드 고정, 바이트 동일 재생성; 전체 약 1.6 GB)
make unittest             # tests/ 단위 테스트
make help                 # make 타깃 목록
```

머신을 옮기면 `make probe` 뒤 `.venv/bin/python -m harness.protocol --suggest-cores N` 으로 측정 코어 후보를 보고, 장비용 설정 파일(`config/experiment.yaml` 사본)의 `measurement.cores`·`single_core`·`threads`·`paths.results_csv` 를 맞춘 뒤 모든 명령에 `CONFIG=<그 파일>` 을 붙인다(`docs/desktop_setup.md` 6·7단계). 다른 장비의 행과 섞어 비교하지 않는다(`env_hash` 가 달라진다).

**모든 Python 실행은 `.venv/bin/python`** 이다. `.venv`는 격리형(시스템 site-packages 미사용)이어야 한다 — 시스템 `python3-numpy`는 참조 BLAS(netlib)라 1024³ GEMM이 ~250 ms이고, pip 휠 `numpy==2.3.5`(OpenBLAS 내장)는 4스레드 ~15 ms(EPP `power`)·~8 ms(EPP `performance`)다 — 절대 시간은 주파수 상태에 따라 달라지므로 비교는 대응 기준선 대비 속도비로 한다(`docs/harness_design.md` §0). 하네스는 BLAS 정체를 행마다 기록하고 OpenBLAS가 아니면 run을 중단한다.

측정 전에 거버너·EPP를 `performance`로 두고 확인한다(`docs/protocol.md` §13):

```sh
sudo cpupower frequency-set -g performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
.venv/bin/python -m harness.protocol --check      # 플래그가 있으면 exit 1
```

### 3.1 측정 규약 요약 (상세는 `docs/protocol.md`, 근거는 `docs/decisions.md` 2026-11-08·2026-11-09·2026-11-11·2026-11-12)

- 입력은 메모리 상주(시드 고정 npz 를 워커가 읽어 둔다), 출력 버퍼는 센티널로 채우고 앞뒤 가드 영역을 둔다(`docs/protocol.md` §7). LLM 코드는 `harness.worker` 서브프로세스에서만 돈다.
- 코어 고정: 워커는 `taskset` 으로 `measurement.cores`(스레드 1개 언어의 import·check 는 `single_core`)에 고정되고 스레드 수는 `measurement.threads` 다.
- 정확성: `tiny`·`rect`·`small` 마다 `check_repeats`(2)번 호출하고, 호출 간 불일치는 `race_condition` 근거다.
- 성능: `large` 로 **교차 측정** — 커널과 기준선을 한 워커에서 번갈아 호출한다(2026-11-09). 호출하는 동안의 주 스레드 친화도(2026-11-11): OpenMP 런타임 위의 코드(`c_openmp`, `numba`)는 libgomp 의 고정 그대로, 순차 코드(`c_seq`)는 고정 코어의 첫 CPU, 그 밖의 스레드 코드(python·halide·rust)는 고정 코어 전체.
- 라운드(2026-11-12): 구현마다 **서로 다른 시각의 별도 워커**에서 `bench_rounds`(3)라운드 × (워밍업 3 + timed 10). 라운드 1 은 check 직후, 라운드 2·3 은 run 의 모든 샘플이 라운드 1 을 마친 뒤 한 바퀴씩(같은 구현의 라운드 사이 최소 `bench_round_gap_s` 60 초). 주 지표는 `speedup_mean`(기준선 평균 / 커널 평균)과 95% 층화 부트스트랩 구간이고, 이전 실험과 비교할 수 있게 중앙값 비 `speedup_vs_baseline` 도 기록한다. 멈춘 run 을 다시 부르면 남은 라운드만 잇는다(라운드 계획 수와 `env_hash` 가 같은 행만).
- OpenMP 코드가 있는 워커의 환경: `OMP_PROC_BIND=true`, `OMP_PLACES=cores`, `OMP_DYNAMIC=false`, `OMP_WAIT_POLICY=passive`. OpenBLAS 의 스핀 대기는 `OPENBLAS_THREAD_TIMEOUT`(설정 `openblas_thread_timeout` 16)으로 줄인다.
- bench 직전 동적 검사: 부하 `load_1m`(정보용 플래그, 2026-11-08), 고정·형제 코어 사용률, 거버너 재확인. 기준선은 run 시작·1스레드·샘플별 대응 값을 남기고 drift 를 플래그로 기록한다.
- A/A 점검(`make aa`): 기준선을 자기 자신과 교차 측정해 측정 장치의 치우침을 잰다. 새 장비에서 측정 전에 돌려 `bench_loop_fsync` 를 정한다(`docs/desktop_setup.md` 9단계).

## 4. 비-Linux 개발 머신에서 되는 것

Windows 개발 머신의 venv는 `.venv-win`(git 제외; Python 3.14, requirements 의 패키지 + `ziglang`, Rust는 MSVC 툴체인)이다. 그 머신에는 make가 없으므로 아래 CLI를 `.venv-win\Scripts\python.exe` 로 직접 부른다. `.venv/bin/python` 이 없으면 하네스는 현재 인터프리터를 워커에도 쓴다.

- **되는 것**: 단위 테스트(Linux 전용은 skip), `tools/devcheck.py`, 더미 제공자로 `harness.generate` → `harness.run` → `harness.repair` → `analysis.aggregate`/`analysis.plots`. `harness.run` 은 python 계열(`numpy`, `numba`, `halide_*`)과 `rust` 를 끝까지 처리한다. bench 는 기준선이 Python 인 과제(`gemm`, `dot`, `sort`, `cholesky`, `fft`)에서 돈다 — C 기준선 과제는 bench 를 요청하면 C 컴파일러 점검에서 exit 7 이므로 `--skip-bench` 로 돌린다. 코어 고정이 없어 시간은 의미가 없다.
- **표시**: 비-Linux에서 만든 행은 `protocol_flags` 에 `platform=<sys.platform>`(정보용 아님)·`governor=unknown`·`taskset_missing` 이 붙어 `protocol_ok=0` 이다. 분석에서 측정값으로 쓰지 않는다.
- **안 되는 것**: C 언어의 `harness.run`(gcc가 없어 행을 쓰기 전에 exit 7), taskset, 거버너, sanitizer, cppcheck·clang-tidy, 시간 측정.
- **devcheck 의 C 툴체인**: gcc가 없으면 `python -m ziglang cc` 로 **직렬** 빌드한다(OpenMP pragma 무시, `omp.h` 스텁 — 논리 검증만; 판정 줄에 `toolchain=zig-serial`). `--cc zig-omp` 를 주면 OpenMP 대상(c_openmp, `openmp: true` 인 C 기준선)을 `clang -fopenmp` + 이 머신의 LLVM OpenMP 런타임(Windows: `libomp140.x86_64.dll`, 다른 경로는 `DEVCHECK_LIBOMP`, 스레드 수는 `DEVCHECK_OMP_THREADS`)으로 실제 스레드로 돌린다(`toolchain=zig-omp`). 어느 쪽도 gcc + libgomp 확인을 대신하지 않는다.
- 개발 머신에서 파이프라인을 돌릴 때는 `config/experiment.yaml` 사본의 `paths` 를 작업 디렉터리 밖으로 바꿔 `--config` 로 넘긴다. 기본 설정 그대로 돌리면 `platform=` 플래그가 붙은 행이 git에 올라가는 `results/runs_dummy.csv` 에 섞인다.

## 5. 단계별 명령

`make` 타깃은 아래 CLI의 얇은 감싸개다. 공통 선택 변수: `TASKS=a,b`, `LANGUAGES=x,y`, `N=<샘플 수>`, `SKIP_BENCH=1`(check까지만), `RUN_ID=…`(없으면 타임스탬프), `CONFIG=path`(모든 harness 명령에 `--config path`). `make help` 가 목록을 보여 준다. 옵션 전체는 각 CLI의 `--help`.

### 5.1 개발용 점검 (`tools/devcheck.py`, 측정 아님·CSV 안 씀)

```sh
make devcheck                          # tools/devcheck.py dummies — 모든 0.md 를 extract → build → worker check(정확성 형태), 표로 출력
make devcheck LANGUAGES=numba,rust TASKS=gemm,dot     # DEVCHECK_CC=gcc|zig|zig-omp 로 C 툴체인을 고른다(기본 auto)
make devcheck-baselines [TASKS=…]      # 과제마다 tools/devcheck.py baseline --task <과제>
.venv/bin/python tools/devcheck.py kernel --task dot --language rust --file my.rs [--args-mode keyword] [--tests tiny,rect,small,large] [--bench]
.venv/bin/python tools/devcheck.py gen --task fft
```

종료 코드 0 통과 / 1 커널이 틀림 / 2 사용법·도구 문제. `dummies` 를 `--languages` 없이 돌리면 설정의 언어와 `0.md` 가 있는 언어 전부가 대상이고, 지원한다고 돼 있는데 `0.md` 가 없는 조합은 `missing`(실패)이다. 기대값은 80 pass, 4 unsupported(Halide × `sort`·`bfs`). `dummies` 는 정확성 형태만 돌린다 — `large` 는 `kernel --tests …` 로 본다.

### 5.2 더미 파이프라인 (API 키 없이 전체 경로)

```sh
make dummy                              # 설정 그대로: run.tasks(12개) × run.languages(numpy,c_seq,c_openmp) × run.n_samples(5)
make dummy RUN_ID=my-dummy TASKS=gemm N=3   # generate 에 --tasks/--languages/--n 을 넘긴다
make dummy-all [LANGUAGES=…] [N=1]      # 언어 7개 × 과제 전부, n=1 → run → aggregate --coverage
make dummy-m3                           # numba,halide_auto,halide_manual,rust 만 (rust 가 있으므로 make rust-setup 먼저)
#   .venv/bin/python -m harness.generate --provider dummy --run-id $RUN_ID [--tasks …] [--languages …] [--n …]
#   .venv/bin/python -m harness.run --run-id $RUN_ID [--skip-bench]
#   .venv/bin/python -m analysis.aggregate --csv results/runs_dummy.csv --run-id $RUN_ID [--coverage]
```

`harness/dummy_responses/<task>/<language>/<k>.md` 의 고정 응답(샘플 i는 `i mod 파일 수`, 수정 턴 t는 `<k>.turn<t>.md`, 없으면 `<k>.md`)으로 생성 단계를 대신한다. 기대 결과표는 `harness/dummy_responses/README.md`(과제 전체)와 과제별 README(`gemm`: 11개 변형으로 pass/`numeric_instability`/`arg_order`/`compile_error`/`runtime_crash`/`race_condition` 각 1건 이상과 수정 턴 2개; `histogram`·`scan`·`floyd_warshall`·`bfs`: 정답 + 경쟁 조건·의존성 위반 변형; 나머지 조합: 정답 `0.md` 하나), 그리고 `docs/error_taxonomy.md` §8에 있다. Tier 2 만 돌리려면 `.venv/bin/python -m harness.generate --provider dummy --run-id dummy-tier2 --tasks histogram,scan,nbody,sort --n 2` 뒤 `harness.run --run-id dummy-tier2`(sample 1 이 k=1 변형을 밟는다). 더미 행은 `provider=dummy` 로 **`results/runs_dummy.csv`에 분리 저장**되며 실제 결과와 섞이지 않는다(키가 없어도 실제 제공자에서 dummy로 자동 폴백하지 않는다). bench를 포함하면 과제·언어 수 × 3라운드만큼 오래 걸린다 — check까지만 보려면 `SKIP_BENCH=1`.

### 5.3 실제 실행

```sh
export ANTHROPIC_API_KEY=...          # config/experiment.yaml model.anthropic.api_key_env (또는 .env, 아래)
# 선택: LLM_PROVIDER=anthropic|openai_compat|dummy, LLM_MODEL=<모델명> 이 experiment.yaml 을 덮어쓴다.
# openai_compat 은 LLM_BASE_URL 과 OPENAI_API_KEY 가 필요하다.
make real RUN_ID=20261105-opus55            # generate(확인 프롬프트) → run → aggregate
make real RUN_ID=… STRICT=1                 # run --strict-protocol: run 시작의 정적 규약 플래그(거버너·EPP·taskset·비-Linux)가 있으면 중단(exit 4)
make real RUN_ID=… TASKS=gemm,dot LANGUAGES=numba,rust N=5
.venv/bin/python -m harness.generate --run-id ID --languages numba,rust --tasks gemm --args keyword --grammar-hint on --yes
```

- **호출 수 주의**: `run.tasks` 가 과제 12개라 `make real` 을 필터 없이 돌리면 12 × 3언어 × 5샘플 = **180회** 호출이다. 새 언어·조건까지 넣으면 그만큼 늘어난다. 대량 호출은 단계마다 확인받는다(`CLAUDE.md` 원칙 5).
- **키 보관**: 저장소 루트에 `.env` 파일을 만들어 `ANTHROPIC_API_KEY=…` 한 줄을 넣어 두면 하네스가 시작할 때 읽는다(`.gitignore` 에 포함, 값은 기록·출력되지 않음). 이미 설정된 환경변수가 있으면 그쪽이 우선한다.
- **확인 게이트**: `harness.generate`는 `preflight()`(모델 조회, 과금 없음) 뒤 계획(`<n> calls to <model> via <provider>, max_tokens=…, effort=…`)과 과제 × 언어 표를 보여주고 `Proceed? [y/N]`을 묻는다(`max_tokens=32000`, 스트리밍). 비대화형(stdin이 tty가 아님)에서는 `--yes`가 없으면 exit 2로 끝나므로, 스크립트에서는 `make` 대신 `.venv/bin/python -m harness.generate --run-id ID --yes` 를 직접 쓴다. 치명 API 오류(인증·404·400)는 raw를 저장한 뒤 즉시 중단(exit 3)한다. `run.tasks` 밖의 과제는 `harness.run`·`harness.repair` 가 건너뛰므로 호출하지 않는다.
- 재실행: 같은 `RUN_ID`로 다시 돌리면 정상 응답(`error` 없음, `stop_reason ∈ {end_turn, stop_sequence}`)은 건너뛰고 실패·절단 샘플만 재호출한다(`--no-retry-failed`로 끔, `--force`로 전부 재호출; 기존 파일은 `sample_<i>.prev<N>.json`으로 보관). `harness.run`도 raw 해시가 같고 요청한 bench 가 이미 있는 샘플은 건너뛴다(`--force`). `--skip-bench` 로 처리한 샘플은 bench 를 요청한 run 이 다시 처리한다. safety·lint 만 빠진 결과는 알리고 건너뛴다(다시 처리하면 측정도 다시 하게 되므로; 채우려면 `--force`).
- 부분 실행: `harness.run --run-id ID [--tasks a,b] [--languages a,b] [--samples 0,1] [--skip-bench] [--no-baseline]`. 끝의 보정(기준선 drift·`model_served_mixed`)은 필터와 무관하게 그 run 의 모든 행에 한다.
- 규약 플래그는 경고와 함께 행의 `protocol_flags`에 기록되고(`protocol_ok=0`) 실험은 계속된다. `STRICT=1`(`--strict-protocol`)은 **run 시작의 정적 검사**(거버너 `powersave`, EPP, `taskset` 없음, 비-Linux)에만 적용돼 그때 exit 4로 중단한다; bench 직전의 **동적** 플래그(부하 `load_1m`, 고정/형제 코어 사용률, 거버너 재확인)와 기준선 `baseline_drift`·`paired_baseline_failed`, bench 의 `call_affinity_failed`(호출 동안의 CPU 고정을 적용·복원하지 못함; `docs/protocol.md` §10.1), run 끝의 `model_served_mixed`는 STRICT=1이어도 **기록만** 되고 run은 멈추지 않는다 — "STRICT=1이면 플래그 없는 행만 남는다"는 보장이 아니므로 분석에서는 행의 `protocol_ok`/`protocol_flags`로 가려낸다(`docs/protocol.md` §8–§10). BLAS가 OpenBLAS가 아니면 STRICT와 무관하게 항상 중단된다.
- `harness.run` 종료 코드: 0 정상, 1 입력 없음 등, 2 설정 오류, 4 규약·BLAS, 5 하네스 오류(워커가 결과 문서를 한 번도 쓰지 못하고 끝남 등; 행 없음), 6 기준선 게이트, 7 도구 없음(행을 쓰기 전에), 130 중단.
- 거부(refusal)는 다른 모델로 대체 실행하지 않고 `stop_reason=refusal`로만 기록한다.

### 5.4 수정 루프

```sh
make repair-dummy [LANGUAGES=numpy,c_seq] [REPAIR_TURNS=3] [SKIP_BENCH=1]
#   generate --provider dummy --tasks gemm --languages numpy,c_seq --n 5 --feedback compiler → run
#   LLM_PROVIDER=dummy harness.repair --run-id $RUN_ID --feedback compiler --max-turns 3
#   harness.repair --run-id $RUN_ID --seed-from-feedback compiler --feedback compiler+lint   # 턴 0 raw 복사(API 호출 없음)
#   run → LLM_PROVIDER=dummy harness.repair --run-id $RUN_ID --feedback compiler+lint --max-turns 3
#   analysis.aggregate --csv results/runs_dummy.csv --run-id $RUN_ID --by-turn --conditions
.venv/bin/python -m harness.repair --run-id ID --max-turns 3 [--feedback compiler+lint] [--tasks …] [--languages …] [--samples …] [--skip-bench] [--yes]
```

전제는 같은 run의 턴 0이 `harness.generate` + `harness.run` 으로 처리돼 있는 것이다. 제공자·모델은 설정과 환경변수(`LLM_PROVIDER`, `LLM_MODEL`)에서 읽고 턴 0을 만든 것과 다르면 호출하지 않고 끝낸다(exit 1). 실제 제공자에서는 턴마다 대상 수를 보여 주고 `generate` 와 같은 확인을 받는다. 수정 턴은 `harness.repair` 가 바로 처리해 행을 쓴다. 종료 코드: 0 정상, 1 설정/입력, 2 확인 거부, 3 치명 API 오류, 5 하네스 오류, 130 중단. `harness.run` 의 준비·처리·라운드 단계가 멈추면 그 코드(4 규약, 6 기준선, 7 도구 없음, 5 하네스 오류)를 그대로 돌려준다. gemm 수정 턴 더미는 `c_seq/1.turn1.md`, `numpy/2.turn1.md` 다(기대표는 `harness/dummy_responses/gemm/README.md`).

### 5.5 sanitizer (C만)

설정 사본에서 `safety: {enabled: true, timeout_s: 300}` 으로 바꿔 `--config` 로 넘긴다. 예(더미 run): `.venv/bin/python -m harness.run --config cfg_san.yaml --run-id ID --languages c_seq,c_openmp --force --skip-bench`. `--force --skip-bench` 는 샘플을 bench 없이 다시 처리해 같은 행 키의 행을 시간 열이 빈 행으로 바꾸므로, 이미 측정한 run 에서는 `--skip-bench` 를 빼고 돌린다(측정도 다시 한다). 로그 확인은 `.venv/bin/python -m harness.safety --parse-logs <샘플>/san`, 린트 단독 실행은 `.venv/bin/python -m harness.safety --lint <파일> --language c_openmp [--task <과제>]`.

### 5.6 집계·그래프

```sh
make aggregate                                   # results/runs.csv, --latest
make aggregate RUN_ID=… [CSV=results/runs_dummy.csv] [AGG_FLAGS="--ci --coverage --by-turn --conditions --halide"]
make aggregate ALL_RUNS=1                        # --all-runs
make plots [RUN_ID=…|ALL_RUNS=1] [CSV=…] [OUT=results/plots] [FORMAT=png|pdf|svg]
.venv/bin/python -m analysis.aggregate [--csv …] [--run-id ID | --latest | --all-runs] [--json] [--ci] [--coverage] [--by-turn] [--conditions] [--halide] [--primary-speedup mean|median] [--config …]
.venv/bin/python -m analysis.plots [--csv …] [--run-id ID | --latest | --all-runs] [--out DIR] [--format png|pdf|svg] [--by-task] [--primary-speedup mean|median] [--config …]
```

pandas로 읽어(`dtype=str, keep_default_na=False` 뒤 수동 변환) 행 키 중복은 `created_at` 최신만 남기고, 그룹 `(run_id, provider, model, task, language, args, grammar_hint, feedback, turn)`별로 아래 열을 낸다. run 단위로는 기준선(run 시작/1스레드/대응 평균·n·최대 |drift|)과 `model_served`·`effort`가 단일값인지(아니면 경고)를 요약한다. turn ≥ 1 행이 있으면 그룹 표는 turn 0만 세고 한 줄로 알린다. 종료 코드: 0 정상(행이 없어도 0), 1 CSV 없음 / run_id 없음, 2 CSV 열이 스키마와 맞지 않음.

| 열 | 의미 |
|---|---|
| `n_rows` | 그룹 행 수(`valid_sample=0` 포함) |
| `n`, `n_pass` | `valid_sample=1`인 행 수(pass@k 분모), 그중 `pass=1` |
| `pass_at_1`, `pass_at_5`, `pass_at_n` | 비편향 추정량 `1 − C(n−c, k)/C(n, k)`; `k > n`이면 NA. `pass_at_1 = n_pass/n` |
| `n_generate_failed`, `n_refusal`, `n_truncated` | 생성 단계: API 오류(`valid_sample=0`), 거절(`valid_sample=1`, `pass=0`), `max_tokens` 절단(`valid_sample=0`) — 서로 겹치지 않음 |
| `benchmarked`, `n_repeats_reduced`, `n_protocol_flagged` | `bench_ran=1`, `repeats_reduced=1`, `protocol_ok=0`이거나 정보용 외 규약 플래그가 있는 행 수 |
| `time_median_ms` | pass 행들의 `time_median_ms` 중앙값(bench 타임아웃 행은 시간 NA라 제외) |
| `speedup_median`, `speedup_min`, `speedup_max`, `speedup_1t_median` | pass 행들의 `speedup_vs_baseline` 중앙값/최소/최대, `speedup_vs_baseline_1t` 중앙값 |
| `speedup_mean_median` | pass 행들의 `speedup_mean`(라운드를 모은 기준선 평균 / 커널 평균; 2026-11-12 규약의 주 지표) 중앙값 |
| `tokens_out_mean`, `tokens_text_mean` | 토큰 값이 있는 모든 행(절단 행 포함)의 평균 |
| `tokens_per_pass`, `tokens_out_per_pass` | `Σ(tokens_in + tokens_out) / n_pass`, `Σ tokens_out / n_pass`(`n_pass=0`이면 NA) |
| `error_types` | `pass=0` 행의 `error_type` 집계 `"유형:수, …"`(수 내림차순; 빈 `error_type`인 생성 단계 행은 제외) |

- **속도비 주 지표**: 설정 `measurement.primary_speedup` 이 `mean` 이면 `speedup_mean`, `median` 이면 `speedup_vs_baseline`. 선택된 통과·측정 행 가운데 `speedup_mean` 이 빈 행(이전 규약으로 잰 행)이 하나라도 있으면 전부 `speedup_vs_baseline` 으로 비교한다. `--ci` 의 속도비 구간, `--halide`, 그림이 이 열을 쓴다(`--primary-speedup` 으로 덮어쓴다).
- `--ci`: `pass_at_1` 과 속도비(주 지표) 중앙값의 95% 백분위 부트스트랩 구간(`_lo`/`_hi` 열; seed 0, 2000회). n=5에서는 구간이 넓고, 작은 n에서 백분위 부트스트랩은 실제보다 좁게 나온다(5/5면 구간이 [1, 1]이지만 Clopper–Pearson 95% 하한은 0.48 — `analysis/stats.py` docstring).
- `--coverage`: 과제 × 언어 표(턴 0 유효 샘플 수, `unsupported` 는 `n/a`, 행 없음은 `-`).
- `--by-turn`: 수정 턴 t까지 허용했을 때의 누적 통과율 `pass_within_t`, 누적 토큰 `tokens_cum_t`, `turns_to_pass` 분포.
- `--conditions`: 조건 수준 쌍의 통과율 차와 구간. `args`·`grammar_hint` 는 턴 0의 독립 표본, `feedback` 은 `(run_id, sample_idx)` 대응 비교로 두 수준 모두 결과를 아는 턴 수 H까지(`pass_within_<H>`).
- `--halide`: Halide 가 지원하는 과제 / 어색한 과제를 뺀 과제 / 어색한 과제만의 언어별 통과율·속도비(§1.1).
- 그림(`results/plots/`, git 제외): `fig1_languages`(x: 성능 결정 부담 순 `numpy, numba, halide_auto, halide_manual, mojo, rust, c_openmp, c_seq` 중 자료가 있는 언어, y: pass@1과 속도비, 과제별 패널), `fig2_error_types[_by_task]`, `fig3_condition_effects`, `fig4_repair_turns`. 자료가 없는 그림은 건너뛰고 알린다.

### 5.7 측정 장비 점검·1차 실험 재측정·보고서

```sh
make aa [TASKS=a,b] [CONFIG=…]          # harness.aa_check: 기준선 대 기준선 교차 측정(fsync/nofsync 변형). 요약 results/aa/<run_id>.json
make unpack-raw                         # results/archive/raw_1st_experiment_laptop.tar.gz 를 results/raw 로 푼다(1차 실험 응답 원문)
make report                             # report/interim/build.py: results/runs.csv → report/interim/interim_report.pdf
                                        #   (먼저 .venv/bin/pip install -r report/requirements.txt; 본문 수치가 데이터와 다르면 빌드가 실패한다)
```

1차 실험 코드를 새 장비에서 다시 재는 절차(API 비용 없음)는 `docs/desktop_setup.md` 10단계다: `make unpack-raw` 뒤 각 run 을 `harness.run --run-id <run> --force --config <장비용 설정>` 으로 다시 잰다.

## 6. 산출물 위치

| 경로 | 내용 |
|---|---|
| `results/raw/<run_id>/<task>/<language>/<cond_tag>/sample_<i>.json` | 턴 0 생성 원문: 프롬프트·응답·토큰·stop_reason·모델·설정 스냅샷 |
| `…/<cond_tag>/sample_<i>.turn<t>.json` | 수정 턴 t 원문(`prompt.messages`, `repair: {parent_raw, feedback, diagnostics, …}`) |
| `…/<cond_tag>/sample_<i>.prev<N>.json` | generate가 재호출할 때 보관한 이전 원문 |
| `results/built/<run_id>/<task>/<language>/<cond_tag>/sample_<i>/` | 턴 0 산출물(아래 표) |
| `…/sample_<i>/turn_<t>/` | 수정 턴 t 산출물(같은 구성) |
| `results/built/<run_id>/<task>/baseline/` | `bench_run.json`(run 시작 기준선), `bench_1t.json`(1스레드), C 기준선이면 `build/`, A/A 점검이면 `aa/` |
| `results/built/<run_id>/_cargo_target/` | rust 샘플이 같이 쓰는 `CARGO_TARGET_DIR`(rayon은 한 번만 빌드). 그 안의 `_toolchain_probe/` 는 run 시작의 도구 점검 빌드(`probe.log`) |
| `results/runs.csv` / `results/runs_dummy.csv` | 실제 제공자 행(git에 올리는 결과) / `provider=dummy` 행 |
| `results/plots/`, `results/aa/` | `analysis.plots` 출력, A/A 점검 요약 |
| `config/machines/<호스트>.yaml` | 머신 프로브 결과(호스트별; 없으면 이전 위치 `config/machine.yaml`) |

샘플(턴) 디렉터리의 파일:

| 파일 | 언제 |
|---|---|
| `kernel.py` · `kernel.c` · `src/lib.rs` + `Cargo.toml`·`Cargo.lock` | 추출된 소스(언어에 따라) |
| `kernel_args.h` · `src/kernel_args.rs` | `args=keyword` 의 C·Rust |
| `kernel.so` | C·Rust 빌드 산출물(Windows rust는 `kernel.dll`) |
| `build.json`, `build.log` | 빌드 결과(+ `extra`), 컴파일러·cargo 출력(C는 벡터화 리마크 포함; python 계열은 `build.log` 없음) |
| `lint.json`, `lint.stderr.log` | 린트 플러그인 출력 원문과 stderr |
| `import.json`, `check.json`, `safety.json` | 워커 결과 문서(`import.json` 의 `inspect` 는 Numba·Halide 관찰값, `safety.json` 은 sanitizer 실행) |
| `bench.json`, `bench_r<r>.json` | 라운드 1, 라운드 r(≥ 2)의 교차 측정 결과(커널·기준선의 호출별 시간, 호출 친화도) |
| `<stage>.stdout.log`, `<stage>.stderr.log` | 워커 단계별 출력(`import`, `check`, `bench`, `bench_r<r>`, `safety`) |
| `san/` | sanitizer 빌드·로그(`kernel.c`, `kernel_san.so`, `build.json`, `build.log`, `asan.<pid>`, `ubsan.<pid>`) |
| `result.json` | `{"row", "raw_sha256", "stages", "requested", "written_at"}`(+ `raw_run_id`, 새 언어는 `toolchain`) — 행 + 단계별 상세(`extract`, `build`, `lint`, `import`, `perf_summary`, `check`, `sanitize`, `dynamic`, `bench`, `rounds` …), `requested` = 이 run이 요청한 선택 단계(`bench`, `safety`, `lint`) |

`result.json` 이 진실의 원천이다. CSV 헤더가 스키마와 달라지면 `.venv/bin/python -m harness.run --rebuild-csv` 로 다시 만든다(result.json이 없는 기존 CSV 행은 남긴다). 같은 행 키 `(run_id, task, language, args, grammar_hint, feedback, sample_idx, turn)` 는 교체(upsert)된다. 행의 `run_id` 는 처리한 run 디렉터리의 이름이다(raw 안의 값은 result.json 의 `raw_run_id`).

## 7. 결과 행의 열 (`harness/schema.py` `COLUMNS`, 121열; 설계 §11, 설계 v3 §0·§9.7)

직렬화: 불리언 `1`/`0`, NA 빈칸, 리스트/딕트 JSON 문자열.

| 열 | 의미 |
|---|---|
| `run_id`, `created_at`, `task`, `language`, `backend`, `args`, `grammar_hint`, `feedback`, `sample_idx`, `turn` | 행 키와 조건. `backend` 는 `python`(numpy), `c`(c_seq·c_openmp), `numba`, `halide`, `rust`. `turn` 0 = 첫 시도, t ≥ 1 = 수정 턴 t. 기본 조건은 `positional`/`off`/`compiler` |
| `provider`, `model`, `model_served`, `sampling`, `effort`, `max_tokens`, `temperature`, `seed`, `prompt_hash` | 요청 모델과 실제 응답 모델, 샘플링 설정 JSON(`temperature_effective: provider_default` 등), 프롬프트 해시(system + user; 수정 턴은 system + 메시지 목록; 바뀌면 실행 간 비교 불가) |
| `stop_reason`, `stop_details_category` | `end_turn` / `max_tokens`(절단) / `refusal`(거절 범주) / … |
| `tokens_in`, `tokens_out`, `tokens_thinking`, `tokens_text`, `n_thinking_blocks`, `latency_ms` | **`tokens_out`은 thinking 포함 총 출력 토큰(과금 기준)**, `tokens_text`는 텍스트 블록 몫. 수정 턴 행은 그 턴 호출의 값(입력에 앞 대화 전체가 들어간다). dummy는 단어 수(가짜) |
| `api_error`, `valid_sample` | API 오류 문자열(`fatal:…`/`transient:…`); `valid_sample=0`(API 오류·절단)은 pass@k 분모에서 제외 |
| `extract_ok`, `signature_ok`, `signature_exact` | 코드 블록 추출·시그니처 검사(한정자 차이는 `signature_exact=0`일 뿐 실패 아님) |
| `build_ok`, `build_time_ms`, `n_warnings`, `undefined_symbols`, `contract_violation` | 빌드 결과; `contract_violation`은 관찰용 JSON 리스트(`threads_in_c_seq`, `stdout_output`, `forbidden_symbol:<s>`, `schedule_in_auto`(halide_auto 응답의 스케줄 호출; 무시하고 기록)) |
| `pass`, `turns_to_pass`, `error_type`, `error_stage`, `error_detail` | **`pass`** = check 전부 통과 ∧ 입력 미변조 ∧ OOB 없음 ∧ (bench 미실행 ∨ 모든 라운드에서 정확성·크래시·race 없음). bench 타임아웃만 `pass=1`·시간 NA. `turns_to_pass` 는 통과한 행의 `turn`(아니면 NA). `error_type`은 고정 분류표의 아홉 값 또는 빈칸(`docs/error_taxonomy.md`). 새 상세: `numba_unsigned_jit:<이름>`·`numba_cache_true`·`numba_signature_rejects_args:…`(`signature_mismatch`), `rust_panic:<메시지>`(`runtime_crash`), sanitizer 가 켜져 있으면 `memory_error` 의 `asan:<종류>; was <이전 type>/<이전 detail>` |
| `inputs_mutated`, `oob_write`, `nondeterministic_runs`, `nondeterministic_bitwise_runs` | 입력 변조, 가드 영역 쓰기, bench 중 허용오차 초과 불일치 수(비트 불일치는 정보용) |
| `check_max_abs_err`, `check_max_tol_ratio`, `check_tests` | 정확성 테스트 최악값과 테스트별 JSON(`{name, classification, pattern, max_tol_ratio, elapsed_ms, nondeterministic}`) |
| `bench_ran`, `time_median_ms`, `time_mean_ms`, `time_min_ms`, `time_max_ms`, `time_cv`, `time_samples_ms`, `warmup`, `repeats`, `repeats_reduced`, `gflops` | `large` 성능: 시간 통계는 **모든 라운드의 timed 호출을 모은 값**이다(`time_cv` 는 표본 표준편차/평균). `time_samples_ms`는 전체 분포, `warmup`·`repeats`는 라운드마다의 **계획값**(축소 행에서는 축소된 값; 즉시 timeout·크래시 행에서는 계획값이 남으므로 실제 호출 수는 `time_samples_ms` 길이로 본다), `repeats_reduced=1`은 느린 커널의 반복 축소(`docs/protocol.md` §5). `gflops` 는 `work` 가 정의되지 않은 정수 과제(`histogram`, `scan`, `sort`, `bfs`)에서 NA |
| `baseline_time_ms`, `baseline_time_cv`, `baseline_samples_ms`, `baseline_run_time_ms`, `baseline_time_1t_ms`, `baseline_drift_pct` | 과제별 기준선(§1.1): 샘플별 **대응** 기준선(교차 측정에서 커널과 번갈아 부른 호출), run 시작 기준선, 1스레드 기준선, drift(%; 그 run 의 과제별 대응 기준선 중앙값 대비) |
| `speedup_vs_baseline`, `speedup_vs_baseline_1t` | `baseline_time_ms / time_median_ms`(중앙값 비; >1이면 기준선보다 빠름), `baseline_time_1t_ms / time_median_ms` |
| `bench_rounds`, `baseline_time_mean_ms`, `speedup_mean`, `speedup_ci_lo`, `speedup_ci_hi`, `speedup_rounds` | 라운드 규약(2026-11-12): 오류 없이 끝난 라운드 수, 기준선 평균, **`speedup_mean` = 기준선 평균 / 커널 평균**(주 지표), 그 95% 층화 부트스트랩 구간, 라운드별 평균 비(JSON 리스트) |
| `cpu_mhz_median`, `cpu_mhz_min`, `throttle_events`, `temp_c_start`, `temp_c_end` | bench 중 고정 코어 주파수·스로틀 횟수·패키지 온도 |
| `compiler`, `compiler_version`, `cflags`, `march_resolved`, `threads`, `cores`, `omp_env` | 빌드·실행 조건. `compiler`: c 는 `gcc`, rust 는 `rustc`, numba·halide 는 패키지 이름, numpy 는 `python`(+ 각 버전). `cflags`: c 는 전체 플래그 문자열, rust 는 `RUSTFLAGS=…; profile.release: …`, halide_auto 는 `autoscheduler=<이름> params=<JSON>`(`parallelism` 포함), halide_manual 은 `autoscheduler=none`, numpy·numba 는 NA. `omp_env` 에는 `NUMBA_`·`RAYON_`·`HL_` 변수도 남는다 |
| `governor`, `governor_ok`, `epp`, `no_turbo`, `load_1m`, `load_ok`, `pinned_busy_pct`, `sibling_busy_pct`, `taskset_used`, `protocol_ok`, `protocol_flags`, `protocol_checked_at` | 측정 규약 상태. `protocol_flags`는 JSON 리스트(`governor=powersave`, `epp=power`, `turbo=on`(정보용), `load_1m=…`(정보용), `pinned_busy=…`, `sibling_busy=…`, `taskset_missing`, `platform=<sys.platform>`(비-Linux), `blas=…`, `baseline_drift=…%`, `paired_baseline_failed`(대응 기준선 측정 실패 → `baseline_time_ms`·`speedup_vs_baseline` NA), `call_affinity_failed`, `multiple_libgomp`(정보용; bench 워커에 서로 다른 libgomp 가 둘 이상), `model_served_mixed`(run 안에서 응답 모델이 단일값이 아님; run 끝에 그 run의 모든 행에 붙음), `env_overrides_dropped`(정보용)). **`protocol_ok=1` = 정보용 외 플래그가 하나도 없음** |
| `hostname`, `cpu_model`, `python_version`, `numpy_version`, `numpy_blas`, `sdk_version`, `test_seed`, `test_hash`, `env_hash` | 재현성 지문. `env_hash`가 다르면 비교 불가. 새 언어의 도구 버전(rustc·numba·halide)은 지문에 넣지 않고 `compiler_version` 열과 result.json `toolchain` 에 남긴다 |
| `asan_findings`, `ubsan_findings` | `safety.enabled` 이고 C 백엔드일 때 보고 수. 돌리지 않았거나 끝까지 못 갔으면 NA(사유는 `result.json` 의 `stages.sanitize`) |
| `lint_findings_total`, `lint_memory`, `lint_numeric`, `lint_concurrency`, `lint_shape_dtype`, `lint_api_misuse`, `lint_performance` | 0 = 돌았고 없음. NA = 린트가 돌지 않음(꺼짐, 플러그인 실패, Rust처럼 도구 없음). 범주 하나만 NA = 그 범주를 볼 수 있는 도구가 끝까지 돌지 않음. `lint_findings_total` 은 검사한 범주의 합 |
| `perf_summary` | 성능 결정 요약 JSON(§1.4). 빌드·로드 실패나 비활성이면 NA |
| `roofline_pct` | 아직 채우지 않는다(NA) |
| `raw_path`, `built_path`, `harness_version` | 원본 위치, 하네스 버전(git describe; git 없으면 `nogit-<소스 해시>`). 소스 해시에는 하네스·분석 코드, 프롬프트, 과제 명세, 설정, `plugins/lint/{run_lint.py,rules.yaml}`, Rust 템플릿이 들어간다 |

## 8. 확정된 실험 조건 (2026-11-05·2026-11-12 사용자 확인; 근거는 `docs/decisions.md`)

1. **LLM** — 제공자 `anthropic`, 키 변수 `ANTHROPIC_API_KEY`, 모델 `claude-sonnet-5-5`, `effort=medium`, `thinking=adaptive`. Claude 4.7+/5.x는 `temperature`/`top_p`/`top_k`를 거부하고 `seed`가 없어 **제공자 기본 샘플링**을 쓴다(CLAUDE.md의 "temperature 0.2" 이탈). 비용은 Max 구독의 월 API 크레딧(Console 조직 연결 필요)으로 충당한다. 모델은 전체 실험 동안 바꾸지 않는다.
2. **머신** — `config/machines/<호스트>.yaml`(노트북: i7-1260P, P-core 4개 고정). 거버너는 측정 세션마다 `sudo cpupower frequency-set -g performance`(재부팅하면 되돌아감). 최종 측정 장비는 리눅스 데스크톱이다(2026-11-12) — `make probe`·`measurement.cores` 를 그 장비에 맞춘다(`docs/desktop_setup.md`).
3. **언어** — M1 언어 `numpy`, `c_seq`, `c_openmp`(설정 기본값). 추가 언어 `numba`, `halide_auto`, `halide_manual`, `rust`, `mojo`(2026-11-12 결정; `mojo` 는 아직 구현하지 않았다). 최종 장비에서 모든 조건을 다시 생성한다(실제 호출 직전에 비용을 다시 확인).
4. **측정 규약** — 구현마다 서로 다른 시각의 별도 워커에서 3라운드 × (워밍업 3 + 10회), 평균·95% 신뢰구간·중앙값 보고, `OMP_WAIT_POLICY=passive` 고정(2026-11-12). bench 루프의 fsync 제거는 새 장비의 A/A 점검 뒤에 정한다.
5. **저장소** — `https://github.com/Ustar2002/graduation_work` (비공개). `harness_version`은 `git describe`.
6. **프롬프트의 하드웨어 사실 공개** — 승인. 모든 언어에 같은 사실을 준다.
7. **느린 커널의 반복 축소 규칙** — 승인(`repeats_reduced=1` 플래그).
8. Python 시그니처의 출력 버퍼 `C`, `atol=1e-3`, 정확성 형태 `tiny`·`rect` 추가 — 승인.

## 9. 상태

- **M1 (수직 슬라이스) — 완료(2026-11-05).** 더미 파이프라인 검증(11개 변형 일치, 단위 테스트 164개) 뒤 첫 실제 실행 `20261105-real`: `claude-sonnet-5-5` 9회 호출 모두 pass, 속도비 중앙값 numpy 1.01 / c_openmp 0.97 / c_seq 0.33, 비용 약 0.21달러. 관찰과 후속 제안은 `docs/decisions.md` "M1 완료" 절.
- **M2a (Tier 1 커널) — 완료(2026-11-08).** 실제 실행 `20261108-tier1`: 60회 호출 중 58 pass(실패 2건은 출력 형식 위반), 비용 약 1.2달러. 결과 해석은 `docs/decisions.md` "M2a 첫 실제 실행" 절.
- **M2b (Tier 2 커널) — 완료(2026-11-08).** 실제 실행 `20261108-tier2`: 60회 중 52 pass. nbody/numpy 는 5/5 가 행렬곱 재배열의 float32 상쇄로 실패, 형식 위반 2건, 병렬 스캔 버그 1건. 비용 약 1.2달러. 해석은 `docs/decisions.md` "M2b 첫 실제 실행" 절.
- **M2c (Tier 3 커널) — 완료(2026-11-09).** 12개 커널 × 3언어 × 5샘플 실제 실행 완료(180행 + M1 9행). 통과 numpy 55/60, c_seq 55/60, c_openmp 53/60. 결과표·해석: `docs/meeting_2026-11.md`, 근거: `docs/decisions.md`.
- **1차 실험 재측정 — 완료(2026-11-12).** 교차 측정(2026-11-09)·호출 친화도 수정(2026-11-11)·3라운드 평균(2026-11-12)으로 네 run 을 `--force --strict-protocol` 로 다시 쟀다(API 비용 0; 이전 측정은 `results/runs_measured_20261109.csv`). 중간보고서(`report/interim/`, `make report`)는 이 측정으로 확정했다. 근거: `docs/decisions.md` "1차 실험 재측정" 절.
- **M3–M6 (새 언어 `numba`·`halide_auto`·`halide_manual`·`rust`, 마이크로 조건·수정 루프, perf_summary·sanitizer·린트, 통계·그림) — 코드 완료, Windows 에서만 검증, Linux 검증 대기(2026-11-23).** 기존 세 언어와 12개 과제의 동작(프롬프트 해시, 워커 명령·환경, 행, 121열, 환경 지문)은 바뀌지 않았다. Windows 에서 확인한 것과 하지 못한 것은 `docs/decisions.md` 2026-11-23 항목에 있다.
  - Linux 에서 처음 확인되는 것: C 언어의 `harness.run`(gcc·nm), gcc + libgomp 의 OpenMP 와 numba 의 omp 계층, taskset·호출 친화도·거버너, 교차 측정의 실제 시간, sanitizer, cppcheck·clang-tidy, Linux 의 Rust 링크 인자와 `~/.cargo/bin` 탐색, Li2018 의 Linux 출력, `make aa`. 절차는 `docs/linux_test_plan.md`.
  - 사용자 확인이 필요한 결정: `docs/decisions.md` 2026-11-23 항목의 "사용자 확인이 필요한 결정"(2026-11-12 결정의 문구와 다른 구조·세부, 기존 동작에서 달라진 하네스 오류 처리, Halide 빌드와 `bench_timeout_s`, Li2018 민감도 분석 절차, 수정 요청의 C 크래시 정보, CLAUDE.md §6 의 `null` 범주). 실제 API 호출 전에 확정한다.
- 남은 일: `mojo`, Linux 일괄 검증(`docs/linux_test_plan.md`), 데스크톱에서 `make probe`·`make aa` 뒤 모든 조건의 재생성·측정(비용 확인 뒤), 대시보드, M7.
