# LLM 생성 HPC 코드 평가 하네스 (졸업프로젝트)

경희대 컴퓨터공학부 졸업프로젝트 **"프로그래밍 언어 선택과 언어 특성이 LLM 생성 HPC 코드의 정확성·성능·안전성에 미치는 영향 — 실증 분석(예비 연구)"** 의 실험 하네스다. 같은 과제를 여러 언어로 LLM에 생성시켜 **추출 → 빌드 → 린트 → 격리 실행 → 정확성 판정 → (sanitizer) → 시간 측정 → 결과 행** 까지를 자동으로 돌리고, 실패한 샘플에는 진단을 돌려주는 수정 루프를 돌린 뒤, pass@k·속도비·토큰·오류 유형을 부트스트랩 구간과 그래프로 집계한다. 핵심 가설은 "성능 격차는 추상화 수준이 아니라 성능 결정(루프 순서·타일링·벡터화·병렬화·메모리 배치)을 LLM이 직접 내려야 하는가로 설명된다"이다(`CLAUDE.md` §1).

하네스는 **Linux 전용 측정 도구**다. M2b–M6 코드는 Windows 개발 머신에서 작성·검증했고, Linux에서만 확인되는 부분은 데스크탑에서 한 번에 시험한다(§9 상태, `docs/linux_test_plan.md`).

- 프로젝트 지시서: `CLAUDE.md` (수정 금지)
- 설계 계약: `docs/harness_design.md`(v2.1, M1·M2a — 모듈 인터페이스·JSON 키·CSV 열), `docs/harness_design_v3.md`(v3, M2b–M6 확장)
- 결정 기록: `docs/decisions.md` · 측정 규약: `docs/protocol.md` · 오류 분류표: `docs/error_taxonomy.md` · Linux 시험 계획: `docs/linux_test_plan.md`

## 1. 하네스가 하는 일

### 1.1 과제 12개 (3 Tier)

공통 계약: 행우선, 입력 버퍼 읽기 전용, 출력 버퍼는 호출자가 할당, 출력 배열 1개. 정확성 형태 `tiny`·`rect`·`small`, 성능 형태 `large`. 명세는 `tasks/<task>/spec.md`(프롬프트에 그대로 들어간다), 계약은 `tasks/<task>/signatures.yaml`·`tolerance.yaml`, 근거는 `docs/decisions.md`.

| Tier | 과제 | `large` 형태 | 출력 비교 | 성능 기준선 (`tasks/<task>/baseline/`) |
|---|---|---|---|---|
| 1 | `gemm` | M=N=K=1024 | float32 허용오차 | NumPy `np.matmul` (OpenBLAS) |
| 1 | `jacobi2d` | 4096², T=20 | float32 허용오차 | 손 구현 C+OpenMP |
| 1 | `spmv_csr` | 10⁶행, nnz 약 1,600만 | float32 허용오차 | 손 구현 C+OpenMP |
| 1 | `dot` | n=2²⁵ | float32 허용오차 | NumPy `np.dot` |
| 2 | `histogram` | n=2²⁶, 256구간 | int32 정확 비교 | 손 구현 C+OpenMP |
| 2 | `prefix_sum` | n=2²⁶ (int32 포함형 누적 합) | int32 정확 비교 | NumPy `np.cumsum` |
| 2 | `nbody` | n=8192 | float32 허용오차 | 손 구현 C+OpenMP |
| 2 | `sort` | n=2²⁴ (int32) | int32 정확 비교 | NumPy 정렬 |
| 3 | `cholesky` | n=2048 | float32 허용오차 | NumPy `np.linalg.cholesky` |
| 3 | `floyd_warshall` | n=1024 | float32 허용오차 | 손 구현 C+OpenMP |
| 3 | `bfs` | 정점 2²², 간선 약 3,350만 | int32 정확 비교 | 손 구현 순차 C |
| 3 | `fft` | n=2²² (복소수, 실수·허수 교차) | float32 허용오차 | NumPy `np.fft.fft` |

`large` 형태는 잠정값이다(데스크탑에서 `make probe` 뒤 다시 정한다). 정수 출력 과제는 원소가 전부 같아야 통과하고 `numeric_instability` 가 없다(`docs/error_taxonomy.md`). Halide는 `sort`·`bfs` 를 지원하지 않는다(`signatures.yaml` 의 `unsupported`; 생성 단계가 건너뛰고 집계 표에 `n/a` 로 나온다).

### 1.2 언어 7개

| 언어 | 백엔드 | 빌드·실행 | 스레드 |
|---|---|---|---|
| `numpy` | `python` | 소스 그대로, 워커가 임포트 | `OMP_NUM_THREADS`·`OPENBLAS_NUM_THREADS` |
| `numba` | `numba` | `@njit("<서명>")` 명시 서명 필수(임포트 때 컴파일; 서명이 없으면 `signature_mismatch`) | `NUMBA_NUM_THREADS`, `NUMBA_THREADING_LAYER=omp` |
| `halide_auto` | `halide` | 파이프라인 빌드 함수(알고리즘만), 하네스가 오토스케줄러 `Mullapudi2016` 적용. JIT 시간은 측정에서 뺀다 | `HL_NUM_THREADS` |
| `halide_manual` | `halide` | 같은 빌드 함수 + LLM이 쓴 스케줄 | `HL_NUM_THREADS` |
| `c_openmp` | `c` | gcc `-O3 -march=native -fopenmp` | `OMP_NUM_THREADS` (+ `OMP_PROC_BIND` 등) |
| `c_seq` | `c` | gcc `-O3 -march=native` (스레드 심벌이 보이면 `contract_violation`) | 1 |
| `rust` | `rust` | `cdylib`, 의존성은 버전 고정 `rayon` 하나, `cargo build --release --offline --locked` | `RAYON_NUM_THREADS` |

빌드 플래그·스레드 수·코어 집합은 `config/experiment.yaml` 이 정한다. `run.languages` 기본값은 지금도 `numpy, c_seq, c_openmp` 이고, M3 언어는 `--languages`(또는 `make … LANGUAGES=…`)로 고른다. 처리할 언어에 필요한 도구(gcc, cargo, numba, halide, 오토스케줄러 플러그인)가 없으면 `harness.run` 은 **행을 쓰기 전에** exit 7로 멈춘다(없는 도구는 모델 오류가 아니다).

### 1.3 마이크로 조건 3개와 수정 루프

| 조건 | 값 | 내용 |
|---|---|---|
| `args` | `positional` · `keyword` | keyword: Python 계열은 키워드 전용 시그니처, C는 하네스가 주는 `kernel_args.h` 구조체, Rust는 `src/kernel_args.rs` |
| `grammar_hint` | `off` · `on` | on: `prompts/hints/<language>.md`(100줄 이하 문법 안내)를 붙인다. 힌트 파일은 `numba`, `halide_auto`, `halide_manual`, `rust` 에만 있고, 없는 언어는 이 조건에서 건너뛴다 |
| `feedback` | `compiler` · `compiler+lint` | 수정 요청에 린트 메시지 절을 더하는지 여부. 두 수준은 같은 턴 0 응답에서 갈라지게 한다(`--seed-from-feedback`, 대응 비교) |

조건 조합은 디렉터리 이름 `cond_tag = args-<args>_hint-<grammar_hint>_fb-<feedback>` 로 갈린다. M2a의 프롬프트 12개(Tier 1 네 과제 × numpy·c_seq·c_openmp, positional·hint off)는 바이트 단위로 그대로다(`prompt_hash` 불변을 테스트로 고정).

**수정 루프**(`harness/repair.py`): 직전 턴 행이 `valid_sample=1 ∧ pass=0 ∧ error_stage ≠ generate` 인 샘플에 진단을 돌려주고 다시 묻는다. 대화는 다중 턴 메시지(원래 프롬프트 → 응답 → 수정 요청 → …), 최대 3턴(`--max-turns`; 설정 `run.max_repair_turns` 기본 0). 진단은 컴파일러·테스트 러너가 줄 법한 정보까지만 준다(컴파일러 출력, 예외·시그널, 형태별 통과/실패와 최대 오차, 타임아웃). 하네스 내부 패턴 이름과 참조값은 주지 않는다. 턴마다 행이 하나 생긴다(`turn` 이 행 키에 있다).

### 1.4 단계와 보조 산출물

```
생성 게이트 → extract → build(백엔드별) → [lint] → import(--inspect) → [perf_summary] → check → [safety] → bench
```

- **린트**(`lint.enabled: true`): `plugins/lint/run_lint.py` 가 `CLAUDE.md` §6 계약의 기성 린터 베이스라인이다 — Python 계열 ruff·pyflakes, C cppcheck·clang-tidy, Rust 없음. 하네스를 임포트하지 않는 독립 CLI라 인지프 ML lint로 바꿔 끼울 수 있다(`plugins/lint/README.md`). 플러그인이 없거나 실패하면 경고하고 계속한다.
- **성능 결정 요약**(`perf_summary.enabled: true`): 빌드가 성공한 샘플마다 `CLAUDE.md` §5 의 JSON(+ `decisions_by_library` 키)을 만든다. 소스: C는 OpenMP pragma와 gcc `-fopt-info-vec` 리마크, Rust는 rayon 호출, Python·Numba는 AST와 Numba 진단, Halide는 `print_loop_nest()` 와 스케줄 호출. 정하지 못하면 `"unknown"` 이다.
- **sanitizer**(`safety.enabled: false` 가 기본): C 백엔드만, 정확성 형태로만 ASan/UBSan 빌드를 따로 돌린다. 판정이 `runtime_crash`·`wrong_output` 이고 ASan 보고가 있으면 `error_type=memory_error` 로 바꾼다. 통과한 샘플의 `pass` 는 바꾸지 않는다. TSan은 쓰지 않는다(`race_condition` 은 반복 실행 간 불일치로 판정).
- **분석**: `analysis/aggregate.py`(pass@k, 속도비, `--ci` 부트스트랩 95% 구간, `--coverage`, `--by-turn`, `--conditions`), `analysis/stats.py`(고정 seed 부트스트랩), `analysis/plots.py`(그림 4개). 대시보드는 아직 없다(`CLAUDE.md` 가 마지막으로 둔 항목).

## 2. 저장소 구조

```
config/experiment.yaml      실험 설정(과제·언어·조건·모델·빌드 플래그·halide·safety·lint·측정 규약·경로)
config/machine.yaml         머신 프로브 결과(`make probe` 가 생성)
tasks/<task>/               spec.md, signatures.yaml, tolerance.yaml, gen_tests.py, baseline/baseline.{py,c}, tests/(생성됨, git 제외)
                            (gemm 에만 reference.py, spec.ko.md 가 더 있다)
prompts/                    system.md, lang/<7개 언어>.md, hints/{numba,halide_auto,halide_manual,rust}.md, repair.md
harness/                    generate → extract → build/{c,python,numba,halide,rust}.py → run(오케스트레이션) ← worker(LLM 코드 실행 유일 장소)
                            repair(수정 루프), llm_client, config, protocol(측정 규약), classify(오류 분류), schema(CSV 스키마),
                            safety(sanitizer·린트 호출), perf_summary, sigs(언어별 시그니처 렌더링), taskspec, testgen, procutil(격리 실행), paths
harness/kernels/            워커의 커널 어댑터: cabi(C·Rust ctypes), pyfunc(numpy·numba), halide_pipe
harness/build/rust_template/  Cargo.toml·Cargo.lock(rayon 고정)·자리표시 src/lib.rs
harness/dummy_responses/<task>/<language>/<k>.md   더미 제공자의 고정 응답(0.md = 검증된 정답)
plugins/lint/               run_lint.py, rules.yaml, README.md
analysis/                   aggregate.py, stats.py, plots.py
tools/devcheck.py           개발용 커널 점검(측정 아님), tools/devcheck_shim/(zig 빌드용 omp.h 스텁·shim.h, --cc zig-omp 용 omp_real/omp.h)
tests/                      unittest (tests/fixtures/ 포함)
results/                    raw/, built/(git 제외), runs.csv, runs_dummy.csv, plots/(git 제외)
docs/                       decisions, protocol, error_taxonomy, harness_design(v2.1), harness_design_v3, linux_test_plan
```

## 3. Linux 설치 (요약 — 상세와 확인 절차는 `docs/linux_test_plan.md`)

필요: Linux, Python 3.14, gcc(`-fopenmp`, libasan·libubsan), `taskset`(util-linux), `nm`, Rust 툴체인(cargo 1.78 이상 — `Cargo.lock` v4), 린트용 `cppcheck`·`clang-tidy`(없으면 C 린트 열이 NA). sudo는 거버너 설정에만 쓰고 하네스는 시스템 설정을 바꾸지 않는다.

```sh
make venv                 # python3 -m venv .venv && .venv/bin/pip install -r requirements.txt  (다른 인터프리터: make venv SYSPY=python3.14)
make rust-setup           # cargo fetch --locked --manifest-path harness/build/rust_template/Cargo.toml  (온라인 1회; 이후 rust 빌드는 --offline)
make probe                # config/machine.yaml 생성 (CPU 토폴로지·거버너·BLAS·컴파일러·rustc·numba·halide·린트 도구 버전)
make tests                # tasks/*/tests/{tiny,rect,small,large}.npz + manifest.json (과제 12개, 약 1.9 GB; 시드 고정, 바이트 동일 재생성)
make unittest             # tests/ 단위 테스트 (Linux 에서 남아도 되는 skip 은 개발 머신 전용 — ziglang·Windows — 테스트뿐)
```

머신을 옮기면 `make probe` 뒤 `config/experiment.yaml` 의 `measurement.cores`·`single_core`·`threads` 를 그 머신에 맞추고, 기존 행과 섞어 비교하지 않는다(`env_hash` 가 달라진다).

**모든 Python 실행은 `.venv/bin/python`** 이다. `.venv`는 격리형(시스템 site-packages 미사용)이어야 한다 — 시스템 `python3-numpy`는 참조 BLAS(netlib)라 1024³ GEMM이 ~250 ms이고, pip 휠 `numpy==2.3.5`(OpenBLAS 내장)는 4스레드 ~15 ms(EPP `power`)·~8 ms(EPP `performance`)다(M1 노트북 실측). 하네스는 BLAS 정체를 행마다 기록하고 OpenBLAS가 아니면 run을 중단한다. 측정 전에 거버너·EPP를 `performance`로 두고 확인한다(`docs/protocol.md` §13):

```sh
sudo cpupower frequency-set -g performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
.venv/bin/python -m harness.protocol --check      # 플래그가 있으면 exit 1
```

## 4. 비-Linux 개발 머신에서 되는 것

Windows 개발 머신의 venv는 `.venv-win`(git 제외; Python 3.14, requirements 의 패키지 + `ziglang`, Rust는 MSVC 툴체인)이다. 그 머신에는 make가 없으므로 아래 CLI를 `.venv-win\Scripts\python.exe` 로 직접 부른다. `.venv/bin/python` 이 없으면 하네스는 현재 인터프리터를 워커에도 쓴다.

- **되는 것**: 단위 테스트(Linux 전용은 skip), `tools/devcheck.py`, 더미 제공자로 `harness.generate` → `harness.run` → `harness.repair` → `analysis.aggregate`/`analysis.plots`. `harness.run` 은 python 계열(`numpy`, `numba`, `halide_*`)과 `rust` 를 끝까지 처리한다. bench도 돌지만 코어 고정이 없어 시간은 의미가 없다.
- **표시**: 비-Linux에서 만든 행은 `protocol_flags` 에 `platform=<sys.platform>`(정보용 아님)·`governor=unknown`·`taskset_missing` 이 붙어 `protocol_ok=0` 이다. 분석에서 측정값으로 쓰지 않는다.
- **안 되는 것**: C 언어의 `harness.run`(gcc가 없어 행을 쓰기 전에 exit 7), taskset, 거버너, sanitizer, cppcheck·clang-tidy, 시간 측정.
- **devcheck 의 C 툴체인**: gcc가 없으면 `python -m ziglang cc` 로 **직렬** 빌드한다(OpenMP pragma 무시, `omp.h` 스텁 — 논리 검증만; 판정 줄에 `toolchain=zig-serial`). `--cc zig-omp` 를 주면 OpenMP 대상(c_openmp, `openmp: true` 인 C 기준선)을 `clang -fopenmp` + 이 머신의 LLVM OpenMP 런타임(Windows: `libomp140.x86_64.dll`, 다른 경로는 `DEVCHECK_LIBOMP`, 스레드 수는 `DEVCHECK_OMP_THREADS`)으로 실제 스레드로 돌린다(`toolchain=zig-omp`). 어느 쪽도 gcc + libgomp 확인을 대신하지 않는다.
- 개발 머신에서 파이프라인을 돌릴 때는 `config/experiment.yaml` 사본의 `paths` 를 작업 디렉터리 밖으로 바꿔 `--config` 로 넘긴다. 기본 설정 그대로 돌리면 `platform=` 플래그가 붙은 행이 git에 올라가는 `results/runs_dummy.csv` 에 섞인다. M2b–M6 작업의 개발 머신 실행은 모두 이런 설정 사본으로 했고 `results/` 에는 쓰지 않았다.

## 5. 단계별 명령

`make` 타깃은 아래 CLI의 얇은 감싸개다. 공통 선택 변수: `TASKS=a,b`, `LANGUAGES=x,y`, `N=<샘플 수>`, `SKIP_BENCH=1`(check까지만), `RUN_ID=…`(없으면 타임스탬프). `make help` 가 목록을 보여 준다. 옵션 전체는 각 CLI의 `--help`.

### 5.1 개발용 점검 (`tools/devcheck.py`, 측정 아님·CSV 안 씀)

```sh
make devcheck                          # tools/devcheck.py dummies — 모든 0.md 를 extract → build → worker check, 표로 출력
make devcheck LANGUAGES=numba,rust TASKS=gemm,dot     # DEVCHECK_CC=gcc|zig|zig-omp 로 C 툴체인을 고른다(기본 auto)
make devcheck-baselines [TASKS=…]      # 과제마다 tools/devcheck.py baseline --task <과제>
.venv/bin/python tools/devcheck.py kernel --task dot --language rust --file my.rs [--args-mode keyword] [--bench]
.venv/bin/python tools/devcheck.py gen --task fft
```

종료 코드 0 통과 / 1 커널이 틀림 / 2 사용법·도구 문제. `dummies` 를 `--languages` 없이 돌리면 설정의 언어와 `0.md` 가 있는 언어 전부가 대상이고, 지원한다고 돼 있는데 `0.md` 가 없는 조합은 `missing`(실패)이다. Linux 기대값은 80 pass, 4 unsupported(Halide × `sort`·`bfs`).

### 5.2 더미 파이프라인 (API 키 없이 전체 경로)

```sh
make dummy                              # 설정 그대로: run.tasks(12개) × run.languages(numpy,c_seq,c_openmp) × run.n_samples(5)
make dummy TASKS=gemm N=3               # generate 에 --tasks/--languages/--n 을 넘긴다
make dummy-all [LANGUAGES=…] [N=1]      # 언어 7개 × 과제 전부, n=1 → run → aggregate --coverage
make dummy-m3                           # numba,halide_auto,halide_manual,rust 만 (rust 가 있으므로 make rust-setup 먼저)
#   .venv/bin/python -m harness.generate --provider dummy --run-id $RUN_ID [--tasks …] [--languages …] [--n …]
#   .venv/bin/python -m harness.run --run-id $RUN_ID [--skip-bench]
#   .venv/bin/python -m analysis.aggregate --csv results/runs_dummy.csv --run-id $RUN_ID [--coverage]
```

`harness/dummy_responses/<task>/<language>/<k>.md` 의 고정 응답(샘플 i는 `i mod 파일 수`, 수정 턴 t는 `<k>.turn<t>.md`, 없으면 `<k>.md`)으로 생성 단계를 대신한다. gemm의 numpy·c_seq·c_openmp에는 실패 변형이 있고 기대 결과표는 `harness/dummy_responses/gemm/README.md` 와 `docs/error_taxonomy.md` §8에 있다. 다른 조합은 정답 `0.md` 하나다. 더미 행은 `provider=dummy` 로 **`results/runs_dummy.csv`에 분리 저장**된다(키가 없어도 실제 제공자에서 dummy로 자동 폴백하지 않는다). bench를 포함하면 과제·언어 수만큼 오래 걸린다 — check까지만 보려면 `SKIP_BENCH=1`.

### 5.3 실제 실행

```sh
export ANTHROPIC_API_KEY=...          # config/experiment.yaml model.anthropic.api_key_env
# 선택: LLM_PROVIDER=anthropic|openai_compat|dummy, LLM_MODEL=<모델명> 이 experiment.yaml 을 덮어쓴다.
# openai_compat 은 LLM_BASE_URL 과 OPENAI_API_KEY 가 필요하다.
make real RUN_ID=m2a-real TASKS=gemm,jacobi2d,spmv_csr,dot    # generate(확인 프롬프트) → run → aggregate
make real RUN_ID=… STRICT=1           # run --strict-protocol
.venv/bin/python -m harness.generate --run-id ID --languages numba,rust --tasks gemm --args keyword --grammar-hint on --yes
```

- **호출 수 주의**: `run.tasks` 가 이제 과제 12개라 `make real` 을 필터 없이 돌리면 12 × 3언어 × 5샘플 = **180회** 호출이다. M2a 계획(Tier 1 네 개, 60회)은 위 첫 줄처럼 `TASKS` 를 준다. 대량 호출은 단계마다 확인받는다(`CLAUDE.md` 원칙 5).
- **확인 게이트**: `harness.generate` 는 `preflight()`(모델 조회, 과금 없음) 뒤 계획(`<n> calls to <model> via <provider>, …`)과 과제 × 언어 표를 보여 주고 `Proceed? [y/N]` 을 묻는다. 비대화형에서는 `--yes` 가 없으면 exit 2로 끝나므로 스크립트에서는 CLI를 `--yes` 로 직접 부른다. 치명 API 오류(인증·404·400)는 raw를 저장한 뒤 즉시 중단(exit 3)한다. `run.tasks` 밖의 `--tasks` 는 호출 전에 거부한다.
- 재실행: 같은 `RUN_ID` 로 다시 돌리면 정상 응답은 건너뛰고 실패·절단 샘플만 재호출한다(`--no-retry-failed` 로 끔, `--force` 로 전부 재호출; 기존 파일은 `sample_<i>.prev<N>.json` 으로 보관). `harness.run` 도 raw 해시가 같고 요청한 bench가 이미 있는 샘플은 건너뛴다(`--force`). 예전에 `--skip-bench` 로 처리한 샘플은 bench를 요청한 run이 다시 처리한다. safety·lint만 빠진 결과는 알리고 건너뛴다(다시 처리하면 측정도 다시 하게 되므로; 채우려면 `--force`).
- 부분 실행: `harness.run --run-id ID [--languages a,b] [--samples 0,1] [--skip-bench] [--no-baseline]`.
- 규약 플래그는 경고와 함께 행의 `protocol_flags` 에 기록되고(`protocol_ok=0`) 실험은 계속된다. `STRICT=1` 은 run 시작의 정적 검사(거버너 `powersave`, EPP, `taskset` 없음, 비-Linux)에만 적용돼 exit 4로 중단한다. bench 직전의 동적 플래그(부하·코어 사용률·거버너 재확인), `baseline_drift`·`paired_baseline_failed`, `model_served_mixed` 는 기록만 된다 — 분석에서는 행의 `protocol_ok`/`protocol_flags` 로 가려낸다(`docs/protocol.md` §8–§10). BLAS가 OpenBLAS가 아니면 STRICT와 무관하게 항상 중단된다.
- `harness.run` 종료 코드: 0 정상, 1 입력 없음 등, 2 설정 오류, 4 규약·BLAS, 5 하네스 오류(행 없음), 6 기준선 게이트, 7 도구 없음(행을 쓰기 전에), 130 중단.
- 거부(refusal)는 다른 모델로 대체 실행하지 않고 `stop_reason=refusal` 로만 기록한다.

### 5.4 수정 루프

```sh
make repair-dummy [LANGUAGES=numpy,c_seq] [REPAIR_TURNS=3] [SKIP_BENCH=1]
#   generate --provider dummy --tasks gemm --languages numpy,c_seq --n 5 --feedback compiler → run
#   LLM_PROVIDER=dummy harness.repair --run-id $RUN_ID --feedback compiler --max-turns 3
#   harness.repair --run-id $RUN_ID --seed-from-feedback compiler --feedback compiler+lint   # 턴 0 raw 복사(API 호출 없음)
#   run → LLM_PROVIDER=dummy harness.repair --run-id $RUN_ID --feedback compiler+lint --max-turns 3
#   analysis.aggregate --csv results/runs_dummy.csv --run-id $RUN_ID --by-turn --conditions
.venv/bin/python -m harness.repair --run-id ID --max-turns 3 [--feedback compiler+lint] [--tasks …] [--languages …] [--yes]
```

전제는 같은 run의 턴 0이 `harness.generate` + `harness.run` 으로 처리돼 있는 것이다. 제공자·모델은 설정과 환경변수(`LLM_PROVIDER`, `LLM_MODEL`)에서 읽고 턴 0을 만든 것과 다르면 거부한다. 실제 제공자에서는 턴마다 대상 수를 보여 주고 `generate` 와 같은 확인을 받는다. 수정 턴은 `harness.repair` 가 바로 처리해 행을 쓴다. 종료 코드: 0 정상, 1 설정/입력, 2 확인 거부, 3 치명 API 오류, 5 하네스 오류, 130 중단. 수정 턴을 처리하는 `harness.run` 의 준비 단계가 멈추면 그 코드(4 규약, 6 기준선, 7 도구 없음)를 그대로 돌려준다. gemm 수정 턴 더미는 `c_seq/1.turn1.md`, `numpy/2.turn1.md` 다(기대표는 `harness/dummy_responses/gemm/README.md`).

### 5.5 sanitizer (C만)

설정 사본에서 `safety: {enabled: true, timeout_s: 300}` 으로 바꿔 `--config` 로 넘긴다. 예(더미 run): `.venv/bin/python -m harness.run --config cfg_san.yaml --run-id ID --languages c_seq,c_openmp --force --skip-bench`. `--force --skip-bench` 는 샘플을 bench 없이 다시 처리해 같은 행 키의 행을 시간 열이 빈 행으로 바꾸므로, 이미 측정한 run 에서는 `--skip-bench` 를 빼고 돌린다(측정도 다시 한다). 로그 확인은 `.venv/bin/python -m harness.safety --parse-logs <샘플>/san`, 린트 단독 실행은 `.venv/bin/python -m harness.safety --lint <파일> --language c_openmp`.

### 5.6 집계·그래프

```sh
make aggregate                                   # results/runs.csv, --latest
make aggregate RUN_ID=… [CSV=results/runs_dummy.csv] [AGG_FLAGS="--ci --coverage --by-turn --conditions"]
make aggregate ALL_RUNS=1                        # --all-runs
make plots [RUN_ID=…|ALL_RUNS=1] [CSV=…] [OUT=results/plots] [FORMAT=png|pdf|svg]
.venv/bin/python -m analysis.aggregate [--csv …] [--run-id ID | --latest | --all-runs] [--json] [--ci] [--coverage] [--by-turn] [--conditions]
.venv/bin/python -m analysis.plots [--csv …] [--run-id ID | --latest | --all-runs] [--out DIR] [--format png|pdf|svg] [--by-task]
```

옵션 없는 `analysis.aggregate` 의 출력은 M2a와 같다. 행 키 중복은 `created_at` 최신만 남기고 그룹 `(run_id, provider, model, task, language, args, grammar_hint, feedback, turn)` 별로 `n`(=`valid_sample=1`), `n_pass`, `pass_at_1/5/n`(비편향 추정량, `k > n` 이면 NA), `speedup_median/min/max`, `tokens_per_pass`, `error_types` 등을 낸다. turn ≥ 1 행이 있으면 그룹 표는 turn 0만 세고 한 줄로 알린다.

- `--ci`: `pass_at_1`·`speedup_median` 의 95% 백분위 부트스트랩 구간(`_lo`/`_hi` 열; seed 0, 2000회). n=5에서는 구간이 넓고, 작은 n에서 백분위 부트스트랩은 실제보다 좁게 나온다(5/5면 구간이 [1, 1]이지만 Clopper–Pearson 95% 하한은 0.48 — `analysis/stats.py` docstring).
- `--coverage`: 과제 × 언어 표(턴 0 유효 샘플 수, `unsupported` 는 `n/a`, 행 없음은 `-`).
- `--by-turn`: 수정 턴 t까지 허용했을 때의 누적 통과율 `pass_within_t`, 누적 토큰 `tokens_cum_t`, `turns_to_pass` 분포.
- `--conditions`: 조건 수준 쌍의 통과율 차와 구간. `args`·`grammar_hint` 는 턴 0의 독립 표본, `feedback` 은 `(run_id, sample_idx)` 대응 비교로 두 수준 모두 결과를 아는 턴 수 H까지(`pass_within_<H>`).
- 그림(`results/plots/`, git 제외): `fig1_languages`(x: 성능 결정 부담 순 `numpy, numba, halide_auto, halide_manual, c_openmp, c_seq, rust`, y: pass@1과 속도비, 과제별 패널), `fig2_error_types[_by_task]`, `fig3_condition_effects`, `fig4_repair_turns`. 자료가 없는 그림은 건너뛰고 알린다.

## 6. 산출물 위치

| 경로 | 내용 |
|---|---|
| `results/raw/<run_id>/<task>/<language>/<cond_tag>/sample_<i>.json` | 턴 0 생성 원문: 프롬프트·응답·토큰·stop_reason·모델·설정 스냅샷 |
| `…/<cond_tag>/sample_<i>.turn<t>.json` | 수정 턴 t 원문(`prompt.messages`, `repair: {parent_raw, feedback, diagnostics, …}`) |
| `…/<cond_tag>/sample_<i>.prev<N>.json` | generate가 재호출할 때 보관한 이전 원문 |
| `results/built/<run_id>/<task>/<language>/<cond_tag>/sample_<i>/` | 턴 0 산출물(아래 표) |
| `…/sample_<i>/turn_<t>/` | 수정 턴 t 산출물(같은 구성) |
| `results/built/<run_id>/<task>/baseline/` | `bench_run.json`(run 시작 기준선), `bench_1t.json`(1스레드), C 기준선이면 `build/` |
| `results/built/<run_id>/_cargo_target/` | rust 샘플이 같이 쓰는 `CARGO_TARGET_DIR`(rayon은 한 번만 빌드). 그 안의 `_toolchain_probe/` 는 run 시작의 도구 점검 빌드(`probe.log`) |
| `results/runs.csv` / `results/runs_dummy.csv` | 실제 제공자 행(git에 올리는 결과) / `provider=dummy` 행 |
| `results/plots/` | `analysis.plots` 출력 |

샘플(턴) 디렉터리의 파일:

| 파일 | 언제 |
|---|---|
| `kernel.py` · `kernel.c` · `src/lib.rs` + `Cargo.toml`·`Cargo.lock` | 추출된 소스(언어에 따라) |
| `kernel_args.h` · `src/kernel_args.rs` | `args=keyword` 의 C·Rust |
| `kernel.so` | C·Rust 빌드 산출물(Windows rust는 `kernel.dll`) |
| `build.json`, `build.log` | 빌드 결과(+ `extra`), 컴파일러·cargo 출력(C는 벡터화 리마크 포함; python 계열은 `build.log` 없음) |
| `lint.json`, `lint.stderr.log` | 린트 플러그인 출력 원문과 stderr |
| `import.json`, `check.json`, `bench.json`, `safety.json` | 워커 결과 문서(`import.json` 의 `inspect` 는 Numba·Halide 관찰값, `safety.json` 은 sanitizer 실행) |
| `<stage>.stdout.log`, `<stage>.stderr.log` | 워커 단계별 출력(`import`, `check`, `bench`, `safety`, `paired_baseline`) |
| `paired_baseline.json` | bench 직전의 샘플별 대응 기준선 |
| `san/` | sanitizer 빌드·로그(`kernel.c`, `kernel_san.so`, `build.json`, `build.log`, `asan.<pid>`, `ubsan.<pid>`) |
| `result.json` | `{"row", "raw_sha256", "stages", "requested", "written_at"}` — 행 + 단계별 상세(`extract`, `build`, `lint`, `import`, `perf_summary`, `check`, `safety`/`sanitize`, `dynamic`, `paired_baseline`, `bench`), `requested` = 이 run이 요청한 선택 단계(`bench`, `safety`, `lint`) |

`result.json` 이 진실의 원천이다. CSV 헤더가 스키마와 달라지면 `.venv/bin/python -m harness.run --rebuild-csv` 로 다시 만든다(result.json이 없는 기존 CSV 행은 남긴다). 같은 행 키 `(run_id, task, language, args, grammar_hint, feedback, sample_idx, turn)` 는 교체(upsert)된다.

## 7. 결과 행 — 뜻이 바뀌었거나 새로 채워지는 열

**열은 추가하지 않았다**(115열, 기존 CSV와 헤더가 같다). 전체 목록은 `harness/schema.py` `COLUMNS`, 뜻은 `docs/harness_design.md` §11과 `docs/harness_design_v3.md` §10. 직렬화: 불리언 `1`/`0`, NA 빈칸, 리스트/딕트 JSON 문자열.

| 열 | 지금의 뜻 |
|---|---|
| `backend` | `python`, `numba`, `halide`, `c`, `rust` |
| `args`, `grammar_hint`, `feedback` | 실제 조건 값(§1.3). M1·M2a 행은 `positional`/`off`/`compiler` |
| `turn` | 0 = 첫 시도, t ≥ 1 = 수정 턴 t. 턴마다 행 하나 |
| `turns_to_pass` | 그 행이 통과했으면 그 행의 `turn`, 아니면 NA. 샘플 수준 분포는 `aggregate --by-turn` |
| `prompt_hash`, `tokens_*`, `latency_ms` | 수정 턴 행은 그 턴 호출의 값(입력에 앞 대화 전체가 들어간다). 턴 행의 `prompt_hash` 는 system + 메시지 목록의 해시 |
| `error_type`, `error_detail` | sanitizer가 켜져 있으면 `memory_error` 가 나온다: 판정이 `runtime_crash`·`wrong_output` 이고 ASan 보고가 있을 때, `error_detail = asan:<종류>; was <이전 type>/<이전 detail>` |
| `contract_violation` | 관찰값에 `schedule_in_halide_auto`(halide_auto 응답이 스케줄을 직접 씀) 추가 |
| `compiler`, `compiler_version` | c: `gcc`+버전, rust: `rustc`+`rustc --version`, numba: `numba`+버전, halide: `halide`+버전, numpy: `python`+버전 |
| `cflags` | c: 전체 플래그 문자열, rust: `RUSTFLAGS=…; profile.release: …`, halide_auto: `autoscheduler=<이름> params=<JSON>`(`parallelism` 포함), halide_manual: `autoscheduler=none`, numpy·numba: NA |
| `omp_env` | 언어별 스레드 변수 추가: `NUMBA_NUM_THREADS`, `NUMBA_THREADING_LAYER`, `RAYON_NUM_THREADS`, `HL_NUM_THREADS` |
| `gflops` | 부동소수 연산 수(`work`)가 정의되지 않은 정수 과제(`histogram`, `prefix_sum`, `sort`, `bfs`)는 NA |
| `baseline_*`, `speedup_vs_baseline` | 과제별 기준선(§1.1) 대비. `baseline_drift_pct` 와 `baseline_drift=…%` 플래그는 **그 run의 과제별 대응 기준선 중앙값** 대비(2026-11-05 변경) |
| `asan_findings`, `ubsan_findings` | `safety.enabled` 이고 C 백엔드일 때 보고 수. 돌리지 않았거나 끝까지 못 갔으면 NA(사유는 `result.json` 의 `stages.sanitize`) |
| `lint_findings_total`, `lint_<범주>` | 범주는 `memory`, `numeric`, `concurrency`, `shape_dtype`, `api_misuse`, `performance`. 0 = 돌았고 없음. NA = 린트가 돌지 않음(꺼짐, 플러그인 실패, Rust처럼 도구 없음). 범주 하나만 NA = 그 범주를 볼 수 있는 도구가 끝까지 돌지 않음. `lint_findings_total` 은 검사한 범주의 합 |
| `perf_summary` | 성능 결정 요약 JSON(§1.4). 빌드 실패·비활성이면 NA |
| `protocol_flags`, `protocol_ok` | `platform=<sys.platform>`(비-Linux, 정보용 아님) 추가 |
| `env_hash`, `harness_version` | 지문에 rustc·numba·halide·린트 도구 버전이, 소스 해시에 `plugins/lint/{run_lint.py,rules.yaml}` 와 Rust 템플릿이 들어간다 |
| `roofline_pct` | 아직 채우지 않는다(NA) |

## 8. 확정된 실험 조건 (2026-11-05 사용자 확인; 근거는 `docs/decisions.md`)

1. **LLM** — 제공자 `anthropic`, 키 변수 `ANTHROPIC_API_KEY`, 모델 `claude-sonnet-5-5`, `effort=medium`, `thinking=adaptive`. Claude 4.7+/5.x는 `temperature`/`top_p`/`top_k`를 거부하고 `seed`가 없어 **제공자 기본 샘플링**을 쓴다(CLAUDE.md의 "temperature 0.2" 이탈). 비용은 Max 구독의 월 API 크레딧으로 충당한다. 모델은 전체 실험 동안 바꾸지 않는다.
2. **머신** — M1·M2a는 `config/machine.yaml`(i7-1260P 노트북, P-core 4개 고정). Linux 데스크탑으로 옮기면 `make probe` 와 `measurement.cores` 를 다시 맞춘다.
3. **언어** — M1 언어 `numpy`, `c_seq`, `c_openmp`(설정 기본값). M3 언어 `numba`, `halide_auto`, `halide_manual`, `rust` 는 코드가 준비돼 있고 실제 실행은 확인 뒤.
4. **저장소** — `https://github.com/Ustar2002/graduation_work` (비공개). `harness_version` 은 `git describe`.
5. **프롬프트의 하드웨어 사실 공개**, **느린 커널의 반복 축소 규칙**(`repeats_reduced=1`), Python 시그니처의 출력 버퍼, 정확성 형태 `tiny`·`rect`, **drift 기준 변경** — 승인.

## 9. 상태

- **M1 (수직 슬라이스) — 완료(2026-11-05, Linux 노트북).** 첫 실제 실행 `20261105-real`: `claude-sonnet-5-5` 9회 호출 모두 pass, 속도비 중앙값 numpy 1.01 / c_openmp 0.97 / c_seq 0.33, 비용 약 0.21달러(`docs/decisions.md` "M1 완료").
- **M2a (Tier 1) — 더미 검증 완료(2026-11-05, Linux 노트북), 실제 호출 대기.** 4과제 × 3언어 × 5샘플 = 60회(`make real … TASKS=gemm,jacobi2d,spmv_csr,dot`). 그 뒤 하네스 코드가 바뀌었으므로 Linux 시험에서 M2a 더미 회귀를 다시 확인한다.
- **M2b–M6 (Tier 2·3 커널, M3 언어, M4 조건·수정 루프, M5 perf_summary·sanitizer·린트, M6 분석·그래프) — 코드 완료, Windows에서 검증, Linux 시험 대기.** 2026-11-22 기준 커밋 전이다.
  - Windows에서 확인한 것(2026-11-22): 단위 테스트 1320개 통과(Linux 전용 52개 skip). `devcheck dummies` 로 새 언어 4개 × 과제 12개 44개 pass·4개 unsupported, `--cc zig-omp` 로 C 더미 24개 pass(c_openmp는 실제 스레드)와 C+OpenMP 기준선 5개 pass, gemm의 race 더미가 3회 모두 `race_condition`. 새 커널 8개는 독립 구현(명세와 시그니처만 보고 numpy·c_seq·c_openmp 24개)이 모두 통과했다. 더미 제공자로 python 계열·rust의 generate → run(bench 포함, 시간은 무의미) → repair(두 피드백 조건) → aggregate(`--ci --coverage --by-turn --conditions`) → plots → `--rebuild-csv`. 리뷰에서 확인된 27건은 모두 고쳤다.
  - Linux에서 처음 확인되는 것: C 언어의 `harness.run`(gcc 빌드·libgomp), taskset·거버너·실제 시간, sanitizer(ASan/UBSan), cppcheck·clang-tidy, gcc `-fopt-info-vec` 로그의 perf_summary 파싱(지금 픽스처는 손으로 조립), C 키워드 모드 끝에서 끝까지(키워드 C 더미가 없다), 큰 형태에서 Halide·Numba·Rust의 측정, Linux용 numba·halide 휠 설치. 절차와 기대값은 `docs/linux_test_plan.md`.
  - 확인이 필요한 결정(`docs/decisions.md` 2026-11-22 항목의 "6. 사용자 확인이 필요한 결정"; 실제 API 호출 전에 확정한다): 값을 바꾸지 않고 남겨 둔 것 — fft 허용오차와 기준선, numba 의 스레드 바인딩, Halide JIT 시간과 bench 제한 시간, 부하 안정 대기 `measurement.max_settle_s`(지금 0 = 꺼짐), Halide manual 의 `allow_race_conditions()`, bfs 명세가 공개하는 작업 부하 사실, 수정 요청에 넣는 컴파일 언어의 크래시 정보, CLAUDE.md §6 린트 예시와 `null` 범주. 구현하면서 정한 것(허용오차 일부, 판정 규칙, 린트·sanitizer 열의 NA, Halide 오토스케줄러 `Mullapudi2016`·Numba 스레딩 계층 `omp` 등)은 이견이 있으면 되돌린다. `large` 형태는 데스크탑에서 다시 정한다.
- 하지 않은 것: 실제 API 호출(M2a 이후 전부), 대시보드, M7.
