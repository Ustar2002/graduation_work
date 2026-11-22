# 측정 규약 (docs/protocol.md)

이 문서는 **코드가 실제로 강제하는** 측정 규약을 적는다. 구현은 `harness/protocol.py`(정적·동적 검사, 코어 고정, 환경 화이트리스트), `harness/worker.py`(버퍼·센티널·반복·주파수 샘플링), `harness/run.py`(기준선·게이트·행 기록), `harness/procutil.py`(프로세스 격리)에 있다. 설정값은 `config/experiment.yaml`의 `measurement`·`build` 절이고, 설계 계약은 `docs/harness_design.md` §0, §7–§9와 `docs/harness_design_v3.md` §3(언어 설정)·§8.6(교차 측정과 호출 친화도의 언어별 적용)·§9.5(준비 점검과 종료 코드)다. 사람이 손으로 지켜야 하는 것은 **거버너/EPP를 `performance`로 두는 것**뿐이며(sudo 필요), 나머지는 전부 코드가 확인하고 행에 기록한다.

**언어 범위.** 이 규약은 `config/experiment.yaml` 의 일곱 언어(`numpy`, `c_seq`, `c_openmp`, `numba`, `halide_auto`, `halide_manual`, `rust`)에 똑같이 적용된다. 새 네 언어가 규약에 들어오는 자리는 §2(빌드 접두·run 시작 시험), §3(언어별 환경변수·`omp_runtime`), §4(빌드·JIT 와 측정 구간), §10.1(호출 친화도)이고, 근거는 `docs/decisions.md` 2026-11-11·2026-11-12·2026-11-23 이다. 새 언어의 측정 경로는 개발용 Windows 에서만 확인했고(시간 값은 의미 없음) Linux 측정 장비에서는 아직 돌지 않았다. GEMM 에 대한 수치(§1, §14)는 M1 의 Linux 노트북 실측이다.

## 1. 측정 머신과 고정 자원

아래 표는 M1 측정 머신(Linux 노트북)이다. 머신 프로필은 호스트별 `config/machines/<hostname>.yaml` 이고, 다른 머신에서는 첫 run 전에 `make probe`(= `python -m harness.protocol --probe --write`)로 그 머신의 프로필을 만들고 `measurement.cores`·`single_core` 를 그 CPU 의 토폴로지에 맞춰 정한다(`--suggest-cores N` 이 후보를 낸다). 프로필의 호스트 이름이 현재와 다르면 하네스가 경고하고, 이 머신에 없는 코어 집합은 §2 의 실행 접두 시험이 run 시작 전에 잡는다. 개발용 Windows 에서 만든 행은 `platform=win32` 플래그로 `protocol_ok=0` 이다(§8).

| 항목 | 값 |
|---|---|
| CPU | 12th Gen Intel Core i7-1260P (Alder Lake 하이브리드): P-core 4개 = CPU 0–7(HT 쌍 0/1, 2/3, 4/5, 6/7, 최대 4.7 GHz), E-core 8개 = CPU 8–15 |
| ISA | AVX2 + FMA, AVX-512 없음. `gcc -march=native` → `-march=alderlake -mtune=alderlake` |
| 캐시 | L1d 48 KiB/P-core, L2 1.25 MiB/P-core, **L3 18 MiB 공유** |
| 메모리 | 14 GiB |
| 전원 | `intel_pstate` HWP. 기본 상태는 governor `powersave`, EPP `power`, turbo 켜짐 — 이 상태에서 P-core는 부하 중에도 1.3–1.7 GHz에 머물고 샘플 간 400–1800 MHz로 요동한다(실측) |
| 컴파일러 | gcc 15.2.0 고정(clang 21은 `omp.h` 없음). `perf stat`은 `perf_event_paranoid=4`라 못 쓴다 |
| Python | `.venv/bin/python` 3.14, NumPy 2.3.5 pip 휠(scipy-openblas 0.3.30 내장). 시스템 python3-numpy는 참조 BLAS라 사용 금지 |

실측 보정값(`taskset -c 0,2,4,6`, powersave): NumPy `np.matmul(out=)` 1024³ 4스레드 14.7 ms / 2스레드 31 ms / 1스레드 51 ms; 순진한 C i-j-k 1스레드 ≈ 9.5 s/호출; C i-k-j(`restrict`) ≈ 240 ms.

**절대 시간은 주파수 상태에 따라 달라진다.** 위 값은 governor `powersave`·EPP `power`(P-core 1.3–1.7 GHz)에서 잰 것이다. 통합 검증(2026-11-04 `make dummy RUN_ID=dummy-m1`, governor `powersave`·EPP `performance`, bench 중 `cpu_mhz_median` 2.6–2.9 GHz)에서는 같은 기준선이 run 시작 4스레드 **7.7–8.3 ms**(대응 기준선 7.7–8.9 ms), 1스레드 19.5–21.7 ms(4.1 GHz)로 약 절반이었고, 더미 c_seq/0(i-k-j, `restrict`)은 77 ms, c_openmp/0은 29 ms였다. 그래서 비교는 대응 기준선 대비 속도비(`speedup_vs_baseline`)로 하고 행마다 `governor`·`epp`·`cpu_mhz_*`·`throttle_events`를 기록한다(§11; 설계 §0).

## 2. 코어 고정 (pinning)

- 모든 측정은 **P-core 4개, 코어당 1스레드**: `measurement.cores = "0,2,4,6"` → `taskset -c 0,2,4,6`. SMT 형제(1, 3, 5, 7)와 E-core(8–15)는 쓰지 않는다. 하이브리드 CPU에서 E-core/SMT 형제를 섞으면 OpenMP 정적 스케줄 불균형과 SMT 간섭으로 분산이 커진다(형제 코어 spin 1개로 OpenBLAS 1024³가 14.6 → 19.9 ms, +36% 실측).
- `threads: 1`인 언어(`c_seq`)는 **단일 P-core** `measurement.single_core = "2"`로 더 좁게 고정한다(`taskset -c 2`; cpu0은 IRQ가 몰려 피한다). 코어를 하나만 주므로 "스레드 사용 금지"가 코드로 강제된다. 이것은 **import·check 단계**의 고정이다. bench 는 교차 측정이라 언어와 무관하게 고정 코어 네 개로 뜨고, `c_seq` 커널은 호출되는 동안 그 첫 CPU 하나에 묶인다(§10.1).
- `protocol.launch_prefix(cfg, threads)`: `threads == 1`이면 `["taskset", "-c", single_core]`, 아니면 `["taskset", "-c", cores]`. `taskset`이 없으면 `[]`로 실행하고 플래그 `taskset_missing`을 단다(행 `taskset_used=0`).
- 컴파일(`gcc`)도 같은 접두로 P-core에서 실행해 `-march=native` 해석을 고정한다(`march_resolved` 열). Rust 의 `cargo build` 도 그 언어의 스레드 수에 맞는 접두로 실행한다(`-C target-cpu=native`). numba·halide 의 빌드 단계는 서브프로세스 없이 구문·AST 만 본다(컴파일은 워커 안에서 한다, §4).
- 기준선(`tasks/gemm/baseline/baseline.py`)도 같은 코어 집합·스레드 수·환경변수로 측정한다.
- **run 시작 시험**(`harness.run` 의 `setup_run`; 행을 쓰기 전, 실패는 모두 exit 7):
  - 도구 점검: 처리할 언어의 C 컴파일러(`build.cc`)·cargo·rustc·numba·halide·(halide_auto 면) 설정한 오토스케줄러의 플러그인, 그리고 bench 를 하는 run 에 C 기준선 과제가 있으면 C 컴파일러.
  - 실행 접두 시험(`probe_launch_prefixes`): run 이 쓸 접두마다 `[*접두, PYTHON_EXE, -c, pass]` 를 한 번 돌린다(명령당 60 s; bench 를 하는 run 은 `measurement.threads` 와 1스레드 접두도 포함). `cores`·`single_core` 가 이 머신에 없는 코어를 가리키면 taskset 이 모든 명령을 rc 1 로 끝내 빌드는 `compile_error`, 워커는 `runtime_crash` 행이 되므로 여기서 멈춘다. 접두가 비어 있으면(taskset 없음, 비-Linux) 건너뛴다.
  - 언어 런타임 시험(`probe_language_runtimes`): halide_auto 의 오토스케줄러·매개변수와 numba 의 스레딩 계층을 그 언어의 접두·워커 환경으로 띄운 서브프로세스에서 워커와 같은 코드로 한 번 써 본다(`python -m harness.protocol --probe-runtime halide|numba`).
  - SciPy 점검(`probe_numba_scipy`): numba 언어가 있으면 워커와 같은 인터프리터·환경에서 `scipy` 가 임포트되는지 본다. 임포트되거나 점검 프로세스가 결과 없이 끝나면 멈춘다 — Numba 는 컴파일한 함수 안의 `np.dot`·`@`·`np.linalg` 를 SciPy 의 BLAS 로 잇는데, 측정 환경에는 SciPy 를 두지 않는다(decisions.md 2026-11-12).
  - Rust 툴체인 시험(`probe_rust_toolchain`): 템플릿의 자리표시 크레이트를 샘플 빌드와 같은 명령·접두로 run 의 공유 cargo target 디렉터리에서 한 번 빌드한다(의존성 빌드는 샘플이 재사용한다).

## 3. 스레드 수와 환경변수 화이트리스트

워커(`harness.worker`)의 환경은 `protocol.launch_env(cfg, threads, openmp=…, extra_env=…)`가 **처음부터 새로** 만든다.

- 상속하는 변수: `PATH`, `HOME`, `LANG`, `TMPDIR`, `VIRTUAL_ENV`. 고정값: `LC_ALL=C`, `PYTHONHASHSEED=0`, `PYTHONPATH=<저장소 루트>`.
- 스레드 수: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 언어의 `threads`(`numpy`·`c_openmp`는 `measurement.threads=4`, `c_seq`는 1). NumPy는 `OPENBLAS_NUM_THREADS`(우선)/`OMP_NUM_THREADS`(폴백)로 제어됨을 실측으로 확인했다.
- OpenMP 런타임 위의 언어(`c_openmp`, `numba`; 아래 `omp_runtime`) 워커에만 `measurement.omp_env_extra`를 추가한다: `OMP_PROC_BIND=true`, `OMP_PLACES=cores`, `OMP_DYNAMIC=false`, `OMP_WAIT_POLICY=passive`(마지막은 2026-11-12, §10.2). bench 워커에는 커널이나 기준선 가운데 하나라도 OpenMP 면 추가한다(교차 측정).
- 사용자 셸의 `OMP_*`, `GOMP_*`, `OPENBLAS_*`, `MKL_*`, `KMP_*`는 **상속하지 않는다**. 그런 변수가 있었으면 한 번 경고하고 정보용 플래그 `env_overrides_dropped`를 단다.
- 행에는 `threads`, `cores`, `omp_env`(실제 넘긴 스레드 관련 변수 JSON; 접두 `OMP_`, `OPENBLAS_`, `MKL_`, `GOMP_`, `KMP_`, `NUMBA_`, `RAYON_`, `HL_`)가 기록된다. bench 를 실행한 행에서는 셋 다 **bench 에서의 값**이다: `threads`·`cores` 는 커널이 호출되는 동안의 스레드 수와 CPU 집합, `omp_env` 는 bench 워커가 받은 변수(§10.1). bench 를 하지 않은 행은 check 단계의 값이다.

**`omp_runtime`(OpenMP 런타임 위의 언어).** 언어 설정의 불리언 `omp_runtime` 은 그 언어의 커널이 libgomp 위에서 도는가를 뜻한다. `c_openmp` 는 `openmp` 값에서 정해지고, `numba` 는 `env.NUMBA_THREADING_LAYER == omp` 인지와 같아야 한다(생략하면 그 값; `experiment.yaml` 에 `omp_runtime: true`). 그 밖의 언어(numpy, c_seq, halide, rust)는 거짓이고 참으로 둘 수 없다 — 어긋나면 `load_config` 가 설정을 거부한다. 위의 "OpenMP 백엔드" 규칙은 이 값으로 정해진다: import·check·sanitizer 워커와 행의 환경은 `openmp=lang.omp_runtime`, bench 워커는 커널의 `omp_runtime` 또는 기준선의 `openmp` 가 참이면 `omp_env_extra` 를 받는다. 기존 세 언어에서는 `omp_runtime` 이 `c_openmp` 에서만 참이라 환경이 그대로다.

**언어별 환경변수.** 새 언어의 런타임은 스레드 수를 `OMP_NUM_THREADS` 가 아닌 변수에서 읽으므로, 언어 설정의 `env` 가 워커 환경에 마지막으로 더해진다(`extra_env=lang.env`; 값의 `${threads}` 는 그 언어의 스레드 수로 치환된다). import·check·sanitizer·bench 워커가 모두 같은 값을 받는다.

| 언어 | `threads` | `omp_runtime` | 더하는 변수 |
|---|---|---|---|
| `numba` | `measurement` (4) | 참 | `NUMBA_NUM_THREADS=4`, `NUMBA_THREADING_LAYER=omp` (+ `omp_env_extra`) |
| `halide_auto`, `halide_manual` | `measurement` (4) | 거짓 | `HL_NUM_THREADS=4` (halide_auto 는 오토스케줄러 매개변수 `parallelism` 도 4) |
| `rust` | `measurement` (4) | 거짓 | `RAYON_NUM_THREADS=4` |

- 이 변수(`NUMBA_NUM_THREADS`, `HL_NUM_THREADS`, `RAYON_NUM_THREADS`)가 빠졌거나 그 언어의 스레드 수와 다르면 `load_config` 가 설정을 거부한다 — 빠뜨리면 런타임이 논리 코어 수만큼 스레드를 띄우고 프롬프트의 스레드 수 서술과 행의 `threads` 가 사실이 아니게 된다. 자리표시는 `${threads}` 만 허용한다.
- halide_auto 의 오토스케줄러는 `halide.autoscheduler`(기본 **Li2018**; decisions.md 2026-11-12)이고 `parallelism` 은 하네스가 스레드 수로 넣는 예약어다. 설정의 `halide.autoscheduler_params` 는 스케줄러별 허용 표로 검사한다: Li2018 없음, Mullapudi2016 `last_level_cache_size`·`balance`, Adams2019 `beam_size`(Halide 는 모르는 매개변수에서 예외 없이 프로세스를 끝낸다). Adams2019·Mullapudi2016 은 같은 응답을 다시 재는 민감도 분석용이다.
- 워커는 스레드 수를 따로 검사하지 않는다. numba 는 실제 스레드 수를 `import.json` 의 `inspect.num_threads` 로 남기고, halide 는 남기지 않는다.
- cargo·rustc 는 워커가 아니라 빌드 환경(`protocol.rust_tool_env()`)으로 띄운다: POSIX 는 `protocol.launch_env_minimal()`(화이트리스트, 스레드 1) + 사용자의 `CARGO_HOME`·`RUSTUP_HOME`·`RUSTUP_TOOLCHAIN`, 개발용 Windows 는 현재 환경 전체이고, 어디서나 PATH 앞에 실행하는 cargo 의 디렉터리를 붙인다. 실행 파일은 `protocol.rust_tool` 이 PATH → `$CARGO_HOME/bin` → `~/.cargo/bin` 순으로 찾는다. `build_rust` 가 그 위에 `RUSTFLAGS`(`protocol.rust_flags`: 설정의 `build.rust.rustflags` + Linux 에서만 `build.rust.rustflags_linux` = `-C link-arg=-Wl,--no-undefined`)·`CARGO_TARGET_DIR` 를 더한다. 프롬프트의 빌드 명령과 행의 `cflags` 도 같은 `protocol.rust_flags` 값을 쓴다.
- 개발용 Windows 에서만 워커 환경에 `PYTHONIOENCODING=utf-8` 을 더한다(Linux 의 워커 환경은 그대로다).

## 4. 워밍업·반복·중앙값

- **정확성(check)**: `correctness_tests = [tiny(7,5,3), rect(200,130,312), small(256³)]`. 테스트마다 `check_repeats = 2`회 호출하고, 호출마다 A·B를 원본으로 복원하고 C를 센티널로 채운다(§7). 두 호출의 출력이 허용오차 밖으로 다르면 `nondeterministic`(→ `race_condition`).
- **성능(bench)**: `performance_test = large(1024³)` 하나. `warmup = 3`회 뒤 `repeats = 10`회를 잰다. 시간은 `time.perf_counter_ns`로 **함수 호출 구간만** 잰다(입력은 이미 메모리에 있음; 파일 I/O 없음).
- 통계는 timed 호출만으로 계산한다: `time_median_ms`(**보고값**), `time_mean_ms`, `time_min_ms`, `time_max_ms`, `time_cv`(표준편차/평균, ddof=1), `gflops = 2·M·N·K / median_s / 1e9`. `time_samples_ms`에 warmup과 timed 전체 분포를 JSON으로 기록한다 — 최선값만 보고하지 않는다.
- bench 중에는 첫 timed 결과 `C_first`를 보관하고 이후 timed 호출마다(계측 밖에서) 비교한다: 비트 불일치 수 `nondeterministic_bitwise_runs`, 허용오차 초과 불일치 수 `nondeterministic_runs`. 마지막 호출 뒤 `final_check`(정확성 재판정), `inputs_mutated`, `oob_write`를 다시 확인한다.
- 과제 일반: 위 형태는 GEMM 의 것이고, 모든 과제가 같은 이름의 네 형태 `tiny, rect, small, large` 를 가진다(`correctness_tests = [tiny, rect, small]`, `performance_test = large`; 크기는 각 `tasks/<task>/gen_tests.py`). `gflops` 는 `signatures.yaml` 에 `work` 식이 있는 과제에서만 값이 있다. int32 출력 과제는 허용오차 0(rtol = atol = 0)이라 호출 간 비교도 정확 일치다(`docs/error_taxonomy.md` 부록 2).
- **빌드·JIT 와 측정 구간**: 측정 구간은 언어와 무관하게 커널 호출 하나다.
  - numba: 진입 함수와 모듈의 모든 `njit`/`jit` 함수가 명시 서명을 가져야 하므로(빌드 단계의 AST 검사) 컴파일은 워커가 모듈을 임포트할 때 끝난다. 이어 워커가 디스패처의 새 컴파일을 끄고, 하네스가 넘길 인자 타입을 컴파일된 서명이 컴파일 없이 받는지 확인한다 — 받지 않으면 로드 실패(`signature_mismatch`)라 측정 호출에서 컴파일이 일어날 수 없다. `cache=True` 는 금지다(빌드 단계에서 거절). 스레딩 계층은 LLM 코드를 임포트하기 전에 띄운다.
  - halide_auto / halide_manual: 워커가 테스트(형태)마다 파이프라인을 새로 만든다(빌드 함수 호출 → auto 면 오토스케줄 → JIT). 이 준비는 측정 구간 밖이고 `build_ms` 로 기록되며, bench 에서는 커널의 호출 친화도 집합(§10.1) 아래에서 한다. 두 모드 모두 빌드 함수가 그 형태의 구체적인 크기를 받는다.
  - rust: cargo 빌드는 run.py 의 빌드 단계(워커 밖)이고, 워커는 C 와 같은 ctypes 경로로 공유 라이브러리를 부른다.
  - 이 준비 시간은 첫 호출 시간으로 정하는 적응 규칙(§5)에는 들어가지 않지만, bench 워커 전체에 걸리는 `bench_timeout_s`(300 s)에는 들어간다 — 라운드마다 새 워커라 Halide 의 형태별 빌드와 Numba 의 임포트 컴파일도 라운드마다 되풀이된다. 빌드가 길면 측정할 수 있는 커널이 bench `timeout` 이 될 수 있으며, 상한을 늘릴지는 측정 규약 변경이라 사용자 확인으로 남아 있다(decisions.md 2026-11-23).

## 5. 느린 커널의 적응 축소 규칙 (`repeats_reduced`)

bench 예산 `budget = measurement.bench_timeout_s = 300 s`, 첫 warmup 호출 시간 `t1`에 대해:

1. `t1 · (1 + min_repeats) > 0.8 · budget` (즉 warmup 1회 + 최소 3회도 예산의 80% 안에 못 들어감) → **즉시** `classification = "timeout"`, `error_detail = "first_call_<t1>ms_projected_<…>s"`로 기록하고 종료한다. 행은 `pass=1`(정확성은 이미 통과), 시간 NA, `error_stage=bench`, `error_type=timeout`.
2. 그렇지 않고 `t1 · (warmup + repeats) > 0.8 · budget`이면 `warmup = 1`, `repeats = clamp(floor(0.8 · budget / t1) − 1, min_repeats=3, repeats=10)`으로 줄이고 `repeats_reduced = 1`을 단다. 행의 `warmup`·`repeats` 열은 **계획값**이다: 축소 행에서는 축소된 값(= 실제 수행 횟수)이고, 규칙 1의 즉시 timeout 행과 bench 크래시 행에서는 기본 계획값(3/10)이 그대로 남는다(실측: `results/runs_dummy.csv`의 dummy-m1x/numpy/sample_3 — `error_type=timeout`, `warmup=3`, `repeats=10`, `time_median_ms` NA, `time_samples_ms={"warmup":[186929],"timed":[]}`). 실제 호출 수는 항상 `time_samples_ms`의 리스트 길이로 본다.
3. 그 밖에는 기본값(3 + 10).

집계(`analysis.aggregate`)는 그룹별 `n_repeats_reduced`를 표시한다. 이 규칙은 CLAUDE.md의 "워밍업 3회 + 10회 반복"을 느린 커널에서 바꾸는 **측정 규약 변경**이므로 `docs/decisions.md`에 **사용자 확인 필요**로 올라가 있다(대안: 축소 없이 timeout 처리 = 데이터 손실).

## 6. 타임아웃과 프로세스 격리

| 단계 | 상한 | 초과 시 |
|---|---|---|
| 컴파일 | `build.compile_timeout_s = 60 s` | `error_stage=build`, `error_type=compile_error`, `error_detail=compile_timeout` |
| import(워커 1회, 로드만; halide 는 첫 정확성 형태의 빌드까지) | `check_timeout_s = 180 s` | **`error_stage=build`**, `error_type=timeout`, `pass=0` (로드 중 시그널·비정상 종료도 `error_stage=build`/`runtime_crash`; 열 enum에 `import`가 없다) |
| check(워커 1회, 세 테스트 전부) | `check_timeout_s = 180 s` | `error_stage=check`, `error_type=timeout`, `pass=0` |
| bench(워커 1회) | `bench_timeout_s = 300 s` | `error_stage=bench`, `error_type=timeout`, `pass=1`, 시간 NA (§5) |
| Rust 빌드(cargo) | `build.rust.build_timeout_s = 600 s` | `Compiling kernel` 뒤(커널 크레이트 컴파일 중)면 `compile_error`/`compile_timeout`, 그 전(의존성 단계)이면 도구 문제로 행 없이 exit 7 |
| 린트 플러그인 | `lint.timeout_s = 60 s` | 린트 열 NA, 실험은 계속 |
| sanitizer check(C 만) | `safety.timeout_s = 300 s` | `asan_findings`·`ubsan_findings` 는 보고가 없으면 NA |

- 모든 서브프로세스는 `procutil.run_isolated`로 **새 세션(프로세스 그룹)** 에서 시작하고, 타임아웃이면 그룹 전체에 `SIGKILL`을 보낸다(`subprocess.run(timeout=)`은 `cc1`·OpenMP 자식을 남긴다 — 실측). 단위 테스트가 타임아웃 뒤 `pgrep -g <pgid>`가 비어 있음을 확인한다.
- 워커는 시작 직후 `prctl(PR_SET_DUMPABLE=0)`(apport 코어 덤프 차단)과 `RLIMIT_AS = max_worker_mem_gb = 8 GiB`를 적용하고 적용값을 결과의 `limits`에 적는다. `RLIMIT_NPROC`은 사용자 전체 기준이라 걸지 않는다. sanitizer 실행만 `--max-mem-gb 0` 으로 `RLIMIT_AS` 를 걸지 않는다(ASan 의 그림자 메모리; `limits.disabled`). 비-Linux 에서는 `resource`·`prctl` 이 없어 제한이 걸리지 않는다(`limits.unsupported`).
- 비-POSIX(개발용 Windows)의 `run_isolated` 는 새 세션 없이 띄우고 타임아웃 때 직접 자식만 `proc.kill()` 한다(프로세스 그룹 정리 없음 — 측정용이 아니다).
- 워커의 결과 파일은 **매 커널 호출 직전과 직후** 원자적으로(tmp → `os.replace`) 갱신된다(`complete`, `in_progress{test, call_idx}`). 시그널로 죽어도 어느 테스트의 몇 번째 호출에서 죽었는지 행의 `error_detail`(`signal SIGSEGV during small call 1`)에 남는다.
- 워커는 stdout에 아무것도 쓰지 않는다. 커널이 stdout에 쓰면 `run.py`가 파일로 받아 `contract_violation`에 `stdout_output`을 적는다(분류에는 영향 없음).

## 7. 버퍼: 64바이트 정렬·가드 영역·센티널

- 모든 배열은 `alloc_aligned(n, guard)`로 만든다: `base = np.empty(n + 16 + guard, float32)`, 시작 오프셋을 64바이트 경계에 맞춘 `view`와 그 뒤 `guard_view`. A는 가드 1024, B는 1024, C는 `max(M,N)² − M·N + 1024`(전치·차원 교환 쓰기도 가드 안에 들어오게). 가드 영역은 센티널로 채우고 호출 뒤 센티널이 아닌 원소가 있으면 `oob_write=1`. 프롬프트의 "A, B, C는 64바이트 정렬"은 이 사실을 서술한 것이다.
- 과제 일반: 모든 배열에 앞쪽 가드 4096 원소, 입력 배열 뒤쪽 가드 1024, 출력 뒤쪽 가드는 `signatures.yaml` 의 `out_guard_extra` 식의 값(없으면 0) + 1024. int32 배열도 같은 규칙이고 채움값은 dtype 에 맞춘다. 출력 센티널은 과제의 `tolerance.yaml` 값이고, 입력 가드 값은 센티널과 다르다(정수 -1073741824, 실수 -3e9; `docs/error_taxonomy.md` 부록 3). sanitizer 실행(`--sanitize`)만 예외로, 배열마다 C 할당자로 정확히 n 원소만 잡고 가드를 두지 않는다(그 실행의 판정은 쓰지 않는다; `docs/error_taxonomy.md` §10).
- C의 사전 채움값(센티널)은 `tolerance.sentinel = -2048.0`(유한값). 입력이 uniform[-1,1)이고 K ≤ 1024이므로 |C| ≤ 1024 < 2048, 정당한 출력이 센티널과 같을 수 없다. 이것으로 "안 썼음(`output_not_written`)·일부만 씀(`partial_write`)·누적함(`accumulated_into_c`)·반환만 함(`returned_array_not_written`)"을 `wrong_output`으로 판정한다(NaN 센티널은 이것들을 `numeric_instability`로 오분류한다 → decisions.md).
- **매 호출 전** `A[...] = A0`, `B[...] = B0`로 복원하고 C를 센티널로 채운다(check 모드는 매 호출; bench 모드는 첫 warmup 전 1회). 호출 뒤 `inputs_mutated = not (A == A0 and B == B0)`.
- python 백엔드에는 2-D 뷰(`view.reshape(M, K)` 등)를, c 백엔드에는 `view.ctypes.data_as(POINTER(c_float))`를 넘긴다. python 백엔드는 반환값 타입명을 `return_type`으로 기록한다. numba 는 python 과 같은 호출 경로이고, rust 산출물은 c 백엔드로 부른다. halide 는 입력·출력 뷰를 복사 없이 `hl.Buffer` 로 감싸 출력 뷰에 직접 실현한다.

## 8. 정적 규약 검사 (run당 1회, `protocol.check_static`)

검사 대상 CPU = `cores ∪ single_core` = {0, 2, 4, 6}. 결과는 `StaticStatus`로, 플래그는 stderr 경고(해결 명령 포함)와 함께 행의 `protocol_flags`에 들어간다.

| 검사 | 통과 조건 | 플래그(의미) | 행 열 |
|---|---|---|---|
| 거버너 | 모든 고정 CPU의 `scaling_governor == performance` | `governor=<값>` — `require_performance_governor=true`일 때만 단다(기본 true) | `governor`, `governor_ok` |
| EPP | 모든 고정 CPU의 `energy_performance_preference == performance`(파일이 없으면 `n/a`·통과) | `epp=<값>` | `epp` |
| turbo | 검사 아님(정보) | `turbo=on` — `intel_pstate/no_turbo == 0`. **정보용**(ok 판정 제외) | `no_turbo` |
| taskset | `taskset` 실행 파일 존재 | `taskset_missing` | `taskset_used` |
| BLAS 정체 | `.venv` 파이썬 서브프로세스에서 NumPy BLAS 이름에 `openblas` 포함 | `blas=<reference\|unknown\|이름>` — `run.py`는 이 경우 **항상 중단**(--strict 무관) | `numpy_blas` |
| 환경변수 | 검사 아님(정보) | `env_overrides_dropped` — 사용자 셸의 OMP_*/OPENBLAS_* 등을 버렸음. **정보용** | — |
| 플랫폼 | `sys.platform` 이 `linux` 로 시작 | `platform=<sys.platform>`(예: `platform=win32`) — **정보용이 아니다**. 비-Linux 에서는 거버너·EPP·taskset·`/proc/stat` 이 없어 측정 규약을 확인할 수 없으므로 그 행은 항상 `protocol_ok=0` 이다. 동적 검사(§9)도 bench 마다 같은 플래그를 단다 | — |

`ok = 정보용(turbo=, env_overrides_dropped) 외 플래그 없음`. 머신 프로필은 호스트별 `config/machines/<hostname>.yaml`(없으면 이전 위치 `config/machine.yaml`)이며, 그 `hostname`이 현재와 다르면 경고(재프로브 안내). `run.py`는 플래그가 있어도 경고 후 계속하고(`protocol_ok=0`으로 기록), `--strict-protocol`(`make real STRICT=1`)이면 **이 정적 검사의** 정보용 외 플래그(`governor=`, `epp=`, `taskset_missing`, 비-Linux 의 `platform=`)가 있을 때 run 시작 시 **exit 4**로 중단한다. `--strict-protocol`의 효력은 여기까지다 — §9의 동적 플래그, §10의 `baseline_drift`·`paired_baseline_failed`, run 끝의 `model_served_mixed`는 strict 모드에서도 행에 **기록만** 하고 run을 멈추지 않는다(`run.py --help`: "정적 규약 플래그가 있으면 중단"). 측정 중 일시적 부하로 run 전체가 죽는 것보다 행 단위 `protocol_ok=0`으로 분석에서 가려내는 쪽을 택했다(decisions.md 2026-11-04).

## 9. 동적 규약 검사 (bench 직전마다, `protocol.check_dynamic`)

통과 샘플마다 bench 워커를 띄우기 직전에 1초 동안 `/proc/stat`을 2회 읽어 검사한다. bench 라운드(§10.2)마다 다시 검사하며, 플래그는 행의 `protocol_flags` 에 더해지고 라운드별 플래그는 `result.json` 의 `stages.rounds[].dynamic_flags` 에 남는다. `/proc/stat`·`os.getloadavg` 가 없는 플랫폼에서는 사용률·load 가 0 으로 남고 `platform=` 플래그가 붙는다.

| 측정 | 정의 | 임계 | 플래그 |
|---|---|---|---|
| `load_1m` | `os.getloadavg()[0]` | `max_load_1m = 2.0` | `load_1m=<x>>2.0` — **정보용**(2026-11-08: 직전 샘플의 다중 스레드 bench 가 올린 1분 평균을 다음 검사가 읽는 자기 부하라 판정에서 뺀다) |
| `pinned_busy_pct` | 고정 코어들의 평균 사용률(idle = idle + iowait, total = user…steal 8개 필드) | `max_pinned_busy_pct = 10 %` | `pinned_busy=<x>%>10%` |
| `sibling_busy_pct` | 고정 코어의 SMT 형제(1, 3, 5, 7) 평균 사용률 | 같은 10 % | `sibling_busy=<x>%>10%` |
| 거버너 재확인 | §8과 같음 | — | `governor=<값>` |

행에는 `load_1m`, `load_ok`, `pinned_busy_pct`, `sibling_busy_pct`, `governor`, `governor_ok`(동적 재확인 값으로 갱신), `protocol_flags`, `protocol_ok`, `protocol_checked_at`(bench를 실행한 행은 동적 검사 시각, 아니면 정적 검사 시각)이 기록된다. `protocol_flags`는 정적(§8) + 동적(이 절) + 기준선(§10: `baseline_drift=<±x.x>%`, `paired_baseline_failed` — 교차 기준선이 실패해 그 라운드가 속도비에서 빠짐) + bench 워커의 `call_affinity_failed`·`multiple_libgomp`(§10.1) + run 끝의 `model_served_mixed`(응답 모델이 단일값이 아님; 설계 §8-10)를 모두 담는 JSON 리스트다. **`protocol_ok` = 그 리스트에 정보용 외 플래그가 하나도 없음**(`protocol.flags_ok`; `finalize_row`에서 계산). 정보용 플래그는 `protocol.INFO_FLAG_PREFIXES` 의 `turbo=`, `env_overrides_dropped`, `load_1m=`(2026-11-08), `multiple_libgomp`(2026-11-23) 넷이다 — 즉 drift·`paired_baseline_failed`·`call_affinity_failed`·`model_served_mixed`는 `protocol_ok=0`을 만든다. 동적 플래그·drift는 `--strict-protocol`이어도 기록만 한다(§8). 집계의 `n_protocol_flagged`는 `protocol_ok=0`이거나 정보용 외 플래그가 있는 행 수다.

## 10. 기준선: run 시작·1스레드·샘플별 대응, drift

기준선은 과제마다 `tasks/<task>/signatures.yaml` 의 `baseline` 이 정한다(`backend: python|c`, `file`, `openmp`, python 기준선의 게이트 `gate_blas`·`gate_min_gflops`). GEMM 은 `tasks/gemm/baseline/baseline.py`(`np.matmul(A, B, out=C)`)이고, 어느 과제든 **LLM 커널과 똑같이** 워커(python 또는 빌드한 C)로 측정한다. bench 를 하는 run 에 C 기준선 과제가 있으면 `build.cc` 가 필요하다(없으면 행을 쓰기 전에 exit 7).

1. **run 시작 기준선**: `threads = measurement.threads = 4`로 bench → `baseline_run_time_ms`, `baseline_run_gflops`. `final_check`가 pass가 아니거나 `gflops < baseline_min_gflops = 30`이면 참조 BLAS(netlib)나 격리되지 않은 venv를 의심해 **중단**한다. 과제 일반: `gflops` 하한은 `gate_min_gflops` 인 과제(GEMM)에만, BLAS 가 OpenBLAS 인지는 `gate_blas` 인 python 기준선에만 적용하고, 게이트 실패는 exit 6 이다. 이어 1스레드로 1회 → `baseline_time_1t_ms`. 파일: `results/built/<run_id>/<task>/baseline/{bench_run.json, bench_1t.json}`. `--no-baseline`이면 기존 파일을 재사용한다. 처리할 샘플이 없는(모두 최신인) 과제는 다시 재지 않고 디스크의 기준선을 읽는다.
2. **대응 기준선**(paired): 샘플의 bench 직전에 같은 설정으로 기준선을 다시 재서 `baseline_time_ms`(대응값), `baseline_time_cv`, `baseline_samples_ms`에 적는다(파일 `baseline/paired_sample_<lang>_<i>.json`). `baseline_drift_pct = (대응 − run 시작) / run 시작 · 100`이고, 절댓값이 `baseline_drift_pct_max = 10`을 넘으면 플래그 `baseline_drift=<x>%`. **(2026-11-09 이후 대체: 대응 기준선은 별도 실행이 아니라 샘플의 bench 워커 안에서 커널과 번갈아 잰다 — §10.1.)** drift 의 기준은 run 이 끝날 때(`finish_run`) 그 run·그 과제의 대응 기준선 중앙값으로 다시 계산하고(2026-11-05 사용자 승인), `finish_run` 은 필터와 무관하게 그 run 의 모든 `result.json` 행을 기준으로 한다.
3. 속도비: `speedup_vs_baseline = baseline_time_ms / time_median_ms`(**대응 기준선 사용**), `speedup_vs_baseline_1t = baseline_time_1t_ms / time_median_ms`. 집계의 `speedup_*`는 pass 행의 이 값들이다.

### 10.1 교차 측정과 호출 친화도 (2026-11-09, 2026-11-11)

- **교차 측정**: 샘플의 bench 워커가 커널과 과제의 기준선을 함께 로드해 번갈아 호출한다(`--baseline-backend`, `--baseline-artifact`). 호출 쌍마다 순서를 바꾸고(짝수 쌍은 기준선 먼저) warmup·timed 모두 쌍으로 진행한다. `baseline_time_ms`·`speedup_vs_baseline` 은 이 교차 기준선에서 온다(`bench.json` 의 `interleaved: true`, `baseline_interleaved{…}`). 노트북에서 지속 부하 뒤 클럭이 절반으로 떨어져, 같은 시점에 잰 두 시간의 비율만 안정적이기 때문이다. `baseline_drift_pct` 는 그 run 의 과제별 교차 기준선 중앙값 대비다.
- **워커 환경**: bench 워커는 언어와 무관하게 `taskset -c <cores>` 와 `measurement.threads` 로 뜬다. 커널이나 기준선이 OpenMP 면 `omp_env_extra` 를 준다(커널은 언어의 `omp_runtime`, §3). 언어의 `env`(`NUMBA_`·`HL_`·`RAYON_` 변수)도 check 와 같은 값으로 준다. OpenBLAS 스레드 풀은 커널 로드 전에 만들고(`prewarm_blas_threads`), `OPENBLAS_THREAD_TIMEOUT=16` 으로 BLAS 스레드의 스핀 대기를 줄인다.
- **호출 친화도**(`protocol.call_cpus`, `run.bench_plan`, 워커 `--kernel-cpus`/`--baseline-cpus`): 한 프로세스에 OpenMP 공유 객체가 로드되면 libgomp 가 주 스레드를 place 0(CPU 하나)에 묶는다. 그래서 워커는 코드를 호출하기 직전에 주 스레드의 친화도를 그 코드의 규약 집합으로 두고 직후에 되돌린다(시간 측정 구간 밖).

  | 호출되는 코드 | 호출 동안 주 스레드의 CPU | 비고 |
  |---|---|---|
  | OpenMP C (`c_openmp` 커널, C+OpenMP 기준선) | 건드리지 않음 | libgomp 의 고정: 주 스레드 = place 0, 작업 스레드는 나머지 place |
  | python (`numpy` 커널, NumPy 기준선) | 고정 코어 전체 | 커널이 만드는 스레드가 전체 마스크를 물려받는다 |
  | 순차 C (`c_seq` 커널) | 고정 코어의 첫 CPU | 프롬프트의 "코어 1개"와 같다. 주 스레드가 이미 있던 CPU 라 옮겨 가지 않는다 |
  | `numba` 커널(스레딩 계층 omp) | 건드리지 않음 | OpenMP C 와 같은 libgomp 의 고정 |
  | `halide_auto`·`halide_manual`·`rust` 커널 | 고정 코어 전체 | Halide 스레드 풀·rayon 스레드 풀이 첫 호출에서 이 마스크를 물려받는다. Halide 의 형태별 빌드도 이 집합에서 한다 |

  규칙은 `call_cpus(cfg, backend=…, openmp=lang.omp_runtime, threads=…)` 하나다: `openmp` 면 None(건드리지 않음), 아니면 `threads == 1` 일 때 고정 코어의 첫 CPU, 그 밖에는 고정 코어 전체. `backend` 는 판정에 쓰지 않는다. 기존 세 언어에서는 `omp_runtime` 이 OpenMP C 에서만 참이라 결과가 2026-11-11 판과 같으므로 `CALL_AFFINITY_POLICY`(`"2026-11-11"`, 환경 지문의 입력)는 그대로다.

  결과는 `bench.json` 의 `call_affinity{main_after_load, kernel, baseline, main_after_bench, libgomp, libgomp_paths, error}` 에 남는다. 요청한 CPU 가 taskset 집합 밖이거나 적용·복원에 실패하면 `error` 에 적고 행에 플래그 `call_affinity_failed` 를 단다. `libgomp_paths` 는 bench 워커에 매핑된 libgomp 파일 전부이고, 서로 다른 경로가 둘 이상이면(numba 의 omp 계층과 C 의 libgomp 가 갈림) 행에 **정보용** 플래그 `multiple_libgomp` 를 단다 — 2026-11-12 결정은 둘이 libgomp 하나를 공유한다고 가정하며, 이 플래그는 그 가정이 깨진 bench 를 표시만 한다(decisions.md 2026-11-23).
- **주파수 샘플**: timed 호출 직후 그 코드가 쓴 CPU(호출 친화도 집합, 없으면 고정 코어 전체)의 주파수를 읽는다.
- **진행 기록의 fsync**: 호출 쌍 사이에 진행 기록(`in_progress`)을 두 번 쓴다. `measurement.bench_loop_fsync`(기본 true)가 false 면 이 기록을 fsync 없이 쓴다(마지막 기록은 언제나 fsync). false 는 규약 변경 후보이며 아직 적용하지 않았다(decisions.md 2026-11-11).
- **기록되지 않는 전제**: 이 머신의 libgomp 는 병렬 구간 뒤 스핀 대기를 하지 않는다(하이브리드 CPU 기본값). 스핀하는 런타임에서는 OpenMP 호출 직후의 호출이 느려지므로, 머신을 옮기면 `OMP_DISPLAY_ENV=verbose` 로 `OMP_WAIT_POLICY`·`GOMP_SPINCOUNT` 를 확인한다(decisions.md 2026-11-11 후보 (a)).

### 10.2 라운드 반복과 평균 (2026-11-12, 11/11 면담 반영)

- **라운드**: 구현마다 bench 를 `measurement.bench_rounds`(3)번, 매번 **새 워커 프로세스**로 돈다. 각 라운드는 워밍업 `warmup`(3) + timed `repeats`(10) 쌍의 교차 측정이다. 라운드 1 은 정확성 검사 직후에, 라운드 2 와 3 은 run 의 모든 샘플이 라운드 1 을 마친 뒤 라운드 번호 순으로 한 바퀴씩 돈다 — 같은 구현의 라운드들이 서로 다른 시각(사이에 다른 구현들의 측정이 낀)에 재진다. `bench_round_gap_s`(60)는 같은 구현의 연속 라운드 사이 최소 간격이다(작은 run 에서만 기다리게 된다). 결과 파일은 `bench.json`(라운드 1), `bench_r2.json`, `bench_r3.json`.
- **행의 값**(`harness/benchstats.py`): 모든 라운드의 timed 호출을 모아 `time_median_ms`·`time_mean_ms`·`time_cv`·`baseline_time_ms`(중앙값)·`baseline_time_mean_ms` 를 낸다. `speedup_vs_baseline` = 기준선 중앙값 / 커널 중앙값(이전 정의), **`speedup_mean` = 기준선 평균 / 커널 평균(주 지표, `measurement.primary_speedup: mean`)**, `speedup_ci_lo`·`speedup_ci_hi` = `speedup_mean` 의 95% 층화 부트스트랩 구간(라운드마다 같은 쌍 번호로 커널·기준선을 함께 복원 추출, 2000회, 씨앗은 행 키에서), `speedup_rounds` = 라운드별 평균 비, `bench_rounds` = 오류 없이 끝난 라운드 수. `time_samples_ms`·`baseline_samples_ms` 에는 모든 호출과 라운드별 개수(`rounds`)가 남는다. 속도비는 모든 라운드가 오류 없이 끝난 pass 행에만 둔다. 어느 라운드에서든 크래시·틀린 출력·입력 변조·호출 간 불일치가 나면 그 라운드에서 멈추고 bench 단계 오류로 기록한다(bench 시간 초과만 pass 유지).
- **이어 재기**: 중단된 run 을 같은 `--run-id` 로 다시 돌리면 남은 라운드만 돈다. 단, 라운드 1 에서 계획한 라운드 수(`rounds_planned`)와 환경 지문(env_hash)이 지금과 같은 행만 이어 잰다 — 이전 규약으로 잰 행은 섞지 않고 `--force` 로 다시 잰다. 수정 루프(`harness.repair`)도 턴을 다 처리한 뒤 자신이 쓴 행의 남은 라운드를 같은 함수로 마저 돈다.
- **OpenMP 대기 방식**: `omp_env_extra` 에 `OMP_WAIT_POLICY=passive` 를 고정했다(병렬 구간 뒤 작업 스레드가 스핀하지 않는다; libgomp 판과 CPU 종류에 따라 기본값이 달라 교차 측정이 장비마다 다르게 흔들리지 않게).
- **A/A 점검**(`make aa` = `python -m harness.aa_check`): 과제의 기준선을 커널 자리와 기준선 자리에 함께 넣어 교차 측정한다. 속도비는 1 이어야 하고, 호출 쌍의 첫 호출 / 둘째 호출 중앙값 비(`first_vs_second`)도 1 이어야 한다. 쌍 사이 진행 기록의 fsync 유무(`fsync`, `nofsync`) 두 변형을 라운드마다 순서를 바꿔 잰다. 새 장비에서 측정 전에 돌려 `measurement.bench_loop_fsync` 를 정한다(요약 `results/aa/<run_id>.json`).

## 11. 주파수·스로틀·온도 샘플링

- 각 timed 호출 직후 그 코드의 호출 친화도 집합(§10.1), 그것이 없으면(OpenMP 런타임 위의 코드) 고정 코어들(`os.sched_getaffinity(0)` — **커널 로드 전에** 읽어 둔 집합; libgomp는 `OMP_PROC_BIND=true`면 로드 시점에 주 스레드를 place 0에 묶어 로드 후 친화도가 CPU 1개로 좁아진다, 실측 2026-11-04)의 `scaling_cur_freq`(kHz → MHz)를 읽어 `cpu_mhz_samples`에 쌓는다 → 행 `cpu_mhz_median`, `cpu_mhz_min`. powersave/EPP=power에서는 이 값이 1.3–1.7 GHz에 머문다(실측) — 거버너를 바꾸지 않은 측정은 이 열로 드러난다.
- bench 시작·끝에 `core_throttle_count` 합(→ `throttle_events` = 차이), `x86_pkg_temp` 패키지 온도(→ `temp_c_start`, `temp_c_end`), `os.getloadavg()[0]`(`loadavg_start/end`)을 기록한다.

## 12. 행에 기록되는 재현성 정보

열 하나하나의 뜻은 `README.md`를 보라. 재현에 필요한 것만 추리면:

- 프롬프트·모델: `prompt_hash`(system + user 프롬프트의 sha256[:16]), `provider`, `model`, `model_served`, `sampling`(JSON: `temperature_sent`, `temperature_effective`, `thinking`, `effort`, `seed`), `effort`, `max_tokens`, `temperature`, `seed`, `sdk_version`.
- 입력: `test_seed`, `test_hash`(입력 바이트만으로 정의; `C_ref`는 BLAS에 따라 비트가 달라 제외).
- 빌드: `compiler`, `compiler_version`, `cflags`(base_flags + cflags + openmp_flag + ldflags 전체 문자열), `march_resolved`. 새 언어에서 `cflags` 는 그 백엔드의 성능을 정하는 빌드 설정이다: rust 는 `RUSTFLAGS=<protocol.rust_flags>; profile.release: <Cargo.toml 의 [profile.release]>`(예: Linux 에서 `RUSTFLAGS=-C target-cpu=native -C link-arg=-Wl,--no-undefined; profile.release: opt-level=3 codegen-units=1 lto="off" panic="abort" debug=false`), halide_auto 는 `autoscheduler=<이름> params=<parallelism 을 포함한 매개변수 JSON>`, halide_manual 은 `autoscheduler=none`, numpy·numba 는 NA. `compiler`·`compiler_version` 은 rust 가 `rustc` 와 `rustc --version` 첫 줄, numba·halide 가 패키지 이름과 그 버전이다.
- 실행: `threads`, `cores`, `omp_env`, `warmup`, `repeats`, `repeats_reduced`, `time_samples_ms`, `baseline_*`.
- 머신·전원: `hostname`, `cpu_model`, `python_version`, `numpy_version`, `numpy_blas`, `governor`, `epp`, `no_turbo`, `cpu_mhz_*`, `throttle_events`, `temp_c_*`, `load_1m`, `pinned_busy_pct`, `sibling_busy_pct`, `protocol_flags`.
- 지문: `env_hash`(`protocol.environment_fingerprint`: cpu_model, kernel, glibc, cc 버전, march_resolved, python, numpy, BLAS, SDK 버전, 전체 플래그, cores, single_core, threads, omp_env, call_affinity_policy, bench_loop_fsync, governor, epp, no_turbo의 sha256[:16]), `harness_version`(git describe; git이 없으면 `nogit-<소스 해시 12자>`). 새 언어의 도구 버전(rustc·cargo·numba·halide)과 Halide 오토스케줄러 설정은 지문에 넣지 않는다 — 행의 `compiler`·`compiler_version`·`cflags` 와 `result.json` 의 `toolchain` 에 남는다(decisions.md 2026-11-23). 소스 해시에는 하네스·분석 코드, 프롬프트, 과제 명세, 설정 외에 `plugins/lint/{run_lint.py,rules.yaml}` 와 `harness/build/rust_template/{Cargo.toml,Cargo.lock}` 도 들어간다.
- 원본 위치: `raw_path`, `built_path`(`result.json`이 진실의 원천; CSV는 `--rebuild-csv`로 재생성 가능 — `result.json` 으로 다시 만들 수 없는 기존 CSV 행은 남긴다).
- `result.json` 에는 행 외에 단계별 상세 `stages`(예: `stages.dynamic`, `stages.rounds`, `stages.call_affinity`, `stages.sanitize`, `stages.lint`), 그 처리가 요청한 선택 단계 `requested = {bench, safety, lint}`, raw 의 run id 가 처리 중인 run 과 다르면 `raw_run_id`, 새 언어의 `toolchain` 이 있다(`docs/harness_design_v3.md` §9.7).

## 13. 거버너/EPP 설정 (측정 전에 사람이 할 일)

```sh
sudo cpupower frequency-set -g performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
.venv/bin/python -m harness.protocol --check     # static + dynamic JSON; 플래그가 있으면 exit 1
```

- 설정은 재부팅하면 `powersave`/`power`로 돌아온다. 측정 세션마다 다시 설정하고 `--check`로 확인한다.
- 새 머신에서는 그 전에 `make probe` 로 머신 프로필을 만든다(§1). `python -m harness.protocol --fingerprint` 는 지금 도구로 만든 환경 지문을 보여 준다.
- 머신 프로필(`config/machines/<hostname>.yaml`)의 `governor`·`epp`는 프로브 시점의 값이다. 실제 측정 시점의 값은 행의 `governor`·`epp`·`protocol_flags`에 따로 기록된다.
- 하네스는 거버너를 **절대 바꾸지 않는다**(sudo 없음; 시스템 설정 변경 금지). 플래그를 달고 계속하거나(`governor_ok=0`), `--strict-protocol`이면 run 시작의 정적 검사에서 중단할 뿐이다(bench 직전의 동적 재확인은 strict여도 플래그만 단다, §8–§9).

## 14. 1024³ 는 L3 상주(warm-cache) 측정이다

성능 형태 (1024, 1024, 1024)의 작업집합은 A + B + C = 3 × 4 MiB = **12 MiB**로 L3 18 MiB 안에 들어간다. 워밍업 뒤의 timed 호출은 모두 **warm-cache** 상태이며 DRAM 대역폭 영역을 보지 않는다. 2048³(48 MiB)는 L3를 넘지만 순진한 C가 호출당 ~80 s라 bench 예산 안에 측정할 수 없어 M1은 1024³를 유지한다(decisions.md). 따라서 이 실험의 속도비는 "캐시 상주 GEMM에서 커널 자체의 연산 효율"을 재는 것이지 메모리 계층 전체의 효율이 아니다.

## 15. `tokens_out` 의 의미

- `tokens_out` = **thinking 토큰을 포함한 총 출력 토큰**(API `usage.output_tokens`, 과금 기준).
- `tokens_thinking` = `usage.output_tokens_details.thinking_tokens`(제공자가 주는 경우), `tokens_text = tokens_out − tokens_thinking`(둘 다 있을 때). `n_thinking_blocks`는 thinking 블록 수.
- `provider=dummy`인 행의 토큰은 단어 수(가짜)다. 집계의 `tokens_out_mean`·`tokens_per_pass`는 토큰 값이 있는 모든 행(절단 행 포함)을 합산한다 — 실패한 호출도 비용이기 때문이다.

## 16. 한계

- **샌드박스는 프로세스 격리뿐이다.** LLM 코드는 사용자 권한으로 `harness.worker` 서브프로세스에서 실행된다. 적용하는 것은 새 프로세스 그룹 + 타임아웃 SIGKILL, `RLIMIT_AS`, 코어 덤프 차단, 환경변수 화이트리스트, 빌드 단계의 금지 심벌 검사(`fork`/`system`/`pthread_create` 등은 `contract_violation`에 기록)뿐이고, seccomp·네임스페이스·파일시스템 격리는 없다. 신뢰할 수 없는 모델 출력을 돌릴 때는 이 점을 전제로 한다.
- **warm-cache 측정**(§14). 메모리 대역폭 병목은 이 과제·크기에서 관찰되지 않는다.
- **비-Linux 실행은 측정이 아니다.** 개발용 Windows 에서도 하네스는 임포트되고 python 계열 백엔드와 Rust(MSVC cargo)의 check·bench 경로까지 돌지만(C 언어는 gcc 가 없어 run 시작에서 exit 7), 코어 고정·거버너·`/proc/stat`·프로세스 그룹 정리·`RLIMIT_AS` 가 없다. 그 행은 `platform=` 플래그로 `protocol_ok=0` 이고 시간 값은 비교에 쓰지 않는다.
- **노트북 열 특성**: i7-1260P는 전력·온도 제한으로 주파수가 요동한다. `performance` 거버너에서도 긴 bench 중 스로틀이 생길 수 있어 `cpu_mhz_*`, `throttle_events`, `temp_c_*`를 행마다 기록하고, 대응 기준선과 drift 플래그로 시점 차이를 드러낸다. 전원 어댑터 연결·방열 상태는 코드가 확인하지 못한다.
- **단일 머신·단일 세션**: `env_hash`가 다른 행끼리의 비교는 보장하지 않는다.
- `perf stat`(하드웨어 카운터)은 쓸 수 없어 루프라인(`roofline_pct`)은 M1에서 NA다.
