# 측정 규약 (docs/protocol.md)

이 문서는 **코드가 실제로 강제하는** 측정 규약을 적는다. 구현은 `harness/protocol.py`(정적·동적 검사, 코어 고정, 환경 화이트리스트), `harness/worker.py`(버퍼·센티널·반복·주파수 샘플링), `harness/run.py`(기준선·게이트·행 기록), `harness/procutil.py`(프로세스 격리)에 있다. 설정값은 `config/experiment.yaml`의 `measurement`·`build` 절이고, 설계 계약은 `docs/harness_design.md` §0, §7–§9와 `docs/harness_design_v3.md` §10.4·§10.6이다. 사람이 손으로 지켜야 하는 것은 **거버너/EPP를 `performance`로 두는 것**뿐이며(sudo 필요), 나머지는 전부 코드가 확인하고 행에 기록한다.

**2026-11-22 갱신(M2b–M6 구현).** 측정 규약 자체(코어 고정, 스레드 수, 워밍업 3 + 반복 10 의 중앙값, 타임아웃, 정적·동적 검사, 대응 기준선)는 **바뀌지 않았다**. 바뀐 것은 그 주위다: 비-Linux 의 `platform=` 플래그(§8), 새 언어의 스레드 환경변수(§3), 지문에 더한 도구 버전(§12), 대응 기준선 파일의 위치(§10), 기본으로 꺼져 있는 부하 대기 `measurement.max_settle_s`(§9), run 시작 전의 실행 접두·런타임 시험(§2). 새 언어(numba, halide_auto, halide_manual, rust)와 새 과제 8개의 측정은 이 Windows 개발 머신에서는 경로만 확인했고(시간 값은 의미 없음), 실제 측정은 Linux 데스크톱에서 처음 한다. GEMM 에 대한 수치(§1, §14)는 M1 의 Linux 노트북 실측이다.

## 1. 측정 머신과 고정 자원

아래 표는 M1 측정 머신(Linux 노트북)이다. 저장소의 `config/machine.yaml` 도 이 머신의 프로브 결과이고, 다른 머신(예: 이후의 Linux 데스크톱)에서는 첫 run 전에 `.venv/bin/python -m harness.protocol --probe --write` 로 다시 쓰고, `measurement.cores`·`single_core` 를 그 CPU 의 토폴로지에 맞춰 정해야 한다(호스트 이름이 다르면 하네스가 경고한다; 맞지 않는 코어 집합은 §2 의 실행 접두 시험이 run 시작 전에 잡는다). 이 Windows 개발 머신에서 만든 행도 machine.yaml 의 호스트 이름·CPU 를 싣지만 `platform=win32` 플래그로 `protocol_ok=0` 이다.

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
- `threads: 1`인 언어(`c_seq`)는 **단일 P-core** `measurement.single_core = "2"`로 더 좁게 고정한다(`taskset -c 2`; cpu0은 IRQ가 몰려 피한다). 코어를 하나만 주므로 "스레드 사용 금지"가 코드로 강제된다.
- `protocol.launch_prefix(cfg, threads)`: `threads == 1`이면 `["taskset", "-c", single_core]`, 아니면 `["taskset", "-c", cores]`. `taskset`이 없으면 `[]`로 실행하고 플래그 `taskset_missing`을 단다(행 `taskset_used=0`).
- 컴파일(`gcc`)도 같은 접두로 P-core에서 실행해 `-march=native` 해석을 고정한다(`march_resolved` 열). Rust 의 `cargo build` 도 그 언어의 스레드 수에 맞는 접두로 실행한다(`-C target-cpu=native`).
- 기준선(과제마다 `tasks/<task>/baseline/`)도 같은 코어 집합·스레드 수·환경변수로 측정한다.
- **실행 접두 시험(2026-11-22 추가)**: `harness.run` 의 준비 단계(`setup_run`)가 run 이 쓸 접두마다 `[*접두, PYTHON_EXE, -c, pass]` 를 한 번 돌린다(명령당 60 s; 기준선을 재는 run 은 `measurement.threads` 와 1스레드 접두도 포함). `cores`·`single_core` 가 이 머신에 없는 코어를 가리키면 taskset 이 모든 명령을 rc 1 로 끝내 빌드는 `compile_error`, 워커는 `runtime_crash` 행이 되므로, 행을 쓰기 전에 exit 7 로 멈춘다. 접두가 비어 있으면(taskset 없음, 비-Linux) 건너뛴다. 같은 단계에서 halide_auto 의 오토스케줄러 매개변수와 numba 의 스레딩 계층도 그 언어의 접두·환경으로 띄운 서브프로세스에서 한 번 써 본다(`python -m harness.protocol --probe-runtime halide|numba`; 실패는 exit 7).

## 3. 스레드 수와 환경변수 화이트리스트

워커(`harness.worker`)의 환경은 `protocol.launch_env(cfg, threads, openmp=…)`가 **처음부터 새로** 만든다.

- 상속하는 변수: `PATH`, `HOME`, `LANG`, `TMPDIR`, `VIRTUAL_ENV`. 고정값: `LC_ALL=C`, `PYTHONHASHSEED=0`, `PYTHONPATH=<저장소 루트>`.
- 스레드 수: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` = 언어의 `threads`(`numpy`·`c_openmp`는 `measurement.threads=4`, `c_seq`는 1). NumPy는 `OPENBLAS_NUM_THREADS`(우선)/`OMP_NUM_THREADS`(폴백)로 제어됨을 실측으로 확인했다.
- OpenMP 백엔드(`c_openmp`) 워커에만 `measurement.omp_env_extra`를 추가한다: `OMP_PROC_BIND=true`, `OMP_PLACES=cores`, `OMP_DYNAMIC=false`.
- 사용자 셸의 `OMP_*`, `GOMP_*`, `OPENBLAS_*`, `MKL_*`, `KMP_*`는 **상속하지 않는다**. 그런 변수가 있었으면 한 번 경고하고 정보용 플래그 `env_overrides_dropped`를 단다.
- 행에는 `threads`, `cores`, `omp_env`(실제 넘긴 스레드 관련 변수 JSON; 접두 `OMP_`, `OPENBLAS_`, `MKL_`, `GOMP_`, `KMP_`, `NUMBA_`, `RAYON_`, `HL_`)가 기록된다.

**언어별 환경변수(2026-11-22 추가).** M3 언어의 런타임은 `OMP_NUM_THREADS` 를 읽지 않으므로, 언어 설정의 `env` 가 워커 환경에 마지막으로 더해진다(`protocol.launch_env(..., extra_env=lang.env)`; 값의 `${threads}` 는 그 언어의 스레드 수로 치환된다).

| 언어 | `threads` | 더하는 변수 |
|---|---|---|
| `numba` | `measurement` (4) | `NUMBA_NUM_THREADS=4`, `NUMBA_THREADING_LAYER=omp` |
| `halide_auto`, `halide_manual` | `measurement` (4) | `HL_NUM_THREADS=4` (halide_auto 는 오토스케줄러 매개변수 `parallelism` 도 4) |
| `rust` | `measurement` (4) | `RAYON_NUM_THREADS=4` |

- 이 변수(`NUMBA_NUM_THREADS`, `HL_NUM_THREADS`, `RAYON_NUM_THREADS`)가 빠졌거나 그 언어의 스레드 수와 다르면 `load_config` 가 설정을 거부한다 — 빠뜨리면 런타임이 논리 코어 수만큼 스레드를 띄우고 프롬프트의 스레드 수 서술과 행의 `threads` 가 사실이 아니게 된다. 자리표시는 `${threads}` 만 허용하고, `halide.autoscheduler_params` 의 `parallelism` 은 하네스가 넣는 예약어다.
- `measurement.omp_env_extra`(`OMP_PROC_BIND`·`OMP_PLACES`·`OMP_DYNAMIC`)는 지금도 C OpenMP 백엔드(`c_openmp`)에만 붙는다. `NUMBA_THREADING_LAYER=omp` 인 numba 는 스레드 묶기 없이 돈다 — 묶기를 줄지는 측정 규약 문제라 사용자 확인이 남아 있다(`docs/decisions.md`).
- 워커는 스레드 수를 따로 검사하지 않는다. numba 는 실제 스레드 수를 `import.json` 의 `inspect.num_threads` 로 남기고, halide 는 남기지 않는다.
- cargo·rustc 는 워커가 아니라 빌드 환경(`protocol.toolchain_env()`)으로 띄운다: POSIX 는 위 화이트리스트(스레드 1; `protocol.launch_env_minimal`) + 사용자의 `CARGO_HOME`·`RUSTUP_HOME`·`RUSTUP_TOOLCHAIN`, 개발용 Windows 는 현재 환경 전체다. `build_rust` 가 그 위에 `RUSTFLAGS`(설정의 `build.rust.rustflags`)·`CARGO_TARGET_DIR` 를 더한다.
- 개발용 Windows 에서만 워커 환경에 `PYTHONIOENCODING=utf-8` 을 더한다(Linux 의 워커 환경은 그대로다).

## 4. 워밍업·반복·중앙값

- **정확성(check)**: `correctness_tests = [tiny(7,5,3), rect(200,130,312), small(256³)]`. 테스트마다 `check_repeats = 2`회 호출하고, 호출마다 A·B를 원본으로 복원하고 C를 센티널로 채운다(§7). 두 호출의 출력이 허용오차 밖으로 다르면 `nondeterministic`(→ `race_condition`).
- **성능(bench)**: `performance_test = large(1024³)` 하나. `warmup = 3`회 뒤 `repeats = 10`회를 잰다. 시간은 `time.perf_counter_ns`로 **함수 호출 구간만** 잰다(입력은 이미 메모리에 있음; 파일 I/O 없음).
- 통계는 timed 호출만으로 계산한다: `time_median_ms`(**보고값**), `time_mean_ms`, `time_min_ms`, `time_max_ms`, `time_cv`(표준편차/평균, ddof=1), `gflops = 2·M·N·K / median_s / 1e9`. `time_samples_ms`에 warmup과 timed 전체 분포를 JSON으로 기록한다 — 최선값만 보고하지 않는다.
- bench 중에는 첫 timed 결과 `C_first`를 보관하고 이후 timed 호출마다(계측 밖에서) 비교한다: 비트 불일치 수 `nondeterministic_bitwise_runs`, 허용오차 초과 불일치 수 `nondeterministic_runs`. 마지막 호출 뒤 `final_check`(정확성 재판정), `inputs_mutated`, `oob_write`를 다시 확인한다.
- 과제 일반(M2 이후): 위 형태는 GEMM 의 것이고, 모든 과제가 같은 이름의 네 형태 `tiny, rect, small, large` 를 가진다(`correctness_tests = [tiny, rect, small]`, `performance_test = large`; 크기는 각 `tasks/<task>/gen_tests.py`). `gflops` 는 GEMM 식이 있는 과제에서만 의미가 있다. 정수 출력 과제(`compare: exact`)의 호출 간 비교는 허용오차 대신 정확 비교다.
- Halide(v3): 형태마다 파이프라인을 새로 만들고(빌드 함수 호출 → 오토스케줄 → JIT) 그 시간은 측정 구간 밖이다(`build_ms` 로 기록). 이 빌드 시간은 워커의 적응 예산(§5)에는 들어가지 않지만 run.py 의 bench 시간 상한(300 s)에는 들어간다 — JIT 가 상한의 약 20 % 를 넘으면 측정할 수 있는 커널이 bench timeout 이 될 수 있다. 처리 방법은 사용자 확인이 남아 있다(`docs/decisions.md`).
- Numba 는 명시 서명으로 임포트 때 즉시 컴파일되므로(서명이 없으면 빌드 단계에서 거절, §6 계약) 첫 측정 호출에 JIT 가 섞이지 않는다.

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
| import(워커 1회, 로드만) | `check_timeout_s = 180 s` | **`error_stage=build`**, `error_type=timeout`, `pass=0` (로드 중 시그널·비정상 종료도 `error_stage=build`/`runtime_crash`; 열 enum에 `import`가 없다) |
| check(워커 1회, 세 테스트 전부) | `check_timeout_s = 180 s` | `error_stage=check`, `error_type=timeout`, `pass=0` |
| bench(워커 1회) | `bench_timeout_s = 300 s` | `error_stage=bench`, `error_type=timeout`, `pass=1`, 시간 NA (§5) |
| Rust 빌드(cargo, v3) | `build.rust.build_timeout_s = 600 s` | 커널 크레이트 컴파일 중이면 `compile_error`/`compile_timeout`, 그 전(의존성 단계)이면 도구 문제로 행 없이 exit 7 |
| 린트 플러그인(v3) | `lint.timeout_s = 60 s` | 린트 열 NA, 실험은 계속 |
| sanitizer check(v3, C 만) | `safety.timeout_s = 300 s` | `asan_findings`·`ubsan_findings` 는 보고가 없으면 NA |

- 모든 서브프로세스는 `procutil.run_isolated`로 **새 세션(프로세스 그룹)** 에서 시작하고, 타임아웃이면 그룹 전체에 `SIGKILL`을 보낸다(`subprocess.run(timeout=)`은 `cc1`·OpenMP 자식을 남긴다 — 실측). 단위 테스트가 타임아웃 뒤 `pgrep -g <pgid>`가 비어 있음을 확인한다.
- 워커는 시작 직후 `prctl(PR_SET_DUMPABLE=0)`(apport 코어 덤프 차단)과 `RLIMIT_AS = max_worker_mem_gb = 8 GiB`를 적용하고 적용값을 결과의 `limits`에 적는다. `RLIMIT_NPROC`은 사용자 전체 기준이라 걸지 않는다. sanitizer 실행만 `--max-mem-gb 0` 으로 `RLIMIT_AS` 를 걸지 않는다(ASan 의 그림자 메모리; `limits.disabled`). 비-Linux 에서는 `resource`·`prctl` 이 없어 제한이 걸리지 않는다(`limits.unsupported`).
- 비-POSIX(개발용 Windows)의 `run_isolated` 는 새 세션 없이 띄우고 타임아웃 때 `proc.kill()` 만 한다(프로세스 그룹 정리 없음 — 측정용이 아니다).
- 워커의 결과 파일은 **매 커널 호출 직전과 직후** 원자적으로(tmp → `os.replace`) 갱신된다(`complete`, `in_progress{test, call_idx}`). 시그널로 죽어도 어느 테스트의 몇 번째 호출에서 죽었는지 행의 `error_detail`(`signal SIGSEGV during small call 1`)에 남는다.
- 워커는 stdout에 아무것도 쓰지 않는다. 커널이 stdout에 쓰면 `run.py`가 파일로 받아 `contract_violation`에 `stdout_output`을 적는다(분류에는 영향 없음).

## 7. 버퍼: 64바이트 정렬·가드 영역·센티널

- 모든 배열은 `alloc_aligned(n, guard)`로 만든다: `base = np.empty(n + 16 + guard, float32)`, 시작 오프셋을 64바이트 경계에 맞춘 `view`와 그 뒤 `guard_view`. A는 가드 1024, B는 1024, C는 `max(M,N)² − M·N + 1024`(전치·차원 교환 쓰기도 가드 안에 들어오게). 과제 일반: 입력 배열마다 가드 1024, 출력은 `signatures.yaml` 의 `out_guard_extra` 식의 값(없으면 0) + 1024. int32 배열(정수 입력·출력)도 같은 규칙이고 센티널은 dtype 에 맞춘다(정수 과제 `-2048`). sanitizer 실행(`--sanitize`)만 예외로, 배열마다 C 할당자로 정확히 n 원소만 잡고 가드를 두지 않는다(그 실행의 판정은 쓰지 않는다; `docs/error_taxonomy.md` §10). 가드 영역은 센티널로 채우고 호출 뒤 센티널이 아닌 원소가 있으면 `oob_write=1`. 프롬프트의 "A, B, C는 64바이트 정렬"은 이 사실을 서술한 것이다.
- C의 사전 채움값(센티널)은 `tolerance.sentinel = -2048.0`(유한값). 입력이 uniform[-1,1)이고 K ≤ 1024이므로 |C| ≤ 1024 < 2048, 정당한 출력이 센티널과 같을 수 없다. 이것으로 "안 썼음(`output_not_written`)·일부만 씀(`partial_write`)·누적함(`accumulated_into_c`)·반환만 함(`returned_array_not_written`)"을 `wrong_output`으로 판정한다(NaN 센티널은 이것들을 `numeric_instability`로 오분류한다 → decisions.md).
- **매 호출 전** `A[...] = A0`, `B[...] = B0`로 복원하고 C를 센티널로 채운다(check 모드는 매 호출; bench 모드는 첫 warmup 전 1회). 호출 뒤 `inputs_mutated = not (A == A0 and B == B0)`.
- python 백엔드에는 2-D 뷰(`view.reshape(M, K)` 등)를, c 백엔드에는 `view.ctypes.data_as(POINTER(c_float))`를 넘긴다. python 백엔드는 반환값 타입명을 `return_type`으로 기록한다.

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
| 플랫폼(v3) | `sys.platform` 이 `linux` 로 시작 | `platform=<sys.platform>`(예: `platform=win32`) — **정보용이 아니다**. 비-Linux 에서는 거버너·EPP·taskset·`/proc/stat` 이 없어 측정 규약을 확인할 수 없으므로 그 행은 항상 `protocol_ok=0` 이다. 동적 검사(§9)도 bench 마다 같은 플래그를 단다 | — |

`ok = 정보용(turbo=, env_overrides_dropped) 외 플래그 없음`. `machine.yaml`의 `hostname`이 현재와 다르면 경고(재프로브 안내). `run.py`는 플래그가 있어도 경고 후 계속하고(`protocol_ok=0`으로 기록), `--strict-protocol`(`make real STRICT=1`)이면 **이 정적 검사의** 정보용 외 플래그(`governor=`, `epp=`, `taskset_missing`, 비-Linux 의 `platform=`)가 있을 때 run 시작 시 **exit 4**로 중단한다. `--strict-protocol`의 효력은 여기까지다 — §9의 동적 플래그, §10의 `baseline_drift`·`paired_baseline_failed`, run 끝의 `model_served_mixed`는 strict 모드에서도 행에 **기록만** 하고 run을 멈추지 않는다(`run.py --help`: "정적 규약 플래그가 있으면 중단"). 측정 중 일시적 부하로 run 전체가 죽는 것보다 행 단위 `protocol_ok=0`으로 분석에서 가려내는 쪽을 택했다(decisions.md 2026-11-04).

## 9. 동적 규약 검사 (bench 직전마다, `protocol.check_dynamic`)

통과 샘플마다 bench를 시작하기 직전(대응 기준선 측정 전)에 1초 동안 `/proc/stat`을 2회 읽어 검사한다. `/proc/stat`·`os.getloadavg` 가 없는 플랫폼에서는 사용률·load 가 0 으로 남고 `platform=` 플래그가 붙는다.

**선택: 부하 대기 `measurement.max_settle_s`(2026-11-22 추가, 기본 0 = 꺼짐).** 1분 load average 는 시상수 60초의 지수 평균이라 하네스 자신의 앞선 다중 스레드 bench 가 다음 샘플의 검사 시점까지 남는다(예: 배경 1.3 + 4스레드 15초 → 5초 뒤 약 2.1). 값을 양수로 두면 검사 직전에 `load_1m` 이 `max_load_1m` 이하가 될 때까지 5초 간격으로 기다린다(최대 `max_settle_s` 초; `protocol.settle_load`). 기다린 시간은 `result.json` 의 `stages.dynamic.settle_wait_s` 에 남고, 기다린 뒤에도 넘으면 평소처럼 `load_1m=` 플래그다(플래그 규칙은 그대로). 기본값 0 에서는 이전 절차와 똑같다. 켤지(제안 60–120 s)는 측정 절차의 변경이라 Linux 데스크톱에서 부하 시험을 본 뒤 사용자가 정한다(`docs/decisions.md`).

| 측정 | 정의 | 임계 | 플래그 |
|---|---|---|---|
| `load_1m` | `os.getloadavg()[0]` | `max_load_1m = 2.0` | `load_1m=<x>>2.0` |
| `pinned_busy_pct` | 고정 코어들의 평균 사용률(idle = idle + iowait, total = user…steal 8개 필드) | `max_pinned_busy_pct = 10 %` | `pinned_busy=<x>%>10%` |
| `sibling_busy_pct` | 고정 코어의 SMT 형제(1, 3, 5, 7) 평균 사용률 | 같은 10 % | `sibling_busy=<x>%>10%` |
| 거버너 재확인 | §8과 같음 | — | `governor=<값>` |

행에는 `load_1m`, `load_ok`, `pinned_busy_pct`, `sibling_busy_pct`, `governor`, `governor_ok`(동적 재확인 값으로 갱신), `protocol_flags`, `protocol_ok`, `protocol_checked_at`(bench를 실행한 행은 동적 검사 시각, 아니면 정적 검사 시각)이 기록된다. `protocol_flags`는 정적(§8) + 동적(이 절) + 기준선(§10: `baseline_drift=<±x.x>%`, `paired_baseline_failed` — 대응 기준선 bench가 실패해 `baseline_time_ms`·`speedup_vs_baseline`이 NA) + run 끝의 `model_served_mixed`(응답 모델이 단일값이 아님; 설계 §8-10)를 모두 담는 JSON 리스트다. **`protocol_ok` = 그 리스트에 정보용(`turbo=`, `env_overrides_dropped`) 외 플래그가 하나도 없음**(`protocol.flags_ok`; `finalize_row`에서 계산) — 즉 drift·`paired_baseline_failed`·`model_served_mixed`도 `protocol_ok=0`을 만든다. 동적 플래그·drift는 `--strict-protocol`이어도 기록만 한다(§8). 집계의 `n_protocol_flagged`는 `protocol_ok=0`이거나 정보용 외 플래그가 있는 행 수다.

## 10. 기준선: run 시작·1스레드·샘플별 대응, drift

기준선은 과제마다 `tasks/<task>/signatures.yaml` 의 `baseline` 이 정한다(`backend: python|c`, `file`, `openmp`, python 기준선의 게이트 `gate_blas`·`gate_min_gflops`). GEMM 은 `tasks/gemm/baseline/baseline.py`(`np.matmul(A, B, out=C)`)이고, 어느 과제든 **LLM 커널과 똑같이** 워커(python 또는 빌드한 C)로 측정한다. C 기준선을 빌드해야 하는 run 은 `build.cc` 가 필요하다(없으면 행을 쓰기 전에 exit 7).

1. **run 시작 기준선**: `threads = measurement.threads = 4`로 bench → `baseline_run_time_ms`, `baseline_run_gflops`. `final_check`가 pass가 아니면 중단한다. `gate_min_gflops` 인 과제(GEMM)는 `gflops < baseline_min_gflops = 30`이면, `gate_blas` 인 python 기준선은 BLAS 가 OpenBLAS 가 아니면 참조 BLAS(netlib)나 격리되지 않은 venv를 의심해 **중단**한다(exit 6). 이어 1스레드로 1회 → `baseline_time_1t_ms`. 파일: `results/built/<run_id>/<task>/baseline/{bench_run.json, bench_1t.json}`. `--no-baseline`이면 기존 파일을 재사용한다. 처리할 샘플이 없는(모두 최신인) 과제는 다시 재지 않고 디스크의 기준선을 읽는다.
2. **대응 기준선**(paired): 샘플의 bench 직전에 같은 설정으로 기준선을 다시 재서 `baseline_time_ms`(대응값), `baseline_time_cv`, `baseline_samples_ms`에 적는다. 파일은 **그 샘플(턴) 디렉터리의 `paired_baseline.json`**(로그 `paired_baseline.stdout.log`·`paired_baseline.stderr.log`)이다 — 2026-11-22 이전 문서의 `baseline/paired_sample_<lang>_<i>.json` 은 예전 위치다(조건·수정 턴이 생기면서 샘플 디렉터리 안으로 옮겼다). 대응 기준선 bench 가 실패하면 플래그 `paired_baseline_failed` 이고 `baseline_time_ms`·`speedup_vs_baseline` 이 NA 다.
   drift: `baseline_drift_pct = (대응 − 기준) / 기준 · 100` 이고, 절댓값이 `baseline_drift_pct_max = 10`을 넘으면 플래그 `baseline_drift=<x>%`. 기준은 run 이 끝날 때(`finish_run`) **그 run·그 과제의 대응 기준선들의 중앙값**으로 다시 계산한다(2026-11-05 사용자 승인으로 run 시작 1회 측정 기준에서 바꿨다 — 그 한 번이 느린 상태에서 재지면 거의 모든 행에 플래그가 붙었다). `finish_run` 은 필터와 무관하게 그 run 의 모든 `result.json` 행을 기준으로 한다.
3. 속도비: `speedup_vs_baseline = baseline_time_ms / time_median_ms`(**대응 기준선 사용**), `speedup_vs_baseline_1t = baseline_time_1t_ms / time_median_ms`. 집계의 `speedup_*`는 pass 행의 이 값들이다.

## 11. 주파수·스로틀·온도 샘플링

- 각 timed 호출 직후 고정 코어들(`os.sched_getaffinity(0)` — **커널 로드 전에** 읽어 둔 집합; libgomp는 `OMP_PROC_BIND=true`면 로드 시점에 주 스레드를 place 0에 묶어 로드 후 친화도가 CPU 1개로 좁아진다, 실측 2026-11-04)의 `scaling_cur_freq`(kHz → MHz)를 읽어 `cpu_mhz_samples`에 쌓는다 → 행 `cpu_mhz_median`, `cpu_mhz_min`. powersave/EPP=power에서는 이 값이 1.3–1.7 GHz에 머문다(실측) — 거버너를 바꾸지 않은 측정은 이 열로 드러난다.
- bench 시작·끝에 `core_throttle_count` 합(→ `throttle_events` = 차이), `x86_pkg_temp` 패키지 온도(→ `temp_c_start`, `temp_c_end`), `os.getloadavg()[0]`(`loadavg_start/end`)을 기록한다.

## 12. 행에 기록되는 재현성 정보

열 하나하나의 뜻은 `README.md`를 보라. 재현에 필요한 것만 추리면:

- 프롬프트·모델: `prompt_hash`(system + user 프롬프트의 sha256[:16]), `provider`, `model`, `model_served`, `sampling`(JSON: `temperature_sent`, `temperature_effective`, `thinking`, `effort`, `seed`), `effort`, `max_tokens`, `temperature`, `seed`, `sdk_version`.
- 입력: `test_seed`, `test_hash`(입력 바이트만으로 정의; `C_ref`는 BLAS에 따라 비트가 달라 제외).
- 빌드: `compiler`, `compiler_version`, `cflags`, `march_resolved`. `cflags` 는 백엔드마다 성능을 정하는 빌드 설정이다(v3): C 는 base_flags + cflags (+ openmp_flag) + ldflags 전체 문자열, Rust 는 `RUSTFLAGS` 와 그 빌드의 `Cargo.toml` `[profile.release]`(`RUSTFLAGS=-C target-cpu=native; profile.release: opt-level=3 codegen-units=1 lto="fat" panic="abort" debug=false`), halide_auto 는 `autoscheduler=<이름> params=<parallelism 을 포함한 매개변수 JSON>`, halide_manual 은 `autoscheduler=none`, numpy·numba 는 NA. `compiler` 는 C 는 `build.cc`, Rust 는 `rustc`, Python 계열은 `python`/`numba`/`halide`(버전은 각 패키지 버전).
- 실행: `threads`, `cores`, `omp_env`, `warmup`, `repeats`, `repeats_reduced`, `time_samples_ms`, `baseline_*`.
- 머신·전원: `hostname`, `cpu_model`, `python_version`, `numpy_version`, `numpy_blas`, `governor`, `epp`, `no_turbo`, `cpu_mhz_*`, `throttle_events`, `temp_c_*`, `load_1m`, `pinned_busy_pct`, `sibling_busy_pct`, `protocol_flags`.
- 지문: `env_hash`(`protocol.environment_fingerprint`: cpu_model, kernel, glibc, cc 버전, march_resolved, python, numpy, BLAS, SDK 버전, **rustc, numba_version, halide_version, lint_tools(ruff·pyflakes·cppcheck·clang-tidy 버전; 없으면 null)**, 전체 C 플래그, cores, single_core, threads, omp_env, governor, epp, no_turbo의 sha256[:16]). 굵게 쓴 도구 버전은 2026-11-22 에 더했다(그래서 그 전의 행과 env_hash 가 다르다 — 측정 머신을 옮기므로 어차피 달라진다). 도구 버전은 machine.yaml 이 아니라 run 시작 때의 도구에서 읽는다(`protocol.toolchain_identity`; rustc 는 빌드와 같은 `toolchain_env()` 환경). cargo 버전과 Halide 오토스케줄러 설정은 지문에 없다(후자는 halide 행의 `cflags` 에 있다). `harness_version`(git describe; git이 없으면 `nogit-<소스 해시 12자>`)의 소스 해시에는 하네스·분석 코드, 프롬프트, 과제 명세, 설정 외에 `plugins/lint/{run_lint.py,rules.yaml}` 와 `harness/build/rust_template/{Cargo.toml,Cargo.lock}` 도 들어간다.
- 원본 위치: `raw_path`, `built_path`(`result.json`이 진실의 원천; CSV는 `--rebuild-csv`로 재생성 가능 — `result.json` 이 이 머신에 없는 기존 CSV 행은 남긴다).
- 단계별 상세는 `result.json` 의 `stages`(예: `stages.dynamic.settle_wait_s`, `stages.sanitize`, `stages.lint`)와 그 run 이 요청한 단계 `requested = {bench, safety, lint}` 에 있다(`docs/harness_design_v3.md` §10.2a).

## 13. 거버너/EPP 설정 (측정 전에 사람이 할 일)

```sh
sudo cpupower frequency-set -g performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
.venv/bin/python -m harness.protocol --check     # static + dynamic JSON; 플래그가 있으면 exit 1
```

- 설정은 재부팅하면 `powersave`/`power`로 돌아온다. 측정 세션마다 다시 설정하고 `--check`로 확인한다.
- 새 머신에서는 그 전에 `.venv/bin/python -m harness.protocol --probe --write` 로 `config/machine.yaml` 을 다시 쓴다(§1). `--fingerprint` 는 지금 도구로 만든 환경 지문을 보여 준다.
- `config/machine.yaml`의 `governor`·`epp`는 프로브 시점의 값이다. 실제 측정 시점의 값은 행의 `governor`·`epp`·`protocol_flags`에 따로 기록된다.
- 하네스는 거버너를 **절대 바꾸지 않는다**(sudo 없음; 시스템 설정 변경 금지). 플래그를 달고 계속하거나(`governor_ok=0`), `--strict-protocol`이면 run 시작의 정적 검사에서 중단할 뿐이다(bench 직전의 동적 재확인은 strict여도 플래그만 단다, §8–§9).

## 14. 1024³ 는 L3 상주(warm-cache) 측정이다

성능 형태 (1024, 1024, 1024)의 작업집합은 A + B + C = 3 × 4 MiB = **12 MiB**로 L3 18 MiB 안에 들어간다. 워밍업 뒤의 timed 호출은 모두 **warm-cache** 상태이며 DRAM 대역폭 영역을 보지 않는다. 2048³(48 MiB)는 L3를 넘지만 순진한 C가 호출당 ~80 s라 bench 예산 안에 측정할 수 없어 M1은 1024³를 유지한다(decisions.md). 따라서 이 실험의 속도비는 "캐시 상주 GEMM에서 커널 자체의 연산 효율"을 재는 것이지 메모리 계층 전체의 효율이 아니다.

## 15. `tokens_out` 의 의미

- `tokens_out` = **thinking 토큰을 포함한 총 출력 토큰**(API `usage.output_tokens`, 과금 기준).
- `tokens_thinking` = `usage.output_tokens_details.thinking_tokens`(제공자가 주는 경우), `tokens_text = tokens_out − tokens_thinking`(둘 다 있을 때). `n_thinking_blocks`는 thinking 블록 수.
- `provider=dummy`인 행의 토큰은 단어 수(가짜)다. 집계의 `tokens_out_mean`·`tokens_per_pass`는 토큰 값이 있는 모든 행(절단 행 포함)을 합산한다 — 실패한 호출도 비용이기 때문이다.

## 16. 한계

- **샌드박스는 프로세스 격리뿐이다.** LLM 코드는 사용자 권한으로 `harness.worker` 서브프로세스에서 실행된다. 적용하는 것은 새 프로세스 그룹 + 타임아웃 SIGKILL, `RLIMIT_AS`, 코어 덤프 차단, 환경변수 화이트리스트, 빌드 단계의 금지 심벌 검사(`fork`/`system`/`pthread_create` 등은 `contract_violation`에 기록)뿐이고, seccomp·네임스페이스·파일시스템 격리는 없다. 신뢰할 수 없는 모델 출력을 돌릴 때는 이 점을 전제로 한다.
- **warm-cache 측정**(§14). 메모리 대역폭 병목은 이 과제·크기에서 관찰되지 않는다. (§14 는 GEMM 1024³ 의 분석이다. 새 과제의 large 형태는 잠정값이고 데스크톱 Linux 에서 다시 정한다 — 작업집합이 L3 를 넘는 과제가 생기면 이 문장은 과제별로 달라진다.)
- **비-Linux 실행은 측정이 아니다.** 개발용 Windows 에서도 하네스는 임포트되고 python 계열 백엔드와 Rust(MSVC cargo)의 check·bench 경로까지 돌지만(C 는 gcc 가 없어 exit 7), 코어 고정·거버너·`/proc/stat`·프로세스 그룹 정리·`RLIMIT_AS` 가 없다. 그 행은 `platform=` 플래그로 `protocol_ok=0` 이고 시간 값은 비교에 쓰지 않는다.
- **노트북 열 특성**: i7-1260P는 전력·온도 제한으로 주파수가 요동한다. `performance` 거버너에서도 긴 bench 중 스로틀이 생길 수 있어 `cpu_mhz_*`, `throttle_events`, `temp_c_*`를 행마다 기록하고, 대응 기준선과 drift 플래그로 시점 차이를 드러낸다. 전원 어댑터 연결·방열 상태는 코드가 확인하지 못한다.
- **단일 머신·단일 세션**: `env_hash`가 다른 행끼리의 비교는 보장하지 않는다.
- `perf stat`(하드웨어 카운터)은 쓸 수 없어 루프라인(`roofline_pct`)은 M1에서 NA다.
