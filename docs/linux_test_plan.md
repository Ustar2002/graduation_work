# Linux 일괄 검증 계획 (docs/linux_test_plan.md)

통합 트리(`docs/decisions.md` 2026-11-23, "M3–M6 통합")를 Linux 측정 장비에서 한 세션에 확인하는 절차다. 장비 설치는 `docs/desktop_setup.md` 가 맡고, 이 문서는 그 위에 통합 하네스가 더 필요로 하는 것과 확인 단계만 적는다. 단계마다 명령, 기대 결과, 다를 때 볼 곳을 둔다. Windows 에서 이미 확인한 것은 2026-11-23 항목의 "검증" 줄, 확인하지 못한 것은 같은 항목의 "검증하지 못한 것" 줄과 이 문서의 부록 A 다.

원칙:

- §2–§12 의 생성은 모두 `--provider dummy`(수정 루프는 `LLM_PROVIDER=dummy`)다. 실제 LLM API 는 §13 에서 비용을 확인받은 뒤에만 부른다.
- §5–§12 의 run 은 경로만 바꾼 설정 사본(§1.3 의 `/tmp/lxplan/lx.yaml`)으로 돌려 저장소의 `results/` 를 건드리지 않는다. 저장소에 쓰는 것은 `make probe`(`config/machines/<호스트>.yaml`), `make tests`(`tasks/*/tests/`, git 무시), `make aa`(`results/aa/`, `results/built/aa-…`), §13 의 실제 측정뿐이다.
- 기대와 다르면 그 자리에서 고치지 말고 명령·출력·파일 경로를 남긴 뒤 다음 단계로 간다. 고친 것과 정한 것은 `docs/decisions.md` 에 날짜와 함께 적는다.
- 명령은 저장소 루트에서, 셸 하나에서 차례로 실행한다. 명령과 옵션은 `Makefile` 과 각 CLI 의 `--help` 그대로다.

---

## 1. 준비

### 1.1 `docs/desktop_setup.md` 의 단계

1–3(패키지, 저장소·venv, rustup), 5(거버너), 6(`make probe`, `--suggest-cores`), 7(`config/experiment_desktop.yaml`)을 마친 상태에서 시작한다. 4(`.env`)는 §13 에서만 필요하다. 8(테스트 입력·단위 테스트)은 이 문서의 §3·§4 가, 9–11(A/A, 1차 실험 재측정, 재생성)은 §10·§13 이 이어받는다.

### 1.2 통합 하네스가 더 필요로 하는 것

```sh
sudo apt install cppcheck clang-tidy           # C 린트 베이스라인(plugins/lint/run_lint.py). 없으면 C 행의 lint_* 가 NA 이고 run 은 계속된다
make rust-setup CARGO=~/.cargo/bin/cargo       # Rust 템플릿의 고정 크레이트(rayon =1.12.0)를 한 번 내려받는다(온라인)
```

- `make rust-setup` 은 `cargo fetch --locked --manifest-path harness/build/rust_template/Cargo.toml` 이다. 빠뜨리면 rust 가 들어간 `harness.run` 이 행을 쓰기 전에 exit 7(`중단: 도구 없음 (Rust 툴체인 점검)`, 사유 끝에 `cargo fetch --locked …` 안내)로 멈춘다. 빌드는 `build.rust.offline: true` 로 `--offline --locked` 다.
- 하네스 자신은 PATH 가 없어도 cargo·rustc 를 PATH → `$CARGO_HOME/bin` → `~/.cargo/bin` 순으로 찾는다(`protocol.rust_tool`, §2.1 에서 확인). 단위 테스트의 Rust 테스트도 같은 함수로 찾으므로(`tests/_support.requires_rust_tool`) PATH 를 바꾸지 않는다 — `~/.cargo/bin` 이 PATH 에 없는 셸(desktop_setup 3단계의 `--no-modify-path`)에서 §4.1 을 돌려야 그 탐색이 시험된다.
- `mojo` 는 하네스에 없다. `requirements.txt` 가 Linux 에서 `mojo`·`max` 를 설치하지만 어떤 단계도 그것을 쓰지 않는다.

### 1.3 검증용 설정 사본과 도우미

```sh
mkdir -p /tmp/lxplan/results
sed -E 's#^(  (results_csv|results_csv_dummy|raw_dir|built_dir): )results/#\1/tmp/lxplan/results/#' \
    config/experiment_desktop.yaml > /tmp/lxplan/lx.yaml
diff config/experiment_desktop.yaml /tmp/lxplan/lx.yaml      # paths 의 네 줄만 달라야 한다
PY=.venv/bin/python
X="--config /tmp/lxplan/lx.yaml"
D=/tmp/lxplan/results/runs_dummy.csv                       # provider dummy 인 행이 가는 CSV(paths.results_csv_dummy)

# 행 보기: $PY /tmp/lxplan/rows.py <csv> <run_id>
cat > /tmp/lxplan/rows.py <<'EOF'
import csv, json, sys
from pathlib import Path
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    if r["run_id"] != sys.argv[2]:
        continue
    res = Path(r["built_path"]) / "result.json"
    stages = json.loads(res.read_text(encoding="utf-8"))["stages"] if res.is_file() else {}
    aff = stages.get("call_affinity") or {}
    env = json.loads(r["omp_env"] or "{}")
    print(r["task"], r["language"], r["feedback"], f's{r["sample_idx"]}', f't{r["turn"]}', f'pass={r["pass"]}',
          f'{r["error_stage"]}/{r["error_type"]}', r["error_detail"][:70], f'rounds={r["bench_rounds"]}',
          f'speedup_mean={r["speedup_mean"]}', f'thr={r["threads"]}', f'cores={r["cores"]}', f'kernel_cpus={aff.get("kernel")}',
          f'libgomp={aff.get("libgomp_paths")}', f'ok={r["protocol_ok"]}', r["protocol_flags"], ",".join(sorted(env)), sep="  ")
EOF
```

설정의 상대 경로(프롬프트, 린트 플러그인)는 설정 파일 위치가 아니라 저장소 루트 기준으로 풀린다(`harness.paths.resolve`). 아래에서 C 는 `measurement.cores`, T 는 `measurement.threads`, c0 는 C 의 첫 CPU 다(`config/experiment_desktop.yaml` 의 값).

---

## 2. 환경과 도구

### 2.1 점검

```sh
$PY -m harness.protocol --check $X; echo rc=$?
make probe CONFIG=config/experiment_desktop.yaml
grep -E '^(cargo|rustc|numba_version|halide_version|halide_autoscheduler|halide_autoscheduler_loadable|taskset_available|governor):' \
     config/machines/$(hostname).yaml
grep -A6 '^lint_tools:' config/machines/$(hostname).yaml
grep -A3 '^numpy_blas:' config/machines/$(hostname).yaml
$PY -m harness.protocol --python-toolchain Li2018
NUMBA_THREADING_LAYER=omp NUMBA_NUM_THREADS=4 $PY -m harness.protocol --probe-runtime numba
$PY -m harness.protocol --probe-runtime halide --halide-autoscheduler Li2018 --halide-params '{}' --threads 4
$PY -c "import scipy" 2>&1 | tail -1
env PATH=/usr/bin:/bin $PY -c "from harness.protocol import rust_tool; print(rust_tool('cargo'), rust_tool('rustc'))"
file -L "$(gcc -print-file-name=libasan.so)"
```

기대:

- `--check`: rc 0, `static.ok`·`dynamic.ok` 가 true. 플래그가 있으면 정보용(`turbo=`, `load_1m=`, `env_overrides_dropped`)뿐이다(정보용 `multiple_libgomp` 는 bench 가 행에만 붙인다; §6.2). `governor=…` 이면 desktop_setup 5단계를, `taskset_missing` 이면 util-linux 를 본다.
- 머신 프로필: `cargo`·`rustc` 가 null 이 아니다(PATH 에 없어도 `rust_tool` 로 찾는다), `halide_autoscheduler: Li2018`, `halide_autoscheduler_loadable: true`, `numba_version: 0.68.1`, `halide_version: 22.1.0`, `taskset_available: true`, `lint_tools` 에 ruff·pyflakes·cppcheck·clang-tidy 의 판, `numpy_blas.name: scipy-openblas`.
- `--python-toolchain Li2018`: `"halide_autoscheduler_loadable": true`, `"errors": {}`.
- 두 `--probe-runtime`: `{"error": null}`.
- `import scipy`: `ModuleNotFoundError`. SciPy 가 있으면 numba 가 든 run 은 exit 7 이다(2026-11-12 결정: numba 측정 환경에는 SciPy 가 없다).
- `rust_tool`: 두 경로가 `~/.cargo/bin/` 아래.
- `file -L`: `ELF 64-bit … shared object`(sanitizer 가 `LD_PRELOAD` 로 앞에 올린다; §7).

### 2.2 exit 7 의 뜻

`harness.run` 은 처리할 언어의 도구를 행을 쓰기 전에 점검하고, 쓸 수 없으면 exit 7 로 멈춘다(없는 도구로 생긴 실패가 모델의 `compile_error` 행이 되지 않게). 메시지와 원인:

| 메시지 | 원인과 볼 곳 |
|---|---|
| `중단: 필요한 도구가 없다 …` + ``C 컴파일러 `gcc` 를 실행할 수 없다 (build.cc) — 필요한 곳: c_seq, …`` | c_seq·c_openmp 가 있거나, bench 를 하는데 C 기준선 과제(jacobi2d, spmv_csr, histogram, scan, nbody, floyd_warshall, bfs)가 있으면(`필요한 곳: C 기준선`) gcc 가 필요하다 |
| 같은 머리 + `` `cargo` 를 실행할 수 없다 (build.rust.cargo=cargo) `` / `` `rustc` 를 실행할 수 없다 `` (`protocol.rust_tool` 이 실행 파일을 찾았으면 이름 대신 그 경로) | rustup 설치(desktop_setup 3단계) |
| 같은 머리 + `… 에서 numba 를 임포트할 수 없다` / `… 에서 halide 를 임포트할 수 없다` | venv(desktop_setup 2단계) |
| 같은 머리 + `Halide 오토스케줄러 Li2018 의 플러그인을 로드할 수 없다 (…)` | halide 휠의 오토스케줄러 플러그인. `$PY -m harness.protocol --python-toolchain Li2018` 의 `errors` |
| `중단: 도구·설정을 쓸 수 없다 …` + `` `taskset -c <코어> … -c pass` 가 실패했다 … measurement.cores=… 가 이 머신의 CPU 와 맞는지 확인하라 `` | `measurement.cores`·`single_core` 가 이 장비에 없는 CPU 를 가리킨다 |
| 같은 머리 + `halide.autoscheduler=… halide.autoscheduler_params=… (threads=…) 를 쓸 수 없다` | 오토스케줄러 설정. 허용하지 않는 매개변수는 그 전에 설정 로더가 `설정 오류`(exit 2)로 막는다(Li2018 은 매개변수 없음, Mullapudi2016 `last_level_cache_size`·`balance`, Adams2019 `beam_size`) |
| 같은 머리 + `languages.numba.env NUMBA_THREADING_LAYER=omp 를 쓸 수 없다` | numba 의 omp 계층이 libgomp 를 올리지 못한다 |
| 같은 머리 + `… 에서 scipy <판> 를 임포트할 수 있다 — numba 측정 환경에는 SciPy 가 없어야 한다` / `SciPy 점검 프로세스가 결과 없이 끝났다` | 워커 venv 에서 SciPy 를 지운다 / 점검 프로세스의 stderr |
| `중단: 도구 없음 (Rust 툴체인 점검): Rust 툴체인 점검 실패 — 자리표시 크레이트 빌드가 …` | `make rust-setup` 을 하지 않았거나 링커·RUSTFLAGS 문제. 메시지 끝의 `전체 로그:` 경로 |
| `중단: 도구 없음 (기준선 빌드): …` | C 기준선 빌드의 컴파일러·`nm` |
| `[run] 도구 없음 — 행을 쓰지 않고 중단한다 (raw …)` | 샘플 처리 중의 도구 문제(예: 워커가 Halide 오토스케줄러 플러그인을 로드하지 못했다) |
| `[run] 도구 없음 — bench 라운드를 멈춘다` | 라운드 2 이후 C 기준선을 다시 빌드하다 컴파일러를 찾지 못했다 |

### 2.3 exit 7 을 일부러 내 보기 (선택)

```sh
$PY -m harness.generate $X --provider dummy --run-id lx-tool --tasks gemm,jacobi2d --languages numpy,c_seq,numba,rust --n 1
sed -E 's/^  cc: gcc/  cc: no-such-cc/' /tmp/lxplan/lx.yaml > /tmp/lxplan/nocc.yaml
$PY -m harness.run --config /tmp/lxplan/nocc.yaml --run-id lx-tool --languages c_seq --skip-bench; echo rc=$?   # 7, 필요한 곳: c_seq
$PY -m harness.run --config /tmp/lxplan/nocc.yaml --run-id lx-tool --languages numpy; echo rc=$?                # 7, 필요한 곳: C 기준선(jacobi2d)
sed -E 's/NUMBA_THREADING_LAYER: omp/NUMBA_THREADING_LAYER: tbb/' /tmp/lxplan/lx.yaml > /tmp/lxplan/tbb.yaml
$PY -m harness.run --config /tmp/lxplan/tbb.yaml --run-id lx-tool --languages numba --skip-bench; echo rc=$?     # 7 (tbb 패키지가 없을 때)
env PATH=/usr/bin:/bin $PY -m harness.run $X --run-id lx-tool --languages rust --skip-bench; echo rc=$?          # 0: PATH 밖의 cargo 를 찾는다
$PY /tmp/lxplan/rows.py $D lx-tool
```

기대: exit 7 인 호출 뒤에는 그 언어의 행이 없다. 마지막 호출은 rust 행 두 개(`pass=1`)를 쓴다.

---

## 3. 테스트 입력과 `test_hash`

```sh
make tests                                     # tasks/*/gen_tests.py (과제 12개, 약 1.6 GB)
$PY - <<'EOF'
import csv, json
from pathlib import Path
recorded = {}
for r in csv.DictReader(open("results/runs.csv", encoding="utf-8")):
    recorded.setdefault(r["task"], set()).add(r["test_hash"])
for task in sorted(recorded):
    now = json.loads(Path(f"tasks/{task}/tests/manifest.json").read_text(encoding="utf-8"))["test_hash"]
    print(f"{task:15s} {now}  {'OK' if recorded[task] == {now} else 'DIFF recorded=' + ','.join(sorted(recorded[task]))}")
EOF
```

기대: 12줄 모두 `OK`. `results/runs.csv` 의 실제 행(189행)이 기록한 값은 다음과 같다.

| 과제 | `test_hash` | 과제 | `test_hash` |
|---|---|---|---|
| gemm | `2077183b4edc9ec8` | nbody | `4733671afc8dfab8` |
| jacobi2d | `2817c6d5544625d8` | sort | `0dd774dfc4d6c0ec` |
| spmv_csr | `fcab7d32e352a1eb` | cholesky | `6c8c90f9caec7b77` |
| dot | `0ed48e0b583e1ac6` | floyd_warshall | `2a9ff0da4e20355e` |
| histogram | `554647534c502bb3` | bfs | `00659fe983b894a1` |
| scan | `5d19e39eaa354980` | fft | `01f8ac04e44b4a83` |

- cholesky 는 Linux 에서 `6c8c90f9caec7b77` 이어야 한다. Windows 에서는 `901d6d94674db16b` 였다: 생성기(`tasks/cholesky/gen_tests.py`)가 `b @ b.T` 로 BLAS 를 거치고, Windows 의 OpenBLAS 결과가 노트북(Linux)과 달랐다.
- 다를 때: 그 과제의 입력이 기록된 행과 다르다는 뜻이므로 그 과제의 재측정(§13)을 하지 않고 원인을 찾는다. 과제 하나만 다시 만들기: `$PY tools/devcheck.py gen --task cholesky` → `cholesky/gen => PASS test_hash=… seed=…`. BLAS 를 거치는 cholesky 면 이 장비의 `numpy_blas.config`(OpenBLAS 가 고른 커널, `MAX_THREADS`)를 노트북 프로필 `config/machines/ustar-Zenbook-UX3402ZA-UX3402ZA.yaml` 의 값(`OpenBLAS 0.3.30  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64`)과 비교하고, `OPENBLAS_NUM_THREADS=1 $PY tasks/cholesky/gen_tests.py --out /tmp/lxplan/chol` 의 `manifest.json` 으로 스레드 수의 영향을 본다. 결과와 결정을 `docs/decisions.md` 에 적는다.

---

## 4. 단위 테스트

### 4.1 전체

```sh
make unittest 2>&1 | tail -3                                                 # = .venv/bin/python -m unittest discover -s tests
$PY -m unittest discover -s tests -v 2>&1 | grep -E "\.\.\. skipped" | sort | uniq -c | sort -rn
```

- 기대: `OK`. Windows 의 개수는 2026-11-23 항목의 검증 줄에 있다. Linux 에서는 Windows 가 건너뛴 Linux 전용 테스트(gcc·nm·taskset·프로세스 그룹·sanitizer·SDK)가 더 돌아 skip 이 줄어든다.
- 플랫폼 때문에 건너뛰는 것은 없다. Linux 에서도 언제나 건너뛰는 것은 개발 머신용 7개뿐이다: `test_devcheck.ZigBuildTest`(1)·`ZigOmpEndToEndTest`(5)는 `ziglang` 패키지(`requirements.txt` 에 없다)가, `test_taskspec … test_platform_without_resource_reports_unsupported_limits`(1)는 `resource` 모듈이 없는 플랫폼이 필요하다.
- 장비에 따라 이유와 함께 건너뛰는 것: `test_worker.CKernelTest` 의 친화도 테스트("서로 다른 물리 코어의 CPU 2개 필요", "친화도가 좁아지지 않아 … 관측할 수 없다", "고정 집합 밖의 CPU 가 필요"), `test_build_c.RealGccTest` 의 `target_clones` 테스트(x86-64 가 아닐 때)와 libasan 테스트(libasan 이 없을 때), `test_generate` 의 Rust 조각 테스트(`make rust-setup` 을 하지 않았을 때).
- 그 밖의 skip 은 문제다. 특히:
  - `Rust 도구 필요: cargo, rustc (PATH·$CARGO_HOME/bin·~/.cargo/bin 어디에도 없음)`(`tests/_support.requires_rust_tool`): rustup 설치(desktop_setup 3단계)가 되지 않았다. 이때 `test_build_rust.RealCargoTest`, `test_worker.RustAbiTest`, `test_sigs.RustKeywordScopeTest`, `test_generate` 의 Rust 조각 테스트가 빠져 Linux 의 링크 인자(`-Wl,--no-undefined`)와 panic 경로가 시험되지 않는다.
  - `실행 파일 필요: gcc, nm (PATH 에 없음)`·`gcc 필요`·`taskset 필요`·`bash/pgrep 필요`: desktop_setup 1단계의 패키지가 빠졌다.
- `requires_modules` 로 표시한 테스트(anthropic·openai SDK, numba, halide, ruff·pyflakes — `test_lint_plugin.EndToEndPythonTest`·`HarnessRunLintTest`)는 Linux 에서 패키지가 없으면 건너뛰지 않고 `ModuleNotFoundError`(`설치되지 않은 패키지: … (requirements.txt)`)로 실패한다: venv 가 `requirements.txt` 대로가 아니다.
- 다를 때: 실패한 클래스의 모듈만 `-v` 로 다시 돌린다(`$PY -m unittest tests.test_<모듈> -v`). gcc·sanitizer 실패는 §2.1 의 `file -L`, Rust 실패는 §1.2 를 먼저 본다.

### 4.2 기존 36개 조건의 프롬프트 해시

```sh
$PY -m unittest tests.test_generate.PromptHashTest -v
```

- 기대: 5개 `ok`. 과제 12 × numpy·c_seq·c_openmp(positional, hint off)를 기록된 프롬프트를 만든 노트북 프로필(`config/machines/ustar-Zenbook-UX3402ZA-UX3402ZA.yaml`)로 렌더링한 해시가 `tests/fixtures/prompt_hashes_upstream.json` 과 같고(`test_default_condition_reproduces_the_up_hashes`), 그 값이 `results/runs.csv` 의 실제 행의 `prompt_hash` 와 같다(`test_fixture_hashes_are_those_of_the_recorded_runs`). 프로필을 파일로 지정해 읽으므로 호스트와 무관하다. Linux 의 LF 체크아웃에서 렌더링 결과에 `\r` 이 없다는 것도 같은 테스트가 본다.
- 다를 때: `prompts/system.md`, `prompts/lang/{numpy,c_seq,c_openmp}.md`, `tasks/*/spec.md`, `tasks/*/signatures.yaml` 의 `languages:` 문자열을 통합 전 main(`6f02d16`)과 비교한다(`git diff 6f02d16 -- <파일>`). 이 해시가 다르면 1차 실험 응답과 같은 프롬프트라는 전제가 깨진다.

---

## 5. 점검 도구 `tools/devcheck.py` (실제 gcc)

devcheck 는 추출 → 빌드 → import → check 를 한 번 돌리는 개발용 도구다(측정 아님, 코어 고정 없음). Linux 에 설정의 gcc 가 있으면 C 는 자동으로 `harness.build.c.build_c` 와 설정의 플래그를 쓰고 판정 줄에 `toolchain=` 표시가 없다. 종료 코드 0 통과, 1 실패, 2 사용법·도구 문제.

### 5.1 더미 전부

```sh
make devcheck CONFIG=config/experiment_desktop.yaml 2>/tmp/lxplan/devcheck.err | tee /tmp/lxplan/devcheck.txt | tail -1
```

- 기대: `summary: 80 pass, 0 fail, 0 missing, 0 problem, 4 unsupported`. 80 = 과제 12 × 언어 7 − 4(sort·bfs × halide_auto·halide_manual; `signatures.yaml` 의 `unsupported`). 어느 줄에도 `toolchain=` 이 없고, c_seq 줄에 `violations=`(`threads_in_c_seq`, `forbidden_symbol:…`)이 없다.
- `dummies` 는 정확성 형태(tiny·rect·small)만 돈다. 성능 형태는 파일마다 `$PY tools/devcheck.py kernel --task <과제> --language <언어> --file harness/dummy_responses/<과제>/<언어>/0.md --tests large` 로 본다(§6 의 bench 가 같은 형태를 덮는다).
- 다를 때: 실패 줄의 `detail` 과 stderr 의 `[devcheck]` 메모로 판단하고, `--keep /tmp/lxplan/keep` 으로 빌드 디렉터리(`build.log`, `build.json`, `import.json`, `check.json`, `*.stderr.log`)를 남긴다. C 만 실패하면 gcc 고유 동작(경고, glibc), numba 만이면 omp 계층, rust 만이면 `make rust-setup`·링크 인자, halide 만이면 Li2018 을 본다.

### 5.2 기준선

```sh
make devcheck-baselines CONFIG=config/experiment_desktop.yaml 2>&1 | grep '=>'
```

기대: 12줄 모두 `=> PASS stage=check classification=pass`(C+OpenMP 기준선 7개: jacobi2d, spmv_csr, histogram, scan, nbody, floyd_warshall, bfs; NumPy 기준선 5개: gemm, dot, sort, cholesky, fft). Windows 에서는 C 기준선을 clang + LLVM OpenMP 로만 확인했다.

### 5.3 틀린 변형 (gcc + libgomp)

```sh
for f in gemm/c_openmp/2 histogram/c_openmp/1 scan/c_openmp/1 floyd_warshall/c_openmp/1 bfs/c_openmp/1 gemm/c_openmp/1; do
  t=${f%%/*}; $PY tools/devcheck.py kernel --task $t --language c_openmp --file harness/dummy_responses/$f.md 2>/dev/null | tail -1
done
```

기대(`harness/dummy_responses/*/README.md`):

- 앞의 다섯 줄: `=> FAIL stage=check classification=race_condition`, detail `nondeterministic:<형태 목록>`. 목록은 타이밍에 달려 있다(gemm 은 rect·small 이 늘 들어가고 tiny 는 때때로). 두 호출이 같게 틀리면 `wrong_output` 이 될 수 있다(README 가 둘 다 허용한다).
- `gemm/c_openmp/1`(NULL 저장): `=> FAIL stage=check classification=runtime_crash detail=worker died (signal SIGSEGV) during tiny call 0`. SIGILL 이면 `volatile` 이 빠진 것이다(README "변형별 주의").
- 스레드 수를 바꿔 보려면 `measurement.threads`(와 `cores`)를 바꾼 설정 사본을 `--config` 로 준다. devcheck 는 워커의 스레드 변수를 언어의 설정값으로 덮어쓴다.

### 5.4 키워드 조건·Rust panic·halide_auto 의 스케줄 호출

```sh
mkdir -p /tmp/lxplan/kw
for l in numpy numba halide_auto halide_manual; do
  sed -E 's/^def dot\(/def dot(*, /' harness/dummy_responses/dot/$l/0.md > /tmp/lxplan/kw/dot_$l.md
  $PY tools/devcheck.py kernel --task dot --language $l --args-mode keyword --file /tmp/lxplan/kw/dot_$l.md 2>/dev/null | tail -1
done
cat > /tmp/lxplan/kw/dot_kw.c <<'EOF'
#include "kernel_args.h"

void dot(const dot_args* args) {
    double s = 0.0;
    for (int i = 0; i < args->n; i++) s += (double)args->x[i] * (double)args->y[i];
    args->out[0] = (float)s;
}
EOF
for l in c_seq c_openmp; do $PY tools/devcheck.py kernel --task dot --language $l --args-mode keyword --file /tmp/lxplan/kw/dot_kw.c 2>/dev/null | tail -1; done
$PY -c "import re,sys; t=open('tests/test_build_rust.py',encoding='utf-8').read(); open(sys.argv[1],'w').write(re.search(r\"RS_DOT_KEYWORD = '''(.*?)'''\", t, re.S).group(1))" /tmp/lxplan/kw/dot_kw.rs
$PY tools/devcheck.py kernel --task dot --language rust --args-mode keyword --file /tmp/lxplan/kw/dot_kw.rs 2>/dev/null | tail -1
$PY tools/devcheck.py dummies --tasks dot --args-mode keyword 2>/dev/null | tail -1

cat > /tmp/lxplan/dot_panic.rs <<'EOF'
#[no_mangle]
pub unsafe extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {
    if n > 100 {
        panic!("n is too large: {}", n);
    }
    let x = std::slice::from_raw_parts(x, n as usize);
    let y = std::slice::from_raw_parts(y, n as usize);
    let mut s = 0.0f64;
    for i in 0..n as usize {
        s += x[i] as f64 * y[i] as f64;
    }
    *out = s as f32;
}
EOF
$PY tools/devcheck.py kernel --task dot --language rust --file /tmp/lxplan/dot_panic.rs 2>&1 | tail -4
$PY tools/devcheck.py kernel --task gemm --language halide_auto --file tests/fixtures/perf_summary/halide/gemm_auto_scheduled.py 2>&1 | tail -2
```

기대(Windows 에서 같은 명령으로 확인한 결과 — c_openmp 키워드 줄은 돌리지 않았다; Linux 에서는 C 줄에 `toolchain=` 이 없다):

- 키워드 커널 7줄: 모두 `=> PASS stage=check classification=pass`. C 는 하네스가 쓴 `kernel_args.h` 의 `dot_args` 구조체, Rust 는 `kernel_args` 모듈의 `DotArgs` 를 거친다.
- 위치 인자 더미를 키워드 조건으로 돌리면 `summary: 0 pass, 7 fail, 0 missing, 0 problem, 0 unsupported`(모두 `extract/signature_mismatch: param_mismatch:…`).
- panic 커널: `=> FAIL stage=check classification=runtime_crash detail=rust_panic:n is too large: 100003` 와 stderr 의 `rust panic at src/lib.rs:4:9`. Linux 에서는 panic = abort 라 워커가 SIGABRT 로 끝난다(Windows 는 0xC0000409).
- halide_auto 픽스처: `=> PASS stage=check classification=pass violations=schedule_in_auto`, stderr 에 워커가 무시한 호출 목록(`Func.vectorize, Func.parallel, Stage.reorder, …`).

---

## 6. 더미 파이프라인 (`harness.run`, 언어 일곱 개, 3라운드)

### 6.1 모든 0.md (`lx-all`)

```sh
$PY -m harness.generate $X --provider dummy --run-id lx-all --languages numpy,c_seq,c_openmp,numba,halide_auto,halide_manual,rust --n 1
$PY -m harness.run $X --run-id lx-all 2>&1 | tee /tmp/lxplan/lx-all.log | tail -12; echo rc=${PIPESTATUS[0]}
$PY /tmp/lxplan/rows.py $D lx-all > /tmp/lxplan/lx-all.rows; wc -l < /tmp/lxplan/lx-all.rows
$PY -m analysis.aggregate --csv $D --run-id lx-all --coverage
```

`make dummy-all` 과 같은 순서이고, 그 타깃의 집계 단계가 `results/runs_dummy.csv` 를 읽기 때문에 사본 설정에서는 명령을 직접 쓴다.

- 생성: `calls=80`, stderr 에 sort·bfs × halide 두 언어의 `지원하지 않는 조합이라 건너뛴다` 네 줄.
- run: rc 0. 로그에 과제마다 run 시작 기준선·1스레드 기준선, 샘플마다 라운드 1, 이어서 `bench 라운드 2/3 ok`·`3/3 ok`. 끝에 `run 요약: 행 80개 (건너뜀 0개)`.
- 행 80개 모두 `pass=1`, 오류 없음, `rounds=3`, `speedup_mean` 이 비지 않는다. `--coverage` 표는 모든 칸 1, sort·bfs 의 halide 두 칸 `n/a`.
- 다시 부르기: `$PY -m harness.run $X --run-id lx-all` → 기준선을 다시 재지 않고(`처리할 샘플이 없어(모두 최신) 기준선 측정을 건너뜀`) 샘플마다 `result.json 이 최신이라 건너뜀`, `run 요약: 행 80개 (건너뜀 80개)`.
- 다를 때: 행의 `built_path` 디렉터리에서 `result.json` 의 `stages`(`extract`, `build`, `import`, `check`, `bench`, `bench_r2`, `bench_r3`, `rounds`)와 그 단계의 `<단계>.json`·`<단계>.stderr.log` 를 본다. C 행만 실패하면 §5.1 과 같은 순서로 본다.

### 6.2 행에서 볼 것

언어마다 bench 워커는 `measurement.threads` 로 뜨고(교차 측정), 행의 `threads`·`cores` 는 커널을 부르는 동안의 값이다(`run.bench_plan`, `protocol.call_cpus`).

| language | `threads` | `cores` | `omp_env` 에 더해지는 것 | bench.json `call_affinity.kernel` |
|---|---|---|---|---|
| numpy | T | C | — | C 의 CPU 목록 |
| c_seq | 1 | c0 | — | `[c0]` |
| c_openmp | T | C | `omp_env_extra` 넷(언제나) | `null` (libgomp 의 고정을 그대로) |
| numba | T | C | `omp_env_extra` 넷(언제나) + `NUMBA_NUM_THREADS`=T, `NUMBA_THREADING_LAYER`=omp | `null` |
| halide_auto / halide_manual | T | C | `HL_NUM_THREADS`=T | C 의 CPU 목록 |
| rust | T | C | `RAYON_NUM_THREADS`=T | C 의 CPU 목록 |

- 모든 행의 `omp_env` 에 `OMP_NUM_THREADS`·`OPENBLAS_NUM_THREADS`·`MKL_NUM_THREADS`(= T)와 `OPENBLAS_THREAD_TIMEOUT`(16)이 있다. `omp_env_extra`(`OMP_PROC_BIND`, `OMP_PLACES`, `OMP_DYNAMIC`, `OMP_WAIT_POLICY`)는 커널이 OpenMP 런타임 위에서 돌거나(c_openmp, numba — 언어 설정의 `omp_runtime`) 과제의 기준선이 C+OpenMP 일 때 붙는다.
- `call_affinity.baseline`: NumPy 기준선 과제(gemm, dot, sort, cholesky, fft)는 C 의 CPU 목록, C+OpenMP 기준선 과제는 `null`. `call_affinity.error` 는 `null` 이고 행에 `call_affinity_failed` 플래그가 없다. `result.json` 의 `stages.call_affinity` 는 라운드 1 의 값이고, 라운드마다의 값은 `bench.json`·`bench_r2.json`·`bench_r3.json` 에 있다.
- `protocol_ok=1`. 0 이면 `protocol_flags` 를 본다. 정보용 플래그(`turbo=`, `load_1m=`, `env_overrides_dropped`, `multiple_libgomp`)는 `protocol_ok` 를 0 으로 만들지 않고, `baseline_drift=…`(행의 대응 기준선이 그 run 의 같은 과제 대응 기준선들의 중앙값과 `measurement.baseline_drift_pct_max` = 10% 넘게 다름)는 만든다.
- `cflags`: c_seq `-std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native -lm -Wl,--no-undefined`(c_openmp 은 `-fopenmp` 가 더해진다), halide_auto `autoscheduler=Li2018 params={"parallelism":"<T>"}`, halide_manual `autoscheduler=none`, rust `RUSTFLAGS=-C target-cpu=native -C link-arg=-Wl,--no-undefined; profile.release: opt-level=3 codegen-units=1 lto="off" panic="abort" debug=false`, numpy·numba 는 비어 있다. `compiler` 는 c 가 gcc, rust `rustc`, numba `numba`, halide `halide`.
- `taskset_used=1`, `governor_ok=1`, `cpu_mhz_median`·`throttle_events`·`temp_c_start`·`temp_c_end` 가 채워진다(Windows 에서는 비었다).

### 6.3 새 언어에서 따로 볼 것

```sh
$PY - $D lx-all <<'EOF'
import csv, json, sys
from pathlib import Path
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    if r["run_id"] != sys.argv[2]:
        continue
    d = Path(r["built_path"])
    if r["language"] == "numba":
        imp = json.loads((d / "import.json").read_text(encoding="utf-8")).get("inspect") or {}
        paths = [json.loads((d / f).read_text(encoding="utf-8")).get("call_affinity", {}).get("libgomp_paths")
                 for f in ("bench.json", "bench_r2.json", "bench_r3.json") if (d / f).is_file()]
        print("numba", r["task"], imp.get("threading_layer"), imp.get("num_threads"), paths, r["protocol_flags"])
    elif r["language"].startswith("halide"):
        ms = [json.loads((d / f).read_text(encoding="utf-8")).get("build_ms")
              for f in ("bench.json", "bench_r2.json", "bench_r3.json") if (d / f).is_file()]
        print(r["language"], r["task"], "bench build_ms", ms, r["contract_violation"])
    elif r["language"] == "rust":
        b = json.loads((d / "result.json").read_text(encoding="utf-8"))["stages"]["build"]
        print("rust", r["task"], (b.get("extra") or {}).get("symbol_check"), r["compiler_version"])
EOF
raw=$(ls /tmp/lxplan/results/raw/lx-all/gemm/rust/*/sample_0.json)
$PY -c "import json,sys; print([l for l in json.load(open(sys.argv[1]))['prompt']['user'].splitlines() if 'RUSTFLAGS' in l])" $raw
```

- numba: `omp` 와 T. `libgomp_paths` 는 라운드마다 경로 목록이다. NumPy 기준선 과제에서는 numba 의 libgomp 만, C+OpenMP 기준선 과제에서는 numba 와 기준선이 올린 것이 함께 보인다. 서로 다른 경로가 둘 이상이면 행에 `multiple_libgomp` 가 붙는다(정보용; 2026-11-12 결정은 둘이 libgomp 하나를 공유한다고 가정했다). 어느 쪽이었는지 `docs/decisions.md` 에 적는다.
- halide: `bench build_ms` 는 라운드마다 형태별 빌드(오토스케줄 + JIT)에 든 시간이다. 측정 구간 밖이지만 `bench_timeout_s`(300 s) 안에 들어간다. 이 값이 2026-11-23 항목의 사용자 확인 3(빌드 시간만큼 제한을 늘릴지)의 근거다. halide_auto 더미에는 스케줄 호출이 없으므로 `contract_violation` 에 `schedule_in_auto` 가 없다.
- rust: `symbol_check` 가 `nm`(ELF 의 동적 심벌 검사), `compiler_version` 은 `rustc --version`.
- rust 프롬프트: `RUSTFLAGS="-C target-cpu=native -C link-arg=-Wl,--no-undefined" cargo build --release --offline --locked` 가 든 줄 하나(Linux 에서만 `rustflags_linux` 가 붙는다). 프롬프트는 도구를 경로 없이 `cargo` 로 적는다.

스레드 마스크(Halide 스레드 풀과 rayon 이 커널 호출의 CPU 집합을 물려받는가):

```sh
$PY -m harness.generate $X --provider dummy --run-id lx-thr --tasks gemm --languages halide_auto,rust --n 1
$PY -m harness.run $X --run-id lx-thr > /tmp/lxplan/lx-thr.log 2>&1 &
for i in $(seq 180); do
  for pid in $(pgrep -f 'harness.worker --backend (halide|c) .*--mode bench'); do
    echo "== $(tr '\0' ' ' < /proc/$pid/cmdline | grep -oE -- '--backend [a-z]+')"
    grep -h Cpus_allowed_list /proc/$pid/task/*/status 2>/dev/null | sort | uniq -c
  done; sleep 1
done > /tmp/lxplan/lx-thr.masks; wait; sort /tmp/lxplan/lx-thr.masks | uniq -c
```

기대: gemm 의 기준선은 NumPy 라서 bench 워커의 모든 스레드가 C 의 마스크를 가진다(커널은 목록을 범위로 적는다, 예: `0-3`). CPU 하나짜리 마스크가 보이면 스레드 풀이 1-CPU 고정을 물려받은 것이다. 스레드 수가 T 와 맞는지도 센다. 워커는 이 일치를 강제하지 않는다.

### 6.4 틀린 변형 (`lx-var`)

```sh
$PY -m harness.generate $X --provider dummy --run-id lx-var --tasks gemm,histogram,scan,floyd_warshall,bfs --languages numpy,c_seq,c_openmp --n 5
$PY -m harness.run $X --run-id lx-var 2>&1 | tail -20
$PY /tmp/lxplan/rows.py $D lx-var | cut -c1-200
```

더미 파일은 `sample_idx mod (그 언어의 숫자 이름 .md 수)` 로 고른다. 기대 행 75개:

| 과제 | language | sample | error_stage / error_type / error_detail | pass |
|---|---|---|---|---|
| gemm | numpy | 0 | 없음 | 1 |
| gemm | numpy | 1 | extract / signature_mismatch / `param_mismatch:…` | 0 |
| gemm | numpy | 2 | check / numeric_instability / `imprecise` | 0 |
| gemm | numpy | 3 | bench / timeout (순수 Python 3중 루프; 1024³ 첫 호출이 `bench_timeout_s` 를 넘는다) | 1 |
| gemm | numpy | 4 | check / wrong_output / `returned_array_not_written` | 0 |
| gemm | c_seq | 0, 3 | 없음 (`signature_exact=0`, `signature_ok=1`) | 1 |
| gemm | c_seq | 1, 4 | build / compile_error / `error: expected ',' or ';' before 'for'` | 0 |
| gemm | c_seq | 2 | check / arg_order / `transposed_output` | 0 |
| gemm | c_openmp | 0, 3 | 없음 | 1 |
| gemm | c_openmp | 1, 4 | check / runtime_crash / `signal SIGSEGV during tiny call 0` | 0 |
| gemm | c_openmp | 2 | check / race_condition / `nondeterministic:…`(rect·small 포함) | 0 |
| histogram·scan·floyd_warshall·bfs | numpy, c_seq | 0–4 | 없음 | 1 |
| histogram·scan·floyd_warshall·bfs | c_openmp | 0, 2, 4 | 없음 | 1 |
| histogram·scan·floyd_warshall·bfs | c_openmp | 1, 3 | check / race_condition / `nondeterministic:<목록>`(두 호출이 같게 틀리면 check / wrong_output / `mismatch`) | 0 |

- 통과한 행은 `rounds=3`, 오류 행은 라운드가 없다(bench timeout 인 gemm numpy/3 은 라운드 1 에서 끝난다).
- 다를 때: 그 행의 `check.json`(형태별 `pattern`, `n_fail`)과 `check.stderr.log`. gemm 의 compile_error 문구가 다르면 gcc 판의 차이다(분류가 같으면 문제가 아니다).

### 6.5 중단

```sh
$PY -m harness.generate $X --provider dummy --run-id lx-int --tasks gemm,bfs --languages numpy,rust --n 1
$PY -m harness.run $X --run-id lx-int &
sleep 20; kill -TERM $!; wait $!; echo rc=$?; pgrep -af harness.worker
```

기대: rc 130(`EXIT_INTERRUPTED`), 진행 중이던 샘플의 행 없음, `pgrep` 출력 없음(실행 중이던 서브프로세스 그룹을 정리한다). 다시 부르면 남은 샘플과 라운드를 잇는다.

---

## 7. sanitizer (ASan / UBSan)

```sh
$PY -m unittest tests.test_safety.RealSanitizerRunTest tests.test_safety.SanitizerEnvTest -v 2>&1 | tail -4
sed -E 's/^safety: \{enabled: false/safety: {enabled: true/' /tmp/lxplan/lx.yaml > /tmp/lxplan/san.yaml
$PY -m harness.generate --config /tmp/lxplan/san.yaml --provider dummy --run-id lx-san --tasks gemm --languages numpy,c_seq,c_openmp --n 5
$PY -m harness.run --config /tmp/lxplan/san.yaml --run-id lx-san --skip-bench
$PY - $D lx-san <<'EOF'
import csv, json, sys
from pathlib import Path
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    if r["run_id"] == sys.argv[2]:
        st = json.loads((Path(r["built_path"]) / "result.json").read_text(encoding="utf-8"))["stages"].get("sanitize") or {}
        print(r["language"], r["sample_idx"], r["error_type"], f'asan={r["asan_findings"]}', f'ubsan={r["ubsan_findings"]}',
              st.get("kinds"), repr(st.get("detail")), r["error_detail"][:90])
EOF
```

- 단위 테스트: `RealSanitizerRunTest` 4개(정상 커널, 커널 안의 버퍼 넘침, `--sanitize` 버퍼에서 하네스 출력 배열 한 칸 뒤 쓰기, 부호 있는 정수 넘침)와 `SanitizerEnvTest.test_real_gcc_resolves_an_existing_libasan` 가 `ok`.
- 규칙: sanitizer 는 `safety.enabled` 이고 C 커널일 때만, check 가 시간 초과가 아닐 때 돈다. 커널을 `build.sanitize_flags` 로 다시 빌드해(`san/kernel_san.so`) `LD_PRELOAD=<libasan>`, `--sanitize`(가드 없는 정확한 크기의 버퍼), RLIMIT_AS 없이 check 를 한 번 돈다. 그 턴의 판정이 `runtime_crash`·`wrong_output` 이고 ASan 보고가 1건 이상이면 `memory_error`, 상세 `asan:<첫 보고의 종류>; was <이전 유형>/<이전 상세>` 로 바꾼다(`docs/error_taxonomy.md` §5a). 통과한 행과 다른 유형은 바꾸지 않는다.
- 기대:
  - numpy 행: `asan`·`ubsan` NA(Python 커널은 대상이 아니다).
  - c_seq 0·3, c_openmp 0·3: `asan=0`, `ubsan=0`, detail `''`(커널을 로드해 끝까지 돈 실행).
  - c_seq 1·4(컴파일 오류): sanitize 단계 없음, NA.
  - c_seq 2(`arg_order`)·c_openmp 2(`race_condition`): 유형 그대로.
  - c_openmp 1·4(NULL 저장): ASan 이 그 크래시를 보고하면 `memory_error`, 상세 `asan:<종류>; was runtime_crash/signal SIGSEGV during tiny call 0`. 보고가 없으면 `runtime_crash` 그대로이고 `asan` 은 NA(detail `incomplete:signal …`). 어느 쪽이었는지 적는다.
- 다를 때: 샘플 디렉터리의 `san/`(`asan.<pid>`, `ubsan.<pid>`, `build.log`)와 `safety.stderr.log`, `result.json` 의 `stages.sanitize.summary`(빌드 명령, 환경변수, 로그 파일, 워커 요약). 로그만 다시 파싱: `$PY -m harness.safety --parse-logs <샘플>/san`. 정상 커널에서 보고가 나오면 CPython·NumPy·OpenBLAS 자체의 보고인지(커널 프레임이 있는지) 본다.

---

## 8. 린트 도구

```sh
mkdir -p /tmp/lxplan/lint
printf '#include <stdlib.h>\n#include <omp.h>\nvoid gemm(const float* A, const float* B, float* C, int M, int N, int K) {\n    float* t = malloc((size_t)M * N * sizeof(float));\n    int acc[8];\n    acc[8] = 0;\n    #pragma omp parallel for\n    for (int i = 0; i < M * N; i++) t[i] = A[0] * B[0] * (float)(1 / K);\n    C[0] = t[0] + (float)acc[8];\n}\n' > /tmp/lxplan/lint/kernel.c
$PY plugins/lint/run_lint.py --language c_openmp --file /tmp/lxplan/lint/kernel.c --task gemm --signature '{}' --out /tmp/lxplan/lint/lint.json
$PY -m json.tool /tmp/lxplan/lint/lint.json | head -40
$PY - $D lx-all <<'EOF'
import csv, sys, collections
c = collections.Counter()
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    if r["run_id"] == sys.argv[2]:
        c[(r["language"], "NA" if r["lint_findings_total"] == "" else "int")] += 1
print(sorted(c.items()))
EOF
```

- `run_lint.py`: `"tool": "baseline:cppcheck+clang-tidy"`, `versions` 에 두 도구, `summary` 의 여섯 범주가 모두 정수(null 없음), findings 에 `rules.yaml` 의 규칙(이 커널이면 `arrayIndexOutOfBounds`(memory), `memleak…`(memory), `bugprone-integer-division`(numeric) 같은 것). stderr 에 `[lint] clang-tidy: …/omp.h 를 -isystem 으로 준다`(OpenMP 언어는 gcc 의 `omp.h` 를 clang-tidy 에 준다).
- 파이프라인(`lx-all`): numpy·numba·halide_* 와 c_seq·c_openmp 행의 `lint_*` 는 정수, rust 행은 NA(Rust 린터 없음). 샘플 디렉터리마다 `lint.json`·`lint.stderr.log`.
- 도구가 `tool` 에서 빠지면(예: `baseline:cppcheck` 만) 그 도구가 보는 범주가 null 이고 행의 그 열이 NA 다. 사유는 플러그인 stderr 의 `[lint] <도구>: …` 한 줄이다(시간 초과 `TOOL_TIMEOUT_S` 20 s, 옵션 거부, `omp.h` 를 찾지 못함, 출력 형식). 실제 커널에서 잡음이 많은 규칙은 `plugins/lint/rules.yaml` 에서 빼거나 info 로 내리고 결정으로 적는다.
- 하네스 경로로 한 파일만: `$PY -m harness.safety $X --lint <샘플>/kernel.c --language c_openmp --task gemm`.

---

## 9. 수정 루프 (두 피드백 조건)

### 9.1 gemm × numpy·c_seq (`make repair-dummy` 의 순서)

```sh
R=lx-repair
$PY -m harness.generate $X --provider dummy --run-id $R --tasks gemm --languages numpy,c_seq --n 5 --feedback compiler
$PY -m harness.run $X --run-id $R
LLM_PROVIDER=dummy $PY -m harness.repair $X --run-id $R --feedback compiler --max-turns 3
$PY -m harness.repair $X --run-id $R --seed-from-feedback compiler --feedback compiler+lint
$PY -m harness.run $X --run-id $R
LLM_PROVIDER=dummy $PY -m harness.repair $X --run-id $R --feedback compiler+lint --max-turns 3
$PY -m analysis.aggregate --csv $D --run-id $R --by-turn --conditions
```

`harness.repair` 는 제공자를 설정·환경변수에서 읽으므로 `LLM_PROVIDER=dummy` 로 턴 0 의 제공자와 맞춘다. 수정 턴 t 의 응답은 같은 k 의 `<k>.turn<t>.md`, 없으면 `<k>.md` 다(gemm 에는 `c_seq/1.turn1.md`, `numpy/2.turn1.md` 만 있다). 대상은 직전 턴 행이 통과하지 못한 샘플이다(bench timeout 인 numpy/3 은 `pass=1` 이라 대상이 아니다).

기대(피드백 조건마다 같다):

| 턴 | 대상 | 통과 |
|---|---|---|
| 0 | 10행 | numpy 0·3, c_seq 0·3 |
| 1 | numpy 1·2·4, c_seq 1·2·4 (6) | numpy 2, c_seq 1·4 (`turns_to_pass=1`) |
| 2 | numpy 1·4, c_seq 2 (3) | 없음 |
| 3 | 같은 3개 | 없음 |

- 행: 조건마다 22, run 전체 44. 턴 1 이상의 행은 `sample_<i>/turn_<t>/` 에 생긴다. 통과한 수정 턴 행도 `rounds=3` 이다: 수정 루프는 턴을 다 처리한 뒤 자신이 쓴 행의 남은 bench 라운드를 `harness.run` 과 같은 함수(`run_pending_rounds`)로 돈다. 그래서 다음 턴의 대상은 라운드 1 의 판정으로 정해진다.
- `--by-turn`: `pass_within_0/1/2/3` numpy 0.4/0.6/0.6/0.6, c_seq 0.4/0.8/0.8/0.8. `turns_to_pass` numpy `0:2, 1:1, NA:2`, c_seq `0:2, 1:2, NA:1`.
- `--conditions`: (gemm, 언어)마다 `factor=feedback` 행 하나, 대응 비교(`n_base = n_alt = 5`), 차 0.000(더미 응답은 프롬프트와 무관). stderr 에 짝이 없는 샘플 경고가 없다.
- 같은 수정 명령을 다시 부르면 `수정할 샘플이 없다 (샘플 …개, 턴 상한 3)` 이고 아무것도 바뀌지 않는다.
- 수정 요청(raw `…/args-positional_hint-off_fb-<조건>/sample_<i>.turn1.json` 의 `repair.diagnostics` 와 마지막 user 메시지): c_seq 1 은 gcc 의 `kernel.c:<줄>:<열>: error: …`(절대 경로 없음), `compiler+lint` 쪽은 `## Static analysis findings` 절이 더 있다(린트가 돌았고 보고가 없으면 `None.`, 돌지 않았으면 `Not available.`). 하네스의 패턴 이름(`transposed_output` 등)과 참조값은 요청에 없다.

### 9.2 새 언어의 수정 턴 (numba 시그니처 규칙, Rust panic)

```sh
cat > /tmp/lxplan/dot_unsigned.py <<'EOF'
import numba


@numba.njit
def dot(x, y, out, n):
    s = 0.0
    for i in range(n):
        s += x[i] * y[i]
    out[0] = s
EOF
$PY -m harness.generate $X --provider dummy --run-id lx-rep --tasks dot --languages numba,rust --n 1
$PY - /tmp/lxplan/results/raw/lx-rep/dot <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for lang, fence, src in (("rust", "rust", "/tmp/lxplan/dot_panic.rs"), ("numba", "python", "/tmp/lxplan/dot_unsigned.py")):
    raw = root / lang / "args-positional_hint-off_fb-compiler" / "sample_0.json"
    d = json.loads(raw.read_text(encoding="utf-8"))
    d["response_text"] = f"Edited for the repair check.\n\n```{fence}\n" + Path(src).read_text(encoding="utf-8") + "```\n"
    raw.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
EOF
$PY -m harness.run $X --run-id lx-rep
LLM_PROVIDER=dummy $PY -m harness.repair $X --run-id lx-rep --max-turns 1
$PY -c "import json,glob; [print(p.split('/')[-3], json.load(open(p))['repair']['diagnostics'][:300]) for p in sorted(glob.glob('/tmp/lxplan/results/raw/lx-rep/dot/*/*/sample_0.turn1.json'))]"
$PY /tmp/lxplan/rows.py $D lx-rep | cut -c1-160
```

raw 를 손으로 고친 run 이므로 분석에 쓰지 않는다. 기대(Windows 에서 `--skip-bench` 로 같은 절차를 돌린 결과):

- 턴 0: numba `build/signature_mismatch`, 상세 `numba_unsigned_jit:dot`; rust `check/runtime_crash`, 상세 `rust_panic:n is too large: 100003`.
- 수정 요청: numba 는 ``The function `dot` is compiled with `njit` or `jit` without an explicit signature string as the first positional argument of its decorator. …``, rust 는 형태별 줄에 `- shape (n=100003): panicked at src/lib.rs:4:9: n is too large: 100003`(종료 코드·시그널 대신 panic 의 메시지와 위치).
- 턴 1(응답은 `dot/<언어>/0.md`): 두 언어 모두 `pass=1`, `turns_to_pass=1`, `rounds=3`.

---

## 10. A/A 점검 (`make aa`)

`docs/desktop_setup.md` 9단계 그대로다: `make aa CONFIG=config/experiment_desktop.yaml`. 기준선을 자기 자신과 교차 측정해(속도비 1 이 정상) 변형(`fsync`, `nofsync`)마다 평균 비·중앙값 비·라운드 범위·첫/둘째 비·커널 CV 를 표로 낸다. 요약은 `results/aa/<run_id>.json`, 워커 결과는 `results/built/<run_id>/<task>/baseline/aa/` 이고 결과 CSV 는 건드리지 않는다. `nofsync` 의 첫/둘째 비가 `fsync` 보다 1 에 가깝고 평균 비가 1 ± 0.02 안이면 `measurement.bench_loop_fsync: false` 로 바꾸고 결과를 `docs/decisions.md` 에 적는다(2026-11-12 승인 조건). 과제를 좁히려면 `TASKS=gemm,histogram`. 다를 때: 같은 시각의 `harness.protocol --check $X`(부하, 거버너)와 `results/aa/<run_id>.json` 의 라운드별 값을 본다.

---

## 11. 분석과 그림

```sh
$PY -W error -m unittest tests.test_stats tests.test_aggregate tests.test_plots 2>&1 | tail -3
$PY -m analysis.aggregate --csv $D --run-id lx-all --ci --coverage --halide
$PY -m analysis.aggregate --csv $D --run-id lx-repair --ci --by-turn --conditions
$PY -m analysis.plots --csv $D --run-id lx-all --out /tmp/lxplan/plots-all --by-task
$PY -m analysis.plots --csv $D --run-id lx-repair --out /tmp/lxplan/plots-repair
$PY -m analysis.plots --csv $D --run-id lx-all --out /tmp/lxplan/plots-pdf --format pdf
$PY -m analysis.aggregate --csv results/runs.csv --all-runs | head -3
$PY -m analysis.plots --csv results/runs.csv --all-runs --out /tmp/lxplan/plots-real
```

- 모두 rc 0. `analysis.plots` 는 쓴 파일 경로를 한 줄씩 출력하고, 자료가 없는 그림은 stderr 에 `<그림>: 건너뜀 — …` 로 알린다. `lx-all` 은 조건·수정 턴이 없어 `fig3_condition_effects`·`fig4_repair_turns` 를 건너뛰고 `fig1_languages`·`fig2_error_types_by_task` 를 쓴다. `lx-repair` 는 `fig1`·`fig2_error_types`·`fig3`(feedback)·`fig4` 네 장이다.
- `--coverage`: sort·bfs 의 halide 칸 `n/a`. `--ci`: 그룹 표에 `pass_at_1` 과 속도비 주 지표(설정 `measurement.primary_speedup: mean` → `speedup_mean`) 중앙값의 95% 부트스트랩 구간 열. `--halide`: Halide 의 어색한 과제(jacobi2d, spmv_csr, floyd_warshall, cholesky, fft)를 넣은/뺀 묶음별 값.
- 실제 결과(`results/runs.csv`, run 4개)도 rc 0 으로 읽히고 그려진다. `speedup_mean` 이 없는 행이 섞이면 `analysis.plots` 가 어느 열을 썼는지 stderr 에 한 줄로 알린다(`analysis.aggregate.choose_speedup`).
- 다를 때: `analysis/` 에는 Linux 의존이 없다. 값이 다르면 입력 행(§6, §9)부터 본다.

---

## 12. `--rebuild-csv` (실제 결과의 행을 잃지 않는가)

`--rebuild-csv` 는 `built_dir` 의 모든 `result.json` 에서 CSV 를 다시 쓴다. provider 가 dummy 인 행은 `results_csv_dummy`, 나머지는 `results_csv` 로 가고, result.json 으로 다시 만들 수 없는 기존 행은 지우지 않는다(2026-11-23 항목 "기존 동작에서 달라진 것" ⑥). 저장소의 CSV 대신 사본으로 확인한다.

```sh
mkdir -p /tmp/lxplan/rebuild && cp results/runs.csv /tmp/lxplan/rebuild/runs.csv
sed -E -e 's#^(  results_csv: ).*#\1/tmp/lxplan/rebuild/runs.csv#' \
       -e 's#^(  results_csv_dummy: ).*#\1/tmp/lxplan/rebuild/runs_dummy.csv#' /tmp/lxplan/lx.yaml > /tmp/lxplan/rebuild.yaml
$PY -m harness.run --config /tmp/lxplan/rebuild.yaml --rebuild-csv
$PY - <<'EOF'
import csv
KEY = ("run_id", "task", "language", "args", "grammar_hint", "feedback", "sample_idx", "turn")
def load(p): return {tuple(r[k] for k in KEY): r for r in csv.DictReader(open(p, encoding="utf-8"))}
a, b = load("results/runs.csv"), load("/tmp/lxplan/rebuild/runs.csv")
print("real:", len(a), len(b), "same keys" if a.keys() == b.keys() else "KEYS DIFFER",
      "cells differing:", sum(a[k][c] != b[k][c] for k in a.keys() & b.keys() for c in a[k]))
c, d = load("/tmp/lxplan/results/runs_dummy.csv"), load("/tmp/lxplan/rebuild/runs_dummy.csv")
print("dummy:", len(c), len(d), "same keys" if c.keys() == d.keys() else "KEYS DIFFER")
EOF
```

- 기대: stderr 에 `[schema] /tmp/lxplan/rebuild/runs.csv: 189행 재생성 (result.json <N>개; result.json 이 없어 기존 CSV 에서 유지한 행 189개)`. 비교는 `real: 189 189 same keys cells differing: 0`, `dummy:` 의 두 수가 같고 `same keys`(더미 행은 이 세션의 result.json 에서 다시 만들어진다).
- 주의: 기본 설정으로 `harness.run --rebuild-csv` 를 부르면 `results/runs.csv` 를 `results/built` 에서 다시 쓴다. §13.2 의 재측정은 노트북 run 과 같은 run_id 로 `results/built/<run_id>` 에 result.json 을 쓰므로, 그 뒤에 기본 설정으로 다시 만들면 같은 행 키의 노트북 행이 데스크톱 측정으로 바뀐다. 데스크톱의 CSV 를 다시 만들 때는 `--config config/experiment_desktop.yaml`(`results_csv: results/runs_desktop.csv`)로 한다.

---

## 13. 첫 실제 측정 (2026-11-12 결정의 순서)

§2–§12 가 끝나고 결과를 `docs/decisions.md` 에 적은 뒤에 한다. 재생성 전에 2026-11-23 항목의 "사용자 확인이 필요한 결정"을 정한다(특히 3: `bench_timeout_s` 와 Halide 빌드 시간 — §6.3 의 `build_ms` 가 근거).

### 13.1 측정 규약

`make aa` 의 결과로 `bench_loop_fsync` 를 정한다(§10). 측정 세션마다 거버너를 `performance` 로 두고 `$PY -m harness.protocol --check --config config/experiment_desktop.yaml` 이 rc 0 인지 본다. `STRICT=1`(`--strict-protocol`)이면 정적 규약 플래그(거버너·EPP·taskset·BLAS)가 있을 때 run 시작에서 exit 4 로 멈춘다. 노트북의 재측정(2026-11-12)은 `--force --strict-protocol` 로 했다.

### 13.2 노트북 응답을 데스크톱에서 다시 재기 (API 비용 없음)

`docs/desktop_setup.md` 10단계 그대로다: `make unpack-raw` 로 `results/archive/raw_1st_experiment_laptop.tar.gz` 를 `results/raw` 에 풀고(`20261105-real`, `20261108-tier1`, `20261108-tier2`, `20261109-tier3`), 세 run 을 `--force --config config/experiment_desktop.yaml` 로 잰 뒤 `analysis.aggregate --csv results/runs_desktop.csv --all-runs` 로 본다. 언어는 numpy·c_seq·c_openmp 뿐이고, 행은 `results/runs_desktop.csv` 로 가며 `run_id` 는 노트북 run 의 이름 그대로다. 노트북 행(`results/runs.csv`)은 바뀌지 않는다.

통과·실패와 오류 분류는 노트북 행과 같아야 한다(같은 코드, 같은 입력, 기존 세 언어의 판정 규칙은 바뀌지 않았다):

```sh
$PY - <<'EOF'
import csv
KEY = ("run_id", "task", "language", "args", "grammar_hint", "feedback", "sample_idx", "turn")
def load(p): return {tuple(r[k] for k in KEY): r for r in csv.DictReader(open(p, encoding="utf-8"))}
lap, desk = load("results/runs.csv"), load("results/runs_desktop.csv")
for k in sorted(lap.keys() & desk.keys()):
    a, b = lap[k], desk[k]
    if (a["pass"], a["error_type"]) != (b["pass"], b["error_type"]) or a["test_hash"] != b["test_hash"]:
        print(k[:3], k[6], "laptop", a["pass"], a["error_type"], "desktop", b["pass"], b["error_type"], b["error_detail"][:60])
print("compared", len(lap.keys() & desk.keys()))
EOF
```

다른 행은 그 `result.json` 을 본다. 경쟁 상태처럼 타이밍에 달린 판정은 측정마다 달라질 수 있다. `test_hash` 가 다른 과제(§3)는 비교하지 않는다.

### 13.3 모든 조건을 데스크톱에서 다시 생성 (비용 확인 먼저)

2026-11-12 결정: 프롬프트의 하드웨어 정보가 장비마다 다르므로 최종 장비에서 모든 조건을 다시 생성한다. 실제 호출 직전에 사용자에게 비용을 다시 확인받는다(CLAUDE.md 원칙 5). `.env`(desktop_setup 4단계)가 있어야 한다.

```sh
make real RUN_ID=<새 run id> CONFIG=config/experiment_desktop.yaml LANGUAGES=numpy,c_seq,c_openmp,numba,halide_auto,halide_manual,rust [STRICT=1]
```

- `harness.generate` 는 실제 제공자에서 `preflight()` 를 먼저 하고(실패하면 exit 3), `[generate] <N> calls to <모델> via <제공자>, max_tokens=…, effort=…` 를 출력한 뒤 `Proceed? [y/N]` 를 묻는다. 비대화형이고 `--yes` 가 없으면 exit 2 로 멈춘다. 이미 응답이 있는 샘플은 다시 부르지 않는다.
- 호출 수(기본 조건, 턴 0): 지원 조합 80(과제 12 × 언어 7 − halide 의 sort·bfs 4) × `run.n_samples` 5 = 400. 2026-11-12 의 추정(약 400~460회, 10~15달러)은 `mojo` 를 넣은 계획이었다. `mojo` 는 하네스에 없다(2026-11-23 항목의 "남은 일"). `--grammar-hint on`·`--args keyword` 조건은 같은 run 에 `harness.generate` 를 조건 옵션과 함께 따로 부르고(힌트가 없는 언어는 `skip`), 호출 수는 생성기가 확인 전에 출력한다.
- 행은 `results/runs_desktop.csv` 로 간다. 이어서 §6.2 의 표대로 행을 보고, 수정 루프·sanitizer 를 켤지는 사용자 결정에 따른다.

---

## 부록 A. Windows 에서 확인하지 못한 가정

한 줄에 하나: 가정 — 판정하는 확인(절).

| 가정 | 판정하는 확인 |
|---|---|
| gcc 경로 전체가 돈다: 실제 `nm -D` 의 심벌 검사, c_seq 의 스레드 심벌(`threads_in_c_seq`)·금지 심벌(`forbidden_symbol:*`) 검사 | `test_build_c.RealGccTest`(§4.1), `make devcheck` 80 pass 이고 C 줄에 `violations=` 없음(§5.1) |
| C 커널과 C 기준선이 `harness.run` 을 끝까지 지난다(Windows 에서는 gcc 가 없어 exit 7) | `lx-all` 의 C 24행, `lx-var` 의 C 50행(§6.1, §6.4), 기준선 12개(§5.2) |
| `extract` 의 C 전처리(`gcc -E`)가 동작하고, C 시그니처의 `int32_t`·`signed int` 는 `int` 와 같지 않다(`signature_mismatch`) | `test_extract.CSignatureTest`·`CKeywordTest`·`CIntSpellingTest.test_real_preprocessor`(§4.1) |
| gcc + libgomp 에서 틀린 OpenMP 변형이 README 의 분류(경쟁 상태)를 낸다(Windows 는 clang + LLVM OpenMP 로만 확인) | §5.3, `lx-var` 의 c_openmp 행(§6.4) |
| NULL 저장 변형의 상세가 `signal SIGSEGV during tiny call 0` 이다(Windows 는 ctypes 예외 문구) | §5.3, `lx-var` gemm c_openmp 1·4(§6.4) |
| Linux 에서만 붙는 `-C link-arg=-Wl,--no-undefined` 로 cdylib 가 링크되고 프롬프트·`cflags` 에 같은 값이 적힌다 | `RealCargoTest`(§4.1), §5.1 의 rust 12줄 pass, rust 행의 `cflags`·프롬프트 줄(§6.2, §6.3) |
| rustup 이 PATH 를 바꾸지 않았어도 하네스가 `~/.cargo/bin` 의 cargo·rustc 를 찾는다 | §2.1 의 `rust_tool`, §2.3 의 `env PATH=/usr/bin:/bin … --languages rust` 가 rc 0 |
| Rust 테스트도 PATH 밖의 `~/.cargo/bin` 에서 cargo·rustc 를 찾아 돈다(`requires_rust_tool`; Windows 에서는 cargo 가 PATH 에 있었다) | `~/.cargo/bin` 이 PATH 에 없는 셸에서 §4.1 에 `Rust 도구 필요` skip 이 없다 |
| Rust panic 이 Linux 에서 SIGABRT 로 끝나고 상세가 `rust_panic:<메시지>` 다 | `RealCargoTest.test_panic_aborts_the_process`(§4.1), §5.4 의 panic 커널, §9.2 |
| numba 의 omp 계층이 Linux 의 libgomp 로 올라가고 스레드 수가 T 다 | §2.1 `--probe-runtime numba`, §6.3 의 `threading_layer`·`num_threads` |
| numba 와 C+OpenMP 기준선이 libgomp 하나를 공유한다(`libgomp_paths` 의 내용, `multiple_libgomp` 가 정보용으로 남는가) | §6.3 의 `libgomp_paths`, 행의 `protocol_flags`·`protocol_ok`; `test_run … test_two_libgomp_files_are_an_informational_flag` 가 통과(§4.1) |
| taskset 과 호출 친화도(`sched_setaffinity`)가 실제로 적용된다 | `test_worker.CKernelTest`(§4.1), §6.2 의 `call_affinity`(`error` null, `call_affinity_failed` 없음) |
| Halide 의 스레드 풀(과 rayon)이 커널 호출의 CPU 집합을 물려받는다 | §6.3 의 스레드 마스크 |
| Li2018 의 Linux 출력이 Windows 에서 만든 perf_summary 픽스처와 맞는다(픽스처의 대상 문자열·스케줄 문구가 다를 수 있다) | `tests.test_perf_summary`(§4.1). 실패하면 `$PY tests/fixtures/perf_summary/make_inspect_fixtures.py --overwrite` 뒤 `git diff tests/fixtures/perf_summary` 로 차이를 보고 정한다 |
| 워커의 RLIMIT_AS(8 GiB)·`prctl`(dumpable 0)·`/sys` 의 주파수·스로틀·온도 샘플러가 동작하고, nbody 등 새 언어 커널이 그 한도 안에서 돈다 | `CKernelTest`(dumpable), 샘플의 `check.json` `limits.rlimit_as_bytes`(8589934592), 행의 `cpu_mhz_median`·`throttle_events`·`temp_c_*`(§6.2), `lx-all` 의 nbody 7행 pass |
| POSIX 프로세스 그룹 정리, `PR_SET_PDEATHSIG`, SIGTERM·SIGHUP 처리 | `test_procutil.RunIsolatedTest`·`ParentDeathTest`·`TerminateHandlersTest`(§4.1), §6.5 |
| sysfs 의 거버너·EPP·부하 값과 taskset 접두가 실제 값이다 | §2.1 `--check`, `test_protocol.LaunchTest`(§4.1) |
| ASan/UBSan 실행: libasan 선행 로드, `--sanitize` 버퍼와 가드에서 출력 배열 한 칸 뒤 쓰기를 ASan 이 잡음, 커널 로드 전의 BLAS 예열이 `LD_PRELOAD=libasan` 아래에서 돈다, `memory_error` 재분류 | `RealSanitizerRunTest`·`SanitizerEnvTest`, `lx-san` 의 정상 C 행이 detail `''`·0건(§7) |
| perf_summary 의 C 벡터화 판정이 실제 gcc `-fopt-info-vec` 로그에서 맞다(단위 테스트는 손으로 조립한 로그) | `lx-all` C 행의 `perf_summary` 에서 `vectorized.source` 가 `compiler_remark`, `status` 가 `unknown` 이 아니다(§6.1); 이상하면 그 샘플의 `build.log` 와 `tests/fixtures/perf_summary/c/*.build.log` 를 비교 |
| 실제 cppcheck·clang-tidy 로 린트 베이스라인이 돈다(gcc 의 `omp.h` 를 `-isystem` 으로) | §8 |
| venv 의 ruff·pyflakes 로 Python 린트 경로가 돈다(없으면 `requires_modules` 가 Linux 에서 테스트를 실패시킨다) | §4.1 에서 `EndToEndPythonTest`·`HarnessRunLintTest` 가 통과, §2.1 의 `lint_tools` |
| `harness/aa_check.py` 의 `main` 이 실제 장비에서 뜻 있는 값을 낸다 | §10 |
| 새 언어 더미의 성능 형태와 bench(교차 측정, 3라운드)가 실제 시간으로 돈다 | `lx-all`(§6.1); 형태별 정확성은 `devcheck kernel … --tests large`(§5.1) |
| 키워드 조건(`--args-mode keyword`)이 새 언어 커널에서 동작한다(더미는 위치 인자뿐) | §5.4 의 키워드 커널, `RealCargoTest.test_keyword_kernel_gets_the_args_module`·`test_sigs.RustKeywordScopeTest`(§4.1) |
| numba·halide·rust 의 수정 턴이 끝에서 끝까지 돈다 | §9.2(numba·rust). halide 수정 턴은 이 계획에 없다 |
| 라운드 2 이후에만 실패하는 수정 턴이 그 라운드의 기록(`bench_r<r>.json`)으로 진단된다 | 단위 테스트(`test_repair` 의 `BenchDiagnosticsTest`·`PendingRoundsTest`)만. 실제로 그런 커널이 생기면 그 샘플의 `bench_r<r>.json` 과 다음 수정 요청을 본다 |
| 이 호스트의 머신 프로필(`config/machines/<호스트>.yaml`)을 찾는다 | §2.1 의 `make probe` 뒤 `--check` 에 프로필 경고가 없고, 행의 `hostname`·`cpu_model` 이 이 장비 |
| cholesky 의 테스트 입력이 Linux 에서 기록된 값과 같다(`6c8c90f9caec7b77`; Windows 는 BLAS 차이로 달랐다) | §3 |
| Halide 의 형태별 빌드와 numba 의 임포트 컴파일이 라운드마다 `bench_timeout_s`(300 s) 안에 든다 | §6.3 의 `build_ms`; 결정은 2026-11-23 항목의 사용자 확인 3 |
