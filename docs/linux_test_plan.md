# Linux 일괄 테스트 계획 (docs/linux_test_plan.md)

이 문서는 Windows 개발 머신에서 작성한 M2b–M6 코드(새 커널 8개, 언어 `numba`·`halide_auto`·`halide_manual`·`rust`, `args=keyword`·`grammar_hint` 조건, 수정 루프, sanitizer, 린트 플러그인, 성능 결정 요약, 부트스트랩 구간과 그림)를 **Linux 측정 머신 한 세션에서** 확인하는 절차다. 저장소만 가진 사람이 위에서 아래로 그대로 실행할 수 있게 썼다. 단계마다 명령, 기대 결과, 대략의 소요 시간, 결과가 다를 때 볼 곳을 적는다.

기준 시점은 2026-11-22 의 작업 트리(아직 커밋 전)다. 명령과 옵션 이름은 그 시점의 `Makefile` 과 각 CLI 의 `--help` 로 확인했다. 그 뒤 코드가 바뀌었으면 실행 전에 `--help` 로 다시 확인한다. 기대값의 출처는 구현·감사·리뷰·수정 에이전트의 보고서와 `harness/dummy_responses/gemm/README.md` 이며, 이 문서를 쓰면서 Windows 에서 다시 돌려 본 것은 그렇다고 표시했다. 시간은 Windows 의 부하 있는 머신에서 잰 값이거나 추정이다. 측정값으로 쓰지 않는다.

---

## 0. 목적·범위·체크리스트

### 0.1 목적

- Windows 에서 확인할 수 없었던 경로를 실제 도구로 처음 돌린다: gcc + libgomp, `taskset`, CPU 거버너·EPP, ASan/UBSan, cppcheck·clang-tidy, ELF(`nm -D`), POSIX 프로세스 그룹·시그널, Rust 의 Linux 빌드, 실제 시간 측정.
- 실제 API 호출(비용이 드는 단계) 전에 정해야 할 것을 모은다(§10).
- 이 세션의 결과(통과·실패·고친 것·정한 것)를 `docs/decisions.md` 에 날짜와 함께 적는다.

### 0.2 범위

- 모든 생성은 `--provider dummy`(또는 `LLM_PROVIDER=dummy`)다. **실제 LLM API 는 이 계획 안에서 부르지 않는다.** `make real` 은 §10 의 결정이 끝난 뒤 사용자 확인을 받고 따로 한다.
- 더미 run 의 행은 `results/runs_dummy.csv` 로 간다(`paths.results_csv_dummy`). 이 파일은 git 이 추적하므로, 세션이 끝나면 `git diff --stat results/runs_dummy.csv` 로 보고 남길지 정한다. 원문·빌드 결과(`results/raw/`, `results/built/`)는 git 이 무시한다.
- 이 세션의 run id 는 `lx-` 로 시작하게 한다(`lx-gemm`, `lx-all`, `lx-cond`, `lx-kw`, `lx-repair`, `lx-san`, `lx-tool`). 기존 M1·M2a run 과 구별하기 위해서다.
- 설정을 바꿔야 하는 확인은 `config/experiment.yaml` 을 고치지 않고 `/tmp/lxplan/` 에 사본을 만들어 `--config` 로 준다(§0.4 의 `mkcfg`). 사본의 상대 경로는 저장소 루트 기준으로 풀린다(`harness.paths.resolve`).

### 0.3 Windows 에서 확인된 것 (요약)

- 단위 테스트: `python -m unittest discover -s tests` → `Ran 1320 tests`, `OK (skipped=52)` (2026-11-22, 이 문서를 쓰며 다시 실행). 건너뛴 52개는 §2.1 표다.
- `tools/devcheck.py dummies --languages numba,halide_auto,halide_manual,rust`(과제 12개): 44 pass, 4 unsupported(`sort`·`bfs` × Halide 두 언어).
- C 더미 24개는 zig 로만 통과했다: `c_seq` 는 직렬 빌드(논리만), `c_openmp` 는 `--cc zig-omp`(clang + LLVM libomp, 실제 스레드). gemm 경쟁 상태 더미는 zig-omp 에서 3번 모두 `race_condition`.
- 파이프라인(`harness.generate --provider dummy` → `harness.run` → `harness.repair` → `analysis.aggregate`·`analysis.plots`)은 Python 계열(numpy, numba, halide_*)과 rust 로만 돌았다. C 언어는 gcc 가 없어 `harness.run` 이 설계대로 exit 7 로 멈췄다. 측정값은 의미가 없다(코어 고정 없음, `platform=win32` 플래그로 `protocol_ok=0`).
- 확인하지 못한 것: C 언어의 `harness.run` 경로, `taskset`, 거버너, sanitizer, cppcheck·clang-tidy, 모든 시간. 한계는 부록 B.

### 0.4 공통 준비 (셸 하나에서 계속 쓴다)

```sh
cd <저장소 루트>
mkdir -p /tmp/lxplan
# 결과를 남긴다: 이 셸의 출력 전체를 /tmp/lxplan/session.log 에 붙인다(끝낼 때 exit)
script -a /tmp/lxplan/session.log

# 설정 사본: mkcfg <이름> <sed 식>... → /tmp/lxplan/<이름>.yaml, 바뀐 줄을 diff 로 보여 준다
mkcfg() {
  local name=$1; shift
  local args=(); for e in "$@"; do args+=(-e "$e"); done
  sed "${args[@]}" config/experiment.yaml > "/tmp/lxplan/$name.yaml"
  diff config/experiment.yaml "/tmp/lxplan/$name.yaml" || true   # 아무것도 안 나오면 sed 식이 맞지 않은 것이다
}

# 행 보기: rows <run_id> [csv] → 그 run 의 행을 한 줄씩
rows() {
  .venv/bin/python - "$1" "${2:-results/runs_dummy.csv}" <<'EOF'
import csv, sys
rid, path = sys.argv[1], sys.argv[2]
for r in csv.DictReader(open(path, encoding="utf-8")):
    if r["run_id"] == rid:
        print(r["task"], r["language"], f'args={r["args"]}', f'hint={r["grammar_hint"]}', f'fb={r["feedback"]}',
              f's{r["sample_idx"]}', f't{r["turn"]}', f'pass={r["pass"]}', r["error_stage"], r["error_type"],
              r["error_detail"][:70], f'ms={r["time_median_ms"]}', sep="  ")
EOF
}
```

### 0.5 체크리스트 (싸고 막히는 것부터)

| | 단계 | 절 | 소요(대략) | 막히면 |
|---|---|---|---|---|
| ☐ | 패키지 설치, venv, `make rust-setup` | §1.1–1.3 | 20–40분 | 이후 전부 |
| ☐ | 도구 전제 확인(nm, libasan, Halide 플러그인, Numba omp, rustc 버전) | §1.4 | 5분 | §3–§7 의 해당 부분 |
| ☐ | `make probe`, 거버너·EPP, `harness.protocol --check` | §1.5–1.6 | 5분 | 시간 측정 전부 |
| ☐ | `make tests` 와 `test_hash` 12개 대조 | §1.7 | 3–10분 | 단위 테스트 일부, 모든 run |
| ☐ | `make unittest`(1320개) | §2 | 5–10분 | 해당 영역 |
| ☐ | `make devcheck`(더미 80개), 경쟁·크래시 더미, 스레드 수 바꾸기 | §3.1–3.3 | 30–60분 | §4 |
| ☐ | 기준선·언어별 `--bench`, 손으로 쓴 픽스처 | §3.4–3.8 | 30–60분 | §9.5 |
| ☐ | 더미 파이프라인: `lx-gemm`, `lx-all`, `lx-cond`, `lx-kw` | §4.1–4.4 | 1–2시간 | §5–§8 |
| ☐ | 준비 단계의 중단 경로(exit 7)와 중단 신호 | §4.6–4.7 | 20분 | — |
| ☐ | 수정 루프와 피드백 대응 비교(`make repair-dummy`) | §5 | 30–40분 (`SKIP_BENCH=1` 이면 10분) | §8 일부 |
| ☐ | sanitizer·린트 | §6 | 30–45분 | — |
| ☐ | perf_summary 픽스처와 실제 로그 | §7 | 20–30분 | — |
| ☐ | 분석·그림 | §8 | 10분 | — |
| ☐ | 측정 규약 자체와 large 형태 재조정 | §9 | 1–2시간 | 실제 실험 |
| ☐ | 실제 API 호출 전 결정 정리 | §10 | — | `make real` |

결과가 기대와 다르면 그 자리에서 고치지 말고 출력과 파일 경로를 `/tmp/lxplan/session.log` 에 남긴 뒤 다음 단계로 간다(막히는 단계는 표의 "막히면" 열). 고친 것은 단위 테스트를 다시 돌리고 `docs/decisions.md` 에 적는다.

---

## 1. 준비

### 1.1 패키지

Ubuntu/Debian 기준 이름이다(다른 배포판은 같은 도구를 찾는다).

```sh
sudo apt install build-essential binutils file util-linux procps git \
                 linux-tools-common linux-tools-$(uname -r) \
                 cppcheck clang-tidy libomp-dev stress-ng
```

| 패키지 | 필요한 이유 | 없으면 |
|---|---|---|
| `build-essential`(gcc, make) | C 언어·C 기준선 빌드(`build.cc: gcc`), `-fopenmp` + libgomp, `-fopt-info-vec`(perf_summary), `gcc -E`(extract 의 전처리), libasan·libubsan(gcc 에 딸려 온다) | `harness.run` 이 C 언어나 C 기준선이 있는 과제에서 행을 쓰기 전에 exit 7. `make` 타깃도 못 쓴다 |
| `binutils`(nm, readelf) | 내보낸 심벌 검사(`nm -D --defined-only`), sanitizer 링크 확인 | C·Rust 빌드 결과를 검사하지 못해 `ToolchainMissing` → exit 7 |
| `util-linux`(taskset) | 코어 고정 `taskset -c <cores>` | 플래그 `taskset_missing`, 행 `taskset_used=0`, `protocol_ok=0`. `--strict-protocol` 이면 exit 4 |
| `linux-tools-*`(cpupower) | 거버너를 `performance` 로 둔다(sudo). sysfs 에 직접 써도 된다(§1.6) | 거버너를 바꿀 방법이 sysfs 쓰기뿐이다. 하네스는 거버너를 바꾸지 않고 플래그만 단다 |
| `cppcheck`, `clang-tidy` | C 린트 베이스라인(`plugins/lint/run_lint.py`) | C 행의 `lint_*` 가 NA(도구 하나만 있으면 그 도구의 범주만 채워진다). 실험은 멈추지 않는다 |
| `libomp-dev` | clang-tidy(clang)가 `#include <omp.h>` 를 해석하게 한다 | 플러그인이 gcc 의 `omp.h` 디렉터리를 `-isystem` 으로 넘겨 대신한다(§6.8). 그것도 실패하면 clang-tidy 가 빠지고 `concurrency`·`api_misuse`·`performance` 가 null |
| `file`, `procps`(pgrep), `stress-ng` | libasan 이 ELF 인지 확인, 남은 프로세스 확인, 부하 실험(§9.3) | 해당 확인만 못 한다 |
| Python 3.14 + venv | `requirements.txt` 의 버전(numpy 2.3.5, numba 0.68.1, halide 22.1.0)은 3.14 휠로 확인했다(Windows) | 다른 버전이면 프롬프트의 환경 서술과 `env_hash` 가 바뀐다 |

### 1.2 Python venv

```sh
make venv SYSPY=python3.14          # = python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt   (5–10분)
.venv/bin/python -c "import numpy, numba, halide, anthropic, openai, pandas, matplotlib; print(numpy.__version__, numba.__version__)"
.venv/bin/python -m harness.paths    # PYTHON_EXE 가 .venv/bin/python 이어야 한다
```

- 기대: `2.3.5 0.68.1`. `harness.paths` 출력의 `PYTHON_EXE` 가 `.venv/bin/python`.
- 다를 때: halide 22.1.0 의 Linux 휠이 3.14 용으로 없으면(부록 A) 설치가 실패한다. 버전을 바꾸면 `requirements.txt`·프롬프트의 버전 서술이 같이 바뀌므로 `docs/decisions.md` 에 적는다.
- `.venv` 없이 `python3 -c "import harness.paths"` 를 돌리면 `[paths] 경고` 가 나와야 한다(POSIX 에서는 경고를 유지한다).
- `ziglang` 은 `requirements.txt` 에 없다(Windows 개발용). 설치하지 않는다 — devcheck 가 Linux 에서 gcc 를 쓰는지 확인하는 것이 목적이다.

### 1.3 Rust

```sh
curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal    # 이미 있으면 건너뛴다
. "$HOME/.cargo/env"
cargo --version; rustc --version      # cargo 1.78 이상(Cargo.lock v4), rustc 1.82 이상 권장
make rust-setup                       # = cargo fetch --locked --manifest-path harness/build/rust_template/Cargo.toml (온라인 1회, 1분)
```

- 의존성은 `rayon = "=1.12.0"` 하나이고 `Cargo.lock` 이 저장소에 있다. 이후 빌드는 `cargo build --release --offline --locked` 다.
- `rustc` 1.82 미만이면 `prompts/hints/rust.md` 19행의 "`#[unsafe(no_mangle)]` is an accepted spelling" 이 사실이 아니다. 그 괄호를 지우거나 툴체인을 올린다.
- `make rust-setup` 을 빼먹으면 rust 를 처리하는 `harness.run` 이 행을 쓰기 전에 exit 7(`중단: 도구 없음 (Rust 툴체인 점검)`)이고, `tests.test_generate` 의 rust 테스트 하나가 skip 된다.

### 1.4 도구 전제 확인 (싸고, 틀리면 뒤의 단계가 막힌다; 5분)

각 줄은 보고서가 세운 가정 하나를 판정한다. 틀리면 부록 A 의 해당 줄을 본다.

```sh
# (a) 심벌이 없는 .so 에서 nm 이 rc=0 인가 (아니면 심벌 없는 커널이 signature_mismatch 가 아니라 ToolchainMissing 이 된다)
printf 'static int f(void){return 0;}\n' > /tmp/lxplan/e.c && gcc -shared -fPIC -o /tmp/lxplan/e.so /tmp/lxplan/e.c && nm -D --defined-only /tmp/lxplan/e.so; echo rc=$?
#   기대: rc=0

# (b) libasan 이 선행 로드 가능한 ELF 인가 (링커 스크립트면 모든 sanitizer 실행이 판정 불가로 기록된다)
file -L "$(gcc -print-file-name=libasan.so)"
#   기대: ELF 64-bit LSB shared object   (ASCII text 면 harness/safety.py 의 sanitizer_env 가 실제 .so 를 찾게 고친다)

# (c) sanitizer 빌드가 링크되고 libasan·libubsan 을 NEEDED 로 붙이는가
#     harness/build/c.py 는 -fsanitize= 가 있으면 ldflags 에서 -Wl,--no-undefined 를 뺀다(미정의 심벌 검사는 일반 빌드가 이미 했다).
#     아래는 그 명령과 같다(build.base_flags + cflags + build.sanitize_flags + ldflags − --no-undefined)
printf 'void gemm(const float* A, const float* B, float* C, int M, int N, int K) { C[0] = A[0] * B[0]; }\n' > /tmp/lxplan/k.c
gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native -fsanitize=address,undefined \
    -fno-omit-frame-pointer -g -o /tmp/lxplan/kernel_san.so /tmp/lxplan/k.c -lm \
  && readelf -d /tmp/lxplan/kernel_san.so | grep -E 'libasan|libubsan'
#   기대: 링크 성공, NEEDED 에 libasan 과 libubsan. (참고: 같은 명령에 -Wl,--no-undefined 를 더해도 링크되는지 본다. 실패해도 하네스에는 영향이 없다)

# (d) -fopenmp 없이 빌드할 때의 경고 문구 (perf_summary 의 "무시된 pragma" 판정 근거)
gcc -std=gnu11 -Wall -O3 -fsyntax-only tests/fixtures/perf_summary/c/jacobi2d_omp.c 2>&1 | grep -n ignoring
#   기대: ignoring '#pragma omp parallel' 과 '#pragma omp simd' (정규식 ignoring\s+\W{0,3}#pragma\s+omp\b)

# (e) target_clones 커널의 심벌 종류 (ifunc → nm 에 'i')
printf '__attribute__((target_clones("avx2","default"))) void gemm(const float* A, const float* B, float* C, int M, int N, int K){ for (int i=0;i<M*N;i++) C[i]=A[0]*B[0]; }\n' > /tmp/lxplan/tc.c
gcc -O3 -march=native -fPIC -shared -o /tmp/lxplan/tc.so /tmp/lxplan/tc.c && nm -D --defined-only /tmp/lxplan/tc.so | grep -w gemm
#   기대: '... i gemm'

# (f) Halide 오토스케줄러 플러그인이 Linux 휠에서 .so 로 찾아지는가
.venv/bin/python -c "import halide as hl; from pathlib import Path; from harness.kernels.halide_pipe import find_autoscheduler_plugin as f; print(f(Path(hl.__file__).resolve().parent, 'Mullapudi2016'))"
#   기대: .../libautoschedule_mullapudi2016.so 의 절대 경로 (None 이면 harness/kernels/halide_pipe.py 의 PLUGIN_SUFFIXES·glob)

# (g) Numba 의 omp 스레딩 계층과 스레드 수, SciPy 부재
NUMBA_THREADING_LAYER=omp NUMBA_NUM_THREADS=4 .venv/bin/python - <<'EOF'
import numba, numpy as np
@numba.njit("float64(float32[::1])", parallel=True)
def f(a):
    s = 0.0
    for i in numba.prange(a.size):
        s += a[i]
    return s
f(np.ones(64, np.float32))
print(numba.threading_layer(), numba.get_num_threads())
EOF
#   기대: omp 4   (omp 를 못 올리면 config/experiment.yaml 의 NUMBA_THREADING_LAYER 를 바꾸고 결정으로 적는다; 프롬프트는 설정을 따른다)
.venv/bin/python -c "import scipy" 2>&1 | tail -1      # 기대: ModuleNotFoundError (prompts/lang/numba.md 가 SciPy 없음을 말한다)

# (h) 워커 환경에 PYTHONIOENCODING 이 들어가지 않는가 (POSIX)
.venv/bin/python -c "from harness.config import load_config; from harness.protocol import launch_env; print('PYTHONIOENCODING' in launch_env(load_config(), 4, openmp=True))"
#   기대: False
```

### 1.5 `config/machine.yaml`

```sh
make probe                                          # = .venv/bin/python -m harness.protocol --probe --write
git diff --stat config/machine.yaml
.venv/bin/python -m harness.protocol --fingerprint | head -40
```

- `machine.yaml` 의 `governor`·`epp` 는 프로브한 순간의 값이다(정적 검사 `--check` 는 sysfs 를 직접 읽는다). 측정 상태를 기록하려면 §1.6 의 거버너 설정을 먼저 하고 `make probe` 를 돌린다.
- 지금 `config/machine.yaml` 은 M1·M2a 를 잰 노트북(i7-1260P, Ubuntu 26.04, gcc 15.2)의 프로브다. 측정 머신이 그 노트북이 아니면 이 파일은 통째로 바뀐다. 그 머신에서 `harness.protocol --check` 는 다시 프로브할 때까지 hostname 이 다르다고 경고한다.
- 기대: `machine.yaml` 에 `cargo`, `rustc`, `numba_version`, `halide_version`, `halide_autoscheduler`, `halide_autoscheduler_loadable`, `lint_tools`(cppcheck·clang-tidy 버전)가 새로 생긴다. `--fingerprint` 의 지문에 `rustc`, `numba_version`, `halide_version`, `lint_tools` 가 있고, `env_hash` 는 M1·M2a 행과 다르다(예상된 차이).
- 같은 머신에 rustup 툴체인이 둘 이상이면 `RUSTUP_TOOLCHAIN=<그 툴체인> .venv/bin/python -m harness.protocol --probe | grep -E 'cargo|rustc'` 가 그 툴체인의 버전을 보여야 한다(rust 행의 `compiler_version` 과 같은 문자열). rustup 을 `RUSTUP_HOME=/opt/rust` 같은 곳에 두고 `~/.rustup` 이 없으면 `RUSTUP_HOME=/opt/rust .venv/bin/python -m harness.protocol --probe` 의 `cargo`·`rustc` 가 null 이 아니어야 한다.
- 새 머신이면 `measurement.cores`·`single_core`·`threads` 가 그 CPU 에 맞는지 §9.2 에서 본다. 이 테스트 세션은 지금 값(`"0,2,4,6"`, `"2"`, 4)으로 진행해도 된다. 그 CPU 들이 존재하기만 하면 된다(`nproc` ≥ 8).

### 1.6 CPU 거버너·EPP

```sh
sudo cpupower frequency-set -g performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference    # intel_pstate HWP 일 때
.venv/bin/python -m harness.protocol --check; echo rc=$?
```

- 기대: `static.flags` 와 `dynamic.flags` 가 비어 있고 rc=0. `platform=` 플래그가 없다. `load_1m`, `pinned_busy_pct` 에 실제 값이 있다.
- 재부팅하면 `powersave`/`power` 로 돌아온다. 세션을 시작할 때마다 다시 설정하고 `--check` 로 본다(`docs/protocol.md` §13).
- 거버너를 못 바꾸면 테스트는 계속할 수 있다. 행에 `governor_ok=0` 이 붙을 뿐이다(`--strict-protocol` 이면 exit 4). 시간 비교(§9)는 그 상태에서 하지 않는다.

### 1.7 테스트 입력 (과제 12개)

```sh
time make tests                                     # tasks/*/gen_tests.py 를 차례로 (3–10분, 약 1.9 GB)
grep -H '"test_hash"' tasks/*/tests/manifest.json
```

과제마다 `tasks/<task>/tests/` 에 `tiny.npz`, `rect.npz`, `small.npz`, `large.npz`, `manifest.json` 이 생긴다. `test_hash` 는 시드·형태·입력 바이트만으로 정의되므로(`harness/testgen.py`) 플랫폼과 무관하게 아래 값과 같아야 한다. 값은 Windows 의 `manifest.json` 과 과제 보고서가 일치한 것이다.

| 과제 | `test_hash` | npz 합계(Windows, MiB) | 비고 |
|---|---|---|---|
| gemm | `2077183b4edc9ec8` | 18 | M1 이후 불변 |
| jacobi2d | `2817c6d5544625d8` | 196 | M2a |
| spmv_csr | `fcab7d32e352a1eb` | 139 | M2a |
| dot | `0ed48e0b583e1ac6` | 265 | M2a |
| histogram | `467726b382000cdd` | 261 | **2026-11-22 변경**(이전 `a380ff26d6f40c46`) |
| prefix_sum | `2bb751afd41770a3` | 521 | 546.1 MB, 설계 상한(과제당 약 550 MB) 바로 아래 |
| nbody | `c15bade031489ac4` | 0.4 | |
| sort | `1e13ecc4e2a44a13` | 130 | |
| cholesky | `5080269a9d095f90` | 49 | |
| floyd_warshall | `845ae92a4a402949` | 13 | |
| bfs | `846392026088677c` | 163 | 보고서 값 170.4 MB |
| fft | `7f78472cc4a65f08` | 98 | 보고서 값 102.3 MB. `out_ref` 의 마지막 비트는 NumPy 빌드에 따라 다를 수 있으나 `test_hash` 에 들어가지 않는다 |

- 과제 하나만 다시 만들 때: `.venv/bin/python tools/devcheck.py gen --task bfs` → `bfs/gen => PASS test_hash=846392026088677c seed=20261105 …`.
- 다를 때: NumPy 버전(`numpy==2.3.5`)부터 본다. 다른 값이 나오면 M2a 이전 행과 입력이 다르다는 뜻이므로 실험을 진행하지 말고 원인을 찾는다. histogram 만 이전 값(`a380ff…`)이면 생성기가 낡은 체크아웃이다.

### 1.8 디스크

- 테스트 입력 약 1.9 GB, `.venv` 약 1 GB(Windows `.venv-win` 1.1 GB 기준), Rust 레지스트리 수 MB.
- run 산출물(`results/built/<run>/…`)은 샘플마다 커널·로그·대응 기준선 문서이고, rust 는 run 마다 `_cargo_target` 을 공유한다(LTO 빌드라 수백 MB 가 될 수 있다 — 추정). 여유 10 GB 이상을 권한다.

---

## 2. 단위 테스트 (5–10분)

`make tests` 뒤에 돌린다(npz 가 없으면 일부 테스트가 skip 된다).

```sh
time make unittest 2>&1 | tail -5         # = .venv/bin/python -m unittest discover -s tests
```

### 2.1 기대 개수

- 기대: `Ran 1320 tests`, `OK`, skipped 는 7개 이하.
- Linux 에서도 건너뛰는 것(7개): `test_devcheck.ZigBuildTest` 1개와 `test_devcheck.ZigOmpEndToEndTest` 5개(ziglang 패키지·LLVM OpenMP 런타임이 필요한 Windows 개발용), `test_taskspec.WorkerPortabilityTest.test_platform_without_resource_reports_unsupported_limits` 1개(`resource` 모듈이 없는 플랫폼 전용).
- 상황에 따라 skip 되는 것(이유가 출력된다): `test_worker.CKernelTest` 의 친화도 테스트("서로 다른 물리 코어의 CPU 2개 필요", "친화도가 좁아지지 않아 … 관측할 수 없다"), `test_build_c` 의 libasan 테스트, `test_generate` 의 rust 테스트(`make rust-setup` 안 함), `test_worker.RustAbiTest`(`rustc` 가 PATH 에 없음), npz 가 필요한 테스트(`make tests` 안 함), `test_schema` 의 git 테스트.

Windows 에서 건너뛰어 **Linux 에서 처음 도는** 52개:

| 테스트 클래스 | 개수 | 필요한 것 |
|---|---|---|
| `test_build_c.RealGccTest` | 4 | gcc, nm, libasan. 기본 인자의 명령·파일·심벌, `args_header`+`extra_flags`+`artifact_name`, `-fsanitize=` 에서 `-Wl,--no-undefined` 를 빼는 것, `test_target_clones_ifunc_is_an_exported_kernel`(x86-64) |
| `test_extract.CSignatureTest` / `CIntSpellingTest` / `CKeywordTest` | 3 / 1 / 1 | `gcc -E`: messy 소스, `preprocess_failed` 폴백, 속성 + typedef, keyword 커널이 `kernel_args.h` 로 전처리됨, `test_real_preprocessor`(`<stdint.h>` 의 `int32_t` → `signature_ok=True`, `signature_exact=False`) |
| `test_procutil.RunIsolatedTest` / `ParentDeathTest` / `TerminateHandlersTest` | 8 / 3 / 1 | POSIX 프로세스 그룹(`killpg`), `PR_SET_PDEATHSIG`, SIGHUP, bash, pgrep |
| `test_protocol.LaunchTest` | 1 | taskset(`test_launch_prefix`) |
| `test_safety.RealSanitizerRunTest` / `SanitizerEnvTest` | 4 / 1 | gcc + libasan·libubsan 선행 로드(§6.2) |
| `test_worker.CKernelTest` | 9 | gcc ELF, 시그널 종료 코드, `OMP_PROC_BIND` 친화도 |
| `test_llm_client.AnthropicStreamMockTest` / `ClassifyErrorTest` / `OpenAICompatMockTest` | 10 / 4 / 2 | `.venv` 의 anthropic·httpx2·openai SDK(Windows 개발 venv 에는 없었다) |

### 2.2 영역별 명령과 개수 (실패를 좁힐 때)

개수는 2026-11-22 Windows 실행의 모듈별 개수다(Linux 에서도 같아야 한다).

| 영역 | 명령 | 개수 |
|---|---|---|
| 기반·워커 | `.venv/bin/python -m unittest tests.test_procutil tests.test_worker tests.test_kernels tests.test_taskspec tests.test_testgen tests.test_sigs tests.test_devcheck tests.test_classify -v` | 44 + 69 + 24 + 28 + 19 + 43 + 70 + 47 |
| 추출·빌드 | `.venv/bin/python -m unittest tests.test_extract tests.test_build_c tests.test_build_numba tests.test_build_halide tests.test_build_rust -v` | 78 + 26 + 24 + 21 + 50 |
| 설정·생성·수정 루프 | `.venv/bin/python -m unittest tests.test_config tests.test_generate tests.test_llm_client tests.test_repair -v` | 223 |
| 실행·스키마·규약 | `.venv/bin/python -m unittest tests.test_run tests.test_schema tests.test_protocol -v` | 194 |
| perf_summary | `.venv/bin/python -m unittest tests.test_perf_summary -v` | 111 |
| 안전성·린트 | `.venv/bin/python -m unittest tests.test_safety tests.test_lint_plugin -v` | 73 + 55 |
| 분석 | `.venv/bin/python -W error -m unittest tests.test_stats tests.test_aggregate tests.test_plots -v` | 121 |

볼 것:

- `tests.test_worker` 의 `HalideFailureTest.test_bad_autoscheduler_parameters_are_a_harness_error_not_a_kernel_error` 가 통과해야 한다. 틀린 오토스케줄러 매개변수가 예외(rc=3)로 끝나는지 SIGABRT 로 끝나는지 어느 쪽이든 `worker_error` 다. 어느 쪽이었는지 `docs/decisions.md` 에 적는다.
- `tests.test_worker.CAbiTest` 가 테스트 도우미 `build_shared_library` 의 gcc 분기로 돈다(Windows 에서는 zig 분기였다).
- `tests.test_worker` 의 Numba·Halide 테스트가 워커 기본값 `RLIMIT_AS` 8 GiB 아래에서 `NUMBA_THREADING_LAYER=omp` 로 처음 돈다(JIT·스레드 풀이 주소 공간 한도 안에서 뜨는지).
- `tests.test_build_rust` 의 `RealCargoTest`(10개)와 `TargetLockTest`(3개): `libkernel.so` → `kernel.so` 복사, `extra.symbol_check == "nm"`, `#[no_mangle]` 없는 커널이 `symbol dot not exported`, 패닉이 SIGABRT(returncode −6), 최소 환경(`launch_env_minimal` + `CARGO_HOME` 등)만으로 cargo·rustc·링커가 도는지, 빈 `CARGO_HOME` 이 `ToolchainMissing` 인지, `fcntl.flock` 잠금(한 프로세스에서 잠금 파일을 두 번 열면 충돌), `test_concurrent_builds_of_different_kernels_get_their_own_library`, `test_probe_builds_the_placeholder_and_leaves_the_dependencies`(뒤 샘플 로그에 `Compiling rayon` 없음), `test_probe_reports_a_missing_dependency_cache`.
- `tests.test_build_numba.RealNumbaTest`: Linux 의 Numba 에서도 명시 서명이면 임포트 때 컴파일된다.
- `tests.test_perf_summary` 의 `test_default_schedule_methods_equal_the_build_module` 은 skip 이 아니라 pass.
- `tests.test_protocol.PythonToolchainLiveTest` 는 numba 나 halide 가 venv 에 없으면 일부러 실패한다.
- `tests.test_generate.PromptHashTest`(4개): LF 체크아웃에서 프롬프트 해시가 고정값과 같다(Windows 작업 트리는 CRLF 였다). `NumbaEnvironmentClaimTest`(3개)는 SciPy 가 임포트되면 실패한다.
- `tests.test_llm_client.OpenAICompatMockTest` 는 SDK 없이 썼다. 설치된 openai SDK 가 `openai.DefaultHttpxClient(transport=httpx.MockTransport(handler))` 를 받지 않으면 테스트 배선만 고친다(요청 조립은 `RequestAssemblyTest` 가 가짜 SDK 로 덮는다).
- `tests.test_lint_plugin.EndToEndPythonTest`: `ruff>=0.17,<1` 이 `--select` 목록의 코드를 전부 알아야 한다. 모르는 코드가 있으면 ruff 가 rc 2 로 끝나 `tool` 이 `baseline:pyflakes` 가 되고 이 테스트가 실패한다. 그러면 `plugins/lint/rules.yaml` 의 선택 목록을 고친다.
- `tests.test_safety.RunLintTest` 의 시간 초과 테스트 뒤 `pgrep -af fake_plugin` 이 비어 있어야 한다(POSIX 의 새 세션·프로세스 그룹 정리).

### 2.3 수동 확인 (각 1분)

```sh
# procutil: 시간 초과 시 그룹 전체 정리
.venv/bin/python -m harness.procutil --timeout 2 -- bash -c "sleep 300 & sleep 300"; pgrep -af 'sleep 300'
#   기대: JSON 의 rc=-9, signal_name=SIGKILL, pgrep 출력 없음

# procutil: UTF-8 출력이 파이프에서도 그대로
.venv/bin/python -m harness.procutil --timeout 5 -- bash -c 'echo 한글' | cat        # 기대: rc 0, stdout_tail 이 "한글\n"
.venv/bin/python -m harness.sigs --task gemm --language c_seq --backend c --args-mode keyword | head -20   # 구조체 안내 + kernel_args.h
.venv/bin/python -m harness.paths --json | cat

# 워커의 RLIMIT_AS 생략과 보고
.venv/bin/python -m harness.worker --backend python --artifact tasks/gemm/baseline/baseline.py --mode import --out /tmp/lxplan/w0.json --max-mem-gb 0 \
  && .venv/bin/python -c "import json;print(json.load(open('/tmp/lxplan/w0.json'))['limits'])"
#   기대: {'rlimit_as_bytes': None, 'rlimit_as_requested_bytes': None, 'rlimit_as_hard_bytes': None, 'disabled': True, 'dumpable': 0}
#   --max-mem-gb 를 빼면 rlimit_as_bytes == 8589934592 이고 disabled 키가 없다
bash -c 'ulimit -Sv 8388608; .venv/bin/python -c "from harness import worker; print(worker.apply_memory_limit(0))"'
#   기대: {'rlimit_as_bytes': 8589934592, 'rlimit_as_requested_bytes': None, 'rlimit_as_hard_bytes': None, 'disabled': True}

# SDK 가 빠진 환경에서 test_llm_client 가 skip 이 아니라 실패하는가
.venv/bin/python -c "import sys, unittest; sys.modules['anthropic'] = None; import tests.test_llm_client as m; r = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(m)); sys.exit(0 if len(r.errors) == 2 and not r.skipped else 1)"; echo rc=$?
#   기대: rc=0 (errors=2, skipped=0)

# Numba 키워드 전용 정의
.venv/bin/python - <<'EOF'
import numba, numpy as np
@numba.njit("void(float32[::1], float32[::1], float32[::1], int64)")
def dot(*, x, y, out, n):
    s = 0.0
    for i in range(n):
        s += x[i] * y[i]
    out[0] = s
x = np.ones(8, np.float32); o = np.zeros(1, np.float32)
dot(x=x, y=x, out=o, n=8); print(o)
EOF
#   기대: [8.]  (Windows numba 0.68.1 에서는 동작했다)
```

다를 때: 실패한 테스트의 클래스로 영역을 찾고 그 영역 모듈만 `-v` 로 다시 돌린다. gcc·taskset·sanitizer 관련 실패는 §1.4 의 해당 확인부터 본다.

---

## 3. 개발용 점검 도구 (`tools/devcheck.py`)

devcheck 는 측정 도구가 아니다: 추출 → 빌드 → `harness.worker` check 를 한 번 돌리고, `--bench` 는 large 형태 1회 호출 시간(참고값)이다. 코어 고정·규약 검사는 하지 않는다. Linux 에 설정의 gcc 가 있으면 C 는 자동으로(`--cc auto`) `harness.build.c.build_c` 와 **설정의 실제 플래그**를 쓰고, 판정 줄에 `toolchain=` 표시가 붙지 않는다. 종료 코드는 0 통과, 1 실패, 2 사용법·도구 문제다.

주의: devcheck 는 워커 환경의 `OMP_NUM_THREADS` 를 그 언어의 설정값(`measurement.threads`, `c_seq` 는 1)으로 덮어쓴다. 셸에서 `OMP_NUM_THREADS=3 python tools/devcheck.py …` 로 준 값은 gcc 경로에서 무시된다(`DEVCHECK_OMP_THREADS` 는 zig-omp 전용). 스레드 수를 바꾸려면 §3.3 처럼 설정 사본을 쓴다.

### 3.1 더미 80개 (20–40분, 추정)

```sh
time make devcheck 2>&1 | tee /tmp/lxplan/devcheck_all.txt | tail -3     # = .venv/bin/python tools/devcheck.py dummies
```

- 기대 마지막 줄: `summary: 80 pass, 0 fail, 0 missing, 0 problem, 4 unsupported`. 어느 행에도 `toolchain=` 표시가 없다. 80 = 과제 12 × 언어 7 − 4(`sort`·`bfs` × `halide_auto`·`halide_manual`).
- 기대 stderr: C 행에 `build: gcc -O3 -march=native [-fopenmp]` 류의 메모.
- 언어·과제별로 나눠 볼 때(보고서들이 각자 확인을 요청한 묶음):
  - `make devcheck LANGUAGES=numpy,c_seq,c_openmp TASKS=gemm,jacobi2d,spmv_csr,dot` → 12 pass.
  - `make devcheck LANGUAGES=halide_auto,halide_manual TASKS=gemm,jacobi2d,spmv_csr,dot` → `8 pass, 0 fail, 0 missing, 0 problem, 0 unsupported`. `TASKS=histogram,prefix_sum,nbody` → 6 pass. `TASKS=cholesky,floyd_warshall,fft` → 6 pass. Mullapudi2016 플러그인이 `.so` 로, halide 패키지 디렉터리에서 절대 경로로 로드됨을 함께 확인한다.
  - `make devcheck LANGUAGES=numba` → 12 pass. omp 계층에는 libgomp(또는 Intel OpenMP)가 있어야 한다. 없으면 첫 병렬 호출에서 threading layer 오류가 난다.
  - `make devcheck LANGUAGES=rust` → 12 pass. `kernel.dll` 이 아니라 `libkernel.so` 를 만들고 설정의 `rustflags`(`-C target-cpu=native`)를 쓴다.
  - `make devcheck LANGUAGES=c_seq TASKS=cholesky,floyd_warshall` → pass, `contract_violation` 없음(c_seq 의 스레드 심벌 검사 `GOMP_`·`omp_`·`pthread_create` 가 아무것도 찾지 않는다).
- 다를 때: 실패 행의 detail(`<stage>/<classification>: …`)로 판단한다. `fail` 이 C 행에만 있으면 gcc 고유 동작(경고·`aligned_alloc`·`qsort` 등), Halide 행에만 있으면 플러그인 로드나 스케줄 차이, rust 행에만 있으면 `make rust-setup` 과 `toolchain_env` 를 본다. `--keep /tmp/lxplan/keep` 으로 빌드 디렉터리를 남겨 `build.log`·`check.json` 을 읽는다.

### 3.2 Windows 에서 볼 수 없던 판정 (2분)

```sh
.venv/bin/python tools/devcheck.py kernel --task gemm --language c_openmp --file harness/dummy_responses/gemm/c_openmp/2.md; echo rc=$?
#   기대: => FAIL stage=check classification=race_condition …, rc=1  (Windows zig 직렬 빌드에서는 PASS … toolchain=zig-serial 였다)
.venv/bin/python tools/devcheck.py kernel --task gemm --language c_openmp --file harness/dummy_responses/gemm/c_openmp/1.md; echo rc=$?
#   기대: => FAIL … classification=runtime_crash detail 에 'worker died (signal SIGSEGV) during tiny call 0', rc=1 (Windows 는 OSError 접근 위반)

# halide_manual 의 스케줄 오류가 compile_error 로 판정되는가 (harness.run 의 행과 같은 stage·class)
mkdir -p /tmp/lxplan/hl && awk '/^```python/{f=1;next}/^```/{f=0}f' harness/dummy_responses/dot/halide_manual/0.md > /tmp/lxplan/hl/dot_manual.py
#   위 파일의 스케줄 부분에 모르는 Var 로 out.vectorize(zz, 4) 한 줄을 넣는다(zz = hl.Var("zz") 를 선언만 하고 정의에는 쓰지 않는다)
.venv/bin/python tools/devcheck.py kernel --task dot --language halide_manual --file /tmp/lxplan/hl/dot_manual.py
#   기대: => FAIL stage=build classification=compile_error detail=HalideError: …
```

### 3.3 스레드 수를 바꿔 본 OpenMP 커널·기준선 (15–30분)

경쟁 상태는 팀 크기에 따라 드러나므로, 팀 크기 1·3(2의 거듭제곱이 아님)·전체 코어 수로 돌린다. 설정 로더는 `threads` 가 `cores` 의 CPU 수보다 크면 거부하므로 전체 코어 수에서는 `cores` 도 넓힌다.

```sh
mkcfg t1 's/^  threads: 4 /  threads: 1 /'
mkcfg t3 's/^  threads: 4 /  threads: 3 /'
mkcfg tmax "s/^  threads: 4 /  threads: $(nproc) /" "s/^  cores: \"0,2,4,6\"/  cores: \"0-$(($(nproc)-1))\"/"

for c in t1 t3 tmax; do
  echo "== $c"
  .venv/bin/python tools/devcheck.py dummies --config /tmp/lxplan/$c.yaml --languages c_openmp \
      --tasks histogram,prefix_sum,nbody,sort,cholesky,floyd_warshall,bfs,fft | tail -1
  for t in histogram nbody floyd_warshall jacobi2d spmv_csr; do       # C+OpenMP 기준선 5개
    .venv/bin/python tools/devcheck.py baseline --config /tmp/lxplan/$c.yaml --task $t | tail -1
  done
done
#   기대: 매번 summary: 8 pass, 0 fail …; 기준선 5개 모두 => PASS
for i in $(seq 20); do
  .venv/bin/python tools/devcheck.py dummies --tasks bfs,fft --languages c_openmp > /tmp/lxplan/bfsfft_$i.txt 2>&1 \
    || { echo "FAIL at $i"; tail -5 /tmp/lxplan/bfsfft_$i.txt; break; }
done; tail -1 /tmp/lxplan/bfsfft_20.txt
#   기대: 20번 모두 통과(마지막 줄 summary: 2 pass …). bfs 의 atomic read/write + reduction 이 gcc 에서 경고 없이 받아들여지고
#   반복 실행에서 nondeterministic/race_condition 이 없다
.venv/bin/python tools/devcheck.py kernel --config /tmp/lxplan/tmax.yaml --task bfs --language c_openmp \
    --file harness/dummy_responses/bfs/c_openmp/0.md --tests tiny,rect,small,large
.venv/bin/python tools/devcheck.py kernel --config /tmp/lxplan/tmax.yaml --task fft --language c_openmp \
    --file harness/dummy_responses/fft/c_openmp/0.md --tests tiny,rect,small,large
#   기대: 모든 형태 PASS. bfs 는 max_abs_err 0, fft 의 large tol_ratio 는 약 4.7e-4
```

과제별로 확인하는 구조(보고서 T1–T4):

| 과제 | 구조 | 특히 볼 것 |
|---|---|---|
| histogram (더미·기준선) | 블록 t 마다 자기 사본 `local[t*stride…]`, 병합은 직렬 | glibc 에서 `aligned_alloc`(크기 `nt*stride*4` 는 64 의 배수) |
| prefix_sum | 두 번의 `parallel for`(블록 합 → 직렬 오프셋 스캔 → 블록 내 스캔) | 스레드 수 > n (tiny n=7) |
| nbody | i 에 대한 `parallel for` + `simd reduction(+:ax,ay,az)` | 기준선의 SoA 복사가 병렬 영역 전에 직렬로 채워진다 |
| sort | 청크별 `qsort` → 라운드마다 병합 `parallel for`, 버퍼 교환은 영역 밖 | 팀 크기 3(홀수 그룹), 스레드 > 9(tiny 키 9개). glibc `qsort` 는 스레드 안전 |
| floyd_warshall | 순차 k 루프 안의 행 `parallel for`(기준선은 j 에 `simd` 추가) | 스레드 수와 무관하게 출력이 비트 단위로 같다 |
| cholesky | 블록 루프 안의 행 `parallel for schedule(static, 8)` | 스레드 수와 무관하게 비트 단위로 같다 |
| bfs | 레벨 동기 sweep, `atomic read`/`atomic write`, `reduction(+:found)` | 반복 실행에서 `nondeterministic` 없음 |
| fft | `parallel for` 네 개(트위들 표, 비트 반전 복사, 블록별·k 별 나비), 단계 루프는 순차 | 1스레드 결과와 비트 단위로 같다 |

다를 때: 스레드 수에 따라 `race_condition`·`wrong_output` 이 나오면 그 커널의 공유 쓰기를 본다. 더미(`harness/dummy_responses/<task>/c_openmp/0.md`)가 틀렸으면 더미를 고치고, 기준선이 틀렸으면 `tasks/<task>/baseline/baseline.c` 를 고친다(기준선이 바뀌면 그 과제의 대응 기준선 값이 바뀐다).

TSan(선택): 보고서 T1·T2·T4 가 제안했다. 설계는 TSan 을 쓰지 않는다(libgomp 가 계측되지 않아 거짓 양성; `docs/decisions.md` 2026-11-20 `memory_error` 규칙). 손으로 해 본다면 `gcc … -fopenmp -fsanitize=thread` 로 빌드한 결과를 참고만 하고 판정에 쓰지 않는다.

### 3.4 기준선 (점검 5분, `--bench` 15–30분)

```sh
make devcheck-baselines                                          # 과제마다 baseline 점검(정확성 형태만)
for t in gemm jacobi2d spmv_csr dot histogram prefix_sum nbody sort cholesky floyd_warshall bfs fft; do
  .venv/bin/python tools/devcheck.py baseline --task $t --bench | tail -1
done | tee /tmp/lxplan/baseline_bench.txt
#   기대: 12줄 모두 => PASS … bench_large_ms=<값>
```

기준선의 종류(`tasks/<task>/signatures.yaml` 의 `baseline`): C+OpenMP 5개(`jacobi2d`, `spmv_csr`, `histogram`, `nbody`, `floyd_warshall`), 순차 C 1개(`bfs`), NumPy 6개(`gemm`, `dot`, `cholesky`, `fft`, `prefix_sum`, `sort`). NumPy 기준선 중 `gemm`·`dot`·`cholesky` 는 BLAS 게이트(`gate_blas: true`; OpenBLAS 가 아니면 exit 4)가 있고, `gemm` 만 `baseline_min_gflops` 게이트(exit 6)가 있다.

`bench_large_ms` 의 뜻: 코어 고정 없는 1회 호출이다. 측정값이 아니라 large 형태가 설계 범위 안인지 보는 참고값이다. 설계 목표는 **순차 C 기준 0.05–3 s, 과제당 npz 약 550 MB 이하**(`docs/harness_design_v3.md` §4)다. 순차 C 기준이므로 각 과제의 `c_seq` 더미 시간을 함께 잰다:

```sh
for t in gemm jacobi2d spmv_csr dot histogram prefix_sum nbody sort cholesky floyd_warshall bfs fft; do
  .venv/bin/python tools/devcheck.py kernel --task $t --language c_seq --file harness/dummy_responses/$t/c_seq/0.md --bench | tail -1
done | tee /tmp/lxplan/cseq_bench.txt
```

Windows 참고값(부하 있는 머신; Linux 값과 비교만 한다): bfs 순차 기준선 1.27–1.31 s(유휴), 3.6–4.2 s(부하); fft 기준선 190 ms(따뜻함)·470 ms(첫 호출); cholesky 기준선 443 ms 가 c_seq 더미 291 ms 보다 느렸다(부하 탓으로 추정 — Linux 에서는 OpenBLAS `spotrf` 기준선이 훨씬 빨라야 한다); nbody 순차 약 155 ms; histogram c_seq 약 0.1 s, numpy 약 0.7 s; prefix_sum 약 0.15 s. 이 값들로 large 형태를 다시 정하는 절차는 §9.5 이다.

### 3.5 새 언어의 `--bench` (20–30분)

```sh
for t in gemm jacobi2d spmv_csr dot histogram prefix_sum nbody sort cholesky floyd_warshall bfs fft; do
  for l in numba rust halide_auto halide_manual; do
    f=harness/dummy_responses/$t/$l/0.md; [ -f "$f" ] || continue
    .venv/bin/python tools/devcheck.py kernel --task $t --language $l --file "$f" --bench | tail -1
  done
done | tee /tmp/lxplan/newlang_bench.txt
#   기대: 44줄 모두 => PASS … bench_large_ms=<값>
```

| 언어 | Windows 참고값(1회 호출, large) | 볼 것 |
|---|---|---|
| numba | histogram 약 47, prefix_sum 약 100, nbody 약 66, sort 약 435, cholesky 약 500, floyd_warshall 약 140, bfs 약 1500, fft 약 440 ms | 모두 1 s 안팎. 프로세스의 첫 JIT 1.7–3.7 s, 이후 0.2–1.3 s |
| rust | Tier 1 약 35–540 ms; histogram 34, prefix_sum 56, nbody 54, sort 173; cholesky 210, floyd_warshall 42, bfs 311, fft 269 ms | 한 호출이 10 s 보다 훨씬 작다 |
| halide_* | spmv_csr/halide_auto 약 0.5 s, jacobi2d/halide_manual 약 0.35 s, prefix_sum/halide_auto 약 1.1 s(가장 느림), Tier 3 0.2–1.0 s | jacobi2d/halide_auto 의 성능 형태 오토스케줄+JIT 4.4 s, fft/halide_manual 의 JIT 약 12 s(n=2²²). 몇 자릿수 늘면 그 조합을 다시 본다 |

- 빌드(JIT) 시간이 `measurement.check_timeout_s`(180 s)와 bench 예산(300 s)의 여유 안인지 본다.
- Halide 의 형태별 정확성: `.venv/bin/python tools/devcheck.py kernel --task <nbody|histogram|prefix_sum> --language <halide_auto|halide_manual> --file harness/dummy_responses/<task>/<language>/0.md --tests large` → PASS. Windows 에서 nbody 의 tol_ratio 약 0.035, histogram·prefix_sum 의 max_abs_err 0.
- 테스트 픽스처 커널의 bench: `.venv/bin/python tools/devcheck.py kernel --task gemm --language halide_auto --file tests/fixtures/kernels/halide/gemm.py --bench`, `… --task jacobi2d --language halide_manual --file tests/fixtures/kernels/halide/jacobi2d_manual.py --bench`, `… --task gemm --language numba --file tests/fixtures/kernels/numba/gemm.py --bench` → 모두 PASS.

### 3.6 손으로 쓴 커널 (10–20분)

```sh
# perf_summary 의 C 픽스처를 실제 플래그·OpenMP 로 (zig 에서는 pragma 가 무시됐다)
.venv/bin/python tools/devcheck.py kernel --task gemm --language c_seq    --file tests/fixtures/perf_summary/c/gemm_blis_avx2.c
.venv/bin/python tools/devcheck.py kernel --task gemm --language c_openmp --file tests/fixtures/perf_summary/c/gemm_blocked_omp_collapse.c
.venv/bin/python tools/devcheck.py kernel --task gemm --language c_seq    --file tests/fixtures/perf_summary/c/gemm_transposed_b.c
.venv/bin/python tools/devcheck.py kernel --task histogram --language c_openmp --file tests/fixtures/perf_summary/c/histogram_omp_private.c
# rayon 을 쓰는 Rust 픽스처 (Windows 에서는 구문 검사만)
for f in gemm_rayon gemm_blocked; do .venv/bin/python tools/devcheck.py kernel --task gemm --language rust --file tests/fixtures/perf_summary/rust/$f.rs; done
for f in dot_rayon_f64 dot_avx2;   do .venv/bin/python tools/devcheck.py kernel --task dot  --language rust --file tests/fixtures/perf_summary/rust/$f.rs; done
#   기대: 모두 => PASS (toolchain 표시 없음). 픽스처가 정확성용이 아니라 실패하면 그 픽스처가 무엇을 보이려는지(tests/test_perf_summary.py) 확인하고 결과를 적는다
```

키워드 모드 C 커널(구조체 경로를 gcc 로; Windows 에서는 zig 직렬과 rustc 로만 확인). 아래 두 파일은 Windows 에서 zig 직렬 빌드와 numpy 로 PASS 한 것과 같은 코드다.

```sh
mkdir -p /tmp/lxplan/kw
cat > /tmp/lxplan/kw/gemm_kw_omp.c <<'EOF'
#include "kernel_args.h"

void gemm(const gemm_args* args) {
    const float* restrict A = args->A;
    const float* restrict B = args->B;
    float* restrict C = args->C;
    const int M = args->M, N = args->N, K = args->K;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) C[i * N + j] = 0.0f;
        for (int k = 0; k < K; k++) {
            const float a = A[i * K + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
    }
}
EOF
grep -v '#pragma omp' /tmp/lxplan/kw/gemm_kw_omp.c > /tmp/lxplan/kw/gemm_kw_seq.c
cat > /tmp/lxplan/kw/gemm_kw.py <<'EOF'
import numpy as np


def gemm(*, A, B, C, M, N, K):
    np.matmul(A, B, out=C)
EOF
.venv/bin/python tools/devcheck.py kernel --task gemm --language c_openmp --args-mode keyword --file /tmp/lxplan/kw/gemm_kw_omp.c
.venv/bin/python tools/devcheck.py kernel --task gemm --language c_seq    --args-mode keyword --file /tmp/lxplan/kw/gemm_kw_seq.c
.venv/bin/python tools/devcheck.py kernel --task gemm --language numpy    --args-mode keyword --file /tmp/lxplan/kw/gemm_kw.py
#   기대: 세 줄 모두 => PASS stage=check classification=pass (toolchain 표시 없음 = gcc)

# Rust 키워드 모드(kernel_args.rs 모듈 경로를 Linux cargo 로). 커널은 tests/test_build_rust.py 의 RS_DOT_KEYWORD 와 같다
.venv/bin/python -c "import re; t=open('tests/test_build_rust.py', encoding='utf-8').read(); open('/tmp/lxplan/kw/dot_kw.rs', 'w').write(re.search(r\"RS_DOT_KEYWORD = '''(.*?)'''\", t, re.S).group(1))"
.venv/bin/python tools/devcheck.py kernel --task dot --language rust --args-mode keyword --file /tmp/lxplan/kw/dot_kw.rs
#   기대: dot/rust => PASS stage=check classification=pass (Windows cargo 에서도 PASS 였다). 같은 경로는 RealCargoTest.test_keyword_kernel_gets_the_args_module 도 본다
```

위치 인자 더미를 키워드 모드로 돌리면 전부 추출 단계에서 떨어진다(Windows 에서 7개 언어 모두 확인): `make devcheck` 대신 `.venv/bin/python tools/devcheck.py dummies --tasks gemm --args-mode keyword` → `summary: 0 pass, 7 fail …`, detail 은 Python 계열 `extract/signature_mismatch: param_mismatch:got (A, B, C, M, N, K) expected (*, A, B, C, M, N, K)`, C·rust `param_mismatch:6 params expected 1`.

### 3.7 Rust 빌드 CLI (5분)

```sh
mkdir -p /tmp/lxplan/rs && awk '/^```rust/{f=1;next}/^```/{f=0}f' harness/dummy_responses/dot/rust/0.md > /tmp/lxplan/rs/dot.rs
.venv/bin/python -m harness.build.rust --src /tmp/lxplan/rs/dot.rs --out-dir /tmp/lxplan/rs_dot --symbol dot; cat /tmp/lxplan/rs_dot/build.json | head -30
#   기대: ok true, command 가 taskset -c <cores> cargo build --release --offline --locked, symbol_check "nm", 결과 kernel.so

# 시간 초과 때 프로세스 그룹 정리 (Windows 에서는 cargo 만 죽었다)
.venv/bin/python -c "from types import SimpleNamespace as N; from pathlib import Path; from harness.build.rust import build_rust; print(build_rust(Path('/tmp/lxplan/rs/dot.rs').read_text(), Path('/tmp/lxplan/rs_to'), N(build_timeout_s=0.3), launch_prefix=[], symbol='dot').error_kind)"; pgrep -a rustc
#   기대: ToolchainMissing(의존성 빌드 중이었으면) 또는 timeout, 그리고 rustc 프로세스가 남지 않는다

# 미정의 외부 심벌: 링크에서 실패하는가 로드(dlopen)에서 실패하는가
sed -e 's/^#\[no_mangle\]/extern "C" { fn nowhere(); }\n#[no_mangle]/' \
    -e 's/^\(pub unsafe extern "C" fn dot(.*{\)$/\1\n    nowhere();/' /tmp/lxplan/rs/dot.rs > /tmp/lxplan/rs/dot_undef.rs
head -8 /tmp/lxplan/rs/dot_undef.rs       # extern 블록이 #[no_mangle] 앞에, nowhere(); 가 dot 본문 첫 줄에 있어야 한다
.venv/bin/python -m harness.build.rust --src /tmp/lxplan/rs/dot_undef.rs --out-dir /tmp/lxplan/rs_undef --symbol dot | tail -5
#   어느 쪽이든 행에서는 compile_error 다. 어느 단계였는지(error_stage 서술)를 적는다. Windows(MSVC)에서는 링크 오류였다
#   빌드가 성공하면 devcheck 로 로드까지 본다: .venv/bin/python tools/devcheck.py kernel --task dot --language rust --file /tmp/lxplan/rs/dot_undef.rs
```

(위 `sed` 식은 Windows 의 GNU sed 로 `dot/rust/0.md` 에서 뽑은 소스에 적용해 결과 텍스트를 확인했다. 빌드는 해 보지 않았다.)

### 3.8 fft 허용오차 여유 (선택, 20분)

`fft` 의 허용오차를 1e-3 으로 둘지는 사용자 확인 대상이다(§10). 판단 자료로, c_seq 더미의 트위들 표 조회를 블록마다 점화식으로 바꾼 커널을 gcc(`-O3 -march=native`, 기본 `-ffp-contract=fast`)로 돌린다: 블록마다 `float wr = 1, wi = 0;`, 나비마다 뒤에 `nwr = wr*wlr - wi*wli; wi = wr*wli + wi*wlr; wr = nwr;`, `wlr = (float)cos(-2*pi/len)`, `wli = (float)sin(-2*pi/len)`. 명령: `.venv/bin/python tools/devcheck.py kernel --task fft --language c_seq --file <그 파일> --bench`. Windows(clang)에서는 통과했고 tol_ratio 가 n=2¹⁶ 에서 0.31, n=2²² 에서 0.88–0.91(시드 하나, 컴파일 옵션 3종)이었다. 시드 7개로 넓히면 n=2²² 에서 0.78–0.91 이다(`docs/decisions.md` 2026-11-22 §6 A1; 권고는 `1e-4` 로 조이는 것). gcc 에서 판정이 달라지면 임계값이 이 구현 계열 안에 걸려 있다는 뜻이다.

---

## 4. 더미 파이프라인 끝에서 끝까지

모든 run 의 행은 `results/runs_dummy.csv` 로 가고, 샘플 산출물은 `results/built/<run_id>/<task>/<language>/<조건 태그>/sample_<i>/`(조건 태그 예: `args-positional_hint-off_fb-compiler`)에 생긴다. 샘플 디렉터리에는 `result.json`, `build.log`(C)·`build.json`, `import.json`, `check.json`, `paired_baseline.json`(bench 를 했으면), `lint.json`·`lint.stderr.log` 가 있다.

### 4.1 gemm 변형 15개 (`lx-gemm`; 15–20분, 대부분 numpy/3 의 bench 시간 초과 300 s)

```sh
time make dummy RUN_ID=lx-gemm TASKS=gemm         # n=5(run.n_samples), 언어 numpy·c_seq·c_openmp(run.languages)
rows lx-gemm
```

더미 파일은 `sample_idx mod 파일 수` 로 고른다(numpy 5개, C 3개). 기대 행(`harness/dummy_responses/gemm/README.md` 의 표를 n=5 에 맞춘 것):

| language | sample | 응답 | error_stage / error_type / error_detail | pass |
|---|---|---|---|---|
| numpy | 0 | `0.md` | 없음 | 1 |
| numpy | 1 | `1.md` | extract / signature_mismatch / `param_mismatch:…` | 0 |
| numpy | 2 | `2.md` | check / numeric_instability / `imprecise` (max_tol_ratio ≈ 10) | 0 |
| numpy | 3 | `3.md` | bench / timeout (시간 NA) | 1 |
| numpy | 4 | `4.md` | check / wrong_output / `returned_array_not_written` | 0 |
| c_seq | 0, 3 | `0.md` | 없음 (`signature_exact=0`, `signature_ok=1`) | 1 |
| c_seq | 1, 4 | `1.md` | build / compile_error / `error: expected ',' or ';' before 'for'` | 0 |
| c_seq | 2 | `2.md` | check / arg_order / `transposed_output` | 0 |
| c_openmp | 0, 3 | `0.md` | 없음 | 1 |
| c_openmp | 1, 4 | `1.md` | check / runtime_crash / `signal SIGSEGV during tiny call 0` | 0 |
| c_openmp | 2 | `2.md` | check / race_condition / `nondeterministic:` + rect·small (tiny 는 포함되기도 한다) | 0 |

추가로 볼 것(M1 대비 새로 생긴 것):

- 모든 행의 `lint_*` 열: numpy 는 정수, C 는 cppcheck·clang-tidy 가 있으면 정수(없으면 NA). `perf_summary` 가 빌드에 성공한 행마다 있다.
- 대응 기준선은 `sample_<i>/paired_baseline.json` 이다(예전의 `baseline/paired_sample_*.json` 이 아니다). 로그 머리말에 조건 태그가 붙는다.
- `taskset_used=1`, `governor_ok=1`(§1.6 을 했으면), `protocol_ok=1`(부하가 `max_load_1m` 2.0 아래면).
- 다시 돌리기: `.venv/bin/python -m harness.run --run-id lx-gemm` → 전부 건너뜀(`run 요약: 행 0개 (건너뜀 15개)` 류), 기준선을 다시 재지 않는다.
- 다를 때: 행의 `built_path` 디렉터리에서 `result.json` 의 `stages` 와 `check.json` 을 본다. c_openmp/2 가 가끔 `wrong_output` 이면 틀린 값이 우연히 두 번 같은 경우다(README "변형별 주의"). c_openmp/1 이 SIGILL 이면 `volatile` 이 빠진 것이다.

### 4.2 언어 일곱 개 × 과제 전부 (`lx-all`; 40–80분, 추정)

```sh
time make dummy-all RUN_ID=lx-all          # generate --languages <7개> --n 1 → run(bench 포함) → aggregate --coverage
rows lx-all | awk '{print $8}' | sort | uniq -c    # pass= 분포
```

- 기대: 생성 80 calls(지원하지 않는 4개는 표에 unsupported), 행 80개, 모두 `pass=1`, `time_median_ms`·`time_cv`·`speedup_vs_baseline` 이 채워짐. `--coverage` 표에서 모든 칸 n=1, `sort`·`bfs` 의 Halide 두 칸은 `n/a` 이고 이유가 표 아래에 나온다.
- 언어 묶음만 따로 돌릴 때: `make dummy-m3 RUN_ID=lx-m3`(새 언어 네 개 × 과제 12개 = 44행), `make dummy-all RUN_ID=lx-t2 LANGUAGES=numpy,c_seq,c_openmp TASKS=histogram,prefix_sum,nbody,sort`.
- 시간 감각: numpy bfs 더미는 호출당 2–5 s 라 bench(워밍업 3 + 10회)에 30–80 s 가 든다(예산 300 s 안). sort 의 large npz 는 134 MB 다. nbody numpy 는 워커 주소 공간 한도 8 GiB 안에서 돌아야 한다(블록 131072//n 행).

행에서 확인할 것(`rows` 출력, CSV, `result.json`):

- `omp_env`: numba 행에 `NUMBA_NUM_THREADS`·`NUMBA_THREADING_LAYER=omp`, halide 행에 `HL_NUM_THREADS`, rust 행에 `RAYON_NUM_THREADS`. 값은 `measurement.threads`. 기준선 실행에는 언어 변수가 붙지 않는다. raw 의 프롬프트에 적힌 `NUMBA_NUM_THREADS`/`HL_NUM_THREADS`/`RAYON_NUM_THREADS` 값도 `measurement.threads` 와 같다.
- rust: `result.json` 의 `stages.build.extra.symbol_check == "nm"`, 산출물 `kernel.so`, `results/built/lx-all/_cargo_target` 를 rust 샘플들이 공유한다(동시 빌드 경쟁 없음). `results/built/lx-all/_cargo_target/_toolchain_probe/probe.log` 가 있다.
- halide: stderr 에 `관찰값(--inspect) 없이 계속한다` 경고가 없고, `result.json` 의 `stages.import.elapsed_s` 가 `check_timeout_s`(180 s)보다 훨씬 작다(특히 jacobi2d, fft, floyd_warshall). halide_auto 행에 `schedule_in_halide_auto` contract_violation 이 없다. floyd_warshall·cholesky 의 halide_auto 에서 `Update definition 0 of function D has not been scheduled` 경고는 Mullapudi 가 impure update 를 직렬로 둔 것이라 무해하다. prefix_sum/halide_auto 의 `Update definition 0 of function offset has not been scheduled` 는 Halide 가 stdout 에 찍는데, 커널 stdout 출력으로 기록되지 않고 stderr 로 가야 한다(`check.stdout.log` 가 비어 있는지).
- `cflags`: halide_auto 행이 `autoscheduler=Mullapudi2016 params={…,"parallelism":"4"}`, halide_manual 행이 `autoscheduler=none`, rust 행에 release 프로필.
- `speedup_vs_baseline`: fft/numpy 와 bfs/c_seq 는 더미가 기준선과 같은 코드라 1.0 근처(bfs 는 1스레드 기준선 대비 `speedup_vs_baseline_1t` 도 본다).
- 각 Tier 의 numba·rust·halide 행에 `perf_summary` 내부 오류가 없다(§7.5 의 스니펫).

### 4.3 조건: `args`·`grammar_hint` (`lx-cond`; 10분)

같은 run 에 세 조건을 만든다. Windows 에서 `dot × numpy,numba` 로 같은 절차를 돌려 아래 기대를 확인했다.

```sh
L=numpy,c_seq,c_openmp,numba,halide_auto,halide_manual,rust
.venv/bin/python -m harness.generate --provider dummy --run-id lx-cond --tasks gemm,dot --languages $L --n 1
.venv/bin/python -m harness.generate --provider dummy --run-id lx-cond --tasks gemm,dot --languages $L --n 1 --args keyword
.venv/bin/python -m harness.generate --provider dummy --run-id lx-cond --tasks gemm,dot --languages $L --n 1 --grammar-hint on
.venv/bin/python -m harness.run --run-id lx-cond --skip-bench
.venv/bin/python -m analysis.aggregate --csv results/runs_dummy.csv --run-id lx-cond --conditions
```

- 기대 행: positional·hint off 14행 pass; keyword 14행 모두 extract / signature_mismatch(위치 인자 더미이므로; §3.6 끝); hint on 8행 pass(힌트 파일 `prompts/hints/` 는 numba·halide_auto·halide_manual·rust 에만 있어서 numpy·c_seq·c_openmp 는 생성 표에 `skip` 으로 나오고 raw 가 없다).
- hint on raw 의 `prompt_hash` 는 hint off 와 다르고 응답은 같다.
- `--conditions` 표: (task, language)마다 `factor=args` 행 하나(base positional, alt keyword, `method=unpaired`, `diff -1.000`)와, 힌트가 있는 네 언어에 `factor=grammar_hint` 행 하나(`diff 0.000`). n=1 이라 구간은 점이다.
- keyword raw 의 rust 프롬프트에는 `kernel_args.rs`, C 프롬프트에는 `kernel_args.h` 가 들어 있다.

### 4.4 키워드 모드의 통과 경로 (`lx-kw`; 5분)

더미 제공자는 조건과 무관하게 같은 파일을 주므로 키워드 모드의 통과 경로는 raw 의 응답을 손으로 바꿔 확인한다. `harness.run` 은 raw 의 `response_text` 에서 코드를 추출하고, raw 가 바뀌면(`raw_sha256`) 다시 처리한다. Windows 에서 numpy 로 이 절차를 확인했다(pass=1, args=keyword, signature_ok=1).

```sh
.venv/bin/python -m harness.generate --provider dummy --run-id lx-kw --tasks gemm --languages numpy,c_seq,c_openmp --n 1 --args keyword
.venv/bin/python - <<'EOF'
import json, pathlib
root = pathlib.Path("results/raw/lx-kw/gemm")
src = {"numpy": ("python", "/tmp/lxplan/kw/gemm_kw.py"),
       "c_seq": ("c", "/tmp/lxplan/kw/gemm_kw_seq.c"),
       "c_openmp": ("c", "/tmp/lxplan/kw/gemm_kw_omp.c")}
for lang, (fence, path) in src.items():
    raw = root / lang / "args-keyword_hint-off_fb-compiler" / "sample_0.json"
    d = json.loads(raw.read_text(encoding="utf-8"))
    d["response_text"] = f"Keyword version.\n\n```{fence}\n" + pathlib.Path(path).read_text(encoding="utf-8") + "```\n"
    raw.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("edited", raw)
EOF
.venv/bin/python -m harness.run --run-id lx-kw
rows lx-kw
ls results/built/lx-kw/gemm/c_seq/args-keyword_hint-off_fb-compiler/sample_0/
```

- 기대: 3행 모두 `pass=1`, `args=keyword`, 시간 채워짐. C 샘플 디렉터리에 `kernel_args.h` 가 있다(extract 의 전처리, `build_c` 의 `args_header`, 워커 `--args-mode keyword` 가 이어진다).
- 이 run 의 raw 는 손으로 고친 것이므로 결과 행을 분석에 쓰지 않는다. 세션 끝에 남길지 정할 때 이 run 은 빼는 편이 낫다.

### 4.5 전체 `make dummy` (선택; 1–2시간, 추정)

`make dummy RUN_ID=lx-dummy` 는 과제 12개 × numpy·c_seq·c_openmp × n=5 = 180 샘플을 bench 까지 돈다. §4.1(gemm 변형)과 §4.2(모든 `0.md`)가 같은 경로를 덮으므로 필수는 아니다. 돌린다면 기대는 gemm 15행이 §4.1 표와 같고, 나머지 165행(다른 과제는 `0.md` 하나뿐이라 다섯 샘플이 같은 응답)이 모두 pass 다.

### 4.6 준비 단계의 중단 경로 (exit 7; 20분)

처리할 언어에 필요한 도구가 없으면 `harness.run` 은 행을 쓰기 전에 exit 7 로 멈춰야 한다(컴파일러가 없어 생긴 실패가 `compile_error` 행으로 남으면 pass 율이 오염된다). 각 확인 뒤 `rows lx-tool` 이 비어 있고 `results/built/lx-tool/` 에 `result.json` 이 생기지 않았는지 본다.

```sh
.venv/bin/python -m harness.generate --provider dummy --run-id lx-tool --tasks gemm,jacobi2d --languages numpy,c_seq,rust,halide_auto,numba --n 1

# cargo 가 ~/.cargo/bin 에만 있을 때 PATH 에서 빼면
env PATH=/usr/bin:/bin .venv/bin/python -m harness.run --run-id lx-tool --languages rust --skip-bench; echo rc=$?          # 기대 7
# CARGO_HOME 이 빈 디렉터리(rayon 캐시 없음, build.rust.offline=true)
mkdir -p /tmp/lxplan/empty_cargo && CARGO_HOME=/tmp/lxplan/empty_cargo .venv/bin/python -m harness.run --run-id lx-tool --languages rust --skip-bench; echo rc=$?
#   기대 7, '중단: 도구 없음 (Rust 툴체인 점검)'
# C 컴파일러가 없는 설정
mkcfg nocc 's/^  cc: gcc/  cc: no-such-cc/'
.venv/bin/python -m harness.run --config /tmp/lxplan/nocc.yaml --run-id lx-tool --languages c_seq --skip-bench; echo rc=$?   # 기대 7
# C 기준선이 있는 과제(jacobi2d)를 numpy 만, bench 포함, 컴파일러 없이
.venv/bin/python -m harness.run --config /tmp/lxplan/nocc.yaml --run-id lx-tool --languages numpy; echo rc=$?               # 기대 7, 'C 기준선'
# 없는 코어. 코어 수는 threads(4) 이상이어야 설정 로더를 지나므로 네 개 중 하나만 없는 CPU 로 바꾼다
#   ("0,999" 처럼 두 개만 주면 taskset 확인 전에 'measurement.threads: 4 가 고정 코어 수 2 보다 크다' 설정 오류로 끝난다)
mkcfg badcores 's/^  cores: "0,2,4,6"/  cores: "0,2,4,999"/'
.venv/bin/python -m harness.run --config /tmp/lxplan/badcores.yaml --run-id lx-tool --languages numpy --skip-bench; echo rc=$?
#   기대 7, 메시지에 measurement.cores=0,2,4,999 와 taskset 의 stderr(예: failed to set pid …'s affinity: Invalid argument)
#   threads: 1 언어(c_seq)는 measurement.single_core 를 쓰므로 같은 확인을 single_core 로도 할 수 있다
# 틀린 Halide 오토스케줄러 매개변수
mkcfg hlbad 's/autoscheduler_params: {}/autoscheduler_params: {no_such_key: "1"}/'
.venv/bin/python -m harness.run --config /tmp/lxplan/hlbad.yaml --run-id lx-tool --languages halide_auto --skip-bench; echo rc=$?
#   기대 7, halide.autoscheduler_params 를 지목, 사유에 signal=SIGABRT (프로브 프로세스가 SIGABRT 로 끝남)
# 설치되지 않은 Numba 스레딩 계층
mkcfg nbtbb 's/NUMBA_THREADING_LAYER: omp/NUMBA_THREADING_LAYER: tbb/'
.venv/bin/python -m harness.run --config /tmp/lxplan/nbtbb.yaml --run-id lx-tool --languages numba --skip-bench; echo rc=$?
#   기대 7, NUMBA_THREADING_LAYER 를 지목 (tbb 패키지가 없을 때)
# 링커가 망가진 Rust. 셸의 RUSTFLAGS 는 빌드가 설정의 build.rust.rustflags 로 덮어쓰므로 설정 사본으로 준다
mkcfg badlinker 's#rustflags: \["-C", "target-cpu=native"\]#rustflags: ["-C", "target-cpu=native", "-C", "linker=/nonexistent/cc"]#'
.venv/bin/python -m harness.run --config /tmp/lxplan/badlinker.yaml --run-id lx-tool --languages rust --skip-bench; echo rc=$?
#   기대 7 (준비 단계의 Rust 툴체인 프로브가 같은 설정으로 자리표시 크레이트를 빌드하다 실패), rust compile_error 행이 생기지 않는다

# 정상 설정에서는 모두 지나간다
.venv/bin/python -m harness.run --run-id lx-tool --skip-bench; echo rc=$?          # 기대 0, 10행 pass
ls results/built/lx-tool/_cargo_target/_toolchain_probe/probe.log
```

### 4.7 중단 신호 (5분)

```sh
.venv/bin/python -m harness.generate --provider dummy --run-id lx-int --tasks gemm,bfs --languages numpy,rust --n 1
# (1) 앞에서 돌리고 bench 나 cargo 빌드가 진행 중일 때 Ctrl-C 를 누른다
.venv/bin/python -m harness.run --run-id lx-int; echo rc=$?; pgrep -f harness.worker
# (2) 뒤에서 돌리고 SIGTERM 을 보낸다 (비대화형 셸의 백그라운드 작업은 SIGINT 를 무시할 수 있어 TERM 을 쓴다)
.venv/bin/python -m harness.run --run-id lx-int --force &
sleep 20; kill -TERM $!; wait $!; echo rc=$?; pgrep -f harness.worker
#   기대: 두 경우 모두 rc=130, 진행 중이던 샘플의 행 없음, pgrep 출력 없음. 한 번은 cargo 빌드 중에, 한 번은 bench 중에 해 본다
```

수정 루프의 SIGTERM 은 §5.3 이다.

---

## 5. 수정 루프와 피드백 조건의 대응 비교

### 5.1 실행 (`lx-repair`; 30–40분, `SKIP_BENCH=1` 이면 약 10분)

```sh
time make repair-dummy RUN_ID=lx-repair LANGUAGES=numpy,c_seq,c_openmp     # REPAIR_TURNS 기본 3
```

이 타깃은 다음을 차례로 한다: gemm × 세 언어 × n=5 생성(`--feedback compiler`) → `harness.run` → `LLM_PROVIDER=dummy harness.repair --feedback compiler --max-turns 3` → `harness.repair --seed-from-feedback compiler --feedback compiler+lint`(턴 0 raw 15개 복사, API 호출 없음) → `harness.run` → `LLM_PROVIDER=dummy harness.repair --feedback compiler+lint --max-turns 3` → `analysis.aggregate --by-turn --conditions`. `harness.repair` 는 제공자를 설정·환경변수에서 읽으므로 `LLM_PROVIDER=dummy` 로 턴 0 의 제공자와 맞춘다(다르면 exit 1).

수정 턴 응답은 같은 k 의 `<k>.turn<t>.md` 이고, 없으면 `<k>.md` 를 다시 준다(같은 실패 반복). gemm 에는 `c_seq/1.turn1.md`, `numpy/2.turn1.md` 만 있다. 대상은 직전 턴 행이 `valid_sample=1 ∧ pass=0 ∧ error_stage ≠ generate` 인 샘플이다(bench timeout 인 numpy/3 은 `pass=1` 이라 대상이 아니다).

기대(피드백 조건마다 같다):

| 턴 | 대상(호출 수) | 통과 |
|---|---|---|
| 0 | — (15행) | numpy 0·3, c_seq 0·3, c_openmp 0·3 |
| 1 | numpy 1·2·4, c_seq 1·2·4, c_openmp 1·2·4 (9) | numpy/2, c_seq/1, c_seq/4 (`turns_to_pass=1`) |
| 2 | numpy 1·4, c_seq 2, c_openmp 1·2·4 (6) | 없음 |
| 3 | 같은 6개 (6) | 없음 |

- 행 수: 조건마다 15 + 9 + 6 + 6 = 36, run 전체 72. 턴 1 이상의 행은 `sample_<i>/turn_<t>/` 아래에 생기고 `run_id`·task·language·조건·`sample_idx` 가 턴 0 과 같으며, `baseline_run_time_ms` 는 기존 기준선 파일에서 가져온다.
- `--by-turn`: `주: 수정 루프 행(turn ≥ 1) N개…` 한 줄. `pass_within_0/1/2/3` = numpy 0.4/0.6/0.6/0.6, c_seq 0.4/0.8/0.8/0.8, c_openmp 0.4/0.4/0.4/0.4. `turns_to_pass` 분포: numpy `0:2, 1:1, NA:2`, c_seq `0:2, 1:2, NA:1`, c_openmp `0:2, NA:3`. 고칠 수 없는 더미도 최대 턴까지 행이 있어 NA 없이 값이 나온다.
- `--conditions`: (task, language)마다 `factor=feedback` 행 하나, `method=paired`, `n_base == n_alt == 5`, `quantity` 는 `pass_within_<H>`(H 는 두 수준의 `--max-turns` 중 작은 값 이하 = 3), `rate_base`·`rate_alt` 는 `--by-turn` 의 그 값, `diff 0.000`(더미 응답은 프롬프트와 무관). stderr 에 `짝이 없는 샘플` 경고가 없어야 한다(나오면 seed 복사가 `sample_idx` 나 `run_id` 를 바꾼 것). 수준이 둘 이상인 조건만 행이 생기므로 이 run(args=positional, grammar_hint=off 뿐)에는 `args`·`grammar_hint` 행이 없다. 그 두 조건의 행은 §4.3 처럼 수준이 둘인 run 에서 나오고, `at` 에 feedback 이 없다(턴 0 은 feedback 과 무관).
- 다시 만들기: `.venv/bin/python -m harness.run --rebuild-csv` 가 같은 행 수를 재현한다(이 명령은 `built_dir` 의 모든 `result.json` 으로 CSV 를 다시 쓰므로 다른 run 행도 함께 다시 쓴다).
- c_seq/1.turn1.md 는 i-j-k 3중 루프라 1024³ 한 호출이 수 초다(노트북 gcc 1스레드 약 9.5 s, Windows zig 7.1 s). bench 예산(300 s) 안에 들어가는지 본다. 넘으면 `repeats_reduced=1` 이거나 bench timeout 이다.
- c_openmp/2 는 턴마다 `race_condition` 이 기대이지만 가끔 `wrong_output` 일 수 있다(§4.1).

### 5.2 수정 요청의 내용 (5분)

```sh
R=results/raw/lx-repair/gemm
.venv/bin/python - <<'EOF'
import json
R = "results/raw/lx-repair/gemm"
def rep(lang, fb, i): return json.load(open(f"{R}/{lang}/args-positional_hint-off_fb-{fb}/sample_{i}.turn1.json", encoding="utf-8"))["repair"]
print(rep("c_seq", "compiler", 1)["diagnostics"][:300]); print("---")
print(rep("c_openmp", "compiler", 1)["diagnostics"][:200]); print("---")
print(rep("c_openmp", "compiler", 2)["diagnostics"][:300]); print("---")
r = rep("c_seq", "compiler+lint", 2); print(r["lint_included"], r["n_lint_findings"])
EOF
.venv/bin/python - <<'EOF'
import json, difflib
base = "results/raw/lx-repair/gemm/c_seq/args-positional_hint-off_fb-{}/sample_2.turn1.json"
a, b = (json.load(open(base.format(f), encoding="utf-8")) for f in ("compiler", "compiler+lint"))
ua, ub = a["prompt"]["messages"][-1]["content"], b["prompt"]["messages"][-1]["content"]
print("\n".join(difflib.unified_diff(ua.splitlines(), ub.splitlines(), lineterm="")))
EOF
```

- c_seq sample_1 의 `diagnostics` 는 `Compiler output:` 으로 시작하고 gcc 의 `kernel.c:<줄>:<열>: error: …` 가 절대 경로 없이 이어진다.
- c_openmp sample_1 은 `crashed: signal SIGSEGV`, sample_2 는 `results differ between two identical calls` 를 담는다.
- `fb-compiler+lint` 쪽은 `lint_included` 가 true 이고, 린트 결과가 있는 C 샘플이면 마지막 user 메시지에 `## Static analysis findings` 절과 `- line N: …` 줄(최대 20개)이 있다. 두 조건의 요청은 그 절만 다르다(위 diff 가 그 절만 보여야 한다).
- 하네스 내부 패턴 이름(`transposed_output` 등)과 참조값이 요청에 없다.

Python 예외 진단(5분): check 단계에서 Python 예외로 실패하는 샘플(예: 0 으로 나누는 numba 커널, 스케줄 오류가 있는 halide_manual)을 §4.4 처럼 raw 를 고쳐 만들고 `harness.run` 뒤 `LLM_PROVIDER=dummy .venv/bin/python -m harness.repair --run-id <그 run> --max-turns 1 --skip-bench` 를 돌린다. 턴 1 raw 의 `repair.diagnostics` 와 마지막 user 메시지에 `call_kernel`, `invoke`, `_build`, `harness/` 프레임이 없어야 하고, numba 의 경우 형태 줄이 `raised ZeroDivisionError: division by zero` 로 끝나야 한다.

### 5.3 수정 루프 중단 (5분)

`lx-repair` 는 모든 턴이 끝나 할 일이 없으므로 새 run 으로 한다.

```sh
.venv/bin/python -m harness.generate --provider dummy --run-id lx-rint --tasks gemm --languages numpy,c_seq --n 5
.venv/bin/python -m harness.run --run-id lx-rint --skip-bench
LLM_PROVIDER=dummy .venv/bin/python -m harness.repair --run-id lx-rint --feedback compiler --max-turns 2 &
sleep 10; kill -TERM $!; wait $!; echo rc=$?; pgrep -f harness.worker
#   기대: rc=130, pgrep 출력 없음 (수정 턴의 처리는 bench 를 포함하므로 10초 뒤에는 워커가 돌고 있다)
```

---

## 6. sanitizer 와 린트

### 6.1 전제

§1.4 의 (b)(c)가 통과해야 한다. sanitizer 는 C 백엔드에서만, 정확성 형태로만 돈다. Rust 에는 돌리지 않는다(nightly 전용). TSan 은 하지 않는다.

### 6.2 단위 테스트 (3분)

```sh
.venv/bin/python -m unittest tests.test_safety.RealSanitizerRunTest tests.test_safety.SanitizerEnvTest -v
```

- `RealSanitizerRunTest` 4개: 정상 커널 → `(True, True, 0, 0, '')`; 커널 내부 malloc 버퍼 한 칸 넘침 → asan 1, `kinds[0]=heap-buffer-overflow`, detail `incomplete:signal SIGABRT`, `san/` 에 로그 파일; 하네스 출력 배열 한 칸 뒤 쓰기(`test_write_one_past_the_harness_output_array_is_an_asan_finding`) → asan ≥ 1, `kinds[0]` `heap-buffer-overflow`; volatile 부호 오버플로 → `(0, 1, ['signed-integer-overflow'], '')`.
- `SanitizerEnvTest.test_real_gcc_resolves_an_existing_libasan`: `LD_PRELOAD` 에 존재하는 파일 두 개, libasan 이 먼저.
- 실패하면 메시지의 summary(env, log_files, worker)를 본다.

### 6.3 워커의 `--sanitize` 버퍼가 가로채지는가 (5분)

```sh
cat > /tmp/lxplan/oob.c <<'EOF'
void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M * N; i++) C[i] = A[0] * B[0] * (float)K;
    C[M * N] = 1.0f;   /* 출력 배열 한 칸 뒤 */
}
EOF
gcc -std=gnu11 -fPIC -shared -O1 -fsanitize=address,undefined -fno-omit-frame-pointer -g -o /tmp/lxplan/oob.so /tmp/lxplan/oob.c -lm
LD_PRELOAD=$(gcc -print-file-name=libasan.so) ASAN_OPTIONS=detect_leaks=0 .venv/bin/python -m harness.worker --backend c \
  --artifact /tmp/lxplan/oob.so --mode check --tests tasks/gemm/tests/tiny.npz --tolerance tasks/gemm/tolerance.yaml \
  --signatures tasks/gemm/signatures.yaml --out /tmp/lxplan/x.json --check-repeats 1 --max-mem-gb 0 --sanitize 2>&1 | grep -E 'AddressSanitizer|posix_memalign' | head
#   기대: 'AddressSanitizer: heap-buffer-overflow' 와 'allocated by' 스택에 posix_memalign
#   같은 명령에서 --sanitize 를 빼면 ASan 보고가 없다(가드 영역 안이라서; 리뷰가 찾은 원래 문제)
```

### 6.4 파이프라인 (`lx-san`; 10분)

```sh
mkcfg san 's/^safety: {enabled: false/safety: {enabled: true/'
.venv/bin/python -m harness.generate --config /tmp/lxplan/san.yaml --provider dummy --run-id lx-san --tasks gemm --languages numpy,c_seq,c_openmp
.venv/bin/python -m harness.run --config /tmp/lxplan/san.yaml --run-id lx-san --skip-bench
rows lx-san
.venv/bin/python - <<'EOF'
import csv
for r in csv.DictReader(open("results/runs_dummy.csv", encoding="utf-8")):
    if r["run_id"] == "lx-san":
        print(r["language"], r["sample_idx"], r["error_type"], r["asan_findings"], r["ubsan_findings"], r["error_detail"][:80])
EOF
```

- C 샘플마다 `san/` 디렉터리, `result.json` 에 `stages.safety`(워커 요약)와 `stages.sanitize`.
- 정상 더미(c_seq 0·3, c_openmp 0·3): `asan_findings=0`, `ubsan_findings=0`. numpy 행: NA.
- c_openmp 1·4(NULL 저장): `error_type=memory_error`, `error_detail` 이 `asan:SEGV; was runtime_crash/signal SIGSEGV` 로 시작한다(실제 종류 문자열을 적는다). `pass` 는 원래 0 이다.
- c_seq 2(전치 저장): `arg_order` 그대로. c_seq 1·4(컴파일 오류)와 check 가 시간 초과한 샘플에는 `sanitize` 단계가 없다.
- 로그 위치와 잡음: `ls results/built/lx-san/gemm/c_seq/args-positional_hint-off_fb-compiler/sample_0/san/` 와 `.venv/bin/python -m harness.safety --parse-logs <그 san 디렉터리>`. (a) UBSan 보고가 `ubsan.<pid>` 에 가는지 `asan.<pid>` 에 가는지(파서는 둘 다 읽는다; `safety.stderr.log` 에 있어도 센다), (b) 정상 커널에서 CPython·NumPy 자체가 ASan 보고(`alloc-dealloc-mismatch` 등)를 내지 않는지. 낸다면 `kernel.c`/`kernel_san.so` 프레임이 있는 보고만 세도록 `harness/safety.py` 의 `asan_reports` 를 고친다.
- `handle_sigfpe=0` 의 효과: `int z = N - N; C[0] = (float)(M / z);` 를 넣은 커널을 같은 방법(§4.4 처럼 raw 를 고쳐서)으로 돌리면 워커가 ASan 보고 없이 SIGFPE 로 죽고(`detail=incomplete:signal SIGFPE`, `asan_findings` None) UBSan 의 `division by zero`(kinds 에 `divide-by-zero`)만 남아야 한다.
- 부호 오버플로 커널을 같은 방법으로 돌리면 `san/` 에 `ubsan.<pid>` 파일이 있고 `runtime error` 줄과 `    #0 ` 스택 프레임이 담긴다.

### 6.5 sanitizer 가 켜진 수정 루프 (5분)

```sh
LLM_PROVIDER=dummy .venv/bin/python -m harness.repair --config /tmp/lxplan/san.yaml --run-id lx-san --max-turns 1; echo rc=$?
grep -rhoE "Kind of failure: [a-z -]+" results/raw/lx-san --include='*.turn1.json' | sort | uniq -c
```

- 기대: rc=0(exit 5 로 멈추지 않는다). 문구는 `compile error` / `runtime crash` / `incorrect output` / `non-deterministic output` / `signature mismatch` 뿐이다. `memory_error` 로 재분류된 행은 수정 요청에서 이전 판정(`runtime crash`)으로 되돌아가야 한다.

### 6.6 clang 으로 빌드할 때만 생기는 경우 (선택, 5분)

`build.cc` 를 clang 으로 바꿀 계획이 없으면 건너뛴다.

```sh
cd /tmp/lxplan && P=$(clang -print-file-name=libasan.so); clang -shared -fPIC -O1 -fsanitize=address,undefined -o k.so k.c \
  && LD_PRELOAD=$P python3 -c "import ctypes; ctypes.CDLL('./k.so')"; cd -
#   깨끗한 OSError(undefined symbol)면 그 상황이 실제로 생긴다: sanitize 단계의 detail 이 'load_failed:…' 이고 개수가 NA 여야 한다.
#   ASan 중단이면 'incomplete:signal SIGABRT'.
```

### 6.7 린트 베이스라인 (15분)

```sh
mkdir -p /tmp/lxplan/lint_c && printf '#include <stdlib.h>\n#include <omp.h>\nvoid gemm(const float* A, const float* B, float* C, int M, int N, int K) {\n    float* t = malloc((size_t)M * N * sizeof(float));\n    int acc[8];\n    acc[8] = 0;\n    #pragma omp parallel for\n    for (int i = 0; i < M * N; i++) t[i] = A[0] * B[0] * (float)(1 / K);\n    C[0] = t[0] + (float)acc[8];\n}\n' > /tmp/lxplan/lint_c/kernel.c
.venv/bin/python plugins/lint/run_lint.py --language c_openmp --file /tmp/lxplan/lint_c/kernel.c --task gemm --signature '{}' --out /tmp/lxplan/lint_c/lint.json; cat /tmp/lxplan/lint_c/lint.json
```

- 기대: `"tool": "baseline:cppcheck+clang-tidy"`, `versions` 에 두 도구, findings 에 `cppcheck/arrayIndexOutOfBounds`(memory, error)·`cppcheck/memleak`·`clang-tidy/bugprone-integer-division`(numeric)·`clang-tidy/openmp-use-default-none`(concurrency) 등, `summary` 의 값이 모두 정수(null 없음). stderr 에 `[lint] cppcheck:`/`[lint] clang-tidy:` 실패 줄이 없다.
- 도구가 `tool` 에서 빠지면 stderr 의 사유를 본다: `--library=posix`·`--config={}` 가 그 버전에서 거부되는지, `omp.h` 를 못 찾는지, cppcheck XML 의 location 순서(첫 location 이 오류 지점이라는 가정)·clang-tidy 진단 줄 형식이 파서와 맞는지.

`omp.h` 경로(1분):

```sh
printf '#include <omp.h>\nint f(void){return omp_get_max_threads();}\n' > /tmp/lxplan/omp_k.c
CC=gcc .venv/bin/python plugins/lint/run_lint.py --language c_openmp --file /tmp/lxplan/omp_k.c --task gemm --signature '{}' --out /tmp/lxplan/l.json; cat /tmp/lxplan/l.json
```

- 기대: `tool` 이 `baseline:cppcheck+clang-tidy`, `summary` 값이 모두 정수, stderr 에 `[lint] clang-tidy: /usr/lib/gcc/…/include/omp.h 를 -isystem 으로 준다`. clang 이 gcc 의 `omp.h` 를 해석하지 못하면 clang-tidy 가 빠지고 `concurrency`·`api_misuse`·`performance` 가 0 이 아니라 null 이다. 어느 경우였는지 `docs/decisions.md` 에 적는다.

실제 커널에서의 시간과 잡음(10분):

```sh
for f in $(ls results/built/lx-all/*/c_openmp/*/sample_0/kernel.c | head -6); do
  /usr/bin/time -f "%e s  $f" .venv/bin/python plugins/lint/run_lint.py --language c_openmp --file "$f" --task gemm --signature '{}' --out /tmp/lxplan/lint.json
  .venv/bin/python -c "import json,collections; d=json.load(open('/tmp/lxplan/lint.json')); print(collections.Counter(x['rule'] for x in d['findings']).most_common(6))"
done
```

- clang-tidy 가 도구당 20 s(`TOOL_TIMEOUT_S`) 안에 끝나는지, `bugprone-implicit-widening-of-multiplication-result`·`bugprone-narrowing-conversions`·`nullPointerOutOfMemory` 가 커널당 몇 건인지 본다. 많으면 `plugins/lint/rules.yaml` 에서 빼거나 info 로 내린다(결정으로 적는다).
- 하네스 경로(계약 검사 포함)를 따로 볼 때: `.venv/bin/python -m harness.safety --lint <샘플>/kernel.c --language c_openmp`.
- 통합 확인: `lx-gemm`·`lx-all` 의 numpy 행 `lint_*` 가 정수, C 행은 정수, 각 샘플 디렉터리에 `lint.json`·`lint.stderr.log`.

---

## 7. perf_summary

### 7.1 단위 테스트

`.venv/bin/python -m unittest tests.test_perf_summary -v` → 111개 OK(§2 에서 이미 돌았으면 건너뛴다).

### 7.2 C 픽스처를 실제 gcc 로그로 (10분)

```sh
for f in gemm_ikj gemm_ijk dot_unrolled gemm_blis_avx2; do
  d=$(mktemp -d); cp tests/fixtures/perf_summary/c/$f.c $d/kernel.c
  (cd $d && gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native -fopt-info-vec-optimized -fopt-info-vec-missed -o kernel.so kernel.c -lm -Wl,--no-undefined 2> build.log)
  echo "== $f"; grep -E ': (optimized|missed): ' $d/build.log | sort -u
  .venv/bin/python -m harness.perf_summary --language c_seq --backend c --file $d/kernel.c --build-log $d/build.log
done
for f in gemm_omp_outer dot_omp_simd_reduction jacobi2d_omp spmv_csr_omp gemm_blocked_omp_collapse histogram_omp_private; do
  d=$(mktemp -d); cp tests/fixtures/perf_summary/c/$f.c $d/kernel.c
  (cd $d && gcc -std=gnu11 -fPIC -shared -Wall -fno-semantic-interposition -O3 -march=native -fopenmp -fopt-info-vec-optimized -fopt-info-vec-missed -o kernel.so kernel.c -lm -Wl,--no-undefined 2> build.log)
  echo "== $f"; grep -E ': (optimized|missed): ' $d/build.log | sort -u
  .venv/bin/python -m harness.perf_summary --language c_openmp --backend c --file $d/kernel.c --build-log $d/build.log
done
gcc -O3 -march=native -fopt-info-vec-optimized -fopt-info-vec-missed -c tests/fixtures/perf_summary/c/gemm_ikj.c -o /dev/null
```

볼 것:

- (a) `loop vectorized` 줄의 줄:열이 `for` 조건식의 비교 연산자 위치와 같은가(가정). gemm_ikj 는 `gemm_ikj.c:12:31` 이 기대다.
- (b) memset 으로 바뀌는 초기화 루프(gemm_ikj.c 7행)에 리마크가 정말 없는가(머리의 missed 리마크도, `not vectorized:` 줄도 없어야 한다). 있으면 status 가 partial 이 아니라 yes 가 될 수 있다. gemm_ikj 는 여전히 status `yes` 여야 한다.
- (c) `versioned`·`basic block` 줄이 가리키는 줄.
- 다르면: 손으로 조립한 `tests/fixtures/perf_summary/c/*.build.log` 를 실제 로그(파일 이름을 `kernel.c` 로 바꿔 만든 것)로 바꾸고 `tests/test_perf_summary.py` 의 기대값이나 `harness/perf_summary.py` 의 `_gcc_vectorization` 을 고친다.

### 7.3 Halide·Numba inspect 픽스처 다시 만들기 (5분)

주의: `tests/fixtures/perf_summary/make_inspect_fixtures.py --write` 는 `tests/fixtures/perf_summary/{halide,numba}/*.import.json` 을 제자리에서 다시 쓴다. 이 파일들이 커밋된 뒤에 돌린다.

```sh
.venv/bin/python tests/fixtures/perf_summary/make_inspect_fixtures.py --write
git diff --stat tests/fixtures/perf_summary
git diff tests/fixtures/perf_summary | head -80
.venv/bin/python -m unittest tests.test_perf_summary
```

- 기대: target·시간(`autoschedule_ms`, `jit_ms`)·`num_threads`·numba 진단 머리말의 경로만 다르다. Windows 에서 같은 스크립트를 다시 돌렸을 때도 halide 파일의 시간 두 값만 달랐다.
- Halide 가 붙이는 자동 변수 이름(`j.v3` 등)이나 Mullapudi 스케줄이 다르면 기대값을 그 출력으로 고친다. 시간·경로만 다르면 변경을 버릴지(`git checkout -- tests/fixtures/perf_summary`) Linux 값으로 남길지 정한다.

### 7.4 워커의 inspect (5분)

```sh
HL_NUM_THREADS=4 .venv/bin/python -m harness.worker --backend halide --halide-mode auto --threads 4 \
  --artifact tests/fixtures/kernels/halide/gemm.py --mode import --tests tasks/gemm/tests/tiny.npz \
  --tolerance tasks/gemm/tolerance.yaml --signatures tasks/gemm/signatures.yaml \
  --inspect --inspect-dims '{"M":1024,"N":1024,"K":1024}' --out /tmp/lxplan/hl_import.json > /tmp/lxplan/hl_stdout.txt
wc -c /tmp/lxplan/hl_stdout.txt
.venv/bin/python -c "import json; i=json.load(open('/tmp/lxplan/hl_import.json'))['inspect']; print(i.get('dims'), 'vectorized' in str(i.get('loop_nest')), i.get('autoschedule_ms'), i.get('jit_ms'), i.get('inspect_error'))"
#   기대: stdout 0바이트, dims 1024³, True, 두 시간 기록, inspect_error None
```

### 7.5 파이프라인의 요약 (5분)

```sh
.venv/bin/python - <<'EOF'
import csv, json, collections
for rid in ("lx-gemm", "lx-all"):
    rows = [r for r in csv.DictReader(open("results/runs_dummy.csv", encoding="utf-8")) if r["run_id"] == rid and r["perf_summary"]]
    c = collections.Counter((r["language"], json.loads(r["perf_summary"])["vectorized"]["status"]) for r in rows)
    print(rid, len(rows), sorted(c.items()))
    print("  errors:", [(r["task"], r["language"]) for r in rows if "perf_summary error" in json.loads(r["perf_summary"]).get("notes", "")])
    for r in rows:
        ps = json.loads(r["perf_summary"])
        if r["language"] == "halide_auto" and r["task"] == "gemm":
            print("  halide_auto gemm loops:", [l.get("bounds") for l in ps.get("loops", [])][:3])
        if r["language"] == "numba" and r["task"] == "gemm":
            print("  numba gemm notes:", ps.get("notes", "")[:160])
EOF
```

- 빌드에 성공한 행마다 `perf_summary` 가 있고 `perf_summary error` notes 가 없다. c_seq 행의 status 가 `unknown` 이 아니다(build.log 가 전달된다).
- halide_auto 행의 `loops[].bounds` 상수 범위가 성능 크기(gemm 이면 `[0, 1023]`)여야 한다. tiny 크기(`[0, 2]`)면 요약이 측정된 스케줄과 다르다.
- numba 행의 notes 에 `numba diagnostics: N parallel loop(s) from prange, …` 가 나온다(`Parallel loop listing` 부분이 8000자 안에 있어야 한다). numba 행의 요약에 `threading_layer=omp`.

---

## 8. 분석과 그림 (10분)

```sh
# 분석 단위 테스트 (경고를 오류로)
.venv/bin/python -W error -m unittest tests.test_stats tests.test_aggregate tests.test_plots -v 2>&1 | tail -3      # 121개 OK

# 기본 출력이 그대로인가 (M1 실제 run; Windows 에서 다시 확인함)
.venv/bin/python -m analysis.aggregate --csv results/runs.csv
#   기대: speedup_median numpy 1.008, c_openmp 0.973, c_seq 0.331 (n=3, pass 3), '주:' 줄 없음
git show HEAD:results/runs_dummy.csv > /tmp/lxplan/runs_dummy_head.csv
.venv/bin/python -m analysis.aggregate --csv /tmp/lxplan/runs_dummy_head.csv --all-runs 2>/dev/null | head -1
#   기대(이 문서 작성 시점의 커밋 대상 CSV): '(선택: all, run 4개, 그룹 16개)' — 이 세션의 lx 행이 들어가기 전 CSV 로 본다

# 이 세션의 run
.venv/bin/python -m analysis.aggregate --csv results/runs_dummy.csv --run-id lx-all --ci --coverage
.venv/bin/python -m analysis.aggregate --csv results/runs_dummy.csv --run-id lx-repair --ci --by-turn --conditions
.venv/bin/python -m analysis.aggregate --csv results/runs_dummy.csv --run-id lx-cond --conditions

# 그림 (디스플레이 없이)
env -u DISPLAY .venv/bin/python -W error -m analysis.plots --csv results/runs_dummy.csv --run-id lx-repair --out /tmp/lxplan/plots-repair
env -u DISPLAY .venv/bin/python -W error -m analysis.plots --csv results/runs_dummy.csv --run-id lx-all --out /tmp/lxplan/plots-all --by-task
env -u DISPLAY .venv/bin/python -W error -m analysis.plots --csv results/runs_dummy.csv --run-id lx-all --out /tmp/lxplan/plots-pdf --format pdf
.venv/bin/python -m analysis.plots --csv results/runs_dummy.csv --all-runs --out /tmp/lxplan/plots-allruns
ls /tmp/lxplan/plots-*
```

- `--coverage`(lx-all): sort·bfs 의 Halide 칸 `n/a`, 이유가 행렬 아래.
- `--ci`: 그룹 표에 `pass_at_1`·`speedup_median` 의 95 % 부트스트랩 구간 열.
- 그림: rc 0, `Glyph … missing` 경고 없음. 수정 루프가 있는 run(lx-repair)은 `fig1_languages`, `fig2_error_types`, `fig3_condition_effects`, `fig4_repair_turns` 네 장(조건이 하나뿐인 run 에서는 fig3 을 건너뛴다는 한 줄). `--by-task` 는 `fig2_error_types_by_task`.
- `--all-runs`: stderr 에 `선택한 run 이 N개다`, fig4 는 (task, language, feedback)마다 선 하나이고 수정 루프를 돌리지 않은 run 의 평평한 선이 없다.
- 수정 루프가 거절·중단된 run 이면 feedback 행이 NA 이고 stderr 에 `비교하지 않는다` 경고 한 줄.
- 다를 때: `analysis/aggregate.py`·`analysis/plots.py` 는 Linux 의존이 없다. 값이 다르면 입력 행(§5.1 의 기대)부터 본다.

---

## 9. 측정 규약 자체와 large 형태 재조정

여기서부터는 실제 실험의 측정 조건을 정하는 단계다. 설정을 바꾸는 것은 측정 규약 변경이므로 사용자에게 먼저 묻고 `docs/decisions.md` 에 적는다(CLAUDE.md 원칙 5).

### 9.1 거버너·EPP·정적 검사 (5분)

§1.6 을 다시 하고 `.venv/bin/python -m harness.protocol --check` 가 rc 0 인지 본다. 그다음 `--strict-protocol` 의 두 경우를 본다.

```sh
.venv/bin/python -m harness.generate --provider dummy --run-id lx-strict --tasks gemm --languages numpy --n 1
.venv/bin/python -m harness.run --run-id lx-strict --strict-protocol; echo rc=$?            # 기대 0 (정적 플래그 없음)
sudo cpupower frequency-set -g powersave
.venv/bin/python -m harness.run --run-id lx-strict --strict-protocol --force; echo rc=$?    # 기대 4, 행을 다시 쓰지 않는다
sudo cpupower frequency-set -g performance
```

### 9.2 코어 고정과 스레드 수 (15분)

- `config/machine.yaml` 의 `p_core_cpus`·`e_core_cpus`·`thread_siblings` 로 코어 집합을 고른다. 지금 설정(`cores: "0,2,4,6"`, `single_core: "2"`, `threads: 4`)은 노트북의 P-core 4개(SMT 형제 제외, cpu0 의 IRQ 회피)다. 다른 CPU 면 같은 원칙(물리 코어당 1스레드, 하이브리드면 P-core 만, cpu0 회피)으로 새 값을 제안하고 사용자 확인을 받는다. 노트북에서 잰 M1·M2a 행과는 섞어 비교하지 않는다(2026-11-05 결정).
- 언어별 실제 스레드 수:

```sh
taskset -c 0,2,4,6 env NUMBA_NUM_THREADS=4 NUMBA_THREADING_LAYER=omp .venv/bin/python -m harness.worker --backend numba \
  --artifact tests/fixtures/kernels/numba/gemm.py --mode import --inspect --signatures tasks/gemm/signatures.yaml --out /tmp/lxplan/nb_import.json
.venv/bin/python -c "import json; d=json.load(open('/tmp/lxplan/nb_import.json')); print(d['inspect'].get('num_threads'), d['inspect'].get('threading_layer'), d.get('pinned_cpus'))"
#   기대: 4 omp [0, 2, 4, 6]
```

  halide·rust 는 `tools/devcheck.py kernel … --bench`(또는 `lx-all` 의 bench) 중에 `top -H -p <워커 pid>` 로 스레드 수가 `HL_NUM_THREADS`·`RAYON_NUM_THREADS` 와 같은지 본다. 워커는 이 일치를 강제하지 않는다(G2a open issue). rayon 이 taskset 코어 집합을 넘지 않는지도 본다.
- `OMP_PROC_BIND=true` 로 libgomp 가 주 스레드를 place 0 에 묶는 회귀는 `tests.test_worker.CKernelTest` 가 본다(§2).

### 9.3 부하와 선택적 대기 (15분)

```sh
stress-ng --cpu 4 --timeout 15s; sleep 5; cat /proc/loadavg
```

- 스스로 만든 부하가 `max_load_1m=2.0` 을 넘는지 본다. 병렬 샘플이 연달아 돌면 1분 부하가 2.0 을 넘어 `load_1m=` 플래그가 붙을 수 있다.
- 선택: `measurement.max_settle_s`(기본 0 = 기다리지 않음)를 켜면 동적 검사 전에 부하가 내려가길 그 시간만큼 기다린다. 측정 절차를 바꾸므로 사용자 확인 전까지 꺼 둔다. 시험해 볼 때:

```sh
mkcfg settle 's/^  # max_settle_s: 0 /  max_settle_s: 60 /'
.venv/bin/python -m harness.generate --provider dummy --run-id lx-settle --tasks gemm,jacobi2d --languages c_openmp,numpy --n 2
.venv/bin/python -m harness.run --config /tmp/lxplan/settle.yaml --run-id lx-settle
grep -ho '"settle_wait_s": [0-9.]*' results/built/lx-settle/*/*/*/sample_*/result.json | sort | uniq -c
#   기대: 느린 병렬 샘플 뒤에 settle_wait_s > 0, load_1m= 플래그 없음, pinned_busy_pct·sibling_busy_pct 는 낮음.
#   기본 설정(0)으로 돌린 run 은 모두 settle_wait_s 0.0
```

### 9.4 기준선 상태 (10분)

- gemm 기준선은 `baseline_min_gflops: 30` 보다 빨라야 한다(아니면 BLAS 정체 의심으로 exit 6). `lx-gemm` 행의 `baseline_time_ms`·`gflops` 와 `numpy_blas`(scipy-openblas 0.3.30)를 본다.
- cholesky 기준선(OpenBLAS `spotrf`)이 측정 스레드 설정에서 c_seq 더미보다 훨씬 빨라야 한다(Windows 에서는 부하 때문에 반대였다): `.venv/bin/python tools/devcheck.py baseline --task cholesky --bench` 와 §3.4 의 c_seq 값.
- `baseline_drift_pct`: M1 에서 기준선이 두 상태(약 8.5 ms, 10.3 ms)를 오가 drift 플래그가 9행 중 7행이었다. 거버너 `performance` 에서 이 현상이 사라지는지 `lx-gemm` 의 `baseline_drift_pct` 로 본다.
- fft 기준선 정의(사용자 확인 대상, §10)를 위한 확인: NumPy 2.3.5 Linux 휠에서 단정밀도 입력의 `np.fft.fft` 가 내부에서 배정밀도로 계산하는지.

```sh
.venv/bin/python - <<'EOF'
import numpy as np, time
n = 1 << 22
rng = np.random.default_rng(0)
x = (rng.standard_normal(2 * n).astype(np.float32) / np.sqrt(n))
xc = x.view(np.complex64)
a = np.fft.fft(xc)
b = np.fft.fft(xc.astype(np.complex128)).astype(np.complex64)
print("dtype", a.dtype, "bitwise equal to double-then-round:", np.array_equal(a.view(np.uint32), b.view(np.uint32)))
oc = np.empty_like(xc)
for name, f in (("plain", lambda: np.fft.fft(xc)), ("out+forward", lambda: (np.fft.fft(xc, out=oc, norm="forward"), oc.__imul__(n)))):
    f(); t = time.perf_counter(); f(); print(name, round((time.perf_counter() - t) * 1e3, 1), "ms")
EOF
#   Windows 값: True, plain 190 ms vs out+forward 104 ms (n = 2^22)
```

### 9.5 large 형태 재조정 (30–60분 + 결정)

새 과제 8개의 large 는 잠정값이다(`docs/decisions.md` 2026-11-20). M2a 4개 과제의 large 도 데스크탑으로 옮기면 다시 정한다는 2026-11-05 결정이 있다. 기준: **순차 C 기준 0.05–3 s, 과제당 npz 약 550 MB 이하**.

| 과제 | 지금 large (`gen_tests.py`) | npz(MiB) | Windows 참고 |
|---|---|---|---|
| gemm | 1024³ | 18 | 노트북 c_seq/0 77 ms, OpenBLAS 4스레드 8–15 ms |
| jacobi2d | 4096², T=20 | 196 | — |
| spmv_csr | 10⁶ 행, 평균 16 | 139 | — |
| dot | 2²⁵ | 265 | — |
| histogram | 2²⁶ 키, 256 구간 | 261 | c_seq 약 0.1 s |
| prefix_sum | 2²⁶ | 521 | 약 0.15 s; npz 가 상한 바로 아래 |
| nbody | 8192 | 0.4 | 순차 약 155 ms |
| sort | 2²⁴ | 130 | c_openmp 2 s 이내 |
| cholesky | 2048 | 49 | c_seq 291 ms |
| floyd_warshall | 1024 | 13 | — |
| bfs | 2²² 정점, 평균 차수 8 | 163 | 순차 기준선 1.27–1.31 s(유휴) |
| fft | 2²² | 98 | 기준선 190 ms |

절차:

1. §3.4 의 `cseq_bench.txt`(순차 C)와 `baseline_bench.txt` 를 보고 범위를 벗어나는 과제를 고른다. 노트북에서 정한 M2a 값은 같은 기준으로 다시 본다.
2. 바꿀 과제마다 `tasks/<task>/gen_tests.py` 의 `("large", {…})` 와 `tasks/<task>/spec.md` 의 성능 형태 문장을 같이 고친다. bfs 는 명세의 수치(최대 거리 10, 도달 불가 약 1,400)와 생성기의 `spec_max_dist`·`spec_unreachable_min`·`spec_unreachable_max` 를 같이 고쳐야 생성이 통과한다.
3. `.venv/bin/python tools/devcheck.py gen --task <task>` → 새 `test_hash`. §1.7 표와 `docs/decisions.md` 를 갱신한다. `spec.md` 가 바뀌면 그 과제의 프롬프트(`prompt_hash`)가 바뀐다. gemm·jacobi2d·spmv_csr 의 프롬프트 해시는 `tests/fixtures/prompt_hashes_m2a.json` 이 고정하므로 이 셋을 바꾸면 그 픽스처와 M2a 행의 비교 가능성도 같이 정해야 한다.
4. `make devcheck TASKS=<task>`, §3.4 의 bench 를 다시 돌린다. prefix_sum 은 형태를 키우면 550 MB 를 넘으므로 줄이는 방향만 가능하다.

### 9.6 행의 재현성 기록 (5분)

`lx-all` 행 몇 개에서 `compiler_version`(gcc·rustc), `cflags`, `march_resolved`, `threads`, `cores`, `omp_env`, `governor`, `epp`, `load_1m`, `taskset_used`, `protocol_ok`, `protocol_flags`, `hostname`, `cpu_model`, `numpy_blas`, `env_hash`, `test_hash` 가 채워졌는지 본다(`docs/protocol.md` §12).

---

## 10. 실제 API 호출 전에 정할 것

아래는 `docs/decisions.md` 에 **사용자 확인 필요** 로 표시됐거나(정리본은 2026-11-22 항목의 "6. 사용자 확인이 필요한 결정") 보고서가 확인을 요청한 항목이다. 실제 호출(`make real`) 전에 답을 받아 `docs/decisions.md` 에 적는다. 최신 목록은 `grep -n '확인 필요' docs/decisions.md` 로 본다.

| 항목 | 내용 | 근거 |
|---|---|---|
| Halide 계약 | 오토스케줄러 선택(Mullapudi2016), JIT 시간을 측정에서 빼는 것 — 측정 규약의 확장 | decisions 2026-11-20 |
| Numba 계약 | 스레딩 계층(`omp`) | decisions 2026-11-20, §1.4 (g) |
| 수정 루프의 피드백 범위 | 컴파일러·테스트 러너가 줄 법한 정보까지만, 하네스 패턴 이름·참조값은 주지 않음 | decisions 2026-11-20, §5.2 |
| `memory_error` 적용 규칙 | `runtime_crash`·`wrong_output` 에 ASan 보고가 있으면 재분류, 통과 샘플의 `pass` 는 유지, TSan 없음 | decisions 2026-11-20, §6.4 |
| 워커 결과 문서가 없을 때 | 결과 문서를 한 번도 못 쓰고 끝난 워커는 `harness_error`(행 없음, exit 5) | G4 보고서(분류 규칙의 적용) |
| import 뒤 비정상 종료 | 로드에 성공한(`load_ok`) 뒤의 비정상 종료·시간 초과는 빌드 실패로 적지 않는다 | G4 보고서 |
| bfs 명세의 정보 범위 | 차수 분포·도달 불가 정점 수·최대 거리 공개 | T4 보고서 |
| fft 허용오차 | rtol·atol 1e-3 유지 여부(점화식 트위들 구현이 경계에 걸림; §3.8). 권고는 `1e-4`(`tasks/fft/tolerance.yaml`·`spec.md` 를 고치면 fft 의 `prompt_hash` 가 바뀐다) | T4 보고서, decisions 2026-11-22 §6 A1 |
| fft 기준선 정의 | 단정밀도 결과지만 NumPy 가 내부에서 배정밀도로 계산(§9.4) | T4 보고서 |
| `prefix_sum` 의 int32 출력 | CLAUDE.md §4 는 정수 출력을 histogram·sort·BFS 로만 적었다 | decisions 2026-11-20 **[CLAUDE.md 이탈]** |
| 측정 머신 설정 | `measurement.cores`·`single_core`·`threads`, large 형태 재조정(§9.2, §9.5) | 2026-11-05 결정 |
| `measurement.max_settle_s` | 켤지(§9.3) | `config/experiment.yaml` 주석 |
| 린트 규칙 | clang-tidy 잡음 규칙을 빼거나 info 로 내릴지(§6.7) | G6 보고서 |
| numba 의 스레드 바인딩 | `measurement.omp_env_extra`(`OMP_PROC_BIND` 등)를 numba 에도 줄지(지금은 C 의 OpenMP 백엔드에만) | decisions 2026-11-22 §6 A3, §9.2 |
| Halide 빌드(JIT) 시간과 bench 제한 | 형태별 빌드 시간은 측정값에서 빠지지만 bench 제한(300 s)에는 들어간다(§3.5 의 JIT 시간) | decisions 2026-11-22 §6 A4 |
| Halide manual 의 `allow_race_conditions()` | 제자리 갱신 병렬화에 필요한 호출을 모델에게 허용할지 | decisions 2026-11-22 §6 A6 |
| 수정 루프의 컴파일 언어 크래시 정보 | C·Rust 크래시는 `crashed: signal …` 한 줄뿐이다(§5.2) | decisions 2026-11-22 §6 A8 |
| CLAUDE.md §6 의 린트 `summary` | 플러그인 계약은 범주 값 `null`(그 도구가 돌지 않음)을 허용한다. CLAUDE.md 는 사용자만 고친다 | decisions 2026-11-22 §6 A9 |
| 구현하면서 정한 것 | nbody·cholesky 허용오차, histogram tiny 형태, 시그니처 표기 허용 범위, 판정 규칙 적용, sanitizer 버퍼, NA 규칙, `--rebuild-csv`, 행의 `run_id`, `--skip-bench` 행, 수정 요청 어휘, Mullapudi2016·omp. 이견이 있을 때만 되돌린다 | decisions 2026-11-22 §6 B1–B12 |
| 보류 중인 M2a 호출 | 4과제 × 3언어 × 5샘플 = 60회. Linux 검증 뒤 확인을 받고 실행 | decisions 2026-11-22 §6 C1 |

비용 가드(코드가 강제하는 것):

- `harness.generate` 는 실제 제공자에서 먼저 `preflight()`(인증·모델명 확인)를 하고, 실패하면 exit 3 이다. 그다음 `[generate] <N> calls to <model> via <provider>, max_tokens=…, effort=…, temperature=…` 를 출력하고 `Proceed? [y/N]` 를 묻는다. 비대화형이고 `--yes` 가 없으면 exit 2 로 멈춘다. `harness.repair` 도 턴마다 같은 게이트를 쓴다.
- 이미 응답이 있는 샘플은 다시 부르지 않는다(`--force` 가 없으면). 실패·절단 샘플만 다시 부른다(`--no-retry-failed` 로 끈다).
- 지금 설정: `model.name: claude-sonnet-5-5`, `effort: medium`, `thinking: adaptive`, `max_tokens: 32000`, 키 변수 `ANTHROPIC_API_KEY`, `run.n_samples: 5`.
- 호출 수 계산(턴 0, 조건 하나): 지원 조합 수 × n. 과제 12 × 언어 7 − 4(Halide 의 sort·bfs) = 80 조합 → n=5 에서 400회. `grammar_hint=on` 은 힌트 파일이 있는 네 언어만이라 (12 × 4 − 4) × 5 = 220회. `args=keyword` 는 400회. `feedback=compiler+lint` 의 턴 0 은 복사(0회)이고 수정 턴은 실패 샘플 수 × 턴 수(최대 3)다. 실제 토큰 비용은 M1 실행(`20261105-real`)의 `tokens_out_mean`(c_openmp 2817, c_seq 2341, numpy 136)을 참고해 사용자와 따로 계산한다.
- 실제 호출은 `make real RUN_ID=<…> [STRICT=1] [TASKS=…] [LANGUAGES=…] [N=…]` 로 한다. CLAUDE.md 원칙 5 에 따라 비용이 드는 대량 호출은 먼저 사용자에게 묻는다.

---

## 부록 A. 확인되지 않은 가정

한 줄에 하나: 가정 — 틀리면 드러나는 모습 — 고칠 곳.

| 가정 | 틀리면 | 고칠 곳 |
|---|---|---|
| 심벌 없는 `.so` 에서 `nm -D --defined-only` 가 rc 0 | 심벌 없는 커널이 `signature_mismatch` 가 아니라 `ToolchainMissing`(run 중단) | `harness/build/c.py` 의 nm 규칙 (§1.4 a) |
| `gcc -print-file-name=libasan.so` 가 ELF | 모든 sanitizer 실행이 `incomplete:… ASan runtime does not come first` | `harness/safety.py` `sanitizer_env` (§1.4 b) |
| gcc 의 공유 라이브러리 sanitizer 빌드(`-Wl,--no-undefined` 를 뺀 명령)가 링크되고 libasan·libubsan 을 NEEDED 로 붙인다 | sanitizer 빌드 실패, 또는 선행 로드가 빠질 때 sanitize 단계가 `load_failed:…` | `harness/build/c.py` 의 sanitize 링크 플래그, `harness/safety.py` `sanitizer_env` (§1.4 c) |
| UBSan 보고가 `ubsan.<pid>` 또는 `asan.<pid>`·`safety.stderr.log` 에 간다 | `ubsan_findings` 가 0 으로 남음 | `harness/safety.py` 파서 (§6.4) |
| 정상 커널에서 CPython·NumPy 가 ASan 보고를 내지 않는다 | 정상 C 행의 `asan_findings` > 0 | `harness/safety.py` `asan_reports` (§6.4) |
| `handle_sigfpe=0` 이면 SIGFPE 가 ASan 보고 없이 시그널로 끝난다 | 0 나누기 커널이 asan 보고로 기록 | `harness/safety.py` 의 `ASAN_OPTIONS` (§6.4) |
| 워커의 `--sanitize` 버퍼(`posix_memalign`)를 ASan 이 가로챈다 | 출력 배열 한 칸 뒤 쓰기를 못 잡음 | `harness/worker.py` (§6.3) |
| gcc 의 `-fopenmp` 없는 경고가 `ignoring '#pragma omp …'` 형식 | 무시된 pragma 판정 누락 | `harness/perf_summary.py` 정규식 (§1.4 d) |
| `loop vectorized` 리마크의 줄:열이 `for` 조건식의 비교 연산자 위치 | 벡터화된 루프를 잘못 짚음 | `_gcc_vectorization`, `tests/test_perf_summary.py` (§7.2) |
| memset 으로 바뀐 초기화 루프에는 리마크가 없다 | `vectorized.status` 가 partial 대신 yes | 같은 곳 (§7.2) |
| Halide Linux 휠의 오토스케줄러 플러그인 파일 이름이 `*autoschedule_mullapudi2016*.so` | 플러그인을 못 찾아 halide_auto 가 exit 7 | `harness/kernels/halide_pipe.py` `PLUGIN_SUFFIXES` (§1.4 f) |
| halide 22.1.0·numba 0.68.1 의 Python 3.14 Linux 휠이 있다 | `make venv` 실패 | `requirements.txt` + 프롬프트 버전 서술 (§1.2) |
| Numba `omp` 계층이 Linux 에서 libgomp 로 올라간다 | 첫 병렬 호출에서 threading layer 오류, run 은 exit 7 | `config/experiment.yaml` `NUMBA_THREADING_LAYER` (§1.4 g) |
| Halide 의 stdout 경고가 커널 stdout 으로 기록되지 않는다 | prefix_sum/halide_auto 가 출력 계약 위반으로 기록 | `harness/worker.py` 의 stdout 처리 (§4.2) |
| Halide inspect 가 성능 형태(`--inspect-dims`)로 모인다 | `loops[].bounds` 가 tiny 크기 | `harness/run.py` 의 import 단계 (§7.5) |
| Halide JIT 코드 안의 크래시(realize 중 SIGSEGV)가 워커의 시그널 종료(rc<0)로 드러나고 `check.json` 의 `in_progress` 가 그 테스트를 가리킨다 | 크래시가 `runtime_crash` 가 아니라 다른 분류나 하네스 오류로 기록 | `harness/worker.py`, `harness/run.py` 의 runtime_crash 규칙. 그런 커널이 생기면 `<샘플 디렉터리>/check.json`·`check.stderr.log` 로 본다(Windows 에서는 만들 수 없었다) |
| 워커가 halide 의 실제 스레드 수를 `HL_NUM_THREADS` 로 둔다(강제하지 않음) | 행의 `threads` 와 실제가 다름 | `harness/worker.py` (§9.2) |
| rayon 이 `RAYON_NUM_THREADS` 와 taskset 을 따른다 | rust 행의 스레드 수가 다름 | 설정 `languages.rust.env` (§9.2) |
| `cargo fetch --locked` 뒤 `--offline --locked` 빌드가 최소 환경(`toolchain_env`)에서 돈다 | rust 가 exit 7 | `harness/protocol.py` `toolchain_env`, `TOOLCHAIN_ENV_KEYS` (§2.2) |
| run 마다 공유하는 `CARGO_TARGET_DIR` 에서 동시 빌드 경쟁이 없다(`fcntl.flock`) | 다른 샘플의 `libkernel.so` 가 섞임 | `harness/build/rust.py` 잠금 (§4.2) |
| Rust 의 미정의 외부 심벌은 링크에서 실패한다 | 로드 단계 실패로 서술이 다름(분류는 같음) | `harness/build/rust.py` 의 error_stage (§3.7) |
| `RUSTFLAGS` 가 Rust 빌드 환경에 전달된다 | §4.6 의 망가진 링커 확인이 다른 결과 | `toolchain_env` (§4.6) |
| clang-tidy 가 gcc 의 `omp.h` 를 `-isystem` 으로 해석한다 | clang-tidy 가 빠지고 세 범주가 null | `plugins/lint/run_lint.py` `openmp_header_dir` (§6.7) |
| cppcheck 의 `--library=posix`·clang-tidy 의 `--config={}` 를 설치 버전이 받는다 | 그 도구가 `tool` 에서 빠짐 | `plugins/lint/run_lint.py` 명령 (§6.7) |
| cppcheck XML 의 첫 location 이 오류 지점이다 | 린트 줄 번호가 틀림 | 같은 파일의 XML 파서 (§6.7) |
| clang-tidy 가 커널당 20 s 안에 끝난다 | 린트 시간 초과, 범주 null | `TOOL_TIMEOUT_S`, `rules.yaml` (§6.7) |
| ruff 가 `--select` 의 코드를 전부 안다 | `tool=baseline:pyflakes`, `EndToEndPythonTest` 실패 | `plugins/lint/rules.yaml` (§2.2) |
| openai SDK 가 `DefaultHttpxClient(transport=MockTransport)` 를 받는다 | `OpenAICompatMockTest` 실패(배선만) | `tests/test_llm_client.py` (§2.2) |
| libgomp 의 `OMP_PROC_BIND=true` 가 주 스레드 친화도를 좁히는 문제를 워커가 피한다 | OpenMP 커널이 한 코어에서 돎 | `harness/worker.py`, `CKernelTest` (§2) |
| 거버너 `performance` 에서 M1 의 두 상태 기준선(8.5/10.3 ms)이 사라진다 | `baseline_drift` 플래그 다수 | 측정 머신 설정 (§9.4) |
| 병렬 샘플이 스스로 만든 부하가 `max_load_1m=2.0` 아래로 빨리 내려간다 | `load_1m=` 플래그, `protocol_ok=0` 다수 | `measurement.max_settle_s` 결정 (§9.3) |
| 지금 large 형태가 측정 머신에서 0.05–3 s 안 | bench 예산 초과 또는 ms 미만 측정 | `tasks/<task>/gen_tests.py`·`spec.md` (§9.5) |
| `fft` 의 `out_ref` 마지막 비트 차이가 판정에 영향이 없다 | fft 경계 판정이 머신마다 다름 | `tasks/fft/tolerance.yaml` (§3.8) |
| gcc 에서 bfs·fft 의 OpenMP pragma 가 경고 없이 받아들여진다 | 빌드 경고·판정 차이 | 더미 `harness/dummy_responses/{bfs,fft}/c_openmp/0.md` (§3.3) |
| glibc 의 `aligned_alloc`·`qsort` 가 더미·기준선의 사용 방식에서 동작한다 | histogram·nbody·sort 의 C 실패 | 해당 더미·기준선 (§3.3) |

## 부록 B. Windows 검증의 한계

- **C 컴파일러.** gcc 대신 `python -m ziglang cc` 로 빌드했다. `c_seq`·`c_openmp` 의 기본 모드는 **직렬**이다: `-fopenmp` 가 없어 OpenMP pragma 가 무시되고 `<omp.h>` 는 `tools/devcheck_shim/omp.h` 스텁, Windows 에 없는 `aligned_alloc`/`posix_memalign` 은 `shim.h` 가 메웠다. 논리만 확인한 것이다. gcc 고유 동작(`-fisolate-erroneous-paths-dereference`, 경고 문구, `-fopt-info-vec` 리마크, ifunc)과 POSIX API 는 확인하지 못했다.
- **실제 스레드 모드.** `tools/devcheck.py --cc zig-omp` 는 clang `-fopenmp` + LLVM OpenMP 런타임(`libomp140.x86_64.dll`)이다. 측정 경로의 gcc + libgomp 가 아니므로 경쟁 상태를 일찍 드러내는 용도일 뿐 Linux 확인을 대신하지 않는다. 이 모드는 Linux 에서 자동으로 고르지 않는다.
- **코어 고정·규약.** taskset, 거버너·EPP, `/proc/loadavg`, sysfs 가 없다. Windows 에서 만든 행에는 `platform=win32`, `governor=unknown`, `taskset_missing` 플래그가 붙어 `protocol_ok=0` 이다. 모든 시간은 의미가 없다.
- **프로세스 관리.** `procutil` 은 POSIX 가 아니면 프로세스 그룹 정리 없이 직접 자식만 `kill()` 한다. 워커는 `resource`/`prctl` 없이 돈다(`RLIMIT_AS` 없음, `PR_SET_PDEATHSIG` 없음). 시그널 종료 코드(음수 rc)는 Linux 에서만 나온다.
- **파이프라인.** C 언어는 `harness.run` 을 통과하지 못했다(gcc 가 없어 exit 7). 파이프라인 검증은 numpy·numba·halide_*·rust 에 한정된다.
- **sanitizer·린트.** ASan/UBSan, cppcheck, clang-tidy 는 고정 텍스트 픽스처로 파서만 검증했다. 워커 호출부는 zig DLL 로만 확인했다.
- **Rust.** MSVC 툴체인(링커 오류 동작이 다르다), `kernel.dll`, 전체 환경 상속. Linux 의 `toolchain_env`(최소 환경 + `TOOLCHAIN_ENV_KEYS`), `libkernel.so`, `fcntl.flock`, 시간 초과 시 그룹 정리는 확인하지 못했다.
- **SDK.** Windows 개발 venv 에는 anthropic·httpx2·openai 가 없어 `test_llm_client` 16개가 skip 됐다.
- **줄바꿈.** Windows 작업 트리의 프롬프트 템플릿은 CRLF 였다(universal newline 으로 읽음). LF 체크아웃에서의 `prompt_hash` 는 `PromptHashTest` 가 Linux 에서 처음 확인한다. `harness/build/rust_template/Cargo.toml`·`Cargo.lock` 은 `.gitattributes` 로 LF 고정이다.
