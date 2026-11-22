# 린트 플러그인 — 기성 린터 베이스라인

`CLAUDE.md` §6 의 린트 플러그인 계약을 기성 린터(ruff·pyflakes, cppcheck·clang-tidy)로 채운 구현이다. 하네스는 이 디렉터리의 내부를 모르고 아래 CLI 와 출력 JSON 만 안다. 같은 CLI 를 지키는 다른 구현(인지프 ML lint)으로 바꿔 끼울 수 있다(§7).

| 파일 | 역할 |
|---|---|
| `run_lint.py` | 플러그인 본체. 하네스를 임포트하지 않는 독립 프로그램이다(표준 라이브러리 + PyYAML). |
| `rules.yaml` | 도구별 규칙 선택과 규칙 ID → 범주·심각도 대응표. 항목마다 근거 한 줄이 있다. |

## 1. 계약

```
PYTHON plugins/lint/run_lint.py --language <lang> --file <path> --task <task> --signature <json> --out <json>
```

- `--language`: 실험 언어 이름(`numpy`, `c_openmp`, …). `--file`: 검사할 소스 파일.
- `--task`, `--signature`: 계약의 인자다. 이 베이스라인은 쓰지 않는다(`--signature` 는 JSON 객체인지만 확인한다). ML lint 가 시그니처 위반을 볼 때 쓰는 자리다.
- `--out`: 출력 JSON 경로(원자적으로 쓴다). 하네스는 이 파일을 샘플 디렉터리의 `lint.json` 으로 남긴다.
- 종료 코드: `0` 정상 — 도구가 없거나 실패해도 0 이고 출력의 `tool` 이 무엇이 돌았는지 말해 준다. `2` 인자·입력 파일·`rules.yaml` 오류(출력을 쓰지 않는다).

출력:

```json
{
  "tool": "baseline:ruff+pyflakes",
  "findings": [
    {"rule": "ruff/F821", "severity": "error", "line": 5, "message": "Undefined name `acc`", "category": "memory"}
  ],
  "summary": {"memory": 1, "numeric": 0, "concurrency": 0, "shape_dtype": 0, "api_misuse": 0, "performance": 0},
  "unmapped": 0,
  "versions": {"ruff": "0.17.6", "pyflakes": "4.1.0"}
}
```

- `tool`, `findings`, `summary` 가 계약이다. `summary` 는 여섯 범주를 전부 가지며 각 값은 `findings` 를 범주별로 센 값이거나 `null` 이다. `null` 은 "검사하지 않음": 그 범주를 볼 수 있는 도구가 하나도 끝까지 돌지 않았다(그 범주의 finding 이 있으면 위반). 하네스(`harness.safety.parse_lint_output`)가 이것을 검사하고, 어긋나면 그 샘플의 린트 결과를 버린다. 하네스는 `null` 범주의 `lint_<범주>` 열을 0 이 아니라 NA 로 둔다(`lint_findings_total` 은 검사한 범주의 합).
- 이 베이스라인에서 도구가 볼 수 있는 범주 = `rules.yaml` 의 그 도구 표에 있는 범주(`tool_categories`): ruff·clang-tidy 는 여섯 범주 전부, pyflakes 는 `pyflakes.messages` 가 가리키는 코드의 범주(memory·numeric·performance), cppcheck 는 memory·numeric·shape_dtype. 예: c_openmp 에서 clang-tidy 가 빠지고 cppcheck 만 돌면 concurrency·api_misuse·performance 는 `null` 이다.
- `tool`: `baseline:<분석을 마친 도구들>`. 하나도 없으면 `"none"`(summary 는 전부 `null`) — 그때 하네스는 결과 행의 `lint_*` 열을 0 이 아니라 NA 로 둔다.
- `findings[].rule`: `<도구>/<도구의 규칙 ID>`. pyflakes 는 규칙 ID 가 없어 같은 뜻의 ruff F 코드로 적는다(`pyflakes/F841`).
- `findings` 의 순서: 심각도(`error` → `warning` → `info`), 같으면 줄 번호. 하네스의 `lint_messages` 는 수정 프롬프트(`feedback=compiler+lint`)에 넣을 `message` 를 앞에서부터 20개까지 고르므로 이 순서가 우선순위다.
- `unmapped`(계약 밖의 추가 키): `rules.yaml` 에 없는 규칙이어서 버린 보고의 수. 어떤 규칙이었는지는 stderr 한 줄(`[lint] 표에 없는 규칙의 보고를 버렸다: ruff/B007 x1, …`)에 나온다. 하네스는 플러그인의 stderr 를 `lint.json` 옆의 `lint.stderr.log` 로 남긴다.
- `versions`(추가 키): 분석을 마친 도구의 버전. 같은 코드라도 도구 버전이 바뀌면 보고가 달라지므로 샘플마다 남긴다.

## 2. 언어별 도구

| 언어 | 도구(이 순서로 돈다) | 실행 |
|---|---|---|
| `numpy`, `numba`, `halide_auto`, `halide_manual` | ruff, pyflakes | `sys.executable -P -m ruff check --output-format json --isolated --no-cache --ignore-noqa --target-version py<XY> --select … <file>`, `sys.executable -P -m pyflakes <file>` |
| `c_seq`, `c_openmp` | cppcheck, clang-tidy | `cppcheck --enable=… --language=c --std=c11 --library=posix --xml --quiet <file>`, `clang-tidy <file> --checks=… --config={} --quiet -- -std=gnu11 -march=native [-isystem <omp.h 디렉터리>] [-fopenmp]` |
| `rust` | 없음 | 항상 `tool: "none"` |

- 설치되지 않은 도구는 조용히 건너뛴다(ruff·pyflakes 는 이 인터프리터에서 임포트 가능한지, cppcheck·clang-tidy 는 `PATH` 에 있는지). 모르는 언어 이름은 stderr 에 한 줄 남기고 `tool: "none"` 을 낸다.
- 도구는 서브프로세스로 돌리고 하나에 20초(`TOOL_TIMEOUT_S`)를 준다. 언어당 도구가 둘이므로 합이 하네스의 `lint.timeout_s`(기본 60초)보다 작다 — 도구가 멈춰도 플러그인은 나머지 결과를 내고 끝난다.
- 주변 환경이 결과를 바꾸지 못하게 한다: ruff `--isolated --no-cache`(설정 파일·캐시 무시), `--ignore-noqa`(생성된 코드의 `# noqa` 무시), clang-tidy `--config={}`(`.clang-tidy` 를 찾지 않음)과 `--`(컴파일 DB 를 찾지 않음). `-P` 는 작업 디렉터리(LLM 이 쓴 `kernel.py` 가 있는 곳)를 `sys.path` 에 넣지 않는다.
- `c_openmp` 는 clang-tidy 의 컴파일 인자에 `-fopenmp` 가 붙는다. clang 에 `omp.h` 가 없을 수 있으므로(예: `libomp-dev` 미설치 — 실험 머신의 clang 21) 설정의 C 컴파일러(하네스가 환경변수 `CC` 로 `build.cc` 를 넘긴다; 없으면 `gcc`)의 `<cc> -print-file-name=include` 디렉터리에 `omp.h` 가 있으면 그 **한 파일만** 임시 디렉터리에 복사해 `-isystem <그 디렉터리>` 로 준다. 컴파일러의 include 디렉터리를 통째로 넣지 않는 것은 그 안의 인트린식 헤더(`immintrin.h` 등)를 clang 이 해석하지 못하기 때문이다. 쓴 `omp.h` 의 경로는 stderr(`lint.stderr.log`)에 한 줄 남는다. 그래도 컴파일 오류가 나면 clang-tidy 결과 없이 cppcheck 만 남고, clang-tidy 만 보는 범주는 `null` 이 된다(§1, §4).
- clang-tidy 의 컴파일 인자(`rules.yaml` 의 `compile_args`)는 `config/experiment.yaml` 의 `build.base_flags`/`cflags` 와 따로 적혀 있다. 빌드 플래그를 바꾸면 여기도 맞춘다.

## 3. `rules.yaml`

```yaml
<도구>:
  <규칙 선택>            # ruff: select / cppcheck: enable / clang-tidy: checks, compile_args — 도구 자신의 문법 그대로
  rules:
    F821:    {category: memory, severity: error, reason: "…"}        # 정확 일치
    "PERF*": {category: performance, severity: info, reason: "…"}    # 접두 일치
```

- 정확 일치가 접두 일치보다, 긴 접두가 짧은 접두보다 우선한다. 예: `clang-analyzer-core.DivideZero`(numeric)가 `clang-analyzer-core.*`(memory)보다 먼저다.
- `category` 는 `CLAUDE.md` §6 의 여섯 범주 중 하나다. 범주 정의에 분명히 들어맞는 규칙만 넣는다. 순수 스타일·위생 규칙(안 쓰는 임포트 등)과 범주가 없는 일반 논리 오류는 넣지 않는다 — 넣지 않은 규칙의 보고는 `unmapped` 로만 세어진다.
- `severity`: `error` = 그 줄이 실행되면 반드시 예외·미정의 동작·누수 / `warning` = 조건에 따라 결함 / `info` = 냄새·방어적 관례. 도구가 붙인 심각도가 아니라 이 표의 값을 쓴다.
- `rules` 의 키는 도구의 규칙 선택에 자동으로 더해진다(ruff `--select`, clang-tidy `-checks`). 표에 줄을 더하면 그 규칙이 켜진다.
- `pyflakes.messages`: pyflakes 메시지의 정규식 → ruff F 코드. ruff 표에 있는 코드만 적는다.
- 형식이 틀리면(`load_rules`) 플러그인은 종료 코드 2 로 끝나고 하네스는 그 샘플의 린트를 빈 결과로 처리한다.

표를 넓히는 순서: `lint.json` 의 `unmapped` 가 0 이 아닌 샘플을 고른다 → 그 샘플의 `lint.stderr.log` 에서 규칙 ID 를 본다 → 범주가 분명하면 `rules.yaml` 에 근거와 함께 한 줄 더한다 → `tests/test_lint_plugin.py` 를 돌린다. 표를 바꾸면 그 뒤의 `lint_*` 열은 이전 행과 같은 기준이 아니다(결정 기록에 남긴다).

## 4. 판정 규칙

- **분석 실패는 "보고 0건"이 아니다.** 도구가 파일을 해석하지 못하면 그 도구는 돌지 않은 것으로 친다: 시간 초과, 비정상 종료, 읽을 수 없는 출력, ruff 의 `invalid-syntax`, pyflakes 의 구문 오류, cppcheck 의 `syntaxError`·`unknownMacro` 등, clang-tidy 의 `error:` 진단(컴파일 오류). stderr(`lint.stderr.log`)에 `[lint] <도구>: <사유>` 한 줄이 남고 `tool` 에서 그 도구가 빠진다. 남은 도구가 볼 수 없는 범주는 `summary` 에서 `null` 이다(0 이 아니다).
- **도구 간 중복 병합.** 두 도구가 같은 줄에서 같은 범주를 보고하면 같은 결함으로 보고 먼저 돈 도구의 보고만 남긴다. 한 도구가 같은 줄·같은 범주에 여러 건을 내면 그 수는 유지된다. `unmapped` 도 같은 방식으로 합쳐 센다.
- **범위.** clang-tidy·cppcheck 는 검사한 파일 자체의 줄에 붙은 보고만 쓴다(하네스가 만든 `kernel_args.h` 안의 보고는 뺀다).

## 5. 이 베이스라인이 보지 못하는 것

기성 린터의 한계가 곧 ML lint 가 채울 자리다.

- **concurrency**: OpenMP 데이터 경쟁·리덕션 누락을 찾는 규칙이 없다. clang-tidy 는 `default(none)` 누락(`openmp-use-default-none`)만 알려 준다. Python 쪽은 `with threading.Lock():` 오용 하나뿐이고 Numba `prange` 는 보지 않는다.
- **performance(Python)**: 배열 원소 단위 Python 루프를 잡는 규칙이 없다(ruff `PERF` 는 리스트·딕셔너리 생성 루프만). 그 관찰은 `harness/perf_summary.py` 가 한다.
- **api_misuse**: 입력 배열 변형, 요구 시그니처 위반을 보지 않는다(`--signature` 미사용).
- **numeric**: float32 누적 정밀도, 0 근처 나눗셈 같은 수치 위험을 보지 않는다.
- **rust**: 도구가 없다. `feedback=compiler+lint` 조건의 린트 절은 항상 비어 있다.

## 6. 손으로 돌려 보기

```
PYTHON plugins/lint/run_lint.py --language numpy --file kernel.py --task gemm --signature '{}' --out /tmp/lint.json
PYTHON -m harness.safety --lint kernel.py --language numpy      # 하네스가 부르는 경로 그대로(계약 검사 포함)
PYTHON -m unittest tests.test_lint_plugin -v
```

## 7. 다른 구현으로 바꾸기

`config/experiment.yaml` 의 `lint.plugin` 을 새 스크립트 경로로 바꾼다. 새 구현이 지킬 것은 §1 의 CLI 와 `tool`·`findings`·`summary` 뿐이다(`unmapped`·`versions` 는 선택). 하네스는 `PYTHON_EXE` 로 스크립트를 띄우고, 종료 코드가 0 이 아니거나 `lint.timeout_s` 를 넘기거나 출력이 계약에 어긋나면 그 샘플의 린트를 빈 결과(열은 NA)로 두고 경고한 뒤 실험을 계속한다.
