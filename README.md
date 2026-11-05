# LLM 생성 HPC 코드 평가 하네스 (졸업프로젝트, M1)

경희대 컴퓨터공학부 졸업프로젝트 **"프로그래밍 언어 선택과 언어 특성이 LLM 생성 HPC 코드의 정확성·성능·안전성에 미치는 영향 — 실증 분석(예비 연구)"** 의 실험 하네스다. 같은 과제(M1: 단정밀도 GEMM)를 여러 언어(`numpy`, `c_seq`, `c_openmp`)로 LLM에 생성시켜 **추출 → 빌드 → 격리 실행 → 정확성 판정 → 시간 측정 → CSV 행** 까지를 자동으로 돌리고, 결과를 pass@k·속도비·토큰·오류 유형으로 집계한다. 핵심 가설은 "성능 격차는 추상화 수준이 아니라 성능 결정(루프 순서·타일링·벡터화·병렬화)을 LLM이 직접 내려야 하는가로 설명된다"이다(`CLAUDE.md` §1).

- 프로젝트 지시서: `CLAUDE.md` (수정 금지)
- 설계 계약(모듈 인터페이스·JSON 키·CSV 열): `docs/harness_design.md`
- 결정 기록: `docs/decisions.md` · 측정 규약: `docs/protocol.md` · 오류 분류표: `docs/error_taxonomy.md`

## 1. 저장소 구조 (M1)

```
config/experiment.yaml      실험 설정(모델·n·언어·빌드 플래그·측정 규약)   config/machine.yaml  머신 프로브 결과(생성됨)
tasks/gemm/                 spec.md(프롬프트에 삽입), signatures.yaml, reference.py, gen_tests.py, tolerance.yaml, baseline/baseline.py
prompts/system.md, prompts/lang/{numpy,c_seq,c_openmp}.md
harness/                    generate(LLM 호출) → extract → build/{c,python} → run(오케스트레이션) ← worker(LLM 코드 실행 유일 장소)
                            llm_client, protocol(측정 규약), classify(오류 분류), schema(CSV 스키마), procutil(격리 실행), dummy_responses/
analysis/aggregate.py       집계         tests/test_*.py  단위 테스트         results/  raw/, built/, runs.csv, runs_dummy.csv
```

## 2. 설치

필요: Linux, Python 3.14, gcc 15(`-fopenmp` 가능), `taskset`(util-linux). sudo는 거버너 설정에만 쓰고 하네스는 시스템 설정을 바꾸지 않는다.

```sh
make venv                 # python3 -m venv .venv && .venv/bin/pip install -r requirements.txt  (다른 인터프리터: make venv SYSPY=python3.14)
make probe                # config/machine.yaml 생성 (CPU 토폴로지·거버너·BLAS 정체·컴파일러)
make tests                # tasks/gemm/tests/{tiny,rect,small,large}.npz + manifest.json (시드 고정, 바이트 동일 재생성)
make unittest             # tests/ 단위 테스트
```

**모든 Python 실행은 `.venv/bin/python`** 이다. `.venv`는 격리형(시스템 site-packages 미사용)이어야 한다 — 시스템 `python3-numpy`는 참조 BLAS(netlib)라 1024³ GEMM이 ~250 ms이고, pip 휠 `numpy==2.3.5`(OpenBLAS 내장)는 4스레드 ~15 ms(EPP `power`)·~8 ms(EPP `performance`)다 — 절대 시간은 주파수 상태에 따라 달라지므로 비교는 대응 기준선 대비 속도비로 한다(`docs/harness_design.md` §0). 하네스는 BLAS 정체를 행마다 기록하고 OpenBLAS가 아니면 run을 중단한다.

측정 전에 거버너·EPP를 `performance`로 두고 확인한다(`docs/protocol.md` §13):

```sh
sudo cpupower frequency-set -g performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference
.venv/bin/python -m harness.protocol --check      # 플래그가 있으면 exit 1
```

## 3. 더미 파이프라인 (API 키 없이 전체 경로 검증)

```sh
make dummy                # RUN_ID=dummy-<timestamp> 로
#   .venv/bin/python -m harness.generate --provider dummy --run-id $RUN_ID
#   .venv/bin/python -m harness.run --run-id $RUN_ID
#   .venv/bin/python -m analysis.aggregate --csv results/runs_dummy.csv --run-id $RUN_ID
make dummy RUN_ID=my-dummy
```

`harness/dummy_responses/gemm/<language>/<k>.md`의 고정 응답(샘플 i는 `i mod 파일 수`)으로 생성 단계를 대신한다. 기대 결과표(pass/`numeric_instability`/`arg_order`/`compile_error`/`runtime_crash`/`race_condition` 각 1건 이상)는 `harness/dummy_responses/gemm/README.md`와 `docs/error_taxonomy.md` §8에 있다. 더미 행은 `provider=dummy`로 **`results/runs_dummy.csv`에 분리 저장**되며 실제 결과와 섞이지 않는다(키가 없어도 실제 제공자에서 dummy로 자동 폴백하지 않는다).

## 4. 실제 실행

```sh
export ANTHROPIC_API_KEY=...          # config/experiment.yaml model.anthropic.api_key_env
# 선택: LLM_PROVIDER=anthropic|openai_compat|dummy, LLM_MODEL=<모델명> 이 experiment.yaml 을 덮어쓴다.
# openai_compat 은 LLM_BASE_URL 과 OPENAI_API_KEY 가 필요하다.
make real RUN_ID=20261105-opus55            # generate(확인 프롬프트) → run → aggregate
make real RUN_ID=20261105-opus55 STRICT=1   # run --strict-protocol: run 시작의 정적 규약 플래그(거버너·EPP·taskset)가 있으면 중단(exit 4)
```

- **확인 게이트**: `harness.generate`는 `preflight()`(모델 조회, 과금 없음) 뒤 계획(`<n> calls to <model> via <provider>, max_tokens=…, effort=…`)을 보여주고 `Proceed? [y/N]`을 묻는다. M1 기본 설정은 3샘플 × 3언어 = **9회 호출**(`max_tokens=32000`, 스트리밍)이다. 비대화형(stdin이 tty가 아님)에서는 `--yes`가 없으면 exit 2로 끝나므로, 스크립트에서는 `make` 대신 `.venv/bin/python -m harness.generate --run-id ID --yes` 를 직접 쓴다. 치명 API 오류(인증·404·400)는 raw를 저장한 뒤 즉시 중단(exit 3)한다.
- 재실행: 같은 `RUN_ID`로 다시 돌리면 정상 응답(`error` 없음, `stop_reason ∈ {end_turn, stop_sequence}`)은 건너뛰고 실패·절단 샘플만 재호출한다(`--no-retry-failed`로 끔, `--force`로 전부 재호출; 기존 파일은 `sample_<i>.prev<N>.json`으로 보관). `harness.run`도 raw 해시가 같은 샘플은 건너뛴다(`--force`).
- 부분 실행: `harness.run --run-id ID [--languages a,b] [--samples 0,1] [--skip-bench] [--no-baseline]`.
- 규약 플래그는 경고와 함께 행의 `protocol_flags`에 기록되고(`protocol_ok=0`) 실험은 계속된다. `STRICT=1`(`--strict-protocol`)은 **run 시작의 정적 검사**(거버너 `powersave`, EPP, `taskset` 없음)에만 적용돼 그때 exit 4로 중단한다; bench 직전의 **동적** 플래그(부하 `load_1m`, 고정/형제 코어 사용률, 거버너 재확인)와 기준선 `baseline_drift`·`paired_baseline_failed`, run 끝의 `model_served_mixed`는 STRICT=1이어도 **기록만** 되고 run은 멈추지 않는다 — "STRICT=1이면 플래그 없는 행만 남는다"는 보장이 아니므로 분석에서는 행의 `protocol_ok`/`protocol_flags`로 가려낸다(`docs/protocol.md` §8–§10). BLAS가 OpenBLAS가 아니면 STRICT와 무관하게 항상 중단된다.
- 거부(refusal)는 다른 모델로 대체 실행하지 않고 `stop_reason=refusal`로만 기록한다.

## 5. 산출물 위치

| 경로 | 내용 |
|---|---|
| `results/raw/<run_id>/gemm/<language>/<cond_tag>/sample_<i>.json` | 생성 원문: 프롬프트·응답 텍스트·토큰·stop_reason·모델·설정 스냅샷. `cond_tag = args-positional_hint-off_fb-compiler` |
| `results/built/<run_id>/gemm/<language>/<cond_tag>/sample_<i>/` | `kernel.c`/`kernel.py`, `kernel.so`, `build.json`, `build.log`(벡터화 리마크 포함), `import/check/bench.json`, `<stage>.{stdout,stderr}.log`, **`result.json`**(행 + 단계별 상세 + `raw_sha256`) |
| `results/built/<run_id>/gemm/baseline/` | `bench_run.json`(run 시작 기준선), `bench_1t.json`(1스레드), `paired_sample_<lang>_<i>.json`(샘플별 대응 기준선) |
| `results/runs.csv` | 실제 제공자 행(git에 올리는 유일한 결과) |
| `results/runs_dummy.csv` | `provider=dummy` 행 |
| `config/machine.yaml` | 머신 프로브 결과 |

`result.json`이 진실의 원천이다. CSV 헤더가 스키마와 달라지면 `.venv/bin/python -m harness.run --rebuild-csv`로 다시 만든다. 같은 행 키 `(run_id, task, language, args, grammar_hint, feedback, sample_idx, turn)`는 교체(upsert)된다.

## 6. 결과 행의 주요 열 (`harness/schema.py` `COLUMNS`, 설계 §11)

직렬화: 불리언 `1`/`0`, NA 빈칸, 리스트/딕트 JSON 문자열.

| 열 | 의미 |
|---|---|
| `run_id`, `created_at`, `task`, `language`, `backend`, `args`, `grammar_hint`, `feedback`, `sample_idx`, `turn` | 행 키와 조건. M1은 `args=positional`, `grammar_hint=off`, `feedback=compiler`, `turn=0` |
| `provider`, `model`, `model_served`, `sampling`, `effort`, `max_tokens`, `temperature`, `seed`, `prompt_hash` | 요청 모델과 실제 응답 모델, 샘플링 설정 JSON(`temperature_effective: provider_default` 등), 프롬프트 해시(system + user; 바뀌면 실행 간 비교 불가) |
| `stop_reason`, `stop_details_category` | `end_turn` / `max_tokens`(절단) / `refusal`(거절 범주) / … |
| `tokens_in`, `tokens_out`, `tokens_thinking`, `tokens_text`, `n_thinking_blocks`, `latency_ms` | **`tokens_out`은 thinking 포함 총 출력 토큰(과금 기준)**, `tokens_text`는 텍스트 블록 몫. dummy는 단어 수(가짜) |
| `api_error`, `valid_sample` | API 오류 문자열(`fatal:…`/`transient:…`); `valid_sample=0`(API 오류·절단)은 pass@k 분모에서 제외 |
| `extract_ok`, `signature_ok`, `signature_exact` | 코드 블록 추출·시그니처 검사(한정자 차이는 `signature_exact=0`일 뿐 실패 아님) |
| `build_ok`, `build_time_ms`, `n_warnings`, `undefined_symbols`, `contract_violation` | 빌드 결과; `contract_violation`은 관찰용 JSON 리스트(`threads_in_c_seq`, `stdout_output`, `forbidden_symbol:<s>`) |
| `pass`, `turns_to_pass`, `error_type`, `error_stage`, `error_detail` | **`pass`** = check 전부 통과 ∧ 입력 미변조 ∧ OOB 없음 ∧ (bench 미실행 ∨ bench 정확성·크래시·race 없음). bench 타임아웃만 `pass=1`·시간 NA. `error_type`은 고정 분류표의 아홉 값 또는 빈칸(`docs/error_taxonomy.md`) |
| `inputs_mutated`, `oob_write`, `nondeterministic_runs`, `nondeterministic_bitwise_runs` | 입력 변조, 가드 영역 쓰기, bench 중 허용오차 초과 불일치 수(비트 불일치는 정보용) |
| `check_max_abs_err`, `check_max_tol_ratio`, `check_tests` | 정확성 테스트 최악값과 테스트별 JSON(`{name, classification, pattern, max_tol_ratio, elapsed_ms, nondeterministic}`) |
| `bench_ran`, `time_median_ms`, `time_mean_ms`, `time_min_ms`, `time_max_ms`, `time_cv`, `time_samples_ms`, `warmup`, `repeats`, `repeats_reduced`, `gflops` | 1024³ 성능: **`time_median_ms`가 보고값**, `time_cv`는 반복 간 변동(ddof=1), `time_samples_ms`는 `{"warmup":[…],"timed":[…]}` 전체 분포, `warmup`·`repeats`는 **계획값**(축소 행에서는 축소된 값 = 실제 횟수; 즉시 timeout·크래시 행에서는 3/10이 남으므로 실제 호출 수는 `time_samples_ms` 길이로 본다), `repeats_reduced=1`은 느린 커널의 반복 축소(`docs/protocol.md` §5) |
| `baseline_time_ms`, `baseline_time_cv`, `baseline_samples_ms`, `baseline_run_time_ms`, `baseline_time_1t_ms`, `baseline_drift_pct` | 기준선(`np.matmul(out=)`, OpenBLAS): 샘플별 **대응** 기준선, run 시작 기준선, 1스레드 기준선, drift(%) |
| `speedup_vs_baseline`, `speedup_vs_baseline_1t` | `baseline_time_ms / time_median_ms`(대응 기준선 대비; >1이면 OpenBLAS보다 빠름), `baseline_time_1t_ms / time_median_ms` |
| `cpu_mhz_median`, `cpu_mhz_min`, `throttle_events`, `temp_c_start`, `temp_c_end` | bench 중 고정 코어 주파수·스로틀 횟수·패키지 온도 |
| `compiler`, `compiler_version`, `cflags`, `march_resolved`, `threads`, `cores`, `omp_env` | 빌드·실행 조건(`cflags`는 전체 명령의 플래그 문자열, `march_resolved`는 `-march=native` 해석값) |
| `governor`, `governor_ok`, `epp`, `no_turbo`, `load_1m`, `load_ok`, `pinned_busy_pct`, `sibling_busy_pct`, `taskset_used`, `protocol_ok`, `protocol_flags`, `protocol_checked_at` | 측정 규약 상태. `protocol_flags`는 JSON 리스트(`governor=powersave`, `epp=power`, `turbo=on`(정보용), `load_1m=…`, `pinned_busy=…`, `sibling_busy=…`, `taskset_missing`, `blas=…`, `baseline_drift=…%`, `paired_baseline_failed`(대응 기준선 bench 실패 → `baseline_time_ms`·`speedup_vs_baseline` NA), `model_served_mixed`(run 안에서 응답 모델이 단일값이 아님; run 끝에 그 run의 모든 행에 붙음), `env_overrides_dropped`(정보용)). **`protocol_ok=1` = 정보용 외 플래그가 하나도 없음**(drift·`paired_baseline_failed`·`model_served_mixed` 포함) |
| `hostname`, `cpu_model`, `python_version`, `numpy_version`, `numpy_blas`, `sdk_version`, `test_seed`, `test_hash`, `env_hash` | 재현성 지문. `env_hash`가 다르면 비교 불가 |
| `asan_findings`, `ubsan_findings`, `lint_*`, `perf_summary`, `roofline_pct` | M5 열(M1은 NA) |
| `raw_path`, `built_path`, `harness_version` | 원본 위치, 하네스 버전(git describe; git 없으면 `nogit-<소스 해시>`) |

## 7. 집계 (`analysis/aggregate.py`)

```sh
make aggregate                                   # results/runs.csv, --latest
make aggregate RUN_ID=… [CSV=results/runs_dummy.csv]
.venv/bin/python -m analysis.aggregate [--csv results/runs.csv] [--run-id ID | --latest | --all-runs] [--json]
```

pandas로 읽어(`dtype=str, keep_default_na=False` 뒤 수동 변환) 행 키 중복은 `created_at` 최신만 남기고, 그룹 `(run_id, provider, model, task, language, args, grammar_hint, feedback, turn)`별로 다음을 낸다. run 단위로는 기준선(run 시작/1스레드/대응 평균·n·최대 |drift|)과 `model_served`·`effort`가 단일값인지(아니면 경고)를 요약한다. CSV가 없으면 안내와 함께 exit 1, 행이 없으면 안내와 함께 exit 0. 부트스트랩 CI는 M6.

| 열 | 의미 |
|---|---|
| `n_rows` | 그룹 행 수(`valid_sample=0` 포함) |
| `n`, `n_pass` | `valid_sample=1`인 행 수(pass@k 분모), 그중 `pass=1` |
| `pass_at_1`, `pass_at_5`, `pass_at_n` | 비편향 추정량 `1 − C(n−c, k)/C(n, k)`; `k > n`이면 NA. `pass_at_1 = n_pass/n` |
| `n_generate_failed`, `n_refusal`, `n_truncated` | 생성 단계: API 오류(`valid_sample=0`), 거절(`valid_sample=1`, `pass=0`), `max_tokens` 절단(`valid_sample=0`) — 서로 겹치지 않음 |
| `benchmarked`, `n_repeats_reduced`, `n_protocol_flagged` | `bench_ran=1`, `repeats_reduced=1`, `protocol_ok=0`이거나 정보용 외 규약 플래그가 있는 행 수 |
| `time_median_ms` | pass 행들의 `time_median_ms` 중앙값(bench 타임아웃 행은 시간 NA라 제외) |
| `speedup_median`, `speedup_min`, `speedup_max`, `speedup_1t_median` | pass 행들의 `speedup_vs_baseline` 중앙값/최소/최대, `speedup_vs_baseline_1t` 중앙값 |
| `tokens_out_mean`, `tokens_text_mean` | 토큰 값이 있는 모든 행(절단 행 포함)의 평균 |
| `tokens_per_pass`, `tokens_out_per_pass` | `Σ(tokens_in + tokens_out) / n_pass`, `Σ tokens_out / n_pass`(`n_pass=0`이면 NA) |
| `error_types` | `pass=0` 행의 `error_type` 집계 `"유형:수, …"`(수 내림차순; 빈 `error_type`인 생성 단계 행은 제외) |

## 8. 확정된 실험 조건 (2026-11-05 사용자 확인; 근거는 `docs/decisions.md`)

1. **LLM** — 제공자 `anthropic`, 키 변수 `ANTHROPIC_API_KEY`, 모델 `claude-sonnet-5-5`, `effort=medium`, `thinking=adaptive`. Claude 4.7+/5.x는 `temperature`/`top_p`/`top_k`를 거부하고 `seed`가 없어 **제공자 기본 샘플링**을 쓴다(CLAUDE.md의 "temperature 0.2" 이탈). 비용은 Max 구독의 월 API 크레딧(Console 조직 연결 필요)으로 충당한다. 모델은 전체 실험 동안 바꾸지 않는다.
2. **머신** — `config/machine.yaml`(i7-1260P, P-core 4개 고정). 거버너는 측정 세션마다 `sudo cpupower frequency-set -g performance`(재부팅하면 되돌아감). 실험 서버가 정해지면 `make probe`와 `measurement.cores`를 다시 맞춘다.
3. **M1 언어** — `numpy`, `c_seq`, `c_openmp`.
4. **저장소** — `https://github.com/Ustar2002/graduation_work` (비공개). `harness_version`은 `git describe`.
5. **프롬프트의 하드웨어 사실 공개** — 승인. 모든 언어에 같은 사실을 준다.
6. **느린 커널의 반복 축소 규칙** — 승인(`repeats_reduced=1` 플래그).
7. Python 시그니처의 출력 버퍼 `C`, `atol=1e-3`, 정확성 형태 `tiny`·`rect` 추가 — 승인.

## 9. 마일스톤 상태

- **M1 (수직 슬라이스) — 완료(2026-11-05).** 더미 파이프라인 검증(11개 변형 일치, 단위 테스트 164개) 뒤 첫 실제 실행 `20261105-real`: `claude-sonnet-5-5` 9회 호출 모두 pass, 속도비 중앙값 numpy 1.01 / c_openmp 0.97 / c_seq 0.33, 비용 약 0.21달러. 관찰과 후속 제안은 `docs/decisions.md` "M1 완료" 절.
- **M2a (Tier 1 커널) — 하네스 일반화와 더미 검증 완료(2026-11-05), 실제 호출 대기.** 과제 `gemm`, `jacobi2d`, `spmv_csr`, `dot`; 단위 테스트 170개; 실제 실행은 4과제 × 3언어 × 5샘플 = 60회 호출(`run.n_samples`를 5로 올린 뒤 `make real`).
- M2b·M2c(Tier 2·3 커널) → M3 언어 확장(numba, halide, rust) → M4 마이크로 조건·수정 루프 → M5 perf_summary·ASan/UBSan·린트 → M6 분석·그래프·대시보드 → M7 확장. 앞 단계 결과를 보고 다음 단계를 넣는다(CLAUDE.md 원칙 1).
