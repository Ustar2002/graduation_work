# 더미 응답 (harness/dummy_responses/) — 과제 전체 기대표

`provider: dummy` 일 때 `harness.llm_client` 가 API 대신 읽는 가짜 모델 응답이다(설계 `docs/harness_design.md` §13·§17).
`<task>/<language>/<k>.md` 가 **`sample_idx mod 파일 수`** 번째 샘플의 응답이 되고, 각 파일은 실제 응답처럼 짧은 설명 + 펜스 코드 블록
하나다. `make dummy`(또는 `generate --provider dummy` → `run` → `aggregate --csv results/runs_dummy.csv`)의 행이 아래 표와 같으면
파이프라인(extract → build → import → check → bench → 행)이 정상이다. 분류 규칙은 설계 §10 과 `docs/error_taxonomy.md`.
더미 행은 `results/runs_dummy.csv` 에만 쌓인다(실제 결과와 섞이지 않음).

| task | numpy | c_seq | c_openmp | 과제별 README(실측·주의) |
|---|---|---|---|---|
| gemm (M1) | 0–4 | 0–2 | 0–2 | `gemm/README.md` — 11개 변형(pass·signature_mismatch·numeric_instability·timeout·wrong_output·compile_error·arg_order·runtime_crash·race_condition) |
| jacobi2d · spmv_csr · dot (M2a, Tier 1) | 0 | 0 | 0 | 정답만(`dummy-m2`, 2026-11-05: 9행 pass) |
| histogram (M2b, Tier 2) | 0 | 0 | 0, 1 | `histogram/README.md` |
| scan (Tier 2) | 0 | 0 | 0, 1 | `scan/README.md` |
| nbody (Tier 2) | 0 | 0 | 0 | `nbody/README.md` |
| sort (Tier 2) | 0 | 0 | 0 | `sort/README.md` |

## Tier 2 기대표 (설계 §17; 2026-11-08)

`k = sample_idx mod 파일 수` 이므로 `--n 2` 로 돌리면 c_openmp 의 sample 1 이 변형(k=1)을 밟고, 나머지 sample 1 은 k=0 을 다시 쓴다.

| task | language | k | 내용 | 기대 (error_stage / error_type / error_detail) | pass |
|---|---|---|---|---|---|
| histogram | numpy | 0 | `hist[...] = np.bincount(data, minlength=nbins)` | 없음 — bench 까지 정상 | 1 |
| histogram | c_seq | 0 | `memset` 으로 0 초기화 뒤 `hist[data[i]]++` | 없음 | 1 |
| histogram | c_openmp | 0 | 스레드별 private 히스토그램(`calloc` 스크래치의 슬라이스) → `omp for` 로 빈을 나눠 합산 | 없음 | 1 |
| histogram | c_openmp | 1 | 0 초기화 뒤 보호 없는 `#pragma omp parallel for` 로 공유 `hist[data[i]]++` (race) | check / race_condition / `nondeterministic:rect,small`(tiny 는 충돌이 거의 없어 보통 pass; 두 호출이 우연히 같게 틀리면 wrong_output / mismatch) | 0 |
| scan | numpy | 0 | `np.cumsum(x, dtype=np.int32, out=out)` | 없음 | 1 |
| scan | c_seq | 0 | 순차 누적 `acc += x[i]; out[i] = acc` | 없음 | 1 |
| scan | c_openmp | 0 | 2-pass 블록 스캔(스레드별 블록 합 → exclusive 오프셋 → 블록 내 순차 스캔) | 없음 | 1 |
| scan | c_openmp | 1 | `out[0] = x[0]` 뒤 `parallel for` 로 `out[i] = out[i-1] + x[i]` — 순차 의존 무시 | check / race_condition / `nondeterministic:<불일치 테스트 목록>`(목록은 타이밍 의존; 세 형태 모두 같게 틀리면 wrong_output / mismatch) | 0 |
| nbody | numpy | 0 | float64 로 256행씩 브로드캐스트 직접 합산, 블록마다 `acc[i0:i1] = …` | 없음 | 1 |
| nbody | c_seq | 0 | i·j 2중 루프, `1.0f/sqrtf`, AoS 그대로 | 없음 (직진 float32 커널의 max_tol_ratio ≤ 0.2 계약) | 1 |
| nbody | c_openmp | 0 | 위 커널에 `#pragma omp parallel for schedule(static)` — 기준선 `tasks/nbody/baseline/baseline.c` 와 같은 코드 | 없음, speedup_vs_baseline ≈ 1 | 1 |
| sort | numpy | 0 | `out[...] = np.sort(x)` — 기준선 `tasks/sort/baseline/baseline.py` 와 같은 코드 | 없음, speedup_vs_baseline ≈ 1 | 1 |
| sort | c_seq | 0 | `memcpy` 뒤 `qsort`(비교자는 오버플로 없는 2비교) | 없음 (large 10⁷ 에 호출당 약 1.3 s, 1스레드) | 1 |
| sort | c_openmp | 0 | 8비트 4패스 병렬 LSD 기수 정렬(스레드별 digit 히스토그램 → bucket·thread 순 오프셋 → 안정 분산; 부호 비트 뒤집기) | 없음, speedup_vs_baseline > 1 | 1 |

정수 출력(histogram·scan·sort)은 허용오차 0(정확 일치)이라 `numeric_instability` 가 나올 수 없다(`docs/error_taxonomy.md` 부록 2).
`histogram`·`scan`·`sort` 는 `work` 가 없어 `gflops` 열이 NA, `nbody` 만 `work = 20·n²`.

## 검증 (2026-11-08, `dummy-tier2`)

`generate --provider dummy --run-id dummy-tier2 --tasks histogram,scan,nbody,sort --n 2`(24 dummy calls, 0.3 s) → `run --run-id dummy-tier2`
(24행, 354 s; governor·EPP `performance`, turbo on) → `aggregate --csv results/runs_dummy.csv --run-id dummy-tier2`(12 그룹, 과제별 기준선 요약).
**24행 모두 위 표와 일치**: 정답 22행 pass(check 세 형태 pass, bench `final_check` pass, 실행 간 불일치 0),
histogram c_openmp/1 `check / race_condition / nondeterministic:rect,small`(tiny 두 호출 pass; rect 호출 1 pass·호출 2 유실 660; small 유실 1607·1793),
scan c_openmp/1 `check / race_condition / nondeterministic:tiny,rect`(small 은 두 호출이 같게 틀림 — max_abs_err 12,185,827; 호출 수준은 전부 `wrong_output/mismatch`).
멱등 재실행(`harness.run --run-id dummy-tier2`): 0.26 s, 24행 전부 "result.json 이 최신이라 건너뜀", 기준선 재측정 없음, CSV·result.json·`bench_run/1t.json` sha256 동일,
행 키 중복 0, `pgrep -af "harness[.]worker|[c]c1"` 비어 있음.

large 의 중앙값(ms; 괄호는 대응 기준선 대비 속도비; 샘플 0 / 샘플 1):

| task | 기준선 run 시작 4T / 1T | numpy/0 | c_seq/0 (1T, core 2) | c_openmp/0 (4T) |
|---|---|---|---|---|
| histogram | 8.34 / 19.45 | 86.1 (0.10) / 89.8 (0.09) | 17.8 (0.46) / 17.7 (0.45) | 8.73 (1.03) |
| scan | 10.58 / 24.37 | 95.9 (0.12) / 72.0 (0.16) | 20.0 (0.55) / 17.2 (0.68) | 20.3 (0.55; cv 0.19) |
| nbody | 229.2 / 643.7 | 5,822 (0.04) / 6,167 (0.04) | 608.7 (0.35) / 606.8 (0.38) | 213.7 (1.01) / 229.2 (1.00) |
| sort | 63.3 / 70.5 | 68.1 (1.03) / 78.9 (0.83) | 1,406 (0.05) / 1,400 (0.05) | 99.5 (0.90) / 46.0 (1.48) |

이 run 은 통합 작업(문서 편집·검증 스크립트)과 **동시에** 돌아 구현자 측정(과제별 README)보다 느리거나 흔들린 행이 있다
(scan c_openmp/0 20.3 ms vs 10.46; sort c_openmp/0 샘플 0 99.5 ms vs 샘플 1 46.0 vs 구현자 37.9; histogram c_seq 17.7 vs 13.5).
drift 플래그 2행(histogram c_openmp/0 `baseline_drift=+10.7%`, sort c_openmp/0 `+30.1%` → `protocol_ok=0`), `load_1m`·`pinned_busy`·`sibling_busy` 플래그 0.
분류 검증이 목적이므로 절대 시간은 참고용이다.
