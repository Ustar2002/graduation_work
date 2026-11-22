# 더미 응답 (harness/dummy_responses/) — 과제 전체 기대표

`provider: dummy` 일 때 `harness.llm_client` 가 API 대신 읽는 가짜 모델 응답이다(설계 `docs/harness_design.md` §13·§17·§18, `docs/harness_design_v3.md` §14).
`<task>/<language>/<k>.md` 가 **`sample_idx mod 파일 수`** 번째 샘플의 응답이 되고, 각 파일은 실제 응답처럼 짧은 설명 + 펜스 코드 블록
하나다. 수정 턴 t ≥ 1 의 응답은 `<k>.turn<t>.md`(없으면 `<k>.md`)이고 파일 수에 들지 않는다(`gemm/README.md`).
`make dummy`(또는 `generate --provider dummy` → `run` → `aggregate --csv results/runs_dummy.csv`)의 행이 아래 표와 같으면
파이프라인(extract → build → import → check → bench → 행)이 정상이다. 분류 규칙은 설계 §10 과 `docs/error_taxonomy.md`.
더미 행은 `results/runs_dummy.csv` 에만 쌓인다(실제 결과와 섞이지 않음). `make dummy` 는 설정의 기본 언어(numpy·c_seq·c_openmp)만 돌리고,
새 언어는 `make dummy-m3`(네 언어) 또는 `make dummy-all`(일곱 언어)로 돌린다. 모든 `0.md` 의 정확성 점검은 `make devcheck` 다.

표의 칸은 그 조합에 있는 `<k>.md` 의 k 다. `—` 는 과제가 그 언어를 지원하지 않는 조합(`tasks/<task>/signatures.yaml` 의 `unsupported`)이라 파일이 없다.

| task | numpy | c_seq | c_openmp | numba | halide_auto | halide_manual | rust | 과제별 README(실측·주의) |
|---|---|---|---|---|---|---|---|---|
| gemm (M1) | 0–4 | 0–2 | 0–2 | 0 | 0 | 0 | 0 | `gemm/README.md` — 11개 변형(pass·signature_mismatch·numeric_instability·timeout·wrong_output·compile_error·arg_order·runtime_crash·race_condition)과 수정 턴 2개 |
| jacobi2d · spmv_csr · dot (M2a, Tier 1) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 과제별 README 없음 — 정답만(`dummy-m2`, 2026-11-05: 기존 세 언어 9행 pass). 새 언어는 아래 "새 언어 더미" |
| histogram (M2b, Tier 2) | 0 | 0 | 0, 1 | 0 | 0 | 0 | 0 | `histogram/README.md` |
| scan (Tier 2) | 0 | 0 | 0, 1 | 0 | 0 | 0 | 0 | `scan/README.md` |
| nbody (Tier 2) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `nbody/README.md` |
| sort (Tier 2) | 0 | 0 | 0 | 0 | — | — | 0 | `sort/README.md` |
| cholesky (M2c, Tier 3) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `cholesky/README.md` |
| floyd_warshall (Tier 3) | 0 | 0 | 0, 1 | 0 | 0 | 0 | 0 | `floyd_warshall/README.md` — c_openmp/1 은 k 루프 병렬화(의존성 위반) |
| bfs (Tier 3) | 0 | 0 | 0, 1 | 0 | — | — | 0 | `bfs/README.md` — c_openmp/1 은 원자 연산 없는 프론티어(race) |
| fft (Tier 3) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `fft/README.md` |

## 새 언어 더미 (numba · halide_auto · halide_manual · rust)

새 언어의 더미는 과제 12개 × 언어 4개에서 Halide 의 `sort`·`bfs` 를 뺀 44개이고, 모두 정답 `0.md` 하나다.
기대는 전부 **오류 없음 — bench 까지 정상, pass = 1** 이다. 공통 규칙:

- numba: 모든 `@njit` 에 명시 시그니처 문자열(첫 위치 인자), `cache=True` 없음, 배열 `float32[::1]`·`float32[:, ::1]`·`int32[::1]`, 크기 `int64`.
- halide_auto: 알고리즘만 쓰고 스케줄 호출이 없다 — 행의 `contract_violation` 에 `schedule_in_auto` 가 붙지 않아야 한다. 병렬화는 오토스케줄러(Li2018) 몫이다.
- halide_manual: 같은 알고리즘에 응답 자신의 스케줄(split·vectorize·parallel·compute_root 등)을 붙인다.
- rust: 내보내는 함수는 `#[no_mangle] pub unsafe extern "C"` 하나, 크기는 `i32`, rayon 외의 크레이트 없음.
- `sort`·`bfs` × `halide_auto`·`halide_manual` 은 파일이 없다: `harness.generate` 가 `unsupported` 로 건너뛰고, `tools/devcheck.py dummies` 는 `unsupported`,
  `analysis.aggregate --coverage` 는 `n/a` 로 낸다. 파일이 없는데 지원 조합이면 devcheck 가 `missing`(실패)으로 낸다.

과제별 README 가 있는 과제(gemm, histogram, scan, nbody, sort, cholesky, floyd_warshall, bfs, fft)는 그 README 의 "새 언어" 절에 파일마다의 구현이 있다.
과제별 README 가 없는 Tier 1 세 과제:

| task | language | 내용 |
|---|---|---|
| jacobi2d | numba | 버퍼 두 개를 스텝마다 바꾸고 각 스텝의 행을 `prange` 로 나눈다(헬퍼 `_step` 과 진입 함수 모두 명시 시그니처). 마지막 스텝이 `dst` 에 쓰이도록 첫 대상을 고른다 |
| jacobi2d | halide_auto | 스텝마다 Func 하나(이전 스텝만 읽음): 내부 칸은 네 이웃 평균, 나머지는 이전 값 복사. 입력만 `repeat_edge` 로 감싼다 |
| jacobi2d | halide_manual | 같은 알고리즘 + 마지막을 뺀 스텝마다 `compute_root()`, `vectorize(x, 8, GuardWithIf).parallel(y)` |
| jacobi2d | rust | `dst` 와 임시 버퍼를 스텝마다 바꾸고, 스텝 안에서는 출력 행을 rayon `par_chunks_mut` 로 나눈다 |
| spmv_csr | numba | 행을 `prange` 로 나누고 행마다 반복 지역 변수에 double 로 누적, 빈 행은 0 |
| spmv_csr | halide_auto | 곱의 누적합 S 를 double 로(RDom 갱신 S[q+1] = S[q] + 곱) 만들고 행 결과 = 행 끝과 시작의 S 차 |
| spmv_csr | halide_manual | 같은 알고리즘 + S 는 한 번의 순차 패스(`compute_root`, 갱신은 `unscheduled`), y 는 4096행 블록 `parallel` + `vectorize` |
| spmv_csr | rust | 1024행 블록을 rayon `par_chunks_mut` 로 나누고 행마다 f64 누적 |
| dot | numba | 고정 1024 청크를 `prange` 로 double 합산한 뒤 청크 합을 고정 순서로 더한다(스레드 수와 무관하게 같은 결과) |
| dot | halide_auto | 16384 원소 청크마다 double 리덕션, 청크 합과 꼬리를 고정 순서로 더한다(크기에 맞춰 파이프라인을 만든다) |
| dot | halide_manual | 같은 알고리즘 + 청크 합 `compute_root().parallel(c)` |
| dot | rust | 2¹⁶ 원소 블록을 rayon `par_chunks` 로 double 합산(누산기 8개), 부분합을 블록 순서로 더한다 |

검증(2026-11-23, Windows 11 개발 머신): `tools/devcheck.py dummies` 에서 새 언어 44개 pass·4개 unsupported(정확성 형태 tiny·rect·small),
더미 파이프라인(`harness.run`, check 까지)에서 numpy·numba·rust·halide_auto·halide_manual × 과제 12개 56행 pass(`docs/decisions.md` 2026-11-23).
Linux 에서의 확인(`large`·bench 포함)은 `docs/linux_test_plan.md`.

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
