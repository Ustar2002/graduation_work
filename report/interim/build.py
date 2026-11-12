"""중간보고서 빌드: results/runs.csv → 수치(JSON)·그림(SVG) → Typst 조판 → PDF.

    .venv/bin/python report/interim/build.py            # 전체(수치 → 그림 → PDF)
    .venv/bin/python report/interim/build.py --no-pdf   # 수치·그림만

- 보고서의 표·그림·본문 수치는 모두 ``results/runs.csv`` 에서 다시 계산한다. 본문은 ``generated/results.json`` 의 ``c``(서식을 입힌
  문자열)를 읽어 끼워 넣으므로 재측정 뒤에도 표와 문장이 함께 바뀐다. 문장의 **뜻**이 기대는 구조(예: "네 과제는 대등, 일곱 과제는
  큰 차이")는 ``check_claims`` 가 검사해, 데이터가 그 구조를 벗어나면 빌드를 실패시킨다(그때는 본문을 고쳐야 한다).
- 대상 run 은 12개 과제 본 실험 세 개(``RUNS``)이며 M1 예비 실행(20261105-real)은 제외한다.
- 생성 코드의 특징(스레드 사용, 내장 함수 사용 등)과 측정 결함 수정 여부는 ``results/built`` 의 파일을 읽어 센다. 이 디렉터리는 git 에
  올리지 않으므로 없으면 그 검사만 건너뛰고 ``generated/results.json`` 에 남아 있는 값을 쓴다.
- 산출물: ``generated/results.json``, ``generated/fig_speedup.svg``, ``interim_report.pdf``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GEN = HERE / "generated"
CSV_PATH = ROOT / "results" / "runs.csv"

RUNS = ("20261108-tier1", "20261108-tier2", "20261109-tier3")
LANGS = ("numpy", "c_seq", "c_openmp")
C_LANGS = ("c_seq", "c_openmp")
# (과제, 난이도 수준) — 표·그림의 행 순서
TASKS: tuple[tuple[str, int], ...] = (
    ("gemm", 1), ("jacobi2d", 1), ("spmv_csr", 1), ("dot", 1),
    ("histogram", 2), ("scan", 2), ("nbody", 2), ("sort", 2),
    ("cholesky", 3), ("floyd_warshall", 3), ("bfs", 3), ("fft", 3),
)
# 기준선이 NumPy 라이브러리 호출인 과제(나머지의 기준선은 C+OpenMP 참조 구현)
LIBRARY_BASELINE = ("gemm", "dot", "sort", "cholesky", "fft")
# 본문 4.3 의 사후 분류: numpy 구현이 연산의 대부분을 최적화된 루틴(BLAS, SIMD 정렬)에 위임한 과제
PARITY_TASKS = ("gemm", "dot", "sort", "cholesky")
# 공시 단가(USD / 100만 토큰; Claude Sonnet 5.5, 2026-10 기준). 비용은 추정치다.
PRICE_IN, PRICE_OUT = 2.0, 10.0

# 실패 17건의 원인. 생성 코드를 직접 읽어 판정했고 2026-11-11 독립 검토에서 재현으로 확인했다(docs/decisions.md).
# 키: (과제, 조건, 샘플). 통과·실패가 달라지면 check_claims 가 빌드를 멈춘다.
FAILURE_CAUSE: dict[tuple[str, str, int], str] = {
    **{("nbody", "numpy", i): "cancel" for i in range(5)},
    ("floyd_warshall", "c_seq", 4): "loop_order", ("floyd_warshall", "c_openmp", 1): "loop_order",
    ("floyd_warshall", "c_openmp", 4): "loop_order",
    ("cholesky", "c_openmp", 2): "index", ("scan", "c_openmp", 3): "index",
    ("bfs", "c_seq", 2): "bottom_up", ("bfs", "c_seq", 4): "bottom_up",
    ("fft", "c_seq", 3): "heap",
    ("jacobi2d", "c_openmp", 2): "format", ("spmv_csr", "c_seq", 2): "format",
    ("scan", "c_openmp", 2): "format", ("sort", "c_openmp", 1): "format",
}
# (원인 키, 표 3 의 서술). 과제 이름은 자동으로 붙인다.
CAUSES: tuple[tuple[str, str], ...] = (
    ("cancel", "위임을 위한 식 재배열의 float32 상쇄"),
    ("loop_order", "블록화 구현의 루프 순서 오류"),
    ("index", "블록·SIMD 구현의 색인·경계 오류"),
    ("bottom_up", "방향 그래프에 bottom-up 탐색 적용"),
    ("heap", "힙 메모리 오류로 실행 중단"),
    ("format", "코드 블록이 없는 응답"),
)
ERR_ABBR = {"wrong_output": "W", "numeric_instability": "N", "runtime_crash": "R", "signature_mismatch": "S"}


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["run_id"] in RUNS]
    keys = Counter((r["run_id"], r["task"], r["language"], r["sample_idx"]) for r in rows)
    dup = [k for k, v in keys.items() if v > 1]
    if dup:
        raise SystemExit(f"[build] 중복 행 키: {dup[:3]}")
    return rows


def _f(x: str) -> float | None:
    return float(x) if x not in ("", None) else None


def _rng(vals: list[float], digits: int = 2) -> list[str]:
    return [f"{min(vals):.{digits}f}", f"{max(vals):.{digits}f}"]


# ----------------------------------------------------------------------------- 생성 코드의 특징(패턴 기반)

_RE_THREADS = re.compile(r"\bimport\s+threading\b|ThreadPoolExecutor|ThreadPool\b|threading\.Thread")
_RE_INTRIN = re.compile(r"immintrin\.h|_mm256_")
_RE_MATMUL = re.compile(r"(?<![\w)\]])@(?!\w)|\w\s*@\s*\w|np\.matmul|np\.dot\(")
_RE_BLOCK_DEF = re.compile(r"#define\s+(?:B|BS|BLK|BLOCK)\b")


def code_features(rows: list[dict[str, str]]) -> dict | None:
    """추출된 커널 소스에서 본문이 인용하는 특징을 센다. results/built 가 없으면 None."""
    if not (ROOT / "results" / "built").is_dir():
        return None

    def src(r: dict[str, str]) -> str | None:
        if r["error_stage"] == "extract":           # 코드 블록이 없던 응답(이전 빌드의 잔여 파일을 세지 않는다)
            return None
        p = ROOT / r["built_path"] / ("kernel.py" if r["language"] == "numpy" else "kernel.c")
        return p.read_text(encoding="utf-8") if p.is_file() else None

    srcs = {(r["task"], r["language"], int(r["sample_idx"])): src(r) for r in rows}
    if all(v is None for v in srcs.values()):
        return None

    def count(task: str | None, langs: tuple[str, ...], pat: re.Pattern) -> list[int]:
        ks = [v for (t, l, _), v in srcs.items() if (task is None or t == task) and l in langs and v is not None]
        return [sum(1 for k in ks if pat.search(k)), len(ks)]

    bottom_up = sorted([l, i] for (t, l, i), v in srcs.items()
                       if t == "bfs" and l in C_LANGS and v is not None and re.search(r"bottom", v, re.I))
    chol_matmul = sorted(i for (t, l, i), v in srcs.items() if t == "cholesky" and l == "numpy" and v and _RE_MATMUL.search(v))
    return {
        "n_extracted": {lang: count(None, (lang,), re.compile(r""))[1] for lang in LANGS},
        "numpy_threads": count(None, ("numpy",), _RE_THREADS),
        "numpy_threads_by_task": {t: count(t, ("numpy",), _RE_THREADS)[0] for t, _ in TASKS},
        "c_intrinsics": count(None, C_LANGS, _RE_INTRIN),
        "c_intrinsics_by_task": {t: count(t, C_LANGS, _RE_INTRIN) for t, _ in TASKS},
        "gemm_numpy_matmul": count("gemm", ("numpy",), _RE_MATMUL),
        "nbody_numpy_matmul": count("nbody", ("numpy",), _RE_MATMUL),
        "nbody_c_rsqrt": count("nbody", C_LANGS, re.compile(r"rsqrt")),
        "cholesky_numpy_matmul": count("cholesky", ("numpy",), _RE_MATMUL),
        "cholesky_numpy_matmul_samples": chol_matmul,
        "floyd_c_blocked": count("floyd_warshall", C_LANGS, _RE_BLOCK_DEF),
        "bfs_c_bottom_up": bottom_up,
        "fft_c_double": count("fft", C_LANGS, re.compile(r"\bdouble\b")),
    }


def affinity_status(rows: list[dict[str, str]]) -> dict | None:
    """측정된 행 가운데 호출 친화도 수정(2026-11-11) 뒤의 워커로 잰 행 수. bench.json 의 ``call_affinity`` 유무로 판정한다.
    results/built 가 없으면 None."""
    if not (ROOT / "results" / "built").is_dir():
        return None
    n = fixed = 0
    for r in rows:
        if r["bench_ran"] != "1":
            continue
        p = ROOT / r["built_path"] / "bench.json"
        if not p.is_file():
            return None
        n += 1
        try:
            fixed += "call_affinity" in json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return {"n_bench": n, "n_fixed": fixed}


# ----------------------------------------------------------------------------- 집계

def summarize(rows: list[dict[str, str]], code: dict | None, aff: dict | None) -> dict:
    by_key = {(r["task"], r["language"], int(r["sample_idx"])): r for r in rows}
    cells: dict[str, dict[str, dict]] = {}
    for task, _tier in TASKS:
        cells[task] = {}
        for lang in LANGS:
            rs = [r for r in rows if r["task"] == task and r["language"] == lang]
            sp = sorted(v for v in (_f(r["speedup_vs_baseline"]) for r in rs) if v is not None)
            cells[task][lang] = {
                "n": len(rs),
                "n_pass": sum(r["pass"] == "1" for r in rs),
                "speedups": sp,
                "speedup": st.median(sp) if sp else None,
                "speedup_min": sp[0] if sp else None,
                "speedup_max": sp[-1] if sp else None,
                "pass_str": f"{sum(r['pass'] == '1' for r in rs)}/{len(rs)}",
                "speedup_str": f"{st.median(sp):.2f}" if sp else "–",
            }
    # 같은 과제 안에서 c_openmp ÷ numpy (속도비 중앙값의 비; 반올림 전 값으로 계산)
    ratio = {t: cells[t]["c_openmp"]["speedup"] / cells[t]["numpy"]["speedup"] for t, _ in TASKS if cells[t]["numpy"]["speedup"]}
    for task, _tier in TASKS:
        cells[task]["ratio"] = ratio.get(task)
        cells[task]["ratio_str"] = f"{ratio[task]:.2f}" if task in ratio else "–"

    totals = {}
    for lang in LANGS:
        rs = [r for r in rows if r["language"] == lang]
        totals[lang] = {"n": len(rs), "n_pass": sum(r["pass"] == "1" for r in rs),
                        "n_extracted": sum(r["error_stage"] != "extract" for r in rs)}
    totals["all"] = {"n": len(rows), "n_pass": sum(r["pass"] == "1" for r in rows),
                     "n_extracted": sum(r["error_stage"] != "extract" for r in rows)}
    failed = [r for r in rows if r["pass"] != "1"]
    failures = [{"task": r["task"], "language": r["language"], "sample_idx": int(r["sample_idx"]),
                 "error_stage": r["error_stage"], "error_type": r["error_type"], "error_detail": r["error_detail"][:80],
                 "cause": FAILURE_CAUSE.get((r["task"], r["language"], int(r["sample_idx"])))}
                for r in sorted(failed, key=lambda r: (r["task"], r["language"], int(r["sample_idx"])))]

    # 표 3: 원인별 × 조건별 실패 수
    ftable = []
    for key, label in CAUSES:
        fs = [f for f in failures if f["cause"] == key]
        auto = Counter(f["error_type"] for f in fs)
        tasks = [t for t, _ in TASKS if any(f["task"] == t for f in fs)]
        ftable.append({"cause": key, "label": label, "tasks": tasks,
                       "auto": ", ".join(f"{ERR_ABBR[k]} {v}" for k, v in sorted(auto.items(), key=lambda kv: -kv[1])),
                       "n": len(fs), **{lang: sum(f["language"] == lang for f in fs) for lang in LANGS}})

    tok_in = sum(int(r["tokens_in"]) for r in rows)
    tok_out = sum(int(r["tokens_out"]) for r in rows)
    tok_think = sum(int(r["tokens_thinking"] or 0) for r in rows)
    think_cells = sorted({(r["task"], r["language"]) for r in rows if int(r["tokens_thinking"] or 0) > 0})
    mean_text = {lang: st.mean(int(r["tokens_text"]) for r in rows if r["language"] == lang) for lang in LANGS}
    benched = [r for r in rows if r["bench_ran"] == "1"]
    cvs = [v for v in (_f(r["time_cv"]) for r in benched) if v is not None]
    flagged = [r for r in benched if r["protocol_ok"] == "0"]

    def sp_of(task: str, lang: str, idx: int) -> float | None:
        r = by_key.get((task, lang, idx))
        return _f(r["speedup_vs_baseline"]) if r else None

    s: dict = {
        "runs": list(RUNS),
        "tasks": [{"task": t, "tier": tier, "library_baseline": t in LIBRARY_BASELINE, "cells": cells[t]} for t, tier in TASKS],
        "totals": totals,
        "failures": failures,
        "failure_table": ftable,
        "failure_by_type": dict(Counter(r["error_type"] for r in failed)),
        "failure_by_lang_type": {lang: dict(Counter(r["error_type"] for r in failed if r["language"] == lang)) for lang in LANGS},
        "tokens": {"in": tok_in, "out": tok_out, "thinking": tok_think,
                   "thinking_cells": [list(c) for c in think_cells],
                   "n_rows_thinking": sum(int(r["tokens_thinking"] or 0) > 0 for r in rows),
                   "mean_text_out": {k: round(v) for k, v in mean_text.items()},
                   "cost_usd": round(tok_in / 1e6 * PRICE_IN + tok_out / 1e6 * PRICE_OUT, 2)},
        "measure": {"n_benchmarked": len(benched),
                    "time_cv_median": st.median(cvs),
                    "n_flagged": len(flagged),
                    "n_drift_flag": sum("baseline_drift" in r["protocol_flags"] for r in flagged),
                    "n_load_info": sum("load_1m=" in r["protocol_flags"] for r in benched),
                    "created_dates": sorted({r["created_at"][:10] for r in rows}),
                    "harness_versions": sorted({r["harness_version"] for r in rows}),
                    "governors": sorted({r["governor"] for r in rows}),
                    "models": sorted({r["model_served"] for r in rows}),
                    "efforts": sorted({r["effort"] for r in rows}),
                    "temperatures": sorted({r["temperature"] for r in rows}),
                    "turns": sorted({r["turn"] for r in rows}),
                    "cpu_models": sorted({r["cpu_model"] for r in rows}),
                    "compilers": sorted({r["compiler_version"] for r in rows if r["compiler"] == "gcc"}),
                    "numpy_versions": sorted({r["numpy_version"] for r in rows}),
                    "blas": sorted({r["numpy_blas"] for r in rows})},
        "code": code,
        "affinity": aff,
    }
    # 초안 여부: 호출 친화도 수정 전의 측정이 하나라도 남아 있으면 초안(본문에 결함을 적고 표 2 의 해당 칸에 표시)
    s["draft"] = not (aff is not None and aff["n_bench"] > 0 and aff["n_fixed"] == aff["n_bench"])

    # ---- 본문에 끼워 넣는 문자열(c)
    non_lib = [t for t, _ in TASKS if t not in LIBRARY_BASELINE]
    non_lib_np = [t for t in non_lib if cells[t]["numpy"]["speedup"] is not None]          # nbody 제외(numpy 통과 없음)
    rest = [t for t, _ in TASKS if t in ratio and t not in PARITY_TASKS]
    aa = cells["gemm"]["numpy"]
    chol_idx = (code or {}).get("cholesky_numpy_matmul_samples") or []
    chol_blocked = [v for v in (sp_of("cholesky", "numpy", i) for i in chol_idx) if v is not None]
    chol_plain = [v for v in (sp_of("cholesky", "numpy", i) for i in range(5) if i not in chol_idx) if v is not None]
    thr5 = [t for t, _ in TASKS if ((code or {}).get("numpy_threads_by_task") or {}).get(t) == 5]
    thr_cmp = [t for t in thr5 if t in rest]                                               # 스레드를 쓰고 c_openmp 와 차이가 큰 과제
    c_mean_tok = st.mean(mean_text[lang] for lang in C_LANGS)
    big = [cells[t][lang]["speedup"] for t in ("nbody", "cholesky", "fft") for lang in C_LANGS]
    affected = sorted(t for t in non_lib_np if ((code or {}).get("numpy_threads_by_task") or {}).get(t, 0) > 0)
    s["c"] = {
        "pass": {lang: f"{totals[lang]['n_pass']}/{totals[lang]['n']}" for lang in LANGS},
        "pass_extracted": {lang: f"{totals[lang]['n_pass']}/{totals[lang]['n_extracted']}" for lang in LANGS},
        "n_resp": str(totals["all"]["n"]), "n_impl": str(totals["all"]["n_extracted"]), "n_fail": str(len(failed)),
        "n_fail_c_algo": str(sum(f["language"] != "numpy" and f["cause"] != "format" for f in failures)),
        "n_fail_numpy": str(sum(f["language"] == "numpy" for f in failures)),
        "n_format": str(sum(f["cause"] == "format" for f in failures)),
        "by_type": {k: str(v) for k, v in Counter(r["error_type"] for r in failed).items()},
        "ratio": {t: f"{ratio[t]:.2f}" for t in ratio},
        "parity": _rng([ratio[t] for t in PARITY_TASKS]),
        "rest": _rng([ratio[t] for t in rest], 1),
        "n_parity": str(len(PARITY_TASKS)), "n_rest": str(len(rest)),
        "np_lib": _rng([cells[t]["numpy"]["speedup"] for t in LIBRARY_BASELINE]),
        "np_c": _rng([cells[t]["numpy"]["speedup"] for t in non_lib_np]),
        "omp_c": _rng([cells[t]["c_openmp"]["speedup"] for t in non_lib_np]),
        "seq_c": _rng([cells[t]["c_seq"]["speedup"] for t in non_lib_np]),
        "n_non_lib_np": str(len(non_lib_np)),
        "big": _rng(big),
        "aa": [f"{aa['speedup_min']:.2f}", f"{aa['speedup_max']:.2f}", f"{aa['speedup']:.2f}"],
        "chol_blocked": _rng(chol_blocked, 1) if chol_blocked else None,
        "chol_plain": f"{chol_plain[0]:.2f}" if chol_plain else None,
        "chol_np": cells["cholesky"]["numpy"]["speedup_str"],
        "dot_np": _rng(cells["dot"]["numpy"]["speedups"], 1),
        "thr_tasks": thr_cmp,
        "thr": _rng([ratio[t] for t in thr_cmp], 1) if thr_cmp else None,
        "tok_in": f"{tok_in / 1e4:.1f}만", "tok_out": f"{tok_out / 1e4:.1f}만", "cost": f"{s['tokens']['cost_usd']:.1f}",
        "tok_np": f"{round(mean_text['numpy'], -1):,.0f}", "tok_c": f"{round(c_mean_tok, -2):,.0f}",
        "think_pct": f"{100 * tok_think / tok_out:.0f}", "n_think": str(s["tokens"]["n_rows_thinking"]),
        "n_bench": str(len(benched)), "cv_pct": f"{100 * st.median(cvs):.1f}",
        "n_flag": str(len(flagged)), "flag_pct": f"{100 * len(flagged) / len(benched):.0f}",
        "n_drift": str(s["measure"]["n_drift_flag"]), "n_load": str(s["measure"]["n_load_info"]),
        "affected_tasks": affected,
        "omp_min": f"{min(cells[t]['c_openmp']['speedup'] for t, _ in TASKS):.2f}",
        # 면담(11/11) 지적 "OpenMP 가 순차보다 느렸다": 중앙값 비와, 개별 구현끼리 역전이 있는 과제
        "omp_seq": _rng([cells[t]["c_openmp"]["speedup"] / cells[t]["c_seq"]["speedup"] for t, _ in TASKS], 1),
        "omp_seq_overlap": [t for t, _ in TASKS if cells[t]["c_openmp"]["speedups"] and cells[t]["c_seq"]["speedups"]
                            and min(cells[t]["c_openmp"]["speedups"]) < max(cells[t]["c_seq"]["speedups"])],
        "n_tests": str(_count_unit_tests()),
        "nbody_fail_ratio": _nbody_numpy_fail_ratio(rows),
        "gen_dates": sorted({rid[:8] for rid in RUNS}),
        "tier_pass": {str(tier): f"{sum(r['pass'] == '1' for r in rows if dict(TASKS)[r['task']] == tier)}/"
                                 f"{sum(1 for r in rows if dict(TASKS)[r['task']] == tier)}" for tier in (1, 2, 3)},
    }
    return s


def _count_unit_tests() -> int:
    """tests/ 의 단위 테스트 메서드 수(본문 3.3)."""
    return sum(len(re.findall(r"^\s+def test_\w+\(", p.read_text(encoding="utf-8"), re.M)) for p in sorted((ROOT / "tests").glob("test_*.py")))


def _nbody_numpy_fail_ratio(rows: list[dict[str, str]]) -> list[str]:
    """nbody 의 numpy 구현이 실패한 정확성 형태에서 허용오차를 넘은 배율의 범위(본문 4.4). 가장 큰 형태(small)는 모두 통과해야 한다."""
    ratios: list[float] = []
    for r in rows:
        if r["task"] == "nbody" and r["language"] == "numpy":
            tests = {t["name"]: t for t in json.loads(r["check_tests"])}
            if tests["small"]["classification"] != "pass" or any(tests[k]["classification"] == "pass" for k in ("tiny", "rect")):
                raise SystemExit("[build] nbody numpy 의 형태별 판정이 본문(작은 두 형태에서만 실패)과 다르다")
            ratios += [float(tests[k]["max_tol_ratio"]) for k in ("tiny", "rect")]
    return [f"{min(ratios):.1f}", f"{max(ratios):.0f}"]


def check_claims(s: dict) -> None:
    """본문이 기대는 사실을 데이터와 대조한다. 어긋나면 빌드를 멈춘다(그때는 main.typ 의 문장도 고쳐야 한다).

    측정을 다시 해도 변하지 않는 값(통과 수, 실패의 종류, 토큰, 코드의 특징)은 값 자체를, 측정값은 문장의 뜻이 성립하는
    조건(대등/큰 차이 등)을 검사한다."""
    cells = {t["task"]: t["cells"] for t in s["tasks"]}
    errs: list[str] = []

    def expect(name: str, got, want) -> None:
        if got != want:
            errs.append(f"{name}: 데이터 {got!r} ≠ 본문 {want!r}")

    def holds(name: str, cond: bool, detail: object = "") -> None:
        if not cond:
            errs.append(f"{name}: 본문의 서술이 성립하지 않는다 {detail}")

    # ---- 측정과 무관한 사실
    expect("전체 응답 수", s["totals"]["all"]["n"], 180)
    expect("코드가 추출된 구현 수", s["totals"]["all"]["n_extracted"], 176)
    expect("통과 수(numpy, c_seq, c_openmp)", [s["totals"][lang]["n_pass"] for lang in LANGS], [55, 55, 53])
    expect("오류 유형별 실패 수", s["failure_by_type"],
           {"wrong_output": 9, "signature_mismatch": 4, "numeric_instability": 3, "runtime_crash": 1})
    expect("실패한 (과제, 조건, 샘플) = 원인을 판정해 둔 집합",
           sorted((f["task"], f["language"], f["sample_idx"]) for f in s["failures"]), sorted(FAILURE_CAUSE))
    expect("표 3 의 조건별 합계 = 조건별 실패 수", {lang: sum(row[lang] for row in s["failure_table"]) for lang in LANGS},
           {lang: s["totals"][lang]["n"] - s["totals"][lang]["n_pass"] for lang in LANGS})
    fmt = [f for f in s["failures"] if f["cause"] == "format"]
    expect("형식 위반은 전부 추출 단계·코드 블록 없음", sorted({(f["error_stage"], f["error_detail"], f["error_type"]) for f in fmt}),
           [("extract", "no_code_block", "signature_mismatch")])
    expect("C 의 알고리즘·메모리 실패 수", s["c"]["n_fail_c_algo"], "8")
    expect("thinking 토큰이 있는 칸", s["tokens"]["thinking_cells"], [["fft", "c_openmp"], ["fft", "c_seq"]])
    expect("thinking 이 있는 응답 수", s["tokens"]["n_rows_thinking"], 10)
    expect("토큰 합(입력, 출력)", [s["tokens"]["in"], s["tokens"]["out"]], [567055, 350876])
    mt = s["tokens"]["mean_text_out"]
    holds("numpy 의 평균 출력 토큰(추론 제외)은 C 조건의 약 절반", 0.4 <= mt["numpy"] / st.mean([mt["c_seq"], mt["c_openmp"]]) <= 0.6, mt)
    m = s["measure"]
    expect("모델", m["models"], ["claude-sonnet-5-5"])
    expect("effort", m["efforts"], ["medium"])
    expect("temperature 미전송", m["temperatures"], [""])
    expect("수정 턴 없음", m["turns"], ["0"])
    expect("거버너", m["governors"], ["performance"])
    expect("CPU", m["cpu_models"], ["12th Gen Intel(R) Core(TM) i7-1260P"])
    expect("컴파일러", m["compilers"], ["gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0"])
    expect("NumPy", m["numpy_versions"], ["2.3.5"])
    expect("BLAS", m["blas"], ["scipy-openblas 0.3.30"])
    expect("측정된 행 수 = 통과 수", m["n_benchmarked"], s["totals"]["all"]["n_pass"])

    # ---- 측정값: 문장의 뜻이 성립하는 조건
    ratio = {t: cells[t]["ratio"] for t, _ in TASKS if cells[t]["ratio"] is not None}
    rest = [t for t in ratio if t not in PARITY_TASKS]
    expect("c_openmp/numpy 비가 정의된 과제 수(nbody 제외)", len(ratio), 11)
    holds("대등한 네 과제: c_openmp/numpy 가 1.5배 이내", all(1 / 1.5 <= ratio[t] <= 1.5 for t in PARITY_TASKS),
          {t: round(ratio[t], 2) for t in PARITY_TASKS})
    holds("나머지 일곱 과제: c_openmp 가 2.5배 넘게 빠르다", len(rest) == 7 and all(ratio[t] > 2.5 for t in rest),
          {t: round(ratio[t], 2) for t in rest})
    aa = cells["gemm"]["numpy"]
    holds("A/A 대조(gemm numpy = 기준선과 같은 호출): 중앙값이 1 의 ±20% 안", 0.8 <= aa["speedup"] <= 1.2, aa["speedup"])
    non_lib_np = [t for t, _ in TASKS if t not in LIBRARY_BASELINE and cells[t]["numpy"]["speedup"] is not None]
    expect("C 기준선 과제 중 numpy 가 통과한 과제 수", len(non_lib_np), 6)
    holds("C 기준선 과제에서 numpy 속도비 < 0.5", all(cells[t]["numpy"]["speedup"] < 0.5 for t in non_lib_np))
    holds("같은 과제에서 c_openmp 속도비 ≥ 0.9", all(cells[t]["c_openmp"]["speedup"] >= 0.9 for t in non_lib_np))
    holds("같은 과제에서 c_seq 속도비 < c_openmp 속도비", all(cells[t]["c_seq"]["speedup"] < cells[t]["c_openmp"]["speedup"] for t in non_lib_np))
    holds("라이브러리 기준선 과제에서 numpy 속도비 ≥ 0.8", all(cells[t]["numpy"]["speedup"] >= 0.8 for t in LIBRARY_BASELINE))
    holds("nbody·cholesky·fft 의 C 속도비 > 2",
          all(cells[t][lang]["speedup"] > 2 for t in ("nbody", "cholesky", "fft") for lang in C_LANGS))
    holds("dot numpy 는 기준선보다 빠르다(최솟값 > 1.2)", min(cells["dot"]["numpy"]["speedups"]) > 1.2)
    holds("c_openmp 속도비 중앙값이 모든 과제에서 c_seq 보다 높다",
          all(cells[t]["c_openmp"]["speedup"] > cells[t]["c_seq"]["speedup"] for t, _ in TASKS))
    holds("c_openmp 속도비는 모든 과제에서 0.8 이상(직접 결정의 속도 손실 없음)", all(cells[t]["c_openmp"]["speedup"] >= 0.8 for t, _ in TASKS),
          {t: round(cells[t]["c_openmp"]["speedup"], 2) for t, _ in TASKS})

    # ---- 생성 코드의 특징
    c = s.get("code")
    if c is None:
        print("[build] 경고: results/built 가 없어 생성 코드 특징 검사를 건너뛴다(본문의 해당 수치는 이전 값).", file=sys.stderr)
    else:
        expect("numpy 스레드 사용 수", c["numpy_threads"], [32, 60])
        expect("numpy 스레드 5/5 과제", sorted(t for t, v in c["numpy_threads_by_task"].items() if v == 5),
               ["dot", "histogram", "jacobi2d", "nbody", "scan"])
        expect("C 내장 함수 사용 수", c["c_intrinsics"], [64, 116])
        expect("gemm C 내장 함수", c["c_intrinsics_by_task"]["gemm"], [10, 10])
        expect("gemm numpy 행렬곱 위임", c["gemm_numpy_matmul"], [5, 5])
        expect("nbody numpy 행렬곱 재배열", c["nbody_numpy_matmul"], [5, 5])
        expect("nbody C rsqrt", c["nbody_c_rsqrt"], [10, 10])
        expect("cholesky numpy 행렬곱", c["cholesky_numpy_matmul"], [4, 5])
        expect("floyd_warshall C 블록화", c["floyd_c_blocked"], [10, 10])
        expect("bfs C 의 bottom-up 코드", c["bfs_c_bottom_up"], [["c_seq", 2], ["c_seq", 4]])
        expect("fft C 의 배정밀도 회전 인자", c["fft_c_double"], [10, 10])
        cc = s["c"]
        holds("cholesky numpy: 블록 분해 구현이 단일 호출 구현보다 1.5배 넘게 빠르다",
              cc["chol_blocked"] is not None and cc["chol_plain"] is not None and float(cc["chol_blocked"][0]) > 1.5 * float(cc["chol_plain"]),
              (cc["chol_blocked"], cc["chol_plain"]))
        expect("스레드를 쓰면서 c_openmp 와 차이가 큰 numpy 과제", cc["thr_tasks"], ["jacobi2d", "histogram", "scan"])
    if errs:
        raise SystemExit("[build] 본문과 데이터가 어긋난다:\n  " + "\n  ".join(errs))


# ----------------------------------------------------------------------------- 그림 2

# dataviz 기준: 범주형 3슬롯(파랑·주황·청록; validate_palette.js --pairs all 통과, 흰 바탕), 표식 모양으로 이중 부호화(인쇄 대비),
# 격자는 실선 헤어라인, 글자는 잉크색(9pt 이상), 표식 ≥ 8px + 바탕색 테두리. 값은 표 2 가 표 보기 역할을 한다.
SERIES = {
    "numpy": {"color": "#2a78d6", "marker": "o", "label": "numpy"},
    "c_seq": {"color": "#eb6834", "marker": "s", "label": "c_seq"},
    "c_openmp": {"color": "#1baf7a", "marker": "^", "label": "c_openmp"},
}
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TICK_LABEL = {"floyd_warshall": "floyd_\nwarshall"}
TIER_NAME = {1: "수준 1: 기본 커널", 2: "수준 2: 병렬화·알고리즘 재구성", 3: "수준 3: 의존성·불규칙 접근"}
FS = 9.0  # 그림 안 글자 크기(pt). 그림을 실제 크기(폭 190 mm)로 넣으므로 양식의 9pt 하한과 같다.


def make_figure(s: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    font = "NanumGothic" if any(f.name == "NanumGothic" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": font, "font.size": FS, "svg.fonttype": "path", "axes.unicode_minus": False})

    mono = "D2Coding ligature" if any(f.name == "D2Coding ligature" for f in font_manager.fontManager.ttflist) else "DejaVu Sans Mono"
    fig, ax = plt.subplots(figsize=(7.48, 2.95))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # x 위치: 난이도 수준 사이에 간격
    xs: dict[str, float] = {}
    x = 0.0
    prev_tier = None
    tier_span: dict[int, list[float]] = {}
    for t in s["tasks"]:
        if prev_tier is not None and t["tier"] != prev_tier:
            x += 0.45
        xs[t["task"]] = x
        tier_span.setdefault(t["tier"], []).append(x)
        prev_tier = t["tier"]
        x += 1.0
    dodge = {"numpy": -0.27, "c_seq": 0.0, "c_openmp": 0.27}
    jitter = (-0.10, 0.10, -0.05, 0.05, 0.0)   # 개별 구현이 중앙값 표식에 가려지지 않게 좌우로 조금 벌린다(속이 빈 원)

    ax.set_yscale("log")
    ymin, ymax = 0.045, 22
    ax.set_ylim(ymin, ymax)
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"]))
    ax.yaxis.set_minor_locator(NullLocator())
    for y in ticks:
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)
    ax.axhline(1.0, color=MUTED, lw=1.0, zorder=1)

    for t in s["tasks"]:
        for lang in LANGS:
            c = t["cells"][lang]
            sty = SERIES[lang]
            px = xs[t["task"]] + dodge[lang]
            if not c["speedups"]:
                continue
            ax.plot([px, px], [c["speedup_min"], c["speedup_max"]], color=sty["color"], lw=1.0, alpha=0.5, zorder=2, solid_capstyle="round")
            ax.scatter([px + jitter[i % len(jitter)] for i in range(len(c["speedups"]))], c["speedups"], s=13, marker="o",
                       facecolors="white", edgecolors=sty["color"], linewidths=0.8, zorder=3)
            ax.scatter([px], [c["speedup"]], s=44, marker=sty["marker"], color=sty["color"], edgecolors="white", linewidths=1.2, zorder=4)

    # 선택적 직접 표기: 극값
    cells = {t["task"]: t["cells"] for t in s["tasks"]}

    def note(task: str, lang: str, dx: float, dy: float, ha: str) -> None:
        c = cells[task][lang]
        ax.annotate(f"{c['speedup']:.2f}", (xs[task] + dodge[lang], c["speedup"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=FS, color=INK2, ha=ha, va="center")
    note("nbody", "c_openmp", 8, 0, "left")
    note("fft", "c_openmp", -8, 0, "right")
    note("floyd_warshall", "numpy", -8, 0, "right")
    # 통과한 구현이 없는 칸: 자료점처럼 보이는 표식을 두지 않고 글로만 적는다
    for t in s["tasks"]:
        for lang in LANGS:
            if not t["cells"][lang]["speedups"]:
                ax.text(xs[t["task"]] + dodge[lang] - 0.12, 0.3, f"{lang}\n통과 0/{t['cells'][lang]['n']}", fontsize=FS, color=INK2,
                        ha="center", va="center", linespacing=1.15)

    # x 눈금 이름은 기준선(baseline) 정렬로 직접 그린다(글자 높이에 따라 줄이 들쭉날쭉해지지 않게). 두 줄 이름은 줄마다 따로 놓는다.
    # 기준선이 NumPy 루틴인 과제에는 표 1 과 같은 * 를 붙인다.
    ax.set_xticks([xs[t["task"]] for t in s["tasks"]])
    ax.set_xticklabels([])
    tr = ax.get_xaxis_transform()
    for t in s["tasks"]:
        lines = TICK_LABEL.get(t["task"], t["task"]).split("\n")
        if t["library_baseline"]:
            lines[-1] += "*"
        for k, line in enumerate(lines):
            ax.text(xs[t["task"]], -0.09 - 0.072 * k, line, transform=tr, ha="center", va="baseline", fontsize=FS, color=INK,
                    fontfamily=mono, clip_on=False)
    ax.set_xlim(-0.6, x - 0.4)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=0, labelsize=FS, labelcolor=INK2, pad=3)
    ax.set_ylabel("기준선 대비 속도비 (로그 축)", fontsize=FS, color=INK2, labelpad=4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.8)

    for tier, pos in tier_span.items():
        ax.text((pos[0] + pos[-1]) / 2, ymax * 0.82, TIER_NAME[tier], ha="center", va="top", fontsize=FS, color=INK2)
    for tier in (1, 2):
        xb = (tier_span[tier][-1] + tier_span[tier + 1][0]) / 2
        ax.axvline(xb, color=GRID, lw=0.6, zorder=0)

    handles = [Line2D([], [], marker=v["marker"], color="none", markerfacecolor=v["color"], markeredgecolor="white",
                      markeredgewidth=1.0, markersize=7, label=v["label"]) for v in SERIES.values()]
    handles += [Line2D([], [], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor="white", markersize=7, label="중앙값"),
                Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=0.8, markersize=3.6,
                       label="개별 구현")]
    leg = fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=FS, handletextpad=0.2, columnspacing=1.6,
                     bbox_to_anchor=(0.53, 1.0))
    for txt in leg.get_texts():
        txt.set_color(INK)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.90, bottom=0.16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="svg", metadata={"Date": None})
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)


def compile_pdf(out: Path) -> None:
    import typst
    result = typst.compile_with_warnings(str(HERE / "main.typ"), output=str(out), root=str(ROOT))
    warnings = result[1] if isinstance(result, tuple) else []
    for w in warnings:
        print(f"[build] typst 경고: {str(w)[:300]}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="중간보고서 빌드 (results/runs.csv → PDF)")
    ap.add_argument("--no-pdf", action="store_true", help="수치·그림만 만들고 조판은 하지 않는다")
    ap.add_argument("--out", type=Path, default=HERE / "interim_report.pdf")
    ns = ap.parse_args(argv)
    rows = load_rows()
    res_path = GEN / "results.json"
    prev = json.loads(res_path.read_text(encoding="utf-8")) if res_path.is_file() else {}
    code = code_features(rows)
    aff = affinity_status(rows)
    code_checked = code is not None
    if code is None:                                  # 새 복제본: 커밋된 값을 유지한다(검사는 건너뛴다)
        code = prev.get("code")
    if aff is None:
        aff = prev.get("affinity")
    s = summarize(rows, code, aff)
    s["code_checked"] = code_checked
    if not code_checked:
        saved, s["code"] = s["code"], None            # check_claims 가 코드 특징 검사를 건너뛰게 한다
        check_claims(s)
        s["code"] = saved
    else:
        check_claims(s)
    GEN.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    make_figure(s, GEN / "fig_speedup.svg")
    state = "초안(호출 친화도 수정 전의 측정 포함)" if s["draft"] else "최종 측정"
    print(f"[build] 수치 {res_path}, 그림 {GEN / 'fig_speedup.svg'} — {state}", file=sys.stderr)
    if not ns.no_pdf:
        compile_pdf(ns.out)
        print(f"[build] PDF {ns.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
