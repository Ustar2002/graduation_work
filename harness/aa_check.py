"""A/A 점검: 과제의 기준선을 **자기 자신과** 교차 측정해 측정 장치의 치우침을 잰다(2026-11-12).

    .venv/bin/python -m harness.aa_check [--tasks gemm,dot] [--rounds 3] [--variants fsync,nofsync] [--run-id aa-…]

같은 코드를 커널 자리와 기준선 자리에 함께 넣으면 속도비는 정확히 1 이어야 한다. 1 에서 벗어난 만큼이 교차 측정 루프
자체의 치우침이다. 두 가지를 본다.

- ``ratio_mean`` / ``ratio_median``: (기준선 자리 평균·중앙값) / (커널 자리 평균·중앙값). 라운드마다와 모은 값.
- ``first_vs_second``: 호출 쌍의 **첫 호출**(쌍 사이 진행 기록·fsync 뒤에 바로 오는 호출) 중앙값 / **둘째 호출** 중앙값.
  같은 코드이므로 1 이어야 한다. 1 보다 크면 쌍 사이의 유휴(진행 기록의 fsync)가 첫 호출을 느리게 만든다는 뜻이다.

변형(variant): ``fsync`` = 현행(쌍 사이 진행 기록을 fsync), ``nofsync`` = ``--no-loop-fsync``. 라운드마다 두 변형의 순서를
바꿔(ABAB / BABA) 시간에 따른 표류가 한쪽에 몰리지 않게 한다. 데스크톱에서 ``measurement.bench_loop_fsync`` 를 정하는 근거로 쓴다
(docs/decisions.md 2026-11-12). 기준선 빌드·코어 고정·환경은 ``harness.run`` 과 같다. 결과 행(runs.csv)은 건드리지 않는다.
요약은 ``results/aa/<run_id>.json``, 워커 결과는 ``results/built/<run_id>/<task>/baseline/aa/``.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from harness import protocol
from harness.benchstats import paired_rounds, ratio_of_means
from harness.config import ConfigError, load_config, load_machine
from harness.paths import ROOT, TASKS_DIR, atomic_write_json, baseline_dir, ensure_dir, rel_to_root
from harness.run import (BaselineError, HarnessError, RunContext, TaskContext, baseline_artifact, bench_args, bench_plan,
                         ensure_tests, info, load_signatures, run_worker, warn)
from harness.schema import harness_version
from harness.taskspec import load_taskspec

_LOG = "[aa_check]"
VARIANTS = ("fsync", "nofsync")


def split_by_position(doc: dict[str, Any]) -> tuple[list[float], list[float]]:
    """교차 bench 결과 문서 → (쌍의 첫 호출 시간들, 둘째 호출 시간들). timed 호출만 쓴다.

    워커는 호출 쌍 번호 ``call_idx``(워밍업 포함, 0부터)가 짝수면 기준선을 먼저, 홀수면 커널을 먼저 부른다(worker.run_bench).
    timed 의 j 번째 쌍은 ``call_idx = (워밍업 수) + j`` 다."""
    k = [float(x) for x in (doc.get("timed_ms") or [])]
    b = [float(x) for x in ((doc.get("baseline_interleaved") or {}).get("timed_ms") or [])]
    warm = len(doc.get("warmup_ms") or [])
    first: list[float] = []
    second: list[float] = []
    for j in range(min(len(k), len(b))):
        baseline_first = (warm + j) % 2 == 0
        first.append(b[j] if baseline_first else k[j])
        second.append(k[j] if baseline_first else b[j])
    return first, second


def summarize_variant(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """한 변형의 라운드 문서들 → A/A 지표."""
    k_rounds = [[float(x) for x in (d.get("timed_ms") or [])] for d in docs]
    b_rounds = [[float(x) for x in ((d.get("baseline_interleaved") or {}).get("timed_ms") or [])] for d in docs]
    pairs = paired_rounds(k_rounds, b_rounds)
    per_round = [ratio_of_means([rp]) for rp in pairs]
    k_all = [k for rp in pairs for k, _ in rp]
    b_all = [b for rp in pairs for _, b in rp]
    first: list[float] = []
    second: list[float] = []
    for d in docs:
        f, s = split_by_position(d)
        first += f
        second += s
    return {
        "rounds": len(docs),
        "n_pairs": len(k_all),
        "ratio_mean": ratio_of_means(pairs) if pairs else None,
        "ratio_median": (st.median(b_all) / st.median(k_all)) if k_all and b_all else None,
        "ratio_mean_rounds": [r for r in per_round if r is not None],
        "first_vs_second": (st.median(first) / st.median(second)) if first and second else None,
        "kernel_cv": (st.stdev(k_all) / st.fmean(k_all)) if len(k_all) >= 2 else None,
        "median_ms": st.median(k_all + b_all) if k_all else None,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m harness.aa_check", description="기준선 대 기준선 교차 측정(A/A)으로 측정 장치의 치우침을 잰다")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--run-id", default=None, help="기본 aa-YYYYmmdd-HHMMSS")
    p.add_argument("--tasks", default=None, help="쉼표 구분(기본: run.tasks 전체)")
    p.add_argument("--rounds", type=int, default=3, help="변형마다 라운드 수(기본 3)")
    p.add_argument("--variants", default=",".join(VARIANTS), help="fsync,nofsync 중 쉼표 구분")
    p.add_argument("--out", type=Path, default=None, help="요약 JSON(기본 results/aa/<run_id>.json)")
    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    try:
        cfg = load_config(ns.config)
    except ConfigError as e:
        print(f"{_LOG} 설정 오류: {e}", file=sys.stderr)
        return 2
    variants = [v.strip() for v in ns.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad or not variants or ns.rounds < 1:
        print(f"{_LOG} --variants 는 {VARIANTS} 중에서, --rounds 는 1 이상", file=sys.stderr)
        return 2
    run_id = ns.run_id or datetime.now().strftime("aa-%Y%m%d-%H%M%S")
    tasks = [t.strip() for t in ns.tasks.split(",")] if ns.tasks else list(cfg.run.tasks)
    machine = load_machine()
    static = protocol.check_static(cfg, machine)
    if [f for f in static.flags if not protocol.is_informational(f)]:
        warn(f"정적 규약 플래그 {[f for f in static.flags if not protocol.is_informational(f)]} — A/A 결과도 그 상태의 값이다")
    fingerprint, env_hash = protocol.environment_fingerprint(cfg, machine, static)
    ctx = RunContext(cfg=cfg, machine=machine, static=static, fingerprint=fingerprint, env_hash=env_hash, run_id=run_id,
                     skip_bench=False, force=False, version=harness_version())
    m = cfg.measurement
    summary: dict[str, Any] = {"run_id": run_id, "created_at": protocol.now_iso(), "harness_version": ctx.version, "env_hash": env_hash,
                               "governor": static.governor, "rounds": ns.rounds, "variants": variants, "tasks": {}}
    for task in tasks:
        test_dir, manifest = ensure_tests(cfg, task)
        sig_path = TASKS_DIR / task / "signatures.yaml"
        tctx = TaskContext(task=task, test_dir=test_dir, manifest=manifest, tolerance_path=TASKS_DIR / task / "tolerance.yaml",
                           signatures=load_signatures(task), spec=load_taskspec(sig_path))
        b = tctx.spec.baseline
        b_openmp = b["backend"] == "c" and bool(b.get("openmp"))
        bdir = ensure_dir(baseline_dir(cfg.paths.built_dir, run_id, task))
        try:
            art = baseline_artifact(ctx, tctx, bdir)
        except (BaselineError, FileNotFoundError) as e:
            warn(f"{task}: 기준선을 준비하지 못해 건너뜀 ({e})")
            continue
        # 기준선을 '커널' 자리에도 넣는다: 같은 규칙(protocol.call_cpus)으로 두 자리의 호출 친화도가 같게 된다
        plan = bench_plan(cfg, backend=str(b["backend"]), openmp=b_openmp, threads=m.threads, baseline=b)
        aa_dir = ensure_dir(bdir / "aa")
        docs: dict[str, list[dict[str, Any]]] = {v: [] for v in variants}
        for r in range(1, ns.rounds + 1):
            order = variants if r % 2 == 1 else list(reversed(variants))
            for v in order:
                extra = [a for a in bench_args(cfg) if a != "--no-loop-fsync"]
                if v == "nofsync":
                    extra.append("--no-loop-fsync")
                extra += ["--baseline-backend", str(b["backend"]), "--baseline-artifact", str(art)]
                extra += plan.worker_args(with_baseline=True)
                name = f"{v}_r{r}"
                try:
                    oc = run_worker(cfg, stage=name, threads=plan.threads, openmp=plan.openmp_env, backend=str(b["backend"]), artifact=art,
                                    mode="bench", tests=[tctx.test_path(m.performance_test)], tolerance=tctx.tolerance_path,
                                    signatures=tctx.spec.path, out_path=aa_dir / f"{name}.json", cwd=aa_dir, timeout_s=m.bench_timeout_s,
                                    extra=extra)
                except HarnessError as e:
                    print(f"{_LOG} harness_error ({task} {name}): {e}", file=sys.stderr)
                    return 3
                verdict = oc.verdict()
                doc = oc.doc or {}
                if verdict is not None or doc.get("classification") != "pass" or \
                        (doc.get("baseline_interleaved") or {}).get("classification") != "pass":
                    warn(f"{task} {name}: 측정 실패({verdict or doc.get('classification')}) — 이 라운드는 빠진다")
                    continue
                docs[v].append(doc)
        summary["tasks"][task] = {"baseline": rel_to_root(art), "results_dir": rel_to_root(aa_dir),
                                  "variants": {v: summarize_variant(docs[v]) for v in variants}}
        row = summary["tasks"][task]["variants"]
        info(f"{task}: " + " | ".join(
            f"{v}: 평균 비 {s['ratio_mean']:.3f} (라운드 {', '.join(f'{x:.3f}' for x in s['ratio_mean_rounds'])}), "
            f"첫/둘째 {s['first_vs_second']:.3f}" if s["ratio_mean"] is not None and s["first_vs_second"] is not None else f"{v}: 결과 없음"
            for v, s in row.items()))
    out = ns.out or (ROOT / "results" / "aa" / f"{run_id}.json")
    atomic_write_json(out, summary)
    print(f"{_LOG} 요약 {rel_to_root(out)}", file=sys.stderr)
    _print_table(summary)
    return 0


def _print_table(summary: dict[str, Any]) -> None:
    print(f"{'과제':15s} {'변형':8s} {'평균 비':>8s} {'중앙값 비':>9s} {'라운드 범위':>17s} {'첫/둘째':>8s} {'커널 CV':>8s} {'시간 ms':>8s}")
    for task, t in summary["tasks"].items():
        for v, s in t["variants"].items():
            rr = s["ratio_mean_rounds"]
            span = f"{min(rr):.3f}~{max(rr):.3f}" if rr else "-"

            def f(x: float | None, d: int = 3) -> str:
                return f"{x:.{d}f}" if x is not None else "-"
            print(f"{task:15s} {v:8s} {f(s['ratio_mean']):>8s} {f(s['ratio_median']):>9s} {span:>17s} {f(s['first_vs_second']):>8s} "
                  f"{f(s['kernel_cv']):>8s} {f(s['median_ms'], 2):>8s}")


if __name__ == "__main__":
    sys.exit(main())
