"""여러 라운드의 교차 bench 를 결과 행 하나로 합친다(2026-11-12 측정 규약).

규약: 구현마다 **서로 다른 시각의 별도 워커 프로세스**에서 bench 를 ``measurement.bench_rounds`` 번(기본 3) 돈다. 각 라운드는
워밍업 + timed 호출이며 커널과 기준선을 같은 프로세스에서 번갈아 부른다(교차 측정, 2026-11-09). 이 모듈은 라운드별 결과
문서(``bench.json``, ``bench_r2.json``, …)를 받아 다음을 계산한다.

- 커널·기준선 시간: 모든 라운드의 timed 호출을 모은 중앙값·평균·최솟값·최댓값·변동계수(표본 표준편차/평균).
- ``speedup_vs_baseline`` = 기준선 중앙값 / 커널 중앙값(이전 규약과 같은 정의, 라운드를 모은 값).
- ``speedup_mean`` = 기준선 평균 / 커널 평균(면담 결정: "반복 횟수를 늘려 평균"), 95% 신뢰구간은 **라운드별 층화 부트스트랩**
  (라운드마다 같은 호출 쌍 번호로 커널·기준선을 함께 복원 추출 → 교차 측정의 짝을 보존). 난수 씨앗은 행 키에서 정해
  같은 데이터면 같은 구간이 나온다.
- ``speedup_rounds``: 라운드별 기준선 평균 / 커널 평균(라운드 사이의 변동을 보는 용도).

순수 파이썬(표준 라이브러리)만 쓴다: 오케스트레이터는 numpy 없이도 돌아야 한다.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics as st
from typing import Any, Iterable, Sequence

N_BOOTSTRAP = 2000
CI_LEVEL = 0.95


def seed_of(key: Iterable[Any]) -> int:
    """행 키 → 결정적 난수 씨앗(같은 행이면 같은 신뢰구간)."""
    text = "|".join(str(k) for k in key)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _cv(xs: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = st.fmean(xs)
    return st.stdev(xs) / m if m > 0 else None


def describe(xs: Sequence[float]) -> dict[str, float | None]:
    """중앙값·평균·최솟값·최댓값·변동계수(표본 표준편차 / 평균). 비어 있으면 모두 None."""
    if not xs:
        return {"median_ms": None, "mean_ms": None, "min_ms": None, "max_ms": None, "cv": None}
    return {"median_ms": float(st.median(xs)), "mean_ms": float(st.fmean(xs)), "min_ms": float(min(xs)),
            "max_ms": float(max(xs)), "cv": _cv(xs)}


def paired_rounds(kernel_rounds: Sequence[Sequence[float]], baseline_rounds: Sequence[Sequence[float]]) -> list[list[tuple[float, float]]]:
    """라운드마다 (커널, 기준선) 호출 쌍. 길이가 다른 라운드는 짧은 쪽에 맞춘다(중간에 기준선이 실패한 경우)."""
    out: list[list[tuple[float, float]]] = []
    for k, b in zip(kernel_rounds, baseline_rounds):
        n = min(len(k), len(b))
        if n:
            out.append([(float(k[i]), float(b[i])) for i in range(n)])
    return out


def ratio_of_means(pairs: Sequence[Sequence[tuple[float, float]]]) -> float | None:
    """모든 쌍을 모은 기준선 평균 / 커널 평균."""
    sk = sum(k for rp in pairs for k, _ in rp)
    sb = sum(b for rp in pairs for _, b in rp)
    return sb / sk if sk > 0 else None


def bootstrap_ratio_ci(pairs: Sequence[Sequence[tuple[float, float]]], *, seed: int, n_boot: int = N_BOOTSTRAP,
                       level: float = CI_LEVEL) -> tuple[float | None, float | None]:
    """기준선 평균 / 커널 평균의 백분위 부트스트랩 신뢰구간. 라운드마다 쌍을 복원 추출한다(층화)."""
    rounds = [list(rp) for rp in pairs if rp]
    if not rounds:
        return None, None
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_boot):
        sk = sb = 0.0
        for rp in rounds:
            m = len(rp)
            for _ in range(m):
                k, b = rp[rng.randrange(m)]
                sk += k
                sb += b
        if sk > 0:
            stats.append(sb / sk)
    if not stats:
        return None, None
    stats.sort()
    alpha = (1.0 - level) / 2.0
    lo_idx = int(math.floor(alpha * (len(stats) - 1)))
    hi_idx = int(math.ceil((1.0 - alpha) * (len(stats) - 1)))
    return stats[lo_idx], stats[hi_idx]


def combine_rounds(docs: Sequence[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    """라운드 순서의 bench 결과 문서들 → 행에 넣을 값.

    반환 키: ``kernel``(describe 결과), ``baseline``(describe 결과 또는 None), ``time_samples_ms``, ``baseline_samples_ms``,
    ``warmup``·``repeats``(라운드 합계), ``repeats_reduced``, ``gflops``(모은 중앙값 기준), ``cpu_mhz_median``, ``cpu_mhz_min``,
    ``throttle_events``(합), ``temp_c_start``(첫 라운드), ``temp_c_end``(마지막 라운드), ``nondeterministic_runs``,
    ``nondeterministic_bitwise_runs``(합), ``inputs_mutated``, ``oob_write``(어느 라운드든), ``rounds``(라운드 수),
    ``speedup_median``, ``speedup_mean``, ``speedup_ci``(lo, hi), ``speedup_rounds``(라운드별 평균 비).
    """
    k_rounds: list[list[float]] = []
    b_rounds: list[list[float]] = []
    k_warm: list[float] = []
    b_warm: list[float] = []
    mhz: list[float] = []
    throttle: int | None = None
    work: float | None = None
    reduced = False
    nondet = nondet_bit = 0
    mutated = oob = False
    for d in docs:
        timed = [float(x) for x in (d.get("timed_ms") or [])]
        k_rounds.append(timed)
        k_warm += [float(x) for x in (d.get("warmup_ms") or [])]
        mhz += [float(x) for x in (d.get("cpu_mhz_samples") or [])]
        if d.get("throttle_events") is not None:
            throttle = (throttle or 0) + int(d["throttle_events"])
        if work is None and d.get("gflops") and d.get("median_ms"):
            work = float(d["gflops"]) * float(d["median_ms"]) * 1e6   # gflops = work / (median_ms/1e3) / 1e9
        reduced = reduced or bool(d.get("repeats_reduced"))
        nondet += int(d.get("nondeterministic_runs") or 0)
        nondet_bit += int(d.get("nondeterministic_bitwise_runs") or 0)
        mutated = mutated or bool(d.get("inputs_mutated"))
        oob = oob or bool(d.get("oob_write"))
        bd = d.get("baseline_interleaved") or {}
        if bd.get("classification") == "pass" and bd.get("timed_ms"):
            b_rounds.append([float(x) for x in bd["timed_ms"]])
            b_warm += [float(x) for x in (bd.get("warmup_ms") or [])]
        else:
            b_rounds.append([])
    k_all = [x for r in k_rounds for x in r]
    b_all = [x for r in b_rounds for x in r]
    kernel = describe(k_all)
    baseline = describe(b_all) if b_all else None
    pairs = paired_rounds(k_rounds, b_rounds)
    sp_median = (baseline["median_ms"] / kernel["median_ms"]) if (baseline and kernel["median_ms"]) else None
    sp_mean = ratio_of_means(pairs) if pairs else None
    ci = bootstrap_ratio_ci(pairs, seed=seed) if pairs else (None, None)
    sp_rounds = [ratio_of_means([rp]) for rp in pairs]
    gflops = (work / (kernel["median_ms"] / 1e3) / 1e9) if (work and kernel["median_ms"]) else None
    return {
        "kernel": kernel,
        "baseline": baseline,
        "time_samples_ms": {"warmup": k_warm, "timed": k_all, "rounds": [len(r) for r in k_rounds]},
        "baseline_samples_ms": ({"warmup": b_warm, "timed": b_all, "rounds": [len(r) for r in b_rounds]} if b_all else None),
        "warmup": sum(len(d.get("warmup_ms") or []) for d in docs),
        "repeats": len(k_all),
        "repeats_reduced": reduced,
        "gflops": gflops,
        "cpu_mhz_median": float(st.median(mhz)) if mhz else None,
        "cpu_mhz_min": float(min(mhz)) if mhz else None,
        "throttle_events": throttle,
        "temp_c_start": docs[0].get("temp_c_start") if docs else None,
        "temp_c_end": docs[-1].get("temp_c_end") if docs else None,
        "nondeterministic_runs": nondet,
        "nondeterministic_bitwise_runs": nondet_bit,
        "inputs_mutated": mutated,
        "oob_write": oob,
        "rounds": len(docs),
        "speedup_median": sp_median,
        "speedup_mean": sp_mean,
        "speedup_ci": ci,
        "speedup_rounds": [r for r in sp_rounds if r is not None],
    }
