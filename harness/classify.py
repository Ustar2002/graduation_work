"""오류 분류 규칙 (설계 §10 v2.1; 고정 — 규칙을 바꾸면 docs/error_taxonomy.md 를 갱신한다).

- ``compare(out, ref64, rtol, atol)``: 허용오차 비교 지표.
- ``classify_output(...)``: 출력 하나의 판정. 규칙 순서(먼저 맞는 규칙이 결정):
  1. NaN/Inf(센티널은 유한값이라 자동 제외) → ``numeric_instability``/``nan_inf``
  2. 허용오차 내 → ``pass``
  3. arg_order 패턴(``transposed_output``, ``swapped_operands``(M==N==K), ``column_major_all``,
     ``swapped_MN_dims``(``out_full`` 이 있을 때)) → ``arg_order``. 센티널·OOB 가 남아 있어도 패턴이 맞으면
     ``arg_order`` 로 보고하고 ``n_unwritten``/``oob_write`` 는 metrics 에 그대로 둔다.
  4. 센티널·계약(전부 미기록 → 누적 → 부분 기록 → OOB 쓰기) → ``wrong_output``
  5. 모든 원소가 허용오차의 ``numeric_pattern_tol_ratio`` 배 이내 → ``numeric_instability``/``imprecise``
  6. 그 밖 → ``wrong_output``/``mismatch``
- ``aggregate_sample(tests, ...)``: 샘플 수준 합산(runtime_crash > race_condition > arg_order > wrong_output > numeric_instability > pass).

v2.1(2026-11-04): arg_order 패턴을 센티널 규칙보다 먼저 본다(그렇지 않으면 ``swapped_MN_dims`` 는 partial_write/
out_of_bounds_write 에 가려 도달 불가). 규칙 4 안에서는 ``accumulated_into_c`` 를 ``partial_write`` 보다 먼저 본다:
float32 에서 ``sentinel + ref`` 는 |ref| < 2⁻¹³ 이면 정확히 센티널로 되돌아가 '미기록'으로 세어지기 때문이다.

이 모듈은 LLM 코드를 실행하지 않는다(순수 numpy 비교). 워커와 run.py 가 함께 쓴다.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_LOG = "[classify]"

# 출력 하나의 판정값
OUTPUT_STATUSES: tuple[str, ...] = ("pass", "wrong_output", "numeric_instability", "arg_order")
# 샘플 수준 판정값(출력 판정 + 실행 결과)
SAMPLE_STATUSES: tuple[str, ...] = ("pass", "runtime_crash", "race_condition", "arg_order", "wrong_output", "numeric_instability")

# swapped_MN_dims 패턴: 전체 가드 버퍼에서 맞는 위치 비율 하한
SWAPPED_MN_MATCH_FRACTION = 0.99


@dataclass(frozen=True)
class Tolerance:
    """``tasks/<task>/tolerance.yaml`` 의 값. 정수 출력 과제는 rtol = atol = 0(정확 일치)으로 쓴다."""

    rtol: float
    atol: float
    numeric_pattern_tol_ratio: float
    sentinel: float
    task: str = ""
    dtype: str = "float32"
    reference_dtype: str = "float64"


def load_tolerance(path: Path) -> Tolerance:
    """tolerance.yaml 을 읽는다. 필수 키가 없으면 ``KeyError``."""
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: 최상위가 매핑이 아니다")
    return Tolerance(
        rtol=float(raw["rtol"]),
        atol=float(raw["atol"]),
        numeric_pattern_tol_ratio=float(raw["numeric_pattern_tol_ratio"]),
        sentinel=float(raw["sentinel"]),
        task=str(raw.get("task", "")),
        dtype=str(raw.get("dtype", "float32")),
        reference_dtype=str(raw.get("reference_dtype", "float64")),
    )


# ----------------------------------------------------------------------------- 비교


def compare(out: Any, ref64: Any, rtol: float, atol: float) -> dict[str, Any]:
    """``|out − ref| ≤ atol + rtol·|ref|`` 비교 지표.

    반환: ``ok``(비유한 원소 없음 ∧ 전부 허용오차 내), ``nan_inf``(out 에 NaN/Inf 존재),
    ``max_abs_err``·``max_tol_ratio``(유한 원소 기준; 유한 원소가 없으면 None),
    ``n_fail``(허용오차 초과 + 비유한 원소 수), ``max_abs_ref``.
    """
    o = np.asarray(out, dtype=np.float64).reshape(-1)
    r = np.asarray(ref64, dtype=np.float64).reshape(-1)
    if o.shape != r.shape:
        raise ValueError(f"compare: 크기가 다르다 out={o.shape} ref={r.shape}")
    finite = np.isfinite(o)
    nan_inf = bool(not finite.all())
    err = np.abs(o - r)
    tol = atol + rtol * np.abs(r)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        # tol == 0(정수 출력의 정확 일치)에서는 err == 0 → 0, err > 0 → inf (어떤 배율로도 '거의 맞음'이 아니다)
        ratio = np.where(finite, np.where(err > 0, err / tol, 0.0), 0.0)
    fail = (~finite) | (err > tol)
    n_fail = int(np.count_nonzero(fail))
    if finite.any():
        max_abs_err: float | None = float(err[finite].max())
        max_tol_ratio: float | None = float(ratio[finite].max())
    else:
        max_abs_err = None
        max_tol_ratio = None
    return {
        "ok": n_fail == 0,
        "nan_inf": nan_inf,
        "max_abs_err": max_abs_err,
        "max_tol_ratio": max_tol_ratio,
        "n_fail": n_fail,
        "max_abs_ref": float(np.abs(r).max()) if r.size else 0.0,
    }


def _close(out2d: np.ndarray, ref2d: np.ndarray, tol: Tolerance) -> bool:
    """유한 원소 전부가 허용오차 내인가(arg_order 패턴 매치용)."""
    return bool(compare(out2d, ref2d, tol.rtol, tol.atol)["ok"])


def match_arg_order(
    out_flat: np.ndarray,
    ref2d: np.ndarray,
    A0: np.ndarray,
    B0: np.ndarray,
    M: int,
    N: int,
    K: int,
    tol: Tolerance,
    out_full: np.ndarray | None = None,
) -> str:
    """§10-3 arg_order 패턴 이름. 매치가 없으면 ``""``. 검사 순서는 설계 순서대로(단순한 가설부터):
    ``transposed_output`` → ``swapped_operands`` → ``column_major_all`` → ``swapped_MN_dims``."""
    o = np.asarray(out_flat, dtype=np.float64).reshape(-1)
    r = np.asarray(ref2d, dtype=np.float64).reshape(M, N)
    a = np.asarray(A0, dtype=np.float64).reshape(-1)
    b = np.asarray(B0, dtype=np.float64).reshape(-1)

    # transposed_output: out.reshape(N, M).T ≈ ref
    if _close(o.reshape(N, M).T, r, tol):
        return "transposed_output"

    # swapped_operands: 정사각일 때 out ≈ B0 @ A0. 정사각에서는 column_major_all 과 대수적으로 동일하므로
    # 먼저 검사해 이 이름으로 보고한다(docs/error_taxonomy.md 에 모호성 명시).
    if M == N == K and _close(o.reshape(M, N), b.reshape(K, N) @ a.reshape(M, K), tol):
        return "swapped_operands"

    # column_major_all: 입력·출력 모두 열우선으로 해석
    A_cm = a.reshape(K, M).T  # (M, K)
    B_cm = b.reshape(N, K).T  # (K, N)
    if _close(o.reshape(N, M).T, A_cm @ B_cm, tol):
        return "column_major_all"

    # swapped_MN_dims: M≠N 일 때 C[i*M + j] 가정 — 가드 포함 전체 버퍼에서 맞는 비율
    if M != N and out_full is not None:
        full = np.asarray(out_full, dtype=np.float64).reshape(-1)
        need = max(M, N) ** 2
        if full.size >= need:
            ii, jj = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
            idx = ii * M + jj
            vals = full[idx]
            okm = np.isfinite(vals) & (np.abs(vals - r) <= tol.atol + tol.rtol * np.abs(r))
            if float(okm.mean()) >= SWAPPED_MN_MATCH_FRACTION:
                return "swapped_MN_dims"
    return ""


def classify_generic(
    out: Any,
    ref64: Any,
    tol: Tolerance,
    *,
    pattern_fn: Any | None = None,
    oob_write: bool = False,
    return_type: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """과제에 무관한 출력 판정(설계 §10 v2.1 규칙 1–6). ``pattern_fn(out_flat, ref)`` 는 과제별 arg_order 패턴 이름
    (없으면 ``""``)을 돌려주는 함수로, 규칙 3 에서만 불린다. ``ref64`` 의 형태가 출력의 논리 형태다."""
    o32 = np.asarray(out).reshape(-1)
    r = np.asarray(ref64, dtype=np.float64)
    n_total = int(r.size)
    if o32.size != n_total:
        raise ValueError(f"classify: out 크기 {o32.size} != 참조 크기 {n_total}")

    sentinel = np.asarray(tol.sentinel, dtype=o32.dtype) if o32.dtype.kind == "f" else tol.sentinel
    n_unwritten = int(np.count_nonzero(o32 == sentinel))

    cmp = compare(o32, r, tol.rtol, tol.atol)
    metrics: dict[str, Any] = {
        "max_abs_err": cmp["max_abs_err"],
        "max_tol_ratio": cmp["max_tol_ratio"],
        "n_fail": cmp["n_fail"],
        "nan_inf": cmp["nan_inf"],
        "n_unwritten": n_unwritten,
        "n_total": n_total,
        "max_abs_ref": cmp["max_abs_ref"],
        "oob_write": bool(oob_write),
    }

    # 1. NaN/Inf (센티널 -2048.0 은 유한값이라 여기서 세어지지 않는다)
    if cmp["nan_inf"]:
        return "numeric_instability", "nan_inf", metrics

    # 2. 허용오차 내
    if cmp["ok"]:
        return "pass", "", metrics

    # 3. arg_order 패턴 — 센티널·OOB 가 남아 있어도 패턴이 전체 버퍼를 설명하면 구조적 오류가 더 정확한 분류다
    pattern = pattern_fn(o32, r) if pattern_fn is not None else ""
    if pattern:
        return "arg_order", pattern, metrics

    # 4. 센티널·계약 검사: 전부 미기록 → 누적 → 부분 기록 → OOB
    if n_total > 0 and n_unwritten == n_total:
        pattern = "returned_array_not_written" if return_type == "ndarray" else "output_not_written"
        return "wrong_output", pattern, metrics
    # 누적(C += A@B)을 부분 기록보다 먼저 본다: float32 에서 sentinel + ref 는 |ref| < 2⁻¹³ 이면 정확히 센티널로
    # 되돌아가 n_unwritten 에 세어진다(실제 small/large 형태에서 실측). 진짜 부분 기록은 모든 원소가 sentinel + ref 와
    # 일치할 수 없으므로 이 검사가 부분 기록을 가리지 않는다.
    accumulated = compare(np.asarray(o32, dtype=np.float64) - float(tol.sentinel), r, tol.rtol, tol.atol)
    if accumulated["ok"]:
        return "wrong_output", "accumulated_into_c", metrics
    if n_unwritten > 0:
        return "wrong_output", f"partial_write:{n_unwritten}/{n_total}", metrics
    if oob_write:
        return "wrong_output", "out_of_bounds_write", metrics

    # 5. 수치 패턴(허용오차 배율)
    if cmp["max_tol_ratio"] is not None and cmp["max_tol_ratio"] <= tol.numeric_pattern_tol_ratio:
        return "numeric_instability", "imprecise", metrics

    # 6. 그 밖
    return "wrong_output", "mismatch", metrics


def classify_output(
    out: Any,
    ref64: Any,
    A0: Any,
    B0: Any,
    M: int,
    N: int,
    K: int,
    tol: Tolerance,
    *,
    oob_write: bool = False,
    return_type: str | None = None,
    out_full: Any | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """GEMM 출력 하나를 판정한다 → ``(status, pattern_or_detail, metrics)``. 규칙은 ``classify_generic``,
    arg_order 패턴은 ``match_arg_order``(GEMM 전용).

    ``out`` 은 센티널(``tol.sentinel``)로 미리 채운 M·N 출력 버퍼(1-D 또는 2-D), ``ref64`` 는 float64 참조(M, N),
    ``A0``/``B0`` 는 원래 입력(행우선 M·K, K·N), ``return_type`` 은 python 백엔드의 반환값 타입명,
    ``out_full`` 은 가드 포함 전체 C 버퍼(``swapped_MN_dims`` 용; 없으면 그 패턴은 생략).
    ``oob_write`` 는 워커가 가드 영역에서 본 플래그: 출력이 허용오차 내(규칙 2)·arg_order(규칙 3)면 판정을 바꾸지 않고
    metrics 에만 남는다(샘플 수준에서는 ``aggregate_sample`` 과 run.py 의 pass 정의가 플래그를 본다).
    metrics: ``max_abs_err, max_tol_ratio, n_fail, nan_inf, n_unwritten, n_total, max_abs_ref, oob_write``.
    """
    n_total = int(M) * int(N)
    if np.asarray(out).size != n_total:
        raise ValueError(f"classify_output: out 크기 {np.asarray(out).size} != M·N {n_total}")
    r = np.asarray(ref64, dtype=np.float64).reshape(M, N)
    return classify_generic(
        out, r, tol,
        pattern_fn=lambda o, rr: match_arg_order(o, rr, A0, B0, M, N, K, tol, out_full),
        oob_write=oob_write, return_type=return_type,
    )


def _transposed_2d(o: np.ndarray, r: np.ndarray, tol: Tolerance) -> str:
    """2-D 출력의 전치 저장(행·열 길이를 바꿔 씀) 패턴. 과제 공통."""
    if r.ndim == 2 and _close(np.asarray(o, dtype=np.float64).reshape(r.shape[::-1]).T, r, tol):
        return "transposed_output"
    return ""


def classify_task(
    task: str,
    out: Any,
    ref64: Any,
    inputs: dict[str, Any],
    dims: dict[str, int],
    tol: Tolerance,
    *,
    oob_write: bool = False,
    return_type: str | None = None,
    out_full: Any | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """과제 이름으로 판정기를 고른다. GEMM 은 전용 arg_order 패턴 4종, 그 밖의 과제는 2-D 출력의 전치 패턴만 본다
    (과제별 패턴을 늘리면 docs/error_taxonomy.md 를 갱신한다)."""
    if task == "gemm":
        return classify_output(out, ref64, inputs["A"], inputs["B"], dims["M"], dims["N"], dims["K"], tol,
                               oob_write=oob_write, return_type=return_type, out_full=out_full)
    return classify_generic(out, ref64, tol, pattern_fn=lambda o, r: _transposed_2d(o, r, tol),
                            oob_write=oob_write, return_type=return_type)


# ----------------------------------------------------------------------------- 샘플 수준 합산


def aggregate_sample(
    tests: list[dict[str, Any]],
    *,
    inputs_mutated: bool = False,
    oob_write: bool = False,
) -> tuple[str, str]:
    """check 테스트(+ bench final_check)들을 샘플 하나의 ``(classification, detail)`` 로 합산한다.

    ``tests`` 원소 키: ``name``, ``classification``, ``pattern``(또는 ``error_detail``), ``nondeterministic``.
    순서: 1) 예외/크래시 → runtime_crash 2) 어떤 테스트든 nondeterministic → race_condition
    3) 실패 전부가 arg_order 이고 같은 패턴 → arg_order 4) wrong_output(입력 변조·OOB 포함) → wrong_output
    5) 나머지 — 실패가 전부 numeric 이면 numeric_instability; 패턴이 섞인 경우(설계가 상정하지 않은 조합)는
    보수적으로 wrong_output/``mixed:<패턴들>`` 6) 전부 pass → pass.
    """

    def _pattern(t: dict[str, Any]) -> str:
        return str(t.get("pattern") or t.get("error_detail") or "")

    def _name(t: dict[str, Any]) -> str:
        return str(t.get("name") or "?")

    crashed = [t for t in tests if t.get("classification") == "runtime_crash"]
    if crashed:
        t = crashed[0]
        return "runtime_crash", f"{_name(t)}: {_pattern(t)}".rstrip(": ")

    nondet = [t for t in tests if t.get("nondeterministic")]
    if nondet:
        return "race_condition", "nondeterministic:" + ",".join(_name(t) for t in nondet)

    failed = [t for t in tests if t.get("classification") not in ("pass", None, "")]
    if failed and all(t.get("classification") == "arg_order" for t in failed):
        patterns = {_pattern(t) for t in failed}
        if len(patterns) == 1:
            return "arg_order", patterns.pop()

    if inputs_mutated:
        return "wrong_output", "inputs_mutated"
    if oob_write:
        return "wrong_output", "out_of_bounds_write"
    wrong = [t for t in failed if t.get("classification") == "wrong_output"]
    if wrong:
        return "wrong_output", _pattern(wrong[0]) or "mismatch"

    if failed:
        if all(t.get("classification") == "numeric_instability" for t in failed):
            return "numeric_instability", _pattern(failed[0]) or "imprecise"
        mixed = sorted({f"{t.get('classification')}/{_pattern(t)}" for t in failed})
        return "wrong_output", "mixed:" + ",".join(mixed)

    return "pass", ""


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """npz(A, B, C_ref, M, N, K)와 출력 배열(.npy, M·N float32)을 받아 판정을 JSON으로 출력한다(진단용)."""
    parser = argparse.ArgumentParser(prog="python -m harness.classify", description="출력 판정 규칙 (설계 §10)")
    parser.add_argument("--tolerance", type=Path, required=True, help="tolerance.yaml")
    parser.add_argument("--npz", type=Path, required=True, help="gen_tests.py 가 만든 테스트 npz")
    parser.add_argument("--out-npy", type=Path, required=True, help="커널 출력(.npy, float32, M·N)")
    parser.add_argument("--return-type", default=None, help="python 백엔드 반환값 타입명(예: ndarray)")
    ns = parser.parse_args(argv)
    tol = load_tolerance(ns.tolerance)
    with np.load(ns.npz) as data:
        A, B, C_ref = data["A"], data["B"], data["C_ref"]
        M, N, K = int(data["M"]), int(data["N"]), int(data["K"])
    out = np.load(ns.out_npy)
    status, pattern, metrics = classify_output(out, C_ref, A, B, M, N, K, tol, return_type=ns.return_type)
    print(json.dumps({"classification": status, "pattern": pattern, "metrics": metrics, "tolerance": asdict(tol)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
