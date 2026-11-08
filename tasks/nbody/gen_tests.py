"""nbody 테스트 입력 생성기. 사용: PYTHON_EXE tasks/nbody/gen_tests.py [--seed S] [--out DIR]

``pos`` 는 float32 uniform[0, 1)^3 의 (n, 3) AoS(행 i 가 물체 i 의 x, y, z), ``mass`` 는 float32 uniform[0.5, 1.5) 의 (n,).
참조 ``acc_ref`` 는 float64 로 **모든 쌍을 직접 합산**한 결과(근사 없음): i 를 ``CHUNK`` 행씩 묶어 (chunk, n, 3) 브로드캐스트로
계산한다(n = 16384 에서 수십 초 허용). EPS2 = 1e-4, G = 1. j = i 항은 p_j − p_i = 0 이라 정확히 0 이다(분모는 EPS2 > 0).
형태마다 참조 acc 성분의 RMS 와 최대 절대값을 로그로 남긴다(large 의 RMS 가 tolerance.yaml 의 atol 근거).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import log, main_for  # noqa: E402

EPS2 = 1e-4  # 연화(softening) 상수: spec.md 와 같아야 한다
CHUNK = 256  # 참조 계산에서 한 번에 처리하는 i 행 수((CHUNK, n, 3) float64 임시 배열 = n=16384 에서 약 100 MB)

SHAPES = [
    ("tiny", {"n": 5}),
    ("rect", {"n": 301}),
    ("small", {"n": 2048}),
    ("large", {"n": 16384}),
]


def reference(pos: np.ndarray, mass: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """float64 직접 합산 참조. ``pos`` (n, 3), ``mass`` (n,) → ``acc`` (n, 3) float64.

    acc_i = Σ_j mass_j · (p_j − p_i) / (|p_j − p_i|² + EPS2)^(3/2). j = i 항은 정확히 0 이므로 따로 빼지 않는다.
    """
    p = np.asarray(pos, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    n = p.shape[0]
    acc = np.empty((n, 3), dtype=np.float64)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        d = p[None, :, :] - p[i0:i1, None, :]  # (chunk, n, 3): p_j − p_i
        r2 = np.einsum("ijk,ijk->ij", d, d) + EPS2  # (chunk, n): |p_j − p_i|² + EPS2
        w = m[None, :] / (r2 * np.sqrt(r2))  # mass_j · (r² + EPS2)^(−3/2)
        acc[i0:i1] = np.einsum("ij,ijk->ik", w, d)
    return acc


def make_case(shape: dict[str, int], rng: np.random.Generator):
    n = shape["n"]
    pos = np.ascontiguousarray(rng.random((n, 3), dtype=np.float32))
    # 0.5 + k·2^-23 (k < 2^23) 격자: float32 로 정확히 표현되고 최댓값이 1.5 − 2^-23 이라 [0.5, 1.5) 가 구성상 보장된다
    mass = (np.float32(0.5) + rng.integers(0, 2**23, size=n).astype(np.float32) * np.float32(2.0**-23)).astype(np.float32)
    ref = reference(pos, mass)
    rms = float(np.sqrt(np.mean(ref * ref)))
    log("nbody", f"n={n}: acc_ref component RMS={rms:.6g} max|acc_ref|={float(np.abs(ref).max()):.6g}")
    return {"n": n}, {"pos": pos, "mass": mass}, "acc", ref


if __name__ == "__main__":
    sys.exit(main_for("nbody", SHAPES, make_case, "numpy float64 direct summation, 256-row chunks", HERE / "tests"))
