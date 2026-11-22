"""nbody 테스트 입력 생성기. 사용: PYTHON_EXE tasks/nbody/gen_tests.py [--seed S] [--out DIR]

입력: ``pos`` 는 float32 uniform[0, 1)³ (n, 3) — 행 i 가 입자 i 의 (x, y, z). ``mass`` 는 float32 로 내린 uniform[0.5, 1.5)/n (n) —
총질량이 약 1 이라 가속도가 형태와 무관하게 O(1) 이다(과제당 상수인 atol 이 성립한다).

참조 ``acc_ref`` 는 float64 로 정의 그대로 구한다:
``acc[i] = Σ_j mass[j]·(pos[j] − pos[i]) / (|pos[j] − pos[i]|² + ε²)^{3/2}``, ε = 0.01 (j = i 항은 0).
n·n·3 개의 double 을 한꺼번에 잡지 않도록 행 블록 단위로 계산한다(블록당 원소 약 2²⁰ 개).

생성기가 확인하는 것(어긋나면 실패): 참조가 유한하고 O(1) 이다(|acc| < ``ACC_BOUND``), 행 몇 개가 정의를 글자 그대로 옮긴
스칼라 계산(``math.fsum``)과 일치한다, 작용·반작용(Σ_i mass[i]·acc[i] = 0)이 성립한다.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

EPS = 0.01              # 연화 길이. spec.md 의 수치와 같아야 한다
SENTINEL = -2048.0      # tolerance.yaml 의 sentinel 과 같은 값
ACC_BOUND = 100.0       # 참조의 크기 상한(가속도가 O(1) 이라는 전제를 코드로 확인한다; 실측 최댓값은 약 8)
BLOCK_ELEMS = 1 << 20   # 참조 계산의 블록당 원소 수(float64 임시 배열 하나가 8 MiB)

SHAPES = [
    ("tiny", {"n": 5}),
    ("rect", {"n": 317}),
    ("small", {"n": 1024}),
    ("large", {"n": 8192}),
]


def reference(pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """float64 참조. 좌표별 연속 배열(SoA)로 바꿔 행 블록마다 (블록 행 수, n) 크기의 임시 배열만 쓴다."""
    n = pos.shape[0]
    p = pos.astype(np.float64)
    px, py, pz = (np.ascontiguousarray(p[:, k]) for k in range(3))
    m = mass.astype(np.float64)
    acc = np.empty((n, 3), dtype=np.float64)
    step = max(1, BLOCK_ELEMS // n)
    for i0 in range(0, n, step):
        i1 = min(n, i0 + step)
        dx = px[None, :] - px[i0:i1, None]
        dy = py[None, :] - py[i0:i1, None]
        dz = pz[None, :] - pz[i0:i1, None]
        r2 = dx * dx + dy * dy + dz * dz + EPS * EPS
        w = m[None, :] / (r2 * np.sqrt(r2))
        acc[i0:i1, 0] = (w * dx).sum(axis=1)  # 마지막 축의 합 = pairwise 누적
        acc[i0:i1, 1] = (w * dy).sum(axis=1)
        acc[i0:i1, 2] = (w * dz).sum(axis=1)
    return acc


def reference_row(pos: np.ndarray, mass: np.ndarray, i: int) -> list[float]:
    """정의를 글자 그대로 옮긴 행 하나의 계산(파이썬 float, ``math.fsum``). ``reference`` 의 벡터화 실수를 잡는 대조용."""
    n = pos.shape[0]
    xi, yi, zi = (float(v) for v in pos[i])
    terms: tuple[list[float], list[float], list[float]] = ([], [], [])
    for j in range(n):
        d = (float(pos[j, 0]) - xi, float(pos[j, 1]) - yi, float(pos[j, 2]) - zi)
        s = float(mass[j]) / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + EPS * EPS) ** 1.5
        for k in range(3):
            terms[k].append(s * d[k])
    return [math.fsum(t) for t in terms]


def make_case(shape, rng):
    n = shape["n"]
    pos = np.ascontiguousarray(rng.random((n, 3), dtype=np.float32))
    mass = ((0.5 + rng.random(n)) / n).astype(np.float32)
    ref = reference(pos, mass)
    if not np.all(np.isfinite(ref)) or float(np.abs(ref).max()) >= ACC_BOUND:
        raise RuntimeError(f"참조 가속도가 O(1) 이 아니다(max |acc| = {float(np.abs(ref).max()):.3g}, 상한 {ACC_BOUND:g})")
    for i in sorted({0, n // 2, n - 1}):
        if not np.allclose(ref[i], reference_row(pos, mass, i), rtol=1e-11, atol=1e-12):
            raise RuntimeError(f"참조의 행 {i} 가 정의를 그대로 옮긴 계산과 다르다")
    # 작용·반작용: 쌍 (i, j) 의 기여가 서로 반대라 질량 가중 합이 0 이다(절댓값 합에 대한 상대 크기로 본다)
    m64 = mass.astype(np.float64)[:, None]
    if float(np.abs((m64 * ref).sum(axis=0)).max()) > 1e-11 * float(np.abs(m64 * ref).sum()):
        raise RuntimeError("참조가 작용·반작용(Σ mass[i]·acc[i] = 0)을 만족하지 않는다")
    return {"n": n}, {"pos": pos, "mass": mass}, "acc", ref


if __name__ == "__main__":
    sys.exit(main_for("nbody", SHAPES, make_case, "numpy float64, all pairs in row blocks", HERE / "tests",
                      sentinel=SENTINEL))
