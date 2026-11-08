"""fft 테스트 입력 생성기. 사용: PYTHON_EXE tasks/fft/gen_tests.py [--seed S] [--out DIR]

``x`` 는 float32 (n, 2) 교차 저장 복소수(열 0 실수부, 열 1 허수부), 두 성분 모두 uniform[-1, 1).
생성은 ``2·r − 1``(r 은 float32 uniform[0, 1) = k·2⁻²⁴) 로, 결과가 2⁻²³ 의 정수배라 float32 에서 정확하고 상한 1 은 닿지 않는다.
참조 ``out_ref`` 는 float64 복소수 입력의 ``np.fft.fft``(정방향, 스케일 없음: X_k = Σ_j x_j·e^{−2πi jk/n})를
실수·허수로 나눠 float64 (n, 2) 로 저장한다. n 은 2 의 거듭제곱(tiny 2³, rect 2¹⁰, small 2¹⁶, large 2²²).
형태마다 참조 성분의 RMS(≈ sqrt(n/3))와 최대 절대값을 로그로 남긴다(tolerance.yaml 의 atol_rms_rel 근거).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import log, main_for  # noqa: E402

SHAPES = [
    ("tiny", {"n": 8}),
    ("rect", {"n": 1024}),
    ("small", {"n": 1 << 16}),
    ("large", {"n": 1 << 22}),
]


def reference(x: np.ndarray) -> np.ndarray:
    """float64 참조. ``x`` (n, 2) float32 교차 저장 → ``out`` (n, 2) float64 (열 0 실수부, 열 1 허수부)."""
    x64 = np.asarray(x, dtype=np.float64)
    z = x64[:, 0] + 1j * x64[:, 1]  # complex128
    X = np.fft.fft(z)  # 정방향, 스케일 없음
    return np.ascontiguousarray(np.stack([X.real, X.imag], axis=1))


def make_case(shape: dict[str, int], rng: np.random.Generator):
    n = shape["n"]
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"fft: n 은 2 의 거듭제곱이어야 한다({n})")
    r = rng.random((n, 2), dtype=np.float32)
    x = np.ascontiguousarray(r * np.float32(2.0) - np.float32(1.0))  # [-1, 1), float32 에서 정확
    ref = reference(x)
    rms = float(np.sqrt(np.mean(ref * ref)))
    log("fft", f"n={n}: out_ref component RMS={rms:.6g} (sqrt(n/3)={np.sqrt(n / 3):.6g}) max|out_ref|={float(np.abs(ref).max()):.6g}")
    return {"n": n}, {"x": x}, "out", ref


if __name__ == "__main__":
    sys.exit(main_for("fft", SHAPES, make_case, "numpy float64 complex128 np.fft.fft, forward, unnormalized", HERE / "tests"))
