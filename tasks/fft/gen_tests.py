"""fft 테스트 입력 생성기. 사용: PYTHON_EXE tasks/fft/gen_tests.py [--seed S] [--out DIR]

입력 ``x`` 는 복소수 n 개를 실수부·허수부 교차로 담은 float32 [2n] 이다(``x[2j]`` 실수부, ``x[2j+1]`` 허수부). 성분은
uniform[-1, 1) 에 1/sqrt(n) 을 곱한 값이다 → 출력 X[k] 의 성분(실수부·허수부) 분산이 n·(1/3)/n = 1/3 로 형태와 무관하다
(성분 표준편차 약 0.58, |X[k]| 의 RMS 약 0.82). 과제당 상수인 atol(1e-3)이 모든 형태에서 같은 뜻을 갖게 하려는 선택이다
(docs/decisions.md 2026-11-20).

참조 ``out_ref`` 는 float64 [2n]: float32 입력을 complex128 로 올려 ``np.fft.fft`` 로 변환하고 같은 방식으로 교차 저장한다.
참조는 정의식 X[k] = Σ_j z[j]·exp(−2πi·jk/n) 의 직접 합(지수 부호 음수, 정규화 없음, 자연 순서)과 대조한다 — n ≤ 1024 는
모든 k, 그보다 크면 고정 k 4개 + 무작위 k 16개 — 그리고 Parseval 등식 Σ|X|² = n·Σ|z|² 으로 전체 크기를 확인한다.
spec.md 가 사실로 적는 출력 크기(|X[k]| 의 RMS 약 0.82)는 n ≥ 1024 인 형태에서 ``verify`` 가 같이 확인한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import log, main_for  # noqa: E402

TASK = "fft"
SENTINEL = -2048.0        # tolerance.yaml 의 sentinel 과 같아야 한다
MAX_ABS_COMPONENT = 16.0  # 출력 성분 절댓값의 상한(센티널과 멀리 떨어져 있음을 보장; 실제 최댓값은 4 안팎)
FULL_CHECK_MAX_N = 1024   # 이 크기까지는 모든 k 를 직접 합과 대조한다
N_RANDOM_K = 16
DIRECT_SUM_TOL = 1e-9     # float64 직접 합과 np.fft.fft 의 차(성분 크기 O(1) 기준; 실제 차는 1e-12 안팎)
# spec.md 가 적은 출력 크기("|X[k]| has a root mean square of about 0.82" = sqrt(2/3)): n 이 이 값 이상인 형태에서 확인한다.
# n = 8 은 표본이 16개뿐이라 RMS 가 0.72–0.90 사이에서 흔들린다(분포에 대한 문장이지 표본에 대한 문장이 아니다).
RMS_CHECK_MIN_N = 1024
RMS_EXPECTED = (2.0 / 3.0) ** 0.5
RMS_TOL = 0.03            # n = 1024 에서 RMS 의 표준편차가 약 0.008 이다

# n 은 2의 거듭제곱(설계 v3 §4). tiny 는 지수가 홀수(2^3), 나머지는 짝수 지수다.
SHAPES = [
    ("tiny", {"n": 8}),
    ("rect", {"n": 1024}),
    ("small", {"n": 1 << 16}),
    ("large", {"n": 1 << 22}),
]


def direct_dft(z: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """정의식의 직접 합 X[k] = Σ_j z[j]·exp(−2πi·jk/n) (float64). 위상은 (j·k mod n) 을 정수로 구해 큰 n 에서도 정확하다."""
    n = z.size
    j = np.arange(n, dtype=np.int64)
    out = np.empty(len(ks), dtype=np.complex128)
    for i, k in enumerate(ks):
        phase = (-2.0 * np.pi / n) * ((j * int(k)) % n)
        out[i] = np.dot(z, np.cos(phase) + 1j * np.sin(phase))
    return out


def verify(z: np.ndarray, X: np.ndarray, rng: np.random.Generator) -> dict:
    """참조 ``X`` 를 정의식과 대조한다(부호·정규화·순서). 어긋나면 RuntimeError. 명세·decisions 에 적는 크기 통계를 돌려준다."""
    n = z.size
    if n <= FULL_CHECK_MAX_N:
        ks = np.arange(n, dtype=np.int64)
    else:
        ks = np.unique(np.concatenate([[0, 1, n // 2, n - 1], rng.integers(0, n, size=N_RANDOM_K)])).astype(np.int64)
    err = float(np.max(np.abs(direct_dft(z, ks) - X[ks])))
    if not err <= DIRECT_SUM_TOL:
        raise RuntimeError(f"n={n}: 참조가 정의식의 직접 합과 다르다(최대 차 {err:.3e}, k {ks.size}개)")
    energy_in, energy_out = float(np.vdot(z, z).real) * n, float(np.vdot(X, X).real)
    if not abs(energy_out - energy_in) <= 1e-10 * energy_in:
        raise RuntimeError(f"n={n}: Parseval 등식이 성립하지 않는다({energy_out!r} != {energy_in!r})")
    comp = X.view(np.float64)
    max_abs = float(np.max(np.abs(comp)))
    if not max_abs < MAX_ABS_COMPONENT:
        raise RuntimeError(f"n={n}: 출력 성분의 절댓값 {max_abs} 가 상한 {MAX_ABS_COMPONENT} 이상이다")
    rms = float(np.sqrt(energy_out / n))
    if n >= RMS_CHECK_MIN_N and not abs(rms - RMS_EXPECTED) <= RMS_TOL:
        raise RuntimeError(f"n={n}: |X[k]| 의 RMS {rms:.4f} 가 spec.md 의 값 {RMS_EXPECTED:.4f} 에서 {RMS_TOL} 넘게 벗어났다")
    return {"direct_sum_err": err, "n_k_checked": int(ks.size), "rms_abs_X": rms,
            "std_component": float(comp.std()), "max_abs_component": max_abs}


def make_case(shape, rng):
    n = shape["n"]
    if n < 2 or n & (n - 1):
        raise ValueError(f"n 은 2 이상의 2의 거듭제곱이어야 한다: {n}")
    x = ((rng.random(2 * n) * 2.0 - 1.0) / np.sqrt(n)).astype(np.float32)
    z = x.astype(np.float64).view(np.complex128)  # 참조는 float32 로 내린 입력에서 계산한다(커널이 받는 값과 같다)
    X = np.fft.fft(z)
    if X.dtype != np.complex128 or X.shape != (n,):
        raise RuntimeError(f"np.fft.fft 의 결과가 complex128 ({n},) 이 아니다: {X.dtype} {X.shape}")
    facts = verify(z, X, rng)
    log(TASK, f"n={n} " + " ".join(f"{k}={v:.6g}" for k, v in facts.items()))
    ref = np.ascontiguousarray(X).view(np.float64)  # (2n,): ref[2k] 실수부, ref[2k+1] 허수부
    return {"n": n}, {"x": x}, "out", ref


if __name__ == "__main__":
    sys.exit(main_for(TASK, SHAPES, make_case, "numpy float64 np.fft.fft of the float32 input, verified against the direct DFT sum",
                      HERE / "tests", sentinel=SENTINEL))
