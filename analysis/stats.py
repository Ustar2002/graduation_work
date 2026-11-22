"""부트스트랩 신뢰구간 (설계 v3 §13; ``analysis.aggregate --ci/--conditions`` 와 ``analysis.plots`` 가 쓴다).

방법은 **백분위 부트스트랩** 하나다: 관측값 n개에서 복원추출로 n개를 뽑는 일을 ``n_boot`` 번 되풀이해
통계량의 분포를 만들고, 그 분포의 ``alpha/2`` · ``1 − alpha/2`` 분위수를 구간의 양 끝으로 삼는다.
난수는 ``np.random.default_rng(seed)`` 하나에서만 나오므로 입력·``n_boot``·``seed`` 가 같으면 결과도 같다
(같은 NumPy 버전 기준). 재추출은 ``(n_boot, n)`` 인덱스 행렬 한 번으로 벡터화한다.

**과제당 샘플 n = 5 에서 이 구간이 뜻하는 것** (예비 연구의 정직한 답은 "넓다"이다):

- 통과율(통과 k개 / 5개)의 95 % 구간은 k = 0 → [0, 0], 1 → [0, 0.6], 2 → [0, 0.8], 3 → [0.2, 1],
  4 → [0.4, 1], 5 → [1, 1] 이다. 즉 3/5 와 5/5 는 구간이 겹친다.
- 서로 다른 값 5개의 중앙값 구간은 [최솟값, 최댓값] 이다(재추출 중앙값이 최솟값이 될 확률이 5.8 % 로 2.5 % 보다 크다).
- 0/5 · 5/5 의 폭 0 은 "5개 안에서 변동을 보지 못했다"는 뜻이지 확실하다는 뜻이 아니다. 5/5 통과여도
  실제 통과율은 48 % 까지 낮을 수 있다(Clopper–Pearson 95 % 하한 0.025^(1/5)).
- 작은 n 에서 백분위 부트스트랩은 실제보다 좁게 나오는 쪽으로 틀린다(포함 확률 부족). 위 구간은 낙관적인 값이다.

따라서 n = 5 의 구간은 차이의 **방향**을 가늠하는 데까지만 쓰고, 크기를 주장하는 근거로 쓰지 않는다.

모든 함수는 ``(점추정, 구간 하한, 구간 상한)`` 을 파이썬 float 튜플로 돌려준다. 값이 하나도 없으면 셋 다 NaN 이다.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from numpy.typing import ArrayLike

Interval = tuple[float, float, float]

_NAN_INTERVAL: Interval = (math.nan, math.nan, math.nan)


def _finite(values: ArrayLike) -> np.ndarray:
    """1차원 float 배열로 바꾸고 NaN 을 뺀다(시간이 없는 bench timeout 행처럼 값이 없는 관측)."""
    x = np.asarray(values, dtype=float).ravel()
    return x[~np.isnan(x)]


def _resample(x: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """복원추출 표본 ``n_boot`` 개를 행으로 쌓은 ``(n_boot, n)`` 배열."""
    return x[rng.integers(0, x.size, size=(n_boot, x.size))]


def _percentile_interval(boot: np.ndarray, alpha: float) -> tuple[float, float]:
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def bootstrap_ci(
    values: ArrayLike,
    stat: Callable[..., Any] = np.median,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """``stat(values)`` 와 그 ``1 − alpha`` 백분위 부트스트랩 구간.

    ``stat`` 은 ``axis`` 인자를 받는 NumPy 축약 함수(``np.median``, ``np.mean`` …)여야 한다. NaN 은 빼고 센다.
    n = 0 이면 ``(NaN, NaN, NaN)``, n = 1 이면 재추출이 모두 같은 값이라 ``(x, x, x)`` 다(구간이 아니라 "모른다"로 읽는다).
    n = 5 의 중앙값 구간은 서로 다른 값이면 [최솟값, 최댓값] 이다(모듈 설명 참조).
    """
    x = _finite(values)
    if x.size == 0:
        return _NAN_INTERVAL
    rng = np.random.default_rng(seed)
    low, high = _percentile_interval(stat(_resample(x, n_boot, rng), axis=1), alpha)
    return float(stat(x)), low, high


def bootstrap_pass_rate(passes: ArrayLike, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """0/1(또는 불리언) 통과 여부의 통과율과 그 부트스트랩 구간.

    n = 5 에서 구간은 통과 수 0..5 에 대해 [0, 0], [0, 0.6], [0, 0.8], [0.2, 1], [0.4, 1], [1, 1] 이다.
    전부 통과·전부 실패의 폭 0 은 변동을 관측하지 못했다는 뜻일 뿐이다(모듈 설명 참조).
    """
    return bootstrap_ci(passes, stat=np.mean, n_boot=n_boot, alpha=alpha, seed=seed)


def paired_bootstrap_diff(a: ArrayLike, b: ArrayLike, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """대응 표본의 평균 차 ``mean(a) − mean(b)`` 와 그 구간. ``a[i]`` 와 ``b[i]`` 는 같은 샘플의 두 조건이다.

    짝을 함께 재추출한다(= 짝별 차이 ``a − b`` 의 평균을 부트스트랩). 한쪽이라도 NaN 인 짝은 뺀다.
    두 조건이 같은 샘플에서 갈라졌을 때만 쓴다: 샘플 간 변동이 차이에서 빠져 독립 표본보다 구간이 좁다.
    그래도 n = 5 짝에서 한 짝만 달라진 경우(차 0.2)의 구간은 [0, 0.6] 이다.
    """
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"대응 표본의 길이가 다르다: {x.size} vs {y.size}")
    return bootstrap_ci(x - y, stat=np.mean, n_boot=n_boot, alpha=alpha, seed=seed)


def unpaired_bootstrap_diff(a: ArrayLike, b: ArrayLike, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Interval:
    """독립 표본의 평균 차 ``mean(a) − mean(b)`` 와 그 구간. 두 표본을 따로(각자의 크기로) 재추출한다.

    NaN 은 빼고 센다. 어느 한쪽이 비면 ``(NaN, NaN, NaN)``.
    n = 5 대 5 에서 통과 3/5 대 5/5 의 차(−0.4)의 구간은 [−0.8, 0] 이다: 0 을 포함하므로 차이가 있다고 말할 수 없다.
    """
    x, y = _finite(a), _finite(b)
    if x.size == 0 or y.size == 0:
        return _NAN_INTERVAL
    rng = np.random.default_rng(seed)
    boot = _resample(x, n_boot, rng).mean(axis=1) - _resample(y, n_boot, rng).mean(axis=1)
    low, high = _percentile_interval(boot, alpha)
    return float(x.mean() - y.mean()), low, high
