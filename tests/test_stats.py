"""stats 테스트: 백분위 부트스트랩의 손계산 기대값(n = 5 구간), n = 0/1, NaN 처리, 대응·독립 표본의 차, 결정성.

기대 구간은 이항분포로 손계산한 값이다. 예: 통과 3/5 의 재추출 통과 수 X ~ Bin(5, 0.6) 에서
P(X = 0) = 0.010 < 0.025 ≤ P(X ≤ 1) = 0.087 이므로 하한 0.2, P(X = 5) = 0.078 > 0.025 이므로 상한 1.0.
경계까지의 여유가 재추출 2000회의 표본오차보다 훨씬 커서(4.5σ 이상) seed 와 무관하게 같은 값이 나온다.
"""

from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from analysis.stats import bootstrap_ci, bootstrap_pass_rate, paired_bootstrap_diff, unpaired_bootstrap_diff


def _assert_interval(case: unittest.TestCase, got: tuple[float, float, float], want: tuple[float, float, float]) -> None:
    case.assertEqual(len(got), 3)
    for g, w in zip(got, want):
        case.assertIsInstance(g, float)
        case.assertAlmostEqual(g, w, places=12)


def _assert_all_nan(case: unittest.TestCase, got: tuple[float, float, float]) -> None:
    case.assertEqual(len(got), 3)
    case.assertTrue(all(math.isnan(v) for v in got), got)


class DefaultsTest(unittest.TestCase):
    def test_contract_defaults(self) -> None:
        # 설계 v3 §13 의 기본값. analysis.aggregate 의 안내 문구("95 %", "재추출 2000회, seed 0")가 이 값에 기대고 있다.
        for function in (bootstrap_ci, bootstrap_pass_rate, paired_bootstrap_diff, unpaired_bootstrap_diff):
            with self.subTest(function=function.__name__):
                defaults = {name: p.default for name, p in inspect.signature(function).parameters.items()}
                self.assertEqual((defaults["n_boot"], defaults["alpha"], defaults["seed"]), (2000, 0.05, 0))
        self.assertIs(inspect.signature(bootstrap_ci).parameters["stat"].default, np.median)


class BootstrapCiTest(unittest.TestCase):
    def test_no_values_is_all_nan(self) -> None:
        _assert_all_nan(self, bootstrap_ci([]))
        _assert_all_nan(self, bootstrap_ci([math.nan, math.nan]))
        _assert_all_nan(self, bootstrap_ci(np.array([], dtype=float), stat=np.mean))

    def test_single_value_is_degenerate(self) -> None:
        _assert_interval(self, bootstrap_ci([3.5]), (3.5, 3.5, 3.5))
        _assert_interval(self, bootstrap_ci([math.nan, 3.5, math.nan]), (3.5, 3.5, 3.5))   # NaN 은 빼고 n = 1

    def test_constant_values_have_zero_width(self) -> None:
        _assert_interval(self, bootstrap_ci([2.0] * 7), (2.0, 2.0, 2.0))
        _assert_interval(self, bootstrap_ci([2.0] * 7, stat=np.mean), (2.0, 2.0, 2.0))

    def test_median_of_five_distinct_values_spans_the_range(self) -> None:
        # 재추출 중앙값이 최솟값일 확률 = P(Bin(5, 0.2) ≥ 3) = 0.058 > 0.025 → 구간은 [min, max]
        _assert_interval(self, bootstrap_ci([2.0, 2.5, 3.0, 3.5, 4.0]), (3.0, 2.0, 4.0))
        _assert_interval(self, bootstrap_ci([4.0, 2.0, math.nan, 3.0, 3.5, 2.5]), (3.0, 2.0, 4.0))   # 순서·NaN 무관

    def test_median_of_three_values(self) -> None:
        # P(재추출 중앙값 = 최솟값) = 7/27 = 0.26
        _assert_interval(self, bootstrap_ci([1.0, 0.9, 1.1]), (1.0, 0.9, 1.1))

    def test_default_stat_is_median(self) -> None:
        values = [1.0, 2.0, 100.0]
        self.assertEqual(bootstrap_ci(values)[0], 2.0)
        self.assertAlmostEqual(bootstrap_ci(values, stat=np.mean)[0], 103.0 / 3)

    def test_stat_is_applied_to_one_resample_matrix(self) -> None:
        shapes: list[tuple[tuple[int, ...], object]] = []

        def recording_mean(x: np.ndarray, axis: int | None = None) -> np.ndarray:
            shapes.append((x.shape, axis))
            return np.mean(x, axis=axis)

        bootstrap_ci([1.0, 2.0, 3.0, 4.0], stat=recording_mean, n_boot=50)
        self.assertEqual(sorted(shapes, key=str), sorted([((50, 4), 1), ((4,), None)], key=str))

    def test_deterministic_for_a_fixed_seed(self) -> None:
        values = np.linspace(0.0, 1.0, 37) ** 2
        first = bootstrap_ci(values, stat=np.mean)
        self.assertEqual(first, bootstrap_ci(values, stat=np.mean))
        self.assertEqual(first, bootstrap_ci(list(values), stat=np.mean, seed=0))
        self.assertNotEqual(first[1:], bootstrap_ci(values, stat=np.mean, seed=1)[1:])
        self.assertEqual(first[0], bootstrap_ci(values, stat=np.mean, seed=1)[0])   # 점추정은 seed 와 무관

    def test_interval_brackets_the_point_and_narrows_with_alpha(self) -> None:
        values = np.linspace(0.0, 1.0, 37) ** 2
        point, low, high = bootstrap_ci(values, stat=np.mean)
        self.assertLess(low, point)
        self.assertLess(point, high)
        self.assertGreaterEqual(low, values.min())
        self.assertLessEqual(high, values.max())
        _, low50, high50 = bootstrap_ci(values, stat=np.mean, alpha=0.5)
        self.assertGreater(low50, low)
        self.assertLess(high50, high)

    def test_large_sample_matches_the_normal_approximation(self) -> None:
        # 0/1 이 200개씩: 평균 0.5, 표준오차 sqrt(0.25/400) = 0.025 → 95 % 구간 ≈ 0.5 ± 0.049
        values = np.array([0.0, 1.0] * 200)
        point, low, high = bootstrap_ci(values, stat=np.mean)
        self.assertEqual(point, 0.5)
        self.assertAlmostEqual(low, 0.451, delta=0.01)
        self.assertAlmostEqual(high, 0.549, delta=0.01)


class BootstrapPassRateTest(unittest.TestCase):
    def test_five_samples_table(self) -> None:
        expected = {0: (0.0, 0.0, 0.0), 1: (0.2, 0.0, 0.6), 2: (0.4, 0.0, 0.8), 3: (0.6, 0.2, 1.0), 4: (0.8, 0.4, 1.0), 5: (1.0, 1.0, 1.0)}
        for k, want in expected.items():
            with self.subTest(n_pass=k):
                _assert_interval(self, bootstrap_pass_rate([1] * k + [0] * (5 - k)), want)

    def test_booleans_and_order(self) -> None:
        _assert_interval(self, bootstrap_pass_rate([False, True, False, True, True]), (0.6, 0.2, 1.0))
        _assert_interval(self, bootstrap_pass_rate(np.array([True, True, False, False, True])), (0.6, 0.2, 1.0))

    def test_no_samples(self) -> None:
        _assert_all_nan(self, bootstrap_pass_rate([]))

    def test_one_sample(self) -> None:
        _assert_interval(self, bootstrap_pass_rate([True]), (1.0, 1.0, 1.0))


class PairedDiffTest(unittest.TestCase):
    def test_identical_conditions(self) -> None:
        _assert_interval(self, paired_bootstrap_diff([1, 0, 1, 1, 0], [1, 0, 1, 1, 0]), (0.0, 0.0, 0.0))

    def test_constant_shift_has_zero_width(self) -> None:
        a = np.array([0.5, 1.5, 4.0, 2.0])
        _assert_interval(self, paired_bootstrap_diff(a + 0.25, a), (0.25, 0.25, 0.25))
        _assert_interval(self, paired_bootstrap_diff(a, a + 0.25), (-0.25, -0.25, -0.25))   # 부호: a − b

    def test_one_of_five_pairs_differs(self) -> None:
        # 짝별 차 [0, 0, 1, 0, 0]: 재추출에 그 짝이 든 횟수 K ~ Bin(5, 0.2); P(K = 0) = 0.33, P(K ≥ 4) = 0.007 < 0.025 ≤ P(K ≥ 3) = 0.058
        _assert_interval(self, paired_bootstrap_diff([1, 1, 1, 1, 0], [1, 1, 0, 1, 0]), (0.2, 0.0, 0.6))

    def test_pairing_removes_between_sample_variation(self) -> None:
        # 같은 자료를 독립 표본으로 보면 구간이 훨씬 넓다
        a, b = [1, 1, 1, 1, 0], [1, 1, 0, 1, 0]
        _, low_p, high_p = paired_bootstrap_diff(a, b)
        _, low_u, high_u = unpaired_bootstrap_diff(a, b)
        self.assertLess(high_p - low_p, high_u - low_u)

    def test_pairs_with_nan_are_dropped(self) -> None:
        _assert_interval(self, paired_bootstrap_diff([1.0, math.nan, 3.0], [0.0, 5.0, 2.0]), (1.0, 1.0, 1.0))

    def test_no_pairs(self) -> None:
        _assert_all_nan(self, paired_bootstrap_diff([], []))

    def test_length_mismatch_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_diff([1, 0, 1], [1, 0])


class UnpairedDiffTest(unittest.TestCase):
    def test_five_of_five_against_three_of_five(self) -> None:
        # 차 = 1 − X/5, X ~ Bin(5, 0.6): P(X = 5) = 0.078 > 0.025 → 하한 0, P(X = 0) = 0.010 < 0.025 ≤ P(X ≤ 1) = 0.087 → 상한 0.8
        _assert_interval(self, unpaired_bootstrap_diff([1, 1, 1, 1, 1], [1, 1, 1, 0, 0]), (0.4, 0.0, 0.8))
        _assert_interval(self, unpaired_bootstrap_diff([1, 1, 1, 0, 0], [1, 1, 1, 1, 1]), (-0.4, -0.8, 0.0))

    def test_three_of_five_against_two_of_five(self) -> None:
        # 차 = (5 − S)/5, S ~ Bin(10, 0.4): P(S ≥ 8) = 0.012 < 0.025 ≤ P(S ≥ 7) = 0.055 → 하한 −0.4,
        # P(S = 0) = 0.006 < 0.025 ≤ P(S ≤ 1) = 0.046 → 상한 0.8
        _assert_interval(self, unpaired_bootstrap_diff([1, 1, 1, 0, 0], [1, 1, 0, 0, 0]), (0.2, -0.4, 0.8))

    def test_different_sample_sizes(self) -> None:
        point, low, high = unpaired_bootstrap_diff([1.0] * 4, [0.0] * 9)
        _assert_interval(self, (point, low, high), (1.0, 1.0, 1.0))

    def test_nan_is_dropped_per_sample(self) -> None:
        _assert_interval(self, unpaired_bootstrap_diff([2.0, math.nan, 2.0], [0.5, 0.5, math.nan, 0.5]), (1.5, 1.5, 1.5))

    def test_empty_side_is_all_nan(self) -> None:
        _assert_all_nan(self, unpaired_bootstrap_diff([], [1, 0]))
        _assert_all_nan(self, unpaired_bootstrap_diff([1, 0], []))
        _assert_all_nan(self, unpaired_bootstrap_diff([1, 0], [math.nan]))

    def test_deterministic_for_a_fixed_seed(self) -> None:
        a = np.linspace(0.0, 1.0, 23)
        b = np.linspace(0.2, 0.9, 31) ** 2
        first = unpaired_bootstrap_diff(a, b)
        self.assertEqual(first, unpaired_bootstrap_diff(a, b))
        self.assertNotEqual(first[1:], unpaired_bootstrap_diff(a, b, seed=7)[1:])
        self.assertLess(first[1], first[0])
        self.assertLess(first[0], first[2])


if __name__ == "__main__":
    unittest.main()
