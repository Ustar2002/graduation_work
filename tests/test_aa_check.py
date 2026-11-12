"""aa_check 의 순수 함수: 호출 쌍의 첫/둘째 분리와 변형 요약(2026-11-12)."""

from __future__ import annotations

import unittest

from harness.aa_check import split_by_position, summarize_variant


def _doc(k, b, warm=1):
    return {"timed_ms": k, "warmup_ms": [1.0] * warm, "baseline_interleaved": {"classification": "pass", "timed_ms": b}}


class SplitTest(unittest.TestCase):
    def test_pair_order_follows_call_index_parity(self) -> None:
        # warmup 1 → timed j=0 은 call_idx 1(홀수: 커널 먼저), j=1 은 call_idx 2(짝수: 기준선 먼저)
        first, second = split_by_position(_doc([10.0, 11.0], [20.0, 21.0], warm=1))
        self.assertEqual(first, [10.0, 21.0])
        self.assertEqual(second, [20.0, 11.0])
        # warmup 0 → j=0 은 call_idx 0(기준선 먼저)
        first, second = split_by_position(_doc([10.0, 11.0], [20.0, 21.0], warm=0))
        self.assertEqual(first, [20.0, 11.0])
        self.assertEqual(second, [10.0, 21.0])

    def test_summary_of_identical_code_is_one(self) -> None:
        docs = [_doc([10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0]), _doc([12.0, 12.0], [12.0, 12.0])]
        s = summarize_variant(docs)
        self.assertEqual((s["rounds"], s["n_pairs"]), (2, 6))
        self.assertAlmostEqual(s["ratio_mean"], 1.0)
        self.assertAlmostEqual(s["ratio_median"], 1.0)
        self.assertEqual(s["ratio_mean_rounds"], [1.0, 1.0])
        self.assertAlmostEqual(s["first_vs_second"], 1.0)

    def test_first_call_penalty_is_visible(self) -> None:
        # 쌍의 첫 호출만 10 % 느린 A/A: 커널·기준선 평균은 같지만 첫/둘째 비가 1.1
        k, b = [], []
        warm = 1
        for j in range(10):
            baseline_first = (warm + j) % 2 == 0
            k.append(10.0 if baseline_first else 11.0)
            b.append(11.0 if baseline_first else 10.0)
        s = summarize_variant([_doc(k, b, warm=warm)])
        self.assertAlmostEqual(s["ratio_mean"], 1.0)
        self.assertAlmostEqual(s["first_vs_second"], 1.1)


if __name__ == "__main__":
    unittest.main()
