"""classify 테스트: 규칙 순서(§10 v2.1: NaN → pass → arg_order → 센티널·계약 → imprecise → mismatch),
더미 기대표(numpy/2 imprecise, c_seq/2 transposed_output), 센티널 규칙, 샘플 합산 우선순위.

v3(§2.2–2.3): 정확 비교(``compare: exact``; 정수 출력) — ``compare_exact``, ``classify_generic``/``classify_task`` 의 정확 비교
규칙(pass / arg_order / output_not_written / accumulated_into_c / partial_write / out_of_bounds_write / mismatch), tolerance.yaml 기본값."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness.classify import (
    COMPARE_MODES,
    Tolerance,
    aggregate_sample,
    classify_generic,
    classify_output,
    classify_task,
    compare,
    compare_exact,
    load_tolerance,
    match_arg_order,
)
from harness.paths import ROOT

TOL = Tolerance(rtol=1e-3, atol=1e-3, numeric_pattern_tol_ratio=20.0, sentinel=-2048.0)
SENT = np.float32(TOL.sentinel)
GEMM_TESTS = ROOT / "tasks" / "gemm" / "tests"


def _guarded_full(M: int, N: int) -> np.ndarray:
    """워커의 C 버퍼 모양(설계 §8): view M·N + 가드 max(M,N)² − M·N + 1024, 전부 센티널."""
    return np.full(max(M, N) ** 2 + 1024, SENT, dtype=np.float32)


def _write_swapped_mn(full: np.ndarray, good: np.ndarray, M: int, N: int) -> None:
    """``C[i*M + j] = C_ref[i, j]`` 로 쓰는 커널(M/N 교환)을 가드 버퍼 위에서 흉내 낸다."""
    ii, jj = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    full[(ii * M + jj).ravel()] = good.ravel()


def _inputs(M: int, N: int, K: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uniform[-1,1) float32 A(M,K), B(K,N) 과 float64 참조."""
    rng = np.random.default_rng(seed)
    A = (rng.random((M, K), dtype=np.float32) * np.float32(2) - np.float32(1))
    B = (rng.random((K, N), dtype=np.float32) * np.float32(2) - np.float32(1))
    ref = A.astype(np.float64) @ B.astype(np.float64)
    return A, B, ref


class Case:
    """형태 하나의 입력·참조·정답 출력 묶음."""

    def __init__(self, M: int, N: int, K: int, seed: int = 0) -> None:
        self.M, self.N, self.K = M, N, K
        self.A, self.B, self.ref = _inputs(M, N, K, seed)
        self.good = (self.A @ self.B).astype(np.float32)  # float32 누적(허용오차 내)

    def classify(self, out: np.ndarray, **kw: object) -> tuple[str, str, dict]:
        return classify_output(out, self.ref, self.A, self.B, self.M, self.N, self.K, TOL, **kw)  # type: ignore[arg-type]


RECT = Case(20, 13, 31)
SMALL = Case(16, 16, 16, seed=1)
TINY = Case(7, 5, 3, seed=2)


class CompareTest(unittest.TestCase):
    def test_metrics(self) -> None:
        c = compare(RECT.good, RECT.ref, TOL.rtol, TOL.atol)
        self.assertTrue(c["ok"])
        self.assertFalse(c["nan_inf"])
        self.assertEqual(c["n_fail"], 0)
        self.assertLess(c["max_tol_ratio"], 1.0)
        self.assertGreater(c["max_abs_ref"], 0.0)
        bad = RECT.good.copy()
        bad[0, 0] += 1.0
        c = compare(bad, RECT.ref, TOL.rtol, TOL.atol)
        self.assertFalse(c["ok"])
        self.assertEqual(c["n_fail"], 1)
        self.assertGreater(c["max_tol_ratio"], 20.0)
        nan = RECT.good.copy()
        nan[1, 1] = np.nan
        c = compare(nan, RECT.ref, TOL.rtol, TOL.atol)
        self.assertTrue(c["nan_inf"])
        self.assertFalse(c["ok"])
        self.assertEqual(c["n_fail"], 1)
        self.assertIsNotNone(c["max_abs_err"])  # 유한 원소 기준

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            compare(np.zeros(3), np.zeros(4), 1e-3, 1e-3)


class ClassifyOutputTest(unittest.TestCase):
    def test_correct_is_pass(self) -> None:
        for case in (RECT, SMALL, TINY):
            status, pattern, metrics = case.classify(case.good)
            self.assertEqual((status, pattern), ("pass", ""), case.__dict__.keys())
            self.assertEqual(metrics["n_unwritten"], 0)
            self.assertEqual(metrics["n_total"], case.M * case.N)
            self.assertFalse(metrics["oob_write"])
        # 2-D 입력도 1-D 입력도 같다
        self.assertEqual(RECT.classify(RECT.good.ravel())[0], "pass")

    def test_scaled_1p01_is_numeric_imprecise(self) -> None:
        # 더미 numpy/2: C[:] = (A @ B) * 1.01 → numeric_instability/imprecise, max_tol_ratio ≈ 10
        for case in (RECT, SMALL):
            status, pattern, metrics = case.classify(case.good * np.float32(1.01))
            self.assertEqual((status, pattern), ("numeric_instability", "imprecise"))
            self.assertGreater(metrics["max_tol_ratio"], 8.0)
            self.assertLessEqual(metrics["max_tol_ratio"], 10.1)

    def test_transposed_store_is_arg_order(self) -> None:
        # 더미 c_seq/2: C[j*M + i] = sum → out.reshape(N, M)[j, i] = ref[i, j]
        for case in (RECT, SMALL, TINY):
            out_flat = np.ascontiguousarray(case.good.T).ravel()
            status, pattern, _ = case.classify(out_flat)
            self.assertEqual((status, pattern), ("arg_order", "transposed_output"), (case.M, case.N, case.K))

    def test_off_by_one_in_k_is_wrong_output_not_numeric(self) -> None:
        for case in (RECT, SMALL):
            off = (case.A[:, : case.K - 1] @ case.B[: case.K - 1, :]).astype(np.float32)
            status, pattern, metrics = case.classify(off)
            self.assertEqual((status, pattern), ("wrong_output", "mismatch"))
            self.assertGreater(metrics["max_tol_ratio"], TOL.numeric_pattern_tol_ratio)

    def test_all_sentinel_is_output_not_written(self) -> None:
        s = np.full((RECT.M, RECT.N), SENT, dtype=np.float32)
        status, pattern, metrics = RECT.classify(s)
        self.assertEqual((status, pattern), ("wrong_output", "output_not_written"))
        self.assertEqual(metrics["n_unwritten"], RECT.M * RECT.N)
        self.assertEqual(RECT.classify(s, return_type="NoneType")[1], "output_not_written")
        # 더미 numpy/4: return A @ B (C 미기록, 반환값이 ndarray)
        self.assertEqual(RECT.classify(s, return_type="ndarray")[:2], ("wrong_output", "returned_array_not_written"))

    def test_partial_sentinel(self) -> None:
        p = RECT.good.copy()
        p[0, :3] = SENT
        status, pattern, metrics = RECT.classify(p)
        self.assertEqual((status, pattern), ("wrong_output", f"partial_write:3/{RECT.M * RECT.N}"))
        self.assertEqual(metrics["n_unwritten"], 3)
        # 참조가 정확히 0 인 행을 건너뛴 진짜 부분 기록은 누적 검사(sentinel + ref ≈ out)에 걸리지 않고 partial_write 다
        zr = Case(12, 9, 10, seed=7)
        zr.A[0, :] = 0  # ref[0, :] = 0
        zr.ref = zr.A.astype(np.float64) @ zr.B.astype(np.float64)
        zr.good = (zr.A @ zr.B).astype(np.float32)
        skipped = zr.good.copy()
        skipped[0, :] = SENT
        status, pattern, metrics = zr.classify(skipped)
        self.assertEqual((status, pattern), ("wrong_output", f"partial_write:{zr.N}/{zr.M * zr.N}"))
        self.assertEqual(metrics["n_unwritten"], zr.N)

    def test_accumulated_into_c(self) -> None:
        for case in (RECT, SMALL):
            acc = (case.good + SENT).astype(np.float32)  # C += A @ B 를 센티널 위에 한 결과
            self.assertEqual(case.classify(acc)[:2], ("wrong_output", "accumulated_into_c"))

    def test_accumulated_into_c_when_tiny_ref_rounds_back_to_sentinel(self) -> None:
        # float32 에서 -2048 근처 간격은 2⁻¹² 이므로 |ref| < 2⁻¹³ 인 원소는 sentinel + ref 가 정확히 센티널로 되돌아간다.
        # 그런 원소가 n_unwritten 에 세어져도 누적 검사가 먼저이므로 partial_write 가 아니라 accumulated_into_c 다.
        zr = Case(24, 17, 20, seed=11)
        zr.A[3, :] = 0  # ref[3, :] = 0 → 누적 결과가 정확히 센티널
        zr.ref = zr.A.astype(np.float64) @ zr.B.astype(np.float64)
        zr.good = (zr.A @ zr.B).astype(np.float32)
        acc = np.full((zr.M, zr.N), SENT, dtype=np.float32)
        acc += zr.good  # C += A @ B
        self.assertEqual(int(np.count_nonzero(acc == SENT)), zr.N, "전제: 한 행이 센티널로 되돌아간다")
        status, pattern, metrics = zr.classify(acc)
        self.assertEqual((status, pattern), ("wrong_output", "accumulated_into_c"))
        self.assertEqual(metrics["n_unwritten"], zr.N, "n_unwritten 지표는 그대로 센다")

    def test_accumulated_into_c_on_real_small_shape(self) -> None:
        # 실제 small(256³) 입력에서는 |ref| < 2⁻¹³ 인 원소가 1개 있어 예전 순서(partial_write 먼저)로는
        # partial_write:1/65536 이 됐다(실측). 파일이 없으면 건너뛴다.
        path = GEMM_TESTS / "small.npz"
        if not path.exists():
            self.skipTest("tasks/gemm/tests/small.npz 없음")
        with np.load(path) as d:
            A, B, C_ref = d["A"], d["B"], d["C_ref"]
            M, N, K = int(d["M"]), int(d["N"]), int(d["K"])
        acc = np.full((M, N), SENT, dtype=np.float32)
        acc += A @ B
        status, pattern, _ = classify_output(acc, C_ref, A, B, M, N, K, TOL)
        self.assertEqual((status, pattern), ("wrong_output", "accumulated_into_c"))

    def test_nan_inf(self) -> None:
        n = RECT.good.copy()
        n[1, 1] = np.nan
        self.assertEqual(RECT.classify(n)[:2], ("numeric_instability", "nan_inf"))
        i = RECT.good.copy()
        i[2, 2] = np.inf
        self.assertEqual(RECT.classify(i)[:2], ("numeric_instability", "nan_inf"))

    def test_oob_flag_with_correct_output_is_pass_with_flag_in_metrics(self) -> None:
        # v2.1: 허용오차 내(2)가 계약 규칙(4)보다 먼저 — 출력 판정은 pass 이고 플래그는 metrics 로 남는다
        # (샘플 수준은 aggregate_sample / run.py 의 pass 정의, 호출 수준은 워커의 _apply_contract_flags 가 본다).
        status, pattern, metrics = RECT.classify(RECT.good, oob_write=True)
        self.assertEqual((status, pattern), ("pass", ""))
        self.assertTrue(metrics["oob_write"])
        # 출력이 틀리면 OOB 가 imprecise(5)·mismatch(6) 보다 먼저
        self.assertEqual(RECT.classify(RECT.good * np.float32(1.01), oob_write=True)[:2], ("wrong_output", "out_of_bounds_write"))
        off = (RECT.A[:, : RECT.K - 1] @ RECT.B[: RECT.K - 1, :]).astype(np.float32)
        self.assertEqual(RECT.classify(off, oob_write=True)[:2], ("wrong_output", "out_of_bounds_write"))

    def test_column_major_all(self) -> None:
        case = RECT
        A_cm = case.A.ravel().reshape(case.K, case.M).T
        B_cm = case.B.ravel().reshape(case.N, case.K).T
        P = (A_cm.astype(np.float64) @ B_cm.astype(np.float64)).astype(np.float32)
        out_flat = np.ascontiguousarray(P.T).ravel()  # out[i + j*M] = P[i, j]
        self.assertEqual(case.classify(out_flat)[:2], ("arg_order", "column_major_all"))

    def test_swapped_operands_reported_on_square_inputs(self) -> None:
        # 정사각(M==N==K)에서 B@A 는 열우선 해석(A_cmᵀ·B_cmᵀ = (B·A)ᵀ)과 출력이 같다. 설계 §10-3 v2.1 은 swapped_operands 를
        # column_major_all 보다 먼저 검사하므로 그 이름으로 보고한다(모호성은 docs/error_taxonomy.md 에 명시).
        for case in (SMALL, Case(8, 8, 8, seed=5)):
            swapped = (case.B @ case.A).astype(np.float32)
            self.assertEqual(case.classify(swapped)[:2], ("arg_order", "swapped_operands"), (case.M, case.N, case.K))
            # 같은 출력이 match_arg_order 에서도 swapped_operands 다(column_major_all 로 가려지지 않는다)
            self.assertEqual(match_arg_order(swapped.ravel(), case.ref, case.A, case.B, case.M, case.N, case.K, TOL), "swapped_operands")
        # 비정사각에서는 swapped_operands 를 검사하지 않는다(형태가 맞지 않아 계산 자체가 불가능)
        self.assertEqual(match_arg_order(RECT.good.ravel() * np.float32(3), RECT.ref, RECT.A, RECT.B, RECT.M, RECT.N, RECT.K, TOL), "")
        # 비정사각의 열우선 해석은 여전히 column_major_all 로 도달한다(test_column_major_all)

    def test_swapped_mn_dims_via_full_buffer(self) -> None:
        case = RECT
        full = np.zeros(max(case.M, case.N) ** 2 + 1024, dtype=np.float32)
        _write_swapped_mn(full, case.good, case.M, case.N)
        pattern = match_arg_order(full[: case.M * case.N], case.ref, case.A, case.B, case.M, case.N, case.K, TOL, full)
        self.assertEqual(pattern, "swapped_MN_dims")
        # out_full 이 없으면 생략
        self.assertEqual(match_arg_order(full[: case.M * case.N], case.ref, case.A, case.B, case.M, case.N, case.K, TOL, None), "")
        self.assertEqual(case.classify(full[: case.M * case.N], out_full=full)[:2], ("arg_order", "swapped_MN_dims"))
        # 정사각이면 검사하지 않는다
        sq_full = np.zeros(SMALL.M ** 2 + 1024, dtype=np.float32)
        _write_swapped_mn(sq_full, SMALL.good, SMALL.M, SMALL.N)
        self.assertEqual(match_arg_order(sq_full[: SMALL.M * SMALL.N], SMALL.ref, SMALL.A, SMALL.B, SMALL.M, SMALL.N, SMALL.K, TOL, sq_full), "")

    def test_swapped_mn_dims_reachable_through_sentinel_and_oob_on_real_rect_shape(self) -> None:
        # 실제 rect(200,130,312) 형태에서 C[i*M + j] 로 쓰는 커널: 워커의 센티널 가드 버퍼 위에서는 view 안에 미기록
        # 원소가 남고(partial_write 조건) 가드에도 쓴다(oob_write 조건). v2.1 전에는 이 둘에 가려 swapped_MN_dims 가
        # 도달 불가였다. 이제 arg_order 로 보고하고 n_unwritten/oob_write 는 metrics 에 그대로 남는다.
        for case in (Case(200, 130, 312, seed=20261103), RECT, TINY):
            full = _guarded_full(case.M, case.N)
            _write_swapped_mn(full, case.good, case.M, case.N)
            view = full[: case.M * case.N]
            oob = bool(np.any(full[case.M * case.N :] != SENT))
            n_sent = int(np.count_nonzero(view == SENT))
            self.assertTrue(oob and n_sent > 0, f"전제: 센티널 {n_sent}개·OOB {oob} ({case.M},{case.N})")
            status, pattern, metrics = case.classify(view, oob_write=oob, out_full=full)
            self.assertEqual((status, pattern), ("arg_order", "swapped_MN_dims"), (case.M, case.N, case.K))
            self.assertEqual(metrics["n_unwritten"], n_sent)
            self.assertTrue(metrics["oob_write"])
        # 같은 버퍼라도 out_full 을 넘기지 않으면(워커가 넘기지 않는 회귀) 센티널 규칙으로 떨어진다
        case = Case(200, 130, 312, seed=20261103)
        full = _guarded_full(case.M, case.N)
        _write_swapped_mn(full, case.good, case.M, case.N)
        status, pattern, _ = case.classify(full[: case.M * case.N], oob_write=True)
        self.assertEqual(status, "wrong_output")
        self.assertTrue(pattern.startswith("partial_write:"), pattern)

    def test_precedence_combinations(self) -> None:
        mn = RECT.M * RECT.N
        # NaN(1) 이 센티널 규칙(4) 보다 먼저: 센티널은 유한값이라 NaN 검사에서 제외되고, 남은 센티널은 n_unwritten 에만 남는다
        p = RECT.good.copy()
        p[0, 0] = SENT
        p[1, 1] = np.nan
        status, pattern, metrics = RECT.classify(p)
        self.assertEqual((status, pattern), ("numeric_instability", "nan_inf"))
        self.assertEqual(metrics["n_unwritten"], 1)
        # NaN(1) 이 OOB(4) 보다 먼저
        n = RECT.good.copy()
        n[1, 1] = np.nan
        self.assertEqual(RECT.classify(n, oob_write=True)[:2], ("numeric_instability", "nan_inf"))
        # NaN(1) 이 arg_order(3) 보다 먼저
        t = np.ascontiguousarray(RECT.good.T).ravel().copy()
        t[0] = np.nan
        self.assertEqual(RECT.classify(t)[:2], ("numeric_instability", "nan_inf"))
        # pass(2) 가 arg_order(3) 보다 먼저: 대칭 입력이면 전치해도 pass
        sym = Case(8, 8, 8, seed=3)
        sym.B = np.ascontiguousarray(sym.A.T)
        sym.ref = sym.A.astype(np.float64) @ sym.B.astype(np.float64)  # A·Aᵀ 는 대칭
        sym.good = (sym.A @ sym.B).astype(np.float32)
        self.assertEqual(sym.classify(np.ascontiguousarray(sym.good.T).ravel())[0], "pass")
        # arg_order(3) 가 센티널·OOB(4) 보다 먼저: 전치 저장 + 가드 쓰기 → transposed_output (플래그는 metrics 에)
        status, pattern, metrics = RECT.classify(np.ascontiguousarray(RECT.good.T).ravel(), oob_write=True)
        self.assertEqual((status, pattern), ("arg_order", "transposed_output"))
        self.assertTrue(metrics["oob_write"])
        # 규칙 4 안: 전부 센티널(+ndarray 반환) → 누적 → 부분 기록 → OOB 순서
        s = np.full((RECT.M, RECT.N), SENT, dtype=np.float32)
        self.assertEqual(RECT.classify(s, oob_write=True, return_type="ndarray")[1], "returned_array_not_written")
        acc = (RECT.good + SENT).astype(np.float32)
        self.assertEqual(RECT.classify(acc, oob_write=True)[1], "accumulated_into_c")
        q = RECT.good.copy()
        q[2, 3] = SENT
        self.assertEqual(RECT.classify(q, oob_write=True)[1], f"partial_write:1/{mn}")
        # 규칙 4(OOB) 가 imprecise(5) 보다 먼저
        self.assertEqual(RECT.classify(RECT.good * np.float32(1.01), oob_write=True)[1], "out_of_bounds_write")
        # imprecise(5) 는 arg_order(3) 뒤: 전치된 채 크게 틀리면 어느 패턴도 아니고 mismatch
        self.assertEqual(RECT.classify(np.ascontiguousarray(RECT.good.T).ravel() * np.float32(1.5))[:2], ("wrong_output", "mismatch"))

    def test_size_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            RECT.classify(RECT.good[:, :-1])


class AggregateSampleTest(unittest.TestCase):
    @staticmethod
    def _t(name: str, cls: str, pattern: str = "", nondet: bool = False) -> dict:
        return {"name": name, "classification": cls, "pattern": pattern, "nondeterministic": nondet}

    def test_all_pass(self) -> None:
        self.assertEqual(aggregate_sample([self._t("tiny", "pass"), self._t("rect", "pass")]), ("pass", ""))
        self.assertEqual(aggregate_sample([]), ("pass", ""))

    def test_crash_beats_everything(self) -> None:
        tests = [self._t("tiny", "arg_order", "transposed_output", nondet=True), self._t("rect", "runtime_crash", "ZeroDivisionError: x")]
        self.assertEqual(aggregate_sample(tests, inputs_mutated=True), ("runtime_crash", "rect: ZeroDivisionError: x"))

    def test_race_beats_arg_order_and_wrong_output(self) -> None:
        tests = [self._t("tiny", "pass"), self._t("rect", "wrong_output", "mismatch", nondet=True), self._t("small", "wrong_output", "mismatch", nondet=True)]
        self.assertEqual(aggregate_sample(tests), ("race_condition", "nondeterministic:rect,small"))

    def test_arg_order_requires_same_pattern_on_all_failed(self) -> None:
        tests = [self._t("tiny", "arg_order", "transposed_output"), self._t("rect", "arg_order", "transposed_output"), self._t("small", "arg_order", "transposed_output")]
        self.assertEqual(aggregate_sample(tests), ("arg_order", "transposed_output"))
        # 일부 pass + 나머지 같은 패턴이어도 arg_order
        tests = [self._t("tiny", "pass"), self._t("rect", "arg_order", "transposed_output")]
        self.assertEqual(aggregate_sample(tests), ("arg_order", "transposed_output"))
        # 패턴이 다르면 arg_order 가 아니다(보수적으로 wrong_output)
        tests = [self._t("rect", "arg_order", "transposed_output"), self._t("small", "arg_order", "swapped_operands")]
        status, detail = aggregate_sample(tests)
        self.assertEqual(status, "wrong_output")
        self.assertTrue(detail.startswith("mixed:"))

    def test_wrong_output_beats_numeric_and_arg_order(self) -> None:
        tests = [self._t("tiny", "arg_order", "transposed_output"), self._t("rect", "wrong_output", "mismatch"), self._t("small", "numeric_instability", "imprecise")]
        self.assertEqual(aggregate_sample(tests), ("wrong_output", "mismatch"))
        # 입력 변조·OOB 는 wrong_output 으로 센다(arg_order 전부 같은 패턴이면 arg_order 가 먼저)
        tests = [self._t("tiny", "numeric_instability", "imprecise")]
        self.assertEqual(aggregate_sample(tests, inputs_mutated=True), ("wrong_output", "inputs_mutated"))
        self.assertEqual(aggregate_sample([self._t("tiny", "pass")], oob_write=True), ("wrong_output", "out_of_bounds_write"))

    def test_numeric_only(self) -> None:
        tests = [self._t("tiny", "pass"), self._t("rect", "numeric_instability", "imprecise"), self._t("small", "numeric_instability", "nan_inf")]
        self.assertEqual(aggregate_sample(tests), ("numeric_instability", "imprecise"))


class ToleranceFileTest(unittest.TestCase):
    def test_load_real_tolerance(self) -> None:
        path = ROOT / "tasks" / "gemm" / "tolerance.yaml"
        if not path.exists():
            self.skipTest("tasks/gemm/tolerance.yaml 없음")
        tol = load_tolerance(path)
        self.assertEqual(tol.task, "gemm")
        self.assertEqual(tol.rtol, 1e-3)
        self.assertEqual(tol.atol, 1e-3)
        self.assertEqual(tol.numeric_pattern_tol_ratio, 20.0)
        self.assertEqual(tol.sentinel, -2048.0)
        self.assertEqual(tol.dtype, "float32")
        self.assertEqual(tol.reference_dtype, "float64")

    def test_existing_tasks_default_to_tolerance_compare(self) -> None:
        for task in ("gemm", "jacobi2d", "spmv_csr", "dot"):
            tol = load_tolerance(ROOT / "tasks" / task / "tolerance.yaml")
            self.assertEqual(tol.compare, "tolerance", task)
            self.assertGreater(tol.rtol, 0.0)
        self.assertEqual(Tolerance(rtol=1e-3, atol=1e-3, numeric_pattern_tol_ratio=20.0, sentinel=-2048.0).compare, "tolerance")
        self.assertEqual(COMPARE_MODES, ("tolerance", "exact"))

    def _load(self, text: str) -> Tolerance:
        with tempfile.TemporaryDirectory(prefix="tol_test_") as tmp:
            path = Path(tmp) / "tolerance.yaml"
            path.write_text(text, encoding="utf-8", newline="\n")
            return load_tolerance(path)

    def test_exact_defaults_do_not_need_rtol_atol(self) -> None:
        # 설계 v3 §2.2 의 예 그대로
        tol = self._load("task: histogram\ndtype: int32\ncompare: exact          # tolerance(기본) | exact\nsentinel: -2048\n")
        self.assertEqual(tol.compare, "exact")
        self.assertEqual((tol.rtol, tol.atol, tol.numeric_pattern_tol_ratio), (0.0, 0.0, 0.0))
        self.assertEqual(tol.sentinel, -2048.0)
        self.assertEqual((tol.task, tol.dtype, tol.reference_dtype), ("histogram", "int32", "int32"))
        # 0 을 명시해도 된다
        tol = self._load("dtype: int32\ncompare: exact\nrtol: 0\natol: 0.0\nnumeric_pattern_tol_ratio: 0\nsentinel: -7\n")
        self.assertEqual((tol.compare, tol.sentinel), ("exact", -7.0))

    def test_exact_rejects_inconsistent_files(self) -> None:
        with self.assertRaises(KeyError):  # sentinel 은 정확 비교에서도 필수
            self._load("dtype: int32\ncompare: exact\n")
        with self.assertRaises(ValueError):  # 정확 비교인데 허용오차가 0 이 아님
            self._load("dtype: int32\ncompare: exact\nrtol: 1.0e-3\nsentinel: -2048\n")
        with self.assertRaises(ValueError):  # 정수 출력의 센티널은 정수
            self._load("dtype: int32\ncompare: exact\nsentinel: -2048.5\n")
        with self.assertRaises(ValueError):  # 모르는 비교 방식
            self._load("dtype: int32\ncompare: fuzzy\nsentinel: -2048\n")

    def test_tolerance_mode_still_requires_all_keys(self) -> None:
        for missing in ("rtol", "atol", "numeric_pattern_tol_ratio", "sentinel"):
            keys = {"rtol": "1.0e-3", "atol": "1.0e-3", "numeric_pattern_tol_ratio": "20", "sentinel": "-2048.0"}
            del keys[missing]
            with self.assertRaises(KeyError, msg=missing):
                self._load("".join(f"{k}: {v}\n" for k, v in keys.items()))


# ----------------------------------------------------------------------------- 정확 비교 (v3 §2.3)

EXACT = Tolerance(rtol=0.0, atol=0.0, numeric_pattern_tol_ratio=0.0, sentinel=-2048.0, task="histogram", dtype="int32",
                  reference_dtype="int32", compare="exact")
ISENT = np.int32(-2048)
HIST_REF = np.array([3, 0, 5, 1, 0, 7, 2, 0], dtype=np.int32)  # 개수가 0 인 칸이 있는 히스토그램


class CompareExactTest(unittest.TestCase):
    def test_keys_match_compare(self) -> None:
        self.assertEqual(set(compare_exact(HIST_REF, HIST_REF)), set(compare(np.zeros(3), np.zeros(3), 1e-3, 1e-3)))

    def test_equal(self) -> None:
        c = compare_exact(HIST_REF.copy(), HIST_REF)
        self.assertEqual(c, {"ok": True, "nan_inf": False, "max_abs_err": 0.0, "max_tol_ratio": None, "n_fail": 0, "max_abs_ref": 7.0})
        self.assertIsInstance(c["max_abs_err"], float)

    def test_differences_are_counted_exactly(self) -> None:
        out = HIST_REF.copy()
        out[0] += 1
        out[5] -= 4
        c = compare_exact(out, HIST_REF)
        self.assertFalse(c["ok"])
        self.assertEqual((c["n_fail"], c["max_abs_err"], c["max_tol_ratio"], c["nan_inf"]), (2, 4.0, None, False))

    def test_int32_extremes_do_not_overflow(self) -> None:
        lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
        c = compare_exact(np.array([lo, hi], dtype=np.int32), np.array([hi, lo], dtype=np.int32))
        self.assertEqual((c["n_fail"], c["max_abs_err"], c["max_abs_ref"]), (2, float(hi) - float(lo), float(-lo)))

    def test_shapes_and_empty(self) -> None:
        self.assertTrue(compare_exact(HIST_REF.reshape(2, 4), HIST_REF)["ok"], "형태가 달라도 원소 순서가 같으면 같다")
        with self.assertRaises(ValueError):
            compare_exact(np.zeros(3, np.int32), np.zeros(4, np.int32))
        c = compare_exact(np.zeros(0, np.int32), np.zeros(0, np.int32))
        self.assertEqual((c["ok"], c["max_abs_err"], c["n_fail"], c["max_abs_ref"]), (True, None, 0, 0.0))

    def test_one_ulp_float_difference_is_a_failure(self) -> None:
        ref = np.array([1.0, 2.0], dtype=np.float32)
        out = ref.copy()
        out[1] = np.nextafter(np.float32(2.0), np.float32(3.0))
        c = compare_exact(out, ref)
        self.assertEqual((c["ok"], c["n_fail"]), (False, 1))
        self.assertTrue(compare(out, ref, 1e-3, 1e-3)["ok"], "허용오차 비교라면 통과하는 차이다")


class ClassifyExactTest(unittest.TestCase):
    """정확 비교 과제의 출력 판정: 규칙 1·5 없음, 2 → 3 → 4(전부 센티널 → 누적 → 부분 기록 → OOB) → 6."""

    def classify(self, out: np.ndarray, ref: np.ndarray = HIST_REF, **kw: object) -> tuple[str, str, dict]:
        return classify_task("histogram", out, ref, {}, {}, EXACT, **kw)  # type: ignore[arg-type]

    def test_pass(self) -> None:
        status, pattern, metrics = self.classify(HIST_REF.copy())
        self.assertEqual((status, pattern), ("pass", ""))
        self.assertEqual(metrics, {"max_abs_err": 0.0, "max_tol_ratio": None, "n_fail": 0, "nan_inf": False, "n_unwritten": 0,
                                   "n_total": 8, "max_abs_ref": 7.0, "oob_write": False})

    def test_mismatch_even_by_one(self) -> None:
        out = HIST_REF.copy()
        out[2] += 1
        status, pattern, metrics = self.classify(out)
        self.assertEqual((status, pattern), ("wrong_output", "mismatch"))
        self.assertEqual((metrics["n_fail"], metrics["max_abs_err"], metrics["n_unwritten"]), (1, 1.0, 0))
        # 허용오차 경로라면 imprecise 였을 작은 차이도 정확 비교에서는 numeric_instability 가 아니다
        self.assertNotEqual(status, "numeric_instability")

    def test_output_not_written(self) -> None:
        out = np.full(HIST_REF.size, ISENT, dtype=np.int32)
        status, pattern, metrics = self.classify(out)
        self.assertEqual((status, pattern), ("wrong_output", "output_not_written"))
        self.assertEqual(metrics["n_unwritten"], 8)
        self.assertEqual(self.classify(out, return_type="NoneType")[1], "output_not_written")
        self.assertEqual(self.classify(out, return_type="ndarray")[:2], ("wrong_output", "returned_array_not_written"))

    def test_accumulated_into_c_for_histogram_that_was_not_zeroed(self) -> None:
        # 출력 버퍼를 0 으로 지우지 않고 hist[k] += 1: 초기 내용이 센티널이므로 out == sentinel + ref
        out = np.full(HIST_REF.size, ISENT, dtype=np.int32)
        out += HIST_REF
        self.assertEqual(int(np.count_nonzero(out == ISENT)), 3, "전제: 개수가 0 인 칸 3개는 센티널 그대로다")
        status, pattern, metrics = self.classify(out)
        self.assertEqual((status, pattern), ("wrong_output", "accumulated_into_c"))
        self.assertEqual(metrics["n_unwritten"], 3, "n_unwritten 지표는 그대로 세지만 부분 기록보다 누적이 먼저다")
        self.assertEqual(metrics["max_abs_err"], 2048.0)
        # 누적 + OOB 도 누적으로 보고한다(규칙 4 의 순서)
        self.assertEqual(self.classify(out, oob_write=True)[:2], ("wrong_output", "accumulated_into_c"))

    def test_partial_write(self) -> None:
        out = HIST_REF.copy()
        out[1] = ISENT
        out[6] = ISENT
        status, pattern, metrics = self.classify(out)
        self.assertEqual((status, pattern), ("wrong_output", "partial_write:2/8"))
        self.assertEqual(metrics["n_unwritten"], 2)
        # 개수가 0 인 칸만 쓰지 않은 커널(0 이 아닌 칸은 맞음)은 누적이 아니라 부분 기록이다
        skipped = HIST_REF.copy()
        skipped[HIST_REF == 0] = ISENT
        self.assertEqual(self.classify(skipped)[:2], ("wrong_output", "partial_write:3/8"))
        # 부분 기록 + OOB 는 부분 기록이 먼저
        self.assertEqual(self.classify(out, oob_write=True)[1], "partial_write:2/8")

    def test_oob_write(self) -> None:
        # 출력이 맞으면 규칙 2 가 먼저(pass) — 플래그는 metrics 에 남고 워커가 호출 수준에서 덮는다
        status, pattern, metrics = self.classify(HIST_REF.copy(), oob_write=True)
        self.assertEqual((status, pattern), ("pass", ""))
        self.assertTrue(metrics["oob_write"])
        # 출력이 틀리고 센티널이 없으면 out_of_bounds_write 가 mismatch 보다 먼저
        out = HIST_REF.copy()
        out[0] += 1
        self.assertEqual(self.classify(out, oob_write=True)[:2], ("wrong_output", "out_of_bounds_write"))
        self.assertEqual(self.classify(out)[:2], ("wrong_output", "mismatch"))

    def test_transposed_2d_int_output_is_arg_order(self) -> None:
        ref = np.arange(12, dtype=np.int32).reshape(3, 4) + 1
        self.assertEqual(self.classify(ref.copy(), ref)[0], "pass")
        transposed = np.ascontiguousarray(ref.T).reshape(-1)
        self.assertEqual(self.classify(transposed, ref)[:2], ("arg_order", "transposed_output"))
        # 전치된 채 한 원소가 틀리면 패턴이 아니다
        transposed[0] += 1
        self.assertEqual(self.classify(transposed, ref)[:2], ("wrong_output", "mismatch"))

    def test_pattern_fn_is_consulted_between_pass_and_sentinel_rules(self) -> None:
        calls: list[int] = []

        def pattern_fn(o: np.ndarray, r: np.ndarray) -> str:
            calls.append(o.size)
            return "custom_pattern"

        self.assertEqual(classify_generic(HIST_REF.copy(), HIST_REF, EXACT, pattern_fn=pattern_fn)[:2], ("pass", ""))
        self.assertEqual(calls, [], "통과한 출력에는 패턴 함수를 부르지 않는다")
        out = np.full(HIST_REF.size, ISENT, dtype=np.int32)
        self.assertEqual(classify_generic(out, HIST_REF, EXACT, pattern_fn=pattern_fn)[:2], ("arg_order", "custom_pattern"))
        self.assertEqual(calls, [8])

    def test_sort_like_full_range_values(self) -> None:
        # 전체 int32 범위의 값(정렬 과제): 차가 int32 를 넘어도 지표가 깨지지 않는다
        lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
        ref = np.array([lo, -5, 0, 7, hi], dtype=np.int32)
        self.assertEqual(self.classify(ref.copy(), ref)[0], "pass")
        status, pattern, metrics = self.classify(ref[::-1].copy(), ref)
        self.assertEqual((status, pattern), ("wrong_output", "mismatch"))
        self.assertEqual(metrics["max_abs_err"], float(hi) - float(lo))

    def test_size_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.classify(HIST_REF[:-1].copy())

    def test_tolerance_path_is_not_affected_by_exact_branch(self) -> None:
        # 같은 float 출력이 허용오차 Tolerance 에서는 예전 규칙(imprecise), 정확 비교 Tolerance 에서는 mismatch
        ref = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        out = (ref * 1.005).astype(np.float32)
        self.assertEqual(classify_generic(out, ref, TOL)[:2], ("numeric_instability", "imprecise"))
        exact_f = Tolerance(rtol=0.0, atol=0.0, numeric_pattern_tol_ratio=0.0, sentinel=-2048.0, compare="exact")
        self.assertEqual(classify_generic(out, ref, exact_f)[:2], ("wrong_output", "mismatch"))


if __name__ == "__main__":
    unittest.main()
