"""Tier 3 ``floyd_warshall`` 과제 테스트(설계 §18).

- 생성기(tasks/floyd_warshall/gen_tests.py)의 결정성: 같은 시드로 두 번 만든 npz 가 바이트 단위로 같다.
- tiny 참조를 순수 Python 삼중 루프(배정밀도)와 대조한다. tiny(n=8, 밀도 0.3)에는 도달 불가 쌍이 있고 참조가 정확히 INF 임을,
  rect(n=37)에는 도달 불가 쌍이 없음을 확인만 한다(밀도는 바꾸지 않는다 — §18).
- 더미 응답 4개(numpy/0, c_seq/0, c_openmp/0, c_openmp/1)의 추출(펜스 1개·시그니처 OK).
- python 정답 더미(numpy/0)의 worker check(tiny·rect) pass.
- 심은 버그: k 루프를 건너뛰고 입력만 복사하는 커널은 샘플 수준 ``wrong_output``/``mismatch`` — rect 가 잡는다. 기본 시드의 tiny 는
  2홉 경로가 없어 참조 == 입력이라 tiny 만으로는 잡히지 않는다(테스트에 사실로 기록).
- C 기준선·C 더미를 실제 빌드 플래그로 컴파일해 worker check: 정답은 pass, k 루프를 병렬화한 c_openmp/1 은 pass 가 아니다
  (실측 3/3 ``race_condition``/``nondeterministic:rect,small``; 두 호출이 같게 틀리면 ``wrong_output`` — 둘 다 허용).

LLM 코드를 흉내 낸 커널은 모두 ``harness.worker`` 서브프로세스에서 실행한다. 테스트 npz 는 임시 디렉터리에 직접 만든다
(tiny·rect 는 실제 평가 크기, small·large 는 작게 줄인다; tasks/floyd_warshall/tests 불필요). gcc 가 없으면 C 테스트를 건너뛴다.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np
import yaml

from harness.build.c import build_c
from harness.classify import classify_task, load_tolerance
from harness.config import load_config
from harness.extract import extract
from harness.paths import PYTHON_EXE, ROOT
from harness.procutil import run_isolated
from harness.protocol import launch_env_minimal
from harness.taskspec import load_taskspec
from harness.testgen import DEFAULT_SEED, TEST_NAMES, generate

TASK_DIR = ROOT / "tasks" / "floyd_warshall"
DUMMY_DIR = ROOT / "harness" / "dummy_responses" / "floyd_warshall"
SIGNATURES: dict = yaml.safe_load((TASK_DIR / "signatures.yaml").read_text(encoding="utf-8"))
TOL = load_tolerance(TASK_DIR / "tolerance.yaml")
INF = np.float32(1.0e30)
REF_METHOD = "unit-test"

# 단위 테스트용 형태: tiny·rect 는 실제 평가 크기(5, 37), small·large 는 빠르게 줄인다(이름은 TEST_NAMES 순서여야 한다)
UNIT_SHAPES = [("tiny", {"n": 8}), ("rect", {"n": 37}), ("small", {"n": 64}), ("large", {"n": 96})]

# 심은 버그: k 루프 없이 입력을 복사만 한다(간선 가중치가 곧 거리라고 가정). 2홉 이상 경로와 도달 가능성이 전부 틀린다.
PY_COPY_ONLY = """import numpy as np

def floyd_warshall(dist_in, dist_out, n):
    np.copyto(dist_out, dist_in)
"""


def _gen_tests_module() -> ModuleType:
    """``tasks/floyd_warshall/gen_tests.py`` 를 모듈로 로드한다(실행하지 않고 make_case/reference 만 쓴다)."""
    spec = importlib.util.spec_from_file_location("floyd_warshall_gen_tests", TASK_DIR / "gen_tests.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dummy_code(lang: str, k: int, backend: str) -> str:
    res = extract((DUMMY_DIR / lang / f"{k}.md").read_text(encoding="utf-8"), backend=backend, signature_spec=SIGNATURES)
    assert res.ok and res.code is not None, f"{lang}/{k}: {res.detail}"
    return res.code


def pure_python_floyd_warshall(dist_in: np.ndarray) -> list[list[float]]:
    """배정밀도 순수 Python 삼중 루프(참조의 참조)."""
    n = dist_in.shape[0]
    d = [[float(dist_in[i, j]) for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            dik = d[i][k]
            for j in range(n):
                via = dik + d[k][j]
                if via < d[i][j]:
                    d[i][j] = via
    return d


def run_worker(backend: str, artifact: Path, mode: str, tests: list[Path], out: Path, *, threads: int,
               timeout_s: float = 300.0) -> dict:
    """``harness.worker`` 를 서브프로세스로 실행하고 결과 JSON 을 돌려준다(rc != 0 이면 AssertionError)."""
    cmd = [str(PYTHON_EXE), "-m", "harness.worker", "--backend", backend, "--artifact", str(artifact), "--mode", mode,
           "--tests", ",".join(str(t) for t in tests), "--tolerance", str(TASK_DIR / "tolerance.yaml"),
           "--signatures", str(TASK_DIR / "signatures.yaml"), "--out", str(out)]
    env = dict(launch_env_minimal())
    env.update({"OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)})
    proc = run_isolated(cmd, timeout_s=timeout_s, cwd=ROOT, env=env)
    if proc.rc != 0:
        raise AssertionError(f"worker rc={proc.rc}: {proc.stderr_tail}")
    return json.loads(out.read_text(encoding="utf-8"))


class SpecFilesTest(unittest.TestCase):
    """과제 파일(signatures/tolerance/spec)이 §18 계약과 맞는가."""

    def test_taskspec_and_tolerance(self) -> None:
        spec = load_taskspec(TASK_DIR / "signatures.yaml")
        self.assertEqual((spec.task, spec.function, spec.params), ("floyd_warshall", "floyd_warshall", ["dist_in", "dist_out", "n"]))
        self.assertEqual((spec.inputs[0].dtype, spec.output.dtype), ("float32", "float32"))
        self.assertEqual(spec.shape_of(spec.inputs[0], {"n": 37}), (37, 37))
        self.assertEqual(spec.shape_of(spec.output, {"n": 37}), (37, 37))
        self.assertEqual(spec.work_of({"n": 1024}), 2.0 * 1024**3)
        self.assertEqual(spec.baseline["backend"], "c")
        self.assertTrue(spec.baseline["openmp"])
        self.assertTrue((TASK_DIR / spec.baseline["file"]).is_file())
        self.assertEqual((TOL.rtol, TOL.atol, TOL.numeric_pattern_tol_ratio, TOL.sentinel), (1.0e-4, 1.0e-3, 20.0, -2048.0))
        self.assertIsNone(TOL.atol_rms_rel)
        self.assertEqual((TOL.task, TOL.dtype, TOL.reference_dtype), ("floyd_warshall", "float32", "float64"))
        for lang in ("numpy", "c_seq", "c_openmp"):
            entry = SIGNATURES["languages"][lang]
            self.assertIn("floyd_warshall(", entry["positional"])
            self.assertIn("1.0e30", entry["param_docs"])
        self.assertEqual(SIGNATURES["languages"]["numpy"]["keyword"], "def floyd_warshall(*, dist_in, dist_out, n):")
        self.assertEqual(SIGNATURES["languages"]["c_seq"]["positional"],
                         "void floyd_warshall(const float* dist_in, float* dist_out, int n)")

    def test_spec_structure_is_language_neutral(self) -> None:
        spec_md = (TASK_DIR / "spec.md").read_text(encoding="utf-8")
        for heading in ("## Definition", "## Data layout", "## Contract", "## Accuracy", "## Evaluation", "## Goal"):
            self.assertIn(heading, spec_md)
        for needle in ("INF = 1.0e30", "rtol = 1e-4", "atol = 1e-3", "8, 37 and 256", "1,024", "8 GiB", "exactly three parameters"):
            self.assertIn(needle, spec_md)
        for word in ("numpy", "NumPy", "OpenMP", "BLAS", "#pragma", "np.", "memcpy"):
            self.assertNotIn(word, spec_md)
        self.assertTrue((TASK_DIR / "spec.ko.md").read_text(encoding="utf-8").startswith("# 과제:"))

    def test_inf_arithmetic_assumptions(self) -> None:
        """spec 이 공개한 INF 성질: float32 에서 INF + w == INF, INF + INF 는 유한, float64 에서도 INF + w == INF."""
        w = np.float32(9.999)
        self.assertEqual(INF + w, INF)
        self.assertTrue(np.isfinite(INF + INF))
        self.assertGreater(INF + INF, INF)
        self.assertEqual(np.float64(INF) + np.float64(w), np.float64(INF))
        # 허용오차 비교: float32 INF 는 참조의 float64 INF 와 통과, 2·INF 는 배율 20 을 훨씬 넘어 wrong_output 으로 떨어진다
        ref = np.full((2, 2), np.float64(INF))
        np.fill_diagonal(ref, 0.0)
        out = ref.astype(np.float32)
        self.assertEqual(classify_task("floyd_warshall", out, ref, {}, {"n": 2}, TOL)[0], "pass")
        bad = out.copy()
        bad[0, 1] = INF + INF
        st, pat, m = classify_task("floyd_warshall", bad, ref, {}, {"n": 2}, TOL)
        self.assertEqual((st, pat), ("wrong_output", "mismatch"))
        self.assertGreater(m["max_tol_ratio"], TOL.numeric_pattern_tol_ratio)


class GenTestsTest(unittest.TestCase):
    """생성기: 손계산 참조, 형태, 도달 불가 쌍, 결정성."""

    def setUp(self) -> None:
        self.gen = _gen_tests_module()

    def test_shapes_and_constants(self) -> None:
        self.assertEqual([n for n, _ in self.gen.SHAPES], list(TEST_NAMES))
        self.assertEqual([s["n"] for _, s in self.gen.SHAPES], [8, 37, 256, 1024])
        self.assertEqual(self.gen.INF, INF)
        self.assertEqual(self.gen.DENSITY, 0.3)

    def test_graph_layout(self) -> None:
        dist = self.gen.make_graph(37, np.random.default_rng(DEFAULT_SEED + 1))
        self.assertEqual((dist.dtype, dist.shape), (np.float32, (37, 37)))
        self.assertTrue(dist.flags["C_CONTIGUOUS"])
        self.assertTrue(np.all(np.diag(dist) == 0.0))
        off = dist[~np.eye(37, dtype=bool)]
        finite = off[off != INF]
        self.assertTrue(np.all((finite >= 1.0) & (finite < 10.0)))
        self.assertTrue(np.all((off == INF) | (off < 10.0)))
        self.assertTrue(0.15 < finite.size / off.size < 0.45)  # 밀도 0.3 근방

    def test_reference_hand_check_tiny(self) -> None:
        dims, inputs, out_name, ref = self.gen.make_case({"n": 8}, np.random.default_rng(DEFAULT_SEED))
        self.assertEqual((dims, out_name, ref.dtype, ref.shape), ({"n": 8}, "dist_out", np.float64, (8, 8)))
        expected = pure_python_floyd_warshall(inputs["dist_in"])
        self.assertEqual(ref.tolist(), expected)
        self.assertTrue(np.all(np.diag(ref) == 0.0))
        # 작은 손계산 그래프: 0→1 (2), 1→2 (3), 2→0 (4), 3 은 고립
        g = np.full((4, 4), INF, dtype=np.float32)
        np.fill_diagonal(g, 0.0)
        g[0, 1], g[1, 2], g[2, 0] = 2.0, 3.0, 4.0
        r = self.gen.reference(g)
        self.assertEqual(r[0].tolist(), [0.0, 2.0, 5.0, float(INF)])
        self.assertEqual(r[1].tolist(), [7.0, 0.0, 3.0, float(INF)])
        self.assertEqual(r[2].tolist(), [4.0, 6.0, 0.0, float(INF)])
        self.assertEqual(r[3].tolist(), [float(INF)] * 3 + [0.0])

    def test_unreachable_pairs_tiny_yes_rect_no(self) -> None:
        """밀도 0.3 그대로 둔 결과(§18): tiny(n=8) 에는 도달 불가 쌍이 있고 참조가 정확히 INF, rect(n=37) 에는 없다.
        기본 시드 기준의 사실 기록이지 요구사항이 아니다 — 시드가 바뀌어 어긋나면 이 테스트와 README 의 문장을 갱신한다."""
        _, inputs_tiny, _, ref_tiny = self.gen.make_case({"n": 8}, np.random.default_rng(DEFAULT_SEED))
        unreachable_tiny = int(np.count_nonzero(ref_tiny == np.float64(INF)))
        self.assertEqual(unreachable_tiny, 13)  # n=8(2026-11-09 변경): 비대각 56쌍 중 13쌍이 도달 불가, 완화되는 쌍 27개
        self.assertTrue(np.all((ref_tiny >= 1e29) == (ref_tiny == np.float64(INF))))  # 큰 값은 정확히 INF 뿐
        # 간선 4개(1→0, 1→2, 3→2, 4→2)로는 2홉 경로가 없어 참조가 입력과 같다 — tiny 는 완화 자체를 검사하지 못한다(알려진 한계)
        self.assertTrue(np.any(ref_tiny < inputs_tiny["dist_in"].astype(np.float64) - 1e-6))  # n=8: 완화되는 쌍이 있다
        _, _, _, ref_rect = self.gen.make_case({"n": 37}, np.random.default_rng(DEFAULT_SEED + 1))
        self.assertEqual(int(np.count_nonzero(ref_rect == np.float64(INF))), 0)
        self.assertLess(float(ref_rect.max()), 100.0)

    def test_generate_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = Path(tmp) / "a", Path(tmp) / "b", Path(tmp) / "c"
            ma = generate("floyd_warshall", DEFAULT_SEED, a, UNIT_SHAPES, self.gen.make_case, REF_METHOD)
            mb = generate("floyd_warshall", DEFAULT_SEED, b, UNIT_SHAPES, self.gen.make_case, REF_METHOD)
            self.assertEqual(ma["test_hash"], mb["test_hash"])
            self.assertEqual(ma["files"], mb["files"])
            for name, shape in UNIT_SHAPES:
                self.assertEqual((a / f"{name}.npz").read_bytes(), (b / f"{name}.npz").read_bytes(), name)
                with np.load(a / f"{name}.npz") as npz:
                    self.assertEqual(set(npz.files), {"dist_in", "dist_out_ref", "n", "seed"})
                    self.assertEqual((npz["dist_in"].dtype, npz["dist_out_ref"].dtype), (np.float32, np.float64))
                    self.assertEqual((int(npz["n"]), npz["dist_in"].shape, npz["dist_out_ref"].shape),
                                     (shape["n"], (shape["n"],) * 2, (shape["n"],) * 2))
                    self.assertTrue(np.all(npz["dist_out_ref"] <= npz["dist_in"]))  # 완화는 거리를 줄이기만 한다
            mc = generate("floyd_warshall", DEFAULT_SEED + 1, c, UNIT_SHAPES, self.gen.make_case, REF_METHOD)
            self.assertNotEqual(ma["test_hash"], mc["test_hash"])


class DummyResponsesTest(unittest.TestCase):
    """더미 응답 4개(numpy/0, c_seq/0, c_openmp/0, c_openmp/1)는 모두 펜스 1개·시그니처 OK 로 추출돼야 한다."""

    def test_all_dummies_extract(self) -> None:
        seen = 0
        for lang, backend in (("numpy", "python"), ("c_seq", "c"), ("c_openmp", "c")):
            for md in sorted((DUMMY_DIR / lang).glob("*.md")):
                res = extract(md.read_text(encoding="utf-8"), backend=backend, signature_spec=SIGNATURES)
                with self.subTest(lang=lang, k=md.stem):
                    self.assertEqual(res.n_blocks, 1)
                    self.assertTrue(res.ok, res.detail)
                    self.assertTrue(res.signature_ok)
                    self.assertEqual(res.detail, "")
                seen += 1
        self.assertEqual(seen, 4)


class WorkerTestBase(unittest.TestCase):
    """단위 테스트용 npz 를 임시 디렉터리에 만들고 worker check 를 돌리는 공통 부분."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls._tmp.name)
        cls.tests_dir = cls.tmp / "tests"
        generate("floyd_warshall", DEFAULT_SEED, cls.tests_dir, UNIT_SHAPES, _gen_tests_module().make_case, REF_METHOD)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def check(self, backend: str, artifact: Path, names: tuple[str, ...], threads: int) -> dict:
        tests = [self.tests_dir / f"{n}.npz" for n in names]
        return run_worker(backend, artifact, "check", tests, self.tmp / f"check_{artifact.stem}_{'_'.join(names)}.json", threads=threads)


class PythonKernelsTest(WorkerTestBase):
    def test_numpy_dummy_passes_check(self) -> None:
        kernel = self.tmp / "numpy0.py"
        kernel.write_text(_dummy_code("numpy", 0, "python"), encoding="utf-8")
        doc = self.check("python", kernel, ("tiny", "rect"), 4)
        self.assertEqual(doc["classification"], "pass", doc.get("error_detail"))
        self.assertEqual(doc["task"], "floyd_warshall")
        self.assertEqual([t["name"] for t in doc["tests"]], ["tiny", "rect"])
        self.assertTrue(all(t["classification"] == "pass" for t in doc["tests"]))
        self.assertFalse(doc["inputs_mutated"])
        self.assertFalse(doc["oob_write"])
        self.assertFalse(doc["nondeterministic"])
        self.assertLess(doc["max_tol_ratio"], 0.2)  # float32 덧셈 ≤ 37회: 허용오차의 20 % 이내

    def test_copy_only_kernel_is_wrong_output(self) -> None:
        """k 루프를 건너뛴 커널. 기본 시드의 tiny(n=8, 간선 4개: 1→0, 1→2, 3→2, 4→2)에는 2홉 경로가 하나도 없어 참조가 입력과 같으므로
        tiny 는 통과한다(tiny 는 복사·INF 보존·대각만 검사하는 셈). rect(n=37) 에서 1369 원소 중 1081 개가 완화로 줄어들어 잡힌다."""
        kernel = self.tmp / "copy_only.py"
        kernel.write_text(PY_COPY_ONLY, encoding="utf-8")
        doc = self.check("python", kernel, ("tiny", "rect"), 4)
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "mismatch"))
        by_name = {t["name"]: t for t in doc["tests"]}
        self.assertEqual((by_name["tiny"]["classification"], by_name["tiny"]["pattern"]), ("wrong_output", "mismatch"))  # n=8 tiny 에는 2홉 경로가 있어 복사만 하는 커널이 tiny 에서도 걸린다
        self.assertEqual((by_name["rect"]["classification"], by_name["rect"]["pattern"]), ("wrong_output", "mismatch"))
        self.assertFalse(any(t["nondeterministic"] for t in doc["tests"]))
        self.assertGreater(doc["max_tol_ratio"], TOL.numeric_pattern_tol_ratio)  # 틀린 원소는 허용오차의 수천 배 → imprecise 아님
        self.assertFalse(doc["oob_write"])
        self.assertFalse(doc["inputs_mutated"])


@unittest.skipIf(shutil.which("gcc") is None, "gcc 없음")
class CKernelsTest(WorkerTestBase):
    """C 기준선과 C 더미를 config 의 실제 플래그로 빌드해 check 한다."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.bcfg = load_config().build

    def build(self, code: str, name: str, *, openmp: bool) -> Path:
        out_dir = self.tmp / name
        out_dir.mkdir(exist_ok=True)
        res = build_c(code, out_dir, self.bcfg, openmp=openmp, launch_prefix=[], symbol="floyd_warshall")
        self.assertTrue(res.ok, f"{name}: {res.error_kind}: {res.error_detail}\n{res.log[-1500:]}")
        self.assertEqual(res.forbidden_symbols, [])
        assert res.artifact is not None
        return res.artifact

    def test_baseline_and_correct_dummies_pass(self) -> None:
        cases = [
            ("baseline", (TASK_DIR / "baseline" / "baseline.c").read_text(encoding="utf-8"), True, 4),
            ("c_seq0", _dummy_code("c_seq", 0, "c"), False, 1),
            ("c_openmp0", _dummy_code("c_openmp", 0, "c"), True, 4),
        ]
        for name, code, openmp, threads in cases:
            with self.subTest(kernel=name):
                so = self.build(code, name, openmp=openmp)
                doc = self.check("c", so, ("tiny", "rect", "small"), threads)
                self.assertEqual(doc["classification"], "pass", f"{name}: {doc.get('error_detail')}")
                self.assertFalse(doc["oob_write"])
                self.assertFalse(doc["inputs_mutated"])
                self.assertFalse(doc["nondeterministic"])
                self.assertLess(doc["max_tol_ratio"], 0.2)

    def test_c_seq_dummy_has_no_thread_symbols(self) -> None:
        out_dir = self.tmp / "cseq_syms"
        out_dir.mkdir(exist_ok=True)
        res = build_c(_dummy_code("c_seq", 0, "c"), out_dir, self.bcfg, openmp=False, launch_prefix=[], symbol="floyd_warshall")
        self.assertTrue(res.ok, res.error_detail)
        self.assertEqual(res.thread_symbols, [])

    def test_k_parallel_kernel_is_not_pass(self) -> None:
        """c_openmp/1: k 루프를 `parallel for` 로 나눠 단계 k 의 의존(단계 k−1 의 결과)을 깬다. 스레드가 서로 다른 k 로 같은 행렬을
        동시에 갱신해 결과가 호출마다 달라진다 — 실측 3/3 race_condition(nondeterministic:rect,small; tiny 는 n=5 라 우연히 맞음).
        두 호출이 같게 틀리면 wrong_output (둘 다 허용; 어느 쪽인지는 타이밍 의존)."""
        so = self.build(_dummy_code("c_openmp", 1, "c"), "c_openmp1", openmp=True)
        doc = self.check("c", so, ("tiny", "rect", "small"), 4)
        self.assertIn(doc["classification"], ("race_condition", "wrong_output"), doc.get("error_detail"))
        self.assertTrue(any(t["classification"] != "pass" for t in doc["tests"]))
        self.assertFalse(doc["oob_write"])
        self.assertFalse(doc["inputs_mutated"])


if __name__ == "__main__":
    unittest.main()
