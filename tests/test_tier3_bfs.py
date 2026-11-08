"""Tier 3 ``bfs`` 과제 테스트(설계 §18).

- gen_tests 의 결정성(같은 시드 → 바이트 동일), CSR 불변식, tiny 참조와 순수 Python BFS 의 대조, 손으로 만든 그래프
  (도달 불가 정점·자기 루프 포함)의 참조, 체크인된 manifest 와 생성기의 일치, rect·small 의 도달 불가 정점 수 보고.
- 더미 응답 4개의 추출(펜스 1개·시그니처 OK).
- python 정답 더미(numpy/0)의 worker check(tiny·rect) pass — 테스트가 임시 디렉터리에 만든 npz 를 쓴다.
- 심은 버그: 도달 불가 정점에 −1 대신 0 을 쓰는 커널은 ``wrong_output``/``mismatch``(정수 출력은 허용오차 0).
- C 기준선·C 더미를 실제 빌드 플래그로 컴파일해 worker check: 정답은 pass, 원자 연산 없는 c_openmp/1 은 pass 가 아니다
  (``race_condition`` 또는 ``wrong_output`` — 타이밍 의존; harness/dummy_responses/bfs/README.md).

LLM 코드를 흉내 낸 커널은 모두 ``harness.worker`` 서브프로세스에서 실행한다(테스트 프로세스 안에서 실행하지 않는다).
gcc 가 없으면 C 테스트를 건너뛴다.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import ModuleType

import numpy as np
import yaml

from harness.build.c import build_c
from harness.classify import Tolerance, classify_task, load_tolerance
from harness.config import load_config
from harness.extract import extract
from harness.paths import PYTHON_EXE, ROOT
from harness.procutil import run_isolated
from harness.protocol import launch_env_minimal
from harness.taskspec import load_taskspec
from harness.testgen import DEFAULT_SEED, TEST_NAMES, generate, write_npz_atomic

TASK_DIR = ROOT / "tasks" / "bfs"
TESTS_DIR = TASK_DIR / "tests"
DUMMY_DIR = ROOT / "harness" / "dummy_responses" / "bfs"
SIGNATURES: dict = yaml.safe_load((TASK_DIR / "signatures.yaml").read_text(encoding="utf-8"))
EXACT = Tolerance(rtol=0.0, atol=0.0, numeric_pattern_tol_ratio=0.0, sentinel=-2048.0)
REF_METHOD = "numpy level-synchronous BFS (test)"

# 시드 20261105 의 tiny(n = 7, max_degree = 2): 도달 불가 정점 2개(1, 6), 최대 레벨 2. 생성기 코드·시드를 바꾸면 이 값과
# spec.md 의 (7, 13), README(harness/dummy_responses/bfs)를 함께 갱신한다.
TINY_DIST = [0, -1, 1, 2, 1, 2, -1]

# 손으로 만든 그래프(n = 6): 0→1, 0→2, 1→3, 2→2(자기 루프), 2→3, 3→0(사이클), 4→5; 4·5 는 도달 불가.
HAND_ROW_PTR = np.array([0, 2, 3, 5, 6, 7, 7], dtype=np.int32)
HAND_COL_IDX = np.array([1, 2, 3, 2, 3, 0, 5], dtype=np.int32)
HAND_DIST = [0, 1, 1, 2, -1, -1]

# 심은 버그: BFS 자체는 맞지만 도달 불가 정점을 −1 이 아닌 0 으로 둔다(dist 를 0 으로 초기화) → wrong_output/mismatch.
PY_ZERO_FOR_UNREACHABLE = """import numpy as np

def bfs(row_ptr, col_idx, dist, n, nnz):
    dist[...] = 0
    seen = np.zeros(n, dtype=bool)
    seen[0] = True
    frontier = np.zeros(1, dtype=np.int64)
    level = 0
    rp = row_ptr.astype(np.int64)
    while frontier.size:
        start = rp[frontier]
        count = rp[frontier + 1] - start
        total = int(count.sum())
        if total == 0:
            break
        shift = np.repeat(start - (np.cumsum(count) - count), count)
        targets = col_idx[np.arange(total, dtype=np.int64) + shift]
        fresh = np.unique(targets[~seen[targets]])
        if fresh.size == 0:
            break
        level += 1
        seen[fresh] = True
        dist[fresh] = level
        frontier = fresh.astype(np.int64)
"""


def _gen_tests_module() -> ModuleType:
    """``tasks/bfs/gen_tests.py`` 를 모듈로 로드한다(실행하지 않고 make_case/reference 만 쓴다)."""
    spec = importlib.util.spec_from_file_location("bfs_gen_tests", TASK_DIR / "gen_tests.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dummy_code(lang: str, k: int, backend: str) -> str:
    res = extract((DUMMY_DIR / lang / f"{k}.md").read_text(encoding="utf-8"), backend=backend, signature_spec=SIGNATURES)
    assert res.ok and res.code is not None, f"{lang}/{k}: {res.detail}"
    return res.code


def _bfs_python(row_ptr: np.ndarray, col_idx: np.ndarray, n: int) -> list[int]:
    """순수 Python 큐 BFS(참조의 대조용)."""
    dist = [-1] * n
    dist[0] = 0
    queue = deque([0])
    while queue:
        u = queue.popleft()
        for p in range(int(row_ptr[u]), int(row_ptr[u + 1])):
            v = int(col_idx[p])
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def write_tests(out_dir: Path, names: tuple[str, ...]) -> list[Path]:
    """생성기와 같은 코드·시드(``DEFAULT_SEED + index``)로 요청한 형태의 npz 를 만든다(large 160 MB 는 피한다)."""
    mod = _gen_tests_module()
    paths: list[Path] = []
    for index, (name, shape) in enumerate(mod.SHAPES):
        if name not in names:
            continue
        dims, inputs, out_name, ref = mod.make_case(shape, np.random.default_rng(DEFAULT_SEED + index))
        path = out_dir / f"{name}.npz"
        write_npz_atomic(path, **inputs, **{f"{out_name}_ref": ref}, **{k: np.int64(v) for k, v in dims.items()},
                         seed=np.int64(DEFAULT_SEED + index))
        paths.append(path)
    return paths


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
        self.assertEqual((spec.task, spec.function, spec.params), ("bfs", "bfs", ["row_ptr", "col_idx", "dist", "n", "nnz"]))
        self.assertEqual([a.dtype for a in spec.inputs], ["int32", "int32"])
        self.assertEqual((spec.output.name, spec.output.dtype), ("dist", "int32"))
        dims = {"n": 7, "nnz": 13}
        self.assertEqual(spec.shape_of(spec.inputs[0], dims), (8,))
        self.assertEqual(spec.shape_of(spec.inputs[1], dims), (13,))
        self.assertEqual(spec.shape_of(spec.output, dims), (7,))
        self.assertIsNone(spec.work)
        self.assertIsNone(spec.work_of(dims))
        self.assertEqual((spec.baseline["backend"], spec.baseline["file"], spec.baseline["openmp"]),
                         ("c", "baseline/baseline.c", True))
        self.assertTrue((TASK_DIR / spec.baseline["file"]).is_file())
        tol = load_tolerance(TASK_DIR / "tolerance.yaml")
        self.assertEqual((tol.rtol, tol.atol, tol.numeric_pattern_tol_ratio, tol.sentinel), (0.0, 0.0, 0.0, -2048.0))
        self.assertEqual((tol.task, tol.dtype), ("bfs", "int32"))
        for lang in ("numpy", "c_seq", "c_openmp"):
            entry = SIGNATURES["languages"][lang]
            self.assertIn("bfs(", entry["positional"])
            self.assertTrue(entry["param_docs"].strip())
        self.assertEqual(SIGNATURES["languages"]["numpy"]["keyword"], "def bfs(*, row_ptr, col_idx, dist, n, nnz):")
        self.assertEqual(SIGNATURES["languages"]["c_seq"]["positional"],
                         "void bfs(const int* row_ptr, const int* col_idx, int* dist, int n, int nnz)")

    def test_spec_structure_is_language_neutral(self) -> None:
        spec_md = (TASK_DIR / "spec.md").read_text(encoding="utf-8")
        for heading in ("## Definition", "## Data layout", "## Contract", "## Accuracy", "## Evaluation", "## Goal"):
            self.assertIn(heading, spec_md)
        for shape in ("(7, 13)", "(1,003, 15,833)", "(65,536, 1,045,504)", "(2,097,152, 33,541,983)"):
            self.assertIn(shape, spec_md)
        for word in ("numpy", "NumPy", "OpenMP", "BLAS", "#pragma", "deque", "scipy", "networkx"):
            self.assertNotIn(word, spec_md)
        for phrase in ("-1", "self loop", "8 GiB", "exactly five parameters"):
            self.assertIn(phrase, spec_md)
        self.assertTrue((TASK_DIR / "spec.ko.md").read_text(encoding="utf-8").startswith("# 과제:"))


class GenTestsTest(unittest.TestCase):
    """생성기: 손계산 참조, CSR 불변식, 결정성, 체크인된 manifest 와의 일치, 도달 불가 정점 수."""

    SMALL_SHAPES = [("tiny", {"n": 7, "max_degree": 2}), ("rect", {"n": 1003}), ("small", {"n": 4096}), ("large", {"n": 16384})]

    def setUp(self) -> None:
        self.gen = _gen_tests_module()

    def test_shapes(self) -> None:
        self.assertEqual([n for n, _ in self.gen.SHAPES], list(TEST_NAMES))
        self.assertEqual([s["n"] for _, s in self.gen.SHAPES], [7, 1003, 1 << 16, 1 << 21])
        self.assertEqual(self.gen.SHAPES[0][1]["max_degree"], 2)
        self.assertEqual(self.gen.MAX_DEGREE, 32)

    def test_reference_hand_graph(self) -> None:
        ref = self.gen.reference(HAND_ROW_PTR, HAND_COL_IDX, 6)
        self.assertEqual(ref.dtype, np.float64)
        self.assertEqual(ref.tolist(), [float(d) for d in HAND_DIST])
        self.assertEqual(_bfs_python(HAND_ROW_PTR, HAND_COL_IDX, 6), HAND_DIST)
        # 간선 없는 그래프·n = 1·정점 0 에 바깥 간선 없음
        self.assertEqual(self.gen.reference(np.zeros(2, dtype=np.int32), np.zeros(0, dtype=np.int32), 1).tolist(), [0.0])
        self.assertEqual(self.gen.reference(np.array([0, 0, 1], dtype=np.int32), np.array([0], dtype=np.int32), 2).tolist(), [0.0, -1.0])

    def test_tiny_reference_equals_pure_python_bfs(self) -> None:
        name, shape = self.gen.SHAPES[0]
        self.assertEqual(name, "tiny")
        dims, inputs, out_name, ref = self.gen.make_case(shape, np.random.default_rng(DEFAULT_SEED + 0))
        self.assertEqual((dims, out_name), ({"n": 7, "nnz": 13}, "dist"))
        self.assertEqual(list(inputs), ["row_ptr", "col_idx"])
        rp, ci = inputs["row_ptr"], inputs["col_idx"]
        self.assertEqual((rp.dtype, ci.dtype, rp.shape, ci.shape), (np.int32, np.int32, (8,), (13,)))
        self.assertEqual(ref.tolist(), [float(d) for d in _bfs_python(rp, ci, 7)])
        self.assertEqual(ref.tolist(), [float(d) for d in TINY_DIST])
        self.assertEqual(int((ref == -1).sum()), 2)

    def _check_csr(self, rp: np.ndarray, ci: np.ndarray, n: int, nnz: int, max_degree: int) -> None:
        self.assertEqual((rp.dtype, ci.dtype), (np.int32, np.int32))
        self.assertEqual((rp.shape, ci.shape), ((n + 1,), (nnz,)))
        self.assertEqual((int(rp[0]), int(rp[-1])), (0, nnz))
        lens = np.diff(rp.astype(np.int64))
        self.assertTrue(bool((lens >= 0).all()))
        self.assertLessEqual(int(lens.max()), min(max_degree, n))
        self.assertTrue(bool(((ci >= 0) & (ci < n)).all()), "열 인덱스가 [0, n) 을 벗어난다")
        row_of = np.repeat(np.arange(n), lens)
        same_row = row_of[1:] == row_of[:-1]
        self.assertTrue(bool((ci[1:][same_row] > ci[:-1][same_row]).all()), "행 안에서 엄격 증가가 아니다")

    def test_rect_reference_and_invariants(self) -> None:
        dims, inputs, _, ref = self.gen.make_case({"n": 1003}, np.random.default_rng(DEFAULT_SEED + 1))
        rp, ci = inputs["row_ptr"], inputs["col_idx"]
        self._check_csr(rp, ci, 1003, dims["nnz"], 32)
        self.assertEqual(ref.tolist(), [float(d) for d in _bfs_python(rp, ci, 1003)])
        self.assertEqual(ref[0], 0.0)

    def test_generate_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = Path(tmp) / "a", Path(tmp) / "b", Path(tmp) / "c"
            ma = generate("bfs", DEFAULT_SEED, a, self.SMALL_SHAPES, self.gen.make_case, REF_METHOD)
            mb = generate("bfs", DEFAULT_SEED, b, self.SMALL_SHAPES, self.gen.make_case, REF_METHOD)
            self.assertEqual(ma["test_hash"], mb["test_hash"])
            self.assertEqual(ma["files"], mb["files"])
            for name, shape in self.SMALL_SHAPES:
                self.assertEqual((a / f"{name}.npz").read_bytes(), (b / f"{name}.npz").read_bytes(), name)
                with np.load(a / f"{name}.npz") as npz:
                    self.assertEqual(set(npz.files), {"row_ptr", "col_idx", "dist_ref", "n", "nnz", "seed"})
                    n, nnz = int(npz["n"]), int(npz["nnz"])
                    self.assertEqual(n, shape["n"])
                    self._check_csr(npz["row_ptr"], npz["col_idx"], n, nnz, shape.get("max_degree", 32))
                    ref = npz["dist_ref"]
                    self.assertEqual((ref.dtype, ref.shape, ref[0]), (np.float64, (n,), 0.0))
                    self.assertTrue(bool((ref >= -1).all()) and bool((ref == np.round(ref)).all()))
                    self.assertLess(float(ref.max()), n)
            mc = generate("bfs", DEFAULT_SEED + 1, c, self.SMALL_SHAPES, self.gen.make_case, REF_METHOD)
            self.assertNotEqual(ma["test_hash"], mc["test_hash"])

    def test_checked_in_tests_match_generator(self) -> None:
        manifest_path = TESTS_DIR / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("tasks/bfs/tests 없음 (tasks/bfs/gen_tests.py)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shapes = dict(self.gen.SHAPES)
        for index, name in enumerate(("tiny", "rect")):
            dims, inputs, _, ref = self.gen.make_case(shapes[name], np.random.default_rng(int(manifest["seed"]) + index))
            self.assertEqual((manifest["files"][name]["n"], manifest["files"][name]["nnz"]), (dims["n"], dims["nnz"]))
            h = hashlib.sha256()
            for arr in inputs.values():
                h.update(np.ascontiguousarray(arr).tobytes())
            self.assertEqual(manifest["files"][name]["sha256_inputs"], h.hexdigest())
            with np.load(TESTS_DIR / f"{name}.npz") as npz:
                self.assertTrue(np.array_equal(npz["row_ptr"], inputs["row_ptr"]), name)
                self.assertTrue(np.array_equal(npz["col_idx"], inputs["col_idx"]), name)
                self.assertTrue(np.array_equal(npz["dist_ref"], ref), name)

    def test_report_unreachable_vertices_in_rect_and_small(self) -> None:
        """rect·small 에 도달 불가 정점이 몇 개인지 보고한다(평균 차수 16 이면 기대값 n·e^-16 ≈ 0 이라 없을 수 있다 —
        없으면 그렇다고 적는다). 도달 불가 처리(-1)는 tiny 와 테스트의 손그래프가 직접 검사한다."""
        shapes = dict(self.gen.SHAPES)
        report = []
        for index, name in ((1, "rect"), (2, "small")):
            _, _, _, ref = self.gen.make_case(shapes[name], np.random.default_rng(DEFAULT_SEED + index))
            k = int((ref == -1).sum())
            report.append(f"{name}: {k} unreachable" if k else f"{name}: no unreachable vertex (all {ref.size} reached)")
            self.assertGreaterEqual(k, 0)
        print(f"[test_tier3_bfs] {'; '.join(report)}", file=sys.stderr)


class ClassifyTest(unittest.TestCase):
    """정수 출력의 정확 일치 판정(설계 §10 규칙을 bfs 출력에 적용)."""

    def test_exact_equality_rules(self) -> None:
        ref = np.array(HAND_DIST, dtype=np.float64)
        ok = ref.astype(np.int32)
        inputs, dims = {"row_ptr": HAND_ROW_PTR, "col_idx": HAND_COL_IDX}, {"n": 6, "nnz": 7}
        self.assertEqual(classify_task("bfs", ok, ref, inputs, dims, EXACT)[0], "pass")
        zero_unreachable = np.where(ok < 0, 0, ok).astype(np.int32)
        st, pat, m = classify_task("bfs", zero_unreachable, ref, inputs, dims, EXACT)
        self.assertEqual((st, pat), ("wrong_output", "mismatch"))
        self.assertTrue(np.isinf(m["max_tol_ratio"]))
        self.assertEqual(m["n_fail"], 2)
        off_by_one = ok.copy()
        off_by_one[3] += 1
        self.assertEqual(classify_task("bfs", off_by_one, ref, inputs, dims, EXACT)[:2], ("wrong_output", "mismatch"))
        unwritten = np.full(6, -2048, dtype=np.int32)
        self.assertEqual(classify_task("bfs", unwritten, ref, inputs, dims, EXACT)[:2], ("wrong_output", "output_not_written"))
        partial = ok.copy()
        partial[4:] = -2048   # 도달 불가 정점을 쓰지 않음
        self.assertEqual(classify_task("bfs", partial, ref, inputs, dims, EXACT)[:2], ("wrong_output", "partial_write:2/6"))


class DummyResponsesTest(unittest.TestCase):
    """더미 응답 4개(numpy/0, c_seq/0, c_openmp/0, c_openmp/1)는 모두 펜스 1개·시그니처 OK 로 추출돼야 한다."""

    def test_all_dummies_extract(self) -> None:
        expected = {("numpy", 0), ("c_seq", 0), ("c_openmp", 0), ("c_openmp", 1)}
        seen: set[tuple[str, int]] = set()
        for lang, backend in (("numpy", "python"), ("c_seq", "c"), ("c_openmp", "c")):
            for md in sorted((DUMMY_DIR / lang).glob("*.md")):
                seen.add((lang, int(md.stem)))
                res = extract(md.read_text(encoding="utf-8"), backend=backend, signature_spec=SIGNATURES)
                with self.subTest(lang=lang, k=md.stem):
                    self.assertEqual(res.n_blocks, 1)
                    self.assertTrue(res.ok, res.detail)
                    self.assertTrue(res.signature_ok)
                    self.assertTrue(res.signature_exact, res.detail)
                    self.assertEqual(res.detail, "")
        self.assertEqual(seen, expected)


class WorkerTestBase(unittest.TestCase):
    """임시 디렉터리에 tiny·rect·small npz 를 한 번 만들고 worker check 를 돌리는 공통 부분."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="bfs_tests_")
        cls.npz_dir = Path(cls._tmp.name)
        cls.npz = {p.stem: p for p in write_tests(cls.npz_dir, ("tiny", "rect", "small"))}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def check(self, backend: str, artifact: Path, names: tuple[str, ...], threads: int, tmp: Path) -> dict:
        tests = [self.npz[n] for n in names]
        return run_worker(backend, artifact, "check", tests, tmp / f"check_{'_'.join(names)}.json", threads=threads)


class PythonKernelsTest(WorkerTestBase):
    def test_numpy_dummy_passes_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kernel = Path(tmp) / "kernel.py"
            kernel.write_text(_dummy_code("numpy", 0, "python"), encoding="utf-8")
            doc = self.check("python", kernel, ("tiny", "rect"), 4, Path(tmp))
        self.assertEqual(doc["classification"], "pass", doc.get("error_detail"))
        self.assertEqual(doc["task"], "bfs")
        self.assertEqual([t["name"] for t in doc["tests"]], ["tiny", "rect"])
        for t in doc["tests"]:
            self.assertEqual(t["classification"], "pass", t)
            self.assertEqual(len(t["runs"]), 2)
            for run in t["runs"]:
                self.assertEqual((run["n_unwritten"], run["max_abs_err"]), (0, 0.0))
        self.assertFalse(doc["inputs_mutated"])
        self.assertFalse(doc["oob_write"])
        self.assertFalse(doc["nondeterministic"])

    def test_zero_for_unreachable_is_wrong_output(self) -> None:
        """도달 불가 정점에 0 을 쓴 커널: tiny 에 도달 불가 정점이 2개 있어 wrong_output/mismatch(원소 2개 틀림);
        rect 는 모두 도달 가능해 pass 다 — 샘플 합산은 wrong_output."""
        with tempfile.TemporaryDirectory() as tmp:
            kernel = Path(tmp) / "kernel.py"
            kernel.write_text(PY_ZERO_FOR_UNREACHABLE, encoding="utf-8")
            doc = self.check("python", kernel, ("tiny", "rect"), 4, Path(tmp))
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "mismatch"))
        by_name = {t["name"]: t for t in doc["tests"]}
        self.assertEqual((by_name["tiny"]["classification"], by_name["tiny"]["pattern"]), ("wrong_output", "mismatch"))
        self.assertEqual(by_name["tiny"]["runs"][0]["max_abs_err"], 1.0)   # 0 대신 −1 → 차이 1
        self.assertFalse(by_name["tiny"]["nondeterministic"])
        self.assertIsNone(doc["max_tol_ratio"])  # 허용오차 0: 배율 inf 는 JSON 에 null; numeric_instability 불가
        self.assertFalse(doc["oob_write"])


@unittest.skipIf(shutil.which("gcc") is None, "gcc 없음")
class CKernelsTest(WorkerTestBase):
    """C 기준선과 C 더미를 config 의 실제 플래그로 빌드해 check 한다."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.bcfg = load_config().build

    def build(self, code: str, out_dir: Path, *, openmp: bool) -> Path:
        res = build_c(code, out_dir, self.bcfg, openmp=openmp, launch_prefix=[], symbol="bfs")
        self.assertTrue(res.ok, f"{res.error_kind}: {res.error_detail}\n{res.log[-1500:]}")
        self.assertEqual(res.forbidden_symbols, [])
        assert res.artifact is not None
        return res.artifact

    def test_baseline_and_correct_dummies_pass(self) -> None:
        cases = [
            ("baseline", (TASK_DIR / "baseline" / "baseline.c").read_text(encoding="utf-8"), True, 4),
            ("c_seq/0", _dummy_code("c_seq", 0, "c"), False, 1),
            ("c_openmp/0", _dummy_code("c_openmp", 0, "c"), True, 4),
        ]
        for name, code, openmp, threads in cases:
            with self.subTest(kernel=name), tempfile.TemporaryDirectory() as tmp:
                so = self.build(code, Path(tmp), openmp=openmp)
                doc = self.check("c", so, ("tiny", "rect", "small"), threads, Path(tmp))
                self.assertEqual(doc["classification"], "pass", f"{name}: {doc.get('error_detail')}")
                self.assertFalse(doc["oob_write"])
                self.assertFalse(doc["inputs_mutated"])
                self.assertFalse(doc["nondeterministic"])
                self.assertEqual(doc["max_abs_err"], 0.0)

    def test_c_seq_dummy_has_no_thread_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = build_c(_dummy_code("c_seq", 0, "c"), Path(tmp), self.bcfg, openmp=False, launch_prefix=[], symbol="bfs")
        self.assertTrue(res.ok, res.error_detail)
        self.assertEqual(res.thread_symbols, [])

    def test_non_atomic_parallel_frontier_is_not_pass(self) -> None:
        """c_openmp/1: 방문 표시도 다음-프론티어 push(`next[next_size++]`)도 원자적이지 않다. 갱신 유실로 프론티어에서
        정점이 빠져 그 뒤 정점들이 −1 로 남는다 — 호출마다 다르면 race_condition, 같게 틀리면 wrong_output(둘 다 허용)."""
        with tempfile.TemporaryDirectory() as tmp:
            so = self.build(_dummy_code("c_openmp", 1, "c"), Path(tmp), openmp=True)
            doc = self.check("c", so, ("tiny", "rect", "small"), 4, Path(tmp))
        self.assertIn(doc["classification"], ("race_condition", "wrong_output"), doc.get("error_detail"))
        self.assertTrue(any(t["classification"] != "pass" for t in doc["tests"]))


if __name__ == "__main__":
    unittest.main()
