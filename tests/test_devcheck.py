"""tools/devcheck.py 테스트 (설계 v3 §14): 언어 → 백엔드 결정, 워커 인자, 종료 코드 규칙, 결과 해석(순수 함수)과
임시 과제 디렉터리에서의 끝-끝 실행(kernel / baseline / dummies / gen; 실제 워커를 띄운다).

끝-끝 테스트는 저장소의 tasks/ 를 건드리지 않는다: ``--tasks-dir``·``--dummy-dir`` 로 임시 디렉터리의 int32 과제(작은
히스토그램)를 쓴다. C 커널 테스트는 이 머신에서 고를 수 있는 툴체인(Linux 의 gcc 또는 ziglang)이 없으면 건너뛴다.

``dummies`` 의 기본 언어 목록에는 설정의 ``languages`` 가 들어가므로, 기본 목록을 쓰는 테스트는 임시 설정(``--config``)을 준다
(저장소 설정에 다른 그룹이 언어를 더해도 결과가 바뀌지 않게). Linux 에서만 도는 빌드 경로(gcc·numba·halide·rust 어댑터)는
대역 모듈로 호출 규약(설계 v3 §6)을 확인한다."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from harness import sigs, testgen
from harness.paths import ROOT
from harness.procutil import ProcResult
from harness.taskspec import parse_taskspec
from tests._support import requires_modules


def _load_devcheck():
    """``tools/`` 는 패키지가 아니므로 파일 경로로 모듈을 올린다."""
    spec = importlib.util.spec_from_file_location("devcheck", ROOT / "tools" / "devcheck.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["devcheck"] = module
    spec.loader.exec_module(module)
    return module


devcheck = _load_devcheck()

SIGNATURES_YAML = """task: histogram
function: histogram
params: [keys, hist, n, n_bins]
args:
  - {name: keys, kind: in,  dtype: int32, shape: [n], doc: "each value is in the range 0 <= keys[i] < n_bins"}
  - {name: hist, kind: out, dtype: int32, shape: [n_bins]}
  - {name: n, kind: dim}
  - {name: n_bins, kind: dim}
baseline: {backend: python, file: baseline/baseline.py, openmp: false, gate_blas: false, gate_min_gflops: false}
unsupported:
  halide_auto: "data-dependent writes are not expressible as a Halide pipeline"
"""
TOLERANCE_YAML = "task: histogram\ndtype: int32\ncompare: exact\nsentinel: -2048\n"
# 끝-끝 테스트의 설정: 언어는 numpy 하나(나머지 언어는 devcheck 의 대응표로 풀린다)
CONFIG_YAML = ("languages:\n  numpy: {backend: python, prompt: p.md, threads: measurement}\n"
               "measurement: {threads: 2, check_repeats: 2}\n")
CONFIG_WITH_C_SEQ_YAML = ("languages:\n  numpy: {backend: python, prompt: p.md, threads: measurement}\n"
                          "  c_seq: {backend: c, prompt: p.md, threads: 1, openmp: false}\n"
                          "measurement: {threads: 2, check_repeats: 2}\n")
SHAPES = [("tiny", {"n": 10, "n_bins": 4}), ("rect", {"n": 1003, "n_bins": 37}),
          ("small", {"n": 4096, "n_bins": 16}), ("large", {"n": 20000, "n_bins": 64})]
GEN_TESTS_PY = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    import numpy as np
    from harness.testgen import main_for

    HERE = Path(__file__).resolve().parent
    SHAPES = [("tiny", {"n": 10, "n_bins": 4}), ("rect", {"n": 1003, "n_bins": 37}),
              ("small", {"n": 4096, "n_bins": 16}), ("large", {"n": 20000, "n_bins": 64})]

    def make_case(shape, rng):
        n, n_bins = shape["n"], shape["n_bins"]
        keys = rng.integers(0, n_bins - 1, size=n, dtype=np.int32)
        return {"n": n, "n_bins": n_bins}, {"keys": keys}, "hist", np.bincount(keys, minlength=n_bins).astype(np.int32)

    if __name__ == "__main__":
        sys.exit(main_for("histogram", SHAPES, make_case, "numpy bincount", HERE / "tests", out_dtype="int32", sentinel=-2048))
    """
)
BASELINE_PY = "import numpy as np\n\n\ndef histogram(keys, hist, n, n_bins):\n    hist[:] = np.bincount(keys, minlength=n_bins)\n"
MD_CORRECT = "Counts with a library call.\n\n```python\n" + BASELINE_PY + "```\n"
MD_ACCUMULATE = ("Adds the counts.\n\n```python\nimport numpy as np\n\n\ndef histogram(keys, hist, n, n_bins):\n"
                 "    hist += np.bincount(keys, minlength=n_bins).astype(np.int32)\n```\n")
MD_BAD_SIGNATURE = "```python\nimport numpy as np\n\n\ndef histogram(keys, n, n_bins):\n    return np.bincount(keys, minlength=n_bins)\n```\n"
# 실패 경로용 커널(파이썬 소스): 임포트에서 예외 / 워커 프로세스를 끝냄 / 돌아오지 않음 / 키워드 전용
KERNEL_IMPORT_ERROR = "raise RuntimeError('boom at import')\n\n\ndef histogram(keys, hist, n, n_bins):\n    pass\n"
KERNEL_EXITS = "import os\n\n\ndef histogram(keys, hist, n, n_bins):\n    os._exit(0)\n"
KERNEL_HANGS = "import time\n\n\ndef histogram(keys, hist, n, n_bins):\n    while True:\n        time.sleep(0.05)\n"
KERNEL_KEYWORD ="import numpy as np\n\n\ndef histogram(*, keys, hist, n, n_bins):\n    hist[:] = np.bincount(keys, minlength=n_bins)\n"
# halide_manual 파이프라인 빌드 함수(출력 인자 없음): 없는 Var 에 스케줄 / Func 가 아닌 반환값
HALIDE_BAD_SCHEDULE = ("import halide as hl\n\n\ndef histogram(keys, n, n_bins):\n    b = hl.Var('b')\n    hist = hl.Func('hist')\n"
                       "    hist[b] = hl.cast(hl.Int(32), 0)\n    hist.vectorize(hl.Var('zz'), 4)\n    return hist\n")
HALIDE_NOT_A_FUNC = "import halide as hl\n\n\ndef histogram(keys, n, n_bins):\n    return 3\n"
UTF8_BOM_BYTES = b"\xef\xbb\xbf"
# UTF-8 이 아닌 파일(cp949 로 저장한 한글 설명 + 맞는 코드)
MD_CP949_BYTES = ("한글 설명 문장.\n\n```python\n" + BASELINE_PY + "```\n").encode("cp949")
# 할당 함수·OpenMP 헤더·pragma 를 일부러 쓴다(zig 경로에서는 shim.h·omp.h 스텁을 지난다)
C_CORRECT = textwrap.dedent(
    """
    #include <stdlib.h>
    #include <string.h>
    #include <omp.h>

    void histogram(const int* keys, int* hist, int n, int n_bins)
    {
        int threads = omp_get_max_threads();
        int *local = calloc((size_t)threads * (size_t)n_bins, sizeof(int));
        int *scratch = aligned_alloc(64, 64 * sizeof(int));
        void *extra = NULL;
        if (!local || !scratch || posix_memalign(&extra, 64, 256) != 0) { free(local); free(scratch); return; }
        #pragma omp parallel
        {
            int *mine = local + (size_t)omp_get_thread_num() * (size_t)n_bins;
            #pragma omp for schedule(static)
            for (int i = 0; i < n; i++) mine[keys[i]]++;
        }
        memset(hist, 0, (size_t)n_bins * sizeof(int));
        for (int t = 0; t < threads; t++)
            for (int b = 0; b < n_bins; b++) hist[b] += local[(size_t)t * (size_t)n_bins + b];
        local = realloc(local, 16);
        free(extra);
        free(scratch);
        free(local);
    }
    """
)
MD_C_CORRECT = "Per-thread counts, then a merge.\n\n```c\n" + C_CORRECT.strip() + "\n```\n"
MD_C_SYNTAX_ERROR = "```c\nvoid histogram(const int* keys, int* hist, int n, int n_bins)\n{\n    int x = 1\n    hist[0] = x;\n}\n```\n"
C_KEYWORD = ('#include "kernel_args.h"\n\nvoid histogram(const histogram_args* args)\n{\n'
             "    for (int b = 0; b < args->n_bins; b++) args->hist[b] = 0;\n"
             "    for (int i = 0; i < args->n; i++) args->hist[args->keys[i]]++;\n}\n")
HISTOGRAM_SPEC = parse_taskspec({
    "task": "histogram", "function": "histogram",
    "args": [{"name": "keys", "kind": "in", "dtype": "int32", "shape": ["n"]},
             {"name": "hist", "kind": "out", "dtype": "int32", "shape": ["n_bins"]},
             {"name": "n", "kind": "dim"}, {"name": "n_bins", "kind": "dim"}],
})


def _proc(rc: int = 0, *, timed_out: bool = False, signal_name: str | None = None, stderr: str = "") -> ProcResult:
    return ProcResult(rc=rc, timed_out=timed_out, elapsed_s=0.1, stdout_tail="", stderr_tail=stderr, signal_name=signal_name)


def _histogram_case(shape, rng):
    n, n_bins = shape["n"], shape["n_bins"]
    keys = rng.integers(0, n_bins - 1, size=n, dtype=np.int32)
    return {"n": n, "n_bins": n_bins}, {"keys": keys}, "hist", np.bincount(keys, minlength=n_bins).astype(np.int32)


def shape_verdict_words(stdout: str) -> list[str]:
    """형태별 판정 줄(``<과제>/<언어> <형태> PASS|FAIL|NOT-RUN …``)의 판정 단어만. 최종 판정 줄(``=>``)과 표는 뺀다."""
    words: list[str] = []
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and "/" in parts[0] and parts[1] != "=>" and parts[2] in ("PASS", "FAIL", "NOT-RUN"):
            words.append(parts[2])
    return words


def table_rows(stdout: str, *tasks: str) -> list[str]:
    """``dummies`` 표의 행(과제 이름으로 시작하는 줄). 형태별 줄(``<과제>/<언어> …``)은 과제 이름 뒤가 ``/`` 라 빠진다."""
    return [ln for ln in stdout.splitlines() if ln.startswith(tuple(f"{t} " for t in tasks))]


def _write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8", newline="\n")
    return path


class SettingsAndLanguageTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="devcheck_cfg_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def settings_from(self, text: str):
        path = self.tmp / "experiment.yaml"
        path.write_text(text, encoding="utf-8", newline="\n")
        with contextlib.redirect_stderr(io.StringIO()):
            return devcheck.load_settings(path)

    def test_repository_config_defines_the_m1_languages(self) -> None:
        settings = devcheck.load_settings(ROOT / "config" / "experiment.yaml")
        for language, backend in (("numpy", "python"), ("c_seq", "c"), ("c_openmp", "c")):
            info = devcheck.resolve_language(settings, language)
            self.assertEqual((info.backend, info.source), (backend, "config"), language)
        self.assertEqual(devcheck.resolve_language(settings, "c_seq").threads, 1)
        self.assertFalse(devcheck.resolve_language(settings, "c_seq").openmp)
        self.assertTrue(devcheck.resolve_language(settings, "c_openmp").openmp)
        self.assertEqual(settings.correctness_tests, ["tiny", "rect", "small"])
        self.assertEqual(settings.performance_test, "large")

    def test_fallback_table_for_languages_missing_from_config(self) -> None:
        settings = self.settings_from("languages:\n  numpy: {backend: python, prompt: p.md, threads: measurement}\n"
                                      "measurement: {threads: 2, check_repeats: 3}\n")
        self.assertEqual(devcheck.FALLBACK_BACKENDS, {"numpy": "python", "numba": "numba", "halide_auto": "halide",
                                                      "halide_manual": "halide", "c_seq": "c", "c_openmp": "c", "rust": "rust"})
        expected = {"numba": ("numba", None), "halide_auto": ("halide", "auto"), "halide_manual": ("halide", "manual"),
                    "c_seq": ("c", None), "c_openmp": ("c", None), "rust": ("rust", None)}
        for language, (backend, mode) in expected.items():
            info = devcheck.resolve_language(settings, language)
            self.assertEqual((info.backend, info.halide_mode, info.source), (backend, mode, "fallback"), language)
        self.assertEqual(devcheck.resolve_language(settings, "numpy").source, "config")
        self.assertEqual(settings.check_repeats, 3)
        self.assertEqual(devcheck.resolve_language(settings, "c_seq").threads, 1)
        self.assertTrue(devcheck.resolve_language(settings, "c_openmp").openmp)
        capped = max(1, min(2, os.cpu_count() or 1))
        self.assertEqual(devcheck.resolve_language(settings, "numba").threads, capped)
        self.assertEqual(devcheck.resolve_language(settings, "numba").env, {"NUMBA_NUM_THREADS": str(capped)})
        self.assertEqual(devcheck.resolve_language(settings, "rust").env, {"RAYON_NUM_THREADS": str(capped)})

    def test_config_entry_wins_over_fallback(self) -> None:
        settings = self.settings_from(
            "languages:\n"
            "  numba: {backend: python, prompt: p.md, threads: 1}\n"
            "  halide_auto: {backend: halide, prompt: p.md, threads: 1, halide_mode: manual, env: {HL_NUM_THREADS: \"${threads}\", X: y}}\n")
        self.assertEqual(devcheck.resolve_language(settings, "numba").backend, "python")
        info = devcheck.resolve_language(settings, "halide_auto")
        self.assertEqual((info.backend, info.halide_mode, info.source), ("halide", "manual", "config"))
        self.assertEqual(info.env, {"HL_NUM_THREADS": "1", "X": "y"})

    def test_unknown_language_and_bad_entries_are_problems(self) -> None:
        settings = self.settings_from("languages:\n  weird: {backend: cobol, prompt: p.md, threads: 1}\n"
                                      "  mystery_halide: {backend: halide, prompt: p.md, threads: 1}\n"
                                      "  bad_threads: {backend: c, prompt: p.md, threads: lots}\n")
        for language in ("fortran", "weird", "mystery_halide", "bad_threads"):
            with self.assertRaises(devcheck.DevcheckProblem, msg=language):
                devcheck.resolve_language(settings, language)

    def test_missing_or_broken_config_falls_back_to_defaults(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            settings = devcheck.load_settings(self.tmp / "does_not_exist.yaml")
        self.assertIn("[devcheck]", err.getvalue())
        self.assertEqual((settings.threads, settings.check_repeats, settings.cc), (4, 2, "gcc"))
        self.assertEqual(devcheck.resolve_language(settings, "numpy").backend, "python")
        broken = self.settings_from("languages: [not, a, mapping\n")
        self.assertEqual(broken.languages, {})
        self.assertEqual(devcheck.resolve_language(broken, "rust").backend, "rust")

    def test_worker_backend_args(self) -> None:
        settings = self.settings_from("halide: {autoscheduler: Adams2019, autoscheduler_params: {beam_size: 8}}\n")
        Lang = devcheck.LanguageInfo
        self.assertEqual(devcheck.worker_backend_args(Lang("c_seq", "c"), settings), ["--backend", "c"])
        self.assertEqual(devcheck.worker_backend_args(Lang("rust", "rust"), settings), ["--backend", "c"])
        self.assertEqual(devcheck.worker_backend_args(Lang("numpy", "python"), settings), ["--backend", "python"])
        self.assertEqual(devcheck.worker_backend_args(Lang("numba", "numba", threads=4), settings), ["--backend", "numba", "--threads", "4"])
        self.assertEqual(devcheck.worker_backend_args(Lang("halide_manual", "halide", threads=2, halide_mode="manual"), settings),
                         ["--backend", "halide", "--halide-mode", "manual", "--threads", "2"])
        self.assertEqual(devcheck.worker_backend_args(Lang("halide_auto", "halide", threads=2, halide_mode="auto"), settings),
                         ["--backend", "halide", "--halide-mode", "auto", "--threads", "2", "--halide-autoscheduler", "Adams2019",
                          "--halide-params", '{"beam_size": 8}'])
        default = self.settings_from("{}\n")
        self.assertEqual(devcheck.worker_backend_args(Lang("halide_auto", "halide", threads=1, halide_mode="auto"), default)[-2:],
                         ["--halide-autoscheduler", "Mullapudi2016"])

    def test_worker_env_sets_pythonpath_threads_and_language_env(self) -> None:
        env = devcheck.worker_env(devcheck.LanguageInfo("numba", "numba", threads=3, env={"NUMBA_NUM_THREADS": "3"}))
        self.assertEqual(env["PYTHONPATH"], str(ROOT))
        self.assertEqual((env["OMP_NUM_THREADS"], env["OPENBLAS_NUM_THREADS"], env["MKL_NUM_THREADS"], env["NUMBA_NUM_THREADS"]),
                         ("3", "3", "3", "3"))
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")

    def test_default_dummy_lists_are_unions(self) -> None:
        """``dummies`` 의 기본 과제 = 과제 디렉터리 ∪ 더미 디렉터리, 기본 언어 = 0.md 가 있는 언어 ∪ 설정의 languages."""
        tasks_dir, dummy_dir = self.tmp / "tasks", self.tmp / "dummies"
        _write(tasks_dir / "alpha" / "signatures.yaml", "task: alpha\n")
        _write(tasks_dir / "no_dummies" / "signatures.yaml", "task: no_dummies\n")
        _write(tasks_dir / "not_a_task" / "README.md", "signatures.yaml 이 없는 디렉터리\n")
        _write(dummy_dir / "alpha" / "numpy" / "0.md", "x")
        _write(dummy_dir / "orphan" / "rust" / "0.md", "x")          # 과제 디렉터리에 없는 과제의 더미
        _write(dummy_dir / "alpha" / "c_seq" / "1.md", "x")           # 0.md 가 아니다
        _write(dummy_dir / ".cache" / "numpy" / "0.md", "x")          # 숨김·내부 디렉터리는 과제가 아니다
        _write(dummy_dir / "__pycache__" / "numpy" / "0.md", "x")
        _write(dummy_dir / "README.md", "파일은 과제가 아니다\n")
        tasks = devcheck.default_dummy_tasks(tasks_dir, dummy_dir)
        self.assertEqual(tasks, ["alpha", "no_dummies", "orphan"])
        settings = self.settings_from("languages:\n  c_seq: {backend: c, prompt: p.md, threads: 1}\n"
                                      "  numba: {backend: numba, prompt: p.md, threads: 1}\n")
        self.assertEqual(devcheck.default_dummy_languages(settings, dummy_dir, tasks), ["numpy", "numba", "c_seq", "rust"])
        # 고른 과제의 더미만 본다; 더미 디렉터리가 없어도 과제·설정의 언어는 나온다
        self.assertEqual(devcheck.default_dummy_languages(settings, dummy_dir, ["alpha"]), ["numpy", "numba", "c_seq"])
        self.assertEqual(devcheck.default_dummy_tasks(tasks_dir, self.tmp / "missing"), ["alpha", "no_dummies"])
        self.assertEqual(devcheck.default_dummy_languages(settings, self.tmp / "missing", ["alpha"]), ["numba", "c_seq"])
        self.assertEqual(devcheck.default_dummy_tasks(self.tmp / "missing", self.tmp / "missing"), [])


class VerdictRulesTest(unittest.TestCase):
    def test_exit_codes(self) -> None:
        O = devcheck.Outcome
        self.assertEqual((devcheck.EXIT_OK, devcheck.EXIT_FAIL, devcheck.EXIT_PROBLEM), (0, 1, 2))
        self.assertEqual(O("a", status="pass").exit_code(), 0)
        self.assertEqual(O("a", status="fail").exit_code(), 1)
        self.assertEqual(O("a", status="problem").exit_code(), 2)
        self.assertEqual(devcheck.combine_exit_codes([O("a", status="pass"), O("b", status="pass")]), 0)
        self.assertEqual(devcheck.combine_exit_codes([O("a", status="pass"), O("b", status="problem")]), 2)
        self.assertEqual(devcheck.combine_exit_codes([O("a", status="problem"), O("b", status="fail")]), 1)
        self.assertEqual(devcheck.combine_exit_codes([]), 0)

    def test_final_line(self) -> None:
        self.assertEqual(devcheck.Outcome("gemm/numpy", status="pass", stage="check", classification="pass").final_line(),
                         "gemm/numpy => PASS stage=check classification=pass")
        self.assertEqual(devcheck.Outcome("gemm/c_seq", status="fail", stage="build", classification="compile_error",
                                          detail="kernel.c:1:1: error: x", bench_ms=None).final_line(),
                         "gemm/c_seq => FAIL stage=build classification=compile_error detail=kernel.c:1:1: error: x")
        self.assertEqual(devcheck.Outcome("t/l", status="pass", stage="check", classification="pass", bench_ms=12.5).final_line(),
                         "t/l => PASS stage=check classification=pass bench_large_ms=12.500")
        self.assertTrue(devcheck.Outcome("t/l").final_line().startswith("t/l => PROBLEM"))

    def test_final_line_marks_the_serial_zig_build(self) -> None:
        """회귀: zig 직렬 빌드로 얻은 PASS 가 판정 줄에서 gcc+OpenMP 로 검증한 것과 구별되지 않았다."""
        self.assertEqual(devcheck.ZIG_TOOLCHAIN, "zig-serial")
        self.assertEqual(devcheck.Outcome("gemm/c_openmp", status="pass", stage="check", classification="pass",
                                          toolchain="zig-serial").final_line(),
                         "gemm/c_openmp => PASS stage=check classification=pass toolchain=zig-serial")
        # detail 은 공백을 포함해 줄 끝까지 가므로 toolchain 은 그 앞에 온다
        self.assertEqual(devcheck.Outcome("t/c_seq", status="fail", stage="build", classification="compile_error",
                                          detail="kernel.c:4:5: error: expected ';'", toolchain="zig-serial").final_line(),
                         "t/c_seq => FAIL stage=build classification=compile_error toolchain=zig-serial detail=kernel.c:4:5: error: expected ';'")

    def test_interpret_worker_rules(self) -> None:
        f = devcheck.interpret_worker
        complete = {"complete": True, "classification": "pass"}
        self.assertIsNone(f(_proc(0), complete, 180))
        status, cls, detail = f(_proc(3), {"complete": False, "worker_error": "Traceback ...\nValueError: boom"}, 180)
        self.assertEqual((status, cls), ("problem", "harness_error"))
        self.assertIn("ValueError: boom", detail)
        status, cls, detail = f(_proc(-9, timed_out=True, signal_name="SIGKILL"), {"complete": False, "in_progress": {"test": "rect", "call_idx": 1}}, 180)
        self.assertEqual((status, cls), ("fail", "timeout"))
        self.assertIn("after 180s during rect call 1", detail)
        # 결과 문서가 없다: 워커 인자 오류(아직 없는 백엔드) → 커널 탓이 아니다
        usage = "usage: python -m harness.worker [-h] ...\npython -m harness.worker: error: argument --backend: invalid choice: 'numba'"
        status, cls, detail = f(_proc(2, stderr=usage), None, 180)
        self.assertEqual((status, cls), ("problem", "worker_failed"))
        self.assertIn("invalid choice: 'numba'", detail)
        self.assertNotIn("usage:", detail)
        # 커널이 프로세스를 죽였다(부분 결과)
        partial = {"complete": False, "in_progress": {"test": "tiny", "call_idx": 0}}
        status, cls, detail = f(_proc(-11, signal_name="SIGSEGV"), partial, 180)
        self.assertEqual((status, cls, detail), ("fail", "runtime_crash", "worker died (signal SIGSEGV) during tiny call 0"))
        status, cls, detail = f(_proc(0xC0000005), partial, 180)
        self.assertEqual((status, cls), ("fail", "runtime_crash"))
        self.assertIn("0xC0000005 (access violation)", detail)
        self.assertEqual(f(_proc(0), {"complete": False}, 180)[:2], ("fail", "runtime_crash"))

    def test_shape_verdicts(self) -> None:
        doc = {"tests": [
            {"name": "tiny", "dims": {"n": 10, "n_bins": 4}, "classification": "pass", "pattern": "", "nondeterministic": False,
             "max_abs_err": 0.0, "max_tol_ratio": None, "runs": [{"elapsed_ms": 0.5, "n_unwritten": 0}, {"elapsed_ms": 0.25, "n_unwritten": 0}]},
            {"name": "rect", "dims": {"n": 1003, "n_bins": 37}, "classification": "wrong_output", "pattern": "accumulated_into_c",
             "nondeterministic": False, "max_abs_err": 2048.0, "max_tol_ratio": None, "runs": [{"elapsed_ms": 1.0, "n_unwritten": 2}]},
        ]}
        tiny, rect, small = devcheck.shape_verdicts_from_check(doc, ["tiny", "rect", "small"])
        self.assertEqual((tiny.name, tiny.verdict), ("tiny", "PASS"))
        self.assertEqual(tiny.info, "n=10 n_bins=4 max_abs_err=0 call_ms=0.25")
        self.assertEqual((rect.verdict, rect.info), ("FAIL", "n=1003 n_bins=37 wrong_output/accumulated_into_c max_abs_err=2.05e+03 n_unwritten=2 call_ms=1"))
        self.assertEqual((small.verdict, small.info), ("NOT-RUN", ""))
        # 판정은 pass 여도 실행 간 불일치면 FAIL
        nondet = {"tests": [{"name": "tiny", "dims": {}, "classification": "pass", "nondeterministic": True, "runs": []}]}
        self.assertEqual(devcheck.shape_verdicts_from_check(nondet, ["tiny"])[0].verdict, "FAIL")

    def test_zig_command_is_serial_and_uses_the_shim(self) -> None:
        cmd = devcheck.zig_command("kernel.dll")
        self.assertEqual(cmd[1:4], ["-m", "ziglang", "cc"])
        self.assertNotIn("-fopenmp", cmd)
        self.assertEqual(cmd[cmd.index("-include") + 1], "shim.h")
        self.assertEqual(Path(cmd[cmd.index("-I") + 1]), ROOT / "tools" / "devcheck_shim")
        self.assertIn("-shared", cmd)
        self.assertEqual(cmd[cmd.index("-o") + 1:cmd.index("-o") + 3], ["kernel.dll", "kernel.c"])
        shim = (ROOT / "tools" / "devcheck_shim" / "shim.h").read_text(encoding="utf-8")
        for needle in ("#define aligned_alloc devcheck_aligned_alloc", "#define free devcheck_free", "#define posix_memalign devcheck_posix_memalign",
                       "_aligned_malloc", "_aligned_free", "#if defined(_WIN32)"):
            self.assertIn(needle, shim)
        omp = (ROOT / "tools" / "devcheck_shim" / "omp.h").read_text(encoding="utf-8")
        for needle in ("#define DEVCHECK_OMP_THREADS 1", "omp_get_max_threads(void) { return DEVCHECK_OMP_THREADS; }",
                       "omp_get_num_threads(void) { return 1; }", "omp_get_thread_num(void) { return 0; }", "omp_lock_t"):
            self.assertIn(needle, omp)
        self.assertNotIn("#define _OPENMP", omp)

    def test_stub_thread_count_comes_from_the_environment(self) -> None:
        # 기본은 1. DEVCHECK_OMP_THREADS 는 스텁 omp_get_max_threads() 의 값만 바꾼다(실행은 직렬 그대로).
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEVCHECK_OMP_THREADS", None)
            self.assertIn("-DDEVCHECK_OMP_THREADS=1", devcheck.zig_command("kernel.dll"))
        with mock.patch.dict(os.environ, {"DEVCHECK_OMP_THREADS": "4"}):
            self.assertIn("-DDEVCHECK_OMP_THREADS=4", devcheck.zig_command("kernel.dll"))
        for bad in ("0", "-2", "four"):
            with mock.patch.dict(os.environ, {"DEVCHECK_OMP_THREADS": bad}):
                with self.assertRaises(devcheck.DevcheckProblem):
                    devcheck.zig_command("kernel.dll")

    def test_zig_omp_commands_compile_with_openmp_and_link_the_runtime(self) -> None:
        runtime = Path("rt") / "libomp140.x86_64.dll"
        compile_cmd, link_cmd = devcheck.zig_omp_commands("kernel.dll", runtime)
        self.assertEqual(compile_cmd[1:4], ["-m", "ziglang", "cc"])
        self.assertIn("-fopenmp", compile_cmd)
        self.assertIn("-c", compile_cmd)
        # <omp.h> 는 선언만 있는 헤더가 직렬 스텁보다 먼저 잡힌다. shim.h(aligned_alloc 등)는 그대로 쓴다.
        includes = [Path(compile_cmd[i + 1]) for i, a in enumerate(compile_cmd) if a == "-I"]
        self.assertEqual(includes, [ROOT / "tools" / "devcheck_shim" / "omp_real", ROOT / "tools" / "devcheck_shim"])
        self.assertEqual(compile_cmd[compile_cmd.index("-include") + 1], "shim.h")
        self.assertEqual(compile_cmd[-3:], ["-o", "kernel.o", "kernel.c"])
        # 링크에는 -fopenmp 를 주지 않는다(드라이버가 zig 에 없는 -lomp 를 찾는다): 런타임 파일이 링크 입력이다
        self.assertEqual(link_cmd[1:4], ["-m", "ziglang", "cc"])
        self.assertEqual(link_cmd[4:], ["-shared", "-o", "kernel.dll", "kernel.o", str(runtime), "-lm"])
        header = (ROOT / "tools" / "devcheck_shim" / "omp_real" / "omp.h").read_text(encoding="utf-8")
        for needle in ("extern int omp_get_max_threads(void);", "extern int omp_get_num_threads(void);",
                       "extern int omp_get_thread_num(void);", "extern double omp_get_wtime(void);", "omp_lock_t"):
            self.assertIn(needle, header)
        self.assertNotIn("static inline", header)   # 정의가 아니라 선언이다(런타임이 구현한다)
        self.assertNotIn("#define _OPENMP", header)  # -fopenmp 를 받은 컴파일러가 정의한다

    def test_openmp_runtime_lookup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="devcheck_rt_") as tmp:
            present = _write(Path(tmp) / "libomp.dll", b"x")
            with mock.patch.dict(os.environ, {"DEVCHECK_LIBOMP": str(present)}):
                self.assertEqual(devcheck.libomp_runtime(), present)
            with mock.patch.dict(os.environ, {"DEVCHECK_LIBOMP": str(Path(tmp) / "missing.dll")}):
                with self.assertRaisesRegex(devcheck.DevcheckProblem, "missing.dll"):
                    devcheck.libomp_runtime()
        # 기본 경로는 Windows 에만 있다: 다른 플랫폼에서는 경로를 직접 주라고 알린다
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(devcheck.os, "name", "posix"):
            os.environ.pop("DEVCHECK_LIBOMP", None)
            with self.assertRaisesRegex(devcheck.DevcheckProblem, "DEVCHECK_LIBOMP"):
                devcheck.libomp_runtime()

    def test_c_toolchain_choice(self) -> None:
        settings = devcheck.Settings(path=Path("unused.yaml"))

        def choose(forced: str, *, has_cc: bool, has_zig: bool, platform: str) -> str:
            with mock.patch.object(devcheck.shutil, "which", return_value="/usr/bin/gcc" if has_cc else None), \
                    mock.patch.object(devcheck, "ziglang_available", return_value=has_zig), \
                    mock.patch.object(devcheck.sys, "platform", platform):
                return devcheck.choose_c_toolchain(settings, forced)

        # auto: Linux 에 설정의 컴파일러가 있으면 하네스의 실제 빌드, 그 밖은 zig(직렬)
        self.assertEqual(choose("auto", has_cc=True, has_zig=True, platform="linux"), "gcc")
        self.assertEqual(choose("auto", has_cc=True, has_zig=True, platform="win32"), "zig")
        self.assertEqual(choose("auto", has_cc=False, has_zig=True, platform="linux"), "zig")
        self.assertEqual(choose("gcc", has_cc=True, has_zig=False, platform="win32"), "gcc")
        self.assertEqual(choose("zig", has_cc=True, has_zig=True, platform="linux"), "zig")
        for forced, has_cc, has_zig in (("auto", False, False), ("gcc", False, True), ("zig", True, False)):
            with self.assertRaises(devcheck.DevcheckProblem, msg=forced):
                choose(forced, has_cc=has_cc, has_zig=has_zig, platform="linux")
        # zig-omp(실제 스레드)는 직접 줄 때만 고르고, ziglang 과 OpenMP 런타임이 둘 다 있어야 한다
        with mock.patch.object(devcheck, "libomp_runtime", return_value=Path("rt.dll")):
            self.assertEqual(choose("zig-omp", has_cc=True, has_zig=True, platform="linux"), "zig-omp")
            self.assertEqual(choose("auto", has_cc=False, has_zig=True, platform="win32"), "zig")
            with self.assertRaises(devcheck.DevcheckProblem):
                choose("zig-omp", has_cc=True, has_zig=False, platform="win32")
        with mock.patch.object(devcheck, "libomp_runtime", side_effect=devcheck.DevcheckProblem("no runtime")):
            with self.assertRaises(devcheck.DevcheckProblem):
                choose("zig-omp", has_cc=True, has_zig=True, platform="win32")

    def test_main_leaves_the_callers_streams_alone(self) -> None:
        """회귀: main() 이 표준 스트림을 재설정해, 이 모듈이 돈 뒤로 테스트 러너의 실제 sys.stdout 인코딩·오류 처리 방식이
        바뀌었다(뒤에 도는 테스트의 출력 코덱이 실행 순서에 달렸다). 스트림 정리는 ``__main__`` 진입점에서만 한다."""

        def state() -> list[tuple[Any, Any]]:
            return [(getattr(s, "encoding", None), getattr(s, "errors", None)) for s in (sys.stdout, sys.__stdout__, sys.__stderr__)]

        before = state()
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            devcheck.main(["kernel", "--task", "histogram"])  # 필수 인자 누락: argparse 가 exit 2 (stdout 은 그대로 둔 채 부른다)
        self.assertEqual(cm.exception.code, 2)
        self.assertEqual(state(), before)
        # 러너의 스트림이 이미 같은 설정이어도 드러나게: main() 은 정리 헬퍼를 부르지 않는다
        with mock.patch.object(devcheck, "setup_stdio") as setup_stdio, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(devcheck.main(["gen", "--tasks-dir", str(ROOT / "tests"), "--task", "no_such_task"]), 2)
            with self.assertRaises(SystemExit):
                devcheck.main(["kernel", "--task", "histogram"])
        setup_stdio.assert_not_called()


class ReadSourceTest(unittest.TestCase):
    """``read_source``: UTF-8 만 받고(아니면 파일을 이름으로 알리는 문제), 맨 앞의 BOM 은 소스·raw JSON 에서만 뗀다."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="devcheck_src_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_bom_is_dropped_from_sources_and_raw_json_but_kept_in_responses(self) -> None:
        for name in ("kernel.py", "kernel.c", "lib.rs", "kernel.PY"):
            text, is_response = devcheck.read_source(_write(self.tmp / name, UTF8_BOM_BYTES + BASELINE_PY.encode("utf-8")))
            self.assertEqual((text, is_response), (BASELINE_PY, False), name)
            compile(text, name, "exec")  # 문자열 compile() 은 U+FEFF 를 구문 오류로 본다 → 떼어야 한다
        raw = json.dumps({"response_text": MD_CORRECT}).encode("utf-8")
        self.assertEqual(devcheck.read_source(_write(self.tmp / "raw.json", UTF8_BOM_BYTES + raw)), (MD_CORRECT, True))
        # 응답 문서: 더미 provider(llm_client)가 읽는 것과 같은 텍스트(BOM 포함)
        text, is_response = devcheck.read_source(_write(self.tmp / "0.md", UTF8_BOM_BYTES + MD_CORRECT.encode("utf-8")))
        self.assertEqual((text, is_response), ("﻿" + MD_CORRECT, True))
        self.assertEqual(text, (self.tmp / "0.md").read_text(encoding="utf-8"))

    def test_files_without_bom_and_crlf_newlines(self) -> None:
        self.assertEqual(devcheck.read_source(_write(self.tmp / "0.md", MD_CORRECT)), (MD_CORRECT, True))
        self.assertEqual(devcheck.read_source(_write(self.tmp / "k.py", BASELINE_PY)), (BASELINE_PY, False))
        crlf = MD_CORRECT.replace("\n", "\r\n").encode("utf-8")
        self.assertEqual(devcheck.read_source(_write(self.tmp / "crlf.md", crlf)), (MD_CORRECT, True))

    def test_unreadable_inputs_are_named_problems(self) -> None:
        cases = {
            "cp949.md": MD_CP949_BYTES,                    # UTF-8 이 아니다
            "latin1.py": "# café\n".encode("latin-1"),
            "bad.json": b"{not json",
            "list.json": b"[1, 2]",                        # raw 문서(매핑)가 아니다
        }
        for name, data in cases.items():
            _write(self.tmp / name, data)
        for name in (*cases, "missing.md"):
            with self.subTest(name=name):
                with self.assertRaises(devcheck.DevcheckProblem) as cm:
                    devcheck.read_source(self.tmp / name)
                self.assertIn(name, str(cm.exception), "어느 파일인지 알려야 한다")
                if name in ("cp949.md", "latin1.py"):
                    self.assertIn("UTF-8", str(cm.exception))


class BuildAdapterTest(unittest.TestCase):
    """빌드 어댑터의 호출 규약(설계 v3 §6)을 대역 모듈로 확인한다 — 실제 gcc·cargo·numba·halide 빌드는 Linux 에서 돈다."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="devcheck_build_")
        self.tmp = Path(self._tmp.name)
        self.settings = devcheck.load_settings(ROOT / "config" / "experiment.yaml")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def result(out_dir: Path, name: str = "kernel.so", **over: Any) -> SimpleNamespace:
        values: dict[str, Any] = {"ok": True, "artifact": Path(out_dir) / name, "error_kind": "", "error_detail": ""}
        values.update(over)
        return SimpleNamespace(**values)

    @staticmethod
    def build_modules(fakes: dict[str, Any]) -> Any:
        """devcheck 가 ``importlib.import_module`` 로 올리는 빌드 모듈을 대역으로 바꾼다(값이 None 이면 그 모듈은 없는 것).

        저장소에 그 모듈이 있든 없든(G2b 가 나중에 더한다) 같은 결과가 나온다. ``sys.modules`` 는 건드리지 않는다 —
        ``patch.dict(sys.modules)`` 는 빠져나올 때 그 사이에 처음 임포트된 모듈을 전부 지운다."""
        real_import = importlib.import_module

        def import_module(name: str, package: str | None = None) -> Any:
            if name in fakes:
                if fakes[name] is None:
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return fakes[name]
            return real_import(name, package)

        return mock.patch.object(devcheck.importlib, "import_module", import_module)

    def test_gcc_adapter_passes_the_configured_build_and_the_args_header(self) -> None:
        import harness.build.c as build_c_mod

        calls: list[dict[str, Any]] = []

        def build_c(code, out_dir, bcfg, *, openmp, launch_prefix, symbol="gemm", args_header=None):
            calls.append({"code": code, "out_dir": Path(out_dir), "cc": bcfg.cc, "openmp": openmp, "launch_prefix": launch_prefix,
                          "symbol": symbol, "args_header": args_header})
            return self.result(out_dir)

        lang = devcheck.LanguageInfo("c_openmp", "c", openmp=True, threads=4)
        with mock.patch.object(build_c_mod, "build_c", build_c):
            built = devcheck.build_c_gcc("int x;", self.tmp, self.settings, lang, "histogram", "HEADER")
        self.assertEqual(calls, [{"code": "int x;", "out_dir": self.tmp, "cc": self.settings.cc, "openmp": True, "launch_prefix": [],
                                  "symbol": "histogram", "args_header": "HEADER"}])
        self.assertEqual((built.ok, built.artifact, built.toolchain), (True, self.tmp / "kernel.so", ""))
        self.assertTrue(built.note.startswith(f"{self.settings.cc} "), built.note)
        self.assertIn("-fopenmp", built.note)
        self.assertFalse((self.tmp / "kernel_args.h").exists(), "args_header 를 받는 build_c 가 헤더를 쓴다")

    def test_gcc_adapter_writes_the_header_itself_for_a_build_c_without_args_header(self) -> None:
        import harness.build.c as build_c_mod

        seen: dict[str, Any] = {}

        def build_c(code, out_dir, bcfg, *, openmp, launch_prefix, symbol="gemm"):  # 설계 v3 §6 이전의 시그니처
            seen.update(openmp=openmp, symbol=symbol)
            return self.result(out_dir, ok=False, artifact=None, error_kind="compile_error", error_detail="kernel.c:1:1: error: x")

        lang = devcheck.LanguageInfo("c_seq", "c", openmp=False, threads=1)
        with mock.patch.object(build_c_mod, "build_c", build_c):
            built = devcheck.build_c_gcc("int x;", self.tmp, self.settings, lang, "histogram", "HEADER")
        self.assertEqual(seen, {"openmp": False, "symbol": "histogram"})
        self.assertEqual((self.tmp / "kernel_args.h").read_text(encoding="utf-8"), "HEADER")
        self.assertEqual((built.ok, built.error_kind, built.error_detail), (False, "compile_error", "kernel.c:1:1: error: x"))
        self.assertNotIn("-fopenmp", built.note)

    def test_gcc_adapter_turns_missing_tools_and_bad_config_into_problems(self) -> None:
        import harness.build.c as build_c_mod

        class FakeToolchainMissing(RuntimeError):
            pass

        lang = devcheck.LanguageInfo("c_seq", "c")
        for error in (FileNotFoundError("gcc"), FakeToolchainMissing("no gcc")):
            def build_c(*args: Any, _error: Exception = error, **kwargs: Any) -> Any:
                raise _error

            with mock.patch.object(build_c_mod, "build_c", build_c), \
                    mock.patch.object(build_c_mod, "ToolchainMissing", FakeToolchainMissing, create=True), \
                    self.assertRaises(devcheck.DevcheckProblem):
                devcheck.build_c_gcc("int x;", self.tmp, self.settings, lang, "histogram", None)

        def crash(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("bug")

        with mock.patch.object(build_c_mod, "build_c", crash), self.assertRaises(RuntimeError):  # 버그는 문제로 바꾸지 않는다
            devcheck.build_c_gcc("int x;", self.tmp, self.settings, lang, "histogram", None)
        with contextlib.redirect_stderr(io.StringIO()):
            no_config = devcheck.load_settings(self.tmp / "missing.yaml")
        with self.assertRaises(devcheck.DevcheckProblem):
            devcheck.build_c_gcc("int x;", self.tmp, no_config, lang, "histogram", None)

    def test_numba_and_halide_adapters_call_the_build_modules(self) -> None:
        calls: dict[str, Any] = {}

        def build_numba(code, out_dir, *, function, expected_signature):
            calls["numba"] = (code, Path(out_dir), function, expected_signature)
            return self.result(out_dir, ok=False, artifact=None, error_kind="signature_mismatch", error_detail="numba_signature_missing")

        def build_halide(code, out_dir, *, function, halide_mode):
            calls["halide"] = (code, Path(out_dir), function, halide_mode)
            return self.result(out_dir, "kernel.py")

        fakes = {"harness.build.numba": SimpleNamespace(build_numba=build_numba),
                 "harness.build.halide": SimpleNamespace(build_halide=build_halide)}
        with self.build_modules(fakes):
            nb = devcheck.build_python_family("CODE", self.tmp, devcheck.LanguageInfo("numba", "numba", threads=2), HISTOGRAM_SPEC)
            hl = devcheck.build_python_family("CODE", self.tmp, devcheck.LanguageInfo("halide_manual", "halide", halide_mode="manual"), HISTOGRAM_SPEC)
        self.assertEqual(calls["numba"], ("CODE", self.tmp, "histogram", "void(int32[::1], int32[::1], int64, int64)"))
        self.assertEqual(calls["halide"], ("CODE", self.tmp, "histogram", "manual"))
        self.assertEqual((nb.ok, nb.error_kind, nb.error_detail), (False, "signature_mismatch", "numba_signature_missing"))
        self.assertEqual((hl.ok, hl.artifact), (True, self.tmp / "kernel.py"))

    def test_numba_and_halide_fall_back_to_a_syntax_check_without_the_build_modules(self) -> None:
        with self.build_modules({"harness.build.numba": None, "harness.build.halide": None}):  # None = 그 모듈이 없다
            nb = devcheck.build_python_family(BASELINE_PY, self.tmp, devcheck.LanguageInfo("numba", "numba"), HISTOGRAM_SPEC)
            hl = devcheck.build_python_family("def histogram(:\n", self.tmp, devcheck.LanguageInfo("halide_auto", "halide", halide_mode="auto"), HISTOGRAM_SPEC)
        self.assertEqual((nb.ok, nb.artifact), (True, self.tmp / "kernel.py"))
        self.assertIn("harness.build.numba", nb.note)
        self.assertEqual((hl.ok, hl.error_kind), (False, "compile_error"))
        self.assertIn("SyntaxError", hl.error_detail)
        self.assertIn("harness.build.halide", hl.note)

    def test_rust_adapter_calls_build_rust_with_the_configured_toolchain(self) -> None:
        calls: list[dict[str, Any]] = []

        def build_rust(code, out_dir, rcfg, *, launch_prefix, symbol, args_module=None, target_dir=None):
            calls.append({"code": code, "out_dir": Path(out_dir), "rcfg": rcfg, "launch_prefix": launch_prefix, "symbol": symbol,
                          "args_module": args_module, "target_dir": target_dir})
            return self.result(out_dir)

        rcfg = object()
        cfg = SimpleNamespace(build=SimpleNamespace(rust=rcfg))
        with self.build_modules({"harness.build.rust": SimpleNamespace(build_rust=build_rust)}), \
                mock.patch("harness.config.load_config", return_value=cfg) as load_config:
            built = devcheck.build_rust("CODE", self.tmp, self.settings, HISTOGRAM_SPEC, "STRUCT")
        load_config.assert_called_once_with(self.settings.path)
        self.assertEqual(calls, [{"code": "CODE", "out_dir": self.tmp, "rcfg": rcfg, "launch_prefix": [], "symbol": "histogram",
                                  "args_module": "STRUCT", "target_dir": Path(tempfile.gettempdir()) / "devcheck_cargo_target"}])
        self.assertEqual((built.ok, built.artifact, built.note, built.toolchain), (True, self.tmp / "kernel.so", "cargo --release", ""))

    def test_rust_adapter_problems(self) -> None:
        def build_rust(*args: Any, **kwargs: Any) -> Any:
            raise FileNotFoundError("cargo")

        cfg = SimpleNamespace(build=SimpleNamespace(rust=object()))
        with self.build_modules({"harness.build.rust": None}), self.assertRaises(devcheck.DevcheckProblem) as cm:
            devcheck.build_rust("CODE", self.tmp, self.settings, HISTOGRAM_SPEC, None)  # 모듈이 아직 없다
        self.assertIn("rust backend not available yet", str(cm.exception))
        fake = {"harness.build.rust": SimpleNamespace(build_rust=build_rust)}
        with self.build_modules(fake), mock.patch("harness.config.load_config", return_value=SimpleNamespace(build=SimpleNamespace())), \
                self.assertRaises(devcheck.DevcheckProblem) as cm:
            devcheck.build_rust("CODE", self.tmp, self.settings, HISTOGRAM_SPEC, None)  # 설정에 build.rust 가 없다
        self.assertIn("build.rust", str(cm.exception))
        with self.build_modules(fake), mock.patch("harness.config.load_config", return_value=cfg), \
                self.assertRaises(devcheck.DevcheckProblem) as cm:
            devcheck.build_rust("CODE", self.tmp, self.settings, HISTOGRAM_SPEC, None)  # cargo 실행 파일이 없다
        self.assertIn("cargo", str(cm.exception))

    def test_python_kernel_is_saved_and_syntax_checked_only(self) -> None:
        built = devcheck.build_python_family(BASELINE_PY, self.tmp, devcheck.LanguageInfo("numpy", "python"), HISTOGRAM_SPEC)
        self.assertEqual((built.ok, built.artifact, built.toolchain), (True, self.tmp / "kernel.py", ""))
        self.assertEqual((self.tmp / "kernel.py").read_text(encoding="utf-8"), BASELINE_PY)

    def test_baseline_request_for_python_and_c_baselines(self) -> None:
        tasks = self.tmp / "tasks"
        _write(tasks / "py" / "signatures.yaml", SIGNATURES_YAML)
        _write(tasks / "py" / "tolerance.yaml", TOLERANCE_YAML)
        _write(tasks / "py" / "baseline" / "baseline.py", BASELINE_PY)
        c_yaml = SIGNATURES_YAML.replace("baseline: {backend: python, file: baseline/baseline.py, openmp: false",
                                         "baseline: {backend: c, file: baseline/baseline.c, openmp: true")
        self.assertIn("backend: c", c_yaml)
        _write(tasks / "c" / "signatures.yaml", c_yaml)
        _write(tasks / "c" / "tolerance.yaml", TOLERANCE_YAML)
        _write(tasks / "c" / "baseline" / "baseline.c", UTF8_BOM_BYTES + b"void histogram(void) {}\n")
        _write(tasks / "nofile" / "signatures.yaml", SIGNATURES_YAML)
        _write(tasks / "nofile" / "tolerance.yaml", TOLERANCE_YAML)

        py = devcheck.baseline_request(devcheck.load_task("py", tasks), self.settings, ["tiny"], bench=True, c_toolchain="auto", timeout_s=None)
        self.assertEqual((py.label, py.lang.backend, py.lang.source, py.code, py.response, py.bench),
                         ("py/baseline", "python", "baseline", None, None, True))
        self.assertEqual(py.artifact, (tasks / "py" / "baseline" / "baseline.py").resolve())
        c = devcheck.baseline_request(devcheck.load_task("c", tasks), self.settings, ["tiny", "rect"], bench=False, c_toolchain="zig", timeout_s=7.0)
        self.assertEqual((c.label, c.lang.backend, c.lang.openmp, c.artifact, c.tests, c.c_toolchain, c.timeout_s),
                         ("c/baseline", "c", True, None, ["tiny", "rect"], "zig", 7.0))
        self.assertEqual(c.code, "void histogram(void) {}\n", "C 기준선 소스의 BOM 은 뗀다")
        with self.assertRaises(devcheck.DevcheckProblem):
            devcheck.baseline_request(devcheck.load_task("nofile", tasks), self.settings, ["tiny"], bench=False, c_toolchain="auto", timeout_s=None)


class EndToEndBase(unittest.TestCase):
    """끝-끝 테스트의 공통 준비(테스트 메서드 없음): 임시 과제 디렉터리에 int32 히스토그램 과제와 더미 응답을 만든다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="devcheck_e2e_")
        cls.tmp = Path(cls._tmpdir.name)
        cls.tasks = cls.tmp / "tasks"
        tdir = cls.tasks / "histogram"
        (tdir / "baseline").mkdir(parents=True)
        (tdir / "signatures.yaml").write_text(SIGNATURES_YAML, encoding="utf-8", newline="\n")
        (tdir / "tolerance.yaml").write_text(TOLERANCE_YAML, encoding="utf-8", newline="\n")
        (tdir / "baseline" / "baseline.py").write_text(BASELINE_PY, encoding="utf-8", newline="\n")
        with contextlib.redirect_stderr(io.StringIO()):
            testgen.generate("histogram", 1234, tdir / "tests", SHAPES, _histogram_case, "numpy bincount", out_dtype="int32", sentinel=-2048)
        cls.dummies = cls.tmp / "dummies"
        (cls.dummies / "histogram" / "numpy").mkdir(parents=True)
        (cls.dummies / "histogram" / "numpy" / "0.md").write_text(MD_CORRECT, encoding="utf-8", newline="\n")
        # dummies 의 기본 언어 목록에는 설정의 languages 가 들어간다 → 저장소 설정에 기대지 않게 고정한 설정을 쓴다
        cls.config = cls.tmp / "experiment.yaml"
        cls.config.write_text(CONFIG_YAML, encoding="utf-8", newline="\n")
        cls._n = 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def write(self, content: str | bytes, name: str) -> Path:
        type(self)._n += 1
        return _write(self.tmp / f"file_{type(self)._n}_{name}", content)

    def new_dir(self, name: str) -> Path:
        type(self)._n += 1
        path = self.tmp / f"dir_{type(self)._n}_{name}"
        path.mkdir()
        return path

    def dummy_dir(self, files: dict[str, str | bytes]) -> Path:
        """``{"<과제>/<언어>/0.md": 내용}`` 으로 새 더미 디렉터리를 만든다."""
        root = self.new_dir("dummies")
        for rel, content in files.items():
            _write(root / rel, content)
        return root

    def run_devcheck(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = devcheck.main([*argv[:1], "--tasks-dir", str(self.tasks), *argv[1:]])
        return rc, out.getvalue(), err.getvalue()

    def run_kernel(self, content: str | bytes, name: str, *extra: str, language: str = "numpy") -> tuple[int, str, str]:
        return self.run_devcheck("kernel", "--task", "histogram", "--language", language, "--file", str(self.write(content, name)), *extra)


class EndToEndTest(EndToEndBase):
    """임시 과제 디렉터리의 int32 히스토그램 과제로 devcheck 의 네 하위 명령을 실제 워커와 함께 돌린다."""

    def test_kernel_md_passes_on_three_shapes(self) -> None:
        rc, out, err = self.run_kernel(MD_CORRECT, "ok.md")
        self.assertEqual(rc, 0, out + err)
        lines = out.strip().splitlines()
        self.assertEqual(len(lines), 4, out)
        for line, name in zip(lines, ("tiny", "rect", "small")):
            self.assertRegex(line, rf"^histogram/numpy {name}\s+PASS\s+n=\d+ n_bins=\d+ max_abs_err=0 ")
        self.assertEqual(lines[3], "histogram/numpy => PASS stage=check classification=pass")

    def test_kernel_that_accumulates_fails_with_classification(self) -> None:
        rc, out, _ = self.run_kernel(MD_ACCUMULATE, "acc.md")
        self.assertEqual(rc, 1, out)
        self.assertEqual(shape_verdict_words(out), ["FAIL", "FAIL", "FAIL"], out)
        self.assertIn("histogram/numpy tiny  FAIL", out)
        self.assertIn("wrong_output/accumulated_into_c", out)
        self.assertTrue(out.strip().endswith("histogram/numpy => FAIL stage=check classification=wrong_output detail=accumulated_into_c"), out)

    def test_extract_failure_is_reported_before_running(self) -> None:
        rc, out, _ = self.run_kernel(MD_BAD_SIGNATURE, "sig.md")
        self.assertEqual(rc, 1, out)
        self.assertEqual(shape_verdict_words(out), ["NOT-RUN", "NOT-RUN", "NOT-RUN"], out)
        self.assertIn("=> FAIL stage=extract classification=signature_mismatch detail=param_mismatch:", out)

    def test_source_file_tests_option_bench_and_keep(self) -> None:
        keep = self.tmp / "kept"
        rc, out, err = self.run_kernel(BASELINE_PY, "k.py", "--tests", "tiny", "--bench", "--keep", str(keep))
        self.assertEqual(rc, 0, out + err)
        lines = out.strip().splitlines()
        self.assertEqual(len(lines), 2, out)
        self.assertRegex(lines[1], r"^histogram/numpy => PASS stage=check classification=pass bench_large_ms=\d+\.\d{3}$")
        for name in ("kernel.py", "check.json", "bench.json", "check.stderr.log"):
            self.assertTrue((keep / name).is_file(), name)
        self.assertEqual(json.loads((keep / "bench.json").read_text(encoding="utf-8"))["test"], "large")

    def test_syntax_error_in_source_is_a_build_failure(self) -> None:
        rc, out, _ = self.run_kernel("def histogram(:\n", "bad.py")
        self.assertEqual(rc, 1, out)
        self.assertIn("=> FAIL stage=build classification=compile_error detail=SyntaxError", out)

    def test_kernel_that_raises_at_import_fails_at_the_build_stage(self) -> None:
        """로드 실패는 harness.run 의 행처럼 build 단계의 오류다(devcheck 와 파이프라인의 단계 이름이 같아야 기대 표가 맞는다)."""
        rc, out, _ = self.run_kernel(KERNEL_IMPORT_ERROR, "import_error.py", "--tests", "tiny")
        self.assertEqual(rc, 1, out)
        self.assertEqual(shape_verdict_words(out), ["NOT-RUN"], out)
        self.assertTrue(out.strip().endswith("histogram/numpy => FAIL stage=build classification=compile_error detail=RuntimeError: boom at import"), out)

    @requires_modules("halide")
    def test_halide_definition_errors_get_the_pipeline_labels(self) -> None:
        """실제 워커: halide 의 스케줄 오류는 harness.run 처럼 build/compile_error, Func 가 아닌 반환값은 build/signature_mismatch."""
        for source, classification in ((HALIDE_BAD_SCHEDULE, "compile_error"), (HALIDE_NOT_A_FUNC, "signature_mismatch")):
            rc, out, err = self.run_kernel(source, "pipe.py", "--tests", "tiny", language="halide_manual")
            self.assertEqual(rc, 1, out + err)
            final = out.strip().splitlines()[-1]
            self.assertTrue(final.startswith(f"histogram/halide_manual => FAIL stage=build classification={classification} "), final)

    def test_kernel_that_ends_the_worker_process_is_a_runtime_crash(self) -> None:
        rc, out, _ = self.run_kernel(KERNEL_EXITS, "exits.py", "--tests", "tiny,rect")
        self.assertEqual(rc, 1, out)
        self.assertEqual(shape_verdict_words(out), ["FAIL", "NOT-RUN"], out)
        self.assertTrue(out.strip().endswith("histogram/numpy => FAIL stage=check classification=runtime_crash "
                                             "detail=worker died (rc=0) during tiny call 0"), out)

    def test_hanging_kernel_is_a_timeout(self) -> None:
        rc, out, _ = self.run_kernel(KERNEL_HANGS, "hangs.py", "--tests", "tiny", "--timeout", "3")
        self.assertEqual(rc, 1, out)
        self.assertIn("histogram/numpy => FAIL stage=check classification=timeout detail=worker timeout after 3s during ", out)

    def test_keyword_args_mode_for_a_python_kernel(self) -> None:
        rc, out, err = self.run_kernel("```python\n" + KERNEL_KEYWORD + "```\n", "keyword.md", "--args-mode", "keyword", "--tests", "tiny")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(out.strip().splitlines()[-1], "histogram/numpy => PASS stage=check classification=pass")
        # positional 시그니처의 커널은 keyword 조건의 추출 검사에 걸린다(워커까지 가지 않는다)
        rc, out, _ = self.run_kernel(MD_CORRECT, "positional.md", "--args-mode", "keyword", "--tests", "tiny")
        self.assertEqual(rc, 1, out)
        self.assertIn("=> FAIL stage=extract classification=signature_mismatch detail=param_mismatch:", out)

    def test_raw_json_input(self) -> None:
        rc, out, err = self.run_kernel(json.dumps({"task": "histogram", "response_text": MD_CORRECT}), "raw.json", "--tests", "tiny")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(out.strip().splitlines()[-1], "histogram/numpy => PASS stage=check classification=pass")
        rc, out, _ = self.run_kernel("{not json", "bad.json", "--tests", "tiny")
        self.assertEqual(rc, 2, out)
        self.assertIn("devcheck => PROBLEM", out)

    def test_utf8_bom_in_a_source_file_is_ignored(self) -> None:
        """회귀: BOM 으로 시작하는 맞는 파이썬 커널을 compile_error(exit 1)로 판정했다 — 워커의 모듈 로더와 컴파일러는 BOM 을
        건너뛰는데 devcheck 의 구문 검사만 걸렸다."""
        rc, out, err = self.run_kernel(UTF8_BOM_BYTES + BASELINE_PY.encode("utf-8"), "bom.py", "--tests", "tiny")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(out.strip().splitlines()[-1], "histogram/numpy => PASS stage=check classification=pass")

    def test_utf8_bom_in_a_response_is_kept_and_reported(self) -> None:
        """응답 문서(.md)의 BOM 은 떼지 않는다: 더미 provider 가 주는 텍스트 그대로 추출해 실제 파이프라인과 같은 판정을 낸다."""
        # 문장으로 시작하는 응답: 추출에 지장이 없다 → 통과하고 BOM 이 있음을 알린다
        rc, out, err = self.run_kernel(UTF8_BOM_BYTES + MD_CORRECT.encode("utf-8"), "bom_prose.md", "--tests", "tiny")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("[devcheck] histogram/numpy: response starts with a UTF-8 BOM", err)
        # 펜스로 시작하는 응답: harness.extract 가 BOM 뒤의 펜스를 인식하지 못한다 → 추출 실패 + 원인 표시
        fence_first = "```python\n" + BASELINE_PY + "```\n"
        rc, out, _ = self.run_kernel(UTF8_BOM_BYTES + fence_first.encode("utf-8"), "bom_fence.md", "--tests", "tiny")
        self.assertEqual(rc, 1, out)
        self.assertIn("=> FAIL stage=extract classification=signature_mismatch detail=", out)
        self.assertTrue(out.strip().endswith("(caused by the UTF-8 BOM at the start of the response file: remove it)"), out)
        # BOM 과 무관한 추출 실패에는 BOM 탓을 하지 않는다
        rc, out, _ = self.run_kernel(UTF8_BOM_BYTES + ("Explanation.\n\n" + MD_BAD_SIGNATURE).encode("utf-8"), "bom_bad_signature.md")
        self.assertEqual(rc, 1, out)
        self.assertIn("detail=param_mismatch:", out)
        self.assertNotIn("caused by the UTF-8 BOM", out)

    def test_non_utf8_file_is_a_problem_that_names_the_file(self) -> None:
        path = self.write(MD_CP949_BYTES, "cp949.md")
        rc, out, _ = self.run_devcheck("kernel", "--task", "histogram", "--language", "numpy", "--file", str(path))
        self.assertEqual(rc, 2, out)
        self.assertIn("devcheck => PROBLEM", out)
        self.assertIn(path.name, out)
        self.assertNotIn("internal error", out)

    def test_baseline(self) -> None:
        rc, out, err = self.run_devcheck("baseline", "--task", "histogram")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(out.strip().splitlines()[-1], "histogram/baseline => PASS stage=check classification=pass")
        self.assertEqual(shape_verdict_words(out), ["PASS", "PASS", "PASS"], out)

    def test_usage_and_tool_problems_exit_2(self) -> None:
        ok_md = str(self.write(MD_CORRECT, "ok.md"))
        cases = [
            ("kernel", "--task", "nosuch", "--language", "numpy", "--file", ok_md),
            ("kernel", "--task", "histogram", "--language", "fortran", "--file", ok_md),
            ("kernel", "--task", "histogram", "--language", "numpy", "--file", str(self.tmp / "missing.md")),
            ("kernel", "--task", "histogram", "--language", "numpy", "--file", ok_md, "--tests", "tiny,huge"),
            ("baseline", "--task", "nosuch"),
            ("gen", "--task", "nosuch"),
        ]
        for argv in cases:
            with self.subTest(argv=argv[:5]):
                rc, out, _ = self.run_devcheck(*argv)
                self.assertEqual(rc, 2, out)
                self.assertIn("PROBLEM", out)
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            devcheck.main(["kernel", "--task", "histogram"])  # 필수 인자 누락: argparse 가 exit 2
        self.assertEqual(cm.exception.code, 2)

    def test_dummies_table_with_missing_and_unsupported(self) -> None:
        rc, out, err = self.run_devcheck("dummies", "--config", str(self.config), "--dummy-dir", str(self.dummies))
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(out.strip().splitlines()[-1], "summary: 1 pass, 0 fail, 0 missing, 0 problem, 0 unsupported")
        self.assertIn("[devcheck] dummies: 기본 목록", err)
        # 요청한 언어에 더미가 없으면 missing(실패), unsupported 로 표시된 언어는 실패가 아니다
        rc, out, err = self.run_devcheck("dummies", "--config", str(self.config), "--dummy-dir", str(self.dummies),
                                         "--languages", "numpy,halide_auto,c_seq")
        self.assertEqual(rc, 1, out)
        self.assertIn("summary: 1 pass, 0 fail, 1 missing, 0 problem, 1 unsupported", out)
        table = table_rows(out, "histogram")
        self.assertEqual(len(table), 3, out)
        self.assertRegex(table[0], r"^histogram\s+numpy\s+PASS\s+PASS\s+PASS\s+pass$")
        self.assertRegex(table[1], r"^histogram\s+halide_auto\s+-\s+-\s+-\s+unsupported\s+data-dependent writes")
        self.assertRegex(table[2], r"^histogram\s+c_seq\s+-\s+-\s+-\s+missing\s+setup/missing_dummy: no histogram/c_seq/0\.md")
        rc, out, err = self.run_devcheck("dummies", "--config", str(self.config), "--dummy-dir", str(self.dummies),
                                         "--tasks", "histogram", "--languages", "halide_auto")
        self.assertEqual(rc, 0, out)
        self.assertNotIn("기본 목록", err, "과제·언어를 둘 다 주면 기본 목록을 쓰지 않는다")

    def test_dummies_default_lists_cover_tasks_without_dummies_and_configured_languages(self) -> None:
        """회귀: 기본 과제·언어 목록이 "0.md 가 있는 것"에서만 나와, 더미 디렉터리가 아예 없는 과제와 설정에는 있지만 더미가
        하나도 없는 언어가 표에 나오지 않고 exit 0 이었다(검증된 더미가 있어야 실험 대상이라는 규칙이 빠졌다)."""
        tasks = self.new_dir("tasks")
        shutil.copytree(self.tasks / "histogram", tasks / "histogram")
        _write(tasks / "newtask" / "signatures.yaml", SIGNATURES_YAML.replace("histogram", "newtask"))  # 더미 디렉터리가 없는 과제
        _write(tasks / "newtask" / "tolerance.yaml", TOLERANCE_YAML.replace("histogram", "newtask"))
        dummies = self.dummy_dir({"histogram/numpy/0.md": MD_CORRECT,
                                  "ghost/numpy/0.md": MD_CORRECT})                                      # 과제가 없는 더미
        config = self.write(CONFIG_WITH_C_SEQ_YAML, "experiment.yaml")                                   # c_seq 는 더미가 하나도 없다
        argv = ("dummies", "--tasks-dir", str(tasks), "--config", str(config), "--dummy-dir", str(dummies), "--tests", "tiny")
        rc, out, err = self.run_devcheck(*argv)
        self.assertEqual(rc, 1, out + err)
        rows = table_rows(out, "ghost", "histogram", "newtask")
        self.assertEqual([tuple(r.split()[:2]) for r in rows],
                         [("ghost", "numpy"), ("ghost", "c_seq"), ("histogram", "numpy"), ("histogram", "c_seq"),
                          ("newtask", "numpy"), ("newtask", "c_seq")], out)
        self.assertRegex(rows[0], r"^ghost\s+numpy\s+-\s+problem\s+setup/devcheck_problem: ")
        self.assertRegex(rows[1], r"^ghost\s+c_seq\s+-\s+problem\s+setup/devcheck_problem: ")
        self.assertRegex(rows[2], r"^histogram\s+numpy\s+PASS\s+pass$")
        self.assertRegex(rows[3], r"^histogram\s+c_seq\s+-\s+missing\s+setup/missing_dummy: no histogram/c_seq/0\.md")
        self.assertRegex(rows[4], r"^newtask\s+numpy\s+-\s+missing\s+setup/missing_dummy: no newtask/numpy/0\.md")
        self.assertRegex(rows[5], r"^newtask\s+c_seq\s+-\s+missing\s+setup/missing_dummy: no newtask/c_seq/0\.md")
        self.assertEqual(out.strip().splitlines()[-1], "summary: 1 pass, 0 fail, 3 missing, 2 problem, 0 unsupported")
        self.assertIn("newtask/numpy => FAIL stage=setup classification=missing_dummy", out)
        self.assertIn("[devcheck] dummies: 기본 목록", err)

    def test_dummies_with_a_failing_dummy_exit_1(self) -> None:
        dummies = self.dummy_dir({"histogram/numpy/0.md": MD_ACCUMULATE})
        rc, out, _ = self.run_devcheck("dummies", "--config", str(self.config), "--dummy-dir", str(dummies))
        self.assertEqual(rc, 1, out)
        self.assertRegex(table_rows(out, "histogram")[0],
                         r"^histogram\s+numpy\s+FAIL\s+FAIL\s+FAIL\s+fail\s+check/wrong_output: accumulated_into_c$")
        self.assertEqual(out.strip().splitlines()[-1], "summary: 0 pass, 1 fail, 0 missing, 0 problem, 0 unsupported")

    def test_dummies_continue_after_a_non_utf8_file(self) -> None:
        """회귀: UTF-8 이 아닌 0.md 하나가 ``dummies`` 전체를 'internal error' 로 끝냈다 — 표·요약이 없고 그 파일의 이름도
        나오지 않았으며 나머지 조합은 점검되지 않았다."""
        tasks = self.new_dir("tasks")
        shutil.copytree(self.tasks / "histogram", tasks / "alpha")  # 이름순으로 histogram 보다 먼저 점검된다
        shutil.copytree(self.tasks / "histogram", tasks / "histogram")
        dummies = self.dummy_dir({"alpha/numpy/0.md": MD_CP949_BYTES, "histogram/numpy/0.md": MD_CORRECT})
        rc, out, err = self.run_devcheck("dummies", "--tasks-dir", str(tasks), "--config", str(self.config),
                                         "--dummy-dir", str(dummies), "--tests", "tiny")
        self.assertEqual(rc, 2, out + err)
        self.assertNotIn("internal error", out)
        self.assertRegex(out, r"(?m)^alpha/numpy => PROBLEM stage=setup classification=devcheck_problem detail=.*UTF-8.*0\.md")
        rows = table_rows(out, "alpha", "histogram")
        self.assertRegex(rows[0], r"^alpha\s+numpy\s+-\s+problem\s+setup/devcheck_problem: .*UTF-8.*0\.md")
        self.assertRegex(rows[1], r"^histogram\s+numpy\s+PASS\s+pass$")
        self.assertEqual(out.strip().splitlines()[-1], "summary: 1 pass, 0 fail, 0 missing, 1 problem, 0 unsupported")

    def test_gen_runs_the_task_generator_as_is(self) -> None:
        tasks = self.tmp / "gen_tasks"
        tdir = tasks / "histogram"
        tdir.mkdir(parents=True)
        (tdir / "gen_tests.py").write_text(GEN_TESTS_PY, encoding="utf-8", newline="\n")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = devcheck.main(["gen", "--tasks-dir", str(tasks), "--task", "histogram"])
        self.assertEqual(rc, 0, out.getvalue() + err.getvalue())
        lines = out.getvalue().strip().splitlines()
        self.assertEqual([ln.split()[1] for ln in lines[:4]], ["tiny", "rect", "small", "large"])
        self.assertRegex(lines[-1], r"^histogram/gen => PASS test_hash=[0-9a-f]{16} seed=20261105 total_mb=\d+\.\d elapsed_s=\d+\.\d$")
        self.assertIn("[gen_tests:histogram] wrote manifest", err.getvalue())
        for name in ("tiny", "rect", "small", "large"):
            self.assertTrue((tdir / "tests" / f"{name}.npz").is_file())
        # 생성기가 실패하면 exit 1
        (tdir / "gen_tests.py").write_text("import sys\nsys.exit(1)\n", encoding="utf-8", newline="\n")
        with contextlib.redirect_stdout(io.StringIO()) as out2, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(devcheck.main(["gen", "--tasks-dir", str(tasks), "--task", "histogram"]), 1)
        self.assertIn("histogram/gen => FAIL rc=1", out2.getvalue())
        # 생성기가 성공으로 끝났는데 파일을 만들지 않았으면 exit 1(없는 것을 알린다)
        _write(tasks / "silent" / "gen_tests.py", "import sys\nsys.exit(0)\n")
        with contextlib.redirect_stdout(io.StringIO()) as out3, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(devcheck.main(["gen", "--tasks-dir", str(tasks), "--task", "silent"]), 1)
        self.assertIn("silent/gen => FAIL missing after generation: tiny, rect, small, large, manifest.json", out3.getvalue())


class CheckLogicTest(EndToEndBase):
    """``check`` 의 워커 결과 해석을 워커 없이(``run_worker`` 대역) 확인한다: 로드 실패, 플래그만 있는 문서, bench 분기."""

    @staticmethod
    def check_doc(**over: Any) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "complete": True, "load_ok": True, "classification": "pass", "error_detail": "", "inputs_mutated": False, "oob_write": False,
            "tests": [{"name": "tiny", "dims": {"n": 10, "n_bins": 4}, "classification": "pass", "pattern": "", "nondeterministic": False,
                       "max_abs_err": 0.0, "max_tol_ratio": None, "runs": [{"elapsed_ms": 0.5, "n_unwritten": 0}]}],
        }
        doc.update(over)
        return doc

    @staticmethod
    def bench_doc(**over: Any) -> dict[str, Any]:
        doc: dict[str, Any] = {"complete": True, "classification": "pass", "error_detail": "", "warmup_ms": [], "timed_ms": [12.5]}
        doc.update(over)
        return doc

    def check_with(self, *results: tuple[ProcResult, dict[str, Any] | None], bench: bool = False, lang: Any = None):
        settings = devcheck.load_settings(self.config)
        tctx = devcheck.load_task("histogram", self.tasks)
        lang = lang or devcheck.LanguageInfo(name="baseline", backend="python", source="baseline")
        req = devcheck.CheckRequest(label="histogram/x", tctx=tctx, lang=lang, tests=["tiny"], bench=bench,
                                    artifact=tctx.dir / "baseline" / "baseline.py")
        with mock.patch.object(devcheck, "run_worker", side_effect=list(results)) as run_worker:
            out = devcheck.check(req, settings, self.tmp)
        return out, run_worker.call_args_list, settings, tctx

    def test_pass_without_bench_runs_one_check_worker(self) -> None:
        out, calls, settings, tctx = self.check_with((_proc(0), self.check_doc()))
        self.assertEqual((out.status, out.stage, out.classification, out.detail, out.bench_ms, out.toolchain),
                         ("pass", "check", "pass", "", None, ""))
        self.assertEqual(len(calls), 1)
        kwargs = calls[0].kwargs
        self.assertEqual((kwargs["mode"], kwargs["tests"], kwargs["args_mode"], kwargs["timeout_s"]),
                         ("check", [tctx.tests_dir / "tiny.npz"], "positional", settings.check_timeout_s))
        self.assertEqual(kwargs["extra"], ["--check-repeats", str(settings.check_repeats)])
        self.assertTrue(Path(kwargs["artifact"]).is_absolute())

    def test_load_failure_is_reported_at_the_build_stage(self) -> None:
        doc = self.check_doc(load_ok=False, classification="signature_mismatch", load_error="module has no attribute histogram", tests=[])
        out, calls, _, _ = self.check_with((_proc(0), doc), bench=True)
        self.assertEqual((out.status, out.stage, out.classification, out.detail),
                         ("fail", "build", "signature_mismatch", "module has no attribute histogram"))
        self.assertEqual(len(calls), 1, "로드에 실패하면 bench 를 돌리지 않는다")

    def test_extraction_preprocesses_with_the_configured_compiler(self) -> None:
        """extract 의 전처리 컴파일러(``cc``)는 설정의 build.cc 다(harness.run 과 같다)."""
        settings = devcheck.load_settings(self.config)
        settings.cc = "clang-18"
        tctx = devcheck.load_task("histogram", self.tasks)
        lang = devcheck.LanguageInfo(name="c_seq", backend="c", threads=1)
        with mock.patch("harness.extract.extract", autospec=True,
                        return_value=SimpleNamespace(ok=False, code=None, detail="no_code", signature_ok=False)) as extract, \
                mock.patch.object(devcheck, "choose_c_toolchain", return_value="gcc"):
            out = devcheck.check(devcheck.CheckRequest(label="histogram/c_seq", tctx=tctx, lang=lang, tests=["tiny"],
                                                       response="no code here"), settings, self.tmp)
        self.assertEqual((out.status, out.stage), ("fail", "extract"))
        self.assertEqual(extract.call_args.kwargs["cc"], "clang-18")

    HALIDE = devcheck.LanguageInfo(name="halide_manual", backend="halide", threads=2, halide_mode="manual")

    def test_halide_runs_an_import_worker_first_like_the_pipeline(self) -> None:
        """회귀: devcheck 는 halide 의 정의·스케줄 오류를 check 의 runtime_crash 로 적었는데 harness.run 은 import 워커의 로드
        실패(build/compile_error, Func 가 아닌 반환값은 build/signature_mismatch)로 적는다. 같은 import 단계를 먼저 돈다."""
        import_ok = {"mode": "import", "complete": True, "load_ok": True, "classification": "pass"}
        out, calls, settings, tctx = self.check_with((_proc(0), import_ok), (_proc(0), self.check_doc()), lang=self.HALIDE)
        self.assertEqual((out.status, out.stage, out.classification), ("pass", "check", "pass"))
        self.assertEqual([c.kwargs["mode"] for c in calls], ["import", "check"])
        self.assertEqual((calls[0].kwargs["tests"], calls[0].kwargs["extra"], calls[0].kwargs["timeout_s"]),
                         ([tctx.tests_dir / "tiny.npz"], [], settings.check_timeout_s), "첫 정확성 형태로 파이프라인을 만든다")
        for classification, detail in (("compile_error", "HalideError: Error: could not find split dimension: zz"),
                                       ("signature_mismatch", "pipeline builder returned int, not a halide Func")):
            doc = {**import_ok, "complete": True, "load_ok": False, "classification": classification, "load_error": detail}
            out, calls, _, _ = self.check_with((_proc(0), doc), lang=self.HALIDE)
            self.assertEqual((out.status, out.stage, out.classification, out.detail), ("fail", "build", classification, detail))
            self.assertEqual(len(calls), 1, "로드에 실패하면 check 를 돌리지 않는다")
            self.assertEqual([s.verdict for s in out.shapes], ["NOT-RUN"])
        # 로드 도중 워커가 죽었다: 그 판정을 build 단계로
        dying = {**import_ok, "complete": False, "load_ok": None, "in_progress": {"test": "tiny", "call_idx": None}}
        out, _, _, _ = self.check_with((_proc(-11, signal_name="SIGSEGV"), dying), lang=self.HALIDE)
        self.assertEqual((out.status, out.stage, out.classification), ("fail", "build", "runtime_crash"))
        # 오토스케줄러 플러그인이 없으면 커널의 잘못이 아니라 도구 문제다
        missing = {**import_ok, "load_ok": False, "classification": "compile_error",
                   "load_error": "halide_autoscheduler_unavailable:Mullapudi2016"}
        with self.assertRaises(devcheck.DevcheckProblem):
            self.check_with((_proc(0), missing), lang=self.HALIDE)

    def test_contract_flags_without_a_failing_classification_still_fail(self) -> None:
        for flag, detail in (("oob_write", "out_of_bounds_write"), ("inputs_mutated", "inputs_mutated")):
            out, _, _, _ = self.check_with((_proc(0), self.check_doc(**{flag: True})))
            self.assertEqual((out.status, out.stage, out.classification, out.detail), ("fail", "check", "wrong_output", detail))

    def test_worker_that_wrote_no_result_is_a_tool_problem(self) -> None:
        usage = "python -m harness.worker: error: argument --backend: invalid choice: 'numba'"
        out, _, _, _ = self.check_with((_proc(2, stderr=usage), None))
        self.assertEqual((out.status, out.stage, out.classification), ("problem", "check", "worker_failed"))
        self.assertEqual([s.verdict for s in out.shapes], ["NOT-RUN"])

    def test_bench_runs_the_performance_shape_once_without_warmup(self) -> None:
        out, calls, settings, tctx = self.check_with((_proc(0), self.check_doc()), (_proc(0), self.bench_doc()), bench=True)
        self.assertEqual((out.status, out.stage, out.classification, out.bench_ms), ("pass", "check", "pass", 12.5))
        self.assertEqual(out.final_line(), "histogram/x => PASS stage=check classification=pass bench_large_ms=12.500")
        kwargs = calls[1].kwargs
        self.assertEqual((kwargs["mode"], kwargs["tests"], kwargs["timeout_s"]),
                         ("bench", [tctx.tests_dir / "large.npz"], settings.bench_timeout_s))
        self.assertEqual(kwargs["extra"], ["--warmup", "0", "--repeats", "1", "--min-repeats", "1", "--budget-s", str(settings.bench_timeout_s)])

    def test_bench_budget_timeout_is_a_note_not_a_failure(self) -> None:
        bench = self.bench_doc(classification="timeout", error_detail="first_call_5000ms_projected_10s", timed_ms=[5000.0])
        out, _, _, _ = self.check_with((_proc(0), self.check_doc()), (_proc(0), bench), bench=True)
        self.assertEqual((out.status, out.stage, out.classification, out.bench_ms), ("pass", "check", "pass", 5000.0))
        self.assertTrue(any(n.startswith("bench: first_call_5000ms_projected_10s") for n in out.notes), out.notes)

    def test_bench_failures(self) -> None:
        wrong = self.bench_doc(classification="wrong_output", error_detail="mismatch")
        out, _, _, _ = self.check_with((_proc(0), self.check_doc()), (_proc(0), wrong), bench=True)
        self.assertEqual((out.status, out.stage, out.classification, out.detail), ("fail", "bench", "wrong_output", "mismatch"))
        partial = {"complete": False, "in_progress": {"test": "large", "call_idx": 0}}
        out, _, _, _ = self.check_with((_proc(0), self.check_doc()), (_proc(1, timed_out=True), partial), bench=True)
        self.assertEqual((out.status, out.stage, out.classification), ("fail", "bench", "timeout"))
        self.assertIn("during large call 0", out.detail)


def _c_toolchain() -> str | None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return devcheck.choose_c_toolchain(devcheck.load_settings(ROOT / "config" / "experiment.yaml"))
    except devcheck.DevcheckProblem:
        return None


@unittest.skipIf(_c_toolchain() is None, "C 툴체인이 없다(Linux 의 gcc 또는 ziglang 패키지 필요)")
class CKernelEndToEndTest(EndToEndBase):
    """C 커널: Linux 는 설정의 gcc 플래그, 그 밖은 zig(직렬, shim.h·omp.h 스텁). int32 포인터 인자로 워커까지 간다.

    zig 로 빌드한 판정에는 ``toolchain=zig-serial`` 표시가 붙는다(논리 검증만 — OpenMP 병렬 실행은 확인하지 않았다)."""

    # 판정 줄의 빌드 방식 표시(이 머신에서 고르는 툴체인에 따라)
    MARKER = " toolchain=zig-serial" if _c_toolchain() == "zig" else ""

    def test_c_kernel_md_passes_on_three_shapes(self) -> None:
        rc, out, err = self.run_kernel(MD_C_CORRECT, "ok_c.md", language="c_openmp")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(shape_verdict_words(out), ["PASS", "PASS", "PASS"], out)
        self.assertEqual(out.strip().splitlines()[-1], "histogram/c_openmp => PASS stage=check classification=pass" + self.MARKER)
        if _c_toolchain() == "zig":
            self.assertIn("zig cc, serial", err)

    def test_c_syntax_error_is_a_build_failure(self) -> None:
        rc, out, _ = self.run_kernel(MD_C_SYNTAX_ERROR, "bad_c.md", language="c_seq")
        self.assertEqual(rc, 1, out)
        self.assertEqual(shape_verdict_words(out), ["NOT-RUN", "NOT-RUN", "NOT-RUN"], out)
        self.assertIn(f"=> FAIL stage=build classification=compile_error{self.MARKER} detail=kernel.c:", out)

    def test_dummies_mark_rows_that_passed_with_the_serial_build(self) -> None:
        """회귀: zig 직렬 빌드로 통과한 c_openmp 행이 표·요약에서 gcc+OpenMP 로 검증한 행과 구별되지 않았다."""
        dummies = self.dummy_dir({"histogram/numpy/0.md": MD_CORRECT, "histogram/c_openmp/0.md": MD_C_CORRECT})
        # 저장소 설정을 쓴다(gcc 경로는 유효한 전체 설정이 필요하다). 언어를 직접 주므로 설정의 languages 와 무관하다.
        rc, out, err = self.run_devcheck("dummies", "--dummy-dir", str(dummies), "--languages", "numpy,c_openmp", "--tests", "tiny")
        self.assertEqual(rc, 0, out + err)
        rows = table_rows(out, "histogram")
        self.assertRegex(rows[0], r"^histogram\s+numpy\s+PASS\s+pass$")
        summary = out.strip().splitlines()[-1]
        if _c_toolchain() == "zig":
            self.assertRegex(rows[1], r"^histogram\s+c_openmp\s+PASS\s+pass\s+toolchain=zig-serial$")
            self.assertEqual(summary, "summary: 2 pass, 0 fail, 0 missing, 0 problem, 0 unsupported; 1 of the pass rows used "
                                      "toolchain=zig-serial (serial build: logic only, OpenMP pragmas ignored)")
        else:
            self.assertRegex(rows[1], r"^histogram\s+c_openmp\s+PASS\s+pass$")
            self.assertEqual(summary, "summary: 2 pass, 0 fail, 0 missing, 0 problem, 0 unsupported")


@unittest.skipUnless(devcheck.ziglang_available(), "ziglang 패키지 필요(개발용 venv)")
class ZigBuildTest(unittest.TestCase):
    def test_zig_build_writes_the_args_header_and_marks_the_toolchain(self) -> None:
        header = sigs.render_args_struct(HISTOGRAM_SPEC, "c")
        with tempfile.TemporaryDirectory(prefix="devcheck_zig_") as tmp:
            built = devcheck.build_c_zig(C_KEYWORD, Path(tmp), header)
            self.assertTrue(built.ok, built.error_detail)
            self.assertEqual(built.toolchain, "zig-serial")
            self.assertIn("serial", built.note)
            self.assertEqual((Path(tmp) / "kernel_args.h").read_text(encoding="utf-8"), header)
            self.assertTrue(built.artifact is not None and built.artifact.is_file())
            broken = devcheck.build_c_zig("int broken(", Path(tmp), None)
            self.assertEqual((broken.ok, broken.artifact, broken.error_kind, broken.toolchain), (False, None, "compile_error", "zig-serial"))
            self.assertIn("kernel.c:", broken.error_detail)


def _zig_omp_available() -> bool:
    if not devcheck.ziglang_available():
        return False
    try:
        devcheck.libomp_runtime()
    except devcheck.DevcheckProblem:
        return False
    return True


def _team_size_kernel_md(expected_team: int) -> str:
    """병렬 영역의 팀 크기가 ``expected_team`` 일 때만 맞는 히스토그램을 쓰는 커널(아니면 전부 0 → wrong_output).
    OpenMP 없이 빌드되면 컴파일 오류다."""
    code = textwrap.dedent(
        f"""
        #include <omp.h>
        #ifndef _OPENMP
        #error "built without OpenMP"
        #endif

        void histogram(const int* keys, int* hist, int n, int n_bins)
        {{
            int team = 0;
            #pragma omp parallel
            {{
                #pragma omp single
                team = omp_get_num_threads();
            }}
            for (int b = 0; b < n_bins; b++) hist[b] = 0;
            if (team != {expected_team}) return;
            for (int i = 0; i < n; i++) hist[keys[i]]++;
        }}
        """
    )
    return "Counts when the team has the expected size.\n\n```c\n" + code.strip() + "\n```\n"


@unittest.skipUnless(_zig_omp_available(), "ziglang 패키지와 LLVM OpenMP 런타임 필요(Windows 개발 머신)")
class ZigOmpEndToEndTest(EndToEndBase):
    """``--cc zig-omp``: OpenMP 대상을 clang ``-fopenmp`` + LLVM OpenMP 런타임으로 빌드해 실제 스레드로 돌린다."""

    # 설정(CONFIG_YAML)의 measurement.threads = 2 가 c_openmp 의 스레드 수다(코어가 하나뿐인 머신에서는 1)
    THREADS = max(1, min(2, os.cpu_count() or 1))

    def test_openmp_kernel_runs_with_a_real_team(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEVCHECK_OMP_THREADS", None)
            rc, out, err = self.run_kernel(_team_size_kernel_md(self.THREADS), "team.md", "--cc", "zig-omp", "--config", str(self.config),
                                           language="c_openmp")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(out.strip().splitlines()[-1], "histogram/c_openmp => PASS stage=check classification=pass toolchain=zig-omp")
        self.assertIn(f"OMP_NUM_THREADS={self.THREADS}", err)
        self.assertIn("real threads", err)

    def test_thread_count_override(self) -> None:
        with mock.patch.dict(os.environ, {"DEVCHECK_OMP_THREADS": "3"}):
            rc, out, err = self.run_kernel(_team_size_kernel_md(3), "team3.md", "--cc", "zig-omp", language="c_openmp")
            self.assertEqual(rc, 0, out + err)
            self.assertIn("OMP_NUM_THREADS=3", err)
            # 팀 크기가 기대와 다르면 커널이 0 만 쓴다 → 실제로 3개짜리 팀이었다는 반대쪽 확인
            rc, out, err = self.run_kernel(_team_size_kernel_md(5), "team5.md", "--cc", "zig-omp", language="c_openmp")
            self.assertEqual(rc, 1, out + err)
            self.assertIn("=> FAIL stage=check classification=wrong_output toolchain=zig-omp", out)

    def test_targets_built_without_openmp_stay_serial(self) -> None:
        # c_seq 는 하네스에서도 -fopenmp 없이 빌드한다: zig-omp 를 줘도 직렬 스텁 빌드다
        rc, out, err = self.run_kernel(MD_C_CORRECT, "seq.md", "--cc", "zig-omp", language="c_seq")
        self.assertEqual(rc, 0, out + err)
        self.assertEqual(out.strip().splitlines()[-1], "histogram/c_seq => PASS stage=check classification=pass toolchain=zig-serial")

    def test_compile_error_is_a_build_failure(self) -> None:
        rc, out, _ = self.run_kernel(MD_C_SYNTAX_ERROR, "bad.md", "--cc", "zig-omp", language="c_openmp")
        self.assertEqual(rc, 1, out)
        self.assertIn("=> FAIL stage=build classification=compile_error toolchain=zig-omp detail=kernel.c:", out)

    def test_dummies_summary_names_the_real_thread_build(self) -> None:
        dummies = self.dummy_dir({"histogram/c_openmp/0.md": MD_C_CORRECT})
        rc, out, err = self.run_devcheck("dummies", "--config", str(self.config), "--dummy-dir", str(dummies), "--languages", "c_openmp",
                                         "--tests", "tiny", "--cc", "zig-omp")
        self.assertEqual(rc, 0, out + err)
        self.assertRegex(table_rows(out, "histogram")[0], r"^histogram\s+c_openmp\s+PASS\s+pass\s+toolchain=zig-omp$")
        self.assertEqual(out.strip().splitlines()[-1],
                         "summary: 1 pass, 0 fail, 0 missing, 0 problem, 0 unsupported; 1 of the pass rows used toolchain=zig-omp "
                         "(real threads with clang + LLVM OpenMP runtime: not gcc/libgomp)")


if __name__ == "__main__":
    unittest.main()
