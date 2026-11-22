"""생성 단계(harness/generate.py)와 프롬프트(prompts/lang, prompts/hints) 테스트.

- 기존 36개 조합(12개 과제 × numpy·c_seq·c_openmp, positional·hint off)의 ``prompt_hash`` 는
  ``tests/fixtures/prompt_hashes_upstream.json`` 의 값(= ``results/runs.csv`` 의 실제 실행 행의 해시)과 같아야 한다
  (docs/decisions.md 2026-11-23). 머신 프로필은 그 해시를 만든 ``config/machines/…`` 파일을 명시해 읽는다(호스트 무관).
- 조건(``args``·``grammar_hint``·``feedback``), 지원하지 않는 조합·힌트 없는 언어 건너뛰기, 새 언어 템플릿의 렌더링(12개 과제),
  환경 값의 출처(머신 프로필 ``config/machines/<host>.yaml`` → 조사 → ``unknown``).
- 프롬프트가 사실로 적는 것이 하네스가 실제로 하는 것과 같은지: 워커 환경변수(``protocol.launch_env``), Rust 빌드 템플릿.

``main`` 은 임시 디렉터리의 과제·더미 응답·설정으로 돌린다(dummy 제공자; 저장소의 ``results/`` 는 건드리지 않는다).
실제 API 호출·네트워크 없음.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from harness import generate, llm_client, protocol, sigs
from harness.build import rust as rust_build
from harness.build.halide import SCHEDULE_METHODS
from harness.config import Config, ConfigError, load_config, load_machine
from harness.llm_client import LLMResponse
from harness.paths import DEFAULT_EXPERIMENT_YAML, PROMPTS_DIR, PYTHON_EXE, ROOT, TASKS_DIR
from harness.procutil import run_isolated
from harness.taskspec import load_taskspec
from tests._support import requires_modules, requires_rust_tool

HASH_FIXTURE = ROOT / "tests" / "fixtures" / "prompt_hashes_upstream.json"
# 기록된 prompt_hash 를 만든 머신 프로필(호스트 이름과 무관하게 이 파일을 읽는다)
MACHINE_PROFILE = ROOT / "config" / "machines" / "ustar-Zenbook-UX3402ZA-UX3402ZA.yaml"
UP_TASKS = ("gemm", "jacobi2d", "spmv_csr", "dot", "histogram", "scan", "nbody", "sort", "cholesky", "floyd_warshall", "bfs", "fft")
HALIDE_UNSUPPORTED_TASKS = ("sort", "bfs")
EXISTING_LANGUAGES = ("numpy", "c_seq", "c_openmp")
NEW_LANGUAGES = ("numba", "halide_auto", "halide_manual", "rust")
ALL_LANGUAGES = (*EXISTING_LANGUAGES, *NEW_LANGUAGES)
SECTION_HEADINGS = ["## Environment", "## Function to implement", "## Execution environment", "## Output format"]
# 설계 v3 §5: 모든 언어 템플릿에 제공되는 변수(기존 것 + v3 에서 더한 것)
V3_TEMPLATE_VARIABLES = (
    "signature", "indexing", "param_docs", "function", "python_version", "numpy_version", "cc", "cc_version", "build_command",
    "threads", "cores_desc", "cpu_summary", "task_spec",
    "numba_signature", "numba_version", "halide_version", "halide_autoscheduler", "rustc_version", "cargo_command", "rust_profile",
    "args_note", "grammar_hint",
)
# machine.yaml 이 M3 도구까지 프로브했을 때 더 갖는 키(harness.protocol.probe_machine)
M3_MACHINE_KEYS = {"numba_version": "0.68.1", "halide_version": "22.1.0", "rustc": "rustc 1.92.0 (ded5c06cf 2025-12-08)"}
RAW_KEYS = [
    "run_id", "task", "language", "backend", "sample_idx", "turn", "conditions", "provider", "model_requested", "model_served",
    "sampling", "max_tokens", "prompt", "prompt_hash", "response_text", "stop_reason", "raw_stop_reason", "stop_details_category",
    "n_thinking_blocks", "n_text_blocks", "tokens_in", "tokens_out", "tokens_thinking", "tokens_text", "latency_ms", "request_id",
    "base_url", "api_key_present", "sdk_version", "error", "created_at", "harness_version", "config_snapshot",
]
_ENV_FACT_RE = re.compile(r"`([A-Z][A-Z0-9_]*)=([^`\s]+)`")
_FENCE_RE = re.compile(r"^```(\w*)\n(.*?)^```$", re.M | re.S)


def hint_fragments(language: str) -> list[tuple[str, str]]:
    """``prompts/hints/<language>.md`` 의 펜스 코드 블록: (태그, 코드) 목록."""
    return _FENCE_RE.findall((PROMPTS_DIR / "hints" / f"{language}.md").read_text(encoding="utf-8"))


def repo_config(**overrides: str) -> Config:
    """저장소의 ``config/experiment.yaml`` (환경변수의 제공자·모델 덮어쓰기는 무시)."""
    with mock.patch.dict(os.environ):
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("LLM_MODEL", None)
        return load_config(overrides=overrides or None)


def profile_machine() -> dict[str, Any]:
    """기록된 실행의 머신 프로필(``MACHINE_PROFILE``)."""
    return load_machine(MACHINE_PROFILE)


def complete_machine() -> dict[str, Any]:
    """머신 프로필에 M3 도구의 버전을 더한 것(= M3 까지 프로브한 머신)."""
    return {**profile_machine(), **M3_MACHINE_KEYS}


def language_part(user: str) -> str:
    """렌더링한 프롬프트에서 과제 명세(마지막 ``---`` 뒤) 앞의 언어 템플릿 부분."""
    head, separator, _ = user.rpartition("\n---\n")
    assert separator, "프롬프트에 '---' 구분 줄이 없다"
    return head


@contextlib.contextmanager
def no_probing() -> Any:
    """환경 값 조사(서브프로세스·도구 실행)가 일어나면 실패하게 한다: 값이 전부 machine.yaml 에서 와야 하는 테스트용."""
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"환경을 조사하면 안 된다: {args} {kwargs}")

    with mock.patch.object(generate, "env_identity_subprocess", forbidden), \
            mock.patch.object(generate, "python_toolchain_subprocess", forbidden), \
            mock.patch.object(generate, "compiler_version", forbidden):
        yield


# ----------------------------------------------------------------------------- 기존 프롬프트의 바이트 동일성


class PromptHashTest(unittest.TestCase):
    """기존 36개 조합의 프롬프트는 기록된 실제 실행(results/runs.csv)의 프롬프트와 바이트 단위로 같다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(HASH_FIXTURE.read_text(encoding="utf-8"))
        cls.cfg = repo_config()
        cls.machine = profile_machine()

    def test_fixture_covers_the_thirty_six_combinations(self) -> None:
        expected = {f"{task}/{language}" for task in UP_TASKS for language in EXISTING_LANGUAGES}
        self.assertEqual(set(self.fixture["hashes"]), expected)
        self.assertEqual(set(self.fixture["user_sha256"]), expected)
        self.assertEqual(ROOT / self.fixture["machine_profile"], MACHINE_PROFILE)
        self.assertTrue(MACHINE_PROFILE.is_file())
        self.assertEqual(tuple(self.cfg.run.tasks), UP_TASKS)

    def test_fixture_hashes_are_those_of_the_recorded_runs(self) -> None:
        """기준값은 results/runs.csv 의 실제 실행 행(dummy 가 아닌 run)의 prompt_hash 와 같다(파일이 있을 때만)."""
        runs = ROOT / "results" / "runs.csv"
        if not runs.is_file():
            self.skipTest("results/runs.csv 가 없다")
        import csv
        with runs.open("r", encoding="utf-8", newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("provider") not in ("dummy", "")]
        recorded: dict[str, set[str]] = {}
        for r in rows:
            key = f"{r['task']}/{r['language']}"
            if key in self.fixture["hashes"] and r.get("args", "positional") == "positional" and r.get("grammar_hint", "off") == "off":
                recorded.setdefault(key, set()).add(r["prompt_hash"])
        self.assertEqual(set(recorded), set(self.fixture["hashes"]))
        for key, hashes in recorded.items():
            with self.subTest(key=key):
                self.assertEqual(hashes, {self.fixture["hashes"][key]})

    def test_default_condition_reproduces_the_up_hashes(self) -> None:
        self.assertEqual((self.cfg.run.conditions.args, self.cfg.run.conditions.grammar_hint), ("positional", "off"))
        for key, expected in self.fixture["hashes"].items():
            task, language = key.split("/")
            with self.subTest(key=key), no_probing():
                rendered = generate.render_prompt(self.cfg, self.machine, task, self.cfg.languages[language], args_mode="positional",
                                                  grammar_hint="off")
                self.assertEqual(rendered.prompt_hash, expected)
                self.assertEqual(hashlib.sha256(rendered.user.encode("utf-8")).hexdigest(), self.fixture["user_sha256"][key])
                self.assertNotIn("\r", rendered.system + rendered.user)

    def test_prompt_hash_definition(self) -> None:
        digest = hashlib.sha256("SYSTEM\n\x00\nUSER".encode("utf-8")).hexdigest()[:16]
        self.assertEqual(generate.prompt_hash("SYSTEM", "USER"), digest)

    def test_existing_templates_carry_exactly_the_two_new_variables(self) -> None:
        for language in EXISTING_LANGUAGES:
            with self.subTest(language=language):
                text = (PROMPTS_DIR / "lang" / f"{language}.md").read_text(encoding="utf-8")
                self.assertEqual(text.count("${args_note}"), 1)
                self.assertEqual(text.count("${grammar_hint}"), 1)
                self.assertIn("${param_docs}${args_note}\n", text)  # 매개변수 설명 바로 뒤
                self.assertIn("\n${grammar_hint}---\n\n${task_spec}", text)  # 마지막 구분 줄 바로 앞
                self.assertEqual(text.count("\n---\n") + text.count("}---\n"), 1)


# ----------------------------------------------------------------------------- 조건: args, grammar_hint


class ConditionRenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = repo_config()
        cls.machine = complete_machine()
        cls.spec = load_taskspec(TASKS_DIR / "gemm" / "signatures.yaml")

    def render(self, language: str, *, args_mode: str = "positional", grammar_hint: str = "off", task: str = "gemm") -> str:
        with no_probing():
            return generate.render_prompt(self.cfg, self.machine, task, self.cfg.languages[language], args_mode=args_mode,
                                          grammar_hint=grammar_hint).user

    def bundle(self, language: str, args_mode: str) -> dict[str, Any]:
        return sigs.signature_bundle(self.spec, language, self.cfg.languages[language].backend, args_mode)

    def test_positional_has_no_args_note(self) -> None:
        for language in ALL_LANGUAGES:
            with self.subTest(language=language):
                bundle = self.bundle(language, "positional")
                self.assertEqual(generate.render_args_note(bundle, self.cfg.languages[language].backend, "positional"), "")
                self.assertIn(bundle["param_docs"] + "\n\n", self.render(language))

    def test_keyword_note_follows_the_parameter_docs_after_one_blank_line(self) -> None:
        for language in ALL_LANGUAGES:
            with self.subTest(language=language):
                bundle = self.bundle(language, "keyword")
                note = generate.render_args_note(bundle, self.cfg.languages[language].backend, "keyword")
                self.assertTrue(note.startswith("\n\n") and not note.startswith("\n\n\n"), repr(note[:10]))
                self.assertFalse(note.endswith("\n"))
                user = self.render(language, args_mode="keyword")
                self.assertIn(bundle["signature"], user)
                self.assertIn(bundle["param_docs"] + note + "\n\n", user)

    def test_keyword_note_of_python_family_states_the_call_by_name(self) -> None:
        calls = {"numpy": "gemm(A=..., B=..., C=..., M=..., N=..., K=...)", "numba": "gemm(A=..., B=..., C=..., M=..., N=..., K=...)",
                 "halide_auto": "gemm(A=..., B=..., M=..., N=..., K=...)", "halide_manual": "gemm(A=..., B=..., M=..., N=..., K=...)"}
        for language, call in calls.items():
            with self.subTest(language=language):
                user = self.render(language, args_mode="keyword")
                self.assertIn("(*, ", user.split("```python\n")[1].split("```")[0])
                self.assertIn(f"The harness passes all arguments by name, as `{call}`, never by position.", user)
                self.assertNotIn("kernel_args", user)

    def test_keyword_note_of_c_shows_the_header_the_harness_writes(self) -> None:
        header = sigs.render_args_struct(self.spec, "c")
        for language in ("c_seq", "c_openmp"):
            with self.subTest(language=language):
                user = self.render(language, args_mode="keyword")
                self.assertIn("void gemm(const gemm_args* args)", user)
                self.assertIn(f"`kernel_args.h` into the directory of `kernel.c` before compiling:\n\n```c\n{header}```\n\n", user)
                self.assertIn('`#include "kernel_args.h"`', user)

    def test_keyword_note_of_rust_shows_the_module_the_harness_writes(self) -> None:
        module = sigs.render_args_struct(self.spec, "rust")
        user = self.render("rust", args_mode="keyword")
        self.assertIn("fn gemm(args: *const GemmArgs)", user)
        self.assertIn(f"`src/kernel_args.rs` next to `src/lib.rs` before building:\n\n```rust\n{module}```\n\n", user)
        self.assertIn("`mod kernel_args;`", user)

    def test_grammar_hint_is_the_hint_file_under_its_heading_before_the_separator(self) -> None:
        for language in NEW_LANGUAGES:
            with self.subTest(language=language):
                hint = (PROMPTS_DIR / "hints" / f"{language}.md").read_text(encoding="utf-8").strip("\n")
                off, on = self.render(language), self.render(language, grammar_hint="on")
                section = f"## Language notes\n\n{hint}\n\n"
                self.assertEqual(generate.render_grammar_hint(language), section)
                self.assertNotIn("## Language notes", off)
                self.assertEqual(on, off.replace("\n---\n", f"\n{section}---\n", 1))  # 두 조건은 이 절만 다르다

    def test_grammar_hint_without_a_hint_file_cannot_be_rendered(self) -> None:
        for language in EXISTING_LANGUAGES:
            with self.subTest(language=language):
                self.assertFalse(generate.hint_path(language).exists())
                with self.assertRaises(ConfigError):
                    self.render(language, grammar_hint="on")

    def test_every_condition_has_its_own_prompt(self) -> None:
        hashes = set()
        for args_mode in ("positional", "keyword"):
            for grammar_hint in ("off", "on"):
                with no_probing():
                    hashes.add(generate.render_prompt(self.cfg, self.machine, "gemm", self.cfg.languages["rust"], args_mode=args_mode,
                                                      grammar_hint=grammar_hint).prompt_hash)
        self.assertEqual(len(hashes), 4)


# ----------------------------------------------------------------------------- 새 언어 템플릿


class NewTemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = repo_config()
        cls.machine = complete_machine()
        with no_probing():
            cls.users = {language: generate.render_prompt(cls.cfg, cls.machine, "gemm", cls.cfg.languages[language],
                                                          args_mode="positional", grammar_hint="off").user
                         for language in ALL_LANGUAGES}

    def test_nothing_is_missing_or_unknown_when_the_machine_file_is_complete(self) -> None:
        spec_text = (TASKS_DIR / "gemm" / "spec.md").read_text(encoding="utf-8").rstrip("\n")
        for language in NEW_LANGUAGES:
            with self.subTest(language=language):
                user = self.users[language]
                head = language_part(user)
                self.assertNotIn("$", head)
                self.assertNotIn("unknown", head.lower())
                self.assertTrue(user.endswith("\n---\n\n" + spec_text + "\n"))

    def test_same_sections_as_the_existing_templates(self) -> None:
        for language in ALL_LANGUAGES:
            with self.subTest(language=language):
                head = language_part(self.users[language])
                self.assertTrue(head.startswith("# Language: "))
                self.assertEqual([line for line in head.splitlines() if line.startswith("## ")], SECTION_HEADINGS)

    def test_hardware_and_thread_facts_are_the_same_for_every_language(self) -> None:
        """정보 균형: 하드웨어 줄은 스레드 수가 같은 언어끼리 글자까지 같고, 모든 언어가 고정 코어·스레드 수를 적는다."""
        hardware: dict[int, set[str]] = {}
        for language in ALL_LANGUAGES:
            lang = self.cfg.languages[language]
            lines = language_part(self.users[language]).splitlines()
            hardware.setdefault(lang.threads, set()).add(next(line for line in lines if line.startswith("- Hardware: ")))
            pinned = next(line for line in lines if line.startswith("- The process is pinned to "))
            self.assertIn(f"Thread count: {lang.threads}.", pinned, language)
        self.assertTrue(all(len(lines) == 1 for lines in hardware.values()), hardware)

    def test_numba_facts(self) -> None:
        head = language_part(self.users["numba"])
        self.assertIn("- Python 3.14.4, NumPy 2.3.5, Numba 0.68.1.", head)
        self.assertIn("Use only the Python standard library, NumPy and Numba", head)
        self.assertIn('@njit("void(float32[:, ::1], float32[:, ::1], float32[:, ::1], int64, int64, int64)")\ndef gemm(A, B, C, M, N, K):', head)
        self.assertIn("called with exactly the signature string shown below as its first argument", head)
        self.assertIn("Keyword options of the decorator, after the signature string, are your choice, except that `cache=True` is not "
                      "allowed.", head)
        # 모듈의 다른 njit/jit 함수도 명시 서명이 있어야 하고 cache=True 는 거절된다(docs/decisions.md 2026-11-23)
        self.assertIn("Every other function in the module that is decorated with `njit` or `jit` must also be given an explicit "
                      "signature string as the first positional argument of its decorator, so that nothing is compiled during a call.", head)
        self.assertIn("A compiled function without one, or any use of `cache=True`, is rejected before the module is imported.", head)
        self.assertIn("Numba compiles `gemm` when the module is imported", head)
        self.assertIn("is not part of the measured time", head)
        self.assertIn("writable, C-contiguous NumPy array of the stated dtype (float32 or int32)", head)
        self.assertIn("every size as a Python `int`", head)
        # OpenMP 환경은 c_openmp 프롬프트와 같은 문장으로 적는다(numba 는 libgomp 를 쓴다)
        self.assertIn("The environment sets `NUMBA_NUM_THREADS=4`, `NUMBA_THREADING_LAYER=omp`, `OMP_NUM_THREADS=4`, `OMP_PROC_BIND=true`, "
                      "`OMP_PLACES=cores` and `OMP_DYNAMIC=false`.", head)
        c_openmp = language_part(self.users["c_openmp"])
        self.assertIn("`OMP_NUM_THREADS=4`, `OMP_PROC_BIND=true`, `OMP_PLACES=cores` and `OMP_DYNAMIC=false`.", c_openmp)

    def test_halide_facts(self) -> None:
        for language in ("halide_auto", "halide_manual"):
            with self.subTest(language=language):
                head = language_part(self.users[language])
                self.assertIn("Halide 22.1.0 (the `halide` Python package, imported as `import halide as hl`)", head)
                self.assertIn("```python\ndef gemm(A, B, M, N, K):\n```", head)
                self.assertIn("builds a new pipeline by calling `gemm` with an `hl.ImageParam` for every input array and the actual "
                              "sizes of that shape as Python `int` values", head)
                self.assertIn("`gemm` must return the `hl.Func` that defines the output array", head)
                # 좌표 규칙(설계 v3 §4)과 그 과제의 매개변수로 든 예
                self.assertIn("`hl.Buffer(array)` (default `reverse_axes=True`), so the last array axis is the first Halide coordinate", head)
                self.assertIn("the element at row i0 and column i1 is `X[i1, i0]`; a 1-D array is indexed as `x[i]`", head)
                self.assertIn("For example, the input `A` has shape (M, K): its element at array index (i0, i1), with 0 <= i0 < M and "
                              "0 <= i1 < K, is `A[i1, i0]`.", head)
                self.assertIn("compiles the pipeline for the host target", head)
                self.assertIn("one realization of the returned `hl.Func` over the full output shape, written directly into the output "
                              "buffer allocated by the caller", head)
                self.assertIn("are not measured; the measured time is the duration of one realization", head)
                self.assertIn("The environment sets `HL_NUM_THREADS=4`.", head)
        auto, manual = language_part(self.users["halide_auto"]), language_part(self.users["halide_manual"])
        # halide_auto: 모델의 스케줄 호출은 효과가 없고 기록만 된다(docs/decisions.md 2026-11-23); 기본 오토스케줄러는 Li2018
        self.assertEqual(self.cfg.halide.autoscheduler, "Li2018")
        self.assertIn("Write the algorithm only. The schedule comes from the autoscheduler: scheduling methods called on an `hl.Func` "
                      "or on one of its update stages, and calls of `apply_autoscheduler`, have no effect; the harness ignores them and "
                      "records that they were called.", auto)
        self.assertNotIn("do not call", auto)
        self.assertIn("sets the size estimates of every input and of the output to the actual sizes and applies the autoscheduler "
                      "`Li2018` with parallelism 4", auto)
        self.assertIn("No autoscheduler is applied", manual)
        self.assertIn("Scheduling calls, if any, are written inside `gemm`, before it returns.", manual)
        self.assertNotIn("Li2018", manual)
        self.assertNotIn("Mullapudi2016", manual)

    def test_halide_coordinate_example_uses_an_input_of_the_task(self) -> None:
        examples = {
            "gemm": "the input `A` has shape (M, K): its element at array index (i0, i1), with 0 <= i0 < M and 0 <= i1 < K, is `A[i1, i0]`.",
            "jacobi2d": "the input `src` has shape (H, W): its element at array index (i0, i1), with 0 <= i0 < H and 0 <= i1 < W, "
                        "is `src[i1, i0]`.",
            "dot": "the input `x` has shape (n,): its element at array index (i0), with 0 <= i0 < n, is `x[i0]`.",
            "spmv_csr": "the input `row_ptr` has shape (n_rows + 1,): its element at array index (i0), with 0 <= i0 < n_rows + 1, "
                        "is `row_ptr[i0]`.",
        }
        for task, sentence in examples.items():
            with self.subTest(task=task):
                spec = load_taskspec(TASKS_DIR / task / "signatures.yaml")
                self.assertEqual(generate.halide_coordinate_example(spec), "For example, " + sentence)

    def test_rust_facts(self) -> None:
        head = language_part(self.users["rust"])
        self.assertIn("- Rust, edition 2021. Compiler: `rustc 1.92.0 (ded5c06cf 2025-12-08)`.", head)
        self.assertIn('saved as `src/lib.rs` of a library crate that is built as a C-compatible shared library (`crate-type = ["cdylib"]`)', head)
        flags = " ".join(protocol.rust_flags(self.cfg.build.rust))
        self.assertIn(f'```\nRUSTFLAGS="{flags}" cargo build --release --offline --locked\n```', head)
        self.assertIn("the Rust standard library (`std`) and `rayon` ", head)
        self.assertIn("A panic aborts the process", head)
        self.assertIn('#[no_mangle]\npub unsafe extern "C" fn gemm(A: *const f32, B: *const f32, C: *mut f32, M: i32, N: i32, K: i32)', head)
        self.assertIn("looks up the symbol `gemm` by name and calls it through the C ABI", head)
        self.assertIn("Every pointer passed by the harness is non-null and valid", head)
        self.assertIn("for the number of elements stated above", head)
        self.assertIn("The environment sets `RAYON_NUM_THREADS=4`.", head)

    def test_rust_facts_are_those_of_the_build_template(self) -> None:
        """프롬프트의 에디션·크레이트 종류·release 프로파일·rayon 버전은 하네스가 빌드에 쓰는 ``Cargo.toml`` 의 것이다."""
        manifest = tomllib.loads(generate.RUST_MANIFEST.read_text(encoding="utf-8"))
        head = language_part(self.users["rust"])
        self.assertIn(f"- Rust, edition {manifest['package']['edition']}.", head)
        self.assertIn(f"`crate-type = {json.dumps(manifest['lib']['crate-type'])}`", head)
        self.assertEqual(list(manifest["dependencies"]), ["rayon"])
        self.assertIn(f"`rayon` {manifest['dependencies']['rayon'].lstrip('=')}.", head)
        profile = manifest["profile"]["release"]
        self.assertEqual(profile["panic"], "abort")
        stated = ", ".join(f"`{key} = {json.dumps(value)}`" for key, value in profile.items())
        self.assertIn(f"- Release profile: {stated}.", head)

    def test_stated_environment_variables_are_what_the_worker_gets(self) -> None:
        """템플릿이 사실로 적는 환경변수(`` `NAME=값` ``)는 ``protocol.launch_env`` 가 워커에 주는 값과 같다."""
        expected_names = {"numpy": {"OMP_NUM_THREADS"}, "c_seq": set(),
                          "c_openmp": {"OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_DYNAMIC"},
                          "numba": {"NUMBA_NUM_THREADS", "NUMBA_THREADING_LAYER", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES",
                                    "OMP_DYNAMIC"},
                          "halide_auto": {"HL_NUM_THREADS"},
                          "halide_manual": {"HL_NUM_THREADS"}, "rust": {"RAYON_NUM_THREADS"}}
        for language in ALL_LANGUAGES:
            with self.subTest(language=language):
                lang = self.cfg.languages[language]
                with contextlib.redirect_stderr(io.StringIO()):
                    env = protocol.launch_env(self.cfg, lang.threads, openmp=lang.omp_runtime, extra_env=lang.env)
                stated = dict(_ENV_FACT_RE.findall(language_part(self.users[language])))
                self.assertEqual(set(stated), expected_names[language])
                self.assertEqual(stated, {name: env.get(name) for name in stated})

    def test_every_v3_variable_is_available_to_every_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "all_variables.md"
            template.write_text("".join(f"{name}=[${{{name}}}]\n" for name in V3_TEMPLATE_VARIABLES), encoding="utf-8")
            for language in ALL_LANGUAGES:
                with self.subTest(language=language), no_probing():
                    lang = dataclasses.replace(self.cfg.languages[language], prompt=template)
                    user = generate.render_prompt(self.cfg, self.machine, "gemm", lang, args_mode="positional", grammar_hint="off").user
                    self.assertNotIn("${", user)
                    self.assertIn("numba_signature=[void(float32[:, ::1], float32[:, ::1], float32[:, ::1], int64, int64, int64)]", user)
                    self.assertIn("halide_autoscheduler=[Li2018]", user)
                    flags = " ".join(protocol.rust_flags(self.cfg.build.rust))
                    self.assertIn(f'cargo_command=[RUSTFLAGS="{flags}" cargo build --release --offline --locked]', user)
                    self.assertIn("args_note=[]\ngrammar_hint=[]", user)

    def test_new_language_prompts_render_for_every_task(self) -> None:
        """새 언어의 프롬프트는 12개 과제 모두에서 렌더링된다(halide 는 sort·bfs 를 지원하지 않는다)."""
        for task in UP_TASKS:
            spec = load_taskspec(TASKS_DIR / task / "signatures.yaml")
            for language in NEW_LANGUAGES:
                lang = self.cfg.languages[language]
                with self.subTest(task=task, language=language):
                    if lang.backend == "halide" and task in HALIDE_UNSUPPORTED_TASKS:
                        self.assertFalse(spec.supports(language))
                        continue
                    self.assertTrue(spec.supports(language))
                    with no_probing():
                        user = generate.render_prompt(self.cfg, self.machine, task, lang, args_mode="positional", grammar_hint="off").user
                    head = language_part(user)
                    self.assertNotIn("$", head)
                    self.assertNotIn("unknown", head.lower())
                    self.assertIn(sigs.signature_bundle(spec, language, lang.backend, "positional")["signature"], head)

    def test_unknown_template_variable_is_a_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "typo.md"
            template.write_text("${signature} ${numba_versoin}\n", encoding="utf-8")
            lang = dataclasses.replace(self.cfg.languages["numpy"], prompt=template)
            with self.assertRaises(ConfigError) as caught:
                generate.render_prompt(self.cfg, self.machine, "gemm", lang, args_mode="positional")
            self.assertIn("numba_versoin", str(caught.exception))

    def test_numba_template_needs_the_threading_layer_of_the_config(self) -> None:
        lang = self.cfg.languages["numba"]
        without = dataclasses.replace(lang, env={"NUMBA_NUM_THREADS": "4"})
        with self.assertRaises(ConfigError) as caught, no_probing():
            generate.render_prompt(self.cfg, self.machine, "gemm", without, args_mode="positional")
        self.assertIn("NUMBA_THREADING_LAYER", str(caught.exception))


# ----------------------------------------------------------------------------- 환경 값의 출처


class EnvironmentValuesTest(unittest.TestCase):
    """프롬프트가 적는 버전: machine.yaml 에 있으면 그 값, 없으면 조사, 조사도 안 되면 경고와 함께 ``unknown``."""

    def setUp(self) -> None:
        self.cfg = repo_config()

    def info(self, machine: dict[str, Any]) -> Any:
        return generate._EnvInfo(self.cfg, machine)

    def test_values_come_from_the_machine_file_without_probing(self) -> None:
        with no_probing():
            info = self.info(complete_machine())
            self.assertEqual((info.numba_version(), info.halide_version(), info.rustc_version()),
                             ("0.68.1", "22.1.0", "rustc 1.92.0 (ded5c06cf 2025-12-08)"))
            self.assertEqual((info.python_version(), info.numpy_version()), ("3.14.4", "2.3.5"))
            self.assertEqual(info.cc_version(), "gcc (Ubuntu 15.2.0-16ubuntu1) 15.2.0")

    def test_missing_values_are_probed_once_through_the_worker_interpreter(self) -> None:
        toolchain = {"numba_version": "0.99.0", "halide_version": "23.0.0", "halide_autoscheduler_loadable": True, "errors": {}}
        with mock.patch.object(generate, "python_toolchain_subprocess", return_value=toolchain) as probe, \
                mock.patch.object(generate, "compiler_version", return_value="rustc 9.9.9 (test)") as version, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            info = self.info({})
            self.assertEqual((info.numba_version(), info.halide_version(), info.rustc_version()), ("0.99.0", "23.0.0", "rustc 9.9.9 (test)"))
            self.assertEqual((info.numba_version(), info.halide_version(), info.rustc_version()), ("0.99.0", "23.0.0", "rustc 9.9.9 (test)"))
        probe.assert_called_once_with(self.cfg.halide.autoscheduler)
        # rustc 는 빌드와 같은 환경(CARGO_HOME·RUSTUP_HOME·RUSTUP_TOOLCHAIN 포함)으로 찾는다
        # PATH 밖(~/.cargo/bin)의 rustup 도 찾는다(protocol.rust_tool)
        version.assert_called_once_with(generate.rust_tool("rustc"), env=generate.toolchain_env())
        self.assertNotIn("unknown", err.getvalue())

    def test_unavailable_values_are_unknown_with_one_warning_each(self) -> None:
        nothing = {"numba_version": None, "halide_version": None, "halide_autoscheduler_loadable": None, "errors": {"python": "x"}}
        with mock.patch.object(generate, "python_toolchain_subprocess", return_value=nothing), \
                mock.patch.object(generate, "compiler_version", return_value=None), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            info = self.info({})
            for _ in range(2):
                self.assertEqual((info.numba_version(), info.halide_version(), info.rustc_version()), ("unknown", "unknown", "unknown"))
        warnings = [line for line in err.getvalue().splitlines() if "'unknown'" in line]
        self.assertEqual(len(warnings), 3, err.getvalue())
        self.assertTrue(all(line.startswith("[generate] ") for line in warnings))

    def test_rust_profile_and_rayon_version_are_read_from_the_build_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "Cargo.toml"
            manifest.write_text('[package]\nname = "kernel"\n\n[dependencies]\nrayon = "=1.2.3"\n\n[profile.release]\n'
                                'opt-level = 3\nlto = "off"\ndebug = false\n', encoding="utf-8")
            with mock.patch.object(generate, "RUST_MANIFEST", manifest), contextlib.redirect_stderr(io.StringIO()) as err:
                info = self.info({})
                self.assertEqual(info.rust_profile(), '`opt-level = 3`, `lto = "off"`, `debug = false`')
                self.assertEqual(info.rayon_version(), "1.2.3")
            self.assertEqual(err.getvalue(), "")
            manifest.write_text('[package]\nname = "kernel"\n\n[dependencies]\nrayon = { version = "1.10", features = [] }\n',
                                encoding="utf-8")
            with mock.patch.object(generate, "RUST_MANIFEST", manifest), contextlib.redirect_stderr(io.StringIO()) as err:
                info = self.info({})
                self.assertEqual((info.rayon_version(), info.rust_profile()), ("1.10", "unknown"))
            self.assertIn("[profile.release]", err.getvalue())

    def test_missing_or_broken_build_template_is_unknown_with_a_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for content in (None, "this is [not toml\n"):
                manifest = Path(tmp) / "Cargo.toml"
                if content is not None:
                    manifest.write_text(content, encoding="utf-8")
                with self.subTest(content=content), mock.patch.object(generate, "RUST_MANIFEST", manifest), \
                        contextlib.redirect_stderr(io.StringIO()) as err:
                    info = self.info({})
                    self.assertEqual((info.rust_profile(), info.rayon_version()), ("unknown", "unknown"))
                    self.assertIn("[generate] ", err.getvalue())

    def test_cargo_command_follows_the_config(self) -> None:
        rust = self.cfg.build.rust
        flags = " ".join(protocol.rust_flags(rust))
        self.assertEqual(generate.cargo_command(rust), f'RUSTFLAGS="{flags}" cargo build --release --offline --locked')
        bare = dataclasses.replace(rust, offline=False, rustflags=[], rustflags_linux=[])
        self.assertEqual(generate.cargo_command(bare), "cargo build --release --locked")

    def test_cargo_command_uses_the_effective_rustflags_of_the_build(self) -> None:
        """RUSTFLAGS 는 빌드·cflags 열과 같은 ``protocol.rust_flags`` 의 값이다: Linux 에서만 ``rustflags_linux`` 가 붙는다.
        프롬프트의 명령은 빌드 모듈이 공개하는 명령(``cargo_command_text``)과 글자까지 같다."""
        rust = self.cfg.build.rust
        for linux, expected in ((True, "-C target-cpu=native -C link-arg=-Wl,--no-undefined"), (False, "-C target-cpu=native")):
            with self.subTest(linux=linux):
                self.assertEqual(" ".join(protocol.rust_flags(rust, linux=linux)), expected)
                command = generate.cargo_command(rust, linux=linux)
                self.assertEqual(command, f'RUSTFLAGS="{expected}" cargo build --release --offline --locked')
                self.assertEqual(command, rust_build.cargo_command_text(rust, linux=linux))

    def test_cargo_command_shows_the_bare_tool_name(self) -> None:
        """설정이 cargo 를 전체 경로로 줘도 프롬프트에는 이름만 나온다(머신마다 프롬프트가 달라지지 않는다)."""
        for configured in ("/home/u/.cargo/bin/cargo", "C:\\Users\\u\\.cargo\\bin\\cargo.exe", "cargo.exe", "cargo"):
            with self.subTest(configured=configured):
                command = generate.cargo_command(dataclasses.replace(self.cfg.build.rust, cargo=configured))
                self.assertTrue(command.endswith('" cargo build --release --offline --locked'), command)
                self.assertNotIn("\\", command)
                self.assertNotIn(".exe", command)

    def test_rustc_version_is_probed_through_the_rust_tool_lookup(self) -> None:
        """rustc 는 PATH 밖(``~/.cargo/bin``)에 있어도 ``protocol.rust_tool`` 로 찾는다(rustup 이 PATH 를 바꾸지 않은 머신)."""
        with mock.patch.object(generate, "rust_tool", return_value="/home/u/.cargo/bin/rustc") as tool, \
                mock.patch.object(generate, "compiler_version", return_value="rustc 1.99.1 (test)") as version:
            self.assertEqual(self.info({}).rustc_version(), "rustc 1.99.1 (test)")
        tool.assert_called_once()
        self.assertEqual(tool.call_args.args[0], "rustc")
        version.assert_called_once_with("/home/u/.cargo/bin/rustc", env=generate.toolchain_env())

    def test_values_are_looked_up_only_for_the_variables_a_template_uses(self) -> None:
        # numpy 프롬프트를 만들면서 rustc·numba·halide·컴파일러를 찾지 않는다(값은 머신 프로필에 없다)
        with no_probing():
            generate.render_prompt(self.cfg, profile_machine(), "gemm", self.cfg.languages["numpy"], args_mode="positional")


# ----------------------------------------------------------------------------- 힌트 파일


class HintFilesTest(unittest.TestCase):
    HINTS = PROMPTS_DIR / "hints"

    def test_hints_exist_only_for_the_new_languages(self) -> None:
        self.assertEqual(sorted(p.stem for p in self.HINTS.glob("*.md")), sorted(NEW_LANGUAGES))

    def test_hints_are_short_plain_english_notes(self) -> None:
        for language in NEW_LANGUAGES:
            with self.subTest(language=language):
                text = (self.HINTS / f"{language}.md").read_text(encoding="utf-8")
                self.assertLessEqual(len(text.splitlines()), 100)
                self.assertTrue(text.isascii())
                self.assertNotIn("$", text)  # 템플릿 변수가 아니다: 렌더링된 값으로 그대로 들어간다
                self.assertEqual(text.count("```") % 2, 0)
                prose = _FENCE_RE.sub("", text)
                self.assertEqual(re.findall(r"^#+ .*$", prose, re.M), [], "제목 줄은 generate 가 붙인다(## Language notes)")
                self.assertTrue(hint_fragments(language), "문법 안내에는 돌려 본 코드 조각이 있어야 한다")

    def test_hints_are_independent_of_the_task(self) -> None:
        task_words = re.compile(r"\b(gemm|matmul|jacobi\w*|stencil|spmv\w*|csr|histogram|prefix[_ ]sum|nbody|n-body|cholesky|floyd\w*|"
                                r"warshall|bfs|fft|fourier)\b", re.IGNORECASE)
        for language in NEW_LANGUAGES:
            with self.subTest(language=language):
                text = (self.HINTS / f"{language}.md").read_text(encoding="utf-8")
                self.assertEqual(task_words.findall(text), [])

    def test_only_the_manual_halide_hint_names_scheduling_methods(self) -> None:
        call = re.compile(r"\.(\w+)\(")
        auto = set(call.findall((self.HINTS / "halide_auto.md").read_text(encoding="utf-8")))
        manual = set(call.findall((self.HINTS / "halide_manual.md").read_text(encoding="utf-8")))
        self.assertEqual(auto & SCHEDULE_METHODS, set())
        self.assertLessEqual({"split", "reorder", "vectorize", "parallel", "tile", "unroll", "fuse", "compute_root", "compute_at"},
                             manual & SCHEDULE_METHODS)

    def test_numba_hint_shows_only_signed_compiled_functions_without_cache(self) -> None:
        """numba 힌트의 예는 하네스 규칙을 지킨다: 모든 njit/jit 에 명시 서명(첫 위치 인자), ``cache=True`` 없음."""
        text = (self.HINTS / "numba.md").read_text(encoding="utf-8")
        decorators = re.findall(r"^\s*@(?:numba\.)?n?jit\b(.*)$", "\n".join(code for _, code in hint_fragments("numba")), re.M)
        self.assertTrue(decorators)
        for rest in decorators:
            with self.subTest(decorator=rest):
                self.assertRegex(rest, r'^\(\s*"')
        self.assertNotIn("cache=True", text)
        self.assertIn("with its own explicit signature string", text)

    def test_manual_halide_hint_is_the_auto_hint_plus_scheduling(self) -> None:
        auto = (self.HINTS / "halide_auto.md").read_text(encoding="utf-8")
        manual = (self.HINTS / "halide_manual.md").read_text(encoding="utf-8")
        self.assertTrue(manual.startswith(auto.rstrip("\n") + "\n\n**Scheduling**\n"))


class HintFragmentsTest(unittest.TestCase):
    """힌트에 실린 코드 조각은 그 언어의 실제 도구에서 돈다(모델에 틀린 문법을 가르치지 않는다).

    Python 계열은 파일의 조각을 차례로 이어 ``PYTHON_EXE`` 로 실행하고(경고도 실패), Rust 는 하네스의 크레이트 템플릿에 조각을
    ``src/lib.rs`` 로 넣어 ``cargo check`` 한다.
    """

    def run_python_fragments(self, language: str, prelude: str = "") -> None:
        fragments = hint_fragments(language)
        self.assertEqual({tag for tag, _ in fragments}, {"python"})
        source = prelude + "\n".join(code for _, code in fragments)
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_isolated([str(PYTHON_EXE), "-W", "error", "-c", source], timeout_s=600, cwd=Path(tmp), tail_chars=4000)
        self.assertEqual((proc.rc, proc.timed_out), (0, False), proc.stderr_tail)

    @requires_modules("numba")
    def test_numba_fragments_compile(self) -> None:
        self.run_python_fragments("numba")

    @requires_modules("halide")
    def test_halide_fragments_build(self) -> None:
        # 빌드 함수 안에서 쓰는 조각이다: 템플릿이 적는 대로 hl 을 임포트해 둔다
        for language in ("halide_auto", "halide_manual"):
            with self.subTest(language=language):
                self.run_python_fragments(language, "import halide as hl\n")

    @requires_rust_tool("cargo", "rustc")
    def test_rust_fragments_compile(self) -> None:
        fragments = hint_fragments("rust")
        self.assertEqual({tag for tag, _ in fragments}, {"rust"})
        with tempfile.TemporaryDirectory() as tmp:
            crate = Path(tmp) / "crate"
            (crate / "src").mkdir(parents=True)
            for name in ("Cargo.toml", "Cargo.lock"):
                (crate / name).write_bytes((generate.RUST_MANIFEST.parent / name).read_bytes())
            # 하네스의 빌드처럼 rust_tool 이 찾은 cargo 를 그 디렉터리가 PATH 앞에 붙은 환경으로 띄운다
            cargo = protocol.rust_tool("cargo")
            env = {**protocol.rust_tool_env(cargo), "CARGO_TARGET_DIR": str(Path(tmp) / "target")}
            for i, (_, code) in enumerate(fragments):
                with self.subTest(fragment=i):
                    (crate / "src" / "lib.rs").write_text(code, encoding="utf-8")
                    proc = run_isolated([cargo, "check", "--offline", "--locked"], timeout_s=900, cwd=crate, env=env, tail_chars=6000)
                    if proc.rc != 0 and "could not compile `kernel`" not in proc.stderr_tail:
                        self.skipTest(f"cargo 가 의존성을 오프라인으로 준비하지 못했다(`make rust-setup`): {proc.stderr_tail[-300:]}")
                    self.assertEqual((proc.rc, proc.timed_out), (0, False), proc.stderr_tail)
                    self.assertNotIn("warning:", proc.stderr_tail)


class NumbaEnvironmentClaimTest(unittest.TestCase):
    """numba 템플릿의 환경 서술 "SciPy is not installed … `np.dot`, `np.vdot`, the `@` operator and `np.linalg` … are not available"
    은 워커 인터프리터에서 사실이어야 한다(SciPy 를 설치하면 이 테스트가 깨진다 → 템플릿을 고쳐야 한다)."""

    STATEMENT = ("- SciPy is not installed. Numba's implementations of `np.dot`, `np.vdot`, the `@` operator and `np.linalg` depend on "
                 "it, so they are not available inside compiled functions.\n")

    def run_python(self, source: str) -> Any:
        with tempfile.TemporaryDirectory() as tmp:
            return run_isolated([str(PYTHON_EXE), "-c", source], timeout_s=600, cwd=Path(tmp), tail_chars=4000)

    def test_template_states_it_and_requirements_do_not_install_scipy(self) -> None:
        self.assertIn(self.STATEMENT, (PROMPTS_DIR / "lang" / "numba.md").read_text(encoding="utf-8"))
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertEqual(re.findall(r"^\s*scipy\b.*$", requirements, re.M | re.I), [])

    def test_scipy_cannot_be_imported_by_the_worker_interpreter(self) -> None:
        proc = self.run_python("import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('scipy') is None else 1)")
        self.assertEqual(proc.rc, 0, "scipy 가 워커 인터프리터에 설치돼 있다: prompts/lang/numba.md 의 서술이 거짓이 된다")

    @requires_modules("numba")
    def test_the_named_functions_do_not_compile(self) -> None:
        bodies = {"np.dot": "return np.dot(a, b)", "np.vdot": "return np.vdot(a[0], b[0])", "@": "return a @ b",
                  "np.linalg": "return np.linalg.cholesky(a)"}
        for name, body in bodies.items():
            source = ("import numpy as np\nfrom numba import njit\ntry:\n"
                      f"    @njit\n    def f(a, b):\n        {body}\n"
                      "    f(np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32))\n"
                      "except ImportError as e:\n    print('IMPORT_ERROR', e)\nelse:\n    print('COMPILED')\n")
            with self.subTest(name=name):
                proc = self.run_python(source)
                self.assertEqual(proc.rc, 0, proc.stderr_tail)
                self.assertIn("IMPORT_ERROR", proc.stdout_tail)
                self.assertIn("scipy", proc.stdout_tail.lower())


# ----------------------------------------------------------------------------- 계획 규칙


class PlanRulesTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "sample_0.json"

    def write(self, **fields: Any) -> None:
        self.path.write_text(json.dumps({"error": None, "stop_reason": "end_turn", **fields}), encoding="utf-8")

    def test_decide_action(self) -> None:
        self.assertEqual(generate.decide_action(self.path, force=False, no_retry_failed=False), ("call", None))
        for stop_reason in ("end_turn", "stop_sequence"):
            self.write(stop_reason=stop_reason)
            self.assertEqual(generate.decide_action(self.path, force=False, no_retry_failed=False)[0], "skip_ok")
            self.assertEqual(generate.decide_action(self.path, force=True, no_retry_failed=False)[0], "call")
        for fields in ({"error": "transient:OverloadedError:529:x", "stop_reason": ""}, {"stop_reason": "max_tokens"},
                       {"stop_reason": "refusal"}):
            self.write(**fields)
            self.assertEqual(generate.decide_action(self.path, force=False, no_retry_failed=False)[0], "call", fields)
            self.assertEqual(generate.decide_action(self.path, force=False, no_retry_failed=True)[0], "skip_failed", fields)
        self.path.write_text("{broken", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(generate.decide_action(self.path, force=False, no_retry_failed=True), ("call", None))

    def test_archive_previous_keeps_every_earlier_file(self) -> None:
        self.assertIsNone(generate.archive_previous(self.path))
        names = []
        for n in (1, 2, 3):
            self.write(n=n)
            names.append(generate.archive_previous(self.path).name)
            self.assertFalse(self.path.exists())
        self.assertEqual(names, ["sample_0.prev1.json", "sample_0.prev2.json", "sample_0.prev3.json"])
        self.assertEqual(json.loads((self.path.parent / "sample_0.prev2.json").read_text(encoding="utf-8"))["n"], 2)

    def test_coverage_table(self) -> None:
        pairs = {("alpha", "numpy"): generate.PairPlan(generate.PAIR_CALL, planned=5, called=3),
                 ("alpha", "halide_auto"): generate.PairPlan(generate.PAIR_SKIP),
                 ("beta", "numpy"): generate.PairPlan(generate.PAIR_CALL, planned=2, called=2),
                 ("beta", "halide_auto"): generate.PairPlan(generate.PAIR_UNSUPPORTED)}
        self.assertEqual(generate.coverage_table(["alpha", "beta"], ["numpy", "halide_auto"], pairs),
                         "task   numpy     halide_auto\n"
                         "alpha  call 3/5  skip\n"
                         "beta   call 2/2  unsupported")


# ----------------------------------------------------------------------------- main (dummy, 임시 디렉터리)


SIGNATURES_ALPHA = """\
task: alpha
function: alpha
args:
  - {name: x, kind: in, dtype: float32, shape: [n]}
  - {name: out, kind: out, dtype: float32, shape: [n]}
  - {name: n, kind: dim}
"""
SIGNATURES_BETA = SIGNATURES_ALPHA.replace("alpha", "beta") + """\
unsupported:
  halide_auto: "the kernel is not expressible as a Halide pipeline"
  rust: "no verified reference implementation"
"""


class FakeClient:
    """``LLMClient`` 자리에 끼우는 가짜(SDK·네트워크 없음). 호출을 기록하고 정해 둔 오류를 순서대로 돌려준다."""

    instances: list["FakeClient"] = []
    errors: list[str | None] = []

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.calls: list[dict[str, Any]] = []
        self.preflighted = False
        FakeClient.instances.append(self)

    def preflight(self) -> None:
        self.preflighted = True

    def complete(self, system: str, user: str, *, messages: Any = None, dummy_key: Any = None) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "messages": messages, "dummy_key": dummy_key})
        error = FakeClient.errors.pop(0) if FakeClient.errors else None
        ok = error is None
        return LLMResponse(
            text="```python\npass\n```\n" if ok else "", model_requested=self.cfg.model.name, model_served="served-model" if ok else "",
            provider=self.cfg.model.provider, tokens_in=100 if ok else 0, tokens_out=20 if ok else 0, tokens_thinking=5 if ok else None,
            tokens_text=15 if ok else None, latency_ms=1.0, stop_reason="end_turn" if ok else "", raw_stop_reason="end_turn" if ok else "",
            stop_details_category=None, n_thinking_blocks=0, n_text_blocks=1 if ok else 0, request_id="req_1" if ok else None,
            raw_usage={}, base_url="https://api.invalid", sdk_version="0.0", api_key_present=True, error=error)


class MainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._fixture.cleanup)
        root = Path(cls._fixture.name)
        cls.tasks_dir, cls.dummy_dir, cls.hints_dir = root / "tasks", root / "dummy", root / "hints"
        for task, signatures in (("alpha", SIGNATURES_ALPHA), ("beta", SIGNATURES_BETA)):
            (cls.tasks_dir / task).mkdir(parents=True)
            (cls.tasks_dir / task / "signatures.yaml").write_text(signatures, encoding="utf-8")
            (cls.tasks_dir / task / "spec.md").write_text(f"# Task: {task}\n\nCopy `x` into `out`.\n", encoding="utf-8")
            for language in ALL_LANGUAGES:
                (cls.dummy_dir / task / language).mkdir(parents=True)
                (cls.dummy_dir / task / language / "0.md").write_text(f"Reply for {task}/{language}.\n\n```\ncode\n```\n", encoding="utf-8")
        cls.hints_dir.mkdir()
        (cls.hints_dir / "numba.md").write_text("A note about the language.\n", encoding="utf-8")
        cls.machine = complete_machine()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        doc = yaml.safe_load(DEFAULT_EXPERIMENT_YAML.read_text(encoding="utf-8"))
        doc["run"]["tasks"] = ["alpha", "beta"]
        doc["paths"] = {"results_csv": str(self.out / "runs.csv"), "results_csv_dummy": str(self.out / "runs_dummy.csv"),
                        "raw_dir": str(self.out / "raw"), "built_dir": str(self.out / "built")}
        self.config_path = self.out / "experiment.yaml"
        self.config_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        FakeClient.instances, FakeClient.errors = [], []
        for patcher in (mock.patch.object(generate, "TASKS_DIR", self.tasks_dir),
                        mock.patch.object(generate, "HINTS_DIR", self.hints_dir),
                        mock.patch.object(llm_client, "DUMMY_ROOT", self.dummy_dir),
                        mock.patch.object(generate, "load_machine", return_value=self.machine),
                        mock.patch.object(generate, "harness_version", return_value="test-version"),
                        mock.patch.dict(os.environ)):  # main 은 --provider/--model 을 환경변수로 넘긴다: 테스트 밖으로 새지 않게
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("LLM_MODEL", None)

    def main(self, *args: str, provider: str = "dummy", run_id: str = "r1") -> tuple[int, str, str]:
        argv = ["--config", str(self.config_path), "--provider", provider, "--run-id", run_id, *args]
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err, no_probing():
            rc = generate.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def raw_path(self, task: str, language: str, i: int, *, tag: str = "args-positional_hint-off_fb-compiler", run_id: str = "r1") -> Path:
        return self.out / "raw" / run_id / task / language / tag / f"sample_{i}.json"

    def raw(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return json.loads(self.raw_path(*args, **kwargs).read_text(encoding="utf-8"))

    def raw_files(self) -> list[str]:
        base = self.out / "raw"
        return sorted(p.relative_to(base).as_posix() for p in base.rglob("*.json")) if base.exists() else []

    # ------------------------------------------------------------------ 기본 흐름

    def test_dummy_run_writes_one_raw_per_sample_with_the_design_keys(self) -> None:
        rc, out, err = self.main("--tasks", "alpha", "--languages", "numpy,rust", "--n", "2")
        self.assertEqual(rc, 0, err)
        tag = "args-positional_hint-off_fb-compiler"
        self.assertEqual(self.raw_files(), [f"r1/alpha/{language}/{tag}/sample_{i}.json" for language in ("numpy", "rust") for i in (0, 1)])
        raw = self.raw("alpha", "rust", 1)
        self.assertEqual(list(raw), RAW_KEYS)
        self.assertEqual((raw["run_id"], raw["task"], raw["language"], raw["backend"], raw["sample_idx"], raw["turn"]),
                         ("r1", "alpha", "rust", "rust", 1, 0))
        self.assertEqual(raw["conditions"], {"args": "positional", "grammar_hint": "off", "feedback": "compiler"})
        self.assertEqual((raw["provider"], raw["model_served"], raw["stop_reason"], raw["error"]), ("dummy", "dummy", "end_turn", None))
        self.assertEqual(raw["response_text"], "Reply for alpha/rust.\n\n```\ncode\n```\n")
        self.assertEqual(list(raw["prompt"]), ["system", "user"])
        self.assertEqual(raw["prompt_hash"], generate.prompt_hash(raw["prompt"]["system"], raw["prompt"]["user"]))
        self.assertIn('pub unsafe extern "C" fn alpha(x: *const f32, out: *mut f32, n: i32)', raw["prompt"]["user"])
        self.assertTrue(raw["prompt"]["user"].endswith("---\n\n# Task: alpha\n\nCopy `x` into `out`.\n"))
        self.assertEqual(raw["harness_version"], "test-version")
        self.assertEqual(raw["config_snapshot"]["languages"]["rust"]["env"], {"RAYON_NUM_THREADS": "4"})
        self.assertEqual(raw["sampling"], {"temperature_sent": None, "temperature_effective": "provider_default", "thinking": None,
                                           "effort": None, "seed": None, "top_p": None})
        self.assertEqual(out.splitlines()[:2], ["task   numpy     rust", "alpha  call 2/2  call 2/2"])
        self.assertIn("run_id=r1 calls=4 skipped_ok=0 skipped_failed=0 failed=0", out.splitlines()[2])

    def test_all_seven_templates_render_for_a_task_without_yaml_signatures(self) -> None:
        rc, _, err = self.main("--tasks", "alpha", "--languages", ",".join(ALL_LANGUAGES), "--n", "1")
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(self.raw_files()), 7)
        self.assertIn('@njit("void(float32[::1], float32[::1], int64)")\ndef alpha(x, out, n):', self.raw("alpha", "numba", 0)["prompt"]["user"])
        halide = self.raw("alpha", "halide_auto", 0)["prompt"]["user"]
        self.assertIn("```python\ndef alpha(x, n):\n```", halide)
        self.assertIn("the input `x` has shape (n,): its element at array index (i0), with 0 <= i0 < n, is `x[i0]`.", halide)

    def test_unsupported_pairs_are_skipped_and_shown_in_the_table(self) -> None:
        rc, out, err = self.main("--languages", "numpy,halide_auto,rust", "--n", "2")
        self.assertEqual(rc, 0, err)
        self.assertEqual(out.splitlines()[:3], ["task   numpy     halide_auto  rust",
                                                "alpha  call 2/2  call 2/2     call 2/2",
                                                "beta   call 2/2  unsupported  unsupported"])
        self.assertEqual(sorted({name.rsplit("/", 2)[0] for name in self.raw_files()}),
                         ["r1/alpha/halide_auto", "r1/alpha/numpy", "r1/alpha/rust", "r1/beta/numpy"])
        self.assertIn("beta/halide_auto", err)
        self.assertIn("the kernel is not expressible as a Halide pipeline", err)
        self.assertIn("no verified reference implementation", err)

    # ------------------------------------------------------------------ 조건

    def test_condition_options_reach_the_raw_and_its_directory(self) -> None:
        rc, _, err = self.main("--tasks", "alpha", "--languages", "numba", "--n", "1", "--args", "keyword", "--grammar-hint", "on",
                               "--feedback", "compiler+lint")
        self.assertEqual(rc, 0, err)
        tag = "args-keyword_hint-on_fb-compiler+lint"
        self.assertEqual(self.raw_files(), [f"r1/alpha/numba/{tag}/sample_0.json"])
        raw = self.raw("alpha", "numba", 0, tag=tag)
        self.assertEqual(raw["conditions"], {"args": "keyword", "grammar_hint": "on", "feedback": "compiler+lint"})
        self.assertEqual(raw["config_snapshot"]["run"]["conditions"], raw["conditions"])
        user = raw["prompt"]["user"]
        self.assertIn("def alpha(*, x, out, n):", user)
        self.assertIn("as `alpha(x=..., out=..., n=...)`, never by position.", user)
        self.assertIn("\n## Language notes\n\nA note about the language.\n\n---\n", user)

    def test_each_condition_value_has_its_own_directory(self) -> None:
        for option, value, tag in (("--args", "keyword", "args-keyword_hint-off_fb-compiler"),
                                   ("--feedback", "compiler+lint", "args-positional_hint-off_fb-compiler+lint"),
                                   ("--grammar-hint", "off", "args-positional_hint-off_fb-compiler")):
            with self.subTest(option=option):
                rc, _, err = self.main("--tasks", "alpha", "--languages", "numpy", "--n", "1", option, value, run_id=f"run{option}")
                self.assertEqual(rc, 0, err)
                self.assertTrue(self.raw_path("alpha", "numpy", 0, tag=tag, run_id=f"run{option}").is_file())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            generate.main(["--config", str(self.config_path), "--args", "named"])

    def test_feedback_does_not_change_the_turn_zero_prompt(self) -> None:
        self.main("--tasks", "alpha", "--languages", "numpy", "--n", "1", run_id="a")
        self.main("--tasks", "alpha", "--languages", "numpy", "--n", "1", "--feedback", "compiler+lint", run_id="b")
        first = self.raw("alpha", "numpy", 0, run_id="a")
        second = self.raw("alpha", "numpy", 0, tag="args-positional_hint-off_fb-compiler+lint", run_id="b")
        self.assertEqual(first["prompt_hash"], second["prompt_hash"])

    def test_languages_without_a_hint_file_are_skipped_when_the_hint_is_on(self) -> None:
        rc, out, err = self.main("--languages", "numpy,numba", "--n", "1", "--grammar-hint", "on")
        self.assertEqual(rc, 0, err)
        self.assertEqual(out.splitlines()[:3], ["task   numpy  numba", "alpha  skip   call 1/1", "beta   skip   call 1/1"])
        self.assertEqual(sorted({name.split("/")[2] for name in self.raw_files()}), ["numba"])
        warning = next(line for line in err.splitlines() if "grammar_hint=on" in line)
        self.assertIn("numpy", warning)
        self.assertEqual(sum("grammar_hint=on" in line for line in err.splitlines()), 1)  # 언어당 한 번

    # ------------------------------------------------------------------ 다시 돌릴 때

    def test_rerun_skips_good_samples_and_recalls_failed_ones(self) -> None:
        args = ("--tasks", "alpha", "--languages", "numpy", "--n", "3")
        self.assertEqual(self.main(*args)[0], 0)
        failed = self.raw_path("alpha", "numpy", 1)
        doc = json.loads(failed.read_text(encoding="utf-8"))
        failed.write_text(json.dumps({**doc, "error": "transient:OverloadedError:529:Overloaded", "stop_reason": ""}), encoding="utf-8")

        rc, out, err = self.main(*args, "--no-retry-failed")
        self.assertEqual(rc, 0, err)
        self.assertEqual(out.splitlines()[1], "alpha  skip")
        self.assertIn("calls=0 skipped_ok=2 skipped_failed=1", out)

        rc, out, err = self.main(*args)
        self.assertEqual(rc, 0, err)
        self.assertEqual(out.splitlines()[1], "alpha  call 1/1")
        self.assertIn("calls=1 skipped_ok=2 skipped_failed=0", out)
        self.assertIsNone(self.raw("alpha", "numpy", 1)["error"])
        archived = failed.with_name("sample_1.prev1.json")
        self.assertEqual(json.loads(archived.read_text(encoding="utf-8"))["error"], "transient:OverloadedError:529:Overloaded")

        rc, out, err = self.main(*args, "--force")
        self.assertEqual(rc, 0, err)
        self.assertIn("calls=3 skipped_ok=0", out)
        self.assertTrue(failed.with_name("sample_1.prev2.json").is_file())

    # ------------------------------------------------------------------ 입력 오류

    def test_input_errors_exit_one_before_any_call(self) -> None:
        cases = [("--languages", "numpy,cobol"), ("--n", "0"), ("--tasks", "gamma"),
                 ("--languages", "numba", "--provider", "nobody")]
        for extra in cases:
            with self.subTest(extra=extra):
                rc, _, err = self.main("--tasks", "alpha", *extra)
                self.assertEqual(rc, 1, err)
                self.assertIn("[generate] ", err)
        self.assertEqual(self.raw_files(), [])

    def test_task_outside_run_tasks_is_refused_before_any_call(self) -> None:
        # harness.run·repair 는 run.tasks 밖의 raw 를 건너뛴다: 그 응답은 행이 되지 못하므로 호출하지 않는다
        doc = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        doc["run"]["tasks"] = ["alpha"]
        self.config_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        rc, _, err = self.main("--tasks", "alpha,beta", "--languages", "numpy", "--n", "1")  # beta 는 명세가 있는 과제다
        self.assertEqual(rc, 1, err)
        self.assertIn("[generate] 설정 run.tasks 에 없는 과제 ['beta']", err)
        self.assertEqual(self.raw_files(), [])
        rc, _, err = self.main("--languages", "numpy", "--n", "1")  # 기본 목록은 run.tasks 다
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.raw_files(), ["r1/alpha/numpy/args-positional_hint-off_fb-compiler/sample_0.json"])
        # run.tasks 에 있어도 명세가 없는 과제는 여전히 호출 전에 거부된다(load_spec)
        doc["run"]["tasks"] = ["alpha", "gamma"]
        self.config_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        rc, _, err = self.main("--tasks", "gamma", "--languages", "numpy", "--n", "1", run_id="r2")
        self.assertEqual(rc, 1, err)
        self.assertIn("[generate] 오류", err)
        self.assertFalse((self.out / "raw" / "r2").exists())

    def test_seed_copy_is_not_called_again(self) -> None:
        # harness.repair --seed-from-feedback 가 만든 복사본: 다시 물으면 대응 비교의 짝이 다른 응답에서 시작한다
        tag = "args-positional_hint-off_fb-compiler+lint"
        self.assertEqual(self.main("--tasks", "alpha", "--languages", "numpy", "--n", "2", "--feedback", "compiler+lint")[0], 0)
        failed, ok = self.raw_path("alpha", "numpy", 0, tag=tag), self.raw_path("alpha", "numpy", 1, tag=tag)
        source = "results/raw/r1/alpha/numpy/args-positional_hint-off_fb-compiler/sample_{}.json"
        failed.write_text(json.dumps({**self.raw("alpha", "numpy", 0, tag=tag), "error": "transient:OverloadedError:529:Overloaded",
                                      "stop_reason": "", "seeded_from": source.format(0)}), encoding="utf-8")
        ok.write_text(json.dumps({**self.raw("alpha", "numpy", 1, tag=tag), "seeded_from": source.format(1)}), encoding="utf-8")
        before = self.raw_files(), failed.read_bytes(), ok.read_bytes()
        for extra in ((), ("--force",)):
            with self.subTest(extra=extra):
                rc, out, err = self.main("--tasks", "alpha", "--languages", "numpy", "--n", "2", "--feedback", "compiler+lint", *extra)
                self.assertEqual(rc, 0, err)
                self.assertIn("calls=0", out)
                self.assertEqual(err.count("시드 복사본이라 다시 호출하지 않는다"), 1 + len(extra))
                self.assertEqual((self.raw_files(), failed.read_bytes(), ok.read_bytes()), before)

    # ------------------------------------------------------------------ 실제 제공자의 게이트(가짜 클라이언트)

    def test_real_provider_needs_confirmation(self) -> None:
        with mock.patch.object(generate, "LLMClient", FakeClient), mock.patch("sys.stdin", io.StringIO("")):
            rc, _, err = self.main("--tasks", "alpha", "--languages", "numpy", "--n", "2", provider="anthropic")
        self.assertEqual(rc, 2)
        self.assertIn("2 calls to", err)
        self.assertIn("--yes", err)
        self.assertEqual((FakeClient.instances[0].preflighted, FakeClient.instances[0].calls), (True, []))
        self.assertEqual(self.raw_files(), [])

    def test_real_provider_with_yes_calls_and_records_the_sampling(self) -> None:
        with mock.patch.object(generate, "LLMClient", FakeClient):
            rc, out, err = self.main("--tasks", "alpha", "--languages", "numpy", "--n", "2", "--yes", provider="anthropic")
        self.assertEqual(rc, 0, err)
        client = FakeClient.instances[0]
        self.assertEqual([call["dummy_key"] for call in client.calls], [("alpha", "numpy", 0), ("alpha", "numpy", 1)])
        self.assertTrue(all(call["messages"] is None for call in client.calls))
        raw = self.raw("alpha", "numpy", 0)
        self.assertEqual((raw["provider"], raw["model_served"], raw["request_id"]), ("anthropic", "served-model", "req_1"))
        self.assertEqual((raw["sampling"]["thinking"], raw["sampling"]["effort"]), ("adaptive", "medium"))
        self.assertEqual(client.calls[0]["user"], raw["prompt"]["user"])
        self.assertIn("tokens_in=200 tokens_out=40 (thinking 10)", out)

    def test_fatal_error_stops_after_saving_the_raw(self) -> None:
        FakeClient.errors = [None, "fatal:AuthenticationError:401:invalid x-api-key"]
        with mock.patch.object(generate, "LLMClient", FakeClient):
            rc, out, _ = self.main("--tasks", "alpha", "--languages", "numpy", "--n", "4", "--yes", provider="anthropic")
        self.assertEqual(rc, 3)
        self.assertEqual(len(FakeClient.instances[0].calls), 2)
        self.assertEqual(self.raw("alpha", "numpy", 1)["error"], "fatal:AuthenticationError:401:invalid x-api-key")
        self.assertFalse(self.raw_path("alpha", "numpy", 2).exists())
        self.assertEqual(out.splitlines()[1], "alpha  call 2/4")

    def test_transient_error_is_recorded_and_the_batch_continues(self) -> None:
        FakeClient.errors = ["transient:OverloadedError:529:Overloaded"]
        with mock.patch.object(generate, "LLMClient", FakeClient):
            rc, out, err = self.main("--tasks", "alpha", "--languages", "numpy", "--n", "3", "--yes", provider="anthropic")
        self.assertEqual(rc, 0, err)
        self.assertIn("calls=3 skipped_ok=0 skipped_failed=0 failed=1", out)
        self.assertEqual([self.raw("alpha", "numpy", i)["error"] is None for i in range(3)], [False, True, True])

    def test_missing_sdk_is_reported_before_any_call(self) -> None:
        with mock.patch.dict(sys.modules, {"anthropic": None}), mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key-not-real"}):
            rc, _, err = self.main("--tasks", "alpha", "--languages", "numpy", "--n", "1", "--yes", provider="anthropic")
        self.assertEqual(rc, 1)
        self.assertIn("anthropic SDK", err)
        self.assertEqual(self.raw_files(), [])


if __name__ == "__main__":
    unittest.main()
