"""설정 로더(harness/config.py) 테스트: 기존 키의 검증, v3(§8)의 선택 섹션·필드와 기본값, 언어의 ``env`` 치환, 조건 덮어쓰기,
스냅샷.

설정 파일은 전부 임시 디렉터리에 만든다. 저장소의 ``config/experiment.yaml`` 은 "설계 v3 §8 의 값 그대로인가"만 본다.
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from harness import config
from harness.config import ConfigError, load_config
from harness.paths import ROOT

# 설계 v3 §8 의 언어 정의(저장소 설정의 M3 언어 넷)
V3_LANGUAGES: dict[str, dict[str, Any]] = {
    "numba": {"backend": "numba", "prompt": "prompts/lang/numba.md", "threads": "measurement",
              "env": {"NUMBA_NUM_THREADS": "${threads}", "NUMBA_THREADING_LAYER": "omp"}},
    "halide_auto": {"backend": "halide", "prompt": "prompts/lang/halide_auto.md", "threads": "measurement",
                    "halide_mode": "auto", "env": {"HL_NUM_THREADS": "${threads}"}},
    "halide_manual": {"backend": "halide", "prompt": "prompts/lang/halide_manual.md", "threads": "measurement",
                      "halide_mode": "manual", "env": {"HL_NUM_THREADS": "${threads}"}},
    "rust": {"backend": "rust", "prompt": "prompts/lang/rust.md", "threads": "measurement",
             "env": {"RAYON_NUM_THREADS": "${threads}"}},
}


def v2_doc() -> dict[str, Any]:
    """v3 의 키가 하나도 없는 설정(M2a 의 ``experiment.yaml`` 과 같은 구조)."""
    return {
        "run": {"tasks": ["gemm"], "languages": ["numpy", "c_seq", "c_openmp"], "n_samples": 5, "max_repair_turns": 0,
                "conditions": {"args": "positional", "grammar_hint": "off", "feedback": "compiler"}},
        "model": {"provider": "dummy", "name": "some-model", "max_tokens": 32000, "temperature": None, "seed": None,
                  "request_timeout_s": 600, "max_retries": 3,
                  "anthropic": {"api_key_env": "ANTHROPIC_API_KEY", "effort": "medium", "thinking": "adaptive"},
                  "openai_compat": {"api_key_env": "OPENAI_API_KEY", "base_url_env": "LLM_BASE_URL", "max_tokens_param": "max_tokens"}},
        "languages": {
            "numpy": {"backend": "python", "prompt": "prompts/lang/numpy.md", "threads": "measurement"},
            "c_seq": {"backend": "c", "prompt": "prompts/lang/c_seq.md", "openmp": False, "threads": 1},
            "c_openmp": {"backend": "c", "prompt": "prompts/lang/c_openmp.md", "openmp": True, "threads": "measurement"},
        },
        "build": {"cc": "gcc", "cflags": ["-O3", "-march=native"], "base_flags": ["-std=gnu11", "-fPIC", "-shared"],
                  "openmp_flag": "-fopenmp", "ldflags": ["-lm"], "diag_flags": [], "compile_timeout_s": 60,
                  "forbidden_symbols": ["system"], "thread_symbols_prefix": ["GOMP_"]},
        "measurement": {
            "cores": "0,2,4,6", "single_core": "2", "threads": 4, "omp_env_extra": {"OMP_PROC_BIND": "true", "OMP_DYNAMIC": False},
            "warmup": 3, "repeats": 10, "min_repeats": 3, "check_repeats": 2, "check_timeout_s": 180, "bench_timeout_s": 300,
            "max_worker_mem_gb": 8, "max_load_1m": 2.0, "max_pinned_busy_pct": 10.0, "require_performance_governor": True,
            "baseline_min_gflops": 30, "baseline_drift_pct_max": 10, "correctness_tests": ["tiny", "rect", "small"],
            "performance_test": "large",
        },
        "paths": {"results_csv": "results/runs.csv", "results_csv_dummy": "results/runs_dummy.csv", "raw_dir": "results/raw",
                  "built_dir": "results/built"},
    }


def v3_doc() -> dict[str, Any]:
    """``v2_doc`` 에 설계 v3 §8 의 언어와 섹션을 전부 더한 설정."""
    doc = v2_doc()
    doc["languages"].update(copy.deepcopy(V3_LANGUAGES))
    doc["build"]["rust"] = {"cargo": "cargo", "rustflags": ["-C", "target-cpu=native"], "build_timeout_s": 600, "offline": True}
    doc["build"]["sanitize_flags"] = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]
    doc["halide"] = {"autoscheduler": "Mullapudi2016", "autoscheduler_params": {}}
    doc["safety"] = {"enabled": False, "timeout_s": 300}
    doc["lint"] = {"enabled": True, "plugin": "plugins/lint/run_lint.py", "timeout_s": 60}
    doc["perf_summary"] = {"enabled": True}
    return doc


class ConfigTestCase(unittest.TestCase):
    """임시 디렉터리에 설정 파일을 쓰고 읽는 공통 틀. ``LLM_PROVIDER``·``LLM_MODEL`` 은 테스트마다 지운다."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("LLM_MODEL", None)

    def write(self, doc: Any) -> Path:
        path = self.tmp / "experiment.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        return path

    def load(self, doc: Any, **kwargs: Any) -> config.Config:
        return load_config(self.write(doc), **kwargs)

    def assert_rejected(self, doc: Any, key_path: str, **kwargs: Any) -> str:
        """``doc`` 이 ``ConfigError`` 로 거부되고 메시지가 ``<키 경로>:`` 로 시작하는지 본다. 메시지를 돌려준다."""
        with self.assertRaises(ConfigError) as caught, contextlib.redirect_stderr(io.StringIO()):
            self.load(doc, **kwargs)
        message = str(caught.exception)
        self.assertTrue(message.startswith(key_path + ":"), message)
        return message


class LoaderBasicsTest(ConfigTestCase):
    """v2 부터 있던 검증 규칙(설계 §2)."""

    def test_v2_values_and_path_resolution(self) -> None:
        cfg = self.load(v2_doc())
        self.assertEqual((cfg.run.tasks, cfg.run.languages, cfg.run.n_samples, cfg.run.max_repair_turns),
                         (["gemm"], ["numpy", "c_seq", "c_openmp"], 5, 0))
        self.assertEqual((cfg.model.provider, cfg.model.name, cfg.model.temperature, cfg.model.seed), ("dummy", "some-model", None, None))
        self.assertEqual({name: lang.threads for name, lang in cfg.languages.items()}, {"numpy": 4, "c_seq": 1, "c_openmp": 4})
        self.assertEqual((cfg.languages["numpy"].backend, cfg.languages["c_seq"].openmp, cfg.languages["c_openmp"].openmp),
                         ("python", False, True))
        self.assertEqual(cfg.languages["numpy"].prompt, ROOT / "prompts" / "lang" / "numpy.md")
        self.assertEqual(cfg.measurement.omp_env_extra, {"OMP_PROC_BIND": "true", "OMP_DYNAMIC": "false"})  # 불리언은 문자열로
        self.assertEqual((cfg.paths.raw_dir, cfg.paths.built_dir), (ROOT / "results" / "raw", ROOT / "results" / "built"))
        self.assertEqual(cfg.source_path, self.tmp / "experiment.yaml")
        absolute = v2_doc()
        absolute["paths"]["raw_dir"] = str(self.tmp / "raw")
        self.assertEqual(self.load(absolute).paths.raw_dir, self.tmp / "raw")

    def test_environment_overrides_provider_and_model(self) -> None:
        os.environ["LLM_PROVIDER"] = "openai_compat"
        os.environ["LLM_MODEL"] = "other-model"
        cfg = self.load(v2_doc())
        self.assertEqual((cfg.model.provider, cfg.model.name), ("openai_compat", "other-model"))
        self.assertEqual(cfg.env_overrides, {"LLM_PROVIDER": "openai_compat", "LLM_MODEL": "other-model"})
        os.environ["LLM_PROVIDER"] = "nobody"
        self.assert_rejected(v2_doc(), "model.provider")

    def test_missing_file_and_broken_yaml(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(self.tmp / "missing.yaml")
        broken = self.tmp / "broken.yaml"
        broken.write_text("run: [unclosed\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_config(broken)
        self.assert_rejected(["not", "a", "mapping"], "(top)")

    def test_rejects_bad_v2_values(self) -> None:
        def changed(path: str, value: Any) -> dict[str, Any]:
            doc = v2_doc()
            node = doc
            *parents, last = path.split(".")
            for key in parents:
                node = node[key]
            node[last] = value
            return doc

        cases = [
            ("run.n_samples", 0), ("run.max_repair_turns", config.MAX_REPAIR_TURNS + 1), ("run.tasks", []),
            ("run.conditions.args", "named"), ("run.conditions.grammar_hint", True), ("run.conditions.feedback", "lint"),
            ("model.max_tokens", 0), ("model.max_retries", -1), ("model.request_timeout_s", 0),
            ("model.openai_compat.max_tokens_param", "tokens"), ("languages.numpy.backend", "fortran"),
            ("languages.numpy.threads", 0), ("languages.numpy.threads", "all"), ("languages.c_seq.openmp", "no"),
            ("build.compile_timeout_s", 0), ("build.cflags", "-O3"), ("measurement.cores", "0;2"),
            ("measurement.single_core", "0,2"), ("measurement.threads", 5), ("measurement.min_repeats", 11),
            ("measurement.performance_test", "small"), ("paths.raw_dir", ""),
        ]
        for path, value in cases:
            with self.subTest(path=path, value=value):
                self.assert_rejected(changed(path, value), path)
        self.assert_rejected(changed("run.languages", ["numpy", "cobol"]), "run.languages")
        self.assert_rejected(changed("run.surprise", 1), "run")  # 알 수 없는 키
        missing = v2_doc()
        del missing["build"]["cc"]
        self.assert_rejected(missing, "build")
        self.assert_rejected({**v2_doc(), "extra": {}}, "(top)")

    def test_provider_specific_rules(self) -> None:
        doc = v2_doc()
        doc["model"].update({"provider": "anthropic", "seed": 7})
        self.assert_rejected(doc, "model.seed")
        doc = v2_doc()
        doc["model"]["provider"] = "anthropic"
        doc["model"]["anthropic"]["effort"] = None
        self.assert_rejected(doc, "model.anthropic.effort")
        doc = v2_doc()
        doc["model"]["provider"] = "openai_compat"
        doc["model"]["openai_compat"]["api_key_env"] = "ANTHROPIC_API_KEY"
        self.assert_rejected(doc, "model.openai_compat.api_key_env")
        doc = v2_doc()
        doc["model"]["temperature"] = 0.2
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(self.load(doc).model.temperature, 0.2)
        self.assertIn("[config]", err.getvalue())  # 4.7 이후 모델은 temperature 를 거부한다는 경고

    def test_parse_cpulist(self) -> None:
        self.assertEqual(config.parse_cpulist("0,2,4,6"), [0, 2, 4, 6])
        self.assertEqual(config.parse_cpulist("0-3,8"), [0, 1, 2, 3, 8])
        for bad in ("", "0, 2", "3-1", "a", "0,,2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                config.parse_cpulist(bad)


class V3DefaultsTest(ConfigTestCase):
    """v3 의 키는 전부 선택이다: 없는 설정 파일도 읽히고 기본값은 설계 v3 §8 의 값이다."""

    def test_backends_constant(self) -> None:
        self.assertEqual(config.BACKENDS, ("python", "c", "numba", "halide", "rust"))

    def test_file_without_v3_keys_gets_the_design_defaults(self) -> None:
        cfg = self.load(v2_doc())
        self.assertEqual(cfg.build.rust, config.RustCfg(cargo="cargo", rustflags=["-C", "target-cpu=native"],
                                                        build_timeout_s=600.0, offline=True))
        self.assertEqual(cfg.build.sanitize_flags, ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"])
        self.assertEqual(cfg.halide, config.HalideCfg(autoscheduler="Mullapudi2016", autoscheduler_params={}))
        self.assertEqual(cfg.safety, config.SafetyCfg(enabled=False, timeout_s=300.0))
        self.assertEqual(cfg.lint, config.LintCfg(enabled=True, plugin=ROOT / "plugins" / "lint" / "run_lint.py", timeout_s=60.0))
        self.assertEqual(cfg.perf_summary, config.PerfSummaryCfg(enabled=True))
        for lang in cfg.languages.values():
            self.assertEqual((lang.env, lang.halide_mode), ({}, None), lang.name)

    def test_explicit_v3_file_equals_the_defaults(self) -> None:
        explicit, implicit = self.load(v3_doc()), self.load(v2_doc())
        for section in ("build", "halide", "safety", "lint", "perf_summary"):
            self.assertEqual(getattr(explicit, section), getattr(implicit, section), section)

    def test_defaults_are_not_shared_between_configs(self) -> None:
        first, second = self.load(v2_doc()), self.load(v2_doc())
        first.build.rust.rustflags.append("-C")
        first.halide.autoscheduler_params["x"] = "1"
        self.assertEqual(second.build.rust.rustflags, ["-C", "target-cpu=native"])
        self.assertEqual(second.halide.autoscheduler_params, {})

    def test_repository_config_is_the_design_v3_config(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.run.languages, ["numpy", "c_seq", "c_openmp"])  # M2a 실제 실행 대기: 기본 언어는 그대로다
        self.assertEqual(cfg.run.tasks, ["gemm", "jacobi2d", "spmv_csr", "dot", "histogram", "prefix_sum", "nbody", "sort",
                                         "cholesky", "floyd_warshall", "bfs", "fft"])
        self.assertEqual({name: (lang.backend, lang.halide_mode) for name, lang in cfg.languages.items()},
                         {"numpy": ("python", None), "c_seq": ("c", None), "c_openmp": ("c", None), "numba": ("numba", None),
                          "halide_auto": ("halide", "auto"), "halide_manual": ("halide", "manual"), "rust": ("rust", None)})
        threads = str(cfg.measurement.threads)
        self.assertEqual({name: cfg.languages[name].env for name in V3_LANGUAGES},
                         {"numba": {"NUMBA_NUM_THREADS": threads, "NUMBA_THREADING_LAYER": "omp"},
                          "halide_auto": {"HL_NUM_THREADS": threads}, "halide_manual": {"HL_NUM_THREADS": threads},
                          "rust": {"RAYON_NUM_THREADS": threads}})
        for name, entry in V3_LANGUAGES.items():
            self.assertEqual((cfg.languages[name].prompt, cfg.languages[name].threads),
                             (ROOT / entry["prompt"], cfg.measurement.threads), name)
        defaults = self.load(v2_doc())
        for section in ("halide", "safety", "lint", "perf_summary"):
            self.assertEqual(getattr(cfg, section), getattr(defaults, section), section)
        self.assertEqual((cfg.build.rust, cfg.build.sanitize_flags), (defaults.build.rust, defaults.build.sanitize_flags))


class V3SectionsTest(ConfigTestCase):
    def test_build_rust(self) -> None:
        doc = v2_doc()
        doc["build"]["rust"] = {"cargo": "/opt/cargo/bin/cargo", "rustflags": [], "build_timeout_s": 90, "offline": False}
        self.assertEqual(self.load(doc).build.rust, config.RustCfg(cargo="/opt/cargo/bin/cargo", rustflags=[], build_timeout_s=90.0,
                                                                   offline=False))
        doc["build"]["rust"] = {"offline": False}  # 준 키만 바뀌고 나머지는 기본값
        self.assertEqual(self.load(doc).build.rust, config.RustCfg(offline=False))
        for key, bad in (("cargo", ""), ("cargo", 3), ("rustflags", "-C opt-level=3"), ("build_timeout_s", 0),
                         ("build_timeout_s", "long"), ("offline", "yes")):
            with self.subTest(key=key, bad=bad):
                doc["build"]["rust"] = {key: bad}
                self.assert_rejected(doc, f"build.rust.{key}")
        for bad in ({"target": "x86_64"}, ["cargo"]):  # 알 수 없는 키, 매핑이 아닌 값
            with self.subTest(bad=bad):
                doc["build"]["rust"] = bad
                self.assert_rejected(doc, "build.rust")

    def test_build_sanitize_flags(self) -> None:
        doc = v2_doc()
        doc["build"]["sanitize_flags"] = ["-fsanitize=address", "-g"]
        self.assertEqual(self.load(doc).build.sanitize_flags, ["-fsanitize=address", "-g"])
        for bad in ([], "-fsanitize=address", [1]):
            with self.subTest(bad=bad):
                doc["build"]["sanitize_flags"] = bad
                self.assert_rejected(doc, "build.sanitize_flags")

    def test_halide(self) -> None:
        doc = v2_doc()
        doc["halide"] = {"autoscheduler": "Adams2019", "autoscheduler_params": {"beam_size": 8, "disable_subtiling": True}}
        self.assertEqual(self.load(doc).halide, config.HalideCfg(autoscheduler="Adams2019",
                                                                 autoscheduler_params={"beam_size": "8", "disable_subtiling": "true"}))
        doc["halide"] = {"autoscheduler": "Li2018"}
        self.assertEqual(self.load(doc).halide, config.HalideCfg(autoscheduler="Li2018", autoscheduler_params={}))
        for bad, key_path in (({"autoscheduler": ""}, "halide.autoscheduler"), ({"autoscheduler": 2016}, "halide.autoscheduler"),
                              ({"autoscheduler_params": ["parallelism"]}, "halide.autoscheduler_params"),
                              ({"autoscheduler_params": {"cache": [1, 2]}}, "halide.autoscheduler_params.cache"),
                              ({"scheduler": "Mullapudi2016"}, "halide"), ("Mullapudi2016", "halide")):
            with self.subTest(bad=bad):
                doc["halide"] = bad
                self.assert_rejected(doc, key_path)

    def test_halide_parallelism_belongs_to_the_harness(self) -> None:
        # 프롬프트는 "parallelism = 그 언어의 스레드 수" 라고 적는다: 설정이 다른 값을 넣을 수 있으면 그 문장이 거짓이 된다
        doc = v2_doc()
        doc["halide"] = {"autoscheduler_params": {"parallelism": 16}}
        self.assertIn("parallelism", self.assert_rejected(doc, "halide.autoscheduler_params"))

    def test_safety(self) -> None:
        doc = v2_doc()
        doc["safety"] = {"enabled": True, "timeout_s": 120}
        self.assertEqual(self.load(doc).safety, config.SafetyCfg(enabled=True, timeout_s=120.0))
        doc["safety"] = {"enabled": True}
        self.assertEqual(self.load(doc).safety, config.SafetyCfg(enabled=True, timeout_s=300.0))
        for bad, key_path in (({"enabled": "true"}, "safety.enabled"), ({"timeout_s": 0}, "safety.timeout_s"),
                              ({"timeout_s": None}, "safety.timeout_s"), ({"tsan": True}, "safety"), ("on", "safety")):
            with self.subTest(bad=bad):
                doc["safety"] = bad
                self.assert_rejected(doc, key_path)

    def test_lint(self) -> None:
        doc = v2_doc()
        plugin = self.tmp / "my_lint.py"
        doc["lint"] = {"enabled": False, "plugin": str(plugin), "timeout_s": 5}
        self.assertEqual(self.load(doc).lint, config.LintCfg(enabled=False, plugin=plugin, timeout_s=5.0))
        doc["lint"] = {"plugin": "plugins/other/run_lint.py"}  # 상대 경로는 저장소 루트 기준
        self.assertEqual(self.load(doc).lint, config.LintCfg(enabled=True, plugin=ROOT / "plugins" / "other" / "run_lint.py",
                                                             timeout_s=60.0))
        for bad, key_path in (({"enabled": 1}, "lint.enabled"), ({"plugin": ""}, "lint.plugin"), ({"plugin": None}, "lint.plugin"),
                              ({"timeout_s": -1}, "lint.timeout_s"), ({"rules": "all"}, "lint"), ([], "lint")):
            with self.subTest(bad=bad):
                doc["lint"] = bad
                self.assert_rejected(doc, key_path)

    def test_perf_summary(self) -> None:
        doc = v2_doc()
        doc["perf_summary"] = {"enabled": False}
        self.assertEqual(self.load(doc).perf_summary, config.PerfSummaryCfg(enabled=False))
        doc["perf_summary"] = {}
        self.assertEqual(self.load(doc).perf_summary, config.PerfSummaryCfg(enabled=True))
        for bad, key_path in (({"enabled": "no"}, "perf_summary.enabled"), ({"format": "json"}, "perf_summary"),
                              (False, "perf_summary")):
            with self.subTest(bad=bad):
                doc["perf_summary"] = bad
                self.assert_rejected(doc, key_path)

    def test_measurement_max_settle_s(self) -> None:
        """선택 설정 ``measurement.max_settle_s``: 없으면 0(기다리지 않음 = 기존 절차), 0 이상의 수, 스냅샷에 들어간다."""
        self.assertEqual(self.load(v2_doc()).measurement.max_settle_s, 0.0)
        self.assertEqual(config.MeasurementCfg.__dataclass_fields__["max_settle_s"].default, 0.0)
        self.assertEqual(load_config().measurement.max_settle_s, 0.0)  # 저장소 설정: 사용자 확인 전에는 꺼져 있다
        doc = v2_doc()
        for value in (0, 30, 12.5):
            with self.subTest(value=value):
                doc["measurement"]["max_settle_s"] = value
                cfg = self.load(doc)
                self.assertEqual(cfg.measurement.max_settle_s, float(value))
                self.assertEqual(json.loads(json.dumps(config.config_snapshot(cfg)))["measurement"]["max_settle_s"], float(value))
        for bad in (-1, "30", None, True):
            with self.subTest(bad=bad):
                doc["measurement"]["max_settle_s"] = bad
                self.assert_rejected(doc, "measurement.max_settle_s")


class LanguageFieldsTest(ConfigTestCase):
    """``languages.<언어>.env`` 와 ``halide_mode`` (설계 v3 §8)."""

    def language(self, entry: dict[str, Any], *, measurement_threads: int = 4) -> config.LanguageCfg:
        doc = v2_doc()
        doc["measurement"]["threads"] = measurement_threads
        doc["languages"]["probe"] = entry
        return self.load(doc).languages["probe"]

    def rejected(self, entry: dict[str, Any], key_path: str) -> str:
        doc = v2_doc()
        doc["languages"]["probe"] = entry
        return self.assert_rejected(doc, key_path)

    def test_threads_placeholder_is_the_resolved_thread_count(self) -> None:
        lang = self.language(V3_LANGUAGES["numba"], measurement_threads=3)
        self.assertEqual((lang.threads, lang.env), (3, {"NUMBA_NUM_THREADS": "3", "NUMBA_THREADING_LAYER": "omp"}))
        pinned = self.language({"backend": "rust", "prompt": "p.md", "threads": 2, "env": {"RAYON_NUM_THREADS": "${threads}"}})
        self.assertEqual((pinned.threads, pinned.env), (2, {"RAYON_NUM_THREADS": "2"}))
        everywhere = self.language({"backend": "python", "prompt": "p.md", "threads": 1,
                                    "env": {"A": "${threads}", "B": "n=${threads},m=${threads}", "C": "plain"}})
        self.assertEqual(everywhere.env, {"A": "1", "B": "n=1,m=1", "C": "plain"})

    def test_env_values_become_strings(self) -> None:
        lang = self.language({"backend": "python", "prompt": "p.md", "threads": 1, "env": {"N": 8, "F": 0.5, "ON": True, "OFF": False}})
        self.assertEqual(lang.env, {"N": "8", "F": "0.5", "ON": "true", "OFF": "false"})
        self.assertEqual(self.language({"backend": "python", "prompt": "p.md", "threads": 1, "env": {}}).env, {})

    def test_env_rejects_what_would_not_reach_the_worker_as_written(self) -> None:
        base = {"backend": "python", "prompt": "p.md", "threads": 1}
        self.rejected({**base, "env": ["A=1"]}, "languages.probe.env")
        self.rejected({**base, "env": {"A": [1]}}, "languages.probe.env.A")
        self.rejected({**base, "env": {"A": None}}, "languages.probe.env.A")
        self.rejected({**base, "env": {1: "x"}}, "languages.probe.env")
        # ${threads} 밖의 자리표시는 오타로 본다: 치환되지 않은 채 워커 환경에 들어가면 런타임이 조용히 기본 스레드 수로 돈다
        for value in ("${thread}", "${cores}", "x${"):
            with self.subTest(value=value):
                self.assertIn("${threads}", self.rejected({**base, "env": {"A": value}}, "languages.probe.env.A"))

    def test_new_backends_must_pin_their_thread_variable(self) -> None:
        self.assertEqual(config.THREAD_ENV, {"numba": "NUMBA_NUM_THREADS", "halide": "HL_NUM_THREADS", "rust": "RAYON_NUM_THREADS"})
        for name, variable in (("numba", "NUMBA_NUM_THREADS"), ("halide_auto", "HL_NUM_THREADS"), ("rust", "RAYON_NUM_THREADS")):
            entry = copy.deepcopy(V3_LANGUAGES[name])
            with self.subTest(name=name, case="ok"):
                self.assertEqual(self.language(entry).env[variable], "4")
            with self.subTest(name=name, case="other value"):
                self.rejected({**entry, "env": {**entry["env"], variable: "8"}}, f"languages.probe.env.{variable}")
            with self.subTest(name=name, case="missing variable"):
                self.rejected({**entry, "env": {k: v for k, v in entry["env"].items() if k != variable}},
                              f"languages.probe.env.{variable}")
            with self.subTest(name=name, case="no env"):
                self.rejected({k: v for k, v in entry.items() if k != "env"}, f"languages.probe.env.{variable}")
        # 스레드 수를 OMP_NUM_THREADS 로 받는 백엔드에는 요구하지 않는다
        self.assertEqual(self.language({"backend": "c", "prompt": "p.md", "threads": 4, "openmp": True}).env, {})

    def test_halide_mode_is_required_for_halide_and_rejected_elsewhere(self) -> None:
        halide = {"backend": "halide", "prompt": "p.md", "threads": 4, "env": {"HL_NUM_THREADS": "${threads}"}}
        for mode in config.HALIDE_MODES:
            self.assertEqual(self.language({**halide, "halide_mode": mode}).halide_mode, mode)
        self.rejected(halide, "languages.probe.halide_mode")
        for bad in ("automatic", "", None, True):
            with self.subTest(bad=bad):
                self.rejected({**halide, "halide_mode": bad}, "languages.probe.halide_mode")
        for backend in ("python", "c", "numba", "rust"):
            with self.subTest(backend=backend):
                self.rejected({"backend": backend, "prompt": "p.md", "threads": 1, "halide_mode": "auto"}, "languages.probe.halide_mode")

    def test_unknown_language_key(self) -> None:
        self.rejected({"backend": "python", "prompt": "p.md", "threads": 1, "environment": {}}, "languages.probe")


class ConditionOverridesTest(ConfigTestCase):
    """``load_config(path, overrides=…)``: CLI 의 ``--args``·``--grammar-hint``·``--feedback`` 이 들어오는 길(설계 v3 §8)."""

    def conditions(self, **kwargs: Any) -> tuple[str, str, str]:
        cond = self.load(v2_doc(), **kwargs).run.conditions
        return cond.args, cond.grammar_hint, cond.feedback

    def test_overrides_replace_only_the_given_conditions(self) -> None:
        self.assertEqual(self.conditions(), ("positional", "off", "compiler"))
        self.assertEqual(self.conditions(overrides=None), ("positional", "off", "compiler"))
        self.assertEqual(self.conditions(overrides={}), ("positional", "off", "compiler"))
        self.assertEqual(self.conditions(overrides={"args": "keyword"}), ("keyword", "off", "compiler"))
        self.assertEqual(self.conditions(overrides={"grammar_hint": "on", "feedback": "compiler+lint"}),
                         ("positional", "on", "compiler+lint"))
        self.assertEqual(self.conditions(overrides={"args": "keyword", "grammar_hint": "on", "feedback": "compiler+lint"}),
                         ("keyword", "on", "compiler+lint"))

    def test_none_keeps_the_value_of_the_file(self) -> None:
        doc = v2_doc()
        doc["run"]["conditions"] = {"args": "keyword", "grammar_hint": "on", "feedback": "compiler+lint"}
        cond = self.load(doc, overrides={"args": None, "grammar_hint": None, "feedback": "compiler"}).run.conditions
        self.assertEqual((cond.args, cond.grammar_hint, cond.feedback), ("keyword", "on", "compiler"))

    def test_overrides_go_through_the_same_validation(self) -> None:
        self.assert_rejected(v2_doc(), "run.conditions.args", overrides={"args": "named"})
        self.assert_rejected(v2_doc(), "run.conditions.grammar_hint", overrides={"grammar_hint": True})
        self.assert_rejected(v2_doc(), "run.conditions.feedback", overrides={"feedback": "lint"})
        self.assert_rejected(v2_doc(), "overrides", overrides={"n_samples": "3"})


class SnapshotTest(ConfigTestCase):
    def test_snapshot_contains_the_v3_values_and_is_json(self) -> None:
        doc = v3_doc()
        doc["halide"]["autoscheduler_params"] = {"last_level_cache_size": 1048576}
        doc["safety"]["enabled"] = True
        cfg = self.load(doc, overrides={"args": "keyword", "feedback": "compiler+lint"})
        snap = json.loads(json.dumps(config.config_snapshot(cfg)))
        self.assertEqual(snap["run"]["conditions"], {"args": "keyword", "grammar_hint": "off", "feedback": "compiler+lint"})
        self.assertEqual(snap["languages"]["numba"]["env"], {"NUMBA_NUM_THREADS": "4", "NUMBA_THREADING_LAYER": "omp"})
        self.assertEqual((snap["languages"]["halide_auto"]["halide_mode"], snap["languages"]["halide_manual"]["halide_mode"],
                          snap["languages"]["numpy"]["halide_mode"]), ("auto", "manual", None))
        self.assertEqual(snap["languages"]["rust"]["backend"], "rust")
        self.assertEqual(snap["build"]["rust"], {"cargo": "cargo", "rustflags": ["-C", "target-cpu=native"], "build_timeout_s": 600.0,
                                                 "offline": True})
        self.assertEqual(snap["build"]["sanitize_flags"], ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"])
        self.assertEqual(snap["halide"], {"autoscheduler": "Mullapudi2016", "autoscheduler_params": {"last_level_cache_size": "1048576"}})
        self.assertEqual(snap["safety"], {"enabled": True, "timeout_s": 300.0})
        self.assertEqual(snap["lint"], {"enabled": True, "plugin": str(ROOT / "plugins" / "lint" / "run_lint.py"), "timeout_s": 60.0})
        self.assertEqual(snap["perf_summary"], {"enabled": True})

    def test_snapshot_of_a_v2_file_has_the_defaults_spelled_out(self) -> None:
        snap = config.config_snapshot(self.load(v2_doc()))
        self.assertEqual(snap["halide"], {"autoscheduler": "Mullapudi2016", "autoscheduler_params": {}})
        self.assertEqual((snap["safety"]["enabled"], snap["lint"]["enabled"], snap["perf_summary"]["enabled"]), (False, True, True))
        self.assertEqual(snap["languages"]["numpy"]["env"], {})

    def test_cli_prints_the_snapshot(self) -> None:
        path = self.write(v3_doc())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(config.main(["--config", str(path)]), 0)
        printed = json.loads(out.getvalue())
        self.assertEqual(printed["config"]["languages"]["halide_auto"]["env"], {"HL_NUM_THREADS": "4"})
        bad = v3_doc()
        bad["languages"]["halide_auto"]["halide_mode"] = "automatic"
        path = self.write(bad)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(config.main(["--config", str(path)]), 2)
        self.assertIn("languages.halide_auto.halide_mode", err.getvalue())


if __name__ == "__main__":
    unittest.main()
