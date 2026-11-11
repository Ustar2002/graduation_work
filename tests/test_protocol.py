"""protocol 테스트: /proc/stat 파서·사용률 계산, launch_prefix/launch_env 규칙, 플래그 판정, 실제 동적 검사."""

from __future__ import annotations

import os
import shutil
import unittest
from unittest import mock

from harness.config import load_config, parse_cpulist
from harness.protocol import (
    DynamicStatus,
    StaticStatus,
    blas_flag_label,
    busy_pct,
    check_dynamic,
    dropped_env_overrides,
    environment_fingerprint,
    flags_ok,
    is_informational,
    launch_env,
    launch_prefix,
    parse_proc_stat,
    sibling_cpus,
)

PROC_STAT_BEFORE = """cpu  100 0 100 800 0 0 0 0 0 0
cpu0 10 0 10 80 0 0 0 0 0 0
cpu1 10 0 10 80 0 0 0 0 0 0
cpu2 0 0 0 100 0 0 0 0 0 0
intr 12345 0 0
ctxt 999
btime 1700000000
"""

PROC_STAT_AFTER = """cpu  200 0 200 1600 0 0 0 0 0 0
cpu0 40 0 20 140 0 0 0 0 0 0
cpu1 10 0 10 180 0 0 0 0 0 0
cpu2 0 0 0 150 50 0 0 0 0 0
"""


class ProcStatParserTest(unittest.TestCase):
    def test_parse_proc_stat_skips_aggregate_and_non_cpu_lines(self) -> None:
        parsed = parse_proc_stat(PROC_STAT_BEFORE)
        self.assertEqual(sorted(parsed), [0, 1, 2])
        self.assertEqual(parsed[0], (10, 0, 10, 80, 0, 0, 0, 0, 0, 0))

    def test_busy_pct(self) -> None:
        before = parse_proc_stat(PROC_STAT_BEFORE)
        after = parse_proc_stat(PROC_STAT_AFTER)
        # cpu0: total 100→200 (+100), idle 80→140 (+60) → busy 40%
        self.assertAlmostEqual(busy_pct(before, after, [0]), 40.0)
        # cpu1: +100 total, +100 idle → 0%
        self.assertAlmostEqual(busy_pct(before, after, [1]), 0.0)
        # cpu2: iowait 도 idle 로 센다 → +100 total, idle+iowait +100 → 0%
        self.assertAlmostEqual(busy_pct(before, after, [2]), 0.0)
        # 평균
        self.assertAlmostEqual(busy_pct(before, after, [0, 1]), 20.0)
        # 없는 CPU 는 무시, 전부 없으면 0.0
        self.assertAlmostEqual(busy_pct(before, after, [0, 99]), 40.0)
        self.assertEqual(busy_pct(before, after, [99]), 0.0)
        # 변화가 없으면(동일 샘플) 0.0
        self.assertEqual(busy_pct(before, before, [0, 1, 2]), 0.0)

    def test_parse_cpulist(self) -> None:
        self.assertEqual(parse_cpulist("0,2,4,6"), [0, 2, 4, 6])
        self.assertEqual(parse_cpulist("0-3,8"), [0, 1, 2, 3, 8])
        self.assertEqual(parse_cpulist("2"), [2])
        with self.assertRaises(ValueError):
            parse_cpulist("0, 2")
        with self.assertRaises(ValueError):
            parse_cpulist("a")


class FlagsTest(unittest.TestCase):
    def test_informational_flags_do_not_break_ok(self) -> None:
        self.assertTrue(is_informational("turbo=on"))
        self.assertTrue(is_informational("env_overrides_dropped"))
        self.assertFalse(is_informational("governor=powersave"))
        self.assertTrue(flags_ok([]))
        self.assertTrue(flags_ok(["turbo=on", "env_overrides_dropped"]))
        self.assertFalse(flags_ok(["turbo=on", "governor=powersave"]))
        self.assertTrue(flags_ok(["load_1m=2.4>2.0"]))  # 2026-11-08: 자기 부하라 정보용
        self.assertFalse(flags_ok(["pinned_busy=23.1%>10%"]))

    def test_blas_flag_label(self) -> None:
        self.assertEqual(blas_flag_label(""), "unknown")
        self.assertEqual(blas_flag_label("unknown"), "unknown")
        self.assertEqual(blas_flag_label("blas"), "reference")
        self.assertEqual(blas_flag_label("MKL"), "mkl")

    def test_dropped_env_overrides(self) -> None:
        env = {"PATH": "/bin", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "2", "KMP_AFFINITY": "x", "GOMP_CPU_AFFINITY": "0", "MKL_DEBUG": "1", "HOME": "/h"}
        self.assertEqual(dropped_env_overrides(env), ["GOMP_CPU_AFFINITY", "KMP_AFFINITY", "MKL_DEBUG", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
        self.assertEqual(dropped_env_overrides({"PATH": "/bin"}), [])


class LaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()

    @unittest.skipIf(shutil.which("taskset") is None, "taskset 필요")
    def test_launch_prefix(self) -> None:
        self.assertEqual(launch_prefix(self.cfg, 1), ["taskset", "-c", self.cfg.measurement.single_core])
        self.assertEqual(launch_prefix(self.cfg, 4), ["taskset", "-c", self.cfg.measurement.cores])
        self.assertEqual(launch_prefix(self.cfg, self.cfg.measurement.threads), ["taskset", "-c", "0,2,4,6"])

    def test_launch_prefix_without_taskset(self) -> None:
        with mock.patch("harness.protocol.shutil.which", return_value=None):
            self.assertEqual(launch_prefix(self.cfg, 4), [])

    def test_launch_env_whitelist_and_threads(self) -> None:
        user_env = {
            "PATH": "/usr/bin:/bin", "HOME": "/home/x", "LANG": "ko_KR.UTF-8", "TMPDIR": "/tmp/x", "VIRTUAL_ENV": "/v",
            "LC_ALL": "ko_KR.UTF-8", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "16", "KMP_BLOCKTIME": "0",
            "ANTHROPIC_API_KEY": "secret", "PYTHONHASHSEED": "123", "SOME_RANDOM_VAR": "1",
        }
        with mock.patch.dict(os.environ, user_env, clear=True), mock.patch("harness.protocol._warned_env_drop", set()):
            with mock.patch("sys.stderr") as err:
                env = launch_env(self.cfg, 4, openmp=False)
            self.assertTrue(any("OMP_NUM_THREADS" in str(c) for c in err.write.call_args_list), "제거된 변수 경고가 있어야 한다")
        # 상속 화이트리스트
        for k in ("PATH", "HOME", "LANG", "TMPDIR", "VIRTUAL_ENV"):
            self.assertEqual(env[k], user_env[k])
        # 고정값
        self.assertEqual(env["LC_ALL"], "C")
        self.assertEqual(env["PYTHONHASHSEED"], "0")
        # 스레드 변수는 사용자 값이 아니라 threads
        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "4")
        self.assertEqual(env["MKL_NUM_THREADS"], "4")
        # 비밀·잡동사니·사용자 스레드 변수는 상속하지 않는다
        for k in ("ANTHROPIC_API_KEY", "SOME_RANDOM_VAR", "KMP_BLOCKTIME"):
            self.assertNotIn(k, env)
        # openmp=False 면 OMP_PROC_BIND 등 없음
        self.assertNotIn("OMP_PROC_BIND", env)

    def test_launch_env_openmp_extra(self) -> None:
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
            env = launch_env(self.cfg, 1, openmp=True)
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["OMP_PROC_BIND"], "true")
        self.assertEqual(env["OMP_PLACES"], "cores")
        self.assertEqual(env["OMP_DYNAMIC"], "false")
        self.assertNotIn("HOME", env)  # 사용자 환경에 없던 것은 생기지 않는다


class CallCpusTest(unittest.TestCase):
    """bench 에서 코드를 호출하는 동안의 주 스레드 CPU 집합(2026-11-11)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()

    def test_openmp_c_is_left_to_libgomp(self) -> None:
        from harness.protocol import call_cpus
        self.assertIsNone(call_cpus(self.cfg, backend="c", openmp=True, threads=self.cfg.measurement.threads))

    def test_python_gets_all_pinned_cores(self) -> None:
        from harness.protocol import call_cpus
        self.assertEqual(call_cpus(self.cfg, backend="python", openmp=False, threads=self.cfg.measurement.threads),
                         self.cfg.measurement.cores)

    def test_sequential_c_gets_first_pinned_cpu(self) -> None:
        """순차 C 는 고정 코어의 첫 CPU(= OpenMP 주 스레드의 place) 하나에서 돈다: libgomp 작업 스레드의 코어와 겹치지 않는다."""
        from harness.protocol import call_cpus
        first = str(parse_cpulist(self.cfg.measurement.cores)[0])
        self.assertEqual(call_cpus(self.cfg, backend="c", openmp=False, threads=1), first)
        self.assertEqual(call_cpus(self.cfg, backend="python", openmp=False, threads=1), first)

    def test_call_cpus_stay_inside_the_bench_taskset(self) -> None:
        """measurement.threads == 1 이면 bench 워커는 single_core 에 고정된다: 호출 CPU 도 그 안에서 골라야 한다."""
        import dataclasses

        from harness.protocol import bench_cpus, call_cpus, launch_prefix
        cfg1 = dataclasses.replace(self.cfg, measurement=dataclasses.replace(self.cfg.measurement, threads=1))
        self.assertEqual(bench_cpus(cfg1), cfg1.measurement.single_core)
        self.assertEqual(bench_cpus(self.cfg), self.cfg.measurement.cores)
        for cfg in (self.cfg, cfg1):
            allowed = set(parse_cpulist(bench_cpus(cfg)))
            prefix = launch_prefix(cfg, cfg.measurement.threads)
            if prefix:
                self.assertEqual(set(parse_cpulist(prefix[-1])), allowed)
            for backend, openmp, threads in (("python", False, cfg.measurement.threads), ("c", False, 1), ("python", False, 1)):
                got = call_cpus(cfg, backend=backend, openmp=openmp, threads=threads)
                self.assertIsNotNone(got)
                self.assertLessEqual(set(parse_cpulist(got)), allowed, (backend, threads, got))

    def test_worker_resolves_cpu_arguments(self) -> None:
        from harness.worker import resolve_call_cpus
        self.assertEqual(resolve_call_cpus("auto", "python", [0, 2, 4, 6]), (0, 2, 4, 6))
        self.assertIsNone(resolve_call_cpus("auto", "c", [0, 2, 4, 6]))
        self.assertIsNone(resolve_call_cpus("none", "python", [0, 2]))
        self.assertEqual(resolve_call_cpus("2", "c", [0, 2, 4, 6]), (2,))
        self.assertEqual(resolve_call_cpus("0,2,4-6", "python", []), (0, 2, 4, 5, 6))
        with self.assertRaises(ValueError):
            resolve_call_cpus(",", "python", [0])


class LiveChecksTest(unittest.TestCase):
    """실제 sysfs/proc 를 읽는 검사(이 머신에서 1초)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()

    def test_sibling_cpus_excludes_pinned(self) -> None:
        pinned = parse_cpulist(self.cfg.measurement.cores)
        sibs = sibling_cpus(pinned)
        self.assertFalse(set(sibs) & set(pinned))

    def test_check_dynamic_shape(self) -> None:
        st = check_dynamic(self.cfg, sample_s=0.3)
        self.assertIsInstance(st, DynamicStatus)
        self.assertGreaterEqual(st.load_1m, 0.0)
        self.assertTrue(0.0 <= st.pinned_busy_pct <= 100.0)
        self.assertTrue(0.0 <= st.sibling_busy_pct <= 100.0)
        self.assertEqual(st.ok, flags_ok(st.flags))
        self.assertEqual(st.busy_ok, st.pinned_busy_pct <= self.cfg.measurement.max_pinned_busy_pct
                         and st.sibling_busy_pct <= self.cfg.measurement.max_pinned_busy_pct)
        self.assertRegex(st.checked_at, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
        if not st.governor_ok and self.cfg.measurement.require_performance_governor:
            self.assertIn(f"governor={st.governor}", st.flags)

    def test_environment_fingerprint_is_deterministic(self) -> None:
        static = StaticStatus(
            governor="powersave", governor_ok=False, epp="performance", epp_ok=True, no_turbo=0, taskset_available=True,
            cores="0,2,4,6", single_core="2", threads=4,
            blas={"name": "scipy-openblas", "version": "0.3.30", "config": "OpenBLAS 0.3.30", "loaded_lib": "/x.so"},
            blas_ok=True, flags=["governor=powersave", "turbo=on"], ok=False, checked_at="2026-11-04T10:00:00+09:00",
        )
        machine = {
            "cpu_model": "cpu", "kernel": "k", "glibc": "2.43", "compilers": {"gcc": "gcc 15"}, "march_resolved": "-march=x",
            "python_version": "3.14.4", "numpy_version": "2.3.5", "sdk_versions": {"anthropic": "2.4.0", "openai": "3.44.0"},
        }
        fp1, h1 = environment_fingerprint(self.cfg, machine, static)
        fp2, h2 = environment_fingerprint(self.cfg, machine, static)
        self.assertEqual(h1, h2)
        self.assertRegex(h1, r"^[0-9a-f]{16}$")
        self.assertEqual(fp1["cores"], "0,2,4,6")
        self.assertIn("-O3", fp1["flags"])
        self.assertIn("-fopenmp", fp1["flags"])
        self.assertEqual(fp1["numpy_blas"]["name"], "scipy-openblas")
        self.assertEqual(fp1["governor"], "powersave")
        # 거버너가 바뀌면 지문도 바뀐다
        static2 = StaticStatus(**{**static.__dict__, "governor": "performance"})
        _, h3 = environment_fingerprint(self.cfg, machine, static2)
        self.assertNotEqual(h1, h3)


if __name__ == "__main__":
    unittest.main()
