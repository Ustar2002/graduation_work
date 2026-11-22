"""plots 테스트: 합성 run(tests.test_aggregate.synthetic_rows)으로 그림 네 장의 핵심 요소(패널 수, 언어 축 순서, 점의 수와 값,
막대 높이의 합, 구간)를 확인하고, CLI 가 파일을 쓰는지와 저장소의 실제 결과 파일이 그려지는지를 본다.

기대값은 tests/test_aggregate.py 의 합성 자료 설명과 같은 손계산이다(그림 1·2 는 기준 조건의 턴 0 행만 쓴다).
"""

from __future__ import annotations

import contextlib
import io
import runpy
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import analysis.aggregate as aggregate
from analysis.aggregate import EXIT_NOT_FOUND, EXIT_OK, LANGUAGE_ORDER, OUTCOME_ORDER, choose_speedup, load_runs, select_runs
from analysis.plots import (
    OUTCOME_STYLE,
    SERIES_COLORS,
    NoData,
    fig_condition_effects,
    fig_error_types,
    fig_languages,
    fig_repair_turns,
    language_styles,
    main,
    reference_condition,
)
from harness import procutil
from tests.test_aggregate import DUMMY_CSV, REAL_CSV, REAL_RUN, synthetic_rows, write_rows, write_task_specs

LANGUAGES = ["numpy", "halide_auto", "rust", "c_seq"]   # 합성 run 의 언어, 성능 결정 부담 순(CLAUDE.md §1: rust 가 C 앞)
OKABE_ITO = {"#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"}


def _lines(ax: Axes) -> dict[str, Line2D]:
    """gid 가 붙은 선·점 묶음(gid → Line2D)."""
    return {line.get_gid(): line for line in ax.lines if line.get_gid()}


def _texts(ax: Axes) -> list[str]:
    return [text.get_text() for text in ax.texts]


def _tick_labels(ax: Axes, axis: str = "x") -> list[str]:
    labels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    return [label.get_text() for label in labels]


def _intervals(ax: Axes, horizontal: bool = False) -> dict[str, tuple[float, float, float]]:
    """errorbar 묶음마다 gid → (점, 구간 하한, 구간 상한)."""
    out: dict[str, tuple[float, float, float]] = {}
    for container in ax.containers:
        data_line, _caps, bars = container
        (start, end), = bars[0].get_segments()
        k = 0 if horizontal else 1
        point = data_line.get_xdata()[0] if horizontal else data_line.get_ydata()[0]
        out[data_line.get_gid()] = (float(point), float(start[k]), float(end[k]))
    return out


def _bar_heights(ax: Axes) -> dict[str, list[float]]:
    """누적 막대의 칸마다 범례 글자 → 언어 순서대로의 높이."""
    return {container.get_label(): [float(patch.get_height()) for patch in container] for container in ax.containers}


class SyntheticFigureCase(unittest.TestCase):
    """합성 run 을 읽어 두는 공통 준비."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="plots_test_")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.csv = self.tmp / "runs.csv"
        write_rows(self.csv, synthetic_rows())
        self.tasks_dir = write_task_specs(self.tmp / "tasks")
        self.df = load_runs(self.csv)

    def assert_close(self, got: object, want: object) -> None:
        np.testing.assert_allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float), rtol=0, atol=1e-12, equal_nan=True)

    def reference_only(self) -> list[dict[str, str]]:
        """기준 조건(args=positional, grammar_hint=off, feedback=compiler)의 턴 0 행만."""
        return [
            r for r in synthetic_rows()
            if (r["args"], r["grammar_hint"], r["feedback"], r["turn"]) == ("positional", "off", "compiler", "0")
        ]


class StyleTest(unittest.TestCase):
    def test_every_outcome_has_a_distinct_colour_and_hatch(self) -> None:
        self.assertEqual(set(OUTCOME_STYLE), set(OUTCOME_ORDER))
        fills = [(color, hatch) for color, hatch, _label in OUTCOME_STYLE.values()]
        self.assertEqual(len(set(fills)), len(fills))
        self.assertEqual(len({label for _color, _hatch, label in OUTCOME_STYLE.values()}), len(OUTCOME_STYLE))
        self.assertLessEqual({color for color, _hatch in fills}, OKABE_ITO | {"#FFFFFF"})
        self.assertEqual(OUTCOME_STYLE["pass"][0], OUTCOME_STYLE["pass_bench_timeout"][0])   # 같은 색, 빗금으로 구분
        self.assertNotEqual(OUTCOME_STYLE["pass"][1], OUTCOME_STYLE["pass_bench_timeout"][1])

    def test_languages_get_distinct_markers_and_safe_colours(self) -> None:
        styles = language_styles(list(LANGUAGE_ORDER))
        self.assertEqual(len({style["marker"] for style in styles.values()}), len(LANGUAGE_ORDER))
        # 이 저장소에서 돌리는 언어 7개는 색도 모두 다르다(Okabe–Ito 의 노랑 외 7색). mojo(범위 밖)는 회색이다.
        experiment = ["numpy", "numba", "halide_auto", "halide_manual", "rust", "c_openmp", "c_seq"]
        self.assertEqual(len({styles[language]["color"] for language in experiment}), 7)
        self.assertLessEqual({styles[language]["color"] for language in experiment}, OKABE_ITO)
        self.assertEqual(styles["mojo"]["color"], "#999999")
        self.assertLessEqual(set(SERIES_COLORS), OKABE_ITO)
        self.assertNotIn("#F0E442", SERIES_COLORS)   # 노랑은 선·점에 쓰지 않는다

    def test_language_style_does_not_depend_on_the_other_languages(self) -> None:
        alone = language_styles(["c_seq"])["c_seq"]
        self.assertEqual(language_styles(["rust", "c_seq", "numpy"])["c_seq"], alone)
        extra = language_styles(["zig", "numpy", "ada"])
        self.assertNotEqual(extra["ada"]["marker"], extra["zig"]["marker"])
        self.assertEqual(extra["numpy"], language_styles(["numpy"])["numpy"])

    def test_reference_condition_is_the_baseline_combination(self) -> None:
        with tempfile.TemporaryDirectory(prefix="plots_ref_") as tmp:
            csv_path = Path(tmp) / "runs.csv"
            write_rows(csv_path, synthetic_rows())
            self.assertEqual(
                reference_condition(load_runs(csv_path)),
                {"args": "positional", "grammar_hint": "off", "feedback": "compiler"},
            )
            write_rows(csv_path, [r for r in synthetic_rows() if r["args"] == "keyword" or r["grammar_hint"] == "on"])
            self.assertEqual(   # positional 이 keyword 보다 앞선다 → grammar_hint=on 조합이 기준
                reference_condition(load_runs(csv_path)),
                {"args": "positional", "grammar_hint": "on", "feedback": "compiler"},
            )


class FigLanguagesTest(SyntheticFigureCase):
    def setUp(self) -> None:
        super().setUp()
        self.fig = fig_languages(self.df, self.tasks_dir)
        self.pass_axes = {ax.get_title(): ax for ax in self.fig.axes if ax.get_title()}
        self.speed_axes = dict(zip(self.pass_axes, (ax for ax in self.fig.axes if ax.get_yscale() == "log")))

    def test_one_pair_of_panels_per_task(self) -> None:
        self.assertIsInstance(self.fig, Figure)
        self.assertEqual(len(self.fig.axes), 4)
        self.assertEqual(list(self.pass_axes), ["gemm", "sort"])
        self.assertEqual(len(self.speed_axes), 2)

    def test_languages_are_in_burden_order_on_every_panel(self) -> None:
        for task, ax in self.speed_axes.items():
            with self.subTest(task=task):
                self.assertEqual(_tick_labels(ax), LANGUAGES)
                self.assertEqual(list(ax.get_xticks()), [0, 1, 2, 3])
                self.assertEqual(ax.get_xlim(), self.pass_axes[task].get_xlim())   # 위아래 패널이 x 축을 공유한다

    def test_pass_rate_with_bootstrap_interval(self) -> None:
        # 기준 조건의 턴 0: gemm numpy 3/5, halide_auto 5/5, c_seq 2/5, rust 3/4 (API 오류 하나는 분모에서 빠진다)
        got = _intervals(self.pass_axes["gemm"])
        self.assertEqual(sorted(got), ["pass:gemm:c_seq", "pass:gemm:halide_auto", "pass:gemm:numpy", "pass:gemm:rust"])
        self.assert_close(got["pass:gemm:numpy"], (0.6, 0.2, 1.0))
        self.assert_close(got["pass:gemm:halide_auto"], (1.0, 1.0, 1.0))
        self.assert_close(got["pass:gemm:c_seq"], (0.4, 0.0, 0.8))
        self.assert_close(got["pass:gemm:rust"], (0.75, 0.25, 1.0))
        got = _intervals(self.pass_axes["sort"])
        self.assertEqual(sorted(got), ["pass:sort:c_seq", "pass:sort:numpy"])   # halide_auto 는 unsupported, rust 는 행이 없다
        self.assert_close(got["pass:sort:numpy"], (0.8, 0.4, 1.0))
        self.assert_close(got["pass:sort:c_seq"], (0.2, 0.0, 0.6))
        x_positions = {gid: line.get_xdata()[0] for gid, line in _lines(self.pass_axes["gemm"]).items()}
        self.assertEqual(x_positions, {"pass:gemm:numpy": 0, "pass:gemm:halide_auto": 1, "pass:gemm:rust": 2, "pass:gemm:c_seq": 3})

    def test_every_passing_sample_is_a_point_and_the_median_is_marked(self) -> None:
        lines = _lines(self.speed_axes["gemm"])
        samples = {gid.split(":")[2]: sorted(line.get_ydata()) for gid, line in lines.items() if gid.startswith("samples:")}
        self.assertEqual(sorted(samples), sorted(LANGUAGES))
        self.assert_close(samples["numpy"], [0.9, 1.0, 1.1])                # 턴 1·2 에서 통과한 샘플은 그리지 않는다
        self.assert_close(samples["halide_auto"], [2.0, 2.5, 3.0, 3.5, 4.0])
        self.assert_close(samples["c_seq"], [0.30, 0.35])
        self.assert_close(samples["rust"], [1.2, 1.5])                      # bench timeout 샘플은 통과지만 시간이 없다
        medians = {gid.split(":")[2]: line.get_ydata() for gid, line in lines.items() if gid.startswith("median:")}
        self.assert_close(medians["numpy"], [1.0, 1.0])
        self.assert_close(medians["halide_auto"], [3.0, 3.0])
        self.assert_close(medians["c_seq"], [0.325, 0.325])
        self.assert_close(medians["rust"], [1.35, 1.35])
        for language, x in zip(LANGUAGES, range(4)):
            self.assertTrue(np.all(np.abs(lines[f"samples:gemm:{language}"].get_xdata() - x) < 0.25))   # 제 언어 자리 안에서만 벌린다
        lines = _lines(self.speed_axes["sort"])
        self.assertEqual(sorted(lines), ["median:sort:c_seq", "median:sort:numpy", "samples:sort:c_seq", "samples:sort:numpy"])
        self.assert_close(sorted(lines["samples:sort:numpy"].get_ydata()), [0.5, 0.6, 0.7, 0.8])
        self.assert_close(lines["samples:sort:c_seq"].get_ydata(), [0.2])
        self.assert_close(lines["median:sort:numpy"].get_ydata(), [0.65, 0.65])

    def test_n_is_annotated_and_unsupported_is_marked(self) -> None:
        self.assertEqual(_texts(self.pass_axes["gemm"]), ["n=5", "n=5", "n=4", "n=5"])
        self.assertEqual(_texts(self.speed_axes["gemm"]), ["n=3", "n=5", "n=2", "n=2"])
        self.assertEqual(_texts(self.pass_axes["sort"]), ["n=5", "n/a", "n=5"])
        self.assertEqual(_texts(self.speed_axes["sort"]), ["n=4", "n=1"])
        na = next(text for text in self.pass_axes["sort"].texts if text.get_text() == "n/a")
        self.assertEqual(na.get_position()[0], LANGUAGES.index("halide_auto"))

    def test_axes_are_labelled_with_units(self) -> None:
        self.assertIn("pass@1", self.pass_axes["gemm"].get_ylabel())
        self.assertIn("fraction", self.pass_axes["gemm"].get_ylabel())
        self.assertIn("speedup vs baseline", self.speed_axes["gemm"].get_ylabel())
        self.assertIn("×", self.speed_axes["gemm"].get_ylabel())
        self.assertIn("log", self.speed_axes["gemm"].get_ylabel())
        for ax in self.speed_axes.values():
            self.assertIn("language", ax.get_xlabel())
        low, high = self.speed_axes["gemm"].get_ylim()
        self.assertLess(low, 0.2)
        self.assertGreater(high, 4.0)
        self.assertEqual(self.speed_axes["sort"].get_ylim(), (low, high))   # 과제 사이에 같은 세로축
        title = self.fig.get_suptitle()
        self.assertIn("args=positional, grammar_hint=off,", title)
        self.assertIn("synth-model", title)

    def test_without_task_specs_only_the_decided_halide_exclusions_are_marked(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            fig = fig_languages(self.df, self.tmp / "no_such_tasks")
        self.assertIn("unsupported 언어를 모른다", err.getvalue())
        pass_axes = {ax.get_title(): ax for ax in fig.axes if ax.get_title()}
        self.assertNotIn("n/a", _texts(pass_axes["gemm"]))
        # sort × Halide 는 명세가 없어도 결정(aggregate.HALIDE_UNSUPPORTED_TASKS)으로 unsupported 다
        self.assertEqual(_texts(pass_axes["sort"]), ["n=5", "n/a", "n=5"])

    def test_tasks_follow_the_task_order_and_bfs_halide_is_not_applicable(self) -> None:
        rows = [
            *(dict(r, task="dot") for r in self.reference_only() if r["task"] == "gemm"),
            *(dict(r, task="jacobi2d") for r in self.reference_only() if r["task"] == "gemm"),
            *(dict(r, task="bfs") for r in self.reference_only() if r["task"] == "sort"),
        ]
        write_rows(self.csv, rows)
        with contextlib.redirect_stderr(io.StringIO()):
            fig = fig_languages(load_runs(self.csv), self.tmp / "no_such_tasks")
        pass_axes = {ax.get_title(): ax for ax in fig.axes if ax.get_title()}
        self.assertEqual(list(pass_axes), ["jacobi2d", "dot", "bfs"])   # 이름순이 아니라 aggregate.TASK_ORDER 순
        self.assertIn("n/a", _texts(pass_axes["bfs"]))

    def test_speedup_axis_uses_the_primary_measure(self) -> None:
        # 모든 통과·측정 행에 speedup_mean(= 중앙값 비의 2배)이 있으면 평균 비를 그린다
        rows = [dict(r, speedup_mean=str(2 * float(r["speedup_vs_baseline"])) if r["speedup_vs_baseline"] else "")
                for r in self.reference_only()]
        write_rows(self.csv, rows)
        df = load_runs(self.csv)
        fig = fig_languages(df, self.tasks_dir, speedup=choose_speedup(df, "mean"))
        speed = next(ax for ax in fig.axes if ax.get_yscale() == "log")
        self.assert_close(sorted(_lines(speed)["samples:gemm:numpy"].get_ydata()), [1.8, 2.0, 2.2])
        self.assertIn("ratio of means", speed.get_ylabel())
        self.assertIn("speedup = speedup_mean", fig.get_suptitle())
        fig = fig_languages(df, self.tasks_dir, speedup=choose_speedup(df, "median"))
        speed = next(ax for ax in fig.axes if ax.get_yscale() == "log")
        self.assert_close(sorted(_lines(speed)["samples:gemm:numpy"].get_ydata()), [0.9, 1.0, 1.1])
        self.assertIn("ratio of medians", speed.get_ylabel())
        # 한 행이라도 speedup_mean 이 비면(이전 규약) 설정이 mean 이어도 중앙값 비로 그린다
        rows[0] = dict(rows[0], speedup_mean="")
        write_rows(self.csv, rows)
        df = load_runs(self.csv)
        fig = fig_languages(df, self.tasks_dir, speedup=choose_speedup(df, "mean"))
        speed = next(ax for ax in fig.axes if ax.get_yscale() == "log")
        self.assert_close(sorted(_lines(speed)["samples:gemm:numpy"].get_ydata()), [0.9, 1.0, 1.1])

    def test_no_valid_first_attempt_is_no_data(self) -> None:
        write_rows(self.csv, [r for r in synthetic_rows() if r["valid_sample"] == "0" and r["turn"] == "0"])
        with self.assertRaises(NoData):
            fig_languages(load_runs(self.csv), self.tasks_dir)
        with self.assertRaises(NoData):
            fig_languages(self.df.iloc[:0], self.tasks_dir)

    def test_runs_without_timed_samples_still_draw(self) -> None:
        rows = [dict(r, speedup_vs_baseline="", time_median_ms="") for r in self.reference_only()]
        write_rows(self.csv, rows)
        fig = fig_languages(load_runs(self.csv), self.tasks_dir)
        speed_axes = [ax for ax in fig.axes if ax.get_yscale() == "log"]
        self.assertEqual([_lines(ax) for ax in speed_axes], [{}, {}])
        self.assertEqual(_texts(speed_axes[0]), ["n=0", "n=0", "n=0", "n=0"])


class FigErrorTypesTest(SyntheticFigureCase):
    def test_stacked_bars_sum_to_the_number_of_rows(self) -> None:
        fig = fig_error_types(self.df)
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        self.assertEqual(_tick_labels(ax), LANGUAGES)
        heights = _bar_heights(ax)
        # 기준 조건의 턴 0 행: numpy 10(gemm 5 + sort 5), halide_auto 5, rust 5(무효인 API 오류 행도 한 칸), c_seq 10
        self.assert_close(np.sum(list(heights.values()), axis=0), [10, 5, 5, 10])
        self.assertEqual(heights, {
            "pass": [7.0, 5.0, 2.0, 3.0],
            "pass (bench timeout)": [0.0, 0.0, 1.0, 0.0],
            "compile_error": [1.0, 0.0, 0.0, 1.0],
            "runtime_crash": [0.0, 0.0, 0.0, 1.0],
            "timeout": [0.0, 0.0, 0.0, 1.0],
            "wrong_output": [2.0, 0.0, 0.0, 2.0],
            "race_condition": [0.0, 0.0, 1.0, 0.0],
            "signature_mismatch": [0.0, 0.0, 0.0, 1.0],
            "generate: refusal": [0.0, 0.0, 0.0, 1.0],
            "generate: API error": [0.0, 0.0, 1.0, 0.0],
        })
        self.assertEqual(_texts(ax), ["n=10", "n=5", "n=5", "n=10"])
        self.assertEqual(fig.get_supylabel(), "samples (count)")
        self.assertIn("language", ax.get_xlabel())
        low, high = ax.get_ylim()
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 10.0)   # 막대 위에 n 을 적을 자리

    def test_segments_are_stacked_without_gaps_in_the_fixed_order(self) -> None:
        ax = fig_error_types(self.df).axes[0]
        order = [OUTCOME_STYLE[outcome][2] for outcome in OUTCOME_ORDER]
        labels = [container.get_label() for container in ax.containers]
        self.assertEqual(labels, [label for label in order if label in labels])
        top = np.zeros(len(LANGUAGES))
        for container in ax.containers:
            self.assert_close([patch.get_y() for patch in container], top)
            top += [patch.get_height() for patch in container]
        legend = fig_error_types(self.df).legends[0]
        self.assertEqual([text.get_text() for text in legend.get_texts()], labels)

    def test_hatches_distinguish_segments_of_the_same_colour(self) -> None:
        containers = {container.get_label(): container for container in fig_error_types(self.df).axes[0].containers}
        plain, timed_out = containers["pass"][0], containers["pass (bench timeout)"][0]
        self.assertEqual(plain.get_facecolor(), timed_out.get_facecolor())
        self.assertNotEqual(plain.get_hatch(), timed_out.get_hatch())
        self.assertNotEqual(containers["generate: refusal"][0].get_hatch(), containers["generate: API error"][0].get_hatch())

    def test_by_task_draws_one_panel_per_task(self) -> None:
        fig = fig_error_types(self.df, by_task=True)
        self.assertEqual([ax.get_title() for ax in fig.axes], ["gemm", "sort"])
        gemm, sort = (_bar_heights(ax) for ax in fig.axes)
        self.assert_close(np.sum(list(gemm.values()), axis=0), [5, 5, 5, 5])
        self.assert_close(np.sum(list(sort.values()), axis=0), [5, 0, 0, 5])
        self.assertEqual(gemm["pass"], [3.0, 5.0, 2.0, 2.0])
        self.assertEqual(sort["pass"], [4.0, 0.0, 0.0, 1.0])
        self.assertEqual(sort["wrong_output"], [1.0, 0.0, 0.0, 2.0])
        self.assertEqual(list(gemm), list(sort))                       # 패널마다 같은 범주·같은 색
        for ax in fig.axes:
            self.assertEqual(_tick_labels(ax), LANGUAGES)
        self.assertEqual(_texts(fig.axes[1]), ["n=5", "n=5"])

    def test_no_rows_is_no_data(self) -> None:
        with self.assertRaises(NoData):
            fig_error_types(self.df.iloc[:0])


class FigConditionEffectsTest(SyntheticFigureCase):
    def test_one_panel_per_comparison_and_one_row_per_task_language(self) -> None:
        fig = fig_condition_effects(self.df)
        self.assertEqual(
            [ax.get_title() for ax in fig.axes],
            ["args: keyword − positional", "grammar_hint: on − off", "feedback: compiler+lint − compiler"],
        )
        self.assertEqual(_tick_labels(fig.axes[0], "y"), ["gemm / numpy", "gemm / c_seq"])
        args, hint, feedback = fig.axes
        # (차, 구간 하한, 구간 상한): tests/test_aggregate.py ConditionEffectsTest 와 같은 값
        self.assertEqual(list(_intervals(args, horizontal=True)), ["effect:args:gemm:numpy"])
        self.assert_close(_intervals(args, horizontal=True)["effect:args:gemm:numpy"], (0.4, 0.0, 0.8))
        self.assert_close(_intervals(hint, horizontal=True)["effect:grammar_hint:gemm:c_seq"], (0.2, -0.4, 0.8))
        self.assert_close(_intervals(feedback, horizontal=True)["effect:feedback:gemm:c_seq"], (0.2, 0.0, 0.6))
        self.assert_close(_lines(args)["effect:args:gemm:numpy"].get_ydata(), [0])       # 행 0 = gemm / numpy
        self.assert_close(_lines(hint)["effect:grammar_hint:gemm:c_seq"].get_ydata(), [1])
        self.assert_close(_lines(feedback)["effect:feedback:gemm:c_seq"].get_ydata(), [1])
        # feedback 은 모든 짝의 결과를 아는 수정 턴 수(H = 2)까지의 통과율이다
        self.assertEqual((_texts(args), _texts(hint), _texts(feedback)), (["n=5/5"], ["n=5/5"], ["n=5 pairs, turns≤2"]))
        for ax in fig.axes:
            self.assertEqual(ax.get_xlim(), (-1.05, 1.05))
            self.assertIn("fraction of samples", ax.get_xlabel())
        self.assertIn("pass@1", args.get_xlabel())
        self.assertIn("repair turns", feedback.get_xlabel())
        markers = {ax.get_title().split(":")[0]: next(iter(_lines(ax).values())).get_marker() for ax in fig.axes}
        self.assertEqual(len(set(markers.values())), 3)   # 조건마다 표식이 다르다

    def test_several_comparisons_in_one_row_are_spread_apart(self) -> None:
        # gemm numpy 의 턴 0 행(positional·keyword)을 grammar_hint=on 으로 한 벌 더 두면, args 비교가 hint=off 와 hint=on
        # 두 군데에서 생기고(둘 다 3/5 대 5/5) grammar_hint 비교는 numpy 행에 둘(차 0), c_seq 행에 하나가 된다
        numpy_first = [r for r in synthetic_rows() if (r["task"], r["language"], r["turn"]) == ("gemm", "numpy", "0")]
        write_rows(self.csv, synthetic_rows() + [dict(r, grammar_hint="on") for r in numpy_first])
        fig = fig_condition_effects(load_runs(self.csv))
        args, hint, _feedback = fig.axes
        self.assertEqual(_tick_labels(args, "y"), ["gemm / numpy", "gemm / c_seq"])
        points = sorted((float(line.get_ydata()[0]), float(line.get_xdata()[0])) for line in args.lines if line.get_gid())
        self.assert_close(points, [(-0.22, 0.4), (0.22, 0.4)])
        points = sorted((float(line.get_ydata()[0]), float(line.get_xdata()[0])) for line in hint.lines if line.get_gid())
        self.assert_close(points, [(-0.22, 0.0), (0.22, 0.0), (1.0, 0.2)])
        # 벌려 그린 점마다 나머지 조건을 적고, 한 행에 하나뿐인 점은 n 만 적는다
        self.assertEqual(sorted(_texts(args)), ["n=5/5; grammar_hint=off", "n=5/5; grammar_hint=on"])
        self.assertEqual(sorted(_texts(hint)), ["n=5/5", "n=5/5; args=keyword", "n=5/5; args=positional"])

    def test_seeded_feedback_copies_are_drawn_once(self) -> None:
        # gemm numpy 의 턴 0 을 compiler+lint 로 복사(--seed-from-feedback)하고 수정하지 않았다: args 비교는 점 하나,
        # feedback 비교는 수정 턴이 없어 그리지 않는다
        numpy_first = [r for r in synthetic_rows() if (r["task"], r["language"], r["turn"]) == ("gemm", "numpy", "0")]
        write_rows(self.csv, synthetic_rows() + [dict(r, feedback="compiler+lint") for r in numpy_first])
        with contextlib.redirect_stderr(io.StringIO()):
            fig = fig_condition_effects(load_runs(self.csv))
        args, _hint, feedback = fig.axes
        self.assertEqual([line.get_gid() for line in args.lines if line.get_gid()], ["effect:args:gemm:numpy"])
        self.assertEqual(_texts(args), ["n=5/5"])
        self.assertEqual(list(_lines(feedback)), ["effect:feedback:gemm:c_seq"])

    def test_no_condition_pair_is_no_data(self) -> None:
        write_rows(self.csv, self.reference_only())
        with self.assertRaises(NoData):
            fig_condition_effects(load_runs(self.csv))
        with self.assertRaises(NoData):
            fig_condition_effects(self.df.iloc[:0])


class FigRepairTurnsTest(SyntheticFigureCase):
    def test_cumulative_pass_rate_lines(self) -> None:
        fig = fig_repair_turns(self.df)
        self.assertEqual([ax.get_title() for ax in fig.axes], ["gemm", "sort"])
        gemm, sort = (_lines(ax) for ax in fig.axes)
        # 기준 args·grammar_hint 의 그룹만: keyword·grammar_hint=on 그룹은 그리지 않고 feedback 두 수준은 모두 그린다
        self.assertEqual(sorted(gemm), [
            "turns:gemm:c_seq:compiler", "turns:gemm:c_seq:compiler+lint", "turns:gemm:halide_auto:compiler",
            "turns:gemm:numpy:compiler", "turns:gemm:rust:compiler",
        ])
        nan = float("nan")
        self.assert_close(gemm["turns:gemm:numpy:compiler"].get_ydata(), [0.6, 0.8, 1.0])
        self.assert_close(gemm["turns:gemm:halide_auto:compiler"].get_ydata(), [1.0, 1.0, 1.0])
        self.assert_close(gemm["turns:gemm:c_seq:compiler"].get_ydata(), [0.4, 0.6, 0.6])
        self.assert_close(gemm["turns:gemm:c_seq:compiler+lint"].get_ydata(), [0.4, 0.8, 0.8])
        self.assert_close(gemm["turns:gemm:rust:compiler"].get_ydata(), [0.75, nan, nan])   # 고쳐 보지 않은 턴은 점이 없다
        self.assert_close(sort["turns:sort:numpy:compiler"].get_ydata(), [0.8, nan, nan])
        self.assert_close(sort["turns:sort:c_seq:compiler"].get_ydata(), [0.2, nan, nan])
        for line in gemm.values():
            self.assert_close(line.get_xdata(), [0, 1, 2])
        for ax in fig.axes:
            self.assertEqual(list(ax.get_xticks()), [0, 1, 2])
            self.assertIn("repair turns", ax.get_xlabel())
        self.assertEqual(fig.get_supylabel(), "cumulative pass rate (fraction of samples)")
        self.assertEqual(fig.axes[0].get_ylim(), (-0.05, 1.08))

    def test_feedback_levels_differ_in_line_style_and_languages_in_colour_and_marker(self) -> None:
        gemm = _lines(fig_repair_turns(self.df).axes[0])
        compiler, lint = gemm["turns:gemm:c_seq:compiler"], gemm["turns:gemm:c_seq:compiler+lint"]
        self.assertEqual((compiler.get_color(), compiler.get_marker()), (lint.get_color(), lint.get_marker()))
        self.assertEqual((compiler.get_linestyle(), lint.get_linestyle()), ("-", "--"))
        per_language = {gid.split(":")[2]: (line.get_color(), line.get_marker()) for gid, line in gemm.items()}
        self.assertEqual(len({color for color, _marker in per_language.values()}), 4)
        self.assertEqual(len({marker for _color, marker in per_language.values()}), 4)

    def test_n_and_legend(self) -> None:
        fig = fig_repair_turns(self.df)
        self.assertEqual(sorted(_texts(fig.axes[0])), ["n=4", "n=5", "n=5", "n=5", "n=5"])
        self.assertEqual(_texts(fig.axes[1]), ["n=5", "n=5"])
        legend = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertEqual(legend, [*LANGUAGES, "feedback=compiler", "feedback=compiler+lint"])

    def assert_same_lines(self, ax: Axes, reference: Axes) -> None:
        """두 패널의 선이 같다: gid 가 겹치지 않고 같은 집합이며 y 값(NaN 포함)이 같다."""
        gids = [line.get_gid() for line in ax.lines if line.get_gid()]
        self.assertEqual(len(gids), len(set(gids)))
        want = _lines(reference)
        self.assertEqual(sorted(gids), sorted(want))
        for gid, line in _lines(ax).items():
            self.assert_close(line.get_ydata(), want[gid].get_ydata())

    def test_runs_without_repair_rows_are_left_out_and_runs_are_pooled(self) -> None:
        reference = fig_repair_turns(self.df).axes[0]
        # 수정 루프를 돌리지 않은 옛 run(gemm numpy 턴 0 만, 모두 통과)은 그리지 않는다
        old = [dict(r, run_id="20261101-000000-old") for r in synthetic_rows()
               if (r["task"], r["language"], r["args"], r["turn"], r["pass"]) == ("gemm", "numpy", "positional", "0", "1")]
        write_rows(self.csv, synthetic_rows() + old)
        fig = fig_repair_turns(load_runs(self.csv))
        self.assert_same_lines(fig.axes[0], reference)
        self.assertEqual(sorted(_texts(fig.axes[0])), ["n=4", "n=5", "n=5", "n=5", "n=5"])
        self.assertNotIn("2 runs", fig.get_suptitle())                      # 제목의 범위도 그린 run 하나
        # 수정 루프가 있는 run 둘은 합쳐 (과제, 언어, feedback)마다 선 하나 — 샘플은 run 마다 따로 센다
        twin = [dict(r, run_id="20261121-000000-twin") for r in synthetic_rows()]
        write_rows(self.csv, synthetic_rows() + twin)
        fig = fig_repair_turns(load_runs(self.csv))
        self.assert_same_lines(fig.axes[0], reference)
        self.assertEqual(sorted(_texts(fig.axes[0])), ["n=10", "n=10", "n=10", "n=10", "n=8"])

    def test_rows_after_the_first_passing_turn_are_not_drawn(self) -> None:
        reference = fig_repair_turns(self.df).axes[0]
        stale = [r for r in synthetic_rows() if (r["task"], r["language"], r["sample_idx"], r["turn"]) == ("gemm", "numpy", "0", "0")]
        write_rows(self.csv, synthetic_rows() + [dict(stale[0], turn="3", error_type="wrong_output", error_stage="check", **{"pass": "0"})])
        with contextlib.redirect_stderr(io.StringIO()) as err:
            fig = fig_repair_turns(load_runs(self.csv))
        self.assert_same_lines(fig.axes[0], reference)          # 낡은 턴 3 행이 x 축을 늘리지 않는다
        self.assertEqual(list(fig.axes[0].get_xticks()), [0, 1, 2])
        self.assertIn("처음 통과한 턴보다 뒤 턴의 행 1개", err.getvalue())

    def test_no_repair_rows_is_no_data(self) -> None:
        write_rows(self.csv, [r for r in synthetic_rows() if r["turn"] == "0"])
        with self.assertRaises(NoData):
            fig_repair_turns(load_runs(self.csv))
        # 수정 루프 행이 기준 args·grammar_hint 밖(keyword)에만 있어도 그릴 것이 없다
        rows = [r for r in synthetic_rows() if r["turn"] == "0"]
        rows.append(dict(rows[0], args="keyword", sample_idx="9", turn="1"))
        write_rows(self.csv, rows)
        with self.assertRaises(NoData):
            fig_repair_turns(load_runs(self.csv))
        with self.assertRaises(NoData):
            fig_repair_turns(self.df.iloc[:0])


class CliTest(SyntheticFigureCase):
    def run_main(self, *args: str) -> tuple[int, list[Path], str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), mock.patch.object(aggregate, "TASKS_DIR", self.tasks_dir):
            rc = main(list(args))
        return rc, [Path(line) for line in out.getvalue().splitlines()], err.getvalue()

    def test_writes_four_figures_and_prints_their_paths(self) -> None:
        out_dir = self.tmp / "plots"
        rc, paths, err = self.run_main("--csv", str(self.csv), "--out", str(out_dir))
        self.assertEqual(rc, EXIT_OK)
        self.assertEqual([p.name for p in paths], [
            "fig1_languages.png", "fig2_error_types.png", "fig3_condition_effects.png", "fig4_repair_turns.png",
        ])
        self.assertEqual(sorted(out_dir.iterdir()), sorted(paths))   # 임시 파일이 남지 않는다
        for path in paths:
            self.assertTrue(path.is_absolute())
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(path.stat().st_size, 10_000)
        self.assertIn("조건 조합이 4개다", err)
        self.assertIn("args=positional, grammar_hint=off, feedback=compiler", err)
        self.assertNotIn("건너뜀", err)

    def test_by_task_and_format(self) -> None:
        write_rows(self.csv, self.reference_only())   # 그림 1·2 만 그려진다(형식마다 네 장을 다 그리면 느리다)
        out_dir = self.tmp / "plots"
        rc, paths, _ = self.run_main("--csv", str(self.csv), "--out", str(out_dir), "--by-task", "--format", "pdf")
        self.assertEqual(rc, EXIT_OK)
        self.assertEqual([p.name for p in paths], ["fig1_languages.pdf", "fig2_error_types_by_task.pdf"])
        for path in paths:
            self.assertEqual(path.read_bytes()[:5], b"%PDF-")
        rc, paths, _ = self.run_main("--csv", str(self.csv), "--out", str(out_dir), "--format", "svg")
        self.assertEqual(rc, EXIT_OK)
        self.assertEqual([p.name for p in paths], ["fig1_languages.svg", "fig2_error_types.svg"])
        self.assertIn('id="samples:gemm:numpy"', paths[0].read_text(encoding="utf-8"))   # gid 가 SVG 요소 id 로 남는다

    def test_figures_without_data_are_skipped_with_one_line_each(self) -> None:
        write_rows(self.csv, self.reference_only())
        rc, paths, err = self.run_main("--csv", str(self.csv), "--out", str(self.tmp / "plots"))
        self.assertEqual(rc, EXIT_OK)
        self.assertEqual([p.name for p in paths], ["fig1_languages.png", "fig2_error_types.png"])
        # 합성 행은 이전 규약(speedup_mean 빈칸)이라 설정의 평균 비 대신 중앙값 비로 그렸다는 경고가 한 줄 먼저 나온다
        warning, *skipped = err.splitlines()
        self.assertIn("speedup_vs_baseline", warning)
        self.assertIn("중앙값 비로 대신한다", warning)
        self.assertEqual([line.split(":")[0] for line in skipped], ["[plots] fig3_condition_effects", "[plots] fig4_repair_turns"])
        self.assertTrue(all("건너뜀" in line for line in skipped))

    def test_empty_csv_writes_nothing(self) -> None:
        write_rows(self.csv, [])
        out_dir = self.tmp / "plots"
        rc, paths, err = self.run_main("--csv", str(self.csv), "--out", str(out_dir))
        self.assertEqual(rc, EXIT_OK)
        self.assertEqual(paths, [])
        self.assertEqual(sum("건너뜀" in line for line in err.splitlines()), 4)
        self.assertFalse(out_dir.exists())

    def test_missing_csv_and_unknown_run(self) -> None:
        rc, paths, err = self.run_main("--csv", str(self.tmp / "absent.csv"), "--out", str(self.tmp / "plots"))
        self.assertEqual((rc, paths), (EXIT_NOT_FOUND, []))
        self.assertIn("CSV가 없다", err)
        rc, paths, err = self.run_main("--csv", str(self.csv), "--run-id", "nope", "--out", str(self.tmp / "plots"))
        self.assertEqual((rc, paths), (EXIT_NOT_FOUND, []))
        self.assertIn("nope", err)
        bad = self.tmp / "bad.csv"
        bad.write_text("run_id,foo\n1,2\n", encoding="utf-8")
        rc, paths, _ = self.run_main("--csv", str(bad), "--out", str(self.tmp / "plots"))
        self.assertEqual((rc, paths), (2, []))

    def test_several_models_are_reported(self) -> None:
        rows = [r for r in self.reference_only() if r["task"] == "sort"]
        rows[0]["model"] = "other-model"
        write_rows(self.csv, rows)
        rc, paths, err = self.run_main("--csv", str(self.csv), "--out", str(self.tmp / "plots"))
        self.assertEqual((rc, len(paths)), (EXIT_OK, 2))
        self.assertIn("모델이 여럿이다 ['other-model', 'synth-model']", err)
        self.assertNotIn("조건 조합", err)   # 조건 조합이 하나면 기준 조합 안내는 없다
        self.assertNotIn("run 이", err)

    def test_several_runs_are_reported(self) -> None:
        rows = self.reference_only()
        write_rows(self.csv, rows + [dict(r, run_id="20261101-000000-old") for r in rows])
        rc, paths, err = self.run_main("--csv", str(self.csv), "--out", str(self.tmp / "plots"), "--all-runs")
        self.assertEqual((rc, len(paths)), (EXIT_OK, 2))
        self.assertIn("선택한 run 이 2개다", err)
        self.assertIn("그림 4 는 수정 루프 행이 있는 run 만", err)
        rc, paths, err = self.run_main("--csv", str(self.csv), "--out", str(self.tmp / "plots"))   # 기본 --latest
        self.assertNotIn("run 이", err)

    def test_streams_are_set_up_in_the_entry_point_only(self) -> None:
        """``python -m analysis.plots`` 는 ``setup_stdio`` 를 부르고, 테스트가 부르는 ``main()`` 은 부르지 않는다."""
        absent = str(self.tmp / "absent.csv")
        with mock.patch.object(procutil, "setup_stdio") as setup:
            rc, _, _ = self.run_main("--csv", absent)
            self.assertEqual(rc, EXIT_NOT_FOUND)
            setup.assert_not_called()
            with mock.patch.object(sys, "argv", ["plots", "--csv", absent]), warnings.catch_warnings(), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
                warnings.simplefilter("ignore", RuntimeWarning)   # 이미 import 한 모듈을 __main__ 으로 다시 실행한다는 경고
                runpy.run_module("analysis.plots", run_name="__main__")
        setup.assert_called_once_with()
        self.assertEqual(stop.exception.code, EXIT_NOT_FOUND)


class RealCsvSmokeTest(unittest.TestCase):
    """저장소의 실제 결과 파일(첫 실제 실행: GEMM × 3언어 × 3샘플, 9/9 통과)이 그려진다."""

    def test_first_real_run_figures(self) -> None:
        df, _ = select_runs(load_runs(REAL_CSV), run_id=REAL_RUN)
        fig = fig_languages(df)
        pass_ax, speed_ax = fig.axes
        self.assertEqual(pass_ax.get_title(), "gemm")
        self.assertEqual(_tick_labels(speed_ax), ["numpy", "c_openmp", "c_seq"])
        for language in ("numpy", "c_openmp", "c_seq"):
            self.assertEqual(_intervals(pass_ax)[f"pass:gemm:{language}"], (1.0, 1.0, 1.0))
            self.assertEqual(len(_lines(speed_ax)[f"samples:gemm:{language}"].get_ydata()), 3)
        self.assertEqual(_texts(pass_ax), ["n=3", "n=3", "n=3"])
        heights = _bar_heights(fig_error_types(df).axes[0])
        self.assertEqual(heights, {"pass": [3.0, 3.0, 3.0]})
        with self.assertRaises(NoData):
            fig_condition_effects(df)
        with self.assertRaises(NoData):
            fig_repair_turns(df)

    def test_cli_on_the_repository_csvs(self) -> None:
        for csv_path in (REAL_CSV, DUMMY_CSV):
            with self.subTest(csv=csv_path.name), tempfile.TemporaryDirectory(prefix="plots_real_") as tmp:
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    rc = main(["--csv", str(csv_path), "--all-runs", "--out", tmp, "--by-task"])
                self.assertEqual(rc, EXIT_OK)
                written = [Path(line) for line in out.getvalue().splitlines()]
                self.assertEqual([p.name for p in written[:2]], ["fig1_languages.png", "fig2_error_types_by_task.png"])
                self.assertTrue(all(p.is_file() for p in written))


if __name__ == "__main__":
    unittest.main()
