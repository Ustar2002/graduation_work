"""결과 그래프 (설계 v3 §13.3; CLAUDE.md §3 M6).

``analysis.aggregate`` 가 읽고 고른 행으로 그림 네 장을 만든다. 그림마다 함수 하나가 ``matplotlib.figure.Figure`` 를
돌려주고(pyplot 전역 상태를 쓰지 않는다; 캔버스는 Agg), CLI 가 파일로 쓴다.

CLI:
    python -m analysis.plots [--csv results/runs.csv] [--run-id ID | --latest | --all-runs]
                             [--out results/plots] [--format png|pdf|svg] [--by-task]
                             [--primary-speedup mean|median] [--config config/experiment.yaml]

    쓴 파일의 경로를 stdout 에 한 줄씩 출력한다. 자료가 없는 그림은 건너뛰고 stderr 에 한 줄로 알린다.
    종료 코드는 ``analysis.aggregate`` 와 같다(0 정상, 1 CSV·run 없음, 2 열 불일치).

그림(파일 이름):
    fig1_languages          과제마다 패널 한 쌍. x = 언어(성능 결정 부담 순). 위: pass@1 과 95 % 부트스트랩 구간,
                            아래: 통과 샘플의 기준선 대비 속도비(로그 축) — 샘플 하나가 점 하나, 중앙값은 검은 가로선.
                            속도비 열은 ``analysis.aggregate.choose_speedup`` 의 주 지표(설정 mean → speedup_mean, 이전 규약
                            행이 섞이면 speedup_vs_baseline)이고 축 이름과 제목에 적는다.
    fig2_error_types        언어별 누적 막대: 통과 / 오류 분류 / 생성 단계 실패(막대 높이 = 행 수).
                            ``--by-task`` 면 과제마다 패널을 나누고 파일 이름은 fig2_error_types_by_task.
    fig3_condition_effects  조건(args, grammar_hint, feedback) 수준 쌍의 통과율 차와 구간. 행 = (과제, 언어).
    fig4_repair_turns       과제마다 패널. 수정 턴을 t 번까지 허용했을 때의 누적 통과율, 언어마다 선 하나.

그림 1·2 는 첫 시도(turn 0)만, 그리고 조건 조합이 여럿이면 기준 조합(``reference_condition``; 보통 args=positional,
grammar_hint=off, feedback=compiler) 하나만 그린다 — 조건이 다른 샘플을 한 통과율로 섞지 않는다. 그림 4 는 기준 조합의
args·grammar_hint 에서 feedback 수준마다 선을 따로 긋는다. 조건 사이의 차이는 그림 3 이 보여 준다. 선택한 run 이 여럿이면
그림 1·2 는 run 을 합쳐 세고, 그림 4 는 수정 루프 행이 있는 run 만 합쳐 그린다(CLI 가 stderr 에 한 줄로 알린다).

색은 Okabe–Ito 팔레트이고 계열마다 표식(막대는 빗금, 선은 선 모양)도 다르다. 그림 안의 글자는 영어다
(matplotlib 기본 글꼴에 한글 글리프가 없다).
"""

from __future__ import annotations

import argparse
import functools
import io
import math
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter

from analysis.aggregate import (
    CONDITION_COLUMNS,
    DEFAULT_CSV,
    EXIT_OK,
    LANGUAGE_ORDER,
    OUTCOME_ORDER,
    PRIMARY_SPEEDUP_CHOICES,
    AggregateError,
    SpeedupChoice,
    by_turn_table,
    choose_speedup,
    condition_effects,
    first_attempt_rows,
    language_sort_key,
    level_sort_key,
    load_runs,
    load_unsupported,
    outcome_labels,
    pass_mask,
    select_runs,
    task_sort_key,
    turn_numbers,
    valid_mask,
)
from analysis.stats import bootstrap_pass_rate
from harness.paths import DEFAULT_EXPERIMENT_YAML, ROOT, atomic_write_bytes, resolve

_LOG = "[plots]"

DEFAULT_OUT: Path = ROOT / "results" / "plots"
FORMATS: tuple[str, ...] = ("png", "pdf", "svg")
DPI = 150
MAX_PANEL_COLUMNS = 4

# Okabe–Ito 색각 이상 안전 팔레트 (+ 무채색)
BLACK, ORANGE, SKY_BLUE, GREEN = "#000000", "#E69F00", "#56B4E9", "#009E73"
YELLOW, BLUE, VERMILION, PURPLE = "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
GREY, WHITE = "#999999", "#FFFFFF"

# 언어 계열의 색·표식(노랑은 흰 바탕에서 흐려 선·점에는 쓰지 않는다). 표식은 ``LANGUAGE_ORDER`` 의 위치로 정해 12개가 모두 다르다.
# 색은 이 저장소에서 돌리는 언어 7개(numpy, numba, halide_auto, halide_manual, rust, c_openmp, c_seq)가 모두 다른 Okabe–Ito 색을
# 받고(Okabe–Ito 의 노랑 외 색이 7개라 여기서 끝난다), mojo(아직 하네스에 없음)는 회색이다. 확장 언어와 목록에 없는 언어는
# ``SERIES_COLORS`` 를 위치 순으로 돌려 쓴다(표식이 다르다).
SERIES_COLORS: tuple[str, ...] = (BLUE, ORANGE, GREEN, PURPLE, VERMILION, SKY_BLUE, BLACK)
LANGUAGE_COLORS: dict[str, str] = {
    "numpy": BLUE, "numba": ORANGE, "halide_auto": GREEN, "halide_manual": PURPLE, "mojo": GREY,
    "rust": BLACK, "c_openmp": VERMILION, "c_seq": SKY_BLUE,
}
SERIES_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "d", "X", "v", "P", "<", ">", "*", "h")

# 결과 범주(``aggregate.OUTCOME_ORDER``) → (색, 빗금, 범례 글자). 생성 단계 실패는 모델의 답에 대한 판정이 아니므로 흰 바탕이다.
OUTCOME_STYLE: dict[str, tuple[str, str, str]] = {
    "pass": (GREEN, "", "pass"),
    "pass_bench_timeout": (GREEN, "//", "pass (bench timeout)"),
    "compile_error": (ORANGE, "", "compile_error"),
    "runtime_crash": (VERMILION, "", "runtime_crash"),
    "timeout": (YELLOW, "", "timeout"),
    "wrong_output": (BLUE, "", "wrong_output"),
    "numeric_instability": (SKY_BLUE, "", "numeric_instability"),
    "arg_order": (BLUE, "xx", "arg_order"),
    "memory_error": (VERMILION, "..", "memory_error"),
    "race_condition": (PURPLE, "", "race_condition"),
    "signature_mismatch": (ORANGE, "\\\\", "signature_mismatch"),
    "refusal": (WHITE, "", "generate: refusal"),
    "truncated": (WHITE, "//", "generate: truncated"),
    "generate_failed": (WHITE, "xx", "generate: API error"),
    "other": (BLACK, "", "other"),
}

# 조건 → (색, 표식), feedback 수준 순서대로의 선 모양
FACTOR_STYLE: dict[str, tuple[str, str]] = {"args": (BLUE, "o"), "grammar_hint": (ORANGE, "s"), "feedback": (GREEN, "^")}
LINE_STYLES: tuple[str, ...] = ("-", "--", ":", "-.")

LANGUAGE_AXIS_LABEL = "language (burden: low → high)"
LANGUAGE_AXIS_NOTE = "x: languages ordered by performance-decision burden"


class NoData(Exception):
    """그림을 그릴 자료가 없다. 메시지는 건너뛴 이유로 그대로 출력한다."""


# ----------------------------------------------------------------------------- 공통 도우미


def language_styles(languages: Sequence[str]) -> dict[str, dict[str, str]]:
    """언어 → ``{"color", "marker"}``. ``LANGUAGE_ORDER`` 의 언어는 어느 그림에서나 같은 색·표식을 받고, 그 밖의 언어는 이름순으로 뒤를 잇는다."""
    unknown = sorted(language for language in languages if language not in LANGUAGE_ORDER)
    styles: dict[str, dict[str, str]] = {}
    for language in languages:
        known = language in LANGUAGE_ORDER
        i = LANGUAGE_ORDER.index(language) if known else len(LANGUAGE_ORDER) + unknown.index(language)
        color = LANGUAGE_COLORS.get(language, SERIES_COLORS[i % len(SERIES_COLORS)])
        styles[language] = {"color": color, "marker": SERIES_MARKERS[i % len(SERIES_MARKERS)]}
    return styles


def reference_condition(df: pd.DataFrame) -> dict[str, str]:
    """행에 있는 조건 조합 가운데 기준 조합: 수준 순서(``aggregate.CONDITION_LEVELS``)가 가장 앞선 것."""
    combos = df[list(CONDITION_COLUMNS)].drop_duplicates().to_dict(orient="records")
    return min(combos, key=lambda combo: tuple(level_sort_key(factor, combo[factor]) for factor in CONDITION_COLUMNS))


def _condition_text(condition: dict[str, str], factors: Sequence[str] = CONDITION_COLUMNS) -> str:
    return ", ".join(f"{factor}={condition[factor]}" for factor in factors)


def _matching(df: pd.DataFrame, condition: dict[str, str], factors: Sequence[str] = CONDITION_COLUMNS) -> pd.DataFrame:
    """``factors`` 의 값이 ``condition`` 과 같은 행."""
    mask = pd.Series(True, index=df.index)
    for factor in factors:
        mask &= df[factor].eq(condition[factor])
    return df[mask]


def _first_attempts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """기준 조건 조합의 첫 시도(turn 0) 행과 그 조합. 행이 없으면 ``NoData``."""
    if df.empty:
        raise NoData("행이 없다")
    condition = reference_condition(df)
    rows = _matching(first_attempt_rows(df), condition)
    if rows.empty:
        raise NoData(f"기준 조건({_condition_text(condition)})의 첫 시도(turn 0) 행이 없다")
    return rows, condition


def _scope(df: pd.DataFrame) -> str:
    """그림 제목에 적는 자료 범위: run 과 모델."""
    runs = sorted(set(df["run_id"]))
    run_text = f"run {runs[0]}" if len(runs) == 1 else f"{len(runs)} runs"
    return f"{run_text}, model {'|'.join(sorted(set(df['model'])))}"


def _new_figure(width: float, height: float) -> Figure:
    """Agg 캔버스에 붙인 빈 그림(pyplot 을 거치지 않으므로 닫을 것도 없다)."""
    fig = Figure(figsize=(width, height), dpi=DPI, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _title(fig: Figure, *lines: str) -> None:
    """그림 제목. 줄마다 그림 너비에 맞춰 접는다(9 pt 글자를 인치당 13자로 어림; 실측은 약 15자)."""
    width = int(fig.get_figwidth() * 13)
    fig.suptitle("\n".join(textwrap.fill(line, width) for line in lines), fontsize=9)


def _legend_below(fig: Figure, handles: Sequence[Artist], title: str | None = None) -> None:
    """범례를 그림 아래 바깥에 둔다(제목과 겹치지 않는다). 열 수는 그림 너비에 맞춘다."""
    ncols = max(1, min(len(handles), int(fig.get_figwidth() / 2.0)))
    fig.legend(handles=handles, loc="outside lower center", ncols=ncols, fontsize=8, title=title, title_fontsize=8)


def _label_bottom_row(panels: Sequence[Axes], ncols: int, label: str) -> None:
    """패널(행 우선 순서)의 마지막 줄에만 x 축 이름을 단다. 그림 전체의 x 이름은 아래 범례와 겹치므로 축에 붙인다."""
    for ax in panels[-(len(panels) % ncols or ncols):]:
        ax.set_xlabel(label, fontsize=9)


def _panel_grid(n_panels: int) -> tuple[int, int]:
    """패널 ``n_panels`` 개를 놓을 (행 수, 열 수). 한 줄에 ``MAX_PANEL_COLUMNS`` 개까지."""
    ncols = min(n_panels, MAX_PANEL_COLUMNS)
    return math.ceil(n_panels / ncols), ncols


def _language_ticks(ax: Axes, languages: Sequence[str]) -> None:
    ax.set_xticks(range(len(languages)), languages, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(languages) - 0.4)


def _y_grid(ax: Axes) -> None:
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def _interval_point(ax: Axes, x: float, y: float, gid: str, **style: Any) -> None:
    """구간(``xerr`` 또는 ``yerr``)이 달린 점 하나. gid 는 점에만 단다(``errorbar`` 에 넘기면 구간 끝 선에도 같은 id 가 붙는다)."""
    container = ax.errorbar([x], [y], linestyle="none", capsize=3, markersize=7, markeredgecolor=BLACK, **style)
    container.lines[0].set_gid(gid)


# ----------------------------------------------------------------------------- 그림 1: 언어 축


def fig_languages(df: pd.DataFrame, tasks_dir: Path | None = None, speedup: SpeedupChoice | None = None) -> Figure:
    """그림 1 — 과제마다 패널 한 쌍, x = 언어(성능 결정 부담 순), 과제는 ``aggregate.TASK_ORDER`` 순.

    위 패널: 첫 시도 통과율(pass@1)과 95 % 백분위 부트스트랩 구간, 언어마다 ``n``(유효 샘플 수). 과제 명세가 돌리지
    않는다고 적은 언어 자리(Halide 의 sort·bfs 포함, ``aggregate.load_unsupported``)에는 ``n/a`` 를 적는다. 아래 패널: 통과
    샘플의 속도비(주 지표 열 ``speedup.column``, 없으면 ``aggregate.choose_speedup(df)``; 로그 축)를 샘플마다 점 하나로 전부
    그리고(측정 규약: 최선값이 아니라 분포를 보고한다) 중앙값을 검은 가로선으로, 기준선(1배)을 점선으로 표시한다.
    ``n`` 은 그린 점의 수다(통과했어도 bench 시간이 없는 샘플은 점이 없다).
    """
    rows, condition = _first_attempts(df)
    valid = rows[valid_mask(rows)]
    if valid.empty:
        raise NoData("첫 시도(turn 0)의 유효 샘플이 없다")
    if speedup is None:
        speedup = choose_speedup(df)
    column = speedup.column
    tasks = sorted(set(rows["task"]), key=task_sort_key)
    languages = sorted(set(rows["language"]), key=language_sort_key)
    styles = language_styles(languages)
    unsupported = load_unsupported(tasks, tasks_dir)
    speedups = valid.loc[pass_mask(valid), column].dropna()
    speedup_limits = (min(speedups.min(), 1.0) / 1.6, max(speedups.max(), 1.0) * 1.6) if len(speedups) else (0.1, 10.0)

    nrows, ncols = _panel_grid(len(tasks))
    fig = _new_figure(max(6.4, 1.2 + ncols * max(0.6 * len(languages) + 0.6, 2.4)), 1.6 + 4.6 * nrows)
    grid = fig.add_gridspec(2 * nrows, ncols)
    pass_axes: list[Axes] = []
    speed_axes: list[Axes] = []
    for i, task in enumerate(tasks):
        r, c = divmod(i, ncols)
        ax_pass = fig.add_subplot(grid[2 * r, c], sharey=pass_axes[0] if pass_axes else None)
        ax_speed = fig.add_subplot(grid[2 * r + 1, c], sharex=ax_pass, sharey=speed_axes[0] if speed_axes else None)
        pass_axes.append(ax_pass)
        speed_axes.append(ax_speed)
        of_task = rows[rows["task"].eq(task)]
        for x, language in enumerate(languages):
            cell = of_task[of_task["language"].eq(language)]
            if cell.empty:
                if language in unsupported[task]:
                    ax_pass.text(x, 0.5, "n/a", ha="center", va="center", fontsize=8, color=GREY, style="italic")
                continue
            cell = cell[valid_mask(cell)]
            passed = pass_mask(cell).to_numpy()
            ax_pass.text(x, -0.12, f"n={len(cell)}", ha="center", va="center", fontsize=7)
            if len(cell):
                rate, low, high = bootstrap_pass_rate(passed)
                _interval_point(
                    ax_pass, x, rate, f"pass:{task}:{language}", yerr=[[rate - low], [high - rate]],
                    ecolor=BLACK, elinewidth=1, **styles[language],
                )
            timed = cell.loc[passed, column].dropna().to_numpy()
            ax_speed.text(x, 0.02, f"n={timed.size}", transform=ax_speed.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
            if timed.size:
                spread = np.linspace(-0.18, 0.18, timed.size) if timed.size > 1 else np.zeros(1)
                ax_speed.plot(
                    x + spread, timed, linestyle="none", markersize=6, markeredgecolor=BLACK, alpha=0.85,
                    gid=f"samples:{task}:{language}", **styles[language],
                )
                median = float(np.median(timed))
                ax_speed.plot([x - 0.3, x + 0.3], [median, median], color=BLACK, linewidth=2, gid=f"median:{task}:{language}")
        ax_pass.set_title(task)
        ax_pass.set_ylim(-0.2, 1.08)
        ax_pass.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_pass.tick_params(labelbottom=False, labelleft=c == 0)
        ax_speed.set_yscale("log")
        ax_speed.set_ylim(*speedup_limits)
        ax_speed.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax_speed.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
        ax_speed.yaxis.set_minor_formatter(NullFormatter())
        ax_speed.axhline(1.0, color=GREY, linestyle="--", linewidth=0.8)
        ax_speed.tick_params(labelleft=c == 0)
        _language_ticks(ax_speed, languages)
        _y_grid(ax_pass)
        _y_grid(ax_speed)
        if c == 0:
            ax_pass.set_ylabel("pass@1\n(fraction of samples)", fontsize=9)
            ax_speed.set_ylabel(f"speedup vs baseline\n({speedup.label}, ×, log scale)", fontsize=9)
    _label_bottom_row(speed_axes, ncols, LANGUAGE_AXIS_LABEL)
    _legend_below(fig, [
        Line2D([], [], color=BLACK, marker="|", linestyle="none", markersize=12, label="95 % bootstrap interval of pass@1"),
        Line2D([], [], color=GREY, marker="o", linestyle="none", markeredgecolor=BLACK, label="one passing sample"),
        Line2D([], [], color=BLACK, linewidth=2, label="median of passing samples"),
        Line2D([], [], color=GREY, linestyle="--", linewidth=0.8, label="baseline (speedup 1)"),
    ])
    _title(
        fig, f"First attempt by language ({LANGUAGE_AXIS_NOTE})",
        f"{_scope(rows)}; {_condition_text(condition)}; speedup = {column} ({speedup.label})",
    )
    return fig


# ----------------------------------------------------------------------------- 그림 2: 오류 분류


def fig_error_types(df: pd.DataFrame, by_task: bool = False) -> Figure:
    """그림 2 — 언어별 누적 막대: 첫 시도 행을 결과 범주(``aggregate.outcome_labels``)로 나눈 수.

    통과(bench timeout 은 빗금으로 구분), 오류 분류표의 유형, 생성 단계 실패(거절·절단·API 오류; 흰 바탕)가 각자 한 칸이다.
    막대 높이의 합은 그 언어의 행 수 ``n`` 이고 막대 위에 적는다. ``by_task`` 면 과제마다 패널을 나눈다(기본은 과제를 합친다).
    """
    rows, condition = _first_attempts(df)
    labels = outcome_labels(rows)
    languages = sorted(set(rows["language"]), key=language_sort_key)
    present = [outcome for outcome in OUTCOME_ORDER if labels.eq(outcome).any()]
    panels = [(task, rows["task"].eq(task)) for task in sorted(set(rows["task"]), key=task_sort_key)] if by_task else [("all tasks", rows["task"].notna())]

    nrows, ncols = _panel_grid(len(panels))
    fig = _new_figure(max(6.4, 1.0 + ncols * max(0.6 * len(languages) + 0.6, 2.4)), 2.4 + 3.0 * nrows)
    axes = fig.subplots(nrows, ncols, sharey=True, squeeze=False)
    x = np.arange(len(languages))
    tallest = 0.0
    for ax, (title, mask) in zip(axes.flat, panels):
        counts = pd.crosstab(rows.loc[mask, "language"], labels[mask]).reindex(index=languages, columns=present, fill_value=0)
        bottom = np.zeros(len(languages))
        for outcome in present:
            color, hatch, label = OUTCOME_STYLE[outcome]
            heights = counts[outcome].to_numpy(dtype=float)
            ax.bar(x, heights, bottom=bottom, color=color, hatch=hatch, edgecolor=BLACK, linewidth=0.6, label=label)
            bottom += heights
        for xi, total in zip(x, bottom):
            if total:
                ax.text(xi, total, f"n={int(total)}", ha="center", va="bottom", fontsize=8)
        tallest = max(tallest, float(bottom.max()))
        ax.set_title(title)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        _language_ticks(ax, languages)
        _y_grid(ax)
    for ax in axes.flat[len(panels):]:
        ax.remove()
    # 막대 위에 n 을 적을 자리. 높이 0 인 칸도 sticky edge 를 남겨 margins() 가 듣지 않으므로 범위를 직접 준다.
    axes.flat[0].set_ylim(0.0, tallest * 1.15)
    _label_bottom_row(list(axes.flat[:len(panels)]), ncols, LANGUAGE_AXIS_LABEL)
    _legend_below(fig, axes.flat[0].get_legend_handles_labels()[0], title="outcome (stacked bottom to top)")
    fig.supylabel("samples (count)", fontsize=9)
    _title(fig, f"First-attempt outcomes by language ({LANGUAGE_AXIS_NOTE})", f"{_scope(rows)}; {_condition_text(condition)}")
    return fig


# ----------------------------------------------------------------------------- 그림 3: 조건 효과


def fig_condition_effects(df: pd.DataFrame) -> Figure:
    """그림 3 — 조건 수준 쌍의 통과율 차(``alt − base``)와 95 % 부트스트랩 구간(``aggregate.condition_effects``).

    비교(조건, base, alt)마다 패널 하나, 행 = (과제, 언어). args·grammar_hint 는 첫 시도 통과율의 차(독립 표본),
    feedback 은 있는 수정 턴 안에서 통과한 비율의 차(sample_idx 대응)다. 점마다 표본 크기를 적는다. 한 행에 비교가
    여럿이면(나머지 조건이 다른 경우) 위아래로 벌려 그리고 점마다 나머지 조건(``at``)을 함께 적는다. feedback 의 x 는
    모든 짝의 결과를 아는 수정 턴 수 H 까지의 통과율이고, 비교할 수정 턴이 없는 비교(차가 NA)는 그리지 않는다.
    x 축은 모든 패널에서 −1..1 로 고정한다.
    """
    effects = condition_effects(df)
    effects = effects[effects["diff"].notna()]
    if effects.empty:
        raise NoData(
            "비교할 조건 수준 쌍이 없다 (수준이 둘 이상인 조건이 없거나, feedback 수준 사이에 짝지을 샘플이나 "
            "두 수준 모두에서 결과를 아는 수정 턴이 없다)"
        )
    comparisons = sorted(
        set(zip(effects["factor"], effects["base"], effects["alt"])),
        key=lambda c: (CONDITION_COLUMNS.index(c[0]), level_sort_key(c[0], c[1]), level_sort_key(c[0], c[2])),
    )
    cells = sorted(set(zip(effects["task"], effects["language"])), key=lambda cell: (task_sort_key(cell[0]), language_sort_key(cell[1])))
    y_of = {cell: i for i, cell in enumerate(cells)}

    fig = _new_figure(max(6.4, 2.2 + 3.0 * len(comparisons)), 1.8 + 0.5 * len(cells))
    axes = fig.subplots(1, len(comparisons), sharey=True, squeeze=False)[0]
    for ax, (factor, base, alt) in zip(axes, comparisons):
        color, marker = FACTOR_STYLE[factor]
        shown = effects[effects["factor"].eq(factor) & effects["base"].eq(base) & effects["alt"].eq(alt)]
        for (task, language), in_cell in shown.groupby(["task", "language"], sort=False):
            spread = np.linspace(-0.22, 0.22, len(in_cell)) if len(in_cell) > 1 else np.zeros(1)
            for offset, effect in zip(spread, in_cell.itertuples(index=False)):
                y = y_of[(task, language)] + offset
                _interval_point(
                    ax, effect.diff, y, f"effect:{factor}:{task}:{language}",
                    xerr=[[effect.diff - effect.ci_low], [effect.ci_high - effect.diff]], color=color, marker=marker,
                )
                if effect.method == "paired":   # quantity = pass_within_<H>
                    sizes = f"n={effect.n_base} pairs, turns≤{effect.quantity.removeprefix('pass_within_')}"
                else:
                    sizes = f"n={effect.n_base}/{effect.n_alt}"
                if len(in_cell) > 1:   # 벌려 그린 점마다 어느 비교(나머지 조건)인지 적는다
                    sizes = f"{sizes}; {effect.at}"
                ax.annotate(sizes, (effect.diff, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7)
        ax.axvline(0.0, color=GREY, linestyle="--", linewidth=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(f"{factor}: {alt} − {base}", fontsize=9)
        quantity = "fraction passed within the repair turns" if factor == "feedback" else "pass@1"
        ax.set_xlabel(f"difference in {quantity}\n(fraction of samples)", fontsize=8)
        ax.grid(axis="x", color="0.9", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(cells)), [f"{task} / {language}" for task, language in cells])
    axes[0].set_ylim(len(cells) - 0.5, -0.5)
    axes[0].set_ylabel("task / language")
    _title(fig, "Condition effects with 95 % bootstrap intervals", _scope(df))
    return fig


# ----------------------------------------------------------------------------- 그림 4: 수정 턴


def fig_repair_turns(df: pd.DataFrame) -> Figure:
    """그림 4 — 과제마다 패널, x = 허용한 수정 턴 수 t, y = 턴 t 이하 누적 통과율(``aggregate.by_turn_table``).

    언어마다 선 하나(색·표식), feedback 수준이 여럿이면 선 모양으로 구분한다. 턴 t 까지 시도하지 않은 실패 샘플이 있는
    그룹은 그 t 부터 점이 없다(시도하지 않은 턴을 실패로 그리지 않는다). 선의 마지막 점 옆에 ``n``(턴 0 유효 샘플 수)을 적는다.

    run 이 여럿이면 수정 루프 행(turn ≥ 1)이 있는 run 만 쓰고, 그림 1·2 처럼 run 을 합쳐 (과제, 언어, feedback)마다 선
    하나를 긋는다(샘플은 run 마다 따로 센다). 수정 루프를 돌리지 않은 run 의 선을 섞지 않는다.
    """
    if df.empty or not turn_numbers(df).ge(1).any():
        raise NoData("수정 루프 행(turn ≥ 1)이 없다")
    condition = reference_condition(df)
    repaired_runs = set(df.loc[turn_numbers(df).ge(1), "run_id"])
    rows = _matching(df[df["run_id"].isin(repaired_runs)], condition, ("args", "grammar_hint"))
    # run 을 합친다: run 이 다르면 같은 sample_idx 라도 다른 샘플이므로 sample_idx 에 run_id 를 붙인다
    pooled = rows.assign(sample_idx=rows["run_id"] + "/" + rows["sample_idx"], run_id="pooled") if len(repaired_runs) > 1 else rows
    table = by_turn_table(pooled)
    rate_columns = [column for column in table.columns if column.startswith("pass_within_")]
    if len(rate_columns) < 2:
        raise NoData(f"기준 조건({_condition_text(condition, ('args', 'grammar_hint'))})에 수정 루프 행(turn ≥ 1)이 없다")
    turns = np.arange(len(rate_columns))
    tasks = sorted(set(table["task"]), key=task_sort_key)
    languages = sorted(set(table["language"]), key=language_sort_key)
    styles = language_styles(languages)
    feedback_levels = sorted(set(table["feedback"]), key=lambda level: level_sort_key("feedback", level))
    line_of = {level: LINE_STYLES[i % len(LINE_STYLES)] for i, level in enumerate(feedback_levels)}

    nrows, ncols = _panel_grid(len(tasks))
    fig = _new_figure(max(6.4, 1.2 + ncols * max(0.9 * len(turns) + 1.2, 3.0)), 1.9 + 3.0 * nrows)
    axes = fig.subplots(nrows, ncols, sharey=True, squeeze=False)
    for ax, task in zip(axes.flat, tasks):
        for group in table[table["task"].eq(task)].to_dict(orient="records"):
            language, feedback = group["language"], group["feedback"]
            rates = np.array([group[column] for column in rate_columns], dtype=float)
            ax.plot(
                turns, rates, linestyle=line_of[feedback], linewidth=1.5, markersize=7, markeredgecolor=BLACK,
                gid=f"turns:{task}:{language}:{feedback}", **styles[language],
            )
            known = np.flatnonzero(~np.isnan(rates))
            if known.size:
                ax.annotate(f"n={group['n']}", (turns[known[-1]], rates[known[-1]]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7)
        ax.set_title(task)
        ax.set_xticks(turns)
        ax.set_xlim(-0.3, turns[-1] + 0.6)
        ax.set_ylim(-0.05, 1.08)
        _y_grid(ax)
    for ax in axes.flat[len(tasks):]:
        ax.remove()
    handles = [Line2D([], [], linestyle="-", markeredgecolor=BLACK, label=language, **styles[language]) for language in languages]
    if len(feedback_levels) > 1:
        handles += [Line2D([], [], color=BLACK, linestyle=line_of[level], label=f"feedback={level}") for level in feedback_levels]
    _label_bottom_row(list(axes.flat[:len(tasks)]), ncols, "repair turns allowed (count)")
    _legend_below(fig, handles)
    fig.supylabel("cumulative pass rate (fraction of samples)", fontsize=9)
    _title(fig, "Cumulative pass rate by repair turn", f"{_scope(rows)}; {_condition_text(condition, ('args', 'grammar_hint'))}")
    return fig


# ----------------------------------------------------------------------------- 저장·CLI


def save_figure(fig: Figure, path: Path, fmt: str) -> None:
    """그림을 ``fmt`` 형식으로 그려 ``path`` 에 원자적으로 쓴다."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=DPI)
    atomic_write_bytes(path, buffer.getvalue())


def main(argv: list[str] | None = None) -> int:
    """CSV 에서 run 을 골라 그림을 그리고 쓴 파일 경로를 stdout 에 출력한다."""
    parser = argparse.ArgumentParser(prog="python -m analysis.plots", description="결과 그래프 (언어 축·오류 분류·조건 효과·수정 턴)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"결과 CSV (기본 {DEFAULT_CSV.relative_to(ROOT)})")
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--run-id", default=None, metavar="ID", help="이 run 만 그린다")
    sel.add_argument("--latest", action="store_true", help="created_at 최신 run 만 그린다 (기본)")
    sel.add_argument("--all-runs", action="store_true", help="모든 run 을 합쳐 그린다")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"그림을 쓸 디렉터리 (기본 {DEFAULT_OUT.relative_to(ROOT)})")
    parser.add_argument("--format", choices=FORMATS, default="png", help="파일 형식 (기본 png)")
    parser.add_argument("--by-task", action="store_true", help="오류 분류 막대(그림 2)를 과제별 패널로 나눈다")
    parser.add_argument("--primary-speedup", choices=PRIMARY_SPEEDUP_CHOICES, default=None,
                        help="그림 1 의 속도비 주 지표 덮어쓰기 (기본: 설정의 measurement.primary_speedup)")
    parser.add_argument("--config", type=Path, default=None,
                        help=f"measurement.primary_speedup 을 읽을 설정 파일 (기본 {DEFAULT_EXPERIMENT_YAML.relative_to(ROOT)})")
    ns = parser.parse_args(argv)

    out_dir = resolve(ns.out)
    try:
        df = load_runs(resolve(ns.csv))
        selected = df if df.empty else select_runs(df, run_id=ns.run_id, all_runs=ns.all_runs)[0]
    except AggregateError as e:
        print(f"{_LOG} {e}", file=sys.stderr)
        return e.exit_code

    n_conditions = len(selected[list(CONDITION_COLUMNS)].drop_duplicates())
    if n_conditions > 1:
        print(
            f"{_LOG} 조건 조합이 {n_conditions}개다 — 그림 1·2 는 {_condition_text(reference_condition(selected))} 만 그리고, "
            "그림 4 는 그 args·grammar_hint 의 feedback 수준을 모두 그린다 (조건 사이의 차이는 그림 3)",
            file=sys.stderr,
        )
    models = sorted(set(selected["model"]))
    if len(models) > 1:
        print(
            f"{_LOG} 경고: 선택한 행에 모델이 여럿이다 {models} — 그림 1·2 는 모델을 합쳐 세고, "
            "그림 3·4 는 모델별 점·선을 표시 없이 함께 그린다",
            file=sys.stderr,
        )
    runs = sorted(set(selected["run_id"]))
    if len(runs) > 1:
        print(
            f"{_LOG} 경고: 선택한 run 이 {len(runs)}개다 — 그림 1·2 는 run 을 합쳐 세고, 그림 4 는 수정 루프 행이 있는 run 만 "
            "합쳐 그린다. 그림 3 의 args·grammar_hint 는 run 을 합치고 feedback 은 같은 run 안에서만 짝짓는다",
            file=sys.stderr,
        )

    speedup = choose_speedup(selected, ns.primary_speedup, ns.config)   # 선택 전체에서 한 번(analysis.aggregate 와 같은 규칙)
    if speedup.fallback and speedup.n_without_mean:   # 그 밖에는 그림 1 의 축 이름·제목이 쓴 열을 밝힌다
        print(f"{_LOG} 경고: {speedup.describe()}", file=sys.stderr)

    figures: list[tuple[str, Callable[[pd.DataFrame], Figure]]] = [
        ("fig1_languages", functools.partial(fig_languages, speedup=speedup)),
        ("fig2_error_types_by_task" if ns.by_task else "fig2_error_types", functools.partial(fig_error_types, by_task=ns.by_task)),
        ("fig3_condition_effects", fig_condition_effects),
        ("fig4_repair_turns", fig_repair_turns),
    ]
    for name, make in figures:
        try:
            fig = make(selected)
        except NoData as e:
            print(f"{_LOG} {name}: 건너뜀 — {e}", file=sys.stderr)
            continue
        path = out_dir / f"{name}.{ns.format}"
        save_figure(fig, path, ns.format)
        print(path)
    return EXIT_OK


if __name__ == "__main__":
    from harness.procutil import setup_stdio

    setup_stdio()   # 진입점에서만: stderr 의 안내 줄이 cp949 콘솔·파이프에서 UnicodeEncodeError 로 죽지 않게
    sys.exit(main())
