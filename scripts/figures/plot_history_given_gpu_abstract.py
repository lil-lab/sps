#!/usr/bin/env python3
"""Render the abstract xl-scale line plot on a training-tokens axis with two Δ arrows."""

from __future__ import annotations

import plotting.mpl_cache  # noqa: F401  (pin MPLCONFIGDIR off NFS before matplotlib)

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "Liberation Serif",
            "DejaVu Serif",
            "serif",
        ],
    }
)
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MultipleLocator

from scripts.figures.plot_validation_nll_history_wandb import (
    HISTORY_PRE_DECAY_TOKEN_CUTOFF_BY_SIZE,
    HISTORY_SIZE_SUFFIX_BY_SCALE,
    _nice_tick_interval,
    _paired_image_paths,
)
from plotting.style import (
    COLORBLIND_METHOD_COLORS,
    LEGEND_TEXT_COLOR,
    PLOT_SPINE_COLOR,
    _apply_axes_background,
    bf,
    enable_latex,
    sc,
)


DEFAULT_CSV_PATH = Path("figures/history_given_gpu.csv")
DEFAULT_OUTPUT_PATH = Path("figures/history_given_gpu_xl_abstract.png")
DEFAULT_SCALE = "xl"
# "Standard (More Tokens)" is the long full-attention run; for xl this is the in-progress
# 100b run, drawn until it matches SPS's pre-decay NLL (mirrors the combined history plot).
STANDARD_MORE_TOKENS_DISPLAY_NAME = "xl_full_attention_100b"
STANDARD_BASE_DISPLAY_NAME = "xl_full_attention_20b"
OURS_DISPLAY_NAME = "xl_sps_w64_20b"

OURS_METHOD_NAME = "State-Prediction Separation"

ARROW_ORANGE = "#E07B00"

FIGSIZE = (10.5, 2.82)
PLOT_DPI = 180
TWENTY_B_SIZE_SUFFIX = "20b"

X_MIN_TOKENS = 8e9
X_PADDING_FACTOR = 1.02
Y_PADDING_FRACTION = 0.20
MIN_Y_SPAN = 0.04
Y_LIMIT_ROUNDING_INTERVAL = 0.005
UPPER_Y_LIMIT = 2.62
LOWER_Y_LIMIT = 2.51

EMPHASIZED_LINEWIDTH = 3.10

AXIS_LABEL_FONT_SIZE = 16.0
AXIS_TICK_LABEL_SIZE = 12.0
LEGEND_FONT_SIZE = 15.0
DELTA_LABEL_FONT_SIZE = 11.0

AXIS_TICK_LENGTH = 4.0
AXIS_TICK_WIDTH = 0.85

@dataclass(frozen=True)
class HistoryPoint:
    tokens_seen: int
    validation_nll: float


@dataclass(frozen=True)
class HistorySeries:
    label: str
    color: str
    linewidth: float
    alpha: float
    zorder: int
    points: tuple[HistoryPoint, ...]


def _pre_decay_token_cutoff_for_size_suffix(size_suffix: str) -> int:
    try:
        return HISTORY_PRE_DECAY_TOKEN_CUTOFF_BY_SIZE[size_suffix]
    except KeyError as exc:
        raise ValueError(f"No pre-decay token cutoff configured for size={size_suffix!r}") from exc


def _pre_decay_token_cutoff(scale: str) -> int:
    try:
        size_suffix = HISTORY_SIZE_SUFFIX_BY_SCALE[scale]
        return _pre_decay_token_cutoff_for_size_suffix(size_suffix)
    except KeyError as exc:
        raise ValueError(f"No pre-decay token cutoff configured for scale={scale!r}") from exc


def _read_points(
    csv_path: Path,
    *,
    display_name: str,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> list[HistoryPoint]:
    points: list[HistoryPoint] = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["display_name"] != display_name:
                continue
            tokens_seen = int(row["tokens_seen"])
            if min_tokens is not None and tokens_seen < min_tokens:
                continue
            if max_tokens is not None and tokens_seen > max_tokens:
                continue
            points.append(
                HistoryPoint(
                    tokens_seen=tokens_seen,
                    validation_nll=float(row["validation_nll"]),
                )
            )
    points.sort(key=lambda p: p.tokens_seen)
    return points


def _read_sps_series(csv_path: Path) -> HistorySeries:
    cutoff_18b = _pre_decay_token_cutoff_for_size_suffix(TWENTY_B_SIZE_SUFFIX)
    points = _read_points(csv_path, display_name=OURS_DISPLAY_NAME, max_tokens=cutoff_18b)
    if not points:
        raise ValueError(f"No pre-decay points found for {OURS_DISPLAY_NAME!r}")
    return HistorySeries(
        label=f"{sc(OURS_METHOD_NAME)} (18B tokens)",
        color=COLORBLIND_METHOD_COLORS["sps"],
        linewidth=EMPHASIZED_LINEWIDTH,
        alpha=0.95,
        zorder=13,
        points=tuple(points),
    )


def _read_standard_more_tokens_series(csv_path: Path, *, target_nll: float) -> HistorySeries:
    # Stitch the base full-attention run's 0-18B pre-decay head onto the long 100b run's
    # >18B tail, then truncate at the first point whose validation NLL reaches target_nll
    # (SPS's pre-decay endpoint) -- where "Standard (More Tokens)" finally matches SPS. The
    # head makes the line start well before the 18B handoff; the tail is the extra tokens.
    cutoff_18b = _pre_decay_token_cutoff_for_size_suffix(TWENTY_B_SIZE_SUFFIX)
    head = _read_points(csv_path, display_name=STANDARD_BASE_DISPLAY_NAME, max_tokens=cutoff_18b)
    tail = _read_points(
        csv_path, display_name=STANDARD_MORE_TOKENS_DISPLAY_NAME, min_tokens=cutoff_18b + 1
    )
    if not head:
        raise ValueError(f"No pre-decay head points found for {STANDARD_BASE_DISPLAY_NAME!r}")
    if not tail:
        raise ValueError(f"No tail points found for {STANDARD_MORE_TOKENS_DISPLAY_NAME!r}")
    points = sorted(head + tail, key=lambda p: p.tokens_seen)
    truncated: list[HistoryPoint] = []
    for point in points:
        truncated.append(point)
        if point.validation_nll <= target_nll:
            break
    end_tokens_b = truncated[-1].tokens_seen / 1e9
    return HistorySeries(
        label=f"{sc('Standard')} ({end_tokens_b:.0f}B tokens)",
        color=COLORBLIND_METHOD_COLORS["standard"],
        linewidth=EMPHASIZED_LINEWIDTH,
        alpha=0.85,
        zorder=11,
        points=tuple(truncated),
    )


def _interp_nll(series: HistorySeries, tokens: float) -> float:
    xs = np.array([p.tokens_seen for p in series.points], dtype=float)
    ys = np.array([p.validation_nll for p in series.points])
    return float(np.interp(tokens, xs, ys))


def _tokens_formatter(value: float, _pos: int) -> str:
    billions = value / 1e9
    if abs(billions - round(billions)) < 1e-6:
        return f"{int(round(billions))}B"
    return f"{billions:g}B"


def _y_bounds(series_list: tuple[HistorySeries, ...], *, x_min: float) -> tuple[float, float]:
    values = [
        point.validation_nll
        for series in series_list
        for point in series.points
        if point.tokens_seen >= x_min
    ]
    if not values:
        values = [series.points[-1].validation_nll for series in series_list]
    y_min, y_max = min(values), max(values)
    span = max(y_max - y_min, MIN_Y_SPAN)
    padded_min = max(0.0, y_min - Y_PADDING_FRACTION * span)
    padded_max = y_max + Y_PADDING_FRACTION * span
    rounded_min = math.floor(padded_min / Y_LIMIT_ROUNDING_INTERVAL) * Y_LIMIT_ROUNDING_INTERVAL
    rounded_max = math.ceil(padded_max / Y_LIMIT_ROUNDING_INTERVAL) * Y_LIMIT_ROUNDING_INTERVAL
    if rounded_min >= rounded_max:
        rounded_max = rounded_min + Y_LIMIT_ROUNDING_INTERVAL
    return rounded_min, rounded_max


def _style_ticks(ax: plt.Axes, *, tick_label_size: float = AXIS_TICK_LABEL_SIZE) -> None:
    ax.tick_params(
        axis="both",
        colors=PLOT_SPINE_COLOR,
        labelcolor=LEGEND_TEXT_COLOR,
        labelsize=tick_label_size,
        length=AXIS_TICK_LENGTH,
        width=AXIS_TICK_WIDTH,
    )
    for tick in [*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks()]:
        tick.tick1line.set_color(PLOT_SPINE_COLOR)
        tick.tick2line.set_color(PLOT_SPINE_COLOR)


def _plot_series(ax: plt.Axes, series: HistorySeries) -> None:
    ax.plot(
        [p.tokens_seen for p in series.points],
        [p.validation_nll for p in series.points],
        color=series.color,
        linewidth=series.linewidth,
        alpha=series.alpha,
        solid_capstyle="round",
        zorder=series.zorder,
    )


def _add_white_halo(text_artist: plt.Text, *, linewidth: float = 2.4, alpha: float = 0.92) -> None:
    text_artist.set_path_effects(
        [
            path_effects.Stroke(linewidth=linewidth, foreground="white", alpha=alpha),
            path_effects.Normal(),
        ]
    )


def _draw_regions_and_deltas(
    ax: plt.Axes,
    *,
    standard: HistorySeries,
    sps: HistorySeries,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    annotation_scale: float = 1.0,
    text_scale: float = 1.0,
) -> None:
    x_span = x_max - x_min
    delta_font_size = DELTA_LABEL_FONT_SIZE * text_scale
    arrow_linewidth = 1.6 * annotation_scale
    star_markersize = 20 * annotation_scale
    halo_linewidth = 2.8 * text_scale

    cutoff_18b = float(_pre_decay_token_cutoff_for_size_suffix(TWENTY_B_SIZE_SUFFIX))

    # SPS reaches target_nll at its 18B pre-decay endpoint; Standard (More Tokens) reaches
    # the same NLL only at standard_match_x (the truncated endpoint of the 100b run).
    target_nll = sps.points[-1].validation_nll
    standard_at_18b = _interp_nll(standard, cutoff_18b)
    standard_match_x = float(standard.points[-1].tokens_seen)

    star_specs = (
        (cutoff_18b, standard_at_18b, standard.color),
        (cutoff_18b, target_nll, sps.color),
        (standard_match_x, target_nll, ARROW_ORANGE),
    )
    for x_pos, y_pos, face_color in star_specs:
        ax.plot(
            [x_pos],
            [y_pos],
            marker="*",
            markersize=star_markersize,
            markerfacecolor=face_color,
            markeredgecolor="white",
            markeredgewidth=1.7,
            linestyle="none",
            zorder=15,
        )

    # Vertical Δ NLL at equal tokens (18B): Standard above, SPS at the target NLL.
    ax.annotate(
        "",
        xy=(cutoff_18b, target_nll),
        xytext=(cutoff_18b, standard_at_18b),
        arrowprops={
            "arrowstyle": "<->",
            "color": standard.color,
            "linewidth": arrow_linewidth,
            "shrinkA": 14 * annotation_scale,
            "shrinkB": 12 * annotation_scale,
        },
        zorder=15,
    )
    delta_18b = standard_at_18b - target_nll
    delta_label = ax.text(
        cutoff_18b + 0.010 * x_span,
        0.5 * (standard_at_18b + target_nll),
        bf(rf"$\Delta$ NLL $= -{delta_18b:.3f}$"),
        ha="left",
        va="center",
        fontsize=delta_font_size,
        fontweight="bold",
        color=standard.color,
        zorder=16,
    )
    _add_white_halo(delta_label, linewidth=halo_linewidth)

    # Horizontal token-efficiency arrow at the SPS NLL: 18B (SPS) -> match point (Standard).
    ax.annotate(
        "",
        xy=(standard_match_x, target_nll),
        xytext=(cutoff_18b, target_nll),
        arrowprops={
            "arrowstyle": "<->",
            "color": ARROW_ORANGE,
            "linewidth": arrow_linewidth,
            "shrinkA": 14 * annotation_scale,
            "shrinkB": 14 * annotation_scale,
        },
        zorder=18,
    )
    saved_ratio = standard_match_x / cutoff_18b if cutoff_18b > 0 else float("inf")
    saved_label = ax.text(
        0.5 * (cutoff_18b + standard_match_x),
        target_nll - 0.006,
        bf(rf"${saved_ratio:.1f}\times$ more token efficient"),
        ha="center",
        va="top",
        fontsize=delta_font_size,
        fontweight="bold",
        color=ARROW_ORANGE,
        zorder=16,
    )
    _add_white_halo(saved_label, linewidth=halo_linewidth)


def _add_legend(
    fig: plt.Figure,
    series_list: tuple[HistorySeries, ...],
    *,
    font_size: float = LEGEND_FONT_SIZE,
) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=series.color,
            linewidth=series.linewidth,
            alpha=min(1.0, series.alpha + 0.05),
        )
        for series in series_list
    ]
    labels = [series.label for series in series_list]
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.005),
        ncol=len(labels),
        frameon=False,
        fontsize=font_size,
        handlelength=1.8,
        handletextpad=0.55,
        columnspacing=1.50,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(LEGEND_TEXT_COLOR)


def render_plot(
    *,
    csv_path: Path,
    output_path: Path,
    scale: str,
    height_scale: float = 1.0,
) -> None:
    _pre_decay_token_cutoff(scale)
    sps = _read_sps_series(csv_path)
    target_nll = sps.points[-1].validation_nll
    standard = _read_standard_more_tokens_series(csv_path, target_nll=target_nll)
    series_list = (standard, sps)

    x_min = X_MIN_TOKENS
    x_max = float(standard.points[-1].tokens_seen) * X_PADDING_FACTOR
    y_min = LOWER_Y_LIMIT
    y_max = UPPER_Y_LIMIT

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figsize = (FIGSIZE[0], FIGSIZE[1] * height_scale)
    fig, ax = plt.subplots(figsize=figsize)
    _apply_axes_background(ax, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    for series in series_list:
        _plot_series(ax, series)

    annotation_scale = height_scale
    text_scale = max(1.0, 1.0 + (height_scale - 1.0) * 0.55)
    chrome_scale = max(1.0, 1.0 + (height_scale - 1.0) * 0.40)
    _draw_regions_and_deltas(
        ax,
        standard=standard,
        sps=sps,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        annotation_scale=annotation_scale,
        text_scale=text_scale,
    )

    ax.set_xlabel(
        bf("Training Tokens"),
        fontsize=AXIS_LABEL_FONT_SIZE * chrome_scale,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
        labelpad=4,
    )
    ax.set_ylabel(
        f"{bf('Pre-Decay')}\n{bf('NLL')}",
        fontsize=AXIS_LABEL_FONT_SIZE * chrome_scale,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
        labelpad=4,
    )
    tick_interval_b = max(1.0, _nice_tick_interval((x_max - x_min) / 1e9))
    ax.xaxis.set_major_locator(MultipleLocator(tick_interval_b * 1e9))
    ax.xaxis.set_major_formatter(FuncFormatter(_tokens_formatter))
    ax.yaxis.set_major_locator(MultipleLocator(0.03))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.minorticks_off()
    _style_ticks(ax, tick_label_size=AXIS_TICK_LABEL_SIZE * chrome_scale)

    _add_legend(fig, series_list, font_size=LEGEND_FONT_SIZE * chrome_scale)
    base_height = FIGSIZE[1]
    top_inches = (1.0 - 0.975) * base_height
    bottom_inches = 0.370 * base_height * (1.0 + (chrome_scale - 1.0) * 0.45)
    new_height = figsize[1]
    fig.subplots_adjust(
        left=0.115,
        right=0.985,
        top=1.0 - top_inches / new_height,
        bottom=bottom_inches / new_height,
    )

    png_path, pdf_path = _paired_image_paths(output_path)
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--scale", default=DEFAULT_SCALE)
    parser.add_argument("--height-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_latex()
    render_plot(
        csv_path=args.csv_path,
        output_path=args.output,
        scale=args.scale,
        height_scale=args.height_scale,
    )


if __name__ == "__main__":
    main()
