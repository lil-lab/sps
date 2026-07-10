#!/usr/bin/env python3
"""Plot average downstream task-accuracy gain over Standard as a function of model scale.

Companion to ``plot_window_ablation.py``. For each model scale we take the same
"Task Accuracy" number the compact main-results table reports -- the mean of 5 zero-shot
benchmarks (ARC-E, HellaSwag, PIQA, SciQ, LAMBADA), i.e. ``TableRow.downstream_avg`` -- and
plot the gain of each method over the Standard baseline (in percentage points) against model
scale on a log x-axis. Ticks are labelled with both the size letter and the parameter count,
e.g. ``S (131M)``. Data is resolved via ``collect_table_rows`` so the numbers match the table
exactly. Missing (scale, method) accuracies are skipped with a warning.
"""

from __future__ import annotations

import plotting.mpl_cache  # noqa: F401  (pin MPLCONFIGDIR off NFS before matplotlib)

import argparse
import csv
import os
import sys
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
from matplotlib.lines import Line2D

# ``export_main_results_table`` lives in scripts/tables (not an installed package); ``plotting.*``
# is importable via the installed ``src`` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tables"))

from plotting.style import (  # noqa: E402
    COLORBLIND_METHOD_COLORS,
    LEGEND_TEXT_COLOR,
    PLOT_SPINE_COLOR,
    _apply_axes_background,
    bf,
    enable_latex,
    sc,
)
from export_main_results_table import collect_table_rows  # noqa: E402


DEFAULT_SCALES = ("xs", "s", "m", "l", "xl")
STANDARD_KEY = "standard"
# Non-standard methods to draw, in legend/order. Keys match ``MainMethodSpec.key``.
METHOD_KEYS = ("delayed_state", "two_x_memory", "sps")

# Non-embedding parameter counts, verified identical across methods within a scale from
# outputs/generation_timing_correctness/THROUGHPUT_b8_all5_h100/results.json.
SCALE_PARAMS = {
    "xs": 53_027_328,
    "s": 130_665_216,
    "m": 378_717_184,
    "l": 831_253_760,
    "xl": 1_678_081_600,
}

METHOD_COLORS = {
    "delayed_state": COLORBLIND_METHOD_COLORS["delayed_state"],
    "two_x_memory": "#0072B2",
    "sps": COLORBLIND_METHOD_COLORS["sps"],
}
METHOD_MARKERS = {
    "delayed_state": "o",
    "two_x_memory": "s",
    "sps": "^",
}
METHOD_LABELS = {
    "delayed_state": "Delayed State",
    "two_x_memory": "2x Memory",
    "sps": "SPS",
}
STANDARD_COLOR = COLORBLIND_METHOD_COLORS["standard"]
STANDARD_LABEL = "Standard"

FIGSIZE = (5.6, 4.6)
PLOT_DPI = 180
Y_PADDING_FRACTION = 0.12
MIN_Y_SPAN = 0.5

LINEWIDTH = 3.4
STROKE_LINEWIDTH = 6.0
MARKER_SIZE = 14
MARKER_EDGE_WIDTH = 1.6

AXIS_LABEL_FONT_SIZE = 22
AXIS_TICK_LABEL_SIZE = 16
LEGEND_FONT_SIZE = 19


def _scale_tick_label(scale: str, params: int) -> str:
    # Two-line tick label: scale name on top, parameter count on the line below.
    if params < 1_000_000_000:
        size = f"{params / 1e6:.0f}M"
    else:
        size = f"{params / 1e9:.1f}B"
    return f"{scale.upper()}\n{size}"


def _collect_gains(
    api, *, entity: str, project: str, scales: tuple[str, ...]
) -> dict[tuple[str, str], float]:
    """Return {(scale, method_key): gain_in_pp} for every non-standard method with data."""
    rows = collect_table_rows(api, entity=entity, project=project, scales=scales)
    acc: dict[tuple[str, str], float | None] = {
        (row.scale, row.method.key): row.downstream_avg for row in rows
    }
    gains: dict[tuple[str, str], float] = {}
    for scale in scales:
        standard = acc.get((scale, STANDARD_KEY))
        if standard is None:
            print(f"WARNING: no Standard task accuracy for scale {scale!r}; skipping scale")
            continue
        for key in METHOD_KEYS:
            value = acc.get((scale, key))
            if value is None:
                print(f"WARNING: no task accuracy for {scale}/{key}; skipping point")
                continue
            gains[(scale, key)] = (float(value) - float(standard)) * 100.0
    return gains


def _y_bounds(values: list[float]) -> tuple[float, float]:
    lo, hi = min([0.0, *values]), max([0.0, *values])
    span = max(hi - lo, MIN_Y_SPAN)
    return lo - Y_PADDING_FRACTION * span, hi + Y_PADDING_FRACTION * span


def _style_ticks(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        colors=PLOT_SPINE_COLOR,
        labelcolor=LEGEND_TEXT_COLOR,
        labelsize=AXIS_TICK_LABEL_SIZE,
        length=4.0,
        width=0.85,
    )
    for tick in (*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks()):
        tick.tick1line.set_color(PLOT_SPINE_COLOR)
        tick.tick2line.set_color(PLOT_SPINE_COLOR)


def _plot_curve(ax: plt.Axes, *, xs: list[float], ys: list[float], key: str) -> None:
    color = METHOD_COLORS[key]
    marker = METHOD_MARKERS[key]
    (line,) = ax.plot(xs, ys, color=color, linewidth=LINEWIDTH, alpha=0.95, zorder=7)
    line.set_path_effects(
        [
            path_effects.Stroke(linewidth=STROKE_LINEWIDTH, foreground="white", alpha=0.85),
            path_effects.Normal(),
        ]
    )
    for x, y in zip(xs, ys):
        (marker_line,) = ax.plot(
            [x],
            [y],
            color=color,
            marker=marker,
            linestyle="None",
            markersize=MARKER_SIZE,
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            zorder=9,
        )
        marker_line.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=MARKER_EDGE_WIDTH + 1.2, foreground="white", alpha=0.9
                ),
                path_effects.Normal(),
            ]
        )


def _legend_handle(key: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=METHOD_COLORS[key],
        marker=METHOD_MARKERS[key],
        linewidth=LINEWIDTH,
        linestyle="-",
        markersize=MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=MARKER_EDGE_WIDTH,
    )


def render_plot(
    *,
    gains: dict[tuple[str, str], float],
    scales: tuple[str, ...],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotted_scales = [s for s in scales if any((s, key) in gains for key in METHOD_KEYS)]
    if not plotted_scales:
        raise RuntimeError("No (scale, method) gains resolved from W&B; check run names/states.")

    xs_by_scale = {s: float(SCALE_PARAMS[s]) for s in plotted_scales}
    all_values = list(gains.values())
    x_positions = [xs_by_scale[s] for s in plotted_scales]
    x_min = min(x_positions) / 1.4
    x_max = max(x_positions) * 1.4
    y_min, y_max = _y_bounds(all_values)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    _apply_axes_background(ax, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    ax.axhline(
        0.0,
        color=STANDARD_COLOR,
        linewidth=LINEWIDTH * 0.7,
        linestyle="--",
        alpha=0.85,
        zorder=5,
    )

    for key in METHOD_KEYS:
        series = [(xs_by_scale[s], gains[(s, key)]) for s in plotted_scales if (s, key) in gains]
        if not series:
            continue
        xs = [pt[0] for pt in series]
        ys = [pt[1] for pt in series]
        _plot_curve(ax, xs=xs, ys=ys, key=key)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([_scale_tick_label(s, SCALE_PARAMS[s]) for s in plotted_scales])
    ax.minorticks_off()
    _style_ticks(ax)

    ax.set_ylabel(
        bf("Accuracy Gain (pp)"),
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
    )
    ax.set_xlabel(
        bf("Model Scale"),
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
        labelpad=8,
    )

    handles = [_legend_handle(key) for key in METHOD_KEYS]
    labels = [sc(METHOD_LABELS[key]) for key in METHOD_KEYS]
    handles.append(Line2D([0], [0], color=STANDARD_COLOR, linewidth=LINEWIDTH * 0.7, linestyle="--"))
    labels.append(sc(STANDARD_LABEL))
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.0,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    for text in legend.get_texts():
        text.set_color(LEGEND_TEXT_COLOR)

    fig.subplots_adjust(left=0.185, right=0.975, top=0.95, bottom=0.42)

    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=PLOT_DPI, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def write_csv(
    gains: dict[tuple[str, str], float], scales: tuple[str, ...], csv_path: Path
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scale", "params", "method", "gain_pp"])
        for scale in scales:
            for key in METHOD_KEYS:
                if (scale, key) not in gains:
                    continue
                writer.writerow(
                    [scale, SCALE_PARAMS[scale], key, f"{gains[(scale, key)]:.6f}"]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default="pretraining_compression")
    parser.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/accuracy_gain_by_scale_20b.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_latex()
    scales = tuple(str(scale) for scale in args.scales)
    unknown = [s for s in scales if s not in SCALE_PARAMS]
    if unknown:
        raise ValueError(f"Unknown scale(s) {unknown}; known: {sorted(SCALE_PARAMS)}")

    import wandb

    if not hasattr(wandb, "Api"):
        raise RuntimeError("Expected the W&B package. Run with `uv run python ...`.")

    api = wandb.Api()
    gains = _collect_gains(api, entity=args.entity, project=args.project, scales=scales)
    if not gains:
        raise RuntimeError("No gains resolved from W&B; check run names/states.")

    render_plot(gains=gains, scales=scales, output_path=args.output)
    write_csv(gains, scales, args.output.with_suffix(".csv"))
    print(f"Resolved gains: {len(gains)} (scale, method) points")


if __name__ == "__main__":
    main()
