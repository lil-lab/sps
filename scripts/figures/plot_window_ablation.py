#!/usr/bin/env python3
"""Plot final validation NLL by window size, comparing delayed_state, sps, and reverse_sps.

By default plots a single S-scale panel using 20B-token runs, with delayed_state
(\\DelayedState), sps (\\SPS), and reverse_sps (Reverse \\SPS) all drawn as full curves
over the requested window sizes. Missing runs are skipped with a warning.
"""

from __future__ import annotations

import plotting.mpl_cache  # noqa: F401  (pin MPLCONFIGDIR off NFS before matplotlib)

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from plotting.style import (
    COLORBLIND_METHOD_COLORS,
    LEGEND_TEXT_COLOR,
    PLOT_SPINE_COLOR,
    TRAINING_METRIC_KEY,
    _apply_axes_background,
    bf,
    enable_latex,
    sc,
)
from plotting.wandb_runs import (
    USABLE_RUN_STATES,
    FinalPoint,
    ResolvedRun,
    RunSpec,
    _latest_run,
    _parse_created_at,
    normalize_history,
)


DEFAULT_SCALES = ("s",)
DEFAULT_WINDOWS = (0, 16, 64, 256)
DEFAULT_SIZE_SUFFIX = "20b"

FAMILY_COLORS = {
    "delayed_state": COLORBLIND_METHOD_COLORS["delayed_state"],
    "sps": COLORBLIND_METHOD_COLORS["sps"],
    "reverse_sps": "#7B61FF",
    "full_attention": COLORBLIND_METHOD_COLORS["standard"],
}
FAMILY_MARKERS = {
    "delayed_state": "o",
    "sps": "^",
    "reverse_sps": "X",
    "full_attention": "",
}
FAMILY_LABELS = {
    "delayed_state": "Delayed State",
    "sps": "SPS",
    "reverse_sps": "Reverse SPS",
    "full_attention": "Standard",
}

CURVE_FAMILIES = ("delayed_state", "sps", "reverse_sps")
REFERENCE_FAMILY = "full_attention"


def _target_tokens_from_suffix(size_suffix: str) -> int | None:
    s = size_suffix.strip().lower()
    if s.endswith("b"):
        try:
            return int(float(s[:-1]) * 1_000_000_000)
        except ValueError:
            return None
    return None

FIGSIZE = (5.6, 4.6)
PLOT_DPI = 180
PLOT_X_PADDING_FACTOR = 1.18
Y_PADDING_FRACTION = 0.12
MIN_Y_SPAN = 0.02
Y_MAJOR_TICK_INTERVAL = 0.03

LINEWIDTH = 3.4
STROKE_LINEWIDTH = 6.0
MARKER_SIZE = 14
MARKER_EDGE_WIDTH = 1.6
OVERLAY_MARKER_SIZE = 20
OVERLAY_MARKER_EDGE_WIDTH = 2.0

PANEL_TITLE_FONT_SIZE = 26
AXIS_LABEL_FONT_SIZE = 22
AXIS_TICK_LABEL_SIZE = 16
LEGEND_FONT_SIZE = 19


@dataclass(frozen=True)
class FetchSpec:
    scale: str
    family: str
    window: int
    size_suffix: str
    display_name: str


def _build_fetch_specs(
    scales: tuple[str, ...],
    windows: tuple[int, ...],
    size_suffix: str,
) -> list[FetchSpec]:
    specs: list[FetchSpec] = []
    for scale in scales:
        for family in CURVE_FAMILIES:
            for window in windows:
                specs.append(
                    FetchSpec(
                        scale=scale,
                        family=family,
                        window=int(window),
                        size_suffix=size_suffix,
                        display_name=f"{scale}_{family}_w{int(window)}_{size_suffix}",
                    )
                )
        specs.append(
            FetchSpec(
                scale=scale,
                family=REFERENCE_FAMILY,
                window=-1,
                size_suffix=size_suffix,
                display_name=f"{scale}_{REFERENCE_FAMILY}_{size_suffix}",
            )
        )
    return specs


def _fetch_final_points(
    api,
    *,
    entity: str,
    project: str,
    fetch_specs: Iterable[FetchSpec],
) -> list[FinalPoint]:
    points: list[FinalPoint] = []
    for spec in fetch_specs:
        run_spec = RunSpec(
            scale=spec.scale,
            family=spec.family,
            window=spec.window,
            display_name=spec.display_name,
        )
        run = _latest_run(api, entity=entity, project=project, spec=run_spec)
        if run is None:
            print(f"WARNING: no usable W&B run for {spec.display_name!r}")
            continue
        rows = list(
            run.scan_history(
                keys=["_step", "tokens_seen", TRAINING_METRIC_KEY],
                page_size=5000,
            )
        )
        tokens_seen, metric_values = normalize_history(rows)
        if not tokens_seen:
            print(f"WARNING: no usable history for {spec.display_name!r}")
            continue
        points.append(
            FinalPoint(
                scale=spec.scale,
                family=spec.family,
                window=spec.window,
                display_name=spec.display_name,
                run_id=str(getattr(run, "id")),
                created_at=str(getattr(run, "created_at")),
                tokens_seen=int(tokens_seen[-1]),
                final_nll=float(metric_values[-1]),
            )
        )
    return points


def _y_bounds(values: list[float]) -> tuple[float, float]:
    y_min, y_max = min(values), max(values)
    span = max(y_max - y_min, MIN_Y_SPAN)
    padded_min = max(0.0, y_min - Y_PADDING_FRACTION * span)
    padded_max = y_max + Y_PADDING_FRACTION * span
    rounded_min = math.floor(padded_min / Y_MAJOR_TICK_INTERVAL) * Y_MAJOR_TICK_INTERVAL
    rounded_max = math.ceil(padded_max / Y_MAJOR_TICK_INTERVAL) * Y_MAJOR_TICK_INTERVAL
    if rounded_min >= rounded_max:
        rounded_max = rounded_min + Y_MAJOR_TICK_INTERVAL
    return rounded_min, rounded_max


def _scale_panel_points(points: list[FinalPoint], scale: str) -> list[FinalPoint]:
    return [point for point in points if point.scale == scale]


def _series_for_family(
    points: list[FinalPoint],
    *,
    family: str,
    windows: tuple[int, ...],
) -> list[FinalPoint]:
    window_order = {int(window): idx for idx, window in enumerate(windows)}
    return sorted(
        [
            point
            for point in points
            if point.family == family and point.window in window_order
        ],
        key=lambda point: window_order[int(point.window)],
    )


def _overlay_point(points: list[FinalPoint], *, family: str, window: int) -> FinalPoint | None:
    for point in points:
        if point.family == family and int(point.window) == int(window):
            return point
    return None


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


def _plot_curve(
    ax: plt.Axes,
    *,
    series: list[FinalPoint],
    family: str,
) -> None:
    color = FAMILY_COLORS[family]
    marker = FAMILY_MARKERS[family]
    line, = ax.plot(
        [point.window for point in series],
        [point.final_nll for point in series],
        color=color,
        linewidth=LINEWIDTH,
        alpha=0.95,
        zorder=7,
    )
    line.set_path_effects(
        [
            path_effects.Stroke(linewidth=STROKE_LINEWIDTH, foreground="white", alpha=0.85),
            path_effects.Normal(),
        ]
    )
    for point in series:
        marker_line, = ax.plot(
            [point.window],
            [point.final_nll],
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
                    linewidth=MARKER_EDGE_WIDTH + 1.2,
                    foreground="white",
                    alpha=0.9,
                ),
                path_effects.Normal(),
            ]
        )


def _plot_overlay(ax: plt.Axes, *, point: FinalPoint, family: str) -> None:
    color = FAMILY_COLORS[family]
    marker = FAMILY_MARKERS[family]
    marker_line, = ax.plot(
        [point.window],
        [point.final_nll],
        color=color,
        marker=marker,
        linestyle="None",
        markersize=OVERLAY_MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=OVERLAY_MARKER_EDGE_WIDTH,
        zorder=10,
    )
    marker_line.set_path_effects(
        [
            path_effects.Stroke(
                linewidth=OVERLAY_MARKER_EDGE_WIDTH + 1.4,
                foreground="white",
                alpha=0.92,
            ),
            path_effects.Normal(),
        ]
    )


def _legend_handle(family: str) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=FAMILY_COLORS[family],
        marker=FAMILY_MARKERS[family],
        linewidth=LINEWIDTH,
        linestyle="-",
        markersize=MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=MARKER_EDGE_WIDTH,
    )


def render_plot(
    *,
    points: list[FinalPoint],
    output_path: Path,
    scales: tuple[str, ...],
    windows: tuple[int, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(scales), figsize=FIGSIZE, sharey=False)
    if len(scales) == 1:
        axes = [axes]

    x_min = 0.0
    x_max = max(1.0, float(max(windows))) * PLOT_X_PADDING_FACTOR

    for ax, scale in zip(axes, scales, strict=False):
        panel_points = _scale_panel_points(points, scale)
        if not panel_points:
            ax.set_title(bf(scale.upper()), fontsize=PANEL_TITLE_FONT_SIZE, fontweight="bold")
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        y_min, y_max = _y_bounds([p.final_nll for p in panel_points])
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.0, base=2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        _apply_axes_background(ax, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

        reference = _overlay_point(panel_points, family=REFERENCE_FAMILY, window=-1)
        if reference is not None:
            ax.axhline(
                reference.final_nll,
                color=FAMILY_COLORS[REFERENCE_FAMILY],
                linewidth=LINEWIDTH * 0.7,
                linestyle="--",
                alpha=0.85,
                zorder=5,
            )

        for family in CURVE_FAMILIES:
            series = _series_for_family(panel_points, family=family, windows=windows)
            if series:
                _plot_curve(ax, series=series, family=family)

        ax.set_xticks(list(windows))
        ax.set_xticklabels([str(int(w)) for w in windows])
        ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR_TICK_INTERVAL))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.minorticks_off()
        _style_ticks(ax)

    axes[0].set_ylabel(
        bf("Validation NLL"),
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
    )
    axes[0].set_xlabel(
        bf("Temporary Window Size"),
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
        # Window ticks are single-line ("0"/"16"/...), whereas the companion gains plot has
        # two-line ticks (scale name over parameter count) that push its x-title down by ~one
        # line. Pad to match so the two figures' x-titles sit at the same height side-by-side.
        labelpad=25,
    )

    handles = [_legend_handle(family) for family in CURVE_FAMILIES]
    labels = [sc(FAMILY_LABELS[family]) for family in CURVE_FAMILIES]
    handles.append(
        Line2D(
            [0],
            [0],
            color=FAMILY_COLORS[REFERENCE_FAMILY],
            linewidth=LINEWIDTH * 0.7,
            linestyle="--",
        )
    )
    labels.append(sc(FAMILY_LABELS[REFERENCE_FAMILY]))
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

    fig.subplots_adjust(left=0.185, right=0.975, top=0.95, bottom=0.42, wspace=0.18)

    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=PLOT_DPI, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def write_csv(points: list[FinalPoint], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "scale",
                "family",
                "window",
                "display_name",
                "run_id",
                "created_at",
                "tokens_seen",
                "final_nll",
            ]
        )
        for point in sorted(points, key=lambda p: (p.scale, p.family, p.window)):
            writer.writerow(
                [
                    point.scale,
                    point.family,
                    point.window,
                    point.display_name,
                    point.run_id,
                    point.created_at,
                    point.tokens_seen,
                    f"{point.final_nll:.8f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--project", default="pretraining_compression")
    parser.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOWS),
        help="Window sizes to plot for each curve.",
    )
    parser.add_argument(
        "--size-suffix",
        default=DEFAULT_SIZE_SUFFIX,
        help="W&B run-name size suffix, e.g. '10b' or '20b'.",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Also accept W&B runs whose state is 'running' or 'crashed' (e.g. for trained runs that did not gracefully close W&B). Use with --min-tokens-fraction to filter out partially-trained runs.",
    )
    parser.add_argument(
        "--min-tokens-fraction",
        type=float,
        default=1.0,
        help="Drop points whose tokens_seen is less than this fraction of the size-suffix target (e.g. 20B for '20b'). Default 1.0 keeps only runs that reached the target. Set to 0 to disable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/window_ablation_s_20b.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_latex()
    scales = tuple(str(scale) for scale in args.scales)
    windows = tuple(int(window) for window in args.windows)
    size_suffix = str(args.size_suffix)

    import wandb

    if not hasattr(wandb, "Api"):
        raise RuntimeError("Expected the W&B package. Run with `uv run python ...`.")

    # Some fully-trained runs (e.g. reverse_sps w0 = s_reverse_sps_w0_20b, which reached 20B with
    # val/token_nll 2.8827) failed to close W&B cleanly and show up as 'crashed'. Accept those
    # by default; --min-tokens-fraction (default 1.0) drops any genuinely partial run.
    USABLE_RUN_STATES.add("crashed")
    if args.include_running:
        USABLE_RUN_STATES.add("running")
    print(f"Using W&B states: {sorted(USABLE_RUN_STATES)}")
    api = wandb.Api()
    fetch_specs = _build_fetch_specs(scales, windows, size_suffix)
    points = _fetch_final_points(
        api,
        entity=args.entity,
        project=args.project,
        fetch_specs=fetch_specs,
    )

    target_tokens = _target_tokens_from_suffix(size_suffix)
    if target_tokens is not None and args.min_tokens_fraction > 0.0:
        threshold = int(target_tokens * float(args.min_tokens_fraction))
        kept: list[FinalPoint] = []
        for point in points:
            if int(point.tokens_seen) < threshold:
                print(
                    f"WARNING: dropping {point.display_name!r} "
                    f"(tokens_seen={point.tokens_seen} < {threshold} = "
                    f"{args.min_tokens_fraction:.2f}*{target_tokens})"
                )
                continue
            kept.append(point)
        points = kept
    if not points:
        raise RuntimeError("No final points resolved from W&B; check run names/states.")

    render_plot(points=points, output_path=args.output, scales=scales, windows=windows)
    write_csv(points, args.output.with_suffix(".csv"))
    print(f"Resolved points: {len(points)}")


if __name__ == "__main__":
    main()
