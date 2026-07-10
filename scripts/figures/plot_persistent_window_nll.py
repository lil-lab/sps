#!/usr/bin/env python3
"""Render the persistent-window NLL (delta-NLL) plot: per-position loss
degradation when each method's persistent state is restricted to a sliding
window, one column per scale (the paper's persistent-state-usage figure,
figures/analysis_plot_delta_nll.png). The gradient-ratio figure is produced
separately by plot_gradient_params.py; both share styling helpers from
plotting.figure_common."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# Importing figure_common selects the Agg backend + serif rcParams (and pins
# MPLCONFIGDIR off NFS) before matplotlib.pyplot is imported below, and provides the
# styling constants + series/plot helpers shared with plot_gradient_params.py.
from plotting.figure_common import (
    AXIS_LABEL_FONT_SIZE,
    LINEWIDTH,
    PANEL_GRID_WSPACE,
    PANEL_TITLE_FONT_SIZE,
    PLOT_DPI,
    SPLIT_FIGURE_HEIGHT,
    SPLIT_PANEL_TOP,
    Y_AXIS_LABEL_COORD_X,
    _array,
    _method_color,
    _method_label,
    _paired_image_paths,
    _plot_line,
    _read_json,
    _rounded_upper,
    _smooth_series,
    _stat_value,
    _style_ticks,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np

from plotting.style import (
    LEGEND_TEXT_COLOR,
    _apply_axes_background,
    bf,
    enable_latex,
    sc,
)


DEFAULT_SCALES = ("xs", "s", "m", "l", "xl")
DEFAULT_WINDOW = 64
# Fixed total figure width (inches): keeps each figure's size constant as the
# number of scale panels grows from four (s,m,l,xl) to five (xs..xl).
FIXED_FIGURE_WIDTH = 17.0
# Sharded/incremental run (without-replacement docs, ~1/sqrt(N) noise); written by
# run_persistent_window_nll.sh. Previous (noisier, n=256, with-replacement) root was
# "outputs/persistent_window_nll_analysis/xs_s_m_l_w64_only_n256_bins64_pw64_256".
DEFAULT_PERSISTENT_ROOT = Path(
    "outputs/persistent_window_nll_analysis/xs_s_m_l_xl_w64_sharded_pw64"
)
DEFAULT_OUTPUT_PATH = Path("figures/analysis_plot.png")
PERSISTENT_METHODS: tuple[tuple[str, int], ...] = (
    ("sps", DEFAULT_WINDOW),
    ("delayed_state", DEFAULT_WINDOW),
)
BOTTOM_X_MAX = 2048.0
SPLIT_BOTTOM_X_MIN = 64.0
SPLIT_BOTTOM_X_TICKS = (64, 512, 1024, 1536, 2048)
ZERO_LINE_COLOR = "#1f1b16"

# Per-position delta-NLL smoothing (mirrors plot_gradient_params.py): the split
# delta-NLL panel plots every document-relative position (no binning) as a faint
# raw curve, with a Savitzky-Golay-smoothed trend line overlaid on top in the
# same color. delta-NLL is a difference on a linear axis (not heavy-tailed), so
# smoothing runs in linear space (log=False). The window is wider than the
# gradient plot's because there are ~2k positions vs ~512 offsets.
DELTA_SMOOTH_CHOICES = ("none", "savgol", "median", "gaussian")
DELTA_SMOOTH = "savgol"
DELTA_SMOOTH_WINDOW = 201      # odd, in position units
DELTA_SMOOTH_POLYORDER = 2     # savgol local-polynomial order
# Lighter than the gradient plot's 0.30: this panel draws ~2k per-position points
# (vs ~512 offsets), so a lower opacity gives the same perceived faint underlay.
DELTA_RAW_ALPHA = 0.22         # opacity of the faint raw underlay
DELTA_RAW_LINEWIDTH_SCALE = 0.45   # raw line width relative to LINEWIDTH
DELTA_RAW_ZORDER = 4               # above grid/axhline (2), below smoothed (6)
LEGEND_FONT_SIZE = 28
SPLIT_PANEL_BOTTOM_DELTA = 0.42
SPLIT_X_AXIS_LABEL_Y_DELTA = 0.30
SPLIT_METHOD_LEGEND_ANCHOR_DELTA = (0.5, 0.04)


def _load_persistent_summary(persistent_root: Path, scale: str, window: int) -> dict[str, Any]:
    path = persistent_root / scale / f"pw{int(window)}" / "persistent_window_nll_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing persistent-window summary: {path}")
    return _read_json(path)


def _persistent_model_by_id(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(model.get("model_id")): model for model in summary.get("models", [])}


def _persistent_model_for_family(
    models: dict[str, dict[str, Any]], family: str
) -> dict[str, Any] | None:
    """Look up a persistent model by family."""
    return models.get(family)


def _persistent_position_series(model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Per-position (un-binned) delta-NLL mean vs document-relative position.

    Reads the dense ``positions`` field (one row per position) rather than the
    pre-binned ``position_bins`` field, so the split delta-NLL panel can plot
    every position and overlay its own smoothed trend (mirroring the param-gradient
    plot). Non-finite means are dropped; rows stay in position order.
    """
    rows = model.get("positions") or []
    x = np.asarray([float(row.get("position", 0)) for row in rows], dtype=float)
    y = _array([_stat_value(row["delta_nll"], "mean") for row in rows])
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def _collect_persistent_position_bound(
    persistent_summaries: dict[str, dict[str, Any]],
    scales: tuple[str, ...],
) -> tuple[float, float]:
    """Y-bound for the per-position split delta-NLL panel: bound to the raw
    per-position means (the smoothed trend sits below the raw spikes, exactly as
    in the param-gradient plot)."""
    values: list[float] = []
    for scale in scales:
        models = _persistent_model_by_id(persistent_summaries[scale])
        for family, _window in PERSISTENT_METHODS:
            model = _persistent_model_for_family(models, family)
            if model is None or not model.get("applicable", True):
                continue
            _x, mean = _persistent_position_series(model)
            values.extend(mean[np.isfinite(mean)].tolist())
    return (0.0, _rounded_upper(values, floor=0.05))


def render_delta_nll_only(
    *,
    persistent_summaries: dict[str, dict[str, Any]],
    persistent_methods: tuple[tuple[str, int], ...],
    persistent_y_bounds: tuple[float, float],
    output_path: Path,
    scales: tuple[str, ...],
    smooth: str = DELTA_SMOOTH,
    smooth_window: int = DELTA_SMOOTH_WINDOW,
    smooth_polyorder: int = DELTA_SMOOTH_POLYORDER,
    raw_alpha: float = DELTA_RAW_ALPHA,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(scales),
        figsize=(FIXED_FIGURE_WIDTH, SPLIT_FIGURE_HEIGHT),
        sharey=True,
    )
    if len(scales) == 1:
        axes = np.asarray([axes])

    for col, scale in enumerate(scales):
        ax = axes[col]
        _apply_axes_background(
            ax,
            x_min=SPLIT_BOTTOM_X_MIN,
            x_max=BOTTOM_X_MAX,
            y_min=persistent_y_bounds[0],
            y_max=persistent_y_bounds[1],
        )
        ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=1.8, alpha=0.75, zorder=2)
        ax.set_xlim(SPLIT_BOTTOM_X_MIN, BOTTOM_X_MAX)
        ax.set_ylim(*persistent_y_bounds)
        models = _persistent_model_by_id(persistent_summaries[scale])
        for family, method_window in persistent_methods:
            model = _persistent_model_for_family(models, family)
            if model is None or not model.get("applicable", True):
                continue
            # Per-position (un-binned) series; trim the leading within-window
            # region where the forced and baseline NLL coincide (delta == 0).
            x, mean = _persistent_position_series(model)
            nonzero = np.isfinite(mean) & (mean != 0.0)
            if not nonzero.any():
                continue
            start = int(np.argmax(nonzero))
            x = x[start:]
            mean = mean[start:]
            color = _method_color(family, method_window)
            if smooth != "none":
                # Faint raw per-position curve underneath (no white halo), with
                # the smoothed trend line on top in the same color -- reads as one
                # series. delta-NLL is linear, so smooth in linear space.
                ax.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=LINEWIDTH * DELTA_RAW_LINEWIDTH_SCALE,
                    alpha=raw_alpha,
                    solid_capstyle="round",
                    zorder=DELTA_RAW_ZORDER,
                )
                mean = _smooth_series(
                    mean,
                    method=smooth,
                    window=smooth_window,
                    polyorder=smooth_polyorder,
                    log=False,
                )
            _plot_line(ax, x, mean, color=color, zorder=6)
        ax.set_title(
            bf(scale.upper()),
            fontsize=PANEL_TITLE_FONT_SIZE,
            fontweight="bold",
            color=LEGEND_TEXT_COLOR,
            pad=9,
        )
        ax.set_xticks(list(SPLIT_BOTTOM_X_TICKS))
        # Force 0.2-spaced (single-decimal) y-ticks: the per-position bound (~0.95)
        # would otherwise auto-pick 0.25 spacing -> wider "0.00"/"0.25" labels that
        # collide with the rotated y-axis label at this narrow 5-panel width.
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        _style_ticks(ax)
        ax.minorticks_off()

    axes[0].set_ylabel(
        bf(r"$\Delta \ell$"),
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
    )
    axes[0].yaxis.set_label_coords(Y_AXIS_LABEL_COORD_X, 0.5)
    fig.text(
        0.5,
        SPLIT_X_AXIS_LABEL_Y_DELTA,
        bf("Document-relative query position"),
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            color=_method_color(family, method_window),
            linewidth=LINEWIDTH,
            label=sc(_method_label(family, method_window)),
        )
        for family, method_window in persistent_methods
    ]
    method_legend = fig.legend(
        handles=method_handles,
        loc="lower center",
        ncol=len(method_handles),
        frameon=False,
        bbox_to_anchor=SPLIT_METHOD_LEGEND_ANCHOR_DELTA,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.35,
        columnspacing=1.05,
        handletextpad=0.5,
    )
    for text in method_legend.get_texts():
        text.set_color(LEGEND_TEXT_COLOR)

    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=SPLIT_PANEL_BOTTOM_DELTA,
        top=SPLIT_PANEL_TOP,
        wspace=PANEL_GRID_WSPACE,
    )
    png_path, pdf_path = _paired_image_paths(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_delta_nll_plot(
    *,
    persistent_root: Path,
    output_path: Path,
    scales: tuple[str, ...],
    window: int,
    delta_smooth: str = DELTA_SMOOTH,
    delta_smooth_window: int = DELTA_SMOOTH_WINDOW,
    delta_smooth_polyorder: int = DELTA_SMOOTH_POLYORDER,
    delta_raw_alpha: float = DELTA_RAW_ALPHA,
) -> Path:
    persistent_methods = tuple((family, int(window)) for family, _method_window in PERSISTENT_METHODS)
    persistent_summaries = {
        scale: _load_persistent_summary(persistent_root, scale, int(window))
        for scale in scales
    }
    persistent_y_bounds = _collect_persistent_position_bound(persistent_summaries, scales)

    base = output_path.with_suffix("") if output_path.suffix.lower() in {".png", ".pdf"} else output_path
    delta_path = base.parent / f"{base.name}_delta_nll.png"

    render_delta_nll_only(
        persistent_summaries=persistent_summaries,
        persistent_methods=persistent_methods,
        persistent_y_bounds=persistent_y_bounds,
        output_path=delta_path,
        scales=scales,
        smooth=delta_smooth,
        smooth_window=delta_smooth_window,
        smooth_polyorder=delta_smooth_polyorder,
        raw_alpha=delta_raw_alpha,
    )
    return delta_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the persistent-window NLL (delta-NLL) plot, one column per scale."
    )
    parser.add_argument("--persistent-root", type=Path, default=DEFAULT_PERSISTENT_ROOT)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path base; the delta-NLL figure is written as <base>_delta_nll.png (a sibling PDF too).",
    )
    parser.add_argument(
        "--delta-smooth",
        choices=DELTA_SMOOTH_CHOICES,
        default=DELTA_SMOOTH,
        help=(
            "Trend line overlaid on the faint raw per-position delta-NLL curve "
            "(split delta plot only). 'savgol' (default) is shape-preserving; "
            "'median'/'gaussian' are alternatives; 'none' draws only the raw curve."
        ),
    )
    parser.add_argument(
        "--delta-smooth-window",
        type=int,
        default=DELTA_SMOOTH_WINDOW,
        help="Delta-NLL smoothing window in position units (forced odd for savgol).",
    )
    parser.add_argument(
        "--delta-smooth-polyorder",
        type=int,
        default=DELTA_SMOOTH_POLYORDER,
        help="Savitzky-Golay local-polynomial order for the delta-NLL trend (savgol only).",
    )
    parser.add_argument(
        "--delta-raw-alpha",
        type=float,
        default=DELTA_RAW_ALPHA,
        help="Opacity of the faint raw per-position curve under the smoothed delta-NLL trend.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    enable_latex()  # \textsc method names + Times, matching the paper
    scales = tuple(str(scale) for scale in args.scales)
    output = args.output or DEFAULT_OUTPUT_PATH
    delta_path = render_delta_nll_plot(
        persistent_root=args.persistent_root,
        output_path=output,
        scales=scales,
        window=int(args.window),
        delta_smooth=args.delta_smooth,
        delta_smooth_window=args.delta_smooth_window,
        delta_smooth_polyorder=args.delta_smooth_polyorder,
        delta_raw_alpha=args.delta_raw_alpha,
    )
    for path in _paired_image_paths(delta_path):
        print(path)


if __name__ == "__main__":
    main()
