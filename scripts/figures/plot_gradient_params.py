#!/usr/bin/env python3
"""Parameter-gradient future-offset plot: one panel per scale.

Plots the future/present gradient ratio vs relative future offset, one line per
method x stream, one column per scale, reading the PARAMETER-gradient analysis
output (``outputs/gradient_analysis_params/...``). The three module types
(attention/mlp/norm) are collapsed into the whole-parameter-vector aggregate that
the analysis emits as ``module_type="all"`` -- so there is no module-type
separation, just one line per method x stream. This produces the paper's Figure 5
gradient panel (``figures/gradient_params_fp.png``); it reuses the shared figure
styling from ``plotting.figure_common``.
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path
from typing import Any

# matplotlib caches usetex output under MPLCONFIGDIR (default ~/.cache/matplotlib),
# which is on NFS home here; TemporaryDirectory cleanup then races on .nfs* lock
# files and crashes usetex ("Directory not empty"). Pin the cache to local disk so
# LaTeX rendering works; setdefault preserves any user-provided MPLCONFIGDIR.
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), f"mplconfig-{os.getuid()}")
)

import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# Shared styling constants + series/plot helpers. Importing figure_common also
# applies the Agg backend + serif rcParams (gradient-figure-specific constants that
# used to be imported alongside these are now defined locally, just below).
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
    _bin_average_series,
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
from plotting.style import (
    LEGEND_TEXT_COLOR,
    PLOT_SPINE_COLOR,
    _apply_axes_background,
)

# Gradient-figure-specific styling (unused by the persistent-window delta-NLL
# plotter; kept local here rather than in the shared figure_common module).
GRADIENT_OFFSET_BIN_SIZE = 8
SLOT_LINESTYLES = {
    "input_token": "-",
    "predict_token": ":",
}
SLOT_LABELS = {
    "input_token": "Input Stream",
    "predict_token": "Prediction Stream",
}
TOP_X_MIN = 0.0
TOP_X_MAX = 512.0
SPLIT_PANEL_BOTTOM_GRADIENT = 0.42
SPLIT_X_AXIS_LABEL_Y_GRADIENT = 0.30
SPLIT_METHOD_LEGEND_ANCHOR_GRADIENT = (0.29, 0.05)
SPLIT_STREAM_LEGEND_ANCHOR_GRADIENT = (0.80, 0.05)

DEFAULT_SCALES = ("xs", "s", "m", "l", "xl")
DEFAULT_WINDOW = 64
# Fixed total figure width (inches): keeps the figure size constant as the number
# of scale panels grows from four (s,m,l,xl) to five (xs..xl); panels compress.
FIXED_FIGURE_WIDTH = 17.0

# The parameter-gradient run behind the paper's Figure-5 ratio panel: the scaled
# multi-position sweep (many sampled positions per document, ~8k samples per offset).
# Matches the default OUTPUT_ROOT of run_gradient_analysis_params.sh; a fresh re-run of
# that submitter writes {scale}_{sps,delayed_state}_w64_20b dirs here that this default
# reads directly. Reproduces figures/gradient_params_fp.png byte-for-byte.
DEFAULT_GRADIENT_ROOT = Path("outputs/gradient_analysis_params/multiposition")
DEFAULT_OUTPUT_PATH = Path("figures/gradient_params_fp.png")

# The plotted metric is the un-normalized future/present gradient ratio, computed
# per record as offset_norms[k]/present_norm during aggregation and read from the
# summary via two fields, selected by aggregate module:
#   "future_over_present"           -> stats over the per-position pooled records
#                                      (n = #positions). Used for the "all"
#                                      aggregate, where layers are already folded
#                                      into the whole-parameter vector (so there is
#                                      no layer axis to average and the per-position
#                                      distribution is the right one; this is what
#                                      makes --stat median meaningful).
#   "layer_mean_future_over_present" -> stats over per-layer means (n = #layers).
#                                      Used for the "all_layered" aggregate.
# The two fields have identical means for the "all" aggregate. The ratio is heavy-
# tailed, so the y-axis is log-scaled.
RATIO_Y_LABEL = "Future / Present Ratio"


def _field_for_aggregate(aggregate_module: str) -> str:
    """Pick the summary field: pooled-over-positions for the whole-vector "all"
    aggregate, per-layer-averaged for "all_layered"."""
    return (
        "layer_mean_future_over_present"
        if aggregate_module == "all_layered"
        else "future_over_present"
    )

# Central statistic to plot. The un-normalized future/present ratio is heavy-
# tailed (records with tiny present_norm inflate the mean), so median is far more
# readable; "mean" is the paper default.
STAT_KEYS = {"mean": "mean", "median": "p50"}
DEFAULT_STAT = "mean"

# The parameter analysis emits two whole-parameter aggregates as synthetic
# modules; we plot one of them -- one line per method x stream, no
# attention/mlp/norm separation:
#   "all"         -> norm of the whole parameter-gradient vector (pooled over
#                    every layer and module), no layer averaging.
#   "all_layered" -> per-layer ratio (pooled over modules within a layer),
#                    averaged over layers, matching the hidden-state analysis.
DEFAULT_AGGREGATE_MODULE = "all"
AGGREGATE_CHOICES = ("all", "all_layered")

# Smoothing of the per-offset curve. The raw curve is always drawn faint
# underneath; a smoothed trend line is overlaid on top in the same color and
# linestyle. future/present is heavy-tailed and log-scaled, so the mean-type
# smoothers (savgol/gaussian) run on log10(y) (geometric) to keep spikes from
# tugging the trend; the rolling median is quantile-invariant (log == linear).
SMOOTH_CHOICES = ("none", "savgol", "median", "gaussian")
DEFAULT_SMOOTH = "savgol"
DEFAULT_SMOOTH_WINDOW = 21      # odd, in offset units
DEFAULT_SMOOTH_POLYORDER = 2    # savgol local-polynomial order
DEFAULT_RAW_ALPHA = 0.30        # opacity of the faint raw underlay
RAW_LINEWIDTH_SCALE = 0.45      # raw line width relative to LINEWIDTH
RAW_ZORDER = 4                  # above grid/axvline (3), below smoothed (6-7)
Y_LOG_PAD_DECADES = 0.03        # tiny log-space margin past data min/max (minimal whitespace)

# --- Method-name typography ----------------------------------------------------
# The paper sets method names in small caps via \textsc. matplotlib's own text
# renderer cannot synthesize small caps with the Times/serif fonts here (the
# small-caps font variant resolves to the same upright face), so to match the
# paper we render the figure through LaTeX (usetex) and wrap method names in
# \textsc. Requires a LaTeX install + mathptmx (Times for LaTeX), both present on
# this cluster. --no-usetex falls back to the plain serif renderer.
USE_TEX_DEFAULT = True
LATEX_PREAMBLE = r"\usepackage{mathptmx}"
# y-axis label sits further left than the shared default under usetex: the LaTeX
# tick labels (10^{-1} with a true minus) are wider and the five narrow panels
# make the tick text occupy a larger axis fraction, so -0.18 overlapped the
# "Future / Present Ratio" label.
Y_AXIS_LABEL_COORD_X_FP = -0.30


def _sc(label: str, usetex: bool) -> str:
    r"""Render a method name in \textsc small caps when going through LaTeX."""
    return rf"\textsc{{{label}}}" if usetex else label


def _bf(label: str, usetex: bool) -> str:
    r"""Bold via \textbf under LaTeX (usetex ignores the fontweight="bold" kwarg)."""
    return rf"\textbf{{{label}}}" if usetex else label

# (family, window) -> run-dir suffix (see run_gradient_analysis_params.sh FAMILY_SPECS).
GRADIENT_METHODS: tuple[tuple[str, int | None], ...] = (
    ("full_attention", None),
    ("sps", DEFAULT_WINDOW),
    ("delayed_state", DEFAULT_WINDOW),
)
FAMILY_DIR_SUFFIX = {
    ("full_attention", None): "full_attention_20b",
    ("sps", DEFAULT_WINDOW): "sps_w64_20b",
    ("delayed_state", DEFAULT_WINDOW): "delayed_state_w64_20b",
}

LEGEND_FONT_SIZE = 20
LEGEND_TITLE_FONT_SIZE = 22


def _run_name(scale: str, family: str, window: int | None) -> str:
    return f"{scale}_{FAMILY_DIR_SUFFIX[(family, window)]}"


def _load_summary(
    gradient_root: Path, scale: str, family: str, window: int | None
) -> dict[str, Any] | None:
    path = (
        gradient_root
        / _run_name(scale, family, window)
        / "gradient_analysis_params_summary.json"
    )
    if not path.exists():
        print(f"WARNING: Missing param-gradient summary: {path}")
        return None
    return _read_json(path)


def _single_model(summary: dict[str, Any]) -> dict[str, Any] | None:
    models = [model for model in summary.get("models", []) if model.get("slots")]
    if not models:
        return None
    return models[0]


def _slots_by_id(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(slot.get("slot_id")): slot for slot in model.get("slots", [])}


def _slots_for_family(family: str) -> tuple[str, ...]:
    if family == "full_attention":
        return ("input_token",)
    return ("input_token", "predict_token")


def _module_for_type(slot: dict[str, Any], module_type: str) -> dict[str, Any] | None:
    for module in slot.get("modules", []):
        if str(module.get("module_type")) == module_type:
            return module
    return None


def _module_series(
    module: dict[str, Any], field: str, stat_key: str, *, bin_size: int
) -> tuple[np.ndarray, np.ndarray]:
    rows = module.get("offsets", [])
    x = np.asarray([float(row.get("offset", 0)) for row in rows], dtype=float)
    y = _array([_stat_value(row[field], stat_key) for row in rows])
    # bin_size <= 1 keeps every offset as its own point (no bucketing of nearby
    # offsets); the helper still sorts by offset and drops non-finite samples.
    return _bin_average_series(x, y, bin_size=max(1, int(bin_size)))


def _collect_y_bound(
    summaries: dict[tuple[str, str, int | None], dict[str, Any] | None],
    scales: tuple[str, ...],
    aggregate_module: str,
    field: str,
    stat_key: str,
    bin_size: int,
    log: bool = False,
) -> tuple[float, float]:
    values: list[float] = []
    for scale in scales:
        for family, window in GRADIENT_METHODS:
            summary = summaries.get((scale, family, window))
            if summary is None:
                continue
            model = _single_model(summary)
            if model is None:
                continue
            slots = _slots_by_id(model)
            for slot_id in _slots_for_family(family):
                slot = slots.get(slot_id)
                if slot is None:
                    continue
                module = _module_for_type(slot, aggregate_module)
                if module is None:
                    continue
                _x, y = _module_series(module, field, stat_key, bin_size=bin_size)
                values.extend(y[np.isfinite(y)].tolist())
    if log:
        positive = [v for v in values if v > 0.0]
        if not positive:
            return (1e-3, 1.0)
        lo = 10.0 ** (math.log10(min(positive)) - Y_LOG_PAD_DECADES)
        hi = 10.0 ** (math.log10(max(positive)) + Y_LOG_PAD_DECADES)
        return (lo, hi)
    return (0.0, _rounded_upper(values, floor=0.05))


def render_plot(
    *,
    gradient_root: Path,
    output_path: Path,
    scales: tuple[str, ...],
    window: int,
    aggregate_module: str = DEFAULT_AGGREGATE_MODULE,
    stat: str = DEFAULT_STAT,
    bin_size: int = 1,
    smooth: str = DEFAULT_SMOOTH,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    smooth_polyorder: int = DEFAULT_SMOOTH_POLYORDER,
    raw_alpha: float = DEFAULT_RAW_ALPHA,
    usetex: bool = USE_TEX_DEFAULT,
) -> None:
    field = _field_for_aggregate(aggregate_module)
    y_label = RATIO_Y_LABEL
    log_y = True
    stat_key = STAT_KEYS[stat]
    methods = tuple(
        (family, None if family == "full_attention" else int(window))
        for family, _w in GRADIENT_METHODS
    )
    summaries = {
        (scale, family, w): _load_summary(gradient_root, scale, family, w)
        for scale in scales
        for family, w in methods
    }
    y_bounds = _collect_y_bound(
        summaries, scales, aggregate_module, field, stat_key, bin_size=bin_size, log=log_y
    )

    n_cols = len(scales)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(FIXED_FIGURE_WIDTH, SPLIT_FIGURE_HEIGHT),
        sharey=True,
        sharex=True,
        squeeze=False,
    )

    for col, scale in enumerate(scales):
        ax = axes[0, col]
        _apply_axes_background(
            ax,
            x_min=TOP_X_MIN,
            x_max=TOP_X_MAX,
            y_min=y_bounds[0],
            y_max=y_bounds[1],
        )
        ax.set_xlim(TOP_X_MIN, TOP_X_MAX)
        ax.set_ylim(*y_bounds)
        if log_y:
            ax.set_yscale("log")
        ax.axvline(
            float(window),
            color=PLOT_SPINE_COLOR,
            linewidth=1.4,
            linestyle=(0, (5, 4)),
            alpha=0.85,
            zorder=3,
        )
        for family, w in methods:
            summary = summaries.get((scale, family, w))
            if summary is None:
                continue
            model = _single_model(summary)
            if model is None:
                continue
            slots = _slots_by_id(model)
            color = _method_color(family, w)
            for slot_id in _slots_for_family(family):
                slot = slots.get(slot_id)
                if slot is None:
                    continue
                module = _module_for_type(slot, aggregate_module)
                if module is None:
                    continue
                x, y = _module_series(module, field, stat_key, bin_size=bin_size)
                top_zorder = 7 if family == "full_attention" else 6
                if smooth != "none":
                    # Faint raw curve underneath (no white halo), smoothed trend
                    # line on top -- same color/linestyle, reads as one series.
                    ax.plot(
                        x,
                        y,
                        color=color,
                        linestyle=SLOT_LINESTYLES[slot_id],
                        linewidth=LINEWIDTH * RAW_LINEWIDTH_SCALE,
                        alpha=raw_alpha,
                        solid_capstyle="round",
                        zorder=RAW_ZORDER,
                    )
                    y = _smooth_series(
                        y,
                        method=smooth,
                        window=smooth_window,
                        polyorder=smooth_polyorder,
                        log=log_y,
                    )
                _plot_line(
                    ax,
                    x,
                    y,
                    color=color,
                    linestyle=SLOT_LINESTYLES[slot_id],
                    zorder=top_zorder,
                )
        ax.set_title(
            _bf(scale.upper(), usetex),
            fontsize=PANEL_TITLE_FONT_SIZE,
            fontweight="bold",
            color=LEGEND_TEXT_COLOR,
            pad=9,
        )
        if col == 0:
            ax.set_ylabel(
                _bf(y_label, usetex),
                fontsize=AXIS_LABEL_FONT_SIZE,
                fontweight="bold",
                color=LEGEND_TEXT_COLOR,
            )
            ax.yaxis.set_label_coords(
                Y_AXIS_LABEL_COORD_X_FP if usetex else Y_AXIS_LABEL_COORD_X, 0.5
            )
        ax.set_xticks([0, 128, 256, 384, 512])
        _style_ticks(ax)
        ax.minorticks_off()

    fig.text(
        0.5,
        SPLIT_X_AXIS_LABEL_Y_GRADIENT,
        _bf("Relative Future Offset", usetex),
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
    )

    method_handles = [
        Line2D([0], [0], color=_method_color(family, w), linewidth=LINEWIDTH, label=_sc(_method_label(family, w), usetex))
        for family, w in methods
    ]
    slot_handles = [
        Line2D([0], [0], color="#4d453b", linestyle=SLOT_LINESTYLES[slot_id], linewidth=LINEWIDTH, label=label)
        for slot_id, label in SLOT_LABELS.items()
    ]
    method_legend = fig.legend(
        handles=method_handles,
        loc="lower center",
        ncol=len(method_handles),
        frameon=False,
        bbox_to_anchor=SPLIT_METHOD_LEGEND_ANCHOR_GRADIENT,
        title=_bf("Method", usetex),
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
        handlelength=2.35,
        columnspacing=1.05,
        handletextpad=0.5,
    )
    stream_legend = fig.legend(
        handles=slot_handles,
        loc="lower center",
        ncol=len(slot_handles),
        frameon=False,
        bbox_to_anchor=SPLIT_STREAM_LEGEND_ANCHOR_GRADIENT,
        title=_bf("Gradient Stream", usetex),
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
        handlelength=2.35,
        columnspacing=1.05,
        handletextpad=0.5,
    )
    for legend in (method_legend, stream_legend):
        legend.get_title().set_color(LEGEND_TEXT_COLOR)
        legend.get_title().set_fontweight("bold")
        for text in legend.get_texts():
            text.set_color(LEGEND_TEXT_COLOR)

    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=SPLIT_PANEL_BOTTOM_GRADIENT,
        top=SPLIT_PANEL_TOP,
        wspace=PANEL_GRID_WSPACE,
    )
    png_path, pdf_path = _paired_image_paths(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-root", type=Path, default=DEFAULT_GRADIENT_ROOT)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument(
        "--bin-size",
        type=int,
        default=1,
        help=(
            "Bucket width (in offsets) for averaging nearby points. 1 = no "
            f"bucketing, plot every offset (default); the legacy value was "
            f"{GRADIENT_OFFSET_BIN_SIZE}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output PNG path (a sibling PDF is also written). Defaults to "
            "figures/gradient_params_fp.png (with a _<stat> suffix for --stat median)."
        ),
    )
    parser.add_argument(
        "--stat",
        choices=tuple(STAT_KEYS),
        default=DEFAULT_STAT,
        help=(
            "Central statistic per offset. 'mean' is the paper default; 'median' "
            "is robust to the heavy tail of future/present."
        ),
    )
    parser.add_argument(
        "--aggregate-module",
        choices=AGGREGATE_CHOICES,
        default=DEFAULT_AGGREGATE_MODULE,
        help=(
            "Which whole-parameter aggregate to plot: 'all' = whole-vector ratio "
            "(no layer averaging); 'all_layered' = per-layer ratio averaged over "
            "layers (matches the hidden-state analysis)."
        ),
    )
    parser.add_argument(
        "--smooth",
        choices=SMOOTH_CHOICES,
        default=DEFAULT_SMOOTH,
        help=(
            "Overlaid trend line on top of the faint raw curve. 'savgol' "
            "(Savitzky-Golay on log10(y), default) is shape-preserving; 'median' "
            "is a robust rolling median; 'gaussian' is a Gaussian-weighted "
            "average on log10(y); 'none' draws only the raw curve (legacy look)."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help="Smoothing window in offset units (forced odd for savgol).",
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=DEFAULT_SMOOTH_POLYORDER,
        help="Savitzky-Golay local-polynomial order (savgol only).",
    )
    parser.add_argument(
        "--raw-alpha",
        type=float,
        default=DEFAULT_RAW_ALPHA,
        help="Opacity of the faint raw curve drawn under the smoothed trend.",
    )
    parser.add_argument(
        "--usetex",
        dest="usetex",
        action="store_true",
        default=USE_TEX_DEFAULT,
        help=(
            "Render through LaTeX so method names use \\textsc small caps, "
            "matching the paper (default; needs a LaTeX install + mathptmx)."
        ),
    )
    parser.add_argument(
        "--no-usetex",
        dest="usetex",
        action="store_false",
        help="Disable LaTeX rendering; plain serif labels (legacy look).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.usetex:
        # Match the paper: LaTeX text engine + Times (mathptmx) so \textsc renders
        # real small caps. Set before any figure is created.
        plt.rcParams.update(
            {"text.usetex": True, "text.latex.preamble": LATEX_PREAMBLE}
        )
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH
        if args.stat != "mean":
            output_path = output_path.with_name(
                f"{output_path.stem}_{args.stat}{output_path.suffix}"
            )
    render_plot(
        gradient_root=args.gradient_root,
        output_path=output_path,
        scales=tuple(str(s) for s in args.scales),
        window=int(args.window),
        aggregate_module=args.aggregate_module,
        stat=args.stat,
        bin_size=args.bin_size,
        smooth=args.smooth,
        smooth_window=args.smooth_window,
        smooth_polyorder=args.smooth_polyorder,
        raw_alpha=args.raw_alpha,
        usetex=args.usetex,
    )


if __name__ == "__main__":
    main()
