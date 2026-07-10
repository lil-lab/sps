#!/usr/bin/env python3
"""Shared styling constants and series/plot helpers for the paper's two analysis
figures: the gradient-ratio panel (``plot_gradient_params.py``) and the
persistent-window delta-NLL panel (``plot_persistent_window_nll.py``). These used to
live in the (formerly combined) persistent-window plotter and were imported across;
they are collected here so neither figure script depends on the other.

Importing this module also pins ``MPLCONFIGDIR`` off NFS, selects the ``Agg`` backend,
and applies the serif (Times) rcParams both figures render with, so importing it
before ``matplotlib.pyplot`` is sufficient matplotlib setup for either plotter."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import plotting.mpl_cache  # noqa: F401  (pin MPLCONFIGDIR off NFS before matplotlib)

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
import numpy as np
from scipy.signal import savgol_filter

from plotting.methods import main_method_for_family_window
from plotting.style import COLORBLIND_METHOD_COLORS, LEGEND_TEXT_COLOR, PLOT_SPINE_COLOR

DEFAULT_WINDOW = 64

# ---- Shared style constants (used by both figures) --------------------------
PLOT_DPI = 180
LINEWIDTH = 3.8
STROKE_LINEWIDTH = 6.4
PANEL_TITLE_FONT_SIZE = 30
AXIS_LABEL_FONT_SIZE = 26
AXIS_TICK_LABEL_SIZE = 19
AXIS_TICK_LENGTH = 4.4
AXIS_TICK_WIDTH = 0.9
AXIS_TICK_COLOR = PLOT_SPINE_COLOR
Y_AXIS_LABEL_COORD_X = -0.18
PANEL_GRID_WSPACE = 0.15
SPLIT_FIGURE_HEIGHT = 5.0
SPLIT_PANEL_TOP = 0.93

# Method labels/colors used when plotting.methods has no entry for a (family, window).
METHOD_FALLBACK_LABELS = {
    ("full_attention", None): "Standard",
    ("sps", DEFAULT_WINDOW): "SPS",
    ("delayed_state", DEFAULT_WINDOW): "Delayed State",
}
METHOD_FALLBACK_COLORS = {
    ("full_attention", None): COLORBLIND_METHOD_COLORS["standard"],
    ("sps", DEFAULT_WINDOW): COLORBLIND_METHOD_COLORS["two_x_memory"],
    ("delayed_state", DEFAULT_WINDOW): COLORBLIND_METHOD_COLORS["delayed_state"],
}


# ---- Shared IO / series helpers ---------------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _paired_image_paths(output_path: Path) -> tuple[Path, Path]:
    base = output_path.with_suffix("") if output_path.suffix.lower() in {".png", ".pdf"} else output_path
    return base.with_suffix(".png"), base.with_suffix(".pdf")


def _stat_value(stats: dict[str, Any], key: str) -> float | None:
    value = stats.get(key)
    if value is None:
        return None
    return float(value)


def _array(values: list[float | None]) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)


def _method_label(family: str, window: int | None) -> str:
    method = main_method_for_family_window(family, window)
    if method is not None:
        return method.label
    return METHOD_FALLBACK_LABELS[(family, window)]


def _method_color(family: str, window: int | None) -> str:
    method = main_method_for_family_window(family, window)
    if method is not None:
        return COLORBLIND_METHOD_COLORS[method.key]
    return METHOD_FALLBACK_COLORS[(family, window)]


def _bin_average_series(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bin_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x = x[finite]
    y = y[finite]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    bin_indices = np.floor((x - 1.0) / float(bin_size)).astype(int)
    binned_x: list[float] = []
    binned_y: list[float] = []
    for bin_index in np.unique(bin_indices):
        mask = bin_indices == bin_index
        binned_x.append(float(np.mean(x[mask])))
        binned_y.append(float(np.mean(y[mask])))
    return np.asarray(binned_x, dtype=float), np.asarray(binned_y, dtype=float)


def _smooth_series(
    y: np.ndarray,
    *,
    method: str,
    window: int,
    polyorder: int,
    log: bool,
) -> np.ndarray:
    """Return a smoothed copy of the per-point series ``y`` for the overlaid trend
    line. ``y`` is assumed finite (callers drop non-finite points). On a log axis
    the mean-type smoothers (savgol/gaussian) run on ``log10(y)`` so spikes don't
    tug the trend; the rolling median is quantile-invariant, so
    ``median(log y) == log median(y)`` and it needs no transform. Returns ``y``
    unchanged for ``method="none"`` or a series too short to smooth (so short
    panels never raise)."""
    n = int(y.shape[0])
    if method == "none" or n < 3:
        return y

    if method == "median":
        radius = max(1, min(int(window), n) // 2)
        out = np.empty(n, dtype=float)
        for i in range(n):
            out[i] = float(np.median(y[max(0, i - radius):min(n, i + radius + 1)]))
        return out

    # savgol / gaussian: optionally smooth in log space (geometric).
    work = y
    if log:
        if not (y > 0.0).any():
            return y
        floor = float(np.min(y[y > 0.0]))
        work = np.log10(np.clip(y, floor, None))

    if method == "savgol":
        w = min(int(window), n if n % 2 == 1 else n - 1)
        if w % 2 == 0:
            w -= 1
        if w < 3 or w <= int(polyorder):
            return y
        smoothed = savgol_filter(work, window_length=w, polyorder=int(polyorder), mode="interp")
    elif method == "gaussian":
        sigma = max(1.0, float(window) / 6.0)  # window ~ +/-3 sigma
        radius = max(1, int(math.ceil(3.0 * sigma)))
        if n < 2 * radius + 1:
            return y
        weights = np.exp(-0.5 * (np.arange(-radius, radius + 1, dtype=float) / sigma) ** 2)
        smoothed = np.empty(n, dtype=float)
        for i in range(n):
            lo, hi = max(0, i - radius), min(n, i + radius + 1)
            wl = weights[lo - i + radius:hi - i + radius]
            smoothed[i] = float(np.sum(work[lo:hi] * wl) / np.sum(wl))
    else:
        return y

    return np.power(10.0, smoothed) if log else smoothed


def _rounded_upper(values: list[float], *, floor: float) -> float:
    if not values:
        return floor
    raw = max(max(values), floor)
    return float(np.ceil(raw * 20.0) / 20.0)


# ---- Shared plot helpers -----------------------------------------------------
def _plot_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    linestyle: object = "-",
    zorder: int = 5,
) -> Line2D:
    line, = ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        linewidth=LINEWIDTH,
        alpha=0.94,
        solid_capstyle="round",
        zorder=zorder,
    )
    line.set_path_effects(
        [
            path_effects.Stroke(linewidth=STROKE_LINEWIDTH, foreground="white", alpha=0.78),
            path_effects.Normal(),
        ]
    )
    return line


def _style_ticks(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        labelsize=AXIS_TICK_LABEL_SIZE,
        length=AXIS_TICK_LENGTH,
        width=AXIS_TICK_WIDTH,
        color=AXIS_TICK_COLOR,
        labelcolor=LEGEND_TEXT_COLOR,
    )
    for tick in (*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks()):
        tick.tick1line.set_color(AXIS_TICK_COLOR)
        tick.tick2line.set_color(AXIS_TICK_COLOR)
    for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick_label.set_fontweight("normal")
