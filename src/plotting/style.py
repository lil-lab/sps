"""Shared matplotlib styling constants and axis helpers for the figure scripts.

This module is intentionally free of import-time side effects: it does NOT call
``matplotlib.use(...)`` or ``matplotlib.rcParams.update(...)``. Each figure script
selects its own backend and fonts before importing from ``plotting``.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# --- Metric keys ---------------------------------------------------------------
TRAINING_METRIC_KEY = "val/token_nll"
TRAINING_METRIC_LABEL = "Validation NLL"

# --- Default scale / window / family selections --------------------------------
DEFAULT_SCALES = ("xs", "s", "m", "l", "xl")
DEFAULT_WINDOWS = (64, 4096)
DEFAULT_FAMILIES = ("delayed_state", "sps")

# --- Method colors -------------------------------------------------------------
COLORBLIND_METHOD_COLORS = {
    "standard": "#000000",
    "two_x_memory": "#0072B2",
    "delayed_state": "#D55E00",
    "sps": "#009E73",
}
FULL_ATTENTION_COLOR = "#1a1a1a"

# --- Per-family window color ramps ---------------------------------------------
SLIDING_WINDOW_COLORS = {
    4: "#facc15",
    16: "#f59e0b",
    64: "#ea580c",
    256: "#c2410c",
}
DELAYED_STATE_VALIDATION_WINDOW_COLORS = {
    0: "#34d399",
    1: "#bbf7d0",
    2: "#86efac",
    4: "#4ade80",
    16: "#22c55e",
    32: "#18b85a",
    64: "#16a34a",
    128: "#158f42",
    256: "#157f3c",
    512: "#15803d",
    4096: "#166534",
}
REVERSE_SPS_VALIDATION_WINDOW_COLORS = {
    0: "#38bdf8",
    1: "#bfdbfe",
    2: "#93c5fd",
    4: "#60a5fa",
    16: "#2563eb",
    32: "#1f5bd8",
    64: "#1d4ed8",
    128: "#1f47c4",
    256: "#1e43b4",
    512: "#1e40af",
    4096: "#1e3a8a",
}
SPS_VALIDATION_WINDOW_COLORS = {
    0: "#f87171",
    1: "#fecdd3",
    2: "#fda4af",
    4: "#fb7185",
    16: "#f43f5e",
    32: "#e11d48",
    64: "#be123c",
    128: "#9f1239",
    256: "#881337",
    512: "#7f1d1d",
    4096: "#450a0a",
}

# --- Background / grid / spine / legend palette --------------------------------
PLOT_BACKGROUND_BASE = "#f8f4ec"
PLOT_BACKGROUND_GRADIENT_LOW = "#fffdf8"
PLOT_BACKGROUND_GRADIENT_HIGH = "#f2eadb"
PLOT_GRID_COLOR = "#d8d0c3"
PLOT_SPINE_COLOR = "#a69d8e"
LEGEND_HEADER_COLOR = "#4d453b"
LEGEND_TEXT_COLOR = "#26221c"

# --- Line / axis weights -------------------------------------------------------
FULL_ATTENTION_LINEWIDTH = 4.8
FULL_ATTENTION_STROKE_LINEWIDTH = 7.0
WINDOWED_SERIES_LINEWIDTH = 4.4
WINDOWED_SERIES_STROKE_LINEWIDTH = 6.8
PANEL_TITLE_FONT_WEIGHT = "bold"
AXIS_LABEL_FONT_WEIGHT = "bold"
AXIS_TICK_LENGTH = 10
AXIS_TICK_WIDTH = 1.9


# --- LaTeX text engine / small-caps method names -------------------------------
# To match the paper's \textsc method names, figures render through LaTeX
# (text.usetex) with Times via mathptmx. enable_latex() must be called explicitly
# (never at import time) so that importing this module -- or a plotter that
# imports it -- does not silently switch other figures to usetex. Wrap method
# names with sc() and bold strings with bf(); both no-op when usetex is off, so
# the non-LaTeX look is unchanged.
LATEX_PREAMBLE = r"\usepackage{mathptmx}"


def enable_latex(preamble: str = LATEX_PREAMBLE) -> None:
    """Switch matplotlib to the LaTeX text engine with Times (mathptmx).

    Requires a LaTeX install + mathptmx, and ``MPLCONFIGDIR`` pinned off NFS (see
    ``plotting.mpl_cache``) or usetex's cache cleanup races on NFS lock files.
    """
    plt.rcParams.update({"text.usetex": True, "text.latex.preamble": preamble})


def sc(label: str) -> str:
    r"""Wrap a method name in \textsc small caps when rendering through LaTeX."""
    return rf"\textsc{{{label}}}" if plt.rcParams.get("text.usetex") else label


def bf(label: str) -> str:
    r"""Bold via \textbf under usetex (which ignores the fontweight kwarg)."""
    return rf"\textbf{{{label}}}" if plt.rcParams.get("text.usetex") else label


def _billions_formatter(value: float, _pos: int) -> str:
    return f"{value / 1_000_000_000:g}B"


def _apply_axes_background(
    ax: plt.Axes,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    ax.set_facecolor(PLOT_BACKGROUND_BASE)
    gradient = np.linspace(0.0, 1.0, 512).reshape(512, 1)
    cmap = LinearSegmentedColormap.from_list(
        "paper_tint",
        [PLOT_BACKGROUND_GRADIENT_LOW, PLOT_BACKGROUND_GRADIENT_HIGH],
    )
    ax.imshow(
        gradient,
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        alpha=0.6,
        interpolation="bicubic",
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.grid(color=PLOT_GRID_COLOR, alpha=0.4, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(PLOT_SPINE_COLOR)
        spine.set_linewidth(1.0)


def _style_legend_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
