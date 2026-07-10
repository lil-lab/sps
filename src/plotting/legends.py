"""Legend dataclasses and the multi-group ("M-classic") legend renderer."""

from __future__ import annotations

import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotting.style import LEGEND_HEADER_COLOR, LEGEND_TEXT_COLOR, _style_legend_axis


STATE_PREDICTION_COUPLING_LABEL = "State-Prediction Coupling"
STATE_PREDICTION_DECOUPLING_LABEL = "State-Prediction Decoupling"
M_CLASSIC_LEGEND_GROUP_WIDTHS = {
    "Standard": 2.8,
    "Standard Attention": 4.9,
    "Standard (More Tokens)": 6.1,
    "2x Memory": 3.6,
    "Delayed State": 4.4,
    "SPS": 2.0,
    "2x Faster Standard": 5.5,
    "Tokens to Match Pre-Decay Standard": 9.7,
    "Sliding window": 4.2,
    STATE_PREDICTION_COUPLING_LABEL: 9.5,
    STATE_PREDICTION_DECOUPLING_LABEL: 10.5,
    "Window Size": 7.4,
}


@dataclass(frozen=True)
class LegendEntry:
    line_key: str
    legend_label: str


@dataclass(frozen=True)
class LegendGroup:
    title: str
    entries: list[LegendEntry]
    ncol: int


# \textsc small caps render wider than the same text in mixed case, so the
# hand-tuned plain-text slot widths are too narrow once a group title is wrapped
# in \textsc{...} (usetex). Strip the wrapper to find the base width and scale up.
SMALL_CAPS_WIDTH_SCALE = 1.4
_TEXTSC_RE = re.compile(r"^\\textsc\{(?P<inner>.*)\}$")


def _group_slot_width(title: str) -> float:
    if title in M_CLASSIC_LEGEND_GROUP_WIDTHS:
        return M_CLASSIC_LEGEND_GROUP_WIDTHS[title]
    match = _TEXTSC_RE.match(title)
    if match is not None:
        inner = match.group("inner")
        if inner in M_CLASSIC_LEGEND_GROUP_WIDTHS:
            return M_CLASSIC_LEGEND_GROUP_WIDTHS[inner] * SMALL_CAPS_WIDTH_SCALE
    return 2.0


def _m_classic_legend_group_widths(groups: list[LegendGroup]) -> list[float]:
    return [_group_slot_width(group.title) for group in groups]


def _m_classic_legend_group_positions(groups: list[LegendGroup]) -> list[tuple[LegendGroup, float]]:
    if [group.title for group in groups] == [
        "Standard Attention",
        STATE_PREDICTION_COUPLING_LABEL,
        STATE_PREDICTION_DECOUPLING_LABEL,
    ]:
        return list(zip(groups, [0.035, 0.30, 0.68], strict=True))

    widths = _m_classic_legend_group_widths(groups)
    gap_units = 0.75
    total_units = sum(widths) + gap_units * (len(groups) + 1)
    cursor = gap_units
    positions: list[tuple[LegendGroup, float]] = []
    for group, width in zip(groups, widths, strict=True):
        positions.append((group, cursor / total_units))
        cursor += width + gap_units
    return positions


def _render_m_classic_legend(
    legend_ax: plt.Axes,
    legend_groups: list[LegendGroup],
    plotted_lines: dict[str, Line2D],
    *,
    header_fontsize: int = 24,
    header_fontweight: str = "semibold",
    entry_fontsize: int = 22,
    entry_fontweight: str = "normal",
) -> None:
    _style_legend_axis(legend_ax)
    for group, anchor_x in _m_classic_legend_group_positions(legend_groups):
        legend_ax.text(
            anchor_x,
            0.84,
            group.title,
            fontsize=header_fontsize,
            fontweight=header_fontweight,
            color=LEGEND_HEADER_COLOR,
            va="top",
            ha="left",
        )
        handles = [plotted_lines[entry.line_key] for entry in group.entries]
        legend = legend_ax.legend(
            handles,
            [entry.legend_label for entry in group.entries],
            loc="upper left",
            bbox_to_anchor=(anchor_x, 0.5),
            frameon=False,
            fontsize=entry_fontsize,
            ncol=min(group.ncol, len(group.entries)),
            columnspacing=1.6,
            handlelength=2.6,
            handletextpad=0.7,
            labelspacing=0.95,
            borderaxespad=0.0,
        )
        for text in legend.get_texts():
            text.set_color(LEGEND_TEXT_COLOR)
            text.set_fontweight(entry_fontweight)
        legend_ax.add_artist(legend)
