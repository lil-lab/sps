#!/usr/bin/env python3
"""Plot validation NLL histories across model scales from Weights & Biases."""

from __future__ import annotations

import plotting.mpl_cache  # noqa: F401  (pin MPLCONFIGDIR off NFS before matplotlib)

import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime
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
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, LinearLocator, MaxNLocator

from plotting.methods import (
    MainMethodSpec,
    main_method_for_family_window,
    main_method_sort_index,
    main_method_specs_for_selection,
)
from plotting.style import (
    AXIS_LABEL_FONT_WEIGHT,
    AXIS_TICK_LENGTH,
    AXIS_TICK_WIDTH,
    COLORBLIND_METHOD_COLORS,
    DEFAULT_FAMILIES,
    DEFAULT_SCALES,
    DEFAULT_WINDOWS,
    FULL_ATTENTION_LINEWIDTH,
    FULL_ATTENTION_STROKE_LINEWIDTH,
    LEGEND_TEXT_COLOR,
    PANEL_TITLE_FONT_WEIGHT,
    PLOT_SPINE_COLOR,
    TRAINING_METRIC_KEY,
    TRAINING_METRIC_LABEL,
    WINDOWED_SERIES_LINEWIDTH,
    WINDOWED_SERIES_STROKE_LINEWIDTH,
    _apply_axes_background,
    _billions_formatter,
    bf,
    enable_latex,
    sc,
)
from plotting.wandb_runs import (
    ResolvedRun,
    RunSpec,
    USABLE_RUN_STATES,
    _latest_run,
    _parse_created_at,
    _repo_root,
    _warn,
    filter_rewound_history_segments,
    resolve_runs,
    windows_for_scale,
)
from plotting.legends import (
    LegendEntry,
    LegendGroup,
    _render_m_classic_legend,
)


Y_PADDING_FRACTION = 0.12
MIN_Y_SPAN = 0.02
HISTORY_X_MIN_FRACTION = 0.10
Y_BOUND_TAIL_FRACTION = 0.35
X_PADDING_FACTOR = 1.02
Y_LIMIT_ROUNDING_INTERVAL = 0.01
HISTORY_TOKEN_X_MAJOR_TICKS = (
    0.0,
    5_000_000_000.0,
    10_000_000_000.0,
    15_000_000_000.0,
    20_000_000_000.0,
)
HISTORY_X_MAJOR_TICK_COUNT = 5
HISTORY_Y_MAJOR_TICK_COUNT = 5
GPU_HOURS_X_MAJOR_TICK_MAX_INTERVALS = 5
GPU_HOURS_X_MAJOR_TICK_STEPS = (1, 2, 2.5, 5, 10)
PANEL_GRID_HSPACE = 0.30
HISTORY_PANEL_TITLE_FONT_SIZE = 50
HISTORY_PENDING_PANEL_FONT_SIZE = 40
HISTORY_AXIS_LABEL_FONT_SIZE = 46
HISTORY_AXIS_TICK_LABEL_SIZE = 32
HISTORY_LEGEND_HEADER_FONT_SIZE = 31
HISTORY_LEGEND_ENTRY_FONT_SIZE = 28
HISTORY_LINEWIDTH_SCALE = 1.15
HISTORY_STROKE_LINEWIDTH_SCALE = 1.05
HISTORY_STROKE_ALPHA = 0.28
HISTORY_SIZE_SUFFIX_BY_SCALE = {
    "xs": "20b",
    "s": "20b",
    "m": "20b",
    "l": "20b",
    "xl": "20b",
}
HISTORY_PRE_DECAY_TOKEN_CUTOFF_BY_SIZE = {
    "10b": 9_000_000_000,
    "20b": 18_000_000_000,
    "40b": 36_000_000_000,
    "100b": 90_000_000_000,
}
STANDARD_2X_TOKENS_KEY = "standard_2x_tokens"
STANDARD_2X_TOKENS_FAMILY = "standard_2x_tokens"
STANDARD_2X_TOKENS_LABEL = "Standard (More Tokens)"
STANDARD_2X_TOKENS_SIZE_SUFFIX_BY_SCALE = {
    "xs": "40b",
    "s": "40b",
    "m": "40b",
    "l": "40b",
    "xl": "100b",
}
# Scales whose "Standard (2x Tokens)" line is stopped once it reaches the SPS run's
# final (pre-decay) validation NLL — used for the xl 100b run so the line ends where
# standard matches SPS instead of running off the chart.
STANDARD_2X_TOKENS_STOP_AT_SPS_FINAL_SCALES = frozenset({"xl"})
STANDARD_2X_TOKENS_STOP_AT_SPS_METHOD_KEY = "sps"
STANDARD_2X_TOKENS_COLOR = "#666666"
STANDARD_2X_TOKENS_LINESTYLE = "-"
# Also accept "crashed": the xl 100b more-tokens run crashed (past the displayed
# range), but its logged history covers the plotted window, so we still want its
# curve. _latest_run_with_states prefers finished/running, so a crashed run is only
# used when it is the sole option.
STANDARD_2X_TOKENS_USABLE_RUN_STATES = frozenset({*USABLE_RUN_STATES, "running", "crashed"})
STANDARD_2X_TOKENS_HANDOFF_GUIDE_COLOR = "#8a8a8a"
STANDARD_2X_TOKENS_HANDOFF_GUIDE_LINESTYLE = (0, (1, 4))
STANDARD_2X_TOKENS_HANDOFF_GUIDE_LINEWIDTH = 2.0
STANDARD_2X_TOKENS_HANDOFF_GUIDE_ALPHA = 0.70
STANDARD_2X_TOKENS_HANDOFF_GUIDE_ZORDER = 4
PRE_DECAY_MATCH_GUIDE_KEY = "pre_decay_match_guide"
PRE_DECAY_MATCH_GUIDE_KEY_PREFIX = "pre_decay_match_guide"
PRE_DECAY_MATCH_GUIDE_LABEL = "Tokens to Match Pre-Decay Standard"
PRE_DECAY_MATCH_GUIDE_LEGEND_COLOR = "#8a8a8a"
PRE_DECAY_MATCH_GUIDE_LINESTYLE = (0, (1, 4))
PRE_DECAY_MATCH_GUIDE_LINEWIDTH = 2.6
PRE_DECAY_MATCH_GUIDE_ALPHA = 0.78
PRE_DECAY_MATCH_GUIDE_ZORDER = 5
HISTORY_REAL_LINE_ALPHA = 0.50
STANDARD_HISTORY_ZORDER = 9
WINDOWED_HISTORY_ZORDER = 8
DELAYED_STATE_HISTORY_ZORDER = 10
COMBINED_HISTORY_FIGSIZE = (24, 17.5)
COMBINED_GRID_HEIGHT_RATIOS = (8.5, 1.0)
COMBINED_GRID_HSPACE = 0.04
COMBINED_PANEL_GRID_HSPACE = 0.32
COMBINED_PANEL_GRID_WSPACE = 0.30
COMBINED_PANEL_TITLE_FONT_SIZE = 38
COMBINED_PENDING_PANEL_FONT_SIZE = 30
COMBINED_AXIS_LABEL_FONT_SIZE = 34
COMBINED_AXIS_TICK_LABEL_SIZE = 22
COMBINED_LEGEND_HEADER_FONT_SIZE = 27
COMBINED_LEGEND_ENTRY_FONT_SIZE = 25
COMBINED_LEGEND_FONT_WEIGHT = "normal"
COMBINED_LINEWIDTH_MULTIPLIER = 1.15
COMBINED_REAL_LINE_ALPHA = 0.90
COMBINED_STROKE_ALPHA = 0.38
COMBINED_SMOOTH_SIGMA_POINTS = 1.2
COMBINED_TOKEN_X_MIN = 7_000_000_000.0
COMBINED_SUBPLOTS_BOTTOM = 0.290
COMBINED_TRAINING_TOKENS_Y_SHIFT = -0.005
COMBINED_GPU_HOURS_X_LABEL_OFFSET = 0.045
COMBINED_LEGEND_Y_SHIFT = -0.060
COMBINED_Y_AXIS_LABEL = "Validation NLL"
X_AXIS_TOKENS = "tokens"
X_AXIS_GPU_HOURS = "gpu-hours"
X_AXIS_COMBINED = "combined"
X_AXIS_CHOICES = (X_AXIS_TOKENS, X_AXIS_GPU_HOURS, X_AXIS_COMBINED)
DEFAULT_GPU_HOURS_REFERENCE_INTERVAL = 1
GPU_HOURS_PER_TOKEN_PRECISION = 12
TIMING_HISTORY_KEYS = ["_step", "_timestamp", "_runtime", "tokens_seen", TRAINING_METRIC_KEY]
CALIBRATION_MODE_SINGLE_RATE = "single-rate"
CALIBRATION_MODE_STARTED_AT = "started-at"
CALIBRATION_MODE_STANDARD_20B_REFERENCE = "standard-20b-reference"


@dataclass(frozen=True)
class HistoryResolvedRun:
    spec: RunSpec
    run_id: str
    created_at: str
    tokens_seen: list[int]
    metric_values: list[float]
    timestamps: list[float | None]
    runtimes: list[float | None]
    gpu_count: int | None
    gpu_hours: list[float]
    calibration_start_index: int
    calibration_end_index: int
    calibration_start_tokens: int
    calibration_end_tokens: int
    calibration_seconds: float
    gpu_hours_per_token: float
    calibration_mode: str = CALIBRATION_MODE_SINGLE_RATE


def _default_output_path(x_axis: str = X_AXIS_TOKENS) -> Path:
    if x_axis == X_AXIS_COMBINED:
        return _repo_root() / "figures" / "history_combined.png"
    if x_axis == X_AXIS_GPU_HOURS:
        return _repo_root() / "figures" / "history_given_gpu.png"
    return _repo_root() / "figures" / "history.png"


def _paired_image_paths(output_path: Path) -> tuple[Path, Path]:
    base = output_path.with_suffix("") if output_path.suffix.lower() in {".png", ".pdf"} else output_path
    return base.with_suffix(".png"), base.with_suffix(".pdf")


HistoryRun = ResolvedRun | HistoryResolvedRun


def history_size_suffix_for_scale(scale: str) -> str:
    try:
        return HISTORY_SIZE_SUFFIX_BY_SCALE[scale]
    except KeyError as exc:
        raise ValueError(f"Unknown scale {scale!r}; add it to HISTORY_SIZE_SUFFIX_BY_SCALE") from exc


def standard_2x_tokens_size_suffix_for_scale(scale: str) -> str:
    # The "Standard (2x Tokens)" reference is the longer-token full-attention run for
    # each scale. xl uses the in-progress 100b run; the rest use 40b (2x of the 20b run).
    return STANDARD_2X_TOKENS_SIZE_SUFFIX_BY_SCALE.get(scale, "40b")


def pre_decay_token_cutoff_for_scale(scale: str) -> int:
    size_suffix = history_size_suffix_for_scale(scale)
    try:
        return HISTORY_PRE_DECAY_TOKEN_CUTOFF_BY_SIZE[size_suffix]
    except KeyError as exc:
        raise ValueError(
            f"No pre-decay token cutoff configured for scale={scale!r}, size={size_suffix!r}"
        ) from exc


def _is_standard_2x_tokens_spec(spec: RunSpec) -> bool:
    return spec.family == STANDARD_2X_TOKENS_FAMILY and spec.window is None


def _is_standard_2x_tokens_run(run: HistoryRun) -> bool:
    return _is_standard_2x_tokens_spec(run.spec)


def history_size_suffix_for_run(run: HistoryRun) -> str:
    if _is_standard_2x_tokens_run(run):
        return standard_2x_tokens_size_suffix_for_scale(run.spec.scale)
    return history_size_suffix_for_scale(run.spec.scale)


def pre_decay_token_cutoff_for_run(run: HistoryRun) -> int:
    size_suffix = history_size_suffix_for_run(run)
    try:
        return HISTORY_PRE_DECAY_TOKEN_CUTOFF_BY_SIZE[size_suffix]
    except KeyError as exc:
        raise ValueError(
            f"No pre-decay token cutoff configured for displayName={run.spec.display_name!r}, "
            f"size={size_suffix!r}"
        ) from exc


def run_specs_for_scale(
    scale: str,
    *,
    windows: Iterable[int] | None = None,
    families: Iterable[str] = DEFAULT_FAMILIES,
) -> list[RunSpec]:
    size_suffix = history_size_suffix_for_scale(scale)
    method_specs = main_method_specs_for_selection(families=families, windows=windows)
    if not method_specs:
        raise ValueError("At least one main method must be selected")
    return [
        RunSpec(
            scale=scale,
            family=method.family,
            window=method.window,
            display_name=(
                f"{scale}_full_attention_{size_suffix}"
                if method.family == "full_attention"
                else f"{scale}_{method.family}_w{int(method.window)}_{size_suffix}"
            ),
        )
        for method in method_specs
    ]


def standard_2x_tokens_run_spec_for_scale(scale: str) -> RunSpec:
    return RunSpec(
        scale=scale,
        family=STANDARD_2X_TOKENS_FAMILY,
        window=None,
        display_name=f"{scale}_full_attention_{standard_2x_tokens_size_suffix_for_scale(scale)}",
    )


def standard_2x_tokens_run_specs_for_scales(scales: Iterable[str]) -> list[RunSpec]:
    return [standard_2x_tokens_run_spec_for_scale(scale) for scale in scales]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _normalize_history_with_timing(
    rows: Iterable[dict[str, object]],
) -> tuple[list[int], list[float], list[float | None], list[float | None]]:
    normalized: list[tuple[int, int, tuple[float, float | None, float | None]]] = []
    for row in rows:
        tokens_seen = row.get("tokens_seen")
        metric_value = row.get(TRAINING_METRIC_KEY)
        if tokens_seen is None or metric_value is None:
            continue
        step_raw = row.get("_step", 0)
        normalized.append(
            (
                int(tokens_seen),
                int(step_raw or 0),
                (
                    float(metric_value),
                    _optional_float(row.get("_timestamp")),
                    _optional_float(row.get("_runtime")),
                ),
            )
        )

    filtered = filter_rewound_history_segments(normalized)
    ordered_tokens: list[int] = []
    metric_values: list[float] = []
    timestamps: list[float | None] = []
    runtimes: list[float | None] = []
    for tokens_seen, _step, payload in filtered:
        metric_value, timestamp, runtime = payload
        ordered_tokens.append(int(tokens_seen))
        metric_values.append(metric_value)
        timestamps.append(timestamp)
        runtimes.append(runtime)
    return ordered_tokens, metric_values, timestamps, runtimes


def _metadata_gpu_count(run, *, default_gpu_count: int | None = None) -> int | None:
    metadata = getattr(run, "metadata", None)
    raw_value = metadata.get("gpu_count") if isinstance(metadata, dict) else None
    if raw_value is None:
        raw_value = default_gpu_count
    if raw_value is None:
        return None
    try:
        gpu_count = int(raw_value)
    except (TypeError, ValueError):
        return None
    if gpu_count <= 0:
        return None
    return gpu_count


def _parse_iso_timestamp(value: object) -> float | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _metadata_started_at_epoch(run) -> float | None:
    metadata = getattr(run, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    started_at = _parse_iso_timestamp(metadata.get("startedAt"))
    if started_at is not None:
        return started_at
    slurm = metadata.get("slurm")
    if not isinstance(slurm, dict):
        return None
    return _optional_float(slurm.get("job_start_time"))


def _scan_timing_history(run) -> tuple[list[int], list[float], list[float | None], list[float | None]]:
    rows = list(
        run.scan_history(
            keys=TIMING_HISTORY_KEYS,
            page_size=5000,
        )
    )
    return _normalize_history_with_timing(rows)


def _latest_run_with_states(
    api,
    *,
    entity: str,
    project: str,
    spec: RunSpec,
    states: Iterable[str],
):
    usable_states = {str(state).lower() for state in states}
    candidates = list(
        api.runs(
            f"{entity}/{project}",
            filters={"displayName": spec.display_name},
            per_page=50,
        )
    )
    runs = [
        run
        for run in candidates
        if str(getattr(run, "state", "")).lower() in usable_states
    ]
    if not runs:
        if candidates:
            seen_states = sorted({str(getattr(run, "state", "")).lower() or "<unset>" for run in candidates})
            _warn(
                f"W&B run displayName={spec.display_name!r} has no usable state "
                f"in {sorted(usable_states)}; saw states={seen_states}"
            )
        return None
    # Prefer a cleanly finished/running run; fall back to other allowed states
    # (e.g. "crashed") only when none exist, so a crashed run is used solely when
    # it is the only option (the xl 100b more-tokens run).
    preferred = [
        run
        for run in runs
        if str(getattr(run, "state", "")).lower() in {"finished", "running"}
    ]
    pool = preferred or runs
    return max(pool, key=lambda run: _parse_created_at(str(getattr(run, "created_at"))))


def _elapsed_seconds_between(
    *,
    timestamps: list[float | None],
    runtimes: list[float | None],
    start_index: int,
    end_index: int,
) -> float | None:
    for values in (timestamps, runtimes):
        start_value = values[start_index]
        end_value = values[end_index]
        if start_value is None or end_value is None:
            continue
        elapsed = float(end_value) - float(start_value)
        if elapsed > 0.0 and math.isfinite(elapsed):
            return elapsed
    return None


def _interval_is_after_started_at(
    *,
    timestamps: list[float | None],
    start_index: int,
    end_index: int,
    started_at_epoch: float | None,
) -> bool:
    if started_at_epoch is None:
        return True
    start_timestamp = timestamps[start_index]
    end_timestamp = timestamps[end_index]
    if start_timestamp is None or end_timestamp is None:
        return True
    return float(start_timestamp) >= started_at_epoch and float(end_timestamp) >= started_at_epoch


def _select_gpu_hour_calibration_interval(
    *,
    tokens_seen: list[int],
    timestamps: list[float | None],
    runtimes: list[float | None],
    reference_interval: int,
    started_at_epoch: float | None = None,
) -> tuple[float, int, int]:
    if reference_interval < 0:
        raise ValueError("GPU-hour reference interval must be non-negative")
    if len(tokens_seen) <= reference_interval + 1:
        raise ValueError(
            f"Need at least {reference_interval + 2} validation rows for GPU-hour calibration; "
            f"saw {len(tokens_seen)}"
        )

    for start_index in range(int(reference_interval), len(tokens_seen) - 1):
        end_index = start_index + 1
        token_delta = int(tokens_seen[end_index]) - int(tokens_seen[start_index])
        if token_delta <= 0:
            continue
        if not _interval_is_after_started_at(
            timestamps=timestamps,
            start_index=start_index,
            end_index=end_index,
            started_at_epoch=started_at_epoch,
        ):
            continue
        elapsed_seconds = _elapsed_seconds_between(
            timestamps=timestamps,
            runtimes=runtimes,
            start_index=start_index,
            end_index=end_index,
        )
        if elapsed_seconds is not None:
            return elapsed_seconds, start_index, end_index

    raise ValueError("GPU-hour calibration could not find a usable live validation interval")


def _gpu_hours_per_token(
    *,
    tokens_seen: list[int],
    timestamps: list[float | None],
    runtimes: list[float | None],
    gpu_count: int,
    reference_interval: int,
    started_at_epoch: float | None = None,
) -> tuple[float, float, int, int]:
    elapsed_seconds, start_index, end_index = _select_gpu_hour_calibration_interval(
        tokens_seen=tokens_seen,
        timestamps=timestamps,
        runtimes=runtimes,
        reference_interval=reference_interval,
        started_at_epoch=started_at_epoch,
    )
    token_delta = int(tokens_seen[end_index]) - int(tokens_seen[start_index])
    if token_delta <= 0:
        raise ValueError(
            "GPU-hour calibration requires increasing tokens between "
            f"rows {start_index} and {end_index}; saw delta={token_delta}"
        )
    gpu_hours_per_token = float(gpu_count) * elapsed_seconds / float(token_delta) / 3600.0
    return gpu_hours_per_token, elapsed_seconds, start_index, end_index


def _with_gpu_hours(
    *,
    spec: RunSpec,
    run_id: str,
    created_at: str,
    tokens_seen: list[int],
    metric_values: list[float],
    timestamps: list[float | None],
    runtimes: list[float | None],
    gpu_count: int,
    reference_interval: int,
    started_at_epoch: float | None = None,
    calibration_mode: str | None = None,
) -> HistoryResolvedRun:
    gpu_hours_per_token, elapsed_seconds, start_index, end_index = _gpu_hours_per_token(
        tokens_seen=tokens_seen,
        timestamps=timestamps,
        runtimes=runtimes,
        gpu_count=gpu_count,
        reference_interval=reference_interval,
        started_at_epoch=started_at_epoch,
    )
    resolved_calibration_mode = calibration_mode
    if resolved_calibration_mode is None:
        resolved_calibration_mode = (
            CALIBRATION_MODE_STARTED_AT
            if start_index != int(reference_interval)
            else CALIBRATION_MODE_SINGLE_RATE
        )
    return HistoryResolvedRun(
        spec=spec,
        run_id=run_id,
        created_at=created_at,
        tokens_seen=tokens_seen,
        metric_values=metric_values,
        timestamps=timestamps,
        runtimes=runtimes,
        gpu_count=gpu_count,
        gpu_hours=[float(tokens) * gpu_hours_per_token for tokens in tokens_seen],
        calibration_start_index=start_index,
        calibration_end_index=end_index,
        calibration_start_tokens=int(tokens_seen[start_index]),
        calibration_end_tokens=int(tokens_seen[end_index]),
        calibration_seconds=elapsed_seconds,
        gpu_hours_per_token=gpu_hours_per_token,
        calibration_mode=resolved_calibration_mode,
    )


def resolve_history_runs_with_gpu_hours(
    api,
    *,
    entity: str,
    project: str,
    scales: Iterable[str] = DEFAULT_SCALES,
    windows: Iterable[int] | None = None,
    families: Iterable[str] = DEFAULT_FAMILIES,
    reference_interval: int = DEFAULT_GPU_HOURS_REFERENCE_INTERVAL,
    default_gpu_count: int | None = None,
) -> list[HistoryResolvedRun]:
    resolved: list[HistoryResolvedRun] = []
    for scale in scales:
        for spec in run_specs_for_scale(scale, windows=windows, families=families):
            run = _latest_run(api, entity=entity, project=project, spec=spec)
            if run is None:
                _warn(f"no usable W&B training run found for displayName={spec.display_name!r}")
                continue
            tokens_seen, metric_values, timestamps, runtimes = _scan_timing_history(run)
            if not tokens_seen:
                _warn(
                    f"W&B run displayName={spec.display_name!r} has no usable "
                    f"{TRAINING_METRIC_KEY!r} history"
                )
                continue
            gpu_count = _metadata_gpu_count(run, default_gpu_count=default_gpu_count)
            if gpu_count is None:
                _warn(
                    f"skipping displayName={spec.display_name!r}; missing positive W&B metadata.gpu_count "
                    "and no --default-gpu-count was provided"
                )
                continue
            try:
                run_id = str(getattr(run, "id"))
                target_started_at_epoch = _metadata_started_at_epoch(run)
                resolved_run = _with_gpu_hours(
                    spec=spec,
                    run_id=run_id,
                    created_at=str(getattr(run, "created_at")),
                    tokens_seen=tokens_seen,
                    metric_values=metric_values,
                    timestamps=timestamps,
                    runtimes=runtimes,
                    gpu_count=gpu_count,
                    reference_interval=reference_interval,
                    started_at_epoch=target_started_at_epoch,
                )
                resolved.append(resolved_run)
            except ValueError as exc:
                _warn(f"skipping displayName={spec.display_name!r}; {exc}")
    return resolved


def _standard_gpu_hour_references_by_scale(
    resolved_runs: Iterable[HistoryResolvedRun],
) -> dict[str, HistoryResolvedRun]:
    references: dict[str, HistoryResolvedRun] = {}
    for run in resolved_runs:
        if run.spec.family == "full_attention" and run.spec.window is None:
            references[run.spec.scale] = run
    return references


def _standard_2x_tokens_history_with_reference_gpu_hours(
    *,
    spec: RunSpec,
    run_id: str,
    created_at: str,
    tokens_seen: list[int],
    metric_values: list[float],
    timestamps: list[float | None],
    runtimes: list[float | None],
    reference_run: HistoryResolvedRun,
) -> HistoryResolvedRun:
    return HistoryResolvedRun(
        spec=spec,
        run_id=run_id,
        created_at=created_at,
        tokens_seen=tokens_seen,
        metric_values=metric_values,
        timestamps=timestamps,
        runtimes=runtimes,
        gpu_count=reference_run.gpu_count,
        gpu_hours=[float(tokens) * reference_run.gpu_hours_per_token for tokens in tokens_seen],
        calibration_start_index=reference_run.calibration_start_index,
        calibration_end_index=reference_run.calibration_end_index,
        calibration_start_tokens=reference_run.calibration_start_tokens,
        calibration_end_tokens=reference_run.calibration_end_tokens,
        calibration_seconds=reference_run.calibration_seconds,
        gpu_hours_per_token=reference_run.gpu_hours_per_token,
        calibration_mode=CALIBRATION_MODE_STANDARD_20B_REFERENCE,
    )


def _filter_tokens_after_reference_pre_decay(
    *,
    tokens_seen: list[int],
    metric_values: list[float],
    timestamps: list[float | None],
    runtimes: list[float | None],
    reference_run: HistoryResolvedRun,
) -> tuple[list[int], list[float], list[float | None], list[float | None]]:
    reference_pre_decay_end = pre_decay_token_cutoff_for_run(reference_run)
    filtered_tokens: list[int] = []
    filtered_metrics: list[float] = []
    filtered_timestamps: list[float | None] = []
    filtered_runtimes: list[float | None] = []
    for tokens, metric_value, timestamp, runtime in zip(
        tokens_seen,
        metric_values,
        timestamps,
        runtimes,
        strict=True,
    ):
        if int(tokens) <= reference_pre_decay_end:
            continue
        filtered_tokens.append(int(tokens))
        filtered_metrics.append(float(metric_value))
        filtered_timestamps.append(timestamp)
        filtered_runtimes.append(runtime)
    return filtered_tokens, filtered_metrics, filtered_timestamps, filtered_runtimes


def resolve_standard_2x_tokens_runs_with_gpu_hours(
    api,
    *,
    entity: str,
    project: str,
    scales: Iterable[str],
    reference_runs: Iterable[HistoryResolvedRun],
) -> list[HistoryResolvedRun]:
    references = _standard_gpu_hour_references_by_scale(reference_runs)
    resolved: list[HistoryResolvedRun] = []
    for spec in standard_2x_tokens_run_specs_for_scales(scales):
        reference_run = references.get(spec.scale)
        if reference_run is None:
            _warn(
                f"skipping displayName={spec.display_name!r}; missing matching 20B standard "
                "GPU-hour reference"
            )
            continue
        run = _latest_run_with_states(
            api,
            entity=entity,
            project=project,
            spec=spec,
            states=STANDARD_2X_TOKENS_USABLE_RUN_STATES,
        )
        if run is None:
            _warn(f"no usable W&B training run found for displayName={spec.display_name!r}")
            continue
        tokens_seen, metric_values, timestamps, runtimes = _scan_timing_history(run)
        if not tokens_seen:
            _warn(
                f"W&B run displayName={spec.display_name!r} has no usable "
                f"{TRAINING_METRIC_KEY!r} history"
            )
            continue
        filtered_tokens, filtered_metrics, filtered_timestamps, filtered_runtimes = (
            _filter_tokens_after_reference_pre_decay(
                tokens_seen=tokens_seen,
                metric_values=metric_values,
                timestamps=timestamps,
                runtimes=runtimes,
                reference_run=reference_run,
            )
        )
        if not filtered_tokens:
            continue
        resolved_run = _standard_2x_tokens_history_with_reference_gpu_hours(
            spec=spec,
            run_id=str(getattr(run, "id")),
            created_at=str(getattr(run, "created_at")),
            tokens_seen=filtered_tokens,
            metric_values=filtered_metrics,
            timestamps=filtered_timestamps,
            runtimes=filtered_runtimes,
            reference_run=reference_run,
        )
        resolved.append(resolved_run)
    return resolved


def _scale_runs(resolved_runs: Iterable[HistoryRun], scale: str) -> list[HistoryRun]:
    return [run for run in resolved_runs if run.spec.scale == scale]


def _filter_main_method_runs(
    resolved_runs: Iterable[HistoryRun],
    *,
    windows: Iterable[int] | None,
    families: Iterable[str],
) -> list[HistoryRun]:
    method_keys = {
        method.key
        for method in main_method_specs_for_selection(
            families=families,
            windows=windows,
        )
    }
    filtered: list[HistoryRun] = []
    for run in resolved_runs:
        if _is_standard_2x_tokens_run(run):
            filtered.append(run)
            continue
        method = main_method_for_family_window(run.spec.family, run.spec.window)
        if method is not None and method.key in method_keys:
            filtered.append(run)
    return filtered


def _line_key(family: str, window: int | None) -> str:
    if family == STANDARD_2X_TOKENS_FAMILY and window is None:
        return STANDARD_2X_TOKENS_KEY
    method = main_method_for_family_window(family, window)
    if method is not None:
        return method.key
    if family == "full_attention":
        return "full_attention"
    if window is None:
        raise ValueError(f"Windowed family {family!r} requires a window")
    return f"{family}_{int(window)}"


def _method_color(method: MainMethodSpec) -> str:
    try:
        return COLORBLIND_METHOD_COLORS[method.key]
    except KeyError:
        return "#000000"


def _history_line_kwargs(
    method: MainMethodSpec,
    *,
    linewidth_multiplier: float = 1.0,
    alpha: float = HISTORY_REAL_LINE_ALPHA,
) -> dict[str, object]:
    if method.family == "full_attention":
        linewidth = FULL_ATTENTION_LINEWIDTH * HISTORY_LINEWIDTH_SCALE
        zorder = STANDARD_HISTORY_ZORDER
    elif method.key == "delayed_state":
        linewidth = WINDOWED_SERIES_LINEWIDTH * HISTORY_LINEWIDTH_SCALE
        zorder = DELAYED_STATE_HISTORY_ZORDER
    else:
        linewidth = WINDOWED_SERIES_LINEWIDTH * HISTORY_LINEWIDTH_SCALE
        zorder = WINDOWED_HISTORY_ZORDER
    return {
        "color": _method_color(method),
        "linewidth": linewidth * linewidth_multiplier,
        "alpha": alpha,
        "label": sc(method.label),
        "zorder": zorder,
    }


def _history_stroke_width(method: MainMethodSpec, *, linewidth_multiplier: float = 1.0) -> float:
    if method.family == "full_attention":
        return FULL_ATTENTION_STROKE_LINEWIDTH * HISTORY_STROKE_LINEWIDTH_SCALE * linewidth_multiplier
    return WINDOWED_SERIES_STROKE_LINEWIDTH * HISTORY_STROKE_LINEWIDTH_SCALE * linewidth_multiplier


def _standard_2x_tokens_line_kwargs(
    *,
    linewidth_multiplier: float = 1.0,
    alpha: float = HISTORY_REAL_LINE_ALPHA,
) -> dict[str, object]:
    return {
        "color": STANDARD_2X_TOKENS_COLOR,
        "linewidth": FULL_ATTENTION_LINEWIDTH * HISTORY_LINEWIDTH_SCALE * linewidth_multiplier,
        "linestyle": STANDARD_2X_TOKENS_LINESTYLE,
        "alpha": alpha,
        "label": sc(STANDARD_2X_TOKENS_LABEL),
        "zorder": STANDARD_HISTORY_ZORDER,
    }


def _standard_2x_tokens_stroke_width(*, linewidth_multiplier: float = 1.0) -> float:
    return FULL_ATTENTION_STROKE_LINEWIDTH * HISTORY_STROKE_LINEWIDTH_SCALE * linewidth_multiplier


def _smooth_metric_values(values: list[float], sigma_points: float) -> list[float]:
    if sigma_points <= 0.0:
        return list(values)

    radius = max(1, int(math.ceil(3.0 * sigma_points)))
    if len(values) < (2 * radius) + 1:
        return list(values)

    offsets = range(-radius, radius + 1)
    weights = [
        math.exp(-0.5 * (float(offset) / sigma_points) ** 2)
        for offset in offsets
    ]
    smoothed: list[float] = []
    for center_index in range(len(values)):
        weighted_sum = 0.0
        weight_sum = 0.0
        for offset, weight in zip(offsets, weights, strict=True):
            value_index = center_index + offset
            if 0 <= value_index < len(values):
                weighted_sum += float(values[value_index]) * weight
                weight_sum += weight
        smoothed.append(weighted_sum / weight_sum if weight_sum else float(values[center_index]))
    return smoothed


def _uses_gpu_hours_axis(x_axis: str) -> bool:
    return x_axis in {X_AXIS_GPU_HOURS, X_AXIS_COMBINED}


def _draw_standard_2x_tokens_handoff_guide(x_axis: str) -> bool:
    return _uses_gpu_hours_axis(x_axis)


def _standard_2x_tokens_handoff_x(run: HistoryRun, *, x_axis: str) -> float | None:
    if not _draw_standard_2x_tokens_handoff_guide(x_axis):
        return None
    gpu_hours_per_token = getattr(run, "gpu_hours_per_token", None)
    if gpu_hours_per_token is None:
        return None
    return float(pre_decay_token_cutoff_for_run(run)) * float(gpu_hours_per_token)


def _standard_2x_tokens_handoff_guide_kwargs() -> dict[str, object]:
    return {
        "color": STANDARD_2X_TOKENS_HANDOFF_GUIDE_COLOR,
        "linewidth": STANDARD_2X_TOKENS_HANDOFF_GUIDE_LINEWIDTH,
        "linestyle": STANDARD_2X_TOKENS_HANDOFF_GUIDE_LINESTYLE,
        "alpha": STANDARD_2X_TOKENS_HANDOFF_GUIDE_ALPHA,
        "zorder": STANDARD_2X_TOKENS_HANDOFF_GUIDE_ZORDER,
    }


def _plot_x_min(max_x: float, *, x_axis: str) -> float:
    if x_axis == X_AXIS_TOKENS:
        return 0.0
    return max(0.0, float(max_x) * HISTORY_X_MIN_FRACTION)


def _plot_x_max(max_x: float, *, x_axis: str) -> float:
    padded_max = float(max_x) * X_PADDING_FACTOR
    if x_axis == X_AXIS_TOKENS:
        return max(padded_max, HISTORY_TOKEN_X_MAJOR_TICKS[-1])
    return padded_max


def _pre_decay_match_guide_key(method: MainMethodSpec) -> str:
    return f"{PRE_DECAY_MATCH_GUIDE_KEY_PREFIX}_{method.key}"


def _pre_decay_match_guide_kwargs(method: MainMethodSpec) -> dict[str, object]:
    return {
        "color": _method_color(method),
        "linewidth": PRE_DECAY_MATCH_GUIDE_LINEWIDTH,
        "linestyle": PRE_DECAY_MATCH_GUIDE_LINESTYLE,
        "alpha": PRE_DECAY_MATCH_GUIDE_ALPHA,
        "zorder": PRE_DECAY_MATCH_GUIDE_ZORDER,
    }


def _pre_decay_match_guide_legend_handle() -> Line2D:
    return Line2D(
        [0],
        [0],
        color=PRE_DECAY_MATCH_GUIDE_LEGEND_COLOR,
        linewidth=PRE_DECAY_MATCH_GUIDE_LINEWIDTH,
        linestyle=PRE_DECAY_MATCH_GUIDE_LINESTYLE,
        alpha=PRE_DECAY_MATCH_GUIDE_ALPHA,
    )


def _pre_decay_match_guide_label(x_axis: str) -> str:
    if x_axis == X_AXIS_GPU_HOURS:
        return "GPU Hours to Match Pre-Decay Standard"
    return PRE_DECAY_MATCH_GUIDE_LABEL


def _draw_pre_decay_match_guides(x_axis: str) -> bool:
    return x_axis == X_AXIS_TOKENS


def _history_value_at_or_before(run: HistoryRun, token_cutoff: int) -> float | None:
    value: float | None = None
    for tokens_seen, metric_value in zip(run.tokens_seen, run.metric_values, strict=True):
        if int(tokens_seen) > token_cutoff:
            continue
        value = float(metric_value)
    return value


def _pre_decay_target_nll(run: HistoryRun, scale: str) -> float | None:
    return _history_value_at_or_before(run, pre_decay_token_cutoff_for_scale(scale))


def _first_token_at_or_below(run: HistoryRun, target_nll: float) -> int | None:
    for tokens_seen, metric_value in zip(run.tokens_seen, run.metric_values, strict=True):
        if float(metric_value) <= float(target_nll):
            return int(tokens_seen)
    return None


def _standard_2x_tokens_sps_match_target(
    scale: str,
    plottable_runs: list[tuple["HistoryRun", "MainMethodSpec"]],
) -> float | None:
    # Target NLL for stopping the "Standard (2x Tokens)" line: the SPS run's final
    # (pre-decay) validation NLL for this scale. None when the scale is not configured
    # for this behavior or the SPS run is absent from the panel.
    if scale not in STANDARD_2X_TOKENS_STOP_AT_SPS_FINAL_SCALES:
        return None
    sps_run = next(
        (
            run
            for run, method in plottable_runs
            if method.key == STANDARD_2X_TOKENS_STOP_AT_SPS_METHOD_KEY
        ),
        None,
    )
    if sps_run is None:
        return None
    return _pre_decay_target_nll(sps_run, scale)


def _truncate_line_at_target_nll(
    x_values: list[float],
    metric_values: list[float],
    target_nll: float,
) -> tuple[list[float], list[float]]:
    # Cut the line at the first point whose (smoothed) NLL reaches target_nll so the
    # series visibly ends where it matches the target rather than continuing past it.
    for idx, value in enumerate(metric_values):
        if value <= target_nll:
            return x_values[: idx + 1], metric_values[: idx + 1]
    return x_values, metric_values


def _is_pre_decay_only_axis(x_axis: str) -> bool:
    return _uses_gpu_hours_axis(x_axis)


def _run_x_values(run: HistoryRun, *, x_axis: str) -> list[float]:
    if x_axis == X_AXIS_TOKENS:
        return [float(tokens_seen) for tokens_seen in run.tokens_seen]
    if _uses_gpu_hours_axis(x_axis):
        gpu_hours = getattr(run, "gpu_hours", None)
        if gpu_hours is None:
            raise ValueError(f"Run {run.spec.display_name!r} has no GPU-hour values")
        return [float(value) for value in gpu_hours]
    raise ValueError(f"Unsupported x-axis {x_axis!r}; expected one of {X_AXIS_CHOICES!r}")


def _run_plot_points(
    run: HistoryRun,
    scale: str,
    *,
    x_axis: str,
) -> tuple[list[int], list[float], list[float]]:
    del scale
    cutoff = pre_decay_token_cutoff_for_run(run) if _is_pre_decay_only_axis(x_axis) else None
    tokens_out: list[int] = []
    x_out: list[float] = []
    y_out: list[float] = []
    for tokens_seen, x_value, metric_value in zip(
        run.tokens_seen,
        _run_x_values(run, x_axis=x_axis),
        run.metric_values,
        strict=True,
    ):
        if cutoff is not None and int(tokens_seen) > cutoff:
            continue
        tokens_out.append(int(tokens_seen))
        x_out.append(float(x_value))
        y_out.append(float(metric_value))
    return tokens_out, x_out, y_out


def _x_value_for_token(run: HistoryRun, token: int, *, x_axis: str) -> float:
    if x_axis == X_AXIS_TOKENS:
        return float(token)
    x_values = _run_x_values(run, x_axis=x_axis)
    for run_token, x_value in zip(run.tokens_seen, x_values, strict=True):
        if int(run_token) == int(token):
            return float(x_value)
    raise ValueError(f"Token {token:,} is not present in run {run.spec.display_name!r}")


def _gpu_hours_formatter(value: float, _pos: int) -> str:
    return f"{value:g}"


def _bounds_series(
    run: HistoryRun,
    scale: str,
    *,
    x_axis: str,
    sps_match_target: float | None,
    smooth_sigma_points: float,
) -> tuple[list[float], list[float]]:
    """(x_values, metric_values) as actually drawn. The "Standard (More Tokens)"
    line is truncated at the SPS-match target on stop-at-SPS scales (xl), so the
    axis bounds track the visible curve rather than the full (crashed) run that
    continues past the match point."""
    _tokens, x_values, metric_values = _run_plot_points(run, scale, x_axis=x_axis)
    if (
        x_values
        and sps_match_target is not None
        and _is_standard_2x_tokens_run(run)
        and _uses_gpu_hours_axis(x_axis)
    ):
        smoothed = _smooth_metric_values(metric_values, smooth_sigma_points)
        x_values, metric_values = _truncate_line_at_target_nll(
            x_values, smoothed, sps_match_target
        )
    return x_values, metric_values


def _late_metric_values(
    run: HistoryRun,
    scale: str,
    *,
    x_axis: str,
    sps_match_target: float | None = None,
    smooth_sigma_points: float = 0.0,
) -> list[float]:
    x_values, metric_values = _bounds_series(
        run,
        scale,
        x_axis=x_axis,
        sps_match_target=sps_match_target,
        smooth_sigma_points=smooth_sigma_points,
    )
    if not x_values:
        return []
    x_min = min(x_values)
    x_max = max(x_values)
    tail_start = x_min + (1.0 - Y_BOUND_TAIL_FRACTION) * max(0.0, x_max - x_min)
    return [
        metric_value
        for x_value, metric_value in zip(x_values, metric_values, strict=True)
        if float(x_value) >= tail_start
    ] or [metric_values[-1]]


def _y_bounds(
    resolved_runs: list[HistoryRun],
    *,
    scale: str,
    x_axis: str,
    sps_match_target: float | None = None,
    smooth_sigma_points: float = 0.0,
) -> tuple[float, float]:
    final_values = [
        metric_values[-1]
        for run in resolved_runs
        if (
            metric_values := _bounds_series(
                run,
                scale,
                x_axis=x_axis,
                sps_match_target=sps_match_target,
                smooth_sigma_points=smooth_sigma_points,
            )[1]
        )
    ]
    if not final_values:
        raise ValueError("No validation histories available for y-axis bounds")
    late_values = [
        value
        for run in resolved_runs
        for value in _late_metric_values(
            run,
            scale,
            x_axis=x_axis,
            sps_match_target=sps_match_target,
            smooth_sigma_points=smooth_sigma_points,
        )
    ]
    bound_values = [*late_values, *final_values]
    y_min = min(bound_values)
    final_y_max = max(final_values)
    y_max = max(bound_values)
    lower_span = max(final_y_max - y_min, MIN_Y_SPAN)
    upper_span = max(y_max - y_min, MIN_Y_SPAN)
    padded_min = max(0.0, y_min - Y_PADDING_FRACTION * lower_span)
    padded_max = y_max + Y_PADDING_FRACTION * upper_span
    rounded_min = math.floor(padded_min / Y_LIMIT_ROUNDING_INTERVAL) * Y_LIMIT_ROUNDING_INTERVAL
    rounded_max = math.ceil(padded_max / Y_LIMIT_ROUNDING_INTERVAL) * Y_LIMIT_ROUNDING_INTERVAL
    if rounded_min >= rounded_max:
        rounded_max = rounded_min + Y_LIMIT_ROUNDING_INTERVAL
    return max(0.0, rounded_min), rounded_max


def _nice_tick_interval(span: float, *, max_intervals: int = GPU_HOURS_X_MAJOR_TICK_MAX_INTERVALS) -> float:
    if span <= 0.0 or not math.isfinite(span):
        return 1.0
    raw_interval = float(span) / max(1, int(max_intervals))
    magnitude = 10.0 ** math.floor(math.log10(raw_interval))
    normalized = raw_interval / magnitude
    for step in GPU_HOURS_X_MAJOR_TICK_STEPS:
        if normalized <= step:
            return float(step) * magnitude
    return 10.0 * magnitude


def _history_x_major_locator(x_axis: str):
    if x_axis == X_AXIS_TOKENS:
        return FixedLocator(HISTORY_TOKEN_X_MAJOR_TICKS)
    if _uses_gpu_hours_axis(x_axis):
        return MaxNLocator(
            nbins=GPU_HOURS_X_MAJOR_TICK_MAX_INTERVALS,
            steps=GPU_HOURS_X_MAJOR_TICK_STEPS,
            min_n_ticks=4,
        )
    raise ValueError(f"Unsupported x-axis {x_axis!r}; expected one of {X_AXIS_CHOICES!r}")


def _history_y_major_locator() -> LinearLocator:
    return LinearLocator(numticks=HISTORY_Y_MAJOR_TICK_COUNT)


def _style_history_ticks(ax, *, labelsize: int = HISTORY_AXIS_TICK_LABEL_SIZE) -> None:
    ax.tick_params(
        axis="both",
        labelsize=labelsize,
        length=AXIS_TICK_LENGTH,
        width=AXIS_TICK_WIDTH,
        color=PLOT_SPINE_COLOR,
        labelcolor=LEGEND_TEXT_COLOR,
    )
    for tick in (*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks()):
        tick.tick1line.set_color(PLOT_SPINE_COLOR)
        tick.tick2line.set_color(PLOT_SPINE_COLOR)


def _x_axis_label(x_axis: str) -> str:
    if x_axis == X_AXIS_TOKENS:
        return "Training Tokens"
    if _uses_gpu_hours_axis(x_axis):
        return "GPU Hours"
    raise ValueError(f"Unsupported x-axis {x_axis!r}; expected one of {X_AXIS_CHOICES!r}")


def _y_axis_label(x_axis: str) -> str:
    if _uses_gpu_hours_axis(x_axis):
        return "Pre-Decay Validation NLL"
    return TRAINING_METRIC_LABEL


def _x_axis_formatter(x_axis: str):
    if x_axis == X_AXIS_TOKENS:
        return FuncFormatter(_billions_formatter)
    if _uses_gpu_hours_axis(x_axis):
        return FuncFormatter(_gpu_hours_formatter)
    raise ValueError(f"Unsupported x-axis {x_axis!r}; expected one of {X_AXIS_CHOICES!r}")


def _hide_pending_panel(
    ax,
    scale: str,
    *,
    title: str | None = None,
    title_fontsize: int = HISTORY_PANEL_TITLE_FONT_SIZE,
    pending_fontsize: int = HISTORY_PENDING_PANEL_FONT_SIZE,
) -> None:
    title_text = scale.upper() if title is None else title
    if title_text:
        ax.set_title(
            bf(title_text),
            fontsize=title_fontsize,
            fontweight=PANEL_TITLE_FONT_WEIGHT,
            pad=12,
        )
    ax.text(
        0.5,
        0.5,
        "pending",
        ha="center",
        va="center",
        fontsize=pending_fontsize,
        color="#6b6258",
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#c8bfb0")


def _legend_groups_for_lines(
    plotted_lines: dict[str, Line2D],
    *,
    method_specs: Iterable[MainMethodSpec],
    x_axis: str = X_AXIS_TOKENS,
) -> list[LegendGroup]:
    groups: list[LegendGroup] = []
    for method in method_specs:
        if method.key in plotted_lines:
            groups.append(
                LegendGroup(
                    title=sc(method.label),
                    entries=[LegendEntry(line_key=method.key, legend_label="")],
                    ncol=1,
                )
            )
        if method.key == "standard" and STANDARD_2X_TOKENS_KEY in plotted_lines:
            groups.append(
                LegendGroup(
                    title=sc(STANDARD_2X_TOKENS_LABEL),
                    entries=[LegendEntry(line_key=STANDARD_2X_TOKENS_KEY, legend_label="")],
                    ncol=1,
                )
            )
    if PRE_DECAY_MATCH_GUIDE_KEY in plotted_lines:
        groups.append(
            LegendGroup(
                title=_pre_decay_match_guide_label(x_axis),
                entries=[LegendEntry(line_key=PRE_DECAY_MATCH_GUIDE_KEY, legend_label="")],
                ncol=1,
            )
        )
    return groups


def _move_legend_axis_down(legend_ax) -> None:
    position = legend_ax.get_position()
    legend_ax.set_position([position.x0, position.y0 - 0.045, position.width, position.height])


def _history_panel_runs(
    resolved_runs: Iterable[HistoryRun],
    scale: str,
    *,
    selected_method_keys: set[str],
    x_axis: str,
) -> list[HistoryRun]:
    return [
        run
        for run in _scale_runs(resolved_runs, scale)
        if (_is_standard_2x_tokens_run(run) and _uses_gpu_hours_axis(x_axis))
        or (
            (method := main_method_for_family_window(run.spec.family, run.spec.window)) is not None
            and method.key in selected_method_keys
        )
    ]


def _draw_history_panel(
    ax,
    resolved_runs: list[HistoryRun],
    *,
    scale: str,
    selected_windows_override: tuple[int, ...] | None,
    selected_families: tuple[str, ...],
    selected_method_keys: set[str],
    x_axis: str,
    plotted_lines: dict[str, Line2D],
    title: str | None = None,
    title_fontsize: int = HISTORY_PANEL_TITLE_FONT_SIZE,
    pending_fontsize: int = HISTORY_PENDING_PANEL_FONT_SIZE,
    tick_labelsize: int = HISTORY_AXIS_TICK_LABEL_SIZE,
    x_min_override: float | None = None,
    line_width_multiplier: float = 1.0,
    line_alpha: float = HISTORY_REAL_LINE_ALPHA,
    stroke_alpha: float = HISTORY_STROKE_ALPHA,
    smooth_sigma_points: float = 0.0,
    draw_handoff_guide: bool = True,
) -> None:
    panel_runs = _history_panel_runs(
        resolved_runs,
        scale,
        selected_method_keys=selected_method_keys,
        x_axis=x_axis,
    )
    if not panel_runs:
        _hide_pending_panel(
            ax,
            scale,
            title=title,
            title_fontsize=title_fontsize,
            pending_fontsize=pending_fontsize,
        )
        return

    panel_runs = [
        run
        for run in panel_runs
        if _run_plot_points(run, scale, x_axis=x_axis)[1]
    ]
    if not panel_runs:
        _hide_pending_panel(
            ax,
            scale,
            title=title,
            title_fontsize=title_fontsize,
            pending_fontsize=pending_fontsize,
        )
        return

    panel_windows = windows_for_scale(scale, selected_windows_override)

    plottable_runs: list[tuple[HistoryRun, MainMethodSpec]] = []
    standard_2x_tokens_runs: list[HistoryRun] = []
    for run in panel_runs:
        if _is_standard_2x_tokens_run(run):
            if _uses_gpu_hours_axis(x_axis):
                standard_2x_tokens_runs.append(run)
            continue
        method = main_method_for_family_window(run.spec.family, run.spec.window)
        if method is None or method.key not in selected_method_keys:
            continue
        if (
            run.spec.family != "full_attention"
            and (run.spec.family not in selected_families or run.spec.window not in panel_windows)
        ):
            continue
        plottable_runs.append((run, method))

    # Stop the "Standard (More Tokens)" line at the SPS final NLL on configured
    # scales (xl). Computed before the axis bound so max_x reflects the *drawn*
    # (truncated) extent of that line -- otherwise the xl 100b more-tokens run,
    # which continues well past the SPS-match point before it crashed, would
    # stretch the GPU-hours axis to its full length.
    sps_match_target = _standard_2x_tokens_sps_match_target(scale, plottable_runs)

    def _drawn_x_values(run: HistoryRun) -> list[float]:
        _t, x_values, metric_values = _run_plot_points(run, scale, x_axis=x_axis)
        if (
            x_values
            and sps_match_target is not None
            and _is_standard_2x_tokens_run(run)
            and _uses_gpu_hours_axis(x_axis)
        ):
            metric_values = _smooth_metric_values(metric_values, smooth_sigma_points)
            x_values, _ = _truncate_line_at_target_nll(
                x_values, metric_values, sps_match_target
            )
        return x_values

    max_x = max(
        max(drawn)
        for run in panel_runs
        if (drawn := _drawn_x_values(run))
    )
    plot_x_min = _plot_x_min(max_x, x_axis=x_axis) if x_min_override is None else float(x_min_override)
    plot_x_max = _plot_x_max(max_x, x_axis=x_axis)
    plot_y_min, plot_y_max = _y_bounds(
        panel_runs,
        scale=scale,
        x_axis=x_axis,
        sps_match_target=sps_match_target,
        smooth_sigma_points=smooth_sigma_points,
    )
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(plot_y_min, plot_y_max)
    _apply_axes_background(
        ax,
        x_min=plot_x_min,
        x_max=plot_x_max,
        y_min=plot_y_min,
        y_max=plot_y_max,
    )

    standard_run = next((run for run, method in plottable_runs if method.family == "full_attention"), None)
    if draw_handoff_guide and standard_run is not None and standard_2x_tokens_runs:
        handoff_x = _standard_2x_tokens_handoff_x(standard_run, x_axis=x_axis)
        if handoff_x is not None and plot_x_min <= handoff_x <= plot_x_max:
            ax.axvline(
                handoff_x,
                **_standard_2x_tokens_handoff_guide_kwargs(),
            )
    if standard_run is not None and _draw_pre_decay_match_guides(x_axis):
        target_nll = _pre_decay_target_nll(standard_run, scale)
        if target_nll is not None:
            for run, method in plottable_runs:
                if method.family == "full_attention":
                    continue
                matched_tokens = _first_token_at_or_below(run, target_nll)
                if matched_tokens is None:
                    continue
                matched_x = _x_value_for_token(run, matched_tokens, x_axis=x_axis)
                guide_line = ax.axvline(
                    matched_x,
                    **_pre_decay_match_guide_kwargs(method),
                )
                if plot_x_min <= matched_x <= plot_x_max:
                    plotted_lines.setdefault(_pre_decay_match_guide_key(method), guide_line)
                    plotted_lines.setdefault(PRE_DECAY_MATCH_GUIDE_KEY, _pre_decay_match_guide_legend_handle())

    for run, method in plottable_runs:
        line_kwargs = _history_line_kwargs(
            method,
            linewidth_multiplier=line_width_multiplier,
            alpha=line_alpha,
        )
        _tokens, x_values, metric_values = _run_plot_points(run, scale, x_axis=x_axis)
        metric_values = _smooth_metric_values(metric_values, smooth_sigma_points)
        line, = ax.plot(x_values, metric_values, **line_kwargs)
        line.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=_history_stroke_width(
                        method,
                        linewidth_multiplier=line_width_multiplier,
                    ),
                    foreground="white",
                    alpha=stroke_alpha,
                ),
                path_effects.Normal(),
            ]
        )
        plotted_lines.setdefault(_line_key(run.spec.family, run.spec.window), line)

    for run in standard_2x_tokens_runs:
        _tokens, x_values, metric_values = _run_plot_points(run, scale, x_axis=x_axis)
        metric_values = _smooth_metric_values(metric_values, smooth_sigma_points)
        if sps_match_target is not None:
            x_values, metric_values = _truncate_line_at_target_nll(
                x_values, metric_values, sps_match_target
            )
        line_kwargs = _standard_2x_tokens_line_kwargs(
            linewidth_multiplier=line_width_multiplier,
            alpha=line_alpha,
        )
        line, = ax.plot(x_values, metric_values, **line_kwargs)
        line.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=_standard_2x_tokens_stroke_width(
                        linewidth_multiplier=line_width_multiplier,
                    ),
                    foreground="white",
                    alpha=stroke_alpha,
                ),
                path_effects.Normal(),
            ]
        )
        plotted_lines.setdefault(STANDARD_2X_TOKENS_KEY, line)

    title_text = scale.upper() if title is None else title
    if title_text:
        ax.set_title(
            bf(title_text),
            fontsize=title_fontsize,
            fontweight=PANEL_TITLE_FONT_WEIGHT,
            pad=12,
        )
    ax.xaxis.set_major_locator(_history_x_major_locator(x_axis))
    ax.xaxis.set_major_formatter(_x_axis_formatter(x_axis))
    ax.yaxis.set_major_locator(_history_y_major_locator())
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    _style_history_ticks(ax, labelsize=tick_labelsize)
    ax.minorticks_off()


def _render_combined_plot(
    resolved_runs: list[HistoryRun],
    output_path: Path,
    *,
    selected_scales: tuple[str, ...],
    selected_windows_override: tuple[int, ...] | None,
    selected_families: tuple[str, ...],
    selected_methods: list[MainMethodSpec],
    selected_method_keys: set[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=COMBINED_HISTORY_FIGSIZE)
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=COMBINED_GRID_HEIGHT_RATIOS,
        hspace=COMBINED_GRID_HSPACE,
    )
    plot_grid = grid[0, 0].subgridspec(
        2,
        5,
        hspace=COMBINED_PANEL_GRID_HSPACE,
        wspace=COMBINED_PANEL_GRID_WSPACE,
    )
    axes_by_row = [
        [fig.add_subplot(plot_grid[row, col]) for col in range(5)]
        for row in range(2)
    ]
    plotted_lines: dict[str, Line2D] = {}
    legend_ax = None

    for col, scale in enumerate(selected_scales):
        _draw_history_panel(
            axes_by_row[0][col],
            resolved_runs,
            scale=scale,
            selected_windows_override=selected_windows_override,
            selected_families=selected_families,
            selected_method_keys=selected_method_keys,
            x_axis=X_AXIS_TOKENS,
            plotted_lines=plotted_lines,
            title=scale.upper(),
            title_fontsize=COMBINED_PANEL_TITLE_FONT_SIZE,
            pending_fontsize=COMBINED_PENDING_PANEL_FONT_SIZE,
            tick_labelsize=COMBINED_AXIS_TICK_LABEL_SIZE,
            x_min_override=COMBINED_TOKEN_X_MIN,
            line_width_multiplier=COMBINED_LINEWIDTH_MULTIPLIER,
            line_alpha=COMBINED_REAL_LINE_ALPHA,
            stroke_alpha=COMBINED_STROKE_ALPHA,
            smooth_sigma_points=COMBINED_SMOOTH_SIGMA_POINTS,
            draw_handoff_guide=True,
        )
        _draw_history_panel(
            axes_by_row[1][col],
            resolved_runs,
            scale=scale,
            selected_windows_override=selected_windows_override,
            selected_families=selected_families,
            selected_method_keys=selected_method_keys,
            x_axis=X_AXIS_GPU_HOURS,
            plotted_lines=plotted_lines,
            title="",
            title_fontsize=COMBINED_PANEL_TITLE_FONT_SIZE,
            pending_fontsize=COMBINED_PENDING_PANEL_FONT_SIZE,
            tick_labelsize=COMBINED_AXIS_TICK_LABEL_SIZE,
            line_width_multiplier=COMBINED_LINEWIDTH_MULTIPLIER,
            line_alpha=COMBINED_REAL_LINE_ALPHA,
            stroke_alpha=COMBINED_STROKE_ALPHA,
            smooth_sigma_points=COMBINED_SMOOTH_SIGMA_POINTS,
            draw_handoff_guide=False,
        )

    for col in range(len(selected_scales), 5):
        axes_by_row[0][col].set_visible(False)
        axes_by_row[1][col].set_visible(False)

    legend_groups = _legend_groups_for_lines(
        plotted_lines,
        method_specs=selected_methods,
        x_axis=X_AXIS_COMBINED,
    )
    if legend_groups:
        legend_ax = fig.add_subplot(grid[1, 0])
        _render_m_classic_legend(
            legend_ax,
            legend_groups,
            plotted_lines,
            header_fontsize=COMBINED_LEGEND_HEADER_FONT_SIZE,
            header_fontweight=COMBINED_LEGEND_FONT_WEIGHT,
            entry_fontsize=COMBINED_LEGEND_ENTRY_FONT_SIZE,
            entry_fontweight=COMBINED_LEGEND_FONT_WEIGHT,
        )

    fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=COMBINED_SUBPLOTS_BOTTOM)
    if legend_ax is not None:
        legend_position = legend_ax.get_position()
        legend_ax.set_position(
            [
                legend_position.x0,
                max(0.02, legend_position.y0 + COMBINED_LEGEND_Y_SHIFT),
                legend_position.width,
                legend_position.height,
            ]
        )
    visible_top_axes = [ax for ax in axes_by_row[0] if ax.get_visible()]
    visible_bottom_axes = [ax for ax in axes_by_row[1] if ax.get_visible()]
    plot_axes = [*visible_top_axes, *visible_bottom_axes]
    top_row_bottom = min(ax.get_position().y0 for ax in visible_top_axes)
    bottom_row_top = max(ax.get_position().y1 for ax in visible_bottom_axes)
    bottom_row_bottom = min(ax.get_position().y0 for ax in visible_bottom_axes)
    label_x = min(ax.get_position().x0 for ax in plot_axes) - 0.045

    fig.text(
        0.5,
        bottom_row_top
        + 0.5 * (top_row_bottom - bottom_row_top)
        + COMBINED_TRAINING_TOKENS_Y_SHIFT,
        bf(_x_axis_label(X_AXIS_TOKENS)),
        ha="center",
        va="center",
        fontsize=COMBINED_AXIS_LABEL_FONT_SIZE,
        fontweight=AXIS_LABEL_FONT_WEIGHT,
        color=LEGEND_TEXT_COLOR,
    )
    fig.text(
        0.5,
        bottom_row_bottom - COMBINED_GPU_HOURS_X_LABEL_OFFSET,
        bf(_x_axis_label(X_AXIS_GPU_HOURS)),
        ha="center",
        va="center",
        fontsize=COMBINED_AXIS_LABEL_FONT_SIZE,
        fontweight=AXIS_LABEL_FONT_WEIGHT,
        color=LEGEND_TEXT_COLOR,
    )
    fig.text(
        max(0.02, label_x),
        0.5
        * (
            max(ax.get_position().y1 for ax in visible_top_axes)
            + min(ax.get_position().y0 for ax in visible_bottom_axes)
        ),
        bf(COMBINED_Y_AXIS_LABEL),
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=COMBINED_AXIS_LABEL_FONT_SIZE,
        fontweight=AXIS_LABEL_FONT_WEIGHT,
        color=LEGEND_TEXT_COLOR,
    )
    png_path, pdf_path = _paired_image_paths(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def render_plot(
    resolved_runs: list[HistoryRun],
    output_path: Path,
    *,
    scales: Iterable[str] = DEFAULT_SCALES,
    windows: Iterable[int] | None = None,
    families: Iterable[str] = DEFAULT_FAMILIES,
    x_axis: str = X_AXIS_TOKENS,
) -> None:
    selected_scales = tuple(scales)
    selected_windows_override = None if windows is None else tuple(int(window) for window in windows)
    selected_families = tuple(families)
    selected_methods = main_method_specs_for_selection(
        families=selected_families,
        windows=selected_windows_override,
    )
    selected_method_keys = {method.key for method in selected_methods}
    if len(selected_scales) > 5:
        raise ValueError("The validation-history plot supports at most five scales (xs..xl)")
    if x_axis == X_AXIS_COMBINED:
        _render_combined_plot(
            resolved_runs,
            output_path,
            selected_scales=selected_scales,
            selected_windows_override=selected_windows_override,
            selected_families=selected_families,
            selected_methods=selected_methods,
            selected_method_keys=selected_method_keys,
        )
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(22, 16))
    grid = fig.add_gridspec(2, 1, height_ratios=[6.2, 1.05], hspace=0.18)
    plot_grid = grid[0, 0].subgridspec(2, 3, hspace=PANEL_GRID_HSPACE, wspace=0.12)
    axes_flat = [fig.add_subplot(plot_grid[row, col]) for row in range(2) for col in range(3)]
    plotted_lines: dict[str, Line2D] = {}
    legend_ax = None

    for ax, scale in zip(axes_flat, selected_scales, strict=False):
        _draw_history_panel(
            ax,
            resolved_runs,
            scale=scale,
            selected_windows_override=selected_windows_override,
            selected_families=selected_families,
            selected_method_keys=selected_method_keys,
            x_axis=x_axis,
            plotted_lines=plotted_lines,
        )

    for extra_ax in axes_flat[len(selected_scales):]:
        extra_ax.set_visible(False)

    legend_groups = _legend_groups_for_lines(
        plotted_lines,
        method_specs=selected_methods,
        x_axis=x_axis,
    )
    if legend_groups:
        legend_ax = fig.add_subplot(grid[1, 0])
        _render_m_classic_legend(
            legend_ax,
            legend_groups,
            plotted_lines,
            header_fontsize=HISTORY_LEGEND_HEADER_FONT_SIZE,
            entry_fontsize=HISTORY_LEGEND_ENTRY_FONT_SIZE,
        )

    fig.text(
        0.5,
        0.185,
        bf(_x_axis_label(x_axis)),
        ha="center",
        va="center",
        fontsize=HISTORY_AXIS_LABEL_FONT_SIZE,
        fontweight=AXIS_LABEL_FONT_WEIGHT,
        color=LEGEND_TEXT_COLOR,
    )
    fig.text(
        0.04,
        0.57,
        bf(_y_axis_label(x_axis)),
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=HISTORY_AXIS_LABEL_FONT_SIZE,
        fontweight=AXIS_LABEL_FONT_WEIGHT,
        color=LEGEND_TEXT_COLOR,
    )
    fig.subplots_adjust(left=0.105, right=0.99, top=0.96, bottom=0.08)
    if legend_ax is not None:
        _move_legend_axis_down(legend_ax)
    png_path, pdf_path = _paired_image_paths(output_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _csv_float(value: object, *, precision: int = 8) -> str:
    if value is None:
        return ""
    return f"{float(value):.{precision}f}"


def _csv_int(value: object) -> str:
    if value is None:
        return ""
    return str(int(value))


def write_csv(
    resolved_runs: list[HistoryRun],
    csv_path: Path,
    *,
    x_axis: str = X_AXIS_TOKENS,
) -> None:
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
                "gpu_hours",
                "validation_nll",
                "gpu_count",
                "calibration_start_tokens",
                "calibration_end_tokens",
                "calibration_seconds",
                "gpu_hours_per_token",
                "calibration_mode",
            ]
        )
        for run in sorted(
            resolved_runs,
            key=lambda item: (
                DEFAULT_SCALES.index(item.spec.scale) if item.spec.scale in DEFAULT_SCALES else len(DEFAULT_SCALES),
                1 if _is_standard_2x_tokens_run(item) else main_method_sort_index(item.spec.family, item.spec.window),
                -1 if item.spec.window is None else int(item.spec.window),
                item.spec.display_name,
            ),
        ):
            window = -1 if run.spec.window is None else int(run.spec.window)
            cutoff = pre_decay_token_cutoff_for_run(run) if _is_pre_decay_only_axis(x_axis) else None
            gpu_hour_values = getattr(run, "gpu_hours", None)
            if gpu_hour_values is None:
                gpu_hour_values = [None] * len(run.tokens_seen)
            for tokens_seen, gpu_hours, metric_value in zip(
                run.tokens_seen,
                gpu_hour_values,
                run.metric_values,
                strict=True,
            ):
                if cutoff is not None and int(tokens_seen) > cutoff:
                    continue
                writer.writerow(
                    [
                        run.spec.scale,
                        run.spec.family,
                        window,
                        run.spec.display_name,
                        run.run_id,
                        run.created_at,
                        int(tokens_seen),
                        _csv_float(gpu_hours),
                        f"{float(metric_value):.8f}",
                        _csv_int(getattr(run, "gpu_count", None)),
                        _csv_int(getattr(run, "calibration_start_tokens", None)),
                        _csv_int(getattr(run, "calibration_end_tokens", None)),
                        _csv_float(getattr(run, "calibration_seconds", None)),
                        _csv_float(
                            getattr(run, "gpu_hours_per_token", None),
                            precision=GPU_HOURS_PER_TOKEN_PRECISION,
                        ),
                        getattr(run, "calibration_mode", ""),
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
        help="Main-method windows to include. Defaults to 64 and 4096.",
    )
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES), choices=sorted(DEFAULT_FAMILIES))
    parser.add_argument(
        "--x-axis",
        default=X_AXIS_TOKENS,
        choices=X_AXIS_CHOICES,
        help=(
            "Plot validation history against training tokens, estimated GPU-hours, "
            "or a compact two-row token/GPU-hour comparison."
        ),
    )
    parser.add_argument(
        "--gpu-hours-reference-interval",
        type=int,
        default=DEFAULT_GPU_HOURS_REFERENCE_INTERVAL,
        help=(
            "Zero-based validation interval used to calibrate tokens to GPU-hours. "
            "The default 1 uses validation row 1 to row 2, i.e. the second validation range."
        ),
    )
    parser.add_argument(
        "--default-gpu-count",
        type=int,
        default=None,
        help="Fallback GPU count when W&B metadata.gpu_count is missing. By default, such runs are skipped.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--history-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_latex()
    output_path = args.output or _default_output_path(args.x_axis)
    history_csv = args.history_csv or output_path.with_suffix(".csv")
    if args.gpu_hours_reference_interval < 0:
        raise ValueError("--gpu-hours-reference-interval must be non-negative")
    if args.default_gpu_count is not None and args.default_gpu_count <= 0:
        raise ValueError("--default-gpu-count must be positive when provided")

    import wandb

    if not hasattr(wandb, "Api"):
        raise RuntimeError("Expected the W&B package. Run this script with `uv run python ...`.")

    print(f"Using W&B states: {sorted(USABLE_RUN_STATES)}")
    if _uses_gpu_hours_axis(args.x_axis):
        print(f"Using Standard (2x Tokens) W&B states: {sorted(STANDARD_2X_TOKENS_USABLE_RUN_STATES)}")
    api = wandb.Api()
    if _uses_gpu_hours_axis(args.x_axis):
        resolved_runs = resolve_history_runs_with_gpu_hours(
            api,
            entity=args.entity,
            project=args.project,
            scales=args.scales,
            windows=args.windows,
            families=args.families,
            reference_interval=args.gpu_hours_reference_interval,
            default_gpu_count=args.default_gpu_count,
        )
        resolved_runs.extend(
            resolve_standard_2x_tokens_runs_with_gpu_hours(
                api,
                entity=args.entity,
                project=args.project,
                scales=args.scales,
                reference_runs=resolved_runs,
            )
        )
    else:
        resolved_runs = resolve_runs(
            api,
            entity=args.entity,
            project=args.project,
            scales=args.scales,
            windows=args.windows,
            families=args.families,
        )
    resolved_runs = _filter_main_method_runs(
        resolved_runs,
        windows=args.windows,
        families=args.families,
    )
    render_plot(
        resolved_runs,
        output_path,
        scales=args.scales,
        windows=args.windows,
        families=args.families,
        x_axis=args.x_axis,
    )
    write_csv(resolved_runs, history_csv, x_axis=args.x_axis)
    history_rows = sum(
        len(_run_plot_points(run, run.spec.scale, x_axis=args.x_axis)[0])
        for run in resolved_runs
    )
    print("Saved plots to:")
    for path in _paired_image_paths(output_path):
        print(f"  {path}")
    print(f"Saved history CSV to: {history_csv}")
    print(f"Resolved runs: {len(resolved_runs)}")
    print(f"Plotted history rows: {history_rows}")
    if _uses_gpu_hours_axis(args.x_axis):
        print(f"GPU-hour reference interval: {args.gpu_hours_reference_interval}")


if __name__ == "__main__":
    main()
