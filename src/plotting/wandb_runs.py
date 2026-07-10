"""W&B run resolution and validation-history normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from plotting.style import (
    DEFAULT_FAMILIES,
    DEFAULT_SCALES,
    DEFAULT_WINDOWS,
    TRAINING_METRIC_KEY,
)


SEGMENT_JUMP_MULTIPLIER = 10
SEGMENT_MIN_LARGE_JUMP_TOKENS = 1_000_000_000
FULL_ATTENTION_WINDOW_SENTINEL = -1
USABLE_RUN_STATES = {"finished"}
FULL_ATTENTION_SIZE_SUFFIX_BY_SCALE = {
    "xs": "20b",
    "s": "20b",
    "m": "20b",
    "l": "20b",
    "xl": "20b",
}
WINDOWED_SIZE_SUFFIX_BY_SCALE = {
    "xs": "20b",
    "s": "20b",
    "m": "20b",
    "l": "20b",
    "xl": "20b",
}
DEFAULT_WINDOWS_BY_SCALE = {
    "xs": (0, 16, 64, 256, 4096),
    "s": (0, 16, 64, 256, 4096),
    "m": DEFAULT_WINDOWS,
    "l": DEFAULT_WINDOWS,
    "xl": DEFAULT_WINDOWS,
}

HistoryPoint = tuple[int, int, Any]


@dataclass(frozen=True)
class RunSpec:
    scale: str
    family: str
    window: int | None
    display_name: str


@dataclass(frozen=True)
class ResolvedRun:
    spec: RunSpec
    run_id: str
    created_at: str
    tokens_seen: list[int]
    metric_values: list[float]


@dataclass(frozen=True)
class FinalPoint:
    scale: str
    family: str
    window: int
    display_name: str
    run_id: str
    created_at: str
    tokens_seen: int
    final_nll: float
    final_nll_err: float | None = None  # 1 SE of the eval mean (None when missing)


def _repo_root() -> Path:
    # This module lives in ``src/plotting/``; the repo root is two levels up.
    return Path(__file__).resolve().parents[2]


def _parse_created_at(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def _large_forward_jump_threshold(points: list[HistoryPoint]) -> int | None:
    positive_deltas = [
        int(tokens_seen) - int(prev_tokens_seen)
        for (prev_tokens_seen, _prev_step, _prev_payload), (tokens_seen, _step, _payload) in zip(
            points,
            points[1:],
            strict=False,
        )
        if int(tokens_seen) > int(prev_tokens_seen)
    ]
    if not positive_deltas:
        return None
    normal_delta = min(positive_deltas)
    return max(
        normal_delta * SEGMENT_JUMP_MULTIPLIER,
        SEGMENT_MIN_LARGE_JUMP_TOKENS,
    )


def _dedupe_segment_by_token(segment: list[HistoryPoint]) -> list[HistoryPoint]:
    deduped: dict[int, HistoryPoint] = {}
    for point in segment:
        deduped[int(point[0])] = point
    return [deduped[tokens_seen] for tokens_seen in sorted(deduped)]


def filter_rewound_history_segments(points: list[HistoryPoint]) -> list[HistoryPoint]:
    """Drop abandoned rewind segments while preserving real resumed overlap."""
    if not points:
        return []

    ordered_points = [
        point
        for _original_index, point in sorted(
            enumerate(points),
            key=lambda item: (int(item[1][1]), item[0]),
        )
    ]
    jump_threshold = _large_forward_jump_threshold(ordered_points)

    segments: list[list[HistoryPoint]] = []
    current_segment: list[HistoryPoint] = []
    previous_tokens: int | None = None
    for point in ordered_points:
        tokens_seen = int(point[0])
        if previous_tokens is not None:
            token_delta = tokens_seen - previous_tokens
            if token_delta < 0 or (
                jump_threshold is not None and token_delta > jump_threshold
            ):
                segments.append(current_segment)
                current_segment = []
        current_segment.append(point)
        previous_tokens = tokens_seen
    if current_segment:
        segments.append(current_segment)

    accepted: list[HistoryPoint] = []
    accepted_max_tokens: int | None = None
    for segment in segments:
        deduped_segment = _dedupe_segment_by_token(segment)
        if not deduped_segment:
            continue
        segment_start_tokens = int(deduped_segment[0][0])
        segment_max_tokens = int(deduped_segment[-1][0])
        if accepted_max_tokens is not None and segment_max_tokens <= accepted_max_tokens:
            continue
        if accepted_max_tokens is not None and segment_start_tokens <= accepted_max_tokens:
            accepted = [
                point for point in accepted if int(point[0]) < segment_start_tokens
            ]
        accepted.extend(deduped_segment)
        accepted_max_tokens = segment_max_tokens

    return accepted


def normalize_history(rows: Iterable[dict[str, object]]) -> tuple[list[int], list[float]]:
    normalized: list[HistoryPoint] = []
    for row in rows:
        tokens_seen = row.get("tokens_seen")
        metric_value = row.get(TRAINING_METRIC_KEY)
        if tokens_seen is None or metric_value is None:
            continue
        step_raw = row.get("_step", 0)
        normalized.append((int(tokens_seen), int(step_raw or 0), float(metric_value)))

    filtered = filter_rewound_history_segments(normalized)
    return (
        [int(tokens_seen) for tokens_seen, _step, _metric_value in filtered],
        [float(metric_value) for _tokens_seen, _step, metric_value in filtered],
    )


def run_specs_for_scale(
    scale: str,
    *,
    windows: Iterable[int] | None = None,
    families: Iterable[str] = DEFAULT_FAMILIES,
) -> list[RunSpec]:
    try:
        full_attention_size = FULL_ATTENTION_SIZE_SUFFIX_BY_SCALE[scale]
    except KeyError as exc:
        raise ValueError(f"Unknown scale {scale!r}; add it to FULL_ATTENTION_SIZE_SUFFIX_BY_SCALE") from exc
    specs = [
        RunSpec(
            scale=scale,
            family="full_attention",
            window=None,
            display_name=f"{scale}_full_attention_{full_attention_size}",
        )
    ]
    try:
        windowed_size = WINDOWED_SIZE_SUFFIX_BY_SCALE[scale]
    except KeyError as exc:
        raise ValueError(f"Unknown scale {scale!r}; add it to WINDOWED_SIZE_SUFFIX_BY_SCALE") from exc
    selected_windows = windows_for_scale(scale, windows)
    for family in families:
        if family not in DEFAULT_FAMILIES:
            raise ValueError(f"Unsupported windowed family: {family!r}")
        for window in selected_windows:
            specs.append(
                RunSpec(
                    scale=scale,
                    family=family,
                    window=int(window),
                    display_name=f"{scale}_{family}_w{int(window)}_{windowed_size}",
                )
            )
    return specs


def windows_for_scale(scale: str, windows_override: Iterable[int] | None = None) -> tuple[int, ...]:
    if windows_override is not None:
        windows = tuple(int(window) for window in windows_override)
    else:
        try:
            windows = tuple(int(window) for window in DEFAULT_WINDOWS_BY_SCALE[scale])
        except KeyError as exc:
            raise ValueError(f"Unknown scale {scale!r}; add it to DEFAULT_WINDOWS_BY_SCALE") from exc
    if not windows:
        raise ValueError("At least one window is required")
    return windows


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _candidate_display_names(spec: RunSpec) -> list[str]:
    return [spec.display_name]


def _latest_run(api, *, entity: str, project: str, spec: RunSpec):
    last_states: list[str] = []
    for display_name in _candidate_display_names(spec):
        candidates = list(
            api.runs(
                f"{entity}/{project}",
                filters={"displayName": display_name},
                per_page=50,
            )
        )
        runs = [
            run
            for run in candidates
            if str(getattr(run, "state", "")).lower() in USABLE_RUN_STATES
        ]
        if runs:
            return max(runs, key=lambda run: _parse_created_at(str(getattr(run, "created_at"))))
        if candidates:
            last_states = sorted({str(getattr(run, "state", "")).lower() or "<unset>" for run in candidates})
    if last_states:
        _warn(
            f"W&B run displayName={spec.display_name!r} (and old-name fallback) has no "
            f"usable state in {sorted(USABLE_RUN_STATES)}; saw states={last_states}"
        )
    return None


def resolve_runs(
    api,
    *,
    entity: str,
    project: str,
    scales: Iterable[str] = DEFAULT_SCALES,
    windows: Iterable[int] | None = None,
    families: Iterable[str] = DEFAULT_FAMILIES,
) -> list[ResolvedRun]:
    resolved: list[ResolvedRun] = []
    for scale in scales:
        for spec in run_specs_for_scale(scale, windows=windows, families=families):
            run = _latest_run(api, entity=entity, project=project, spec=spec)
            if run is None:
                _warn(f"no usable W&B training run found for displayName={spec.display_name!r}")
                continue
            rows = list(
                run.scan_history(
                    keys=["_step", "tokens_seen", TRAINING_METRIC_KEY],
                    page_size=5000,
                )
            )
            tokens_seen, metric_values = normalize_history(rows)
            if not tokens_seen:
                _warn(
                    f"W&B run displayName={spec.display_name!r} has no usable "
                    f"{TRAINING_METRIC_KEY!r} history"
                )
                continue
            resolved.append(
                ResolvedRun(
                    spec=spec,
                    run_id=str(getattr(run, "id")),
                    created_at=str(getattr(run, "created_at")),
                    tokens_seen=tokens_seen,
                    metric_values=metric_values,
                )
            )
    return resolved
