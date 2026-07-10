#!/usr/bin/env python3
"""Export the main results table as LaTeX.

Also houses the generation-efficiency row model and summaries (``collect_rows``,
``render_summary_tsv``, ``render_plain_summary``, ``write_efficiency_summary``): the
table's Peak-Memory / Throughput columns are computed from a benchmark ``results.json``
here, and the benchmark driver imports the same helpers to write its per-run summary.
The paper reports these numbers inside the main results table, so there is no
standalone efficiency table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from plotting.methods import MAIN_METHOD_SPECS, MainMethodSpec, main_run_name
from plotting.eval_data import (
    CLASSIC_DEFAULT_TASKS,
    CLASSIC_TASK_METRIC_PRIORITY,
    PPL_TASK_METRIC_PRIORITY,
    classic_is_selected_eval_run,
    parse_checkpoint_tokens,
    ppl_is_selected_eval_run,
    _lookup_run_value,
    _metric_name_and_value_for_run,
)
from plotting.style import TRAINING_METRIC_KEY
from plotting.wandb_runs import (
    _parse_created_at,
    _repo_root,
    normalize_history,
)


DEFAULT_SCALES = ("xs", "s", "m", "l", "xl")
DEFAULT_OUTPUT = _repo_root() / "outputs" / "tables" / "main_results_table.tex"
# Both speed columns of the compact table (Throughput and Peak Memory) are derived from
# this single benchmark run's results.json -- see scripts/benchmark/REPRODUCE.md.
DEFAULT_RESULTS = (
    _repo_root()
    / "outputs"
    / "generation_timing_correctness"
    / "THROUGHPUT_b16_all5_h100"
    / "results.json"
)
TABLE_SCALE_SIZE_SUFFIX = {
    "xs": "20b",
    "s": "20b",
    "m": "20b",
    "l": "20b",
    "xl": "20b",
}
FINAL_TRAINING_RUN_STATES = {"finished"}
DOWNSTREAM_TASKS = CLASSIC_DEFAULT_TASKS
DOWNSTREAM_LABELS = {
    "arc_easy": "ARC-E",
    "hellaswag": "HS",
    "piqa": "PIQA",
    "sciq": "SciQ",
    "lambada_openai": "LAMBADA",
}
PPL_NLL_TASKS = ("wikitext", "c4", "pile_books3", "gov_report_nll")
PPL_NLL_LABELS = {
    "wikitext": "WT",
    "c4": "C4",
    "pile_books3": "Books3",
    "gov_report_nll": "GR",
}
GOV_REPORT_KEY = "gov_report_nll"
MISSING_VALUE = "--"


@dataclass(frozen=True)
class TableRow:
    scale: str
    method: MainMethodSpec
    run_name: str
    fineweb_edu: float | None
    downstream: dict[str, float | None]
    nll: dict[str, float | None]
    memory_ratio: float | None = None
    throughput_ratio: float | None = None  # combined prefill+decode at p=1024 / n=3072

    @property
    def downstream_avg(self) -> float | None:
        return mean_available(self.downstream.values())

    @property
    def nll_avg(self) -> float | None:
        return mean_available(self.nll.values())


METHODS = MAIN_METHOD_SPECS


def method_for_key(method_key: str) -> MainMethodSpec:
    for method in METHODS:
        if method.key == method_key:
            return method
    raise ValueError(f"Unsupported method key {method_key!r}")


def run_name_for(scale: str, method: MainMethodSpec) -> str:
    try:
        size_suffix = TABLE_SCALE_SIZE_SUFFIX[scale]
    except KeyError as exc:
        raise ValueError(f"Unsupported scale {scale!r}") from exc
    if method.family == "full_attention":
        return f"{scale}_full_attention_{size_suffix}"
    if method.window is None:
        raise ValueError(f"Windowed method {method.key!r} requires a window")
    return f"{scale}_{method.family}_w{int(method.window)}_{size_suffix}"


def run_name_candidates(scale: str, method: MainMethodSpec) -> list[str]:
    """The run name for this scale/method."""
    return [run_name_for(scale, method)]


def mean_available(values: Iterable[float | None]) -> float | None:
    finite_values = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite_values:
        return None
    return sum(finite_values) / len(finite_values)


def format_value(value: float | None, *, digits: int = 3, scale: float = 1.0) -> str:
    if value is None:
        return MISSING_VALUE
    value = float(value)
    if not math.isfinite(value):
        return MISSING_VALUE
    return f"{value * scale:.{digits}f}"


def _latex_emphasis(value: str, *, rank: int | None, include_underline: bool = False) -> str:
    if value == MISSING_VALUE:
        return value
    if rank == 0:
        return rf"{{\bfseries {value}}}"
    if include_underline and rank == 1:
        return rf"{{\underline{{{value}}}}}"
    return value


def _rank_values(values: list[float | None], *, higher_is_better: bool) -> list[int | None]:
    finite_items = [
        (idx, float(value))
        for idx, value in enumerate(values)
        if value is not None and math.isfinite(float(value))
    ]
    finite_items.sort(key=lambda item: item[1], reverse=higher_is_better)
    ranks: list[int | None] = [None] * len(values)
    rank = 0
    previous_value: float | None = None
    for position, (idx, value) in enumerate(finite_items):
        if previous_value is None or not math.isclose(value, previous_value, rel_tol=1e-12, abs_tol=1e-12):
            rank = position
            previous_value = value
        ranks[idx] = rank
    return ranks


def _row_metric_values(row: TableRow) -> list[float | None]:
    nll_values = [row.nll.get(task_name) for task_name in PPL_NLL_TASKS]
    downstream_values = [row.downstream.get(task_name) for task_name in DOWNSTREAM_TASKS]
    return [
        *nll_values,
        *downstream_values,
    ]


def _higher_is_better_by_column() -> list[bool]:
    return [
        *[False for _ in PPL_NLL_TASKS],
        *[True for _ in DOWNSTREAM_TASKS],
    ]


# Layout indices (0-based) for the metric columns in `_row_metric_values`.
_NLL_START = 0
_NLL_END = _NLL_START + len(PPL_NLL_TASKS)
_DOWN_START = _NLL_END
_DOWN_END = _DOWN_START + len(DOWNSTREAM_TASKS)


def _format_metric_cell(column_idx: int, value: float | None, rank: int | None) -> str:
    is_downstream = _DOWN_START <= column_idx < _DOWN_END
    formatted = format_value(
        value,
        digits=1 if is_downstream else 3,
        scale=100.0 if is_downstream else 1.0,
    )
    return _latex_emphasis(formatted, rank=rank, include_underline=True)


def _latest_finished_training_run(api, *, entity: str, project: str, display_names: Iterable[str]):
    for display_name in display_names:
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
            if str(getattr(run, "state", "")).lower() in FINAL_TRAINING_RUN_STATES
        ]
        if runs:
            return max(runs, key=lambda run: _parse_created_at(str(getattr(run, "created_at"))))
    return None


def resolve_final_fineweb_edu_nll(
    api, *, entity: str, project: str, display_names: Iterable[str]
) -> float | None:
    run = _latest_finished_training_run(
        api, entity=entity, project=project, display_names=display_names
    )
    if run is None:
        return None
    rows = list(
        run.scan_history(
            keys=["_step", "tokens_seen", TRAINING_METRIC_KEY],
            page_size=5000,
        )
    )
    _, metric_values = normalize_history(rows)
    if not metric_values:
        return None
    return metric_values[-1]


def _final_checkpoint_tokens(checkpoint_name: str) -> int | None:
    checkpoint_name = str(checkpoint_name)
    if not checkpoint_name.endswith("_final"):
        return None
    try:
        return parse_checkpoint_tokens(checkpoint_name)
    except ValueError:
        return None


def _resolve_final_eval_values(
    api,
    *,
    entity: str,
    project: str,
    group_names: Iterable[str],
    tasks: Iterable[str],
    is_selected_eval_run: Callable[[object], bool],
    task_metric_priorities: Mapping[str, tuple[str, ...]],
    value_transform: Callable[[str, float], float | None] | None = None,
) -> dict[str, float | None]:
    selected_tasks = tuple(tasks)
    values: dict[str, tuple[int, str, float]] = {}
    for group_name in group_names:
        for run in api.runs(
            f"{entity}/{project}",
            filters={"group": group_name},
            per_page=500,
        ):
            if not is_selected_eval_run(run):
                continue
            task_name = str(_lookup_run_value(run, "task_name", "eval_task_name") or "").strip()
            if task_name not in selected_tasks:
                continue
            checkpoint_name = str(_lookup_run_value(run, "checkpoint_name", "eval_checkpoint_name") or "")
            tokens_seen = _final_checkpoint_tokens(checkpoint_name)
            if tokens_seen is None:
                continue
            metric_payload = _metric_name_and_value_for_run(
                run,
                task_name,
                task_metric_priorities=task_metric_priorities,
            )
            if metric_payload is None:
                continue
            _, metric_value = metric_payload
            transformed_value = metric_value if value_transform is None else value_transform(task_name, metric_value)
            if transformed_value is None:
                continue
            created_at = str(getattr(run, "created_at", ""))
            current = values.get(task_name)
            candidate_key = (tokens_seen, created_at)
            current_key = None if current is None else (current[0], current[1])
            if current_key is None or candidate_key >= current_key:
                values[task_name] = (tokens_seen, created_at, transformed_value)

    return {task_name: values.get(task_name, (0, "", None))[2] for task_name in selected_tasks}


def resolve_final_downstream_values(
    api, *, entity: str, project: str, group_names: Iterable[str]
) -> dict[str, float | None]:
    return _resolve_final_eval_values(
        api,
        entity=entity,
        project=project,
        group_names=group_names,
        tasks=DOWNSTREAM_TASKS,
        is_selected_eval_run=classic_is_selected_eval_run,
        task_metric_priorities=CLASSIC_TASK_METRIC_PRIORITY,
    )


def _word_ppl_to_nll(_task_name: str, value: float) -> float | None:
    if value <= 0:
        return None
    return math.log(value)


def resolve_final_public_nll_values(
    api, *, entity: str, project: str, group_names: Iterable[str]
) -> dict[str, float | None]:
    return _resolve_final_eval_values(
        api,
        entity=entity,
        project=project,
        group_names=group_names,
        tasks=PPL_NLL_TASKS,
        is_selected_eval_run=ppl_is_selected_eval_run,
        task_metric_priorities=PPL_TASK_METRIC_PRIORITY,
        value_transform=_word_ppl_to_nll,
    )


# ---------------------------------------------------------------------------
# Generation-efficiency row model and summaries.
#
# The table's Peak-Memory and Throughput columns are derived from a benchmark
# ``results.json`` via ``collect_rows``. The benchmark driver
# (``scripts/benchmark/benchmark_generation_speed.py``) imports
# ``render_plain_summary`` / ``write_efficiency_summary`` from here to write its
# per-run summary. The paper reports these numbers inside the main results table,
# so there is no standalone efficiency table.
# ---------------------------------------------------------------------------
METHOD_ORDER = {method.key: idx for idx, method in enumerate(MAIN_METHOD_SPECS)}
SUPPORTED_SCALES = DEFAULT_SCALES


@dataclass(frozen=True)
class GenerationEfficiencyRow:
    scale: str
    prompt_len: int
    batch_size: int
    new_tokens: int
    method: MainMethodSpec
    run_name: str
    mode: str
    tokens_per_sec: float | None
    decode_tokens_per_sec: float | None
    total_ms: float | None
    prefill_ms: float | None
    decode_ms: float | None
    peak_cuda_memory_mib: float | None
    peak_prefill_mib: float | None = None
    peak_decode_mib: float | None = None
    speedup_vs_standard: float | None = None
    memory_ratio_vs_standard: float | None = None

    @property
    def tokens_per_sec_per_gib(self) -> float | None:
        # Prefer decode-only peak when available (reflects steady-state KV
        # cache cost), fall back to legacy combined peak.
        mem = self.peak_decode_mib if (
            self.peak_decode_mib is not None and math.isfinite(float(self.peak_decode_mib))
        ) else self.peak_cuda_memory_mib
        if self.tokens_per_sec is None or mem is None:
            return None
        if not math.isfinite(self.tokens_per_sec) or not math.isfinite(mem):
            return None
        if mem <= 0.0:
            return None
        return self.tokens_per_sec / (mem / 1024.0)


def _summary_mean(payload: dict[str, Any] | None) -> float | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get("mean")
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _infer_run_name(model: dict[str, Any]) -> str:
    run_name = model.get("run_name")
    if run_name:
        return str(run_name)
    checkpoint_path = model.get("checkpoint_path")
    if checkpoint_path:
        return Path(str(checkpoint_path)).parent.name
    return str(model.get("label") or model.get("kind") or "unknown")


def _run_name_variants(scale: str, method: MainMethodSpec) -> tuple[str, ...]:
    """The run name for this scale/method."""
    return (main_run_name(scale, method),)


def _method_for_run_name(run_name: str) -> tuple[str, MainMethodSpec]:
    for scale in SUPPORTED_SCALES:
        for method in MAIN_METHOD_SPECS:
            if run_name in _run_name_variants(scale, method):
                return scale, method
    raise ValueError(f"Could not map run name to main method: {run_name!r}")


def _row_sort_key(row: GenerationEfficiencyRow) -> tuple[int, int, int, int, int]:
    scale_idx = SUPPORTED_SCALES.index(row.scale) if row.scale in SUPPORTED_SCALES else len(SUPPORTED_SCALES)
    return (
        scale_idx,
        int(row.prompt_len),
        int(row.batch_size),
        int(row.new_tokens),
        METHOD_ORDER.get(row.method.key, len(METHOD_ORDER)),
    )


def _peak_memory_for_table(row: GenerationEfficiencyRow) -> float | None:
    """Pick the peak-memory number to report in the table.

    Prefer decode-only peak when available (reflects KV cache size, the thing
    we want to compare across methods). Fall back to legacy combined peak so
    older results.json files still render.
    """
    if row.peak_decode_mib is not None and math.isfinite(float(row.peak_decode_mib)):
        return row.peak_decode_mib
    return row.peak_cuda_memory_mib


def _with_relative_metrics(rows: list[GenerationEfficiencyRow]) -> list[GenerationEfficiencyRow]:
    standard_by_group = {
        (row.scale, row.prompt_len, row.batch_size, row.new_tokens): row
        for row in rows
        if row.method.key == "standard"
    }
    updated: list[GenerationEfficiencyRow] = []
    for row in rows:
        standard = standard_by_group.get((row.scale, row.prompt_len, row.batch_size, row.new_tokens))
        speedup = None
        memory_ratio = None
        if standard is not None:
            if (
                row.tokens_per_sec is not None
                and standard.tokens_per_sec is not None
                and standard.tokens_per_sec > 0.0
            ):
                speedup = row.tokens_per_sec / standard.tokens_per_sec
            row_mem = _peak_memory_for_table(row)
            std_mem = _peak_memory_for_table(standard)
            if (
                row_mem is not None
                and std_mem is not None
                and std_mem > 0.0
            ):
                memory_ratio = row_mem / std_mem
        updated.append(
            GenerationEfficiencyRow(
                scale=row.scale,
                prompt_len=row.prompt_len,
                batch_size=row.batch_size,
                new_tokens=row.new_tokens,
                method=row.method,
                run_name=row.run_name,
                mode=row.mode,
                tokens_per_sec=row.tokens_per_sec,
                decode_tokens_per_sec=row.decode_tokens_per_sec,
                total_ms=row.total_ms,
                prefill_ms=row.prefill_ms,
                decode_ms=row.decode_ms,
                peak_cuda_memory_mib=row.peak_cuda_memory_mib,
                peak_prefill_mib=row.peak_prefill_mib,
                peak_decode_mib=row.peak_decode_mib,
                speedup_vs_standard=speedup,
                memory_ratio_vs_standard=memory_ratio,
            )
        )
    return updated


def collect_rows(payload: dict[str, Any]) -> list[GenerationEfficiencyRow]:
    rows: list[GenerationEfficiencyRow] = []
    for model in payload.get("models", []):
        run_name = _infer_run_name(model)
        scale, method = _method_for_run_name(run_name)
        for result in model.get("prompt_length_results", []):
            timing = result.get("timing", {})
            rows.append(
                GenerationEfficiencyRow(
                    scale=scale,
                    prompt_len=int(result["prompt_len"]),
                    batch_size=int(result.get("batch_size", result.get("prompt_count", 1))),
                    new_tokens=int(result.get("new_tokens", 0)),
                    method=method,
                    run_name=run_name,
                    mode=str(result.get("mode") or ""),
                    tokens_per_sec=_summary_mean(timing.get("tokens_per_sec")),
                    decode_tokens_per_sec=_summary_mean(timing.get("decode_tokens_per_sec")),
                    total_ms=_summary_mean(timing.get("total_ms")),
                    prefill_ms=_summary_mean(timing.get("prefill_ms")),
                    decode_ms=_summary_mean(timing.get("decode_ms")),
                    peak_cuda_memory_mib=_summary_mean(timing.get("peak_cuda_memory_mib")),
                    peak_prefill_mib=_summary_mean(timing.get("peak_prefill_mib")),
                    peak_decode_mib=_summary_mean(timing.get("peak_decode_mib")),
                )
            )
    return _with_relative_metrics(sorted(rows, key=_row_sort_key))


def _format_float(value: float | None, digits: int) -> str:
    if value is None or not math.isfinite(float(value)):
        return MISSING_VALUE
    return f"{float(value):.{digits}f}"


def _format_ratio(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return MISSING_VALUE
    return f"{float(value):.2f}x"


def render_summary_tsv(rows: Iterable[GenerationEfficiencyRow]) -> str:
    lines = [
        "\t".join(
            [
                "scale",
                "prompt_len",
                "batch_size",
                "new_tokens",
                "method_key",
                "method",
                "run_name",
                "mode",
                "tokens_per_sec",
                "decode_tokens_per_sec",
                "speedup_vs_standard",
                "total_ms",
                "prefill_ms",
                "decode_ms",
                "peak_cuda_memory_mib",
                "peak_prefill_mib",
                "peak_decode_mib",
                "memory_ratio_vs_standard",
                "tokens_per_sec_per_gib",
            ]
        )
    ]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    row.scale,
                    str(row.prompt_len),
                    str(row.batch_size),
                    str(row.new_tokens),
                    row.method.key,
                    row.method.label,
                    row.run_name,
                    row.mode,
                    _format_float(row.tokens_per_sec, 6),
                    _format_float(row.decode_tokens_per_sec, 6),
                    _format_float(row.speedup_vs_standard, 6),
                    _format_float(row.total_ms, 6),
                    _format_float(row.prefill_ms, 6),
                    _format_float(row.decode_ms, 6),
                    _format_float(row.peak_cuda_memory_mib, 6),
                    _format_float(row.peak_prefill_mib, 6),
                    _format_float(row.peak_decode_mib, 6),
                    _format_float(row.memory_ratio_vs_standard, 6),
                    _format_float(row.tokens_per_sec_per_gib, 6),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def render_plain_summary(rows: Iterable[GenerationEfficiencyRow]) -> str:
    output = [
        "Generation efficiency summary",
        "Scale Prompt Batch NewTok Method         Tok/s  DecTok/s Speedup  Total ms  Prefill  Decode  Decode MiB  Mem x  Tok/s/GiB",
    ]
    for row in rows:
        mem = _peak_memory_for_table(row)
        output.append(
            f"{row.scale.upper():>5} {row.prompt_len:>6} "
            f"{row.batch_size:>5} {row.new_tokens:>6} "
            f"{row.method.label:<12} "
            f"{_format_float(row.tokens_per_sec, 1):>7} "
            f"{_format_float(row.decode_tokens_per_sec, 1):>8} "
            f"{_format_ratio(row.speedup_vs_standard):>8} "
            f"{_format_float(row.total_ms, 1):>9} "
            f"{_format_float(row.prefill_ms, 1):>8} "
            f"{_format_float(row.decode_ms, 1):>7} "
            f"{_format_float(mem, 1):>11} "
            f"{_format_ratio(row.memory_ratio_vs_standard):>6} "
            f"{_format_float(row.tokens_per_sec_per_gib, 1):>10}"
        )
    return "\n".join(output) + "\n"


def write_efficiency_summary(
    payload: dict[str, Any],
    *,
    summary_tsv: Path,
) -> list[GenerationEfficiencyRow]:
    """Collect efficiency rows from a benchmark payload and write the TSV summary."""
    rows = collect_rows(payload)
    summary_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary_tsv.write_text(render_summary_tsv(rows))
    return rows


def load_throughput_lookup(path: Path | None) -> dict[tuple[str, str], float | None]:
    """Read a combined-workload `results.json` and return {(scale, method_key): throughput_ratio}.

    The throughput for a given (scale, method) is `(prompt_len + new_tokens) / total_s`
    where total_s = prefill_ms + decode_ms. The ratio is
    `method.tokens_per_sec / standard.tokens_per_sec` within the same scale, so
    Standard is always 1.00 and higher is better. Used for the merged "Throughput
    (prefill 1k + decode 3k)" column.
    """
    if path is None:
        return {}
    if not Path(path).exists():
        print(
            f"WARNING: latency results.json not found at {path!r}; rendering with `--` in latency column"
        )
        return {}
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not load latency results.json at {path!r}: {exc}")
        return {}

    def _mean(x):
        return float(x["mean"]) if isinstance(x, dict) else float(x)

    label_to_key = {method.label: method.key for method in METHODS}
    throughputs: dict[tuple[str, str], float] = {}
    for model in payload.get("models", []):
        scale = model["run_name"].split("_")[0]
        method_key = label_to_key.get(model.get("label"))
        if method_key is None:
            continue
        plr = model.get("prompt_length_results") or []
        if not plr:
            continue
        # Pick the largest (prompt_len, batch_size, new_tokens) entry — mirrors
        # other loaders.
        chosen = max(
            plr,
            key=lambda r: (int(r.get("prompt_len", 0)), int(r.get("batch_size", 0)), int(r.get("new_tokens", 0))),
        )
        timing = chosen.get("timing", {})
        try:
            total_ms = _mean(timing["prefill_ms"]) + _mean(timing["decode_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(total_ms) or total_ms <= 0:
            continue
        prompt_count = int(chosen.get("prompt_count", chosen.get("batch_size", 0)))
        seq_tokens = int(chosen.get("prompt_len", 0)) + int(chosen.get("new_tokens", 0))
        if prompt_count <= 0 or seq_tokens <= 0:
            continue
        throughputs[(scale, method_key)] = (prompt_count * seq_tokens) / (total_ms / 1000.0)
    standards = {scale: tps for (scale, key), tps in throughputs.items() if key == "standard"}
    out: dict[tuple[str, str], float | None] = {}
    for (scale, method_key), tps in throughputs.items():
        std = standards.get(scale)
        out[(scale, method_key)] = (tps / std) if (std and std > 0) else None
    return out


def load_memory_lookup(path: Path | None) -> dict[tuple[str, str], float | None]:
    """Read the speed bench `results.json` and return {(scale, method_key): mem_ratio}.

    Peak-memory ratio is `method.peak_bytes / standard.peak_bytes` within a scale
    (computed by collect_rows). Missing path or
    unreadable file -> empty dict + a warning, so the table still renders (with `--`
    in the Peak Memory column) when the bench output isn't available.
    """
    if path is None:
        return {}
    if not Path(path).exists():
        print(f"WARNING: results.json not found at {path!r}; rendering without the Peak Memory column")
        return {}
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: could not load results.json at {path!r}: {exc}")
        return {}
    rows = collect_rows(payload)
    grouped: dict[tuple[str, str], list[GenerationEfficiencyRow]] = {}
    for row in rows:
        grouped.setdefault((row.scale, row.method.key), []).append(row)
    out: dict[tuple[str, str], float | None] = {}
    for key, group in grouped.items():
        # If multiple (prompt_len, batch_size, new_tokens) groups exist for the
        # same (scale, method), pick the largest tuple — the headline number.
        chosen = max(group, key=lambda r: (int(r.prompt_len), int(r.batch_size), int(r.new_tokens)))
        out[key] = chosen.memory_ratio_vs_standard
    return out


def collect_table_rows(
    api,
    *,
    entity: str,
    project: str,
    scales: Iterable[str] = DEFAULT_SCALES,
    repo_root: Path | None = None,
    memory_lookup: Mapping[tuple[str, str], float | None] | None = None,
    throughput_lookup: Mapping[tuple[str, str], float | None] | None = None,
) -> list[TableRow]:
    del repo_root
    memory_lookup = memory_lookup or {}
    throughput_lookup = throughput_lookup or {}
    rows: list[TableRow] = []
    for scale in scales:
        for method in METHODS:
            run_name = run_name_for(scale, method)
            name_candidates = run_name_candidates(scale, method)
            fineweb_edu = resolve_final_fineweb_edu_nll(
                api,
                entity=entity,
                project=project,
                display_names=name_candidates,
            )
            if fineweb_edu is None:
                print(f"WARNING: skipping unfinished or missing training run {run_name!r}")
                continue
            nll_values = resolve_final_public_nll_values(
                api,
                entity=entity,
                project=project,
                group_names=name_candidates,
            )
            memory_ratio = memory_lookup.get((scale, method.key))
            throughput_ratio = throughput_lookup.get((scale, method.key))
            rows.append(
                TableRow(
                    scale=scale,
                    method=method,
                    run_name=run_name,
                    fineweb_edu=fineweb_edu,
                    downstream=resolve_final_downstream_values(
                        api,
                        entity=entity,
                        project=project,
                        group_names=name_candidates,
                    ),
                    nll=nll_values,
                    memory_ratio=memory_ratio,
                    throughput_ratio=throughput_ratio,
                )
            )
    return rows


def _ranked_metric_cells(rows_for_scale: list[TableRow]) -> dict[str, list[str]]:
    values_by_row = [_row_metric_values(row) for row in rows_for_scale]
    if not values_by_row:
        return {}

    rank_by_column = [
        _rank_values(
            [row_values[column_idx] for row_values in values_by_row],
            higher_is_better=higher_is_better,
        )
        for column_idx, higher_is_better in enumerate(_higher_is_better_by_column())
    ]

    cells_by_method: dict[str, list[str]] = {}
    for row_idx, row in enumerate(rows_for_scale):
        row_values = values_by_row[row_idx]
        cells_by_method[row.method.key] = [
            _format_metric_cell(column_idx, value, rank_by_column[column_idx][row_idx])
            for column_idx, value in enumerate(row_values)
        ]
    return cells_by_method


def _render_size_block(
    *,
    scale: str,
    rows_for_scale: list[TableRow],
    cells_by_method: dict[str, list[str]],
    is_first_block: bool,
    inter_size_separator: str,
) -> list[str]:
    lines: list[str] = []
    if not is_first_block:
        lines.extend(
            [
                r"        \addlinespace[3pt]",
                f"        {inter_size_separator}",
                r"        \addlinespace[3pt]",
            ]
        )
    lines.append(f"        % ---------- {scale.upper()} ----------")
    lines.append(rf"        \multirow{{{len(rows_for_scale)}}}{{*}}{{\textbf{{{scale.upper()}}}}}")
    for row in rows_for_scale:
        method = row.method
        if method.key == "sps":
            lines.append(r"        \rowcolor{winrow}")
        cells = " & ".join([method.label, *cells_by_method.get(method.key, [])])
        lines.append(f"         & {cells} \\\\")
    return lines


def render_latex_table(rows: list[TableRow], *, scales: Iterable[str] | None = None) -> str:
    selected_scales = tuple(DEFAULT_SCALES if scales is None else scales)
    row_lookup = {(row.scale, row.method.key): row for row in rows}
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \vspace*{2em}",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{6pt}",
        r"    \renewcommand{\arraystretch}{1.2}",
        r"    \begin{tabular}{@{}c l",
        r"        S[table-format=1.3] S[table-format=1.3] S[table-format=1.3] S[table-format=1.3]",
        r"        S[table-format=2.1] S[table-format=2.1] S[table-format=2.1] S[table-format=2.1] S[table-format=2.1]@{}}",
        r"        \toprule",
        r"        & & \multicolumn{4}{c}{NLL Generalization ($\downarrow$)}",
        r"        & \multicolumn{5}{c}{Task Generalization (\%, $\uparrow$)} \\",
        r"        \cmidrule(lr){3-6} \cmidrule(lr){7-11}",
        r"        Size & Method",
        r"        & {WT} & {C4} & {Books3} & {GR}",
        r"        & {ARC-E} & {HS} & {PIQA} & {SciQ} & {LAMB} \\",
        r"        \midrule",
    ]

    inter_size_separator = (
        r"\arrayrulecolor{groupline}\cmidrule[0.3pt](l{2pt}r{2pt}){1-11}\arrayrulecolor{black}"
    )
    rendered_scale_count = 0
    for scale in selected_scales:
        rows_for_scale = [
            row
            for method in METHODS
            for row in [row_lookup.get((scale, method.key))]
            if row is not None
        ]
        if not rows_for_scale:
            continue
        cells_by_method = _ranked_metric_cells(rows_for_scale)
        lines.extend(
            _render_size_block(
                scale=scale,
                rows_for_scale=rows_for_scale,
                cells_by_method=cells_by_method,
                is_first_block=(rendered_scale_count == 0),
                inter_size_separator=inter_size_separator,
            )
        )
        rendered_scale_count += 1

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \label{tab:main-full}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def _compact_metric_values(row: TableRow) -> list[float | None]:
    """Return the 5 metric values shown in the compact table, in column order:
    FW-E, Corpus NLL, Task Acc., Throughput (prefill 1k + decode 3k), Mem.
    """
    return [
        row.fineweb_edu,
        row.nll_avg,
        row.downstream_avg,
        row.throughput_ratio,
        row.memory_ratio,
    ]


def _compact_higher_is_better() -> list[bool]:
    # FW-E (lower better), Corpus NLL (lower), Task Acc (higher),
    # Throughput (higher), Memory (lower).
    return [False, False, True, True, False]


def _format_compact_cell(
    column_idx: int,
    value: float | None,
    rank: int | None,
) -> str:
    if column_idx == 0:  # FW-E
        formatted = format_value(value, digits=3)
    elif column_idx == 1:  # Corpus NLL
        formatted = format_value(value, digits=3)
    elif column_idx == 2:  # Task Acc.
        formatted = format_value(value, digits=1, scale=100.0)
    else:  # Throughput / Mem. — efficiency columns get no rank emphasis.
        if value is None or not math.isfinite(float(value)):
            return MISSING_VALUE
        return f"{float(value):.2f}"
    return _latex_emphasis(formatted, rank=rank)


# Compact-table columns that get a bracketed delta-vs-Standard annotation. All
# six metric columns are annotated: the three quality columns carry absolute
# deltas, and the three efficiency columns carry the (ratio - 1) delta so the
# coloring is consistent across the whole row.
_COMPACT_ABSOLUTE_COLS = (0, 1, 2, 3, 4)


# Three-stop palette for the smooth red -> amber -> green gradient. Endpoints are
# pinned at norm=0 (worst non-Standard) and norm=1 (best non-Standard); the middle
# stop kicks in at norm=0.5. Using xcolor's [rgb] selector so no preamble setup
# is needed.
_DELTA_COLOR_RED = (0.75, 0.15, 0.15)
_DELTA_COLOR_AMBER = (0.85, 0.65, 0.10)
_DELTA_COLOR_GREEN = (0.20, 0.60, 0.20)


def _color_from_norm(norm: float) -> str:
    """Map a normalized goodness `norm` in [0, 1] to a `[rgb]{r,g,b}` color spec.

    norm=0  -> red (worst non-Standard delta in this column)
    norm=.5 -> amber
    norm=1  -> green (best non-Standard delta)
    Two cells with the same `norm` get the same color string, so e.g. two methods
    that tie at +2.2 percentage points render with identical color.
    """
    norm = 0.0 if norm != norm else max(0.0, min(1.0, float(norm)))  # NaN -> 0
    if norm <= 0.5:
        t = norm * 2.0
        c0, c1 = _DELTA_COLOR_RED, _DELTA_COLOR_AMBER
    else:
        t = (norm - 0.5) * 2.0
        c0, c1 = _DELTA_COLOR_AMBER, _DELTA_COLOR_GREEN
    r = c0[0] + (c1[0] - c0[0]) * t
    g = c0[1] + (c1[1] - c0[1]) * t
    b = c0[2] + (c1[2] - c0[2]) * t
    return rf"[rgb]{{{r:.2f},{g:.2f},{b:.2f}}}"


def _format_compact_delta(column_idx: int, delta: float) -> str:
    """Format a delta vs Standard for inclusion next to a cell value.

    Renders inside math mode so the sign uses a proper minus glyph rather than a
    hyphen. Returned text is meant to be concatenated after the formatted value.
    """
    if column_idx == 2:  # Task Accuracy: percentage points, 1 decimal.
        return rf"${delta * 100.0:+.1f}$"
    if column_idx == 3:  # Throughput ratio: 2 decimals (e.g. +0.05).
        return rf"${delta:+.2f}$"
    if column_idx == 4:  # Memory ratio: percent (e.g. +89\%).
        return rf"${delta * 100.0:+.0f}\%$"
    # Validation Loss / Corpus NLL: 3 decimals, raw NLL units.
    return rf"${delta:+.3f}$"


def _annotate_with_delta(
    cell: str, column_idx: int, delta: float, norm: float | None
) -> str:
    """Wrap a non-Standard cell so the delta vs Standard appears in brackets.

    Uses an outer `{...}` so siunitx S columns treat the cell as text rather than
    parsing the parenthesized delta as part of the number. `norm` is unused (was
    used previously for a red->green coloring scheme).
    """
    del norm  # coloring removed
    if cell == MISSING_VALUE:
        return cell
    delta_str = _format_compact_delta(column_idx, delta)
    return rf"{{{cell}\,\scriptsize {{({delta_str})}}}}"


def _compute_compact_delta_norms(
    rows_for_scale: list[TableRow],
    values_by_row: list[list[float | None]],
    standard_idx: int,
) -> dict[tuple[int, int], float]:
    """For each absolute column, normalize the non-Standard goodness to [0, 1].

    `goodness` is direction-aware: for lower-is-better columns it is `-(delta)`,
    for higher-is-better it is `+delta`. Linear normalization between the min and
    max goodness across the non-Standard methods means tied deltas produce
    identical norms (and thus identical colors). When all non-Standard methods tie,
    everyone gets norm=0.5 (amber neutral).
    """
    higher_by_col = _compact_higher_is_better()
    non_std_indices = [
        idx for idx, row in enumerate(rows_for_scale) if row.method.key != "standard"
    ]
    out: dict[tuple[int, int], float] = {}
    for col_idx in _COMPACT_ABSOLUTE_COLS:
        std_value = values_by_row[standard_idx][col_idx]
        if std_value is None or not math.isfinite(float(std_value)):
            continue
        higher = higher_by_col[col_idx]
        goodness_by_idx: dict[int, float] = {}
        for row_idx in non_std_indices:
            value = values_by_row[row_idx][col_idx]
            if value is None or not math.isfinite(float(value)):
                continue
            delta = float(value) - float(std_value)
            goodness_by_idx[row_idx] = delta if higher else -delta
        if not goodness_by_idx:
            continue
        gmin = min(goodness_by_idx.values())
        gmax = max(goodness_by_idx.values())
        spread = gmax - gmin
        for row_idx, goodness in goodness_by_idx.items():
            if spread <= 0.0:
                out[(col_idx, row_idx)] = 0.5  # all tied -> neutral
            else:
                out[(col_idx, row_idx)] = (goodness - gmin) / spread
    return out


def _ranked_compact_cells(rows_for_scale: list[TableRow]) -> dict[str, list[str]]:
    values_by_row = [_compact_metric_values(row) for row in rows_for_scale]
    if not values_by_row:
        return {}
    rank_by_column = [
        _rank_values(
            [row_values[col_idx] for row_values in values_by_row],
            higher_is_better=higher,
        )
        for col_idx, higher in enumerate(_compact_higher_is_better())
    ]

    standard_idx = next(
        (idx for idx, row in enumerate(rows_for_scale) if row.method.key == "standard"),
        None,
    )
    delta_norm_by_cell = (
        _compute_compact_delta_norms(rows_for_scale, values_by_row, standard_idx)
        if standard_idx is not None
        else {}
    )

    cells_by_method: dict[str, list[str]] = {}
    for row_idx, row in enumerate(rows_for_scale):
        row_cells: list[str] = []
        for col_idx, value in enumerate(values_by_row[row_idx]):
            cell = _format_compact_cell(col_idx, value, rank_by_column[col_idx][row_idx])
            if (
                row.method.key != "standard"
                and standard_idx is not None
                and col_idx in _COMPACT_ABSOLUTE_COLS
                and value is not None
            ):
                std_value = values_by_row[standard_idx][col_idx]
                if (
                    std_value is not None
                    and math.isfinite(float(value))
                    and math.isfinite(float(std_value))
                ):
                    cell = _annotate_with_delta(
                        cell,
                        col_idx,
                        float(value) - float(std_value),
                        delta_norm_by_cell.get((col_idx, row_idx)),
                    )
            row_cells.append(cell)
        cells_by_method[row.method.key] = row_cells
    return cells_by_method


def render_latex_table_compact(
    rows: list[TableRow], *, scales: Iterable[str] | None = None
) -> str:
    """Compact 7-column table: Size | Method | FW-E | Corpus NLL | Task Acc. | Throughput | Mem."""
    selected_scales = tuple(DEFAULT_SCALES if scales is None else scales)
    row_lookup = {(row.scale, row.method.key): row for row in rows}
    lines = [
        r"\begin{table}[ht]",
        r"    \centering",
        r"    \vspace*{2em}",
        r"    \setlength{\tabcolsep}{6pt}",
        r"    \renewcommand{\arraystretch}{1.2}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{@{}c l",
        r"        S[table-format=1.3]",
        r"        S[table-format=1.3] S[table-format=2.1]",
        r"        S[table-format=1.2] S[table-format=1.2]@{}}",
        r"        \toprule",
        r"        & & {\textsc{Validation Loss}}",
        r"        & \multicolumn{2}{c}{\textsc{Generalization}}",
        r"        & \multicolumn{2}{c}{\textsc{Inference Efficiency}} \\",
        r"        \cmidrule(lr){3-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"        Size & Method",
        r"        & {FineWeb-Edu ($\downarrow$)}",
        r"        & {Corpus NLL ($\downarrow$)}",
        r"        & {Task Accuracy (\%, $\uparrow$)}",
        r"        & {Throughput ($\times$, $\uparrow$)}",
        r"        & {Peak Memory ($\times$, $\downarrow$)} \\",
        r"        \midrule",
    ]

    inter_size_separator = (
        r"\arrayrulecolor{groupline}\cmidrule[0.3pt](l{2pt}r{2pt}){1-7}\arrayrulecolor{black}"
    )
    rendered_scale_count = 0
    for scale in selected_scales:
        rows_for_scale = [
            row
            for method in METHODS
            for row in [row_lookup.get((scale, method.key))]
            if row is not None
        ]
        if not rows_for_scale:
            continue
        cells_by_method = _ranked_compact_cells(rows_for_scale)
        lines.extend(
            _render_size_block(
                scale=scale,
                rows_for_scale=rows_for_scale,
                cells_by_method=cells_by_method,
                is_first_block=(rendered_scale_count == 0),
                inter_size_separator=inter_size_separator,
            )
        )
        rendered_scale_count += 1

    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"    \label{tab:main-compact}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"), help="W&B entity")
    parser.add_argument("--project", default="pretraining_compression", help="W&B project")
    parser.add_argument("--scales", nargs="+", default=list(DEFAULT_SCALES), choices=DEFAULT_SCALES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--compact-output",
        type=Path,
        default=DEFAULT_OUTPUT.with_name("main_results_table_compact.tex"),
        help="Path for the compact (downstream-Avg-only, NLL-Avg-only) sibling table.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path to the speed benchmark results.json feeding BOTH speed columns "
        "(Throughput and Peak Memory). See scripts/benchmark/REPRODUCE.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import wandb

    if not hasattr(wandb, "Api"):
        raise RuntimeError("Expected the W&B package. Run this script with `uv run python ...`.")

    api = wandb.Api()
    memory_lookup = load_memory_lookup(args.results_json)
    throughput_lookup = load_throughput_lookup(args.results_json)
    rows = collect_table_rows(
        api,
        entity=args.entity,
        project=args.project,
        scales=args.scales,
        memory_lookup=memory_lookup,
        throughput_lookup=throughput_lookup,
    )
    table = render_latex_table(rows, scales=args.scales)
    compact_table = render_latex_table_compact(rows, scales=args.scales)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.compact_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    args.compact_output.write_text(compact_table)
    print(table, end="")
    print(compact_table, end="")
    print(f"\nWrote full table to:    {args.output}")
    print(f"Wrote compact table to: {args.compact_output}")


if __name__ == "__main__":
    main()
