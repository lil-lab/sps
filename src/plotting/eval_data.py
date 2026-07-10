"""W&B eval-run lookup helpers for downstream-accuracy and public-perplexity tables.

Extracted from the (deleted) plot_m_classic_benchmarks_wandb /
plot_perplexity_public_benchmarks_wandb scripts. The two ``*_is_selected_eval_run``
variants differ only in the ``eval_profile`` they accept.
"""

from __future__ import annotations

import json
import re
from typing import Mapping


# --- Classic downstream-accuracy task config ----------------------------------
CLASSIC_DEFAULT_TASKS = (
    "arc_easy",
    "hellaswag",
    "piqa",
    "sciq",
    "lambada_openai",
)
CLASSIC_TASK_METRIC_PRIORITY = {
    "arc_easy": ("acc_norm,none", "acc,none"),
    "hellaswag": ("acc_norm,none", "acc,none"),
    "piqa": ("acc_norm,none", "acc,none"),
    "sciq": ("acc_norm,none", "acc,none"),
    "lambada_openai": ("acc,none",),
}

# --- Public-perplexity task config --------------------------------------------
PPL_TASKS = (
    "wikitext",
    "c4",
    "pile_10k",
    "pile_books3",
    "paloma_falcon-refinedweb",
    "paloma_wikitext_103",
    "paloma_m2d2_wikipedia_unsplit",
    "paloma_m2d2_s2orc_unsplit",
    "gov_report_nll",
)
PPL_TASK_METRIC_PRIORITY = {task: ("word_perplexity,none",) for task in PPL_TASKS}

CHECKPOINT_TOKENS_RE = re.compile(r"ckpt_tokens_(\d+)(?:_final)?$")
SUPPORTED_WANDB_STATES = {"finished"}
CLASSIC_EVAL_PROFILE = "classic"
PPL_EVAL_PROFILE = "perplexity_public"


def parse_checkpoint_tokens(checkpoint_name: str) -> int:
    match = CHECKPOINT_TOKENS_RE.fullmatch(str(checkpoint_name))
    if match is None:
        raise ValueError(f"Unsupported checkpoint name: {checkpoint_name!r}")
    return int(match.group(1))


def _sanitize_wandb_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("._-")
    return sanitized or "unknown"


def _summary_metric_key(metric_name: str) -> str:
    return "eval/" + _sanitize_wandb_component(metric_name).replace("-", "_")


def _metric_priority(
    task_name: str,
    task_metric_priorities: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    metric_priorities = (
        CLASSIC_TASK_METRIC_PRIORITY if task_metric_priorities is None else task_metric_priorities
    )
    if task_name not in metric_priorities:
        raise ValueError(f"Unsupported task {task_name!r}")
    return metric_priorities[task_name]


def _mapping_from_obj(obj) -> dict[str, object]:
    if obj is None:
        return {}
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    if isinstance(obj, Mapping):
        return dict(obj)
    for attr_name in ("_json_dict", "_as_dict"):
        candidate = getattr(obj, attr_name, None)
        if isinstance(candidate, str):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                return dict(parsed)
        if isinstance(candidate, Mapping):
            return dict(candidate)
        if callable(candidate):
            try:
                resolved = candidate()
            except Exception:
                continue
            if isinstance(resolved, Mapping):
                return dict(resolved)
    try:
        return dict(obj)
    except Exception:
        return {}


def _summary_mapping(run) -> dict[str, object]:
    return _mapping_from_obj(getattr(run, "summary", None))


def _config_mapping(run) -> dict[str, object]:
    raw = _mapping_from_obj(getattr(run, "config", None))
    if not raw:
        return {}
    flattened: dict[str, object] = {}
    for key, value in raw.items():
        if isinstance(value, Mapping) and "value" in value:
            flattened[key] = value["value"]
        else:
            flattened[key] = value
    return flattened


def _lookup_run_value(run, *keys: str):
    summary = _summary_mapping(run)
    config = _config_mapping(run)
    for key in keys:
        if key in summary:
            return summary[key]
        if key in config:
            return config[key]
    return None


def _coerce_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_name_and_value_for_run(
    run,
    task_name: str,
    *,
    task_metric_priorities: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[str, float] | None:
    summary = _summary_mapping(run)
    for metric_name in _metric_priority(task_name, task_metric_priorities):
        metric_value = _coerce_float(summary.get(_summary_metric_key(metric_name)))
        if metric_value is not None:
            return metric_name, metric_value
    return None


def _is_selected_eval_run_for_profile(run, *, eval_profile: str) -> bool:
    if str(getattr(run, "state", "")).lower() not in SUPPORTED_WANDB_STATES:
        return False
    if str(getattr(run, "job_type", "")).strip() != "eval":
        return False
    if str(_lookup_run_value(run, "status")).strip().lower() != "completed":
        return False
    if str(_lookup_run_value(run, "eval_profile")).strip() != eval_profile:
        return False
    return _coerce_int(_lookup_run_value(run, "num_fewshot")) == 0


def classic_is_selected_eval_run(run) -> bool:
    return _is_selected_eval_run_for_profile(run, eval_profile=CLASSIC_EVAL_PROFILE)


def ppl_is_selected_eval_run(run) -> bool:
    return _is_selected_eval_run_for_profile(run, eval_profile=PPL_EVAL_PROFILE)
