"""Helpers for choosing a primary scalar metric from lm-eval task results."""

from __future__ import annotations

from typing import Mapping


DEFAULT_METRIC_PRIORITY = (
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "f1,none",
    "mc1,none",
    "mc2,none",
    "bleu,none",
    "rouge1,none",
    "rouge2,none",
    "rougeL,none",
)

LAMBADA_METRIC_PRIORITY = (
    "perplexity,none",
    "word_perplexity,none",
    "byte_perplexity,none",
)

ROLLING_PERPLEXITY_TASKS = {
    "wikitext",
    "c4",
    "pile_10k",
    "paloma_falcon-refinedweb",
    "paloma_wikitext_103",
    "paloma_m2d2_wikipedia_unsplit",
    "paloma_m2d2_s2orc_unsplit",
}

ROLLING_PERPLEXITY_METRIC_PRIORITY = (
    "bits_per_byte,none",
    "word_perplexity,none",
    "byte_perplexity,none",
)


def choose_primary_metric_name(
    task_name: str,
    metrics: Mapping[str, object],
) -> str | None:
    """Pick the preferred scalar metric for a task result payload."""
    if task_name == "lambada_openai":
        for metric_name in LAMBADA_METRIC_PRIORITY:
            if isinstance(metrics.get(metric_name), (int, float)):
                return metric_name

    if task_name in ROLLING_PERPLEXITY_TASKS:
        for metric_name in ROLLING_PERPLEXITY_METRIC_PRIORITY:
            if isinstance(metrics.get(metric_name), (int, float)):
                return metric_name

    for metric_name in DEFAULT_METRIC_PRIORITY:
        if isinstance(metrics.get(metric_name), (int, float)):
            return metric_name

    for metric_name, value in metrics.items():
        if metric_name == "alias" or metric_name.endswith("_stderr,none"):
            continue
        if isinstance(value, (int, float)):
            return metric_name

    return None
