#!/usr/bin/env python3
"""Resolve a benchmark config (conf/benchmark/<name>.yaml) into driver arguments.

This is the bridge between the cluster-independent YAML configs and the argparse-based
benchmark_generation_speed.py driver. It reads the named config, substitutes the
cluster-specific checkpoint root / filename, and emits either:

  --emit argv         newline-delimited argv for the driver (read with `mapfile -t`)
  --emit checkpoints  one resolved checkpoint path per line (for existence validation)

Keeping spaces intact matters: a "--spec" value like `sps:2x Memory:/path` is a single
token (one line), so labels with spaces survive the round-trip through bash arrays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONF_DIR = _REPO_ROOT / "conf" / "benchmark"


def _load_config(config_name: str) -> dict:
    named_path = _CONF_DIR / f"{config_name}.yaml"
    if not named_path.is_file():
        available = sorted(
            p.stem for p in _CONF_DIR.glob("*.yaml") if not p.stem.startswith("_")
        )
        raise SystemExit(
            f"Unknown benchmark config {config_name!r}. "
            f"Available: {', '.join(available)}"
        )
    return yaml.safe_load(named_path.read_text()) or {}


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def _resolve_specs(cfg: dict, out_root: str, ckpt_name: str, scales: list[str], method_labels=None):
    """Yield (spec_string, checkpoint_path) for every (scale, method).

    A method's checkpoint directory comes from its ``run_name`` template
    (``{scale}`` substituted).
    """
    for scale in scales:
        for method in cfg["methods"]:
            if method_labels is not None and method["label"] not in method_labels:
                continue
            run_name = method["run_name"].format(scale=scale)
            kind = method["kind"]
            ckpt_path = f"{out_root.rstrip('/')}/{run_name}/{ckpt_name}"
            spec = f"{kind}:{method['label']}:{ckpt_path}"
            yield spec, ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name", help="Benchmark config name (e.g. speed)")
    parser.add_argument("--out-root", required=True, help="Checkpoint root ($OUT_ROOT)")
    parser.add_argument(
        "--ckpt-name",
        default="ckpt_tokens_20000145408_final.pt",
        help="Checkpoint filename under each run_name directory",
    )
    parser.add_argument(
        "--scales",
        default=None,
        help="Space-separated scale override (default: from config, e.g. 'xs s m l')",
    )
    parser.add_argument(
        "--num-prompts",
        default=None,
        help="Override the config's batch size / num_prompts (e.g. 16 for XL w4096 OOM).",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method labels to include (default: all). "
        "e.g. '2x Memory' to probe a single model.",
    )
    parser.add_argument(
        "--warp-specialize",
        choices=["keep", "on", "off"],
        default=None,
        help="Pass through to the driver (XL SPS kernels need 'off').",
    )
    parser.add_argument(
        "--emit",
        choices=["argv", "checkpoints"],
        default="argv",
        help="argv: driver arguments; checkpoints: resolved checkpoint paths",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config_name)
    scales = args.scales.split() if args.scales else list(cfg["scales"])
    method_labels = (
        [m.strip() for m in args.methods.split(",")] if args.methods else None
    )
    num_prompts = args.num_prompts if args.num_prompts is not None else cfg["num_prompts"]
    specs = list(_resolve_specs(cfg, args.out_root, args.ckpt_name, scales, method_labels))

    if args.emit == "checkpoints":
        for _spec, ckpt_path in specs:
            print(ckpt_path)
        return

    # emit == "argv": newline-delimited tokens for `mapfile -t`.
    tokens: list[str] = []
    tokens += ["--prompt-lens", *[str(v) for v in _as_list(cfg["prompt_lens"])]]
    tokens += ["--num-prompts", *[str(v) for v in _as_list(num_prompts)]]
    tokens += ["--max-new-tokens", *[str(v) for v in _as_list(cfg["max_new_tokens"])]]
    tokens += ["--warmup-iters", str(cfg["warmup_iters"])]
    tokens += ["--timed-iters", str(cfg["timed_iters"])]
    tokens += ["--seed", str(cfg["seed"])]
    tokens += ["--config-name", args.config_name]
    warp = args.warp_specialize if args.warp_specialize is not None else cfg.get("warp_specialize")
    if warp:
        tokens += ["--warp-specialize", warp]
    if cfg.get("allow_step_prefill"):
        tokens += ["--allow-step-prefill"]
    for spec, _ckpt_path in specs:
        tokens += ["--spec", spec]
    print("\n".join(tokens))


if __name__ == "__main__":
    main()
