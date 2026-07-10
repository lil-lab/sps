"""Plot mean and 95% CI (Student-t) of final validation NLL across seeds at S/10B.

For each of the four methods (\\Standard, \\DelayedState, \\Twomem, \\SPS) we
run three seeds (the unseeded base run + seed0 + seed1) and pull the final
``val/token_nll`` from W&B. The figure is a single panel showing per-method
mean with 95% CI bars computed via the Student-t distribution
(t(0.025, df=n-1); for n=3 the multiplier is ~4.303).

Style matches ``plot_window_ablation.py``: bold tick/axis fonts, warm
background, white-edged markers with path effects.
"""

from __future__ import annotations

import plotting.mpl_cache  # noqa: F401  (pin MPLCONFIGDIR off NFS before matplotlib)

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

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
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from plotting.style import (
    COLORBLIND_METHOD_COLORS,
    LEGEND_TEXT_COLOR,
    PLOT_SPINE_COLOR,
    _apply_axes_background,
    bf,
    enable_latex,
    sc,
)


# (label, W&B base run name, color slug)
METHODS: tuple[tuple[str, str, str], ...] = (
    ("Standard",      "s_full_attention_10b", "standard"),
    ("Delayed State", "s_delayed_state_w64_10b", "delayed_state"),
    ("2x Memory",     "s_sps_w4096_10b",         "two_memory"),
    ("SPS",           "s_sps_w64_10b",           "sps"),
)
SEED_SUFFIXES: tuple[str, ...] = ("", "_seed0", "_seed1")
# W&B run path "<entity>/<project>". Set WANDB_ENTITY (and optionally WANDB_PROJECT
# or the whole WANDB_PATH) to point this at your own runs.
WANDB_PATH = os.environ.get(
    "WANDB_PATH",
    f"{os.environ.get('WANDB_ENTITY', '')}/{os.environ.get('WANDB_PROJECT', 'pretraining_compression')}",
)
METRIC_KEY = "val/token_nll"

# --- Style (compact ablation-style panel) ---
FIGSIZE = (4.6, 3.4)
LINEWIDTH = 2.4
MARKER_SIZE = 11
MARKER_EDGE_WIDTH = 1.4
ERR_LINEWIDTH = 1.5
ERR_CAPSIZE = 4.0
ERR_CAPTHICK = 1.3
AXIS_LABEL_FONT_SIZE = 16
AXIS_TICK_LABEL_SIZE = 13
PLOT_DPI = 200


@dataclass
class RunRow:
    method: str
    base_name: str
    seed_tag: str
    run_id: str
    nll: float
    created_at: str


def _color(slug: str) -> str:
    return COLORBLIND_METHOD_COLORS.get(slug, "#444444")


def _final_nll(run) -> float | None:
    sm = getattr(run, "summary_metrics", None)
    if isinstance(sm, str):
        try:
            sm = json.loads(sm)
        except Exception:
            sm = None
    if isinstance(sm, dict):
        v = sm.get(METRIC_KEY)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    last = None
    try:
        for row in run.scan_history(keys=[METRIC_KEY]):
            value = row.get(METRIC_KEY)
            if value is not None:
                last = value
    except Exception:
        return None
    return float(last) if last is not None else None


def fetch_runs() -> list[RunRow]:
    import wandb

    api = wandb.Api()
    rows: list[RunRow] = []
    for label, base_name, _ in METHODS:
        for suffix in SEED_SUFFIXES:
            display_name = base_name + suffix
            try:
                runs = list(
                    api.runs(
                        WANDB_PATH,
                        filters={"display_name": display_name, "state": "finished"},
                        per_page=20,
                    )
                )
            except Exception as exc:
                print(f"  [warn] {display_name}: API error {exc}")
                continue
            if not runs:
                print(f"  [warn] {display_name}: no finished runs")
                continue
            runs.sort(key=lambda r: r.created_at, reverse=True)
            run = runs[0]
            nll = _final_nll(run)
            if nll is None:
                print(f"  [warn] {display_name}: missing {METRIC_KEY}")
                continue
            rows.append(
                RunRow(
                    method=label,
                    base_name=base_name,
                    seed_tag=suffix or "base",
                    run_id=run.id,
                    nll=nll,
                    created_at=str(run.created_at),
                )
            )
    return rows


def write_csv(rows: list[RunRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["method", "base_name", "seed_tag", "run_id", "final_val_token_nll", "created_at"]
        )
        for r in rows:
            writer.writerow(
                [r.method, r.base_name, r.seed_tag, r.run_id, f"{r.nll:.6f}", r.created_at]
            )


_T_MULT_BY_DF: dict[int, float] = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
}


def _t_mult(n: int) -> float:
    df = max(n - 1, 1)
    if df in _T_MULT_BY_DF:
        return _T_MULT_BY_DF[df]
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df))
    except Exception:
        return 1.96


def _stats(values: list[float]) -> tuple[float, float, float]:
    """(mean, stderr, half-width of 95% CI via Student-t)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, float("nan"), float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    return mean, se, _t_mult(n) * se


def _style_ticks(ax: plt.Axes) -> None:
    ax.tick_params(
        axis="both",
        colors=PLOT_SPINE_COLOR,
        labelcolor=LEGEND_TEXT_COLOR,
        labelsize=AXIS_TICK_LABEL_SIZE,
        length=4.0,
        width=0.85,
    )
    for tick in (*ax.xaxis.get_major_ticks(), *ax.yaxis.get_major_ticks()):
        tick.tick1line.set_color(PLOT_SPINE_COLOR)
        tick.tick2line.set_color(PLOT_SPINE_COLOR)


def render(rows: list[RunRow], out_path: Path) -> None:
    by_method: dict[str, list[float]] = {label: [] for label, _, _ in METHODS}
    for r in rows:
        if r.method in by_method:
            by_method[r.method].append(r.nll)

    method_labels = [label for label, _, _ in METHODS]
    method_slugs = [slug for _, _, slug in METHODS]
    means: list[float] = []
    serrs: list[float] = []
    cis: list[float] = []
    n_per_method: list[int] = []
    for label in method_labels:
        m, se, ci = _stats(by_method[label])
        means.append(m)
        serrs.append(se)
        cis.append(ci)
        n_per_method.append(len(by_method[label]))

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Axis bounds: pad around all means +/- 95% CI.
    finite_low = [m - c for m, c in zip(means, cis) if math.isfinite(m) and math.isfinite(c)]
    finite_high = [m + c for m, c in zip(means, cis) if math.isfinite(m) and math.isfinite(c)]
    pad = 0.005
    y_min = (min(finite_low) - pad) if finite_low else 0.0
    y_max = (max(finite_high) + pad) if finite_high else 1.0

    xs = list(range(len(method_labels)))
    x_min, x_max = -0.6, len(method_labels) - 0.4
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    _apply_axes_background(ax, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # Per-method points: 95% CI bars (Student-t) + colored markers.
    for x, label, slug, m, c in zip(xs, method_labels, method_slugs, means, cis):
        color = _color(slug)
        if math.isfinite(c) and c > 0:
            ax.errorbar(
                [x],
                [m],
                yerr=[[c], [c]],
                fmt="none",
                ecolor=color,
                elinewidth=ERR_LINEWIDTH,
                capsize=ERR_CAPSIZE,
                capthick=ERR_CAPTHICK,
                alpha=0.95,
                zorder=8,
            )
        marker_line, = ax.plot(
            [x],
            [m],
            color=color,
            marker="o",
            linestyle="None",
            markersize=MARKER_SIZE,
            markeredgecolor="white",
            markeredgewidth=MARKER_EDGE_WIDTH,
            zorder=9,
        )
        marker_line.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=MARKER_EDGE_WIDTH + 1.4, foreground="white", alpha=0.92
                ),
                path_effects.Normal(),
            ]
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [sc(label) for label in method_labels],
        fontsize=AXIS_TICK_LABEL_SIZE,
        color=LEGEND_TEXT_COLOR,
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel(
        bf("Validation NLL"),
        fontsize=AXIS_LABEL_FONT_SIZE,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
    )
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.minorticks_off()
    _style_ticks(ax)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    png_path = out_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figures/seed_variance_s_10b"),
        help="Output stem (.pdf, .png, .csv are appended).",
    )
    args = parser.parse_args()
    enable_latex()

    print("Fetching runs from W&B...")
    rows = fetch_runs()
    if not rows:
        raise SystemExit("no runs collected; aborting")

    csv_path = args.out.with_suffix(".csv")
    write_csv(rows, csv_path)
    print(f"Wrote {csv_path}")

    print("Rendering plot...")
    render(rows, args.out)

    # Summary print
    print()
    print(f"{'Method':16s} {'#':>2s} {'mean':>8s} {'stderr':>8s} {'95% CI':>10s}")
    print("-" * 50)
    by_method: dict[str, list[float]] = {label: [] for label, _, _ in METHODS}
    for r in rows:
        by_method[r.method].append(r.nll)
    for label, _, _ in METHODS:
        m, se, ci = _stats(by_method[label])
        n = len(by_method[label])
        print(f"{label:16s} {n:2d} {m:8.4f} {se:8.4f} {ci:10.4f}")


if __name__ == "__main__":
    main()
