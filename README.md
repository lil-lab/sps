# The State-Prediction Separation Hypothesis

Code for the paper **["The State-Prediction Separation Hypothesis"](https://arxiv.org/abs/2607.01218)**
(Giovanni Monea, Nathan Godey, Kianté Brantley, Yoav Artzi).

Transformers use the same forward computation stream both to predict the next token
and to store state that future positions read from the KV cache. We formulate the
**state–prediction separation (SPS) hypothesis**: disentangling these two roles yields
better language modeling. The **SPS Transformer** realizes the separation by inserting a
dedicated `<predict>` token after every input token, giving two interleaved streams — a
persistent *input* stream that carries state forward, and an ephemeral *prediction* stream
(kept only within a sliding window `w`) that emits next-token predictions. Across scales
from 53M to 1.68B parameters, SPS lowers FineWeb-Edu validation loss, improves held-out
generalization, and raises zero-shot accuracy at matched inference cost.

## Installation

The project uses [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

This installs the pinned dependencies (PyTorch 2.9, Hydra, a Triton FlashAttention build,
and the EleutherAI `lm-evaluation-harness`). Prefix commands with `uv run` to use the
environment (e.g. `uv run python scripts/train.py ...`). A CUDA GPU with Triton is required
to run the fused attention kernels; CPU fallbacks exist for the tests.

## Data preparation

We pretrain on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
tokenized with the GPT-2 tokenizer and packed across document boundaries:

```bash
# writes train/val memmap bins under <DATA_ROOT>/data
# (downloads FineWeb-Edu to <DATA_ROOT>/.hf_cache on first run)
uv run python src/data/prepare.py system.data_root=<DATA_ROOT>
```

## Training

Training is configured with [Hydra](https://hydra.cc/). Experiment recipes live under
`conf/experiment/` and follow the naming convention
`{scale}_{family}[_w{window}]_{tokens}b`, where:

- **scale** ∈ `{xs, s, m, l, xl}` (53M / 131M / 379M / 831M / 1.678B parameters),
- **family** ∈ `{full_attention, sps, reverse_sps, delayed_state}`,
- **window** is the sliding `<predict>`-window size `w` (e.g. `w64`; the paper uses `w=64`),
- **tokens** is the pre-decay training budget (`10b` / `20b`).

> **Every `train.py` / `evaluate.py` command needs `system.data_root=<DATA_ROOT>`** — the same
> directory you used for data prep. It has no default (Hydra errors if it is missing); the
> examples show it on the first command of each block, so append it to the variants too.

```bash
# STANDARD baseline (xs, 20B tokens)
uv run python scripts/train.py +experiment=xs_full_attention_20b system.data_root=<DATA_ROOT>

# SPS (s, w=64, 20B tokens)
uv run python scripts/train.py +experiment=s_sps_w64_20b

# 2X MEMORY ablation is SPS with a full window (w=4096)
uv run python scripts/train.py +experiment=s_sps_w4096_20b

# DELAYED STATE and REVERSE SPS baselines
uv run python scripts/train.py +experiment=s_delayed_state_w64_20b
uv run python scripts/train.py +experiment=s_reverse_sps_w64_20b
```

The available model families are `standard`, `sps`, `reverse_sps`, and `delayed_state`
(`conf/model/`). See `conf/README.md` for the full config surface. For distributed data
parallel:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py +experiment=xs_full_attention_20b system.data_root=<DATA_ROOT>
```

### Runtime and logging overrides

- **Weights & Biases** logging is on by default and writes to your default W&B entity. Turn it
  off with `logging.wandb_log=false`, or point it somewhere specific with
  `logging.wandb_entity=<entity> logging.wandb_project=<project>`.
- **Device / compile** — add `system.device=cpu` or `system.compile=false` to any command to
  run on CPU or skip `torch.compile`.

## Evaluation

Evaluation runs through the `lm-evaluation-harness` and reports FineWeb-Edu validation
loss, out-of-distribution corpus NLL (WikiText, C4, Pile-Books3, GovReport — see
`lm_eval_tasks/`), and zero-shot accuracy on ARC-Easy, HellaSwag, PIQA, SciQ, and LAMBADA:

```bash
uv run python scripts/evaluate.py +experiment=s_sps_w64_20b +checkpoint=final system.data_root=<DATA_ROOT>
```

## Reproducing the paper's figures and tables

- **Figures** — `scripts/figures/plot_*.py` (e.g. `plot_validation_nll_history_wandb.py` for
  Figure 2, `plot_accuracy_gain_by_scale.py` for Figure 3, `plot_window_ablation.py` for
  Figure 4, and — for Figure 5 — `plot_gradient_params.py` (the future/present gradient-ratio
  panel) and `plot_persistent_window_nll.py` (the persistent-window Δℓ panel)).
  Figures are written to `figures/`.
- **Tables** — `scripts/tables/export_main_results_table.py` produces the main results
  table in two forms: the compact headline (validation loss, generalization, and
  inference-efficiency ratios; `tab:main-compact`) and its per-corpus / per-benchmark
  expansion (`tab:main-full`). Both `.tex` files are written to `outputs/tables/`.
- **Inference-efficiency benchmarks** — `scripts/benchmark/`; see
  [`scripts/benchmark/REPRODUCE.md`](scripts/benchmark/REPRODUCE.md) for the throughput /
  peak-memory measurement protocol.

## Repository layout

```text
src/modeling/        model implementations (standard, sps, reverse_sps, delayed_state),
                     Triton attention kernels, evaluation adapter, and tests
scripts/             train.py, evaluate.py, and figures/ tables/ analysis/ benchmark/ utilities
conf/                Hydra configs (model/ data/ experiment/ optimizer/ scheduler/ ...)
src/data/            dataset tokenization (FineWeb-Edu)
lm_eval_tasks/       custom lm-eval corpus-NLL task definitions
```

## Tests

```bash
uv run pytest                 # fast CPU-safe tests
uv run pytest -m cuda         # GPU tests (Triton kernels)
```

## Citation

```bibtex
@article{monea2026sps,
  title   = {The State-Prediction Separation Hypothesis},
  author  = {Monea, Giovanni and Godey, Nathan and Brantley, Kiant\'e and Artzi, Yoav},
  journal = {arXiv preprint arXiv:2607.01218},
  year    = {2026},
}
```
