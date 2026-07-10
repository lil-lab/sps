# Hydra Configuration Guide

This repo uses [Hydra](https://hydra.cc/) to configure training and evaluation. The active
config surface:

```text
conf/
├── config.yaml     # top-level defaults list
├── model/          # model family: standard, sps, reverse_sps, delayed_state
├── training/       # training-loop hyperparameters
├── optimizer/      # adamw
├── scheduler/      # constant, linear_decay
├── data/           # fineweb-edu
├── logging/        # Weights & Biases settings
├── system/         # device / dtype / compile / data_root
├── eval/           # evaluation defaults
├── experiment/     # concrete {scale}_{family}[_w{window}]_{tokens}b recipes
└── benchmark/      # generation-speed benchmark configs (loaded by scripts/benchmark)
```

## Quick Start

Every run also needs `system.data_root=<DATA_ROOT>` (see the main
[README](../README.md)); it is omitted below for brevity.

Run the default config:

```bash
uv run python scripts/train.py
```

Use a specific model:

```bash
uv run python scripts/train.py model=standard
uv run python scripts/train.py model=sps
uv run python scripts/train.py model=reverse_sps
uv run python scripts/train.py model=delayed_state
```

Override model parameters:

```bash
uv run python scripts/train.py model=sps model.config.window_size=8
uv run python scripts/train.py learning_rate=3e-4
```

## Experiments

Experiment files live under `conf/experiment/`. Browse them with:

```bash
rg --files conf/experiment
```

Representative examples:

```bash
uv run python scripts/train.py +experiment=xs_full_attention_20b
uv run python scripts/train.py +experiment=s_sps_w64_10b
uv run python scripts/train.py +experiment=s_sps_w4096_10b
uv run python scripts/train.py +experiment=s_reverse_sps_w0_20b
```

Use `--multirun` for sweeps:

```bash
uv run python scripts/train.py +experiment=s_sps_w64_10b \
  'model.config.window_size=16,32,64' --multirun
```

## Distributed Training

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py +experiment=xs_full_attention_20b
```

For multi-node runs, use the same `torchrun` pattern with `--nnodes`,
`--node_rank`, `--master_addr`, and `--master_port`.

## Inspecting Configs

```bash
uv run python scripts/train.py --cfg job
uv run python scripts/train.py +experiment=xs_full_attention_20b --cfg job
```

## Notes

- `model/` contains the supported model families only.
- `experiment/` files encode concrete training recipes; the naming convention is the
  current source of truth.
