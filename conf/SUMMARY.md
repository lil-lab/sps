# Configuration Summary

## Active Config Surface

```text
conf/
├── config.yaml
├── model/
├── training/
├── optimizer/
├── scheduler/
├── data/
├── logging/
├── system/
├── eval/
├── experiment/
└── benchmark/
```

## Supported Model Configs

- `standard`
- `sps`
- `reverse_sps`
- `delayed_state`

## Experiment Naming

Current experiment files use the naming convention already present in
`conf/experiment/`, for example:

- `xs_full_attention_20b`
- `s_sps_w64_10b`
- `s_sps_w4096_10b`
- `s_reverse_sps_w0_20b`

Use the filenames as the source of truth rather than older baseline docs.

## Typical Commands

Every run also needs `system.data_root=<DATA_ROOT>` (see the main README).

```bash
uv run python scripts/train.py
uv run python scripts/train.py model=standard
uv run python scripts/train.py model=sps model.config.window_size=8
uv run python scripts/train.py +experiment=xs_full_attention_20b
uv run python scripts/train.py +experiment=s_sps_w64_10b
```

## Notes

- `config.yaml` composes the training/eval defaults listed above.
- Run `rg --files conf/experiment` to see all available recipes.
