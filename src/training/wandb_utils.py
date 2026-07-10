"""Helpers for keeping local W&B run data off the home filesystem."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    try:
        return getattr(obj, key)
    except (AttributeError, TypeError):
        pass
    try:
        return obj[key]
    except Exception:
        return default


def _clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _expand_path(path: str | os.PathLike[str]) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(os.fspath(path)))
    result = Path(expanded)
    if not result.is_absolute():
        result = Path.cwd() / result
    return result


def resolve_wandb_dir(
    *,
    wandb_dir: Any = None,
    data_root: Any = None,
    create: bool = True,
) -> str | None:
    """Resolve the local W&B run directory.

    Explicit ``wandb_dir`` wins. If absent, an existing ``WANDB_DIR`` is honored.
    If neither is set and ``data_root`` exists, W&B run files go under
    ``<data_root>/wandb``.
    """
    raw_dir = _clean_optional_str(wandb_dir)
    if raw_dir is None:
        raw_dir = _clean_optional_str(os.environ.get("WANDB_DIR"))
    if raw_dir is None and (raw_data_root := _clean_optional_str(data_root)) is not None:
        raw_dir = str(Path(raw_data_root) / "wandb")
    if raw_dir is None:
        return None

    path = _expand_path(raw_dir)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_wandb_dir_from_config(cfg: Any, *, create: bool = True) -> str | None:
    logging_cfg = _cfg_get(cfg, "logging", cfg)
    system_cfg = _cfg_get(cfg, "system", None)
    return resolve_wandb_dir(
        wandb_dir=_cfg_get(logging_cfg, "wandb_dir", None),
        data_root=_cfg_get(system_cfg, "data_root", None),
        create=create,
    )


def prepare_wandb_dir(
    *,
    wandb_dir: Any = None,
    data_root: Any = None,
    create: bool = True,
) -> str | None:
    path = resolve_wandb_dir(wandb_dir=wandb_dir, data_root=data_root, create=create)
    if path is not None:
        os.environ.setdefault("WANDB_DIR", path)
    return path


def prepare_wandb_dir_from_config(cfg: Any, *, create: bool = True) -> str | None:
    path = resolve_wandb_dir_from_config(cfg, create=create)
    if path is not None:
        os.environ.setdefault("WANDB_DIR", path)
    return path


def _read_simple_yaml_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        if raw_key.strip() != key:
            continue
        return _clean_optional_str(raw_value.strip().strip("\"'"))
    return None


def default_wandb_dir_from_repo_config(repo_root: str | os.PathLike[str]) -> str | None:
    """Resolve the repo default W&B dir without requiring Hydra."""
    root = Path(repo_root)
    data_root = _read_simple_yaml_value(root / "conf" / "system" / "default.yaml", "data_root")
    configured = _read_simple_yaml_value(root / "conf" / "logging" / "default.yaml", "wandb_dir")
    if configured is not None and data_root is not None:
        configured = configured.replace("${system.data_root}", data_root)
    return resolve_wandb_dir(wandb_dir=configured, data_root=data_root, create=False)
