from __future__ import annotations

from pathlib import Path

import torch

from training.checkpointing import (
    CheckpointManager,
    ROLLING_CHECKPOINT_NAME,
    latest_checkpoint_path,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint")


def test_latest_checkpoint_path_excludes_final_and_ignores_ckpt_alias(tmp_path: Path) -> None:
    _touch(tmp_path / "ckpt.pt")
    _touch(tmp_path / "ckpt_tokens_18000248832_pre_decay.pt")
    _touch(tmp_path / "ckpt_tokens_28001697792.pt")
    _touch(tmp_path / "ckpt_tokens_40000000000_final.pt")

    assert latest_checkpoint_path(tmp_path) == tmp_path / "ckpt_tokens_28001697792.pt"


def test_checkpoint_manager_resolves_explicit_filename(tmp_path: Path) -> None:
    _touch(tmp_path / "ckpt_tokens_28001697792.pt")
    manager = CheckpointManager(str(tmp_path), save_every=100)

    assert manager.resolve_checkpoint_path("ckpt_tokens_28001697792") == str(
        tmp_path / "ckpt_tokens_28001697792.pt"
    )


def test_save_checkpoint_does_not_write_ckpt_alias(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.ByteTensor([0]))
    manager = CheckpointManager(str(tmp_path), save_every=100)

    saved_path = manager.save_checkpoint(
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={},
        model_args={},
        iter_num=1,
        best_val_loss=1.0,
        tokens_seen=100,
        config={},
        wandb_run_id="run",
        next_eval_tokens=200,
    )

    assert saved_path == str(tmp_path / "ckpt_tokens_100.pt")
    assert (tmp_path / "ckpt_tokens_100.pt").exists()
    assert not (tmp_path / "ckpt.pt").exists()


def test_rolling_save_writes_ckpt_alias_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.ByteTensor([0]))
    manager = CheckpointManager(str(tmp_path), save_every=1_000_000_000, rolling_save_every=100_000_000)

    saved_path = manager.save_checkpoint(
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={},
        model_args={},
        iter_num=5,
        best_val_loss=1.0,
        tokens_seen=500_000_000,
        config={},
        wandb_run_id="run",
        is_rolling=True,
    )

    assert saved_path == str(tmp_path / ROLLING_CHECKPOINT_NAME)
    assert (tmp_path / "ckpt.pt").exists()
    # Rolling save must not create a named snapshot, and must advance only the
    # rolling cadence (not the named save_every cadence).
    assert not list(tmp_path.glob("ckpt_tokens_*.pt"))
    assert manager.last_rolling_save_tokens == 500_000_000
    assert manager.last_save_tokens == 0


def test_should_save_rolling_cadence_and_decay_gating() -> None:
    manager = CheckpointManager(
        "unused",
        save_every=1_000_000_000,
        rolling_save_every=100_000_000,
        decay_start_tokens=1_000_000_000,
        tokens_per_iter=1,
    )
    assert manager.should_save_rolling(50_000_000, iter_num=0) is False  # iter 0 never saves
    assert manager.should_save_rolling(50_000_000, iter_num=10) is False  # < cadence
    assert manager.should_save_rolling(100_000_000, iter_num=10) is True  # reached cadence
    manager.last_rolling_save_tokens = 100_000_000
    assert manager.should_save_rolling(150_000_000, iter_num=20) is False  # since last < cadence
    # Decay phase: keep current (named-only) behavior — no rolling refresh.
    assert manager.should_save_rolling(1_000_000_000, iter_num=99) is False
    assert manager.should_save_rolling(1_500_000_000, iter_num=99) is False


def test_resolve_prefers_rolling_pre_decay_and_named_in_decay(tmp_path: Path) -> None:
    _touch(tmp_path / "ckpt.pt")
    _touch(tmp_path / "ckpt_tokens_1000000000.pt")  # a 1B named snapshot
    decay_start = 18_000_000_000

    # Pre-decay: latest named (1B) is below decay start, so the rolling file wins.
    manager = CheckpointManager(str(tmp_path), save_every=1_000_000_000, decay_start_tokens=decay_start)
    assert manager.resolve_checkpoint_path() == str(tmp_path / "ckpt.pt")

    # Once a named checkpoint reaches the decay phase, it is the more advanced one.
    _touch(tmp_path / "ckpt_tokens_18000248832_pre_decay.pt")
    assert manager.resolve_checkpoint_path() == str(tmp_path / "ckpt_tokens_18000248832_pre_decay.pt")


def test_resolve_falls_back_to_named_without_rolling(tmp_path: Path) -> None:
    _touch(tmp_path / "ckpt_tokens_1000000000.pt")
    manager = CheckpointManager(str(tmp_path), save_every=1_000_000_000, decay_start_tokens=18_000_000_000)
    assert manager.resolve_checkpoint_path() == str(tmp_path / "ckpt_tokens_1000000000.pt")
