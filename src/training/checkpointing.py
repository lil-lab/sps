"""Checkpoint management for training."""

import os
from pathlib import Path
import re
from typing import Dict, Any, Optional, Tuple
import torch


_CHECKPOINT_RE = re.compile(r"^ckpt_tokens_(\d+)(?:_(?:pre_decay|final))?\.pt$")

# Single rolling resume checkpoint, overwritten frequently (every
# ``rolling_save_every`` tokens) during the pre-decay phase so a crash loses at
# most that many tokens of compute. Distinct from the named ``ckpt_tokens_*.pt``
# snapshots saved every ``save_every`` tokens.
ROLLING_CHECKPOINT_NAME = "ckpt.pt"


def parse_checkpoint_tokens(path: str | os.PathLike[str]) -> int | None:
    """Return the token count encoded in a named checkpoint filename."""
    match = _CHECKPOINT_RE.match(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def is_final_checkpoint_path(path: str | os.PathLike[str]) -> bool:
    """Return whether a named checkpoint is a final checkpoint."""
    return "_final" in Path(path).name


def _checkpoint_sort_priority(path: Path) -> int:
    if is_final_checkpoint_path(path):
        return 2
    if "_pre_decay" in path.name:
        return 0
    return 1


def named_checkpoint_paths(
    out_dir: str | os.PathLike[str],
    *,
    include_final: bool = False,
) -> list[Path]:
    """List named checkpoints sorted by token count, excluding final by default."""
    out_path = Path(out_dir)
    if not out_path.is_dir():
        return []

    candidates: list[tuple[int, int, str, Path]] = []
    for path in out_path.glob("ckpt_tokens_*.pt"):
        tokens = parse_checkpoint_tokens(path)
        if tokens is None:
            continue
        if not include_final and is_final_checkpoint_path(path):
            continue
        candidates.append((tokens, _checkpoint_sort_priority(path), path.name, path))

    return [path for _tokens, _priority, _name, path in sorted(candidates)]


def latest_checkpoint_path(
    out_dir: str | os.PathLike[str],
    *,
    include_final: bool = False,
) -> Path | None:
    """Return the latest named checkpoint, excluding final checkpoints by default."""
    candidates = named_checkpoint_paths(out_dir, include_final=include_final)
    if not candidates:
        return None
    return candidates[-1]


class CheckpointManager:
    """Manages saving and loading checkpoints during training.
    
    Features:
    - Saves checkpoints at regular token intervals (save_every)
    - Each checkpoint has a unique filename based on token count
    - Handles final checkpoint saving with special naming
    - Saves a special pre-decay checkpoint before LR decay starts
    
    Args:
        out_dir: Directory to save checkpoints to
        save_every: Save checkpoint every N tokens
        master_process: Whether this is the master process (only master saves)
        decay_start_tokens: Token count when LR decay starts (optional)
        tokens_per_iter: Tokens processed per iteration (optional)
    """
    
    def __init__(self, out_dir: str, save_every: int, master_process: bool = True,
                 decay_start_tokens: Optional[int] = None, tokens_per_iter: Optional[int] = None,
                 rolling_save_every: Optional[int] = None):
        self.out_dir = out_dir
        self.save_every = save_every
        self.master_process = master_process
        self.last_save_tokens = 0
        self.saved_pre_decay_checkpoint = False
        self.decay_start_tokens = decay_start_tokens
        self.tokens_per_iter = tokens_per_iter
        # Rolling resume checkpoint cadence (tokens). None disables it.
        self.rolling_save_every = rolling_save_every
        self.last_rolling_save_tokens = 0
        
        # Create output directory if master process
        if self.master_process:
            os.makedirs(self.out_dir, exist_ok=True)
    
    def should_save(self, tokens_seen: int, iter_num: int) -> Tuple[bool, bool]:
        """Check if we should save a checkpoint and what type.
        
        Args:
            tokens_seen: Total tokens processed so far
            iter_num: Current iteration number
            
        Returns:
            Tuple of (should_save, is_pre_decay)
            - should_save: True if any checkpoint should be saved
            - is_pre_decay: True if this should be the pre-decay checkpoint
        """
        if not self.master_process or iter_num == 0:
            return False, False
        
        # Check for pre-decay checkpoint first (higher priority)
        if (self.decay_start_tokens is not None and 
            self.tokens_per_iter is not None and 
            not self.saved_pre_decay_checkpoint):
            # Save if we've just crossed or are about to cross the decay threshold
            if tokens_seen >= self.decay_start_tokens and tokens_seen < self.decay_start_tokens + self.tokens_per_iter:
                return True, True
        
        # Check for regular checkpoint
        if tokens_seen - self.last_save_tokens >= self.save_every:
            return True, False

        return False, False

    def should_save_rolling(self, tokens_seen: int, iter_num: int) -> bool:
        """Whether to refresh the rolling resume checkpoint (ckpt.pt).

        Active only during the pre-decay phase (the decay-phase checkpoints are
        kept as named snapshots, per the original behavior). Returns True every
        ``rolling_save_every`` tokens.
        """
        if not self.master_process or iter_num == 0 or self.rolling_save_every is None:
            return False
        # Keep current (named-only) behavior during the decay phase.
        if self.decay_start_tokens is not None and tokens_seen >= self.decay_start_tokens:
            return False
        return tokens_seen - self.last_rolling_save_tokens >= self.rolling_save_every

    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        model_args: Dict[str, Any],
        iter_num: int,
        best_val_loss: float,
        tokens_seen: int,
        config: Dict[str, Any],
        wandb_run_id: str,
        sampler_offset: int = 0,
        sampler_samples_seen_per_rank: int = 0,
        next_eval_tokens: Optional[int] = None,
        is_final: bool = False,
        is_pre_decay: bool = False,
        is_rolling: bool = False
    ) -> Optional[str]:
        """Save a checkpoint to disk.

        Args:
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            model_args: Model configuration arguments
            iter_num: Current iteration number
            best_val_loss: Best validation loss so far
            tokens_seen: Total tokens processed
            config: Full training configuration
            wandb_run_id: Wandb run ID for resuming runs
            sampler_offset: Sampler offset (next unread position) for data loading resumption
            is_final: Whether this is the final checkpoint
            is_pre_decay: Whether this is the pre-decay checkpoint (saved before LR decay)

        Returns:
            Path to saved checkpoint, or None if not master process
        """
        if not self.master_process:
            return None

        # Build checkpoint dictionary
        checkpoint = {
            'model': model_state,
            'optimizer': optimizer_state,
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'last_save_tokens': tokens_seen,
            'config': config,
            'sampler_offset': sampler_offset,
            'sampler_samples_seen_per_rank': int(sampler_samples_seen_per_rank),
            'wandb_run_id': wandb_run_id,
            'cpu_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
        }
        if next_eval_tokens is not None:
            checkpoint['next_eval_tokens'] = int(next_eval_tokens)
        
        # Generate filename. The rolling resume checkpoint is a single file that
        # is overwritten; the others are unique, token-named snapshots.
        if is_rolling:
            ckpt_filename = ROLLING_CHECKPOINT_NAME
        elif is_final:
            ckpt_filename = f'ckpt_tokens_{tokens_seen}_final.pt'
        elif is_pre_decay:
            ckpt_filename = f'ckpt_tokens_{tokens_seen}_pre_decay.pt'
        else:
            ckpt_filename = f'ckpt_tokens_{tokens_seen}.pt'

        ckpt_path = os.path.join(self.out_dir, ckpt_filename)

        # Save checkpoint. The rolling checkpoint is written atomically (tmp +
        # os.replace) so an interrupted write can never corrupt the sole resume
        # file that overwrites itself.
        if is_rolling:
            print(f"Updating rolling resume checkpoint {ckpt_path} (tokens: {tokens_seen:,})")
            tmp_path = ckpt_path + ".tmp"
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, ckpt_path)
        else:
            if is_pre_decay:
                print(f"Saving pre-decay checkpoint to {ckpt_path} (tokens: {tokens_seen:,})")
            else:
                print(f"Saving checkpoint to {ckpt_path} (tokens: {tokens_seen:,})")
            torch.save(checkpoint, ckpt_path)

        if is_pre_decay:
            self.saved_pre_decay_checkpoint = True

        # Update bookkeeping for the relevant cadence.
        if is_rolling:
            self.last_rolling_save_tokens = tokens_seen
        elif not is_final and not is_pre_decay:
            self.last_save_tokens = tokens_seen

        return ckpt_path
    
    def _explicit_checkpoint_candidates(self, checkpoint_path: str) -> list[Path]:
        requested = Path(checkpoint_path)
        bases = [requested] if requested.is_absolute() else [requested, Path(self.out_dir) / requested]

        candidates: list[Path] = []
        for base in bases:
            candidates.append(base)
            if base.suffix != ".pt":
                candidates.append(base.with_suffix(".pt"))
        return candidates

    def resolve_checkpoint_path(self, checkpoint_path: Optional[str] = None) -> Optional[str]:
        """Resolve an explicit checkpoint or the latest non-final named checkpoint."""
        if checkpoint_path:
            candidates = self._explicit_checkpoint_candidates(str(checkpoint_path))
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)
            tried = "\n  - ".join(str(candidate) for candidate in candidates)
            raise FileNotFoundError(f"Checkpoint not found. Tried:\n  - {tried}")

        named = latest_checkpoint_path(self.out_dir, include_final=False)
        rolling = Path(self.out_dir) / ROLLING_CHECKPOINT_NAME
        if rolling.is_file():
            if named is None:
                return str(rolling)
            named_tokens = parse_checkpoint_tokens(named) or 0
            # The rolling file is only refreshed pre-decay, so once a named
            # checkpoint reaches the decay phase it is the more advanced one;
            # otherwise the rolling file is the most recent state.
            if self.decay_start_tokens is not None and named_tokens >= self.decay_start_tokens:
                return str(named)
            return str(rolling)
        if named is None:
            return None
        return str(named)

    def load_checkpoint(self, device: str, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Load a checkpoint from disk.
        
        Args:
            device: Device to load checkpoint to
            checkpoint_path: Explicit checkpoint path. If None, loads the latest
                named non-final checkpoint.
            
        Returns:
            Checkpoint dictionary containing model, optimizer, etc.
        """
        resolved_checkpoint_path = self.resolve_checkpoint_path(checkpoint_path)
        if resolved_checkpoint_path is None:
            raise FileNotFoundError(
                f"No named non-final checkpoints found in {self.out_dir}"
            )
        
        print(f"Loading checkpoint from {resolved_checkpoint_path}")
        checkpoint = torch.load(resolved_checkpoint_path, map_location=device)
        
        # Restore last_save_tokens if present
        if 'last_save_tokens' in checkpoint:
            self.last_save_tokens = checkpoint['last_save_tokens']
            # Resume the rolling cadence from the loaded checkpoint's token count
            # so the next rolling save happens ~rolling_save_every tokens later.
            self.last_rolling_save_tokens = int(checkpoint['last_save_tokens'])
        
        # Check if we've already saved a pre-decay checkpoint by looking for the file
        pre_decay_files = [f for f in os.listdir(self.out_dir) if '_pre_decay.pt' in f]
        if pre_decay_files:
            self.saved_pre_decay_checkpoint = True
            print(f"   Found existing pre-decay checkpoint(s), will not save another")
        
        return checkpoint
    
    def get_latest_checkpoint_path(self) -> str:
        """Get path to the latest named non-final checkpoint."""
        checkpoint_path = latest_checkpoint_path(self.out_dir, include_final=False)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No named non-final checkpoints found in {self.out_dir}"
            )
        return str(checkpoint_path)
    
    def checkpoint_exists(self, checkpoint_path: Optional[str] = None) -> bool:
        """Check if a checkpoint exists.
        
        Args:
            checkpoint_path: Explicit checkpoint path. If None, checks for the
                latest named non-final checkpoint.
            
        Returns:
            True if checkpoint exists
        """
        try:
            resolved_checkpoint_path = self.resolve_checkpoint_path(checkpoint_path)
        except FileNotFoundError:
            return False
        return resolved_checkpoint_path is not None and os.path.exists(resolved_checkpoint_path)
