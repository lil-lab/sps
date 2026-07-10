"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py batch_size=32 compile=false

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with specific model config:
$ python train.py model=sps

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
from contextlib import nullcontext
from dataclasses import asdict
import math
import random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp
from torch.utils.data import DataLoader

from training import (
    TokenDataset,
    DistributedSequentialSampler,
    SequentialOffsetSampler,
    FixedRandomChunkDistributedSampler,
    CheckpointManager,
    prepare_wandb_dir_from_config,
)
from tqdm import trange


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Print config for debugging
    if int(os.environ.get('RANK', -1)) <= 0:
        print(OmegaConf.to_yaml(cfg))
    
    if not cfg.system.data_root:
        raise ValueError("cfg.system.data_root is required and must be a non-empty path")

    # Determine output directory from wandb run name
    dir_name = cfg.logging.wandb_run_name
    out_dir = os.path.join(cfg.system.data_root, "out", dir_name)
    
    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=cfg.system.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        print(f"DDP rank: {ddp_rank}, local rank: {ddp_local_rank}, world size: {ddp_world_size}, device: {device}")
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_rank = 0
        ddp_local_rank = 0
        device = cfg.system.device
    
    # Compute gradient accumulation steps from batch sizes
    assert cfg.training.global_batch_size % (cfg.training.micro_batch_size * ddp_world_size) == 0, \
        f"global_batch_size ({cfg.training.global_batch_size}) must be divisible by (micro_batch_size ({cfg.training.micro_batch_size}) * num_gpus ({ddp_world_size}))"
    gradient_accumulation_steps = cfg.training.global_batch_size // (cfg.training.micro_batch_size * ddp_world_size)
    if master_process:
        print(f"Batch configuration:")
        print(f"  micro_batch_size: {cfg.training.micro_batch_size} (per GPU)")
        print(f"  global_batch_size: {cfg.training.global_batch_size}")
        print(f"  num_gpus: {ddp_world_size}")
        print(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
    
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    
    base_seed = int(getattr(cfg.training, "seed", 1337))
    torch.manual_seed(base_seed + seed_offset)
    random.seed(base_seed + seed_offset)
    if getattr(cfg.system, 'deterministic', False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        if master_process:
            print("Deterministic mode enabled")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.system.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # LR-decay start tokens (validated/recomputed below). Needed up-front so that
    # resume resolution prefers the rolling ckpt.pt pre-decay and the latest named
    # checkpoint once in the decay phase.
    decay_start_tokens = (
        cfg.training.max_tokens - cfg.scheduler.lr_decay_tokens
        if cfg.scheduler.decay_lr else None
    )
    # Initialize checkpoint manager for loading (will be re-initialized with full params later)
    checkpoint_manager = CheckpointManager(
        out_dir=out_dir,
        save_every=cfg.training.save_every,
        master_process=master_process,
        decay_start_tokens=decay_start_tokens,
        rolling_save_every=cfg.training.rolling_save_every,
    )
    
    # Model initialization (moved earlier to get block_size)
    iter_num = 0
    best_val_loss = 1e9
    train_sampler = None  # Will be set later if needed
    wandb_run_id = None  # Will be loaded from checkpoint or generated by wandb
    checkpoint = None  # Initialize checkpoint variable
    next_eval_tokens = None

    if cfg.training.init_from == 'scratch':
        print("Initializing a new model from scratch")

        # Instantiate model using Hydra
        model = instantiate(cfg.model)
        print("Model instantiated:", type(model).__name__)
        print("Model config:", model.config)

        # Extract model args for checkpointing
        model_args = asdict(model.config)

    elif cfg.training.init_from == 'resume':
        resume_checkpoint = cfg.training.resume_checkpoint
        if resume_checkpoint is not None:
            checkpoint_path = checkpoint_manager.resolve_checkpoint_path(str(resume_checkpoint))
        else:
            checkpoint_path = checkpoint_manager.resolve_checkpoint_path()

        if checkpoint_path is not None:
            print(f"Resuming training from {out_dir}")
            checkpoint = checkpoint_manager.load_checkpoint(device, checkpoint_path=checkpoint_path)
            checkpoint_model_args = checkpoint['model_args']

            # Update config with checkpoint model args
            OmegaConf.set_struct(cfg, False)
            for k, v in checkpoint_model_args.items():
                cfg.model.config[k] = v
            OmegaConf.set_struct(cfg, True)

            # Instantiate model using Hydra
            model = instantiate(cfg.model)

            # Load state dict
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            state_dict.pop("freqs_cis", None)
            load_result = model.load_state_dict(state_dict, strict=False)
            missing_keys = load_result.missing_keys
            unexpected_keys = load_result.unexpected_keys
            if missing_keys or unexpected_keys:
                print("Checkpoint load with non-matching keys.")
                if missing_keys:
                    print(f"   Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"   Unexpected keys: {unexpected_keys}")

            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']
            model_args = checkpoint_model_args
            wandb_run_id = checkpoint.get('wandb_run_id')  # Load wandb run ID for resuming
            next_eval_tokens = int(checkpoint['next_eval_tokens'])
            if master_process:
                last_save_tokens = int(checkpoint.get('last_save_tokens', iter_num))
                print(
                    "Resume state: "
                    f"checkpoint={checkpoint_path}, iter_num={iter_num}, "
                    f"last_save_tokens={last_save_tokens:,}"
                )
        else:
            print(f"No checkpoint found in {out_dir}, starting from scratch instead")
            # Fall back to scratch initialization
            model = instantiate(cfg.model)
            print("Model instantiated:", type(model).__name__)
            print("Model config:", model.config)

            # Extract model args for checkpointing
            model_args = asdict(model.config)
    else:
        raise ValueError(f"Unknown init_from: {cfg.training.init_from}")
    
    # Get block_size from model config
    block_size = model.config.block_size
    
    # Calculate tokens per iteration (now that we have block_size)
    tokens_per_iter = cfg.training.global_batch_size * block_size
    
    if master_process:
        print(f"Batch configuration:")
        print(f"  micro_batch_size: {cfg.training.micro_batch_size} (per GPU)")
        print(f"  global_batch_size: {cfg.training.global_batch_size}")
        print(f"  num_gpus: {ddp_world_size}")
        print(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
        print(f"  tokens per iteration: {tokens_per_iter:,}")
    
    # Data loading
    data_dir = os.path.join(cfg.system.data_root, "data", cfg.data.dataset)
    train_bin_path = os.path.join(data_dir, "train.bin")
    val_bin_path = os.path.join(data_dir, "val.bin")
    if not os.path.exists(train_bin_path):
        raise FileNotFoundError(f"Missing train.bin at {train_bin_path}")
    if not os.path.exists(val_bin_path):
        raise FileNotFoundError(f"Missing val.bin at {val_bin_path}")
    
    # Create training dataset
    train_dataset = TokenDataset(
        train_bin_path,
        block_size
    )
    
    # Sampler configuration.
    sampler_start_offset = 0
    sampler_samples_seen_per_rank = 0
    sampler_type = str(getattr(cfg.training, "sampler_type", "sequential"))
    if sampler_type not in {"sequential", "fixed_random_chunk"}:
        raise ValueError(f"Unknown training.sampler_type={sampler_type!r}. Expected 'sequential' or 'fixed_random_chunk'.")
    if ddp:
        print(f"Creating train sampler type='{sampler_type}' (len={len(train_dataset):,}, world_size={ddp_world_size}, rank={ddp_rank})")
        # If resuming, restore starting offset
        if cfg.training.init_from == 'resume' and checkpoint is not None and 'sampler_offset' in checkpoint:
            sampler_start_offset = int(checkpoint['sampler_offset'])
            print(f"   Restored sampler start_offset to {sampler_start_offset}")
            if 'sampler_samples_seen_per_rank' in checkpoint:
                sampler_samples_seen_per_rank = int(checkpoint['sampler_samples_seen_per_rank'])
                print(f"   Restored sampler_samples_seen_per_rank to {sampler_samples_seen_per_rank}")
        else:
            # Randomize starting position for fresh training runs
            sampler_start_offset = random.randint(0, min(1_000_000, len(train_dataset) // 2))
            print(f"   Randomized starting position (offset={sampler_start_offset:,})")
        if sampler_type == "fixed_random_chunk":
            train_sampler = FixedRandomChunkDistributedSampler(
                dataset_len=len(train_dataset),
                num_replicas=ddp_world_size,
                rank=ddp_rank,
                block_size=block_size,
                chunk_size_units=int(getattr(cfg.training, "chunk_size_units", 262_144)),
                seed=int(getattr(cfg.training, "chunk_shuffle_seed", 1337)),
                start_offset=sampler_start_offset,
                resume_samples_seen_per_rank=sampler_samples_seen_per_rank,
            )
        else:
            train_sampler = DistributedSequentialSampler(
                dataset_len=len(train_dataset),
                num_replicas=ddp_world_size,
                rank=ddp_rank,
                block_size=block_size,
                drop_last=False,
                start_offset=sampler_start_offset,
            )
    else:
        if cfg.training.init_from == 'resume' and checkpoint is not None and 'sampler_offset' in checkpoint:
            sampler_start_offset = int(checkpoint['sampler_offset'])
            if 'sampler_samples_seen_per_rank' in checkpoint:
                sampler_samples_seen_per_rank = int(checkpoint['sampler_samples_seen_per_rank'])
            print(f"No DDP, restoring sampler offset={sampler_start_offset:,}")
        else:
            print(f"No DDP, using sampler from offset=0")
        if sampler_type == "fixed_random_chunk":
            train_sampler = FixedRandomChunkDistributedSampler(
                dataset_len=len(train_dataset),
                num_replicas=1,
                rank=0,
                block_size=block_size,
                chunk_size_units=int(getattr(cfg.training, "chunk_size_units", 262_144)),
                seed=int(getattr(cfg.training, "chunk_shuffle_seed", 1337)),
                start_offset=sampler_start_offset,
                resume_samples_seen_per_rank=sampler_samples_seen_per_rank,
            )
        else:
            train_sampler = SequentialOffsetSampler(
                dataset_len=len(train_dataset),
                start_offset=sampler_start_offset,
            )
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.micro_batch_size,
        sampler=train_sampler,
        shuffle=False,  # No shuffling - sequential access
        num_workers=0,  # 0 for memmap (shared memory)
        pin_memory=(device_type == 'cuda'),
        drop_last=False
    )
    
    # Calculate dataset statistics
    samples_per_iter = cfg.training.micro_batch_size * gradient_accumulation_steps
    num_train_positions = len(train_dataset)
    
    print(f"Dataset statistics:")
    print(f"   Dataset has {len(train_dataset.data):,} tokens")
    print(f"   Training positions: {num_train_positions:,}")
    print(f"   Samples per iteration: {samples_per_iter:,}")
    estimated_iters = cfg.training.max_tokens // tokens_per_iter
    print(f"   Training for {cfg.training.max_tokens:,} tokens (~{estimated_iters:,} iterations)")
    
    # Token-based evaluation schedule is required for deterministic resume behavior.
    if cfg.training.eval_interval_tokens is None:
        raise ValueError("cfg.training.eval_interval_tokens must be set (token-based eval scheduling only)")
    if cfg.training.eval_interval_tokens <= 0:
        raise ValueError(f"eval_interval_tokens must be > 0, got {cfg.training.eval_interval_tokens}")
    if cfg.training.eval_total_tokens is None:
        raise ValueError("cfg.training.eval_total_tokens must be set (evaluation workload is token-based)")
    if cfg.training.eval_total_tokens <= 0:
        raise ValueError(f"eval_total_tokens must be > 0, got {cfg.training.eval_total_tokens}")

    # Validate learning rate schedule configuration
    assert hasattr(cfg.scheduler, 'warmup_tokens'), "warmup_tokens must be specified in scheduler config"
    assert hasattr(cfg.scheduler, 'lr_decay_tokens'), "lr_decay_tokens must be specified in scheduler config"
    assert cfg.scheduler.warmup_tokens >= 0, f"warmup_tokens must be >= 0, got {cfg.scheduler.warmup_tokens}"
    assert cfg.scheduler.lr_decay_tokens > 0, f"lr_decay_tokens must be > 0, got {cfg.scheduler.lr_decay_tokens}"
    assert cfg.scheduler.warmup_tokens + cfg.scheduler.lr_decay_tokens <= cfg.training.max_tokens, \
        f"warmup_tokens ({cfg.scheduler.warmup_tokens:,}) + lr_decay_tokens ({cfg.scheduler.lr_decay_tokens:,}) " \
        f"exceeds max_tokens ({cfg.training.max_tokens:,})"
    
    # Calculate when LR decay starts (needed for pre-decay checkpoint)
    decay_start_tokens = cfg.training.max_tokens - cfg.scheduler.lr_decay_tokens if cfg.scheduler.decay_lr else None
    
    # Re-initialize checkpoint manager now that we have tokens_per_iter and decay_start_tokens
    checkpoint_manager = CheckpointManager(
        out_dir=out_dir,
        save_every=cfg.training.save_every,
        master_process=master_process,
        decay_start_tokens=decay_start_tokens,
        tokens_per_iter=tokens_per_iter,
        rolling_save_every=cfg.training.rolling_save_every,
    )
    # On resume, restore last_save_tokens and pre-decay state from the loaded checkpoint
    if cfg.training.init_from == 'resume' and checkpoint is not None:
        if 'last_save_tokens' in checkpoint:
            checkpoint_manager.last_save_tokens = checkpoint['last_save_tokens']
            # Resume the rolling cadence from the loaded token count.
            checkpoint_manager.last_rolling_save_tokens = int(checkpoint['last_save_tokens'])
        # Check if we've already saved a pre-decay checkpoint
        pre_decay_files = [f for f in os.listdir(out_dir) if '_pre_decay.pt' in f]
        if pre_decay_files:
            checkpoint_manager.saved_pre_decay_checkpoint = True
    
    if master_process and cfg.scheduler.decay_lr:
        print(f"Learning rate schedule:")
        if cfg.scheduler.warmup_tokens > 0:
            print(f"  Warmup: 0 -> {cfg.scheduler.warmup_tokens:,} tokens (~{cfg.scheduler.warmup_tokens // tokens_per_iter:,} iters)")
        print(f"  Constant LR: {cfg.scheduler.warmup_tokens:,} -> {decay_start_tokens:,} tokens (~{(decay_start_tokens - cfg.scheduler.warmup_tokens) // tokens_per_iter:,} iters)")
        print(f"  Linear decay: {decay_start_tokens:,} -> {cfg.training.max_tokens:,} tokens (~{cfg.scheduler.lr_decay_tokens // tokens_per_iter:,} iters)")
        print(f"  Pre-decay checkpoint will be saved at ~{decay_start_tokens:,} tokens")
    
    # Create iterator for training data
    train_iter = iter(train_loader)
    
    def get_batch():
        nonlocal train_iter, sampler_samples_seen_per_rank
        try:
            x, y = next(train_iter)
            sampler_samples_seen_per_rank += x.shape[0]
        except StopIteration:
            if master_process:
                print(f"Dataset pass complete at iter {iter_num}. Creating new iterator...")
            if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(train_sampler.epoch + 1)
            train_iter = iter(train_loader)
            x, y = next(train_iter)
            sampler_samples_seen_per_rank += x.shape[0]
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    
    # Move model to device (was already initialized above to get block_size)
    model.to(device)
    
    # GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.system.dtype == 'float16'))
    
    # Optimizer
    optimizer = model.configure_optimizers(cfg.optimizer.weight_decay, cfg.optimizer.learning_rate,
                                          (cfg.optimizer.beta1, cfg.optimizer.beta2), device_type)
    if cfg.training.init_from == 'resume' and checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as exc:
            print("Optimizer state mismatch; reinitializing optimizer state.")
            print(f"   Reason: {exc}")
        # Restore RNG state for exact resume reproducibility
        # RNG states must be CPU ByteTensors, but map_location may have moved them
        if 'cpu_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['cpu_rng_state'].cpu().byte())
        if 'cuda_rng_state' in checkpoint:
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu().byte())
    checkpoint = None
    
    # Compile
    if cfg.system.compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)
    
    # DDP wrap
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Evaluation function
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['val']:
            eval_batch_size = cfg.training.micro_batch_size * gradient_accumulation_steps
            eval_tokens_per_iter = eval_batch_size * block_size
            eval_iters = max(1, math.ceil(cfg.training.eval_total_tokens / eval_tokens_per_iter))
            losses = torch.zeros(eval_iters)
            eval_stats_sums = {}
            eval_stats_occurrences = {}
            val_rng = torch.Generator().manual_seed(42)
            for k in trange(eval_iters, disable=not master_process):
                batch_size = eval_batch_size
                data = np.memmap(val_bin_path, dtype=np.uint16, mode='r')
                ix = torch.randint(len(data) - block_size, (batch_size,), generator=val_rng)
                X = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
                Y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
                if device_type == 'cuda':
                    X, Y = X.pin_memory().to(device, non_blocking=True), Y.pin_memory().to(device, non_blocking=True)
                else:
                    X, Y = X.to(device), Y.to(device)
                with ctx:
                    loss_total = 0.0
                    stats_sums_iter = {}
                    stats_occurrences_iter = {}
                    for micro_step in range(gradient_accumulation_steps):
                        start = micro_step * cfg.training.micro_batch_size
                        end = start + cfg.training.micro_batch_size
                        X_slice = X[start:end]
                        Y_slice = Y[start:end]
                        model_out = model(X_slice, Y_slice)
                        _, loss_slice, stats = model_out
                        loss_total += float(loss_slice.detach())
                        if stats:
                            _accumulate_stats(stats, stats_sums_iter, stats_occurrences_iter)
                stats_iters = max(max(stats_occurrences_iter.values(), default=0), 1)
                losses[k] = loss_total / stats_iters
                for key, total in stats_sums_iter.items():
                    eval_stats_sums[key] = eval_stats_sums.get(key, 0.0) + total
                    eval_stats_occurrences[key] = eval_stats_occurrences.get(key, 0) + stats_occurrences_iter.get(key, 0)
            out[split] = losses.mean()
            for key, value in _summarize_stats(eval_stats_sums, eval_stats_occurrences).items():
                out[f"{split}_{key}"] = value
        model.train()
        return out
    
    # Learning rate scheduler
    def get_lr(it):
        if it == 0:
            it = 1
            
        tokens_seen = it * tokens_per_iter
        
        # Phase 1: Warmup
        if tokens_seen < cfg.scheduler.warmup_tokens:
            return cfg.optimizer.learning_rate * tokens_seen / cfg.scheduler.warmup_tokens
        
        # Calculate when decay starts
        decay_start_tokens = cfg.training.max_tokens - cfg.scheduler.lr_decay_tokens
        
        # Phase 2: Constant LR
        if tokens_seen < decay_start_tokens:
            return cfg.optimizer.learning_rate
        
        # Phase 3: Linear decay
        if tokens_seen >= cfg.training.max_tokens:
            return cfg.scheduler.min_lr
        
        tokens_in_decay = tokens_seen - decay_start_tokens
        decay_ratio = tokens_in_decay / cfg.scheduler.lr_decay_tokens
        coeff = 1.0 - decay_ratio
        return cfg.scheduler.min_lr + coeff * (cfg.optimizer.learning_rate - cfg.scheduler.min_lr)
    
    # Wandb logging
    if cfg.logging.wandb_log and master_process:
        import wandb
        # Convert config to dict for wandb
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb_dir = prepare_wandb_dir_from_config(cfg)
        wandb_init_kwargs = {"dir": wandb_dir} if wandb_dir is not None else {}

        # Extract experiment metadata if it exists
        wandb_tags = []
        if hasattr(cfg, 'experiment'):
            wandb_tags = cfg.experiment.get('tags', [])

        # Initialize wandb with resume support
        run = wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.logging.wandb_run_name,
            config=config_dict,
            entity=cfg.logging.wandb_entity,
            tags=wandb_tags,
            id=wandb_run_id,  # Use existing run ID when resuming
            resume="allow",    # Allow resuming if run ID exists
            **wandb_init_kwargs,
        )

        # Save the run ID for checkpointing (will be used when resuming)
        wandb_run_id = run.id
    elif wandb_run_id is None:
        # Generate a unique ID even if wandb is disabled (needed for checkpointing)
        wandb_run_id = f"local_{int(time.time())}"
    
    # Training loop
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model
    base_model = getattr(raw_model, "_orig_mod", raw_model)
    running_mfu = -1.0
    
    # For storing validation metrics to log together with train metrics
    val_metrics = None
    tokens_seen_start = iter_num * tokens_per_iter
    eval_interval_tokens = int(cfg.training.eval_interval_tokens)
    if next_eval_tokens is None:
        if getattr(cfg.training, "skip_first_eval", False) and not cfg.training.eval_only:
            next_eval_tokens = tokens_seen_start + eval_interval_tokens
        else:
            next_eval_tokens = tokens_seen_start
    
    # Ensure model is in training mode
    model.train()
    
    print(f"Starting training loop at iter_num={iter_num}, tokens_seen={tokens_seen_start:,}")

    def _format_tokens(count: int) -> str:
        units = ["", "K", "M", "B", "T"]
        value = float(count)
        unit = 0
        while abs(value) >= 1000.0 and unit < len(units) - 1:
            value /= 1000.0
            unit += 1
        if unit == 0:
            return f"{int(value)}"
        return f"{value:.2f}{units[unit]}"

    def _to_float(value):
        return float(value.item()) if torch.is_tensor(value) else float(value)

    def _accumulate_stats(stats_dict, stats_sums, stats_occurrences):
        for key, value in stats_dict.items():
            scalar_value = _to_float(value)
            stats_sums[key] = stats_sums.get(key, 0.0) + scalar_value
            stats_occurrences[key] = stats_occurrences.get(key, 0) + 1

    def _summarize_stats(stats_sums, stats_occurrences):
        metrics = {}
        for key, total in stats_sums.items():
            if key.endswith("_sum") or key.endswith("_count"):
                continue
            count = stats_occurrences.get(key, 0)
            if count > 0:
                metrics[key] = total / count
        for key, total_sum in stats_sums.items():
            if not key.endswith("_sum"):
                continue
            base = key[:-4]
            count_key = f"{base}_count"
            total_count = stats_sums.get(count_key, 0.0)
            if total_count > 0:
                metrics[base] = total_sum / total_count
        if "token_nll" in metrics:
            metrics["token_ppl"] = math.exp(metrics["token_nll"])
        return metrics

    def _reduce_stats_for_logging(local_total_loss, local_stats_sums, local_stats_occurrences):
        """All-reduce scalar logging accumulators across DDP ranks."""
        if not ddp:
            avg_loss_local = local_total_loss / gradient_accumulation_steps
            metrics_local = _summarize_stats(local_stats_sums, local_stats_occurrences)
            return avg_loss_local, metrics_local

        device_for_reduce = torch.device(device)
        world_size = float(ddp_world_size)

        # Diagnostic: log per-rank values before reduce
        local_nll_sum_val = local_stats_sums.get("token_nll_sum", float("nan"))
        local_nll_count_val = local_stats_sums.get("token_nll_count", 0.0)
        local_nll_val = local_nll_sum_val / max(local_nll_count_val, 1.0)
        if local_nll_val < 0.5 or not math.isfinite(local_nll_val):
            print(
                f"PRE-REDUCE rank={ddp_rank} | "
                f"local_loss={local_total_loss:.6f} | "
                f"nll_sum={local_nll_sum_val:.6f} | nll_count={local_nll_count_val:.0f} | "
                f"nll={local_nll_val:.6f}",
                flush=True,
            )

        loss_tensor = torch.tensor([local_total_loss], device=device_for_reduce, dtype=torch.float64)
        all_reduce(loss_tensor, op=ReduceOp.SUM)
        global_avg_loss = (loss_tensor.item() / world_size) / gradient_accumulation_steps

        all_keys = sorted(set(local_stats_sums.keys()) | set(local_stats_occurrences.keys()))
        if not all_keys:
            return global_avg_loss, {}

        packed = torch.zeros((len(all_keys), 2), device=device_for_reduce, dtype=torch.float64)
        for i, key in enumerate(all_keys):
            packed[i, 0] = float(local_stats_sums.get(key, 0.0))
            packed[i, 1] = float(local_stats_occurrences.get(key, 0))
        all_reduce(packed, op=ReduceOp.SUM)

        global_stats_sums = {key: float(packed[i, 0].item()) for i, key in enumerate(all_keys)}
        global_stats_occurrences = {key: int(round(packed[i, 1].item())) for i, key in enumerate(all_keys)}
        global_metrics = _summarize_stats(global_stats_sums, global_stats_occurrences)
        return global_avg_loss, global_metrics
    
    while True:
        # Set learning rate
        lr = get_lr(iter_num) if cfg.scheduler.decay_lr else cfg.optimizer.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate only on token schedule.
        tokens_seen = iter_num * tokens_per_iter
        should_eval = tokens_seen >= next_eval_tokens
        if should_eval and master_process:
            eval_start = time.perf_counter()
            losses = estimate_loss()
            eval_time = time.perf_counter() - eval_start
            tokens_seen = iter_num * tokens_per_iter
            val_nll = losses.get('val_token_nll', float("nan"))
            val_ppl = losses.get('val_token_ppl', float("nan"))
            print(
                f"eval | iter {iter_num:>6} | val nll: {val_nll:.4f} (ppl {val_ppl:.2f}) | "
                f"tokens: {_format_tokens(tokens_seen)} | eval: {eval_time*1000:.0f}ms"
            )
            while next_eval_tokens <= tokens_seen:
                next_eval_tokens += eval_interval_tokens
            
            # Store validation metrics to log together with train metrics later
            if cfg.logging.wandb_log:
                val_metrics = {"sys/eval_ms": eval_time * 1000.0}
                for key, value in losses.items():
                    if key.startswith("val_"):
                        val_metrics[f"val/{key[4:]}"] = value
            
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
        
        # Save checkpoint(s): named regular/pre-decay snapshots at save_every, plus
        # a frequently-refreshed rolling resume checkpoint (ckpt.pt) at
        # rolling_save_every during the pre-decay phase (so a crash wastes less compute).
        tokens_seen = iter_num * tokens_per_iter
        should_save, is_pre_decay = checkpoint_manager.should_save(tokens_seen, iter_num)
        should_save_rolling = checkpoint_manager.should_save_rolling(tokens_seen, iter_num)
        if should_save or should_save_rolling:
            if isinstance(train_sampler, FixedRandomChunkDistributedSampler):
                current_sampler_offset = sampler_start_offset
            elif ddp and hasattr(train_sampler, "stride"):
                current_sampler_offset = sampler_start_offset + sampler_samples_seen_per_rank * train_sampler.stride
            else:
                current_sampler_offset = sampler_start_offset + sampler_samples_seen_per_rank
            save_kwargs = dict(
                model_state=raw_model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                model_args=model_args,
                iter_num=iter_num,
                best_val_loss=best_val_loss,
                tokens_seen=tokens_seen,
                config=OmegaConf.to_container(cfg, resolve=True),
                wandb_run_id=wandb_run_id,
                sampler_offset=int(current_sampler_offset),
                sampler_samples_seen_per_rank=int(sampler_samples_seen_per_rank),
                next_eval_tokens=next_eval_tokens,
            )
            if should_save:
                checkpoint_manager.save_checkpoint(**save_kwargs, is_final=False, is_pre_decay=is_pre_decay)
            if should_save_rolling:
                checkpoint_manager.save_checkpoint(**save_kwargs, is_rolling=True)

        if iter_num == 0 and cfg.training.eval_only:
            break
        
        # Forward backward update
        total_loss_tensor = torch.zeros(1, device=device)
        train_stats_sums = {}
        train_stats_occurrences = {}
        for micro_step in range(gradient_accumulation_steps):
            # Get batch: counter naturally advances through the epoch
            X, Y = get_batch()

            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                out = model(X, Y)  # unscaled loss
                logits, loss, stats = out
                total_loss_tensor += loss.detach()
                loss = loss / gradient_accumulation_steps
            if stats is not None:
                _accumulate_stats(stats, train_stats_sums, train_stats_occurrences)
            scaler.scale(loss).backward()
        total_loss = total_loss_tensor.item()  # single GPU sync after all micro-steps

        # --- Diagnostic: per-rank anomaly detection ---
        local_nll_sum = train_stats_sums.get("token_nll_sum", float("nan"))
        local_nll_count = train_stats_sums.get("token_nll_count", 0.0)
        local_nll = local_nll_sum / max(local_nll_count, 1.0)
        loss_finite = torch.isfinite(total_loss_tensor).all().item()
        _is_anomalous = local_nll < 0.5 or not loss_finite or not math.isfinite(local_nll)

        if _is_anomalous:
            print(
                f"ANOMALY rank={ddp_rank} iter={iter_num} | "
                f"local_total_loss={total_loss:.6f} | "
                f"local_nll_sum={local_nll_sum:.6f} | "
                f"local_nll_count={local_nll_count:.0f} | "
                f"local_nll={local_nll:.6f} | "
                f"loss_finite={loss_finite}",
                flush=True,
            )

        if cfg.optimizer.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip)
            if _is_anomalous or not math.isfinite(grad_norm.item()):
                print(
                    f"GRAD rank={ddp_rank} iter={iter_num} | "
                    f"grad_norm={grad_norm.item():.6f} | "
                    f"finite={torch.isfinite(grad_norm).item()}",
                    flush=True,
                )

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if _is_anomalous:
            torch.cuda.synchronize()
            print(f"POST-OPTIM rank={ddp_rank} iter={iter_num} | CUDA sync OK", flush=True)
        
        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        global_avg_loss, global_train_metrics = _reduce_stats_for_logging(
            total_loss, train_stats_sums, train_stats_occurrences
        )

        if iter_num % cfg.training.log_interval == 0 and master_process:
            # Average loss across micro-steps for this iteration.
            avg_loss = global_avg_loss
            train_metrics = global_train_metrics
            avg_nll_loss = train_metrics.get("token_nll")
            train_ppl = train_metrics.get("token_ppl", float("nan"))
            tokens_seen = iter_num * tokens_per_iter
            if hasattr(base_model, 'estimate_mfu'):
                if local_iter_num >= 5:
                    mfu = base_model.estimate_mfu(cfg.training.micro_batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                nll_str = f" | nll: {avg_nll_loss:.4f}" if avg_nll_loss is not None else ""
                print(f"train | iter {iter_num:>6} | total: {avg_loss:.4f}{nll_str} | ppl {train_ppl:>7.2f} | lr: {lr:.2e} | {dt*1000:.0f}ms | mfu: {running_mfu*100:.2f}% | tokens: {tokens_seen:,}")
            else:
                nll_str = f" | nll: {avg_nll_loss:.4f}" if avg_nll_loss is not None else ""
                print(f"train | iter {iter_num:>6} | total: {avg_loss:.4f}{nll_str} | ppl {train_ppl:>7.2f} | lr: {lr:.2e} | {dt*1000:.0f}ms | tokens: {tokens_seen:,}")
            
            # Log training loss to wandb (combine with val metrics if available)
            if cfg.logging.wandb_log:
                log_dict = {
                    "iter": iter_num,
                    "tokens_seen": tokens_seen,
                    "train/total_loss": avg_loss,
                    "train/perplexity": train_ppl,
                    "lr": lr,
                    "mfu": running_mfu * 100,
                }
                for key, value in train_metrics.items():
                    log_dict[f"train/{key}"] = value
                # Keep old dashboard key name for compatibility.
                if "token_nll" in train_metrics:
                    log_dict["train/nll_loss"] = train_metrics["token_nll"]
                # Add validation metrics if they were computed this iteration
                if val_metrics is not None:
                    log_dict.update(val_metrics)
                    val_metrics = None  # Clear after logging
                
                wandb.log(log_dict)
        
        iter_num += 1
        local_iter_num += 1
        
        # Termination - check if we've reached max_tokens
        tokens_seen = iter_num * tokens_per_iter
        if tokens_seen >= cfg.training.max_tokens:
            if master_process:
                print(f"Reached max_tokens: {tokens_seen:,} >= {cfg.training.max_tokens:,}")
                # Final evaluation at end of training
                print(f"Running final evaluation...")
                eval_start = time.perf_counter()
                losses = estimate_loss()
                eval_time = time.perf_counter() - eval_start
                tokens_seen = iter_num * tokens_per_iter
                val_nll_eval = losses.get('val_token_nll', float("nan"))
                val_ppl_eval = losses.get('val_token_ppl', float("nan"))
                print(
                    f"final | iter {iter_num:>6} | val nll: {val_nll_eval:.4f} (ppl {val_ppl_eval:.2f}) | "
                    f"tokens: {_format_tokens(tokens_seen)} | eval: {eval_time*1000:.0f}ms"
                )
                
                if cfg.logging.wandb_log:
                    final_eval_log = {
                        "iter": iter_num,
                        "tokens_seen": tokens_seen,
                        "sys/eval_ms": eval_time * 1000.0,
                    }
                    for key, value in losses.items():
                        if key.startswith("val_"):
                            final_eval_log[f"val/{key[4:]}"] = value
                    wandb.log(final_eval_log)
                
                # Save final checkpoint
                if isinstance(train_sampler, FixedRandomChunkDistributedSampler):
                    current_sampler_offset = sampler_start_offset
                elif ddp and hasattr(train_sampler, "stride"):
                    current_sampler_offset = sampler_start_offset + sampler_samples_seen_per_rank * train_sampler.stride
                else:
                    current_sampler_offset = sampler_start_offset + sampler_samples_seen_per_rank
                checkpoint_manager.save_checkpoint(
                    model_state=raw_model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    model_args=model_args,
                    iter_num=iter_num,
                    best_val_loss=best_val_loss,
                    tokens_seen=tokens_seen,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    wandb_run_id=wandb_run_id,
                    sampler_offset=int(current_sampler_offset),
                    sampler_samples_seen_per_rank=int(sampler_samples_seen_per_rank),
                    next_eval_tokens=next_eval_tokens,
                    is_final=True
                )
            break
    
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
