"""
Evaluation script using EleutherAI lm-evaluation-harness.

Automatically loads checkpoints based on config and evaluates on standard benchmarks.

Usage:
    # Evaluate specific experiment on final checkpoint
    python evaluate.py +experiment=s_sps_w64_20b +checkpoint=final

    # Evaluate on a specific task
    python evaluate.py +experiment=s_sps_w64_20b tasks=hellaswag

    # Override evaluation parameters
    python evaluate.py +experiment=s_sps_w64_20b batch_size=8 limit=100
"""

import os
import base64
import json
import pickle
import re
import glob
import tempfile
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
try:
    import tiktoken
except ImportError:
    tiktoken = None

from lm_eval import simple_evaluate
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from modeling.evaluation.lm_eval_adapter import create_hflm_eval_model
from modeling.evaluation.metric_selection import choose_primary_metric_name
from training import CheckpointManager, prepare_wandb_dir_from_config


LOCAL_LM_EVAL_TASKS = frozenset({"gov_report_nll", "pile_books3"})


def _local_lm_eval_tasks_dir() -> str:
    # evaluate.py lives in scripts/; the lm_eval_tasks/ dir is at the repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "lm_eval_tasks")


def _resolve_checkpoint_path(checkpoint_path: str, out_dir: str) -> str:
    """
    Resolve a checkpoint path.

    Preference order for relative paths:
      1) As provided relative to CWD
      2) Relative to out_dir (back-compat)
    """
    if os.path.isabs(checkpoint_path):
        return checkpoint_path

    cwd_candidate = os.path.normpath(checkpoint_path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    return os.path.join(out_dir, checkpoint_path)


def _resolve_checkpoint_name_in_out_dir(checkpoint_name: str, out_dir: str) -> str:
    direct_candidate = os.path.join(out_dir, checkpoint_name)
    if os.path.exists(direct_candidate):
        return direct_candidate

    if checkpoint_name.endswith(".pt"):
        return direct_candidate

    suffixed_candidate = f"{direct_candidate}.pt"
    if os.path.exists(suffixed_candidate):
        return suffixed_candidate
    return suffixed_candidate


def _sanitize_task_name(task_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name).strip("._")


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return sanitized or "unknown"


def _sanitize_wandb_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("._-")
    return sanitized or "unknown"


def _normalize_eval_profile(eval_profile: str | None) -> str | None:
    if eval_profile is None:
        return None
    profile = str(eval_profile).strip()
    return profile or None


def _profile_subdir(eval_profile: str | None) -> str | None:
    normalized = _normalize_eval_profile(eval_profile)
    if normalized in (None, "classic"):
        return None
    return f"profile_{_sanitize_path_component(normalized)}"


def _build_eval_task_slug(tasks: list[str]) -> str:
    sanitized_tasks = [_sanitize_wandb_component(task) for task in tasks]
    full_slug = "-".join(sanitized_tasks)
    if len(full_slug) <= 48:
        return full_slug
    return f"{sanitized_tasks[0]}-plus{len(sanitized_tasks) - 1}"


def _build_eval_wandb_run_name(
    experiment_name: str,
    tasks: list[str],
    checkpoint_name: str,
    num_fewshot: int,
    eval_profile: str | None = None,
) -> str:
    task_slug = _build_eval_task_slug(tasks)
    exp_slug = _sanitize_wandb_component(experiment_name)
    prefix = f"eval_fs{int(num_fewshot)}"
    parts = [prefix, task_slug, exp_slug]
    normalized_profile = _normalize_eval_profile(eval_profile)
    if normalized_profile not in (None, "classic"):
        parts.insert(1, _sanitize_wandb_component(normalized_profile))
    if checkpoint_name and not checkpoint_name.endswith("_final"):
        parts.append(_sanitize_wandb_component(checkpoint_name))
    return "_".join(parts)


def _flatten_numeric_metrics(prefix: str, payload: dict) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            metric_key = _sanitize_wandb_component(key).replace("-", "_")
            flat[f"{prefix}/{metric_key}"] = float(value)
    return flat


def _select_primary_metric(task_name: str, task_results: dict[str, object]) -> tuple[str, float] | None:
    metric_name = choose_primary_metric_name(task_name, task_results)
    if metric_name is None:
        return None
    return metric_name, float(task_results[metric_name])


def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items() if not callable(v)}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    if callable(obj):
        return None
    return obj


def _checkpoint_eval_dir(
    out_dir: str,
    checkpoint_name: str,
    eval_profile: str | None = None,
    num_fewshot: int = 0,
) -> str:
    base_dir = os.path.join(out_dir, "eval", checkpoint_name)
    profile_subdir = _profile_subdir(eval_profile)
    if profile_subdir is not None:
        base_dir = os.path.join(base_dir, profile_subdir)
    if int(num_fewshot) <= 0:
        return base_dir
    return os.path.join(base_dir, f"fewshot_{int(num_fewshot)}")


def _task_output_path(
    out_dir: str,
    checkpoint_name: str,
    task_name: str,
    eval_profile: str | None = None,
    num_fewshot: int = 0,
) -> str:
    safe_task_name = _sanitize_task_name(task_name)
    if not safe_task_name:
        raise ValueError(f"Could not derive a safe filename for task {task_name!r}")
    return os.path.join(
        _checkpoint_eval_dir(
            out_dir,
            checkpoint_name,
            eval_profile=eval_profile,
            num_fewshot=num_fewshot,
        ),
        f"{safe_task_name}.json",
    )


def _task_result_exists(output_path: str, task_name: str) -> bool:
    if not os.path.exists(output_path):
        return False

    try:
        with open(output_path, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    return (
        isinstance(payload, dict)
        and isinstance(payload.get("results"), dict)
        and task_name in payload["results"]
    )


def _normalize_tasks(raw_tasks) -> list[str]:
    if isinstance(raw_tasks, str):
        tasks = [task.strip() for task in raw_tasks.split(",") if task.strip()]
    else:
        tasks = OmegaConf.to_container(raw_tasks, resolve=True)
        if not isinstance(tasks, list):
            tasks = [tasks]
        tasks = [str(task).strip() for task in tasks if str(task).strip()]
    return list(dict.fromkeys(tasks))


def _decode_b64_json_object(raw_payload: str | None, *, label: str) -> dict[str, object] | None:
    if raw_payload is None:
        return None
    encoded = str(raw_payload).strip()
    if not encoded:
        return None
    padded = encoded + ("=" * (-len(encoded) % 4))
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid {label} payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must decode to a JSON object")
    return payload


def _decode_task_metadata_b64(raw_metadata: str | None) -> dict[str, object] | None:
    return _decode_b64_json_object(raw_metadata, label="task_metadata_b64")


def _decode_task_config_b64(raw_config: str | None) -> dict[str, object] | None:
    return _decode_b64_json_object(raw_config, label="task_config_b64")


class _SingleCustomTaskManager:
    def __init__(self, task_name: str) -> None:
        self.task_index = {
            task_name: {
                "type": "task",
                "yaml_path": f"task_config_b64/{_sanitize_task_name(task_name)}.yaml",
            }
        }


def _custom_task_manager(task_name: str) -> _SingleCustomTaskManager:
    # lm-eval can evaluate ConfigurableTask objects directly, but its selected-task
    # logger still expects every string task key to exist in task_index.
    return _SingleCustomTaskManager(task_name)


def _task_manager_for_task(
    task_name: str,
    task_metadata: dict[str, object] | None,
) -> TaskManager | None:
    if task_name not in LOCAL_LM_EVAL_TASKS:
        return None
    return TaskManager(
        include_path=_local_lm_eval_tasks_dir(),
        include_defaults=False,
        metadata=task_metadata,
    )


def _extract_task_payload(results: dict, task_name: str) -> dict:
    available_tasks = set(results.get("results", {}).keys())
    task_payload = {}

    for key, value in results.items():
        if (
            isinstance(value, dict)
            and available_tasks
            and set(value.keys()).issubset(available_tasks)
        ):
            if task_name in value:
                task_payload[key] = {task_name: value[task_name]}
            continue

        task_payload[key] = value

    return task_payload


def _stage_json_payload(output_path: str, payload: dict) -> str:
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=".tmp_eval_",
        suffix=".json",
        dir=output_dir,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(_make_serializable(payload), f, indent=2)
    except Exception:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise
    return temp_path


def _wandb_run_dirs(wandb_dir: str | None) -> set[str]:
    root = wandb_dir if wandb_dir is not None else os.path.join(os.getcwd(), "wandb")
    return set(glob.glob(os.path.join(root, "run-*")))


def _latest_wandb_run_dir(previous_run_dirs: set[str], wandb_dir: str | None) -> str | None:
    current_run_dirs = _wandb_run_dirs(wandb_dir)
    new_dirs = sorted(current_run_dirs - previous_run_dirs)
    if new_dirs:
        return new_dirs[-1]
    existing_dirs = sorted(current_run_dirs)
    return existing_dirs[-1] if existing_dirs else None


def _is_retryable_wandb_init_error(exc: Exception) -> bool:
    message = str(exc)
    retry_markers = (
        "Failed to connect to service on socket",
        "Failed to connect to service on port",
        "Failed to read port info",
        "wandb-core exited with code",
    )
    return any(marker in message for marker in retry_markers)


def _clear_wandb_service_env() -> None:
    os.environ.pop("WANDB_SERVICE", None)


def _safe_finish_wandb_run(task_wandb_run, *, status: str, error: str | None = None, exit_code: int = 0) -> None:
    try:
        task_wandb_run.summary["status"] = status
        if error is not None:
            task_wandb_run.summary["error"] = error
        task_wandb_run.finish(exit_code=exit_code)
    except Exception as cleanup_exc:
        print(f"WARNING: failed to finalize wandb run after {status}: {cleanup_exc}")


def _init_eval_wandb_run(
    wandb_module,
    cfg: DictConfig,
    *,
    experiment_name: str,
    checkpoint_name: str,
    task_name: str,
    output_dir: str,
):
    eval_config_dict = OmegaConf.to_container(cfg, resolve=True)
    eval_config_dict["eval_task_name"] = task_name
    eval_config_dict["eval_pending_tasks"] = [task_name]
    eval_config_dict["eval_completed_tasks_at_start"] = []
    eval_config_dict["eval_checkpoint_name"] = checkpoint_name
    eval_config_dict["eval_output_dir"] = output_dir
    eval_profile = _normalize_eval_profile(getattr(cfg, "eval_profile", None)) or "classic"
    eval_config_dict["eval_profile"] = eval_profile

    wandb_tags = ["eval", f"fewshot_{int(cfg.num_fewshot)}"]
    if eval_profile != "classic":
        wandb_tags.append(f"profile_{_sanitize_wandb_component(eval_profile)}")
    if hasattr(cfg, "experiment"):
        wandb_tags.extend(cfg.experiment.get("tags", []))

    run_name = _build_eval_wandb_run_name(
        experiment_name=experiment_name,
        tasks=[task_name],
        checkpoint_name=checkpoint_name,
        num_fewshot=int(cfg.num_fewshot),
        eval_profile=eval_profile,
    )
    wandb_dir = prepare_wandb_dir_from_config(cfg)
    wandb_init_kwargs = {"dir": wandb_dir} if wandb_dir is not None else {}
    last_exc = None
    for attempt in (1, 2):
        existing_run_dirs = _wandb_run_dirs(wandb_dir)
        try:
            task_wandb_run = wandb_module.init(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                name=run_name,
                group=experiment_name,
                job_type="eval",
                tags=wandb_tags,
                config=eval_config_dict,
                **wandb_init_kwargs,
            )
            task_wandb_run.summary["experiment_name"] = experiment_name
            task_wandb_run.summary["checkpoint_name"] = checkpoint_name
            task_wandb_run.summary["task_name"] = task_name
            task_wandb_run.summary["num_fewshot"] = int(cfg.num_fewshot)
            task_wandb_run.summary["eval_profile"] = eval_profile
            task_wandb_run.summary["requested_tasks"] = task_name
            task_wandb_run.summary["output_dir"] = output_dir
            task_wandb_run.summary["status"] = "running"
            return task_wandb_run
        except Exception as exc:
            last_exc = exc
            debug_dir = _latest_wandb_run_dir(existing_run_dirs, wandb_dir)
            print(f"ERROR: failed to initialize wandb for task {task_name}: {exc}")
            print(f"       WANDB_SERVICE set: {'WANDB_SERVICE' in os.environ}")
            if debug_dir is not None:
                print(f"       W&B debug dir: {debug_dir}")
            if attempt == 1 and _is_retryable_wandb_init_error(exc):
                print("       Retrying wandb initialization once after teardown...")
                try:
                    wandb_module.teardown(exit_code=1)
                except Exception as teardown_exc:
                    print(f"       wandb.teardown() failed during retry prep: {teardown_exc}")
                _clear_wandb_service_env()
                continue
            break

    raise RuntimeError(f"wandb initialization failed for task {task_name}: {last_exc}") from last_exc


def _apply_eval_checkpoint_model_args(
    model_config: DictConfig,
    checkpoint_model_args: dict[str, object],
    *,
    checkpoint_name: str,
    model_target: str | None = None,
) -> None:
    """Apply saved checkpoint model args for eval, skipping obsolete config keys."""
    current_keys = set(model_config.keys())
    skipped_keys = sorted(key for key in checkpoint_model_args if key not in current_keys)

    for key, value in checkpoint_model_args.items():
        if key in current_keys:
            model_config[key] = value

    if skipped_keys:
        model_label = model_target or "unknown_model"
        skipped = ", ".join(skipped_keys)
        print(
            "WARNING: skipping obsolete checkpoint model args during eval "
            f"for {checkpoint_name} ({model_label}): {skipped}"
        )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Temporarily disable struct mode to add evaluation options if needed
    OmegaConf.set_struct(cfg, False)
    
    # Add checkpoint config if provided via command line
    if 'checkpoint' not in cfg:
        cfg.checkpoint = None
    
    # Add evaluation-specific configs with defaults
    if 'tasks' not in cfg:
        cfg.tasks = None
    
    if 'num_fewshot' not in cfg:
        cfg.num_fewshot = 0
    
    if 'limit' not in cfg:
        cfg.limit = None  # None means evaluate on full dataset
    
    if 'eval_batch_size' not in cfg:
        cfg.eval_batch_size = "auto"
    
    if 'max_batch_size' not in cfg:
        cfg.max_batch_size = 4096
    elif cfg.max_batch_size is not None:
        cfg.max_batch_size = int(cfg.max_batch_size)

    if "min_eval_seq_len" not in cfg:
        cfg.min_eval_seq_len = None
    elif cfg.min_eval_seq_len is not None:
        cfg.min_eval_seq_len = int(cfg.min_eval_seq_len)
    
    if 'output_path' not in cfg:
        cfg.output_path = None  # Will be set based on out_dir

    if "eval_profile" not in cfg:
        cfg.eval_profile = None

    if "task_metadata_b64" not in cfg:
        cfg.task_metadata_b64 = None

    if "task_config_b64" not in cfg:
        cfg.task_config_b64 = None

    if "required_block_size" not in cfg:
        cfg.required_block_size = None
    elif cfg.required_block_size is not None:
        cfg.required_block_size = int(cfg.required_block_size)

    if "eval_seed" not in cfg:
        cfg.eval_seed = 42

    if "use_forward_efficient" not in cfg:
        # Default to the dense training forward (Triton), matching the paper evaluations.
        # forward_efficient is numerically identical but far slower; opt in explicitly.
        cfg.use_forward_efficient = False

    # If false, we only load the checkpoint weights; model behavior comes entirely from `+experiment=...`.
    # This is useful for evaluating a checkpoint under a different predict scheme.
    if "apply_checkpoint_model_args" not in cfg:
        cfg.apply_checkpoint_model_args = False
    
    # Re-enable struct mode
    OmegaConf.set_struct(cfg, True)
    
    # Print config for debugging
    print(OmegaConf.to_yaml(cfg))

    if not cfg.tasks:
        raise ValueError("evaluate.py requires tasks to be passed explicitly")

    tasks = _normalize_tasks(cfg.tasks)
    if len(tasks) != 1:
        raise ValueError(
            f"evaluate.py requires exactly one task, got {len(tasks)}: {tasks}"
        )
    task_name = tasks[0]

    # Determine output directory from wandb run name (same as train.py)
    if hasattr(cfg, 'out_dir'):
        out_dir = cfg.out_dir
    else:
        dir_name = cfg.logging.wandb_run_name
        out_dir = os.path.join(cfg.system.data_root, "out", dir_name)

    device = cfg.system.device
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    print(f"\nEvaluation configuration:")
    print(f"  Output directory: {out_dir}")
    print(f"  Device: {device}")
    print(f"  Dtype: {cfg.system.dtype}")
    print(f"  Task: {task_name}")
    print(f"  Eval batch size: {cfg.eval_batch_size}")
    print(f"  Max batch size: {cfg.max_batch_size}")
    print(f"  Min eval seq len: {cfg.min_eval_seq_len}")
    print(f"  Use forward_efficient: {cfg.use_forward_efficient}")
    print(f"  Num fewshot: {cfg.num_fewshot}")
    print(f"  Eval profile: {_normalize_eval_profile(cfg.eval_profile) or 'classic'}")
    if cfg.limit:
        print(f"  Limit: {cfg.limit} examples")

    task_metadata = _decode_task_metadata_b64(cfg.task_metadata_b64)
    if task_metadata:
        print(f"  Task metadata: {task_metadata}")

    task_config = _decode_task_config_b64(cfg.task_config_b64)
    if task_config:
        print(f"  Task config override: {task_config}")

    checkpoint_manager = CheckpointManager(
        out_dir=out_dir,
        save_every=cfg.training.save_every,
        master_process=True
    )

    checkpoint_path = None
    if hasattr(cfg, 'checkpoint_path') and cfg.checkpoint_path:
        checkpoint_path = _resolve_checkpoint_path(str(cfg.checkpoint_path), out_dir)
    elif hasattr(cfg, 'checkpoint') and cfg.checkpoint:
        requested_checkpoint = cfg.checkpoint
        if requested_checkpoint == "final":
            if os.path.exists(out_dir):
                final_checkpoints = [f for f in os.listdir(out_dir) if '_final.pt' in f]
                if final_checkpoints:
                    final_checkpoints.sort(reverse=True)
                    checkpoint_path = os.path.join(out_dir, final_checkpoints[0])
                    print(f"Found final checkpoint: {final_checkpoints[0]}")
                else:
                    raise FileNotFoundError(
                        f"No final checkpoint found in {out_dir}. "
                        f"Available checkpoints: {[f for f in os.listdir(out_dir) if f.endswith('.pt')]}"
                    )
            else:
                raise FileNotFoundError(f"Output directory {out_dir} does not exist.")
        else:
            checkpoint_path = _resolve_checkpoint_name_in_out_dir(str(requested_checkpoint), out_dir)
    else:
        if os.path.exists(out_dir):
            final_checkpoints = [f for f in os.listdir(out_dir) if '_final.pt' in f]
            if final_checkpoints:
                final_checkpoints.sort(reverse=True)
                checkpoint_path = os.path.join(out_dir, final_checkpoints[0])
                print(f"Auto-detected final checkpoint: {final_checkpoints[0]}")

    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
    else:
        if not checkpoint_manager.checkpoint_exists():
            raise FileNotFoundError(
                f"No checkpoint found in {out_dir}. "
                f"Make sure you've trained a model first or specify a different out_dir."
            )
        checkpoint_path = checkpoint_manager.get_latest_checkpoint_path()
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')

    if cfg.output_path:
        task_output_path = cfg.output_path
        output_dir = os.path.dirname(cfg.output_path) or "."
    else:
        output_dir = _checkpoint_eval_dir(
            out_dir,
            checkpoint_name,
            eval_profile=cfg.eval_profile,
            num_fewshot=cfg.num_fewshot,
        )
        task_output_path = _task_output_path(
            out_dir,
            checkpoint_name,
            task_name,
            eval_profile=cfg.eval_profile,
            num_fewshot=cfg.num_fewshot,
        )

    print(f"\nEvaluating on task: {task_name}")
    print(f"Per-task results directory: {output_dir}")
    if _task_result_exists(task_output_path, task_name):
        print(f"Skipping already evaluated task: {task_name}")
        return

    os.makedirs(output_dir or ".", exist_ok=True)

    wandb = None
    task_wandb_run = None
    experiment_name = (
        cfg.experiment.name
        if hasattr(cfg, "experiment") and "name" in cfg.experiment
        else cfg.logging.wandb_run_name
    )
    if cfg.logging.wandb_log:
        import wandb

        try:
            task_wandb_run = _init_eval_wandb_run(
                wandb,
                cfg,
                experiment_name=experiment_name,
                checkpoint_name=checkpoint_name,
                task_name=task_name,
                output_dir=output_dir,
            )
        except Exception:
            try:
                wandb.teardown(exit_code=1)
            except Exception as teardown_exc:
                print(f"WARNING: wandb teardown failed after init error: {teardown_exc}")
            raise

    staged_output_path = None
    final_output_written = False
    try:
        checkpoint_load_path = checkpoint_path
        print(f"\nLoading checkpoint from {checkpoint_load_path}...")
        checkpoint = checkpoint_manager.load_checkpoint(device, checkpoint_path=checkpoint_load_path)
        checkpoint_model_args = checkpoint['model_args']

        OmegaConf.set_struct(cfg, False)
        if cfg.apply_checkpoint_model_args:
            _apply_eval_checkpoint_model_args(
                cfg.model.config,
                checkpoint_model_args,
                checkpoint_name=checkpoint_name,
                model_target=str(cfg.model.get("_target_", "unknown_model")),
            )

        if cfg.model.config.pad_token_id >= cfg.model.config.vocab_size:
            print(f"WARNING: pad_token_id ({cfg.model.config.pad_token_id}) >= vocab_size ({cfg.model.config.vocab_size})")
            print(f"         Correcting to pad_token_id = {cfg.model.config.vocab_size - 1}")
            cfg.model.config.pad_token_id = cfg.model.config.vocab_size - 1

        OmegaConf.set_struct(cfg, True)

        print("Instantiating model...")
        model = instantiate(cfg.model)
        print(f"Model type: {type(model).__name__}")

        if (
            cfg.required_block_size is not None
            and int(model.config.block_size) < int(cfg.required_block_size)
        ):
            raise ValueError(
                f"Eval profile {_normalize_eval_profile(cfg.eval_profile) or 'classic'} "
                f"requires block_size >= {cfg.required_block_size}, "
                f"but model checkpoint provides block_size={model.config.block_size}"
            )

        state_dict = checkpoint["model"]
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        model_sd = model.state_dict()

        strict = True
        if "freqs_cis" in state_dict:
            if "freqs_cis" in model_sd and state_dict["freqs_cis"].shape != model_sd["freqs_cis"].shape:
                state_dict = dict(state_dict)
                state_dict.pop("freqs_cis", None)
                strict = False

        model.load_state_dict(state_dict, strict=strict)

        model.eval()
        model.to(device)

        load_meta = False
        if 'config' in checkpoint and 'dataset' in checkpoint['config']:
            meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
            load_meta = os.path.exists(meta_path)

        if load_meta:
            print(f"Loading tokenizer from {meta_path}...")
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
            invalid_token_ids = list(range(len(itos), int(model.config.vocab_size)))
            lm_eval_model_args = None
        else:
            if tiktoken is None:
                raise ImportError(
                    "No meta.pkl found and tiktoken is not installed. "
                    "Please install tiktoken: pip install tiktoken"
                )
            print("No meta.pkl found, using GPT-2 tokenization...")
            enc = tiktoken.get_encoding("gpt2")
            encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
            decode = lambda l: enc.decode(l)
            tokenizer_vocab_size = int(getattr(enc, "n_vocab", model.config.vocab_size))
            invalid_token_ids = list(range(tokenizer_vocab_size, int(model.config.vocab_size)))
            lm_eval_model_args = {"pretrained": "gpt2"}

        if isinstance(cfg.eval_batch_size, str):
            if cfg.eval_batch_size.lower().startswith("auto"):
                eval_batch_size = cfg.eval_batch_size
            else:
                eval_batch_size = int(cfg.eval_batch_size)
        else:
            eval_batch_size = int(cfg.eval_batch_size)

        print("\nCreating lm-eval adapter...")
        lm_eval_model = create_hflm_eval_model(
            model=model,
            config=model.config,
            tokenizer_encode=encode,
            tokenizer_decode=decode,
            invalid_token_ids=invalid_token_ids,
            device=device,
            batch_size=eval_batch_size,
            max_batch_size=cfg.max_batch_size,
            min_eval_seq_len=cfg.min_eval_seq_len,
            use_forward_efficient=cfg.use_forward_efficient,
        )

        print("\n" + "="*70)
        print("Starting evaluation...")
        print("="*70 + "\n")
        print(f"[1/1] Evaluating task: {task_name}")

        torch.manual_seed(cfg.eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.eval_seed)

        if hasattr(lm_eval_model, "reset_eval_run_state"):
            lm_eval_model.reset_eval_run_state()
        if device_type == "cuda":
            torch.cuda.empty_cache()

        task_spec: str | ConfigurableTask = task_name
        task_manager = _task_manager_for_task(task_name, task_metadata)
        if task_config is not None:
            task_config_with_name = dict(task_config)
            task_config_with_name["task"] = task_name
            task_spec = ConfigurableTask(config=task_config_with_name)
            task_manager = _custom_task_manager(task_name)

        simple_evaluate_kwargs = dict(
            model=lm_eval_model,
            tasks=[task_spec],
            num_fewshot=cfg.num_fewshot,
            batch_size=eval_batch_size,
            device=device,
            max_batch_size=cfg.max_batch_size,
            limit=cfg.limit,
        )
        if lm_eval_model_args is not None:
            simple_evaluate_kwargs["model_args"] = lm_eval_model_args
        if task_metadata is not None:
            simple_evaluate_kwargs["metadata"] = task_metadata
        if task_manager is not None:
            simple_evaluate_kwargs["task_manager"] = task_manager
        results = simple_evaluate(**simple_evaluate_kwargs)

        staged_output_path = _stage_json_payload(task_output_path, results)

        print("\nResults Summary:")
        print("-" * 70)
        if 'results' in results:
            for result_task_name, task_results in results['results'].items():
                print(f"\n{result_task_name}:")
                primary_metric = _select_primary_metric(result_task_name, task_results)
                if primary_metric is not None:
                    metric_name, metric_value = primary_metric
                    print(f"  primary_metric: {metric_name} = {metric_value:.4f}")
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"  {metric_name}: {metric_value:.4f}")

        os.replace(staged_output_path, task_output_path)
        staged_output_path = None
        final_output_written = True

        if task_wandb_run is not None:
            task_results = results.get("results", {}).get(task_name, {})
            primary_metric = _select_primary_metric(task_name, task_results)
            log_payload = {
                "eval/task_index_in_bundle": 1,
                "eval/tasks_total_in_bundle": 1,
            }
            log_payload.update(_flatten_numeric_metrics("eval", task_results))
            if primary_metric is not None:
                _, primary_score = primary_metric
                log_payload["eval/primary_score"] = primary_score
            if len(log_payload) > 2:
                task_wandb_run.log(log_payload, step=1)

            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    summary_key = (
                        "eval/"
                        f"{_sanitize_wandb_component(metric_name).replace('-', '_')}"
                    )
                    task_wandb_run.summary[summary_key] = float(metric_value)
            if primary_metric is not None:
                primary_metric_name, primary_score = primary_metric
                task_wandb_run.summary["eval/primary_metric_name"] = primary_metric_name
                task_wandb_run.summary["eval/primary_score"] = primary_score
            task_wandb_run.summary["result_path"] = task_output_path
            task_wandb_run.summary["status"] = "completed"
            task_wandb_run.finish()
            task_wandb_run = None

        print(f"\nSaved {task_name} results to: {task_output_path}")
        print("\n" + "="*70)
        print("Evaluation complete!")
        print("="*70)

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        traceback.print_exc()
        if staged_output_path and os.path.exists(staged_output_path):
            try:
                os.unlink(staged_output_path)
            except OSError as cleanup_exc:
                print(f"WARNING: failed to remove staged eval output {staged_output_path}: {cleanup_exc}")
        if final_output_written and os.path.exists(task_output_path):
            try:
                os.unlink(task_output_path)
            except OSError as cleanup_exc:
                print(f"WARNING: failed to remove failed eval output {task_output_path}: {cleanup_exc}")
        if task_wandb_run is not None:
            _safe_finish_wandb_run(
                task_wandb_run,
                status="failed",
                error=str(e),
                exit_code=1,
            )
        raise
    finally:
        if wandb is not None:
            try:
                wandb.teardown()
            except Exception as teardown_exc:
                print(f"WARNING: wandb teardown failed: {teardown_exc}")


if __name__ == "__main__":
    main()
