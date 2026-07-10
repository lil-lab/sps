"""
Interactive generation script for local checkpoints.

This script:
1. loads a checkpoint using the same Hydra config conventions as training/eval
2. runs a prepared batch of prompts
3. drops into an interactive REPL so you can try your own inputs

Prompt sources:
- `batch_prompts=[...]` on the command line
- `prompt_file=...` with `.txt`, `.json`, or `.jsonl`
- built-in toy prompts if neither is provided

Examples:
    python generate.py +experiment=s_sps_w64_10b +checkpoint=final
    python generate.py +experiment=... +checkpoint=final +prompt_file=prompts.jsonl +interactive=false
    python generate.py +experiment=... +checkpoint=final +batch_prompts='[\"Hello\",\"The answer is\"]' +do_sample=true +top_k=40
"""

from __future__ import annotations

import json
import os
import pickle
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
from torch import Tensor

try:
    import tiktoken
except ImportError:
    tiktoken = None

from training import CheckpointManager


DEFAULT_BATCH_PROMPTS = [
    "The meaning of life is",
    "Once upon a time",
    "Artificial intelligence will",
    "The key to success is",
    "In a world where",
    "The most important thing about science is",
]


@dataclass
class TokenizerBundle:
    encode: Callable[[str], list[int]]
    decode: Callable[[Sequence[int]], str]
    name: str
    invalid_token_ids: list[int]


def _resolve_checkpoint_path(checkpoint_path: str, out_dir: str) -> str:
    if os.path.isabs(checkpoint_path):
        return checkpoint_path

    cwd_candidate = os.path.normpath(checkpoint_path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    return os.path.join(out_dir, checkpoint_path)


def _coerce_prompt_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(item) for item in value]
    raise TypeError(f"Unsupported prompt list type: {type(value)!r}")


def _extract_prompt_record(item, *, source: str) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and "prompt" in item and isinstance(item["prompt"], str):
        return item["prompt"]
    raise ValueError(f"Could not parse prompt from {source}: {item!r}")


def _load_prompts_from_file(path: str) -> list[str]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    suffix = prompt_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(prompt_path.read_text())
        if isinstance(payload, dict) and "prompts" in payload:
            payload = payload["prompts"]
        if not isinstance(payload, list):
            raise ValueError(f"JSON prompt file must contain a list or a {{\"prompts\": [...]}} object: {prompt_path}")
        return [_extract_prompt_record(item, source=str(prompt_path)) for item in payload]

    if suffix == ".jsonl":
        prompts = []
        for line_no, line in enumerate(prompt_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            prompts.append(_extract_prompt_record(payload, source=f"{prompt_path}:{line_no}"))
        return prompts

    return [line.rstrip("\n") for line in prompt_path.read_text().splitlines() if line.strip()]


def _load_tokenizer(checkpoint: dict, model_config) -> TokenizerBundle:
    meta_path = None
    if isinstance(checkpoint.get("config"), dict):
        dataset_name = checkpoint["config"].get("dataset")
        if dataset_name:
            candidate = os.path.join("data", dataset_name, "meta.pkl")
            if os.path.exists(candidate):
                meta_path = candidate

    if meta_path is not None:
        print(f"Loading tokenizer from {meta_path}...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        invalid_token_ids = list(range(len(itos), int(model_config.vocab_size)))
        return TokenizerBundle(
            encode=lambda s: [stoi[c] for c in s],
            decode=lambda ids: "".join(itos[i] for i in ids),
            name=f"meta:{meta_path}",
            invalid_token_ids=invalid_token_ids,
        )

    if tiktoken is None:
        raise ImportError(
            "No meta.pkl found and tiktoken is not installed. "
            "Please install tiktoken or provide a dataset meta.pkl."
        )

    print("No meta.pkl found, using GPT-2 tokenization...")
    enc = tiktoken.get_encoding("gpt2")
    invalid_token_ids = list(range(enc.n_vocab, int(model_config.vocab_size)))
    return TokenizerBundle(
        encode=lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}),
        decode=lambda ids: enc.decode(list(ids)),
        name="gpt2",
        invalid_token_ids=invalid_token_ids,
    )


def _sample_from_logits(
    logits_BxV: Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    forbidden_token_ids: Optional[Tensor],
) -> Tensor:
    logits = logits_BxV.float().clone()
    if forbidden_token_ids is not None and forbidden_token_ids.numel() > 0:
        logits[:, forbidden_token_ids] = float("-inf")

    if do_sample:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0 when do_sample=True, got {temperature}")
        logits = logits / float(temperature)
        if top_k is not None:
            if top_k <= 0:
                raise ValueError(f"top_k must be positive when provided, got {top_k}")
            top_k = min(int(top_k), logits.size(-1))
            cutoff = torch.topk(logits, top_k, dim=-1).values[..., -1]
            logits = logits.masked_fill(logits < cutoff.unsqueeze(-1), float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(1)

    return logits.argmax(dim=-1)


def _forward_logits(model, idx_BxT: Tensor) -> Tensor:
    try:
        result = model(idx_BxT)
        return result[0] if isinstance(result, tuple) else result
    except TypeError:
        pass

    if hasattr(model, "forward_hidden_states") and hasattr(model, "lm_head"):
        try:
            x = model.forward_hidden_states(idx_BxT, idx_BxT.clone())[0]
        except TypeError:
            x = model.forward_hidden_states(idx_BxT)[0]
        return model.lm_head(x)

    raise TypeError(f"Do not know how to obtain logits from model type {type(model).__name__}")


def _left_pad_batch(token_lists: Sequence[Sequence[int]], pad_token_id: int, device: torch.device) -> Tensor:
    max_len = max(len(tokens) for tokens in token_lists)
    batch = torch.full((len(token_lists), max_len), pad_token_id, device=device, dtype=torch.long)
    for row, tokens in enumerate(token_lists):
        batch[row, max_len - len(tokens):] = torch.tensor(tokens, device=device, dtype=torch.long)
    return batch


def _clean_generated_tokens(tokens: Sequence[int], *, pad_token_id: int, eos_token_id: int) -> list[int]:
    cleaned = [tok for tok in tokens if tok != pad_token_id]
    if cleaned and cleaned[-1] == eos_token_id:
        cleaned = cleaned[:-1]
    return cleaned


def _truncate_prompt_tokens(
    prompt_tokens: list[int],
    *,
    block_size: int,
    max_new_tokens: int,
    native_generate: bool,
    truncate_left: bool,
    prompt_label: str,
) -> tuple[list[int], bool]:
    max_prompt_tokens = block_size - max_new_tokens if native_generate else block_size
    if max_prompt_tokens <= 0:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} leaves no room for the prompt with block_size={block_size}"
        )
    if len(prompt_tokens) <= max_prompt_tokens:
        return prompt_tokens, False
    if not truncate_left:
        raise ValueError(
            f"{prompt_label} has {len(prompt_tokens)} tokens but the model only allows "
            f"{max_prompt_tokens} prompt tokens in this generation mode"
        )
    return prompt_tokens[-max_prompt_tokens:], True


def _batched_generate_native(
    model,
    prompt_token_lists: list[list[int]],
    *,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    stop_on_eos: bool,
    forbidden_token_ids: Optional[Tensor],
    device: torch.device,
) -> list[list[int]]:
    completions: list[list[int]] = []
    for start in range(0, len(prompt_token_lists), batch_size):
        chunk = prompt_token_lists[start:start + batch_size]
        padded_prompt_BxT = _left_pad_batch(chunk, model.config.pad_token_id, device)
        generated_BxT = model.generate(
            padded_prompt_BxT,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            stop_on_eos=stop_on_eos,
            forbidden_token_ids=forbidden_token_ids,
        )
        prompt_width = padded_prompt_BxT.size(1)
        generated_only = generated_BxT[:, prompt_width:]
        for row in range(generated_only.size(0)):
            completions.append(generated_only[row].tolist())
    return completions


def _generate_single_fallback(
    model,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    stop_on_eos: bool,
    forbidden_token_ids: Optional[Tensor],
    device: torch.device,
) -> list[int]:
    tokens = list(prompt_tokens)
    generated: list[int] = []
    block_size = int(model.config.block_size)

    for _ in range(max_new_tokens):
        context = tokens[-block_size:]
        idx = torch.tensor([context], device=device, dtype=torch.long)
        logits = _forward_logits(model, idx)
        next_token = int(
            _sample_from_logits(
                logits[:, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                forbidden_token_ids=forbidden_token_ids,
            )[0].item()
        )
        generated.append(next_token)
        tokens.append(next_token)
        if stop_on_eos and next_token == model.config.eos_token_id:
            break

    return generated


def _generate_completions(
    model,
    prompt_token_lists: list[list[int]],
    *,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    stop_on_eos: bool,
    forbidden_token_ids: Optional[Tensor],
    device: torch.device,
) -> list[list[int]]:
    if hasattr(model, "generate"):
        return _batched_generate_native(
            model,
            prompt_token_lists,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            stop_on_eos=stop_on_eos,
            forbidden_token_ids=forbidden_token_ids,
            device=device,
        )

    print(
        f"Model type {type(model).__name__} has no native batched generate(); "
        "falling back to prompt-by-prompt autoregressive decoding."
    )
    return [
        _generate_single_fallback(
            model,
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            stop_on_eos=stop_on_eos,
            forbidden_token_ids=forbidden_token_ids,
            device=device,
        )
        for prompt_tokens in prompt_token_lists
    ]


def _print_completion(prompt: str, completion: str, *, index: Optional[int] = None, total: Optional[int] = None) -> None:
    if index is not None and total is not None:
        print(f"[{index}/{total}]")
    print("Prompt:")
    print(prompt)
    print("Completion:")
    print(completion)
    print("-" * 80)


def _sanitize_output_record(prompt: str, prompt_tokens: list[int], completion_tokens: list[int], tokenizer: TokenizerBundle) -> dict:
    completion_text = tokenizer.decode(completion_tokens)
    full_text = tokenizer.decode(prompt_tokens + completion_tokens)
    return {
        "prompt": prompt,
        "prompt_token_count": len(prompt_tokens),
        "completion_token_count": len(completion_tokens),
        "completion": completion_text,
        "full_text": full_text,
    }


def _write_output_records(output_path: str, records: list[dict]) -> None:
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} generation records to {output_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    if "checkpoint" not in cfg:
        cfg.checkpoint = None
    if "checkpoint_path" not in cfg:
        cfg.checkpoint_path = None
    if "apply_checkpoint_model_args" not in cfg:
        cfg.apply_checkpoint_model_args = True
    if "batch_prompts" not in cfg:
        cfg.batch_prompts = None
    if "prompt_file" not in cfg:
        cfg.prompt_file = None
    if "generation_batch_size" not in cfg:
        cfg.generation_batch_size = 8
    if "interactive" not in cfg:
        cfg.interactive = True
    if "max_new_tokens" not in cfg:
        cfg.max_new_tokens = 128
    if "do_sample" not in cfg:
        cfg.do_sample = False
    if "temperature" not in cfg:
        cfg.temperature = 1.0
    if "top_k" not in cfg:
        cfg.top_k = None
    if "stop_on_eos" not in cfg:
        cfg.stop_on_eos = True
    if "truncate_left" not in cfg:
        cfg.truncate_left = False
    if "seed" not in cfg:
        cfg.seed = 1337
    if "output_path" not in cfg:
        cfg.output_path = None
    OmegaConf.set_struct(cfg, True)

    print(OmegaConf.to_yaml(cfg))

    if hasattr(cfg, "out_dir"):
        out_dir = cfg.out_dir
    else:
        out_dir = os.path.join(cfg.system.data_root, "out", cfg.logging.wandb_run_name)

    device = torch.device(cfg.system.device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.system.dtype]
    def autocast_context():
        if device_type == "cpu":
            return nullcontext()
        return torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    print("\nGeneration configuration:")
    print(f"  Output directory: {out_dir}")
    print(f"  Device: {device}")
    print(f"  Dtype: {cfg.system.dtype}")
    print(f"  Max new tokens: {cfg.max_new_tokens}")
    print(f"  Batch size: {cfg.generation_batch_size}")
    print(f"  Sampling: {cfg.do_sample}")
    print(f"  Temperature: {cfg.temperature}")
    print(f"  Top-k: {cfg.top_k}")
    print(f"  Interactive: {cfg.interactive}")
    if cfg.truncate_left:
        raise ValueError(
            "Prompt truncation is disabled in generate.py. "
            "Shorten the prompt or reduce +max_new_tokens instead."
        )

    checkpoint_manager = CheckpointManager(
        out_dir=out_dir,
        save_every=cfg.training.save_every,
        master_process=True,
    )

    checkpoint_path = None
    if cfg.checkpoint_path:
        checkpoint_path = _resolve_checkpoint_path(str(cfg.checkpoint_path), out_dir)
    elif cfg.checkpoint:
        checkpoint_name = str(cfg.checkpoint)
        if checkpoint_name == "final":
            if not os.path.exists(out_dir):
                raise FileNotFoundError(f"Output directory {out_dir} does not exist.")
            final_checkpoints = sorted(f for f in os.listdir(out_dir) if "_final.pt" in f)
            if not final_checkpoints:
                raise FileNotFoundError(
                    f"No final checkpoint found in {out_dir}. "
                    f"Available checkpoints: {[f for f in os.listdir(out_dir) if f.endswith('.pt')]}"
                )
            checkpoint_path = os.path.join(out_dir, final_checkpoints[-1])
            print(f"Found final checkpoint: {os.path.basename(checkpoint_path)}")
        else:
            checkpoint_path = os.path.join(out_dir, checkpoint_name)
    elif os.path.exists(out_dir):
        final_checkpoints = sorted(f for f in os.listdir(out_dir) if "_final.pt" in f)
        if final_checkpoints:
            checkpoint_path = os.path.join(out_dir, final_checkpoints[-1])
            print(f"Auto-detected final checkpoint: {os.path.basename(checkpoint_path)}")

    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        if not checkpoint_manager.checkpoint_exists():
            raise FileNotFoundError(
                f"No checkpoint found in {out_dir}. "
                "Train a model first or specify checkpoint_path/checkpoint."
            )
        checkpoint_path = checkpoint_manager.get_latest_checkpoint_path()

    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = checkpoint_manager.load_checkpoint(str(device), checkpoint_path=checkpoint_path)
    checkpoint_model_args = checkpoint["model_args"]

    OmegaConf.set_struct(cfg, False)
    if cfg.apply_checkpoint_model_args:
        for key, value in checkpoint_model_args.items():
            cfg.model.config[key] = value
    if cfg.model.config.pad_token_id >= cfg.model.config.vocab_size:
        print(
            f"WARNING: pad_token_id ({cfg.model.config.pad_token_id}) >= vocab_size ({cfg.model.config.vocab_size}); "
            f"correcting to {cfg.model.config.vocab_size - 1}"
        )
        cfg.model.config.pad_token_id = cfg.model.config.vocab_size - 1
    OmegaConf.set_struct(cfg, True)

    print("Instantiating model...")
    model = instantiate(cfg.model)
    print(f"Model type: {type(model).__name__}")

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    model_sd = model.state_dict()

    strict = True
    if "freqs_cis" in state_dict and "freqs_cis" in model_sd:
        if state_dict["freqs_cis"].shape != model_sd["freqs_cis"].shape:
            state_dict = dict(state_dict)
            state_dict.pop("freqs_cis", None)
            strict = False

    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    model.to(device)

    tokenizer = _load_tokenizer(checkpoint, model.config)
    print(f"Tokenizer: {tokenizer.name}")

    forbidden_token_ids = set(tokenizer.invalid_token_ids)
    forbidden_token_ids.add(int(model.config.pad_token_id))
    predict_token_id = getattr(model.config, "predict_token_id", None)
    if predict_token_id is not None:
        forbidden_token_ids.add(int(predict_token_id))
    forbidden_token_ids.discard(int(model.config.eos_token_id))
    forbidden_token_ids_tensor = torch.tensor(
        sorted(tok for tok in forbidden_token_ids if 0 <= tok < int(model.config.vocab_size)),
        device=device,
        dtype=torch.long,
    )
    print(f"Masked {int(forbidden_token_ids_tensor.numel())} non-text/special token ids during sampling")

    prompt_sources = []
    prompt_sources.extend(_coerce_prompt_list(cfg.batch_prompts))
    if cfg.prompt_file:
        prompt_sources.extend(_load_prompts_from_file(str(cfg.prompt_file)))
    if cfg.batch_prompts is None and not cfg.prompt_file:
        prompt_sources = list(DEFAULT_BATCH_PROMPTS)

    if not prompt_sources:
        if not cfg.interactive:
            raise ValueError("No prompts provided and interactive=false; nothing to do.")
        print("\nNo prepared prompts configured. Entering interactive mode only.\n")

    prepared_prompts = [prompt for prompt in prompt_sources if prompt]
    native_generate = hasattr(model, "generate")
    prepared_prompt_tokens: list[list[int]] = []
    prepared_prompt_texts: list[str] = []
    truncated_count = 0
    for idx, prompt in enumerate(prepared_prompts):
        tokens = tokenizer.encode(prompt)
        if not tokens:
            raise ValueError(f"Prepared prompt {idx} tokenized to an empty sequence: {prompt!r}")
        tokens, was_truncated = _truncate_prompt_tokens(
            list(tokens),
            block_size=int(model.config.block_size),
            max_new_tokens=int(cfg.max_new_tokens),
            native_generate=native_generate,
            truncate_left=False,
            prompt_label=f"Prepared prompt {idx}",
        )
        truncated_count += int(was_truncated)
        prepared_prompt_tokens.append(tokens)
        prepared_prompt_texts.append(tokenizer.decode(tokens))

    output_records: list[dict] = []
    if prepared_prompt_tokens:
        print(f"\nRunning {len(prepared_prompt_tokens)} prepared prompts...")
        if truncated_count:
            print(f"Left-truncated {truncated_count} prepared prompts to fit the model context window")
        with torch.no_grad():
            with autocast_context():
                completion_token_lists = _generate_completions(
                    model,
                    prepared_prompt_tokens,
                    batch_size=int(cfg.generation_batch_size),
                    max_new_tokens=int(cfg.max_new_tokens),
                    do_sample=bool(cfg.do_sample),
                    temperature=float(cfg.temperature),
                    top_k=(None if cfg.top_k is None else int(cfg.top_k)),
                    stop_on_eos=bool(cfg.stop_on_eos),
                    forbidden_token_ids=forbidden_token_ids_tensor,
                    device=device,
                )
        for idx, (prompt_text, prompt_tokens, completion_tokens) in enumerate(
            zip(prepared_prompt_texts, prepared_prompt_tokens, completion_token_lists),
            start=1,
        ):
            cleaned_completion = _clean_generated_tokens(
                completion_tokens,
                pad_token_id=int(model.config.pad_token_id),
                eos_token_id=int(model.config.eos_token_id),
            )
            record = _sanitize_output_record(prompt_text, prompt_tokens, cleaned_completion, tokenizer)
            output_records.append(record)
            _print_completion(record["prompt"], record["completion"], index=idx, total=len(prepared_prompt_tokens))

    if cfg.output_path:
        _write_output_records(str(cfg.output_path), output_records)

    if not cfg.interactive:
        return

    print("\nInteractive mode. Type ':quit' or press Ctrl-D to exit.")
    while True:
        try:
            prompt = input("prompt> ")
        except EOFError:
            print()
            break
        if prompt.strip() in {":quit", ":exit"}:
            break
        if not prompt.strip():
            continue

        prompt_tokens = tokenizer.encode(prompt)
        if not prompt_tokens:
            print("Prompt tokenized to an empty sequence; try a different input.")
            continue
        prompt_tokens, was_truncated = _truncate_prompt_tokens(
            list(prompt_tokens),
            block_size=int(model.config.block_size),
            max_new_tokens=int(cfg.max_new_tokens),
            native_generate=native_generate,
            truncate_left=False,
            prompt_label="Interactive prompt",
        )
        prompt_text = tokenizer.decode(prompt_tokens)
        if was_truncated:
            print("Prompt was left-truncated to fit the model context window.")

        with torch.no_grad():
            with autocast_context():
                completion_tokens = _generate_completions(
                    model,
                    [prompt_tokens],
                    batch_size=1,
                    max_new_tokens=int(cfg.max_new_tokens),
                    do_sample=bool(cfg.do_sample),
                    temperature=float(cfg.temperature),
                    top_k=(None if cfg.top_k is None else int(cfg.top_k)),
                    stop_on_eos=bool(cfg.stop_on_eos),
                    forbidden_token_ids=forbidden_token_ids_tensor,
                    device=device,
                )[0]

        cleaned_completion = _clean_generated_tokens(
            completion_tokens,
            pad_token_id=int(model.config.pad_token_id),
            eos_token_id=int(model.config.eos_token_id),
        )
        completion_text = tokenizer.decode(cleaned_completion)
        print(completion_text)
        print("-" * 80)


if __name__ == "__main__":
    main()
