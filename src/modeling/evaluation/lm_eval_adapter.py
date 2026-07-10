"""
Custom adapter for EleutherAI lm-evaluation-harness.

This adapter scores examples through the model's training-time ``forward()``
path and also supports decoder-only ``generate_until`` benchmarks through a
stepwise generation loop. That lets the SPS-family models (SPS, Reverse-SPS,
Delayed-State) participate in both loglikelihood and generation tasks without a
shared HF causal-LM interface.
"""

from contextlib import contextmanager
from typing import Optional

import torch
import lm_eval.models.huggingface as lm_eval_huggingface
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model

class _FunctionTokenizer:
    """
    Minimal tokenizer wrapper to satisfy HFLM expectations while reusing
    pre-existing encode/decode callables.
    """

    def __init__(self, encode_fn, decode_fn, pad_token_id: int, eos_token_id: int, block_size: int):
        self._encode = encode_fn
        self._decode = decode_fn
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        # Fall back to EOS for BOS if none is provided by the config
        self.bos_token_id = eos_token_id
        # HFLM uses this when deciding truncation length
        self.model_max_length = block_size
        self.padding_side = "left"
        self.name_or_path = "function_tokenizer"
        self.vocab_size = max(pad_token_id, eos_token_id) + 1

        # Print all the attributes in a nice format, altogether
        print(
            f"_FunctionTokenizer initialized with:\n"
            f"  pad_token_id = {self.pad_token_id}\n"
            f"  eos_token_id = {self.eos_token_id}\n"
            f"  bos_token_id = {self.bos_token_id}\n"
            f"  model_max_length = {self.model_max_length}"
        )
        

    def encode(self, text: str, add_special_tokens: bool = False, truncation: bool = False, max_length: int = None):
        tokens = list(self._encode(text))
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return tokens

    def decode(self, tokens, skip_special_tokens: bool = True):
        if isinstance(tokens, int):
            tokens = [tokens]
        if skip_special_tokens:
            tokens = [
                token
                for token in tokens
                if token not in {self.pad_token_id, self.eos_token_id, self.bos_token_id}
            ]
        return self._decode(tokens)

    def __call__(self, texts, padding=False, return_tensors=None, truncation=False, max_length=None):
        # Support the subset of the HF tokenizer API that HFLM calls into
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.encode(t, truncation=truncation, max_length=max_length) for t in texts]
        max_len = max(len(t) for t in encoded)

        input_ids = []
        attention_mask = []
        for t in encoded:
            pad_len = max_len - len(t)
            if self.padding_side == "right":
                input_ids.append(t + [self.pad_token_id] * pad_len)
                attention_mask.append([1] * len(t) + [0] * pad_len)
            else:
                input_ids.append([self.pad_token_id] * pad_len + t)
                attention_mask.append([0] * pad_len + [1] * len(t))

        result = {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask)}
        return result


@register_model("custom_hflm")
class CustomHFLMAdapter(HFLM):
    """
    Lightweight adapter that reuses HFLM batching while swapping in the
    preloaded model, tokenizer, and our training-time forward for accuracy.

    To run under Slurm with uv resolving deps:
        sbatch --wrap="uv run python evaluate.py +experiment=baseline_standard +checkpoint=final"
    """

    def __init__(
        self,
        model,
        tokenizer_encode,
        tokenizer_decode,
        invalid_token_ids,
        config,
        device: str = "cuda",
        batch_size: int | str = 1,
        max_batch_size: int | None = None,
        min_eval_seq_len: int | None = None,
        use_forward_efficient: bool = False,
    ):
        # Don't call super().__init__ - it will try to load a model
        # Instead, initialize LM base class and set up everything manually
        LM.__init__(self)
        
        # Store our custom pieces
        self._raw_model = model
        self._use_forward_efficient = use_forward_efficient
        self._min_eval_seq_len = (
            int(min_eval_seq_len) if min_eval_seq_len is not None else None
        )
        if self._min_eval_seq_len is not None and self._min_eval_seq_len < 1:
            raise ValueError("min_eval_seq_len must be >= 1 when provided")
        
        # Store actual vocab size (for logit slicing)
        self._vocab_size = config.vocab_size
        self.tokenizer = _FunctionTokenizer(
            encode_fn=tokenizer_encode,
            decode_fn=tokenizer_decode,
            pad_token_id=config.pad_token_id,
            eos_token_id=config.eos_token_id,
            block_size=config.block_size,
        )

        self._model = model
        self._model.eval()
        self._device = torch.device(device)
        self._model.to(self._device)

        # Set private attributes that HFLM properties use
        if str(batch_size).startswith("auto"):
            batch_size_parts = str(batch_size).split(":")
            self.batch_size_per_gpu = batch_size_parts[0]
            self.batch_schedule = (
                float(batch_size_parts[1]) if len(batch_size_parts) > 1 else 1
            )
            self.max_batch_size = (
                int(max_batch_size) if max_batch_size is not None else 64
            )
        else:
            self.batch_size_per_gpu = int(batch_size)
            self.max_batch_size = (
                int(max_batch_size)
                if max_batch_size is not None
                else self.batch_size_per_gpu
            )
        self._max_length = config.block_size
        self._max_gen_toks = config.block_size
        self._config = config
        self.truncation = False
        self.logits_cache = True
        self.vocab_size = self._vocab_size
        self.backend = "causal"
        self.add_bos_token = None
        self.custom_prefix_token_id = None
        self.think_end_token = None
        self.batch_sizes = {}
        if not str(batch_size).startswith("auto"):
            self.batch_schedule = 1
        self._last_padding_side = self.tokenizer.padding_side

        # Additional HFLM attributes that might be needed
        self.softmax_dtype = None  # Use model's dtype
        self.dtype = None  # Will be inferred from model
        self.accelerator = None  # No distributed setup
        self._rank = 0
        self._world_size = 1
        self.revision = None  # No git revision for custom model
        # tokenizer_name is a property, will be handled by HFLM if needed

        forbidden_token_ids = {
            int(token_id)
            for token_id in (invalid_token_ids or [])
            if 0 <= int(token_id) < self._vocab_size
        }
        forbidden_token_ids.add(int(config.pad_token_id))
        predict_token_id = getattr(config, "predict_token_id", None)
        if predict_token_id is not None:
            forbidden_token_ids.add(int(predict_token_id))
        forbidden_token_ids.discard(int(config.eos_token_id))
        self._forbidden_token_ids = torch.tensor(
            sorted(forbidden_token_ids),
            device=self._device,
            dtype=torch.long,
        )
        self._max_gen_toks = min(256, max(1, int(config.block_size) - 1))

    # Reuse HFLM's batching but make sure we encode/decode via our tokenizer
    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.encode(string, add_special_tokens=False, truncation=self.truncation, max_length=self._max_length)

    def tok_decode(self, tokens, skip_special_tokens: bool = True, **kwargs):
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _pad_and_concat_with_pad_token(
        self,
        max_length: int,
        tensors: list[torch.Tensor],
        padding_side: str = "left",
    ) -> torch.Tensor:
        padding_side = "left"
        target_length = self._effective_eval_seq_len(max_length)

        padded = []
        for tensor in tensors:
            if len(tensor.shape) == 2:
                tensor = tensor.squeeze(0)

            tensor_len = tensor.shape[0]
            if tensor_len < target_length:
                pad = torch.full(
                    (target_length - tensor_len,),
                    fill_value=self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=tensor.device,
                )
                tensor = (
                    torch.cat([tensor, pad], dim=0)
                    if padding_side == "right"
                    else torch.cat([pad, tensor], dim=0)
                )
            padded.append(tensor.unsqueeze(0))

        result = torch.cat(padded, dim=0)
        self._last_padding_side = padding_side
        return result

    def _effective_eval_seq_len(self, seq_len: int) -> int:
        if self._min_eval_seq_len is None:
            return seq_len
        return max(seq_len, self._min_eval_seq_len)

    @contextmanager
    def _use_pad_aware_batching(self):
        original_pad_and_concat = lm_eval_huggingface.pad_and_concat
        lm_eval_huggingface.pad_and_concat = self._pad_and_concat_with_pad_token
        try:
            yield
        finally:
            lm_eval_huggingface.pad_and_concat = original_pad_and_concat

    def _model_generate(self, context, max_length: int, stop: list[str], **generation_kwargs):
        generation_kwargs.pop("attention_mask", None)
        do_sample = bool(generation_kwargs.pop("do_sample", False))
        temperature = float(generation_kwargs.pop("temperature", 0.0 if not do_sample else 1.0))
        top_k = generation_kwargs.pop("top_k", None)
        top_p = generation_kwargs.pop("top_p", None)
        if generation_kwargs:
            unsupported = ", ".join(sorted(generation_kwargs.keys()))
            raise NotImplementedError(
                "This lm-eval adapter currently supports only `do_sample`, `temperature`, "
                f"`top_k`, and `top_p` generation kwargs; got unsupported kwargs: {unsupported}"
            )

        max_new_tokens = max(0, int(max_length) - int(context.shape[1]))
        if max_new_tokens == 0:
            return context
        if not hasattr(self._raw_model, "generate"):
            raise AttributeError(
                f"{type(self._raw_model).__name__} must implement generate() for lm-eval generation benchmarks."
            )

        with torch.no_grad():
            generated = self._raw_model.generate(
                context,
                max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_on_eos=True,
                forbidden_token_ids=self._forbidden_token_ids,
            )
        return generated

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, override_bs: int | None = None):
        with self._use_pad_aware_batching():
            return super()._loglikelihood_tokens(
                requests,
                disable_tqdm=disable_tqdm,
                override_bs=override_bs,
            )

    def _detect_batch_size(self, requests=None, pos: int = 0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            )
            max_context_enc = len(context_enc[-(self.max_length + 1) :])
            max_cont_enc = len(continuation_enc[-(self.max_length + 1) :])
        else:
            max_length = self.max_length
            max_context_enc = max_length
            max_cont_enc = max_length

        max_length = self._effective_eval_seq_len(max_length)
        max_context_enc = self._effective_eval_seq_len(max_context_enc)
        max_cont_enc = self._effective_eval_seq_len(max_cont_enc)

        @lm_eval_huggingface.find_executable_batch_size(
            starting_batch_size=self.max_batch_size
        )
        def forward_batch(batch_size: int):
            if self.backend == "seq2seq":
                length = max(max_context_enc, max_cont_enc)
                batched_conts = torch.ones(
                    (batch_size, length), device=self.device
                ).long()
                test_batch = torch.ones((batch_size, length), device=self.device).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones(
                    (batch_size, max_length), device=self.device
                ).long()

            for _ in range(5):
                torch.nn.functional.log_softmax(
                    self._model_call(test_batch, **call_kwargs),
                    dim=-1,
                    dtype=self.softmax_dtype,
                )

            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        if self.world_size > 1:
            max_rnk_bs = torch.tensor([batch_size], device=self.device)
            gathered = (
                self.accelerator.gather(max_rnk_bs).cpu().detach().numpy().tolist()
            )
            batch_size = min(gathered)
            lm_eval_huggingface.clear_torch_cache()
            return batch_size

        lm_eval_huggingface.clear_torch_cache()
        return batch_size

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Use the training forward for accurate log-likelihoods (full-sequence logits).
        """
        if attn_mask is not None or labels is not None:
            raise NotImplementedError(
                "This lm-eval adapter only supports decoder-only loglikelihood evaluation."
            )

        with torch.no_grad():
            targets = inps.clone()
            targets[:, :-1] = inps[:, 1:]
            used_forward_efficient = self._use_forward_efficient and hasattr(
                self._raw_model, "forward_efficient"
            )
            if used_forward_efficient:
                outputs = self._raw_model.forward_efficient(
                    inps,
                    targets,
                )
            else:
                outputs = self._raw_model.forward(inps, targets)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            # Slice logits to actual vocab_size to exclude any extra entries (e.g., predict tokens)
            # that might inflate perplexity calculations
            logits = logits[:, :, :self._vocab_size]

        return logits

    def _select_cont_toks(
        self,
        logits: torch.Tensor,
        contlen: int | None = None,
        inplen: int | None = None,
    ) -> torch.Tensor:
        if self.backend == "causal" and self._last_padding_side == "left":
            assert contlen is not None, "Must pass continuation length for causal LM"
            # Under left padding, valid continuation logits are always the trailing tokens.
            return logits[-contlen:]
        return super()._select_cont_toks(logits, contlen=contlen, inplen=inplen)

    def reset_eval_run_state(self) -> None:
        self.batch_sizes = {}

    def get_model_info(self):
        """Override to avoid accessing properties that don't exist."""
        return {
            "model_type": type(self._raw_model).__name__,
            "model_config": str(self._config),
        }


def create_hflm_eval_model(
    model,
    config,
    tokenizer_encode,
    tokenizer_decode,
    invalid_token_ids=None,
    device: str = "cuda",
    batch_size: int | str = 1,
    max_batch_size: int | None = None,
    min_eval_seq_len: int | None = None,
    use_forward_efficient: bool = False,
) -> CustomHFLMAdapter:
    """
    Factory for the HFLM-based adapter.

    `use_forward_efficient` defaults to False: log-likelihood evaluation runs the dense
    training forward (the Triton path), which is what the paper evaluations used. The
    streaming `forward_efficient` path is numerically identical but far slower (an O(T)
    per-token loop); pass True only to exercise it explicitly.
    """
    return CustomHFLMAdapter(
        model=model,
        tokenizer_encode=tokenizer_encode,
        tokenizer_decode=tokenizer_decode,
        invalid_token_ids=invalid_token_ids,
        config=config,
        device=device,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        min_eval_seq_len=min_eval_seq_len,
        use_forward_efficient=use_forward_efficient,
    )
