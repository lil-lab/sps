"""
Full attention model with standard causal attention.
"""

import math
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks
from modeling.masking import (
    causal_mask,
    document_mask_factory_method,
    left_padding_mask_factory_method,
    get_mask_mod_w_offset,
    cached_tokens_padding_mask_factory_method,
    sliding_window_mask_factory_method,
)

# Import shared components from model.py
from modeling.models.model import (
    RMSNorm,
    Block,
    CausalSelfAttention,
    MLP,
    precompute_freqs_cis,
    apply_rotary_emb,
    compute_left_padded_position_ids,
    generate_left_padded_document_idx,
    infer_is_real_tokens,
    ModelConfig as BaseModelConfig,
    IGNORE_INDEX,
    validate_left_padded_tokens,
)
from modeling.models.utils.generation import generate_with_batched_prefill
from modeling.models.utils.decode_attention import KVSegment, masked_kv_attention
from modeling.models.utils.sampling import sample_next_token

try:
    from modeling.models.attention.triton_full_flash_attention import full_attention as triton_full_attention
except Exception:
    triton_full_attention = None



@dataclass
class ModelConfig(BaseModelConfig):
    """Configuration for full attention models."""
    use_triton_full_attention: bool = False
    warp_specialize: bool = False


@dataclass
class _FullAttentionDecodeState:
    k: list[Tensor]
    v: list[Tensor]
    documents_idx_BxK: Tensor
    current_documents_idx_B: Tensor
    current_doc_tokens_B: Tensor
    processed_tokens_B: Tensor
    attn_dtype: torch.dtype
    head_dim: int


class TritonFullAttention(CausalSelfAttention):
    """Standard causal attention backed by the Triton document-masked causal kernel."""

    def __init__(self, config):
        super().__init__(config)
        self.warp_specialize = getattr(config, "warp_specialize", False)

    def forward(
        self,
        x,
        freqs_cis: torch.Tensor,
        attn_block_mask: Optional[torch.Tensor] = None,
        past_key_values: Tuple[torch.Tensor, torch.Tensor] = None,
        documents_idx_BxT: Optional[torch.Tensor] = None,
    ):
        if (
            triton_full_attention is None
            or not x.is_cuda
            or attn_block_mask is not None
            or past_key_values is not None
            or documents_idx_BxT is None
        ):
            return super().forward(
                x,
                freqs_cis,
                attn_block_mask=attn_block_mask,
                past_key_values=past_key_values,
            )

        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2).to(torch.bfloat16)
        k = k.transpose(1, 2).to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        y = triton_full_attention(
            q,
            k,
            v,
            1.0 / math.sqrt(q.shape[-1]),
            warp_specialize=self.warp_specialize,
            documents_idx_BxT=documents_idx_BxT.contiguous(),
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = y.to(self.c_proj.weight.dtype)
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v)


class TritonFullAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attn = TritonFullAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x,
        freqs_cis: torch.Tensor,
        attn_block_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        documents_idx_BxT: Optional[torch.Tensor] = None,
    ):
        attn_output, past_key_values = self.attn(
            self.attention_norm(x),
            freqs_cis,
            attn_block_mask=attn_block_mask,
            past_key_values=past_key_values,
            documents_idx_BxT=documents_idx_BxT,
        )
        x = x + attn_output
        x = x + self.mlp(self.mlp_norm(x))
        return x, past_key_values

class Model(nn.Module):
    """
    Full attention model with standard causal attention.
    This is the concrete implementation of the transformer model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pad_token_id is not None, "pad_token_id must be provided in config"
        self.config = config
        self.use_triton_full_attention = bool(
            getattr(config, "use_triton_full_attention", False)
        )
        block_cls = self._get_block_cls()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.hidden_size),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([block_cls(config) for _ in range(config.n_layer)]),
            output_norm = RMSNorm(config)
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Standard input/output weight tying: the token-embedding table (wte) shares its
        # weight matrix with the lm_head projection. Under torch.compile this surfaces a
        # benign "functional_call was passed multiple values for tied weights" warning
        # that does not affect correctness.
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.config.hidden_size // self.config.n_head, self.config.block_size 
            ),
            persistent=False,
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

        self.last_predict_idx = None

    def _get_block_cls(self):
        return TritonFullAttentionBlock if self.use_triton_full_attention else Block

    def generate_document_idx(self, idx_BxT: Tensor) -> Tensor:
        """
        Generate document indices for each token based on EOS token positions.
        """
        return generate_left_padded_document_idx(
            idx_BxT,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
        )

    def get_prefilling_mask_function(
        self,
        idx_BxT: Tensor,
        documents_idx_BxT: Optional[Tensor] = None,
    ) -> Callable:
        """
        Get the attention mask function for full-sequence training/evaluation without padding.
        Padding is applied by create_attention_mask.
        Can be overridden by subclasses to add model-specific masks (e.g., sliding window).
        """
        mask_fn = causal_mask

        if documents_idx_BxT is None:
            documents_idx_BxT = self.generate_document_idx(idx_BxT)
        mask_fn = and_masks(mask_fn, document_mask_factory_method(documents_idx_BxT))
        persistent_key_window = getattr(self, "forced_persistent_key_window", None)
        if persistent_key_window is not None:
            mask_fn = and_masks(
                mask_fn,
                sliding_window_mask_factory_method(int(persistent_key_window)),
            )

        return mask_fn

    def apply_padding_masks(
        self,
        mask_fn: Callable,
        idx_BxT: Tensor,
        is_real_BxT: Tensor,
        cache_lengths: Optional[Tensor] = None,
    ) -> Callable:
        """
        Apply padding masks to a base mask function.
        Used during padded training/evaluation to ensure consistent padding handling.
        """
        device = idx_BxT.device

        padding_offsets = is_real_BxT.long().argmax(dim=1)
        mask_fn = and_masks(mask_fn, left_padding_mask_factory_method(padding_offsets))

        if cache_lengths is not None and (cache_lengths != 0).any():
            batched_kv_len = cache_lengths.max()
            lengths_tensor = cache_lengths.to(device)
            cached_pad_mask_fn = cached_tokens_padding_mask_factory_method(lengths_tensor, past_max_len=batched_kv_len)
            mask_fn = and_masks(mask_fn, cached_pad_mask_fn)

            if batched_kv_len == 0:
                seq_lengths = is_real_BxT.sum(dim=1)
                max_seq_len = seq_lengths.max()
                if max_seq_len < idx_BxT.shape[1]:
                    right_pad_mask_fn = cached_tokens_padding_mask_factory_method(
                        seq_lengths.to(device), past_max_len=idx_BxT.shape[1]
                    )
                    mask_fn = and_masks(mask_fn, right_pad_mask_fn)

            mask_fn = get_mask_mod_w_offset(mask_fn, batched_kv_len)

        return mask_fn

    def create_attention_mask(
        self,
        idx_BxT: Tensor,
        cache_lengths: Tensor,
        is_real_BxT: Optional[Tensor] = None,
        documents_idx_BxT: Optional[Tensor] = None,
        combined_documents_idx_BxKV: Optional[Tensor] = None,
    ) -> Optional[Callable]:
        """
        Model-level hook for building attention masks.
        """
        if is_real_BxT is None:
            is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        batched_kv_len = cache_lengths.max()
        prefilling_mode = batched_kv_len == 0

        if prefilling_mode:
            mask_fn = self.get_prefilling_mask_function(idx_BxT, documents_idx_BxT=documents_idx_BxT)
            if not bool(is_real_BxT.all()):
                padding_offsets = is_real_BxT.long().argmax(dim=1)
                mask_fn = and_masks(mask_fn, left_padding_mask_factory_method(padding_offsets))
        else:
            mask_fn = causal_mask
            if combined_documents_idx_BxKV is not None:
                mask_fn = and_masks(
                    mask_fn,
                    document_mask_factory_method(combined_documents_idx_BxKV),
                )
            mask_fn = self.apply_padding_masks(mask_fn, idx_BxT, is_real_BxT, cache_lengths)

        return mask_fn

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _can_use_triton_prefill(
        self,
        idx_BxT: Tensor,
        is_real_BxT: Tensor,
    ) -> bool:
        # Note: left-padding is handled by the Triton kernel's document masking.
        # generate_left_padded_document_idx puts pad tokens in isolated "fake"
        # documents disjoint from the real suffix, so real queries never attend
        # pad keys -- no is_real_BxT.all() gate needed.
        return (
            self.use_triton_full_attention
            and triton_full_attention is not None
            and idx_BxT.is_cuda
            and getattr(self, "forced_persistent_key_window", None) is None
        )

    def _forward_block(
        self,
        block: nn.Module,
        x: Tensor,
        freqs_cis: Tensor,
        *,
        attn_block_mask: Optional[torch.Tensor],
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]],
        documents_idx_BxT: Optional[Tensor] = None,
    ) -> tuple[Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self.use_triton_full_attention:
            return block(
                x,
                freqs_cis,
                attn_block_mask=attn_block_mask,
                past_key_values=past_key_values,
                documents_idx_BxT=documents_idx_BxT,
            )
        return block(
            x,
            freqs_cis,
            attn_block_mask=attn_block_mask,
            past_key_values=past_key_values,
        )


    def forward_hidden_states(
        self, idx_BxT: Tensor, targets_BxT: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for training when inputs already include predicts/padding.
        """
        device = idx_BxT.device
        b, t = idx_BxT.size()

        is_not_pad_mask = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_not_pad_mask,
            allow_all_pad=True,
            context="forward inputs",
        )
        use_triton_prefill = self._can_use_triton_prefill(idx_BxT, is_not_pad_mask)

        # RoPE is relative, so left-padded position ids agree with arange on the
        # real tokens when there is no padding; using them unconditionally keeps
        # the Triton prefill path correct for padded batches too.
        position_ids = compute_left_padded_position_ids(is_not_pad_mask)

        assert t <= self.freqs_cis.shape[0], f"Cannot forward sequence of length {t}, block size is only {self.freqs_cis.shape[0]}"

        all_freqs_cis = self.freqs_cis.to(device)
        freqs_cis = all_freqs_cis[position_ids]

        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        if use_triton_prefill:
            attn_block_mask = None
        else:
            cache_lengths = torch.zeros(b, dtype=torch.long, device=device)
            attn_mask_function = self.create_attention_mask(
                idx_BxT=idx_BxT,
                cache_lengths=cache_lengths,
                is_real_BxT=is_not_pad_mask,
                documents_idx_BxT=documents_idx_BxT,
            )
            attn_block_mask = create_block_mask(
                attn_mask_function,
                B=b,
                H=None,
                Q_LEN=t,
                KV_LEN=t,
                device=device
            )

        tok_emb = self.transformer.wte(idx_BxT)
        x = self.transformer.drop(tok_emb)

        past_key_values_Lx2 = [None for _ in range(self.config.n_layer)]
        for layer_idx, block in enumerate(self.transformer.h):
            x, _ = self._forward_block(
                block,
                x,
                freqs_cis,
                attn_block_mask=attn_block_mask,
                past_key_values=past_key_values_Lx2[layer_idx],
                documents_idx_BxT=documents_idx_BxT if use_triton_prefill else None,
            )

        x = self.transformer.output_norm(x)

        return x, targets_BxT, is_not_pad_mask

    def forward(self, idx_BxT: Tensor, targets_BxT: Tensor):
        """
        Forward pass for training. Computes logits and loss.
        """
        device = idx_BxT.device

        x, targets_BxT, is_not_pad_mask = self.forward_hidden_states(idx_BxT, targets_BxT)
        original_t = idx_BxT.size(1)

        b, t = x.size(0), x.size(1)

        logits = self.lm_head(x)

        # Return logits without predicts (if any)
        original_logits = logits[targets_BxT != IGNORE_INDEX].view(b, original_t, logits.size(-1))

        # Mask out padded positions and EOS positions in targets
        targets_BxT[~is_not_pad_mask] = IGNORE_INDEX
        targets_BxT[idx_BxT == self.config.eos_token_id] = IGNORE_INDEX

        # Compute loss, ignoring padded positions
        token_count = (targets_BxT != IGNORE_INDEX).sum()
        token_nll_sum = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets_BxT.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        loss = token_nll_sum / token_count.clamp(min=1)

        stats = {
            "token_nll_sum": token_nll_sum.detach(),
            "token_nll_count": token_count.detach(),
            "token_count": token_count.detach(),
        }

        return original_logits, loss, stats

    @contextmanager
    def _temporary_disable_generation_triton(self):
        previous_use_triton = self.use_triton_full_attention
        previous_config_flag = getattr(self.config, "use_triton_full_attention", previous_use_triton)
        self.use_triton_full_attention = False
        if hasattr(self.config, "use_triton_full_attention"):
            self.config.use_triton_full_attention = False
        try:
            yield
        finally:
            self.use_triton_full_attention = previous_use_triton
            if hasattr(self.config, "use_triton_full_attention"):
                self.config.use_triton_full_attention = previous_config_flag

    def _sample_next_token(
        self,
        logits_BxV: Tensor,
        active_mask_B: Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        forbidden_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        return sample_next_token(
            logits_BxV,
            active_mask_B,
            pad_token_id=self.config.pad_token_id,
            suppressed_token_ids=(self.config.pad_token_id,),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            forbidden_token_ids=forbidden_token_ids,
        )

    def _build_decode_state(
        self,
        *,
        batch_size: int,
        max_tokens: int,
        device: torch.device,
    ) -> _FullAttentionDecodeState:
        n_layer = int(self.config.n_layer)
        n_head = int(self.config.n_head)
        head_dim = int(self.config.hidden_size // self.config.n_head)
        # Match the SPS-family models: store KV cache in bf16 on CUDA so memory
        # comparisons across methods are fair. Model weights stay in their
        # original dtype.
        attn_dtype = torch.bfloat16 if device.type == "cuda" else self.transformer.wte.weight.dtype

        return _FullAttentionDecodeState(
            k=[
                torch.zeros((batch_size, n_head, max_tokens, head_dim), device=device, dtype=attn_dtype)
                for _ in range(n_layer)
            ],
            v=[
                torch.zeros((batch_size, n_head, max_tokens, head_dim), device=device, dtype=attn_dtype)
                for _ in range(n_layer)
            ],
            documents_idx_BxK=torch.zeros((batch_size, max_tokens), device=device, dtype=torch.long),
            current_documents_idx_B=torch.zeros((batch_size,), device=device, dtype=torch.long),
            current_doc_tokens_B=torch.zeros((batch_size,), device=device, dtype=torch.long),
            processed_tokens_B=torch.zeros((batch_size,), device=device, dtype=torch.long),
            attn_dtype=attn_dtype,
            head_dim=head_dim,
        )

    def _attend_one_token_from_decode_state(
        self,
        state: _FullAttentionDecodeState,
        layer_idx: int,
        q_BxHxD: Tensor,
        k_BxHxD: Tensor,
        v_BxHxD: Tensor,
        active_mask_B: Tensor,
    ) -> Tensor:
        b, n_head, _ = q_BxHxD.shape
        active_mask_B = active_mask_B.to(dtype=torch.bool, device=q_BxHxD.device)
        previous_doc_len_B = state.current_doc_tokens_B
        attn_out_BxHxD = masked_kv_attention(
            q_BxHxD,
            [
                KVSegment(
                    state.k[layer_idx],
                    state.v[layer_idx],
                    previous_doc_len_B,
                )
            ],
            extra_kv=[(k_BxHxD, v_BxHxD)],
            attn_dtype=state.attn_dtype,
            prefer_triton=True,
            require_triton=q_BxHxD.is_cuda,
            error_context=f" in full attention decode state layer={layer_idx}",
        )

        batch_idx_B = torch.arange(b, device=q_BxHxD.device)
        old_k_BxHxD = state.k[layer_idx][batch_idx_B, :, previous_doc_len_B, :]
        old_v_BxHxD = state.v[layer_idx][batch_idx_B, :, previous_doc_len_B, :]
        write_mask_Bx1x1 = active_mask_B.view(b, 1, 1)
        state.k[layer_idx][batch_idx_B, :, previous_doc_len_B, :] = torch.where(
            write_mask_Bx1x1,
            k_BxHxD,
            old_k_BxHxD,
        )
        state.v[layer_idx][batch_idx_B, :, previous_doc_len_B, :] = torch.where(
            write_mask_Bx1x1,
            v_BxHxD,
            old_v_BxHxD,
        )
        state.documents_idx_BxK[batch_idx_B, previous_doc_len_B] = torch.where(
            active_mask_B,
            state.current_documents_idx_B,
            state.documents_idx_BxK[batch_idx_B, previous_doc_len_B],
        )
        return attn_out_BxHxD

    def _decode_one_token_step(
        self,
        state: _FullAttentionDecodeState,
        token_B: Tensor,
        active_mask_B: Tensor,
    ) -> Tensor:
        b = token_B.size(0)
        device = token_B.device
        active_mask_B = active_mask_B.to(dtype=torch.bool, device=device)
        zero_logits = self.lm_head.weight.new_zeros((b, self.config.vocab_size))
        if not bool(active_mask_B.any()):
            return zero_logits

        if bool((state.processed_tokens_B[active_mask_B] >= self.freqs_cis.shape[0]).any()):
            raise ValueError(
                f"Cannot decode beyond block size {self.freqs_cis.shape[0]} in full attention generation"
            )

        old_lengths_B = state.processed_tokens_B.clone()
        safe_token_B = torch.where(active_mask_B, token_B, torch.zeros_like(token_B))
        position_ids_B = torch.where(
            active_mask_B,
            old_lengths_B,
            torch.zeros_like(old_lengths_B),
        )
        freqs_cis_Bx1xD = self.freqs_cis.to(device)[position_ids_B].unsqueeze(1)

        x = self.transformer.drop(self.transformer.wte(safe_token_B.view(b, 1)))
        for layer_idx, block in enumerate(self.transformer.h):
            x_norm = block.attention_norm(x)
            q, k, v = block.attn.c_attn(x_norm).split(block.attn.hidden_size, dim=2)
            q = q.view(b, 1, block.attn.n_head, state.head_dim)
            k = k.view(b, 1, block.attn.n_head, state.head_dim)
            v = v.view(b, 1, block.attn.n_head, state.head_dim)
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis_Bx1xD)

            q_BxHxD = q[:, 0].to(state.attn_dtype)
            k_BxHxD = k[:, 0].to(state.attn_dtype)
            v_BxHxD = v[:, 0].to(state.attn_dtype)
            attn_out_BxHxD = self._attend_one_token_from_decode_state(
                state,
                layer_idx,
                q_BxHxD,
                k_BxHxD,
                v_BxHxD,
                active_mask_B,
            )
            attn_out_Bx1xC = attn_out_BxHxD.reshape(b, 1, self.config.hidden_size).to(x.dtype)
            attn_out_Bx1xC = block.attn.resid_dropout(block.attn.c_proj(attn_out_Bx1xC))
            x = x + attn_out_Bx1xC
            x = x + block.mlp(block.mlp_norm(x))

        hidden_states_Bx1xC = self.transformer.output_norm(x)
        logits_BxV = self.lm_head(hidden_states_Bx1xC)[:, -1, : self.config.vocab_size]
        return torch.where(active_mask_B.unsqueeze(1), logits_BxV, zero_logits)

    def _advance_decode_state(
        self,
        state: _FullAttentionDecodeState,
        token_B: Tensor,
        active_mask_B: Tensor,
    ) -> None:
        active_mask_B = active_mask_B.to(dtype=torch.bool, device=token_B.device)
        state.processed_tokens_B[active_mask_B] = state.processed_tokens_B[active_mask_B] + 1
        generated_eos_B = active_mask_B & (token_B == self.config.eos_token_id)
        state.current_doc_tokens_B = torch.where(
            generated_eos_B,
            torch.zeros_like(state.current_doc_tokens_B),
            torch.where(
                active_mask_B,
                state.current_doc_tokens_B + 1,
                state.current_doc_tokens_B,
            ),
        )
        state.current_documents_idx_B = torch.where(
            generated_eos_B,
            state.current_documents_idx_B + 1,
            state.current_documents_idx_B,
        )

    def _prefill_generation_state(
        self,
        idx_BxT: Tensor,
        max_new_tokens: int,
        *,
        require_batched: bool = False,
    ) -> tuple[_FullAttentionDecodeState, Tensor, str]:
        del require_batched
        device = idx_BxT.device
        b, _ = idx_BxT.size()
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=False,
            context="full attention generation prefill prompts",
        )
        real_lengths_B = is_real_BxT.sum(dim=1).to(dtype=torch.long)
        max_real_prompt_tokens = int(real_lengths_B.max().item())
        total_real_tokens = max_real_prompt_tokens + int(max_new_tokens)
        if total_real_tokens > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Cannot generate {max_new_tokens} new tokens from a prompt with "
                f"{max_real_prompt_tokens} real tokens when block size is {self.freqs_cis.shape[0]}"
            )

        hidden_states_BxTxC, past_key_values_Lx2 = self._forward_generation_hidden_states(idx_BxT)
        state = self._build_decode_state(
            batch_size=b,
            max_tokens=total_real_tokens,
            device=device,
        )
        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        last_tokens_B = idx_BxT[:, -1]
        last_doc_idx_B = documents_idx_BxT[:, -1]
        current_doc_mask_BxT = is_real_BxT & (documents_idx_BxT == last_doc_idx_B.unsqueeze(1))
        current_doc_mask_BxT = current_doc_mask_BxT & (last_tokens_B != self.config.eos_token_id).unsqueeze(1)

        for batch_idx in range(b):
            doc_positions = current_doc_mask_BxT[batch_idx].nonzero(as_tuple=False).flatten()
            doc_len = int(doc_positions.numel())
            if doc_len == 0:
                continue
            state.documents_idx_BxK[batch_idx, :doc_len] = documents_idx_BxT[
                batch_idx,
                doc_positions,
            ]
            for layer_idx, (layer_k_BxHxTxD, layer_v_BxHxTxD) in enumerate(past_key_values_Lx2):
                state.k[layer_idx][batch_idx, :, :doc_len, :] = layer_k_BxHxTxD[
                    batch_idx,
                    :,
                    doc_positions,
                    :,
                ].to(dtype=state.k[layer_idx].dtype)
                state.v[layer_idx][batch_idx, :, :doc_len, :] = layer_v_BxHxTxD[
                    batch_idx,
                    :,
                    doc_positions,
                    :,
                ].to(dtype=state.v[layer_idx].dtype)

        state.processed_tokens_B.copy_(real_lengths_B)
        state.current_doc_tokens_B.copy_(current_doc_mask_BxT.sum(dim=1).to(dtype=torch.long))
        state.current_documents_idx_B.copy_(
            documents_idx_BxT[:, -1]
            + (last_tokens_B == self.config.eos_token_id).to(documents_idx_BxT.dtype)
        )
        next_logits_BxV = self.lm_head(hidden_states_BxTxC[:, -1, :])[:, : self.config.vocab_size]
        return state, next_logits_BxV, "batched_prefill"

    def _decode_generation_state(
        self,
        state: _FullAttentionDecodeState,
        token_B: Tensor,
        active_mask_B: Tensor,
    ) -> Tensor:
        return self._decode_one_token_step(state, token_B, active_mask_B)

    def _forward_generation_hidden_states(
        self,
        idx_BxT: Tensor,
        *,
        past_key_values_Lx2: Optional[list[Optional[Tuple[Tensor, Tensor]]]] = None,
        cache_lengths_B: Optional[Tensor] = None,
        cached_documents_idx_BxK: Optional[Tensor] = None,
        current_documents_idx_BxT: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tuple[Tensor, Tensor]]]:
        device = idx_BxT.device
        b, t = idx_BxT.size()
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=False,
            context="generation inputs",
        )

        if cache_lengths_B is None:
            cache_lengths_B = torch.zeros(b, dtype=torch.long, device=device)
        batched_kv_len = int(cache_lengths_B.max().item())

        position_ids_BxT = compute_left_padded_position_ids(is_real_BxT) + cache_lengths_B.unsqueeze(1)
        max_position = int(position_ids_BxT[is_real_BxT].max().item())
        if max_position >= self.freqs_cis.shape[0]:
            raise ValueError(
                f"Cannot generate position {max_position} when block size is {self.freqs_cis.shape[0]}"
            )

        all_freqs_cis = self.freqs_cis.to(device)
        freqs_cis = all_freqs_cis[position_ids_BxT]

        documents_idx_BxT = None
        combined_documents_idx_BxKV = None
        if batched_kv_len == 0:
            documents_idx_BxT = self.generate_document_idx(idx_BxT)
        elif cached_documents_idx_BxK is not None or current_documents_idx_BxT is not None:
            if cached_documents_idx_BxK is None or current_documents_idx_BxT is None:
                raise ValueError(
                    "cached_documents_idx_BxK and current_documents_idx_BxT must be provided together"
                )
            if cached_documents_idx_BxK.shape != (b, batched_kv_len):
                raise ValueError(
                    "cached_documents_idx_BxK must have shape "
                    f"{(b, batched_kv_len)}, got {tuple(cached_documents_idx_BxK.shape)}"
                )
            if current_documents_idx_BxT.shape != idx_BxT.shape:
                raise ValueError(
                    "current_documents_idx_BxT must have shape "
                    f"{tuple(idx_BxT.shape)}, got {tuple(current_documents_idx_BxT.shape)}"
                )
            combined_documents_idx_BxKV = torch.cat(
                [
                    cached_documents_idx_BxK.to(device=device),
                    current_documents_idx_BxT.to(device=device),
                ],
                dim=1,
            )
        # The Triton flash kernel handles causal + document masking internally
        # (via documents_idx_BxT) and bails out of its fast path when given a
        # non-None attn_block_mask. Skip the (expensive, flex-only) block-mask
        # build when we know the Triton path will trigger: prefill (no cached
        # KV), all-real tokens (no padding), and the model is configured to use
        # the Triton kernel. This keeps the Triton fast-path conditions in
        # `TritonFullAttention.forward` happy without changing semantics.
        skip_block_mask = (
            self.use_triton_full_attention
            and triton_full_attention is not None
            and idx_BxT.is_cuda
            and batched_kv_len == 0
            and bool(is_real_BxT.all())
            and combined_documents_idx_BxKV is None
            and documents_idx_BxT is not None
        )
        if skip_block_mask:
            attn_block_mask = None
        else:
            attn_mask_function = self.create_attention_mask(
                idx_BxT=idx_BxT,
                cache_lengths=cache_lengths_B,
                is_real_BxT=is_real_BxT,
                documents_idx_BxT=documents_idx_BxT,
                combined_documents_idx_BxKV=combined_documents_idx_BxKV,
            )
            attn_block_mask = create_block_mask(
                attn_mask_function,
                B=b,
                H=None,
                Q_LEN=t,
                KV_LEN=t + batched_kv_len,
                device=device,
            )

        x = self.transformer.drop(self.transformer.wte(idx_BxT))
        if past_key_values_Lx2 is None:
            past_key_values_Lx2 = [None for _ in range(self.config.n_layer)]
        next_past_key_values_Lx2 = []
        for layer_idx, block in enumerate(self.transformer.h):
            x, layer_past_key_values = self._forward_block(
                block,
                x,
                freqs_cis,
                attn_block_mask=attn_block_mask,
                past_key_values=past_key_values_Lx2[layer_idx],
                documents_idx_BxT=documents_idx_BxT if self.use_triton_full_attention else None,
            )
            next_past_key_values_Lx2.append(layer_past_key_values)
        x = self.transformer.output_norm(x)
        return x, next_past_key_values_Lx2

    def _generate_single_unpadded(
        self,
        prompt_T: Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        stop_on_eos: bool,
        forbidden_token_ids: Optional[Tensor],
    ) -> Tensor:
        device = prompt_T.device
        prompt_1xT = prompt_T.unsqueeze(0)
        current_len = prompt_1xT.size(1)
        cached_documents_idx_BxK = self.generate_document_idx(prompt_1xT)
        next_document_idx_B = cached_documents_idx_BxK[:, -1] + (
            prompt_1xT[:, -1] == self.config.eos_token_id
        ).to(cached_documents_idx_BxK.dtype)
        hidden_states_BxTxC, past_key_values_Lx2 = self._forward_generation_hidden_states(prompt_1xT)
        next_logits_BxV = self.lm_head(hidden_states_BxTxC)[:, -1, : self.config.vocab_size]

        generated_T = torch.full(
            (max_new_tokens,),
            self.config.pad_token_id,
            device=device,
            dtype=prompt_T.dtype,
        )
        active_mask_B = torch.ones((1,), device=device, dtype=torch.bool)

        for step in range(max_new_tokens):
            next_token_B = self._sample_next_token(
                next_logits_BxV,
                active_mask_B,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                forbidden_token_ids=forbidden_token_ids,
            ).to(prompt_T.dtype)
            generated_T[step] = next_token_B[0]
            if stop_on_eos and int(next_token_B[0].item()) == self.config.eos_token_id:
                active_mask_B[0] = False
                break

            current_documents_idx_BxT = next_document_idx_B.view(1, 1)
            hidden_states_BxTxC, past_key_values_Lx2 = self._forward_generation_hidden_states(
                next_token_B.view(1, 1),
                past_key_values_Lx2=past_key_values_Lx2,
                cache_lengths_B=torch.tensor([current_len], device=device, dtype=torch.long),
                cached_documents_idx_BxK=cached_documents_idx_BxK,
                current_documents_idx_BxT=current_documents_idx_BxT,
            )
            cached_documents_idx_BxK = torch.cat(
                [cached_documents_idx_BxK, current_documents_idx_BxT],
                dim=1,
            )
            next_document_idx_B = next_document_idx_B + (
                next_token_B == self.config.eos_token_id
            ).to(next_document_idx_B.dtype)
            current_len += 1
            next_logits_BxV = self.lm_head(hidden_states_BxTxC)[:, -1, : self.config.vocab_size]

        return torch.cat([prompt_T, generated_T], dim=0)

    @torch.no_grad()
    def generate(
        self,
        idx_BxT: Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_on_eos: bool = True,
        forbidden_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        def prefill(prompt_BxT: Tensor, new_tokens: int):
            return self._prefill_generation_state(prompt_BxT, new_tokens)

        def sample(logits_BxV: Tensor, active_mask_B: Tensor) -> Tensor:
            return self._sample_next_token(
                logits_BxV,
                active_mask_B,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                forbidden_token_ids=forbidden_token_ids,
            )

        return generate_with_batched_prefill(
            self,
            idx_BxT,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stop_on_eos=stop_on_eos,
            forbidden_token_ids=forbidden_token_ids,
            prefill_prompt=prefill,
            decode_one_token=self._decode_generation_state,
            advance_state=self._advance_decode_state,
            sample_next_token=sample,
            context="full attention",
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
