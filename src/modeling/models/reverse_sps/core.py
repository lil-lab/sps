from __future__ import annotations

"""Core deterministic sliding-predicts model definition."""

import inspect
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from modeling.models.full_attention_model import IGNORE_INDEX, ModelConfig
from modeling.models.model import (
    MLP,
    RMSNorm,
    apply_rotary_emb,
    generate_left_padded_document_idx,
    infer_is_real_tokens,
    precompute_freqs_cis,
    validate_left_padded_tokens,
)
from modeling.models.utils.masked_stats import (
    add_distribution_stats,
    add_empty_distribution_stats,
)
from modeling.models.utils.segmented_ops import (
    masked_per_document_count,
)

try:
    from modeling.models.attention.triton_reverse_sps_flash_attention import (
        reverse_sps_sliding_attention as triton_reverse_sps_sliding_attention,
    )
except Exception:
    triton_reverse_sps_sliding_attention = None


class ReverseSPSFlashAttention(nn.Module):
    def __init__(self, config: ReverseSPSConfig):
        super().__init__()
        self.enable_triton_attention = config.enable_triton_attention
        self.warp_specialize = config.warp_specialize
        assert config.hidden_size % config.n_head == 0
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        self.window_size = int(config.window_size)
        self.persistent_key_window: int | None = None

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        documents_idx_Bx2T: Optional[Tensor] = None,
    ) -> Tensor:
        b, two_t, c = x.size()
        t = two_t // 2

        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        q = q.view(b, two_t, self.n_head, c // self.n_head)
        k = k.view(b, two_t, self.n_head, c // self.n_head)
        v = v.view(b, two_t, self.n_head, c // self.n_head).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2).to(torch.bfloat16)
        k = k.transpose(1, 2).to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        use_triton = (
            self.enable_triton_attention
            and triton_reverse_sps_sliding_attention is not None
            and self.dropout == 0.0
            and q.shape[-1] in {16, 32, 64, 128, 256}
        )
        assert use_triton, (
            "Triton attention is required for training but not available. "
            "Check enable_triton_attention, dropout, and head_dim."
        )
        y = triton_reverse_sps_sliding_attention(
            q,
            k,
            v,
            1.0 / math.sqrt(q.shape[-1]),
            self.window_size,
            warp_specialize=self.warp_specialize,
            documents_idx_BxT=documents_idx_Bx2T,
            persistent_key_window=self.persistent_key_window,
        )

        y = y.transpose(1, 2).contiguous().view(b, two_t, c)
        y = y.to(self.c_proj.weight.dtype)
        y = self.resid_dropout(self.c_proj(y))
        return y


class ReverseSPSBlock(nn.Module):
    def __init__(self, config: ReverseSPSConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attn = ReverseSPSFlashAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        documents_idx_Bx2T: Optional[Tensor] = None,
    ) -> Tensor:
        attn_output = self.attn(
            self.attention_norm(x),
            freqs_cis,
            documents_idx_Bx2T=documents_idx_Bx2T,
        )
        x = x + attn_output
        x = x + self.mlp(self.mlp_norm(x))
        return x


@dataclass
class ReverseSPSConfig(ModelConfig):
    """Configuration for deterministic sliding-predicts models."""

    predict_token_id: int = 50257
    window_size: int = 64
    enable_triton_attention: bool = True
    warp_specialize: bool = False


def build_reverse_sps_loss_and_stats(
    *,
    token_nll_sum: Tensor,
    token_count: Tensor,
    documents_idx_BxT: Tensor,
    is_real_BxT: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    device = documents_idx_BxT.device
    stats_mask_BxT = is_real_BxT

    token_loss = token_nll_sum / token_count.clamp(min=1)
    stats = {
        "token_nll_sum": token_nll_sum.detach(),
        "token_nll_count": token_count.detach(),
        "token_count": token_count.detach(),
    }

    with torch.no_grad():
        doc_counts, doc_count_mask = masked_per_document_count(documents_idx_BxT, stats_mask_BxT)
        doc_lengths = doc_counts[doc_count_mask].float()
        if doc_lengths.numel() > 0:
            add_distribution_stats(stats, "document_length", doc_lengths)
        else:
            add_empty_distribution_stats(stats, "document_length", device=device)

    return token_loss, stats


class ReverseSPSModelBase(nn.Module):
    def __init__(self, config: ReverseSPSConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pad_token_id is not None, "pad_token_id must be provided in config"
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([ReverseSPSBlock(config) for _ in range(config.n_layer)]),
                output_norm=RMSNorm(config),
            )
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.config.hidden_size // self.config.n_head,
                self.config.block_size,
            ),
            persistent=False,
        )

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True):
        del non_embedding
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def generate_document_idx(self, idx_BxT: Tensor) -> Tensor:
        return generate_left_padded_document_idx(
            idx_BxT,
            eos_token_id=self.config.eos_token_id,
            pad_token_id=self.config.pad_token_id,
        )

    def _expand_real_and_document_idx(self, idx_BxT: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=True,
            context="reverse_sps inputs",
        )
        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        documents_idx_Bx2T = documents_idx_BxT.repeat_interleave(2, dim=1)
        return is_real_BxT, documents_idx_BxT, documents_idx_Bx2T

    def forward_hidden_states(
        self,
        idx_Bx2T: Tensor,
        *,
        documents_idx_Bx2T: Tensor,
    ) -> Tensor:
        device = idx_Bx2T.device
        b, two_t = idx_Bx2T.size()
        t = two_t // 2
        assert two_t % 2 == 0, "Doubled sequence length must be even"

        position_ids_Bx2T = torch.arange(t, device=device).repeat_interleave(2).unsqueeze(0).expand(b, -1)
        assert t <= self.freqs_cis.shape[0], (
            f"Cannot forward sequence of length {two_t}, block size is only {self.freqs_cis.shape[0]}"
        )
        freqs_cis = self.freqs_cis.to(device)[position_ids_Bx2T]

        x = self.transformer.drop(self.transformer.wte(idx_Bx2T))
        for block in self.transformer.h:
            x = block(
                x,
                freqs_cis,
                documents_idx_Bx2T=documents_idx_Bx2T,
            )

        x = self.transformer.output_norm(x)
        return x

    def add_predict_tokens(self, idx_BxT: Tensor) -> Tensor:
        idx_Bx2T = idx_BxT.repeat_interleave(2, dim=1)
        idx_Bx2T[:, 1::2] = self.config.predict_token_id
        return idx_Bx2T

    def forward(
        self,
        idx_BxT: Tensor,
        targets_BxT: Optional[Tensor] = None,
    ):
        is_real_BxT, documents_idx_BxT, documents_idx_Bx2T = self._expand_real_and_document_idx(idx_BxT)
        idx_Bx2T = self.add_predict_tokens(idx_BxT)
        x_Bx2T = self.forward_hidden_states(
            idx_Bx2T,
            documents_idx_Bx2T=documents_idx_Bx2T,
        )
        token_logits_BxTxV = self.lm_head(x_Bx2T[:, ::2])
        if targets_BxT is None:
            return token_logits_BxTxV

        masked_targets_BxT = torch.where(
            is_real_BxT & (idx_BxT != self.config.eos_token_id),
            targets_BxT,
            torch.full_like(targets_BxT, IGNORE_INDEX),
        )
        token_count = (masked_targets_BxT != IGNORE_INDEX).sum()
        token_nll_sum = F.cross_entropy(
            token_logits_BxTxV.view(-1, token_logits_BxTxV.size(-1)),
            masked_targets_BxT.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        ).float()
        loss, stats = build_reverse_sps_loss_and_stats(
            token_nll_sum=token_nll_sum,
            token_count=token_count,
            documents_idx_BxT=documents_idx_BxT,
            is_real_BxT=is_real_BxT,
        )
        return token_logits_BxTxV, loss, stats

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
