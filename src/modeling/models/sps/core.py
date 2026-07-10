from __future__ import annotations

"""Core ReverseSPS-P model definition."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from modeling.models.model import precompute_freqs_cis
from modeling.models.reverse_sps.core import (
    ReverseSPSBlock,
    ReverseSPSConfig,
    ReverseSPSFlashAttention,
    ReverseSPSModelBase,
    IGNORE_INDEX,
    ModelConfig,
    RMSNorm,
    MLP,
    build_reverse_sps_loss_and_stats,
)

try:
    from modeling.models.attention.triton_sps_flash_attention import (
        sps_sliding_attention as triton_sps_sliding_attention,
    )
except Exception:
    triton_sps_sliding_attention = None


class SPSFlashAttention(ReverseSPSFlashAttention):
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

        from modeling.models.model import apply_rotary_emb

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2).to(torch.bfloat16)
        k = k.transpose(1, 2).to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        use_triton = (
            self.enable_triton_attention
            and triton_sps_sliding_attention is not None
            and self.dropout == 0.0
            and q.shape[-1] in {16, 32, 64, 128, 256}
        )
        if not use_triton:
            raise RuntimeError(
                "Triton attention is required for ReverseSPS-P dense forward but not available. "
                "Check enable_triton_attention, dropout, and head_dim."
            )
        y = triton_sps_sliding_attention(
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


class SPSBlock(ReverseSPSBlock):
    def __init__(self, config: "SPSConfig"):
        nn.Module.__init__(self)
        self.attention_norm = RMSNorm(config)
        self.attn = SPSFlashAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)


@dataclass
class SPSConfig(ReverseSPSConfig):
    """Configuration for ReverseSPS-P models."""


class SPSModelBase(ReverseSPSModelBase):
    def __init__(self, config: SPSConfig):
        nn.Module.__init__(self)
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pad_token_id is not None, "pad_token_id must be provided in config"
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([SPSBlock(config) for _ in range(config.n_layer)]),
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
        token_logits_BxTxV = self.lm_head(x_Bx2T[:, 1::2])
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
