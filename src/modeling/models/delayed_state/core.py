from __future__ import annotations

"""Core delayed-state model definition."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from modeling.models.model import infer_is_real_tokens, validate_left_padded_tokens
from modeling.models.reverse_sps.core import (
    ReverseSPSConfig,
    ReverseSPSModelBase,
    IGNORE_INDEX,
    ModelConfig,
    build_reverse_sps_loss_and_stats,
    triton_reverse_sps_sliding_attention,
)


@dataclass
class DelayedStateConfig(ReverseSPSConfig):
    """Configuration for delayed-state models."""


class DelayedStateModelBase(ReverseSPSModelBase):
    def _expand_real_and_document_idx(self, idx_BxT: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=True,
            context="delayed_state inputs",
        )
        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        documents_idx_Bx2T = documents_idx_BxT.repeat_interleave(2, dim=1)
        return is_real_BxT, documents_idx_BxT, documents_idx_Bx2T

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
