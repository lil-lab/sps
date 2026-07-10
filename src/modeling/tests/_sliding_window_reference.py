from __future__ import annotations

"""Test-only reference for the deterministic sliding-window visibility that the
Triton sliding-attention kernels (used by the SPS / Reverse-SPS / Delayed-State
models) implement over the doubled input/predict (2T) sequence.

Exists purely to validate the model window semantics in tests; not imported by
any runtime model code.
"""

import torch
from torch import Tensor


def window_score_mod_factory(window_size: int) -> callable:
    """FlexAttention score modifier encoding the deterministic window semantics.

    Predict (odd-slot) keys are always visible; normal (input) keys are visible
    only within ``window_size`` of the query in token space.
    """

    def window_score_mod(
        score: Tensor,
        batch: Tensor,
        head: Tensor,
        q_idx: Tensor,
        k_idx: Tensor,
    ) -> Tensor:
        is_predict = (k_idx % 2 == 1)
        k_out_of_sliding_window = (q_idx - k_idx) // 2 > window_size
        normal_score = torch.where(k_out_of_sliding_window, float("-inf"), score)
        return torch.where(is_predict, score, normal_score)

    return window_score_mod
