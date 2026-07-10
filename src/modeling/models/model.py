"""
Shared components for transformer models.

This module contains shared components used across all model implementations:
- Transformer components: RMSNorm, Block, CausalSelfAttention, MLP
- Base class: ModelConfig
- Helper functions: precompute_freqs_cis, apply_rotary_emb, reshape_for_broadcast, repeat_kv
- Constants: IGNORE_INDEX

The concrete standard-attention Model implementation is in full_attention_model.py;
the SPS-family models (sps, reverse_sps, delayed_state) reuse these shared components.

References:
1) the official Llama Torch implementation released by Meta:
https://github.com/meta-llama/llama/blob/main/llama/model.py
"""

import math
import inspect
from dataclasses import dataclass


import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from torch.nn.attention.flex_attention import flex_attention, create_block_mask, and_masks
from modeling.masking import causal_mask, document_mask_factory_method, get_mask_mod_w_offset, left_padding_mask_factory_method

from typing import Optional, Tuple, Callable

IGNORE_INDEX = -100


def infer_is_real_tokens(idx_BxT: Tensor, pad_token_id: int) -> Tensor:
    """Infer which token positions are real from a dedicated pad token id."""
    return idx_BxT != pad_token_id


def validate_left_padded_tokens(
    is_real_BxT: Tensor,
    *,
    allow_all_pad: bool,
    context: str,
) -> None:
    """Validate contiguous left padding and optionally reject all-pad rows."""
    is_real_BxT = is_real_BxT.to(dtype=torch.bool)
    ever_real = is_real_BxT.cumsum(dim=1) > 0
    has_pad_after_real = ((~is_real_BxT) & ever_real).any(dim=1)
    if bool(has_pad_after_real.any()):
        bad_rows = has_pad_after_real.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            f"{context} must be left padded only; found pad tokens after real tokens in rows {bad_rows}"
        )
    if not allow_all_pad:
        empty_rows = (is_real_BxT.sum(dim=1) == 0)
        if bool(empty_rows.any()):
            bad_rows = empty_rows.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"{context} must contain at least one real token per row; empty rows: {bad_rows}")


def compute_left_padded_position_ids(is_real_BxT: Tensor) -> Tensor:
    """Assign consecutive positions to real tokens while leaving pad positions at 0."""
    is_real_BxT = is_real_BxT.to(dtype=torch.bool)
    cumsum = is_real_BxT.long().cumsum(dim=1) - 1
    return torch.where(is_real_BxT, cumsum, torch.zeros_like(cumsum))


def generate_left_padded_document_idx(
    idx_BxT: Tensor,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> Tensor:
    """
    Generate document ids where EOS ends the current real document and left-pad
    tokens occupy isolated fake documents that are disjoint from the real suffix.
    """
    is_real_BxT = infer_is_real_tokens(idx_BxT, pad_token_id)
    validate_left_padded_tokens(
        is_real_BxT,
        allow_all_pad=True,
        context="document index inputs",
    )

    b, t = idx_BxT.shape
    is_real_long = is_real_BxT.long()
    pad_count_Bx1 = (~is_real_BxT).sum(dim=1, keepdim=True)

    is_real_eos_BxT = is_real_BxT & (idx_BxT == eos_token_id)
    document_idx = torch.zeros((b, t), dtype=idx_BxT.dtype, device=idx_BxT.device)
    if t > 1:
        document_idx[:, 1:] = torch.cumsum(is_real_eos_BxT[:, :-1], dim=1, dtype=idx_BxT.dtype)
    document_idx = document_idx + pad_count_Bx1.to(dtype=idx_BxT.dtype)

    pad_doc_idx = torch.cumsum((~is_real_BxT).long(), dim=1) - 1
    pad_doc_idx = torch.clamp(pad_doc_idx, min=0).to(dtype=idx_BxT.dtype)
    return torch.where(is_real_BxT, document_idx, pad_doc_idx)

class RMSNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.norm_eps

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
    
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Handles two cases:
    1. freqs_cis shape (T, head_dim) - same freqs for all sequences (old behavior)
    2. freqs_cis shape (B, T, head_dim) - per-sequence freqs (new behavior with padding)

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility (B, T, nh, hs).

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert ndim == 4, f"Expected x to have 4 dimensions (B, T, nh, hs), got {ndim}"

    # Check if freqs_cis is per-sequence (B, T, head_dim) or global (T, head_dim)
    if freqs_cis.ndim == 3:
        # Per-sequence freqs: (B, T, head_dim)
        # x shape: (B, T, nh, hs)
        assert freqs_cis.shape[0] == x.shape[0], f"Batch size mismatch: freqs_cis {freqs_cis.shape[0]} vs x {x.shape[0]}"
        assert freqs_cis.shape[1] == x.shape[1], f"Sequence length mismatch: freqs_cis {freqs_cis.shape[1]} vs x {x.shape[1]}"
        assert freqs_cis.shape[2] == x.shape[-1], f"Head dim mismatch: freqs_cis {freqs_cis.shape[2]} vs x {x.shape[-1]}"
        # Reshape to (B, T, 1, hs) for broadcasting with (B, T, nh, hs)
        return freqs_cis.unsqueeze(2)
    else:
        # Global freqs: (T, head_dim) - same for all sequences
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"Shape mismatch: freqs_cis {freqs_cis.shape} vs expected ({x.shape[1]}, {x.shape[-1]})"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.hidden_size = config.hidden_size
        self.dropout = config.dropout
        # Use non-compiled flex_attention (torch.compile causes issues on CPU)
        # Compilation works for some input shapes but fails during recompilation with different shapes
        # See: https://github.com/pytorch/pytorch/issues/139434
        self.flex_attention_fn = flex_attention

    def forward(
        self,
        x,
        freqs_cis: torch.Tensor,
        attn_block_mask: Optional[torch.Tensor] = None,
        past_key_values: Tuple[torch.Tensor, torch.Tensor] = None
        ):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (hidden_size)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.hidden_size, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.transpose(1, 2) # (B, nh, T, hs)
        k = k.transpose(1, 2) # (B, nh, T, hs)

        if past_key_values is not None:
            # Update past key and value tensors by concatenating with cached values
            k = torch.cat([past_key_values[0], k], dim=2)
            v = torch.cat([past_key_values[1], v], dim=2)

        past_key_values = (k, v)




        y = self.flex_attention_fn(q, k, v, block_mask=attn_block_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, past_key_values


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.gate_proj    = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.up_proj  = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down_proj  = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis: torch.Tensor, attn_block_mask: Optional[torch.Tensor] = None, past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None):
        attn_output, past_key_values = self.attn(self.attention_norm(x), freqs_cis, attn_block_mask=attn_block_mask, past_key_values=past_key_values)
        x = x + attn_output
        x = x + self.mlp(self.mlp_norm(x))
        return x, past_key_values


@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3 * hidden_size
    norm_eps: float = 1e-6
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    eos_token_id: int = 50256
    pad_token_id: int = 50303

# Model class moved to full_attention_model.py
# This file now contains only shared components:
# - RMSNorm, Block, CausalSelfAttention, MLP (transformer components)
# - ModelConfig (base config class)
# - Helper functions: precompute_freqs_cis, apply_rotary_emb, etc.
# - IGNORE_INDEX constant
#
# The concrete standard-attention Model implementation is in full_attention_model.py.

    
