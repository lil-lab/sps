import torch
from torch import Tensor
from typing import Callable

# --- GENERATIO ONLY MASKS ---
def get_mask_mod_w_offset(mask_mod: Callable[[int, int, Tensor, Tensor], Tensor], _offset: Tensor) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    def _mask_mod(b, h, q, kv):
        return mask_mod(b, h, q + _offset, kv)
    return _mask_mod

def left_padding_mask_factory_method(padding_offsets: Tensor) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    """
    Create a mask function for left-padded sequences.

    Args:
        padding_offsets: 1D tensor of shape (batch,) indicating the number of
                        left-padding tokens at the beginning of each sequence.

    Returns:
        Mask function that returns False for padding positions, True for valid positions.
    """
    def mask(b, h, q_idx, kv_idx):
        return kv_idx >= padding_offsets[b]
    return mask

def cached_tokens_padding_mask_factory_method(valid_lengths: Tensor, past_max_len: int) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    """
    Mask out padded positions in the middle of batched cached tokens.
    
    When sequences have different cache lengths, shorter sequences have padding
    in the middle range [valid_lengths[b], past_max_len) between their cached
    tokens and the new tokens being added.

    Args:
        valid_lengths: 1D tensor of shape (batch,) indicating number of valid cached tokens.
        past_max_len: maximum cached length across batch (before adding new tokens).

    Returns:
        Mask function that returns False for padded positions [valid_lengths[b], past_max_len),
        and True for all other positions (valid cached tokens and new tokens).
    """
    def mask(b, h, q_idx, kv_idx):
        # Padding region is [valid_lengths[b], past_max_len)
        is_padding = (kv_idx >= valid_lengths[b]) & (kv_idx < past_max_len)
        return ~is_padding
    return mask


# --- GENERAL MASKS ---

def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def sliding_window_mask_factory_method(window_size: int) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    return lambda b, h, q_idx, kv_idx: kv_idx >= q_idx - window_size

def document_mask_factory_method(documents_idx: Tensor) -> Callable[[int, int, Tensor, Tensor], Tensor]:
    return lambda b, h, q_idx, kv_idx: documents_idx[b][q_idx] == documents_idx[b][kv_idx]
