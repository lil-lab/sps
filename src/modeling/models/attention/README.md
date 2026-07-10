# Attention kernels

Four Triton attention kernels. `original_flash_attn.py` is the upstream Flash-Attention-2
tutorial kernel, vendored unchanged. The other three are that kernel plus one edit to the
**score step**: the few lines in `_attn_fwd_inner` between `qk = q·kᵀ` and the row max where
a mask is added to `qk`. The online-softmax core (`m_i`, `l_i`, rescaled `acc`) is textually
identical across all four files; the backward, the numeric-safety conventions, and the
sequence padding are shared by the three that ship. So the score step is the entire
difference, and SPS is a copy of reverse-SPS with a single parity bit flipped.


| File                                    | Entry point                     | Used by                                 |
| --------------------------------------- | ------------------------------- | --------------------------------------- |
| `original_flash_attn.py`                | `attention`                     | reference only (untouched FA2 baseline) |
| `triton_full_flash_attention.py`        | `full_attention`                | Standard                                |
| `triton_reverse_sps_flash_attention.py` | `reverse_sps_sliding_attention` | Reverse-SPS, Delayed-State              |
| `triton_sps_flash_attention.py`         | `sps_sliding_attention`         | SPS                                     |


## The one block that differs

Every difference lives in `_attn_fwd_inner` (`_attn_fwd_inner_sps` for SPS), in the lines
right after `qk = qk * qk_scale`. `original` adds nothing there (causal mask only).

`full` adds a single intra-document mask:

```python
if HAS_DOCUMENT_MASK:
    same_doc = docs_q[:, None] == docs_k[None, :]
    qk += tl.where(same_doc, 0.0, -1.0e6)                # keep only same-document keys
```

`reverse_sps` keeps that same document mask and, before it, adds a sliding window that the
persistent key is exempt from:

```python
if HAS_PREDICT_BIAS:
    rel = q_tok[:, None] - k_tok[None, :]                # query/key distance, in token pairs
    normal_bias = tl.where(rel > temporary_key_window, -1.0e6, 0.0)   # sliding window (default 64)
    k_is_predict = (offs_n_abs % 2) == 1                 # persistent key = odd <predict> slot
    attn_bias = tl.where(k_is_predict, 0.0, normal_bias) # persistent key is global, skips the window
    is_self = (offs_m[:, None] == offs_n_abs[None, :]) & k_is_predict
    qk += tl.where(is_self, 0.0, attn_bias)              # persistent key always self-attends
```

`sps` is that same kernel with one line flipped, moving the persistent key to the even input
slot (the local is renamed `k_is_persistent` to match):

```python
k_is_persistent = (offs_n_abs % 2) == 0                  # reverse-SPS uses == 1, the odd <predict> slot
```

The layout doubles the sequence: even slot = input token, odd slot = `<predict>`,
`tok = pos // 2`. Each pair has one persistent key (global context, window-exempt) and one
windowed readout slot whose hidden state feeds `lm_head`. Flipping which parity is persistent
is all that separates SPS from reverse-SPS; it swaps which slot the model reads its next-token
logits from, making the two mirror images.

## Additional details

- **Softmax core**: byte-identical to `original_flash_attn.py`.
- **Numeric safety**: masks add a finite `-1e6` (never `-inf`) and `m_i` starts at `-1e9`, so
a fully-masked row yields `0`, not `NaN`. The baseline keeps `m_i = -inf`, safe only because
plain causal attention never fully masks a row.
- **Sequence padding**: `N_CTX` (the doubled `2T` for SPS) is rounded up to a multiple of 128
before launch. The flat `[B·H·N_CTX, D]` descriptor has no per-row bounds check, so an  
unaligned final block would write into the next head's rows, a cross-program race. Padded  
positions sit after all real tokens (masked out), padded query rows are sliced off the  
output, padded document ids use a `-1` sentinel, and the backward pads the incoming gradient
and slices `dq`/`dk`/`dv` the same way. It is a no-op at aligned training lengths.
- **Backward**: recomputes the same position-derived mask instead of storing the score
matrix, and returns only `dQ`/`dK`/`dV` (there is nothing else to differentiate).

## Tests

Under `src/modeling/tests/`. Each kernel is checked forward and backward against an
additive-mask PyTorch reference (`scaled_dot_product_attention` or `flex_attention`), plus
NaN-poison tests for cross-document leakage and model-level parity against each
`forward_efficient` path. The SPS decode path uses a separate kernel
(`triton_segmented_q2_attention` in `models/utils/decode_attention.py`).