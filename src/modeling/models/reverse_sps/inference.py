from __future__ import annotations

"""Autoregressive evaluation and generation utilities for ReverseSPS models."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from modeling.models.model import (
    apply_rotary_emb,
    compute_left_padded_position_ids,
    infer_is_real_tokens,
    validate_left_padded_tokens,
)
from modeling.models.utils.decode_attention import (
    KVSegment,
    masked_kv_attention,
    triton_segmented_q2_attention,
)
from modeling.models.utils.decode_attention_fast import fused_sps_kv_update
from modeling.models.utils.generation import generate_with_batched_prefill
from modeling.models.utils.sampling import sample_next_token

from .core import IGNORE_INDEX, build_reverse_sps_loss_and_stats


@dataclass
class _KVMemory:
    k: list[Tensor]
    v: list[Tensor]
    len: list[Tensor]


@dataclass
class _WindowMemory:
    k: list[Tensor]
    v: list[Tensor]
    len: list[Tensor]
    pos: list[Tensor]


@dataclass
class _ReverseSPSDecodeState:
    normal_window: _WindowMemory
    predict_retained: _KVMemory
    processed_tokens_B: Tensor
    next_doc_start_B: Tensor
    attn_dtype: torch.dtype
    normal_window_tokens: int
    head_dim: int
    live_token_cap: int


class ReverseSPSInferenceMixin:
    _prediction_slot_index = 0
    _window_slot_index = 0
    _retained_slot_index = 1
    _decode_error_name = "Reverse SPS"

    def _build_decode_state(
        self,
        *,
        batch_size: int,
        max_tokens: int,
        device: torch.device,
        attn_dtype: torch.dtype,
    ) -> _ReverseSPSDecodeState:
        n_layer = self.config.n_layer
        n_head = self.config.n_head
        head_dim = self.config.hidden_size // self.config.n_head
        normal_window_tokens = int(self.config.window_size)
        normal_window_storage = max(normal_window_tokens, 1)

        def _make_kv_memory(max_t: int) -> _KVMemory:
            return _KVMemory(
                k=[
                    torch.zeros((batch_size, n_head, max_t, head_dim), device=device, dtype=attn_dtype)
                    for _ in range(n_layer)
                ],
                v=[
                    torch.zeros((batch_size, n_head, max_t, head_dim), device=device, dtype=attn_dtype)
                    for _ in range(n_layer)
                ],
                len=[torch.zeros((batch_size, n_head), device=device, dtype=torch.long) for _ in range(n_layer)],
            )

        return _ReverseSPSDecodeState(
            normal_window=_WindowMemory(
                k=[
                    torch.zeros((batch_size, n_head, normal_window_storage, head_dim), device=device, dtype=attn_dtype)
                    for _ in range(n_layer)
                ],
                v=[
                    torch.zeros((batch_size, n_head, normal_window_storage, head_dim), device=device, dtype=attn_dtype)
                    for _ in range(n_layer)
                ],
                len=[torch.zeros((batch_size,), device=device, dtype=torch.long) for _ in range(n_layer)],
                pos=[torch.zeros((batch_size,), device=device, dtype=torch.long) for _ in range(n_layer)],
            ),
            predict_retained=_make_kv_memory(max_tokens),
            processed_tokens_B=torch.zeros((batch_size,), device=device, dtype=torch.long),
            next_doc_start_B=torch.ones((batch_size,), device=device, dtype=torch.bool),
            attn_dtype=attn_dtype,
            normal_window_tokens=normal_window_tokens,
            head_dim=head_dim,
            live_token_cap=0,
        )

    def _reset_layer_memory(
        self,
        state: _ReverseSPSDecodeState,
        layer_idx: int,
        reset_mask_B: Tensor,
    ) -> None:
        reset_mask_B = reset_mask_B.to(dtype=torch.bool, device=state.processed_tokens_B.device)
        state.normal_window.len[layer_idx].masked_fill_(reset_mask_B, 0)
        state.normal_window.pos[layer_idx].masked_fill_(reset_mask_B, 0)
        state.predict_retained.len[layer_idx].masked_fill_(reset_mask_B.unsqueeze(1), 0)

    def _update_normal_memory(
        self,
        state: _ReverseSPSDecodeState,
        layer_idx: int,
        k_BxHxD: Tensor,
        v_BxHxD: Tensor,
        active_mask_B: Tensor,
    ) -> None:
        # Defer: the window and retained buffers are updated together in a single
        # fused Triton kernel in `_update_predict_memory`, once both K/V tensors are
        # in hand. Stash this step's window K/V and do nothing else here.
        self._fused_kv_pending = (state, layer_idx, k_BxHxD, v_BxHxD, active_mask_B)

    def _update_normal_memory_unfused(
        self,
        state: _ReverseSPSDecodeState,
        layer_idx: int,
        k_BxHxD: Tensor,
        v_BxHxD: Tensor,
        active_mask_B: Tensor,
    ) -> None:
        if state.normal_window_tokens == 0:
            return

        nw = state.normal_window
        active_mask_B = active_mask_B.to(dtype=torch.bool, device=k_BxHxD.device)
        b = k_BxHxD.size(0)
        batch_idx_B = torch.arange(b, device=k_BxHxD.device)
        len_B = nw.len[layer_idx]
        pos_B = nw.pos[layer_idx]
        full_B = len_B >= state.normal_window_tokens
        insert_B = torch.where(full_B, pos_B, len_B)

        old_k_BxHxD = nw.k[layer_idx][batch_idx_B, :, insert_B, :]
        old_v_BxHxD = nw.v[layer_idx][batch_idx_B, :, insert_B, :]
        write_mask_Bx1x1 = active_mask_B.view(b, 1, 1)
        nw.k[layer_idx][batch_idx_B, :, insert_B, :] = torch.where(write_mask_Bx1x1, k_BxHxD, old_k_BxHxD)
        nw.v[layer_idx][batch_idx_B, :, insert_B, :] = torch.where(write_mask_Bx1x1, v_BxHxD, old_v_BxHxD)

        grow_B = active_mask_B & (~full_B)
        advance_B = active_mask_B & full_B
        nw.len[layer_idx].copy_(torch.where(grow_B, len_B + 1, len_B))
        next_pos_B = torch.remainder(pos_B + 1, state.normal_window_tokens)
        nw.pos[layer_idx].copy_(torch.where(advance_B, next_pos_B, pos_B))

    def _update_predict_memory(
        self,
        state: _ReverseSPSDecodeState,
        layer_idx: int,
        k_BxHxD: Tensor,
        v_BxHxD: Tensor,
        active_mask_B: Tensor,
    ) -> None:
        # Second half of the fused KV update: `_update_normal_memory` stashed this
        # step's window K/V; now that the retained K/V are in hand, update both
        # buffers in a single Triton launch (identical to the un-fused path, ~25%
        # faster by collapsing ~20 micro-launches per layer into one).
        pending = getattr(self, "_fused_kv_pending", None)
        if pending is None:
            # Paranoid path — the caller always queues the window update first.
            return self._update_predict_memory_unfused(
                state, layer_idx, k_BxHxD, v_BxHxD, active_mask_B
            )

        prev_state, prev_layer, k_window_BxHxD, v_window_BxHxD, prev_active = pending
        if prev_state is not state or prev_layer != layer_idx:
            # Mismatch — fall back to the un-fused path so we never silently corrupt state.
            self._update_normal_memory_unfused(prev_state, prev_layer, k_window_BxHxD, v_window_BxHxD, prev_active)
            self._update_predict_memory_unfused(state, layer_idx, k_BxHxD, v_BxHxD, active_mask_B)
            self._fused_kv_pending = None
            return

        nw = state.normal_window
        br = state.predict_retained
        # Cast new k/v to cache dtype.
        cache_dtype = nw.k[layer_idx].dtype if state.normal_window_tokens > 0 else br.k[layer_idx].dtype
        k_window_cast = k_window_BxHxD.to(cache_dtype)
        v_window_cast = v_window_BxHxD.to(cache_dtype)
        k_retained_cast = k_BxHxD.to(cache_dtype)
        v_retained_cast = v_BxHxD.to(cache_dtype)

        if state.normal_window_tokens > 0:
            fused_sps_kv_update(
                nw_k=nw.k[layer_idx],
                nw_v=nw.v[layer_idx],
                nw_len=nw.len[layer_idx],
                nw_pos=nw.pos[layer_idx],
                br_k=br.k[layer_idx],
                br_v=br.v[layer_idx],
                br_len=br.len[layer_idx],
                k_window_BxHxD=k_window_cast,
                v_window_BxHxD=v_window_cast,
                k_retained_BxHxD=k_retained_cast,
                v_retained_BxHxD=v_retained_cast,
                active_mask_B=active_mask_B,
                window_tokens=state.normal_window_tokens,
            )
        else:
            # Window disabled — only the retained buffer matters.
            fused_sps_kv_update(
                nw_k=nw.k[layer_idx] if len(nw.k) > layer_idx else br.k[layer_idx],
                nw_v=nw.v[layer_idx] if len(nw.v) > layer_idx else br.v[layer_idx],
                nw_len=nw.len[layer_idx] if len(nw.len) > layer_idx else br.len[layer_idx, :1].view(-1).contiguous(),
                nw_pos=nw.pos[layer_idx] if len(nw.pos) > layer_idx else br.len[layer_idx, :1].view(-1).contiguous(),
                br_k=br.k[layer_idx],
                br_v=br.v[layer_idx],
                br_len=br.len[layer_idx],
                k_window_BxHxD=k_window_cast,
                v_window_BxHxD=v_window_cast,
                k_retained_BxHxD=k_retained_cast,
                v_retained_BxHxD=v_retained_cast,
                active_mask_B=active_mask_B,
                window_tokens=0,
            )
        self._fused_kv_pending = None

    def _update_predict_memory_unfused(
        self,
        state: _ReverseSPSDecodeState,
        layer_idx: int,
        k_BxHxD: Tensor,
        v_BxHxD: Tensor,
        active_mask_B: Tensor,
    ) -> None:
        br = state.predict_retained
        active_mask_B = active_mask_B.to(dtype=torch.bool, device=k_BxHxD.device)
        b, n_head, _ = k_BxHxD.shape
        batch_idx_BxH = torch.arange(b, device=k_BxHxD.device).view(b, 1).expand(b, n_head)
        head_idx_BxH = torch.arange(n_head, device=k_BxHxD.device).view(1, n_head).expand(b, n_head)
        len_BxH = br.len[layer_idx]

        old_k_BxHxD = br.k[layer_idx][batch_idx_BxH, head_idx_BxH, len_BxH, :]
        old_v_BxHxD = br.v[layer_idx][batch_idx_BxH, head_idx_BxH, len_BxH, :]
        write_mask_Bx1x1 = active_mask_B.view(b, 1, 1)
        br.k[layer_idx][batch_idx_BxH, head_idx_BxH, len_BxH, :] = torch.where(
            write_mask_Bx1x1,
            k_BxHxD,
            old_k_BxHxD,
        )
        br.v[layer_idx][batch_idx_BxH, head_idx_BxH, len_BxH, :] = torch.where(
            write_mask_Bx1x1,
            v_BxHxD,
            old_v_BxHxD,
        )
        br.len[layer_idx].copy_(torch.where(active_mask_B.unsqueeze(1), len_BxH + 1, len_BxH))

    def _attend_from_decode_state(
        self,
        state: _ReverseSPSDecodeState,
        layer_idx: int,
        q_BxHxD: Tensor,
        *,
        curr_normal_k_BxHxD: Optional[Tensor] = None,
        curr_normal_v_BxHxD: Optional[Tensor] = None,
        curr_predict_k_BxHxD: Optional[Tensor] = None,
        curr_predict_v_BxHxD: Optional[Tensor] = None,
        check_finite: bool = False,
    ) -> Tensor:
        # Cap segment lengths at the (constant) retained-storage size rather than
        # `state.live_token_cap`, which grows every step. A constant `scan_t` lets
        # the Triton kernel reuse one compiled binary across all decode steps; the
        # kernel still masks invalid keys via `seg.lengths`, so results are unchanged.
        max_tokens = int(state.predict_retained.k[layer_idx].size(2))
        segments = []
        if state.normal_window_tokens > 0:
            segments.append(
                KVSegment(
                    state.normal_window.k[layer_idx],
                    state.normal_window.v[layer_idx],
                    state.normal_window.len[layer_idx],
                    max_len_cap=min(state.normal_window_tokens, max_tokens),
                )
            )
        segments.append(
            KVSegment(
                state.predict_retained.k[layer_idx],
                state.predict_retained.v[layer_idx],
                state.predict_retained.len[layer_idx],
                max_len_cap=max_tokens,
            )
        )

        extra_kv: list[tuple[Tensor, Tensor]] = []
        if curr_normal_k_BxHxD is not None and curr_normal_v_BxHxD is not None:
            extra_kv.append((curr_normal_k_BxHxD, curr_normal_v_BxHxD))
        if curr_predict_k_BxHxD is not None and curr_predict_v_BxHxD is not None:
            extra_kv.append((curr_predict_k_BxHxD, curr_predict_v_BxHxD))

        return masked_kv_attention(
            q_BxHxD,
            segments,
            extra_kv=extra_kv or None,
            attn_dtype=state.attn_dtype,
            check_finite=check_finite,
            error_context=f" in decode state layer={layer_idx}",
            prefer_triton=True,
        )

    def _attend_pair_from_decode_state(
        self,
        state: _ReverseSPSDecodeState,
        layer_idx: int,
        q_Bx2xHxD: Tensor,
        *,
        curr_normal_k_BxHxD: Tensor,
        curr_normal_v_BxHxD: Tensor,
        curr_predict_k_BxHxD: Tensor,
        curr_predict_v_BxHxD: Tensor,
    ) -> Tensor:
        # Constant `max_len_cap` (= retained storage) for cross-step kernel reuse;
        # see `_attend_from_decode_state`.
        max_tokens = int(state.predict_retained.k[layer_idx].size(2))
        segments = []
        if state.normal_window_tokens > 0:
            segments.append(
                KVSegment(
                    state.normal_window.k[layer_idx],
                    state.normal_window.v[layer_idx],
                    state.normal_window.len[layer_idx],
                    max_len_cap=min(state.normal_window_tokens, max_tokens),
                )
            )
        segments.append(
            KVSegment(
                state.predict_retained.k[layer_idx],
                state.predict_retained.v[layer_idx],
                state.predict_retained.len[layer_idx],
                max_len_cap=max_tokens,
            )
        )
        return triton_segmented_q2_attention(
            q_Bx2xHxD,
            segments,
            current_normal_k_BxHxD=curr_normal_k_BxHxD,
            current_normal_v_BxHxD=curr_normal_v_BxHxD,
            current_predict_k_BxHxD=curr_predict_k_BxHxD,
            current_predict_v_BxHxD=curr_predict_v_BxHxD,
            attn_dtype=state.attn_dtype,
        )

    def _decode_one_token_step(
        self,
        state: _ReverseSPSDecodeState,
        token_B: Tensor,
        active_mask_B: Tensor,
        freqs_all: Tensor,
        *,
        check_finite: bool = False,
        require_triton_pair_attention: bool = False,
    ) -> Tensor:
        b = token_B.size(0)
        device = token_B.device
        n_head = self.config.n_head
        head_dim = state.head_dim

        active_mask_B = active_mask_B.to(dtype=torch.bool, device=device)
        zero_logits = self.lm_head.weight.new_zeros((b, self.config.vocab_size))
        if not bool(active_mask_B.any()):
            return zero_logits

        if bool((state.processed_tokens_B[active_mask_B] >= self.freqs_cis.shape[0]).any()):
            raise ValueError(
                f"Cannot decode beyond block size {self.freqs_cis.shape[0]} in "
                f"{self._decode_error_name} generation"
            )

        safe_token_B = torch.where(active_mask_B, token_B, torch.zeros_like(token_B))
        idx_pair = torch.stack(
            [
                safe_token_B,
                torch.full_like(safe_token_B, self.config.predict_token_id),
            ],
            dim=1,
        )
        position_ids_B = torch.where(
            active_mask_B,
            state.processed_tokens_B,
            torch.zeros_like(state.processed_tokens_B),
        )
        freqs_pair = freqs_all[position_ids_B].unsqueeze(1).expand(-1, 2, -1)
        x = self.transformer.drop(self.transformer.wte(idx_pair))

        for layer_idx, block in enumerate(self.transformer.h):
            doc_start_normal_B = state.next_doc_start_B & active_mask_B
            self._reset_layer_memory(state, layer_idx, doc_start_normal_B)

            x_norm = block.attention_norm(x)
            q, k, v = block.attn.c_attn(x_norm).split(block.attn.hidden_size, dim=2)
            q = q.view(b, 2, n_head, head_dim)
            k = k.view(b, 2, n_head, head_dim)
            v = v.view(b, 2, n_head, head_dim)
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_pair)
            q_Bx2xHxD = q.to(state.attn_dtype)
            k_Bx2xHxD = k.to(state.attn_dtype)
            v_Bx2xHxD = v.to(state.attn_dtype)

            k_slot0_BxHxD = k_Bx2xHxD[:, 0]
            v_slot0_BxHxD = v_Bx2xHxD[:, 0]
            k_slot1_BxHxD = k_Bx2xHxD[:, 1]
            v_slot1_BxHxD = v_Bx2xHxD[:, 1]
            if require_triton_pair_attention:
                attn_out_pair_Bx2xHxD = self._attend_pair_from_decode_state(
                    state,
                    layer_idx,
                    q_Bx2xHxD,
                    curr_normal_k_BxHxD=k_slot0_BxHxD,
                    curr_normal_v_BxHxD=v_slot0_BxHxD,
                    curr_predict_k_BxHxD=k_slot1_BxHxD,
                    curr_predict_v_BxHxD=v_slot1_BxHxD,
                )
                attn_out_slot0_BxHxD = attn_out_pair_Bx2xHxD[:, 0]
                attn_out_slot1_BxHxD = attn_out_pair_Bx2xHxD[:, 1]
            else:
                q_slot0_BxHxD = q_Bx2xHxD[:, 0]
                attn_out_slot0_BxHxD = self._attend_from_decode_state(
                    state,
                    layer_idx,
                    q_slot0_BxHxD,
                    curr_normal_k_BxHxD=k_slot0_BxHxD,
                    curr_normal_v_BxHxD=v_slot0_BxHxD,
                    check_finite=check_finite,
                )

                q_slot1_BxHxD = q_Bx2xHxD[:, 1]
                attn_out_slot1_BxHxD = self._attend_from_decode_state(
                    state,
                    layer_idx,
                    q_slot1_BxHxD,
                    curr_normal_k_BxHxD=k_slot0_BxHxD,
                    curr_normal_v_BxHxD=v_slot0_BxHxD,
                    curr_predict_k_BxHxD=k_slot1_BxHxD,
                    curr_predict_v_BxHxD=v_slot1_BxHxD,
                    check_finite=check_finite,
                )

            k_window_BxHxD = k_Bx2xHxD[:, self._window_slot_index]
            v_window_BxHxD = v_Bx2xHxD[:, self._window_slot_index]
            k_retained_BxHxD = k_Bx2xHxD[:, self._retained_slot_index]
            v_retained_BxHxD = v_Bx2xHxD[:, self._retained_slot_index]
            self._update_normal_memory(
                state,
                layer_idx,
                k_window_BxHxD,
                v_window_BxHxD,
                active_mask_B,
            )
            self._update_predict_memory(
                state,
                layer_idx,
                k_retained_BxHxD,
                v_retained_BxHxD,
                active_mask_B,
            )

            attn_out_pair = torch.stack(
                [attn_out_slot0_BxHxD, attn_out_slot1_BxHxD],
                dim=1,
            ).reshape(b, 2, self.config.hidden_size).to(x.dtype)
            attn_out_pair = block.attn.resid_dropout(block.attn.c_proj(attn_out_pair))
            x = x + attn_out_pair
            x = x + block.mlp(block.mlp_norm(x))
            if check_finite and (not torch.isfinite(x).all()):
                raise RuntimeError(f"Non-finite hidden state in decode step layer={layer_idx}")

        x = self.transformer.output_norm(x)
        logits_pair_Bx2xV = self.lm_head(x)
        token_logits_BxV = logits_pair_Bx2xV[:, self._prediction_slot_index, :]
        return torch.where(active_mask_B.unsqueeze(1), token_logits_BxV, torch.zeros_like(token_logits_BxV))

    def _advance_decode_state(
        self,
        state: _ReverseSPSDecodeState,
        token_B: Tensor,
        active_mask_B: Tensor,
    ) -> None:
        active_mask_B = active_mask_B.to(dtype=torch.bool, device=token_B.device)
        state.processed_tokens_B[active_mask_B] = state.processed_tokens_B[active_mask_B] + 1
        if bool(active_mask_B.any()):
            state.live_token_cap = min(state.live_token_cap + 1, int(self.freqs_cis.shape[0]))
        state.next_doc_start_B = torch.where(
            active_mask_B,
            token_B == self.config.eos_token_id,
            state.next_doc_start_B,
        )

    def _batched_prefill_attention_fn(self):
        if self._decode_error_name == "SPS":
            from modeling.models.sps.core import triton_sps_sliding_attention

            return triton_sps_sliding_attention
        from modeling.models.reverse_sps.core import triton_reverse_sps_sliding_attention

        return triton_reverse_sps_sliding_attention

    def _can_use_batched_generation_prefill(self, idx_BxT: Tensor) -> bool:
        if not idx_BxT.is_cuda:
            return False
        if self._batched_prefill_attention_fn() is None:
            return False
        head_dim = int(self.config.hidden_size // self.config.n_head)
        if head_dim not in {16, 32, 64, 128, 256}:
            return False
        return all(
            bool(getattr(block.attn, "enable_triton_attention", False))
            and float(getattr(block.attn, "dropout", 0.0)) == 0.0
            for block in self.transformer.h
        )

    def _prefill_attention_from_qkv(
        self,
        block,
        q_Bx2TxHxD: Tensor,
        k_Bx2TxHxD: Tensor,
        v_Bx2TxHxD: Tensor,
        documents_idx_Bx2T: Tensor,
    ) -> Tensor:
        b, two_t, _, head_dim = q_Bx2TxHxD.shape
        t = two_t // 2
        attn = block.attn
        attn_dtype = torch.bfloat16 if q_Bx2TxHxD.is_cuda else q_Bx2TxHxD.dtype
        q_BxHx2TxD = q_Bx2TxHxD.transpose(1, 2).to(attn_dtype)
        k_BxHx2TxD = k_Bx2TxHxD.transpose(1, 2).to(attn_dtype)
        v_BxHx2TxD = v_Bx2TxHxD.transpose(1, 2).to(attn_dtype)
        attention_fn = self._batched_prefill_attention_fn()
        if attention_fn is None:
            raise RuntimeError(f"Triton attention is not available for {self._decode_error_name} prefill")
        y_BxHx2TxD = attention_fn(
            q_BxHx2TxD,
            k_BxHx2TxD,
            v_BxHx2TxD,
            1.0 / math.sqrt(head_dim),
            attn.window_size,
            warp_specialize=attn.warp_specialize,
            documents_idx_BxT=documents_idx_Bx2T.contiguous(),
            persistent_key_window=attn.persistent_key_window,
        )
        y_Bx2TxC = y_BxHx2TxD.transpose(1, 2).contiguous().view(
            b,
            two_t,
            self.config.hidden_size,
        )
        y_Bx2TxC = y_Bx2TxC.to(attn.c_proj.weight.dtype)
        return attn.resid_dropout(attn.c_proj(y_Bx2TxC))

    def _materialize_prefill_layer_state(
        self,
        state: _ReverseSPSDecodeState,
        layer_idx: int,
        k_Bx2TxHxD: Tensor,
        v_Bx2TxHxD: Tensor,
        current_doc_mask_BxT: Tensor,
    ) -> None:
        k_window_BxTxHxD = k_Bx2TxHxD[:, self._window_slot_index :: 2].to(
            dtype=state.normal_window.k[layer_idx].dtype
        )
        v_window_BxTxHxD = v_Bx2TxHxD[:, self._window_slot_index :: 2].to(
            dtype=state.normal_window.v[layer_idx].dtype
        )
        k_retained_BxTxHxD = k_Bx2TxHxD[:, self._retained_slot_index :: 2].to(
            dtype=state.predict_retained.k[layer_idx].dtype
        )
        v_retained_BxTxHxD = v_Bx2TxHxD[:, self._retained_slot_index :: 2].to(
            dtype=state.predict_retained.v[layer_idx].dtype
        )

        for batch_idx in range(current_doc_mask_BxT.size(0)):
            doc_positions = current_doc_mask_BxT[batch_idx].nonzero(as_tuple=False).flatten()
            doc_len = int(doc_positions.numel())
            if doc_len == 0:
                continue

            state.predict_retained.k[layer_idx][batch_idx, :, :doc_len, :] = (
                k_retained_BxTxHxD[batch_idx, doc_positions].transpose(0, 1)
            )
            state.predict_retained.v[layer_idx][batch_idx, :, :doc_len, :] = (
                v_retained_BxTxHxD[batch_idx, doc_positions].transpose(0, 1)
            )
            state.predict_retained.len[layer_idx][batch_idx, :] = doc_len

            if state.normal_window_tokens > 0:
                window_positions = doc_positions[-state.normal_window_tokens :]
                window_len = int(window_positions.numel())
                state.normal_window.k[layer_idx][batch_idx, :, :window_len, :] = (
                    k_window_BxTxHxD[batch_idx, window_positions].transpose(0, 1)
                )
                state.normal_window.v[layer_idx][batch_idx, :, :window_len, :] = (
                    v_window_BxTxHxD[batch_idx, window_positions].transpose(0, 1)
                )
                state.normal_window.len[layer_idx][batch_idx] = window_len
                state.normal_window.pos[layer_idx][batch_idx] = 0

    def _prefill_generation_state_batched(
        self,
        idx_BxT: Tensor,
        max_new_tokens: int,
    ) -> tuple[_ReverseSPSDecodeState, Tensor, str]:
        device = idx_BxT.device
        b, t = idx_BxT.size()
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=False,
            context=f"{self._decode_error_name} generation prefill prompts",
        )
        real_lengths_B = is_real_BxT.sum(dim=1).to(dtype=torch.long)
        max_real_prompt_tokens = int(real_lengths_B.max().item())
        total_real_tokens = max_real_prompt_tokens + int(max_new_tokens)
        if total_real_tokens > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Cannot generate {max_new_tokens} new tokens from a prompt with "
                f"{max_real_prompt_tokens} real tokens when block size is {self.freqs_cis.shape[0]}"
            )

        attn_dtype = torch.bfloat16 if device.type == "cuda" else self.transformer.wte.weight.dtype
        state = self._build_decode_state(
            batch_size=b,
            max_tokens=total_real_tokens,
            device=device,
            attn_dtype=attn_dtype,
        )
        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        documents_idx_Bx2T = documents_idx_BxT.repeat_interleave(2, dim=1)
        last_doc_idx_B = documents_idx_BxT[:, -1]
        current_doc_mask_BxT = is_real_BxT & (documents_idx_BxT == last_doc_idx_B.unsqueeze(1))
        position_ids_BxT = compute_left_padded_position_ids(is_real_BxT)
        position_ids_Bx2T = position_ids_BxT.repeat_interleave(2, dim=1)
        freqs_cis_Bx2TxD = self.freqs_cis.to(device)[position_ids_Bx2T]

        idx_Bx2T = self.add_predict_tokens(idx_BxT)
        x = self.transformer.drop(self.transformer.wte(idx_Bx2T))
        for layer_idx, block in enumerate(self.transformer.h):
            x_norm = block.attention_norm(x)
            q_Bx2TxC, k_Bx2TxC, v_Bx2TxC = block.attn.c_attn(x_norm).split(
                block.attn.hidden_size,
                dim=2,
            )
            head_dim = self.config.hidden_size // self.config.n_head
            q_Bx2TxHxD = q_Bx2TxC.view(b, 2 * t, self.config.n_head, head_dim)
            k_Bx2TxHxD = k_Bx2TxC.view(b, 2 * t, self.config.n_head, head_dim)
            v_Bx2TxHxD = v_Bx2TxC.view(b, 2 * t, self.config.n_head, head_dim)
            q_Bx2TxHxD, k_Bx2TxHxD = apply_rotary_emb(
                q_Bx2TxHxD,
                k_Bx2TxHxD,
                freqs_cis=freqs_cis_Bx2TxD,
            )
            self._materialize_prefill_layer_state(
                state,
                layer_idx,
                k_Bx2TxHxD,
                v_Bx2TxHxD,
                current_doc_mask_BxT,
            )
            attn_output = self._prefill_attention_from_qkv(
                block,
                q_Bx2TxHxD,
                k_Bx2TxHxD,
                v_Bx2TxHxD,
                documents_idx_Bx2T,
            )
            x = x + attn_output
            x = x + block.mlp(block.mlp_norm(x))

        x = self.transformer.output_norm(x)
        pred_hidden_BxTxC = x[:, self._prediction_slot_index :: 2]
        next_logits_BxV = self.lm_head(pred_hidden_BxTxC[:, -1, :])[:, : self.config.vocab_size]
        state.processed_tokens_B.copy_(real_lengths_B)
        state.next_doc_start_B.copy_(idx_BxT[:, -1] == self.config.eos_token_id)
        state.live_token_cap = int(current_doc_mask_BxT.sum(dim=1).max().item())
        return state, next_logits_BxV, "batched_prefill"

    def _prefill_generation_state_stepwise(
        self,
        idx_BxT: Tensor,
        max_new_tokens: int,
    ) -> tuple[_ReverseSPSDecodeState, Tensor, str]:
        device = idx_BxT.device
        b = idx_BxT.size(0)
        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=False,
            context=f"{self._decode_error_name} generation prefill prompts",
        )
        max_real_prompt_tokens = int(is_real_BxT.sum(dim=1).max().item())
        total_real_tokens = max_real_prompt_tokens + int(max_new_tokens)
        if total_real_tokens > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Cannot generate {max_new_tokens} new tokens from a prompt with "
                f"{max_real_prompt_tokens} real tokens when block size is {self.freqs_cis.shape[0]}"
            )

        attn_dtype = torch.bfloat16 if device.type == "cuda" else self.transformer.wte.weight.dtype
        state = self._build_decode_state(
            batch_size=b,
            max_tokens=total_real_tokens,
            device=device,
            attn_dtype=attn_dtype,
        )
        freqs_all = self.freqs_cis.to(device)
        last_logits_BxV = self.lm_head.weight.new_zeros((b, self.config.vocab_size))
        for col in range(idx_BxT.size(1)):
            active_col_B = is_real_BxT[:, col]
            logits_step_BxV = self._decode_one_token_step(
                state,
                idx_BxT[:, col],
                active_col_B,
                freqs_all,
            )
            last_logits_BxV = torch.where(
                active_col_B.unsqueeze(1),
                logits_step_BxV,
                last_logits_BxV,
            )
            self._advance_decode_state(state, idx_BxT[:, col], active_col_B)
        return state, last_logits_BxV, "step_prefill"

    def _prefill_generation_state(
        self,
        idx_BxT: Tensor,
        max_new_tokens: int,
        *,
        require_batched: bool = False,
    ) -> tuple[_ReverseSPSDecodeState, Tensor, str]:
        if self._can_use_batched_generation_prefill(idx_BxT):
            return self._prefill_generation_state_batched(idx_BxT, max_new_tokens)
        if require_batched:
            raise RuntimeError(
                f"{self._decode_error_name} batched generation prefill requires CUDA Triton attention"
            )
        return self._prefill_generation_state_stepwise(idx_BxT, max_new_tokens)

    def _decode_generation_state(
        self,
        state: _ReverseSPSDecodeState,
        token_B: Tensor,
        active_mask_B: Tensor,
    ) -> Tensor:
        return self._decode_one_token_step(
            state,
            token_B,
            active_mask_B,
            self.freqs_cis.to(token_B.device),
            require_triton_pair_attention=True,
        )

    def forward_efficient(
        self,
        idx_BxT: Tensor,
        targets_BxT: Optional[Tensor] = None,
        *,
        progress: bool = False,
        progress_every: Optional[int] = None,
        progress_prefix: str = "forward_efficient",
        check_finite: bool = False,
        store_token_logits: bool = True,
    ):
        device = idx_BxT.device
        b, t = idx_BxT.size()
        if t == 0:
            empty_logits = self.lm_head.weight.new_empty((b, 0, self.config.vocab_size))
            if targets_BxT is None:
                return empty_logits
            zero = empty_logits.new_zeros(())
            is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
            documents_idx_BxT = self.generate_document_idx(idx_BxT)
            _, stats = build_reverse_sps_loss_and_stats(
                token_nll_sum=zero,
                token_count=torch.tensor(0, device=device, dtype=torch.long),
                documents_idx_BxT=documents_idx_BxT,
                is_real_BxT=is_real_BxT,
            )
            return empty_logits, zero, stats

        is_real_BxT = infer_is_real_tokens(idx_BxT, self.config.pad_token_id)
        validate_left_padded_tokens(
            is_real_BxT,
            allow_all_pad=True,
            context="forward_efficient inputs",
        )
        max_real_tokens = int(is_real_BxT.sum(dim=1).max().item()) if t > 0 else 0
        if max_real_tokens > self.freqs_cis.shape[0]:
            raise ValueError(
                f"Cannot forward {max_real_tokens} real tokens, block size is only {self.freqs_cis.shape[0]}"
            )
        if progress and (progress_every is None or progress_every <= 0):
            progress_every = max(1, t // 20)

        attn_dtype = torch.bfloat16 if device.type == "cuda" else self.transformer.wte.weight.dtype
        freqs_all = self.freqs_cis.to(device)
        state = self._build_decode_state(
            batch_size=b,
            max_tokens=max_real_tokens,
            device=device,
            attn_dtype=attn_dtype,
        )

        collect_token_logits = bool(store_token_logits) or (targets_BxT is None)
        token_logits_BxTxV = None
        if collect_token_logits:
            token_logits_BxTxV = self.lm_head.weight.new_zeros((b, t, self.config.vocab_size))

        if targets_BxT is not None:
            token_nll_sum = torch.zeros((), device=device, dtype=torch.float32)
            token_count = torch.zeros((), device=device, dtype=torch.long)

        for step in range(t):
            if progress:
                done = step + 1
                if done == 1 or done == t or (done % progress_every == 0):
                    pct = 100.0 * done / max(t, 1)
                    print(f"[{progress_prefix}] {done}/{t} ({pct:.1f}%)", flush=True)

            logits_step_BxV = self._decode_one_token_step(
                state,
                idx_BxT[:, step],
                is_real_BxT[:, step],
                freqs_all,
                check_finite=check_finite,
            )
            if collect_token_logits:
                token_logits_BxTxV[:, step, :] = logits_step_BxV

            if targets_BxT is not None:
                target_step_B = torch.where(
                    is_real_BxT[:, step],
                    targets_BxT[:, step],
                    torch.full_like(targets_BxT[:, step], IGNORE_INDEX),
                )
                target_step_B = torch.where(
                    idx_BxT[:, step] == self.config.eos_token_id,
                    torch.full_like(target_step_B, IGNORE_INDEX),
                    target_step_B,
                )
                token_nll_sum = token_nll_sum + F.cross_entropy(
                    logits_step_BxV,
                    target_step_B,
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                ).float()
                token_count = token_count + (target_step_B != IGNORE_INDEX).sum()

            self._advance_decode_state(state, idx_BxT[:, step], is_real_BxT[:, step])

        if not collect_token_logits:
            token_logits_BxTxV = self.lm_head.weight.new_empty((b, 0, self.config.vocab_size))

        if targets_BxT is None:
            return token_logits_BxTxV

        documents_idx_BxT = self.generate_document_idx(idx_BxT)
        loss, stats = build_reverse_sps_loss_and_stats(
            token_nll_sum=token_nll_sum,
            token_count=token_count,
            documents_idx_BxT=documents_idx_BxT,
            is_real_BxT=is_real_BxT,
        )
        return token_logits_BxTxV, loss, stats

    def _sample_next_token(
        self,
        logits_BxV: Tensor,
        active_mask_B: Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        forbidden_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        return sample_next_token(
            logits_BxV,
            active_mask_B,
            pad_token_id=self.config.pad_token_id,
            suppressed_token_ids=(self.config.predict_token_id, self.config.pad_token_id),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            forbidden_token_ids=forbidden_token_ids,
        )

    @torch.no_grad()
    def generate(
        self,
        idx_BxT: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        stop_on_eos: bool = True,
        forbidden_token_ids: Optional[Tensor] = None,
    ) -> Tensor:
        def prefill(prompt_BxT: Tensor, new_tokens: int):
            return self._prefill_generation_state(
                prompt_BxT,
                new_tokens,
                require_batched=True,
            )

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
            context=self._decode_error_name,
        )
