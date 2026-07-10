from __future__ import annotations

from . import core, inference
from .core import (
    ReverseSPSBlock,
    ReverseSPSConfig,
    ReverseSPSFlashAttention,
    ReverseSPSModelBase,
    IGNORE_INDEX,
    ModelConfig,
    build_reverse_sps_loss_and_stats,
    triton_reverse_sps_sliding_attention,
)
from .inference import ReverseSPSInferenceMixin

class ReverseSPSModel(ReverseSPSInferenceMixin, ReverseSPSModelBase):
    pass


__all__ = [
    "core",
    "inference",
    "ReverseSPSBlock",
    "ReverseSPSConfig",
    "ReverseSPSFlashAttention",
    "ReverseSPSInferenceMixin",
    "ReverseSPSModel",
    "ReverseSPSModelBase",
    "IGNORE_INDEX",
    "ModelConfig",
    "build_reverse_sps_loss_and_stats",
    "triton_reverse_sps_sliding_attention",
]
