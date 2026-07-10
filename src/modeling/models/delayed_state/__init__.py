from __future__ import annotations

from . import core, inference
from .core import (
    IGNORE_INDEX,
    ModelConfig,
    DelayedStateConfig,
    DelayedStateModelBase,
    build_reverse_sps_loss_and_stats,
    triton_reverse_sps_sliding_attention,
)
from .inference import DelayedStateInferenceMixin

class DelayedStateModel(DelayedStateInferenceMixin, DelayedStateModelBase):
    pass


__all__ = [
    "core",
    "inference",
    "IGNORE_INDEX",
    "ModelConfig",
    "DelayedStateConfig",
    "DelayedStateInferenceMixin",
    "DelayedStateModel",
    "DelayedStateModelBase",
    "build_reverse_sps_loss_and_stats",
    "triton_reverse_sps_sliding_attention",
]
