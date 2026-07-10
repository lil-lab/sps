from __future__ import annotations

"""Autoregressive evaluation and generation utilities for delayed-state models."""

from modeling.models.reverse_sps.inference import ReverseSPSInferenceMixin


class DelayedStateInferenceMixin(ReverseSPSInferenceMixin):
    _prediction_slot_index = 1
    _decode_error_name = "Delayed State"
