from __future__ import annotations

"""Autoregressive evaluation and generation utilities for ReverseSPS-P models."""

from modeling.models.reverse_sps.inference import ReverseSPSInferenceMixin


class SPSInferenceMixin(ReverseSPSInferenceMixin):
    _prediction_slot_index = 1
    _window_slot_index = 1
    _retained_slot_index = 0
    _decode_error_name = "SPS"
