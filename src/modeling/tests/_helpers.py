"""Shared model factories and test utilities.

Each factory accepts **overrides for any config field, so tests can
customise vocab_size, hidden_size, etc. while sharing the boilerplate.
"""

from __future__ import annotations

import torch

from modeling.models.full_attention_model import Model, ModelConfig
from modeling.models.reverse_sps import ReverseSPSConfig, ReverseSPSModel
from modeling.models.sps import SPSConfig, SPSModel
from modeling.models.delayed_state import DelayedStateConfig, DelayedStateModel


_BASE_DEFAULTS = dict(
    block_size=16,
    vocab_size=32,
    n_layer=2,
    n_head=2,
    hidden_size=32,
    intermediate_size=96,
    dropout=0.0,
    bias=False,
    eos_token_id=30,
    pad_token_id=31,
)


def make_full_model(**overrides) -> Model:
    cfg = {**_BASE_DEFAULTS, **overrides}
    model = Model(ModelConfig(**cfg))
    model.eval()
    return model


def make_reverse_sps_model(**overrides) -> ReverseSPSModel:
    cfg = {
        **_BASE_DEFAULTS,
        "hidden_size": 128,
        "intermediate_size": 256,
        "window_size": 2,
        "predict_token_id": 29,
        **overrides,
    }
    model = ReverseSPSModel(ReverseSPSConfig(**cfg))
    model.eval()
    return model


def make_delayed_state_model(**overrides) -> DelayedStateModel:
    cfg = {
        **_BASE_DEFAULTS,
        "hidden_size": 128,
        "intermediate_size": 256,
        "window_size": 2,
        "predict_token_id": 29,
        **overrides,
    }
    model = DelayedStateModel(DelayedStateConfig(**cfg))
    model.eval()
    return model


def make_sps_model(**overrides) -> SPSModel:
    cfg = {
        **_BASE_DEFAULTS,
        "hidden_size": 128,
        "intermediate_size": 256,
        "window_size": 2,
        "predict_token_id": 29,
        **overrides,
    }
    model = SPSModel(SPSConfig(**cfg))
    model.eval()
    return model


def forward_logits(model, idx_BxT: torch.Tensor) -> torch.Tensor:
    """Run model forward and return only logits, handling both call signatures."""
    try:
        outputs = model(idx_BxT)
    except TypeError:
        outputs = model(idx_BxT, idx_BxT.clone())
    return outputs[0] if isinstance(outputs, tuple) else outputs
