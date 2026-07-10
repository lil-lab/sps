from __future__ import annotations

import math

import pytest
import torch

from modeling.evaluation.lm_eval_adapter import create_hflm_eval_model
from modeling.tests._helpers import (
    make_reverse_sps_model,
    make_full_model,
    make_delayed_state_model,
)

_SMALL = dict(
    vocab_size=16, hidden_size=16, intermediate_size=48,
    eos_token_id=14, pad_token_id=15,
)

# Batched generation prefill runs a Triton kernel that only supports
# head_dim in {16, 32, 64, 128, 256}. The generate_until tests need a hidden_size
# whose head_dim (hidden_size // n_head) lands in that set, so they use
# head_dim=64 (hidden_size=128) rather than the tiny _SMALL default (head_dim=8).
_SMALL_GEN = {**_SMALL, "hidden_size": 128}


class _Request:
    def __init__(self, *args):
        self.args = args


def _encode(text: str) -> list[int]:
    table = {"a": 1, "b": 2, "c": 3}
    return [table[ch] for ch in text]


def _decode(tokens) -> str:
    table = {0: "<eos>", 1: "a", 2: "b", 3: "c", 14: "<eos14>", 15: "<pad>"}
    return "".join(table.get(int(tok), "?") for tok in tokens)


def test_real_full_attention_adapter_loglikelihood_runs_on_mixed_lengths():
    torch.manual_seed(0)
    model = make_full_model(**_SMALL)
    adapter = create_hflm_eval_model(
        model=model,
        config=model.config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=8,
        use_forward_efficient=False,
    )

    results = adapter.loglikelihood([
        _Request("ab", "c"),
        _Request("a", "b"),
    ])

    assert len(results) == 2
    assert all(math.isfinite(logprob) for logprob, _ in results)


def test_real_full_attention_adapter_matches_unbatched_on_mixed_lengths():
    torch.manual_seed(0)
    model = make_full_model(**_SMALL)
    requests = [
        _Request("ab", "c"),
        _Request("a", "b"),
    ]

    batched_adapter = create_hflm_eval_model(
        model=model,
        config=model.config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=8,
        use_forward_efficient=False,
    )
    single_adapter = create_hflm_eval_model(
        model=model,
        config=model.config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=1,
        use_forward_efficient=False,
    )

    batched_results = batched_adapter.loglikelihood(requests)
    single_results = [single_adapter.loglikelihood([request])[0] for request in requests]

    for (batched_logprob, batched_greedy), (single_logprob, single_greedy) in zip(
        batched_results,
        single_results,
        strict=True,
    ):
        assert batched_logprob == pytest.approx(single_logprob, abs=1e-5)
        assert batched_greedy is single_greedy


def test_real_full_attention_adapter_generate_until_runs():
    torch.manual_seed(0)
    model = make_full_model(**_SMALL)
    adapter = create_hflm_eval_model(
        model=model,
        config=model.config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
        use_forward_efficient=False,
    )

    results = adapter.generate_until(
        [
            _Request("ab", {"until": [], "max_gen_toks": 2, "do_sample": False, "temperature": 0.0}),
            _Request("a", {"until": [], "max_gen_toks": 2, "do_sample": False, "temperature": 0.0}),
        ]
    )

    assert len(results) == 2
    assert all(isinstance(text, str) for text in results)


def test_real_full_attention_adapter_generate_until_matches_unbatched():
    torch.manual_seed(0)
    model = make_full_model(**_SMALL)
    requests = [
        _Request("ab", {"until": [], "max_gen_toks": 3, "do_sample": False, "temperature": 0.0}),
        _Request("a", {"until": [], "max_gen_toks": 3, "do_sample": False, "temperature": 0.0}),
    ]

    batched_adapter = create_hflm_eval_model(
        model=model,
        config=model.config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
        use_forward_efficient=False,
    )
    single_adapter = create_hflm_eval_model(
        model=model,
        config=model.config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=1,
        use_forward_efficient=False,
    )

    batched_results = batched_adapter.generate_until(requests)
    single_results = [single_adapter.generate_until([request])[0] for request in requests]

    assert batched_results == single_results


@pytest.mark.cuda
def test_real_reverse_sps_adapter_loglikelihood_runs():
    torch.manual_seed(4)
    model = make_reverse_sps_model(**_SMALL, n_layer=1, predict_token_id=13)
    adapter = create_hflm_eval_model(
        model=model,
        config=model.config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cuda",
        batch_size=8,
        use_forward_efficient=True,
    )

    results = adapter.loglikelihood([
        _Request("ab", "c"),
        _Request("a", "b"),
    ])

    assert len(results) == 2
    assert all(math.isfinite(logprob) for logprob, _ in results)


@pytest.mark.cuda
def test_real_reverse_sps_adapter_generate_until_runs():
    torch.manual_seed(4)
    model = make_reverse_sps_model(**_SMALL_GEN, n_layer=1, predict_token_id=13)
    adapter = create_hflm_eval_model(
        model=model,
        config=model.config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cuda",
        batch_size=2,
        use_forward_efficient=True,
    )

    results = adapter.generate_until(
        [
            _Request("ab", {"until": [], "max_gen_toks": 2, "do_sample": False, "temperature": 0.0}),
            _Request("a", {"until": [], "max_gen_toks": 2, "do_sample": False, "temperature": 0.0}),
        ]
    )

    assert len(results) == 2
    assert all(isinstance(text, str) for text in results)


@pytest.mark.cuda
def test_real_delayed_state_adapter_loglikelihood_runs():
    torch.manual_seed(5)
    model = make_delayed_state_model(**_SMALL, n_layer=1, predict_token_id=13)
    adapter = create_hflm_eval_model(
        model=model,
        config=model.config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cuda",
        batch_size=8,
        use_forward_efficient=True,
    )

    results = adapter.loglikelihood([
        _Request("ab", "c"),
        _Request("a", "b"),
    ])

    assert len(results) == 2
    assert all(math.isfinite(logprob) for logprob, _ in results)


@pytest.mark.cuda
def test_real_delayed_state_adapter_generate_until_runs():
    torch.manual_seed(5)
    model = make_delayed_state_model(**_SMALL_GEN, n_layer=1, predict_token_id=13)
    adapter = create_hflm_eval_model(
        model=model,
        config=model.config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cuda",
        batch_size=2,
        use_forward_efficient=True,
    )

    results = adapter.generate_until(
        [
            _Request("ab", {"until": [], "max_gen_toks": 2, "do_sample": False, "temperature": 0.0}),
            _Request("a", {"until": [], "max_gen_toks": 2, "do_sample": False, "temperature": 0.0}),
        ]
    )

    assert len(results) == 2
    assert all(isinstance(text, str) for text in results)
