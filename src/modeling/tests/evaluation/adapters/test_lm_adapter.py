from dataclasses import dataclass

import pytest
import torch
from torch import nn

from modeling.evaluation.lm_eval_adapter import create_hflm_eval_model


@dataclass
class _DummyConfig:
    block_size: int = 16
    vocab_size: int = 8
    pad_token_id: int = 7
    eos_token_id: int = 0


class _DummyTupleModel(nn.Module):
    def __init__(self, config: _DummyConfig):
        super().__init__()
        self.config = config
        self.output_vocab_size = config.vocab_size + 2
        self.proj = nn.Linear(1, 1)
        self.seen_inputs = []
        self.seen_forward_targets = []
        self.seen_forward_efficient_targets = []
        self.seen_generate_inputs = []
        self.seen_generate_kwargs = []
        self.forward_calls = 0
        self.forward_efficient_calls = 0

    def forward(self, idx_BxT: torch.Tensor, targets_BxT: torch.Tensor | None = None):
        self.forward_calls += 1
        self.seen_inputs.append(idx_BxT.detach().clone())
        batch, seq_len = idx_BxT.shape
        logits = torch.full(
            (batch, seq_len, self.output_vocab_size),
            -50.0,
            device=idx_BxT.device,
        )
        next_tokens = (idx_BxT + 1) % self.config.vocab_size
        logits.scatter_(2, next_tokens.unsqueeze(-1), 50.0)
        if targets_BxT is None:
            return logits
        self.seen_forward_targets.append(targets_BxT.detach().clone())
        return logits, logits.new_tensor(0.0), {"dummy": logits.new_tensor(1.0)}

    def forward_efficient(
        self,
        idx_BxT: torch.Tensor,
        targets_BxT: torch.Tensor | None = None,
    ):
        self.forward_efficient_calls += 1
        self.seen_inputs.append(idx_BxT.detach().clone())
        batch, seq_len = idx_BxT.shape
        logits = torch.full(
            (batch, seq_len, self.output_vocab_size),
            -50.0,
            device=idx_BxT.device,
        )
        next_tokens = (idx_BxT + 1) % self.config.vocab_size
        logits.scatter_(2, next_tokens.unsqueeze(-1), 50.0)
        if targets_BxT is None:
            return logits
        self.seen_forward_efficient_targets.append(targets_BxT.detach().clone())

        return logits, logits.new_tensor(0.0), {}

    def generate(
        self,
        idx_BxT: torch.Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_on_eos: bool = True,
        forbidden_token_ids: torch.Tensor | None = None,
    ):
        del do_sample, temperature, top_k, top_p, stop_on_eos, forbidden_token_ids
        self.seen_generate_inputs.append(idx_BxT.detach().clone())
        self.seen_generate_kwargs.append({"max_new_tokens": max_new_tokens})
        generated = idx_BxT.clone()
        suffix = torch.full(
            (idx_BxT.size(0), max_new_tokens),
            self.config.pad_token_id,
            dtype=idx_BxT.dtype,
            device=idx_BxT.device,
        )
        current = idx_BxT.clone()
        for step in range(max_new_tokens):
            next_token = (current[:, -1] + 1) % self.config.vocab_size
            suffix[:, step] = next_token
            current = torch.cat([current, next_token.unsqueeze(1)], dim=1)
        return torch.cat([generated, suffix], dim=1)


class _DenseOnlyTupleModel(nn.Module):
    def __init__(self, config: _DummyConfig):
        super().__init__()
        self.config = config
        self.output_vocab_size = config.vocab_size + 2
        self.proj = nn.Linear(1, 1)
        self.seen_forward_targets = []
        self.forward_calls = 0

    def forward(self, idx_BxT: torch.Tensor, targets_BxT: torch.Tensor | None = None):
        self.forward_calls += 1
        batch, seq_len = idx_BxT.shape
        logits = torch.full(
            (batch, seq_len, self.output_vocab_size),
            -50.0,
            device=idx_BxT.device,
        )
        next_tokens = (idx_BxT + 1) % self.config.vocab_size
        logits.scatter_(2, next_tokens.unsqueeze(-1), 50.0)
        if targets_BxT is None:
            return logits
        self.seen_forward_targets.append(targets_BxT.detach().clone())
        return logits, logits.new_tensor(0.0), {"dummy": logits.new_tensor(1.0)}


class _PadBiasedGenerationModel(nn.Module):
    def __init__(self, config: _DummyConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(1, 1)

    def forward_efficient(
        self,
        idx_BxT: torch.Tensor,
        targets_BxT: torch.Tensor | None = None,
    ):
        batch, seq_len = idx_BxT.shape
        logits = torch.full(
            (batch, seq_len, self.config.vocab_size),
            -20.0,
            device=idx_BxT.device,
        )
        logits[:, :, self.config.pad_token_id] = 50.0
        logits[:, :, 1] = 10.0
        if targets_BxT is None:
            return logits
        return logits, logits.new_tensor(0.0), {}

    def generate(
        self,
        idx_BxT: torch.Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_on_eos: bool = True,
        forbidden_token_ids: torch.Tensor | None = None,
    ):
        del do_sample, temperature, top_k, top_p, stop_on_eos
        chosen_token = self.config.pad_token_id
        if forbidden_token_ids is not None and int(self.config.pad_token_id) in forbidden_token_ids.tolist():
            chosen_token = 1
        suffix = torch.full(
            (idx_BxT.size(0), max_new_tokens),
            chosen_token,
            dtype=idx_BxT.dtype,
            device=idx_BxT.device,
        )
        return torch.cat([idx_BxT, suffix], dim=1)


class _GenerateOnlyModel(nn.Module):
    def __init__(self, config: _DummyConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(1, 1)
        self.forward_calls = 0
        self.generate_calls = 0
        self.seen_generate_kwargs = []

    def forward(self, idx_BxT: torch.Tensor, targets_BxT: torch.Tensor | None = None):
        del idx_BxT, targets_BxT
        self.forward_calls += 1
        raise AssertionError("adapter should not call forward() for generation")

    def generate(
        self,
        idx_BxT: torch.Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_on_eos: bool = True,
        forbidden_token_ids: torch.Tensor | None = None,
    ):
        self.generate_calls += 1
        self.seen_generate_kwargs.append(
            {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "stop_on_eos": stop_on_eos,
                "forbidden_token_ids": None if forbidden_token_ids is None else forbidden_token_ids.detach().cpu().tolist(),
            }
        )
        suffix = torch.tensor([[2, 3]], dtype=idx_BxT.dtype, device=idx_BxT.device)
        return torch.cat([idx_BxT, suffix[:, :max_new_tokens]], dim=1)


class _NoGenerateModel(nn.Module):
    def __init__(self, config: _DummyConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(1, 1)


class _Request:
    def __init__(self, *args):
        self.args = args


def _encode(text: str) -> list[int]:
    table = {"a": 1, "b": 2, "c": 3}
    return [table[ch] for ch in text]


def _decode(tokens) -> str:
    table = {0: "<eos>", 1: "a", 2: "b", 3: "c", 7: "<pad>"}
    return "".join(table.get(tok, "?") for tok in tokens)


def test_model_call_uses_first_forward_output_and_slices_vocab():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
    )

    logits = adapter._model_call(torch.tensor([[0, 1, 2]], dtype=torch.long))

    assert logits.shape == (1, 3, config.vocab_size)
    assert logits.argmax(dim=-1).tolist() == [[1, 2, 3]]


def test_model_call_builds_shifted_targets_for_forward_efficient():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
        use_forward_efficient=True,
    )

    adapter._model_call(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert len(model.seen_forward_efficient_targets) == 1
    assert model.seen_forward_targets == []
    assert torch.equal(
        model.seen_forward_efficient_targets[0],
        torch.tensor([[2, 3, 3]], dtype=torch.long),
    )


def test_model_call_builds_shifted_targets_for_dense_forward_when_efficient_disabled():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
        use_forward_efficient=False,
    )

    adapter._model_call(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert len(model.seen_forward_targets) == 1
    assert model.seen_forward_efficient_targets == []
    assert torch.equal(
        model.seen_forward_targets[0],
        torch.tensor([[2, 3, 3]], dtype=torch.long),
    )


def test_model_call_falls_back_to_dense_forward_when_efficient_method_absent():
    config = _DummyConfig()
    model = _DenseOnlyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
        use_forward_efficient=True,
    )

    adapter._model_call(torch.tensor([[1, 2, 3]], dtype=torch.long))

    assert model.forward_calls == 1
    assert torch.equal(
        model.seen_forward_targets[0],
        torch.tensor([[2, 3, 3]], dtype=torch.long),
    )


def test_loglikelihood_works_for_empty_context_requests():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
    )

    result = adapter.loglikelihood([_Request("", "a")])

    assert len(result) == 1
    logprob, is_greedy = result[0]
    assert is_greedy is True
    assert logprob > -1e-3


def test_tok_decode_skips_special_tokens_by_default():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
    )

    assert adapter.tok_decode([config.pad_token_id, 1, config.eos_token_id, 2]) == "ab"
    assert (
        adapter.tok_decode(
            [config.pad_token_id, 1, config.eos_token_id, 2],
            skip_special_tokens=False,
        )
        == "<pad>a<eos>b"
    )


def test_generate_until_runs_for_greedy_requests():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
    )

    result = adapter.generate_until(
        [
            _Request(
                "a",
                {"until": [], "max_gen_toks": 2, "do_sample": False, "temperature": 0.0},
            )
        ]
    )

    assert result == ["bc"]


def test_generate_until_calls_model_generate_only():
    config = _DummyConfig()
    model = _GenerateOnlyModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
        use_forward_efficient=False,
    )

    result = adapter.generate_until(
        [
            _Request(
                "a",
                {"until": [], "max_gen_toks": 2, "do_sample": False, "temperature": 0.0},
            )
        ]
    )

    assert result == ["bc"]
    assert model.generate_calls == 1
    assert model.forward_calls == 0
    assert model.seen_generate_kwargs[0]["max_new_tokens"] == 2
    assert model.seen_generate_kwargs[0]["do_sample"] is False
    assert model.seen_generate_kwargs[0]["top_k"] is None
    assert model.seen_generate_kwargs[0]["stop_on_eos"] is True
    assert config.pad_token_id in model.seen_generate_kwargs[0]["forbidden_token_ids"]


def test_generate_until_requires_model_generate():
    config = _DummyConfig()
    model = _NoGenerateModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
        use_forward_efficient=False,
    )

    with pytest.raises(AttributeError, match="must implement generate\\(\\)"):
        adapter.generate_until(
            [
                _Request(
                    "a",
                    {"until": [], "max_gen_toks": 2, "do_sample": False, "temperature": 0.0},
                )
            ]
        )


def test_generate_until_respects_stop_sequences():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
    )

    result = adapter.generate_until(
        [
            _Request(
                "a",
                {"until": ["c"], "max_gen_toks": 3, "do_sample": False, "temperature": 0.0},
            ),
            _Request(
                "b",
                {"until": ["c"], "max_gen_toks": 3, "do_sample": False, "temperature": 0.0},
            ),
        ]
    )

    assert result == ["b", ""]


def test_generate_until_masks_pad_token_during_sampling():
    config = _DummyConfig()
    model = _PadBiasedGenerationModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=2,
    )

    result = adapter.generate_until(
        [
            _Request(
                "a",
                {"until": [], "max_gen_toks": 2, "do_sample": False, "temperature": 0.0},
            )
        ]
    )

    assert result == ["aa"]


def test_loglikelihood_batches_use_configured_pad_token():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=8,
    )

    adapter.loglikelihood([
        _Request("ab", "c"),
        _Request("a", "b"),
    ])

    assert model.seen_inputs, "expected at least one forward call"
    batched_inputs = model.seen_inputs[0]
    assert batched_inputs.shape == (2, 2)
    assert batched_inputs[1, 0].item() == config.pad_token_id


def test_loglikelihood_mixed_length_batched_matches_unbatched_under_left_padding():
    config = _DummyConfig()
    batched_model = _DummyTupleModel(config)
    single_model = _DummyTupleModel(config)
    requests = [
        _Request("ab", "c"),
        _Request("a", "b"),
    ]

    batched_adapter = create_hflm_eval_model(
        model=batched_model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=8,
        use_forward_efficient=False,
    )
    single_adapter = create_hflm_eval_model(
        model=single_model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=1,
        use_forward_efficient=False,
    )

    batched_results = batched_adapter.loglikelihood(requests)
    single_results = [single_adapter.loglikelihood([request])[0] for request in requests]

    assert len(batched_results) == len(single_results)
    for (batched_logprob, batched_greedy), (single_logprob, single_greedy) in zip(
        batched_results,
        single_results,
        strict=True,
    ):
        assert batched_logprob == pytest.approx(single_logprob)
        assert batched_greedy is single_greedy


def test_pad_and_concat_forces_left_padding_even_if_lm_eval_requests_right_padding():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=8,
    )

    batched = adapter._pad_and_concat_with_pad_token(
        max_length=3,
        tensors=[torch.tensor([1, 2]), torch.tensor([3])],
        padding_side="right",
    )

    assert batched.tolist() == [[config.pad_token_id, 1, 2], [config.pad_token_id, config.pad_token_id, 3]]


def test_auto_batch_size_configuration_preserves_schedule_and_cap():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size="auto:2",
        max_batch_size=16,
    )

    assert adapter.batch_size == "auto"
    assert adapter.batch_schedule == pytest.approx(2.0)
    assert adapter.max_batch_size == 16


def test_min_eval_seq_len_pads_batches_to_minimum_length():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=8,
        min_eval_seq_len=4,
    )

    adapter.loglikelihood([
        _Request("ab", "c"),
        _Request("a", "b"),
    ])

    assert model.seen_inputs, "expected at least one forward call"
    assert model.seen_inputs[0].shape == (2, 4)


def test_min_eval_seq_len_applies_to_auto_batch_detection():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size="auto",
        max_batch_size=8,
        min_eval_seq_len=4,
    )

    detected = adapter._detect_batch_size([
        (("a", "b"), [1], [2]),
    ])

    assert detected >= 1
    assert model.seen_inputs, "expected auto batch-size detection to call forward"
    assert model.seen_inputs[0].shape[1] == 4


def test_disabling_forward_efficient_uses_forward():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size=8,
        use_forward_efficient=False,
    )

    adapter.loglikelihood([
        _Request("ab", "c"),
        _Request("a", "b"),
    ])

    assert model.forward_calls > 0
    assert model.forward_efficient_calls == 0


def test_reset_eval_run_state_clears_auto_batch_cache():
    config = _DummyConfig()
    model = _DummyTupleModel(config)
    adapter = create_hflm_eval_model(
        model=model,
        config=config,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        device="cpu",
        batch_size="auto:2",
        max_batch_size=16,
        use_forward_efficient=True,
    )

    adapter.loglikelihood([
        _Request("ab", "c"),
        _Request("a", "b"),
    ])
    adapter.batch_sizes[0] = 8

    assert adapter.batch_sizes

    adapter.reset_eval_run_state()

    assert adapter.batch_sizes == {}
