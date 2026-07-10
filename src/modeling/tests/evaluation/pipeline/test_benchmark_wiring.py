from __future__ import annotations

import base64
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import scripts.evaluate as evaluate_mod


class _FakeEncoding:
    def encode(self, text: str, allowed_special=None):
        del allowed_special
        return [1 for _ in text]

    def decode(self, tokens):
        return "".join("a" for _ in tokens)


class _DummyCheckpointManager:
    def __init__(self, out_dir: str, save_every: int, master_process: bool):
        self.out_dir = out_dir
        self.save_every = save_every
        self.master_process = master_process

    def checkpoint_exists(self) -> bool:
        return True

    def get_latest_checkpoint_path(self) -> str:
        return os.path.join(self.out_dir, "ckpt_tokens_1.pt")

    def load_checkpoint(self, device: str, checkpoint_path: str | None = None):
        del device, checkpoint_path
        return {
            "model_args": {},
            "model": {},
            "config": {},
        }


class _DummyEvalModel(torch.nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config


class _FakeAdapter:
    def __init__(self):
        self.reset_calls = 0

    def reset_eval_run_state(self) -> None:
        self.reset_calls += 1


class _FakeWandbRun:
    def __init__(self, *, fail_finish: bool = False):
        self.summary: dict[str, object] = {}
        self.logged: list[tuple[dict[str, float], int | None]] = []
        self.finish_calls: list[int | None] = []
        self.fail_finish = fail_finish

    def log(self, payload: dict[str, float], step: int | None = None) -> None:
        self.logged.append((dict(payload), step))

    def finish(self, exit_code: int | None = None) -> None:
        self.finish_calls.append(exit_code)
        if self.fail_finish:
            raise RuntimeError("finish boom")


class _FakeWandbModule:
    def __init__(self, *, fail_first_init: bool = False, fail_finish: bool = False, events: list[str] | None = None):
        self.fail_first_init = fail_first_init
        self.fail_finish = fail_finish
        self.events = events if events is not None else []
        self.init_calls = 0
        self.init_kwargs: list[dict[str, object]] = []
        self.seen_service_values: list[str | None] = []
        self.teardown_calls: list[int | None] = []
        self.runs: list[_FakeWandbRun] = []

    def init(self, **kwargs):
        self.init_calls += 1
        self.init_kwargs.append(dict(kwargs))
        self.seen_service_values.append(os.environ.get("WANDB_SERVICE"))
        self.events.append(f"wandb.init.{self.init_calls}")
        if self.fail_first_init and self.init_calls == 1:
            raise RuntimeError("Failed to connect to service on socket /tmp/fake-wandb/socket")
        run = _FakeWandbRun(fail_finish=self.fail_finish)
        self.runs.append(run)
        return run

    def teardown(self, exit_code: int | None = None) -> None:
        self.teardown_calls.append(exit_code)
        self.events.append(f"wandb.teardown.{exit_code}")


def _make_cfg(tmp_path, *, tasks: str, output_path: str | None, use_forward_efficient: bool):
    return OmegaConf.create(
        {
            "out_dir": str(tmp_path / "out"),
            "checkpoint_path": str(tmp_path / "ckpt.pt"),
            "checkpoint": None,
            "tasks": tasks,
            "num_fewshot": 2,
            "limit": 7,
            "eval_batch_size": 4,
            "max_batch_size": 32,
            "min_eval_seq_len": None,
            "output_path": output_path,
            "eval_profile": None,
            "task_metadata_b64": None,
            "required_block_size": None,
            "eval_seed": 123,
            "use_forward_efficient": use_forward_efficient,
            "apply_checkpoint_model_args": False,
            "system": {"device": "cpu", "dtype": "float32"},
            "training": {"save_every": 10},
            "logging": {
                "wandb_log": False,
                "wandb_run_name": "unit-test-run",
                "wandb_project": "unit-test-project",
                "wandb_entity": "unit-test-entity",
            },
            "model": {
                "config": {
                    "block_size": 16,
                    "vocab_size": 8,
                    "pad_token_id": 99,
                    "eos_token_id": 0,
                }
            },
        }
    )


def test_apply_eval_checkpoint_model_args_skips_obsolete_keys(capsys):
    model_config = OmegaConf.create(
        {
            "block_size": 16,
            "vocab_size": 8,
            "pad_token_id": 7,
        }
    )

    evaluate_mod._apply_eval_checkpoint_model_args(
        model_config,
        {
            "block_size": 32,
            "efficiency_aggregation": "global_margin",
            "use_continuous_train_path": True,
        },
        checkpoint_name="ckpt_tokens_123_final",
        model_target="modeling.models.sps.SPSModel",
    )

    assert OmegaConf.to_container(model_config, resolve=True) == {
        "block_size": 32,
        "vocab_size": 8,
        "pad_token_id": 7,
    }
    captured = capsys.readouterr()
    assert "skipping obsolete checkpoint model args during eval" in captured.out
    assert "efficiency_aggregation" in captured.out
    assert "use_continuous_train_path" in captured.out


def test_evaluate_main_passes_adapter_flags(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    captured = {}
    fake_adapter = _FakeAdapter()

    def _fake_instantiate(cfg_model):
        config = SimpleNamespace(**OmegaConf.to_container(cfg_model.config, resolve=True))
        return _DummyEvalModel(config)

    def _fake_create_hflm_eval_model(**kwargs):
        captured["adapter_kwargs"] = kwargs
        return fake_adapter

    def _fake_simple_evaluate(
        *,
        model,
        tasks,
        num_fewshot,
        batch_size,
        device,
        max_batch_size,
        limit,
        model_args=None,
        metadata=None,
    ):
        captured["simple_evaluate"] = {
            "model": model,
            "tasks": tasks,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "device": device,
            "max_batch_size": max_batch_size,
            "limit": limit,
            "model_args": model_args,
            "metadata": metadata,
        }
        return {
            "results": {
                tasks[0]: {"acc,none": 0.5},
            },
            "configs": {tasks[0]: {"dataset_name": tasks[0]}},
            "versions": {tasks[0]: 1},
        }

    monkeypatch.setattr(evaluate_mod, "CheckpointManager", _DummyCheckpointManager)
    monkeypatch.setattr(evaluate_mod, "instantiate", _fake_instantiate)
    monkeypatch.setattr(evaluate_mod, "create_hflm_eval_model", _fake_create_hflm_eval_model)
    monkeypatch.setattr(evaluate_mod, "simple_evaluate", _fake_simple_evaluate)
    monkeypatch.setattr(evaluate_mod, "tiktoken", SimpleNamespace(get_encoding=lambda _: _FakeEncoding()))

    output_path = tmp_path / "result.json"
    cfg = _make_cfg(
        tmp_path,
        tasks="hellaswag",
        output_path=str(output_path),
        use_forward_efficient=False,
    )

    evaluate_mod.main.__wrapped__(cfg)

    assert captured["adapter_kwargs"]["use_forward_efficient"] is False
    assert captured["adapter_kwargs"]["config"].pad_token_id == 7
    assert captured["simple_evaluate"]["model"] is fake_adapter
    assert captured["simple_evaluate"]["tasks"] == ["hellaswag"]
    assert captured["simple_evaluate"]["batch_size"] == 4
    assert captured["simple_evaluate"]["model_args"] == {"pretrained": "gpt2"}
    assert captured["simple_evaluate"]["metadata"] is None
    assert fake_adapter.reset_calls == 1

    payload = json.loads(output_path.read_text())
    assert payload["results"]["hellaswag"]["acc,none"] == 0.5


def test_local_lm_eval_task_manager_uses_repo_task_dir(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeTaskManager:
        def __init__(self, *, include_path, include_defaults, metadata):
            captured["include_path"] = include_path
            captured["include_defaults"] = include_defaults
            captured["metadata"] = metadata

    monkeypatch.setattr(evaluate_mod, "TaskManager", _FakeTaskManager)
    for task_name in ("gov_report_nll", "pile_books3"):
        task_manager = evaluate_mod._task_manager_for_task(task_name, {"max_seq_lengths": [4096]})

        assert task_manager is not None
        assert captured["include_path"].endswith("lm_eval_tasks")
        assert captured["include_defaults"] is False
        assert os.path.exists(os.path.join(captured["include_path"], f"{task_name}.yaml"))
        assert captured["metadata"] == {"max_seq_lengths": [4096]}

    with open(os.path.join(captured["include_path"], "pile_books3.yaml")) as f:
        assert "lighteval/pile@refs/convert/parquet/pile_books3/pile-test.parquet" in f.read()
    assert evaluate_mod._task_manager_for_task("wikitext", None) is None


def test_evaluate_main_filters_legacy_checkpoint_model_args_for_eval(tmp_path, monkeypatch, capsys):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    fake_adapter = _FakeAdapter()
    captured: dict[str, object] = {}

    class _LegacyCheckpointManager(_DummyCheckpointManager):
        def load_checkpoint(self, device: str, checkpoint_path: str | None = None):
            del device, checkpoint_path
            return {
                "model_args": {
                    "block_size": 24,
                    "pad_token_id": 6,
                    "efficiency_aggregation": "global_margin",
                    "use_continuous_train_path": True,
                },
                "model": {},
                "config": {},
            }

    def _fake_instantiate(cfg_model):
        config_dict = OmegaConf.to_container(cfg_model.config, resolve=True)
        captured["config"] = config_dict
        assert "efficiency_aggregation" not in config_dict
        assert "use_continuous_train_path" not in config_dict
        return _DummyEvalModel(SimpleNamespace(**config_dict))

    def _fake_create_hflm_eval_model(**kwargs):
        return fake_adapter

    def _fake_simple_evaluate(
        *,
        model,
        tasks,
        num_fewshot,
        batch_size,
        device,
        max_batch_size,
        limit,
        model_args=None,
        metadata=None,
    ):
        del model, num_fewshot, batch_size, device, max_batch_size, limit, metadata
        assert model_args == {"pretrained": "gpt2"}
        return {
            "results": {tasks[0]: {"acc,none": 0.25}},
            "configs": {tasks[0]: {"dataset_name": tasks[0]}},
            "versions": {tasks[0]: 1},
        }

    monkeypatch.setattr(evaluate_mod, "CheckpointManager", _LegacyCheckpointManager)
    monkeypatch.setattr(evaluate_mod, "instantiate", _fake_instantiate)
    monkeypatch.setattr(evaluate_mod, "create_hflm_eval_model", _fake_create_hflm_eval_model)
    monkeypatch.setattr(evaluate_mod, "simple_evaluate", _fake_simple_evaluate)
    monkeypatch.setattr(evaluate_mod, "tiktoken", SimpleNamespace(get_encoding=lambda _: _FakeEncoding()))

    output_path = tmp_path / "result.json"
    cfg = _make_cfg(
        tmp_path,
        tasks="hellaswag",
        output_path=str(output_path),
        use_forward_efficient=False,
    )
    cfg.apply_checkpoint_model_args = True

    evaluate_mod.main.__wrapped__(cfg)

    assert captured["config"] == {
        "block_size": 24,
        "vocab_size": 8,
        "pad_token_id": 6,
        "eos_token_id": 0,
    }
    captured_output = capsys.readouterr()
    assert "efficiency_aggregation" in captured_output.out
    assert "use_continuous_train_path" in captured_output.out
    payload = json.loads(output_path.read_text())
    assert payload["results"]["hellaswag"]["acc,none"] == 0.25


def test_evaluate_main_rejects_multiple_tasks_before_checkpoint_work(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    class _ShouldNotConstructCheckpointManager:
        def __init__(self, *args, **kwargs):
            raise AssertionError("checkpoint manager should not be constructed")

    monkeypatch.setattr(evaluate_mod, "CheckpointManager", _ShouldNotConstructCheckpointManager)

    cfg = _make_cfg(
        tmp_path,
        tasks="hellaswag,arc_easy",
        output_path=None,
        use_forward_efficient=True,
    )

    with pytest.raises(ValueError, match="exactly one task"):
        evaluate_mod.main.__wrapped__(cfg)


def test_evaluate_main_requires_tasks_to_be_passed(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    class _ShouldNotConstructCheckpointManager:
        def __init__(self, *args, **kwargs):
            raise AssertionError("checkpoint manager should not be constructed")

    monkeypatch.setattr(evaluate_mod, "CheckpointManager", _ShouldNotConstructCheckpointManager)

    cfg = _make_cfg(
        tmp_path,
        tasks="hellaswag",
        output_path=None,
        use_forward_efficient=True,
    )
    del cfg["tasks"]

    with pytest.raises(ValueError, match="requires tasks to be passed explicitly"):
        evaluate_mod.main.__wrapped__(cfg)


def test_evaluate_main_resolves_checkpoint_stem_to_pt_file(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    checkpoint_file = out_dir / "ckpt_tokens_123_final.pt"
    checkpoint_file.write_bytes(b"checkpoint")
    captured: dict[str, object] = {}
    fake_adapter = _FakeAdapter()

    class _CapturingCheckpointManager(_DummyCheckpointManager):
        def load_checkpoint(self, device: str, checkpoint_path: str | None = None):
            captured["checkpoint_path"] = checkpoint_path
            return super().load_checkpoint(device, checkpoint_path)

    def _fake_instantiate(cfg_model):
        config = SimpleNamespace(**OmegaConf.to_container(cfg_model.config, resolve=True))
        return _DummyEvalModel(config)

    def _fake_create_hflm_eval_model(**kwargs):
        return fake_adapter

    def _fake_simple_evaluate(
        *,
        model,
        tasks,
        num_fewshot,
        batch_size,
        device,
        max_batch_size,
        limit,
        model_args=None,
        metadata=None,
    ):
        del model, num_fewshot, batch_size, device, max_batch_size, limit, metadata
        assert model_args == {"pretrained": "gpt2"}
        return {
            "results": {tasks[0]: {"acc,none": 0.75}},
            "configs": {tasks[0]: {"dataset_name": tasks[0]}},
            "versions": {tasks[0]: 1},
        }

    monkeypatch.setattr(evaluate_mod, "CheckpointManager", _CapturingCheckpointManager)
    monkeypatch.setattr(evaluate_mod, "instantiate", _fake_instantiate)
    monkeypatch.setattr(evaluate_mod, "create_hflm_eval_model", _fake_create_hflm_eval_model)
    monkeypatch.setattr(evaluate_mod, "simple_evaluate", _fake_simple_evaluate)
    monkeypatch.setattr(evaluate_mod, "tiktoken", SimpleNamespace(get_encoding=lambda _: _FakeEncoding()))

    cfg = _make_cfg(
        tmp_path,
        tasks="hellaswag",
        output_path=str(tmp_path / "result.json"),
        use_forward_efficient=False,
    )
    cfg.out_dir = str(out_dir)
    cfg.checkpoint_path = None
    cfg.checkpoint = "ckpt_tokens_123_final"

    evaluate_mod.main.__wrapped__(cfg)

    assert captured["checkpoint_path"] == str(checkpoint_file)


def test_evaluate_main_initializes_wandb_before_checkpoint_load_and_retries_socket_once(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    events: list[str] = []
    fake_adapter = _FakeAdapter()
    fake_wandb = _FakeWandbModule(fail_first_init=True, events=events)

    class _OrderedCheckpointManager(_DummyCheckpointManager):
        def load_checkpoint(self, device: str, checkpoint_path: str | None = None):
            events.append("load_checkpoint")
            return super().load_checkpoint(device, checkpoint_path)

    def _fake_instantiate(cfg_model):
        events.append("instantiate")
        config = SimpleNamespace(**OmegaConf.to_container(cfg_model.config, resolve=True))
        return _DummyEvalModel(config)

    def _fake_create_hflm_eval_model(**kwargs):
        return fake_adapter

    def _fake_simple_evaluate(
        *,
        model,
        tasks,
        num_fewshot,
        batch_size,
        device,
        max_batch_size,
        limit,
        model_args=None,
        metadata=None,
    ):
        del model, num_fewshot, batch_size, device, max_batch_size, limit, metadata
        assert model_args == {"pretrained": "gpt2"}
        events.append("simple_evaluate")
        return {
            "results": {tasks[0]: {"acc,none": 0.75}},
            "configs": {tasks[0]: {"dataset_name": tasks[0]}},
            "versions": {tasks[0]: 1},
        }

    monkeypatch.setattr(evaluate_mod, "CheckpointManager", _OrderedCheckpointManager)
    monkeypatch.setattr(evaluate_mod, "instantiate", _fake_instantiate)
    monkeypatch.setattr(evaluate_mod, "create_hflm_eval_model", _fake_create_hflm_eval_model)
    monkeypatch.setattr(evaluate_mod, "simple_evaluate", _fake_simple_evaluate)
    monkeypatch.setattr(evaluate_mod, "tiktoken", SimpleNamespace(get_encoding=lambda _: _FakeEncoding()))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_SERVICE", "stale-service-token")

    cfg = _make_cfg(
        tmp_path,
        tasks="hellaswag",
        output_path=str(tmp_path / "result.json"),
        use_forward_efficient=True,
    )
    cfg.logging.wandb_log = True

    evaluate_mod.main.__wrapped__(cfg)

    assert fake_adapter.reset_calls == 1
    assert events[:5] == [
        "wandb.init.1",
        "wandb.teardown.1",
        "wandb.init.2",
        "load_checkpoint",
        "instantiate",
    ]
    assert "simple_evaluate" in events
    assert fake_wandb.seen_service_values == ["stale-service-token", None]
    assert "id" not in fake_wandb.init_kwargs[-1]
    assert "resume" not in fake_wandb.init_kwargs[-1]
    assert fake_wandb.runs[0].summary["status"] == "completed"
    payload = json.loads((tmp_path / "result.json").read_text())
    assert payload["results"]["hellaswag"]["acc,none"] == 0.75


def test_evaluate_main_wandb_finish_failure_removes_output_and_raises(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    fake_adapter = _FakeAdapter()
    fake_wandb = _FakeWandbModule(fail_finish=True)

    def _fake_instantiate(cfg_model):
        config = SimpleNamespace(**OmegaConf.to_container(cfg_model.config, resolve=True))
        return _DummyEvalModel(config)

    def _fake_create_hflm_eval_model(**kwargs):
        return fake_adapter

    def _fake_simple_evaluate(
        *,
        model,
        tasks,
        num_fewshot,
        batch_size,
        device,
        max_batch_size,
        limit,
        model_args=None,
        metadata=None,
    ):
        del model, num_fewshot, batch_size, device, max_batch_size, limit, metadata
        assert model_args == {"pretrained": "gpt2"}
        return {
            "results": {tasks[0]: {"acc,none": 0.75}},
            "configs": {tasks[0]: {"dataset_name": tasks[0]}},
            "versions": {tasks[0]: 1},
        }

    monkeypatch.setattr(evaluate_mod, "CheckpointManager", _DummyCheckpointManager)
    monkeypatch.setattr(evaluate_mod, "instantiate", _fake_instantiate)
    monkeypatch.setattr(evaluate_mod, "create_hflm_eval_model", _fake_create_hflm_eval_model)
    monkeypatch.setattr(evaluate_mod, "simple_evaluate", _fake_simple_evaluate)
    monkeypatch.setattr(evaluate_mod, "tiktoken", SimpleNamespace(get_encoding=lambda _: _FakeEncoding()))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    output_path = tmp_path / "result.json"
    cfg = _make_cfg(
        tmp_path,
        tasks="hellaswag",
        output_path=str(output_path),
        use_forward_efficient=True,
    )
    cfg.logging.wandb_log = True

    with pytest.raises(RuntimeError, match="finish boom"):
        evaluate_mod.main.__wrapped__(cfg)

    assert not output_path.exists()


def test_evaluate_main_forwards_task_metadata_and_uses_profile_output_dir(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    captured: dict[str, object] = {}
    fake_adapter = _FakeAdapter()

    def _fake_instantiate(cfg_model):
        config = SimpleNamespace(**OmegaConf.to_container(cfg_model.config, resolve=True))
        return _DummyEvalModel(config)

    def _fake_create_hflm_eval_model(**kwargs):
        return fake_adapter

    def _fake_simple_evaluate(
        *,
        model,
        tasks,
        num_fewshot,
        batch_size,
        device,
        max_batch_size,
        limit,
        model_args=None,
        metadata=None,
    ):
        del model, num_fewshot, batch_size, device, max_batch_size, limit
        captured["model_args"] = model_args
        captured["metadata"] = metadata
        return {
            "results": {tasks[0]: {"acc,none": 0.42}},
            "configs": {tasks[0]: {"dataset_name": tasks[0]}},
            "versions": {tasks[0]: 1},
        }

    monkeypatch.setattr(evaluate_mod, "CheckpointManager", _DummyCheckpointManager)
    monkeypatch.setattr(evaluate_mod, "instantiate", _fake_instantiate)
    monkeypatch.setattr(evaluate_mod, "create_hflm_eval_model", _fake_create_hflm_eval_model)
    monkeypatch.setattr(evaluate_mod, "simple_evaluate", _fake_simple_evaluate)
    monkeypatch.setattr(evaluate_mod, "tiktoken", SimpleNamespace(get_encoding=lambda _: _FakeEncoding()))

    cfg = _make_cfg(
        tmp_path,
        tasks="hellaswag",
        output_path=None,
        use_forward_efficient=True,
    )
    cfg.eval_profile = "long_context"
    cfg.task_metadata_b64 = base64.urlsafe_b64encode(
        json.dumps({"pretrained": "gpt2", "max_seq_lengths": [4096]}).encode("utf-8")
    ).decode("ascii")

    evaluate_mod.main.__wrapped__(cfg)

    assert captured["model_args"] == {"pretrained": "gpt2"}
    assert captured["metadata"] == {"pretrained": "gpt2", "max_seq_lengths": [4096]}
    assert (
        tmp_path
        / "out"
        / "eval"
        / "ckpt"
        / "profile_long_context"
        / "fewshot_2"
        / "hellaswag.json"
    ).exists()


def test_evaluate_main_rejects_profile_when_block_size_is_too_small(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "ckpt.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    fake_adapter = _FakeAdapter()

    def _fake_instantiate(cfg_model):
        config = SimpleNamespace(**OmegaConf.to_container(cfg_model.config, resolve=True))
        return _DummyEvalModel(config)

    def _fake_create_hflm_eval_model(**kwargs):
        return fake_adapter

    def _should_not_run_simple_evaluate(**kwargs):
        raise AssertionError("simple_evaluate should not run when block size is too small")

    monkeypatch.setattr(evaluate_mod, "CheckpointManager", _DummyCheckpointManager)
    monkeypatch.setattr(evaluate_mod, "instantiate", _fake_instantiate)
    monkeypatch.setattr(evaluate_mod, "create_hflm_eval_model", _fake_create_hflm_eval_model)
    monkeypatch.setattr(evaluate_mod, "simple_evaluate", _should_not_run_simple_evaluate)
    monkeypatch.setattr(evaluate_mod, "tiktoken", SimpleNamespace(get_encoding=lambda _: _FakeEncoding()))

    cfg = _make_cfg(
        tmp_path,
        tasks="hellaswag",
        output_path=str(tmp_path / "result.json"),
        use_forward_efficient=True,
    )
    cfg.eval_profile = "long_context"
    cfg.required_block_size = 4096

    with pytest.raises(ValueError, match="requires block_size >= 4096"):
        evaluate_mod.main.__wrapped__(cfg)
