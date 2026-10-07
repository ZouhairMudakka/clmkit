"""Training loop tests on a tiny Qwen3 encoder (CPU, seconds)."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from clmkit.data import ContrastiveExample, save_examples  # noqa: E402
from clmkit.encoders.hf import HFEncoder  # noqa: E402
from clmkit.eval import RetrievalEvaluator  # noqa: E402
from clmkit.training import ContrastiveTrainer, LoraSettings, TrainConfig, train_from_config  # noqa: E402

pytestmark = pytest.mark.hf

QWEN_LIKE = {
    "pooling": "last_token",
    "padding_side": "left",
    "ensure_eos": True,
    "eos_token": "<|endoftext|>",
    "query_template": "Instruct: {instruction}\nQuery:{text}",
    "default_instruction": "find the document",
}


def _encoder(path):  # type: ignore[no-untyped-def]
    return HFEncoder(str(path), device="cpu", **QWEN_LIKE)


def test_training_improves_retrieval(tmp_path, tiny_model_dir, toy_examples) -> None:  # type: ignore[no-untyped-def]
    enc = _encoder(tiny_model_dir)
    evaluator = RetrievalEvaluator.from_examples(toy_examples, ks=[1, 5])
    before = evaluator(enc)
    cfg = TrainConfig(
        output_dir=str(tmp_path / "run"),
        epochs=25,
        batch_size=10,
        learning_rate=3e-3,
        warmup_ratio=0.1,
        log_every=5,
        eval_every=25,
        metric_for_best="mrr@5",
        loss_kwargs={"temperature": 0.05},
    )
    logs: list[dict] = []
    trainer = ContrastiveTrainer(enc, cfg, toy_examples, evaluator=evaluator, callbacks=[logs.append])
    result = trainer.train()
    after = evaluator(enc)
    assert result.global_step == 50
    losses = [r["loss"] for r in result.history if "loss" in r]
    assert losses[-1] < losses[0] * 0.5, losses
    assert after["mrr@5"] > before["mrr@5"] + 0.2, (before, after)
    assert any("eval/mrr@5" in r for r in logs)
    run = tmp_path / "run"
    assert (run / "final" / "clmkit_config.json").is_file() and (run / "best").is_dir()
    state = json.loads((run / "trainer_state.json").read_text())
    assert state["global_step"] == 50 and state["best_metric"] == result.best_metric
    # the saved final checkpoint reproduces the trained encoder
    reloaded = HFEncoder(str(run / "final"), device="cpu")
    np.testing.assert_allclose(reloaded.encode(["cats eat fish"]), enc.encode(["cats eat fish"]), atol=1e-5)


def _grads(enc, cfg, examples):  # type: ignore[no-untyped-def]
    trainer = ContrastiveTrainer(enc, cfg, examples)
    enc.model.train()
    loss = trainer.training_step(examples)
    return loss, {n: p.grad.clone() for n, p in enc.model.named_parameters() if p.grad is not None}


def test_gradcache_matches_full_batch_gradients(tmp_path, tiny_model_dir, toy_examples) -> None:  # type: ignore[no-untyped-def]
    examples = [
        ContrastiveExample(e.query, e.positive, [toy_examples[(i + 3) % 20].positive])
        for i, e in enumerate(toy_examples[:8])
    ]
    base = dict(output_dir=str(tmp_path), batch_size=8, max_negatives=1)
    loss_full, g_full = _grads(_encoder(tiny_model_dir), TrainConfig(**base), examples)
    loss_gc, g_gc = _grads(_encoder(tiny_model_dir), TrainConfig(**base, mini_batch_size=3), examples)
    assert loss_full == pytest.approx(loss_gc, rel=1e-5)
    assert g_full.keys() == g_gc.keys() and g_full
    for name in g_full:
        torch.testing.assert_close(g_gc[name], g_full[name], atol=1e-5, rtol=1e-4, msg=name)


def test_lora_training_saves_adapter_and_merged_model(tmp_path, tiny_model_dir, toy_examples) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("peft")
    enc = _encoder(tiny_model_dir)
    frozen = {n: p.detach().clone() for n, p in enc.model.named_parameters()}
    cfg = TrainConfig(
        output_dir=str(tmp_path / "lora"),
        max_steps=4,
        batch_size=5,
        learning_rate=1e-2,
        lora=LoraSettings(r=4, alpha=8, target_modules=["q_proj", "v_proj"]),
        save_every=2,
        mini_batch_size=4,
        gradient_checkpointing=True,
    )
    trainer = ContrastiveTrainer(enc, cfg, toy_examples)
    trainable = [n for n, p in enc.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    trainer.train()
    ckpt = tmp_path / "lora" / "checkpoint-2"
    assert (ckpt / "adapter_config.json").is_file() and not (ckpt / "config.json").exists()
    # adapter-only checkpoint loads base + adapter automatically
    from_adapter = HFEncoder(str(ckpt), device="cpu")
    assert from_adapter.encode("cats").shape == (32,)
    # final checkpoint is merged -> plain model, loads without peft wrappers
    final = tmp_path / "lora" / "final"
    assert (final / "config.json").is_file() and not (final / "adapter_config.json").exists()
    merged = HFEncoder(str(final), device="cpu")
    np.testing.assert_allclose(merged.encode(["cats eat fish"]), enc.encode(["cats eat fish"]), atol=1e-5)
    base_after = dict(HFEncoder(str(tiny_model_dir), device="cpu").model.named_parameters())
    assert all(torch.equal(base_after[n], frozen[n]) for n in frozen)  # base checkpoint untouched


def test_matryoshka_and_cosent_objectives(tmp_path, tiny_model_dir, toy_examples) -> None:  # type: ignore[no-untyped-def]
    cfg = TrainConfig(output_dir=str(tmp_path / "mrl"), max_steps=2, batch_size=4, matryoshka_dims=[32, 16, 8])
    assert ContrastiveTrainer(_encoder(tiny_model_dir), cfg, toy_examples).train().global_step == 2
    scored = [ContrastiveExample(e.query, e.positive, score=float(i % 3)) for i, e in enumerate(toy_examples)]
    cfg2 = TrainConfig(output_dir=str(tmp_path / "cosent"), max_steps=2, batch_size=6, loss="cosent")
    assert ContrastiveTrainer(_encoder(tiny_model_dir), cfg2, scored).train().global_step == 2


@pytest.mark.parametrize("loss", ["infonce", "cosent", "triplet"])
def test_selected_loss_uses_its_own_defaults(tmp_path, tiny_model_dir, toy_examples, loss) -> None:
    examples = [
        ContrastiveExample(e.query, e.positive, [toy_examples[(i + 3) % len(toy_examples)].positive], score=float(i))
        for i, e in enumerate(toy_examples[:4])
    ]
    cfg = TrainConfig(output_dir=str(tmp_path / loss), max_steps=1, batch_size=4, loss=loss)
    trainer = ContrastiveTrainer(_encoder(tiny_model_dir), cfg, examples)
    if loss == "infonce":
        assert trainer.loss_fn.temperature == 0.05
    result = trainer.train()
    assert result.global_step == 1 and np.isfinite(result.train_loss)


def test_config_validation(tmp_path, tiny_model_dir) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="fp16"):
        TrainConfig(precision="fp16")
    with pytest.raises(ValueError):
        TrainConfig(batch_size=0)
    with pytest.raises(ValueError, match="log_every"):
        TrainConfig(log_every=0)
    with pytest.raises(ValueError, match="max_steps"):
        TrainConfig(max_steps=0)
    with pytest.raises(ValueError, match="empty"):
        ContrastiveTrainer(_encoder(tiny_model_dir), TrainConfig(output_dir=str(tmp_path)), [])
    assert isinstance(TrainConfig(lora={"r": 2}).lora, LoraSettings)  # type: ignore[arg-type]


def test_train_from_config_end_to_end(tmp_path, tiny_model_dir, toy_examples) -> None:  # type: ignore[no-untyped-def]
    train = save_examples(tmp_path / "train.jsonl", toy_examples)
    cfg = {
        "encoder": {"type": "hf", "model": str(tiny_model_dir), "device": "cpu", **QWEN_LIKE},
        "data": {"train": str(train), "eval_fraction": 0.2},
        "train": {"output_dir": str(tmp_path / "out"), "max_steps": 3, "batch_size": 4, "precision": "bf16"},
        "eval": {"ks": [1, 3]},
    }
    result = train_from_config(cfg)
    assert result.global_step == 3 and "ndcg@3" in result.eval
    from clmkit.config import ConfigError

    with pytest.raises(ConfigError, match=r"did you mean 'learning_rate'"):
        train_from_config({**cfg, "train": {"learnig_rate": 1}})
    with pytest.raises(ConfigError, match="top-level"):
        train_from_config({**cfg, "trian": {}})
    with pytest.raises(ConfigError, match=r"data\.train"):
        train_from_config({**cfg, "data": {}})


def test_cli_train_with_overrides(tmp_path, tiny_model_dir, toy_examples, capsys) -> None:  # type: ignore[no-untyped-def]
    from clmkit.cli import main

    train = save_examples(tmp_path / "train.jsonl", toy_examples)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "encoder": {"type": "hf", "model": str(tiny_model_dir), "device": "cpu", **QWEN_LIKE},
                "data": {"train": str(train)},
                "train": {"output_dir": str(tmp_path / "cli-run"), "max_steps": 5, "batch_size": 4},
            }
        )
    )
    assert main(["train", "--config", str(cfg_path), "--set", "train.max_steps=2"]) == 0
    assert json.loads(capsys.readouterr().out)["global_step"] == 2
