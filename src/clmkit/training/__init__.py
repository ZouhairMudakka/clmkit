"""Contrastive fine-tuning (requires ``pip install "clmkit[train]"``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from clmkit.config import ConfigError, from_dict, load_config
from clmkit.training.trainer import ContrastiveTrainer, LoraSettings, TrainConfig, TrainResult

__all__ = ["ContrastiveTrainer", "LoraSettings", "TrainConfig", "TrainResult", "train_from_config"]

logger = logging.getLogger(__name__)

_TOP_LEVEL = {"encoder", "data", "train", "eval"}


def train_from_config(config: str | Path | dict[str, Any]) -> TrainResult:
    """Run a full training job from a YAML/JSON file or dict::

    encoder: {type: hf, model: Qwen/Qwen3-Embedding-8B, dtype: bfloat16, max_length: 512}
    data:
      train: data/train.jsonl
      eval: data/eval.jsonl          # optional; else `eval_fraction` of train is held out
      eval_fraction: 0.05
    train: {output_dir: runs/qwen3-8b, batch_size: 64, mini_batch_size: 8, lora: {r: 16}}
    eval: {ks: [1, 5, 10]}
    """
    from clmkit.data import load_examples, split_examples
    from clmkit.encoders import load_encoder
    from clmkit.eval import RetrievalEvaluator

    cfg = load_config(config) if isinstance(config, (str, Path)) else dict(config)
    unknown = set(cfg) - _TOP_LEVEL
    if unknown:
        raise ConfigError(f"unknown top-level section(s): {sorted(unknown)}; expected {sorted(_TOP_LEVEL)}")
    if "encoder" not in cfg or "data" not in cfg:
        raise ConfigError("config needs 'encoder' and 'data' sections")
    data = dict(cfg["data"])
    if "train" not in data:
        raise ConfigError("data.train is required")
    train_cfg = from_dict(TrainConfig, cfg.get("train"), where="train")
    enc_spec = cfg["encoder"]
    encoder = load_encoder(enc_spec) if isinstance(enc_spec, (str, dict)) else enc_spec

    train_examples = load_examples(data["train"])
    if data.get("eval"):
        eval_examples = load_examples(data["eval"])
    elif data.get("eval_fraction"):
        train_examples, eval_examples = split_examples(
            train_examples, float(data["eval_fraction"]), seed=train_cfg.seed
        )
    else:
        eval_examples = []
    evaluator = RetrievalEvaluator.from_examples(eval_examples, **(cfg.get("eval") or {})) if eval_examples else None
    logger.info("training on %d examples (%d eval)", len(train_examples), len(eval_examples))
    trainer = ContrastiveTrainer(encoder, train_cfg, train_examples, evaluator=evaluator)  # type: ignore[arg-type]
    return trainer.train()
