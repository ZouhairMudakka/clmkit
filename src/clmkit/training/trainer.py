"""A compact, readable contrastive fine-tuning loop (PyTorch).

Highlights:

* **GradCache** (``mini_batch_size``): large contrastive batches in small memory.
  In-batch negatives make the *effective* batch size matter; GradCache computes
  the loss over the whole batch while only ever back-propagating ``mini_batch_size``
  sequences at a time (Gao et al., 2021). Results match full-batch training.
* **LoRA** via ``peft`` for 4B/8B models (e.g. Qwen3-Embedding-8B) on one GPU.
* bf16 autocast, gradient checkpointing, warmup + linear decay, grad clipping,
  periodic eval with best-checkpoint tracking, and pluggable loss functions.

It deliberately avoids hiding the loop inside a framework: read ``train()`` top to bottom.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from clmkit.config import to_dict
from clmkit.data import ContrastiveExample, iter_batches
from clmkit.losses import ContrastiveLoss, build_loss
from clmkit.types import EncodeKind
from clmkit.utils import require, set_seed

if TYPE_CHECKING:
    import torch

    from clmkit.encoders.base import Encoder

logger = logging.getLogger(__name__)


class TrainableEncoder(Protocol):
    """What the trainer needs from an encoder (``HFEncoder`` implements it)."""

    model: Any
    device: str

    def forward(self, texts: Sequence[str], kind: EncodeKind = ..., instruction: Any = ...) -> torch.Tensor: ...
    def save_pretrained(self, path: str | Path, *, merge_adapter: bool = ...) -> Path: ...


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    #: ``"all-linear"`` or explicit names, e.g. ``["q_proj", "k_proj", "v_proj", "o_proj"]``.
    target_modules: list[str] | str = "all-linear"


@dataclass
class TrainConfig:
    output_dir: str = "runs/clmkit"
    epochs: int = 1
    max_steps: int | None = None
    batch_size: int = 32
    #: GradCache chunk size. ``None`` = back-prop the whole batch at once.
    mini_batch_size: int | None = None
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float | None = 1.0
    loss: str = "infonce"
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    matryoshka_dims: list[int] | None = None
    #: Hard negatives per example (the batch uses min(available, max_negatives)).
    max_negatives: int | None = 7
    avoid_duplicates: bool = True
    seed: int = 42
    #: ``"auto"`` (bf16 on capable CUDA, else fp32), ``"bf16"`` or ``"fp32"``.
    precision: str = "auto"
    gradient_checkpointing: bool = False
    lora: LoraSettings | None = None
    merge_lora_on_save: bool = True
    log_every: int = 10
    eval_every: int | None = None
    save_every: int | None = None
    #: Evaluator metric used to keep ``output_dir/best`` (higher is better), e.g. ``"ndcg@10"``.
    metric_for_best: str | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1 and self.max_steps is None:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.mini_batch_size is not None and self.mini_batch_size < 1:
            raise ValueError("mini_batch_size must be >= 1")
        if self.log_every < 1:
            raise ValueError("log_every must be >= 1")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_negatives is not None and self.max_negatives < 0:
            raise ValueError("max_negatives must be >= 0")
        if self.precision not in ("auto", "bf16", "fp32"):
            raise ValueError("precision must be 'auto', 'bf16' or 'fp32' (fp16 is not supported)")
        if isinstance(self.lora, dict):
            self.lora = LoraSettings(**self.lora)


@dataclass
class TrainResult:
    global_step: int
    train_loss: float
    history: list[dict[str, Any]]
    eval: dict[str, float]
    output_dir: str
    best_metric: float | None = None


Evaluator = Callable[["Encoder"], dict[str, float]]
Callback = Callable[[dict[str, Any]], None]


class ContrastiveTrainer:
    def __init__(
        self,
        encoder: TrainableEncoder,
        config: TrainConfig,
        train_examples: Sequence[ContrastiveExample],
        *,
        evaluator: Evaluator | None = None,
        loss_fn: ContrastiveLoss | None = None,
        callbacks: Sequence[Callback] = (),
    ) -> None:
        self._torch = torch = require("torch")
        if not train_examples:
            raise ValueError("train_examples is empty")
        self.encoder = encoder
        self.config = config
        self.examples = list(train_examples)
        self.evaluator = evaluator
        self.callbacks = list(callbacks)
        set_seed(config.seed)

        model = encoder.model
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            if config.lora is not None and hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        if config.lora is not None:
            model = self._apply_lora(model, config.lora)
            encoder.model = model

        self.loss_fn = loss_fn or build_loss(config.loss, matryoshka_dims=config.matryoshka_dims, **config.loss_kwargs)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("model has no trainable parameters")
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]  # biases and norm weights
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=config.learning_rate,
        )
        # Duplicate avoidance can expand (and shuffling can change) an epoch's
        # batch count. Count the same deterministic batches the loop will use.
        self.total_steps = config.max_steps or sum(
            sum(1 for _ in self._epoch_batches(epoch)) for epoch in range(config.epochs)
        )
        warmup = int(self.total_steps * config.warmup_ratio)

        def schedule(step: int) -> float:
            if warmup and step < warmup:
                return (step + 1) / warmup
            return max(0.0, (self.total_steps - step) / max(1, self.total_steps - warmup))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)
        self.global_step = 0
        self.history: list[dict[str, Any]] = []
        self.best_metric: float | None = None

    # ------------------------------------------------------------- set-up --
    @staticmethod
    def _apply_lora(model: Any, lora: LoraSettings) -> Any:
        peft = require("peft")
        cfg = peft.LoraConfig(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=lora.target_modules,
            task_type=peft.TaskType.FEATURE_EXTRACTION,
        )
        model = peft.get_peft_model(model, cfg)
        trainable, total = model.get_nb_trainable_parameters()
        logger.info("LoRA: %s trainable / %s total params (%.3f%%)", trainable, total, 100 * trainable / total)
        return model

    def _autocast(self) -> contextlib.AbstractContextManager[Any]:
        torch = self._torch
        device_type = str(self.encoder.device).split(":")[0]
        precision = self.config.precision
        if precision == "auto":
            precision = "bf16" if device_type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
        if precision == "bf16":
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # --------------------------------------------------------------- step --
    def _epoch_batches(self, epoch: int) -> Iterator[list[ContrastiveExample]]:
        return iter_batches(
            self.examples,
            self.config.batch_size,
            shuffle=True,
            seed=self.config.seed + epoch,
            avoid_duplicates=self.config.avoid_duplicates,
        )

    def _check_finite_loss(self, loss: torch.Tensor) -> None:
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError(f"non-finite loss at step {self.global_step + 1}")

    def _prepare(self, batch: list[ContrastiveExample]) -> tuple[list[str], list[str | None], list[str], int, Any]:
        n_per = min(len(ex.negatives) for ex in batch)
        if self.config.max_negatives is not None:
            n_per = min(n_per, self.config.max_negatives)
        q_texts = [ex.query for ex in batch]
        q_inst = [ex.instruction for ex in batch]
        d_texts = [ex.positive for ex in batch] + [neg for ex in batch for neg in ex.negatives[:n_per]]
        scores = None
        if all(ex.score is not None for ex in batch):
            scores = self._torch.tensor(
                [ex.score for ex in batch], dtype=self._torch.float32, device=self.encoder.device
            )
        return q_texts, q_inst, d_texts, n_per, scores

    def _loss(self, q: torch.Tensor, d: torch.Tensor, n_per: int, scores: Any) -> torch.Tensor:
        bsz = q.shape[0]
        pos = d[:bsz]
        neg = d[bsz:].reshape(bsz, n_per, -1) if n_per else None
        return self.loss_fn(q.float(), pos.float(), neg.float() if neg is not None else None, scores)

    def _rng_state(self) -> tuple[Any, Any]:
        torch = self._torch
        cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return torch.get_rng_state(), cuda

    def _set_rng_state(self, state: tuple[Any, Any]) -> None:
        torch = self._torch
        torch.set_rng_state(state[0])
        if state[1] is not None:  # pragma: no cover - GPU only
            torch.cuda.set_rng_state_all(state[1])

    def _chunks(self, n: int) -> Iterator[slice]:
        size = self.config.mini_batch_size or n
        for start in range(0, n, size):
            yield slice(start, start + size)

    def training_step(self, batch: list[ContrastiveExample]) -> float:
        """Compute loss and gradients for one batch (no optimizer step)."""
        q_texts, q_inst, d_texts, n_per, scores = self._prepare(batch)
        total = len(q_texts) + len(d_texts)
        if self.config.mini_batch_size is None or self.config.mini_batch_size >= total:
            with self._autocast():
                q = self.encoder.forward(q_texts, "query", q_inst)
                d = self.encoder.forward(d_texts, "document", None)
            loss = self._loss(q, d, n_per, scores)
            self._check_finite_loss(loss)
            loss.backward()
            return float(loss.detach())
        return self._gradcache_step(q_texts, q_inst, d_texts, n_per, scores)

    def _gradcache_step(
        self, q_texts: list[str], q_inst: list[str | None], d_texts: list[str], n_per: int, scores: Any
    ) -> float:
        torch = self._torch
        groups: list[tuple[EncodeKind, list[str], list[str | None] | None]] = [
            ("query", q_texts, q_inst),
            ("document", d_texts, None),
        ]

        # 1) Representation pass without a graph; remember RNG so dropout replays identically.
        reps: list[list[Any]] = []
        states: list[list[tuple[Any, Any]]] = []
        for kind, texts, inst in groups:
            g_reps, g_states = [], []
            for sl in self._chunks(len(texts)):
                g_states.append(self._rng_state())
                with torch.no_grad(), self._autocast():
                    g_reps.append(self.encoder.forward(texts[sl], kind, inst[sl] if inst else None))
            reps.append(g_reps)
            states.append(g_states)

        # 2) Full-batch loss on detached reps -> gradient w.r.t. every representation.
        q_all = torch.cat(reps[0]).float().detach().requires_grad_()
        d_all = torch.cat(reps[1]).float().detach().requires_grad_()
        loss = self._loss(q_all, d_all, n_per, scores)
        self._check_finite_loss(loss)
        loss.backward()
        cached = [q_all.grad, d_all.grad]
        if any(grad is not None and not torch.isfinite(grad).all() for grad in cached):
            raise FloatingPointError(f"non-finite cached gradients at step {self.global_step + 1}")

        # 3) Re-run each chunk with a graph and push the cached gradient through it.
        for (kind, texts, inst), g_states, grad in zip(groups, states, cached, strict=True):
            if grad is None:  # pragma: no cover - only if the loss ignores an input entirely
                raise RuntimeError(f"GradCache: the loss produced no gradient for the {kind} embeddings")
            for sl, state in zip(self._chunks(len(texts)), g_states, strict=True):
                self._set_rng_state(state)
                with self._autocast():
                    rep = self.encoder.forward(texts[sl], kind, inst[sl] if inst else None)
                (rep.float() * grad[sl]).sum().backward()
        return float(loss.detach())

    # --------------------------------------------------------------- loop --
    def train(self) -> TrainResult:
        torch = self._torch
        cfg = self.config
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "train_config.json").write_text(json.dumps(to_dict(cfg), indent=2, default=str), encoding="utf-8")
        model = self.encoder.model
        model.train()
        params = [p for p in model.parameters() if p.requires_grad]
        running: list[float] = []
        last_eval: dict[str, float] = {}
        started = time.time()
        done = False
        epoch = 0
        while not done and (cfg.max_steps is not None or epoch < cfg.epochs):
            for batch in self._epoch_batches(epoch):
                try:
                    loss = self.training_step(batch)
                    if not math.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss at step {self.global_step + 1}")
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
                        raise FloatingPointError(f"non-finite gradients at step {self.global_step + 1}")
                    if cfg.max_grad_norm:
                        norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                        if not torch.isfinite(norm):
                            raise FloatingPointError(f"non-finite gradient norm at step {self.global_step + 1}")
                except FloatingPointError:
                    self.optimizer.zero_grad(set_to_none=True)
                    raise
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                running.append(loss)
                if self.global_step % cfg.log_every == 0:
                    self._log(
                        {
                            "step": self.global_step,
                            "epoch": epoch,
                            "loss": sum(running[-cfg.log_every :]) / len(running[-cfg.log_every :]),
                            "lr": self.scheduler.get_last_lr()[0],
                            "elapsed_s": round(time.time() - started, 1),
                        }
                    )
                if cfg.eval_every and self.global_step % cfg.eval_every == 0:
                    last_eval = self.evaluate()
                    model.train()
                if cfg.save_every and self.global_step % cfg.save_every == 0:
                    self.encoder.save_pretrained(out / f"checkpoint-{self.global_step}", merge_adapter=False)
                if cfg.max_steps is not None and self.global_step >= cfg.max_steps:
                    done = True
                    break
            epoch += 1

        if self.evaluator is not None and (not cfg.eval_every or self.global_step % cfg.eval_every):
            last_eval = self.evaluate()
        final = out / "final"
        self.encoder.save_pretrained(final, merge_adapter=cfg.merge_lora_on_save and cfg.lora is not None)
        tail = running[-cfg.log_every :] or [float("nan")]
        result = TrainResult(
            self.global_step, sum(tail) / len(tail), self.history, last_eval, str(out), self.best_metric
        )
        (out / "trainer_state.json").write_text(json.dumps(to_dict(result), indent=2, default=str), encoding="utf-8")
        return result

    def evaluate(self) -> dict[str, float]:
        if self.evaluator is None:
            return {}
        metrics = self.evaluator(self.encoder)  # type: ignore[arg-type]
        self._log({"step": self.global_step, **{f"eval/{k}": v for k, v in metrics.items()}})
        key = self.config.metric_for_best
        if key is not None:
            if key not in metrics:
                raise KeyError(f"metric_for_best={key!r} not in evaluator output {sorted(metrics)}")
            if self.best_metric is None or metrics[key] > self.best_metric:
                self.best_metric = metrics[key]
                self.encoder.save_pretrained(Path(self.config.output_dir) / "best", merge_adapter=False)
        return metrics

    def _log(self, record: dict[str, Any]) -> None:
        self.history.append(record)
        logger.info(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
        for cb in self.callbacks:
            cb(record)
