"""Bounded F03 extension: finite reported loss need not imply finite gradients.

The custom backward deliberately injects infinity; no claim is made that ordinary
supported losses trigger this on typical finite inputs. This checks failure handling.
"""
import copy
import json
from pathlib import Path

import pytest
import torch

from test_scientific_contracts import ToyEncoder, pairs
from clmkit.training import ContrastiveTrainer, TrainConfig

OUT = Path(__file__).resolve().parent


class FiniteValueInfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.new_tensor(1.0)

    @staticmethod
    def backward(ctx, upstream):
        return upstream.new_tensor(float('inf'))


@pytest.mark.parametrize('mini_batch_size', [None, 2], ids=['ordinary', 'gradcache'])
def test_finite_loss_nonfinite_gradient_rejected_before_update(tmp_path, mini_batch_size):
    encoder = ToyEncoder()
    initial = copy.deepcopy(encoder.model.state_dict())
    observed_losses = []

    def bad_backward_loss(q, p, n=None, scores=None):
        value = FiniteValueInfiniteGradient.apply(q.sum() + p.sum())
        observed_losses.append(float(value.detach()))
        return value

    trainer = ContrastiveTrainer(encoder, TrainConfig(output_dir=str(tmp_path), max_steps=1,
        mini_batch_size=mini_batch_size), pairs(), loss_fn=bad_backward_loss)
    scheduler_before = copy.deepcopy(trainer.scheduler.state_dict())
    error = None
    try:
        trainer.train()
    except Exception as exc:
        error = exc
    payload = {
        'mini_batch_size': mini_batch_size, 'observed_loss_values': observed_losses,
        'exception': None if error is None else repr(error), 'global_step': trainer.global_step,
        'nonfinite_parameter_names': [name for name, value in encoder.model.state_dict().items()
                                     if not bool(torch.isfinite(value).all())],
        'changed_parameter_names': [name for name, value in encoder.model.state_dict().items()
                                   if not torch.equal(value, initial[name])],
        'optimizer_state_entries': len(trainer.optimizer.state),
        'scheduler_changed': trainer.scheduler.state_dict() != scheduler_before,
    }
    suffix = 'ordinary' if mini_batch_size is None else 'gradcache'
    (OUT / f'finite-loss-gradient-{suffix}.json').write_text(json.dumps(payload, indent=2))
    assert observed_losses == [1.0]
    assert isinstance(error, FloatingPointError), payload
    assert payload['changed_parameter_names'] == [], payload
    assert payload['global_step'] == 0, payload
    assert payload['optimizer_state_entries'] == 0, payload
    assert payload['scheduler_changed'] is False, payload
