"""Independent scientific contracts; failures intentionally expose baseline defects.

No downloaded models. Expected tolerances established before execution:
float64 loss oracles 1e-10; float32 GradCache gradients 1e-5 absolute/1e-4 relative.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from clmkit.data import ContrastiveExample
from clmkit.losses import CoSENTLoss, MatryoshkaLoss, info_nce
from clmkit.training import ContrastiveTrainer, TrainConfig

torch.set_num_threads(2)


class ToyEncoder:
    device = 'cpu'

    def __init__(self, dropout=0.0):
        self.model = nn.Sequential(nn.Embedding(64, 8), nn.Dropout(dropout), nn.Linear(8, 6))
        self.seen_queries = []

    def forward(self, texts, kind='document', instruction=None):
        if kind == 'query':
            self.seen_queries.extend(texts)
        ids = torch.tensor([sum(t.encode('utf-8')) % 64 for t in texts])
        return self.model(ids)

    def save_pretrained(self, path, *, merge_adapter=False):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)


def pairs(n=5):
    return [ContrastiveExample(f'q{i}', f'p{i}', [f'n{i}']) for i in range(n)]


def test_infonce_matches_independent_logsumexp_and_gradcheck():
    gen = torch.Generator().manual_seed(173)
    q = torch.randn(3, 5, dtype=torch.float64, generator=gen, requires_grad=True)
    p = torch.randn(3, 5, dtype=torch.float64, generator=gen, requires_grad=True)
    n = torch.randn(3, 2, 5, dtype=torch.float64, generator=gen, requires_grad=True)
    temp = 0.3
    rows = []
    candidates = torch.cat([p, n.reshape(-1, 5)])
    for i in range(3):
        scores = torch.stack([torch.dot(q[i], c) / (q[i].norm() * c.norm()) / temp for c in candidates])
        rows.append(torch.logsumexp(scores, 0) - scores[i])
    torch.testing.assert_close(info_nce(q, p, n, temperature=temp), torch.stack(rows).mean(), atol=1e-10, rtol=1e-10)
    assert torch.autograd.gradcheck(lambda a,b,c: info_nce(a,b,c,temperature=temp), (q,p,n))


def test_cosent_matches_explicit_pair_ordering():
    q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    p = torch.tensor([[1., 1.], [1., 0.], [1., 2.]], dtype=torch.float64)
    labels = torch.tensor([2., 0., 1.], dtype=torch.float64)
    cos = F.cosine_similarity(q, p)
    terms = [torch.exp(2 * (cos[j] - cos[i])) for i in range(3) for j in range(3) if labels[i] > labels[j]]
    expected = torch.log(1 + sum(terms))
    torch.testing.assert_close(CoSENTLoss(scale=2)(q,p,scores=labels), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('dropout', [0.0, 0.25])
@pytest.mark.parametrize('mrl', [False, True])
def test_gradcache_matches_chunked_autograd_with_rng_replay(tmp_path, dropout, mrl):
    # Dropout RNG draws can differ with batch shape; the oracle uses the same
    # chunk partition while retaining graphs, rather than comparing different masks.
    torch.manual_seed(91)
    oracle = ToyEncoder(dropout)
    cached = copy.deepcopy(oracle)
    examples = pairs(5)
    cfg = TrainConfig(output_dir=str(tmp_path), batch_size=5, mini_batch_size=2,
                      matryoshka_dims=[6, 3] if mrl else None)
    trainer = ContrastiveTrainer(cached, cfg, examples)
    oracle.model.train()
    cached.model.train()
    qtexts = [e.query for e in examples]
    dtexts = [e.positive for e in examples] + [n for e in examples for n in e.negatives]
    torch.manual_seed(333)
    q = torch.cat([oracle.forward(qtexts[i:i+2], 'query') for i in range(0, len(qtexts), 2)])
    d = torch.cat([oracle.forward(dtexts[i:i+2]) for i in range(0, len(dtexts), 2)])
    loss = trainer.loss_fn(q, d[:5], d[5:].reshape(5, 1, 6))
    loss.backward()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(333)
    actual_loss = trainer.training_step(examples)
    assert actual_loss == pytest.approx(float(loss.detach()), abs=1e-5, rel=1e-4)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for a, b in zip(oracle.model.parameters(), cached.model.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-4)
    # Match the optimizer's parameter groups, learning rate and schedule starting LR.
    reference_opt = torch.optim.AdamW([
        {'params': [p for p in oracle.model.parameters() if p.ndim >= 2], 'weight_decay': cfg.weight_decay},
        {'params': [p for p in oracle.model.parameters() if p.ndim < 2], 'weight_decay': 0.0},
    ], lr=trainer.optimizer.param_groups[0]['lr'])
    reference_opt.step()
    trainer.optimizer.step()
    for a,b in zip(oracle.model.parameters(), cached.model.parameters(), strict=True):
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)


def test_epoch_processes_all_duplicate_avoiding_batches(tmp_path):
    examples = [ContrastiveExample('same-query', f'positive-{i}') for i in range(5)]
    enc = ToyEncoder()
    trainer = ContrastiveTrainer(enc, TrainConfig(output_dir=str(tmp_path), epochs=1, batch_size=4), examples)
    result = trainer.train()
    assert result.global_step == 5, f'Expected five singleton batches; got {result.global_step}'
    assert len(enc.seen_queries) == len(examples)


def test_nonfinite_loss_does_not_mutate_model(tmp_path):
    enc = ToyEncoder()
    initial = copy.deepcopy(enc.model.state_dict())
    def bad_loss(q,p,n=None,scores=None):
        return (q.sum() + p.sum()) * torch.tensor(float('nan'))
    trainer = ContrastiveTrainer(enc, TrainConfig(output_dir=str(tmp_path), max_steps=1), pairs(), loss_fn=bad_loss)
    with pytest.raises(FloatingPointError):
        trainer.train()
    for name,value in enc.model.state_dict().items():
        torch.testing.assert_close(value, initial[name], atol=0, rtol=0)


def test_matryoshka_preserves_dimension_weight_pairing():
    def dimension_loss(q,p,n=None,scores=None):
        return q.new_tensor(float(q.shape[-1]))
    q = torch.ones(2, 4)
    actual = MatryoshkaLoss(dimension_loss, [2,4], weights=[1,3])(q,q)
    assert float(actual) == pytest.approx((1*2 + 3*4) / 4)


@pytest.mark.parametrize('dims', [[0], [-1]])
def test_matryoshka_rejects_nonpositive_dimensions(dims):
    q = torch.tensor([[1.,2.,3.]])
    with pytest.raises(ValueError):
        MatryoshkaLoss(lambda q,p,n=None,scores=None: q.sum(), dims)(q,q)


def test_hf_save_reload_preserves_effective_output_dimension(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen3Config, Qwen3Model, PreTrainedTokenizerFast
    from clmkit.encoders.hf import HFEncoder
    tok = Tokenizer(models.WordLevel({'<pad>':0,'<unk>':1,'hello':2}, unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok,pad_token='<pad>',unk_token='<unk>')
    model = Qwen3Model(Qwen3Config(vocab_size=3,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8,pad_token_id=0))
    enc = HFEncoder(model=model,tokenizer=tokenizer,device='cpu',output_dim=8)
    before = enc.encode('hello')
    enc.save_pretrained(tmp_path / 'model')
    loaded = HFEncoder(str(tmp_path / 'model'),device='cpu')
    assert loaded.dim == enc.dim, f'Saved dimension {enc.dim}, reloaded {loaded.dim}'
    np.testing.assert_allclose(loaded.encode('hello'),before,atol=1e-5,rtol=1e-4)
