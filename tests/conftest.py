"""Shared fixtures. Nothing here downloads anything: HF models are built from tiny configs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from clmkit import HashingEncoder
from clmkit.data import ContrastiveExample

# A toy paired dataset with clear topical structure (animals / space / cooking / sports / code).
TOY_PAIRS = [
    ("what do cats eat", "cats eat fish and meat and drink milk"),
    ("how fast can a cheetah run", "the cheetah is the fastest land animal running at high speed"),
    ("where do penguins live", "penguins live in the cold southern ocean near antarctica"),
    ("why do dogs bark", "dogs bark to warn and to communicate with people"),
    ("how far is the moon", "the moon orbits the earth at a large distance in space"),
    ("what is a black hole", "a black hole is a region of space where gravity traps light"),
    ("how hot is the sun", "the sun is a very hot star at the centre of our solar system"),
    ("what planet has rings", "saturn is the planet with bright rings of ice and rock"),
    ("how to boil an egg", "boil the egg in hot water for eight minutes then cool it"),
    ("how to bake bread", "mix flour water yeast and salt then bake the dough in an oven"),
    ("how to make tea", "pour hot water over tea leaves and wait a few minutes"),
    ("what is pasta made of", "pasta is made of flour water and sometimes egg"),
    ("how many players in football", "a football team has eleven players on the field"),
    ("who invented basketball", "basketball was invented by a teacher who nailed a basket to a wall"),
    ("how long is a marathon", "a marathon race is a long run of about forty two kilometres"),
    ("what is tennis", "tennis is played with a racket and a ball over a net"),
    ("what is python", "python is a programming language that is easy to read"),
    ("what is a compiler", "a compiler translates source code into machine code"),
    ("what is a database", "a database stores and queries structured data"),
    ("what is recursion", "recursion is when a function calls itself"),
]

SPECIALS = ["<pad>", "<unk>", "<eos>", "<|endoftext|>", "<|im_end|>", "yes", "no"]


@pytest.fixture
def hashing() -> HashingEncoder:
    return HashingEncoder(dim=256)


@pytest.fixture
def toy_examples() -> list[ContrastiveExample]:
    return [ContrastiveExample(q, p) for q, p in TOY_PAIRS]


def _vocab() -> list[str]:
    words: set[str] = set()
    for q, p in TOY_PAIRS:
        words.update(re.findall(r"\w+|[^\w\s]+", f"{q} {p}".lower()))
    words.update(
        [
            "instruct",
            "query",
            ":",
            "given",
            "a",
            "web",
            "search",
            "retrieve",
            "relevant",
            "passages",
            "that",
            "answer",
            "the",
            "find",
            "document",
            "hello",
            "world",
            "unrelated",
            "text",
        ]
    )
    return SPECIALS + sorted(words - set(SPECIALS))


def build_tiny_tokenizer():  # type: ignore[no-untyped-def]
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(_vocab())}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>")


def _tiny_config():  # type: ignore[no-untyped-def]
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=len(_vocab()),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=True,
    )


@pytest.fixture
def tiny_hf_parts():  # type: ignore[no-untyped-def]
    """(config factory, tokenizer factory) for building other tiny HF model heads in tests."""
    pytest.importorskip("transformers")
    return _tiny_config, build_tiny_tokenizer


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-layer, 32-dim Qwen3 encoder saved to disk (like a hub checkpoint)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch
    from transformers import Qwen3Model

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-qwen3")
    Qwen3Model(_tiny_config()).save_pretrained(path)
    build_tiny_tokenizer().save_pretrained(path)
    return path


@pytest.fixture(scope="session")
def tiny_causal_lm_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny Qwen3ForCausalLM for LLM yes/no reranker tests."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import torch
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny-qwen3-lm")
    Qwen3ForCausalLM(_tiny_config()).save_pretrained(path)
    build_tiny_tokenizer().save_pretrained(path)
    return path
