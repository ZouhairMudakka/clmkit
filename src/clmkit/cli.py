"""``clmkit`` command-line interface.

clmkit info
clmkit encode   --model hashing "some text" "more text" --output emb.npy
clmkit index    --model Qwen/Qwen3-Embedding-0.6B --input docs.jsonl --output ./idx
clmkit search   --index ./idx "what is contrastive learning?" -k 5
clmkit mine     --model hashing --train train.jsonl --corpus corpus.txt --output mined.jsonl
clmkit train    --config configs/qwen3-embedding-8b-lora.yaml
clmkit eval     --model hashing --data eval.jsonl
clmkit serve    --model hashing --index ./idx --port 8000
clmkit mcp      --model hashing --index ./idx
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from clmkit import __version__
from clmkit.utils import parse_value


def _kv(pairs: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"error: expected key=value, got {pair!r}")
        out[key.replace("-", "_")] = parse_value(value)
    return out


def _encoder_spec(args: argparse.Namespace) -> dict[str, Any]:
    return {"model": args.model, "kwargs": _kv(args.model_arg)}


def _build_encoder(spec: dict[str, Any]) -> Any:
    from clmkit.encoders import load_encoder

    return load_encoder(spec["model"], **spec.get("kwargs", {}))


def _validate_persisted_encoder_spec(spec: dict[str, Any]) -> None:
    """Reject credential-bearing options before saving a reconstructible CLI spec."""
    credential_keys = {"api_key", "token", "use_auth_token", "access_token", "password", "authorization", "headers"}

    def has_credentials(value: Any) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                name = str(key).lower().replace("-", "_")
                if name in credential_keys and item is not None:
                    # HF token=True/False selects cached auth or disables it; no literal secret.
                    if name in {"token", "use_auth_token"} and isinstance(item, bool):
                        continue
                    if item:
                        return True
                if name == "base_url" and isinstance(item, str):
                    url = urlsplit(item)
                    if url.username is not None or url.password is not None or url.query or url.fragment:
                        return True
                if has_credentials(item):
                    return True
        elif isinstance(value, list):
            return any(has_credentials(item) for item in value)
        return False

    if has_credentials(spec):
        raise SystemExit(
            "error: index snapshots store encoder options; literal credentials, custom headers, and credentials "
            "in base_url are not allowed. Use --model-arg api_key_env=ENV_NAME (OPENAI_API_KEY by default), "
            "or HF_TOKEN/cached Hugging Face login instead of literal token options."
        )


def _read_texts(path: str | None, positional: Sequence[str], text_field: str) -> list[str]:
    if positional:
        return list(positional)
    if path in (None, "-"):
        lines = sys.stdin.read().splitlines()
    else:
        p = Path(path)
        if p.suffix == ".jsonl":
            return [json.loads(line)[text_field] for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        lines = p.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip()]


def _read_documents(
    path: str, text_field: str, id_field: str
) -> tuple[list[str], list[str] | None, list[dict[str, Any]]]:
    p = Path(path)
    texts: list[str] = []
    ids: list[str] = []
    metas: list[dict[str, Any]] = []
    if p.suffix == ".jsonl":
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if text_field not in rec:
                raise SystemExit(f"error: {path}:{lineno} has no {text_field!r} field")
            texts.append(str(rec.pop(text_field)))
            if id_field in rec:
                ids.append(str(rec.pop(id_field)))
            metas.append(rec)
    else:
        texts = [line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        metas = [{} for _ in texts]
    if ids and len(ids) != len(texts):
        raise SystemExit(f"error: some records in {path} have {id_field!r} and some do not")
    return texts, ids or None, metas


# ------------------------------------------------------------------ commands --
def cmd_info(args: argparse.Namespace) -> int:
    from clmkit.registry import ENCODERS, INDEXES, LOSSES, RERANKERS
    from clmkit.utils import is_available

    info: dict[str, Any] = {
        "clmkit": __version__,
        "python": sys.version.split()[0],
        "backends": {m: is_available(m) for m in ("torch", "transformers", "peft", "faiss", "fastapi", "mcp", "yaml")},
        "encoders": ENCODERS.names(),
        "indexes": INDEXES.names(),
        "rerankers": RERANKERS.names(),
        "losses": LOSSES.names(),
    }
    if info["backends"]["torch"]:
        from clmkit.utils import resolve_device

        info["device"] = resolve_device("auto")
    print(json.dumps(info, indent=2))
    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    import numpy as np

    texts = _read_texts(args.input, args.texts, args.text_field)
    encoder = _build_encoder(_encoder_spec(args))
    emb = encoder.encode(texts, kind=args.kind, instruction=args.instruction, dim=args.dim, batch_size=args.batch_size)
    if args.output and args.output.endswith(".npy"):
        np.save(args.output, emb, allow_pickle=False)
        print(f"wrote {emb.shape} embeddings to {args.output}", file=sys.stderr)
    else:
        out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout  # noqa: SIM115
        try:
            for text, vec in zip(texts, emb, strict=True):
                out.write(json.dumps({"text": text, "embedding": [round(float(x), 6) for x in vec]}) + "\n")
        finally:
            if out is not sys.stdout:
                out.close()
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from clmkit.registry import INDEXES
    from clmkit.retrieval import Retriever
    from clmkit.text import chunk_text

    texts, ids, metas = _read_documents(args.input, args.text_field, args.id_field)
    if args.chunk_size:
        c_texts, c_ids, c_metas = [], [], []
        for i, (text, meta) in enumerate(zip(texts, metas, strict=True)):
            parent = ids[i] if ids else str(i)
            for j, chunk in enumerate(chunk_text(text, args.chunk_size, args.chunk_overlap)):
                c_texts.append(chunk)
                c_ids.append(f"{parent}#{j}")
                c_metas.append({**meta, "parent_id": parent, "chunk": j})
        texts, ids, metas = c_texts, c_ids, c_metas
    spec = _encoder_spec(args)
    _validate_persisted_encoder_spec(spec)
    encoder = _build_encoder(spec)
    retriever = Retriever(encoder, INDEXES.build(args.index_type, encoder.dim), query_instruction=args.instruction)
    retriever.add(texts, ids=ids, metadata=metas, batch_size=args.batch_size)
    retriever.save(args.output, extra_meta={"encoder_spec": spec})
    print(f"indexed {len(retriever)} documents into {args.output}", file=sys.stderr)
    return 0


def _load_retriever(index_dir: str, args: argparse.Namespace) -> Any:
    from clmkit.retrieval import Retriever

    spec: dict[str, Any] | None
    if args.model:
        spec = _encoder_spec(args)
    else:
        spec = Retriever.read_meta(index_dir).get("encoder_spec")
        if not spec:
            raise SystemExit("error: index has no stored encoder spec; pass --model")
    reranker = None
    if getattr(args, "reranker", None):
        from clmkit.rerank import load_reranker

        reranker = load_reranker(args.reranker)
    return Retriever.load(index_dir, _build_encoder(spec), reranker=reranker)


def cmd_search(args: argparse.Namespace) -> int:
    retriever = _load_retriever(args.index, args)
    hits = retriever.search(args.query, k=args.k, instruction=args.instruction)
    if args.json:
        print(json.dumps([h.to_dict() for h in hits], ensure_ascii=False, indent=2))
    else:
        for rank, h in enumerate(hits, start=1):
            snippet = (h.text or "").replace("\n", " ")
            print(f"{rank:>2}. [{h.score:.4f}] {h.id}  {snippet[:160]}")
    return 0


def cmd_mine(args: argparse.Namespace) -> int:
    from clmkit.data import load_examples, mine_hard_negatives, save_examples

    examples = load_examples(args.train)
    corpus = _read_texts(args.corpus, [], args.text_field)
    mined = mine_hard_negatives(
        _build_encoder(_encoder_spec(args)),
        examples,
        corpus,
        num_negatives=args.num_negatives,
        skip_top=args.skip_top,
        max_relative_score=args.max_relative_score,
        batch_size=args.batch_size,
    )
    save_examples(args.output, mined)
    print(f"wrote {len(mined)} examples with hard negatives to {args.output}", file=sys.stderr)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from clmkit.config import load_config
    from clmkit.training import train_from_config

    cfg = load_config(args.config)
    for key, value in _kv(args.set).items():  # --set train.learning_rate=1e-5
        section, _, name = key.partition(".")
        if not name:
            raise SystemExit(f"error: --set keys look like section.option, got {key!r}")
        cfg.setdefault(section, {})[name] = value
    result = train_from_config(cfg)
    print(
        json.dumps(
            {
                "global_step": result.global_step,
                "train_loss": result.train_loss,
                "eval": result.eval,
                "output_dir": result.output_dir,
            },
            indent=2,
        )
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from clmkit.eval import RetrievalEvaluator, load_beir

    ks = [int(k) for k in args.ks.split(",")]
    if args.beir:
        queries, corpus, qrels = load_beir(args.beir, args.split)
        evaluator = RetrievalEvaluator(
            queries, corpus, qrels, ks=ks, instruction=args.instruction, batch_size=args.batch_size
        )
    elif args.data:
        from clmkit.data import load_examples

        options = {"instruction": args.instruction} if args.instruction is not None else {}
        evaluator = RetrievalEvaluator.from_examples(
            load_examples(args.data), ks=ks, batch_size=args.batch_size, **options
        )
    else:
        raise SystemExit("error: pass --data or --beir")
    metrics = evaluator(_build_encoder(_encoder_spec(args)))
    text = json.dumps(metrics, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - long-running
    from clmkit.utils import require

    uvicorn = require("uvicorn")
    from clmkit.retrieval import Retriever
    from clmkit.serve.app import create_app

    if args.index:
        spec = _encoder_spec(args) if args.model else Retriever.read_meta(args.index).get("encoder_spec")
        if spec is not None and args.allow_writes and args.save_on_exit:
            _validate_persisted_encoder_spec(spec)
        retriever = _load_retriever(args.index, args)
        encoder, reranker = retriever.encoder, retriever.reranker
    else:
        retriever = None
        encoder = _build_encoder(_encoder_spec(args))
        reranker = None
        if args.reranker:
            from clmkit.rerank import load_reranker

            reranker = load_reranker(args.reranker)
    api_key = os.environ.get(args.api_key_env) or None
    if args.host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        print(f"warning: serving on {args.host} without auth; set ${args.api_key_env}", file=sys.stderr)
    app = create_app(encoder, retriever=retriever, reranker=reranker, api_key=api_key, allow_writes=args.allow_writes)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    if retriever is not None and args.allow_writes and args.save_on_exit:
        retriever.save(args.index, extra_meta={"encoder_spec": spec})
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:  # pragma: no cover - needs an MCP client
    from clmkit.agents.tools import retriever_tools
    from clmkit.serve.mcp_server import run_stdio

    retriever = _load_retriever(args.index, args)
    run_stdio(retriever_tools(retriever, allow_write=args.allow_writes))
    return 0


# -------------------------------------------------------------------- parser --
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clmkit", description="Contrastive Language Model toolkit")
    p.add_argument("--version", action="version", version=f"clmkit {__version__}")
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="command", required=True)

    def model_args(sp: argparse.ArgumentParser, required: bool = True) -> None:
        sp.add_argument(
            "--model",
            required=required,
            help="'hashing', 'openai:<model>', or a Hugging Face id/path (e.g. Qwen/Qwen3-Embedding-8B)",
        )
        sp.add_argument(
            "--model-arg",
            action="append",
            metavar="KEY=VALUE",
            help="encoder option, repeatable (e.g. dim=256, dtype=bfloat16, device=cuda)",
        )
        sp.add_argument("--batch-size", type=int, default=32)

    sp = sub.add_parser("info", help="show version, backends and registered components")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("encode", help="embed texts")
    model_args(sp)
    sp.add_argument("texts", nargs="*")
    sp.add_argument("--input", help="text file (one per line), .jsonl, or '-' for stdin")
    sp.add_argument("--text-field", default="text")
    sp.add_argument("--kind", choices=["query", "document"], default="document")
    sp.add_argument("--instruction")
    sp.add_argument("--dim", type=int, help="Matryoshka truncation")
    sp.add_argument("--output", help=".npy or .jsonl (default: JSONL to stdout)")
    sp.set_defaults(func=cmd_encode)

    sp = sub.add_parser("index", help="build a searchable index from documents")
    model_args(sp)
    sp.add_argument("--input", required=True, help=".txt (one doc per line) or .jsonl")
    sp.add_argument("--output", required=True)
    sp.add_argument("--text-field", default="text")
    sp.add_argument("--id-field", default="id")
    sp.add_argument("--index-type", default="numpy", choices=["numpy", "faiss"])
    sp.add_argument("--chunk-size", type=int, default=0, help="chunk long docs to N chars (0 = off)")
    sp.add_argument("--chunk-overlap", type=int, default=100)
    sp.add_argument("--instruction", help="default query instruction stored with the index")
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("search", help="query an index")
    model_args(sp, required=False)
    sp.add_argument("--index", required=True)
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=5)
    sp.add_argument("--instruction")
    sp.add_argument("--reranker", help="reranker id, e.g. Qwen/Qwen3-Reranker-0.6B")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("mine", help="mine hard negatives for training data")
    model_args(sp)
    sp.add_argument("--train", required=True)
    sp.add_argument("--corpus", required=True, help=".txt (one doc per line) or .jsonl")
    sp.add_argument("--text-field", default="text")
    sp.add_argument("--output", required=True)
    sp.add_argument("-n", "--num-negatives", type=int, default=4)
    sp.add_argument("--skip-top", type=int, default=0)
    sp.add_argument("--max-relative-score", type=float, default=0.95)
    sp.set_defaults(func=cmd_mine)

    sp = sub.add_parser("train", help="fine-tune from a YAML/JSON config")
    sp.add_argument("--config", required=True)
    sp.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE", help="override a config value")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("eval", help="retrieval evaluation (nDCG, MRR, Recall, MAP)")
    model_args(sp)
    sp.add_argument("--data", help="JSONL contrastive examples")
    sp.add_argument("--beir", help="BEIR-format dataset directory")
    sp.add_argument("--split", default="test")
    sp.add_argument("--ks", default="1,5,10")
    sp.add_argument("--instruction")
    sp.add_argument("--output")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("serve", help="run the REST API (OpenAI-compatible /v1/embeddings)")
    model_args(sp, required=False)
    sp.add_argument("--index")
    sp.add_argument("--reranker")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--api-key-env", default="CLMKIT_API_KEY", help="env var holding the bearer token")
    sp.add_argument("--allow-writes", action="store_true")
    sp.add_argument("--save-on-exit", action="store_true", help="persist writes to --index on shutdown")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("mcp", help="run an MCP (Model Context Protocol) server over stdio")
    model_args(sp, required=False)
    sp.add_argument("--index", required=True)
    sp.add_argument("--allow-writes", action="store_true")
    sp.set_defaults(func=cmd_mcp)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if args.command == "serve" and not (args.model or args.index):
        parser.error("serve needs --model or --index")
    if getattr(args, "model_arg", None) and not args.model:
        parser.error("--model-arg requires --model; otherwise the saved encoder spec is used")
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
