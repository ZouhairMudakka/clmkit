"""Source-checkout examples for BANKING77 matching and separately calibrated CLINC routing."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.evidence_data import load_prepared
from validation.evidence_run import digest, load_model, require_cloud

from clmkit import Retriever


def prepared_gallery(data_root: Path, protocol_path: Path, dataset: str) -> tuple[dict, dict]:
    """Check provenance and load the complete training gallery; never open test."""
    if dataset not in {"banking77", "clinc150"}:
        raise ValueError("This example supports banking77 and clinc150")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    folder = data_root / dataset
    manifest_hash = digest(folder / "manifest.json")
    if manifest_hash != protocol["dataset_manifests"][dataset]:
        raise ValueError("Prepared manifest differs from the pinned protocol")
    data = load_prepared(folder, "dev")
    corpus, manifest = data["corpus"], data["manifest"]
    if (
        manifest["dataset"] != dataset
        or len(corpus) != manifest["processed_counts"]["corpus"]
        or [row["id"] for row in corpus] != manifest["corpus_ids"]
    ):
        raise ValueError("The complete registered gallery is required")
    if not corpus or any(not row.get("in_scope", True) or not isinstance(row.get("label"), str) for row in corpus):
        raise ValueError("Gallery must contain labeled in-scope training examples only")
    contract = {
        "dataset": dataset,
        "protocol_sha256": digest(protocol_path),
        "manifest_sha256": manifest_hash,
        "corpus_sha256": manifest["files"]["corpus.jsonl"]["sha256"],
        "corpus_documents": len(corpus),
    }
    return data, contract


def _check_documents(retriever: Retriever, corpus: list[dict]) -> None:
    if set(retriever.documents) != {row["id"] for row in corpus}:
        raise ValueError("Snapshot does not contain the full prepared gallery")
    for row in corpus:
        document = retriever.documents[row["id"]]
        if document.text != row["text"] or document.metadata.get("label") != row["label"]:
            raise ValueError("Snapshot document text/label differs from the prepared gallery")


def open_index(path: Path, encoder, identity: str, corpus: list[dict], contract: dict) -> Retriever:
    """Reject gallery/model/config drift before any query is encoded."""
    if Retriever.read_meta(path).get("intent_example") != contract:
        raise ValueError("Snapshot belongs to a different protocol or gallery; rebuild into a fresh directory")
    retriever = Retriever.load(path, encoder, strict=True, encoder_identity=identity)
    _check_documents(retriever, corpus)
    return retriever


def build_index(path: Path, encoder, identity: str, corpus: list[dict], contract: dict) -> Retriever:
    """Stage and strictly reload a complete new snapshot before naming it usable.

    This is a single-writer recipe, not a transactional/concurrent index service.
    Failed staging directories are preserved for inspection; existing indexes are
    never overwritten. Switch consumers to the new path only after success.
    """
    path = path.resolve()
    if path.exists():
        raise FileExistsError("Choose a fresh index directory for each model/gallery rebuild")
    if len(corpus) != contract["corpus_documents"]:
        raise ValueError("Cannot build a partial gallery")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{path.name}-staging-", dir=path.parent))
    retriever = Retriever(encoder, encoder_identity=identity)
    retriever.add(
        [row["text"] for row in corpus],
        ids=[row["id"] for row in corpus],
        metadata=[{"label": row["label"]} for row in corpus],
    )
    retriever.save(stage, extra_meta={"intent_example": contract})
    checked = open_index(stage, encoder, identity, corpus, contract)
    if path.exists():
        raise FileExistsError("Target appeared during staging; preserve it and choose another directory")
    stage.rename(path)
    return checked


def calibrated_threshold(path: Path, seal_path: Path, contract: dict, identity: str, model: str = "minilm") -> float:
    """Only accept this CLINC gallery/model's full-development, sealed threshold."""
    if contract["dataset"] != "clinc150":
        raise ValueError("A CLINC threshold cannot calibrate BANKING77")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("schema") != 2
        or not seal.get("frozen_before_test")
        or seal.get("protocol_sha256") != contract["protocol_sha256"]
        or digest(path) not in seal.get("threshold_files", {}).values()
    ):
        raise ValueError("Threshold is not bound to the matching frozen protocol")
    threshold = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "clinc150",
        "method": "dense",
        "model": model,
        "encoder_identity": identity,
        "protocol_sha256": contract["protocol_sha256"],
        "manifest_sha256": contract["manifest_sha256"],
        "query_limit": 0,
        "selection_split": "dev",
    }
    if any(threshold.get(key) != value for key, value in expected.items()):
        raise ValueError("Threshold does not match this CLINC dataset/model/gallery configuration")
    value = threshold["selected"]["threshold"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Threshold must be finite")
    return float(value)


def match(retriever: Retriever, query: str, dataset: str, *, threshold: float | None = None, k: int = 5) -> dict:
    if dataset not in {"banking77", "clinc150"}:
        raise ValueError("Unknown dataset")
    if (dataset == "clinc150") != (threshold is not None):
        raise ValueError("CLINC requires its calibrated threshold; BANKING77 has no validated rejection threshold")
    hits = retriever.search(query, k=k)
    accepted = bool(hits) and (threshold is None or hits[0].score >= threshold)
    return {
        "dataset": dataset,
        "gallery_documents": len(retriever),
        "encoder_identity": retriever.encoder_identity,
        "status": ("matched" if accepted else "no_match")
        if dataset == "banking77"
        else ("accepted" if accepted else "rejected"),
        "intent": hits[0].metadata["label"] if accepted else None,
        "threshold": threshold,
        "matches": [
            {"id": hit.id, "label": hit.metadata["label"], "text": hit.text, "score": hit.score} for hit in hits
        ],
        "action_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "query"])
    parser.add_argument("--dataset", choices=["banking77", "clinc150"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", choices=["minilm", "checkpoint"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threshold", type=Path)
    parser.add_argument("--seal", type=Path)
    parser.add_argument("--text")
    args = parser.parse_args()
    if (args.model == "checkpoint") != (args.checkpoint is not None):
        parser.error("--model checkpoint requires --checkpoint; pinned minilm does not accept one")
    if args.dataset == "clinc150" and args.checkpoint:
        parser.error("Adaptation is registered for BANKING77; CLINC uses its own pinned baseline")
    if args.command == "query" and not args.text:
        parser.error("query requires --text")
    if args.dataset == "banking77" and (args.threshold or args.seal):
        parser.error("Do not transfer CLINC rejection calibration to BANKING77")
    if args.command == "query" and args.dataset == "clinc150" and not (args.threshold and args.seal):
        parser.error("CLINC query requires its --threshold and matching --seal")
    require_cloud()  # This evidence recipe keeps dataset/model workloads in the authorized Codespace.
    data, contract = prepared_gallery(args.data_root, args.protocol, args.dataset)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    encoder, identity = load_model(protocol, "minilm", args.dataset, args.checkpoint)
    if args.command == "build":
        retriever = build_index(args.index, encoder, identity, data["corpus"], contract)
        print(json.dumps({"index": str(args.index), "gallery_documents": len(retriever), "encoder_identity": identity}))
    else:
        retriever = open_index(args.index, encoder, identity, data["corpus"], contract)
        threshold = (
            calibrated_threshold(args.threshold, args.seal, contract, identity) if args.dataset == "clinc150" else None
        )
        print(json.dumps(match(retriever, args.text, args.dataset, threshold=threshold), indent=2))


if __name__ == "__main__":
    main()
