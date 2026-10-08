"""Pinned, stdlib-only preparation for the development evidence milestone.

Downloads are permitted only inside Codespaces. No raw data belongs in Git.
The test files are prepared for reproducibility, but load_prepared refuses test
unless the caller explicitly opts in after the experiment freeze.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "evidence-data-v1"
BANKING_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
CLINC_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
SCIFACT_LICENSE_REVISION = "68b98a56d93e0f9da0d2aab4e6c3294699a0f72e"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_MD5 = "5f7d1de60b170fc8027bb7898e2efca1"
CHECKSUM_SOURCE = "https://github.com/beir-cellar/beir/wiki/Datasets-available"
GROUP_RULE = (
    "NFKC + casefold + Unicode word tokens; identical sorted token multisets form a group. "
    "This groups punctuation/case/spacing changes and token reorderings, retains numbers/negations, "
    "and does not claim to detect semantic paraphrases or all near duplicates."
)
RIGHTS = {
    "banking77": {
        "license": "CC BY 4.0",
        "attribution": "PolyAI; Casanueva et al., Efficient Intent Detection with Dual Sentence Encoders (2020)",
        "source": "https://github.com/PolyAI-LDN/task-specific-datasets",
        "limitation": "Intent labels are relevance proxies; collection is not verified production ticket data.",
    },
    "clinc150": {
        "license": "CC BY 3.0",
        "attribution": (
            "Larson et al., An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction (2019)"
        ),
        "source": "https://github.com/clinc/oos-eval",
        "limitation": "Crowdworker intent examples; OOS is never a semantic positive class.",
    },
    "scifact": {
        "license": "Claims/annotations CC BY 4.0; abstracts ODC-By 1.0; upstream code Apache-2.0",
        "attribution": "Wadden et al., Fact or Fiction (2020); Thakur et al., BEIR (2021)",
        "source": "https://github.com/allenai/scifact",
        "limitation": "Judged evidence retrieval only; unjudged documents are not established negatives.",
    },
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def fingerprint(text: str) -> str:
    tokens = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    return hashlib.sha256(" ".join(sorted(tokens)).encode()).hexdigest()


def _records(rows: list[Any], prefix: str, *, in_scope: bool = True) -> list[dict]:
    result = []
    for i, row in enumerate(rows):
        text, label = (row["text"], row["category"]) if isinstance(row, dict) else row
        if not isinstance(text, str) or not text.strip() or not isinstance(label, str):
            raise ValueError(f"Malformed {prefix} row {i}")
        result.append(
            {
                "id": f"{prefix}-{i:05d}",
                "text": text,
                "label": label,
                "in_scope": in_scope,
                "metadata": {"group_id": fingerprint(text)},
            }
        )
    return result


def grouped_split(
    records: list[dict], *, seed: int = 42, dev_fraction: float = 0.2, group_ids: dict[str, str] | None = None
) -> tuple[list[dict], list[dict]]:
    """Deterministic per-label closest-target greedy split, retaining gallery labels.

    Conflicting-label groups stay in training and are disclosed by the audit.
    A single group for a label cannot be split; its dev count is zero.
    """
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must lie strictly between zero and one")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        key = group_ids[row["id"]] if group_ids is not None else fingerprint(row["text"])
        groups[key].append(row)
    strata: dict[str, list[tuple[str, list[dict]]]] = defaultdict(list)
    for key, rows in groups.items():
        labels = {row.get("label", "") for row in rows}
        if len(labels) == 1:
            strata[next(iter(labels))].append((key, rows))
    dev_ids = set()
    for label, items in sorted(strata.items()):
        items.sort(key=lambda item: hashlib.sha256(f"{seed}:{label}:{item[0]}".encode()).hexdigest())
        total = sum(len(rows) for _, rows in items)
        target = total * dev_fraction
        chosen = 0
        for _, rows in items:
            if chosen + len(rows) < total and abs(chosen + len(rows) - target) < abs(chosen - target):
                dev_ids.update(row["id"] for row in rows)
                chosen += len(rows)
    return ([r for r in records if r["id"] not in dev_ids], [r for r in records if r["id"] in dev_ids])


def _audit(splits: dict[str, list[dict]]) -> dict:
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for split, records in splits.items():
        for record in records:
            groups[fingerprint(record["text"])].append((split, record))
    duplicated = {key: rows for key, rows in groups.items() if len(rows) > 1}
    return {
        "rule": GROUP_RULE,
        "duplicate_groups": len(duplicated),
        "duplicate_records": sum(len(rows) for rows in duplicated.values()),
        "cross_split_groups": [
            {"group_id": key, "members": [{"split": split, "id": row["id"]} for split, row in rows]}
            for key, rows in sorted(duplicated.items())
            if len({split for split, _ in rows}) > 1
        ],
        "conflicting_label_groups": [
            {
                "group_id": key,
                "ids": [row["id"] for _, row in rows],
                "labels": sorted({row.get("label", "") for _, row in rows}),
            }
            for key, rows in sorted(duplicated.items())
            if len({row.get("label", "") for _, row in rows}) > 1
        ],
        "policy": (
            "No official test/validation rows excluded. Overlap IDs support separately labeled sensitivity reports."
        ),
    }


def _intent_qrels(corpus: list[dict], queries: dict[str, list[dict]]) -> dict:
    labels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in corpus:
        labels[row["label"]][row["id"]] = 1
    return {
        split: {
            q["id"]: (
                {doc: score for doc, score in labels[q["label"]].items() if doc != q["id"]} if q["in_scope"] else {}
            )
            for q in rows
        }
        for split, rows in queries.items()
    }


def prepare_banking77(train_rows: list, test_rows: list, *, seed: int = 42, dev_fraction: float = 0.2) -> dict:
    """Pure adapter for pinned CSV rows (also usable with tiny synthetic fixtures)."""
    official_train = _records(train_rows, "banking77-train")
    train, dev = grouped_split(official_train, seed=seed, dev_fraction=dev_fraction)
    test = _records(test_rows, "banking77-test")
    splits = {"train": train, "dev": dev, "test": test}
    return _bundle(
        "banking77",
        train,
        splits,
        _intent_qrels(train, splits),
        {"official_train": len(official_train), "official_test": len(test)},
        seed,
        dev_fraction,
        "Same-intent proxy relevance; train qrels exclude self; no actual resolutions or duplicate-ticket gold.",
    )


def prepare_clinc150(data: dict) -> dict:
    """Keep official full splits; OOS rows remain queries and never enter the gallery."""
    splits = {}
    for original, target in [("train", "train"), ("val", "dev"), ("test", "test")]:
        splits[target] = _records(data[original], f"clinc150-{original}") + _records(
            data[f"oos_{original}"], f"clinc150-oos-{original}", in_scope=False
        )
    corpus = [row for row in splits["train"] if row["in_scope"]]
    return _bundle(
        "clinc150",
        corpus,
        splits,
        _intent_qrels(corpus, splits),
        {key: len(data[key]) for key in ("train", "val", "test", "oos_train", "oos_val", "oos_test")},
        None,
        None,
        "Same-intent proxy relevance for in-scope queries; OOS qrels empty; preserve denominator.",
    )


def prepare_scifact(
    corpus: list[dict],
    queries: list[dict],
    train_qrels: dict,
    test_qrels: dict,
    *,
    seed: int = 42,
    dev_fraction: float = 0.2,
) -> dict:
    """Split BEIR train by connected components of shared positive evidence/text."""
    corpus = [
        {
            "id": str(r["_id"]),
            "text": " ".join(x for x in (r.get("title", ""), r["text"]) if x),
            "metadata": r.get("metadata", {}),
        }
        for r in corpus
    ]
    query_map = {str(r["_id"]): {"id": str(r["_id"]), "text": r["text"]} for r in queries}
    if len(query_map) != len(queries):
        raise ValueError("Duplicate SciFact query IDs")
    if set(train_qrels) & set(test_qrels):
        raise ValueError("SciFact train/test query ID overlap")
    missing = (set(train_qrels) | set(test_qrels)) - set(query_map)
    if missing:
        raise ValueError(f"SciFact qrels reference missing queries: {sorted(missing)[:5]}")
    parent = {qid: qid for qid in train_qrels}

    def root(qid: str) -> str:
        while parent[qid] != qid:
            parent[qid] = parent[parent[qid]]
            qid = parent[qid]
        return qid

    seen = {}
    for qid in sorted(train_qrels):
        keys = ["text:" + fingerprint(query_map[qid]["text"])]
        keys += ["doc:" + doc for doc, score in train_qrels[qid].items() if score > 0]
        for key in keys:
            if key in seen:
                a, b = root(qid), root(seen[key])
                parent[max(a, b)] = min(a, b)
            seen[key] = qid
    group_ids = {qid: root(qid) for qid in parent}
    train, dev = grouped_split(
        [query_map[qid] for qid in sorted(train_qrels)], seed=seed, dev_fraction=dev_fraction, group_ids=group_ids
    )
    splits = {"train": train, "dev": dev, "test": [query_map[qid] for qid in sorted(test_qrels)]}
    for records in splits.values():
        for record in records:
            record["metadata"] = {"group_id": group_ids.get(record["id"], fingerprint(record["text"]))}
    qrels = {
        split: {r["id"]: (test_qrels if split == "test" else train_qrels)[r["id"]] for r in records}
        for split, records in splits.items()
    }
    bundle = _bundle(
        "scifact",
        corpus,
        splits,
        qrels,
        {
            "corpus": len(corpus),
            "queries_file": len(queries),
            "beir_train": len(train_qrels),
            "beir_test": len(test_qrels),
            "queries_without_split_qrels": len(query_map.keys() - train_qrels.keys() - test_qrels.keys()),
        },
        seed,
        dev_fraction,
        "Original BEIR graded judgments, all relevant documents preserved.",
    )
    bundle["manifest"]["split_policy"] = (
        "BEIR training-only connected components of shared positive evidence or text fingerprint; "
        "20% target is approximate. Full corpus retained. Test stays official. "
        "Related claims without shared evidence/text may remain across splits."
    )
    bundle["manifest"]["training_evidence_groups"] = {qid: group_ids[qid] for qid in sorted(group_ids)}
    bundle["manifest"]["exclusions"] = [
        {"id": qid, "reason": "No original BEIR train/test qrels membership"}
        for qid in sorted(query_map.keys() - train_qrels.keys() - test_qrels.keys())
    ]
    return bundle


def _bundle(
    dataset: str,
    corpus: list,
    queries: dict,
    qrels: dict,
    original: dict,
    seed: int | None,
    fraction: float | None,
    relevance: str,
) -> dict:
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "rights": dict(RIGHTS[dataset]),
        "original_counts": original,
        "processed_counts": {
            "corpus": len(corpus),
            "queries": {split: len(rows) for split, rows in queries.items()},
            "in_scope": {split: sum(r.get("in_scope", True) for r in rows) for split, rows in queries.items()},
        },
        "seed": seed,
        "dev_fraction": fraction,
        "relevance": relevance,
        "test_sealed": True,
        "preprocessing": "Text preserved; SciFact concatenates title and text with one space.",
        "split_policy": "Official splits for CLINC; grouped intent-stratified training-derived dev for BANKING77.",
        "audit": _audit(queries),
        "exclusions": [],
        "label_counts": {
            split: dict(sorted(Counter(r.get("label", "") for r in rows).items())) for split, rows in queries.items()
        },
        "split_ids": {split: [r["id"] for r in rows] for split, rows in queries.items()},
        "corpus_ids": [r["id"] for r in corpus],
        "transformations": (
            "Derived retrieval gallery, ID assignment, split mapping and relevance conversion as documented."
        ),
    }
    bundle = {"corpus": corpus, "queries": queries, "qrels": qrels, "manifest": manifest}
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict) -> None:
    corpus = bundle["corpus"]
    doc_ids = {r["id"] for r in corpus}
    if len(doc_ids) != len(corpus):
        raise ValueError("Duplicate corpus IDs")
    all_query_ids = set()
    for split, queries in bundle["queries"].items():
        ids = {q["id"] for q in queries}
        if len(ids) != len(queries) or ids & all_query_ids:
            raise ValueError(f"Duplicate query IDs in/across {split}")
        all_query_ids.update(ids)
        if set(bundle["qrels"][split]) != ids:
            raise ValueError(f"qrels must cover every query, including no-positive queries: {split}")
        for qid, judgments in bundle["qrels"][split].items():
            if set(judgments) - doc_ids:
                raise ValueError(f"Missing corpus positives for query {qid}")
            if any(
                not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0
                for score in judgments.values()
            ):
                raise ValueError(f"Invalid relevance score for query {qid}")


def write_prepared(bundle: dict, output_dir: str | Path) -> dict:
    """Write immutable prepared files; refuse changing a previously frozen artifact."""
    validate_bundle(bundle)
    output = Path(output_dir)
    files = {
        "corpus.jsonl": "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in bundle["corpus"])
    }
    for split, rows in bundle["queries"].items():
        files[f"queries/{split}.jsonl"] = "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows
        )
        files[f"qrels/{split}.json"] = _json(bundle["qrels"][split])
    manifest = dict(bundle["manifest"])
    manifest["files"] = {
        name: {"sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text.encode())}
        for name, text in files.items()
    }
    files["manifest.json"] = _json(manifest)
    for name, text in files.items():
        path = output / name
        if path.exists() and path.read_bytes() != text.encode():
            raise ValueError(f"Refusing to change frozen file: {path}")
    for name, text in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
    return manifest


def load_prepared(output_dir: str | Path, split: str = "dev", *, allow_test: bool = False) -> dict:
    if split == "test" and not allow_test:
        raise PermissionError("Test is sealed. Freeze the protocol before explicitly passing allow_test=True.")
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown split: {split}")
    output = Path(output_dir)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    def checked(name: str) -> str:
        data = (output / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"Prepared file checksum mismatch: {name}")
        return data.decode("utf-8")

    result = {
        "corpus": [json.loads(line) for line in checked("corpus.jsonl").splitlines()],
        "queries": [json.loads(line) for line in checked(f"queries/{split}.jsonl").splitlines()],
        "qrels": json.loads(checked(f"qrels/{split}.json")),
        "manifest": manifest,
    }
    validate_bundle({**result, "queries": {split: result["queries"]}, "qrels": {split: result["qrels"]}})
    return result


def _github(repo: str, revision: str, path: str, git_blob: str, size: int) -> dict:
    return {
        "url": f"https://raw.githubusercontent.com/{repo}/{revision}/{path}",
        "revision": revision,
        "git_blob_sha1": git_blob,
        "expected_bytes": size,
    }


def sources(dataset: str) -> dict:
    if dataset == "banking77":
        return {
            "train.csv": _github(
                "PolyAI-LDN/task-specific-datasets",
                BANKING_REVISION,
                "banking_data/train.csv",
                "98e2543cf482d0dca7bfb175ebe35d98efad95be",
                839073,
            ),
            "test.csv": _github(
                "PolyAI-LDN/task-specific-datasets",
                BANKING_REVISION,
                "banking_data/test.csv",
                "799687a8367359432985b8b13d85a2baf73f92dd",
                239961,
            ),
            "LICENSE": _github(
                "PolyAI-LDN/task-specific-datasets",
                BANKING_REVISION,
                "LICENSE",
                "2f244ac814036ecd9ba9f69782e89ce6b1dca9eb",
                18650,
            ),
        }
    if dataset == "clinc150":
        return {
            "data_full.json": _github(
                "clinc/oos-eval",
                CLINC_REVISION,
                "data/data_full.json",
                "7a7b26c5f2dfbbf213f3e67d2dd0727e1af545aa",
                2495390,
            ),
            "LICENSE": _github(
                "clinc/oos-eval", CLINC_REVISION, "LICENSE", "1a16e05564d2aaa880bbe9e506a0a0226d8742cc", 19467
            ),
        }
    if dataset == "scifact":
        return {
            "scifact.zip": {
                "url": SCIFACT_URL,
                "md5": SCIFACT_MD5,
                "checksum_source": CHECKSUM_SOURCE,
                "revision": "BEIR archive identified by published MD5; SHA256 locked after verification",
            },
            "LICENSE.md": _github(
                "allenai/scifact",
                SCIFACT_LICENSE_REVISION,
                "LICENSE.md",
                "18a738e109d7e3b28935a75d2134c5ccd602f403",
                542,
            ),
        }
    raise ValueError(f"Unknown dataset: {dataset}")


def _verify_source(data: bytes, spec: dict) -> dict:
    if "expected_bytes" in spec and len(data) != spec["expected_bytes"]:
        raise ValueError("Source byte count mismatch")
    if "git_blob_sha1" in spec:
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()
        if digest != spec["git_blob_sha1"]:
            raise ValueError("Pinned Git blob checksum mismatch")
    if "md5" in spec and hashlib.md5(data, usedforsecurity=False).hexdigest() != spec["md5"]:
        raise ValueError("Published BEIR MD5 checksum mismatch")
    return {**spec, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def fetch_sources(dataset: str, cache_dir: str | Path) -> tuple[dict[str, Path], dict]:
    if os.environ.get("CODESPACES", "").lower() != "true":
        raise RuntimeError(
            "Dataset downloads/preparation must run remotely in the authorized Codespace (CODESPACES=true)."
        )
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache).free < 256 * 1024 * 1024:
        raise RuntimeError("Dataset preparation requires at least 256 MiB free disk")
    paths, verified = {}, {}
    for name, spec in sources(dataset).items():
        path = cache / name
        if not path.exists():
            # URLs are hardcoded HTTPS source pins above; no caller-provided URL is accepted.
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "clmkit-evidence-v1"})  # noqa: S310
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                data = response.read(32 * 1024 * 1024 + 1)
            if len(data) > 32 * 1024 * 1024:
                raise ValueError("Source exceeds the declared 32 MiB per-file budget")
            _verify_source(data, spec)
            path.write_bytes(data)
        verified[name] = _verify_source(path.read_bytes(), spec)
        paths[name] = path
    lock = cache / "source-lock.json"
    lock_text = _json(verified)
    if lock.exists() and lock.read_text(encoding="utf-8") != lock_text:
        raise ValueError("Immutable source-lock mismatch; do not replace existing evidence inputs")
    lock.write_text(lock_text, encoding="utf-8")
    return paths, verified


def _read_qrels(text: str) -> dict:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
        qid, docid = row["query-id"], row["corpus-id"]
        if docid in result[qid]:
            raise ValueError(f"Duplicate qrel pair: {qid}, {docid}")
        result[qid][docid] = int(row["score"])
    return dict(result)


def prepare_dataset(
    dataset: str,
    output_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    seed: int = 42,
    dev_fraction: float = 0.2,
) -> dict:
    output = Path(output_dir)
    paths, provenance = fetch_sources(dataset, cache_dir or output / "sources")
    if dataset == "banking77":
        rows = {
            key: list(csv.DictReader(io.StringIO(paths[key].read_text(encoding="utf-8"))))
            for key in ("train.csv", "test.csv")
        }
        bundle = prepare_banking77(rows["train.csv"], rows["test.csv"], seed=seed, dev_fraction=dev_fraction)
    elif dataset == "clinc150":
        bundle = prepare_clinc150(json.loads(paths["data_full.json"].read_text(encoding="utf-8")))
    else:
        with zipfile.ZipFile(paths["scifact.zip"]) as archive:

            def read(name: str) -> str:
                info = archive.getinfo(f"scifact/{name}")
                if info.file_size > 64 * 1024 * 1024:
                    raise ValueError("Unexpected oversized archive member")
                return archive.read(info).decode("utf-8")

            bundle = prepare_scifact(
                [json.loads(line) for line in read("corpus.jsonl").splitlines()],
                [json.loads(line) for line in read("queries.jsonl").splitlines()],
                _read_qrels(read("qrels/train.tsv")),
                _read_qrels(read("qrels/test.tsv")),
                seed=seed,
                dev_fraction=dev_fraction,
            )
    bundle["manifest"]["sources"] = provenance
    bundle["manifest"]["rights"]["notice_files"] = [name for name in paths if name.startswith("LICENSE")]
    return write_prepared(bundle, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset", choices=sorted(RIGHTS), required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--dev-fraction", type=float, default=0.2)
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.dataset, args.output, cache_dir=args.cache_dir, seed=args.seed, dev_fraction=args.dev_fraction
    )
    print(
        _json(
            {
                "dataset": args.dataset,
                "processed_counts": manifest["processed_counts"],
                "manifest": str(args.output / "manifest.json"),
                "test_sealed": True,
            }
        )
    )


if __name__ == "__main__":
    main()
