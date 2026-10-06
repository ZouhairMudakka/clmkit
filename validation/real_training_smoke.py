"""Bounded real Qwen3-Embedding-0.6B CPU LoRA smoke test for a 4-core/16-GB Codespace.

Run from the repository root, separately from other model tests:
    timeout 1200s python -u validation/real_training_smoke.py --threads 4

Downloads only official, SHA-pinned safetensors/tokenizer/config files at runtime.
One step on two synthetic examples is execution evidence, never a quality result.
No package installs, remote code, merged-model save or product modifications.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid


MODEL = "Qwen/Qwen3-Embedding-0.6B"
GIB = 1024**3


def memory_snapshot():
    """Linux RSS and host/cgroup bounds; no optional monitoring dependency."""
    if platform.system() != "Linux":
        return {}
    import resource

    result = {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            key, value, _ = line.split()
            result[key.rstrip(":")] = int(value) * 1024
    for key, file in (("cgroup_limit", "memory.max"), ("cgroup_current", "memory.current")):
        path = Path("/sys/fs/cgroup") / file
        if path.exists():
            value = path.read_text().strip()
            result[key] = int(value) if value.isdigit() else value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Resolve this ref to an immutable full HF SHA")
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--output-dir", type=Path, help="New/empty artifact directory below validation-results")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.dont_write_bytecode = True
    results_root = root / "validation-results"
    out = (args.output_dir or results_root / (
        "real-training-smoke-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid.uuid4().hex[:6])).resolve()
    if not out.is_relative_to(results_root.resolve()):
        raise ValueError("--output-dir must be below the repository validation-results directory")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing to overwrite existing evidence: {out}")
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    evidence = {
        "status": "running", "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": MODEL, "requested_revision": args.revision, "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()), "artifact_directory": str(out),
        "runtime": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count()},
        "limits": {"optimizer_steps": 1, "synthetic_pairs": 2, "max_length": 48,
                   "threads": args.threads, "dtype": "float32", "lora_rank": 2,
                   "mini_batch_size": 1, "gradient_checkpointing": True,
                   "merged_save": False, "quality_claim": False},
        "limitations": "Synthetic one-step execution/save-reload probe; no held-out quality, improvement, GPU, 4B/8B or production claim.",
        "stages": [],
    }

    def persist(stage):
        evidence["stage"] = stage
        evidence["elapsed_seconds"] = round(time.monotonic() - start, 3)
        evidence["memory"] = memory_snapshot()
        evidence["stages"].append({"stage": stage, "elapsed_seconds": evidence["elapsed_seconds"],
                                   "memory": evidence["memory"]})
        (out / "result.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps({"stage": stage, "status": evidence["status"], "result": str(out / "result.json"),
                          "elapsed_seconds": evidence["elapsed_seconds"]}), flush=True)

    try:
        if platform.system() != "Linux":
            raise RuntimeError("Run this real-weight probe in the Linux Codespace, not on the constrained laptop")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(args.threads)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        evidence["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=10).strip()
        evidence["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src" / "clmkit").rglob("*.py"))}
        evidence["packages"] = {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "peft", "huggingface-hub", "safetensors", "numpy", "tokenizers")}
        persist("preflight")
        mem = evidence["memory"]
        available = mem["MemAvailable"]
        if isinstance(mem.get("cgroup_limit"), int) and isinstance(mem.get("cgroup_current"), int):
            available = min(available, mem["cgroup_limit"] - mem["cgroup_current"])
        if available < 8 * GIB:
            raise RuntimeError(f"Need at least 8 GiB available memory before loading; available={available}")
        # Use the normal HF cache so a previously pinned embedding download can be reused.
        from huggingface_hub import model_info, snapshot_download
        from huggingface_hub.constants import HF_HUB_CACHE
        cache = Path(HF_HUB_CACHE)
        cache.mkdir(parents=True, exist_ok=True)
        evidence["cache_directory"] = str(cache)
        if shutil.disk_usage(cache).free < 4 * GIB:
            raise RuntimeError("Need at least 4 GiB free in the HF cache filesystem")
        info = model_info(MODEL, revision=args.revision)
        revision = info.sha
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise RuntimeError(f"Expected immutable 40-character source revision, got {revision!r}")
        evidence["model_revision"] = revision
        names = [entry.rfilename for entry in info.siblings if "/" not in entry.rfilename
                 and entry.rfilename.endswith((".json", ".safetensors", ".txt", ".model"))]
        if not any(name.endswith(".safetensors") for name in names):
            raise RuntimeError("Official revision has no root safetensors weights")
        evidence["download_allowlist"] = sorted(names)
        persist("downloading-pinned-official-snapshot")
        snapshot = Path(snapshot_download(MODEL, revision=revision, allow_patterns=names, max_workers=2))
        evidence["snapshot_directory"] = str(snapshot)
        evidence["weight_files"] = [{"name": p.name, "bytes": p.stat().st_size}
                                    for p in sorted(snapshot.glob("*.safetensors"))]
        if not evidence["weight_files"]:
            raise RuntimeError("Safetensors snapshot is incomplete")
        persist("loading-cpu-encoder")
        import numpy as np
        import torch
        import peft
        import transformers
        from clmkit.data import ContrastiveExample
        from clmkit.encoders.hf import HFEncoder
        from clmkit.training import ContrastiveTrainer, LoraSettings, TrainConfig

        evidence["imported_modules"] = {module.__name__: {"version": module.__version__, "file": module.__file__}
                                        for module in (np, torch, peft, transformers)}
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(1847)
        evidence["runtime"].update({"torch_threads": torch.get_num_threads(),
                                   "torch_interop_threads": torch.get_num_interop_threads(),
                                   "cuda_available": torch.cuda.is_available()})
        loader = {"device": "cpu", "dtype": "float32", "trust_remote_code": False,
                  "model_kwargs": {"use_safetensors": True, "local_files_only": True,
                                   "attn_implementation": "eager"},
                  "tokenizer_kwargs": {"local_files_only": True}}
        # Snapshot path includes the official repo name, preserving the Qwen preset.
        encoder = HFEncoder(str(snapshot), max_length=48, **loader)
        assert encoder.pooling == "last_token" and encoder.padding_side == "left"
        examples = [ContrastiveExample("Where do penguins live?", "Penguins live in Antarctica."),
                    ContrastiveExample("What is Python?", "Python is a programming language.")]
        probes = [e.query for e in examples]
        documents = [e.positive for e in examples]

        def embeddings(enc):
            query = enc.encode(probes, kind="query", batch_size=1)
            document = enc.encode(documents, kind="document", batch_size=1)
            assert np.isfinite(query).all() and np.isfinite(document).all()
            assert query.shape == document.shape == (2, enc.dim)
            return query, document

        def assert_finite_parameters(model):
            count = 0
            for name, parameter in model.named_parameters():
                # Avoid allocating a full-size boolean tensor for the vocabulary matrix.
                for chunk in parameter.detach().reshape(-1).split(1_000_000):
                    assert bool(torch.isfinite(chunk).all()), f"Non-finite parameter: {name}"
                count += parameter.numel()
            return count

        before_query, before_document = embeddings(encoder)
        evidence["parameter_count"] = assert_finite_parameters(encoder.model)
        cfg = TrainConfig(output_dir=str(out / "training"), max_steps=1, batch_size=2,
            mini_batch_size=1, max_negatives=0, learning_rate=1e-4, warmup_ratio=0.0,
            weight_decay=0.0, precision="fp32", gradient_checkpointing=True,
            lora=LoraSettings(r=2, alpha=4, dropout=0.0, target_modules=["q_proj", "v_proj"]),
            merge_lora_on_save=False, log_every=1, seed=1847)
        trainer = ContrastiveTrainer(encoder, cfg, examples)
        trainable = {name: p for name, p in encoder.model.named_parameters() if p.requires_grad}
        assert trainable and all("lora_" in name for name in trainable)
        initial_adapters = {name: p.detach().clone() for name, p in trainable.items()}
        evidence["trainable_parameter_count"] = sum(p.numel() for p in trainable.values())
        evidence["trainable_parameter_names"] = list(trainable)
        persist("training-one-step")
        result = trainer.train()
        assert result.global_step == 1 and math.isfinite(result.train_loss)
        assert_finite_parameters(encoder.model)
        updates = {name: float((p.detach() - initial_adapters[name]).abs().max())
                   for name, p in trainable.items()}
        assert any(delta > 0 for delta in updates.values()), "No adapter tensor changed"
        after_query, after_document = embeddings(encoder)
        adapter = out / "training" / "final"
        assert (adapter / "adapter_model.safetensors").is_file()
        assert (adapter / "adapter_config.json").is_file()
        assert not (adapter / "config.json").exists(), "Expected adapter-only save"
        adapter_config = json.loads((adapter / "adapter_config.json").read_text())
        assert Path(adapter_config["base_model_name_or_path"]).resolve() == snapshot.resolve()
        evidence["training"] = {"steps": result.global_step, "loss": result.train_loss,
            "adapter_max_abs_updates": updates, "adapter_directory": str(adapter),
            "query_embedding_max_abs_change": float(np.max(np.abs(after_query - before_query))),
            "document_embedding_max_abs_change": float(np.max(np.abs(after_document - before_document))),
            "all_parameters_finite": True, "nonzero_adapter_update": True,
            "adapter_base_matches_pinned_snapshot": True}
        # Release the model and optimizer before constructing the reload to bound peak memory.
        del trainer, encoder, trainable, initial_adapters
        gc.collect()
        persist("reloading-saved-adapter")
        reloaded = HFEncoder(str(adapter), **loader)
        assert reloaded.max_length == 48
        assert_finite_parameters(reloaded.model)
        reload_query, reload_document = embeddings(reloaded)
        evidence["reload"] = {"query_max_abs_error": float(np.max(np.abs(reload_query - after_query))),
            "document_max_abs_error": float(np.max(np.abs(reload_document - after_document))),
            "atol": 1e-5, "rtol": 1e-4, "all_parameters_finite": True}
        np.testing.assert_allclose(reload_query, after_query, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(reload_document, after_document, atol=1e-5, rtol=1e-4)
        final_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src" / "clmkit").rglob("*.py"))}
        assert final_hashes == evidence["source_sha256"], "Product source changed during probe"
        evidence["product_source_unchanged"] = True
        evidence["status"] = "passed"
        persist("complete")
        return 0
    except Exception:
        evidence["status"] = "failed"
        evidence["exception"] = traceback.format_exc()
        persist("failed")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
