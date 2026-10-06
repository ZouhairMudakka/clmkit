"""Resolve and record a model revision before downloading reference-test files."""
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import model_info, snapshot_download

kind = os.environ["MODEL_KIND"]
model = {"embedding": "Qwen/Qwen3-Embedding-0.6B", "reranker": "Qwen/Qwen3-Reranker-0.6B"}[kind]
revision = model_info(model).sha
free = shutil.disk_usage(os.environ["HF_HOME"] if Path(os.environ["HF_HOME"]).exists() else "/tmp").free
if free < 4 * 1024**3:
    raise RuntimeError(f"Need at least 4 GiB free before model download; available={free}")
path = snapshot_download(model, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
Path("reference-model.json").write_text(json.dumps({"model": model, "revision": revision, "kind": kind}, indent=2))
name = "CLMKIT_TEST_MODEL" if kind == "embedding" else "CLMKIT_TEST_RERANKER"
with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as stream:
    stream.write(f"{name}={path}\n")
