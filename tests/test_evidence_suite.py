"""Bounded retry orchestration tests; no child processes or cloud downloads."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from validation import evidence_suite as suite


def test_timeout_retry_preserves_partial_artifacts_and_history(tmp_path, monkeypatch):
    root = tmp_path / "results"
    output = root / "dev" / "banking77" / "dense"
    suite.write_json(root / "protocol.json", {"fixed": True})
    argv = [sys.executable, "validation/evidence_run.py", "run", "--output", str(output)]
    monkeypatch.setattr(suite, "require_cloud", lambda: None)
    monkeypatch.setattr(suite, "source_identity", lambda: "fixed-source")
    monkeypatch.setattr(suite, "commands", lambda *args: [("dev-dense", argv, 50)])
    calls = []

    def child(command, *, stdout, stderr, timeout, check):
        calls.append(timeout)
        output.mkdir(parents=True)
        if len(calls) == 1:
            (output / "partial.txt").write_text("preserve me")
            stdout.write("timeout attempt\n")
            raise subprocess.TimeoutExpired(command, timeout)
        suite.write_json(
            output / "result.json",
            {
                "status": "passed",
                "protocol_sha256": suite.digest(root / "protocol.json"),
                "source_sha256": "fixed-source",
            },
        )

    monkeypatch.setattr(suite.subprocess, "run", child)
    with pytest.raises(subprocess.TimeoutExpired):
        suite.run_suite("dev", root, tmp_path / "data", max_seconds=20, resume=False)
    assert 0 < calls[0] <= 20
    suite.run_suite("dev", root, tmp_path / "data", max_seconds=20, resume=True)
    status = json.loads((root / "suite-dev.json").read_text())
    archive = Path(status["jobs"][0]["previous_attempt"])
    assert (archive / "output" / "partial.txt").read_text() == "preserve me"
    assert (archive / "run.log").read_text() == "timeout attempt\n"
    assert status["previous_invocations"][0]["status"] == "failed"
    assert status["status"] == "passed"
    suite.run_suite("dev", root, tmp_path / "data", max_seconds=20, resume=True)
    assert len(calls) == 2


def test_archiving_refuses_paths_outside_experiment_root(tmp_path):
    outside = tmp_path / "keep.txt"
    outside.write_text("keep")
    with pytest.raises(ValueError):
        suite.archive_incomplete(tmp_path / "results", outside, tmp_path / "results" / "run.log", "job")
    assert outside.read_text() == "keep"
