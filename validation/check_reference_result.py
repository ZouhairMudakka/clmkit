"""Reject a reference job whose intended test did not actually pass."""

import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


EXPECTED_TESTS = {
    "embedding": "test_reproduces_model_card_scores",
    "reranker": "test_llm_reranker_matches_reference_implementation",
}


def check_result(path: Path, kind: str) -> dict:
    expected = EXPECTED_TESTS[kind]
    cases = list(ET.parse(path).getroot().iter("testcase"))
    target = [case for case in cases if case.get("name") == expected]
    problems = []
    if len(target) != 1:
        problems.append(f"Expected exactly one {expected!r} result, found {len(target)}")
    elif any(target[0].find(tag) is not None for tag in ("skipped", "failure", "error")):
        problems.append(f"Required reference test {expected!r} did not pass")
    if any(case.find(tag) is not None for case in cases for tag in ("failure", "error")):
        problems.append("The reference report contains a failure or error")
    return {
        "model_kind": kind,
        "required_test": expected,
        "observed_tests": [case.get("name") for case in cases],
        "passed": not problems,
        "problems": problems,
    }


if __name__ == "__main__":
    result = check_result(Path(sys.argv[1]), os.environ["MODEL_KIND"])
    Path("reference-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)
