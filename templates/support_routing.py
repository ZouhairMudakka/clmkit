"""Select a synthetic support queue without executing tools or changing accounts.

Run: python templates/support_routing.py
Hashing is a deterministic lexical baseline, not a learned semantic model.
Calibration below illustrates mechanics on a tiny synthetic set, not a quality
claim. Recalibrate on representative held-out requests before real deployment.
"""

from __future__ import annotations

import json
import math
from typing import Any

from clmkit import Encoder, load_encoder
from clmkit.agents import Route, SemanticRouter

ROUTE_EXAMPLES = {
    "billing": ["billing invoice payment refund", "duplicate charge on my invoice"],
    "account": ["account password reset login", "change my account email address"],
    "delivery": ["delivery parcel tracking shipping", "my parcel delivery is late"],
}
# Distinct from the routing exemplars. The tests use additional held-out requests.
CALIBRATION_KNOWN = (
    ("invoice payment refund request", "billing"),
    ("account login password reset help", "account"),
    ("parcel shipping tracking update", "delivery"),
)
CALIBRATION_UNKNOWN = ("sourdough bread baking recipe", "astronomy telescope nebula")


def calibration_report(router: SemanticRouter) -> dict[str, float]:
    """Choose a demo threshold halfway across the observed acceptance gap.

    Fail visibly if this tiny set cannot be separated with the supplied encoder.
    A real application should choose thresholds using its false-accept cost and
    evaluate on a separate dataset, including ambiguous and out-of-domain input.
    """
    positives = []
    for query, expected in CALIBRATION_KNOWN:
        scores = router.scores(query)
        if max(scores, key=scores.get) != expected:
            raise ValueError("demo calibration misclassified a known request; use application calibration")
        positives.append(scores[expected])
    known_min = min(positives)
    unknown_max = max(max(router.scores(query).values()) for query in CALIBRATION_UNKNOWN)
    if unknown_max >= known_min:
        raise ValueError("no demo calibration gap; use application calibration")
    return {
        "known_min_score": known_min,
        "unknown_max_score": unknown_max,
        "threshold": (known_min + unknown_max) / 2,
    }


def build_router(encoder: Encoder, *, threshold: float | None = None) -> SemanticRouter:
    router = SemanticRouter(
        encoder, [Route(name, list(examples)) for name, examples in ROUTE_EXAMPLES.items()], aggregation="max"
    )
    if threshold is None:
        threshold = calibration_report(router)["threshold"]
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    router.threshold = threshold
    return router


def select_route(router: SemanticRouter, request: str) -> dict[str, Any]:
    """Select only. The application separately authorizes and dispatches actions.

    Similarity scores are not probabilities. Unknown requests can still score
    above a threshold; this demo does not guarantee out-of-domain detection.
    """
    match = router.route(request) if request.strip() else None
    return {
        "request": request,
        "status": "selected" if match else "abstained",
        "route": match.name if match else None,
        "score": match.score if match else None,
        "threshold": router.threshold,
        "scores": match.scores if match else (router.scores(request) if request.strip() else {}),
        "action_executed": False,
    }


def demo(encoder: Encoder | None = None) -> dict[str, Any]:
    encoder = encoder if encoder is not None else load_encoder("hashing")
    router = build_router(encoder)
    return {
        "use_case": "support_routing",
        "synthetic_data": True,
        "encoder": encoder.name,
        "calibration": calibration_report(router),
        "requests": [
            select_route(router, request)
            for request in (
                "please refund the duplicate invoice charge",
                "help reset my account password",
                "track my parcel delivery",
                "volcanic basalt tectonics",
                "",
            )
        ],
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, allow_nan=False))
