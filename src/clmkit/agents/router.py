"""Semantic routing: map a request to a tool / intent / sub-agent by embedding similarity.

Much cheaper and more deterministic than asking an LLM to pick a tool, and a
good first stage before one (route -> shortlist tools -> LLM decides).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from clmkit.encoders.base import Encoder

Aggregation = Literal["max", "mean", "centroid"]


@dataclass
class Route:
    name: str
    utterances: list[str]
    description: str | None = None
    #: Per-route override of the router threshold.
    threshold: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("route name must be non-empty")
        if not self.utterances and not self.description:
            raise ValueError(f"route {self.name!r} needs utterances or a description")


@dataclass(frozen=True)
class RouteMatch:
    name: str
    score: float
    scores: dict[str, float]
    route: Route


class SemanticRouter:
    """Route queries to the most similar :class:`Route`.

    Args:
        threshold: minimum score to accept a match; below it :meth:`route` returns ``None``
            (let your agent fall back to an LLM or a default route).
        aggregation: how utterance similarities become a route score:
            ``"max"`` (nearest example; robust default), ``"mean"``, or ``"centroid"``.
        instruction: query instruction (e.g. for Qwen3: "Given a user request, retrieve the
            tool that can fulfil it").
    """

    def __init__(
        self,
        encoder: Encoder,
        routes: Sequence[Route] = (),
        *,
        threshold: float = 0.5,
        aggregation: Aggregation = "max",
        instruction: str | None = None,
    ) -> None:
        if aggregation not in ("max", "mean", "centroid"):
            raise ValueError("aggregation must be 'max', 'mean' or 'centroid'")
        self.encoder = encoder
        self.threshold = threshold
        self.aggregation: Aggregation = aggregation
        self.instruction = instruction
        self.routes: dict[str, Route] = {}
        self._vecs: dict[str, np.ndarray] = {}
        for r in routes:
            self.add_route(r)

    def add_route(self, route: Route) -> None:
        if route.name in self.routes:
            raise ValueError(f"duplicate route {route.name!r}")
        texts = list(route.utterances) + ([route.description] if route.description else [])
        vecs = self.encoder.encode(texts, kind="document")
        if self.aggregation == "centroid":
            c = vecs.mean(axis=0, keepdims=True)
            vecs = c / max(float(np.linalg.norm(c)), 1e-12)
        self.routes[route.name] = route
        self._vecs[route.name] = vecs

    def remove_route(self, name: str) -> None:
        self.routes.pop(name)
        self._vecs.pop(name)

    def scores(self, query: str) -> dict[str, float]:
        if not self.routes:
            return {}
        q = self.encoder.encode(query, kind="query", instruction=self.instruction)
        out = {}
        for name, vecs in self._vecs.items():
            sims = vecs @ q
            out[name] = float(sims.mean() if self.aggregation == "mean" else sims.max())
        return out

    def route(self, query: str) -> RouteMatch | None:
        """Best route, or ``None`` if nothing clears its threshold."""
        scores = self.scores(query)
        for name, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            route = self.routes[name]
            limit = route.threshold if route.threshold is not None else self.threshold
            if score >= limit:
                return RouteMatch(name, score, scores, route)
            break  # only the top route is eligible; a lower one never overrides it
        return None

    def top_k(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Shortlist of routes, e.g. to give an LLM only the most relevant tools."""
        return sorted(self.scores(query).items(), key=lambda kv: kv[1], reverse=True)[:k]
