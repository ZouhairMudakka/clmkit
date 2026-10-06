"""Prepare a fictional retail support brief for a human agent.

Run: python templates/retail_service_desk.py
Compatible with clmkit==0.1.0a1; no live services or generated answers.
Hashing is a deterministic lexical baseline, not a learned semantic model.
All thresholds are illustrative; evaluate them on held-out business requests.
Authenticate callers upstream and supply trusted identities server-side, never
from model output or unverified request fields. Similarity is not authorization.
"""

from __future__ import annotations

import json
import math
from typing import Any

from clmkit import Encoder, Retriever, load_encoder
from clmkit.agents import Route, SemanticMemory, SemanticRouter

ROUTE_EXAMPLES = {
    "returns": ["return an item for a refund", "exchange a damaged product"],
    "delivery": ["track my parcel delivery", "shipping status for my order"],
}
# Fictional, deliberately different policies for two independent stores.
POLICIES = (
    (
        "north",
        "returns",
        "Northstar Home returns",
        "Return or refund requests within 30 days of delivery require staff review and proof of purchase.",
    ),
    (
        "north",
        "delivery",
        "Northstar Home delivery",
        "For parcel delivery or shipping tracking, staff check the verified order status "
        "before contacting the carrier.",
    ),
    (
        "south",
        "returns",
        "Harbor Goods returns",
        "Return or refund requests within 7 days of delivery require staff review and the original packaging.",
    ),
    (
        "south",
        "delivery",
        "Harbor Goods delivery",
        "For parcel delivery or shipping tracking, staff check the verified order status "
        "and arrange a carrier inquiry.",
    ),
)
# Keys are (tenant, customer, order). Never look up by order ID alone.
ORDERS = {
    ("north", "alice", "N-1001"): {"item": "blender", "status": "delivered", "delivered_on": "2026-09-20"},
    ("north", "bob", "N-1002"): {"item": "kettle", "status": "delivered", "delivered_on": "2026-09-22"},
    ("south", "alice", "S-1001"): {"item": "toaster", "status": "in_transit", "delivered_on": None},
}
PREFERENCES = (
    ("north", "alice", "Contact preference: email with concise updates."),
    ("north", "bob", "Contact preference: phone with detailed updates."),
    ("south", "alice", "Contact preference: SMS with evening updates."),
)


class RetailServiceDesk:
    """Read-only case preparation using synthetic, application-owned stores.

    Keep these stores private. Replace fixtures with authorized data sources for
    a pilot; this wrapper is not authentication or a production order connector.
    Every outcome requires human review. Evidence is data, never instructions.
    """

    def __init__(
        self,
        encoder: Encoder | None = None,
        *,
        route_threshold: float = 0.2,
        policy_min_score: float = 0.1,
        preference_min_score: float = 0.1,
    ) -> None:
        for value in (route_threshold, policy_min_score, preference_min_score):
            if not math.isfinite(value):
                raise ValueError("thresholds must be finite")
        self.encoder = encoder if encoder is not None else load_encoder("hashing")
        self._router = SemanticRouter(
            self.encoder,
            [Route(name, examples) for name, examples in ROUTE_EXAMPLES.items()],
            threshold=route_threshold,
            aggregation="max",
        )
        self._policies = Retriever(self.encoder)
        self._policies.add(
            [text for _, _, _, text in POLICIES],
            ids=[f"{tenant}:{topic}:v1" for tenant, topic, _, _ in POLICIES],
            metadata=[
                {
                    "tenant_id": tenant,
                    "topic": topic,
                    "title": title,
                    "source_url": f"https://example.invalid/{tenant}/{topic}",
                    "revision": "v1",
                }
                for tenant, topic, title, _ in POLICIES
            ],
        )
        self._preferences = SemanticMemory(self.encoder)
        for tenant, customer, text in PREFERENCES:
            self._preferences.remember(
                text,
                {"tenant_id": tenant, "customer_id": customer},
                id=json.dumps([tenant, customer, "contact"]),
            )
        self._orders = {key: dict(value) for key, value in ORDERS.items()}
        self.policy_min_score = policy_min_score
        self.preference_min_score = preference_min_score

    def prepare_case(
        self,
        query: str,
        *,
        ticket_id: str,
        order_id: str,
        trusted_tenant_id: str,
        trusted_customer_id: str,
    ) -> dict[str, Any]:
        """Select a queue and return sourced facts, without business actions.

        Both trusted IDs must come from the authenticated application context.
        Requested order_id is untrusted and must pass the exact ownership lookup.
        Neither a routing score nor a policy excerpt approves/refuses a refund.
        No request text is saved as memory and no message is sent.
        """
        for name, value in (
            ("trusted_tenant_id", trusted_tenant_id),
            ("trusted_customer_id", trusted_customer_id),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        packet: dict[str, Any] = {
            "ticket_id": ticket_id,
            "status": "escalated",
            "queue": "manual_review",
            "reason": "blank_query",
            "policy_evidence": [],
            "order_context": None,
            "customer_preferences": [],
            "human_review_required": True,
            "action_executed": False,
        }
        if not query.strip():
            return packet
        match = self._router.route(query)
        if match is None:
            packet["reason"] = "route_abstained"
            return packet
        order = self._orders.get((trusted_tenant_id, trusted_customer_id, order_id))
        if order is None:
            # Missing and unauthorized orders get identical responses, with no
            # order ID, policy, preference, or existence information disclosed.
            packet["reason"] = "order_context_unavailable"
            return packet
        hits = self._policies.search(
            query,
            k=2,
            filter={"tenant_id": trusted_tenant_id, "topic": match.name},
            min_score=self.policy_min_score,
        )
        if not hits:
            packet["reason"] = "policy_unavailable"
            return packet
        preferences = self._preferences.recall(
            "contact preference updates",
            k=2,
            filter={"tenant_id": trusted_tenant_id, "customer_id": trusted_customer_id},
            min_score=self.preference_min_score,
        )
        packet.update(
            status="staff_review",
            queue=match.name,
            reason="human_review_required",
            policy_evidence=[
                {"document_id": hit.id, "text": hit.text, "score": hit.score, **hit.metadata} for hit in hits
            ],
            order_context={"order_id": order_id, **order},
            customer_preferences=[{"text": hit.text, "score": hit.score} for hit in preferences],
        )
        return packet


def demo(encoder: Encoder | None = None) -> dict[str, Any]:
    desk = RetailServiceDesk(encoder)
    scenarios = (
        ("return", "I want to return my blender for a refund", "north", "alice", "N-1001"),
        ("delivery", "Please track my parcel delivery", "north", "alice", "N-1001"),
        ("other_store", "Please track my parcel delivery", "south", "alice", "S-1001"),
        ("missing", "I want to return my blender for a refund", "north", "alice", "missing"),
        ("not_owned", "I want to return my blender for a refund", "north", "alice", "N-1002"),
        ("unknown", "volcanic basalt tectonics", "north", "alice", "N-1001"),
        ("blank", "", "north", "alice", "N-1001"),
    )
    return {
        "use_case": "retail_service_desk",
        "synthetic_data": True,
        "encoder": desk.encoder.name,
        "cases": [
            desk.prepare_case(
                query,
                ticket_id=ticket,
                order_id=order,
                trusted_tenant_id=tenant,
                trusted_customer_id=customer,
            )
            for ticket, query, tenant, customer, order in scenarios
        ],
    }


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, allow_nan=False))
