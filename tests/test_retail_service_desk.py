"""Business workflow checks against the installed clmkit public API."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from clmkit import load_encoder

SCRIPT = Path(__file__).resolve().parents[1] / "templates" / "retail_service_desk.py"
spec = importlib.util.spec_from_file_location("retail_service_desk", SCRIPT)
assert spec is not None and spec.loader is not None
template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template)
RETURN_QUERY = "I want to return my blender for a refund"


def prepare(desk, query=RETURN_QUERY, tenant="north", customer="alice", order="N-1001"):
    return desk.prepare_case(
        query, ticket_id="T-1", order_id=order, trusted_tenant_id=tenant, trusted_customer_id=customer
    )


def test_return_packet_cites_policy_and_requires_human_review():
    result = prepare(template.RetailServiceDesk())
    assert (result["status"], result["queue"], result["reason"]) == ("staff_review", "returns", "human_review_required")
    evidence = result["policy_evidence"][0]
    assert evidence["document_id"] == "north:returns:v1"
    assert evidence["source_url"] == "https://example.invalid/north/returns"
    assert evidence["revision"] == "v1"
    assert evidence["title"] == "Northstar Home returns"
    assert "30 days" in evidence["text"]
    assert result["order_context"]["item"] == "blender"
    assert result["human_review_required"] is True
    assert result["action_executed"] is False


@pytest.mark.parametrize(
    ("tenant", "customer", "order", "channel", "item"),
    [
        ("north", "alice", "N-1001", "email", "blender"),
        ("north", "bob", "N-1002", "phone", "kettle"),
        ("south", "alice", "S-1001", "SMS", "toaster"),
    ],
)
def test_delivery_scopes_policy_topic_tenant_and_customer(tenant, customer, order, channel, item):
    result = prepare(template.RetailServiceDesk(), "Track my parcel delivery", tenant, customer, order)
    assert result["queue"] == "delivery"
    assert result["order_context"]["item"] == item
    assert len(result["customer_preferences"]) == 1
    assert channel in result["customer_preferences"][0]["text"]
    assert {row["tenant_id"] for row in result["policy_evidence"]} == {tenant}
    assert {row["topic"] for row in result["policy_evidence"]} == {"delivery"}


def test_policy_filter_excludes_more_similar_other_tenant_and_topic():
    desk = template.RetailServiceDesk(route_threshold=-1, policy_min_score=-1)
    south_return = next(text for tenant, topic, _, text in template.POLICIES if (tenant, topic) == ("south", "returns"))
    assert desk._policies.search(south_return, k=1)[0].id == "south:returns:v1"
    result = prepare(desk, south_return)
    assert result["queue"] == "returns"
    assert [row["document_id"] for row in result["policy_evidence"]] == ["north:returns:v1"]


@pytest.mark.parametrize(
    ("tenant", "customer", "order"),
    [
        ("north", "alice", "missing"),
        ("north", "alice", "N-1002"),
        ("south", "alice", "N-1001"),
        ("north", "unknown", "N-1001"),
    ],
)
def test_missing_and_unauthorized_orders_return_identical_empty_packet(tenant, customer, order):
    desk = template.RetailServiceDesk()
    result = prepare(desk, tenant=tenant, customer=customer, order=order)
    assert result == prepare(desk, order="missing")
    assert result["reason"] == "order_context_unavailable"
    assert result["status"] == "escalated"
    assert result["order_context"] is None
    assert result["policy_evidence"] == result["customer_preferences"] == []


@pytest.mark.parametrize("scope", ["tenant", "customer"])
def test_blank_trusted_scope_rejected(scope):
    with pytest.raises(ValueError, match=f"trusted_{scope}_id"):
        prepare(template.RetailServiceDesk(), **{scope: " "})


@pytest.mark.parametrize(("query", "reason"), [("  ", "blank_query"), ("volcanic basalt tectonics", "route_abstained")])
def test_blank_and_unknown_requests_escalate(query, reason):
    result = prepare(template.RetailServiceDesk(), query)
    assert result["status"] == "escalated"
    assert result["reason"] == reason
    assert result["policy_evidence"] == result["customer_preferences"] == []


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [({"route_threshold": 1.1}, "route_abstained"), ({"policy_min_score": 1.1}, "policy_unavailable")],
)
def test_injected_encoder_and_threshold_override(overrides, reason):
    encoder = load_encoder("hashing", dim=1024)
    desk = template.RetailServiceDesk(encoder, **overrides)
    assert desk.encoder is encoder
    result = prepare(desk)
    assert result["reason"] == reason
    assert result["order_context"] is None


def test_preparation_does_not_mutate_orders_policies_or_preferences():
    desk = template.RetailServiceDesk()
    before = deepcopy((desk._orders, desk._policies.documents, desk._preferences.retriever.documents))
    first = prepare(desk)
    assert first == prepare(desk)
    assert before == (desk._orders, desk._policies.documents, desk._preferences.retriever.documents)
    first["order_context"]["status"] = "refunded"
    assert prepare(desk)["order_context"]["status"] == "delivered"


def test_copied_cli_isolated_json_and_no_generated_files(tmp_path):
    copied = tmp_path / SCRIPT.name
    shutil.copyfile(SCRIPT, copied)
    result = subprocess.run(
        [sys.executable, "-I", str(copied)], cwd=tmp_path, check=True, text=True, capture_output=True, timeout=60
    )
    assert result.stderr == ""
    output = json.loads(result.stdout)
    assert output["use_case"] == "retail_service_desk"
    assert output["synthetic_data"] is True
    assert output["encoder"] == "hashing-512"
    assert [case["status"] for case in output["cases"]] == ["staff_review"] * 3 + ["escalated"] * 4
    assert all(case["action_executed"] is False for case in output["cases"])
    assert list(tmp_path.iterdir()) == [copied]
