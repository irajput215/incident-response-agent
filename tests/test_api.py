"""The HTTP API, exercised through the real app factory.

Requests go through the same ``create_app`` the server uses, so a change that
breaks the factory breaks these tests rather than only production. The approval
flow is covered end to end here, because that is the part an operator actually
touches.
"""
from __future__ import annotations

from app.scenarios import SCENARIOS_BY_ID


def alert_json(scenario_id: str = "missing_partition") -> dict:
    return dict(SCENARIOS_BY_ID[scenario_id].incident)


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #
def test_health_reports_capability_not_just_liveness(client) -> None:
    """`status: ok` alone tells an operator nothing about what is configured."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["capabilities"]["llm"] == {
        "enabled": False,
        "model": "stub",
        "max_calls_per_incident": 25,
    }
    assert body["database"]["reachable"] is True
    assert body["database"]["counts"]["incidents"] == 0
    assert body["tools"] == 9


def test_health_never_leaks_a_password(client) -> None:
    assert "secret" not in client.get("/health").text


def test_metrics_endpoint_exposes_counters(client) -> None:
    body = client.get("/metrics").json()
    assert "counters" in body
    assert "latency" in body


# --------------------------------------------------------------------------- #
# creating incidents
# --------------------------------------------------------------------------- #
def test_reporting_an_incident_investigates_it(client) -> None:
    response = client.post("/incidents", json=alert_json())
    assert response.status_code == 201
    body = response.json()
    assert body["incident_id"].startswith("INC-")
    assert body["status"] == "AWAITING_APPROVAL"
    assert body["interrupted"] is True
    assert body["awaiting_approval_url"].endswith("/approve")
    assert body["report"] is None


def test_an_alert_can_be_recorded_without_investigating(client) -> None:
    response = client.post("/incidents?investigate=false", json=alert_json())
    assert response.status_code == 201
    assert response.json()["status"] == "OPEN"
    assert response.json()["investigation_id"] == ""


def test_a_redelivered_alert_does_not_duplicate_the_incident(client) -> None:
    first = client.post("/incidents?investigate=false", json=alert_json()).json()
    second = client.post("/incidents?investigate=false", json=alert_json()).json()
    assert first["incident_id"] == second["incident_id"]
    assert len(client.get("/incidents").json()) == 1


def test_a_malformed_alert_is_rejected_with_a_useful_body(client) -> None:
    response = client.post("/incidents", json={"pipeline": "", "run_id": ""})
    assert response.status_code == 422
    assert "detail" in response.json()


# --------------------------------------------------------------------------- #
# reading incidents back
# --------------------------------------------------------------------------- #
def test_incidents_are_listed_newest_first(client) -> None:
    client.post("/incidents?investigate=false", json=alert_json("missing_partition"))
    client.post("/incidents?investigate=false", json=alert_json("spark_oom"))

    listed = client.get("/incidents").json()
    assert len(listed) == 2
    assert {row["pipeline"] for row in listed} == {
        "customer_claims_daily",
        "claims_aggregation_monthly",
    }


def test_incidents_can_be_filtered_by_status(client) -> None:
    client.post("/incidents?investigate=false", json=alert_json())
    assert len(client.get("/incidents?status=OPEN").json()) == 1
    assert client.get("/incidents?status=RESOLVED").json() == []


def test_an_incident_detail_carries_its_trace(client) -> None:
    incident_id = client.post("/incidents", json=alert_json()).json()["incident_id"]
    detail = client.get(f"/incidents/{incident_id}").json()

    assert detail["incident"]["pipeline"] == "customer_claims_daily"
    assert detail["investigation"]["status"] == "AWAITING_APPROVAL"
    assert detail["counts"]["evidence"] >= 5
    assert detail["counts"]["tool_calls"] >= 5
    assert detail["counts"]["steps"] >= 4


def test_the_timeline_lists_every_node_that_ran(client) -> None:
    incident_id = client.post("/incidents", json=alert_json()).json()["incident_id"]
    nodes = [step["node"] for step in client.get(f"/incidents/{incident_id}/timeline").json()]
    assert nodes == ["triage", "investigate", "assess_root_cause", "plan_remediation"]


def test_evidence_is_available_as_a_first_class_resource(client) -> None:
    incident_id = client.post("/incidents", json=alert_json()).json()["incident_id"]
    evidence = client.get(f"/incidents/{incident_id}/evidence").json()
    assert len(evidence) >= 5
    assert {item["tool"] for item in evidence} >= {"get_pipeline_logs", "check_row_count"}


def test_unknown_incidents_are_404(client) -> None:
    assert client.get("/incidents/INC-nope").status_code == 404
    assert client.get("/incidents/INC-nope/report").status_code == 404
    assert client.get("/incidents/INC-nope/timeline").status_code == 404


def test_a_report_is_404_until_one_exists(client) -> None:
    """Awaiting approval means there is deliberately no report yet."""
    incident_id = client.post("/incidents", json=alert_json()).json()["incident_id"]
    assert client.get(f"/incidents/{incident_id}/report").status_code == 404


# --------------------------------------------------------------------------- #
# human-in-the-loop over HTTP
# --------------------------------------------------------------------------- #
def test_approving_over_http_resolves_the_incident(client) -> None:
    incident_id = client.post("/incidents", json=alert_json()).json()["incident_id"]

    response = client.post(
        f"/incidents/{incident_id}/approve",
        json={"approver": "oncall@acme.example", "note": "upstream outage confirmed"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "RESOLVED"
    assert body["report"]["status"] == "RESOLVED"
    assert body["report"]["remediation"]["action"] == "RERUN_UPSTREAM"
    assert body["report"]["remediation"]["target"] == "claims_ingestion"


def test_the_report_becomes_available_after_approval(client) -> None:
    incident_id = client.post("/incidents", json=alert_json()).json()["incident_id"]
    client.post(f"/incidents/{incident_id}/approve", json={"approver": "a@x"})

    report = client.get(f"/incidents/{incident_id}/report").json()
    assert report["category"] == "UPSTREAM_DEPENDENCY_FAILURE"
    assert report["approval"]["approved"] is True
    assert report["approval"]["approver"] == "a@x"


def test_rejecting_over_http_leaves_the_incident_open(client) -> None:
    incident_id = client.post("/incidents", json=alert_json()).json()["incident_id"]
    body = client.post(
        f"/incidents/{incident_id}/reject",
        json={"approver": "sre@acme.example", "reason": "wait for the upstream backfill"},
    ).json()

    assert body["status"] == "UNRESOLVED"
    assert body["report"]["approval"]["approved"] is False
    assert body["report"]["approval"]["note"] == "wait for the upstream backfill"


def test_approval_requires_who_is_accountable(client) -> None:
    """An audit trail with no name in it is not an audit trail."""
    incident_id = client.post("/incidents", json=alert_json()).json()["incident_id"]
    response = client.post(f"/incidents/{incident_id}/approve", json={"approver": ""})
    assert response.status_code == 422


def test_deciding_twice_is_a_conflict_not_a_second_execution(client) -> None:
    """409, not 400: the request is fine, the resource is in the wrong state.

    Re-approving a finished investigation must not re-run the remediation.
    """
    incident_id = client.post("/incidents", json=alert_json()).json()["incident_id"]
    assert client.post(f"/incidents/{incident_id}/approve", json={"approver": "a@x"}).status_code == 200

    second = client.post(f"/incidents/{incident_id}/approve", json={"approver": "b@x"})
    assert second.status_code == 409
    assert "not awaiting approval" in second.json()["detail"]


def test_deciding_on_an_incident_with_no_investigation_is_404(client) -> None:
    incident_id = client.post("/incidents?investigate=false", json=alert_json()).json()["incident_id"]
    assert client.post(f"/incidents/{incident_id}/approve", json={"approver": "a@x"}).status_code == 404


def test_the_approval_is_audited_over_http(client) -> None:
    incident_id = client.post("/incidents", json=alert_json()).json()["incident_id"]
    client.post(
        f"/incidents/{incident_id}/approve",
        json={"approver": "auditor@acme.example", "note": "checked the upstream"},
    )
    detail = client.get(f"/incidents/{incident_id}").json()
    assert detail["approval"]["approver"] == "auditor@acme.example"
    assert detail["approval"]["approved"] is True


# --------------------------------------------------------------------------- #
# a low-risk scenario needs no human
# --------------------------------------------------------------------------- #
def test_a_low_risk_scenario_completes_without_any_approval(client) -> None:
    body = client.post("/incidents", json=alert_json("schema_change")).json()
    assert body["interrupted"] is False
    assert body["status"] == "RESOLVED"
    assert body["report"]["remediation"]["action"] == "ALERT_OWNER"
