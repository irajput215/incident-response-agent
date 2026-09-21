"""Graph construction: the workflow as an explicit state machine.

The shape is the point of the project. A single ``agent.invoke()`` with a big
prompt is not a workflow — it has no places to intervene, no bounded loops, and
no way to require a human before something dangerous happens. Here the control
flow is data you can read:

    START → triage → investigate → assess ─┬─ conclusive ──→ plan_remediation ─┬─ needs approval → approval ─┬─ approved → execute → report
                                           │                                    └─ no approval needed ──────┼─ rejected ─────────→ report
                                           ├─ inconclusive, rounds left → investigate (loop)
                                           └─ inconclusive, out of rounds ────────────────────────────────────→ report

Two properties fall out of writing it this way, and both are load-bearing:

* **The loop is bounded by state**, not by a prompt asking the model to stop.
  ``rounds`` and ``max_rounds`` are compared in :meth:`AgentNodes.route_after_assessment`.
* **Approval is structurally unavoidable for anything that mutates data.** The
  only edge into ``execute_remediation`` from ``plan_remediation`` runs through
  ``approval`` whenever ``requires_approval`` is set — which defaults to True.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import AgentNodes
from app.agents.state import IncidentState


def build_graph(nodes: AgentNodes, checkpointer: Any | None = None) -> Any:
    """Compile the incident-response workflow.

    A checkpointer is not optional in practice: ``interrupt()`` needs one to
    persist state across the pause, so a graph compiled without one cannot ask
    for approval at all.
    """
    graph = StateGraph(IncidentState)

    graph.add_node("triage", nodes.triage)
    graph.add_node("investigate", nodes.investigate)
    graph.add_node("assess_root_cause", nodes.assess_root_cause)
    graph.add_node("plan_remediation", nodes.plan_remediation)
    graph.add_node("approval", nodes.approval)
    graph.add_node("execute_remediation", nodes.execute_remediation)
    graph.add_node("report", nodes.report)

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "investigate")
    graph.add_edge("investigate", "assess_root_cause")

    graph.add_conditional_edges(
        "assess_root_cause",
        nodes.route_after_assessment,
        {
            "plan_remediation": "plan_remediation",
            "investigate": "investigate",
            "report": "report",
        },
    )
    graph.add_conditional_edges(
        "plan_remediation",
        nodes.route_after_remediation,
        {"approval": "approval", "execute_remediation": "execute_remediation"},
    )
    graph.add_conditional_edges(
        "approval",
        nodes.route_after_approval,
        {"execute_remediation": "execute_remediation", "report": "report"},
    )

    graph.add_edge("execute_remediation", "report")
    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer)


def graph_mermaid(nodes: AgentNodes, checkpointer: Any | None = None) -> str:
    """The compiled graph as Mermaid, for the README.

    Generated rather than hand-drawn so the diagram cannot drift from the code.
    """
    compiled = build_graph(nodes, checkpointer)
    return compiled.get_graph().draw_mermaid()


__all__ = ["build_graph", "graph_mermaid"]
