"""The agent layer: state, prompts, nodes, graph, and the orchestrator.

The workflow is a LangGraph state machine rather than a single ``agent.invoke()``
so that the control flow is inspectable, the loops are bounded by state, and a
human can be required before anything mutates data.

    from app.agents import IncidentAgent, build_graph

``AgentNodes`` groups the node implementations; ``build_graph`` wires them;
``IncidentAgent`` adds persistence, the two-phase investigation record, and the
resume path used by the approval endpoints.
"""

from app.agents.agent import IncidentAgent, InvestigationOutcome
from app.agents.graph import build_graph, graph_mermaid
from app.agents.nodes import AgentNodes, BudgetExceeded, Remediator, SimulatedRemediator
from app.agents.state import IncidentState, build_report, initial_state

__all__ = [
    "AgentNodes",
    "BudgetExceeded",
    "IncidentAgent",
    "IncidentState",
    "InvestigationOutcome",
    "Remediator",
    "SimulatedRemediator",
    "build_graph",
    "build_report",
    "graph_mermaid",
    "initial_state",
]
