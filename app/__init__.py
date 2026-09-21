"""AI Data Engineering Incident Response Agent.

An agentic system that investigates failed data pipelines: it triages the
incident, investigates logs and data, forms a root-cause hypothesis, proposes a
remediation, waits for human approval, and writes an incident report.

The package is deliberately layered so that each concern is testable in
isolation:

    app.config      typed settings, and the single source of truth for what the
                    running process believes it has (keys, model, database)
    app.llm         provider-agnostic LLM access via LiteLLM, plus a
                    deterministic offline stub used by tests and CI
    app.db          PostgreSQL schema, migrations and repositories
    app.tools       the typed tools the agent may call
    app.agents      the LangGraph workflow: state, nodes, graph
    app.evaluation  LangSmith datasets, evaluators and the CI quality gate
    app.api         FastAPI routes
"""

__version__ = "0.1.0"
