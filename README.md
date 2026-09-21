# AI Data Engineering Incident Response Agent

An agentic AI system that **investigates failed data pipelines**. Given an alert that
`customer_claims_daily` failed at 2am, it triages the incident, investigates the logs and
the data with typed tools, follows the dependency chain to a root cause, proposes a
remediation, **waits for a human before anything that changes data**, and writes an
incident report.

The point is not "an LLM that describes an error". It is a controlled, observable,
**evaluated** agentic workflow: a stateful graph, nine typed tools behind a real
read-only SQL boundary, persisted state, a human in the loop, LangSmith tracing, and a CI
gate that fails the build when root-cause accuracy drops.

> **Status: all six roadmap phases implemented.** 228 tests, 8/8 evaluation tasks, clean
> `ruff` and `mypy`. See [What I verified](#what-i-verified) for the raw numbers and
> [Limitations](#limitations) for what this deliberately does not do.

---

## The one-paragraph version

A pipeline fails and posts an alert. A LangGraph state machine triages it into one of nine
failure categories, then runs bounded rounds of tool-assisted investigation — reading the
run's log, pulling its history, reading the pipeline's dependency metadata, and measuring
the warehouse directly with read-only SQL. When the evidence supports a single explanation
it commits to a root cause and proposes a fix; when it does not, it says so and either
gathers more evidence or ends `UNRESOLVED` rather than guessing. Anything that would
change data pauses the graph for human approval, and the whole run is recorded — every
node, tool call, observation, token and millisecond — so the report can state its own cost.

---

## Architecture

### Control flow

A LangGraph state machine. This diagram is generated from the compiled graph
(`app.agents.graph_mermaid`), so it cannot drift from the code:

```mermaid
graph TD;
	__start__([start]):::first
	triage(triage)
	investigate(investigate)
	assess_root_cause(assess_root_cause)
	plan_remediation(plan_remediation)
	approval(approval)
	execute_remediation(execute_remediation)
	report(report)
	__end__([end]):::last
	__start__ --> triage;
	approval -.-> execute_remediation;
	approval -.-> report;
	assess_root_cause -.-> investigate;
	assess_root_cause -.-> plan_remediation;
	assess_root_cause -.-> report;
	execute_remediation --> report;
	investigate --> assess_root_cause;
	plan_remediation -.-> approval;
	plan_remediation -.-> execute_remediation;
	triage --> investigate;
	report --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

Three properties fall out of writing the workflow as a graph rather than one
`agent.invoke()`:

* **The loop is bounded by state, not by a prompt.** `rounds` vs `max_rounds` decides
  whether to investigate again — a model cannot talk the agent into looping forever.
* **Approval is structurally unavoidable.** The only edge into `execute_remediation` from
  `plan_remediation` runs through `approval` whenever `requires_approval` is set, and that
  defaults to `True`. Getting this wrong requires editing the graph, not the prompt.
* **Refusing to answer is a first-class outcome.** An inconclusive assessment routes to
  `report` with status `UNRESOLVED`, not to a confident-sounding guess.

### Data flow

```
  S3 / logs ──▶ tools ──▶ evidence ──▶ root cause ──▶ remediation ──▶ report
                  │                                        │
                  └──────────▶ meta schema ◀───────────────┘
                     incidents · investigations · agent_steps
                     evidence · tool_calls · approvals · reports
```

Every node writes to the `platform` schema as it runs, which is what makes an interrupted
investigation resumable and a finished one auditable.

### The nodes

| Node | What it does | Why it exists here |
|---|---|---|
| `triage` | Classifies into one of 9 categories, sets severity, seeds a hypothesis | Cheap. Runs before anything expensive |
| `investigate` | Bounded agentic tool loop; collects typed observations | Where evidence actually comes from |
| `assess_root_cause` | Decides whether the evidence establishes a cause; may say no | The refusal is what drives another round |
| `plan_remediation` | Chooses one fix and assesses its risk | Where the agent stops being read-only |
| `approval` | `interrupt()` — suspends the graph and checkpoints it | Nothing mutating happens without a human |
| `execute_remediation` | Runs the approved action through a `Remediator` seam | The integration point for Airflow/Dagster |
| `report` | Assembles the timeline, findings and cost; persists it | The artefact a human reads |

---

## Offline by default

**Every credential is optional.** With an empty `.env` the system runs a deterministic
offline analyst, and the full test suite *and* the agent evaluation suite pass with no API
key and no network. That is the default path, not a bolted-on fallback — it is what makes
the CI gate meaningful, because a gate that only passes on the author's machine is not a
gate. The CI workflow **asserts that it has no credentials** before running the evaluation.

`GET /health` reports what the process actually believes it has:

```json
{
  "status": "ok",
  "capabilities": {
    "environment": "local",
    "llm": {"enabled": false, "model": "stub", "max_calls_per_incident": 25},
    "tracing": {"langsmith_enabled": false, "project": null},
    "storage": {"database": "postgresql://localhost:***@localhost:5432/ai_incidents",
                "s3_enabled": false, "log_dir": "data/logs"}
  },
  "database": {"reachable": true, "version": 1, "counts": {"incidents": 0}},
  "estate": {"seeded": true, "pipelines": 7},
  "tools": 9,
  "checkpointer": "PostgresSaver"
}
```

Switching to a real model is a config change, not a code change — **LiteLLM** is the
provider-agnostic layer:

```bash
LLM_MODEL=anthropic/claude-sonnet-4-5   ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=openai/gpt-4o-mini            OPENAI_API_KEY=sk-...
LLM_MODEL=deepseek/deepseek-chat        DEEPSEEK_API_KEY=...
LLM_MODEL=ollama/qwen2.5:7b             # local, free
```

---

## Quickstart

PostgreSQL is the only prerequisite.

```bash
createdb ai_incidents
cp .env.example .env          # every value is optional
uv sync --extra dev

uv run adp-agent seed         # build the simulated estate + warehouse
uv run pytest                 # 228 tests, no keys, no network

uv run python -m app.evaluation     # the agent gate: 8/8 tasks
uv run adp-agent demo missing_partition
```

Or the whole stack in containers:

```bash
docker compose up --build
docker compose exec api adp-agent seed
curl localhost:8000/health
```

### Reporting an incident over HTTP

```bash
curl -sX POST localhost:8000/incidents -H 'content-type: application/json' -d '{
  "pipeline": "customer_claims_daily", "run_id": "run_98321", "status": "FAILED",
  "timestamp": "2026-09-21T02:14:00",
  "error": "SparkException: File not found: s3://acme-lake/claims-raw/dt=2026-09-20/"
}' | jq '{incident_id, status, awaiting_approval_url}'
```

The graph pauses. A human decides:

```bash
curl -sX POST localhost:8000/incidents/$ID/approve \
  -H 'content-type: application/json' \
  -d '{"approver": "oncall@acme.example", "note": "upstream outage confirmed"}' | jq '.report'
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/incidents` | Report a failure and investigate it (`?investigate=false` to only record) |
| `GET` | `/incidents` | List, filterable by `status`, paginated |
| `GET` | `/incidents/{id}` | Incident + latest investigation + counts |
| `GET` | `/incidents/{id}/report` | The incident report |
| `GET` | `/incidents/{id}/evidence` | Every observation recorded |
| `GET` | `/incidents/{id}/timeline` | The node-by-node trace |
| `POST` | `/incidents/{id}/approve` | Approve and resume |
| `POST` | `/incidents/{id}/reject` | Reject; the incident ends `UNRESOLVED` |
| `GET` | `/health` | Liveness **and capability** |
| `GET` | `/metrics` | Counters and latency percentiles |

---

## The SQL boundary

The agent needs to query the warehouse. The naive guard — `if not sql.startswith("select")`
— is bypassable in one line: `select 1; drop table victim` starts with `select` and the
`DROP` still runs. This project does not ship that. Four layers, because each catches
something the others do not:

1. **Parse the statement** (`sqlglot`): exactly one statement, read-only root, and unknown
   syntax fails closed as `exp.Command`.
2. **Reject dangerous nodes anywhere in the tree.** A root-only check misses this:
   `WITH deleted AS (DELETE FROM claims RETURNING *) SELECT * FROM deleted` has a `SELECT`
   root and deletes your data.
3. **Constrain the schemas** that may be read: `warehouse` and `information_schema` only.
4. **The engine enforces it.** Queries run on a connection opened with
   `default_transaction_read_only=on`, plus a `statement_timeout`.

**Threat model — what this does *not* defend against**, because a guard whose limits are
undocumented gets trusted beyond them:

* A read-only role can still read any row it can see. There is no row- or column-level
  policy; if the warehouse holds PII, this tool returns it.
* `statement_timeout` bounds runtime, not the cost of a scan already underway.
* Resource exhaustion via a pathological query is mitigated, not prevented.

The guard is tested against 18 adversarial inputs plus 8 legitimate ones
(`tests/test_sql_guard.py`), including the multi-statement bypass, the data-modifying CTE,
`SELECT ... INTO`, `COPY ... TO` and cross-schema reads.

---

## Evaluation

`evals/tasks.yaml` is the suite, written as **data** so a change to what "good" means is
reviewable by someone who does not read the agent code. It is simultaneously the demo
fixture, the evaluation dataset and the CI gate: if you break the agent, all three break.

Every assertion is on an artefact or a behaviour — the category concluded, the action
chosen, the tools actually called, the observations gathered, the calls and tokens spent —
never on prose. Each check reports the **actual numbers**:

```
[PASS] missing_partition  (48ms)  RESOLVED
        ok   category        'UPSTREAM_DEPENDENCY_FAILURE' == 'UPSTREAM_DEPENDENCY_FAILURE'
        ok   action          'RERUN_UPSTREAM' == 'RERUN_UPSTREAM'
        ok   target          'claims_ingestion' == 'claims_ingestion'
        ok   min_evidence    8 >= 5
        ok   required_tools  used=[...]; missing=[]
        ok   max_write_attempts  0 <= 0
```

Three design choices worth naming:

* **A task that asserts failure.** `unknown_estate_stays_unresolved` requires a clean,
  recorded `UNRESOLVED`. A suite that only tests success makes it trivially easy to "fix" a
  system by making it never fail.
* **A malformed suite raises.** A typo'd assertion key, a duplicate id, an unknown scenario
  — all hard errors. Silently falling back to a built-in task list means your new task never
  runs while the suite reports green, and for a CI gate a failure that looks like success is
  the worst outcome.
* **An empty `expect` block fails.** `all([])` is `True` in Python, so a task with nothing
  to assert would otherwise sail through and quietly weaken the suite.

The suite also has a **pattern-matching trap**: `missing_partition_vague_error` is the same
incident with the error text stripped to `"Job run_98321 failed"`. An agent that classifies
by regex cannot solve it and must investigate the estate instead.

---

## What I verified

Numbers from an actual run on this machine, not aspirations:

| Claim | Evidence |
|---|---|
| Test suite passes | `228 passed` (config, observability, SQL guard, tools, analyst, persistence, graph, API, evaluation) |
| Agent evaluation passes | `8/8 = 100%`, exit code 0, no API key present |
| Lint and types clean | `ruff check` → *All checks passed*; `mypy` → *no issues found in 38 source files* |
| SQL guard holds | 18/18 adversarial inputs rejected, 8/8 legitimate queries allowed, target table intact after every attack |
| Root causes are right | 5/5 scenario categories, and the upstream case names `claims_ingestion` as the fix target |
| It investigates rather than pattern-matches | The vague-error task still reaches `UPSTREAM_DEPENDENCY_FAILURE` |
| It refuses to guess | The unknown-pipeline task ends `UNRESOLVED` with confidence 0.20 |
| The human gate works | Data-changing actions pause (`interrupted: true`); an owner notification does not |
| Budgets are enforced | At `llm_max_calls_per_incident=1` the run ends `FAILED` with the reason recorded |
| A broken model degrades | With a model that always raises, the run still completes via the offline analyst, recorded as `agent.triage.degraded` |
| Migrations are safe | Applying twice is a no-op; `platform.schema_migrations` records the version |
| Fixtures are reproducible | Two consecutive seeds produce byte-identical `claims` rows |

Typical per-incident cost on the offline analyst: **~12 LLM calls, 8 tool calls, ~8
observations, ~17k estimated tokens**, in under 150 ms.

---

## Limitations

The most important section. A project with no stated limitations reads as one whose author
has not looked.

1. **Confidence scores are LLM self-assessments, not calibrated probabilities.** They are
   deliberately *not* computed from evidence counts — turning "3 of 4 signals agree" into
   "0.91 confident" is numerology dressed as statistics. The right treatment is to carry
   the number as an untrusted signal and measure whether it correlates with being right.
   **That measurement is not built.** The evaluation asserts a confidence *range*, not
   calibration. Doing this properly needs a labelled set large enough to bin, which five
   scenarios are not.

2. **The pipeline estate is simulated.** Logs and warehouse tables are generated fixtures.
   This is deliberate — the investigation logic is the subject, not the integration surface
   — but it means the agent has never been tested against a real orchestrator's log format,
   real clock skew, or a real warehouse's permission model.

3. **Remediation is simulated.** `SimulatedRemediator` records the intended action and says
   so, loudly. There is no orchestrator connected, so "the fix was applied" means "the fix
   was recorded". The `Remediator` protocol is the seam; wiring it is real work that has not
   been done.

4. **LangSmith tracing is wired but unverified without a key.** `configure_tracing()` sets
   the environment and enables LiteLLM's LangSmith callback; nodes carry `@traceable`. With
   no key configured this is exercised as a no-op, which is what CI proves. **The upload
   path itself, dataset sync and `langsmith.evaluate()` have not been run against a live
   LangSmith account.** The local harness — not LangSmith — is what gates CI.

5. **The offline analyst is keyword-based and its limits are real.** It fails on failure
   vocabulary it does not recognise, its severity rule is a lookup table rather than an
   impact assessment, and its regression is scoped to a documented grammar (the upstream
   pattern cannot cross brackets or newlines — see
   `test_upstream_pattern_does_not_cross_brackets_or_lines`). It is a **baseline to beat**,
   not a substitute for a model.

6. **The API serves one database connection.** psycopg serialises access internally, so this
   is correct but not concurrent: two simultaneous investigations queue. A production
   deployment needs a per-request repository from a pool. The refactor was not worth doing
   before the concurrency was needed, and saying so is more useful than pretending.

7. **There is no authentication, authorisation or rate limiting.** Anyone who can reach
   `POST /incidents/{id}/approve` can approve a production remediation. The approval
   endpoint records *who* approved, but nothing verifies that claim.

8. **Metrics are per-process and unbounded in key space.** Counters and latency samples are
   bounded per key (`MAX_SAMPLES`), but the number of distinct keys is not, and with
   multiple uvicorn workers each holds a disjoint snapshot. A real deployment wants
   OpenTelemetry export.

9. **The Docker image is built in CI but was not built locally** — the Docker daemon is not
   running on this machine — so the `docker compose` path is authored and reviewed, not
   executed end to end here.

10. **No coverage measurement.** The suite is broad but nothing asserts a coverage floor, so
    an untested branch can be added without the build noticing.

---

## Roadmap

- [x] **Phase 1** — core LangGraph workflow
- [x] **Phase 2** — nine typed tools with a real read-only SQL boundary
- [x] **Phase 3** — human-in-the-loop approval with interrupt/resume
- [x] **Phase 4** — evaluation harness with a CI quality gate, plus LangSmith wiring
- [x] **Phase 5** — GitHub Actions CI/CD and container packaging
- [x] **Phase 6** — structured logging, injected metrics, budgets, timeouts, orphan sweeping
- [ ] **Next** — calibrate confidence against a labelled set; connect a real orchestrator
      behind `Remediator`; per-request DB repositories; API authentication
- [ ] **Then** — dependency-aware execution (a DAG rather than a list, derived from the
      lineage already recorded) and an async run queue returning `202`

---

## Layout

```
ai-incident-agent/
├── app/
│   ├── config.py          typed settings + derived capability
│   ├── observability.py   JSON logs + an injected, bounded metrics registry
│   ├── tracing.py         LangSmith configuration (inert without a key)
│   ├── schemas.py         the domain: incidents, evidence, root causes, reports
│   ├── estate.py          the simulated company, loaded as data
│   ├── scenarios.py       five incident fixtures, used by demo and evaluation alike
│   ├── platform.py        the composition root
│   ├── cli.py             seed · scenarios · demo · eval · serve · sweep · health
│   ├── llm/               LiteLLM client + the deterministic offline analyst
│   ├── tools/             registry + 9 tools + the SQL guard
│   ├── db/                connection, versioned migrations, repositories
│   ├── agents/            state, prompts, nodes, graph, orchestrator
│   ├── evaluation/        task loading, evaluators, the CI gate
│   └── api/               FastAPI routes and schemas
├── evals/tasks.yaml       the suite, as reviewable data
├── tests/                 228 tests
├── docker/, docker-compose.yml
└── .github/workflows/     ci.yml · deploy.yml
```

---

## Licence

MIT
