# NOTES — the build log

Every bug hit, trade-off made, and thing I would redo. Written as it happened, while the
confusion was still fresh. This is the one artefact nobody can fake, because nobody else
was there.

Legend: **E** = what I expected, **A** = what actually happened.

---

## Phase 1 — Foundation

### The evidence-payload bug that made the agent pattern-match instead of investigate
**E** The offline analyst reads evidence structurally: it looks for `role: "tool"` messages
and joins "this pipeline depends on X" with "X failed" to conclude an upstream failure.
**A** It concluded `DATA_SOURCE_FAILURE` for every run — the *triage* answer — and never
found the upstream cause, even though the evidence plainly contained both facts.
**Why** The investigation loop passes tool results as real `role: "tool"` messages, so the
join worked *while gathering*. But the root-cause prompt renders the same evidence **into a
user message as text**. Looking only for tool messages found nothing, the structural join
returned `None`, and the analyst silently fell back to keyword matching on the error string.
**What I changed** `_evidence_payloads()` now recovers structured data from *either*
representation: tool messages during the loop, embedded JSON objects while concluding.
**What I'd tell someone hitting this** The two representations are not equivalent and the
code did not know that. Whenever the same data crosses a boundary in two shapes, write the
function that handles both, and test the path that reads it *after* the round trip — not
just the one that produced it. I only caught this by printing what the analyst actually
received, which is the debugging move I keep having to relearn.

### Re-classifying the whole prompt made every remediation identical
**E** Remediation reads the established root cause and picks the matching action.
**A** Every incident got `RERUN_UPSTREAM` — including the out-of-memory one.
**Why** `_remediation` called `detect_category()` over the *entire* prompt. The evidence
stream contains every pipeline's metadata, including its `upstream` list, so the upstream
pattern matched on every single run regardless of the actual cause.
**What I changed** Read the category out of the explicit `ROOT CAUSE` JSON block in the
prompt; only fall back to text classification when that block is absent.
**What I'd tell someone** "Classify this text" is a different operation from "read the field
that already contains the answer". The second is cheaper, more accurate, and I reached for
the first out of habit.

### A tool with no argument mapping is a tool that can never succeed
**E** The analyst's plan calls `get_pipeline_metadata(pipeline)`, learns the upstream
pipeline, then queries that upstream's history.
**A** The metadata call failed every time with a validation error, the upstream was never
discovered, and the dependency chain was never followed.
**Why** `get_pipeline_metadata` was in the plan but **missing from `_arguments_for`**, so it
was invoked with `{}`. `additionalProperties: false` plus a required field meant the
registry correctly rejected it — and the rejection was recorded, which is the only reason I
found it.
**What I'd tell someone** The registry's strict validation is what turned a silent
no-op into a visible failed tool call. That is the entire argument for validating arguments
against a schema before calling, and it paid for itself here.

### The plan replayed every round and burned the call budget
**E** Three investigation rounds, well inside the 25-call ceiling.
**A** `budget exceeded: LLM call budget of 25 exhausted`, on a task with no recognisable
cause.
**Why** Round two is handed a *freshly built* message list in which the evidence appears as
text. Progress was measured by counting `role: "tool"` messages — which reset to zero — so
the analyst re-asked every question, three rounds of eight calls.
**What I changed** Measure progress against the *evidence* (the `[source:tool]` markers)
rather than against message positions.
**What I'd tell someone** A loop that restarts its work on every iteration is a loop that
never terminates. If the loop counter lives in a structure that gets rebuilt, it is not a
counter.

### Four tool-layer bugs found by writing the tests
- `check_row_count` raised `NameError: comparable` — an earlier edit deleted the assignment
  and kept the use.
- `registry.call("no_such_tool")` **raised `KeyError`**, violating its own documented
  "never raises" contract. The planner is supposed to drop hallucinated tools before this
  point, but defence in depth means the call path has to survive one that slipped through.
- Database errors arrived multi-line (`... does not exist\nLINE 2: ...`), which breaks a
  structured log field and the prompt text. Collapsed to one line at the boundary.
- `get_pipeline_logs("../../etc/passwd")` returned "no runs recorded" — safe, but misleading:
  it told the caller the pipeline does not exist when the real problem was the name.
  Validation now happens before the lookup.
**What I'd tell someone** Three of those four were contract violations rather than wrong
answers, and contracts are exactly what tests are for.

---

## Phase 2 — The simulated estate

### `hash()` is not reproducible across processes
**E** A fixed RNG seed makes the fixture byte-identical on every seed.
**A** It would have differed between runs.
**Why** Python randomises string hashing per process (`PYTHONHASHSEED`), so
`hash("missing_partition")` is a different number every time.
**What I changed** `zlib.crc32(scenario.id.encode())`.
**What I'd tell someone** "I seeded the RNG" is not the same as "this is reproducible". I
verified it by hashing the resulting rows and seeding twice.

### The fixture's own log line was a lie
**E** The data-quality scenario's log says "62 of 200 rows have a null eligibility_status".
**A** The table was generated with `random() < 0.31`, so the real count was 61 or 63.
**Why** Probabilistic generation with a stated exact number.
**What I changed** Generate exactly `round(rate * count)` nulls, and derive the numbers
quoted in the log from the same constants.
**What I'd tell someone** In a fixture, an inconsistency between the narrative and the data
teaches the agent to distrust evidence — and teaches you to distrust the fixture. Derive
the prose from the data or do not quote numbers.

### An off-by-one between run dates and partition dates
Six successful runs on days D produced partitions D, when a run on day D processes D−1. The
missing-partition fixture was therefore only missing by accident. Fixed by making the
partition list explicitly `ANCHOR-7 … ANCHOR-2`, with the comment explaining why the failing
run's partition is absent *by default*.

---

## Phase 3 — The SQL boundary

### A security guard that raises `AttributeError` on import is worse than a slightly weaker one
**E** Reject `Insert`, `Update`, `Delete`, `Drop`, `Alter`, `AlterTable`, `Call`, …
**A** `AttributeError: module 'sqlglot.expressions' has no attribute 'AlterTable'`.
**Why** sqlglot 30.x renamed and removed node classes between majors.
**What I changed** Build the forbidden-node list **by name**, skipping names this version
does not provide, and expose `missing_guard_nodes()` so a test can fail if a dependency bump
silently removes a protection.
**What I'd tell someone** A security boundary that depends on third-party class names has an
implicit upgrade hazard. Make the hazard observable — a silent hole in a guard is the worst
kind, because the guard still looks like it is working.

### The bypass I designed for, and the one I nearly missed
The multi-statement attack (`select 1; drop table x`) was the known target. The one worth
naming is the **data-modifying CTE**:

```sql
WITH deleted AS (DELETE FROM warehouse.claims RETURNING *) SELECT * FROM deleted
```

The root node is a `SELECT`. A root-only type check passes it and it deletes the data. The
guard walks *every* node in the tree for that reason, and there is a test named after it.

---

## Phase 4 — The workflow

### The graph worked on the first run, which was suspicious
It did — because the analyst was answering from the error string alone and never needed the
tools. The first honest measurement was after fixing the evidence join: 4/5 categories,
1/5 actions. The remaining defects were the ones above.

### `interrupted` was asserted on the wrong state
**E** A data-changing action pauses for approval.
**A** The check failed for every task that required approval.
**Why** The observation read `interrupted` from the **final** state, which is naturally
un-paused after resuming. The property being evaluated is "did this action require a human",
which is a fact about the *run*, not the *end state*.
**What I changed** Record the pause when it happens.
**What I'd tell someone** An evaluation asserting on final state can only see what survived,
not what happened. If the behaviour is "paused for a human", the assertion has to observe
the pause, not the aftermath.

---

## Decisions and trade-offs

| Decision | Alternative rejected | Why | Cost accepted |
|---|---|---|---|
| The offline analyst is a real rule-based baseline | A mock that returns canned answers | A mock makes the evaluation tautological; a baseline gives CI a *real* number and something to beat | The baseline's limitations are now part of the eval's limitations |
| Heuristic + LiteLLM behind one `LLMClient` protocol | Calling LiteLLM directly from nodes | Nodes cannot couple to a provider; the offline path exercises the same prompts | One more layer to read |
| Parse + read-only transaction + schema allow-list | A prefix check | The prefix check is bypassable; the engine is not | `COPY … TO` still needs the parse layer, and that is documented rather than hidden |
| `interrupted` recorded at pause time | Read from the final state | The final state cannot answer a question about the run | One extra local variable |
| Domain models don't inherit across the alert/persisted boundary | `Incident(IncidentCreate)` | The alert's `status` is `Literal["FAILED"]`; an incident's is a lifecycle. Sharing the name across inheritance silently conflates them | A few duplicated field declarations |
| `platform` schema over `public` for platform state | Everything in `public` | The LangGraph checkpointer owns `public.*`; keeping them apart makes "what is ours?" a schema-qualified question | One more schema |
| `/health` reports capability, not just liveness | `{"status": "ok"}` | "The thing is up" does not tell an operator whether it is configured | Slightly more code in one endpoint |
| An unseeded estate warns and serves | Refuse to start | A missing fixture should not be an outage — but it must be visible, so `/health` reports `estate.seeded: false` | A server can run in a useless state, loudly |
| Evaluation suite in YAML | Python task list | Reviewable in a PR by someone who does not read the agent code | A loader to write, and a schema to validate |
| Malformed suite raises | Fall back to a built-in list | A typo means your task never runs while the report is green. For a CI gate, a failure that looks like success is the worst outcome | A missing file breaks the build instead of quietly substituting |
| Budget exhaustion fails the run | Stop calling and continue | A ceiling that does not fail is not a ceiling | A task can fail for a reason unrelated to the agent's reasoning |
| Single shared DB connection in the API | Per-request repository from the pool | The alternative threads a repository through every graph node; the concurrency is not needed yet | Two simultaneous requests serialise — documented, not hidden |

---

## Things I could not answer yet

- **Is the confidence signal any good?** Five scenarios cannot answer it. Doing this
  properly needs enough labelled incidents to bin confidence against correctness, and the
  honest current answer is "unknown".
- **Would a real model do better or worse than the baseline?** Unknown without an API key.
  The harness is built to answer it: point `LLM_MODEL` at a provider, run
  `python -m app.evaluation --json`, compare. I expect better root causes and worse
  determinism, and I would want the number before claiming either.
- **Does LiteLLM's LangSmith callback actually emit traced spans?** Configured, not
  observed. The local harness is what gates CI, so this is an observability gap rather than
  a correctness one — but it is unverified and the README says so.

---

## Things I would do differently

- **Write the evaluation before the third node.** The harness found the `interrupted` bug,
  the evidence-payload bug and the plan-replay bug within minutes of existing; the same bugs
  had survived a working demo and a green unit suite.
- **Give the analyst one source of structured evidence from the start.** Both the
  "evidence as text" and "remediation re-classifies everything" bugs came from the same
  root cause: the prompt was treated as the only interface, so structure was thrown away and
  re-derived by regex.
- **Assert on the run, not just the end state,** everywhere — not only for `interrupted`.
  `remediation_executed`, the number of rounds, and whether approval happened are all
  run-level facts that a final state can obscure.
