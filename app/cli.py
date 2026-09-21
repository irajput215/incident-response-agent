"""The command-line interface.

``argparse`` with ``set_defaults(func=...)`` dispatch rather than an if/elif
chain: adding a command is then a function plus one registration, and the
dispatch table is inspectable.

Every command goes through the same three steps — configure logging, configure
tracing, build the platform — so a command cannot accidentally run without
observability, and ``main(argv)`` takes its arguments explicitly so the whole
CLI is testable in-process.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from app import __version__
from app.config import get_settings
from app.llm import build_llm
from app.observability import get_logger, log, setup_logging
from app.tracing import configure_tracing

_log = get_logger("app.cli")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_seed(args: argparse.Namespace) -> int:
    """Materialise the simulated estate: metadata, logs and warehouse rows."""
    from app.scenarios import seed

    manifest = seed()
    for key, value in manifest.items():
        print(f"{key:18} {value}")
    return 0


def cmd_scenarios(args: argparse.Namespace) -> int:
    """List the incidents in the scenario catalogue."""
    from app.scenarios import SCENARIOS

    print(f"{'id':22} {'pipeline':32} {'expected root cause':32} {'expected action'}")
    print("-" * 120)
    for scenario in SCENARIOS:
        print(
            f"{scenario.id:22} {scenario.pipeline:32} "
            f"{scenario.expected_category:32} {scenario.expected_action}"
        )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Investigate one incident end to end and print the report."""
    from app.db import IncidentRepository, connect
    from app.scenarios import SCENARIOS_BY_ID
    from app.schemas import ApprovalDecision, IncidentCreate

    settings = get_settings()
    if args.scenario not in SCENARIOS_BY_ID:
        print(
            f"unknown scenario {args.scenario!r}; try one of "
            f"{', '.join(SCENARIOS_BY_ID)}",
            file=sys.stderr,
        )
        return 2

    scenario = SCENARIOS_BY_ID[args.scenario]
    alert = IncidentCreate.model_validate(scenario.incident)

    with connect(settings) as conn:
        from app.agents import IncidentAgent
        from app.db.schema import ensure_schema

        ensure_schema(conn)
        agent = IncidentAgent(
            settings=settings,
            repository=IncidentRepository(conn),
            llm=build_llm(settings),
        )

        print(f"\nscenario   {scenario.id} — {scenario.title}")
        print(f"incident   {scenario.pipeline} / {scenario.incident['run_id']}")
        print(f"error      {str(scenario.incident.get('error', ''))[:110]}")

        outcome = agent.investigate(alert)
        _print_timeline(outcome)

        if outcome.awaiting_approval:
            request = outcome.approval_request or {}
            remediation = request.get("remediation", {})
            print("\n⏸  paused for human approval")
            print(f"   proposed  {remediation.get('action')} on {remediation.get('target')}")
            print(f"   risk      {remediation.get('risk')}")
            if args.reject:
                print("   decision  REJECTED (--reject)")
                outcome = agent.resume(
                    outcome.investigation_id,
                    ApprovalDecision(approved=False, approver="cli", note="rejected via CLI"),
                )
            else:
                print("   decision  approved (use --reject to decline)")
                outcome = agent.resume(
                    outcome.investigation_id,
                    ApprovalDecision(approved=True, approver="cli", note="approved via CLI"),
                )

        _print_report(outcome, expected=scenario.expected_category)
    return 0


def _print_timeline(outcome: Any) -> None:
    print("\ntimeline")
    for entry in outcome.state.get("timeline", []):
        print(f"  [{entry['node']:20}] {entry['summary'][:104]}")


def _print_report(outcome: Any, *, expected: str | None = None) -> None:
    report = outcome.report
    if report is None:
        print(f"\nno report; status={outcome.status.value} error={outcome.error}")
        return

    print("\n" + "=" * 76)
    print(f"INCIDENT REPORT  {report.incident_id}")
    print("=" * 76)
    print(f"status      {report.status.value}")
    print(f"pipeline    {report.pipeline} / {report.run_id}")
    print(f"severity    {report.severity.value if report.severity else '-'}")
    print(f"category    {report.category.value if report.category else '-'}"
          + (f"   (expected {expected})" if expected else ""))
    print(f"summary     {report.summary}")

    if report.root_cause:
        print(f"\nroot cause  {report.root_cause.root_cause}")
        print(f"confidence  {report.root_cause.confidence:.2f} "
              f"(conclusive={report.root_cause.is_conclusive})")
        for line in report.root_cause.evidence[:6]:
            print(f"            · {line}")

    if report.remediation:
        print(f"\nremediation {report.remediation.action.value} → {report.remediation.target}")
        print(f"risk        {report.remediation.risk.value}")
        print(f"expected    {report.remediation.expected_effect}")

    if report.approval:
        verdict = "approved" if report.approval.approved else "rejected"
        print(f"approval    {verdict} by {report.approval.approver}")

    print(f"\nevidence    {len(report.evidence)} observation(s)")
    for item in report.evidence:
        print(f"            [{item.source.value}] {item.summary[:92]}")

    print(f"\ncost        {report.llm_calls} llm calls · {report.tool_calls} tool calls · "
          f"{report.tokens_in + report.tokens_out} tokens · {report.duration_ms:.0f}ms · "
          f"model={report.model}")
    print(f"rounds      {report.investigation_rounds}")


def cmd_eval(args: argparse.Namespace) -> int:
    """Run the agent evaluation suite."""
    from app.evaluation.harness import main as eval_main

    argv = ["--min-accuracy", str(args.min_accuracy)]
    if args.suite:
        argv += ["--suite", args.suite]
    if args.report:
        argv += ["--report", args.report]
    if args.json:
        argv.append("--json")
    for task_id in args.task or []:
        argv += ["--task", task_id]
    return eval_main(argv)


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the HTTP API."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=args.host or settings.api_host,
        port=args.port or settings.api_port,
        reload=args.reload,
        log_config=None,
    )
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Mark investigations that will never finish."""
    from app.db import IncidentRepository, connect
    from app.db.schema import ensure_schema

    with connect(get_settings()) as conn:
        ensure_schema(conn)
        swept = IncidentRepository(conn).sweep_orphaned_investigations(args.max_age_minutes)

    print(f"swept {len(swept)} orphaned investigation(s)")
    for investigation_id in swept:
        print(f"  {investigation_id}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Probe a running server's ``/health``."""
    import httpx

    settings = get_settings()
    url = args.url or f"http://{settings.api_host}:{settings.api_port}/health"
    try:
        response = httpx.get(url, timeout=5.0)
    except httpx.HTTPError as exc:
        print(f"cannot reach {url}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(response.json(), indent=2, default=str))
    return 0 if response.status_code == 200 else 1


# --------------------------------------------------------------------------- #
# wiring
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adp-agent",
        description="AI Data Engineering Incident Response Agent",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed = subparsers.add_parser("seed", help="create the simulated estate and warehouse")
    seed.set_defaults(func=cmd_seed)

    scenarios = subparsers.add_parser("scenarios", help="list the incident catalogue")
    scenarios.set_defaults(func=cmd_scenarios)

    demo = subparsers.add_parser("demo", help="investigate one incident end to end")
    demo.add_argument("scenario", nargs="?", default="missing_partition")
    demo.add_argument(
        "--reject", action="store_true", help="decline the remediation instead of approving it"
    )
    demo.set_defaults(func=cmd_demo)

    evaluate = subparsers.add_parser("eval", help="run the agent evaluation suite")
    evaluate.add_argument("--min-accuracy", type=float, default=1.0)
    evaluate.add_argument("--suite", default=None)
    evaluate.add_argument("--report", default=None)
    evaluate.add_argument("--task", action="append", default=None)
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(func=cmd_eval)

    serve = subparsers.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    sweep = subparsers.add_parser("sweep", help="mark orphaned investigations as failed")
    sweep.add_argument("--max-age-minutes", type=int, default=30)
    sweep.set_defaults(func=cmd_sweep)

    health = subparsers.add_parser("health", help="probe a running server")
    health.add_argument("--url", default=None)
    health.set_defaults(func=cmd_health)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = get_settings()
    setup_logging(settings.log_level, json_output=settings.log_json)
    configure_tracing(settings)
    log(_log, 20, "cli_invoked", command=args.command)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive path
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
