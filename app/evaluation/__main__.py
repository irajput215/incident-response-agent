"""``python -m app.evaluation`` — run the agent evaluation suite.

A dedicated ``__main__`` rather than telling people to run
``python -m app.evaluation.harness``: the package ``__init__`` imports the
harness, so the latter triggers a ``RuntimeWarning`` about a module found in
``sys.modules`` before execution. Noisy warnings in a CI log train people to
ignore warnings.
"""

from app.evaluation.harness import main

if __name__ == "__main__":
    raise SystemExit(main())
