"""Keep the committed seed database out of every benchmarks test run.

``dashboard.backend.database`` runs schema migrations AT IMPORT TIME against
whatever ``DATABASE_PATH`` resolves to. Unset, that is
``dashboard/storage/data/backtest.db`` — the committed seed database that this
suite asserts is byte-identical. So merely importing a dashboard module from a
test rewrites the fixture the next test checks.

That happened twice while Phase 20 was being written: once from the tracing
script and once from its own test module. Pointing DATABASE_PATH at a
throwaway file here, before any test imports anything, makes it impossible
rather than remembered.

This runs at collection time, which is early enough: pytest imports conftest
before the test modules whose top-level imports would trigger the migration.
"""

import os
import tempfile

_SCRATCH = os.path.join(tempfile.gettempdir(), "atl_benchmarks_scratch.db")

if not os.environ.get("DATABASE_PATH"):
    os.environ["DATABASE_PATH"] = _SCRATCH
