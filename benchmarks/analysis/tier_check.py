"""Check that every numeric claim in a generated report carries an evidence tier.

The rule this enforces: **a line that asserts a number must say where the number
came from.** Untagged numbers are how an assumption becomes a finding, which is
the specific failure this report exists to avoid.

Exemptions are narrow and each has a reason:

* headings, table separators, and table header rows assert nothing;
* fenced code blocks are quoted source, not claims — the claim is the prose
  around them;
* the tier legend itself defines the vocabulary;
* bare `file:line` citations and hashes are provenance, and provenance IS the
  evidence, so requiring a tag on them would be circular.

Used by ``make_advisor_report.py --check`` and by
``tests/test_advisor_report.py``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

__all__ = ["TIER_TOKENS", "find_untagged_numeric_lines", "check_report"]

TIER_TOKENS = ("[MEASURED]", "[DERIVED]", "[NOT MEASURED]", "[ASSUMED",
               "| MEASURED |", "| DERIVED |", "| NOT MEASURED |", "| ASSUMED",
               "`MEASURED`", "`DERIVED`", "`NOT MEASURED`", "`ASSUMED")

# ASSUMED is a fourth tier used where an input is present and load-bearing but
# unconfirmed — the Nof1 cadence is the only current instance. It is NOT
# MEASURED (no run), NOT DERIVED (an input, not arithmetic), and NOT
# "NOT MEASURED" (that tier means absent). Matched as a prefix so the
# qualifier — "[ASSUMED — not confirmed with advisor]" — travels with it.

# A digit that is part of a real quantity, not part of an identifier. Excludes
# digits inside backticked spans, which are code/identifiers rather than claims.
_NUMERIC = re.compile(r"\d")
_BACKTICKED = re.compile(r"`[^`]*`")
_CITATION_ONLY = re.compile(r"^\s*[-*|]?\s*`?[\w/.\-]+\.(py|json|md):\d+`?")
_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


_BARE_TIER = re.compile(r"\b(MEASURED|DERIVED|NOT MEASURED|ASSUMED)\b")
_ORDERED_MARKER = re.compile(r"^\s*\d+\.\s")


def _strip_noise(line: str) -> str:
    """Remove spans that carry identifiers rather than quantitative claims."""
    out = _BACKTICKED.sub("", line)
    out = re.sub(r"\]\([^)]*\)", "]", out)   # link targets
    out = _ORDERED_MARKER.sub("", out)       # "3." is a list marker, not a claim
    return out


def _row_is_tagged(line: str) -> bool:
    """A table row is tagged when any cell names a tier.

    Checked per cell rather than by exact string so a Tier cell may carry a
    qualifier — "MEASURED (ratio DERIVED)" is a legitimate cell.
    """
    return any(_BARE_TIER.search(cell) for cell in line.split("|"))


def find_untagged_numeric_lines(text: str) -> List[Dict[str, object]]:
    """Return every line that states a number without an evidence tier."""
    violations: List[Dict[str, object]] = []
    in_fence = False
    in_legend = False
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # The legend block defines the three tags; it runs from the "How to
        # read this" heading to the blank line after its table.
        if stripped.startswith("## How to read this"):
            in_legend = True
            continue
        if in_legend:
            if stripped.startswith("##"):
                in_legend = False
            else:
                continue

        if not stripped:
            continue
        if stripped.startswith("#"):          # headings
            continue
        if _TABLE_SEP.match(line):            # |---|---|
            continue
        if stripped.startswith("|") and "Tier" in line:   # table header
            continue
        if _CITATION_ONLY.match(line):        # provenance line
            continue

        payload = _strip_noise(line)
        if not _NUMERIC.search(payload):
            continue
        if stripped.startswith("|"):
            if _row_is_tagged(line):
                continue
        elif any(tok in line for tok in TIER_TOKENS):
            continue
        violations.append({"line_no": i, "text": stripped})
    return violations


def check_report(text: str) -> Dict[str, object]:
    v = find_untagged_numeric_lines(text)
    return {
        "ok": not v,
        "violations": v,
        "n_violations": len(v),
        "rule": ("every line asserting a number must carry [MEASURED], "
                 "[DERIVED] or [NOT MEASURED]"),
    }


def format_violations(violations: Sequence[Dict[str, object]]) -> str:
    return "\n".join(
        f"  line {v['line_no']}: {str(v['text'])[:110]}" for v in violations)
