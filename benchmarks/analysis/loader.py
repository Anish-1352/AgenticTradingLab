"""Load run artifacts and refuse to compare runs that are not comparable.

Every analysis entry point goes through :func:`load_run` and
:func:`require_comparable`. The guard is the reason this package exists in the
form it does: the entire study is a defence against comparing numbers that came
from different hardware, different prompts, or different software, and an
analysis layer that quietly averaged across those would reintroduce exactly the
failure the harness was built to prevent.

So a mismatch is a **hard refusal**, not a warning. A warning gets scrolled
past; a non-zero exit does not.

Pure stdlib — no torch, no pandas, no GPU.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "GUARDED_FIELDS",
    "ALWAYS_GUARDED",
    "CROSS_ARM_CONTROLS",
    "config_check_mode",
    "ProvenanceMismatch",
    "RunSet",
    "load_run",
    "load_runs",
    "require_comparable",
    "describe_provenance",
    "shared_concurrencies",
]

# A difference in any of these makes two runs incomparable. Each one has a
# concrete failure attached to it:
#   gpu_uuid          - Colab reallocates cards; a B-vs-C delta across two
#                       A100s contains an unknown hardware term.
#   fixture_sha256    - different prompts means different prefill work and,
#                       for arm C, a different prefix-cache hit rate.
#   config_sha256     - the resolved config carries every controlled variable.
#   max_new_tokens    - decode length dominates every per-token metric.
#   pip_freeze_sha256 - a moved torch/vLLM is a different serving stack.
GUARDED_FIELDS: Tuple[str, ...] = (
    "gpu_uuid",
    "fixture_sha256",
    "config_sha256",
    "max_new_tokens",
    "pip_freeze_sha256",
)

# Enforced no matter which arms are involved.
ALWAYS_GUARDED: Tuple[str, ...] = (
    "gpu_uuid",
    "fixture_sha256",
    "max_new_tokens",
    "pip_freeze_sha256",
)

# ``config_sha256`` hashes the RESOLVED config, which for arm C includes the
# vLLM serving knobs (enable_prefix_caching, max_num_seqs,
# gpu_memory_utilization). Those knobs ARE the independent variable of a B-vs-C
# comparison, so two different arms can never share a config hash — enforcing
# equality across arms would make the study's headline comparison permanently
# impossible, and routinely waiving it would hollow the guard out.
#
# So the check is arm-aware:
#   * same arm  -> config_sha256 must match. A workload change inside one arm
#                  is a genuine red flag.
#   * different arms -> config_sha256 is EXPECTED to differ, and the shared
#                  controlled variables below are checked individually instead.
#                  Nothing is relaxed: every quantity that must be held constant
#                  is still verified, just from first-class manifest fields
#                  rather than through an aggregate hash.
SAME_ARM_ONLY_GUARDED: Tuple[str, ...] = ("config_sha256",)
CROSS_ARM_CONTROLS: Tuple[str, ...] = (
    "model",
    "context_tokens",
    "fixture_name",
)


class ProvenanceMismatch(RuntimeError):
    """Raised when runs disagree on a field that makes them incomparable."""


@dataclass
class RunSet:
    """One run's artifacts: summary + manifest, with raw/monitor loaded lazily."""

    run_id: str
    summary_path: str
    summary: Dict[str, Any]
    manifest: Dict[str, Any] = field(default_factory=dict)
    manifest_path: Optional[str] = None
    raw_path: Optional[str] = None
    label: Optional[str] = None

    # ---- identity ----

    @property
    def arm(self) -> str:
        return (
            self.summary.get("arm")
            or self.manifest.get("arm")
            or "?"
        )

    @property
    def display(self) -> str:
        return self.label or f"Arm {self.arm} ({self.run_id})"

    def guarded(self) -> Dict[str, Any]:
        return {k: self.manifest.get(k) for k in GUARDED_FIELDS}

    # ---- levels ----

    @property
    def levels(self) -> List[Dict[str, Any]]:
        return list(self.summary.get("levels") or [])

    def level(self, concurrency: int) -> Optional[Dict[str, Any]]:
        for lv in self.levels:
            if int(lv.get("concurrency", -1)) == int(concurrency):
                return lv
        return None

    def concurrencies(self) -> List[int]:
        return sorted(int(lv["concurrency"]) for lv in self.levels
                      if lv.get("concurrency") is not None)

    # ---- lazy artifacts ----

    def raw(self) -> Dict[str, Any]:
        """Per-request records, including per-token timestamps."""
        path = self.raw_path or self.summary_path.replace("_summary.json", "_raw.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"raw records not found for {self.run_id}: {path}")
        with open(path) as fh:
            return json.load(fh)

    def records(self) -> List[Dict[str, Any]]:
        return list(self.raw().get("records") or [])

    def monitor_path(self, concurrency: int) -> str:
        base = os.path.dirname(self.summary_path)
        return os.path.join(base, f"{self.run_id}_c{concurrency}_monitor.csv")

    def monitor(self, concurrency: int) -> List[Dict[str, Any]]:
        """20 Hz resource samples for one level. Numeric fields coerced."""
        path = self.monitor_path(concurrency)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"monitor CSV not found for {self.run_id} c={concurrency}: {path}"
            )
        rows: List[Dict[str, Any]] = []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                out: Dict[str, Any] = {}
                for key, value in row.items():
                    if value is None or value == "":
                        out[key] = None
                        continue
                    try:
                        out[key] = float(value)
                    except (TypeError, ValueError):
                        out[key] = value
                rows.append(out)
        return rows


def load_run(
    summary_path: str,
    label: Optional[str] = None,
    manifest_path: Optional[str] = None,
) -> RunSet:
    """Load a ``*_summary.json`` plus its sibling manifest."""
    with open(summary_path) as fh:
        summary = json.load(fh)

    run_id = summary.get("run_id") or os.path.basename(summary_path).replace(
        "_summary.json", ""
    )

    mpath = manifest_path or summary_path.replace("_summary.json", "_manifest.json")
    manifest: Dict[str, Any] = {}
    if os.path.exists(mpath):
        with open(mpath) as fh:
            manifest = json.load(fh)
    else:
        mpath = None  # type: ignore[assignment]

    return RunSet(
        run_id=run_id,
        summary_path=summary_path,
        summary=summary,
        manifest=manifest,
        manifest_path=mpath,
        label=label,
    )


def load_runs(paths: Sequence[str], labels: Optional[Sequence[str]] = None) -> List[RunSet]:
    labels = list(labels or [])
    out = []
    for i, p in enumerate(paths):
        out.append(load_run(p, label=labels[i] if i < len(labels) else None))
    return out


def require_comparable(runs: Sequence[RunSet], allow: Sequence[str] = ()) -> None:
    """Raise ProvenanceMismatch unless every run agrees on the guarded fields.

    ``allow`` names fields to exempt. It exists for the deliberate case — e.g.
    an intentional cross-stack comparison — and every exemption is echoed into
    the emitted report, so a waived guard can never be invisible in the output.
    """
    if len(runs) < 2:
        return

    missing_manifests = [r.run_id for r in runs if not r.manifest]
    if missing_manifests:
        raise ProvenanceMismatch(
            "REFUSING TO COMPARE: no manifest found for "
            + ", ".join(missing_manifests)
            + "\n\nA run without a manifest cannot be shown to be comparable to "
              "anything. Locate its *_manifest.json or exclude the run."
        )

    def disagreement(field_name: str, subset: Sequence[RunSet]) -> Optional[str]:
        seen: Dict[Any, List[str]] = {}
        for r in subset:
            seen.setdefault(r.manifest.get(field_name), []).append(r.run_id)
        if len(seen) <= 1:
            return None
        lines = [f"  {field_name}:"]
        for value, ids in seen.items():
            shown = value if value is not None else "<missing>"
            lines.append(f"    {shown}")
            for rid in ids:
                lines.append(f"      <- {rid}")
        return "\n".join(lines)

    arms = {r.arm for r in runs}
    cross_arm = len(arms) > 1
    problems: List[str] = []

    for field_name in ALWAYS_GUARDED:
        if field_name in allow:
            continue
        found = disagreement(field_name, runs)
        if found:
            problems.append(found)

    if cross_arm:
        # config_sha256 legitimately differs between arms; verify the shared
        # controlled variables directly instead.
        for field_name in CROSS_ARM_CONTROLS:
            if field_name in allow:
                continue
            found = disagreement(field_name, runs)
            if found:
                problems.append(found)
        # …and require it to be internally consistent WITHIN each arm.
        for arm in sorted(arms):
            same = [r for r in runs if r.arm == arm]
            if len(same) < 2 or "config_sha256" in allow:
                continue
            found = disagreement("config_sha256", same)
            if found:
                problems.append(
                    f"  (within arm {arm}) " + found.lstrip().replace("\n  ", "\n  ")
                )
    else:
        for field_name in SAME_ARM_ONLY_GUARDED:
            if field_name in allow:
                continue
            found = disagreement(field_name, runs)
            if found:
                problems.append(found)

    if problems:
        raise ProvenanceMismatch(
            "REFUSING TO COMPARE — runs disagree on fields that make them "
            "incomparable:\n\n"
            + "\n\n".join(problems)
            + "\n\n"
            + _mismatch_guidance()
        )


def _mismatch_guidance() -> str:
    return (
        "This is a hard refusal, not a warning. A silent cross-card or\n"
        "cross-fixture comparison is the exact failure this study was built to\n"
        "avoid: the resulting delta would contain an unknown hardware or\n"
        "workload term and would not be a property of the serving stack.\n\n"
        "Options:\n"
        "  - Re-run the arms so they match on the differing field.\n"
        "  - Exclude the offending run.\n"
        "  - If the difference is deliberate, pass --allow-mismatch <field>.\n"
        "    The waiver is recorded in the output so the reader sees it too."
    )


def config_check_mode(runs: Sequence[RunSet]) -> Dict[str, Any]:
    """How config_sha256 was verified, so the report can state it."""
    arms = sorted({r.arm for r in runs})
    if len(arms) <= 1:
        return {"cross_arm": False, "mode": "config_sha256 equality",
                "note": "All runs share an arm, so the resolved config hash "
                        "must match exactly."}
    return {
        "cross_arm": True,
        "arms": arms,
        "mode": "shared controlled variables",
        "checked": list(CROSS_ARM_CONTROLS),
        "note": (
            "Runs span arms " + ", ".join(arms) + ". config_sha256 hashes the "
            "RESOLVED config, which for arm C includes the vLLM serving knobs "
            "(enable_prefix_caching, max_num_seqs, gpu_memory_utilization) — "
            "those knobs are the independent variable, so the hashes differ by "
            "design. The shared controlled variables were verified individually "
            "instead (" + ", ".join(CROSS_ARM_CONTROLS) + "), alongside "
            + ", ".join(ALWAYS_GUARDED) + ". Nothing that must be held constant "
            "went unchecked."
        ),
    }


def describe_provenance(runs: Sequence[RunSet]) -> Dict[str, Any]:
    """Common provenance across runs, plus per-run identity."""
    common: Dict[str, Any] = {}
    for key in GUARDED_FIELDS + ("gpu_name", "driver_version", "cuda_version",
                                 "upstream_sha", "branch_sha", "model",
                                 "context_tokens", "fixture_name",
                                 "torchvision_shim"):
        values = {r.manifest.get(key) for r in runs}
        common[key] = values.pop() if len(values) == 1 else None
    return {
        "common": common,
        "config_check": config_check_mode(runs),
        "runs": [
            {
                "run_id": r.run_id,
                "arm": r.arm,
                "label": r.display,
                "concurrencies": r.concurrencies(),
                "profiling_layer": r.manifest.get("profiling_layer"),
                "utc_timestamp": r.manifest.get("utc_timestamp"),
            }
            for r in runs
        ],
    }


def shared_concurrencies(runs: Sequence[RunSet]) -> List[int]:
    """Concurrency levels present in EVERY run.

    Comparison happens only at matched levels — reporting a level one arm
    reached and another did not would read as a gap in performance rather than
    a gap in the data.
    """
    if not runs:
        return []
    common = set(runs[0].concurrencies())
    for r in runs[1:]:
        common &= set(r.concurrencies())
    return sorted(common)
