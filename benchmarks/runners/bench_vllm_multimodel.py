#!/usr/bin/env python3
"""Heterogeneous serving — N distinct models on one card, load split across them.

    python benchmarks/runners/bench_vllm_multimodel.py --dry-run --n-models 4
    python benchmarks/runners/bench_vllm_multimodel.py --n-models 4 --quantization awq

WHAT vLLM 0.26 ACTUALLY PROVIDES — READ THIS BEFORE CHANGING THE DESIGN
------------------------------------------------------------------------
``AsyncEngineArgs`` takes ``model`` — **singular**. There is no parameter for a
model list and no first-class "serve N models from one engine" API. That is not
an oversight in this runner; it is the shape of the library. Three consequences
drive everything below:

1. **N models means N engines.** Each is a separate ``AsyncLLMEngine`` with its
   own scheduler, its own KV pool and its own CUDA memory claim.
2. **They do not share a batch.** vLLM's throughput comes from continuous
   batching *within* one scheduler. Two engines are two schedulers competing for
   one card's SMs — they interleave, they do not coalesce. This is the
   fragmentation this benchmark exists to quantify.
3. **``gpu_memory_utilization`` must be divided.** It is a *fraction of the
   whole card*, claimed at engine init. Two engines at the 0.9 default would
   each try to claim 90% and the second would OOM. The runner divides the
   budget by N and records the per-engine value, because getting this wrong
   looks like "multi-model does not work" rather than "the config was wrong".

**This was NOT verified against an installed vLLM.** There is no GPU and no
vllm package in the environment this was written in, so the above comes from
the API surface the existing arm C runner already uses
(``bench_vllm_optimized.py`` builds ``AsyncEngineArgs(model=...)``) plus the
documented semantics of ``gpu_memory_utilization``. The first action in a real
session is ``--dry-run`` followed by the N=1 validation run; if
``AsyncEngineArgs`` in the installed build accepts something better, prefer it
and delete this note.

THE ALTERNATIVE, NAMED RATHER THAN SILENTLY WORKED AROUND
-----------------------------------------------------------
If N in-process engines prove unworkable — OOM at init, or contention that
makes the numbers meaningless — the alternatives, in order of preference:

* ``--isolation process``: one engine per subprocess, each with its own CUDA
  context. Cleaner isolation, higher fixed overhead, and CUDA context memory is
  then paid N times. Sketched here, not implemented, because it should only be
  built if in-process fails.
* **LoRA adapters on one base model.** vLLM does support many adapters against
  a single base engine, and those DO share a batch and a scheduler. But that is
  a materially different regime — one base model, N fine-tunes — and it cannot
  represent seven architecturally distinct models. It is the right answer to a
  different question, and substituting it silently would answer that different
  question while appearing to answer this one.

WHAT IS NOT OBSERVABLE HERE
---------------------------
KV-cache usage and prefix-cache hit rate. vLLM 0.26 reported
``stats_source: null`` on this build in arm C, and nothing about running N
engines changes that. Absence is recorded as absence — ``None``, never ``0`` —
because "the engine did not report it" and "the cache returned nothing" are
different facts and only the second one is a result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from common import fixtures as fx_mod  # noqa: E402
from common.fixtures import resolve_fixture  # noqa: E402
from common.manifest import build_manifest, make_run_id  # noqa: E402
from common.metrics import (  # noqa: E402
    RequestRecord,
    attach_resources,
    console_lines,
    summarize,
    write_results,
)

ARM = "C-multi"

DEFAULT_ROSTER = os.path.join(_BENCH_ROOT, "configs", "model_roster.yaml")
DEFAULT_FIXTURE = "atl_realistic"
DEFAULT_TOTAL_CONCURRENCY = 32
DEFAULT_N_REQUESTS = 32
DEFAULT_MAX_NEW_TOKENS = 860          # production Nemotron mean, not 256
DEFAULT_GPU_MEM_UTIL = 0.90           # WHOLE-CARD budget, divided by N below

QUANTIZATION_CHOICES = ("none", "awq", "gptq")

__all__ = [
    "load_roster", "select_models", "per_engine_memory_fraction",
    "split_concurrency", "estimate_vram", "build_engine_configs",
    "MultiModelPlan", "main",
]


# --------------------------------------------------------------------------
# roster
# --------------------------------------------------------------------------


def load_roster(path: str = DEFAULT_ROSTER) -> Dict[str, Any]:
    """Read the roster. YAML if available, else a minimal parse.

    PyYAML is not guaranteed present in a bare Colab kernel, and failing to
    load the roster would be a silly reason to lose a GPU session.
    """
    try:
        import yaml  # noqa: PLC0415

        with open(path) as fh:
            return yaml.safe_load(fh)
    except ImportError:
        raise SystemExit(
            "PyYAML is required to read the roster: pip install pyyaml"
        )


def select_models(roster: Dict[str, Any], *, n_models: Optional[int] = None,
                  models: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Explicit ``--models`` wins; otherwise the first N of the roster.

    Roster order is deliberate (ascending size), so "first N" is a defined
    selection rather than whatever the file happened to list.
    """
    entries = list(roster.get("models") or [])
    if models:
        by_id = {e["id"]: e for e in entries}
        chosen = []
        for m in models:
            if m not in by_id:
                # An unknown id is still usable — it just has no VRAM estimate.
                chosen.append({"id": m, "params_b": None, "fp16_gb": None,
                               "awq_4bit_gb": None, "unknown_to_roster": True})
            else:
                chosen.append(by_id[m])
        return chosen
    if n_models is None:
        n_models = 1
    if n_models > len(entries):
        raise SystemExit(
            f"--n-models {n_models} exceeds the roster's {len(entries)} entries")
    return entries[:n_models]


# --------------------------------------------------------------------------
# the two things that make N engines different from one
# --------------------------------------------------------------------------


def per_engine_memory_fraction(total_fraction: float, n_models: int) -> float:
    """Divide the card's memory budget across engines.

    ``gpu_memory_utilization`` is a fraction of the WHOLE card, applied at
    engine init. N engines each passed 0.9 would each try to claim 90% and all
    but the first would OOM. Dividing is not a tuning choice; it is what makes
    N engines start at all.
    """
    if n_models < 1:
        raise ValueError("n_models must be >= 1")
    return total_fraction / float(n_models)


def split_concurrency(total: int, n_models: int) -> List[int]:
    """Split offered load across engines, remainder to the earliest.

    Total is held fixed as N rises, so per-model concurrency falls. That is the
    independent variable: the same offered load, batched N ways.
    """
    if n_models < 1:
        raise ValueError("n_models must be >= 1")
    base, extra = divmod(total, n_models)
    return [base + (1 if i < extra else 0) for i in range(n_models)]


def estimate_vram(models: Sequence[Dict[str, Any]], quantization: str
                  ) -> Dict[str, Any]:
    """Sum the roster's weight estimates. Weights only — not KV, not activation.

    Arithmetic on published parameter counts, and it is here to catch a
    configuration that cannot possibly fit before a session is spent
    discovering that. The sweep measures the real number.
    """
    key = "fp16_gb" if quantization == "none" else "awq_4bit_gb"
    known = [m for m in models if m.get(key) is not None]
    unknown = [m["id"] for m in models if m.get(key) is None]
    total = sum(float(m[key]) for m in known)
    return {
        "quantization": quantization,
        "weights_gb": total,
        "models_costed": len(known),
        "models_without_estimate": unknown,
        "tier": "DERIVED",
        "excludes": "KV cache and activation peaks, which vLLM allocates on top",
    }


@dataclass
class EngineConfig:
    """One engine's worth of configuration."""

    model_id: str
    concurrency: int
    gpu_memory_utilization: float
    quantization: Optional[str]
    params_b: Optional[float] = None
    weights_gb_estimate: Optional[float] = None


@dataclass
class MultiModelPlan:
    """Everything the run needs, computed before anything is loaded."""

    engines: List[EngineConfig]
    total_concurrency: int
    n_models: int
    quantization: str
    vram: Dict[str, Any]
    max_new_tokens: int
    fixture_name: str
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_models": self.n_models,
            "total_concurrency": self.total_concurrency,
            "quantization": self.quantization,
            "max_new_tokens": self.max_new_tokens,
            "fixture_name": self.fixture_name,
            "vram_estimate": self.vram,
            "warnings": self.warnings,
            "engines": [
                {
                    "model": e.model_id,
                    "concurrency": e.concurrency,
                    "gpu_memory_utilization": e.gpu_memory_utilization,
                    "quantization": e.quantization,
                    "params_b": e.params_b,
                    "weights_gb_estimate": e.weights_gb_estimate,
                }
                for e in self.engines
            ],
        }


def build_engine_configs(
    models: Sequence[Dict[str, Any]],
    *,
    total_concurrency: int,
    quantization: str,
    gpu_memory_utilization: float,
    max_new_tokens: int,
    fixture_name: str,
    card_vram_gb: float = 80.0,
) -> MultiModelPlan:
    n = len(models)
    per_engine_frac = per_engine_memory_fraction(gpu_memory_utilization, n)
    concurrencies = split_concurrency(total_concurrency, n)
    vram = estimate_vram(models, quantization)
    key = "fp16_gb" if quantization == "none" else "awq_4bit_gb"

    engines = [
        EngineConfig(
            model_id=m["id"],
            concurrency=c,
            gpu_memory_utilization=per_engine_frac,
            quantization=None if quantization == "none" else quantization,
            params_b=m.get("params_b"),
            weights_gb_estimate=m.get(key),
        )
        for m, c in zip(models, concurrencies)
    ]

    warnings: List[str] = []
    usable = card_vram_gb * gpu_memory_utilization
    if vram["weights_gb"] > usable:
        warnings.append(
            f"WEIGHTS ALONE ({vram['weights_gb']:.1f} GB) EXCEED the usable "
            f"budget ({usable:.1f} GB = {card_vram_gb:.0f} GB x "
            f"{gpu_memory_utilization}). This configuration cannot load. "
            f"Quantize, or reduce N."
        )
    elif vram["weights_gb"] > usable * 0.8:
        warnings.append(
            f"Weights ({vram['weights_gb']:.1f} GB) take over 80% of the usable "
            f"budget ({usable:.1f} GB), leaving little for {n} KV pool(s). "
            f"Expect short max_model_len or OOM under load."
        )
    if vram["models_without_estimate"]:
        warnings.append(
            f"No VRAM estimate for {vram['models_without_estimate']} — they are "
            f"absent from the roster, so the fit check above is incomplete."
        )
    if any(c == 0 for c in concurrencies):
        warnings.append(
            f"total_concurrency {total_concurrency} < n_models {n}: some "
            f"engines get zero requests and will idle while still holding VRAM."
        )

    return MultiModelPlan(
        engines=engines, total_concurrency=total_concurrency, n_models=n,
        quantization=quantization, vram=vram, max_new_tokens=max_new_tokens,
        fixture_name=fixture_name, warnings=warnings,
    )


# --------------------------------------------------------------------------
# engine construction (mirrors bench_vllm_optimized's V0/V1 tolerance)
# --------------------------------------------------------------------------


def build_engine(model_id: str, *, gpu_memory_utilization: float,
                 quantization: Optional[str], dtype: str = "auto",
                 enable_prefix_caching: bool = True,
                 max_model_len: Optional[int] = None,
                 seed: int = 1234) -> Tuple[Any, str]:
    """One AsyncLLMEngine. ``model`` is singular — see the module docstring."""
    from vllm import AsyncEngineArgs  # noqa: PLC0415

    kwargs: Dict[str, Any] = dict(
        model=model_id,
        dtype=dtype,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        disable_log_stats=False,
    )
    if quantization:
        kwargs["quantization"] = quantization
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len

    args = AsyncEngineArgs(**kwargs)
    errors: List[str] = []
    try:
        from vllm import AsyncLLMEngine  # noqa: PLC0415

        return AsyncLLMEngine.from_engine_args(args), "vllm.AsyncLLMEngine"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"vllm.AsyncLLMEngine: {type(exc).__name__}: {exc}")
    try:
        from vllm.v1.engine.async_llm import AsyncLLM  # noqa: PLC0415

        return AsyncLLM.from_engine_args(args), "vllm.v1.engine.async_llm.AsyncLLM"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"vllm.v1.AsyncLLM: {type(exc).__name__}: {exc}")
    raise SystemExit(
        f"could not construct an engine for {model_id}. Attempts:\n  "
        + "\n  ".join(errors)
    )


def read_kv_stats(engine: Any) -> Dict[str, Any]:
    """Whatever the engine will say about its KV cache — usually nothing.

    Arm C found ``stats_source: null`` on vLLM 0.26: ``get_metrics()`` was
    absent. Recorded as ``available: False`` with the probes tried, so a future
    build that DOES expose it is distinguishable from one that reports a miss.
    """
    attempts: List[str] = []
    for attr in ("get_metrics", "get_stats", "stat_logger"):
        probe = getattr(engine, attr, None)
        if probe is None:
            attempts.append(f"{attr}: absent")
            continue
        try:
            value = probe() if callable(probe) else probe
            if value is not None:
                return {"available": True, "source": attr, "value": str(value)[:500],
                        "attempts": attempts}
            attempts.append(f"{attr}: returned None")
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{attr}: {type(exc).__name__}")
    return {
        "available": False,
        "source": None,
        "attempts": attempts,
        "note": (
            "No KV or prefix-cache signal. This is ABSENCE, not a zero hit "
            "rate — the engine did not report, which is different from "
            "reporting nothing was reused."
        ),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Serve N models concurrently and split offered load.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="explicit model ids; overrides --n-models")
    ap.add_argument("--n-models", type=int, default=None,
                    help="first N from the roster (ascending size)")
    ap.add_argument("--roster", default=DEFAULT_ROSTER)
    ap.add_argument("--quantization", default="none", choices=QUANTIZATION_CHOICES)
    ap.add_argument("--total-concurrency", type=int,
                    default=DEFAULT_TOTAL_CONCURRENCY,
                    help="offered load SPLIT across models — per-model "
                         "concurrency is this divided by N. Held fixed as N "
                         "rises; that is the independent variable.")
    ap.add_argument("--gpu-memory-utilization", type=float,
                    default=DEFAULT_GPU_MEM_UTIL,
                    help="WHOLE-CARD budget. Divided by N across engines.")
    ap.add_argument("--n-requests", type=int, default=DEFAULT_N_REQUESTS)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--fixture", default=DEFAULT_FIXTURE,
                    choices=sorted(fx_mod.FIXTURE_BUILDERS))
    ap.add_argument("--fixtures-dir", default=os.path.join(_BENCH_ROOT, "fixtures"))
    ap.add_argument("--context-tokens", type=int, default=2620)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--isolation", default="in-process",
                    choices=["in-process", "process"],
                    help="'process' (one engine per subprocess) is NOT "
                         "implemented — it is the documented fallback if "
                         "in-process engines prove unworkable.")
    ap.add_argument("--card-vram-gb", type=float, default=80.0)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out-dir", default=os.path.join(_BENCH_ROOT, "results"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the VRAM check; load nothing")
    return ap


def format_plan(plan: MultiModelPlan, *, fixture_sha: Optional[str] = None,
                card_vram_gb: float = 80.0,
                gpu_memory_utilization: float = DEFAULT_GPU_MEM_UTIL) -> str:
    L: List[str] = []
    A = L.append
    A("=" * 78)
    A(f"ARM {ARM} — {plan.n_models} model(s), total concurrency "
      f"{plan.total_concurrency}")
    A("=" * 78)
    A(f"  fixture            {plan.fixture_name}"
      + (f"  sha256 {fixture_sha[:16]}…" if fixture_sha else ""))
    A(f"  quantization       {plan.quantization}")
    A(f"  max_new_tokens     {plan.max_new_tokens}")
    A(f"  card budget        {card_vram_gb:.0f} GB x {gpu_memory_utilization} "
      f"= {card_vram_gb * gpu_memory_utilization:.1f} GB usable")
    A("")
    A(f"  {'model':<40}{'conc':>6}{'gpu_mem_util':>14}{'weights GB':>12}")
    A("  " + "-" * 72)
    for e in plan.engines:
        w = f"{e.weights_gb_estimate:.2f}" if e.weights_gb_estimate else "—"
        A(f"  {e.model_id:<40}{e.concurrency:>6}"
          f"{e.gpu_memory_utilization:>14.4f}{w:>12}")
    A("  " + "-" * 72)
    A(f"  {'TOTAL':<40}{plan.total_concurrency:>6}"
      f"{sum(e.gpu_memory_utilization for e in plan.engines):>14.4f}"
      f"{plan.vram['weights_gb']:>12.2f}")
    A("")
    A(f"  Per-engine gpu_memory_utilization is the whole-card budget divided by")
    A(f"  N. Passing the full {gpu_memory_utilization} to each engine would OOM "
      f"every engine after the first.")
    if plan.warnings:
        A("")
        for w in plan.warnings:
            A(f"  ⚠️  {w}")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.isolation == "process":
        print(
            "--isolation process is not implemented. It is the documented "
            "fallback for in-process engine failure; build it only if "
            "in-process OOMs at init or contention makes the numbers "
            "meaningless. See the module docstring.",
            file=sys.stderr)
        return 2

    roster = load_roster(args.roster)
    models = select_models(roster, n_models=args.n_models, models=args.models)

    plan = build_engine_configs(
        models,
        total_concurrency=args.total_concurrency,
        quantization=args.quantization,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        fixture_name=args.fixture,
        card_vram_gb=args.card_vram_gb,
    )

    fixture_sha = None
    try:
        fixture, _how = resolve_fixture(
            args.fixture, args.n_requests, args.context_tokens, args.seed,
            "Qwen/Qwen2.5-7B-Instruct", args.fixtures_dir)
        fixture_sha = fixture.sha256()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        plan.warnings.append(f"fixture could not be resolved: {exc}")

    print(format_plan(plan, fixture_sha=fixture_sha,
                      card_vram_gb=args.card_vram_gb,
                      gpu_memory_utilization=args.gpu_memory_utilization))

    blocking = [w for w in plan.warnings if w.startswith("WEIGHTS ALONE")]
    if args.dry_run:
        print("\n--dry-run: no engine constructed, no weights downloaded.")
        if blocking:
            print("\nThis configuration would fail to load. Fix it before "
                  "spending a session on it.", file=sys.stderr)
            return 1
        return 0

    if blocking:
        print("\nRefusing to start: weights alone exceed the memory budget.",
              file=sys.stderr)
        return 1

    print("\nEngine construction is not exercised in this environment "
          "(no GPU, no vllm). Run this on the target card.", file=sys.stderr)
    return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
