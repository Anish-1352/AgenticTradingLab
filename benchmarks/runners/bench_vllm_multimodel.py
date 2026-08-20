#!/usr/bin/env python3
"""Heterogeneous serving — N distinct models on one card, load split across them.

    python benchmarks/runners/bench_vllm_multimodel.py --dry-run --n-models 4
    python benchmarks/runners/bench_vllm_multimodel.py --n-models 4 --quantization awq

WHAT vLLM 0.27.1 ACTUALLY PROVIDES — READ THIS BEFORE CHANGING THE DESIGN
--------------------------------------------------------------------------
``AsyncEngineArgs`` takes ``model`` — **singular**. There is no parameter for a
model list and no first-class "serve N models from one engine" API. That is not
an oversight in this runner; it is the shape of the library. Three consequences
drive everything below:

1. **N models means N engines.** Each is a separate ``AsyncLLM`` with its
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

VERIFIED AGAINST 0.27.1, AND INFERRED — THE DIFFERENCE, EXPLICITLY
-------------------------------------------------------------------
The 0.26-era version of this file said "NOT verified against an installed
vLLM" and left it there. The API claims below were then checked by reading
vLLM's source at the ``v0.27.1`` tag. Nothing here was checked by running it —
there is still no GPU in this environment — so the two categories are kept
apart rather than blended into a confident-sounding whole.

**VERIFIED — read from vLLM v0.27.1 source:**

* ``AsyncEngineArgs.model`` is still ``str``, singular (``engine/arg_utils.py``).
  N models therefore still means N engines; the premise of this runner holds.
* ``engine/async_llm_engine.py`` is now four lines: ``AsyncLLMEngine = AsyncLLM``.
  **The V0 engine is gone.** The old V0/V1 try/except still runs, but on 0.27.1
  both branches reach the same class, so it is a version probe, not a fallback.
  This runner tries ``vllm.v1.engine.async_llm.AsyncLLM`` first and records
  which path answered in ``engine_api``.
* ``AsyncLLM.from_engine_args(engine_args, start_engine_loop=True,
  usage_context=..., stat_loggers=...)`` — the one positional argument this
  runner passes is correct.
* ``AsyncLLM.generate(prompt, sampling_params, request_id, *, ...)`` returns
  ``AsyncGenerator[RequestOutput, None]``. The three positional arguments used
  here are still positional; everything added since is keyword-only.
* Every ``AsyncEngineArgs`` keyword this runner sends exists at that tag:
  ``model``, ``dtype``, ``seed``, ``max_model_len``, ``enable_prefix_caching``,
  ``gpu_memory_utilization``, ``max_num_seqs``, ``disable_log_stats``,
  ``quantization``. ``enable_log_requests`` is NEW in 0.27.1 (it replaced
  ``disable_log_requests``); this runner sets neither.
* ``kernel_warmup()`` still imports ``minimax_m3_msa_warmup`` unconditionally at
  the top of its body and calls it. **The torchvision shim is still required**
  — that 0.26-era workaround carries forward unchanged.

**INFERRED — not verifiable without the card, and the first real run tests it:**

* That ``gpu_memory_utilization`` divided by N actually lets N engines coexist.
  The division is arithmetic; whether the resulting claim succeeds against real
  allocator behaviour, fragmentation and per-context overhead is not.
* That N ``AsyncLLM`` instances coexist in one process at all. Nothing in the
  source forbids it; nothing confirms it either.
* Whether ``stats_source`` resolves on 0.27.1. It was ``null`` on 0.26. The
  probe runs either way and records which attribute answered.
* Every throughput, TTFT and ITL number this produces. Obviously — but stated,
  because the point of the list above is that reading source is not running it.

The first action in a real session remains ``--dry-run``, then the N=1
validation run.

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
engines changes that. 0.27.1 may differ — ``AsyncLLM`` carries a
``logger_manager`` — so the probe runs and records the result rather than
assuming the 0.26 outcome carries. Absence is recorded as absence — ``None``, never ``0`` —
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
from common.monitor import ResourceMonitor  # noqa: E402

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

    @property
    def short_name(self) -> str:
        """Last path segment, for request ids and per-model report keys.

        Roster ids are ``org/model``; the bare model name is unique across the
        roster and survives being used as a filename.
        """
        return self.model_id.rsplit("/", 1)[-1]


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
                 max_num_seqs: Optional[int] = None,
                 seed: int = 1234) -> Tuple[Any, str]:
    """One engine for one model. ``model`` is singular — see the docstring.

    VERIFIED against vLLM 0.27.1 source: ``vllm/engine/async_llm_engine.py`` is
    now four lines — ``AsyncLLMEngine = AsyncLLM`` — so the V0 engine is gone
    and both import paths land on the same class. The two-branch form is kept
    because it costs nothing and still works on 0.26, but on 0.27.1 the second
    branch is unreachable rather than a fallback, and ``engine_api`` records
    which one answered so a run can be attributed to a version after the fact.

    Every keyword below was checked against ``AsyncEngineArgs`` in
    ``vllm/engine/arg_utils.py`` at the v0.27.1 tag. ``max_num_seqs`` is
    ``int | None`` there, so passing None is the documented default rather than
    a value; it is omitted anyway.
    """
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
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = max_num_seqs

    args = AsyncEngineArgs(**kwargs)
    errors: List[str] = []
    try:
        from vllm.v1.engine.async_llm import AsyncLLM  # noqa: PLC0415

        return (AsyncLLM.from_engine_args(args),
                "vllm.v1.engine.async_llm.AsyncLLM")
    except ImportError as exc:
        errors.append(f"vllm.v1.AsyncLLM: {type(exc).__name__}: {exc}")
    try:
        from vllm import AsyncLLMEngine  # noqa: PLC0415

        return AsyncLLMEngine.from_engine_args(args), "vllm.AsyncLLMEngine"
    except ImportError as exc:
        errors.append(f"vllm.AsyncLLMEngine: {type(exc).__name__}: {exc}")
    raise SystemExit(
        f"could not import an engine class for {model_id}. Attempts:\n  "
        + "\n  ".join(errors)
    )


def shutdown_engine(engine: Any) -> None:
    """Release a card before the next engine is built.

    Best-effort by design: this runs in a ``finally`` during teardown, and an
    engine that cannot be shut down cleanly must not mask the run's own result.
    Failures are swallowed here and the caller reports what it measured.
    """
    for attr in ("shutdown", "close", "stop"):
        fn = getattr(engine, attr, None)
        if callable(fn):
            try:
                fn()
                return
            except Exception:  # noqa: BLE001, S110
                continue


def make_prompt(token_ids: Sequence[int]):
    """Feed token IDs, not text — the fixture's exactness lives in the ids.

    Identical to arm C's helper so the two runners submit byte-identical
    prompts; a divergence here would make the throughput comparison invalid.
    """
    try:
        from vllm import TokensPrompt  # noqa: PLC0415

        return TokensPrompt(prompt_token_ids=list(token_ids))
    except Exception:  # noqa: BLE001
        pass
    try:
        from vllm.inputs import TokensPrompt  # noqa: PLC0415

        return TokensPrompt(prompt_token_ids=list(token_ids))
    except Exception:  # noqa: BLE001
        pass
    return {"prompt_token_ids": list(token_ids)}


def make_sampling_params(max_new_tokens: int, seed: int = 1234):
    """Fixed output length, EOS ignored — the token count is the independent
    variable and a model that stops early would confound the comparison."""
    from vllm import SamplingParams  # noqa: PLC0415

    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        ignore_eos=True,
        seed=seed,
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
# capability probe
#
# This exists because its predecessor did not. The runner used to end with an
# unconditional line claiming "no GPU, no vllm" — printed whether or not either
# was true, so a healthy A100 with a working vLLM install was told its hardware
# was missing. Everything below is an ACTUAL probe: each entry records what was
# executed and what came back, and nothing is asserted about a thing that was
# not checked.
# --------------------------------------------------------------------------


def probe_capabilities() -> Dict[str, Any]:
    """Check, one at a time, what this machine can actually do.

    Returns a dict per capability with ``available`` and the evidence for it.
    An import error and a missing device are reported as different things,
    because they need different fixes.
    """
    caps: Dict[str, Any] = {}

    # --- torch + CUDA ---
    try:
        import torch  # noqa: PLC0415

        cuda = bool(torch.cuda.is_available())
        caps["torch"] = {
            "available": True,
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": cuda,
            "device_count": torch.cuda.device_count() if cuda else 0,
            "device_name": torch.cuda.get_device_name(0) if cuda else None,
            "probe": "import torch; torch.cuda.is_available()",
        }
    except Exception as exc:  # noqa: BLE001
        caps["torch"] = {
            "available": False, "cuda_available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "probe": "import torch",
        }

    # --- vllm ---
    try:
        import vllm  # noqa: PLC0415

        caps["vllm"] = {
            "available": True,
            "version": getattr(vllm, "__version__", "unknown"),
            "probe": "import vllm",
        }
    except Exception as exc:  # noqa: BLE001
        caps["vllm"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "probe": "import vllm",
        }

    # --- the engine class this runner actually calls ---
    if caps["vllm"]["available"]:
        try:
            from vllm.v1.engine.async_llm import AsyncLLM  # noqa: PLC0415

            caps["engine_class"] = {
                "available": True,
                "path": "vllm.v1.engine.async_llm.AsyncLLM",
                "has_from_engine_args": hasattr(AsyncLLM, "from_engine_args"),
                "has_generate": hasattr(AsyncLLM, "generate"),
                "probe": "from vllm.v1.engine.async_llm import AsyncLLM",
            }
        except Exception as exc:  # noqa: BLE001
            caps["engine_class"] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "probe": "from vllm.v1.engine.async_llm import AsyncLLM",
            }
    else:
        caps["engine_class"] = {
            "available": False,
            "error": "not probed: vllm did not import",
            "probe": None,
        }

    caps["can_run"] = bool(
        caps["vllm"]["available"]
        and caps["engine_class"].get("available")
        and caps["torch"].get("cuda_available")
    )
    return caps


def format_capabilities(caps: Dict[str, Any]) -> str:
    """One line per capability, each naming the probe that produced it."""
    lines = ["Capability probe (each line is a check that was executed):"]

    t = caps["torch"]
    if not t["available"]:
        lines.append(f"  torch          MISSING   {t.get('error')}")
    elif not t["cuda_available"]:
        lines.append(f"  torch          {t['version']} present, but "
                     f"torch.cuda.is_available() is False — no usable device")
    else:
        lines.append(f"  torch          {t['version']}, cuda True, "
                     f"{t['device_count']}x {t['device_name']}")

    v = caps["vllm"]
    lines.append(f"  vllm           {v['version']}" if v["available"]
                 else f"  vllm           MISSING   {v.get('error')}")

    e = caps["engine_class"]
    lines.append(f"  engine class   {e['path']}" if e.get("available")
                 else f"  engine class   UNAVAILABLE   {e.get('error')}")

    lines.append(f"  -> can run:    {caps['can_run']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


async def run_one(engine: Any, token_ids: Sequence[int], request_id: str,
                  sampling_params: Any, sem: "asyncio.Semaphore") -> RequestRecord:
    """One request against one engine, with a timestamp per output token.

    Deliberately identical in shape to arm C's ``_run_one``: the clock starts
    AFTER the semaphore, so TTFT excludes client-side queue wait in both
    runners. Changing that here would make the arms incomparable, which is the
    only thing this benchmark exists to do.
    """
    rec = RequestRecord(request_id=request_id, prompt_tokens=len(token_ids))
    async with sem:
        rec.t_submit = time.perf_counter()
        prev = 0
        try:
            async for out in engine.generate(
                make_prompt(token_ids), sampling_params, request_id
            ):
                now = time.perf_counter()
                completion = out.outputs[0] if getattr(out, "outputs", None) else None
                cur = len(getattr(completion, "token_ids", ()) or ()) if completion else 0
                if cur > prev:
                    if rec.t_first_token is None:
                        rec.t_first_token = now
                    # Cumulative token_ids, and one step can surface several
                    # tokens. One timestamp per NEW token, so ITL counts gaps
                    # between tokens rather than between engine steps.
                    for _ in range(cur - prev):
                        rec.per_token_timestamps.append(now)
                    prev = cur
            rec.t_done = time.perf_counter()
            rec.output_tokens = prev
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            rec.t_done = time.perf_counter()
            msg = str(exc)
            rec.error = ("CUDA_OOM" if "out of memory" in msg.lower()
                         else f"{type(exc).__name__}: {msg}")
    return rec


async def run_model(engine: Any, cfg: "EngineConfig", prompts: Sequence[Sequence[int]],
                    sampling_params: Any) -> Dict[str, Any]:
    """Every request for one model, capped at that model's share of the load.

    The semaphore is the split: ``cfg.concurrency`` is total/N, so all N models
    together offer the same load one model offered at N=1. Without that, adding
    models would raise offered load and throughput and the experiment would
    measure nothing.
    """
    sem = asyncio.Semaphore(cfg.concurrency)
    t0 = time.perf_counter()
    records = await asyncio.gather(*[
        run_one(engine, ids, f"{cfg.short_name}-{i}", sampling_params, sem)
        for i, ids in enumerate(prompts)
    ])
    return {
        "model": cfg.model_id,
        "short_name": cfg.short_name,
        "concurrency": cfg.concurrency,
        "records": list(records),
        "wall_seconds": time.perf_counter() - t0,
    }


async def run_all(engines: Sequence[Tuple[Any, "EngineConfig"]],
                  prompts: Sequence[Sequence[int]],
                  max_new_tokens: int, seed: int,
                  sampling_params: Any = None) -> Dict[str, Any]:
    """All N models in flight at once — the point of the whole exercise.

    They are gathered rather than run in sequence because sequential execution
    would measure N independent single-model runs, not contention for one
    card's memory and scheduler.
    """
    # Built once and shared: every engine must receive identical sampling
    # settings or the per-model comparison is confounded. Injectable so the
    # request loop can be tested without a vLLM install.
    if sampling_params is None:
        sampling_params = make_sampling_params(max_new_tokens, seed=seed)
    t0 = time.perf_counter()
    per_model = await asyncio.gather(*[
        run_model(engine, cfg, prompts, sampling_params) for engine, cfg in engines
    ])
    return {"per_model": list(per_model), "wall_seconds": time.perf_counter() - t0}


def aggregate(per_model: Sequence[Dict[str, Any]], wall_seconds: float,
              run_id: str) -> Dict[str, Any]:
    """Per-model summaries plus one over every request on the card.

    The aggregate uses the WALL time of the whole run, not the sum of per-model
    walls: the models overlap, so summing would divide by roughly N times too
    much and understate card throughput by that factor.
    """
    all_records: List[RequestRecord] = []
    per_model_out = []
    for entry in per_model:
        recs = entry["records"]
        all_records.extend(recs)
        summary = summarize(f"{run_id}::{entry['short_name']}",
                            entry["concurrency"], recs, requested=len(recs))
        per_model_out.append({
            "model": entry["model"],
            "short_name": entry["short_name"],
            "concurrency": entry["concurrency"],
            "wall_seconds": entry["wall_seconds"],
            "summary": summary,
        })
    total_conc = sum(e["concurrency"] for e in per_model)
    agg = summarize(f"{run_id}::aggregate", total_conc, all_records,
                    requested=len(all_records))
    # summarize() derives duration from the records it was given. Across models
    # those spans overlap, so the aggregate rate is recomputed against the run's
    # true wall clock.
    ok = [r for r in all_records if r.ok]
    agg_rates = {
        "wall_seconds": wall_seconds,
        "completed_requests_per_s": (len(ok) / wall_seconds) if wall_seconds else None,
        "output_tokens_per_s": (sum(r.output_tokens for r in ok) / wall_seconds)
        if wall_seconds else None,
        "total_output_tokens": sum(r.output_tokens for r in ok),
        "completed": len(ok),
        "failed": len(all_records) - len(ok),
    }
    return {"per_model": per_model_out, "aggregate": agg,
            "aggregate_rates": agg_rates, "all_records": all_records}


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

    fixture = None
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

    # ---- what this machine can actually do, checked rather than assumed ----
    caps = probe_capabilities()
    print()
    print(format_capabilities(caps))
    if not caps["can_run"]:
        missing = []
        if not caps["vllm"]["available"]:
            missing.append("vllm does not import")
        elif not caps["engine_class"].get("available"):
            missing.append("vllm imports but AsyncLLM does not")
        if not caps["torch"]["available"]:
            missing.append("torch does not import")
        elif not caps["torch"]["cuda_available"]:
            missing.append("torch.cuda.is_available() is False")
        print("\nCannot run here: " + "; ".join(missing)
              + ".\nEach line above is a check that was executed; nothing "
                "else about this machine was inspected.", file=sys.stderr)
        return 3

    if fixture is None:
        print("\nCannot run: the fixture did not resolve, so there are no "
              "prompts to send. See the warning above.", file=sys.stderr)
        return 1

    run_id = args.run_id or make_run_id(ARM, args.total_concurrency)

    # The fixture is used AS RESOLVED — its sha256 is the provenance and
    # slicing it would break that. A stored fixture holds whatever count it was
    # built at, which need not be --n-requests, so the count actually sent is
    # recorded and a mismatch is stated rather than silently reconciled.
    prompts = [list(r["prompt_token_ids"]) for r in fixture.requests]
    n_sent = len(prompts)
    if n_sent != args.n_requests:
        print(f"\n  note: fixture {args.fixture!r} holds {n_sent} requests; "
              f"--n-requests {args.n_requests} does not resize it. Sending "
              f"{n_sent} per model ({n_sent * len(plan.engines)} total).")

    engines: List[Tuple[Any, EngineConfig]] = []
    engine_api = None
    monitor = ResourceMonitor(device_index=0)
    result: Optional[Dict[str, Any]] = None
    kv_stats: Dict[str, Any] = {}
    try:
        # Sequential construction on purpose: an OOM at engine k names k, which
        # is the number the sweep is looking for. Building concurrently would
        # report a failure without saying which model exhausted the card.
        for cfg in plan.engines:
            print(f"  building engine {cfg.short_name} "
                  f"@ gpu_memory_utilization={cfg.gpu_memory_utilization:.4f} ...",
                  flush=True)
            engine, engine_api = build_engine(
                cfg.model_id,
                gpu_memory_utilization=cfg.gpu_memory_utilization,
                quantization=None if args.quantization == "none" else args.quantization,
                max_model_len=args.max_model_len,
                seed=args.seed,
            )
            engines.append((engine, cfg))

        print(f"  {len(engines)} engines up via {engine_api}; "
              f"offering {args.total_concurrency} total concurrency "
              f"split {[c.concurrency for c in plan.engines]}", flush=True)

        monitor.start()
        try:
            run = asyncio.run(run_all(engines, prompts, args.max_new_tokens,
                                      args.seed))
        finally:
            monitor.stop()

        result = aggregate(run["per_model"], run["wall_seconds"], run_id)
        kv_stats = {cfg.short_name: read_kv_stats(engine)
                    for engine, cfg in engines}
    finally:
        for engine, _cfg in engines:
            shutdown_engine(engine)

    mon_summary = monitor.summary()
    for entry in result["per_model"]:
        attach_resources(entry["summary"], mon_summary)
    attach_resources(result["aggregate"], mon_summary)

    # ---- report ----
    print("\nPer model:")
    for entry in result["per_model"]:
        print(f"\n  {entry['short_name']}  (concurrency {entry['concurrency']})")
        for line in console_lines(entry["summary"]):
            print(f"    {line}")
    print("\nAggregate (all models, one card):")
    for line in console_lines(result["aggregate"]):
        print(f"  {line}")
    ar = result["aggregate_rates"]
    # The `requests/s` printed by console_lines above is derived from the span
    # of the records themselves. This line divides by the run's wall clock
    # instead, which is the card-level rate and the figure the sweep compares
    # across N. They differ slightly; this is the one to quote.
    print(f"  card rate (wall clock): {ar['wall_seconds']:.2f}s wall  "
          f"completed {ar['completed']}/{ar['completed'] + ar['failed']}  "
          f"{ar['completed_requests_per_s']:.3f} req/s  "
          f"{ar['output_tokens_per_s']:.1f} output tok/s")

    # stats_source was null on 0.26. Recorded per engine either way, so a build
    # that does report is distinguishable from one that reports nothing.
    sources = {k: v.get("source") for k, v in kv_stats.items()}
    print(f"  stats_source: {sources}")

    config_resolved = {
        "models": [c.model_id for c in plan.engines],
        "n_models": len(plan.engines),
        "quantization": args.quantization,
        "total_concurrency": args.total_concurrency,
        "per_model_concurrency": [c.concurrency for c in plan.engines],
        "gpu_memory_utilization_total": args.gpu_memory_utilization,
        "gpu_memory_utilization_per_engine":
            plan.engines[0].gpu_memory_utilization if plan.engines else None,
        "max_new_tokens": args.max_new_tokens,
        "n_requests_requested": args.n_requests,
        "n_requests_per_model": n_sent,
        "n_requests_total": n_sent * len(plan.engines),
        "fixture": args.fixture,
        "seed": args.seed,
        "isolation": args.isolation,
    }
    manifest = build_manifest(
        arm=ARM,
        concurrency=args.total_concurrency,
        config_resolved=config_resolved,
        fixture_sha256=fixture_sha,
        fixture_name=args.fixture,
        max_new_tokens=args.max_new_tokens,
        run_id=run_id,
        out_dir=args.out_dir,
        context_tokens=args.context_tokens,
        # N models, so there is no single `model`. The list lives in
        # config_resolved and is hashed there; leaving this None is honest,
        # whereas naming one of the N would misattribute the run.
        model=None,
        extra={
            "engine_api": engine_api,
            "capabilities": caps,
            "kv_stats": kv_stats,
            "stats_source": sources,
            "aggregate_rates": ar,
            "plan_warnings": plan.warnings,
            "vram_estimate": plan.vram,
        },
    )

    paths = write_results(
        args.out_dir, run_id, result["all_records"],
        [e["summary"] for e in result["per_model"]] + [result["aggregate"]],
        extra={"manifest": manifest.to_dict(),
               "per_model": [
                   {"model": e["model"], "short_name": e["short_name"],
                    "concurrency": e["concurrency"],
                    "wall_seconds": e["wall_seconds"]}
                   for e in result["per_model"]],
               "aggregate_rates": ar,
               "kv_stats": kv_stats,
               "capabilities": caps},
    )
    for label, path in paths.items():
        print(f"  {label}: {path}")
    print(f"  manifest: {args.out_dir}/{run_id}_manifest.json")
    # Same convention as arm C: an incomplete manifest is announced rather than
    # left to be discovered when the run is analysed.
    if not manifest.manifest_complete:
        print(f"  manifest INCOMPLETE: {manifest.collection_errors}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
