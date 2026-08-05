#!/usr/bin/env python3
"""Arm C — optimized self-hosted serving: vLLM AsyncLLMEngine + asyncio.

Same fixtures, same ``workload.yaml``, same ``common/`` metrics, monitor and
manifest modules as arm B. The serving stack is the only thing that differs,
which is the entire point of the comparison.

CONCURRENCY MEANS OFFERED LOAD, IN BOTH ARMS
--------------------------------------------
The *mechanism* differs and the *offered load* does not.

  Arm B  concurrency=15 -> ThreadPoolExecutor(max_workers=15): 15 OS threads,
         each independently calling ``model.generate``. Nothing batches them;
         they contend for one GIL and one model.
  Arm C  concurrency=15 -> ONE engine, ``asyncio.Semaphore(15)`` around
         submission: at most 15 requests outstanding at any instant, which
         vLLM's scheduler then batches continuously as it sees fit.

Both therefore present **15 simultaneously-outstanding requests** to the
serving stack. That is the controlled variable. If arm C instead submitted all
105 at once and let the scheduler queue them, it would be answering a different
question than arm B at the same nominal "concurrency", and every B->C delta in
the study would be uninterpretable.

``t_submit`` is recorded **after** the semaphore is acquired, mirroring arm B —
where ``_run_one`` starts only once a pool worker is free. So TTFT excludes
client-side queue wait in both arms and measures the server's response to an
admitted request.

MEMORY: NVML ALONE IS MISLEADING FOR THIS ARM
---------------------------------------------
vLLM **pre-allocates its KV cache pool at startup** to
``gpu_memory_utilization`` of the device. NVML process VRAM therefore reports
what the pool was *configured* to reserve, not what the workload *demanded*,
and it barely moves with concurrency or with prefix caching on/off. Reporting
only NVML would make the prefix-cache ablation look like it did nothing.

So three quantities are recorded, and they answer different questions:

  * ``nvml_*``            — process VRAM. Directly comparable to arm B's number,
                            but for arm C it is bounded by config, not demand.
  * ``kv_cache_usage``    — vLLM's own KV utilization. This is the demand signal
                            and the one the prefix-cache ablation turns on.
  * ``num_gpu_blocks`` x ``block_size`` — lets absolute KV bytes be derived
                            rather than inferred from a percentage.

``gpu_memory_utilization`` is carried in the manifest because it *determines*
the NVML figure; a reader comparing NVML across arms without it would be
comparing a config value to a measurement.

THE vLLM STATS API IS PROBED, NOT ASSUMED
-----------------------------------------
Metric plumbing moved between vLLM releases (V0 stat loggers -> V1
``get_metrics``), and this file was written without the ability to execute
against 0.26.0. ``VllmIntrospector`` therefore tries a documented list of
access paths in order, records **which one worked** in ``stats_source``, and
records every failed attempt in ``stats_probe_attempts``. A run whose stats
came back null is visible as such rather than silently reporting zeros — check
``stats_source`` before quoting a KV or hit-rate number.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _BENCH_ROOT not in sys.path:
    sys.path.insert(0, _BENCH_ROOT)

from common import fixtures as fx_mod  # noqa: E402
from common.config import apply_overrides, config_sha256, load_config  # noqa: E402
from common.manifest import build_manifest, make_run_id  # noqa: E402
from common.metrics import (  # noqa: E402
    RequestRecord,
    Summary,
    attach_resources,
    console_lines,
    summarize,
    write_results,
)
from common.fixtures import resolve_fixture  # noqa: E402
from common.monitor import ResourceMonitor  # noqa: E402

ARM = "C"


# --------------------------------------------------------------------------
# vLLM introspection
# --------------------------------------------------------------------------


class VllmIntrospector:
    """Best-effort extraction of KV cache and prefix-cache stats from an engine.

    Every getter returns ``None`` rather than raising or guessing, and every
    attempt is logged. ``stats_source`` names the path that worked so a number
    in the results can be traced to how it was obtained.
    """

    # (label, dotted path from the engine object) — tried in order.
    CACHE_CONFIG_PATHS = (
        "vllm_config.cache_config",
        "engine.cache_config",
        "cache_config",
        "engine.vllm_config.cache_config",
        "llm_engine.cache_config",
    )
    # Metric-name fragments, matched case-insensitively against whatever the
    # installed version exposes.
    KV_USAGE_KEYS = ("gpu_cache_usage_perc", "gpu_cache_usage", "kv_cache_usage")
    HIT_RATE_KEYS = ("gpu_prefix_cache_hit_rate", "prefix_cache_hit_rate")
    HIT_COUNT_KEYS = ("prefix_cache_hits", "gpu_prefix_cache_hits")
    QUERY_COUNT_KEYS = ("prefix_cache_queries", "gpu_prefix_cache_queries")

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.attempts: List[str] = []
        self.stats_source: Optional[str] = None

    # ---- helpers ----

    def _dig(self, dotted: str) -> Any:
        node = self.engine
        for part in dotted.split("."):
            node = getattr(node, part, None)
            if node is None:
                return None
        return node

    def _metric_pairs(self) -> List[Tuple[str, float]]:
        """Flatten whatever the installed version calls 'current metrics'."""
        pairs: List[Tuple[str, float]] = []

        # V1: engine.get_metrics() -> iterable of prometheus-like objects.
        getter = getattr(self.engine, "get_metrics", None)
        if callable(getter):
            try:
                for m in getter() or []:
                    name = getattr(m, "name", None)
                    value = getattr(m, "value", None)
                    if value is None:
                        samples = getattr(m, "samples", None) or []
                        for s in samples:
                            sname = getattr(s, "name", name)
                            sval = getattr(s, "value", None)
                            if sname is not None and sval is not None:
                                pairs.append((str(sname), float(sval)))
                        continue
                    if name is not None:
                        pairs.append((str(name), float(value)))
                if pairs:
                    self.stats_source = self.stats_source or "engine.get_metrics()"
                    return pairs
                self.attempts.append("engine.get_metrics(): returned no usable pairs")
            except Exception as exc:  # noqa: BLE001
                self.attempts.append(f"engine.get_metrics(): {type(exc).__name__}: {exc}")
        else:
            self.attempts.append("engine.get_metrics(): absent")

        # V0: stat loggers holding the last Stats object.
        for path in ("stat_loggers", "engine.stat_loggers"):
            loggers = self._dig(path)
            if not loggers:
                continue
            try:
                items = loggers.values() if hasattr(loggers, "values") else loggers
                for lg in items:
                    last = getattr(lg, "last_local_stats", None) or getattr(lg, "stats", None)
                    if last is None:
                        continue
                    for attr in dir(last):
                        if attr.startswith("_"):
                            continue
                        val = getattr(last, attr, None)
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            pairs.append((attr, float(val)))
                if pairs:
                    self.stats_source = self.stats_source or path
                    return pairs
                self.attempts.append(f"{path}: no numeric fields found")
            except Exception as exc:  # noqa: BLE001
                self.attempts.append(f"{path}: {type(exc).__name__}: {exc}")

        return pairs

    @staticmethod
    def _find(pairs: List[Tuple[str, float]], keys: Tuple[str, ...]) -> Optional[float]:
        for key in keys:
            for name, value in pairs:
                if key in name.lower():
                    return value
        return None

    # ---- public ----

    def cache_config(self) -> Dict[str, Any]:
        """``num_gpu_blocks`` and ``block_size`` so absolute KV bytes derive."""
        for path in self.CACHE_CONFIG_PATHS:
            cc = self._dig(path)
            if cc is None:
                continue
            num_blocks = getattr(cc, "num_gpu_blocks", None)
            if num_blocks is None:
                num_blocks = getattr(cc, "num_gpu_blocks_override", None)
            block_size = getattr(cc, "block_size", None)
            if num_blocks is not None or block_size is not None:
                return {
                    "source": path,
                    "num_gpu_blocks": num_blocks,
                    "block_size": block_size,
                    "gpu_memory_utilization": getattr(cc, "gpu_memory_utilization", None),
                    "enable_prefix_caching": getattr(cc, "enable_prefix_caching", None),
                    # KV tokens the pool can hold at all; the ceiling every
                    # usage percentage is a fraction of.
                    "kv_capacity_tokens": (
                        num_blocks * block_size
                        if isinstance(num_blocks, int) and isinstance(block_size, int)
                        else None
                    ),
                }
            self.attempts.append(f"{path}: present but no num_gpu_blocks/block_size")
        self.attempts.append("cache_config: no path matched")
        return {"source": None, "num_gpu_blocks": None, "block_size": None,
                "gpu_memory_utilization": None, "enable_prefix_caching": None,
                "kv_capacity_tokens": None}

    def sample(self) -> Dict[str, Any]:
        """Point-in-time KV usage and prefix-cache hit rate."""
        pairs = self._metric_pairs()
        hits = self._find(pairs, self.HIT_COUNT_KEYS)
        queries = self._find(pairs, self.QUERY_COUNT_KEYS)
        hit_rate = self._find(pairs, self.HIT_RATE_KEYS)
        if hit_rate is None and hits is not None and queries:
            hit_rate = hits / queries

        return {
            "kv_cache_usage_perc": self._find(pairs, self.KV_USAGE_KEYS),
            "prefix_cache_hit_rate": hit_rate,
            "prefix_cache_hits": hits,
            "prefix_cache_queries": queries,
            "stats_source": self.stats_source,
            "metric_names_seen": sorted({n for n, _ in pairs})[:60],
        }

    def reset_prefix_cache(self) -> Dict[str, Any]:
        """Flush the prefix cache between levels.

        Without this, a level's hit rate is contaminated by the level before it:
        the sweep reuses the same fixture, so level 2 would start against a cache
        already warmed by level 1 and report a hit rate the workload did not
        earn. Recorded rather than assumed — if no reset path exists in this
        version, the results say so and the level ordering has to be treated as
        a confound.
        """
        for name in ("reset_prefix_cache", "reset_prefix_caching"):
            fn = getattr(self.engine, name, None) or self._dig(f"engine.{name}")
            if callable(fn):
                try:
                    result = fn()
                    if asyncio.iscoroutine(result):
                        return {"reset": "coroutine", "path": name, "awaitable": result}
                    return {"reset": True, "path": name}
                except Exception as exc:  # noqa: BLE001
                    return {"reset": False, "path": name,
                            "error": f"{type(exc).__name__}: {exc}"}
        return {"reset": False, "path": None,
                "error": "no reset_prefix_cache path found on this vLLM version"}

    def report(self) -> Dict[str, Any]:
        return {"stats_source": self.stats_source, "stats_probe_attempts": self.attempts}


# --------------------------------------------------------------------------
# engine construction
# --------------------------------------------------------------------------


def build_engine(
    model_id: str,
    dtype: str,
    enable_prefix_caching: bool,
    gpu_memory_utilization: float,
    max_num_seqs: Optional[int],
    max_model_len: Optional[int],
    seed: int,
):
    """Construct an AsyncLLMEngine, tolerating the V0/V1 import split."""
    from vllm import AsyncEngineArgs  # noqa: PLC0415

    kwargs: Dict[str, Any] = dict(
        model=model_id,
        dtype=dtype,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=gpu_memory_utilization,
        seed=seed,
        disable_log_stats=False,  # stats are the point; keep the loggers alive
    )
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = max_num_seqs
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
        "could not construct a vLLM async engine. Attempts:\n  "
        + "\n  ".join(errors)
    )


def make_prompt(token_ids: List[int]):
    """Feed token IDs, not text — the fixture's exactness lives in the ids."""
    try:
        from vllm import TokensPrompt  # noqa: PLC0415

        return TokensPrompt(prompt_token_ids=list(token_ids))
    except Exception:
        pass
    try:
        from vllm.inputs import TokensPrompt  # noqa: PLC0415

        return TokensPrompt(prompt_token_ids=list(token_ids))
    except Exception:
        pass
    # Every version accepts the plain mapping form.
    return {"prompt_token_ids": list(token_ids)}


def make_sampling_params(max_new_tokens: int, temperature: float, ignore_eos: bool):
    from vllm import SamplingParams  # noqa: PLC0415

    return SamplingParams(
        max_tokens=max_new_tokens,
        min_tokens=max_new_tokens if ignore_eos else 0,
        temperature=temperature,
        # ignore_eos keeps output length exactly max_new_tokens, matching arm
        # B's min_new_tokens==max_new_tokens. Without it, output length becomes
        # a function of the prompt and output_tok_per_s stops being comparable
        # between arms.
        ignore_eos=ignore_eos,
    )


# --------------------------------------------------------------------------
# one request
# --------------------------------------------------------------------------


async def _run_one(
    engine,
    token_ids: List[int],
    request_id: str,
    sampling_params,
    sem: "asyncio.Semaphore",
) -> RequestRecord:
    rec = RequestRecord(request_id=request_id, prompt_tokens=len(token_ids))

    async with sem:
        # AFTER the semaphore, mirroring arm B where _run_one begins only once a
        # pool worker frees up. TTFT excludes client-side queue wait in both.
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
                    # RequestOutput carries CUMULATIVE token_ids and a step can
                    # surface more than one new token. Emit one timestamp per
                    # new token so ITL counts gaps between tokens, matching arm
                    # B's per-chunk stream; collapsing a multi-token step into a
                    # single timestamp would understate the token count and
                    # inflate ITL.
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
            rec.error = (
                "CUDA_OOM" if "out of memory" in msg.lower()
                else f"{type(exc).__name__}: {msg}"
            )
    return rec


# --------------------------------------------------------------------------
# one level
# --------------------------------------------------------------------------


async def run_level(
    engine,
    introspector: VllmIntrospector,
    prompts: List[Tuple[str, List[int]]],
    concurrency: int,
    sampling_params,
    kv_sample_hz: float = 5.0,
) -> Tuple[List[RequestRecord], Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)

    # vLLM's KV usage is only meaningful while requests are in flight; sampling
    # it after the level finishes reads an idle pool. This polls during it and
    # keeps the peak.
    kv_samples: List[Dict[str, Any]] = []
    stop = asyncio.Event()

    async def _poll() -> None:
        interval = 1.0 / kv_sample_hz
        while not stop.is_set():
            try:
                s = introspector.sample()
                s["t"] = time.perf_counter()
                kv_samples.append(s)
            except Exception:  # noqa: BLE001
                pass
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    poller = asyncio.create_task(_poll())
    t0 = time.perf_counter()
    try:
        records = await asyncio.gather(
            *[_run_one(engine, ids, rid, sampling_params, sem) for rid, ids in prompts]
        )
    finally:
        stop.set()
        await poller

    section_wall = time.perf_counter() - t0

    usages = [s["kv_cache_usage_perc"] for s in kv_samples
              if s.get("kv_cache_usage_perc") is not None]
    hit_rates = [s["prefix_cache_hit_rate"] for s in kv_samples
                 if s.get("prefix_cache_hit_rate") is not None]
    final = introspector.sample()

    info: Dict[str, Any] = {
        "concurrency": concurrency,
        "section_wall_s": section_wall,
        "kv_cache": {
            "usage_perc_peak": max(usages) if usages else None,
            "usage_perc_mean": (sum(usages) / len(usages)) if usages else None,
            "usage_sample_count": len(usages),
            "prefix_cache_hit_rate_final": final.get("prefix_cache_hit_rate"),
            "prefix_cache_hit_rate_max": max(hit_rates) if hit_rates else None,
            "prefix_cache_hits": final.get("prefix_cache_hits"),
            "prefix_cache_queries": final.get("prefix_cache_queries"),
            "stats_source": final.get("stats_source"),
            "metric_names_seen": final.get("metric_names_seen"),
        },
        "memory_note": (
            "NVML process VRAM for this arm is bounded by gpu_memory_utilization "
            "(the KV pool is pre-allocated at startup), so it reflects CONFIG, "
            "not demand. kv_cache.usage_perc_* is the demand signal. Compare "
            "NVML across arms only with gpu_memory_utilization in hand."
        ),
    }
    if not usages:
        info["kv_cache"]["warning"] = (
            "No KV usage samples obtained — the stats path did not resolve on "
            "this vLLM build. See stats_probe_attempts in the manifest; do not "
            "read the absence as zero usage."
        )
    return list(records), info


# --------------------------------------------------------------------------
# CLI — mirrors arm B for every shared flag
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Arm C: vLLM optimized serving benchmark.")
    # --- shared with arm B, identical names and semantics ---
    ap.add_argument("--config", default=None)
    ap.add_argument("--concurrency", type=int, nargs="*", default=None)
    ap.add_argument("--n-requests", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--fixture", default=None, choices=sorted(fx_mod.FIXTURE_BUILDERS))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--traces-dir", default=None)
    ap.add_argument("--fixtures-dir", default=os.path.join(_BENCH_ROOT, "fixtures"))
    ap.add_argument("--profile", action="store_true",
                    help="Layer 3 marker. vLLM runs its own CUDA graphs and "
                         "workers; torch.profiler around the engine is not "
                         "equivalent to arm B's in-process capture. Use "
                         "profiling/run_nsys.sh (Layer 2) for this arm.")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stub-tokenizer", action="store_true")
    ap.add_argument("--rebuild-fixture", action="store_true")
    ap.add_argument("--no-resume", action="store_true")

    # --- arm C only ---
    cache = ap.add_mutually_exclusive_group(required=True)
    cache.add_argument("--enable-prefix-caching", dest="prefix_caching",
                       action="store_true",
                       help="REQUIRED (with its counterpart): prefix caching is "
                            "never defaulted, because it is the variable the "
                            "prefix_cache ablation isolates and an implicit "
                            "value would silently decide the result.")
    cache.add_argument("--no-prefix-caching", dest="prefix_caching",
                       action="store_false")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="scheduler cap on concurrently running sequences. "
                         "1 approximates sequential scheduling; used by the "
                         "continuous_batching ablation.")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                    help="fraction of device memory vLLM pre-allocates. "
                         "Determines the NVML figure; recorded in the manifest.")
    ap.add_argument("--max-model-len", type=int, default=None)
    return ap


async def _amain(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, {
        "workload.fixture": args.fixture,
        "workload.requests_per_level": args.n_requests,
        "workload.concurrency": args.concurrency,
        "generation.max_new_tokens": args.max_new_tokens,
        # Arm C knobs belong in the hashed config: they change the result and
        # must therefore change config_sha256.
        "vllm.enable_prefix_caching": args.prefix_caching,
        "vllm.max_num_seqs": args.max_num_seqs,
        "vllm.gpu_memory_utilization": args.gpu_memory_utilization,
    })

    model_id = cfg["model"]["id"]
    gen = cfg["generation"]
    wl = cfg["workload"]
    max_new_tokens = int(gen["max_new_tokens"])
    ignore_eos = bool(gen.get("ignore_eos", True))
    temperature = float(gen.get("temperature", 0.0))
    n_requests = int(wl["requests_per_level"])
    levels = list(wl["concurrency"])
    context_tokens = int(wl["context_tokens"])
    seed = int(cfg.get("seed", 0))
    fixture_name = wl["fixture"]

    out_dir = args.out_dir or os.path.join(_BENCH_ROOT, "results")

    fixture, how = resolve_fixture(
        fixture_name, n_requests, context_tokens, seed, model_id,
        args.fixtures_dir, use_stub=args.stub_tokenizer, rebuild=args.rebuild_fixture,
    )
    meta = fixture.to_meta()

    print("=" * 72)
    print(f"ARM {ARM} — vLLM optimized serving")
    print("=" * 72)
    print(f"model              {model_id} ({cfg['model']['dtype']})")
    print(f"fixture            {fixture_name} ({how})")
    print(f"  sha256           {meta['sha256']}")
    print(f"  tokenizer        {meta['tokenizer_name']}"
          f"{'  ** STUB **' if meta['tokenizer_is_stub'] else ''}")
    print(f"  requests         {meta['n_requests']}")
    if meta["token_counts_all_equal"]:
        counts_desc = f"{meta['token_count_min']} (all equal)"
    else:
        counts_desc = (f"{meta['token_count_min']}..{meta['token_count_max']} "
                       f"** NOT EQUAL — context length is a controlled variable **")
    print(f"  tokens/request   {counts_desc}")
    print(f"  common prefix    {meta['common_prefix_tokens']} tokens "
          f"({meta['common_prefix_fraction'] * 100:.1f}% of context)")
    print(f"max_new_tokens     {max_new_tokens}  (ignore_eos: {ignore_eos})")
    print(f"concurrency        {levels}   (offered load, asyncio.Semaphore)")
    print(f"prefix caching     {'ON' if args.prefix_caching else 'OFF'}")
    print(f"max_num_seqs       {args.max_num_seqs if args.max_num_seqs else 'engine default'}")
    print(f"gpu_mem_util       {args.gpu_memory_utilization}")
    print(f"out-dir            {out_dir}")

    if args.dry_run:
        print("\n--dry-run: workload built, no engine constructed. Exiting.")
        total = len(levels) * n_requests
        print(f"  would issue      {total} requests "
              f"({len(levels)} levels x {n_requests})")
        print(f"  prompt tokens    {meta['token_count_min']} per request")
        print(f"  output tokens    {max_new_tokens} per request")
        print(f"  total prompt tok {total * meta['token_count_min']:,}")
        print(f"  total output tok {total * max_new_tokens:,}")
        if meta["tokenizer_is_stub"]:
            print("\n  !! STUB TOKENIZER: these counts are NOT Qwen token counts.")
        return 0

    if meta["tokenizer_is_stub"]:
        raise SystemExit(
            "refusing to run a real benchmark against a stub-tokenizer fixture: "
            "its token counts are not the model's, so context length — a "
            "controlled variable — would be wrong."
        )

    run_id = args.run_id or make_run_id(ARM, levels[0] if levels else 0)

    completed: Dict[int, Any] = {}
    summary_path = os.path.join(out_dir, f"{run_id}_summary.json")
    if os.path.exists(summary_path) and not args.no_resume:
        try:
            with open(summary_path) as fh:
                prev = json.load(fh)
            for lv in prev.get("levels", []):
                completed[int(lv["concurrency"])] = lv
            if completed:
                print(f"\n[resume] skipping levels {sorted(completed)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[resume] could not read {summary_path}: {exc}")

    print(f"\n[engine] constructing vLLM engine ...")
    t_load = time.perf_counter()
    engine, engine_kind = build_engine(
        model_id=model_id,
        dtype=cfg["model"].get("dtype", "float16"),
        enable_prefix_caching=args.prefix_caching,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=seed,
    )
    print(f"[engine] {engine_kind} ready in {time.perf_counter() - t_load:.1f}s")

    introspector = VllmIntrospector(engine)
    cache_cfg = introspector.cache_config()
    print(f"[engine] num_gpu_blocks={cache_cfg['num_gpu_blocks']} "
          f"block_size={cache_cfg['block_size']} "
          f"kv_capacity_tokens={cache_cfg['kv_capacity_tokens']} "
          f"(via {cache_cfg['source']})")

    sampling_params = make_sampling_params(max_new_tokens, temperature, ignore_eos)
    prompts_all = [(r["request_id"], r["prompt_token_ids"]) for r in fixture.requests]

    n_warm = int(cfg.get("warmup", {}).get("requests", 3))
    if n_warm > 0:
        print(f"[warmup] {n_warm} discarded requests ...")
        warm_sem = asyncio.Semaphore(n_warm)
        warm_params = make_sampling_params(min(max_new_tokens, 32), temperature, ignore_eos)
        await asyncio.gather(*[
            _run_one(engine, ids, f"warmup-{rid}", warm_params, warm_sem)
            for rid, ids in prompts_all[:n_warm]
        ])

    all_records: List[RequestRecord] = []
    summaries: List[Summary] = []
    level_infos: List[Dict[str, Any]] = []
    for lv in completed.values():
        level_infos.append({"concurrency": lv["concurrency"], "resumed": True})

    for concurrency in levels:
        if concurrency in completed:
            summaries.append(Summary(**{
                k: v for k, v in completed[concurrency].items()
                if k in Summary.__dataclass_fields__
            }))
            continue

        # Flush between levels: the sweep reuses one fixture, so without this a
        # later level inherits a cache warmed by an earlier one and reports a
        # hit rate it did not earn.
        reset = introspector.reset_prefix_cache()
        if isinstance(reset.get("awaitable"), object) and reset.get("reset") == "coroutine":
            try:
                await reset.pop("awaitable")
                reset["reset"] = True
            except Exception as exc:  # noqa: BLE001
                reset["reset"] = False
                reset["error"] = f"{type(exc).__name__}: {exc}"
        reset.pop("awaitable", None)

        prompts = prompts_all[:n_requests]
        print(f"\n[level] concurrency={concurrency}  requests={len(prompts)}")
        if not reset.get("reset"):
            print(f"  [warn] prefix cache NOT reset: {reset.get('error')}")

        monitor = ResourceMonitor(
            out_csv=os.path.join(out_dir, f"{run_id}_c{concurrency}_monitor.csv"),
            hz=20.0,
            warmup_s=float(cfg.get("warmup", {}).get("seconds", 5)),
        ).start()
        try:
            records, info = await run_level(
                engine, introspector, prompts, concurrency, sampling_params
            )
        finally:
            monitor.stop()

        info["monitor"] = monitor.summary()
        info["prefix_cache_reset"] = reset
        info["cache_config"] = cache_cfg

        summary = summarize(run_id, concurrency, records, requested=len(prompts))
        attach_resources(summary, info["monitor"])
        summary.notes["kv_cache"] = info["kv_cache"]
        summary.notes["gpu_memory_utilization"] = args.gpu_memory_utilization
        summary.notes["enable_prefix_caching"] = args.prefix_caching
        summary.notes["max_num_seqs"] = args.max_num_seqs

        all_records.extend(records)
        summaries.append(summary)
        level_infos.append(info)

        for line in console_lines(summary):
            print(line)
        kv = info["kv_cache"]
        print(f"  KV cache      peak {kv['usage_perc_peak']}"
              f"   mean {kv['usage_perc_mean']}"
              f"   (source: {kv['stats_source']})")
        print(f"  prefix cache  hit rate {kv['prefix_cache_hit_rate_final']}"
              f"   hits {kv['prefix_cache_hits']}/{kv['prefix_cache_queries']}")
        if kv.get("warning"):
            print(f"  [warn] {kv['warning']}")

        write_results(
            out_dir, run_id, all_records, summaries,
            extra={
                "arm": ARM,
                "engine_kind": engine_kind,
                "levels_detail": level_infos,
                "vllm": {
                    "enable_prefix_caching": args.prefix_caching,
                    "max_num_seqs": args.max_num_seqs,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "cache_config": cache_cfg,
                    **introspector.report(),
                },
            },
        )
        print(f"  [write] {out_dir}/{run_id}_summary.json")

    manifest = build_manifest(
        arm=ARM,
        concurrency=levels[0] if levels else 0,
        config_resolved=cfg,
        fixture_sha256=meta["sha256"],
        fixture_name=fixture_name,
        max_new_tokens=max_new_tokens,
        profiling_layer=1,
        run_id=run_id,
        out_dir=out_dir,
        context_tokens=context_tokens,
        model=model_id,
        extra={
            "config_sha256_recomputed": config_sha256(cfg),
            "concurrency_sweep": levels,
            "common_prefix_tokens": meta["common_prefix_tokens"],
            "requests_per_level": n_requests,
            "engine_kind": engine_kind,
            # gpu_memory_utilization determines the NVML VRAM figure for this
            # arm; an NVML comparison against arm B without it is meaningless.
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enable_prefix_caching": args.prefix_caching,
            "max_num_seqs": args.max_num_seqs,
            "vllm_cache_config": cache_cfg,
            **introspector.report(),
        },
    )
    print(f"\n[manifest] {out_dir}/{run_id}_manifest.json")
    if not manifest.manifest_complete:
        print(f"[manifest] INCOMPLETE: {manifest.collection_errors}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_amain(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
