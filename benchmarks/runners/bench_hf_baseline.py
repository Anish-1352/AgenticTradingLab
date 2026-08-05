#!/usr/bin/env python3
"""Arm B — naive self-hosted serving: transformers + ThreadPoolExecutor.

This is the "just run it yourself" configuration, done properly. One model
instance, no batching layer, one ``TextIteratorStreamer`` and one ``generate``
thread per in-flight request. At concurrency N that is **2N threads** against a
single model, and that is the point of the arm: it is what a naive
implementation does, and its failure mode is the thing arm C is supposed to
fix. It is not a strawman as long as it is measured honestly.

THE v1 ERROR THIS FILE EXISTS TO NOT REPEAT
-------------------------------------------
v1 ran ``torch.profiler`` in a **separate single-threaded loop after** the
concurrent section had finished, on a different (much shorter) prompt. The
resulting trace described a sequential single-request generation, and was then
used to explain the behaviour of a 105-task / 15-thread concurrent run.

Here the profiler context **wraps the concurrent execution itself**. Its
``schedule`` is stepped from the completion loop *inside* that section, so the
recorded window is a slice of the real workload under real contention.

Because it is a slice, the manifest records exactly which slice:
``profiled_window_fraction`` is the fraction of the concurrent section's wall
time that the active window covered, alongside the step indices and the number
of completions inside it. A bounded window is unavoidable — a full trace at 105
concurrency is ~1.5 GB — but an unlabelled bounded window is how v1 happened.

``--profile`` defaults **off**. Layer 1 (timing + NVML) and Layer 3
(torch.profiler) are separate runs because CUPTI instrumentation perturbs the
timings Layer 1 exists to measure.

MEMORY: A SLOPE, NOT AN ASSERTION
---------------------------------
v1 reported one global ``torch.cuda.max_memory_allocated()`` and the write-up
then attributed it per-thread. That per-thread number was never measured.

This runner calls ``reset_peak_memory_stats()`` **per concurrency level** and
records, for each level, the allocator peak and the NVML process VRAM. Marginal
per-request KV cost is then the **slope** of VRAM against concurrency across
levels — a quantity that is actually measured, with a fit quality reported
alongside it so a bad fit is visible rather than silently rounded into a
headline number.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import os
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# Import as a package when possible; fall back to a path insert so the file
# also runs as `python benchmarks/runners/bench_hf_baseline.py` from the repo
# root, which is how COLAB.md invokes it.
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
from common.monitor import ResourceMonitor  # noqa: E402

ARM = "B"


# --------------------------------------------------------------------------
# fixture handling
# --------------------------------------------------------------------------


# Lives in common/fixtures.py so both arms resolve fixtures identically.
# Re-exported here because that is where it originally lived and where the
# Colab-validated invocations import it from.
resolve_fixture = fx_mod.resolve_fixture


# --------------------------------------------------------------------------
# single request
# --------------------------------------------------------------------------


def _run_one(
    model,
    tokenizer_obj,
    token_ids: List[int],
    request_id: str,
    max_new_tokens: int,
    force_full_length: bool,
    torch_mod,
) -> RequestRecord:
    """One request: a generate thread plus this thread consuming its streamer."""
    from transformers import TextIteratorStreamer  # noqa: PLC0415

    rec = RequestRecord(request_id=request_id, prompt_tokens=len(token_ids))
    rec.t_submit = time.perf_counter()

    try:
        input_ids = torch_mod.tensor([token_ids], dtype=torch_mod.long, device=model.device)
        attention_mask = torch_mod.ones_like(input_ids)

        streamer = TextIteratorStreamer(
            tokenizer_obj, skip_prompt=True, skip_special_tokens=True
        )
        kwargs: Dict[str, Any] = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        if force_full_length:
            # `ignore_eos` in the config. transformers has no such flag; forcing
            # min == max is the equivalent, and it is what keeps output length
            # from becoming a function of the prompt (which would make
            # output_tok_per_s incomparable across arms).
            kwargs["min_new_tokens"] = max_new_tokens

        result: Dict[str, Any] = {}

        def _generate() -> None:
            try:
                result["out"] = model.generate(**kwargs)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                result["error"] = f"{type(exc).__name__}: {exc}"

        gen_thread = threading.Thread(target=_generate, name=f"gen-{request_id}")
        gen_thread.start()

        # Every chunk gets a timestamp, including whitespace-only ones. v1
        # filtered on `new_text.strip()`, which dropped chunks from the token
        # count while leaving the time they took in the numerator, inflating its
        # per-token figure. Filtering here is exactly the bug.
        for _chunk in streamer:
            now = time.perf_counter()
            if rec.t_first_token is None:
                rec.t_first_token = now
            rec.per_token_timestamps.append(now)

        gen_thread.join()
        rec.t_done = time.perf_counter()

        if "error" in result:
            rec.error = result["error"]
            return rec

        out = result.get("out")
        if out is not None:
            # Exact, from ids — not inferred from the number of stream chunks,
            # which can differ when the streamer buffers partial UTF-8.
            rec.output_tokens = int(out.shape[1]) - len(token_ids)
        else:
            rec.output_tokens = len(rec.per_token_timestamps)

    except Exception as exc:  # noqa: BLE001
        rec.t_done = time.perf_counter()
        msg = str(exc)
        rec.error = (
            "CUDA_OOM" if "out of memory" in msg.lower() else f"{type(exc).__name__}: {msg}"
        )
        try:
            torch_mod.cuda.empty_cache()
        except Exception:
            pass

    return rec


# --------------------------------------------------------------------------
# one concurrency level
# --------------------------------------------------------------------------


def run_level(
    model,
    tokenizer_obj,
    prompts: List[Tuple[str, List[int]]],
    concurrency: int,
    max_new_tokens: int,
    force_full_length: bool,
    torch_mod,
    profile: bool = False,
    profiler_schedule: Optional[Dict[str, int]] = None,
    traces_dir: Optional[str] = None,
    run_id: str = "run",
    gzip_traces: bool = True,
) -> Tuple[List[RequestRecord], Dict[str, Any]]:
    """Execute one concurrency level. Returns (records, level_info)."""
    torch_mod.cuda.reset_peak_memory_stats()

    records: List[RequestRecord] = []
    step_timestamps: List[float] = []
    prof_ctx = None
    trace_path: Optional[str] = None

    if profile:
        sched = profiler_schedule or {"wait": 5, "warmup": 2, "active": 4, "repeat": 1}
        os.makedirs(traces_dir or ".", exist_ok=True)
        trace_path = os.path.join(traces_dir or ".", f"{run_id}_c{concurrency}_torch.json")

        prof_ctx = torch_mod.profiler.profile(
            activities=[
                torch_mod.profiler.ProfilerActivity.CPU,
                torch_mod.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch_mod.profiler.schedule(
                wait=sched["wait"],
                warmup=sched["warmup"],
                active=sched["active"],
                repeat=sched.get("repeat", 1),
            ),
            record_shapes=True,
            profile_memory=True,
            # with_stack adds substantial overhead and balloons the trace; the
            # cuda_runtime breakdown this study needs does not require it.
            with_stack=False,
        )

    t_section_start = time.perf_counter()

    # ---- THE CONCURRENT SECTION. The profiler wraps THIS. ----
    def _execute() -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    _run_one, model, tokenizer_obj, ids, rid,
                    max_new_tokens, force_full_length, torch_mod,
                )
                for rid, ids in prompts
            ]
            for fut in concurrent.futures.as_completed(futures):
                records.append(fut.result())
                if prof_ctx is not None:
                    # Steps are driven by completions *inside* the concurrent
                    # section, so the active window is a slice of the real
                    # contended workload rather than a separate quiet run.
                    prof_ctx.step()
                    step_timestamps.append(time.perf_counter())

    if prof_ctx is not None:
        with prof_ctx:
            _execute()
    else:
        _execute()

    t_section_end = time.perf_counter()
    section_wall = t_section_end - t_section_start

    # ---- memory, per level ----
    peak_alloc_mb = torch_mod.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved_mb = torch_mod.cuda.max_memory_reserved() / (1024 ** 2)

    level_info: Dict[str, Any] = {
        "concurrency": concurrency,
        "section_wall_s": section_wall,
        "torch_peak_allocated_mb": peak_alloc_mb,
        "torch_peak_reserved_mb": peak_reserved_mb,
        "memory_note": (
            "Global allocator peak for this level. NOT a per-thread figure and "
            "must not be divided by concurrency. Marginal per-request cost is "
            "the slope across levels — see vram_slope in the summary."
        ),
    }

    # ---- what the profiler actually captured ----
    if prof_ctx is not None:
        sched = profiler_schedule or {"wait": 5, "warmup": 2, "active": 4, "repeat": 1}
        s = sched["wait"] + sched["warmup"]
        a = sched["active"]
        n_steps = len(step_timestamps)

        if n_steps >= s + a:
            t_a0 = step_timestamps[s - 1] if s > 0 else t_section_start
            t_a1 = step_timestamps[s + a - 1]
            captured = True
        elif n_steps > s:
            t_a0 = step_timestamps[s - 1] if s > 0 else t_section_start
            t_a1 = step_timestamps[-1]
            captured = True
        else:
            t_a0 = t_a1 = t_section_start
            captured = False

        window_s = max(t_a1 - t_a0, 0.0)
        level_info["profiling"] = {
            "captured": captured,
            "schedule": sched,
            "total_steps": n_steps,
            "active_step_range": [s, s + a],
            "active_window_s": window_s,
            "section_wall_s": section_wall,
            "profiled_window_fraction": (window_s / section_wall) if section_wall > 0 else None,
            "completions_in_window": min(a, max(n_steps - s, 0)),
            "completions_total": n_steps,
            "note": (
                "The active window is a bounded slice OF THE CONCURRENT SECTION "
                "(not a separate sequential run). Steps are request completions."
            ),
            "warning": (
                None if captured else
                f"Only {n_steps} completions occurred but the schedule needs "
                f"{s + a} to close its active window. No trace was recorded. "
                f"Lower wait/warmup or raise n-requests."
            ),
        }

        if captured and trace_path:
            try:
                prof_ctx.export_chrome_trace(trace_path)
                if gzip_traces:
                    gz = trace_path + ".gz"
                    with open(trace_path, "rb") as src, gzip.open(gz, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    os.remove(trace_path)
                    trace_path = gz
                level_info["profiling"]["trace_path"] = trace_path
                level_info["profiling"]["trace_bytes"] = os.path.getsize(trace_path)
            except Exception as exc:  # noqa: BLE001
                level_info["profiling"]["export_error"] = f"{type(exc).__name__}: {exc}"

    return records, level_info


# --------------------------------------------------------------------------
# VRAM slope
# --------------------------------------------------------------------------


def vram_slope(levels: List[Dict[str, Any]], key: str = "torch_peak_allocated_mb") -> Dict[str, Any]:
    """Least-squares fit of VRAM against concurrency.

    The slope is the marginal per-request cost (KV cache plus per-request
    activations); the intercept is the fixed cost (weights plus context).
    ``r_squared`` is reported so a poor fit is visible: if VRAM does not grow
    linearly — because the allocator is reusing cached blocks, or because a
    level partially OOMed — the slope is not a per-request cost and should not
    be quoted as one.
    """
    pts = [
        (float(l["concurrency"]), float(l[key]))
        for l in levels
        if l.get(key) is not None and l.get("concurrency")
    ]
    if len(pts) < 2:
        return {"available": False, "reason": "need at least 2 levels"}

    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return {"available": False, "reason": "all concurrency values identical"}

    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n

    mean_y = sy / n
    ss_tot = sum((p[1] - mean_y) ** 2 for p in pts)
    ss_res = sum((p[1] - (slope * p[0] + intercept)) ** 2 for p in pts)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None

    return {
        "available": True,
        "metric": key,
        "slope_mb_per_request": slope,
        "intercept_mb": intercept,
        "r_squared": r2,
        "points": pts,
        "interpretation": (
            "slope = marginal VRAM per concurrent request (KV cache + per-request "
            "activations); intercept = fixed cost (weights + runtime). This is "
            "measured across levels, NOT a per-thread division of a global peak."
        ),
        "caveat": (
            None if (r2 is not None and r2 >= 0.9) else
            "Poor linear fit (r^2 < 0.9). Do not quote the slope as a per-request "
            "cost; inspect the per-level points for allocator caching or a "
            "partial OOM."
        ),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Arm B: naive HuggingFace serving benchmark.")
    ap.add_argument("--config", default=None, help="path to workload.yaml")
    ap.add_argument("--concurrency", type=int, nargs="*", default=None,
                    help="override the config's concurrency sweep")
    ap.add_argument("--n-requests", type=int, default=None,
                    help="override requests_per_level")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--fixture", default=None, choices=sorted(fx_mod.FIXTURE_BUILDERS),
                    help="override the config's fixture")
    ap.add_argument("--out-dir", default=None,
                    help="results dir; point at mounted Drive on Colab")
    ap.add_argument("--traces-dir", default=None)
    ap.add_argument("--fixtures-dir", default=os.path.join(_BENCH_ROOT, "fixtures"))
    ap.add_argument("--profile", action="store_true",
                    help="enable torch.profiler (Layer 3). OFF by default so "
                         "Layer 1 timings stay unperturbed.")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the workload, print token counts, exit before "
                         "loading any model")
    ap.add_argument("--stub-tokenizer", action="store_true",
                    help="offline whitespace tokenizer; --dry-run only")
    ap.add_argument("--rebuild-fixture", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="do not skip levels already present in the summary file")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, {
        "workload.fixture": args.fixture,
        "workload.requests_per_level": args.n_requests,
        "workload.concurrency": args.concurrency,
        "generation.max_new_tokens": args.max_new_tokens,
    })

    model_id = cfg["model"]["id"]
    gen = cfg["generation"]
    wl = cfg["workload"]
    max_new_tokens = int(gen["max_new_tokens"])
    force_full_length = bool(gen.get("ignore_eos", True))
    n_requests = int(wl["requests_per_level"])
    levels = list(wl["concurrency"])
    context_tokens = int(wl["context_tokens"])
    seed = int(cfg.get("seed", 0))
    fixture_name = wl["fixture"]

    out_dir = args.out_dir or os.path.join(_BENCH_ROOT, "results")
    traces_dir = args.traces_dir or os.path.join(_BENCH_ROOT, "traces")

    # ---- fixture ----
    fixture, how = resolve_fixture(
        fixture_name, n_requests, context_tokens, seed, model_id,
        args.fixtures_dir, use_stub=args.stub_tokenizer, rebuild=args.rebuild_fixture,
    )
    meta = fixture.to_meta()

    print("=" * 72)
    print(f"ARM {ARM} — naive HuggingFace serving")
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
        counts_desc = (
            f"{meta['token_count_min']}..{meta['token_count_max']} "
            f"** NOT EQUAL — context length is a controlled variable **"
        )
    print(f"  tokens/request   {counts_desc}")
    print(f"  common prefix    {meta['common_prefix_tokens']} tokens "
          f"({meta['common_prefix_fraction'] * 100:.1f}% of context)")
    print(f"max_new_tokens     {max_new_tokens}  (forced full length: {force_full_length})")
    print(f"concurrency        {levels}")
    print(f"profiling layer    {3 if args.profile else 1}")
    print(f"out-dir            {out_dir}")

    if args.dry_run:
        print("\n--dry-run: workload built, no model loaded. Exiting.")
        total = len(levels) * n_requests
        print(f"  would issue      {total} requests "
              f"({len(levels)} levels x {n_requests})")
        print(f"  prompt tokens    {meta['token_count_min']} per request")
        print(f"  output tokens    {max_new_tokens} per request")
        print(f"  total prompt tok {total * meta['token_count_min']:,}")
        print(f"  total output tok {total * max_new_tokens:,}")
        if meta["tokenizer_is_stub"]:
            print("\n  !! STUB TOKENIZER: these counts are NOT Qwen token counts.")
            print("     Rebuild with the real tokenizer before any real run.")
        return 0

    if meta["tokenizer_is_stub"]:
        raise SystemExit(
            "refusing to run a real benchmark against a stub-tokenizer fixture: "
            "its token counts are not the model's, so context length — a "
            "controlled variable — would be wrong. Rebuild with "
            "`python -m common.fixtures --model <id>`."
        )

    # ---- heavy imports, only now ----
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible; this arm requires a GPU")

    run_id = args.run_id or make_run_id(ARM, levels[0] if levels else 0)

    # ---- resume ----
    completed: Dict[int, Any] = {}
    summary_path = os.path.join(out_dir, f"{run_id}_summary.json")
    if os.path.exists(summary_path) and not args.no_resume:
        try:
            with open(summary_path) as fh:
                prev = json.load(fh)
            for lv in prev.get("levels", []):
                completed[int(lv["concurrency"])] = lv
            if completed:
                print(f"\n[resume] {summary_path} has levels "
                      f"{sorted(completed)} — skipping them")
        except Exception as exc:  # noqa: BLE001
            print(f"[resume] could not read {summary_path}: {exc}")

    print(f"\n[load] {model_id} ...")
    t_load = time.perf_counter()
    dtype = getattr(torch, cfg["model"].get("dtype", "float16"))
    tokenizer_obj = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=cfg["model"].get("device_map", "auto"),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", False)),
    )
    model.eval()
    print(f"[load] done in {time.perf_counter() - t_load:.1f}s")

    prompts_all = [(r["request_id"], r["prompt_token_ids"]) for r in fixture.requests]

    # ---- warmup, discarded ----
    n_warm = int(cfg.get("warmup", {}).get("requests", 3))
    if n_warm > 0:
        print(f"[warmup] {n_warm} discarded requests ...")
        for rid, ids in prompts_all[:n_warm]:
            _run_one(model, tokenizer_obj, ids, f"warmup-{rid}",
                     min(max_new_tokens, 32), force_full_length, torch)
        torch.cuda.synchronize()

    all_records: List[RequestRecord] = []
    summaries: List[Summary] = []
    level_infos: List[Dict[str, Any]] = []
    for lv in completed.values():
        level_infos.append({"concurrency": lv["concurrency"], "resumed": True})

    prof_cfg = cfg.get("profiling", {}).get("torch_profiler", {})
    gzip_traces = bool(cfg.get("output", {}).get("gzip_traces", True))
    last_window_fraction: Optional[float] = None

    for concurrency in levels:
        if concurrency in completed:
            summaries.append(Summary(**{
                k: v for k, v in completed[concurrency].items()
                if k in Summary.__dataclass_fields__
            }))
            continue

        prompts = prompts_all[:n_requests]
        print(f"\n[level] concurrency={concurrency}  requests={len(prompts)}")

        monitor = ResourceMonitor(
            out_csv=os.path.join(out_dir, f"{run_id}_c{concurrency}_monitor.csv"),
            hz=20.0,
            warmup_s=float(cfg.get("warmup", {}).get("seconds", 5)),
        ).start()

        try:
            records, info = run_level(
                model, tokenizer_obj, prompts, concurrency, max_new_tokens,
                force_full_length, torch,
                profile=args.profile,
                profiler_schedule=prof_cfg,
                traces_dir=traces_dir,
                run_id=run_id,
                gzip_traces=gzip_traces,
            )
        finally:
            monitor.stop()

        info["monitor"] = monitor.summary()
        if "profiling" in info and info["profiling"].get("profiled_window_fraction"):
            last_window_fraction = info["profiling"]["profiled_window_fraction"]

        summary = summarize(run_id, concurrency, records, requested=len(prompts))
        attach_resources(summary, info["monitor"])
        summary.notes["torch_peak_allocated_mb"] = info["torch_peak_allocated_mb"]
        summary.notes["torch_peak_reserved_mb"] = info["torch_peak_reserved_mb"]
        all_records.extend(records)
        summaries.append(summary)
        level_infos.append(info)

        for line in console_lines(summary):
            print(line)
        print(f"  torch alloc   peak {info['torch_peak_allocated_mb']:.0f} MB"
              f"   reserved {info['torch_peak_reserved_mb']:.0f} MB")
        if "profiling" in info:
            p = info["profiling"]
            if p.get("warning"):
                print(f"  [profiler] {p['warning']}")
            else:
                print(f"  [profiler] window {p['active_window_s']:.2f}s = "
                      f"{(p['profiled_window_fraction'] or 0) * 100:.1f}% of section, "
                      f"steps {p['active_step_range']}")

        # Written after EVERY level: a Colab preemption mid-sweep keeps
        # everything finished so far.
        write_results(
            out_dir, run_id, all_records, summaries,
            extra={
                "arm": ARM,
                "levels_detail": level_infos,
                "vram_slope": vram_slope(level_infos),
                "vram_slope_nvml": vram_slope(
                    [
                        {"concurrency": i["concurrency"],
                         "nvml_peak_mb": (i.get("monitor") or {}).get("vram_peak_mb")}
                        for i in level_infos if "monitor" in i
                    ],
                    key="nvml_peak_mb",
                ),
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
        profiling_layer=3 if args.profile else 1,
        profiled_window_fraction=last_window_fraction,
        run_id=run_id,
        out_dir=out_dir,
        context_tokens=context_tokens,
        model=model_id,
        extra={
            "config_sha256_recomputed": config_sha256(cfg),
            "concurrency_sweep": levels,
            "common_prefix_tokens": meta["common_prefix_tokens"],
            "requests_per_level": n_requests,
        },
    )
    print(f"\n[manifest] {out_dir}/{run_id}_manifest.json")
    if not manifest.manifest_complete:
        print(f"[manifest] INCOMPLETE: {manifest.collection_errors}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
