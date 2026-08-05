#!/usr/bin/env python3
"""Ablation: how much of arm B's host-side cost is actually the GIL.

WHAT THIS REPLACES
------------------
The v1 write-up attributed >90% of host time to the GIL, inferred from
queue-wait durations in a torch.profiler trace. That trace came from
``advanced_gpu_stress_test.py``, which runs **one request at a time in a
sequential loop**. There was no thread contention in it to measure — a single
request's streamer queue wait is the GPU producing tokens, not threads fighting
for the interpreter lock. The number had no support.

This is the controlled experiment that can support a number.

THE DESIGN
----------
Identical work, three execution mechanisms, matched offered concurrency:

    (a) ThreadPoolExecutor   N workers, ONE shared GIL   <- arm B's mechanism
    (b) ProcessPoolExecutor  N workers, one GIL EACH
    (c) sequential           no concurrency at all       <- contention-free floor

The attributable GIL cost is the (a)-(b) wall-clock gap as a fraction of (a).
(c) anchors the scale: it shows what the work costs with no concurrency
machinery of any kind, so a small (a)-(b) gap can be read against the total
rather than in isolation.

WHY A SYNTHETIC MODE IS NECESSARY, NOT A SHORTCUT
-------------------------------------------------
(b) requires one model replica **per process**. At ~15 GB for Qwen2.5-7B in
fp16, an 80 GB card fits about four — so a real-model (b) cannot reach the
concurrency where arm B's contention actually appears. Testing GIL contention
only at concurrency 2-3 would answer a question nobody asked.

GIL contention scales with **control flow**, not model size. The synthetic mode
therefore reproduces arm B's exact threading structure — a producer thread per
request pushing tokens into a ``queue.Queue`` that the worker consumes, which is
precisely what ``TextIteratorStreamer`` is — with the model replaced by a
calibrated cost model. That preserves what the GIL sees and drops what it does
not care about.

THE PART A NAIVE MOCK GETS WRONG
--------------------------------
``time.sleep()`` **releases the GIL**. A mock token producer built only from
sleeps would show near-zero thread contention no matter how many threads ran,
and would "prove" the GIL costs nothing — an artifact of the mock, exactly the
kind of error this ablation exists to correct.

A real decode step is two parts:

    GPU compute        releases the GIL (the CUDA call blocks outside it)
    Python framework   HOLDS the GIL (sampling, stopping criteria, streamer
    overhead           callback, incremental detokenization)

Only the second part contends. So each synthetic token costs
``python_fraction`` of its time in a **GIL-holding busy loop** and the remainder
in a **GIL-releasing sleep**, summing to the measured ~37 ms/token from arm B at
C=1 (26.7 output tok/s).

``python_fraction`` is not known a priori — assuming it is what produced the v1
error. So the ablation **sweeps it** and reports GIL cost as a function of it.
The honest claim has the form "if per-token Python work is F of the step, the
attributable GIL cost at concurrency N is X%", and ``--real`` anchors which F is
plausible by measuring the true (a)-(b) gap at concurrency 2-3.

Report both modes. Synthetic alone is a model; real alone cannot reach the
concurrency of interest. Neither is sufficient by itself.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _ablation_util import (  # noqa: E402
    BENCH_ROOT,
    fmt,
    render_table,
    write_comparison,
)

# Arm B, concurrency 1: 26.7 output tok/s -> 37.45 ms per token.
DEFAULT_MS_PER_TOKEN = 37.45
DEFAULT_TOKENS = 64
DEFAULT_PYTHON_FRACTIONS = (0.05, 0.10, 0.20, 0.40)

_STOP = object()


# --------------------------------------------------------------------------
# calibrated busy loop (GIL-HOLDING work)
# --------------------------------------------------------------------------


def _spin(iterations: int) -> int:
    """Pure-Python work. Holds the GIL for its whole duration.

    Deliberately plain arithmetic on ints: no C-extension call that might
    release the lock, no allocation pattern that could hit a GIL-dropping
    boundary. This is the part of a decode step that contends.
    """
    acc = 0
    for i in range(iterations):
        acc = (acc + i * 7) % 1000003
    return acc


def calibrate_spin(target_s: float = 0.05) -> float:
    """Iterations per second for ``_spin``, measured on this host.

    Re-measured per run rather than hardcoded: a constant tuned on one machine
    would silently change the GIL-held fraction on another, which is the one
    variable this ablation is trying to control.
    """
    n = 20_000
    while True:
        t0 = time.perf_counter()
        _spin(n)
        dt = time.perf_counter() - t0
        if dt >= target_s * 0.5:
            return n / dt
        n *= 4
        if n > 200_000_000:  # pragma: no cover - defensive
            return n / max(dt, 1e-9)


# --------------------------------------------------------------------------
# synthetic request — mirrors arm B's producer-thread + queue structure
# --------------------------------------------------------------------------


def _producer(q: "queue.Queue", n_tokens: int, spin_iters: int, sleep_s: float) -> None:
    """Stands in for ``model.generate`` feeding a TextIteratorStreamer.

    Per token: GIL-holding work, then GIL-releasing wait, then hand the token to
    the consumer through a Queue — the same handoff TextIteratorStreamer makes.
    """
    for i in range(n_tokens):
        if spin_iters > 0:
            _spin(spin_iters)
        if sleep_s > 0:
            time.sleep(sleep_s)
        q.put(f"tok{i} ")
    q.put(_STOP)


def synthetic_request(payload: Tuple[str, int, int, float, float]) -> Dict[str, Any]:
    """One request. Module-level and picklable so ProcessPoolExecutor can run it.

    Structure matches arm B exactly: this callable is the pool worker, and it
    spawns one producer thread and consumes its queue. So both (a) and (b) have
    two threads per request; the only difference is whether requests share an
    interpreter.
    """
    request_id, n_tokens, spin_iters, sleep_s, _pyfrac = payload

    t_submit = time.perf_counter()
    q: "queue.Queue" = queue.Queue()
    th = threading.Thread(
        target=_producer, args=(q, n_tokens, spin_iters, sleep_s),
        name=f"producer-{request_id}",
    )
    th.start()

    stamps: List[float] = []
    while True:
        item = q.get()
        if item is _STOP:
            break
        stamps.append(time.perf_counter())
    th.join()
    t_done = time.perf_counter()

    return {
        "request_id": request_id,
        "t_submit": t_submit,
        "t_done": t_done,
        "e2e": t_done - t_submit,
        "ttft": (stamps[0] - t_submit) if stamps else None,
        "tokens": len(stamps),
        "itl_mean": (
            (stamps[-1] - stamps[0]) / (len(stamps) - 1) if len(stamps) > 1 else None
        ),
        "pid": os.getpid(),
    }


# --------------------------------------------------------------------------
# real request — actual model, low concurrency only
# --------------------------------------------------------------------------

_REAL_MODEL: Dict[str, Any] = {}


def _real_init(model_id: str, dtype_name: str) -> None:
    """Per-worker model load. One replica per process in the (b) condition."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, dtype_name), device_map="auto"
    )
    model.eval()
    _REAL_MODEL["tok"] = tok
    _REAL_MODEL["model"] = model
    _REAL_MODEL["torch"] = torch


def real_request(payload: Tuple[str, List[int], int, str, str]) -> Dict[str, Any]:
    """One real generation. Module-level and picklable, same as the synthetic."""
    request_id, token_ids, max_new_tokens, model_id, dtype_name = payload
    if "model" not in _REAL_MODEL:
        _real_init(model_id, dtype_name)

    import threading as _th  # noqa: PLC0415

    from transformers import TextIteratorStreamer  # noqa: PLC0415

    torch = _REAL_MODEL["torch"]
    model = _REAL_MODEL["model"]
    tok = _REAL_MODEL["tok"]

    t_submit = time.perf_counter()
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
    )
    th = _th.Thread(target=lambda: model.generate(**kwargs))
    th.start()

    stamps: List[float] = []
    for _chunk in streamer:
        stamps.append(time.perf_counter())
    th.join()
    t_done = time.perf_counter()

    return {
        "request_id": request_id,
        "t_submit": t_submit,
        "t_done": t_done,
        "e2e": t_done - t_submit,
        "ttft": (stamps[0] - t_submit) if stamps else None,
        "tokens": len(stamps),
        "itl_mean": (
            (stamps[-1] - stamps[0]) / (len(stamps) - 1) if len(stamps) > 1 else None
        ),
        "pid": os.getpid(),
    }


# --------------------------------------------------------------------------
# execution mechanisms
# --------------------------------------------------------------------------


def _rollup(label: str, results: List[Dict[str, Any]], wall: float,
            concurrency: int) -> Dict[str, Any]:
    e2es = [r["e2e"] for r in results if r.get("e2e") is not None]
    ttfts = [r["ttft"] for r in results if r.get("ttft") is not None]
    itls = [r["itl_mean"] for r in results if r.get("itl_mean") is not None]
    tokens = sum(r.get("tokens", 0) for r in results)
    return {
        "mechanism": label,
        "concurrency": concurrency,
        "requests": len(results),
        "wall_time_s": wall,
        "tokens_total": tokens,
        "tokens_per_s": tokens / wall if wall > 0 else None,
        "e2e_mean": sum(e2es) / len(e2es) if e2es else None,
        "e2e_max": max(e2es) if e2es else None,
        "ttft_mean": sum(ttfts) / len(ttfts) if ttfts else None,
        "itl_mean": sum(itls) / len(itls) if itls else None,
        "distinct_pids": len({r.get("pid") for r in results}),
    }


def run_threads(fn, payloads, concurrency: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(fn, payloads))
    return _rollup("threadpool", results, time.perf_counter() - t0, concurrency)


def run_processes(fn, payloads, concurrency: int,
                  initializer=None, initargs=()) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=concurrency, initializer=initializer, initargs=initargs
    ) as pool:
        results = list(pool.map(fn, payloads))
    return _rollup("processpool", results, time.perf_counter() - t0, concurrency)


def run_sequential(fn, payloads) -> Dict[str, Any]:
    t0 = time.perf_counter()
    results = [fn(p) for p in payloads]
    return _rollup("sequential", results, time.perf_counter() - t0, 1)


# --------------------------------------------------------------------------
# py-spy
# --------------------------------------------------------------------------


def start_pyspy(out_path: str, duration: int, pid: Optional[int] = None) -> Dict[str, Any]:
    """Attach ``py-spy record --gil`` to this process. Degrades gracefully.

    Independent evidence: py-spy samples which thread actually *holds* the GIL,
    rather than inferring contention from wall-clock deltas. Where ptrace is
    restricted (common in containers; Colab may or may not permit it) this is
    simply unavailable and the ablation still stands on the (a)-(b) gap.
    """
    exe = shutil.which("py-spy")
    if not exe:
        return {"available": False,
                "reason": "py-spy not on PATH (pip install py-spy)"}

    cmd = [exe, "record", "--gil", "--pid", str(pid or os.getpid()),
           "--duration", str(duration), "--output", out_path]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    time.sleep(1.0)  # let it attach or fail fast
    if proc.poll() is not None:
        err = (proc.stderr.read() or b"").decode(errors="replace")
        reason = "ptrace denied" if "permission" in err.lower() or "ptrace" in err.lower() \
            else f"py-spy exited {proc.returncode}"
        return {"available": False, "reason": reason, "stderr": err[-500:]}

    return {"available": True, "proc": proc, "output": out_path,
            "command": " ".join(cmd)}


def stop_pyspy(handle: Dict[str, Any]) -> Dict[str, Any]:
    proc = handle.pop("proc", None)
    if proc is None:
        return handle
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()
    handle["exists"] = os.path.exists(handle.get("output", ""))
    return handle


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------


def run_synthetic(concurrency: int, n_requests: int, n_tokens: int,
                  ms_per_token: float, python_fraction: float,
                  spin_per_s: float) -> Dict[str, Any]:
    step_s = ms_per_token / 1000.0
    python_s = step_s * python_fraction
    sleep_s = max(step_s - python_s, 0.0)
    spin_iters = int(python_s * spin_per_s)

    payloads = [
        (f"syn-{i:04d}", n_tokens, spin_iters, sleep_s, python_fraction)
        for i in range(n_requests)
    ]

    threads = run_threads(synthetic_request, payloads, concurrency)
    procs = run_processes(synthetic_request, payloads, concurrency)
    seq = run_sequential(synthetic_request, payloads)

    return {
        "python_fraction": python_fraction,
        "ms_per_token": ms_per_token,
        "spin_iters_per_token": spin_iters,
        "sleep_s_per_token": sleep_s,
        "threadpool": threads,
        "processpool": procs,
        "sequential": seq,
        "gil_cost_pct": _gil_cost(threads, procs),
    }


def _gil_cost(threads: Dict[str, Any], procs: Dict[str, Any]) -> Optional[float]:
    """(a)-(b) wall-clock gap as a fraction of (a).

    Positive means the thread pool took longer than the process pool — the
    portion of arm B's wall time attributable to sharing one interpreter lock.
    Negative means process overhead (spawn, pickling) dominated, which at low
    concurrency or short workloads is a real and reportable outcome, not a bug.
    """
    a = threads.get("wall_time_s")
    b = procs.get("wall_time_s")
    if not a or b is None or a == 0:
        return None
    return (a - b) / a * 100.0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--concurrency", type=int, default=15,
                    help="matched offered concurrency for (a) and (b)")
    ap.add_argument("--n-requests", type=int, default=30)
    ap.add_argument("--tokens", type=int, default=DEFAULT_TOKENS,
                    help="tokens per request (synthetic mode)")
    ap.add_argument("--ms-per-token", type=float, default=DEFAULT_MS_PER_TOKEN,
                    help="calibrated from arm B C=1: 26.7 tok/s -> 37.45 ms")
    ap.add_argument("--python-fraction", type=float, nargs="*",
                    default=list(DEFAULT_PYTHON_FRACTIONS),
                    help="fraction of each token step spent in GIL-HOLDING "
                         "Python work; swept because it is not known a priori")
    ap.add_argument("--synthetic", action="store_true", default=True)
    ap.add_argument("--real", action="store_true",
                    help="also run the real model at low concurrency as an anchor")
    ap.add_argument("--real-concurrency", type=int, default=2,
                    help="2-3 only: ProcessPoolExecutor loads one ~15GB replica "
                         "per worker")
    ap.add_argument("--real-requests", type=int, default=4)
    ap.add_argument("--real-max-new-tokens", type=int, default=64)
    ap.add_argument("--fixture", default="low_overlap")
    ap.add_argument("--fixtures-dir", default=os.path.join(BENCH_ROOT, "fixtures"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--with-pyspy", action="store_true")
    ap.add_argument("--out-dir", default=os.path.join(BENCH_ROOT, "results"))
    ap.add_argument("--traces-dir", default=os.path.join(BENCH_ROOT, "traces"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stamp = args.tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print("=" * 78)
    print("ABLATION: GIL attribution")
    print("=" * 78)
    print(f"concurrency        {args.concurrency} (matched for threadpool and processpool)")
    print(f"requests           {args.n_requests}")
    print(f"tokens/request     {args.tokens}")
    print(f"ms/token           {args.ms_per_token}  (arm B C=1: 26.7 tok/s)")
    print(f"python fractions   {args.python_fraction}")
    print(f"real mode          {'ON' if args.real else 'off'}")
    print()
    print("Replaces the v1 claim that >90% of host time was GIL, which was")
    print("inferred from a SINGLE-REQUEST sequential trace containing no thread")
    print("contention to measure.")

    if args.dry_run:
        print("\n--dry-run: plan only, nothing executed.")
        total = len(args.python_fraction) * 3 * args.n_requests
        print(f"  synthetic conditions  {len(args.python_fraction)} fractions "
              f"x 3 mechanisms")
        print(f"  synthetic requests    {total} total")
        print(f"  est. sequential wall  "
              f"{args.n_requests * args.tokens * args.ms_per_token / 1000:.1f}s "
              f"per fraction")
        if args.real:
            print(f"  real: concurrency {args.real_concurrency}, "
                  f"{args.real_requests} requests, "
                  f"{args.real_max_new_tokens} tokens — loads "
                  f"{args.real_concurrency} model replicas in processpool mode")
        print("  py-spy: " + ("requested" if args.with_pyspy else "not requested"))
        return 0

    print("\n[calibrate] measuring GIL-holding busy loop rate ...")
    spin_per_s = calibrate_spin()
    print(f"[calibrate] {spin_per_s:,.0f} spin iterations/s on this host")

    pyspy: Dict[str, Any] = {"available": False, "reason": "not requested"}
    if args.with_pyspy:
        os.makedirs(args.traces_dir, exist_ok=True)
        out_svg = os.path.join(args.traces_dir, f"{stamp}_gil_pyspy.svg")
        est = int(args.n_requests * args.tokens * args.ms_per_token / 1000 * 1.5) + 10
        pyspy = start_pyspy(out_svg, duration=est)
        print(f"[py-spy] {'recording -> ' + out_svg if pyspy['available'] else 'unavailable: ' + pyspy['reason']}")

    # ---- synthetic sweep ----
    synthetic: List[Dict[str, Any]] = []
    for frac in args.python_fraction:
        print(f"\n--- synthetic: python_fraction={frac} ---")
        res = run_synthetic(
            args.concurrency, args.n_requests, args.tokens,
            args.ms_per_token, frac, spin_per_s,
        )
        synthetic.append(res)
        print(f"  threadpool  {res['threadpool']['wall_time_s']:.2f}s"
              f"   processpool {res['processpool']['wall_time_s']:.2f}s"
              f"   sequential {res['sequential']['wall_time_s']:.2f}s")
        print(f"  attributable GIL cost: {fmt(res['gil_cost_pct'], '%')}")

    if args.with_pyspy and pyspy.get("available"):
        pyspy = stop_pyspy(pyspy)
        print(f"[py-spy] finished; output exists: {pyspy.get('exists')}")

    # ---- real anchor ----
    real: Optional[Dict[str, Any]] = None
    if args.real:
        print(f"\n--- real model, concurrency {args.real_concurrency} ---")
        print("    (processpool loads one model replica per worker; this is why")
        print("     real mode cannot reach the concurrency arm B runs at)")
        try:
            from common import fixtures as fx_mod  # noqa: PLC0415
            from common.config import load_config  # noqa: PLC0415

            cfg = load_config(args.config)
            model_id = cfg["model"]["id"]
            dtype_name = cfg["model"].get("dtype", "float16")
            fixture, _ = fx_mod.resolve_fixture(
                args.fixture, args.real_requests,
                int(cfg["workload"]["context_tokens"]), int(cfg.get("seed", 0)),
                model_id, args.fixtures_dir,
            )
            payloads = [
                (r["request_id"], r["prompt_token_ids"],
                 args.real_max_new_tokens, model_id, dtype_name)
                for r in fixture.requests[:args.real_requests]
            ]
            threads = run_threads(real_request, payloads, args.real_concurrency)
            procs = run_processes(
                real_request, payloads, args.real_concurrency,
                initializer=_real_init, initargs=(model_id, dtype_name),
            )
            seq = run_sequential(real_request, payloads)
            real = {
                "concurrency": args.real_concurrency,
                "requests": args.real_requests,
                "max_new_tokens": args.real_max_new_tokens,
                "threadpool": threads,
                "processpool": procs,
                "sequential": seq,
                "gil_cost_pct": _gil_cost(threads, procs),
            }
            print(f"  threadpool  {threads['wall_time_s']:.2f}s"
                  f"   processpool {procs['wall_time_s']:.2f}s"
                  f"   sequential {seq['wall_time_s']:.2f}s")
            print(f"  attributable GIL cost: {fmt(real['gil_cost_pct'], '%')}")
        except Exception as exc:  # noqa: BLE001
            real = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  real mode failed: {real['error']}")

    # ---- report ----
    print("\n" + "=" * 78)
    print("SYNTHETIC: GIL COST vs ASSUMED PYTHON FRACTION")
    print("=" * 78)
    print(render_table(
        ["python frac", "threadpool", "processpool", "sequential",
         "GIL cost", "thread tok/s", "proc tok/s"],
        [[fmt(r["python_fraction"], "", 2),
          fmt(r["threadpool"]["wall_time_s"], "s", 2),
          fmt(r["processpool"]["wall_time_s"], "s", 2),
          fmt(r["sequential"]["wall_time_s"], "s", 2),
          fmt(r["gil_cost_pct"], "%"),
          fmt(r["threadpool"]["tokens_per_s"], ""),
          fmt(r["processpool"]["tokens_per_s"], "")]
         for r in synthetic],
    ))

    print("\n" + "=" * 78)
    print("HOW TO STATE THIS RESULT")
    print("=" * 78)
    print("The GIL cost is CONDITIONAL on how much of a decode step is Python-side")
    print("work. That fraction is not known a priori — assuming it is what made the")
    print("v1 claim unsupportable. State it in this form:")
    print()
    print('  "If per-token Python framework overhead is F of the decode step,')
    print(f'   the GIL accounts for X% of arm B wall time at concurrency'
          f' {args.concurrency}."')
    print()
    if real and "gil_cost_pct" in real and real.get("gil_cost_pct") is not None:
        rc = real["gil_cost_pct"]
        print(f"The real-model anchor measured {fmt(rc, '%')} at concurrency "
              f"{args.real_concurrency}.")
        closest = min(
            (r for r in synthetic if r["gil_cost_pct"] is not None),
            key=lambda r: abs(r["gil_cost_pct"] - rc), default=None,
        )
        if closest:
            print(f"The synthetic fraction reproducing that is "
                  f"~{closest['python_fraction']}, which is the value to carry"
                  f" into the higher-concurrency synthetic reading.")
        print("Caveat: the anchor is at concurrency 2-3, where contention is")
        print("weakest. It constrains F; it does not by itself establish the")
        print("cost at arm B's operating concurrency.")
    else:
        print("No real-model anchor was run (--real). The synthetic sweep alone")
        print("gives a RANGE conditional on F, not a point estimate. Run --real")
        print("before quoting any single figure.")

    payload = {
        "ablation": "gil_attribution",
        "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "concurrency": args.concurrency,
        "n_requests": args.n_requests,
        "tokens_per_request": args.tokens,
        "ms_per_token": args.ms_per_token,
        "ms_per_token_source": "arm B concurrency=1: 26.7 output tok/s",
        "spin_iterations_per_s": spin_per_s,
        "synthetic": synthetic,
        "real": real,
        "pyspy": {k: v for k, v in pyspy.items() if k != "proc"},
        "replaces": (
            "The v1 claim that >90% of host time was GIL, which was inferred "
            "from queue-wait durations in a trace of a SINGLE-REQUEST sequential "
            "run — a run containing no thread contention to measure."
        ),
        "method_note": (
            "time.sleep releases the GIL, so a sleep-only mock would show no "
            "contention regardless of thread count. Each synthetic token spends "
            "python_fraction of its step in a calibrated GIL-HOLDING busy loop "
            "and the remainder in a GIL-RELEASING sleep, modelling framework "
            "overhead and GPU compute respectively. python_fraction is swept "
            "because its true value is unknown; --real anchors which value is "
            "plausible."
        ),
        "limitations": (
            "Synthetic mode models control flow, not the model. Real mode is "
            "limited to concurrency 2-3 because ProcessPoolExecutor loads one "
            "~15GB replica per worker. A negative GIL cost means process-pool "
            "overhead exceeded lock contention at that point, which is a real "
            "result at low concurrency."
        ),
    }
    path = write_comparison(args.out_dir, f"ablation_gil_attribution_{stamp}", payload)
    print(f"\n[write] {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
