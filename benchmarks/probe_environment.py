#!/usr/bin/env python3
"""Probe the GPU host and (over)write benchmarks/ENVIRONMENT.md.

**Run this first, before any measurement.** Two of its checks can invalidate a
planned profiling layer, and it is much cheaper to learn that now than after a
sweep has been designed around a metric the host will not produce.

The three permission tiers are tested **separately and in order**, because they
gate on different things and one result does not imply the others:

  (a) nsys CUDA trace        - CUPTI tracing of the CUDA API and kernel
                               timeline. Usually available to unprivileged
                               users.
  (b) nsys GPU metrics       - hardware counter *sampling*
      (--gpu-metrics-device)   (SM activity, warp occupancy over time).
                               Needs the same counter permission ncu does.
  (c) ncu counters           - full per-kernel counter collection with replay.

A host can pass (a) and fail (b) and (c). Recording "profiling worked" from (a)
alone and then planning on occupancy numbers is how a study discovers in its
final week that its central metric was never collectable.

ERR_NVGPUCTRPERM means the driver restricts performance counters to
administrators. On Colab the NVIDIA kernel module is loaded by the host, not by
the notebook, so `modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0` cannot
be applied from inside the session — this is very likely permanent for the
environment, not a fixable misconfiguration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ENVIRONMENT.md")

# A100-SXM4-80GB reports exactly this many bytes. Recorded as a reference point
# only: earlier write-ups asserted 40GB, and the probe's job is to settle the
# question from the device rather than to confirm either claim.
A100_80GB_BYTES = 85_094_825_984
A100_40GB_BYTES = 42_949_672_960

TINY_CUDA_SNIPPET = (
    "import torch;"
    "a=torch.randn(8,8,device='cuda');"
    "b=torch.randn(8,8,device='cuda');"
    "c=a@b;"
    "torch.cuda.synchronize();"
    "print('matmul ok', float(c.sum()))"
)

OBTAINABLE = "OBTAINABLE"
BLOCKED = "BLOCKED"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _run(cmd: List[str], timeout: int = 300) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except Exception as exc:  # pragma: no cover
        return 1, "", f"{type(exc).__name__}: {exc}"


def _tail(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "... (truncated) ...\n" + text[-limit:]


def _classify(rc: int, out: str, err: str, artifact: Optional[str]) -> Dict[str, Any]:
    blob = f"{out}\n{err}"
    perm = "ERR_NVGPUCTRPERM" in blob or "insufficient permissions" in blob.lower()
    missing = rc == 127 or "command not found" in blob

    if missing:
        status, reason = BLOCKED, "not installed"
    elif perm:
        status, reason = BLOCKED, "ERR_NVGPUCTRPERM (driver restricts counters to admin)"
    elif rc != 0:
        status, reason = BLOCKED, f"exit code {rc}"
    elif artifact and not _artifact_exists(artifact):
        status, reason = BLOCKED, "command succeeded but produced no output file"
    else:
        status, reason = OBTAINABLE, "ok"

    return {
        "status": status,
        "reason": reason,
        "returncode": rc,
        "permission_error": perm,
        "not_installed": missing,
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }


def _artifact_exists(prefix: str) -> bool:
    # nsys/ncu append their own suffix (.nsys-rep, .ncu-rep, .sqlite).
    d = os.path.dirname(prefix) or "."
    base = os.path.basename(prefix)
    try:
        return any(f.startswith(base) for f in os.listdir(d))
    except OSError:
        return False


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------


def probe_torch_device() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import torch  # noqa: PLC0415

        info["torch_version"] = torch.__version__
        info["cuda_runtime_version"] = torch.version.cuda
        info["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        )
        info["available"] = bool(torch.cuda.is_available())
        if not info["available"]:
            info["error"] = "torch.cuda.is_available() is False"
            return info

        props = torch.cuda.get_device_properties(0)
        total = int(props.total_memory)
        info.update(
            {
                "device_name": props.name,
                "total_memory_bytes": total,
                "total_memory_gib": round(total / (1024 ** 3), 2),
                "total_memory_mb": total // (1024 ** 2),
                "sm_count": props.multi_processor_count,
                "compute_capability": f"{props.major}.{props.minor}",
                "matches_a100_80gb": total == A100_80GB_BYTES,
                "matches_a100_40gb": total == A100_40GB_BYTES,
            }
        )
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def probe_nvml() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False}
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)

        def _s(v):
            return v.decode() if isinstance(v, bytes) else str(v)

        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        info.update(
            {
                "available": True,
                "gpu_name": _s(pynvml.nvmlDeviceGetName(h)),
                "gpu_uuid": _s(pynvml.nvmlDeviceGetUUID(h)),
                "driver_version": _s(pynvml.nvmlSystemGetDriverVersion()),
                "total_vram_bytes": int(mem.total),
                "total_vram_mb": int(mem.total // (1024 ** 2)),
                "device_count": pynvml.nvmlDeviceGetCount(),
            }
        )
        try:
            info["persistence_mode"] = pynvml.nvmlDeviceGetPersistenceMode(h)
        except Exception:
            info["persistence_mode"] = None
        pynvml.nvmlShutdown()
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def probe_nsys_cuda_trace(tmpdir: str) -> Dict[str, Any]:
    """Tier (a): plain CUDA trace. No hardware counters involved."""
    out = os.path.join(tmpdir, "probe_nsys_a")
    cmd = [
        "nsys", "profile", "--trace=cuda", "--force-overwrite=true",
        "-o", out, sys.executable, "-c", TINY_CUDA_SNIPPET,
    ]
    rc, so, se = _run(cmd)
    res = _classify(rc, so, se, out)
    res["command"] = " ".join(cmd)
    return res


def probe_nsys_gpu_metrics(tmpdir: str) -> Dict[str, Any]:
    """Tier (b): counter *sampling*. Gates separately from tier (a)."""
    out = os.path.join(tmpdir, "probe_nsys_b")
    cmd = [
        "nsys", "profile", "--trace=cuda", "--gpu-metrics-device=0",
        "--force-overwrite=true", "-o", out, sys.executable, "-c", TINY_CUDA_SNIPPET,
    ]
    rc, so, se = _run(cmd)
    res = _classify(rc, so, se, out)
    res["command"] = " ".join(cmd)
    return res


def probe_ncu(tmpdir: str) -> Dict[str, Any]:
    """Tier (c): full per-kernel counter collection with replay."""
    cmd = [
        "ncu", "--metrics", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "--launch-count", "1", sys.executable, "-c", TINY_CUDA_SNIPPET,
    ]
    rc, so, se = _run(cmd)
    res = _classify(rc, so, se, None)
    res["command"] = " ".join(cmd)
    return res


def probe_vllm() -> Dict[str, Any]:
    """Does vLLM import, and did installing it move torch underneath us?"""
    info: Dict[str, Any] = {"importable": False}
    try:
        import torch  # noqa: PLC0415

        info["torch_version"] = torch.__version__
    except Exception as exc:
        info["torch_version"] = None
        info["torch_error"] = str(exc)
    try:
        import vllm  # noqa: PLC0415

        info["importable"] = True
        info["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def probe_disk(paths: Optional[List[str]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in paths or ["/content", "/content/drive/MyDrive", "/tmp", REPO_ROOT]:
        try:
            if not os.path.exists(p):
                out[p] = {"exists": False}
                continue
            u = shutil.disk_usage(p)
            out[p] = {
                "exists": True,
                "total_gb": round(u.total / 1e9, 2),
                "used_gb": round(u.used / 1e9, 2),
                "free_gb": round(u.free / 1e9, 2),
            }
        except Exception as exc:
            out[p] = {"exists": True, "error": str(exc)}
    return out


def probe_pip_freeze() -> Dict[str, Any]:
    rc, so, se = _run([sys.executable, "-m", "pip", "freeze"], timeout=300)
    if rc != 0:
        return {"ok": False, "error": _tail(se)}
    return {
        "ok": True,
        "sha256": hashlib.sha256(so.encode()).hexdigest(),
        "package_count": len([ln for ln in so.splitlines() if ln.strip()]),
        "text": so,
    }


def probe_tool_versions() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tool in ("nsys", "ncu", "nvidia-smi"):
        path = shutil.which(tool)
        if not path:
            out[tool] = {"present": False}
            continue
        rc, so, se = _run([tool, "--version"], timeout=60)
        out[tool] = {
            "present": True,
            "path": path,
            "version": _tail(so or se, 300),
        }
    return out


# --------------------------------------------------------------------------
# capability roll-up
# --------------------------------------------------------------------------


def derive_capabilities(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Which requested metrics are actually measurable given the probe results.

    This is the section the study plan has to be reconciled against. Every entry
    names the layer that would produce it and, when blocked, says so plainly
    rather than leaving it to be discovered later.
    """
    nsys_a = results["nsys_cuda_trace"]["status"] == OBTAINABLE
    nsys_b = results["nsys_gpu_metrics"]["status"] == OBTAINABLE
    ncu_ok = results["ncu_counters"]["status"] == OBTAINABLE
    torch_ok = results["torch_device"].get("available", False)
    nvml_ok = results["nvml"].get("available", False)

    def entry(metric: str, ok: bool, layer: str, note: str) -> Dict[str, str]:
        return {
            "metric": metric,
            "measurable": "YES" if ok else "NO",
            "layer": layer,
            "note": note,
        }

    return [
        entry("Latency: TTFT, ITL, e2e (p50/p95/p99)", torch_ok, "1",
              "Application timing around a streamer. No special permission."),
        entry("Throughput: input/output/total tok per s", torch_ok, "1",
              "Application timing. Reported separately, never merged."),
        entry("VRAM peak and steady-state", nvml_ok, "1",
              "NVML process memory + torch allocator peak."),
        entry("CPU per-core utilization", True, "1",
              "psutil. Host-side; no GPU permission needed."),
        entry("GPU 'utilization' (kernel residency)", nvml_ok, "1",
              "NVML utilization.gpu. NOT occupancy — see monitor.py."),
        entry("Kernel timeline, durations, stream overlap", nsys_a, "2/3",
              "nsys CUDA trace and/or torch.profiler. CUPTI tracing only."),
        entry("cudaLaunchKernel / sync / memcpy CPU time breakdown", nsys_a, "3",
              "torch.profiler cuda_runtime rows; this is the launch/schedule/"
              "wait metric."),
        entry("Prefill vs decode kernel time split", nsys_a, "3",
              "Requires the explicit t_first_token boundary from Layer 1."),
        entry("SM activity over time (duration-weighted)", nsys_b, "2",
              "nsys --gpu-metrics-device. Needs counter permission."
              if not nsys_b else "nsys --gpu-metrics-device."),
        entry("Achieved occupancy (warps active vs peak)", ncu_ok, "4",
              "ncu only. No substitute exists — NVML and nsys tracing cannot "
              "produce this." if not ncu_ok else "ncu per-kernel counters."),
        entry("Tensor Core / HMMA pipe utilization", ncu_ok, "4",
              "ncu only. Kernel *names* containing s16816gemm show tensor-core "
              "GEMMs ran, which is not the same as how well they were used."
              if not ncu_ok else "ncu sm__pipe_tensor_op_hmma_cycles_active."),
        entry("Memory bandwidth achieved vs peak", ncu_ok, "4",
              "ncu throughput counters." if ncu_ok else "ncu only; blocked."),
    ]


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def render_markdown(results: Dict[str, Any]) -> str:
    td = results["torch_device"]
    nv = results["nvml"]
    caps = results["capabilities"]

    lines: List[str] = []
    A = lines.append

    A("# Benchmark Environment")
    A("")
    A(f"**Machine-generated by `benchmarks/probe_environment.py` at "
      f"{results['utc_timestamp']}.** Do not hand-edit — re-run the probe.")
    A("")
    A("Re-run this on every new Colab session that produces committed results. "
      "Colab reallocates hardware between sessions; a GPU UUID from a previous "
      "session is not evidence about this one.")
    A("")

    # ---- device ----
    A("## Device")
    A("")
    if td.get("available"):
        total_b = td["total_memory_bytes"]
        A(f"| Field | Value |")
        A(f"|---|---|")
        A(f"| Device name | `{td['device_name']}` |")
        A(f"| **Total memory (bytes)** | **{total_b:,}** |")
        A(f"| Total memory | {td['total_memory_gib']} GiB ({td['total_memory_mb']:,} MB) |")
        A(f"| SM count | {td['sm_count']} |")
        A(f"| Compute capability | {td['compute_capability']} |")
        A(f"| GPU UUID | `{nv.get('gpu_uuid', 'UNKNOWN')}` |")
        A(f"| Driver version | `{nv.get('driver_version', 'UNKNOWN')}` |")
        A(f"| CUDA runtime (torch linked) | `{td.get('cuda_runtime_version')}` |")
        A(f"| torch version | `{td.get('torch_version')}` |")
        A("")
        if td["matches_a100_80gb"]:
            A(f"**Memory: this is the 80GB part.** {total_b:,} bytes matches "
              f"A100-SXM4-80GB exactly.")
        elif td["matches_a100_40gb"]:
            A(f"**Memory: this is the 40GB part.** {total_b:,} bytes matches "
              f"A100-SXM4-40GB exactly.")
        else:
            A(f"**Memory: {total_b:,} bytes — matches neither the 80GB "
              f"({A100_80GB_BYTES:,}) nor the 40GB ({A100_40GB_BYTES:,}) A100 "
              f"reference value.** Use the measured figure; do not substitute "
              f"either reference.")
        A("")
        A("Every VRAM headroom claim in the write-up must be computed against "
          "the measured figure above, not against a remembered card spec.")
    else:
        A("**No CUDA device visible.**")
        A("")
        A(f"```\n{td.get('error', 'unknown error')}\n```")
        A("")
        A("Every GPU capability below is consequently reported as BLOCKED. "
          "Re-run on a GPU runtime before drawing any conclusion from this file.")
    A("")

    # ---- permission tiers ----
    A("## Profiling permission tiers")
    A("")
    A("Tested separately. Passing (a) does **not** imply (b) or (c) — (a) is "
      "CUPTI tracing, while (b) and (c) need hardware counter access.")
    A("")
    A("| Tier | Capability | Status | Reason |")
    A("|---|---|---|---|")
    for key, label in (
        ("nsys_cuda_trace", "(a) nsys CUDA trace"),
        ("nsys_gpu_metrics", "(b) nsys GPU metrics sampling"),
        ("ncu_counters", "(c) ncu counter collection"),
    ):
        r = results[key]
        A(f"| {label} | {'trace' if key != 'ncu_counters' else 'counters'} | "
          f"**{r['status']}** | {r['reason']} |")
    A("")

    for key, label in (
        ("nsys_cuda_trace", "(a) nsys CUDA trace"),
        ("nsys_gpu_metrics", "(b) nsys GPU metrics sampling"),
        ("ncu_counters", "(c) ncu counter collection"),
    ):
        r = results[key]
        A(f"### {label} — {r['status']}")
        A("")
        A(f"```\n{r['command']}\n```")
        A("")
        if r["stdout_tail"]:
            A("stdout:")
            A(f"```\n{r['stdout_tail']}\n```")
        if r["stderr_tail"]:
            A("stderr:")
            A(f"```\n{r['stderr_tail']}\n```")
        if r["permission_error"]:
            A("")
            A("`ERR_NVGPUCTRPERM`: the driver restricts performance counters to "
              "administrators. On Colab the NVIDIA kernel module is loaded by "
              "the **host**, so the usual fix "
              "(`NVreg_RestrictProfilingToAdminUsers=0`) cannot be applied from "
              "inside the session. Treat this as a permanent property of the "
              "environment and plan the study without this metric.")
        if r["not_installed"]:
            A("")
            A("Tool not on PATH. Nsight Systems and Nsight Compute come from "
              "the **NVIDIA apt repository, not pip** — see COLAB.md.")
        A("")

    # ---- vllm ----
    A("## vLLM vs torch")
    A("")
    vl = results["vllm"]
    if vl.get("importable"):
        A(f"- vLLM `{vl.get('vllm_version')}` imports successfully.")
        A(f"- torch after install: `{vl.get('torch_version')}`.")
        A("")
        A("If that torch version differs from the one Layer 1 / arm B ran "
          "against, **the environment moved mid-study**. Arm C then needs its "
          "own `pip_freeze_sha256` and the difference must be disclosed in the "
          "write-up — it is not a detail to smooth over.")
    else:
        A(f"- vLLM does **not** import: `{vl.get('error', 'unknown')}`")
        A(f"- torch: `{vl.get('torch_version')}`")
        A("")
        A("Arm C (optimized self-hosted) is blocked until this resolves.")
    A("")

    # ---- disk ----
    A("## Disk")
    A("")
    A("| Path | Exists | Free (GB) | Total (GB) |")
    A("|---|---|---|---|")
    for path, d in results["disk"].items():
        if not d.get("exists"):
            A(f"| `{path}` | no | — | — |")
        elif "error" in d:
            A(f"| `{path}` | yes | error: {d['error']} | — |")
        else:
            A(f"| `{path}` | yes | {d['free_gb']} | {d['total_gb']} |")
    A("")
    A("A full profiled run emits ~1.5 GB of raw trace. Check free space against "
      "the number of traced runs planned, and write results to mounted Drive — "
      "the local disk does not survive preemption.")
    A("")

    # ---- provenance ----
    A("## Provenance")
    A("")
    pf = results["pip_freeze"]
    A("| Field | Value |")
    A("|---|---|")
    A(f"| `pip_freeze_sha256` | `{pf.get('sha256', 'FAILED')}` |")
    A(f"| packages | {pf.get('package_count', '—')} |")
    A(f"| branch SHA | `{results['git']['branch_sha']}` |")
    A(f"| upstream SHA | `{results['git']['upstream_sha']}` "
      f"(ref: `{results['git']['upstream_ref']}`) |")
    A("")
    for tool, d in results["tools"].items():
        if d.get("present"):
            A(f"- `{tool}`: `{d['path']}`")
        else:
            A(f"- `{tool}`: **not on PATH**")
    A("")

    # ---- the section that matters ----
    A("## Measurable vs not measurable")
    A("")
    A("Given the probe results above, this is what this environment can and "
      "cannot produce. Reconcile the study plan against it **before** running a "
      "sweep.")
    A("")
    A("| Metric | Measurable | Layer | Note |")
    A("|---|---|---|---|")
    for c in caps:
        A(f"| {c['metric']} | **{c['measurable']}** | {c['layer']} | {c['note']} |")
    A("")

    blocked = [c for c in caps if c["measurable"] == "NO"]
    if blocked:
        A("### Not obtainable in this environment")
        A("")
        for c in blocked:
            A(f"- **{c['metric']}** — {c['note']}")
        A("")
        A("These must be either dropped from the study or obtained on different "
          "hardware. Do not substitute a Layer 1 proxy for a Layer 4 metric: "
          "NVML `utilization.gpu` is not occupancy, and a kernel-name match is "
          "not tensor pipe utilization.")
    else:
        A("All requested metrics are obtainable in this environment.")
    A("")

    # ---- machine-readable ----
    A("## Machine-readable probe results")
    A("")
    A("Parsed by `profiling/run_nsys.sh` and `profiling/run_ncu.sh`. Do not "
      "reformat.")
    A("")
    A("<!-- PROBE_RESULTS_BEGIN -->")
    A("```")
    A(f"nsys_cuda_trace={results['nsys_cuda_trace']['status']}")
    A(f"nsys_gpu_metrics={results['nsys_gpu_metrics']['status']}")
    A(f"ncu_counters={results['ncu_counters']['status']}")
    A(f"gpu_uuid={nv.get('gpu_uuid', 'UNKNOWN')}")
    A(f"gpu_name={nv.get('gpu_name', td.get('device_name', 'UNKNOWN'))}")
    A(f"total_vram_bytes={td.get('total_memory_bytes', 0)}")
    A(f"sm_count={td.get('sm_count', 0)}")
    A(f"driver_version={nv.get('driver_version', 'UNKNOWN')}")
    A(f"cuda_version={td.get('cuda_runtime_version', 'UNKNOWN')}")
    A(f"torch_version={td.get('torch_version', 'UNKNOWN')}")
    A(f"pip_freeze_sha256={pf.get('sha256', 'UNKNOWN')}")
    A(f"probe_utc={results['utc_timestamp']}")
    A("```")
    A("<!-- PROBE_RESULTS_END -->")
    A("")
    return "\n".join(lines)


def _git_info() -> Dict[str, Any]:
    def g(*a):
        try:
            p = subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=15)
            return p.stdout.strip() if p.returncode == 0 else None
        except Exception:
            return None

    branch = g("rev-parse", "HEAD")
    if branch and g("status", "--porcelain"):
        branch = f"{branch}-dirty"
    upstream_sha = upstream_ref = None
    for ref in ("upstream/main", "origin/main", "main"):
        s = g("rev-parse", ref)
        if s:
            upstream_sha, upstream_ref = s, ref
            break
    return {
        "branch_sha": branch,
        "upstream_sha": upstream_sha,
        "upstream_ref": upstream_ref,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=ENV_MD, help="path to ENVIRONMENT.md")
    ap.add_argument("--json-out", default=None, help="also write raw results JSON")
    ap.add_argument("--skip-nsys", action="store_true")
    ap.add_argument("--skip-ncu", action="store_true")
    ap.add_argument("--skip-pip-freeze", action="store_true")
    args = ap.parse_args(argv)

    skipped = {
        "status": BLOCKED,
        "reason": "skipped by flag",
        "returncode": -1,
        "permission_error": False,
        "not_installed": False,
        "stdout_tail": "",
        "stderr_tail": "",
        "command": "(skipped)",
    }

    print("=" * 72)
    print("BENCHMARK ENVIRONMENT PROBE")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as tmp:
        results: Dict[str, Any] = {
            "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "torch_device": probe_torch_device(),
            "nvml": probe_nvml(),
            "tools": probe_tool_versions(),
        }
        td = results["torch_device"]
        print(f"\n[device] {td.get('device_name', 'NO CUDA DEVICE')}")
        if td.get("available"):
            print(f"[device] total memory: {td['total_memory_bytes']:,} bytes "
                  f"({td['total_memory_gib']} GiB), SMs: {td['sm_count']}")
            if td["matches_a100_80gb"]:
                print("[device] -> matches A100-SXM4-80GB exactly")
            elif td["matches_a100_40gb"]:
                print("[device] -> matches A100-SXM4-40GB exactly")
            else:
                print("[device] -> matches NEITHER the 80GB nor 40GB reference value")
        print(f"[nvml]   uuid: {results['nvml'].get('gpu_uuid', 'UNKNOWN')}")

        print("\n[tier a] nsys CUDA trace ...")
        results["nsys_cuda_trace"] = skipped if args.skip_nsys else probe_nsys_cuda_trace(tmp)
        print(f"[tier a] {results['nsys_cuda_trace']['status']} — "
              f"{results['nsys_cuda_trace']['reason']}")

        print("[tier b] nsys GPU metrics sampling ...")
        results["nsys_gpu_metrics"] = skipped if args.skip_nsys else probe_nsys_gpu_metrics(tmp)
        print(f"[tier b] {results['nsys_gpu_metrics']['status']} — "
              f"{results['nsys_gpu_metrics']['reason']}")

        print("[tier c] ncu counters ...")
        results["ncu_counters"] = skipped if args.skip_ncu else probe_ncu(tmp)
        print(f"[tier c] {results['ncu_counters']['status']} — "
              f"{results['ncu_counters']['reason']}")

    results["vllm"] = probe_vllm()
    results["disk"] = probe_disk()
    results["pip_freeze"] = (
        {"ok": False, "error": "skipped"} if args.skip_pip_freeze else probe_pip_freeze()
    )
    results["git"] = _git_info()
    results["capabilities"] = derive_capabilities(results)

    md = render_markdown(results)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(md)
    print(f"\n[write] {args.out}")

    if args.json_out:
        payload = json.loads(json.dumps(results, default=str))
        payload.get("pip_freeze", {}).pop("text", None)
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[write] {args.json_out}")

    print("\n" + "=" * 72)
    print("MEASURABLE vs NOT")
    print("=" * 72)
    for c in results["capabilities"]:
        mark = "YES" if c["measurable"] == "YES" else "NO "
        print(f"  [{mark}] L{c['layer']:<3} {c['metric']}")
    blocked = [c for c in results["capabilities"] if c["measurable"] == "NO"]
    if blocked:
        print(f"\n  {len(blocked)} metric(s) NOT obtainable here. "
              f"Reconcile the study plan before sweeping.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
