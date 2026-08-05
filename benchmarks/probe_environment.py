#!/usr/bin/env python3
"""Probe the GPU host and (over)write benchmarks/ENVIRONMENT.md.

**Run this first, before any measurement.** Its results gate which profiling
layers the study can use, and it is much cheaper to learn that now than after a
sweep has been designed around a metric the host will not produce.

TOOLS ARE RESOLVED BY SEARCHING, NOT BY NAME
--------------------------------------------
``nsys`` and ``ncu`` are frequently **not on PATH**, and on some images nsys is
not separately installable at all — it ships bundled *inside* the Nsight
Compute tree, e.g.::

    /opt/nvidia/nsight-compute/<version>/host/target-linux-x64/nsys

So each tool is resolved by trying ``shutil.which`` first and then globbing a
list of known install layouts, picking the **highest version** when several are
present. Versions are never hardcoded: the image changes between Colab sessions
and lab hardware will differ again. The resolved **absolute path** and the
version string reported by the binary itself are both recorded in the
machine-readable block, and the profiling wrappers invoke those paths rather
than relying on PATH.

FOUND-BUT-BLOCKED IS NOT NOT-FOUND
----------------------------------
These are different states with different remedies, and collapsing them costs
real time: "install the tool" and "the driver will not let this tool read
counters" are not the same problem. Every tier result therefore records the
resolved path alongside its status, and a tier whose tool was never found is
reported distinctly from one whose tool ran and failed.

THE THREE PERMISSION TIERS ARE TESTED SEPARATELY
------------------------------------------------
  (a) nsys CUDA trace        - CUPTI tracing of the CUDA API and kernel
                               timeline. Usually available unprivileged.
  (b) nsys GPU metrics       - hardware counter *sampling* (SM activity, warp
      sampling                 occupancy over time). Needs the same counter
                               permission ncu does.
  (c) ncu counters           - full per-kernel counter collection with replay.

A host can pass (a) and fail (b) and (c). ``ERR_NVGPUCTRPERM`` means the driver
restricts performance counters to administrators; where the NVIDIA kernel
module is loaded by the host rather than the session,
``NVreg_RestrictProfilingToAdminUsers=0`` cannot be applied from inside and the
restriction is permanent for that environment.
"""

from __future__ import annotations

import argparse
import glob as glob_mod
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ENVIRONMENT.md")

# Reference values used ONLY to tell the reader which part the measured figure
# matches. Never substituted for a measurement.
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

# Install layouts to glob when a tool is not on PATH. Ordered by preference
# only for tie-breaking; selection is by version, highest wins.
#
# The first nsys pattern is the important one: on images where Nsight Systems
# has no standalone package, nsys still exists inside the Nsight Compute tree.
NSYS_SEARCH_PATTERNS: Tuple[str, ...] = (
    "/opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys",
    "/opt/nvidia/nsight-systems/*/target-linux-x64/nsys",
    "/opt/nvidia/nsight-systems/*/bin/nsys",
    "/usr/local/cuda/bin/nsys",
    "/usr/local/cuda-*/bin/nsys",
)
NCU_SEARCH_PATTERNS: Tuple[str, ...] = (
    "/usr/local/cuda/bin/ncu",
    "/usr/local/cuda-*/bin/ncu",
    "/opt/nvidia/nsight-compute/*/ncu",
    "/opt/nvidia/nsight-compute/*/target/*/ncu",
)

_VERSION_COMPONENT = re.compile(r"^\d+(?:\.\d+)*$")
_VERSION_IN_TEXT = re.compile(r"(\d+\.\d+(?:\.\d+)*)")


# --------------------------------------------------------------------------
# tool resolution
# --------------------------------------------------------------------------


def parse_path_version(path: str) -> Tuple[int, ...]:
    """Extract a version tuple from a path's directory components.

    ``/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`` ->
    ``(2025, 1, 1)``. Components like ``target-linux-x64`` contain digits but
    are not pure dotted numbers, so they do not match. A path carrying no
    version yields ``()``, which sorts below every real version — so a
    versioned install wins a tie, while an unversioned one is still usable when
    it is the only candidate.
    """
    best: Tuple[int, ...] = ()
    for part in path.split(os.sep):
        if _VERSION_COMPONENT.match(part):
            try:
                candidate = tuple(int(x) for x in part.split("."))
            except ValueError:  # pragma: no cover - regex already guards
                continue
            if candidate > best:
                best = candidate
    return best


def resolve_tool(
    name: str,
    patterns: Sequence[str],
    which_fn: Callable[[str], Optional[str]] = shutil.which,
    glob_fn: Callable[[str], List[str]] = glob_mod.glob,
    version_fn: Optional[Callable[[str], Optional[str]]] = None,
) -> Dict[str, Any]:
    """Locate ``name``: PATH first, then the glob patterns, highest version wins.

    ``which_fn``/``glob_fn``/``version_fn`` are injectable so the selection
    logic can be unit-tested against a mocked filesystem with no GPU present.
    """
    on_path = which_fn(name)
    candidates: List[str] = []
    for pattern in patterns:
        try:
            candidates.extend(glob_fn(pattern))
        except Exception:  # pragma: no cover - defensive
            continue
    # Deduplicate while keeping a stable order for the recorded candidate list.
    seen = set()
    unique: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    if on_path:
        path, source = on_path, "PATH"
    elif unique:
        # Highest version wins; path as a deterministic tie-break so two
        # installs of the same version resolve reproducibly.
        path = max(unique, key=lambda p: (parse_path_version(p), p))
        source = "search"
    else:
        path, source = None, None

    version = None
    if path is not None:
        vf = version_fn if version_fn is not None else _tool_version
        try:
            version = vf(path)
        except Exception:  # pragma: no cover - defensive
            version = None

    return {
        "name": name,
        "found": path is not None,
        "path": path,
        "source": source,
        "version": version,
        "candidates": unique,
        "on_path": on_path,
        "search_patterns": list(patterns),
    }


def _tool_version(path: str) -> Optional[str]:
    """Version string reported by the binary itself, not inferred from its path."""
    rc, out, err = _run([path, "--version"], timeout=60)
    blob = (out or "") + "\n" + (err or "")
    blob = blob.strip()
    if not blob:
        return None
    m = _VERSION_IN_TEXT.search(blob)
    if m:
        return m.group(1)
    return blob.splitlines()[0][:120]


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


def _artifact_exists(prefix: str) -> bool:
    # nsys/ncu append their own suffix (.nsys-rep, .ncu-rep, .sqlite).
    d = os.path.dirname(prefix) or "."
    base = os.path.basename(prefix)
    try:
        return any(f.startswith(base) for f in os.listdir(d))
    except OSError:
        return False


def _not_found_result(tool: Dict[str, Any], label: str) -> Dict[str, Any]:
    """A tier whose tool was never located. Distinct from found-but-failing."""
    return {
        "status": BLOCKED,
        "reason": "tool not found",
        "returncode": -1,
        "permission_error": False,
        "tool_found": False,
        "tool_path": None,
        "tool_version": None,
        "stdout_tail": "",
        "stderr_tail": (
            f"{tool['name']} not on PATH and not matched by any search pattern:\n"
            + "\n".join(f"  {p}" for p in tool["search_patterns"])
        ),
        "command": f"({label}: {tool['name']} not found)",
    }


def _classify(
    rc: int,
    out: str,
    err: str,
    artifact: Optional[str],
    tool: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify a tier test whose tool WAS found. ``found`` is already true."""
    blob = f"{out}\n{err}"
    perm = "ERR_NVGPUCTRPERM" in blob or "insufficient permissions" in blob.lower()

    if perm:
        status, reason = BLOCKED, "ERR_NVGPUCTRPERM (driver restricts counters to admin)"
    elif rc == 127:
        # The resolved path stopped working between resolution and invocation.
        status, reason = BLOCKED, "resolved path failed to execute"
    elif rc != 0:
        status, reason = BLOCKED, f"tool ran but exited {rc}"
    elif artifact and not _artifact_exists(artifact):
        status, reason = BLOCKED, "command succeeded but produced no output file"
    else:
        status, reason = OBTAINABLE, "ok"

    return {
        "status": status,
        "reason": reason,
        "returncode": rc,
        "permission_error": perm,
        "tool_found": True,
        "tool_path": tool["path"],
        "tool_version": tool["version"],
        "stdout_tail": _tail(out),
        "stderr_tail": _tail(err),
    }


# --------------------------------------------------------------------------
# device probes
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
        # Provided by nvidia-ml-py (NOT the deprecated pynvml package); the
        # import name is the same. See benchmarks/requirements.txt.
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


# --------------------------------------------------------------------------
# tier probes
# --------------------------------------------------------------------------


def probe_nsys_cuda_trace(tool: Dict[str, Any], tmpdir: str) -> Dict[str, Any]:
    """Tier (a): plain CUDA trace. No hardware counters involved."""
    if not tool["found"]:
        return _not_found_result(tool, "tier a")
    out = os.path.join(tmpdir, "probe_nsys_a")
    cmd = [
        tool["path"], "profile", "--trace=cuda", "--force-overwrite=true",
        "-o", out, sys.executable, "-c", TINY_CUDA_SNIPPET,
    ]
    rc, so, se = _run(cmd)
    res = _classify(rc, so, se, out, tool)
    res["command"] = " ".join(cmd)
    return res


def probe_nsys_gpu_metrics(tool: Dict[str, Any], tmpdir: str) -> Dict[str, Any]:
    """Tier (b): counter *sampling*. Gates separately from tier (a).

    Tries the plural ``--gpu-metrics-devices`` first. The singular
    ``--gpu-metrics-device`` is deprecated in nsys 2025.x and removed later; the
    plural form is unrecognised by older builds. Rather than branching on a
    parsed version — which would need updating every time nsys changes — the
    probe *tries* the plural form and falls back to the singular only when the
    tool rejects the option itself. The flag that worked is recorded so
    ``run_nsys.sh`` never has to guess.
    """
    if not tool["found"]:
        res = _not_found_result(tool, "tier b")
        res["flag_used"] = None
        return res

    attempts: List[Dict[str, Any]] = []
    for idx, flag in enumerate(("--gpu-metrics-devices", "--gpu-metrics-device")):
        out = os.path.join(tmpdir, f"probe_nsys_b{idx}")
        cmd = [
            tool["path"], "profile", "--trace=cuda", f"{flag}=0",
            "--force-overwrite=true", "-o", out, sys.executable, "-c", TINY_CUDA_SNIPPET,
        ]
        rc, so, se = _run(cmd)
        res = _classify(rc, so, se, out, tool)
        res["command"] = " ".join(cmd)
        res["flag_used"] = flag
        attempts.append({"flag": flag, "status": res["status"], "reason": res["reason"]})

        if res["status"] == OBTAINABLE:
            res["attempts"] = attempts
            return res

        # Only fall through to the deprecated spelling when the tool rejected
        # the option. A permission failure or any other error means the flag was
        # understood and the capability is genuinely unavailable — retrying with
        # a different spelling would just mislabel the reason.
        blob = f"{so}\n{se}".lower()
        option_rejected = (
            "unrecognized" in blob or "unrecognised" in blob
            or "invalid option" in blob or "unknown option" in blob
            or "not a valid" in blob
        )
        if not option_rejected:
            res["attempts"] = attempts
            return res

    res["attempts"] = attempts
    return res


def probe_nsys_metric_sets(tool: Dict[str, Any]) -> Dict[str, Any]:
    """List the available ``--gpu-metrics-set`` values.

    The default set is "General Metrics". Whether a set with better Tensor-pipe
    coverage exists is a decision that has to be made *before* the real runs —
    the set is baked into the capture and cannot be changed afterwards.
    """
    if not tool["found"]:
        return {"available": False, "reason": "nsys not found", "raw": "", "sets": []}

    cmd = [tool["path"], "profile", "--gpu-metrics-set=help"]
    rc, so, se = _run(cmd, timeout=120)
    blob = (so or "") + ("\n" + se if se else "")

    # Output shape varies by nsys version, so the raw text is recorded verbatim
    # and the parse is best-effort on top of it.
    sets: List[Dict[str, str]] = []
    for line in blob.splitlines():
        m = re.match(r"\s*\[?(\d+)\]?\s+([A-Za-z0-9_.\-]+)\s*(?:\((.*)\))?\s*$", line)
        if m:
            sets.append({"index": m.group(1), "name": m.group(2),
                         "description": (m.group(3) or "").strip()})

    return {
        "available": rc == 0 or bool(sets),
        "returncode": rc,
        "command": " ".join(cmd),
        "raw": _tail(blob, 4000),
        "sets": sets,
    }


def probe_ncu(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Tier (c): full per-kernel counter collection with replay."""
    if not tool["found"]:
        return _not_found_result(tool, "tier c")
    cmd = [
        tool["path"], "--metrics", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "--launch-count", "1", sys.executable, "-c", TINY_CUDA_SNIPPET,
    ]
    rc, so, se = _run(cmd)
    res = _classify(rc, so, se, None, tool)
    res["command"] = " ".join(cmd)
    return res


# --------------------------------------------------------------------------
# other probes
# --------------------------------------------------------------------------


def probe_vllm() -> Dict[str, Any]:
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


# --------------------------------------------------------------------------
# capability roll-up
# --------------------------------------------------------------------------


def derive_capabilities(results: Dict[str, Any]) -> List[Dict[str, str]]:
    """Which requested metrics are measurable given the probe results."""
    nsys_a = results["nsys_cuda_trace"]["status"] == OBTAINABLE
    nsys_b = results["nsys_gpu_metrics"]["status"] == OBTAINABLE
    ncu_ok = results["ncu_counters"]["status"] == OBTAINABLE
    torch_ok = results["torch_device"].get("available", False)
    nvml_ok = results["nvml"].get("available", False)

    def entry(metric: str, ok: bool, layer: str, note: str) -> Dict[str, str]:
        return {"metric": metric, "measurable": "YES" if ok else "NO",
                "layer": layer, "note": note}

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
              "nsys GPU metrics sampling."
              if nsys_b else "nsys GPU metrics sampling. Needs counter permission."),
        entry("Achieved occupancy (warps active vs peak)", ncu_ok, "4",
              "ncu per-kernel counters."
              if ncu_ok else "ncu only. No substitute exists — NVML and nsys "
                             "tracing cannot produce this."),
        entry("Tensor Core / HMMA pipe utilization", ncu_ok, "4",
              "ncu sm__pipe_tensor_op_hmma_cycles_active."
              if ncu_ok else "ncu only. Kernel *names* containing s16816gemm show "
                             "tensor-core GEMMs ran, which is not the same as how "
                             "well they were used."),
        entry("Memory bandwidth achieved vs peak", ncu_ok, "4",
              "ncu throughput counters." if ncu_ok else "ncu only; blocked."),
    ]


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

EFA_NOTE = (
    "**The `efa_metrics` warning from nsys is benign — do not re-investigate.** "
    "A message like `Executable path does not exist: "
    ".../plugins/efa_metrics/nic_sampler` refers to a sampler for AWS Elastic "
    "Fabric Adapter network interfaces. It is absent from bundled/partial Nsight "
    "builds, has nothing to do with GPU profiling, and does not affect CUDA "
    "tracing, GPU metrics sampling, or the resulting report."
)


def _tool_row(tool: Dict[str, Any]) -> str:
    if not tool["found"]:
        return f"| `{tool['name']}` | **NOT FOUND** | — | — |"
    return (
        f"| `{tool['name']}` | found ({tool['source']}) | "
        f"`{tool['path']}` | `{tool['version'] or 'unknown'}` |"
    )


def render_markdown(results: Dict[str, Any]) -> str:
    td = results["torch_device"]
    nv = results["nvml"]
    caps = results["capabilities"]
    nsys = results["tools"]["nsys"]
    ncu = results["tools"]["ncu"]

    lines: List[str] = []
    A = lines.append

    A("# Benchmark Environment")
    A("")
    A(f"**Machine-generated by `benchmarks/probe_environment.py` at "
      f"{results['utc_timestamp']}.** Do not hand-edit — re-run the probe.")
    A("")
    A("Re-run this on every new Colab session that produces committed results. "
      "Colab reallocates hardware between sessions; a GPU UUID, driver version, "
      "or tool path from a previous session is not evidence about this one.")
    A("")

    # ---- device ----
    A("## Device")
    A("")
    if td.get("available"):
        total_b = td["total_memory_bytes"]
        A("| Field | Value |")
        A("|---|---|")
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

    # ---- resolved tools ----
    A("## Resolved profiling tools")
    A("")
    A("Located by `shutil.which` first, then by globbing known install layouts "
      "and taking the highest version. **Neither binary is assumed to be on "
      "PATH** — on some images `nsys` has no standalone package and ships "
      "inside the Nsight Compute tree. `run_nsys.sh` and `run_ncu.sh` invoke "
      "the absolute paths recorded here.")
    A("")
    A("| Tool | State | Resolved path | Version |")
    A("|---|---|---|---|")
    A(_tool_row(nsys))
    A(_tool_row(ncu))
    A("")
    for tool in (nsys, ncu):
        if tool["candidates"]:
            A(f"`{tool['name']}` candidates considered:")
            A("")
            for c in tool["candidates"]:
                mark = " **<- selected**" if c == tool["path"] else ""
                A(f"- `{c}`{mark}")
            A("")
        elif not tool["found"]:
            A(f"`{tool['name']}` matched none of:")
            A("")
            for p in tool["search_patterns"]:
                A(f"- `{p}`")
            A("")
    A(EFA_NOTE)
    A("")

    # ---- permission tiers ----
    A("## Profiling permission tiers")
    A("")
    A("Tested separately. Passing (a) does **not** imply (b) or (c) — (a) is "
      "CUPTI tracing, while (b) and (c) need hardware counter access. "
      "*Tool not found* and *tool found but blocked* are reported as distinct "
      "states: they have different remedies.")
    A("")
    A("| Tier | Capability | Status | Tool found | Reason |")
    A("|---|---|---|---|---|")
    for key, label, cap in (
        ("nsys_cuda_trace", "(a) nsys CUDA trace", "trace"),
        ("nsys_gpu_metrics", "(b) nsys GPU metrics sampling", "counters"),
        ("ncu_counters", "(c) ncu counter collection", "counters"),
    ):
        r = results[key]
        A(f"| {label} | {cap} | **{r['status']}** | "
          f"{'yes' if r.get('tool_found') else 'NO'} | {r['reason']} |")
    A("")

    for key, label in (
        ("nsys_cuda_trace", "(a) nsys CUDA trace"),
        ("nsys_gpu_metrics", "(b) nsys GPU metrics sampling"),
        ("ncu_counters", "(c) ncu counter collection"),
    ):
        r = results[key]
        A(f"### {label} — {r['status']}")
        A("")
        if r.get("tool_path"):
            A(f"Tool: `{r['tool_path']}` (version `{r.get('tool_version') or 'unknown'}`)")
            A("")
        A(f"```\n{r['command']}\n```")
        A("")
        if r.get("flag_used"):
            A(f"Flag that worked: `{r['flag_used']}`")
            A("")
        if r.get("attempts") and len(r["attempts"]) > 1:
            A("Flag attempts:")
            A("")
            for att in r["attempts"]:
                A(f"- `{att['flag']}` -> {att['status']} ({att['reason']})")
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
              "administrators. Where the NVIDIA kernel module is loaded by the "
              "host rather than the session, "
              "`NVreg_RestrictProfilingToAdminUsers=0` cannot be applied from "
              "inside — treat this as a permanent property of the environment.")
        if not r.get("tool_found"):
            A("")
            A("The tool itself was not located. This is **not** a permission "
              "problem: install it, or add its install layout to the search "
              "patterns in `probe_environment.py`.")
        A("")

    # ---- gpu metric sets ----
    A("## nsys GPU metric sets")
    A("")
    ms = results.get("nsys_metric_sets", {})
    if ms.get("available"):
        A("Output of `--gpu-metrics-set=help`. The **default is General "
          "Metrics**; the set is baked into a capture and cannot be changed "
          "afterwards, so pick it before the real runs. Look for a set with "
          "explicit Tensor-pipe coverage if occupancy of the HMMA pipes matters "
          "at Layer 2.")
        A("")
        if ms.get("sets"):
            A("| Index | Set | Description |")
            A("|---|---|---|")
            for s in ms["sets"]:
                A(f"| {s['index']} | `{s['name']}` | {s['description']} |")
            A("")
        A("Raw:")
        A("")
        A(f"```\n{ms.get('raw', '')}\n```")
    else:
        A(f"Not available: {ms.get('reason', 'command failed')}")
        if ms.get("raw"):
            A("")
            A(f"```\n{ms['raw']}\n```")
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
        A("**All requested metrics are obtainable in this environment.** All "
          "four profiling layers are available.")
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
    A(f"nsys_found={'true' if nsys['found'] else 'false'}")
    A(f"nsys_path={nsys['path'] or ''}")
    A(f"nsys_version={nsys['version'] or ''}")
    A(f"ncu_found={'true' if ncu['found'] else 'false'}")
    A(f"ncu_path={ncu['path'] or ''}")
    A(f"ncu_version={ncu['version'] or ''}")
    A(f"nsys_gpu_metrics_flag={results['nsys_gpu_metrics'].get('flag_used') or ''}")
    A(f"gpu_uuid={nv.get('gpu_uuid', 'UNKNOWN')}")
    A(f"gpu_name={nv.get('gpu_name', td.get('device_name', 'UNKNOWN'))}")
    A(f"total_vram_bytes={td.get('total_memory_bytes', 0)}")
    A(f"total_vram_mb={nv.get('total_vram_mb', td.get('total_memory_mb', 0))}")
    A(f"sm_count={td.get('sm_count', 0)}")
    A(f"compute_capability={td.get('compute_capability', 'UNKNOWN')}")
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
    return {"branch_sha": branch, "upstream_sha": upstream_sha,
            "upstream_ref": upstream_ref}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Probe the GPU host for benchmarking.")
    ap.add_argument("--out", default=ENV_MD, help="path to ENVIRONMENT.md")
    ap.add_argument("--json-out", default=None, help="also write raw results JSON")
    ap.add_argument("--skip-nsys", action="store_true")
    ap.add_argument("--skip-ncu", action="store_true")
    ap.add_argument("--skip-pip-freeze", action="store_true")
    args = ap.parse_args(argv)

    def skipped(tool: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": BLOCKED, "reason": "skipped by flag", "returncode": -1,
            "permission_error": False, "tool_found": tool["found"],
            "tool_path": tool["path"], "tool_version": tool["version"],
            "stdout_tail": "", "stderr_tail": "", "command": "(skipped)",
        }

    print("=" * 72)
    print("BENCHMARK ENVIRONMENT PROBE")
    print("=" * 72)

    nsys = resolve_tool("nsys", NSYS_SEARCH_PATTERNS)
    ncu = resolve_tool("ncu", NCU_SEARCH_PATTERNS)
    for t in (nsys, ncu):
        if t["found"]:
            print(f"[tool]   {t['name']}: {t['path']}  "
                  f"(via {t['source']}, version {t['version'] or 'unknown'})")
        else:
            print(f"[tool]   {t['name']}: NOT FOUND "
                  f"(searched {len(t['search_patterns'])} patterns)")

    with tempfile.TemporaryDirectory() as tmp:
        results: Dict[str, Any] = {
            "utc_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "torch_device": probe_torch_device(),
            "nvml": probe_nvml(),
            "tools": {"nsys": nsys, "ncu": ncu},
        }
        td = results["torch_device"]
        print(f"\n[device] {td.get('device_name', 'NO CUDA DEVICE')}")
        if td.get("available"):
            print(f"[device] total memory: {td['total_memory_bytes']:,} bytes "
                  f"({td['total_memory_gib']} GiB), SMs: {td['sm_count']}, "
                  f"CC {td['compute_capability']}")
            if td["matches_a100_80gb"]:
                print("[device] -> matches A100-SXM4-80GB exactly")
            elif td["matches_a100_40gb"]:
                print("[device] -> matches A100-SXM4-40GB exactly")
            else:
                print("[device] -> matches NEITHER the 80GB nor 40GB reference value")
        print(f"[nvml]   uuid: {results['nvml'].get('gpu_uuid', 'UNKNOWN')}")

        print("\n[tier a] nsys CUDA trace ...")
        results["nsys_cuda_trace"] = (
            skipped(nsys) if args.skip_nsys else probe_nsys_cuda_trace(nsys, tmp)
        )
        print(f"[tier a] {results['nsys_cuda_trace']['status']} — "
              f"{results['nsys_cuda_trace']['reason']}")

        print("[tier b] nsys GPU metrics sampling ...")
        results["nsys_gpu_metrics"] = (
            skipped(nsys) if args.skip_nsys else probe_nsys_gpu_metrics(nsys, tmp)
        )
        flag = results["nsys_gpu_metrics"].get("flag_used")
        print(f"[tier b] {results['nsys_gpu_metrics']['status']} — "
              f"{results['nsys_gpu_metrics']['reason']}"
              + (f"  (flag: {flag})" if flag else ""))

        print("[tier c] ncu counters ...")
        results["ncu_counters"] = (
            skipped(ncu) if args.skip_ncu else probe_ncu(ncu)
        )
        print(f"[tier c] {results['ncu_counters']['status']} — "
              f"{results['ncu_counters']['reason']}")

    print("[sets  ] nsys --gpu-metrics-set=help ...")
    results["nsys_metric_sets"] = (
        {"available": False, "reason": "skipped", "raw": "", "sets": []}
        if args.skip_nsys else probe_nsys_metric_sets(nsys)
    )
    n_sets = len(results["nsys_metric_sets"].get("sets") or [])
    print(f"[sets  ] {n_sets} metric set(s) parsed"
          f"{' (raw output recorded)' if results['nsys_metric_sets'].get('raw') else ''}")

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
    else:
        print("\n  All metrics obtainable. All four profiling layers available.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
