"""Run manifest emission. See ../RUN_MANIFEST_SCHEMA.md.

Every run emits one manifest. A number without a manifest does not go in the
write-up — that is the rule this module exists to make cheap to follow.

Collection is best-effort and never fatal: a field that cannot be read becomes
``null`` and the reason is appended to ``collection_errors``, with
``manifest_complete: false``. A run that produced real measurements must not be
thrown away because NVML hiccuped, but a manifest with holes must also never be
mistaken for a complete one.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

__all__ = ["Manifest", "build_manifest", "make_run_id", "sha256_file", "sha256_text"]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Tried in order; the one that resolves is recorded in `upstream_ref`. A fork
# clone commonly has no `upstream` remote, and silently substituting
# origin/main without saying so would misrepresent what the run was pinned to.
UPSTREAM_REF_CANDIDATES = ("upstream/main", "origin/main", "main")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def _branch_sha(errors: List[str]) -> Optional[str]:
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        errors.append("branch_sha: git rev-parse HEAD failed")
        return None
    dirty = _git("status", "--porcelain")
    if dirty:
        # Schema: a dirty tree is marked and its results are non-citable.
        return f"{sha}-dirty"
    return sha


def _upstream_sha(errors: List[str]) -> tuple:
    for ref in UPSTREAM_REF_CANDIDATES:
        sha = _git("rev-parse", ref)
        if sha:
            return sha, ref
    errors.append(
        "upstream_sha: none of " + ", ".join(UPSTREAM_REF_CANDIDATES) + " resolved"
    )
    return None, None


def _pip_freeze(errors: List[str]) -> tuple:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=180,
        )
        if out.returncode != 0:
            errors.append("pip_freeze: pip freeze returned non-zero")
            return None, None
        text = out.stdout
        return sha256_text(text), text
    except Exception as exc:
        errors.append(f"pip_freeze: {exc}")
        return None, None


def _gpu_info(errors: List[str], device_index: int = 0) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "gpu_name": None,
        "gpu_uuid": None,
        "total_vram_mb": None,
        "driver_version": None,
        "cuda_version": None,
    }
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        name = pynvml.nvmlDeviceGetName(h)
        info["gpu_name"] = name.decode() if isinstance(name, bytes) else str(name)

        uuid = pynvml.nvmlDeviceGetUUID(h)
        info["gpu_uuid"] = uuid.decode() if isinstance(uuid, bytes) else str(uuid)

        info["total_vram_mb"] = int(
            pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 ** 2)
        )
        drv = pynvml.nvmlSystemGetDriverVersion()
        info["driver_version"] = drv.decode() if isinstance(drv, bytes) else str(drv)
        pynvml.nvmlShutdown()
    except Exception as exc:
        errors.append(f"nvml: {exc}")

    try:
        import torch  # noqa: PLC0415

        # The CUDA the process actually linked, not what nvidia-smi advertises
        # as the driver's maximum supported version.
        info["cuda_version"] = torch.version.cuda
    except Exception as exc:
        errors.append(f"torch_cuda_version: {exc}")

    return info


def make_run_id(arm: str, concurrency: int, branch_sha: Optional[str] = None) -> str:
    """``<utc_compact>-<arm>-c<concurrency>-<short_hash>`` per the schema."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = (branch_sha or "nogit")[:6]
    return f"{ts}-{arm}-c{concurrency}-{short}"


@dataclass
class Manifest:
    run_id: str
    utc_timestamp: str
    arm: str
    concurrency: int

    upstream_sha: Optional[str]
    upstream_ref: Optional[str]
    branch_sha: Optional[str]

    gpu_name: Optional[str]
    gpu_uuid: Optional[str]
    total_vram_mb: Optional[int]
    driver_version: Optional[str]
    cuda_version: Optional[str]

    pip_freeze_sha256: Optional[str]
    config_sha256: Optional[str]
    fixture_sha256: Optional[str]

    # Controlled variables that must travel with every number.
    max_new_tokens: Optional[int]
    # 1 = app timing + NVML, 2 = nsys, 3 = torch.profiler, 4 = ncu.
    profiling_layer: int = 1
    # Fraction of run wall time inside the profiler's active window. null when
    # profiling_layer == 1. v1's trace covered a section that was not the
    # workload at all; this field makes the captured window explicit.
    profiled_window_fraction: Optional[float] = None

    trace_url: Optional[str] = None
    trace_sha256: Optional[str] = None

    fixture_name: Optional[str] = None
    context_tokens: Optional[int] = None
    model: Optional[str] = None

    manifest_complete: bool = True
    collection_errors: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{self.run_id}_manifest.json")
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path


def build_manifest(
    arm: str,
    concurrency: int,
    config_path: Optional[str] = None,
    config_resolved: Optional[Dict[str, Any]] = None,
    fixture_sha256: Optional[str] = None,
    fixture_name: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    profiling_layer: int = 1,
    profiled_window_fraction: Optional[float] = None,
    run_id: Optional[str] = None,
    out_dir: Optional[str] = None,
    context_tokens: Optional[int] = None,
    model: Optional[str] = None,
    collect_pip_freeze: bool = True,
    device_index: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Manifest:
    """Collect everything and return a Manifest.

    ``config_resolved`` is hashed in preference to ``config_path``: the schema
    requires the hash of the config *after* CLI overrides are applied, since
    that is what the run actually used. Hashing the on-disk template would make
    two runs with different overrides look identical.
    """
    errors: List[str] = []

    branch_sha = _branch_sha(errors)
    upstream_sha, upstream_ref = _upstream_sha(errors)
    gpu = _gpu_info(errors, device_index=device_index)

    pip_sha, pip_text = (None, None)
    if collect_pip_freeze:
        pip_sha, pip_text = _pip_freeze(errors)

    if config_resolved is not None:
        config_sha = sha256_text(
            json.dumps(config_resolved, sort_keys=True, separators=(",", ":"))
        )
    elif config_path:
        config_sha = sha256_file(config_path)
        if config_sha is None:
            errors.append(f"config_sha256: could not read {config_path}")
    else:
        config_sha = None
        errors.append("config_sha256: no config supplied")

    if fixture_sha256 is None:
        errors.append("fixture_sha256: not supplied")

    rid = run_id or make_run_id(arm, concurrency, branch_sha)

    man = Manifest(
        run_id=rid,
        utc_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        arm=arm,
        concurrency=concurrency,
        upstream_sha=upstream_sha,
        upstream_ref=upstream_ref,
        branch_sha=branch_sha,
        gpu_name=gpu["gpu_name"],
        gpu_uuid=gpu["gpu_uuid"],
        total_vram_mb=gpu["total_vram_mb"],
        driver_version=gpu["driver_version"],
        cuda_version=gpu["cuda_version"],
        pip_freeze_sha256=pip_sha,
        config_sha256=config_sha,
        fixture_sha256=fixture_sha256,
        max_new_tokens=max_new_tokens,
        profiling_layer=profiling_layer,
        profiled_window_fraction=profiled_window_fraction,
        fixture_name=fixture_name,
        context_tokens=context_tokens,
        model=model,
        collection_errors=errors,
        manifest_complete=not errors,
        extra=extra or {},
    )

    if out_dir:
        man.write(out_dir)
        if pip_text is not None:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{rid}_pip_freeze.txt"), "w") as fh:
                fh.write(pip_text)

    return man
