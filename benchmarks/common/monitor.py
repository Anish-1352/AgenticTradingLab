"""20 Hz background sampler for GPU and CPU utilization.

Layer 1 instrumentation. Runs alongside every benchmark run, including the
unprofiled ones, because it is the only layer cheap enough to leave on.

WHAT NVML CAN AND CANNOT MEASURE
--------------------------------
Read this before quoting any number this module produces.

``utilization.gpu`` **is not occupancy.** NVML defines it as the percentage of
the sampling period during which at least one kernel was resident on the
device. A single tiny kernel occupying one SM of 108 pins this field at 100%.
It answers "was the GPU doing anything", not "was the GPU doing much".

Specifically, NVML **cannot** measure:

* achieved occupancy (active warps per SM vs. theoretical maximum)
* Tensor Core / HMMA pipe utilization
* SM-level activity distribution
* memory bandwidth achieved vs. peak

Those require hardware performance counters:

* duration-weighted SM activity across a whole run -> **nsys** (Layer 2)
* exact per-kernel achieved occupancy on a slice -> **ncu** (Layer 4)

Reporting ``utilization.gpu`` as though it were occupancy is the single easiest
way to conclude a GPU is saturated when it is in fact idling between small
memory-bound kernels — which is the expected shape of batch-1 decode.

PEAK VS STEADY-STATE VRAM
-------------------------
Both are reported, and they answer different questions:

* **peak** — the provisioning number. What the deployment must have available
  or it OOMs.
* **steady_state** — the median over post-warmup samples. What the workload
  actually sits at once weights are loaded and allocator caches have settled.

Peak alone overstates the working set (it includes transient allocation
spikes); steady-state alone understates the requirement. v1 reported a single
``torch.cuda.max_memory_allocated()`` figure and it was then read as both.
"""

from __future__ import annotations

import csv
import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["Sample", "ResourceMonitor", "nvml_available"]

DEFAULT_HZ = 20.0
DEFAULT_WARMUP_S = 5.0


def nvml_available() -> bool:
    try:
        import pynvml  # noqa: F401,PLC0415

        return True
    except Exception:
        return False


@dataclass
class Sample:
    t: float                      # perf_counter seconds, run-relative
    wall: float                   # unix epoch seconds
    vram_process_mb: Optional[float]
    vram_device_mb: Optional[float]
    util_gpu_pct: Optional[float]
    util_mem_pct: Optional[float]
    cpu_per_core: List[float] = field(default_factory=list)

    def row(self, n_cores: int) -> List[Any]:
        cores = list(self.cpu_per_core) + [""] * (n_cores - len(self.cpu_per_core))
        return [
            f"{self.t:.4f}", f"{self.wall:.4f}",
            "" if self.vram_process_mb is None else f"{self.vram_process_mb:.2f}",
            "" if self.vram_device_mb is None else f"{self.vram_device_mb:.2f}",
            "" if self.util_gpu_pct is None else f"{self.util_gpu_pct:.1f}",
            "" if self.util_mem_pct is None else f"{self.util_mem_pct:.1f}",
            *cores[:n_cores],
        ]


class ResourceMonitor:
    """Background sampler. Start before the workload, stop after.

    Degrades rather than raises: with no pynvml the GPU columns are empty and
    ``summary()`` reports nulls with ``nvml_available: false``, so a CPU-only
    smoke test still produces a well-formed artifact.
    """

    def __init__(
        self,
        out_csv: Optional[str] = None,
        hz: float = DEFAULT_HZ,
        device_index: int = 0,
        warmup_s: float = DEFAULT_WARMUP_S,
    ) -> None:
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.out_csv = out_csv
        self.interval = 1.0 / hz
        self.hz = hz
        self.device_index = device_index
        self.warmup_s = warmup_s

        self.samples: List[Sample] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

        self._nvml = None
        self._handle = None
        self._psutil = None
        self._n_cores = 0
        self._pid = os.getpid()
        self._vram_source = "none"

    # ---- lifecycle ----

    def _init_backends(self) -> None:
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            self._nvml = None
            self._handle = None

        try:
            import psutil  # noqa: PLC0415

            self._psutil = psutil
            self._n_cores = psutil.cpu_count(logical=True) or 0
            psutil.cpu_percent(percpu=True)  # prime the delta baseline
        except Exception:
            self._psutil = None
            self._n_cores = 0

    def start(self) -> "ResourceMonitor":
        self._init_backends()
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="resource-monitor", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> "ResourceMonitor":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self.out_csv:
            self.write_csv(self.out_csv)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return self

    def __enter__(self) -> "ResourceMonitor":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ---- sampling ----

    def _sample_gpu(self):
        if self._nvml is None or self._handle is None:
            return None, None, None, None
        proc_mb = None
        dev_mb = None
        util_gpu = None
        util_mem = None
        try:
            mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            dev_mb = mem.used / (1024 ** 2)
        except Exception:
            pass
        try:
            # Per-process VRAM: the device-wide figure includes anything else
            # sharing the card, which on a hosted runtime is not always nothing.
            for p in self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                if p.pid == self._pid and getattr(p, "usedGpuMemory", None):
                    proc_mb = p.usedGpuMemory / (1024 ** 2)
                    self._vram_source = "nvml_process"
                    break
        except Exception:
            pass
        if proc_mb is None and dev_mb is not None:
            self._vram_source = "nvml_device_fallback"
        try:
            u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            util_gpu = float(u.gpu)
            util_mem = float(u.memory)
        except Exception:
            pass
        return proc_mb, dev_mb, util_gpu, util_mem

    def _loop(self) -> None:
        i = 0
        while not self._stop.is_set():
            target = self._t0 + i * self.interval
            now = time.perf_counter()
            if target > now:
                # Wait on the event, not sleep(): a stop during the interval
                # returns immediately instead of hanging for up to 50 ms.
                if self._stop.wait(target - now):
                    break
            proc_mb, dev_mb, util_gpu, util_mem = self._sample_gpu()
            cpu: List[float] = []
            if self._psutil is not None:
                try:
                    cpu = list(self._psutil.cpu_percent(percpu=True))
                except Exception:
                    cpu = []
            self.samples.append(
                Sample(
                    t=time.perf_counter() - self._t0,
                    wall=time.time(),
                    vram_process_mb=proc_mb,
                    vram_device_mb=dev_mb,
                    util_gpu_pct=util_gpu,
                    util_mem_pct=util_mem,
                    cpu_per_core=cpu,
                )
            )
            i += 1
            # If sampling fell behind (a long GC pause, a stalled NVML call),
            # resync the schedule rather than sprinting to catch up.
            if time.perf_counter() > self._t0 + (i + 1) * self.interval:
                i = int((time.perf_counter() - self._t0) / self.interval) + 1

    # ---- output ----

    def write_csv(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        n = self._n_cores or (
            max((len(s.cpu_per_core) for s in self.samples), default=0)
        )
        header = [
            "t_rel_s", "wall_epoch_s", "vram_process_mb", "vram_device_mb",
            "util_gpu_pct", "util_mem_pct",
            *[f"cpu{i}_pct" for i in range(n)],
        ]
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for s in self.samples:
                w.writerow(s.row(n))
        return path

    def summary(self) -> Dict[str, Any]:
        def _vals(attr: str) -> List[float]:
            return [
                getattr(s, attr) for s in self.samples if getattr(s, attr) is not None
            ]

        vram = _vals("vram_process_mb") or _vals("vram_device_mb")
        post_warmup = [
            (s.vram_process_mb if s.vram_process_mb is not None else s.vram_device_mb)
            for s in self.samples
            if s.t >= self.warmup_s
        ]
        post_warmup = [v for v in post_warmup if v is not None]

        util = _vals("util_gpu_pct")
        umem = _vals("util_mem_pct")
        cpu_means = [
            sum(s.cpu_per_core) / len(s.cpu_per_core)
            for s in self.samples
            if s.cpu_per_core
        ]

        achieved_hz = None
        if len(self.samples) > 1:
            span = self.samples[-1].t - self.samples[0].t
            if span > 0:
                achieved_hz = (len(self.samples) - 1) / span

        return {
            "nvml_available": self._nvml is not None,
            "vram_source": self._vram_source,
            "sample_count": len(self.samples),
            "requested_hz": self.hz,
            "achieved_hz": achieved_hz,
            "warmup_s": self.warmup_s,
            "vram_peak_mb": max(vram) if vram else None,
            "vram_steady_state_mb": statistics.median(post_warmup) if post_warmup else None,
            "vram_steady_state_sample_count": len(post_warmup),
            "util_gpu_mean_pct": (sum(util) / len(util)) if util else None,
            "util_gpu_max_pct": max(util) if util else None,
            "util_mem_mean_pct": (sum(umem) / len(umem)) if umem else None,
            "cpu_mean_pct_across_cores": (
                sum(cpu_means) / len(cpu_means) if cpu_means else None
            ),
            "cpu_core_count": self._n_cores,
            "caveats": [
                "util_gpu_pct is NVML's 'at least one kernel resident' measure. "
                "It is NOT achieved occupancy and must not be reported as such.",
                "NVML cannot measure Tensor Core utilization or SM activity. "
                "Use nsys (Layer 2) and ncu (Layer 4).",
                "vram_peak_mb is the provisioning number; "
                "vram_steady_state_mb is the post-warmup median working set.",
            ],
        }
