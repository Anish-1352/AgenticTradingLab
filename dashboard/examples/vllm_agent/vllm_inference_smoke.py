#!/usr/bin/env python3
"""Bounded GPU inference smoke test for the ATL vLLM adapter (Phase 28-infra Stage 1).

Sends a fixed set of synthetic ATL steps (none from real market data, chosen
before any model output was seen) through ``VLLMTradingAgent`` against a live
vLLM server, and writes one JSON report of measured facts only:

  - server: /health, /version, /v1/models identity vs. the configured model
  - startup: seconds to healthy, when this script launches the server (--launch)
  - per request: status, reason, tokens, latency, response hash (agent records)
  - TTFT: mean time-to-first-token from vLLM's own Prometheus /metrics delta
  - GPU: peak memory.used polled from nvidia-smi while requests run
  - a free-form probe (no JSON schema) to exercise the strict parser on raw text

Scenarios lean HOLD / BUY / SELL by construction; what the model actually does
is recorded, never steered. Nothing here tunes the prompt.

    PYTHONPATH=packaging/agentictrading/src python3 dashboard/examples/vllm_agent/vllm_inference_smoke.py \\
        --out /path/outside/git/inference-smoke --launch --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentictrading import Step
from agentictrading.integrations.vllm import (
    ChatTransportError,
    DecisionRejected,
    VLLMChatClient,
    VLLMConfig,
    VLLMTradingAgent,
    build_messages,
    parse_decision,
    summarize_records,
)

HERE = Path(__file__).resolve().parent
DJIA_30 = ["AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS", "HD", "HON", "IBM",
           "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"]


def _bar(ts: str, close: float) -> Dict[str, Any]:
    return {"timestamp": ts, "open": close * 0.998, "high": close * 1.004, "low": close * 0.995,
            "close": close, "volume": 1_500_000.0}


def _features(close: float, rsi: float, trend: float) -> Dict[str, Any]:
    return {"price": close, "rsi": rsi, "macd": trend, "macd_signal": trend * 0.5,
            "sma20": close * (1 - trend / 100), "sma50": close * (1 - trend / 50),
            "bb_upper": close * 1.03, "bb_lower": close * 0.97}


def synthetic_step(name: str, symbols: List[str], *, rsi: float, trend: float,
                   position: Optional[Dict[str, Any]] = None, cash: float = 1000.0) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    ts = now.isoformat()
    closes = {s: 100.0 + 7.0 * i for i, s in enumerate(symbols)}
    positions = [position] if position else []
    equity = cash + sum(p["market_value"] for p in positions)
    return {
        "run_id": "inference-smoke", "step_id": name, "sequence": 0, "timestamp": ts,
        "deadline_at": (now + timedelta(seconds=60)).isoformat(), "status": "awaiting_decision",
        "observation": {
            "market": {"bars": {s: _bar(ts, c) for s, c in closes.items()},
                       "features": {s: _features(c, rsi, trend) for s, c in closes.items()}, "events": []},
            "portfolio": {"cash": cash, "equity": equity, "positions": positions},
        },
        "constraints": {"allowed_symbols": symbols, "allow_short": False, "max_position_weight": 0.25,
                        "max_orders": 10},
    }


def scenarios() -> List[Dict[str, Any]]:
    held = {"symbol": "AAPL", "quantity": 2, "entry_price": 80.0, "current_price": 100.0,
            "market_value": 200.0, "unrealized_pnl_pct": 0.25}
    return [
        synthetic_step("short_neutral_hold", ["AAPL"], rsi=50.0, trend=0.0),
        synthetic_step("oversold_buy", ["AAPL"], rsi=22.0, trend=-1.5),
        synthetic_step("overbought_sell", ["AAPL"], rsi=84.0, trend=2.5, position=held, cash=800.0),
        synthetic_step("long_djia30", DJIA_30, rsi=55.0, trend=0.4),
    ]


class GpuSampler:
    """Polls nvidia-smi; reports nothing at all when there is no NVIDIA GPU."""

    QUERY = ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"]

    def __init__(self, interval: float = 0.5) -> None:
        self.available = shutil.which("nvidia-smi") is not None
        self.interval = interval
        self.peak_mib: Dict[str, float] = {}
        self.info: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _read(self) -> List[List[str]]:
        out = subprocess.run(self.QUERY, capture_output=True, text=True, timeout=10, check=True).stdout
        return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines()]

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for idx, _, _, used, _, _ in self._read():
                    self.peak_mib[idx] = max(self.peak_mib.get(idx, 0.0), float(used))
            except (subprocess.SubprocessError, ValueError, OSError):
                pass  # a missed sample only lowers the observed peak; never invents one
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self.available:
            self.info = [dict(zip(("index", "name", "driver_version", "memory_used_mib", "memory_total_mib",
                                   "utilization_pct"), row)) for row in self._read()]
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)


def scrape_metrics(base_url: str, timeout: float = 5.0) -> Dict[str, float]:
    import urllib.request

    root = re.sub(r"/v1/?$", "", base_url)
    try:
        with urllib.request.urlopen(root + "/metrics", timeout=timeout) as resp:
            text = resp.read().decode()
    except OSError:
        return {}
    out: Dict[str, float] = {}
    for name in ("vllm:time_to_first_token_seconds", "vllm:e2e_request_latency_seconds"):
        for suffix in ("_sum", "_count"):
            total = sum(float(m.group(1)) for m in re.finditer(rf"^{re.escape(name + suffix)}(?:{{[^}}]*}})? (\S+)$",
                                                               text, re.M))
            out[name + suffix] = total
    return out


def mean_delta(before: Dict[str, float], after: Dict[str, float], name: str) -> Optional[float]:
    if not before or not after:
        return None
    count = after.get(name + "_count", 0) - before.get(name + "_count", 0)
    return (after.get(name + "_sum", 0) - before.get(name + "_sum", 0)) / count if count > 0 else None


def wait_healthy(client: VLLMChatClient, deadline_s: float, proc: Optional[subprocess.Popen]) -> Optional[float]:
    start = time.monotonic()
    while time.monotonic() - start < deadline_s:
        if proc is not None and proc.poll() is not None:
            return None
        try:
            if client.health(timeout=2):
                return time.monotonic() - start
        except ChatTransportError:
            pass
        time.sleep(1.0)
    return None


def free_form_probe(client: VLLMChatClient, step: Step) -> Dict[str, Any]:
    """Same prompt without the JSON schema: whatever comes back goes through the strict parser."""
    body = client.request_body(build_messages(step))
    body.pop("response_format")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if client.config.api_key:
        headers["Authorization"] = f"Bearer {client.config.api_key}"
    from agentictrading.integrations.vllm import urllib_transport

    status, raw = urllib_transport("POST", client.config.base_url + "/chat/completions",
                                   json.dumps(body).encode(), headers, client.config.request_timeout)
    content = json.loads(raw)["choices"][0]["message"].get("content") if status == 200 else None
    try:
        parse_decision(content, step)
        verdict, reason = "valid", None
    except DecisionRejected as exc:
        verdict, reason = "rejected", exc.reason
    return {"http_status": status, "verdict": verdict, "reason": reason, "content_chars": len(content or "")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="report directory (keep it outside git)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--launch", action="store_true", help="start serve_vllm.sh, time startup, stop it at the end")
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--record-freeze", action="store_true", help="include `pip freeze` of this interpreter")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = VLLMConfig.from_env()
    client = VLLMChatClient(config)
    report: Dict[str, Any] = {"started_at": datetime.now(timezone.utc).isoformat(), "config": config.public_dict(),
                              "host": {"platform": platform.platform(), "python": sys.version.split()[0]}}
    gpu = GpuSampler()
    report["gpu_available"] = gpu.available
    proc = None
    if args.launch:
        log = open(args.out / "vllm-server.log", "w")
        proc = subprocess.Popen(["bash", str(HERE / "serve_vllm.sh")], stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    try:
        gpu.start()
        report["gpu"] = gpu.info
        startup = wait_healthy(client, args.startup_timeout if proc else 10.0, proc)
        report["server_healthy"] = startup is not None
        report["startup_seconds"] = startup if proc else None
        if startup is None:
            report["error"] = "server never became healthy" + (f" (exit {proc.poll()})" if proc else "")
            return _finish(args.out, report, 2)
        report["served_models"] = client.list_models(timeout=10)
        report["model_identity_ok"] = config.model_id in report["served_models"]
        try:
            import urllib.request

            with urllib.request.urlopen(re.sub(r"/v1/?$", "", config.base_url) + "/version", timeout=5) as r:
                report["vllm_version"] = json.loads(r.read()).get("version")
        except (OSError, ValueError):
            report["vllm_version"] = None

        agent = VLLMTradingAgent(client)
        before = scrape_metrics(config.base_url)
        hashes: Dict[str, List[Optional[str]]] = {}
        for _ in range(args.repeats):
            for payload in scenarios():
                step = Step.from_dict(payload)
                agent.on_step(step)
                agent.decide(step.observation)
                hashes.setdefault(step.step_id, []).append(agent.records[-1].response_sha256)
        after = scrape_metrics(config.base_url)
        report["records"] = [r.to_dict() for r in agent.records]
        report["summary"] = summarize_records(agent.records)
        report["by_scenario"] = {
            name: {"actions": [r.action for r in agent.records if r.step_id == name],
                   "reasons": [r.reason for r in agent.records if r.step_id == name],
                   "prompt_tokens": next((r.prompt_tokens for r in agent.records if r.step_id == name), None),
                   "identical_replies_across_repeats": len(set(h)) == 1}
            for name, h in hashes.items()}
        report["ttft_mean_seconds"] = mean_delta(before, after, "vllm:time_to_first_token_seconds")
        report["server_e2e_mean_seconds"] = mean_delta(before, after, "vllm:e2e_request_latency_seconds")
        report["free_form_probe"] = free_form_probe(client, Step.from_dict(scenarios()[0]))
    finally:
        gpu.stop()
        report["gpu_peak_memory_used_mib"] = gpu.peak_mib or None
        if args.record_freeze:
            report["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                                  text=True).stdout.splitlines()
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    return _finish(args.out, report, 0)


def _finish(out: Path, report: Dict[str, Any], code: int) -> int:
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out / "inference-smoke.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: report.get(k) for k in ("server_healthy", "model_identity_ok", "startup_seconds",
                                                 "summary", "ttft_mean_seconds", "gpu_peak_memory_used_mib")},
                     indent=1, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
