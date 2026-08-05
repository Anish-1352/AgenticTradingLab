#!/usr/bin/env bash
#
# Layer 4 — Nsight Compute, short slice.
#
#   ./run_ncu.sh <runner-script> [args...]
#
# NSYS AND NCU ANSWER DIFFERENT QUESTIONS. Report both; neither substitutes.
#
#   nsys (Layer 2) samples SM activity over time across the WHOLE run. It gives
#   a duration-weighted picture: how busy the device was, when, and how that
#   tracked the workload's phases. It cannot tell you how efficiently any
#   individual kernel used the SMs it occupied.
#
#   ncu (Layer 4) collects exact hardware counters PER KERNEL, replaying each
#   kernel many times to gather every metric. It gives achieved occupancy and
#   tensor-pipe utilization for the kernels it profiles — but only for a small
#   slice, and with the timeline destroyed by replay.
#
# "The GPU was busy 95% of the time" (nsys) and "those kernels ran at 8%
# achieved occupancy" (ncu) are both true simultaneously, and together they are
# the actual finding. Either alone is misleading.
#
# BECAUSE OF REPLAY, THIS RUNS AGAINST concurrency=1 AND n-requests=3 ONLY.
# Pointing ncu at the 105-request sweep would replay every kernel of every
# request and take hours to days.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${HERE}/.." && pwd)"
ENV_MD="${BENCH_ROOT}/ENVIRONMENT.md"
TRACES_DIR="${TRACES_DIR:-${BENCH_ROOT}/traces}"

if [ "$#" -lt 1 ]; then
  echo "usage: $(basename "$0") <runner-script> [args...]" >&2
  exit 2
fi

probe_value() {
  local key="$1"
  [ -f "${ENV_MD}" ] || return 0
  awk -v key="${key}" '
    /PROBE_RESULTS_BEGIN/ { inblock = 1; next }
    /PROBE_RESULTS_END/   { inblock = 0 }
    inblock && index($0, key "=") == 1 {
      sub(/^[^=]*=/, "", $0); print $0; exit
    }
  ' "${ENV_MD}"
}

# ---- gate on the probe ---------------------------------------------------
if [ ! -f "${ENV_MD}" ]; then
  cat >&2 <<EOF
ERROR: ${ENV_MD} not found.

Run the probe first — it determines whether ncu can collect counters at all:

  python ${BENCH_ROOT}/probe_environment.py
EOF
  exit 2
fi

NCU_STATUS="$(probe_value ncu_counters || true)"

if [ "${NCU_STATUS}" != "OBTAINABLE" ]; then
  cat >&2 <<EOF
Layer 4 (ncu) is NOT available on this host.

  probe result: ncu_counters=${NCU_STATUS:-UNKNOWN}
  see:          ${ENV_MD}

If the reason is ERR_NVGPUCTRPERM, the driver restricts performance counters to
administrators. On Colab the NVIDIA kernel module is loaded by the host, so
NVreg_RestrictProfilingToAdminUsers=0 cannot be set from inside the session.
This is a property of the environment, not a misconfiguration you can fix.

Consequences to carry into the write-up — do NOT substitute a proxy:
  * achieved occupancy        : NOT MEASURABLE here
  * Tensor Core utilization   : NOT MEASURABLE here
  * memory bandwidth achieved : NOT MEASURABLE here

NVML utilization.gpu is kernel residency, not occupancy. A kernel name matching
's16816gemm' shows a tensor-core GEMM ran, not how well it used the pipes.
Report these as unavailable rather than approximating them.
EOF
  exit 3
fi

if ! command -v ncu >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: ncu not found on PATH, though the probe recorded it as OBTAINABLE.
Re-run probe_environment.py — the environment has changed.

Nsight Compute installs from the NVIDIA apt repository, NOT pip:
  apt-get install -y nsight-compute
EOF
  exit 127
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-B-ncu}"
OUT_BASE="${TRACES_DIR}/${RUN_ID}_ncu"
LAUNCH_COUNT="${LAUNCH_COUNT:-40}"
mkdir -p "${TRACES_DIR}"

# sm__warps_active...                        -> achieved occupancy
# sm__pipe_tensor_op_hmma_cycles_active...   -> Tensor Core (HMMA) pipe utilization
# sm__throughput...                          -> overall SM throughput vs peak
METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,sm__throughput.avg.pct_of_peak_sustained_elapsed"

echo "[ncu] run_id       : ${RUN_ID}"
echo "[ncu] export       : ${OUT_BASE}.ncu-rep"
echo "[ncu] launch-count : ${LAUNCH_COUNT}"
echo "[ncu] NOTE: forcing --concurrency 1 --n-requests 3 (kernel replay)."
echo

ncu \
  --metrics "${METRICS}" \
  --launch-count "${LAUNCH_COUNT}" \
  --export "${OUT_BASE}" \
  --force-overwrite \
  python "$@" --concurrency 1 --n-requests 3

echo
echo "[ncu] done: ${OUT_BASE}.ncu-rep"
echo "[ncu] to CSV: ncu --import ${OUT_BASE}.ncu-rep --csv --page raw > ${OUT_BASE}.csv"
echo
echo "Report alongside the nsys (Layer 2) numbers, not instead of them:"
echo "  nsys = duration-weighted SM activity across the whole run"
echo "  ncu  = exact per-kernel achieved occupancy on this slice"
