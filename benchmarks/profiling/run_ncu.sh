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
#
# ncu IS NOT ASSUMED TO BE ON PATH — it commonly lives at
# /usr/local/cuda/bin/ncu or under /opt/nvidia/nsight-compute/<version>/. The
# absolute path is read from ENVIRONMENT.md, where probe_environment.py recorded
# it after resolving it by search. Versions are never hardcoded here.
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

# ---- gate on the probe -------------------------------------------------------
if [ ! -f "${ENV_MD}" ]; then
  cat >&2 <<EOF
ERROR: ${ENV_MD} not found.

Run the probe first — it resolves the ncu path (ncu is often NOT on PATH) and
determines whether counters can be collected at all:

  python ${BENCH_ROOT}/probe_environment.py
EOF
  exit 2
fi

NCU_STATUS="$(probe_value ncu_counters || true)"
NCU_FOUND="$(probe_value ncu_found || true)"

if [ "${NCU_STATUS}" != "OBTAINABLE" ]; then
  if [ "${NCU_FOUND}" = "false" ]; then
    cat >&2 <<EOF
Layer 4 (ncu) is unavailable: THE TOOL WAS NOT FOUND.

This is an install/path problem, not a permission problem — a different fix.
ncu commonly lives at one of:

  /usr/local/cuda/bin/ncu
  /opt/nvidia/nsight-compute/<version>/ncu

Install Nsight Compute, or add the layout to NCU_SEARCH_PATTERNS in
probe_environment.py, then re-run the probe.
EOF
    exit 4
  fi

  cat >&2 <<EOF
Layer 4 (ncu) is NOT available on this host.

  probe result: ncu_counters=${NCU_STATUS:-UNKNOWN}
  ncu found:    ${NCU_FOUND:-unknown}
  see:          ${ENV_MD}

If the reason is ERR_NVGPUCTRPERM, the driver restricts performance counters to
administrators. Where the NVIDIA kernel module is loaded by the host rather than
the session, NVreg_RestrictProfilingToAdminUsers=0 cannot be set from inside.
That is a property of the environment, not a misconfiguration you can fix.

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

# ---- resolve the binary ------------------------------------------------------
NCU_BIN="$(probe_value ncu_path || true)"
NCU_VERSION="$(probe_value ncu_version || true)"

if [ -z "${NCU_BIN}" ]; then
  if command -v ncu >/dev/null 2>&1; then
    NCU_BIN="$(command -v ncu)"
    echo "WARNING: ${ENV_MD} records no ncu_path; falling back to PATH (${NCU_BIN})." >&2
    echo "         Re-run probe_environment.py to refresh it." >&2
  else
    echo "ERROR: ${ENV_MD} records no ncu_path and ncu is not on PATH." >&2
    echo "       Re-run: python ${BENCH_ROOT}/probe_environment.py" >&2
    exit 127
  fi
fi

if [ ! -x "${NCU_BIN}" ]; then
  echo "ERROR: recorded ncu path is not executable: ${NCU_BIN}" >&2
  echo "       The image may have changed. Re-run probe_environment.py." >&2
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

echo "[ncu] binary       : ${NCU_BIN} (version ${NCU_VERSION:-unknown})"
echo "[ncu] run_id       : ${RUN_ID}"
echo "[ncu] export       : ${OUT_BASE}.ncu-rep"
echo "[ncu] launch-count : ${LAUNCH_COUNT}"
echo "[ncu] NOTE: forcing --concurrency 1 --n-requests 3 (kernel replay)."
echo

"${NCU_BIN}" \
  --metrics "${METRICS}" \
  --launch-count "${LAUNCH_COUNT}" \
  --export "${OUT_BASE}" \
  --force-overwrite \
  python "$@" --concurrency 1 --n-requests 3

echo
echo "[ncu] done: ${OUT_BASE}.ncu-rep"
echo "[ncu] to CSV: ${NCU_BIN} --import ${OUT_BASE}.ncu-rep --csv --page raw > ${OUT_BASE}.csv"
echo
echo "Report alongside the nsys (Layer 2) numbers, not instead of them:"
echo "  nsys = duration-weighted SM activity across the whole run"
echo "  ncu  = exact per-kernel achieved occupancy on this slice"
