#!/usr/bin/env bash
#
# Layer 2 — Nsight Systems, full workload.
#
#   ./run_nsys.sh <runner-script> [args...]
#
# Wraps a runner in `nsys profile` and then reduces the capture to CSV with
# `nsys stats`, so the numbers land in files that can be read, diffed, and
# committed rather than requiring the Nsight GUI.
#
# LAYER 2 IS ITS OWN RUN. Do not pass --profile to the runner here: nsys and
# torch.profiler both subscribe to CUPTI, and running them together produces
# either an error or a silently truncated capture. Layer 3 is a separate
# invocation of the runner with --profile and no nsys.
#
# The --gpu-metrics-device flag is appended ONLY if probe_environment.py marked
# tier (b) OBTAINABLE. It is parsed out of ENVIRONMENT.md rather than hardcoded,
# because on a counter-restricted host adding that flag makes nsys fail
# outright — turning "we lack SM sampling" into "we have no Layer 2 at all".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${HERE}/.." && pwd)"
ENV_MD="${BENCH_ROOT}/ENVIRONMENT.md"
TRACES_DIR="${TRACES_DIR:-${BENCH_ROOT}/traces}"
RESULTS_DIR="${RESULTS_DIR:-${BENCH_ROOT}/results}"

if [ "$#" -lt 1 ]; then
  echo "usage: $(basename "$0") <runner-script> [args...]" >&2
  echo "  e.g. $(basename "$0") ${BENCH_ROOT}/runners/bench_hf_baseline.py --concurrency 15" >&2
  exit 2
fi

if ! command -v nsys >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: nsys not found on PATH.

Nsight Systems comes from the NVIDIA apt repository, NOT from pip. On Colab:

  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  dpkg -i cuda-keyring_1.1-1_all.deb
  apt-get update -qq
  apt-get install -y nsight-systems-cli

Then re-run. See benchmarks/COLAB.md.
EOF
  exit 127
fi

# ---- probe results -------------------------------------------------------
probe_value() {
  # Reads KEY=VALUE from the machine-readable block written by
  # probe_environment.py. Absent file or absent key -> empty.
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

GPU_METRICS_STATUS="$(probe_value nsys_gpu_metrics || true)"
CUDA_TRACE_STATUS="$(probe_value nsys_cuda_trace || true)"

if [ ! -f "${ENV_MD}" ]; then
  echo "WARNING: ${ENV_MD} not found — run probe_environment.py first." >&2
  echo "         Proceeding WITHOUT --gpu-metrics-device (the safe default)." >&2
elif [ -z "${CUDA_TRACE_STATUS}" ]; then
  echo "WARNING: no machine-readable probe block in ${ENV_MD}." >&2
  echo "         Re-run probe_environment.py. Proceeding without GPU metrics." >&2
elif [ "${CUDA_TRACE_STATUS}" != "OBTAINABLE" ]; then
  echo "ERROR: probe recorded nsys_cuda_trace=${CUDA_TRACE_STATUS}." >&2
  echo "       Layer 2 is not available on this host. See ${ENV_MD}." >&2
  exit 3
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-B-nsys}"
OUT_BASE="${TRACES_DIR}/${RUN_ID}_nsys"
mkdir -p "${TRACES_DIR}" "${RESULTS_DIR}"

NSYS_ARGS=(
  profile
  --trace=cuda,osrt,nvtx
  --output="${OUT_BASE}"
  --force-overwrite=true
)

if [ "${GPU_METRICS_STATUS}" = "OBTAINABLE" ]; then
  echo "[nsys] tier (b) OBTAINABLE — enabling --gpu-metrics-device=0"
  NSYS_ARGS+=(--gpu-metrics-device=0)
else
  echo "[nsys] tier (b) = ${GPU_METRICS_STATUS:-UNKNOWN} — omitting --gpu-metrics-device"
  echo "[nsys] SM activity sampling will NOT be in this capture."
fi

echo "[nsys] run_id : ${RUN_ID}"
echo "[nsys] output : ${OUT_BASE}.nsys-rep"
echo "[nsys] command: nsys ${NSYS_ARGS[*]} python $*"
echo

nsys "${NSYS_ARGS[@]}" python "$@"

REP="${OUT_BASE}.nsys-rep"
if [ ! -f "${REP}" ]; then
  echo "ERROR: expected ${REP} was not produced." >&2
  exit 4
fi

# ---- reduce to CSV -------------------------------------------------------
# gpukernsum : per-kernel total/avg GPU time -> which kernels dominate
# cudaapisum : per-CUDA-API total CPU time  -> launch/schedule/wait cost
for report in gpukernsum cudaapisum; do
  csv="${RESULTS_DIR}/${RUN_ID}_${report}.csv"
  echo "[nsys] stats --report ${report} -> ${csv}"
  if ! nsys stats --report "${report}" --format csv --output - "${REP}" > "${csv}" 2>"${csv}.err"; then
    echo "WARNING: nsys stats ${report} failed; see ${csv}.err" >&2
  else
    rm -f "${csv}.err"
  fi
done

cat <<EOF

[nsys] done.
  trace : ${REP}
  csv   : ${RESULTS_DIR}/${RUN_ID}_gpukernsum.csv
          ${RESULTS_DIR}/${RUN_ID}_cudaapisum.csv

Reminder: this capture is Layer 2. Run Layer 3 (torch.profiler) as a SEPARATE
invocation — CUPTI cannot serve both at once.
EOF
