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
# NSYS IS NOT ASSUMED TO BE ON PATH. On some images it has no standalone
# package and ships bundled inside the Nsight Compute tree, e.g.
# /opt/nvidia/nsight-compute/<version>/host/target-linux-x64/nsys. The absolute
# path is read from ENVIRONMENT.md, where probe_environment.py recorded it
# after resolving it by search. Versions are never hardcoded here — the image
# changes between sessions.
#
# The GPU-metrics flag is likewise read from the probe rather than guessed:
# --gpu-metrics-device (singular) is deprecated in nsys 2025.x, while the plural
# --gpu-metrics-devices is unrecognised by older builds. The probe tried both
# and recorded which one this binary accepts.
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

# ---- fail closed without a probe --------------------------------------------
if [ ! -f "${ENV_MD}" ]; then
  cat >&2 <<EOF
ERROR: ${ENV_MD} not found.

Run the probe first — it resolves the nsys path (nsys is often NOT on PATH) and
determines which capabilities this host permits:

  python ${BENCH_ROOT}/probe_environment.py
EOF
  exit 2
fi

CUDA_TRACE_STATUS="$(probe_value nsys_cuda_trace || true)"
if [ -z "${CUDA_TRACE_STATUS}" ]; then
  echo "ERROR: no machine-readable probe block in ${ENV_MD}." >&2
  echo "       Re-run: python ${BENCH_ROOT}/probe_environment.py" >&2
  exit 2
fi

if [ "${CUDA_TRACE_STATUS}" != "OBTAINABLE" ]; then
  echo "ERROR: probe recorded nsys_cuda_trace=${CUDA_TRACE_STATUS}." >&2
  echo "       Layer 2 is not available on this host. See ${ENV_MD}." >&2
  exit 3
fi

# ---- resolve the binary ------------------------------------------------------
NSYS_BIN="$(probe_value nsys_path || true)"
NSYS_VERSION="$(probe_value nsys_version || true)"

if [ -z "${NSYS_BIN}" ]; then
  # Probe said the tier works but recorded no path: stale ENVIRONMENT.md from
  # before path resolution existed. Fall back to PATH, but say so.
  if command -v nsys >/dev/null 2>&1; then
    NSYS_BIN="$(command -v nsys)"
    echo "WARNING: ${ENV_MD} records no nsys_path; falling back to PATH (${NSYS_BIN})." >&2
    echo "         Re-run probe_environment.py to refresh it." >&2
  else
    cat >&2 <<EOF
ERROR: ${ENV_MD} records no nsys_path and nsys is not on PATH.

nsys is frequently NOT separately installable. It ships inside the Nsight
Compute tree, e.g.:

  /opt/nvidia/nsight-compute/<version>/host/target-linux-x64/nsys

Re-run the probe so it can search for it:

  python ${BENCH_ROOT}/probe_environment.py

If it still is not found, add the install layout to NSYS_SEARCH_PATTERNS in
probe_environment.py.
EOF
    exit 127
  fi
fi

if [ ! -x "${NSYS_BIN}" ]; then
  echo "ERROR: recorded nsys path is not executable: ${NSYS_BIN}" >&2
  echo "       The image may have changed. Re-run probe_environment.py." >&2
  exit 127
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

# ---- GPU metrics sampling ----------------------------------------------------
GPU_METRICS_STATUS="$(probe_value nsys_gpu_metrics || true)"
GPU_METRICS_FLAG="$(probe_value nsys_gpu_metrics_flag || true)"

if [ "${GPU_METRICS_STATUS}" = "OBTAINABLE" ]; then
  # Prefer whatever the probe actually got to work. Default to the plural form
  # when the probe predates this field: singular is deprecated in nsys 2025.x.
  FLAG="${GPU_METRICS_FLAG:---gpu-metrics-devices}"
  echo "[nsys] tier (b) OBTAINABLE — enabling ${FLAG}=0"
  NSYS_ARGS+=("${FLAG}=0")
  if [ -n "${GPU_METRICS_SET:-}" ]; then
    echo "[nsys] using --gpu-metrics-set=${GPU_METRICS_SET}"
    NSYS_ARGS+=("--gpu-metrics-set=${GPU_METRICS_SET}")
  else
    echo "[nsys] --gpu-metrics-set not set; nsys default (General Metrics) applies."
    echo "[nsys] Available sets are listed in ${ENV_MD}."
  fi
else
  echo "[nsys] tier (b) = ${GPU_METRICS_STATUS:-UNKNOWN} — omitting GPU metrics flag"
  echo "[nsys] SM activity sampling will NOT be in this capture."
fi

echo "[nsys] binary : ${NSYS_BIN} (version ${NSYS_VERSION:-unknown})"
echo "[nsys] run_id : ${RUN_ID}"
echo "[nsys] output : ${OUT_BASE}.nsys-rep"
echo "[nsys] command: ${NSYS_BIN} ${NSYS_ARGS[*]} python $*"
echo

# A warning like "Executable path does not exist: .../plugins/efa_metrics/
# nic_sampler" is BENIGN — an AWS network-adapter sampler absent from bundled
# Nsight builds, irrelevant to GPU profiling. See ENVIRONMENT.md.
"${NSYS_BIN}" "${NSYS_ARGS[@]}" python "$@"

REP="${OUT_BASE}.nsys-rep"
if [ ! -f "${REP}" ]; then
  echo "ERROR: expected ${REP} was not produced." >&2
  exit 4
fi

# ---- reduce to CSV -----------------------------------------------------------
# gpukernsum : per-kernel total/avg GPU time -> which kernels dominate
# cudaapisum : per-CUDA-API total CPU time  -> launch/schedule/wait cost
for report in gpukernsum cudaapisum; do
  csv="${RESULTS_DIR}/${RUN_ID}_${report}.csv"
  echo "[nsys] stats --report ${report} -> ${csv}"
  if ! "${NSYS_BIN}" stats --report "${report}" --format csv --output - "${REP}" \
       > "${csv}" 2>"${csv}.err"; then
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
