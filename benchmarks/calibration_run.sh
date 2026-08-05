#!/usr/bin/env bash
#
# Calibration run — the study's measured noise floor.
#
#   ./calibration_run.sh              # run and append to the calibration log
#   ./calibration_run.sh --analyze    # report spread across all recorded runs
#
# A short, FIXED arm-B configuration, run at the start of every session:
#
#   arm B  |  shared_prefix  |  concurrency 15  |  20 requests
#
# The configuration is fixed on purpose. Its only job is to be identical every
# time, so that the variation between its results is variation in the
# ENVIRONMENT — session, card, driver, neighbours on the host — and nothing
# else. Do not "improve" these numbers; changing them resets the history.
#
# WHY BOTHER
# ----------
# An arm-to-arm delta is only meaningful against a known noise floor. Without
# one, "arm C is 12% faster" cannot be distinguished from "those two sweeps ran
# on different days, possibly on different physical A100s". This measures that
# floor directly so the delta can be reported against it.
#
# Runs are keyed by GPU UUID + timestamp, and --analyze separates within-card
# spread from between-card spread. If a metric's variance is explained by which
# card you landed on, any cross-session comparison of it is a hardware
# comparison, and the analysis says so.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="${HERE}"
REPO_ROOT="$(cd "${BENCH_ROOT}/.." && pwd)"

# `python -m common.calibration` resolves against benchmarks/, but the run
# itself executes from the repo root (so the runner's own git/provenance lookups
# behave). Exporting the path rather than cd-ing twice keeps both working.
export PYTHONPATH="${BENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Drive by default: the local disk does not survive preemption, and a
# calibration history that resets every session measures nothing.
CALIBRATION_LOG="${ATL_CALIBRATION_LOG:-/content/drive/MyDrive/atl_bench/calibration_log.jsonl}"
OUT_DIR="${ATL_CALIBRATION_OUT:-/content/drive/MyDrive/atl_bench/calibration_results}"

# --- the fixed configuration. Changing these invalidates the history. ---
CAL_ARM_SCRIPT="${BENCH_ROOT}/runners/bench_hf_baseline.py"
CAL_FIXTURE="shared_prefix"
CAL_CONCURRENCY=15
CAL_REQUESTS=20

# Colab's `python` is python3, but a shim on PATH (pyenv, conda) can point at a
# version that does not exist. Prefer an explicit $PYTHON, then python3.
if [ -n "${PYTHON:-}" ]; then
  PY="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  PY="python"
fi

# ---- analyze mode --------------------------------------------------------
if [ "${1:-}" = "--analyze" ]; then
  shift
  exec "${PY}" -m common.calibration --log "${CALIBRATION_LOG}" --analyze "$@"
fi

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

# ---- run mode ------------------------------------------------------------
cd "${REPO_ROOT}"

if [ ! -f "${CAL_ARM_SCRIPT}" ]; then
  echo "ERROR: ${CAL_ARM_SCRIPT} not found." >&2
  exit 2
fi

# GPU identity from nvidia-smi: works before any CUDA context exists and is not
# perturbed by a half-resolved torch install.
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d ' ')"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^ *//;s/ *$//')"
else
  echo "ERROR: nvidia-smi not found — calibration must be keyed by GPU UUID." >&2
  echo "       Without it the log cannot separate card changes from session noise." >&2
  exit 2
fi

BRANCH_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null)" ]; then
  BRANCH_SHA="${BRANCH_SHA}-dirty"
fi
PIP_SHA="$("${PY}" -m pip freeze 2>/dev/null | shasum -a 256 2>/dev/null | cut -d' ' -f1 || true)"
if [ -z "${PIP_SHA}" ]; then
  PIP_SHA="$("${PY}" -m pip freeze 2>/dev/null | sha256sum 2>/dev/null | cut -d' ' -f1 || echo unknown)"
fi

RUN_ID="${RUN_ID:-cal-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

cat <<EOF
========================================================================
CALIBRATION RUN — fixed configuration, do not modify
========================================================================
  arm            B (bench_hf_baseline.py)
  fixture        ${CAL_FIXTURE}
  concurrency    ${CAL_CONCURRENCY}
  requests       ${CAL_REQUESTS}
  run_id         ${RUN_ID}
  GPU            ${GPU_NAME}
  GPU UUID       ${GPU_UUID}
  branch         ${BRANCH_SHA}
  log            ${CALIBRATION_LOG}
========================================================================
EOF

"${PY}" "${CAL_ARM_SCRIPT}" \
  --fixture "${CAL_FIXTURE}" \
  --concurrency "${CAL_CONCURRENCY}" \
  --n-requests "${CAL_REQUESTS}" \
  --run-id "${RUN_ID}" \
  --out-dir "${OUT_DIR}" \
  --no-resume

SUMMARY="${OUT_DIR}/${RUN_ID}_summary.json"
if [ ! -f "${SUMMARY}" ]; then
  echo "ERROR: expected ${SUMMARY} was not produced; nothing appended." >&2
  exit 4
fi

"${PY}" -m common.calibration \
  --log "${CALIBRATION_LOG}" \
  --append "${SUMMARY}" \
  --concurrency "${CAL_CONCURRENCY}" \
  --gpu-uuid "${GPU_UUID}" \
  --gpu-name "${GPU_NAME}" \
  --branch-sha "${BRANCH_SHA}" \
  --pip-freeze-sha256 "${PIP_SHA}"

echo
echo "Spread across all recorded calibration runs:"
echo
"${PY}" -m common.calibration --log "${CALIBRATION_LOG}" --analyze
