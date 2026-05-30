#!/usr/bin/env bash
# run_on_nebius.sh — copy the repo to a Nebius GPU instance, run hw1 and hw2,
# and download the results back to the local machine.
#
# Usage:
#   ./deploy/run_on_nebius.sh [--hw1-only | --hw2-only | --setup-only]
#
# Reads connection settings from .env in the repo root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Load .env ──────────────────────────────────────────────────────────────
if [[ ! -f ".env" ]]; then
    echo "ERROR: .env not found. Copy .env.example to .env and fill in your values."
    exit 1
fi
# Export each non-comment, non-empty line as an environment variable.
set -o allexport
# shellcheck disable=SC1091
source .env
set +o allexport

: "${NEBIUS_INSTANCE_IP:?Set NEBIUS_INSTANCE_IP in .env}"
: "${NEBIUS_SSH_USER:=karke}"
: "${NEBIUS_SSH_KEY_PATH:=$HOME/.ssh/id_ed25519}"
: "${NEBIUS_REMOTE_DIR:=/home/karke/gpu_and_inference_hw}"
: "${NEBIUS_SSH_PORT:=22}"

# ── SSH agent — load key once so passphrase is never asked during the run ──
_load_key_into_agent() {
    # Start a new agent if there isn't one, then add the key.
    # If the key is already in the agent this is a no-op.
    if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
        eval "$(ssh-agent -s)" >/dev/null
    fi
    # ssh-add exits 0 if the key is already loaded, otherwise prompts once.
    ssh-add -l 2>/dev/null | grep -qF "$(ssh-keygen -lf "${NEBIUS_SSH_KEY_PATH}" 2>/dev/null | awk '{print $2}')" \
        || ssh-add "${NEBIUS_SSH_KEY_PATH}"
}
# Only ask for passphrase when this script is run directly (not when called
# from nebius_create_and_run.sh, which already set up the agent).
if [[ "${NEBIUS_AGENT_READY:-0}" != "1" ]]; then
    _load_key_into_agent
fi

SSH_OPTS="-i ${NEBIUS_SSH_KEY_PATH} -p ${NEBIUS_SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=15"
REMOTE="${NEBIUS_SSH_USER}@${NEBIUS_INSTANCE_IP}"

run_remote() {
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "$REMOTE" "$@"
}

# ── Parse flags ────────────────────────────────────────────────────────────
RUN_HW1=true
RUN_HW2=true
SETUP_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --hw1-only)    RUN_HW2=false ;;
        --hw2-only)    RUN_HW1=false ;;
        --setup-only)  SETUP_ONLY=true ;;
        *) echo "Unknown flag: $arg" && exit 1 ;;
    esac
done

# ── Step 1: Copy repo to instance ─────────────────────────────────────────
echo "===================================================================="
echo " Copying repo to ${REMOTE}:${NEBIUS_REMOTE_DIR}"
echo "===================================================================="

rsync -avz --progress \
    -e "ssh $SSH_OPTS" \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.py[cod]' \
    --exclude='hw1/results/' \
    --exclude='hw2/results/' \
    --exclude='.env' \
    "$REPO_ROOT/" \
    "${REMOTE}:${NEBIUS_REMOTE_DIR}/"

# ── Step 2: Set up Python environment ─────────────────────────────────────
echo ""
echo "===================================================================="
echo " Setting up Python environment on instance"
echo "===================================================================="

run_remote bash "${NEBIUS_REMOTE_DIR}/deploy/setup_instance.sh" "${NEBIUS_REMOTE_DIR}"

if [[ "$SETUP_ONLY" == "true" ]]; then
    echo "Setup complete (--setup-only). Exiting."
    exit 0
fi

# ── Step 3: Run HW1 ───────────────────────────────────────────────────────
if [[ "$RUN_HW1" == "true" ]]; then
    echo ""
    echo "===================================================================="
    echo " Running HW1 (roofline benchmark)"
    echo "===================================================================="
    run_remote bash -c "
        cd ${NEBIUS_REMOTE_DIR}
        source .venv/bin/activate
        mkdir -p hw1/results
        python3 hw1/hw1_task.py 2>&1 | tee hw1/results/hw1_run.log
    "
fi

# ── Step 4: Run HW2 ───────────────────────────────────────────────────────
if [[ "$RUN_HW2" == "true" ]]; then
    echo ""
    echo "===================================================================="
    echo " Running HW2 (inference optimization)"
    echo "===================================================================="
    run_remote bash -c "
        cd ${NEBIUS_REMOTE_DIR}
        source .venv/bin/activate
        mkdir -p hw2/results
        python3 hw2/hw2_task.py 2>&1 | tee hw2/results/hw2_run.log
    "
fi

# ── Step 5: Download results ───────────────────────────────────────────────
echo ""
echo "===================================================================="
echo " Downloading results"
echo "===================================================================="

LOCAL_RESULTS="$REPO_ROOT/nebius_results"
mkdir -p "$LOCAL_RESULTS"

if [[ "$RUN_HW1" == "true" ]]; then
    mkdir -p "$LOCAL_RESULTS/hw1"
    rsync -avz \
        -e "ssh $SSH_OPTS" \
        "${REMOTE}:${NEBIUS_REMOTE_DIR}/hw1/results/" \
        "$LOCAL_RESULTS/hw1/"
fi

if [[ "$RUN_HW2" == "true" ]]; then
    mkdir -p "$LOCAL_RESULTS/hw2"
    rsync -avz \
        -e "ssh $SSH_OPTS" \
        "${REMOTE}:${NEBIUS_REMOTE_DIR}/hw2/results/" \
        "$LOCAL_RESULTS/hw2/"
fi

echo ""
echo "===================================================================="
echo " Done. Results saved to: $LOCAL_RESULTS"
echo "===================================================================="

if [[ "$RUN_HW1" == "true" ]]; then
    echo "  HW1 roofline plot : $LOCAL_RESULTS/hw1/roofline.png"
    echo "  HW1 data          : $LOCAL_RESULTS/hw1/roofline_data.json"
    echo "  HW1 log           : $LOCAL_RESULTS/hw1/hw1_run.log"
fi
if [[ "$RUN_HW2" == "true" ]]; then
    echo "  HW2 slow trace    : $LOCAL_RESULTS/hw2/v0_slow_trace.json"
    echo "  HW2 opt trace     : $LOCAL_RESULTS/hw2/v1_optimized_trace.json"
    echo "  HW2 log           : $LOCAL_RESULTS/hw2/hw2_run.log"
fi
