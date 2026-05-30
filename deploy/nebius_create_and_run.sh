#!/usr/bin/env bash
# nebius_create_and_run.sh
# End-to-end: create Nebius GPU VM → setup → run hw1+hw2 → download results → destroy VM.
#
# Usage:
#   ./deploy/nebius_create_and_run.sh [--hw1-only | --hw2-only | --no-destroy]
#
# Flags:
#   --hw1-only    Run only HW1 (skip HW2)
#   --hw2-only    Run only HW2 (skip HW1)
#   --no-destroy  Keep the VM running after the run (useful for debugging)
#
# Prerequisites:
#   - .env configured (see .env.example)
#   - nebius CLI authenticated  (nebius iam whoami must succeed)
#   - jq installed              (sudo apt-get install jq)
set -euo pipefail

NEBIUS="${HOME}/.nebius/bin/nebius"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Helpers ────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
sep()  { echo; echo "════════════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════════════"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ── Load .env ──────────────────────────────────────────────────────────────
[[ -f ".env" ]] || die ".env not found — copy .env.example and fill in your values"
set -o allexport; source .env; set +o allexport

: "${NEBIUS_PROJECT_ID:?Set NEBIUS_PROJECT_ID in .env}"
: "${NEBIUS_SUBNET_ID:?Set NEBIUS_SUBNET_ID in .env}"
: "${NEBIUS_SSH_KEY_PATH:=$HOME/.ssh/id_ed25519}"
: "${NEBIUS_SSH_PUBLIC_KEY_PATH:=${NEBIUS_SSH_KEY_PATH}.pub}"
: "${NEBIUS_SSH_USER:=karke}"
: "${NEBIUS_REMOTE_DIR:=/home/${NEBIUS_SSH_USER}/gpu_and_inference_hw}"
: "${NEBIUS_SSH_PORT:=22}"
: "${NEBIUS_GPU_PLATFORM:=gpu-h100-sxm}"
: "${NEBIUS_INSTANCE_PRESET:=1gpu-16vcpu-200gb}"
: "${NEBIUS_DISK_SIZE_BYTES:=214748364800}"

[[ -f "$NEBIUS_SSH_PUBLIC_KEY_PATH" ]] \
    || die "Public key not found at $NEBIUS_SSH_PUBLIC_KEY_PATH"

# ── Parse flags ────────────────────────────────────────────────────────────
EXTRA_FLAGS=()
DESTROY=true
for arg in "$@"; do
    case "$arg" in
        --no-destroy)  DESTROY=false ;;
        --hw1-only|--hw2-only) EXTRA_FLAGS+=("$arg") ;;
        *) die "Unknown flag: $arg" ;;
    esac
done

# ── SSH agent — prompt for passphrase exactly once ─────────────────────────
sep "Setting up SSH agent"
if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
    eval "$(ssh-agent -s)" >/dev/null
    log "Started new ssh-agent (PID $SSH_AGENT_PID)"
fi
KEY_FP=$(ssh-keygen -lf "$NEBIUS_SSH_KEY_PATH" 2>/dev/null | awk '{print $2}')
if ssh-add -l 2>/dev/null | grep -qF "$KEY_FP"; then
    log "Key already loaded in agent — no passphrase needed"
else
    log "Adding $NEBIUS_SSH_KEY_PATH to agent (you will be prompted once)"
    ssh-add "$NEBIUS_SSH_KEY_PATH"
fi
export NEBIUS_AGENT_READY=1   # tells run_on_nebius.sh not to re-add the key

# ── Resource naming ────────────────────────────────────────────────────────
SUFFIX=$(date '+%Y%m%d%H%M%S')
DISK_NAME="hw-disk-${SUFFIX}"
INSTANCE_NAME="hw-instance-${SUFFIX}"
INSTANCE_ID=""
DISK_ID=""

# ── Cleanup trap ───────────────────────────────────────────────────────────
cleanup() {
    local exit_code=$?
    if [[ "$DESTROY" == "true" ]]; then
        sep "Tearing down VM"
        if [[ -n "$INSTANCE_ID" ]]; then
            log "Deleting instance $INSTANCE_ID …"
            "$NEBIUS" compute instance delete --id "$INSTANCE_ID" 2>/dev/null || true
            log "Waiting 20 s for instance to terminate before deleting disk …"
            sleep 20
        fi
        if [[ -n "$DISK_ID" ]]; then
            log "Deleting disk $DISK_ID …"
            "$NEBIUS" compute disk delete --id "$DISK_ID" 2>/dev/null || true
        fi
        log "Resources deleted."
    else
        log "--no-destroy: VM left running (instance=$INSTANCE_ID, disk=$DISK_ID)"
        log "Public IP: ${PUBLIC_IP:-unknown}  |  user: $NEBIUS_SSH_USER"
        log "Delete manually when done:"
        log "  $NEBIUS compute instance delete --id $INSTANCE_ID"
        log "  $NEBIUS compute disk delete     --id $DISK_ID"
    fi
    [[ $exit_code -eq 0 ]] \
        && log "All done — exit 0" \
        || log "Exited with code $exit_code"
}
trap cleanup EXIT

# ── 1. Verify nebius auth ──────────────────────────────────────────────────
sep "Verifying Nebius authentication"
"$NEBIUS" iam whoami --format json | jq -r '
    .service_account_profile.info.metadata.name // .user_profile.name // "unknown"
    | "Logged in as: " + .' \
    || die "nebius CLI not authenticated — check ~/.nebius/config.yaml"

# ── 2. Create boot disk ────────────────────────────────────────────────────
sep "Creating boot disk"
log "Name: $DISK_NAME  |  Size: $(( NEBIUS_DISK_SIZE_BYTES / 1073741824 )) GB"
DISK_ID=$(
    "$NEBIUS" compute disk create \
        --name              "$DISK_NAME" \
        --parent-id         "$NEBIUS_PROJECT_ID" \
        --type              network_ssd \
        --block-size-bytes  4096 \
        --size-bytes        "$NEBIUS_DISK_SIZE_BYTES" \
        --source-image-family-image-family ubuntu24.04-cuda13.0 \
        --disk-encryption-type disk_encryption_unspecified \
        --format json | jq -r '.metadata.id'
)
log "Disk created: $DISK_ID"

# ── 3. Create GPU instance ─────────────────────────────────────────────────
sep "Creating GPU instance"
PUBLIC_KEY_CONTENT="$(cat "$NEBIUS_SSH_PUBLIC_KEY_PATH")"
log "Name:    $INSTANCE_NAME"
log "GPU:     $NEBIUS_GPU_PLATFORM / $NEBIUS_INSTANCE_PRESET"

INSTANCE_ID=$(
    "$NEBIUS" compute instance create \
        --name              "$INSTANCE_NAME" \
        --parent-id         "$NEBIUS_PROJECT_ID" \
        --stopped           false \
        --resources-platform "$NEBIUS_GPU_PLATFORM" \
        --resources-preset  "$NEBIUS_INSTANCE_PRESET" \
        --boot-disk-existing-disk-id "$DISK_ID" \
        --boot-disk-attach-mode read_write \
        --boot-disk-device-id boot-disk \
        --network-interfaces "[{\"name\":\"eth0\",\"ip_address\":{\"allocationId\":\"\"},\"subnet_id\":\"${NEBIUS_SUBNET_ID}\",\"public_ip_address\":{}}]" \
        --cloud-init-user-data $"users:
 - name: ${NEBIUS_SSH_USER}
   sudo: ALL=(ALL) NOPASSWD:ALL
   shell: /bin/bash
   ssh_authorized_keys:
    - ${PUBLIC_KEY_CONTENT}" \
        --reservation-policy-policy auto \
        --format json | jq -r '.metadata.id'
)
log "Instance created: $INSTANCE_ID"

# ── 3b. Start the instance ─────────────────────────────────────────────────
log "Starting instance …"
"$NEBIUS" compute instance start --id "$INSTANCE_ID"
log "Start command sent"

# ── 4. Wait for RUNNING state ──────────────────────────────────────────────
sep "Waiting for instance to reach RUNNING state"
for i in $(seq 1 60); do
    INSTANCE_JSON=$("$NEBIUS" compute instance get --id "$INSTANCE_ID" --format json 2>/dev/null)
    STATE=$(echo "$INSTANCE_JSON" | jq -r '
        .status.state        //
        .metadata.state      //
        .spec.state          //
        "UNKNOWN"' 2>/dev/null)
    log "(${i}) State: $STATE"
    [[ "$STATE" == "RUNNING" ]] && break
    [[ $i -eq 60 ]] && die "Instance did not reach RUNNING within 10 minutes"
    sleep 10
done

# ── 5. Get public IP ───────────────────────────────────────────────────────
sep "Retrieving public IP"
PUBLIC_IP=""
for attempt in $(seq 1 12); do
    INSTANCE_JSON=$("$NEBIUS" compute instance get --id "$INSTANCE_ID" --format json 2>/dev/null)
    PUBLIC_IP=$(echo "$INSTANCE_JSON" | jq -r '
        .status.network_interfaces[0].public_ip_address.address //
        .spec.network_interfaces[0].public_ip_address.address   //
        .status.network_interfaces[0].public_ip.address         //
        empty' 2>/dev/null | grep -v '^null$' | head -1 | cut -d/ -f1)
    [[ -n "$PUBLIC_IP" ]] && break
    log "IP not yet assigned (attempt $attempt/12) — waiting 10 s …"
    sleep 10
done
[[ -n "$PUBLIC_IP" ]] || die "Could not retrieve public IP after 2 minutes"
log "Public IP: $PUBLIC_IP"

# Update .env so run_on_nebius.sh sees the correct IP
sed -i "s|^NEBIUS_INSTANCE_IP=.*|NEBIUS_INSTANCE_IP=${PUBLIC_IP}|" .env
export NEBIUS_INSTANCE_IP="$PUBLIC_IP"

# ── 6. Wait for SSH to become available ───────────────────────────────────
sep "Waiting for SSH"
SSH_OPTS="-i ${NEBIUS_SSH_KEY_PATH} -p ${NEBIUS_SSH_PORT} -o StrictHostKeyChecking=no -o ConnectTimeout=5"
REMOTE="${NEBIUS_SSH_USER}@${PUBLIC_IP}"
for attempt in $(seq 1 24); do
    # shellcheck disable=SC2086
    if ssh $SSH_OPTS "$REMOTE" "true" 2>/dev/null; then
        log "SSH is ready"
        break
    fi
    log "SSH not yet available (attempt $attempt/24) — waiting 10 s …"
    [[ $attempt -eq 24 ]] && die "SSH did not become available within 4 minutes"
    sleep 10
done

# ── 7. Deploy and run ─────────────────────────────────────────────────────
sep "Running deployment"
./deploy/run_on_nebius.sh "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"

sep "Complete"
log "Results are in: $REPO_ROOT/nebius_results/"
