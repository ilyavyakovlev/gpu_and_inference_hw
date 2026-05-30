# Running HW1 and HW2 on Nebius Cloud

This guide covers everything from creating a GPU instance to downloading your results.

## Prerequisites

On your **local machine**:
- `ssh` and `rsync` installed
- An SSH key pair (to authenticate to the Nebius instance)
- Access to your Nebius Cloud account at [console.nebius.ai](https://console.nebius.ai)

---

## Step 1 — Create an L40S GPU Instance on Nebius

HW2 speedup tiers are calibrated against L40S; HW1 also supports it. Use the Nebius console or CLI.

### Via the Web Console

1. Go to **Compute → Virtual Machines → Create**.
2. Choose an image: **Ubuntu 22.04 LTS** (or the Nebius Deep Learning Image if available — it ships with CUDA pre-installed).
3. Under **GPU**, select **NVIDIA L40S × 1** (or the closest available option).
4. Add your **SSH public key** (paste the contents of `~/.ssh/id_rsa.pub` or whichever key you use).
5. Assign a **public IP** so you can SSH in from your laptop.
6. Create the instance and wait until its status is **Running**.
7. Note the **public IP address** shown in the instance details page.

### Recommended instance specs
| Field        | Value            |
|-------------|-----------------|
| GPU          | NVIDIA L40S × 1 |
| vCPU         | 8+              |
| RAM          | 32 GB+          |
| Disk         | 50 GB SSD       |
| OS           | Ubuntu 22.04    |

---

## Step 2 — Configure `.env`

In the repo root, copy the example file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set:

```dotenv
NEBIUS_INSTANCE_IP=<public-ip-from-nebius-console>
NEBIUS_SSH_USER=ubuntu
NEBIUS_SSH_KEY_PATH=~/.ssh/id_rsa      # path to your private key
NEBIUS_REMOTE_DIR=/home/ubuntu/gpu_and_inference_hw
NEBIUS_SSH_PORT=22
```

> `.env` is git-ignored and will not be committed.

---

## Step 3 — Verify SSH Connectivity

Test that you can reach the instance before running the full script:

```bash
source .env
ssh -i "$NEBIUS_SSH_KEY_PATH" -p "$NEBIUS_SSH_PORT" \
    "${NEBIUS_SSH_USER}@${NEBIUS_INSTANCE_IP}" "nvidia-smi"
```

You should see an `nvidia-smi` table with the L40S listed. If the connection hangs, check that:
- The instance has a public IP assigned.
- TCP port 22 is open in the instance's security group / firewall.
- You are using the correct private key.

---

## Step 4 — Run everything in one command

From the **repo root**, run:

```bash
./deploy/nebius_create_and_run.sh
```

This script does everything end-to-end with clear timestamped logging:

1. **Starts ssh-agent** and prompts for your key passphrase **once** — no further prompts.
2. **Verifies Nebius auth** (`nebius iam whoami`).
3. **Creates a boot disk** (Ubuntu 24.04 + CUDA 13.0).
4. **Creates the GPU instance** with your SSH key embedded via cloud-init.
5. **Polls until RUNNING**, then retrieves the public IP and writes it to `.env`.
6. **Polls until SSH is available** (~60–90 s after RUNNING).
7. **Rsyncs the repo**, installs dependencies, runs HW1 + HW2, downloads results.
8. **Destroys the instance and disk** when done (saves cost automatically).

### Optional flags

| Flag           | Effect                                               |
|---------------|-----------------------------------------------------|
| `--hw1-only`  | Run only HW1 (skip HW2)                             |
| `--hw2-only`  | Run only HW2 (skip HW1)                             |
| `--no-destroy`| Leave the VM running after the run (for debugging)  |

```bash
./deploy/nebius_create_and_run.sh --hw1-only
./deploy/nebius_create_and_run.sh --no-destroy   # inspect the instance afterward
```

### Manual flow (if you already have a running VM)

If you created a VM yourself and just want to rsync + run + download:

```bash
# Set the IP in .env, then:
./deploy/run_on_nebius.sh
# Flags: --hw1-only | --hw2-only | --setup-only
```

`run_on_nebius.sh` also manages the ssh-agent automatically — one passphrase prompt at the start.

### First-run note on torch.compile

HW1 uses `torch.compile` internally. The **first** execution compiles the kernel graph (30–120 s, depending on the GPU and CUDA version). Subsequent runs on the same instance will be faster because the compiled artifacts are cached.

---

## Step 5 — Inspect Results

After the script completes, results are in `nebius_results/` locally:

```
nebius_results/
├── hw1/
│   ├── roofline.png          ← roofline diagram
│   ├── roofline_data.json    ← raw measurements
│   └── hw1_run.log           ← full stdout
└── hw2/
    ├── v0_slow_trace.json    ← Chrome trace for slow baseline
    ├── v1_optimized_trace.json ← Chrome trace for optimized loop
    └── hw2_run.log           ← full stdout with speedup summary
```

Open the Chrome traces at [ui.perfetto.dev](https://ui.perfetto.dev) to compare the GPU stream density between the slow and optimized runs.

---

## Step 6 — Re-running After Code Changes

If you edit local code and want to re-run on the same instance without repeating the setup:

```bash
./deploy/run_on_nebius.sh        # rsync re-syncs only changed files; setup is idempotent
```

Or to re-run a single homework after a fix:

```bash
./deploy/run_on_nebius.sh --hw2-only
```

---

## Troubleshooting

### `RuntimeError: Unsupported GPU '<name>'`
HW1's `hw1_runtime.py` only recognises `H100` and `L40S` in the GPU name string. If your instance reports a different name (e.g. `NVIDIA L40S Ada`), add the variant to `GPU_SPECS` in `hw1/hw1_runtime.py`:

```python
"L40S Ada": {
    "label": "NVIDIA L40S Ada 48GB",
    "peak_flops": 91.6e12,
    "peak_bw": 864e9,
},
```

### `torch.cuda.is_available()` returns `False`
The PyTorch wheel installed by `setup_instance.sh` targets CUDA 12.4 (`cu124`). If the instance has a different CUDA version, edit the `--index-url` line in `deploy/setup_instance.sh`:

```bash
# For CUDA 12.1:
pip install "torch==${TORCH_VER}" --index-url https://download.pytorch.org/whl/cu121
```

Check the instance's CUDA version with `nvidia-smi` (top-right corner) or `nvcc --version`.

### SSH connection refused / timeout
- Confirm the instance is in **Running** state in the Nebius console.
- Confirm the instance has a **public IP** (not just a private/internal IP).
- Confirm TCP/22 is permitted in the instance's network policy.

### `rsync: command not found` (on the remote)
`rsync` should be present on Ubuntu 22.04. If missing:
```bash
ssh ... "sudo apt-get install -y rsync"
```

### `torch.compile` hangs indefinitely
This is rare but can happen if Triton's compilation cache gets corrupted. Clear it with:
```bash
ssh ... "rm -rf ~/.triton/cache && rm -rf /tmp/torchinductor_*"
```
Then re-run.

---

## Tearing Down the Instance

When you are done, delete the instance from the Nebius console to stop billing. Results are already downloaded to `nebius_results/` locally.
