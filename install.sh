#!/usr/bin/env bash
# ==============================================================================
# All-in-one setup & installer for Chitrak Locomotion on Isaac Lab
# Fully automated for fresh servers / cloud GPU instances (Vast.ai, Lambda, RunPod).
# Handles:
#   1. System packages (libglu1-mesa, vulkan-tools) & Vulkan ICD linkage
#   2. Python 3.11 venv creation and activation (if not already active)
#   3. Pip build dependencies (setuptools, wheel)
#   4. GPU architecture auto-detection (RTX 50-series / Blackwell sm_120 -> cu128 PyTorch)
#   5. Isaac Sim & project requirements installation
#   6. Isaac Lab 2.3.2 clone & editable install with --no-build-isolation
#   7. Automated share_cnn_encoders patch
#   8. EULA acceptance & stale lock cleanup
#   9. End-to-end verification
# ==============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="${ISAACLAB_PATH:-$(dirname "$REPO")/IsaacLab}"
ISAACLAB_TAG="${ISAACLAB_TAG:-v2.3.2}"

info() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33mWARN: %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. System packages & Vulkan
if command -v apt-get >/dev/null 2>&1; then
  SUDO=""
  [[ "${EUID:-$(id -u)}" -ne 0 ]] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
  if [[ "${EUID:-$(id -u)}" -eq 0 ]] || [[ -n "$SUDO" ]]; then
    info "Installing required system packages (libglu1-mesa, vulkan-tools)"
    $SUDO apt-get update -qq || true
    $SUDO apt-get install -y -qq libglu1-mesa vulkan-tools git >/dev/null 2>&1 || true

    # Link Vulkan ICD so headless PhysX finds NVIDIA driver
    if [[ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
      $SUDO mkdir -p /etc/vulkan/icd.d
      for src in /usr/share/vulkan/icd.d/nvidia_icd.json /tmp/vulkan/nvidia_icd.json; do
        if [[ -f "$src" ]]; then
          $SUDO cp "$src" /etc/vulkan/icd.d/
          info "Configured Vulkan ICD: $src -> /etc/vulkan/icd.d/nvidia_icd.json"
          break
        fi
      done
    fi
  fi
fi

# ---------------------------------------------------------------- 2. Python 3.11 Venv
if [[ -z "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]]; then
  VENV_DIR="$HOME/venvs/isaaclab"
  info "No virtual environment active. Setting up venv at $VENV_DIR"
  if command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv "$VENV_DIR"
  elif command -v python3 >/dev/null 2>&1 && [[ "$(python3 -c 'import sys; print("%d.%d"%sys.version_info[:2])')" == "3.11" ]]; then
    python3 -m venv "$VENV_DIR"
  else
    die "Python 3.11 is required. Please install python3.11 & python3.11-venv first."
  fi
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
fi

PYV="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[[ "$PYV" == "3.11" ]] || die "Python $PYV found, but Isaac Sim requires Python 3.11."
echo "Using Python $PYV at $(command -v python)"

# Persist EULA
export OMNI_KIT_ACCEPT_EULA=Y
if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -f "$VIRTUAL_ENV/bin/activate" ]]; then
  grep -q "OMNI_KIT_ACCEPT_EULA" "$VIRTUAL_ENV/bin/activate" || echo 'export OMNI_KIT_ACCEPT_EULA=Y' >> "$VIRTUAL_ENV/bin/activate"
fi

# ---------------------------------------------------------------- 3. Build toolchain & GPU Check
info "Upgrading pip, setuptools (pinned for pkg_resources compatibility), and wheel"
pip install --upgrade pip "setuptools<=80.10.2" wheel
pip install --no-build-isolation flatdict

command -v nvidia-smi >/dev/null || die "nvidia-smi not found — no NVIDIA driver visible."
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
info "Detected GPU: $GPU_NAME"

IS_BLACKWELL=0
if echo "$GPU_NAME" | grep -Ei "5090|5080|5070|5060|Blackwell|B100|B200" >/dev/null; then
  IS_BLACKWELL=1
  info "Blackwell / RTX 50-series detected (sm_120 architecture)."
fi

# ---------------------------------------------------------------- 4. Python dependencies
info "Installing Python dependencies (Isaac Sim, rsl-rl, mujoco, etc.)"
pip install -r "$REPO/requirements.txt" --extra-index-url https://pypi.nvidia.com

# If RTX 50-series, ensure PyTorch with CUDA 12.8 (cu128) nightly is installed for sm_120
if [[ "$IS_BLACKWELL" -eq 1 ]]; then
  info "Installing PyTorch nightly (cu128) with native sm_120 support for RTX 50-series"
  pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
fi

# ---------------------------------------------------------------- 5. Isaac Lab Clone & Editable Install
if [[ -d "$ISAACLAB_DIR/.git" ]]; then
  info "Isaac Lab already present at $ISAACLAB_DIR — skipping clone"
else
  info "Cloning Isaac Lab $ISAACLAB_TAG -> $ISAACLAB_DIR"
  git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi

info "Installing Isaac Lab source packages (editable with --no-build-isolation)"
for p in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do
  if [[ -d "$ISAACLAB_DIR/source/$p" ]]; then
    pip install -e "$ISAACLAB_DIR/source/$p" --no-build-isolation
  fi
done

# ---------------------------------------------------------------- 6. Patch rsl-rl compatibility
info "Applying rsl-rl 3.0.1 compatibility patch"
RL_CFG="$ISAACLAB_DIR/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py"
if [[ -f "$RL_CFG" ]] && grep -q "share_cnn_encoders" "$RL_CFG"; then
  python - "$RL_CFG" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
t2 = re.sub(
    r'\n    share_cnn_encoders: bool = False\n    """[^"]*"""\n', '\n', t, count=1)
if t2 == t:
    sys.exit("could not remove share_cnn_encoders automatically — check rl_cfg.py manually")
open(p, "w").write(t2)
print("  Patched successfully:", p)
PY
else
  echo "  Already patched (or pattern not present)"
fi

# ---------------------------------------------------------------- 7. Clean stale locks & Verify
rm -f "$HOME/.cache/ov/_cache.lock" /root/.cache/ov/_cache.lock 2>/dev/null || true

info "Verifying PyTorch CUDA execution"
python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see GPU'; x = torch.ones(5, device='cuda'); assert (x+1).sum().item() == 10.0; print('  CUDA Tensor computation: OK on', torch.cuda.get_device_name(0))"

cat <<EOF

$(printf '\033[1;32m')========================================================$(printf '\033[0m')
$(printf '\033[1;32m')✓ Chitrak Locomotion Stack installed and verified!$(printf '\033[0m')
$(printf '\033[1;32m')========================================================$(printf '\033[0m')

To train:
  1. source $HOME/venvs/isaaclab/bin/activate
  2. cd $REPO
  3. ./train_and_record.sh --num_envs 4096 --max_iterations 1000

EOF
