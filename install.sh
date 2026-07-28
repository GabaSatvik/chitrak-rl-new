#!/usr/bin/env bash
# Streamlined installer for the Chitrak locomotion stack.
#
#   ./install.sh                    # installs into the currently-active venv
#   ISAACLAB_PATH=$HOME/IsaacLab ./install.sh
#
# What it does:
#   1. sanity-checks Python 3.11 + CUDA
#   2. pip installs requirements.txt (Isaac Sim, rsl-rl, mujoco, ...)
#   3. clones Isaac Lab 2.3.2 next to this repo (or uses $ISAACLAB_PATH)
#   4. installs Isaac Lab's source packages and applies the required local patch
#   5. prints the verification ladder
#
# It does NOT install `libglu1-mesa` (needs sudo) — do that yourself, see README §1.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="${ISAACLAB_PATH:-$(dirname "$REPO")/IsaacLab}"
ISAACLAB_TAG="${ISAACLAB_TAG:-v2.3.2}"

info() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. sanity
info "Checking interpreter"
PYV="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
[[ "$PYV" == "3.11" ]] || die "Python $PYV found, but Isaac Sim 5.1.0 requires 3.11.
  Create one:  python3.11 -m venv \$HOME/venvs/isaaclab && source \$HOME/venvs/isaaclab/bin/activate"
[[ -n "${VIRTUAL_ENV:-}${CONDA_PREFIX:-}" ]] || die "No venv/conda env active. Activate one first —
  isaaclab.sh resolves its interpreter from \$CONDA_PREFIX, then \$VIRTUAL_ENV."
echo "python $PYV at $(command -v python)"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found — no NVIDIA driver visible."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------- 2. python deps
info "Installing Python dependencies (this pulls ~10 GB of Isaac Sim, be patient)"
pip install --upgrade pip
pip install -r "$REPO/requirements.txt" --extra-index-url https://pypi.nvidia.com

python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; \
print('torch', torch.__version__, '|', torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------- 3. Isaac Lab
if [[ -d "$ISAACLAB_DIR/.git" ]]; then
  info "Isaac Lab already present at $ISAACLAB_DIR — skipping clone"
else
  info "Cloning Isaac Lab $ISAACLAB_TAG -> $ISAACLAB_DIR"
  git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi

info "Installing Isaac Lab source packages (editable)"
for p in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do
  [[ -d "$ISAACLAB_DIR/source/$p" ]] && pip install -e "$ISAACLAB_DIR/source/$p"
done

# ---------------------------------------------------------------- 4. patch
# rsl-rl 3.0.1's PPO.__init__ does not accept `share_cnn_encoders`, which Isaac
# Lab 2.3.2's RslRlPpoAlgorithmCfg declares -> TypeError on every launch.
info "Applying the rsl-rl 3.0.1 compatibility patch"
RL_CFG="$ISAACLAB_DIR/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py"
if grep -q "share_cnn_encoders" "$RL_CFG"; then
  python - "$RL_CFG" <<'PY'
import re, sys
p = sys.argv[1]
t = open(p).read()
t2 = re.sub(
    r'\n    share_cnn_encoders: bool = False\n    """[^"]*"""\n', '\n', t, count=1)
if t2 == t:
    sys.exit("could not remove share_cnn_encoders automatically — see README section 2.3")
open(p, "w").write(t2)
print("  patched", p)
PY
else
  echo "  already patched (or not present in this Isaac Lab version)"
fi

# ---------------------------------------------------------------- 5. done
cat <<EOF

$(printf '\033[1;32m')Install complete.$(printf '\033[0m')

  ISAACLAB_PATH = $ISAACLAB_DIR
  repo          = $REPO

Next, in order (stop at the first failure):

  1. sudo apt-get install -y libglu1-mesa        # if you have not already
  2. export ISAACLAB_PATH=$ISAACLAB_DIR
  3. python -c "import chitrak_isaac, gymnasium as gym; \\
       print([k for k in gym.registry if 'Chitrak' in k])"
  4. python train_chitrak.py --task=Isaac-Velocity-Flat-Chitrak-Walk-v0 \\
       --headless --num_envs 4 --max_iterations 3
  5. ./train_and_record.sh --num_envs 4096 --max_iterations 1000

See README section 6 if step 4 hangs with no output — that is the headless
Vulkan/PhysX issue and it has a known fix.
EOF
