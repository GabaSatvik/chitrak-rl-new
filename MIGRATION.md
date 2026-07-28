# Chitrak × Isaac Lab — Migration / Fresh-Server Setup Guide

**Audience:** a Claude Code (or human) instance on a *brand-new* server that has never run this
project. This file is self-contained: follow it top to bottom and you end with a working training
run. It encodes everything that was learned the hard way on the original machine (Lightning AI
studio, Ubuntu 24.04, Tesla T4) — read the **Gotchas** section before debugging anything.

Written 2026-07-28. Source machine reference: `/teamspace/studios/this_studio/`.

---

## 0. TL;DR for the impatient

```bash
# after the install steps below are done:
cd <WORKSPACE>/chitrak_isaac
TASK=Isaac-Velocity-Flat-Chitrak-Walk-v0 EXPERIMENT=chitrak_walk \
  ./train_and_record.sh --num_envs 4096 --max_iterations 1000
```
If that trains, you are migrated. Everything else in this file exists to get you to that line.

---

## 1. What is being migrated

Three directories, all siblings under one workspace root (call it `<WORKSPACE>`). **The sibling
layout is load-bearing** — `chitrak_isaac/train_chitrak.py` reaches `../IsaacLab/scripts/...` by
relative path, and `train_and_record.sh` does `cd "$SCRIPT_DIR/../IsaacLab"`.

```
<WORKSPACE>/
  IsaacLab/          # NVIDIA Isaac Lab 2.3.2, cloned + pip-installed (+2 local patches, see §5)
  chitrak/           # robot description: xacro/URDF, STL meshes, converted USD
  chitrak_isaac/     # OUR CODE — env configs, robot cfg, training/record/analysis scripts
  walk-these-ways/   # optional, Task 2 only; needs Isaac Gym Preview 4, NOT Isaac Lab
```

| Repo | Origin | Notes |
|---|---|---|
| `chitrak_isaac` | `https://github.com/mrmilohd/chitrak-isaac.git` | its own git repo; this is the thing you actually maintain |
| `chitrak` | `https://github.com/marsiitr/chitrak.git` | MARS IITR robot description |
| `IsaacLab` | `https://github.com/isaac-sim/IsaacLab.git` | pin to the 2.3.2 line (source machine: commit `b4c3210247`) |

> ⚠️ **Do not copy `.git/config` files across machines.** Clone each repo fresh over SSH or with
> `gh` so no credential rides along in a remote URL.

### What you do NOT need to copy
- `IsaacLab/logs/` — old training runs (copy only if you want the checkpoints; see §11).
- `chitrak_isaac/train_run_*.log`, `video_logs/`, `replay_mp4/`, `__pycache__/` — artifacts.
- Any conda env — see §3, we deliberately do not use one.

---

## 2. Hardware / OS assumptions and what changes if yours differ

Source machine: Ubuntu 24.04, x86_64, **Tesla T4 16 GB, headless, no RT cores**, NVIDIA driver with
working CUDA.

| Your server | What to change |
|---|---|
| Any NVIDIA GPU ≥ 8 GB, headless Linux | nothing; follow as written |
| GPU with RT cores (A100 is *also* RT-less; RTX A6000/L40/4090 have them) | you may be able to use `--video` in `play.py` directly and skip the whole MuJoCo replay pipeline (§10). Test before relying on it. |
| < 16 GB VRAM | lower `--num_envs`: ~512 ≈ 4 GB, ~2048 ≈ 8 GB, ~4096 ≈ 14–16 GB |
| Multiple GPUs | Isaac Lab supports `--distributed`; untested here. Start single-GPU. |
| No `sudo` | you need it exactly once, for `apt-get install libglu1-mesa` (§4). Ask your admin. |
| Shared/SLURM cluster | Isaac Sim wants a real filesystem for its extension cache (`~/.cache/ov`, `~/.local/share/ov`). Make sure `$HOME` is writable and not purged between jobs, or the ~10 min first-run download repeats every job. |

Non-negotiable: **Python 3.11**. Isaac Sim 5.1.0 ships no wheels for 3.12+.

---

## 3. Python environment — read this whole section before typing anything

### The situation on the source machine (and why it looks weird)
The Lightning AI studio **blocked creating conda envs and venvs**, and its default conda Python was
3.12 (wrong version). So Isaac Sim was installed into the **user site-packages of a standalone
Python 3.11** at:

```
/system/conda/miniconda3/uv/python/cpython-3.11-linux-x86_64-gnu/    # the prefix
~/.local/lib/python3.11/site-packages/                               # where isaacsim lives
```

and every command had to *pretend* that prefix was a virtualenv:

```bash
CONDA_PREFIX="" VIRTUAL_ENV=/system/conda/miniconda3/uv/python/cpython-3.11-linux-x86_64-gnu ...
```

**This is a workaround for a broken environment, not a recommendation.** On a normal college server
you almost certainly CAN make a venv — do that instead, it is strictly better.

### Recommended on the new server: a real venv

```bash
# 1. get a Python 3.11 (any of these is fine)
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev   # Ubuntu w/ deadsnakes or 24.04+
#   or, no sudo:  uv python install 3.11        (uv: https://astral.sh/uv)
#   or, no sudo:  conda create -n isaac311 python=3.11

# 2. create the venv (adjust the interpreter path)
python3.11 -m venv $HOME/venvs/isaaclab
source $HOME/venvs/isaaclab/bin/activate
python -V        # must print 3.11.x
pip install --upgrade pip
```

**Why a venv is the right answer here:** `IsaacLab/isaaclab.sh` resolves its interpreter in this
exact order (function `extract_python_exe`):

```
if $CONDA_PREFIX is set   -> uses  $CONDA_PREFIX/bin/python
elif $VIRTUAL_ENV is set  -> uses  $VIRTUAL_ENV/bin/python
else                      -> tries _isaac_sim/python.sh, else the bundled kit python
```

A venv sets `VIRTUAL_ENV` for you on `activate`, so with a venv you can drop the ugly prefix
entirely and just run `./isaaclab.sh -p script.py`. If you use **conda** instead, `CONDA_PREFIX`
gets set and that also works — as long as the conda env is the 3.11 one.

The one requirement either way: **`$VIRTUAL_ENV/bin/python` (unsuffixed) must exist.** venv and
conda both create it. (The original machine's pain was a standalone prefix that only had
`python3.11`, needing a manual symlink every session; the later relocated prefix shipped `python`,
which is why that chore disappeared.)

### If you must reproduce the no-venv setup
Install into user site-packages of the 3.11 interpreter and export, for every command:
```bash
export CONDA_PREFIX="" VIRTUAL_ENV=/abs/path/to/py311/prefix
```
Then **update the three files that hardcode the old prefix** (§7).

### Exact package set (source machine, all verified working together)

| Package | Version | Note |
|---|---|---|
| `isaacsim` (all extras) | **5.1.0.0** | pip install, see §4 |
| `torch` | **2.7.0+cu128** | pinned by `isaacsim-core`; do not upgrade |
| `rsl-rl-lib` | **3.0.1** | `train.py` hard-checks a minimum; 3.0.1 is what our patch (§5) matches |
| `gymnasium` | 1.2.1 | |
| `numpy` | 1.26.0 | |
| `scipy` | 1.15.3 | IK / FK analysis scripts |
| `wandb` | 0.28.0 | logging |
| `mujoco` | 3.10.0 | rendering + FK only, **never** for quantitative dynamics (§10) |
| `imageio`, `imageio-ffmpeg` | 2.37.0 | mp4 writing |

---

## 4. Install, step by step

```bash
export WORKSPACE=$HOME/chitrak_ws        # pick your root
mkdir -p $WORKSPACE && cd $WORKSPACE
source $HOME/venvs/isaaclab/bin/activate  # the 3.11 venv from §3
```

### 4.1 System package (needs sudo, once per machine)
```bash
sudo apt-get update && sudo apt-get install -y libglu1-mesa
```
Without this you get `libGLU.so.1: cannot open shared object file` from iray/the RTX renderer.
Physics still runs, but it pollutes every log and breaks rendering paths.

Optional but useful for diagnosing GPU problems:
```bash
sudo apt-get install -y vulkan-tools     # gives you `vulkaninfo --summary`
```

### 4.2 Isaac Sim 5.1.0
```bash
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
```
- The `extscache` extra pre-bakes the Omniverse extension cache; without it the **first run
  downloads ~10 min of extensions**. Either way, expect a long first launch — that is normal, not a
  hang.
- Accept the EULA non-interactively, one of:
  ```bash
  export OMNI_KIT_ACCEPT_EULA=Y                       # per-command (what our scripts do)
  touch $(python -c "import isaacsim,os;print(os.path.dirname(isaacsim.__file__))")/EULA_ACCEPTED
  ```
- Smoke test: `python -c "import isaacsim, torch; print(torch.cuda.is_available())"` → `True`.

### 4.3 Isaac Lab
```bash
cd $WORKSPACE
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.3.2        # or the exact source-machine commit b4c3210247
./isaaclab.sh --install     # installs isaaclab, isaaclab_assets, isaaclab_tasks, isaaclab_rl, isaaclab_mimic
```
> **Verify where it installed.** On the source machine `--install` once put packages into the wrong
> (3.12) interpreter. Check with `pip show isaaclab` and confirm the location is inside your venv.
> If it went elsewhere, install manually:
> ```bash
> for p in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do
>   pip install -e source/$p; done
> ```

```bash
pip install rsl-rl-lib==3.0.1 wandb mujoco imageio imageio-ffmpeg scipy
```

### 4.4 The two project repos
```bash
cd $WORKSPACE
git clone git@github.com:mrmilohd/chitrak-isaac.git chitrak_isaac
git clone git@github.com:marsiitr/chitrak.git chitrak     # SSH/gh — NOT the token URL
```
`chitrak_isaac` is **not** pip-installed; `train_chitrak.py` inserts its parent dir on `sys.path`
and imports it as a package. That is why the sibling layout matters.

---

## 5. Local patches to Isaac Lab you MUST reapply

`git status` in `IsaacLab/` on the source machine shows these modifications. A fresh clone will not
have them.

### 5.1 REQUIRED — `share_cnn_encoders` / rsl-rl 3.0.1 mismatch
`RslRlPpoAlgorithmCfg` declares a field that rsl-rl 3.0.1's `PPO.__init__` does not accept →
`TypeError` at startup. Delete these three lines from
`source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` (~line 211):

```python
    share_cnn_encoders: bool = False
    """Whether to share the CNN networks between actor and critic, in case CNNModels are used. Defaults to False."""

```
(i.e. the field, its docstring, and the blank line — leaving `rnd_cfg` as the next entry.)

*Alternative if you prefer not to patch vendor code:* install a newer `rsl-rl-lib` that accepts the
kwarg. Untested here; if you try it, re-verify the whole ladder in §8.

### 5.2 OPTIONAL — Go1 reference configs
`.../locomotion/velocity/config/go1/flat_env_cfg.py` and `.../go1/agents/rsl_rl_ppo_cfg.py` were
edited to disable the three GPU-tensor event terms (see §6.2) and toggle the wandb logger. These
only affect the **Go1 baseline runs** used for comparison, not Chitrak. Reapply only if you want a
Go1 reference run.

### 5.3 OPTIONAL — `apps/isaaclab.python.headless.rendering.raster.kit`
An untracked experimental kit file from the (ultimately failed) attempt to force rasterization-only
rendering on the T4. It did not solve the RTX hang (§10). Skip it unless you're revisiting that.

**Record your patches:** rather than re-doing this by hand next time, keep them as a patch file:
```bash
cd $WORKSPACE/IsaacLab && git diff > $WORKSPACE/chitrak_isaac/isaaclab_local.patch
# on the new machine:  git apply ../chitrak_isaac/isaaclab_local.patch
```

---

## 6. GPU / headless configuration (the section that saves you a day)

### 6.1 Vulkan
Isaac Sim routes PhysX GPU interop through Vulkan. On a headless box with no Vulkan ICD file, you
get `VkResult: ERROR_INCOMPATIBLE_DRIVER`, `IGpuFoundation` fails, PhysX silently falls back to
**CPU**, and Isaac Lab's GPU tensor API then **deadlocks** waiting on buffers that were never
allocated (process sits in `sigsuspend`, no error message).

Diagnose and fix:
```bash
vulkaninfo --summary          # should list your GPU as a physical device
ls /etc/vulkan/icd.d/         # loader search path — often empty on headless images
ls /tmp/vulkan/ /usr/share/vulkan/icd.d/   # the file frequently exists here instead
sudo cp /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/   # or from /tmp/vulkan/
```
**`VK_ICD_FILENAMES` is not required** — confirmed on the source machine. Training worked with
Vulkan still failing, once `libglu1-mesa` was installed *and* the events in §6.2 were disabled.
If your server has a working Vulkan ICD, you are in better shape than the source machine and can
try re-enabling those events.

### 6.2 The three deadlocking event terms
These call `root_physx_view` GPU ops before the first sim step initializes GPU buffers:
`physics_material`, `add_base_mass`, `apply_external_force_torque`. They are **already disabled** in
our Chitrak env configs. They are domain-randomization terms — fine to leave off for training, but
**re-enable them (and verify no hang) before any sim-to-real transfer**, since DR is what makes a
policy survive hardware.

### 6.3 Vulkan errors that are NOT fatal
`libGLU.so.1`, `libneuray.so`, `VkResult: ERROR_INCOMPATIBLE_DRIVER`, and the noisy errors on
`simulation_app.close()` are all rendering/teardown issues. Physics is unaffected. Do not chase them.

---

## 7. Path fixes — the absolute paths you must edit

Four files hardcode `/teamspace/studios/this_studio/...`:

| File | Line | What |
|---|---|---|
| `chitrak_isaac/robots/chitrak.py` | 35 | `usd_path=".../chitrak/chitrak_description/usd_xacro_full/chitrak.usd"` |
| `chitrak_isaac/verify_usd.py` | 31 | URDF path (also hardcodes the py3.11 prefix) |
| `chitrak_isaac/mujoco/chitrak_full_mesh.xml` | 2 | `meshdir=".../chitrak_description/meshes"` |
| `chitrak_isaac/mujoco/make_mesh_mjcf.py` | 23 | `MESHDIR = Path(...)` |

Three files hardcode the Python 3.11 prefix (`VIRTUAL_ENV=/system/conda/.../cpython-3.11-...`):
`train_and_record.sh`, `record_chitrak.sh`, `verify_usd.py`.

Fastest correct fix on the new server:
```bash
cd $WORKSPACE/chitrak_isaac
grep -rl '/teamspace/studios/this_studio' --include='*.py' --include='*.sh' --include='*.xml' . \
  | xargs sed -i "s#/teamspace/studios/this_studio#$WORKSPACE#g"

# If you use a venv (recommended), also neutralise the hardcoded interpreter prefix:
grep -rl 'cpython-3.11-linux-x86_64-gnu' --include='*.sh' --include='*.py' . \
  | xargs sed -i "s#/system/conda/miniconda3/uv/python/cpython-3.11-linux-x86_64-gnu#$VIRTUAL_ENV#g"
```
Then re-read the diffs (`git diff`) before committing — blind sed on this repo also touches the
prose in `results/*.md` if you widen the include list, which you should not.

---

## 8. The USD asset (the robot model)

`chitrak_isaac/robots/chitrak.py` points at
`chitrak/chitrak_description/usd_xacro_full/chitrak.usd`. If that directory came across in the
`chitrak` clone, **use it as-is and skip regeneration** — the checked-in USD is the validated one.

If you must regenerate it, these are the exact parameters used (from
`usd_xacro_full/config.yaml`, generated 2026-07-26):

```yaml
asset_path:        <WORKSPACE>/chitrak/chitrak_description/urdf/chitrak_full.urdf
usd_dir:           <WORKSPACE>/chitrak/chitrak_description/usd_xacro_full
usd_file_name:     chitrak.usd
force_usd_conversion: true
make_instanceable:    true
fix_base:             false
link_density:         0.0
merge_fixed_joints:   true
collider_type:        convex_hull
self_collision:       false
joint_drive: {drive_type: force, target_type: position, gains: {stiffness: 100.0, damping: 1.0}}
```
```bash
cd $WORKSPACE/IsaacLab
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  $WORKSPACE/chitrak/chitrak_description/urdf/chitrak_full.urdf \
  $WORKSPACE/chitrak/chitrak_description/usd_xacro_full/chitrak.usd \
  --merge-joints --headless
```
Notes / traps:
- Source URDF is `chitrak_full.urdf` — the **real SolidWorks export**, not
  `chitrak_simplified.urdf` and not the simplified-collision xacro. Using the simplified model
  invalidates every measured number in `results/`.
- If you start from `.xacro`: `xacro chitrak.xacro > chitrak_full.urdf`. Mesh refs use
  `$(find chitrak_description)/...` which must resolve to absolute paths (or set `ROS_PACKAGE_PATH`).
- `self_collision: false` was inherited from Go2 and is flagged in `CLAUDE.md` as a **known
  questionable choice** for Chitrak (short legs can fold through the torso). Revisit before hardware.
- **After any re-conversion, run `verify_usd.py`** — it loads the USD in PhysX, checks per-body mass
  against the URDF, prints joint limits, and prints init-pose foot heights. Total mass should be
  ~1.3045 kg, torso 0.531 kg, each calf/foot ~0.049 kg.

---

## 9. Verification ladder — run these in order, stop at the first failure

Do not skip to full training. Each rung isolates one failure class.

```bash
export WORKSPACE=$HOME/chitrak_ws
source $HOME/venvs/isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=Y
cd $WORKSPACE/IsaacLab
```

**1. CUDA visible to torch**
```bash
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**2. Isaac Sim launches headless** (first run: 10+ min of extension downloads — be patient)
```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

**3. Stock Isaac Lab training works (proves the framework + patches, not our code)**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Unitree-Go1-v0 --headless --num_envs 1 --max_iterations 5
```
This downloads the Go1 USD from Nucleus — **needs internet on first run.** If your college server is
behind a firewall, this is where you find out; pre-seed `~/.cache/ov` from the source machine or
whitelist `omniverse-content-production.s3-us-west-2.amazonaws.com`.

**4. Our USD is sane**
```bash
cd $WORKSPACE/chitrak_isaac && ./isaaclab.sh-equivalent -p verify_usd.py   # see the header of that file
```

**5. Our tasks are registered**
```bash
cd $WORKSPACE/IsaacLab
python -c "
import sys; sys.path.insert(0,'$WORKSPACE')
import chitrak_isaac, gymnasium as gym
print([k for k in gym.registry if 'Chitrak' in k])"
```
Expect: `Isaac-Velocity-Flat-Chitrak-{Stand,Walk}-v0` (+ `-Play-v0`), `Isaac-Velocity-{Flat,Rough}-Chitrak-v0`.

**6. Tiny Chitrak run**
```bash
cd $WORKSPACE/chitrak_isaac
./isaaclab.sh-equivalent -p train_chitrak.py \
  --task=Isaac-Velocity-Flat-Chitrak-Walk-v0 --headless --num_envs 4 --max_iterations 3
```

**7. Full run** — §10.

*(“`./isaaclab.sh-equivalent`” = `cd $WORKSPACE/IsaacLab && ./isaaclab.sh -p <abs path to script>`;
with a venv active you can equivalently just run `python <script>` for the non-sim ones.)*

---

## 10. Running training

### The one command
```bash
cd $WORKSPACE/chitrak_isaac
TASK=Isaac-Velocity-Flat-Chitrak-Walk-v0 EXPERIMENT=chitrak_walk \
  ./train_and_record.sh --num_envs 4096 --max_iterations 1000
#   stand task:  TASK=Isaac-Velocity-Flat-Chitrak-Stand-v0 EXPERIMENT=chitrak_stand
```
`TASK` and `EXPERIMENT` **must be set together** — `EXPERIMENT` must match `experiment_name` in the
PPO cfg for that task, or the video watcher watches the wrong log dir. `RECORD_TASK` is derived
automatically from `TASK` (it used to silently default wrong; do not un-derive it).

The script: launches training detached (`setsid nohup`) → waits for the new
`IsaacLab/logs/rsl_rl/<EXPERIMENT>/<timestamp>/` dir → starts `auto_record.sh` against it, which
rolls out checkpoints to mp4 periodically into `chitrak_isaac/video_logs/runN/`.

### Getting the wandb URL (always report it)
```bash
grep -oiE "https://wandb.ai/[^ ]+/runs/[a-z0-9]+" train_run_<timestamp>.log | head -1
```

### wandb setup on the new server
```bash
wandb login      # writes ~/.netrc
```
Source machine used entity `aaditya_b-iit-roorkee`, project `chitrak-locomotion`. The walk/flat
runners log to wandb; the stand runner logs to TensorBoard (`logs/rsl_rl/<experiment>/`).
If the college server has no outbound internet, set `WANDB_MODE=offline` and `wandb sync` later.

### Killing a run
```bash
pkill -9 -f train_chitrak.py; pkill -9 -f auto_record.sh
```

### Rules of engagement
- **Never run two Isaac Sim processes at once.** They fight over VRAM and both hang. Check first:
  `ps aux | grep -E "train_chitrak|isaaclab.sh" | grep -v grep`
- **Video: never use `--video` with `play.py` on an RT-core-less GPU (T4, A100).** It loads
  `rtx.raytracing.plugin` and hangs *forever* compiling ray-tracing shaders — not slow, hung
  (tested past 190 s, zero progress). Our workaround pipeline is: log the trajectory fully headless
  (`play_log_chitrak.py` → `.npz` of `joint_pos`/`root_pos_w`/`root_quat_w`/`applied_torque` at
  50 Hz), then render it with MuJoCo's EGL backend (`mujoco/replay_to_mp4.py`, `MUJOCO_GL=egl`).
  One-shot wrapper: `record_chitrak.sh`. On a GPU *with* RT cores, try native `--video` first.
- **Resolve run-dir → `runN` via `video_logs/.registry`, never by guessing.** Guessing this produced
  two confidently-wrong diagnoses in one session.
  ```bash
  grep "<run dir>" video_logs/.registry
  ```
- **MuJoCo is for kinematics and rendering only.** Do not use it for torque/contact magnitudes — a
  MuJoCo static-torque test was built, used, and *retracted* after its results proved to be contact-
  stiffness artifacts. For any quantitative torque claim, measure in Isaac Lab's PhysX: zero or
  constant action, read `robot.data.applied_torque` (the real post-saturation value, traced to
  `DCMotor._clip_effort` in Isaac Lab source). The old `static_holding_isaaclab.py` that did this is
  no longer in the tree — rewrite it from that one-line recipe if you need it.

---

## 11. Carrying over checkpoints (optional but recommended)

Best-known checkpoints on the source machine:
```
IsaacLab/logs/rsl_rl/chitrak_walk/2026-07-27_19-58-09/    # BEST walk — Go2-original hyperparams, mean reward 50.77
IsaacLab/logs/rsl_rl/chitrak_walk/2026-07-27_06-35-21/model_999.pt   # mean reward 49.95, +1.416 m / 6 s
IsaacLab/logs/rsl_rl/chitrak_stand/2026-07-27_13-30-59/   # the standing policy
```
```bash
rsync -avz --progress \
  <source>:/teamspace/studios/this_studio/IsaacLab/logs/rsl_rl/chitrak_walk/2026-07-27_19-58-09 \
  $WORKSPACE/IsaacLab/logs/rsl_rl/chitrak_walk/
```
Resume: `train_chitrak.py --task=... --resume --load_run <run name>`.
Each run dir also has `params/env.yaml` + `params/agent.yaml` — the ground truth of what config
produced it. Copy those even if you skip the weights.

Also copy the trajectory `.npz` files under `video_logs/runN/` if you want to re-run the offline
foot-height / duty-factor analyses without a GPU.

---

## 12. Project state you are inheriting (so you don't re-derive it)

Read these first, in this order — they supersede older prose:
`chitrak_isaac/results/results1.md` (root cause + standing fix + first walk attempts),
`results/results2.md` (the BASE walk config, git tag `walk-base`), `results/results3.md`
(exploration sweep), then `ABLATIONS.md` entries #25–38. **Entries #1–24 predate the root-cause
discovery and their "torque ceiling" framing is a known misdiagnosis — ignore their conclusions.**

**The headline result:** years of "Chitrak cannot stand" traced to a **65 mm spawn penetration** —
the Go2-inherited init pose buried the calf links below the floor plane, and PhysX's depenetration
impulse launched the robot on frame 1 of every reset. Fixed init pose (now in the base robot cfg):
`pos=(0,0,0.242081)`, `hip_pitch ±0.722868`, `knee ∓1.586182`, `hip_roll 0`.
Torque is **not** the constraint (9 % of 2.5 N·m standing); the binding limit is the **8 rad/s joint
velocity cap**, and only in swing.

| Task | Status |
|---|---|
| `Isaac-Velocity-Flat-Chitrak-Stand-v0` | solved — 0.2380 m, 0.00 mm std, L/R asymmetry 0.3°, falls 1.6 % |
| `Isaac-Velocity-Flat-Chitrak-Walk-v0` | walks — 0.236–0.242 m/s vs 0.25 commanded, falls < 0.5 % |

**The open problem:** `track_lin_vel_xy_exp` sits at 97 % of ceiling, so a limping shuffle and a
clean trot score identically. Duty factor is 91–95 % per leg (a trot is ~50 %) — it walks by
shuffling. Next levers, in order: (1) a real foot-clearance term / raise `feet_air_time`;
(2) re-add `hip_roll_deviation` ≈ −0.5; (3) restore environment stochasticity (obs noise,
`reset_base` yaw) — required regardless before hardware; (4) explicit gait clock + contact schedule,
walk-these-ways style.

**Config files are ground truth — read them, not any numbers quoted in prose:**
`robots/chitrak.py`, `tasks/velocity/chitrak_{stand,walk,flat,rough}_env_cfg.py`,
`tasks/velocity/agents/rsl_rl_ppo_cfg.py`.

---

## 13. Gotchas index (skim before debugging anything)

**Environment**
- `isaaclab.sh --install` can install into the *wrong* interpreter. Always `pip show isaaclab` after.
- `stable-baselines3 2.9.0` wants torch ≥ 2.8 while we're pinned to 2.7.0. Harmless — RSL-RL doesn't use sb3.
- First Isaac Sim run downloads ~10 min of extensions. Not a hang.
- Isaac Lab fetches robot USDs / actuator nets from Nucleus on first use. Needs internet.
- The GPU driver dropped out once mid-session on the source machine (`libcuda.so.1` missing) causing
  a spurious "Found no NVIDIA driver". It recovered by itself. **Check `torch.cuda.is_available()`
  before assuming a config bug.**

**Isaac Lab semantics**
- `heading_command=True` **silently overwrites `ang_vel_z`** with an internal heading controller
  driven by `ranges.heading`, so setting `ranges.ang_vel_z` does nothing. Set it `False` for direct
  yaw commands.
- Querying `command_manager` right after `env.reset()` shows pre-override random values —
  `reset()` resamples but never calls `_update_command()`, which only runs inside `step()`.
  **Check after at least one `env.step()`.**
- Reward arithmetic: `Episode_Reward/<term> = episodic_sum / episode_length_s`, and each term is
  `func × weight × dt` → **for any bounded-[0,1] kernel, max `Episode_Reward` == its weight.**
  wandb `Mean reward` is the episodic *return*. Ladder: falling ≈ −110, standing ≈ +12,
  rocking ≈ +29, current walk ≈ +50, estimated good walk ≈ +57.
- wandb reports `state: crashed` even on cleanly-completed runs — Isaac Sim's teardown kills the
  session before the finish handshake. **The logged data is intact**; verify via `scan_history()` /
  `run.summary` against the local log tail rather than trusting `state`.

**Robot-specific traps**
- **Chitrak's URDF has a non-intuitive joint-zero convention — `knee_joint=0` is NOT a straight leg**
  (leg reach only ~0.05 m at `knee=0`). Never infer a pose from joint numbers; verify with real FK
  (`mujoco/chitrak_full_mesh.xml`, set `qpos`, `mj_forward`, read `d.xpos`/`d.geom_xpos`).
- **Analytic 2-link IK does not work on this robot and must never be used.** It's off by 0.28 rad
  (16°) at the knee for the nominal stance — the leg isn't a clean 2-link chain from the hip
  (offsets hip→thigh `(0.022055, −0.021, 0.00041)`, foot sphere at calf-local `(0.14, −0.009, −0.025)`).
  A scripted-gait probe built on it produced `vx = −0.014 m/s` with saturated torque — that looked
  exactly like a physics blocker and was purely an IK bug. Solve against real FK
  (`scipy.least_squares` on the MuJoCo mesh model) and **assert the solution reproduces the
  configured stance pose** before trusting anything downstream.
- Joint ranges are one-sided for some joints (`hip_pitch: 0..π`, not `±π`).
- Naming: `{fr|fl|br|bl}_{hip_roll|hip_pitch|knee}_joint`.

**Analysis discipline**
- **Compare reward terms at matched fall rates, not matched iteration.** `dof_acc_l2` looked 60×
  too heavy at iteration 50 but is 0.87–2.55× Go2's at matched fall rates — the spike was fall
  impacts. Acting on the naive read (÷100) would have made things strictly worse.
- A negative `feet_air_time` is normal (Go2 reads −0.0316 while walking at 95 % tracking).
- Chitrak is **under**-penalised, not over-penalised: total penalties 2.7 % of positives vs Go2's 16.3 %.
- Known reward exploits, all real: instantaneous `track_lin_vel_xy_exp` pays a *rocking* robot 50 %
  of ceiling with zero net displacement (fixed by bounding commands away from zero + `std` 0.5→0.15;
  raising anti-rocking penalties 5× did **not** fix it — the exploit is structural to an
  instantaneous kernel); `track_ang_vel_z_exp` paid a statue 100 % of ceiling for not rotating;
  `base_height_exp` paid 3.75/4.0 for doing nothing and froze the walk task entirely.
- Hyperparameter rule that generalises: what matters is `init_noise_std × action_scale` = joint
  dither. **Sustained dither above ~±0.07 rad breaks the gait**, whichever knob gets it there.
  Go2's originals (`action_scale 0.25`, `init_noise_std 1.0`, `entropy_coef 0.01`) **work on
  Chitrak** and produced the best run — the std anneals 1.00 → 0.16 on its own.

**Reuse, don't rewrite:** `verify_usd.py`, `mujoco/chitrak_full_mesh.xml` +
`mujoco/make_mesh_mjcf.py`, and the `.npz` trajectory logs in `video_logs/runN/` (FK, foot heights,
duty factors — all computable off-GPU).

---

## 14. Migration checklist

- [ ] Clone all repos fresh over SSH/`gh` (never copy `.git/config` across machines)
- [ ] Python 3.11 venv created and activated; `$VIRTUAL_ENV/bin/python` exists
- [ ] `sudo apt-get install -y libglu1-mesa`
- [ ] `isaacsim[all,extscache]==5.1.0` installed; `torch.cuda.is_available()` is True
- [ ] Isaac Lab 2.3.2 cloned + `--install`, verified into the *right* interpreter
- [ ] `rsl-rl-lib==3.0.1`, wandb, mujoco, imageio, imageio-ffmpeg, scipy installed
- [ ] Patch §5.1 (`share_cnn_encoders`) reapplied
- [ ] Vulkan ICD checked (`vulkaninfo --summary`); if broken, confirm the §6.2 events stay disabled
- [ ] `chitrak_isaac` + `chitrak` cloned as siblings of `IsaacLab`
- [ ] All hardcoded `/teamspace/...` paths and py3.11-prefix paths rewritten (§7); `git diff` reviewed
- [ ] `chitrak.usd` present (or regenerated with §8's exact config) and `verify_usd.py` passes
- [ ] Verification ladder §9 rungs 1–6 all green
- [ ] `wandb login` done (or `WANDB_MODE=offline`)
- [ ] Checkpoints / `params/*.yaml` rsynced if you want to resume
- [ ] Full run launched; wandb URL captured and reported
