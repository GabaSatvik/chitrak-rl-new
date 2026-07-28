# Chitrak — Locomotion RL on Isaac Lab

Training walk and stand policies for **Chitrak**, a 1.3 kg 12-DOF quadruped from
[MARS, IIT Roorkee](https://github.com/marsiitr/chitrak), in NVIDIA Isaac Lab with RSL-RL PPO.

Self-contained: the robot description (URDF + meshes + converted USD), the Isaac Lab task/reward
configs, the training launcher, and an offline MuJoCo render pipeline all live here. The only
external piece is Isaac Lab itself, which `install.sh` clones and patches for you.

**Status:** standing is solved. Walking works — the policy tracks 94–97 % of the commanded forward
velocity with a < 0.5 % fall rate. Gait *quality* is the open problem: see §8.

```bash
git clone https://github.com/mrmilohd/chitrak-rl-new.git && cd chitrak-rl-new
python3.11 -m venv ~/venvs/isaaclab && source ~/venvs/isaaclab/bin/activate
./install.sh
./train_and_record.sh --num_envs 4096 --max_iterations 1000
```

---

## Contents

| Path | What |
|---|---|
| `chitrak_isaac/robots/chitrak.py` | `CHITRAK_CFG` — articulation, init pose, actuator model |
| `chitrak_isaac/tasks/velocity/chitrak_walk_env_cfg.py` | **the walk task** — commands, rewards, events |
| `chitrak_isaac/tasks/velocity/chitrak_stand_env_cfg.py` | the stand task |
| `chitrak_isaac/tasks/velocity/chitrak_{flat,rough}_env_cfg.py` | Go2-derived baselines |
| `chitrak_isaac/tasks/velocity/mdp/` | custom reward + curriculum terms |
| `chitrak_isaac/tasks/velocity/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters per task |
| `train_chitrak.py` | registers the tasks, then delegates to Isaac Lab's `rsl_rl/train.py` |
| `train_and_record.sh` | detached training **+** automatic periodic video capture |
| `play_log_chitrak.py` → `mujoco/replay_to_mp4.py` | headless rollout → `.npz` → mp4 (see §7) |
| `record_chitrak.sh` | one-shot wrapper for the above |
| `verify_usd.py` | loads the USD in PhysX, checks masses/limits/foot heights |
| `description/` | `urdf/`, `meshes/` (13 STLs), `usd/` (converted, ready to use) |
| `MIGRATION.md` | long-form setup + the full accumulated-gotchas dossier |

Registered gym IDs:
`Isaac-Velocity-Flat-Chitrak-{Walk,Stand}-v0`, `Isaac-Velocity-{Flat,Rough}-Chitrak-v0`,
plus a `-Play-v0` variant of each.

---

## 1. Requirements

| | |
|---|---|
| OS | Linux (developed on Ubuntu 24.04) |
| GPU | NVIDIA, ≥ 8 GB VRAM. Developed on a **Tesla T4 16 GB, headless** |
| Python | **3.11 exactly** — Isaac Sim 5.1.0 ships no 3.12+ wheels |
| Disk | ~25 GB (Isaac Sim is large) |
| Internet | needed on first launch (Omniverse extension + asset download) |

One system package, needs `sudo`, once per machine:

```bash
sudo apt-get install -y libglu1-mesa
```
Without it you get `libGLU.so.1: cannot open shared object file` on every launch.

VRAM → `--num_envs`: ~512 ≈ 4 GB · ~2048 ≈ 8 GB · ~4096 ≈ 14–16 GB.

---

## 2. Install

### 2.1 The venv (this matters — read it)

```bash
python3.11 -m venv ~/venvs/isaaclab
source ~/venvs/isaaclab/bin/activate
python -V   # must print 3.11.x
```

Isaac Lab's `isaaclab.sh` picks its interpreter in this exact order:

```
$CONDA_PREFIX set   -> $CONDA_PREFIX/bin/python
$VIRTUAL_ENV set    -> $VIRTUAL_ENV/bin/python
otherwise           -> bundled kit python
```

so **a venv or a conda env must be active**, and `$VIRTUAL_ENV/bin/python` (unsuffixed) must exist —
both venv and conda create it. Conda works equally well:
`conda create -n isaac311 python=3.11 && conda activate isaac311`.

> If you are on a host that forbids venvs (some managed notebook platforms do), you can point at a
> bare Python 3.11 prefix instead:
> `export CONDA_PREFIX="" VIRTUAL_ENV=/abs/path/to/py311-prefix` before every command. The shell
> scripts here honour `$CHITRAK_PY_PREFIX` for the same purpose.

### 2.2 Everything else

```bash
./install.sh
```

which pip-installs [`requirements.txt`](requirements.txt), clones Isaac Lab `v2.3.2` as a sibling
directory (override with `ISAACLAB_PATH`), installs its five source packages editable, and applies
the patch below. Manual equivalent:

```bash
pip install -r requirements.txt --extra-index-url https://pypi.nvidia.com
git clone --depth 1 -b v2.3.2 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
for p in isaaclab isaaclab_assets isaaclab_mimic isaaclab_rl isaaclab_tasks; do
  pip install -e ../IsaacLab/source/$p; done
```

> **Check where it landed.** `pip show isaaclab` — the location must be inside your venv. Isaac
> Lab's own `./isaaclab.sh --install` has been observed installing into a *different* interpreter
> than the one running the sim, which produces baffling `ModuleNotFoundError`s later.

### 2.3 The one required patch to Isaac Lab

Isaac Lab 2.3.2's `RslRlPpoAlgorithmCfg` declares `share_cnn_encoders`, which `rsl-rl-lib==3.0.1`'s
`PPO.__init__` does not accept → `TypeError` on every launch. `install.sh` removes it; by hand,
delete these two lines from `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` (~line 211):

```python
    share_cnn_encoders: bool = False
    """Whether to share the CNN networks between actor and critic, in case CNNModels are used. Defaults to False."""
```

### 2.4 Environment variables

```bash
export ISAACLAB_PATH=$HOME/IsaacLab   # only if not a sibling of this repo
export OMNI_KIT_ACCEPT_EULA=Y         # the shell scripts set this themselves
```

---

## 3. Verify before training

Run in order; stop at the first failure.

```bash
# 1. torch sees the GPU
python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. Isaac Sim launches headless   (FIRST RUN TAKES ~10 MIN — extension download, not a hang)
cd $ISAACLAB_PATH && ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless

# 3. stock Isaac Lab trains (proves the framework + patch, independent of this repo)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-Unitree-Go1-v0 --headless --num_envs 1 --max_iterations 5

# 4. our USD is sane — expect total ~1.3045 kg, torso 0.531 kg, each calf ~0.049 kg
python verify_usd.py

# 5. our tasks are registered
python -c "import chitrak_isaac, gymnasium as gym; print([k for k in gym.registry if 'Chitrak' in k])"

# 6. a 3-iteration Chitrak run
python train_chitrak.py --task=Isaac-Velocity-Flat-Chitrak-Walk-v0 --headless --num_envs 4 --max_iterations 3
```

Step 3 downloads the Go1 USD from Omniverse Nucleus. **Behind a firewall this is where you find
out** — whitelist `omniverse-content-production.s3-us-west-2.amazonaws.com` or pre-seed `~/.cache/ov`.

---

## 4. Train

```bash
TASK=Isaac-Velocity-Flat-Chitrak-Walk-v0 EXPERIMENT=chitrak_walk \
  ./train_and_record.sh --num_envs 4096 --max_iterations 1000
```

`TASK` and `EXPERIMENT` **must be set together** — `EXPERIMENT` has to match `experiment_name` in
the PPO config for that task (`chitrak_walk` / `chitrak_stand`), or the video watcher watches the
wrong log directory. `RECORD_TASK` is derived from `TASK` automatically.

The script launches training detached with `setsid nohup`, waits for Isaac Lab to create
`$ISAACLAB_PATH/logs/rsl_rl/$EXPERIMENT/<timestamp>/`, then starts `auto_record.sh` against it,
which renders checkpoints to mp4 in `video_logs/runN/` as training proceeds. Console output goes to
`train_run_<timestamp>.log`.

Plain foreground training, no video:

```bash
python train_chitrak.py --task=Isaac-Velocity-Flat-Chitrak-Walk-v0 --headless --num_envs 4096
```

**Stop:** `pkill -9 -f train_chitrak.py; pkill -9 -f auto_record.sh`

> **Never run two Isaac Sim processes at once** — they fight over VRAM and both hang with no error.
> Check first: `ps aux | grep -E "train_chitrak|isaaclab.sh" | grep -v grep`

### Logging

The walk/flat runners log to Weights & Biases; the stand runner logs to TensorBoard. `wandb login`
once, or `export WANDB_MODE=offline` if the machine has no outbound internet. Grab the run URL with:

```bash
grep -oiE "https://wandb.ai/[^ ]+/runs/[a-z0-9]+" train_run_<timestamp>.log | head -1
```

> W&B reports `state: crashed` even on runs that finished all iterations cleanly — Isaac Sim's
> teardown kills the session before the finish handshake. **The logged data is intact**; check
> `run.summary` / `scan_history()` rather than the `state` field.

### Resume / evaluate

```bash
python train_chitrak.py --task=... --resume --load_run <run-name>
python play_chitrak.py  --task=Isaac-Velocity-Flat-Chitrak-Walk-Play-v0 --headless --load_run <run-name>
```

---

## 5. Current configuration

The walk task as committed (`chitrak_walk_env_cfg.py`):

| | value |
|---|---|
| `lin_vel_x` command | `(0.0, 0.5)` m/s |
| `lin_vel_y` command | `(0.0, 0.3)` m/s — one-sided, see note |
| `ang_vel_z` command | `(0.0, 0.0)` — yaw pinned, `heading_command=False` |
| `action_scale` | 0.25 |
| `init_noise_std` / `entropy_coef` | 1.0 / 0.01 |
| envs × iterations | 4096 × 1000 |
| `episode_length_s` | 20 s, `decimation` 4, `sim.dt` 0.005 (policy at 50 Hz) |

Command ranges are deliberately **one-sided (0 → +max)**, not symmetric. `track_lin_vel_xy_exp` is
evaluated *instantaneously*, so with a symmetric range a robot that merely oscillates about zero
collects ~50 % of the reward ceiling with **zero net displacement** — a real, measured exploit here,
not a hypothetical. See §8.

`action_scale 0.25 / init_noise_std 1.0 / entropy_coef 0.01` are Unitree Go2's stock values, and
they are the best-performing setting found on Chitrak (mean reward 53.06). What actually matters is
their product: `init_noise_std × action_scale` = per-joint dither. **Sustained dither above
±0.07 rad breaks the gait**, whichever knob gets it there.

### The robot

12 DOF (`{fr,fl,br,bl}_{hip_roll,hip_pitch,knee}_joint`), 1.3045 kg total, torso 0.531 kg,
calf/foot 0.049 kg each. Thigh 0.13 m, calf 0.15 m. Motors: **2.5 N·m**, **8 rad/s**.

Init pose — do not change casually, it is the fix for a two-year bug (§8):

```
pos = (0, 0, 0.242081)
.*_hip_roll_joint         0.0
(fr|br)_hip_pitch_joint  +0.722868     (fl|bl)_hip_pitch_joint  -0.722868
(fr|br)_knee_joint       -1.586182     (fl|bl)_knee_joint       +1.586182
```

> **Chitrak's URDF does not use the textbook joint-zero convention — `knee=0` is NOT a straight
> leg** (leg reach is only ~0.05 m there). Never infer a pose from joint numbers; verify with real
> forward kinematics against `mujoco/chitrak_full_mesh.xml`. Relatedly, **analytic 2-link IK does
> not work on this robot** — it is off by 0.28 rad at the knee, because the leg is not a clean
> 2-link chain from the hip. A scripted-gait experiment built on that IK produced numbers that
> looked exactly like a physics blocker and were purely an IK bug.

### Regenerating the USD (not normally needed — `description/usd/` is ready to use)

```bash
cd $ISAACLAB_PATH
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  <repo>/description/urdf/chitrak_full.urdf <repo>/description/usd/chitrak.usd \
  --merge-joints --headless
python verify_usd.py       # ALWAYS, after any re-conversion
```
Source is `chitrak_full.urdf` — the real SolidWorks export, **not** the simplified-collision variant.
Conversion settings used are recorded in `description/usd/config.yaml`.

---

## 6. Headless GPU — the failure that looks like a hang

Isaac Sim routes PhysX GPU interop through Vulkan. On a headless box with no Vulkan ICD file you get
`VkResult: ERROR_INCOMPATIBLE_DRIVER`, `IGpuFoundation` fails, PhysX **silently falls back to CPU**,
and Isaac Lab's GPU tensor API then **deadlocks** on buffers that were never allocated — the process
sits in `sigsuspend` forever with no error.

```bash
sudo apt-get install -y vulkan-tools
vulkaninfo --summary            # should list your GPU
ls /etc/vulkan/icd.d/           # loader search path, often empty on headless images
sudo cp /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/   # or from /tmp/vulkan/
```

`VK_ICD_FILENAMES` is **not** required. Three domain-randomization event terms
(`physics_material`, `add_base_mass`, `apply_external_force_torque`) call `root_physx_view` GPU ops
before the first sim step and are the ones that deadlock — they are **already disabled** in these
configs, which is what makes training work even with Vulkan broken. **Re-enable them before any
sim-to-real transfer**; domain randomization is what makes a policy survive hardware.

Errors that are noisy but **not** fatal: `libGLU.so.1`, `libneuray.so`,
`VkResult: ERROR_INCOMPATIBLE_DRIVER`, and anything printed during `simulation_app.close()`.

---

## 7. Video

> **Never pass `--video` to Isaac Lab's `play.py` on a GPU without RT cores** (T4, A100). It loads
> `rtx.raytracing.plugin`, which tries to compile ray-tracing pipeline shaders and **hangs forever**
> — verified past 190 s with zero progress. It is not a slow first-run shader cache.

The pipeline here sidesteps rendering in Isaac Sim entirely:

```
play_log_chitrak.py       roll out headless, log joint_pos / root_pos_w / root_quat_w /
                          applied_torque at 50 Hz to .npz   (no rendering at all)
        ↓
mujoco/replay_to_mp4.py   render that trajectory with MuJoCo's EGL backend  ->  mp4
```

One-shot:

```bash
./record_chitrak.sh --checkpoint logs/rsl_rl/chitrak_walk/<run>/model_999.pt \
                    --task Isaac-Velocity-Flat-Chitrak-Walk-Play-v0
```

`auto_record.sh` runs this automatically during training. The `.npz` files it leaves in
`video_logs/runN/` are also the input for offline foot-height and duty-factor analysis — no GPU
needed.

> **Resolve run-dir → `runN` via `video_logs/.registry`, never by guessing.** Guessing that mapping
> produced two confidently-wrong diagnoses in a single session.
>
> **MuJoCo here is for kinematics and rendering only.** Do not use it for torque or contact
> magnitudes — a MuJoCo static-torque test was built, used, and then *retracted* when its results
> turned out to be contact-stiffness artifacts. For quantitative torque, measure in PhysX: constant
> action, read `robot.data.applied_torque` (the real post-saturation value).

---

## 8. Results, and what is actually hard here

### The root cause of "Chitrak cannot stand"

The Go2-inherited init pose (`z=0.17`, `hip_pitch ±0.8`, `knee ±1.5`) buried the calf links
**65.3 mm below the floor plane**. PhysX resolves that overlap with a depenetration impulse, which
on a 49 g foot launches the robot on frame 1 of *every* reset. Years of "the motors are too weak"
was this. Adding foot mass appeared to help only because heavier feet resist the impulse — which is
what generated the earlier torque-ceiling misdiagnosis.

**Torque is not the constraint**: 9 % of 2.5 N·m standing on four legs, 50 % at a 2× dynamic trot
peak. The binding limit is the **8 rad/s joint velocity cap**, and only during swing.

### Where it stands

| Task | Result |
|---|---|
| `...-Chitrak-Stand-v0` | solved — 0.2380 m height, 0.00 mm std, L/R asymmetry 0.3°, 1.6 % falls |
| `...-Chitrak-Walk-v0` | walks — 0.236–0.242 m/s vs 0.25 commanded, `error_vel_xy` 0.037–0.040, < 0.5 % falls |

### The open problem: gait quality is invisible to the reward

`track_lin_vel_xy_exp` sits at **97 % of its ceiling**, so there is almost nothing left to gain on
the task as specified — and **a limping shuffle scores identically to a clean trot**. Measured duty
factors are 91–95 % per leg (a trot is ~50 %) with 1.2–2.0 cm of foot lift: it walks by shuffling.
Hip-roll splay and ~10°/6 s yaw drift are likewise unpriced.

Next levers, in order:

1. A real foot-clearance term (or raise `feet_air_time`, currently ≈ −0.002, i.e. inert) — the only
   way to make swing phase pay.
2. Re-add `hip_roll_deviation` ≈ −0.5. It took the stand task's asymmetry from 8° to 0.3°.
3. Restore environment stochasticity (obs noise, `reset_base` yaw) so asymmetric solutions stop
   being free. **Required regardless before hardware.**
4. Explicit gait structure (phase clock + contact schedule, walk-these-ways style) if 1–3 fall short
   — leg alternation does not reliably emerge from a velocity scalar alone.

### Reward exploits found here (all real, all measured)

- **Standing still paid +11.9 and rocking paid +29**, because `track_lin_vel_xy_exp` is
  instantaneous: oscillating through the commanded band collects 50 % of the ceiling with zero net
  displacement. Fixed by bounding commands away from zero and tightening `std` 0.5 → 0.15. Raising
  the anti-rocking penalties 5× did **not** fix it — the exploit is structural to an instantaneous
  kernel.
- `track_ang_vel_z_exp` paid a motionless robot **100 % of its ceiling** for not rotating when yaw
  was commanded to 0. Weight cut 0.75 → 0.25.
- `base_height_exp` paid **3.75 of 4.0 for doing nothing**, freezing the walk task completely.
  Removed.

### Reading the numbers

`Episode_Reward/<term> = episodic_sum / episode_length_s`, and each term is `func × weight × dt`, so
**for any bounded-[0,1] kernel the maximum `Episode_Reward` equals its weight.** W&B's `Mean reward`
is the episodic *return*. Ladder: falling ≈ −110 · standing ≈ +12 · rocking ≈ +29 · current walk
≈ +50 · estimated good walk ≈ +57.

Two analysis rules learned expensively:

- **Compare reward terms at matched fall rates, not matched iteration.** `dof_acc_l2` looked 60×
  too heavy at iteration 50 but is 0.87–2.55× Go2's at matched fall rates — the spike was fall
  impacts, not commanded motion. Acting on the naive read would have made things strictly worse.
- A negative `feet_air_time` is normal (Go2 reads −0.0316 while tracking at 95 %). And Chitrak is
  **under**-penalised, not over-penalised: total penalties are 2.7 % of positives vs Go2's 16.3 %.

---

## 9. Credits

Robot: [MARS, IIT Roorkee](https://github.com/marsiitr/chitrak) ·
Framework: [Isaac Lab](https://github.com/isaac-sim/IsaacLab) ·
PPO: [RSL-RL](https://github.com/leggedrobotics/rsl_rl) ·
Task/reward structure derived from Isaac Lab's Unitree Go2 velocity-locomotion configs.
