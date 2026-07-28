#!/usr/bin/env bash
# Launches Chitrak training detached, then starts auto_record.sh detached
# against its run directory once Isaac Lab creates it (periodic mp4s every
# 150 iterations via record_chitrak.sh -- see auto_record.sh).
# Extra args are passed straight through to train_chitrak.py, e.g.:
#   ./train_and_record.sh --num_envs 4096 --max_iterations 1500
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="${ISAACLAB_PATH:-$(cd "$SCRIPT_DIR/../IsaacLab" && pwd)}"
# EXPERIMENT selects which logs/rsl_rl/<experiment> dir to watch -- must match
# the PPO cfg's experiment_name for the chosen TASK (chitrak_flat for the walk
# task, chitrak_stand for the stand task). RECORD_TASK is the Play task the
# video watcher rolls checkpoints out with (defaults to the training TASK's
# Play variant).
EXPERIMENT="${EXPERIMENT:-chitrak_walk}"
LOG_ROOT="$ISAACLAB_DIR/logs/rsl_rl/$EXPERIMENT"
TASK="${TASK:-Isaac-Velocity-Flat-Chitrak-Walk-v0}"
# The comment above used to claim RECORD_TASK "defaults to the training TASK's
# Play variant" -- but nothing ever computed that default, so it stayed empty,
# auto_record.sh passed no --task, and record_chitrak.sh silently fell back to
# its hardcoded Isaac-Velocity-Flat-Chitrak-Play-v0 (the WALK task). Result:
# every stand-run video was the stand policy rolled out in the WALK env -- wrong
# spawn height, wrong default pose, wrong action_scale, non-zero velocity
# commands. Now actually derived, so training and recording can't diverge.
RECORD_TASK="${RECORD_TASK:-${TASK%-v0}-Play-v0}"

mkdir -p "$LOG_ROOT" "$SCRIPT_DIR/video_logs"
LOGFILE="$SCRIPT_DIR/train_run_$(date +%Y%m%d_%H%M%S).log"
BEFORE="$(ls -1 "$LOG_ROOT" 2>/dev/null || true)"

# The Python 3.11 install relocated from /usr/lib/python-build-standalone/3.11
# to the uv-managed prefix below, which already ships the unsuffixed `python`
# symlink isaaclab.sh requires -- so no more manual re-symlinking each session.

echo "Launching training -> $LOGFILE"
cd "$ISAACLAB_DIR"
setsid nohup env CONDA_PREFIX="" VIRTUAL_ENV="${CHITRAK_PY_PREFIX:-${VIRTUAL_ENV:?activate your Python 3.11 venv, or set CHITRAK_PY_PREFIX}}" OMNI_KIT_ACCEPT_EULA=Y \
  ./isaaclab.sh -p "$SCRIPT_DIR/train_chitrak.py" --task="$TASK" --headless "$@" \
  > "$LOGFILE" 2>&1 < /dev/null &
disown

echo "Waiting for the new run directory under $LOG_ROOT ..."
NEW_RUN=""
for _ in $(seq 1 60); do
  AFTER="$(ls -1 "$LOG_ROOT" 2>/dev/null || true)"
  NEW_RUN="$(comm -13 <(echo "$BEFORE" | sort) <(echo "$AFTER" | sort) | tail -1)"
  [[ -n "$NEW_RUN" ]] && break
  sleep 2
done

if [[ -z "$NEW_RUN" ]]; then
  echo "Run directory never appeared within 2 minutes -- not starting the video watcher."
  echo "$LOGFILE"
  exit 0
fi

RUN_DIR="$LOG_ROOT/$NEW_RUN"
echo "Training run dir: $RUN_DIR"
setsid nohup env ${RECORD_TASK:+RECORD_TASK="$RECORD_TASK"} "$SCRIPT_DIR/auto_record.sh" --run_dir "$RUN_DIR" \
  > "$SCRIPT_DIR/video_logs/watcher_${NEW_RUN}.log" 2>&1 < /dev/null &
disown
echo "Video watcher started (every ${INTERVAL:-50} iterations) -> chitrak_isaac/video_logs/"

echo "$LOGFILE"
