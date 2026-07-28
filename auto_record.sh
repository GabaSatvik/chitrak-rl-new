#!/usr/bin/env bash
# Watches a training run's checkpoint dir and renders a video via
# record_chitrak.sh every INTERVAL iterations, without touching the training
# process itself. Videos go to video_logs/runN/ -- N is assigned the first
# time a given run dir is seen and reused after that (video_logs/.registry).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_LOGS_DIR="$SCRIPT_DIR/video_logs"
REGISTRY="$VIDEO_LOGS_DIR/.registry"
INTERVAL=50
POLL_SECONDS=15
RUN_DIR=""

usage() {
  echo "Usage: $0 --run_dir <path/to/logs/rsl_rl/chitrak_flat/TIMESTAMP> [--interval N] [--poll_seconds N]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_dir) RUN_DIR="$2"; shift 2;;
    --interval) INTERVAL="$2"; shift 2;;
    --poll_seconds) POLL_SECONDS="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done
[[ -z "$RUN_DIR" ]] && usage
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

mkdir -p "$VIDEO_LOGS_DIR"
touch "$REGISTRY"

LABEL="$(awk -v d="$RUN_DIR" '$1==d {print $2}' "$REGISTRY")"
if [[ -z "$LABEL" ]]; then
  LABEL="run$(( $(wc -l < "$REGISTRY") + 1 ))"
  echo "$RUN_DIR $LABEL" >> "$REGISTRY"
fi
OUT_DIR="$VIDEO_LOGS_DIR/$LABEL"
mkdir -p "$OUT_DIR"
DONE_FILE="$OUT_DIR/.rendered"
touch "$DONE_FILE"

echo "Watching $RUN_DIR -> $OUT_DIR (every $INTERVAL iterations)"

render_new_checkpoints() {
  for CKPT in "$RUN_DIR"/model_*.pt; do
    [[ -e "$CKPT" ]] || continue
    ITER="$(basename "$CKPT" .pt | sed 's/model_//')"
    [[ "$ITER" =~ ^[0-9]+$ ]] || continue
    grep -qx "$ITER" "$DONE_FILE" && continue
    if (( ITER > 0 && ITER % INTERVAL == 0 )); then
      echo "[$(date +%T)] iteration $ITER checkpoint -- rendering"
      "$SCRIPT_DIR/record_chitrak.sh" --checkpoint "$CKPT" --out_dir "$OUT_DIR" ${RECORD_TASK:+--task "$RECORD_TASK"} && echo "$ITER" >> "$DONE_FILE"
    fi
  done
}

while true; do
  render_new_checkpoints
  if ! pgrep -f train_chitrak.py > /dev/null; then
    sleep 5
    render_new_checkpoints
    # always grab the final checkpoint too, even if not on the interval
    LAST_CKPT="$(ls -t "$RUN_DIR"/model_*.pt 2>/dev/null | head -1)"
    if [[ -n "$LAST_CKPT" ]]; then
      LAST_ITER="$(basename "$LAST_CKPT" .pt | sed 's/model_//')"
      if ! grep -qx "$LAST_ITER" "$DONE_FILE"; then
        echo "[$(date +%T)] training finished -- rendering final checkpoint $LAST_ITER"
        "$SCRIPT_DIR/record_chitrak.sh" --checkpoint "$LAST_CKPT" --out_dir "$OUT_DIR" ${RECORD_TASK:+--task "$RECORD_TASK"} && echo "$LAST_ITER" >> "$DONE_FILE"
      fi
    fi
    echo "Training process gone -- stopping watcher for $LABEL."
    break
  fi
  sleep "$POLL_SECONDS"
done
