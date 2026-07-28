"""Thin launcher: registers the Chitrak tasks, then runs Isaac Lab's rsl_rl train.py.

Isaac Lab is located via $ISAACLAB_PATH, falling back to a sibling ../IsaacLab
directory next to this repo.
"""

import os
import runpy
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _REPO)
import chitrak_isaac  # noqa: F401,E402  -- runs gym.register() via the __init__ chain

_ISAACLAB = os.environ.get("ISAACLAB_PATH") or os.path.join(os.path.dirname(_REPO), "IsaacLab")
_train = os.path.abspath(
    os.path.join(_ISAACLAB, "scripts", "reinforcement_learning", "rsl_rl", "train.py")
)
if not os.path.isfile(_train):
    raise SystemExit(
        f"Could not find Isaac Lab's train.py at {_train}.\n"
        "Set ISAACLAB_PATH to your IsaacLab checkout, e.g.\n"
        "  export ISAACLAB_PATH=$HOME/IsaacLab"
    )
sys.path.insert(0, os.path.dirname(_train))
runpy.run_path(_train, run_name="__main__")
