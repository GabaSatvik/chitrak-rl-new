import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chitrak_isaac  # noqa: F401  -- runs gym.register() via the __init__ chain

_play = os.path.abspath(
    os.path.join(os.environ.get("ISAACLAB_PATH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "IsaacLab"), "scripts", "reinforcement_learning", "rsl_rl", "play.py")
)
sys.path.insert(0, os.path.dirname(_play))
runpy.run_path(_play, run_name="__main__")
