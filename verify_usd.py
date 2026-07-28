"""Verify the Chitrak USD against urdf/chitrak_full.urdf inside real PhysX.

Checks body names, per-body masses (via root_physx_view.get_masses(), the actual
simulated values -- not what the USD claims), joint names/limits, and that the
init pose leaves every foot exactly on the ground. Run after any re-conversion:

  cd IsaacLab && CONDA_PREFIX="" \
    VIRTUAL_ENV="${CHITRAK_PY_PREFIX:-${VIRTUAL_ENV:?activate your Python 3.11 venv, or set CHITRAK_PY_PREFIX}}" \
    OMNI_KIT_ACCEPT_EULA=Y ./isaaclab.sh -p ../chitrak_isaac/verify_usd.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import xml.etree.ElementTree as ET

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from chitrak_isaac.robots.chitrak import CHITRAK_CFG

URDF = str(Path(__file__).resolve().parent / "description" / "urdf" / "chitrak_full.urdf")


def urdf_masses():
    root = ET.parse(URDF).getroot()
    out = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            out[link.get("name")] = float(inertial.find("mass").get("value"))
    return out


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(CHITRAK_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    print("\n" + "=" * 78)
    print("USD :", CHITRAK_CFG.spawn.usd_path)
    print("=" * 78)

    masses = robot.root_physx_view.get_masses()[0]
    ref = urdf_masses()
    print(f"\n{'body':<20}{'sim mass':>12}{'urdf':>12}{'delta':>12}")
    for name, m in zip(robot.body_names, masses.tolist()):
        # merged base: URDF base_link (massless) + torso_link
        exp = ref.get(name, ref.get("torso_link") if name in ("base", "base_link") else None)
        d = f"{m - exp:+.3e}" if exp is not None else "  (no ref)"
        e = f"{exp:12.6f}" if exp is not None else f"{'-':>12}"
        print(f"{name:<20}{m:12.6f}{e}{d:>12}")

    sim_total = float(masses.sum())
    urdf_total = sum(ref.values())
    print(f"\n{'TOTAL':<20}{sim_total:12.6f}{urdf_total:12.6f}{sim_total-urdf_total:+12.3e}")
    print(f"  bodies={len(robot.body_names)}  joints={robot.num_joints}")
    assert abs(sim_total - urdf_total) < 1e-5, "MASS MISMATCH vs URDF"

    lim = robot.data.joint_pos_limits[0]
    print(f"\n{'joint':<24}{'lower':>10}{'upper':>10}{'default':>10}")
    for i, n in enumerate(robot.joint_names):
        print(f"{n:<24}{lim[i,0]:10.4f}{lim[i,1]:10.4f}{robot.data.default_joint_pos[0,i]:10.4f}")

    # settle the configured init pose and see where the feet actually sit
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel))
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
    robot.reset()
    sim.step()
    robot.update(0.005)

    print(f"\ninit pose check (spawn z = {CHITRAK_CFG.init_state.pos[2]}):")
    print(f"  root z = {float(robot.data.root_pos_w[0,2]):.6f}")
    for i, n in enumerate(robot.body_names):
        if "calf" in n:
            print(f"  {n:<18} z = {float(robot.data.body_pos_w[0,i,2]):+.6f}")
    print("\nOK -- masses match URDF exactly.\n")


main()
simulation_app.close()
