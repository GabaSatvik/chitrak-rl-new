"""Render a 360-degree orbit around the FINAL (settled) frame of a logged
trajectory -- holds the robot still in its last pose and sweeps the camera
azimuth a full circle, so you can inspect a settled standing/crouching pose
from every angle. Use replay_to_mp4.py instead for a normal time-lapse replay
of the whole trajectory."""
import argparse

import imageio.v2 as imageio
import mujoco
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--traj", type=str, default="/tmp/chitrak_traj.npz")
parser.add_argument("--model", type=str, default=__file__.replace("replay_360.py", "chitrak_full_mesh.xml"))
parser.add_argument("--out", type=str, default="/tmp/chitrak_360.mp4")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--num_frames", type=int, default=180)
parser.add_argument("--distance", type=float, default=0.6)
parser.add_argument("--elevation", type=float, default=-15)
parser.add_argument("--frame_index", type=int, default=-1, help="Which logged step to freeze on (-1 = last)")
args = parser.parse_args()

data = np.load(args.traj, allow_pickle=True)
joint_names = [str(n) for n in data["joint_names"]]
joint_pos = data["joint_pos"][args.frame_index]  # (12,)
root_pos = data["root_pos"][args.frame_index]  # (3,)
root_quat = data["root_quat"][args.frame_index]  # (4,) wxyz

m = mujoco.MjModel.from_xml_path(args.model)
d = mujoco.MjData(m)

qpos_idx = {}
for name in joint_names:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid == -1:
        raise ValueError(f"joint {name} not found in mujoco model")
    qpos_idx[name] = m.jnt_qposadr[jid]

d.qpos[0:3] = root_pos
d.qpos[3:7] = root_quat
for i, name in enumerate(joint_names):
    d.qpos[qpos_idx[name]] = joint_pos[i]
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, height=args.height, width=args.width)
cam = mujoco.MjvCamera()
cam.distance = args.distance
cam.elevation = args.elevation
cam.lookat = root_pos

frames = []
for i in range(args.num_frames):
    cam.azimuth = 360.0 * i / args.num_frames
    renderer.update_scene(d, camera=cam)
    frames.append(renderer.render().copy())

imageio.mimsave(args.out, frames, fps=args.fps)
print(f"[INFO]: Saved {len(frames)} frames (360-degree orbit) to {args.out}")
print(f"[INFO]: frozen pose -- root z={root_pos[2]:.4f}, joint_pos={dict(zip(joint_names, joint_pos))}")
