"""Generate chitrak_full_mesh.xml from chitrak_floating.xml + the real STL meshes.

chitrak_floating.xml has the correct kinematic tree (body offsets, joint axes,
ranges, inertials) but its geoms are the SIMPLIFIED collision primitives -- 7
boxes, 8 cylinders, 4 spheres -- because it was derived from
urdf/chitrak_simplified.urdf. Videos rendered from it therefore show a blocky
stand-in, not the real robot.

This keeps that verified tree untouched and only replaces the geometry: every
body gets exactly one mesh geom, matching urdf/chitrak_full.urdf (the original
non-simplified chitrak.xacro), where all 13 <visual> AND all 13 <collision>
entries are the same STL mesh at zero offset/rotation.

Run:  python3.11 make_mesh_mjcf.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "chitrak_floating.xml"
DST = HERE / "chitrak_full_mesh.xml"
MESHDIR = Path(__file__).resolve().parent.parent / "description" / "meshes"

# Current shipped gains -- IdealPDActuatorCfg in chitrak_isaac/robots/chitrak.py,
# and the same PD that holds the validated MuJoCo stand.
KP, KV = 3.0, 0.2

tree = ET.parse(SRC)
root = tree.getroot()

# --- compiler: resolve mesh filenames relative to the real mesh dir ----------
compiler = root.find("compiler")
compiler.set("meshdir", str(MESHDIR))

# --- swap every body's primitive geoms for one mesh geom ---------------------
world = root.find("worldbody")
used = []
for body in world.iter("body"):
    name = body.get("name")
    stl = f"{name}.STL"
    if not (MESHDIR / stl).is_file():
        raise SystemExit(f"missing mesh for body {name}: {MESHDIR/stl}")
    for g in list(body.findall("geom")):
        body.remove(g)
    # URDF <visual>/<collision> for every link is the mesh at xyz=0 rpy=0,
    # so no pos/quat is needed here.
    geom = ET.Element(
        "geom",
        {"name": f"{name}_mesh", "type": "mesh", "mesh": name, "rgba": "0.752941 0.752941 0.752941 1"},
    )
    body.insert(len(body), geom)
    used.append(name)

# --- declare the meshes in <asset> ------------------------------------------
asset = root.find("asset")
for name in used:
    asset.append(ET.Element("mesh", {"name": name, "file": f"{name}.STL"}))

# --- refresh actuator gains to the shipped values ---------------------------
for pos in root.find("actuator").findall("position"):
    pos.set("kp", str(KP))
    pos.set("kv", str(KV))

ET.indent(tree, space="  ")
tree.write(DST, encoding="unicode", xml_declaration=False)
print(f"wrote {DST}  ({len(used)} mesh geoms: {', '.join(used)})")
