import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Originally a direct port of isaaclab_assets/robots/unitree.py's UNITREE_GO2_CFG
# (usd_path and joint-name regexes renamed, every numeric field left at Go2's
# literal value). The actuator block below has since been retuned to Chitrak's
# own real spec/gains instead -- see the actuators= comment below.
# Asset path: <repo>/description/usd/chitrak.usd by default. Override with
# $CHITRAK_USD if you keep the converted USD somewhere else.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHITRAK_USD = os.environ.get(
    "CHITRAK_USD", os.path.join(_REPO_ROOT, "description", "usd", "chitrak.usd")
)

CHITRAK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_full_xacro_merged/, converted directly from chitrak.xacro (the
        # real, non-simplified, SolidWorks-exported URDF), not usd_go2_merged/
        # (which was actually built from chitrak_footlinks.urdf, per its own
        # config.yaml's asset_path). Converted with --merge-joints, which folds
        # the massless base_link/torso_link split into one body -- without it,
        # both default to a phantom 1.0kg each. Masses independently verified
        # via root_physx_view.get_masses() against chitrak.xacro's own values:
        # total 1.3045kg, hip=0.1086kg x4, thigh=0.0358kg x4, calf=0.0490kg x4 --
        # exact match.
        # Regenerated 2026-07-26 from urdf/chitrak_full.urdf, which is now a
        # PERMANENT, checked-in expansion of chitrak.xacro (the original,
        # non-simplified SolidWorks export) -- chitrak.xacro turned out to have
        # no xacro directives at all, so the only edit needed was rewriting the
        # 13 `package://chitrak_description/...` mesh refs to absolute paths.
        # Supersedes usd_full_xacro_merged/, whose source URDF lived in a
        # since-deleted /tmp scratchpad and therefore could not be regenerated
        # or diffed.
        #
        # NON-SIMPLIFIED is the point: chitrak.xacro declares all 13 <collision>
        # geometries as real STL MESHES. The alternatives in urdf/ do not --
        # chitrak_simplified.urdf and chitrak_simplified_collision.xacro replace
        # them with 7 boxes + 8 cylinders + 4 spheres. Converted with
        # collider_type=convex_hull + collision_from_visuals=false, so the
        # colliders are convex hulls of the true collision meshes.
        usd_path=CHITRAK_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    # Chitrak's hip_pitch/knee joints are one-sided and mirrored left/right (not
    # front/back like Go2's) -- e.g. fl_hip_pitch's range is [-3.14, 0] while
    # fr_hip_pitch's is [0, 3.14]. Hence grouping by SIDE, not front/back.
    #
    # These values are the MuJoCo PID-validated standing configuration: a plain
    # position PD (kp=3.0, kv=0.2, forcerange=+-2.5 -- the same actuator model
    # and gains configured below) holds a stable stand from this state with no RL
    # policy at all. Analytic IK for a foot 0.20 m below the hip (thigh=0.13,
    # calf=0.15) gives hip_pitch=0.722868, knee=1.586182 and puts the torso at
    # z=0.242081 with the feet exactly on the ground.
    #
    # THE OLD VALUES HERE WERE A REAL BUG, not just a suboptimal pose. They were
    # Go2's literal magnitudes (0.8 / 1.5) at Go2-ish spawn height 0.17. Measured
    # on the actual collision meshes (mujoco/make_mesh_mjcf.py + FK), that pose
    # buries the calf links **65.3 mm BELOW the floor plane** at spawn. PhysX
    # resolves the overlap with a depenetration impulse, and on a 49 g foot that
    # impulse is enormous -- the robot is flung/bounced on frame 1 of every single
    # reset. Much of the "light-foot contact instability" this project has been
    # fighting was that self-inflicted spawn penetration; it also explains why
    # ADDING foot mass appeared to help (heavier feet resist the impulse), which
    # is what led to the torque-ceiling misdiagnosis.
    #
    # At the values below the deepest vertex of any link sits +0.5 mm above the
    # floor -- resting on it, not through it. Verified via verify_usd.py.
    # hip_roll is 0.0 (was +-0.1): JointPositionAction uses use_default_offset,
    # so the PD target at action=0 IS this pose -- +-0.1 is a standing initial
    # error the PD must immediately fight, against the hip_roll_deviation penalty.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.242081),
        joint_pos={
            ".*_hip_roll_joint": 0.0,
            "(fr|br)_hip_pitch_joint": 0.722868,
            "(fl|bl)_hip_pitch_joint": -0.722868,
            "(fr|br)_knee_joint": -1.586182,
            "(fl|bl)_knee_joint": 1.586182,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # IdealPDActuatorCfg, not DCMotorCfg: Chitrak's real controller (the
        # Gazebo ignition-gazebo-joint-position-controller-system plugin in
        # chitrak_sim/urdf/chitrak_gazebo.xacro) is a flat-clamped PID with no
        # velocity-dependent torque-speed derating -- that's IdealPDActuator's
        # model (tau = clip(Kp*err + Kd*err_vel, +-effort_limit)), not DCMotor's
        # linear-derate-to-zero-at-velocity_limit curve.
        #
        # stiffness/damping: tried Chitrak's own real Gazebo gains (p_gain=1.0,
        # d_gain=0.01) first -- fixed the saturation-headroom problem (see
        # CHITRAK_VS_SOLO12_COMPARISON.md sec 4) but empirically left velocity
        # tracking flat/un-learning (error_vel_xy stuck ~1.5 m/s from iteration
        # 100 through 1600/4096, never improving -- confirmed in
        # logs/rsl_rl/chitrak_flat/2026-07-03_16-53-54). Root cause: Chitrak's
        # real p_gain=1 was validated for smooth IK/bezier-trajectory tracking
        # (small, slow target changes), never for absorbing a neural net's raw
        # per-step position-target outputs the way RL training demands -- a
        # different task even on the same hardware. Solo12's Kp=3.0/Kd=0.2
        # (soloRL/Environment.hpp) is the more relevant reference: validated in
        # the *same* paradigm (RL policy -> direct PD position target), just on
        # a different robot of similar torque budget. Switched to that.
        "base_legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit=2.5,
            velocity_limit=8,
            stiffness=3.0,
            damping=0.2,
            friction=0.0,
        ),
    },
)
