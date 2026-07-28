from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


# Direct port of isaaclab_tasks/.../config/go2/agents/rsl_rl_ppo_cfg.py.
@configclass
class ChitrakRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "chitrak_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ChitrakFlatPPORunnerCfg(ChitrakRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "chitrak_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
        self.logger = "wandb"
        self.wandb_project = "chitrak-locomotion"


@configclass
class ChitrakStandPPORunnerCfg(ChitrakFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 500
        self.save_interval = 50
        self.experiment_name = "chitrak_stand"
        # tensorboard for this run (inherited "wandb" from the flat runner)
        self.logger = "tensorboard"

        # init_noise_std 1.0 -> 0.25 and entropy_coef 0.01 -> 0.002.
        #
        # Both were Go2's literal values -- a 15 kg robot with 23.5 Nm motors,
        # where this much exploration dither is proportionally trivial. On
        # Chitrak it is not. JointPositionAction uses use_default_offset=True, so
        # the target at action=0 IS init_state.joint_pos, which is now the
        # PID-validated pose -- meaning THE ZERO-ACTION POLICY IS EXACTLY THE
        # PLAIN PID CONTROLLER THAT PROVABLY HOLDS A STAND. The network only has
        # to learn to output ~0. But with init_noise_std=1.0 and action_scale
        # =0.125, every joint target gets +-0.125 rad of RANDOM dither at 50 Hz
        # permanently, and entropy_coef actively pays to keep that noise large.
        # So even a policy whose mean is exactly the PID solution would still be
        # shaking hard -- which is why action_rate_l2 (-1.28) was the single
        # largest cost in the whole reward sum in run16. Shrinking the noise lets
        # the policy actually settle onto the zero-action/PID solution.
        self.policy.init_noise_std = 0.25
        self.algorithm.entropy_coef = 0.002


@configclass
class ChitrakWalkPPORunnerCfg(ChitrakStandPPORunnerCfg):
    """Mild velocity tracking. Inherits the stand runner's fixes."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000  # locomotion is harder than holding a pose
        self.experiment_name = "chitrak_walk"
        self.logger = "wandb"
        self.wandb_project = "chitrak-locomotion"

        # Exploration: a middle value, bracketed by two measured failures.
        #
        # UPPER bound -- Go2's stock 1.0: robot fell in EVERY episode
        #   (base_contact 1.0000, mean episode length 46 of 1000 = under 1 s).
        #   At action_scale=0.25 that is +-0.25 rad of RANDOM dither per joint at
        #   50 Hz. On a 15 kg Go2 with 23.5 Nm motors it is a nudge; on a 1.3 kg
        #   Chitrak with 49 g feet and 2.5 Nm it topples the robot unaided, before
        #   the policy has any say.
        #
        # LOWER bound -- effectively deterministic at 0.05 with entropy_coef=0.0
        #   (run 2026-07-26_15-32-31, 367 iterations): never fell
        #   (base_contact 0.0008, episode length 991) but never learned either.
        #   error_vel_xy settled at 0.3865 against a max command of 0.25 -- i.e.
        #   drifting backwards ~0.19 m/s, WORSE than a statue's 0.20. With no
        #   sampling variation, PPO has no gradient to escape whatever arbitrary
        #   drift the initial random weights produced. Confirmed that the command
        #   range and kernel-width fixes were necessary but not sufficient.
        #
        # Modest increase from BASE (0.2 / 0.002).
        #
        # BASE halved action_scale to 0.125, which ALSO halved exploration measured
        # in joint space -- 0.2 * 0.125 = +-0.025 rad (1.4 deg) of dither, which is
        # tiny. And the noise std COLLAPSED from 0.20 to 0.06 over the BASE run, so
        # entropy_coef=0.002 was not holding it up.
        #
        # Headroom check against the run that toppled the robot at 100% falls:
        # that was init_noise_std=1.0 at action_scale=0.25, i.e. +-0.25 rad. At the
        # current 0.125 scale, matching that would need std=2.0. So 0.4 is +-0.05 rad
        # -- one FIFTH of the toppling amplitude, and double BASE.
        #
        # entropy_coef 0.002 -> 0.005 to keep the std from collapsing again; still
        # half of Go2's 0.01.
        # 0.4 -> 0.8, entropy_coef 0.005 -> 0.01 (Go2's value).
        #
        # The 0.4 run WORKED -- it walks (error_vel_xy 0.0391, Mean reward 49.95,
        # 996/1000 episode length, 0.34% falls at iteration 1000). The remaining
        # problems are a hopping rather than alternating gait, and a back-right
        # hip_roll flare that grows monotonically over training (+0.1 deg at ckpt
        # 50 -> +2.6 deg at ckpt 200) while bl drifts the other way. Both are the
        # signature of a solution that locked in early and could not be escaped.
        #
        # Exploration is decaying exactly when it is needed. init_noise_std is in
        # ACTION units, so the meaningful comparison is after action_scale:
        #
        #                init_noise_std  x action_scale  =  joint dither
        #   Go2               1.00            0.25          +-0.250 rad (14.3 deg)
        #   Chitrak @0.4      0.40            0.125         +-0.050 rad ( 2.9 deg)
        #
        # and the live decay widens it further -- Go2 still dithers +-0.090 rad at
        # iteration 270 (std 0.36) while Chitrak fell to +-0.014 rad (std 0.11) by
        # iteration 813, a 6.4x gap.
        #
        # Headroom: init_noise_std=1.0 at action_scale=0.25 is what toppled the
        # robot at 100% falls -- but that was +-0.25 rad, and reproducing it at
        # today's 0.125 scale would need std=2.0. So 0.8 is +-0.10 rad, still 2.5x
        # BELOW Go2's joint-space amplitude and 2.5x below the toppling point.
        # Sweep so far at action_scale=0.125 (see results/results3.md):
        #   0.2 / 0.002  -> too little: rocks in place, zero net displacement
        #   0.4 / 0.005  -> WORKS: walks, error_vel_xy 0.0391, Mean reward 49.95,
        #                   peak base_contact 0.276, 996/1000 episode length
        #   0.8 / 0.01   -> too much: front feet DRAG, error_vel_xy 0.2768 (7x
        #                   worse), peak base_contact 0.9114 (back to toppling),
        #                   flat_orientation 8x worse, action_rate 15x worse
        #   0.6 / 0.0075 -> also DRAGS: error_vel_xy 0.2994, flat_orientation
        #                   -0.0963 (worst of sweep), peak base_contact 0.6850
        #   0.5 / 0.00625 -> also DRAGS: error_vel_xy 0.4054 (WORST of the sweep),
        #                   peak base_contact 0.5584
        #
        # init_noise_std is SETTLED at 0.4: it is a narrow sweet spot, not the low
        # end of a gradient. 0.5 / 0.6 / 0.8 are all ~7-10x worse on error_vel_xy
        # and none is monotonically ordered, so the differences between them are
        # mostly run-to-run variance -- all three are simply outside the band.
        #
        # Now isolating entropy_coef, holding init_noise_std at the working 0.4.
        # Motivation: run27 (0.4 / 0.005) walks, but its noise std DECAYED 0.40 ->
        # 0.11 over 1000 iterations, and the back-right hip_roll flare locked in
        # between checkpoints 100 and 200 (+0.4 deg -> +2.6 deg) -- i.e. exploration
        # died right when the asymmetry was forming and could no longer be escaped.
        # Raising entropy_coef slows that collapse WITHOUT raising the initial
        # dither amplitude, which is what broke 0.5/0.6/0.8 (joint dither goes as
        # init_noise_std * action_scale; entropy_coef does not touch it).
        # entropy_coef sweep at init_noise_std=0.4 (joint dither fixed at +-0.05 rad):
        #   0.005 -> walks, but HOPPING/bounding gait. noise std collapsed
        #            0.40 -> 0.11, and the back-right hip_roll flare locked in
        #            between checkpoints 100-200 (+0.4 -> +2.6 deg).
        #   0.01  -> NORMAL ALTERNATING GAIT. noise std held at 0.22 instead of
        #            collapsing. error_vel_xy 0.0401, Mean reward 49.59,
        #            1000/1000 episode length, falls 0.5%, flat_orientation
        #            -0.0062. This is the mechanism working as intended: same
        #            dither amplitude, exploration simply survives long enough for
        #            the gait to keep improving past iteration 200.
        #   0.02  -> testing now. Exceeds Go2's 0.01, justified empirically by the
        #            gait change above rather than by matching the reference.
        #
        # entropy_coef is the right knob here precisely because it does NOT touch
        # dither amplitude (that is init_noise_std * action_scale, and raising it to
        # 0.5/0.6/0.8 made the robot drag its feet). It only resists the decay of
        # the std, which is what was killing exploration mid-training.
        #   0.02  -> entropy bonus OUTWEIGHS the policy gradient's pressure to
        #            sharpen, so the std INFLATES instead of merely holding:
        #            0.40 -> 0.78 (1.9x its initial value). That puts live joint
        #            dither at 0.78*0.125 = +-0.098 rad, essentially the +-0.100 rad
        #            that made the robot drag its feet in the init_noise_std=0.8 run.
        #            error_vel_xy 0.0805 (2x worse than 0.01's 0.0401).
        #
        # Convergent finding across both knobs: sustained joint dither above roughly
        # +-0.07 rad breaks the gait, whichever parameter gets it there.
        #
        # NOW TESTING GO2'S LITERAL ORIGINALS for all three:
        #   action_scale 0.25, init_noise_std 1.0, entropy_coef 0.01
        #     -> joint dither +-0.250 rad, 5x the value that works here.
        # This combination was tried once before (2026-07-26_14-47-50) and fell in
        # 100% of episodes -- but that was a materially different config: commands
        # +-0.5 m/s, track std 0.25, Kd 0.2, feet_air_time threshold 0.15. Since
        # then the command range, kernel widths, damping and spawn pose have all
        # changed, so it is a genuinely new test.
        self.policy.init_noise_std = 1.0
        self.algorithm.entropy_coef = 0.01
