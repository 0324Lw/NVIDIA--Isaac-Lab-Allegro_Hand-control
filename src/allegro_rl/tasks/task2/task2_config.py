from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Task2Config:
    """Allegro Hand Task2: in-hand object reorientation.

    Objective:
        Keep an object in the palm and rotate it toward a target quaternion.

    Observation:
        Actor obs dim = 83
        Privileged obs dim = 88

    Project positioning:
        This is an educational pure-RL baseline. High-quality dexterous
        manipulation may benefit from demonstrations / mocap, but this task
        preserves a pure-RL reorientation baseline.
    """

    # ----------------------------- Runtime -----------------------------
    num_envs: int = 4096
    device: str = "cuda:0"
    seed: int = 42

    # ----------------------------- Isaac / sim -----------------------------
    sim_dt: float = 0.005
    decimation: int = 4
    control_dt: float = 0.020
    max_episode_length: int = 300
    env_spacing: float = 1.5

    hand_init_height: float = 0.50
    object_spawn_height: float = 0.54
    inactive_object_z: float = -10.0

    # ----------------------------- Spaces -----------------------------
    num_actions: int = 16
    num_observations: int = 83
    num_privileged_obs: int = 88
    frame_stack: int = 5
    stacked_actor_obs_dim: int = 83 * 5

    # ----------------------------- Control -----------------------------
    ema_alpha: float = 0.60
    action_scale: float = 0.05

    # ----------------------------- Object / contact -----------------------------
    use_only_cube: bool = True
    cube_size: float = 0.06
    sphere_radius: float = 0.035

    contact_force_threshold: float = 0.02

    fingertip_body_names: Tuple[str, str, str, str] = (
        "index_link_3",
        "middle_biotac_tip",
        "ring_biotac_tip",
        "thumb_biotac_tip",
    )

    # ----------------------------- Domain randomization -----------------------------
    mass_range: Tuple[float, float] = (0.05, 0.20)
    friction_range: Tuple[float, float] = (0.60, 1.20)
    com_offset_std: float = 0.005

    # ----------------------------- Curriculum -----------------------------
    curriculum_total_steps: int = 200_000_000
    min_target_angle: float = 0.10
    max_target_angle: float = 3.141592653589793

    # ----------------------------- Success / termination -----------------------------
    success_theta_threshold: float = 0.15
    success_lin_vel_threshold: float = 0.50
    success_ang_vel_threshold: float = 0.50
    success_min_height: float = 0.48
    success_xy_radius: float = 0.08
    success_min_contacts: float = 1.0

    drop_height: float = 0.40
    pre_drop_height: float = 0.45

    # ----------------------------- Reward weights -----------------------------
    w_rot: float = 0.55
    w_prog: float = 0.15
    w_contact: float = 0.14

    w_safe: float = 1.50
    w_height: float = 0.70
    target_height: float = 0.535
    bonus_success: float = 2.0

    penalty_drop: float = -28.0
    w_pre_drop: float = 3.50

    w_excess_vel: float = 0.022
    w_act_rate: float = 0.004
    w_joint_vel: float = 0.002

    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0

    # ----------------------------- Safety / debug -----------------------------
    actor_obs_clip: float = 20.0
    privileged_obs_clip: float = 30.0
    debug_print_names: bool = True

    def validate(self) -> None:
        assert self.num_envs > 0
        assert self.device in ("cpu", "cuda", "cuda:0") or self.device.startswith("cuda")

        assert self.sim_dt > 0.0
        assert self.decimation >= 1
        assert self.control_dt > 0.0
        assert self.max_episode_length > 0
        assert self.env_spacing > 0.0

        assert self.num_actions == 16
        assert self.num_observations == 83
        assert self.num_privileged_obs == 88
        assert self.stacked_actor_obs_dim == self.num_observations * self.frame_stack

        assert 0.0 <= self.ema_alpha <= 1.0
        assert self.action_scale > 0.0

        assert self.cube_size > 0.0
        assert self.sphere_radius > 0.0
        assert self.object_spawn_height > 0.0
        assert self.contact_force_threshold >= 0.0
        assert len(self.fingertip_body_names) == 4

        assert self.mass_range[0] > 0.0 and self.mass_range[1] > self.mass_range[0]
        assert self.friction_range[0] > 0.0 and self.friction_range[1] > self.friction_range[0]
        assert self.com_offset_std >= 0.0

        assert self.curriculum_total_steps > 0
        assert self.min_target_angle > 0.0
        assert self.max_target_angle >= self.min_target_angle

        assert self.success_theta_threshold > 0.0
        assert self.success_lin_vel_threshold > 0.0
        assert self.success_ang_vel_threshold > 0.0
        assert self.success_min_height > self.drop_height
        assert self.success_xy_radius > 0.0
        assert self.success_min_contacts >= 0.0

        assert self.drop_height < self.pre_drop_height
        assert self.reward_clip_min < self.reward_clip_max


Task2ConfigAlias = Task2Config
