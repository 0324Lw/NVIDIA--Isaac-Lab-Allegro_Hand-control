from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Task3Config:
    """Allegro Hand Task3: dynamic grasping and tool use.

    Objective:
        Use Allegro Hand with a controlled floating base to approach a tool,
        form stable contacts, lift it from the table, and optionally align
        its orientation according to curriculum stage.

    Action:
        22-D action:
            16 hand residual joint commands
             6 floating-base twist-like residual commands

    Observation:
        Actor obs dim = 147
        Privileged obs dim = 168

    Project positioning:
        This is an educational pure-RL baseline. It records the experience of
        building contact-rich dexterous manipulation tasks with reward shaping.
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
    env_spacing: float = 1.3

    # ----------------------------- Spaces -----------------------------
    num_hand_actions: int = 16
    num_base_actions: int = 6
    num_actions: int = 22

    num_observations: int = 147
    num_privileged_obs: int = 168
    frame_stack: int = 5
    stacked_actor_obs_dim: int = 147 * 5
    stacked_critic_obs_dim: int = 168 * 5

    # ----------------------------- Table / tools -----------------------------
    table_height: float = 0.0
    table_size_xy: float = 0.8
    table_thickness: float = 0.04

    use_only_pen: bool = True
    pen_size: Tuple[float, float, float] = (0.14, 0.016, 0.016)
    cup_size: Tuple[float, float, float] = (0.07, 0.07, 0.09)

    object_spawn_xy_range: float = 0.16
    object_yaw_random: bool = True
    inactive_object_z: float = -10.0

    # ----------------------------- Floating base control -----------------------------
    base_init_z: float = 0.42
    base_pregrasp_height: float = 0.10

    base_workspace_low: Tuple[float, float, float] = (-0.32, -0.32, 0.16)
    base_workspace_high: Tuple[float, float, float] = (0.32, 0.32, 0.62)
    base_max_tilt_rad: float = 0.75

    hand_action_scale: float = 0.05
    base_xyz_scale: Tuple[float, float, float] = (0.008, 0.008, 0.006)
    base_rot_scale: float = 0.035

    ema_hand: float = 0.60
    ema_base: float = 0.28

    # ----------------------------- Fingertips / tactile contact -----------------------------
    fingertip_body_names: Tuple[str, str, str, str] = (
        "index_link_3",
        "middle_biotac_tip",
        "ring_biotac_tip",
        "thumb_biotac_tip",
    )
    palm_body_name: str = "palm_link"

    contact_force_threshold: float = 0.0003
    contact_distance_gate: float = 0.28
    contact_force_clip: float = 15.0
    safe_contact_force: float = 10.0

    # ----------------------------- Curriculum -----------------------------
    curriculum_total_steps: int = 300_000_000
    phase_thresholds: Tuple[float, float, float, float, float, float] = (
        0.04,
        0.12,
        0.25,
        0.42,
        0.62,
        0.82,
    )

    lift_heights_by_phase: Tuple[float, float, float, float, float, float, float] = (
        0.00,
        0.00,
        0.02,
        0.05,
        0.10,
        0.12,
        0.14,
    )

    orient_max_angle_by_phase: Tuple[float, float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.15,
        0.50,
        1.20,
    )

    success_hold_frames: int = 12

    # ----------------------------- Reward weights: approach / grasp -----------------------------
    w_approach: float = 0.04
    w_reach_progress: float = 0.03
    w_pregrasp: float = 0.30
    w_contact: float = 2.50
    w_force_closure: float = 1.20
    w_grip: float = 0.90

    w_descend_to_contact: float = 1.20
    w_no_contact_stall: float = 0.35
    w_base_escape: float = 0.40
    w_far_from_object: float = 0.08
    w_soft_contact: float = 0.35

    # ----------------------------- Reward weights: lift / orientation / stability -----------------------------
    w_lift: float = 0.15
    w_lift_progress: float = 0.10
    w_orient: float = 0.08
    w_axis: float = 0.05
    w_stable: float = 0.015
    w_hover: float = 0.008

    # ----------------------------- Penalties -----------------------------
    w_base_jerk: float = 0.003
    w_base_rate: float = 0.0015
    w_action_rate: float = 0.0015
    w_action_mag: float = 0.0006
    w_joint_vel: float = 0.0006
    w_excess_force: float = 0.010
    w_workspace_violation: float = 0.20

    penalty_drop: float = -35.0
    penalty_slide_out: float = -25.0
    w_table_crash: float = 5.0
    bonus_success: float = 4.0

    # ----------------------------- Numerical guards -----------------------------
    continuous_reward_clip: float = 1.0
    episode_return_abs_limit: float = 200.0

    actor_obs_clip: float = 30.0
    privileged_obs_clip: float = 50.0

    # ----------------------------- Domain randomization buffers -----------------------------
    mass_range: Tuple[float, float] = (0.06, 0.18)
    object_friction_range: Tuple[float, float] = (0.60, 1.20)
    table_friction_range: Tuple[float, float] = (0.60, 1.20)
    com_offset_std: float = 0.004

    # ----------------------------- Debug -----------------------------
    debug_print_names: bool = True

    def validate(self) -> None:
        assert self.num_envs > 0
        assert self.device in ("cpu", "cuda", "cuda:0") or self.device.startswith("cuda")

        assert self.sim_dt > 0.0
        assert self.decimation >= 1
        assert self.control_dt > 0.0
        assert self.max_episode_length > 0
        assert self.env_spacing > 0.0

        assert self.num_hand_actions == 16
        assert self.num_base_actions == 6
        assert self.num_actions == self.num_hand_actions + self.num_base_actions
        assert self.num_actions == 22

        assert self.num_observations == 147
        assert self.num_privileged_obs == 168
        assert self.frame_stack >= 1
        assert self.stacked_actor_obs_dim == self.num_observations * self.frame_stack
        assert self.stacked_critic_obs_dim == self.num_privileged_obs * self.frame_stack

        assert self.table_size_xy > 0.0
        assert self.table_thickness > 0.0
        assert len(self.pen_size) == 3 and all(v > 0.0 for v in self.pen_size)
        assert len(self.cup_size) == 3 and all(v > 0.0 for v in self.cup_size)
        assert self.object_spawn_xy_range >= 0.0

        assert self.base_init_z > 0.0
        assert self.base_pregrasp_height >= 0.0
        assert len(self.base_workspace_low) == 3
        assert len(self.base_workspace_high) == 3
        assert all(lo < hi for lo, hi in zip(self.base_workspace_low, self.base_workspace_high))
        assert self.base_max_tilt_rad > 0.0

        assert self.hand_action_scale > 0.0
        assert len(self.base_xyz_scale) == 3 and all(v > 0.0 for v in self.base_xyz_scale)
        assert self.base_rot_scale > 0.0
        assert 0.0 <= self.ema_hand <= 1.0
        assert 0.0 <= self.ema_base <= 1.0

        assert len(self.fingertip_body_names) == 4
        assert isinstance(self.palm_body_name, str) and len(self.palm_body_name) > 0
        assert self.contact_force_threshold >= 0.0
        assert self.contact_distance_gate > 0.0
        assert self.contact_force_clip > 0.0
        assert self.safe_contact_force > 0.0

        assert self.curriculum_total_steps > 0
        assert len(self.phase_thresholds) == 6
        assert all(0.0 <= v <= 1.0 for v in self.phase_thresholds)
        assert all(self.phase_thresholds[i] <= self.phase_thresholds[i + 1] for i in range(len(self.phase_thresholds) - 1))
        assert len(self.lift_heights_by_phase) == 7
        assert len(self.orient_max_angle_by_phase) == 7
        assert self.success_hold_frames > 0

        assert self.continuous_reward_clip > 0.0
        assert self.episode_return_abs_limit > 0.0

        assert self.mass_range[0] > 0.0 and self.mass_range[1] > self.mass_range[0]
        assert self.object_friction_range[0] > 0.0 and self.object_friction_range[1] > self.object_friction_range[0]
        assert self.table_friction_range[0] > 0.0 and self.table_friction_range[1] > self.table_friction_range[0]
        assert self.com_offset_std >= 0.0


Task3ConfigAlias = Task3Config
