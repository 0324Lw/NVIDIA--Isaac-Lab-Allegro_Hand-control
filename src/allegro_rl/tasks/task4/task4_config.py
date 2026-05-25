from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Task4Config:
    """Allegro Hand Task4: Blind Sim2Real / RMA robust reorientation.

    Task objective:
        Robustly reorient an in-hand object under heavy Sim2Real domain
        randomization. The policy-facing student observation is blind: it
        does not directly expose object pose/velocity. Teacher and privileged
        observations are provided for asymmetric training / RMA-style learning.

    Observation design:
        - obs: blind student obs, 108-D
        - teacher_obs: teacher obs, 139-D
        - privileged_obs: teacher_obs + DR vector, 206-D
        - history_obs: 104-D frame x 50 frames = 5200-D

    Project positioning:
        This task is still an educational baseline. It is not a full
        production-grade real robot deployment stack, but it keeps the
        important Sim2Real mechanisms explicit and testable.
    """

    # ===================================================================
    # 1. Basic
    # ===================================================================
    num_envs: int = 2048
    device: str = "cuda:0"
    seed: int = 42

    sim_dt: float = 0.005
    decimation: int = 4
    max_episode_length: int = 300
    env_spacing: float = 1.5

    num_actions: int = 16
    num_joints: int = 16

    # ===================================================================
    # 2. Observation spaces
    # ===================================================================
    # Blind Student obs:
    # q 16 + qd 16 + q_err 16 + raw_action_prev 16
    # + applied_action 16 + motor_effort 16
    # + contact_bools 4 + force_norms 4 + target_quats 4 = 108
    num_observations: int = 108

    # Teacher obs:
    # blind_obs 108
    # + obj_rel 3 + obj_quat 4 + obj_lin 3 + obj_ang 3
    # + theta 1 + axis_err 3 + fingertip_rel 12 + shape_onehot 2 = 139
    num_teacher_obs: int = 139

    # Privileged obs:
    # teacher_obs 139 + DR vector 67 = 206
    num_privileged_obs: int = 206

    # RMA history:
    # q 16 + qd 16 + q_err 16 + applied_action 16 + action_delta 16
    # + motor_effort 16 + contact_bools 4 + force_norms 4 = 104
    history_frame_dim: int = 104
    history_length: int = 50
    history_obs_dim: int = 104 * 50

    # ===================================================================
    # 3. Control / actuator model
    # ===================================================================
    action_scale: float = 0.04
    action_noise_std: float = 0.004
    default_action_alpha: float = 0.55
    max_action_delay_steps: int = 6

    # ===================================================================
    # 4. Object / scene
    # ===================================================================
    hand_init_height: float = 0.50
    object_spawn_height: float = 0.58
    drop_height: float = 0.40
    inactive_object_z: float = -10.0

    cube_size: float = 0.06
    sphere_radius: float = 0.035
    use_mixed_shapes: bool = True

    # ===================================================================
    # 5. Contact / tactile
    # ===================================================================
    fingertip_body_names: List[str] = field(
        default_factory=lambda: [
            "index_link_3",
            "middle_biotac_tip",
            "ring_biotac_tip",
            "thumb_biotac_tip",
        ]
    )

    contact_force_threshold: float = 0.002
    contact_force_clip: float = 20.0
    safe_contact_force: float = 12.0

    # ===================================================================
    # 6. Automatic curriculum schedule
    # ===================================================================
    curriculum_total_steps: int = 300_000_000

    # DR starts at 15% progress and reaches full DR at 75%.
    dr_start_k: float = 0.15
    dr_end_k: float = 0.75

    # Reward transitions from warmup to full between 10% and 65%.
    reward_start_k: float = 0.10
    reward_end_k: float = 0.65

    target_min_angle: float = 0.10
    target_max_angle: float = math.pi

    # ===================================================================
    # 7. Warmup DR
    # ===================================================================
    mass_range_warmup: Tuple[float, float] = (0.08, 0.25)
    friction_range_warmup: Tuple[float, float] = (0.40, 1.50)
    scale_range_warmup: Tuple[float, float] = (0.85, 1.15)
    com_offset_range_warmup: Tuple[float, float] = (-0.008, 0.008)
    inertia_scale_range_warmup: Tuple[float, float] = (0.70, 1.50)

    action_delay_range_warmup: Tuple[int, int] = (0, 3)
    joint_efficiency_range_warmup: Tuple[float, float] = (0.70, 1.10)
    joint_stiffness_scale_range_warmup: Tuple[float, float] = (0.70, 1.30)
    joint_damping_scale_range_warmup: Tuple[float, float] = (0.70, 1.30)
    actuator_deadzone_range_warmup: Tuple[float, float] = (0.00, 0.015)
    action_alpha_range_warmup: Tuple[float, float] = (0.35, 0.75)

    joint_pos_noise_range_warmup: Tuple[float, float] = (0.001, 0.005)
    joint_vel_noise_range_warmup: Tuple[float, float] = (0.005, 0.025)
    tactile_noise_range_warmup: Tuple[float, float] = (0.00, 0.030)
    tactile_dropout_range_warmup: Tuple[float, float] = (0.00, 0.10)
    state_dropout_range_warmup: Tuple[float, float] = (0.00, 0.04)

    slip_velocity_scale_warmup: float = 0.002
    slip_angular_scale_warmup: float = 0.012
    push_probability_warmup: float = 0.0005
    push_delta_v_range_warmup: Tuple[float, float] = (0.002, 0.025)

    # ===================================================================
    # 8. Full Sim2Real DR
    # ===================================================================
    mass_range_full: Tuple[float, float] = (0.05, 0.50)
    friction_range_full: Tuple[float, float] = (0.10, 2.00)
    scale_range_full: Tuple[float, float] = (0.70, 1.30)
    com_offset_range_full: Tuple[float, float] = (-0.015, 0.015)
    inertia_scale_range_full: Tuple[float, float] = (0.50, 2.00)

    action_delay_range_full: Tuple[int, int] = (0, 6)
    joint_efficiency_range_full: Tuple[float, float] = (0.50, 1.20)
    joint_stiffness_scale_range_full: Tuple[float, float] = (0.50, 1.50)
    joint_damping_scale_range_full: Tuple[float, float] = (0.50, 1.50)
    actuator_deadzone_range_full: Tuple[float, float] = (0.00, 0.030)
    action_alpha_range_full: Tuple[float, float] = (0.20, 0.80)

    joint_pos_noise_range_full: Tuple[float, float] = (0.001, 0.010)
    joint_vel_noise_range_full: Tuple[float, float] = (0.005, 0.050)
    tactile_noise_range_full: Tuple[float, float] = (0.00, 0.050)
    tactile_dropout_range_full: Tuple[float, float] = (0.00, 0.20)
    state_dropout_range_full: Tuple[float, float] = (0.00, 0.08)

    slip_velocity_scale_full: float = 0.012
    slip_angular_scale_full: float = 0.080
    push_probability_full: float = 0.004
    push_delta_v_range_full: Tuple[float, float] = (0.020, 0.120)

    disturbance_recovery_window: int = 30

    # ===================================================================
    # 9. Optional direct PhysX randomization
    # ===================================================================
    # Default off for cross-version stability.
    enable_physx_mass_com_write: bool = False
    debug_physx_dr: bool = False

    max_object_linvel: float = 1.20
    max_object_angvel: float = 8.00
    max_joint_vel_abs: float = 25.0
    disturbance_warmup_steps: int = 12

    # ===================================================================
    # 10. Reward warmup -> full
    # ===================================================================
    # 60% mainline: reorientation + progress + contact + keeping object.
    w_rot_warmup: float = 0.55
    w_rot_full: float = 0.55

    w_progress_warmup: float = 0.60
    w_progress_full: float = 0.45

    w_contact_warmup: float = 0.50
    w_contact_full: float = 0.45

    w_center_warmup: float = 1.20
    w_center_full: float = 1.70

    w_height_warmup: float = 1.10
    w_height_full: float = 1.20

    target_height: float = 0.535

    # 20% robust steady state / disturbance recovery.
    w_stable_warmup: float = 0.01
    w_stable_full: float = 0.08

    w_recovery_warmup: float = 0.10
    w_recovery_full: float = 0.22

    bonus_success_warmup: float = 4.0
    bonus_success_full: float = 5.0

    # 20% physical constraints.
    penalty_drop_warmup: float = -20.0
    penalty_drop_full: float = -35.0

    w_action_rate_warmup: float = 0.004
    w_action_rate_full: float = 0.006

    w_action_mag_warmup: float = 0.0012
    w_action_mag_full: float = 0.0020

    w_joint_vel_warmup: float = 0.002
    w_joint_vel_full: float = 0.002

    w_energy_warmup: float = 0.0025
    w_energy_full: float = 0.004

    w_torque_spike_warmup: float = 0.006
    w_torque_spike_full: float = 0.006

    w_contact_force_spike_warmup: float = 0.010
    w_contact_force_spike_full: float = 0.010

    # ===================================================================
    # 11. Numeric stability
    # ===================================================================
    continuous_reward_clip: float = 1.0
    episode_return_abs_limit: float = 200.0
    success_hold_frames: int = 10

    actor_obs_clip: float = 50.0
    teacher_obs_clip: float = 50.0
    privileged_obs_clip: float = 80.0
    history_obs_clip: float = 50.0

    # ===================================================================
    # 12. Debug
    # ===================================================================
    debug_print_names: bool = True

    # ===================================================================
    # 13. Backward-compatible aliases as properties
    # ===================================================================
    @property
    def control_dt(self) -> float:
        return float(self.sim_dt) * int(self.decimation)

    @property
    def mass_range(self) -> Tuple[float, float]:
        return self.mass_range_warmup

    @property
    def friction_range(self) -> Tuple[float, float]:
        return self.friction_range_warmup

    @property
    def scale_range(self) -> Tuple[float, float]:
        return self.scale_range_warmup

    @property
    def com_offset_range(self) -> Tuple[float, float]:
        return self.com_offset_range_warmup

    @property
    def inertia_scale_range(self) -> Tuple[float, float]:
        return self.inertia_scale_range_warmup

    @property
    def action_delay_range(self) -> Tuple[int, int]:
        return self.action_delay_range_warmup

    @property
    def joint_efficiency_range(self) -> Tuple[float, float]:
        return self.joint_efficiency_range_warmup

    @property
    def joint_stiffness_scale_range(self) -> Tuple[float, float]:
        return self.joint_stiffness_scale_range_warmup

    @property
    def joint_damping_scale_range(self) -> Tuple[float, float]:
        return self.joint_damping_scale_range_warmup

    @property
    def actuator_deadzone_range(self) -> Tuple[float, float]:
        return self.actuator_deadzone_range_warmup

    @property
    def action_alpha_range(self) -> Tuple[float, float]:
        return self.action_alpha_range_warmup

    @property
    def joint_pos_noise_range(self) -> Tuple[float, float]:
        return self.joint_pos_noise_range_warmup

    @property
    def joint_vel_noise_range(self) -> Tuple[float, float]:
        return self.joint_vel_noise_range_warmup

    @property
    def tactile_noise_range(self) -> Tuple[float, float]:
        return self.tactile_noise_range_warmup

    @property
    def tactile_dropout_range(self) -> Tuple[float, float]:
        return self.tactile_dropout_range_warmup

    @property
    def state_dropout_range(self) -> Tuple[float, float]:
        return self.state_dropout_range_warmup

    @property
    def slip_velocity_scale(self) -> float:
        return self.slip_velocity_scale_warmup

    @property
    def slip_angular_scale(self) -> float:
        return self.slip_angular_scale_warmup

    @property
    def push_probability(self) -> float:
        return self.push_probability_warmup

    @property
    def push_delta_v_range(self) -> Tuple[float, float]:
        return self.push_delta_v_range_warmup

    @property
    def w_rot(self) -> float:
        return self.w_rot_warmup

    @property
    def w_progress(self) -> float:
        return self.w_progress_warmup

    @property
    def w_contact(self) -> float:
        return self.w_contact_warmup

    @property
    def w_center(self) -> float:
        return self.w_center_warmup

    @property
    def w_height(self) -> float:
        return self.w_height_warmup

    @property
    def w_stable(self) -> float:
        return self.w_stable_warmup

    @property
    def w_recovery(self) -> float:
        return self.w_recovery_warmup

    @property
    def bonus_success(self) -> float:
        return self.bonus_success_warmup

    @property
    def penalty_drop(self) -> float:
        return self.penalty_drop_warmup

    @property
    def w_action_rate(self) -> float:
        return self.w_action_rate_warmup

    @property
    def w_action_mag(self) -> float:
        return self.w_action_mag_warmup

    @property
    def w_joint_vel(self) -> float:
        return self.w_joint_vel_warmup

    @property
    def w_energy(self) -> float:
        return self.w_energy_warmup

    @property
    def w_torque_spike(self) -> float:
        return self.w_torque_spike_warmup

    @property
    def w_contact_force_spike(self) -> float:
        return self.w_contact_force_spike_warmup

    # ===================================================================
    # Validation
    # ===================================================================
    def validate(self) -> None:
        assert self.num_envs > 0
        assert self.device in ("cpu", "cuda", "cuda:0") or self.device.startswith("cuda")

        assert self.sim_dt > 0.0
        assert self.decimation >= 1
        assert self.control_dt > 0.0
        assert self.max_episode_length > 0
        assert self.env_spacing > 0.0

        assert self.num_actions == 16
        assert self.num_joints == 16

        assert self.num_observations == 108
        assert self.num_teacher_obs == 139
        assert self.num_privileged_obs == 206

        assert self.history_frame_dim == 104
        assert self.history_length == 50
        assert self.history_obs_dim == self.history_frame_dim * self.history_length

        assert self.action_scale > 0.0
        assert self.action_noise_std >= 0.0
        assert 0.0 <= self.default_action_alpha <= 1.0
        assert self.max_action_delay_steps >= 0

        assert self.object_spawn_height > self.drop_height
        assert self.cube_size > 0.0
        assert self.sphere_radius > 0.0

        assert len(self.fingertip_body_names) == 4
        assert self.contact_force_threshold >= 0.0
        assert self.contact_force_clip > 0.0
        assert self.safe_contact_force > 0.0

        assert self.curriculum_total_steps > 0
        assert 0.0 <= self.dr_start_k <= self.dr_end_k <= 1.0
        assert 0.0 <= self.reward_start_k <= self.reward_end_k <= 1.0
        assert self.target_min_angle > 0.0
        assert self.target_max_angle >= self.target_min_angle

        self._validate_range("mass_range_warmup", self.mass_range_warmup, positive=True)
        self._validate_range("mass_range_full", self.mass_range_full, positive=True)
        self._validate_range("friction_range_warmup", self.friction_range_warmup, positive=True)
        self._validate_range("friction_range_full", self.friction_range_full, positive=True)

        self._validate_range("scale_range_warmup", self.scale_range_warmup, positive=True)
        self._validate_range("scale_range_full", self.scale_range_full, positive=True)
        self._validate_range("com_offset_range_warmup", self.com_offset_range_warmup, allow_negative=True)
        self._validate_range("com_offset_range_full", self.com_offset_range_full, allow_negative=True)
        self._validate_range("inertia_scale_range_warmup", self.inertia_scale_range_warmup, positive=True)
        self._validate_range("inertia_scale_range_full", self.inertia_scale_range_full, positive=True)

        assert self.action_delay_range_warmup[0] >= 0
        assert self.action_delay_range_warmup[1] >= self.action_delay_range_warmup[0]
        assert self.action_delay_range_full[0] >= 0
        assert self.action_delay_range_full[1] >= self.action_delay_range_full[0]
        assert self.action_delay_range_full[1] <= self.max_action_delay_steps

        self._validate_range("joint_efficiency_range_warmup", self.joint_efficiency_range_warmup, positive=True)
        self._validate_range("joint_efficiency_range_full", self.joint_efficiency_range_full, positive=True)
        self._validate_range("joint_stiffness_scale_range_warmup", self.joint_stiffness_scale_range_warmup, positive=True)
        self._validate_range("joint_stiffness_scale_range_full", self.joint_stiffness_scale_range_full, positive=True)
        self._validate_range("joint_damping_scale_range_warmup", self.joint_damping_scale_range_warmup, positive=True)
        self._validate_range("joint_damping_scale_range_full", self.joint_damping_scale_range_full, positive=True)

        self._validate_range("actuator_deadzone_range_warmup", self.actuator_deadzone_range_warmup)
        self._validate_range("actuator_deadzone_range_full", self.actuator_deadzone_range_full)
        self._validate_range("action_alpha_range_warmup", self.action_alpha_range_warmup)
        self._validate_range("action_alpha_range_full", self.action_alpha_range_full)

        assert 0.0 <= self.action_alpha_range_warmup[0] <= self.action_alpha_range_warmup[1] <= 1.0
        assert 0.0 <= self.action_alpha_range_full[0] <= self.action_alpha_range_full[1] <= 1.0

        self._validate_range("joint_pos_noise_range_warmup", self.joint_pos_noise_range_warmup)
        self._validate_range("joint_pos_noise_range_full", self.joint_pos_noise_range_full)
        self._validate_range("joint_vel_noise_range_warmup", self.joint_vel_noise_range_warmup)
        self._validate_range("joint_vel_noise_range_full", self.joint_vel_noise_range_full)
        self._validate_range("tactile_noise_range_warmup", self.tactile_noise_range_warmup)
        self._validate_range("tactile_noise_range_full", self.tactile_noise_range_full)
        self._validate_range("tactile_dropout_range_warmup", self.tactile_dropout_range_warmup)
        self._validate_range("tactile_dropout_range_full", self.tactile_dropout_range_full)
        self._validate_range("state_dropout_range_warmup", self.state_dropout_range_warmup)
        self._validate_range("state_dropout_range_full", self.state_dropout_range_full)

        assert 0.0 <= self.tactile_dropout_range_warmup[0] <= self.tactile_dropout_range_warmup[1] <= 1.0
        assert 0.0 <= self.tactile_dropout_range_full[0] <= self.tactile_dropout_range_full[1] <= 1.0
        assert 0.0 <= self.state_dropout_range_warmup[0] <= self.state_dropout_range_warmup[1] <= 1.0
        assert 0.0 <= self.state_dropout_range_full[0] <= self.state_dropout_range_full[1] <= 1.0

        assert self.slip_velocity_scale_warmup >= 0.0
        assert self.slip_velocity_scale_full >= 0.0
        assert self.slip_angular_scale_warmup >= 0.0
        assert self.slip_angular_scale_full >= 0.0
        assert self.push_probability_warmup >= 0.0
        assert self.push_probability_full >= 0.0
        self._validate_range("push_delta_v_range_warmup", self.push_delta_v_range_warmup)
        self._validate_range("push_delta_v_range_full", self.push_delta_v_range_full)

        assert self.disturbance_recovery_window > 0
        assert self.max_object_linvel > 0.0
        assert self.max_object_angvel > 0.0
        assert self.max_joint_vel_abs > 0.0
        assert self.disturbance_warmup_steps >= 0

        assert self.continuous_reward_clip > 0.0
        assert self.episode_return_abs_limit > 0.0
        assert self.success_hold_frames > 0

        assert self.actor_obs_clip > 0.0
        assert self.teacher_obs_clip > 0.0
        assert self.privileged_obs_clip > 0.0
        assert self.history_obs_clip > 0.0

    @staticmethod
    def _validate_range(
        name: str,
        value: Tuple[float, float],
        positive: bool = False,
        allow_negative: bool = False,
    ) -> None:
        assert len(value) == 2, f"{name} must be a tuple/list of length 2"
        lo, hi = float(value[0]), float(value[1])
        assert hi >= lo, f"{name} invalid: hi < lo"
        if positive:
            assert lo > 0.0 and hi > 0.0, f"{name} must be positive"
        if not allow_negative and not positive:
            assert lo >= 0.0, f"{name} lower bound must be >= 0"


Task4ConfigAlias = Task4Config
