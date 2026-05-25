from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor

from allegro_rl.tasks.task3.task3_config import Task3Config
from allegro_rl.tasks.task3.task3_scene import make_allegro_task3_scene_cfg


# ======================================================================
# Quaternion / geometry utilities
# ======================================================================

def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-8)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q.unbind(-1)
    w2, x2, y2, z2 = r.unbind(-1)

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)

    if q.dim() == v.dim() - 1:
        q = q.unsqueeze(-2)

    q = torch.broadcast_to(q, v.shape[:-1] + (4,))
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)

    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


def quat_from_rotvec(rv: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(rv, dim=-1, keepdim=True)
    axis = rv / torch.clamp(angle, min=1e-8)
    half = 0.5 * angle

    return quat_normalize(torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1))


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.cos(yaw * 0.5),
            torch.zeros_like(yaw),
            torch.zeros_like(yaw),
            torch.sin(yaw * 0.5),
        ],
        dim=-1,
    )


def so3_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = torch.clamp(torch.abs(torch.sum(q1 * q2, dim=-1)), 0.0, 1.0)
    return 2.0 * torch.acos(dot)


def sigmoid_gate(x: torch.Tensor, sharpness: float = 8.0) -> torch.Tensor:
    return torch.sigmoid(float(sharpness) * x)


# ======================================================================
# Environment
# ======================================================================

class AllegroHandTask3Env(gym.Env):
    """Allegro Hand Task3: dynamic grasping and tool use.

    The environment is fully independent from Task1 / Task2. It exposes
    actor obs and privileged critic obs in a dict, and controls the Allegro
    hand by joint position targets plus a scripted floating-base root pose.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task3Config):
        super().__init__()

        cfg.validate()
        self.cfg = cfg
        self.device = str(cfg.device)

        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(cfg.num_actions),),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(int(cfg.num_observations),),
                    dtype=np.float32,
                ),
                "privileged_obs": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(int(cfg.num_privileged_obs),),
                    dtype=np.float32,
                ),
            }
        )

        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(cfg.num_privileged_obs),),
            dtype=np.float32,
        )

        sim_cfg = sim_utils.SimulationCfg(
            dt=float(cfg.sim_dt),
            device=str(cfg.device),
            physx=sim_utils.PhysxCfg(
                enable_external_forces_every_iteration=True,
                min_position_iteration_count=4,
                max_position_iteration_count=8,
                min_velocity_iteration_count=1,
                max_velocity_iteration_count=2,
            ),
        )

        self.sim = sim_utils.SimulationContext(sim_cfg)

        SceneCfg = make_allegro_task3_scene_cfg(cfg)
        self.scene = InteractiveScene(SceneCfg(num_envs=int(cfg.num_envs)))

        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.pen: RigidObject = self.scene["pen"]
        self.cup: RigidObject = self.scene["cup"]
        self.contact_sensor: ContactSensor = self.scene["hand_contact"]

        self.num_envs = int(cfg.num_envs)
        self.num_actions = int(cfg.num_actions)
        self.num_observations = int(cfg.num_observations)
        self.num_privileged_obs = int(cfg.num_privileged_obs)

        if self.robot.num_joints != int(cfg.num_hand_actions):
            raise RuntimeError(
                f"Allegro Task3 expects robot.num_joints == {cfg.num_hand_actions}, "
                f"but got {self.robot.num_joints}. joint_names={self.robot.joint_names}"
            )

        self.robot_joint_names = list(self.robot.joint_names)
        self.robot_body_names = list(self.robot.body_names)
        self.contact_body_names = list(self.contact_sensor.body_names)

        self.global_steps = 0
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # 0 = pen, 1 = cup
        self.active_tool = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.curriculum_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.base_pos_rel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.base_quat[:, 0] = 1.0
        self.base_lin_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self.u_hand_prev = torch.zeros((self.num_envs, cfg.num_hand_actions), dtype=torch.float32, device=self.device)
        self.u_base_prev = torch.zeros((self.num_envs, cfg.num_base_actions), dtype=torch.float32, device=self.device)
        self.u_base_prev2 = torch.zeros_like(self.u_base_prev)

        self.prev_tcp_dist = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_lift_amount = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.target_height = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.target_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.target_quat[:, 0] = 1.0

        self.dr_mass = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.dr_obj_friction = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.dr_table_friction = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.dr_com_offset = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.dr_inertia_diag = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self.base_was_clamped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.debug_soft_contact = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.debug_hard_contact = torch.zeros_like(self.debug_soft_contact)
        self.debug_dist_score = torch.zeros_like(self.debug_soft_contact)
        self.debug_force_score = torch.zeros_like(self.debug_soft_contact)

        self.total_done_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_drop_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_slide_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_crash_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_success_count = torch.zeros((), dtype=torch.float32, device=self.device)

        self.joint_limits = self.robot.data.joint_pos_limits.clone()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()

        self.palm_index = self._name_to_index(
            cfg.palm_body_name,
            self.robot_body_names,
            "robot.body_names",
            allow_fallback=True,
        )

        self.fingertip_indices = self._names_to_indices(
            list(cfg.fingertip_body_names),
            self.robot_body_names,
            "robot.body_names",
            allow_fallback=True,
        )

        self.contact_tip_indices = torch.tensor(
            self._names_to_indices(
                list(cfg.fingertip_body_names),
                self.contact_body_names,
                "contact_sensor.body_names",
                allow_fallback=True,
            ),
            dtype=torch.long,
            device=self.device,
        )

        if bool(cfg.debug_print_names):
            self._print_debug_names()

        self.reset()

    # ------------------------------------------------------------------
    # Name mapping
    # ------------------------------------------------------------------
    def _print_debug_names(self) -> None:
        print("\n" + "=" * 120)
        print("[AllegroHandTask3Env] selected body/contact mapping")
        print("=" * 120)
        print(f"[DEBUG] palm index: {self.palm_index} -> {self.robot_body_names[self.palm_index]}")
        print("[DEBUG] requested fingertips:", list(self.cfg.fingertip_body_names))
        print("[DEBUG] robot fingertip indices:", self.fingertip_indices)
        print("[DEBUG] contact fingertip indices:", self.contact_tip_indices.tolist())
        print("[DEBUG] selected robot fingertip names:")
        for idx in self.fingertip_indices:
            print(f"  robot[{idx:02d}] = {self.robot_body_names[idx]}")
        print("[DEBUG] selected contact fingertip names:")
        for idx in self.contact_tip_indices.tolist():
            print(f"  contact[{idx:02d}] = {self.contact_body_names[idx]}")
        print("=" * 120 + "\n")

    @staticmethod
    def _finger_keyword(name: str) -> str:
        lower = name.lower()
        if "index" in lower or "ff" in lower:
            return "index"
        if "middle" in lower or "mf" in lower:
            return "middle"
        if "ring" in lower or "rf" in lower:
            return "ring"
        if "thumb" in lower or "th" in lower:
            return "thumb"
        if "palm" in lower:
            return "palm"
        return lower

    def _fallback_find_body(self, requested_name: str, source: list[str]) -> Optional[int]:
        keyword = self._finger_keyword(requested_name)

        candidates = []
        for i, name in enumerate(source):
            lower = name.lower()
            if keyword not in lower:
                continue

            score = 0
            if keyword == "palm":
                if "palm" in lower:
                    score += 20
            else:
                if "tip" in lower:
                    score += 10
                if "biotac" in lower:
                    score += 8
                if "link_3" in lower or "link3" in lower:
                    score += 6
                if "joint" not in lower:
                    score += 1

            candidates.append((score, i, name))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return int(candidates[0][1])

    def _name_to_index(
        self,
        name: str,
        source: list[str],
        source_name: str,
        allow_fallback: bool = True,
    ) -> int:
        if name in source:
            return int(source.index(name))

        fallback_idx = self._fallback_find_body(name, source) if allow_fallback else None
        if fallback_idx is not None:
            return int(fallback_idx)

        raise RuntimeError(
            f"[Task3 Name Error] {name} not found in {source_name}.\n"
            f"Available names:\n{source}"
        )

    def _names_to_indices(
        self,
        names: list[str],
        source: list[str],
        source_name: str,
        allow_fallback: bool = True,
    ) -> list[int]:
        out = [
            self._name_to_index(name, source, source_name, allow_fallback=allow_fallback)
            for name in names
        ]

        if len(set(out)) != len(out):
            raise RuntimeError(
                f"[Task3 Name Error] duplicated indices from {source_name}: {out}\n"
                f"Requested names: {names}\nAvailable names: {source}"
            )

        return out

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _env_origins(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        origins = self.scene.env_origins.to(self.device)
        return origins if env_ids is None else origins[env_ids]

    def _identity_quat(self, n: int) -> torch.Tensor:
        q = torch.zeros((int(n), 4), dtype=torch.float32, device=self.device)
        q[:, 0] = 1.0
        return q

    def _zero_vel6(self, n: int) -> torch.Tensor:
        return torch.zeros((int(n), 6), dtype=torch.float32, device=self.device)

    def _get_phase(self) -> int:
        k = min(1.0, float(self.global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0))
        return int(sum(k >= th for th in self.cfg.phase_thresholds))

    def _tool_half_extents(self) -> torch.Tensor:
        pen_half = torch.tensor(self.cfg.pen_size, dtype=torch.float32, device=self.device) * 0.5
        cup_half = torch.tensor(self.cfg.cup_size, dtype=torch.float32, device=self.device) * 0.5
        return torch.where((self.active_tool == 0).unsqueeze(-1), pen_half, cup_half)

    def _rest_center_height(self) -> torch.Tensor:
        return float(self.cfg.table_height) + self._tool_half_extents()[:, 2]

    def _get_active_tool_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        is_pen = (self.active_tool == 0).unsqueeze(-1)

        pos = torch.where(is_pen, self.pen.data.root_pos_w, self.cup.data.root_pos_w)
        quat = torch.where(is_pen, self.pen.data.root_quat_w, self.cup.data.root_quat_w)
        lin_vel = torch.where(is_pen, self.pen.data.root_lin_vel_w, self.cup.data.root_lin_vel_w)
        ang_vel = torch.where(is_pen, self.pen.data.root_ang_vel_w, self.cup.data.root_ang_vel_w)

        return pos, quat_normalize(quat), lin_vel, ang_vel

    def _get_tcp_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.palm_index, :]

    def _get_fingertip_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.fingertip_indices, :]

    def _get_tool_keypoints_w(self) -> torch.Tensor:
        obj_pos, obj_quat, _, _ = self._get_active_tool_state()

        pen_l = float(self.cfg.pen_size[0]) * 0.5
        pen_r = float(self.cfg.pen_size[1]) * 0.5
        cup_x = float(self.cfg.cup_size[0]) * 0.5
        cup_y = float(self.cfg.cup_size[1]) * 0.5
        cup_z = float(self.cfg.cup_size[2]) * 0.5

        pen_local = torch.tensor(
            [
                [pen_l, 0.0, 0.0],
                [-pen_l, 0.0, 0.0],
                [0.0, pen_r, 0.0],
                [0.0, -pen_r, 0.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        cup_local = torch.tensor(
            [
                [cup_x, 0.0, 0.0],
                [-cup_x, 0.0, 0.0],
                [0.0, cup_y + 0.035, 0.0],
                [0.0, 0.0, cup_z],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        local = torch.where(
            (self.active_tool == 0).view(self.num_envs, 1, 1),
            pen_local.unsqueeze(0),
            cup_local.unsqueeze(0),
        )

        return obj_pos.unsqueeze(1) + quat_rotate(obj_quat.unsqueeze(1), local)

    def _get_target_axis(self) -> torch.Tensor:
        axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        axis = axis.repeat(self.num_envs, 1)
        return quat_rotate(self.target_quat, axis)

    def _get_tactile(self):
        forces = self.contact_sensor.data.net_forces_w_history

        if forces.ndim != 4:
            raise RuntimeError(f"[Task3 Contact Error] expected [N,H,B,3], got {tuple(forces.shape)}")

        tip_force_hist = forces[:, :, self.contact_tip_indices, :]
        force_norm_hist = torch.norm(tip_force_hist, dim=-1)

        peak_ids = torch.argmax(force_norm_hist, dim=1)
        gather_ids = peak_ids[:, None, :, None].expand(-1, 1, -1, 3)
        peak_force = torch.gather(tip_force_hist, dim=1, index=gather_ids).squeeze(1)

        force_norm = torch.norm(peak_force, dim=-1)

        obj_pos, _, _, _ = self._get_active_tool_state()
        tip_pos = self._get_fingertip_pos_w()

        tip_to_obj = torch.norm(tip_pos - obj_pos.unsqueeze(1), dim=-1)
        above_table = tip_pos[..., 2] > (
            self._env_origins()[:, None, 2] + float(self.cfg.table_height) + 0.004
        )

        dist_score = torch.clamp(1.0 - tip_to_obj / float(self.cfg.contact_distance_gate), 0.0, 1.0)
        force_score = torch.clamp(force_norm / max(float(self.cfg.contact_force_threshold), 1e-8), 0.0, 1.0)

        soft_contact = ((tip_to_obj < float(self.cfg.contact_distance_gate)) & above_table).float()
        hard_contact = ((force_norm > float(self.cfg.contact_force_threshold)) & soft_contact.bool()).float()

        self.debug_soft_contact = soft_contact
        self.debug_hard_contact = hard_contact
        self.debug_dist_score = dist_score
        self.debug_force_score = force_score

        force_norm_clipped = torch.clamp(force_norm, 0.0, float(self.cfg.contact_force_clip)) / float(
            self.cfg.contact_force_clip
        )

        return hard_contact, force_norm_clipped, peak_force

    def _workspace_clamp(self, pos_rel: torch.Tensor, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        low = torch.tensor(self.cfg.base_workspace_low, dtype=torch.float32, device=self.device)
        high = torch.tensor(self.cfg.base_workspace_high, dtype=torch.float32, device=self.device)

        clamped = torch.max(torch.min(pos_rel, high), low)
        clamped_mask = torch.any(torch.abs(clamped - pos_rel) > 1e-6, dim=-1)

        if env_ids is None:
            self.base_was_clamped = clamped_mask
        else:
            self.base_was_clamped[env_ids] = clamped_mask

        return clamped

    def _limit_base_tilt(self, q: torch.Tensor) -> torch.Tensor:
        q = quat_normalize(q.clone())

        angle = 2.0 * torch.acos(torch.clamp(torch.abs(q[:, 0]), 0.0, 1.0))
        over = angle > float(self.cfg.base_max_tilt_rad)

        if over.any():
            axis = q[:, 1:] / torch.clamp(torch.norm(q[:, 1:], dim=-1, keepdim=True), min=1e-6)
            limited_angle = torch.full_like(angle, float(self.cfg.base_max_tilt_rad))
            limited = torch.cat(
                [
                    torch.cos(limited_angle * 0.5).unsqueeze(-1),
                    axis * torch.sin(limited_angle * 0.5).unsqueeze(-1),
                ],
                dim=-1,
            )
            q[over] = limited[over]
            self.base_was_clamped[over] = True

        return quat_normalize(q)

    def _sample_targets(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        phase = self._get_phase()
        self.curriculum_phase[env_ids] = int(phase)

        _, obj_quat, _, _ = self._get_active_tool_state()

        lift = float(self.cfg.lift_heights_by_phase[phase])
        rest_h = self._rest_center_height()[env_ids]
        self.target_height[env_ids] = rest_h + lift

        max_angle = float(self.cfg.orient_max_angle_by_phase[phase])

        if max_angle <= 1e-6:
            self.target_quat[env_ids] = obj_quat[env_ids]
        else:
            axis = torch.randn((int(env_ids.numel()), 3), dtype=torch.float32, device=self.device)
            axis = axis / torch.clamp(torch.norm(axis, dim=-1, keepdim=True), min=1e-6)

            angle = torch.rand(int(env_ids.numel()), dtype=torch.float32, device=self.device) * max_angle
            dq = quat_from_rotvec(axis * angle.unsqueeze(-1))

            self.target_quat[env_ids] = quat_normalize(quat_mul(obj_quat[env_ids], dq))

    def _apply_base_pose(self, env_ids: Optional[torch.Tensor] = None) -> None:
        origins = self._env_origins(env_ids)

        pos = self.base_pos_rel if env_ids is None else self.base_pos_rel[env_ids]
        quat = self.base_quat if env_ids is None else self.base_quat[env_ids]
        lin_vel = self.base_lin_vel if env_ids is None else self.base_lin_vel[env_ids]
        ang_vel = self.base_ang_vel if env_ids is None else self.base_ang_vel[env_ids]

        self.robot.write_root_pose_to_sim(torch.cat([origins + pos, quat], dim=-1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset(
        self,
        env_ids: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        m = int(env_ids.numel())

        if m == 0:
            return self._compute_obs(), {}

        origins = self._env_origins(env_ids)

        if bool(self.cfg.use_only_pen):
            self.active_tool[env_ids] = 0
        else:
            self.active_tool[env_ids] = torch.randint(0, 2, (m,), dtype=torch.long, device=self.device)

        obj_xy = (torch.rand((m, 2), dtype=torch.float32, device=self.device) * 2.0 - 1.0) * float(
            self.cfg.object_spawn_xy_range
        )

        if bool(self.cfg.object_yaw_random):
            yaw = (torch.rand(m, dtype=torch.float32, device=self.device) * 2.0 - 1.0) * math.pi
        else:
            yaw = torch.zeros(m, dtype=torch.float32, device=self.device)

        quat = quat_from_yaw(yaw)

        pen_z = float(self.cfg.table_height) + float(self.cfg.pen_size[2]) * 0.5
        cup_z = float(self.cfg.table_height) + float(self.cfg.cup_size[2]) * 0.5

        pen_pos = origins + torch.cat(
            [obj_xy, torch.full((m, 1), pen_z, dtype=torch.float32, device=self.device)],
            dim=-1,
        )
        cup_pos = origins + torch.cat(
            [obj_xy, torch.full((m, 1), cup_z, dtype=torch.float32, device=self.device)],
            dim=-1,
        )
        inactive = origins + torch.tensor(
            [0.0, 0.0, float(self.cfg.inactive_object_z)],
            dtype=torch.float32,
            device=self.device,
        )

        pen_active = (self.active_tool[env_ids] == 0).unsqueeze(-1)

        pen_state = torch.cat(
            [
                torch.where(pen_active, pen_pos, inactive),
                quat,
                torch.zeros((m, 6), dtype=torch.float32, device=self.device),
            ],
            dim=-1,
        )
        cup_state = torch.cat(
            [
                torch.where(~pen_active, cup_pos, inactive),
                quat,
                torch.zeros((m, 6), dtype=torch.float32, device=self.device),
            ],
            dim=-1,
        )

        self.pen.write_root_state_to_sim(pen_state, env_ids=env_ids)
        self.cup.write_root_state_to_sim(cup_state, env_ids=env_ids)

        rel_obj_pos = torch.where(pen_active, pen_pos - origins, cup_pos - origins)
        base_xy_noise = torch.randn((m, 2), dtype=torch.float32, device=self.device) * 0.04

        self.base_pos_rel[env_ids] = torch.cat(
            [
                rel_obj_pos[:, :2] + base_xy_noise,
                torch.full((m, 1), float(self.cfg.base_init_z), dtype=torch.float32, device=self.device),
            ],
            dim=-1,
        )
        self.base_pos_rel[env_ids] = self._workspace_clamp(self.base_pos_rel[env_ids], env_ids=env_ids)

        self.base_quat[env_ids] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.base_lin_vel[env_ids] = 0.0
        self.base_ang_vel[env_ids] = 0.0

        self._apply_base_pose(env_ids)

        self.robot.write_joint_state_to_sim(
            self.default_joint_pos[env_ids],
            torch.zeros_like(self.default_joint_pos[env_ids]),
            env_ids=env_ids,
        )

        self.robot.reset(env_ids)
        self.pen.reset(env_ids)
        self.cup.reset(env_ids)

        self.u_hand_prev[env_ids] = 0.0
        self.u_base_prev[env_ids] = 0.0
        self.u_base_prev2[env_ids] = 0.0
        self.episode_steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.success_counter[env_ids] = 0

        self.dr_mass[env_ids] = torch.empty(m, dtype=torch.float32, device=self.device).uniform_(
            float(self.cfg.mass_range[0]),
            float(self.cfg.mass_range[1]),
        )
        self.dr_obj_friction[env_ids] = torch.empty(m, dtype=torch.float32, device=self.device).uniform_(
            float(self.cfg.object_friction_range[0]),
            float(self.cfg.object_friction_range[1]),
        )
        self.dr_table_friction[env_ids] = torch.empty(m, dtype=torch.float32, device=self.device).uniform_(
            float(self.cfg.table_friction_range[0]),
            float(self.cfg.table_friction_range[1]),
        )
        self.dr_com_offset[env_ids] = torch.randn((m, 3), dtype=torch.float32, device=self.device) * float(
            self.cfg.com_offset_std
        )

        pen_size_sq = torch.tensor(self.cfg.pen_size, dtype=torch.float32, device=self.device).pow(2)
        cup_size_sq = torch.tensor(self.cfg.cup_size, dtype=torch.float32, device=self.device).pow(2)
        self.dr_inertia_diag[env_ids] = torch.where(pen_active, pen_size_sq, cup_size_sq)

        self.scene.update(dt=0.0)
        self._sample_targets(env_ids)

        tcp = self._get_tcp_pos_w()[env_ids]
        obj_pos, _, _, _ = self._get_active_tool_state()

        lift_amount = (obj_pos[env_ids, 2] - origins[:, 2]) - self._rest_center_height()[env_ids]

        self.prev_tcp_dist[env_ids] = torch.norm(tcp - obj_pos[env_ids], dim=-1)
        self.prev_lift_amount[env_ids] = lift_amount

        return self._compute_obs(), {}

    @torch.no_grad()
    def step(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)

        a_hand = actions[:, : int(self.cfg.num_hand_actions)]
        a_base = actions[:, int(self.cfg.num_hand_actions) :]

        u_hand = float(self.cfg.ema_hand) * a_hand + (1.0 - float(self.cfg.ema_hand)) * self.u_hand_prev
        u_base = float(self.cfg.ema_base) * a_base + (1.0 - float(self.cfg.ema_base)) * self.u_base_prev

        old_base_pos = self.base_pos_rel.clone()
        old_base_quat = self.base_quat.clone()

        trans_scale = torch.tensor(self.cfg.base_xyz_scale, dtype=torch.float32, device=self.device)
        self.base_pos_rel = self._workspace_clamp(self.base_pos_rel + u_base[:, :3] * trans_scale)

        dq = quat_from_rotvec(u_base[:, 3:] * float(self.cfg.base_rot_scale))
        self.base_quat = self._limit_base_tilt(quat_mul(dq, self.base_quat))

        self.base_lin_vel = (self.base_pos_rel - old_base_pos) / float(self.cfg.control_dt)

        q_delta = quat_mul(self.base_quat, quat_conj(old_base_quat))
        self.base_ang_vel = q_delta[:, 1:] * 2.0 / float(self.cfg.control_dt)

        cmd = torch.clamp(
            self.robot.data.joint_pos + u_hand * float(self.cfg.hand_action_scale),
            self.joint_limits[..., 0],
            self.joint_limits[..., 1],
        )

        self._apply_base_pose()
        self.robot.set_joint_position_target(cmd)

        self.scene.write_data_to_sim()

        for _ in range(int(self.cfg.decimation)):
            self.sim.step()
            self.scene.update(dt=float(self.cfg.sim_dt))

        self.global_steps += self.num_envs
        self.episode_steps += 1

        rewards, info, terminated = self._compute_rewards(actions, u_hand, u_base)
        self.episode_return += rewards

        self.u_hand_prev = u_hand.clone()
        self.u_base_prev2 = self.u_base_prev.clone()
        self.u_base_prev = u_base.clone()

        truncated = self.episode_steps >= int(self.cfg.max_episode_length)
        done = terminated | truncated

        if done.any():
            self.total_done_count += done.float().sum()
            self.total_timeout_count += truncated.float().sum()

            flat_events = info.get("events", {})
            self.total_drop_count += float(flat_events.get("Drop_Count", 0.0))
            self.total_slide_count += float(flat_events.get("SlideOut_Count", 0.0))
            self.total_crash_count += float(flat_events.get("TableCrash_Count", 0.0))
            self.total_success_count += float(flat_events.get("Success_Count", 0.0))

        obs = self._compute_obs()

        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            reset_obs, _ = self.reset(reset_ids)
            obs["obs"][reset_ids] = reset_obs["obs"][reset_ids]
            obs["privileged_obs"][reset_ids] = reset_obs["privileged_obs"][reset_ids]

        return obs, rewards, terminated, truncated, info

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _compute_obs(self) -> Dict[str, torch.Tensor]:
        origins = self._env_origins()

        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel

        obj_pos, obj_quat, obj_lin_vel, obj_ang_vel = self._get_active_tool_state()
        obj_rel_pos = obj_pos - origins

        tcp = self._get_tcp_pos_w()
        tip_pos = self._get_fingertip_pos_w()
        keypoints = self._get_tool_keypoints_w()
        contact_bools, force_norms, tip_forces = self._get_tactile()

        phase_oh = torch.nn.functional.one_hot(
            self.curriculum_phase.clamp(0, 6),
            num_classes=7,
        ).float()

        tool_oh = torch.nn.functional.one_hot(
            self.active_tool.clamp(0, 1),
            num_classes=2,
        ).float()

        target_axis = self._get_target_axis()

        tcp_to_obj = obj_pos - tcp
        fingertip_rel_obj = (tip_pos - obj_pos.unsqueeze(1)).reshape(self.num_envs, -1)
        fingertip_rel_keypoints = (tip_pos - keypoints).reshape(self.num_envs, -1)
        keypoints_rel_tcp = (keypoints - tcp.unsqueeze(1)).reshape(self.num_envs, -1)

        lift_amount = obj_rel_pos[:, 2] - self._rest_center_height()
        base_height = self.base_pos_rel[:, 2] - float(self.cfg.table_height)
        tcp_dist = torch.norm(tcp_to_obj, dim=-1, keepdim=True)

        actor_obs = torch.cat(
            [
                self.base_pos_rel,
                self.base_quat,
                self.base_lin_vel,
                self.base_ang_vel,
                q,
                qd,
                self.u_hand_prev,
                self.u_base_prev,
                obj_rel_pos,
                obj_quat,
                obj_lin_vel,
                obj_ang_vel,
                tool_oh,
                self.target_height.unsqueeze(-1),
                self.target_quat,
                target_axis,
                phase_oh,
                tcp_to_obj,
                fingertip_rel_obj,
                fingertip_rel_keypoints,
                contact_bools,
                force_norms,
                keypoints_rel_tcp,
                lift_amount.unsqueeze(-1),
                base_height.unsqueeze(-1),
                tcp_dist,
            ],
            dim=-1,
        )

        if actor_obs.shape[-1] != int(self.cfg.num_observations):
            raise RuntimeError(
                f"Task3 actor obs dim mismatch: got {actor_obs.shape[-1]}, expected {self.cfg.num_observations}"
            )

        actor_obs = torch.nan_to_num(
            torch.clamp(actor_obs, -float(self.cfg.actor_obs_clip), float(self.cfg.actor_obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.actor_obs_clip),
            neginf=-float(self.cfg.actor_obs_clip),
        )

        privileged_obs = torch.cat(
            [
                actor_obs,
                self.dr_mass.unsqueeze(-1),
                self.dr_obj_friction.unsqueeze(-1),
                self.dr_table_friction.unsqueeze(-1),
                self.dr_com_offset,
                tip_forces.reshape(self.num_envs, -1),
                self.dr_inertia_diag,
            ],
            dim=-1,
        )

        if privileged_obs.shape[-1] != int(self.cfg.num_privileged_obs):
            raise RuntimeError(
                f"Task3 privileged obs dim mismatch: got {privileged_obs.shape[-1]}, "
                f"expected {self.cfg.num_privileged_obs}"
            )

        privileged_obs = torch.nan_to_num(
            torch.clamp(privileged_obs, -float(self.cfg.privileged_obs_clip), float(self.cfg.privileged_obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.privileged_obs_clip),
            neginf=-float(self.cfg.privileged_obs_clip),
        )

        return {"obs": actor_obs, "privileged_obs": privileged_obs}

    def _compute_states(self) -> torch.Tensor:
        return self._compute_obs()["privileged_obs"]

    def get_privileged_observations(self) -> torch.Tensor:
        return self._compute_obs()["privileged_obs"]

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_rewards(self, raw_actions: torch.Tensor, u_hand: torch.Tensor, u_base: torch.Tensor):
        origins = self._env_origins()

        obj_pos, obj_quat, obj_lin_vel, obj_ang_vel = self._get_active_tool_state()
        obj_rel_pos = obj_pos - origins

        tcp = self._get_tcp_pos_w()
        tip_pos = self._get_fingertip_pos_w()
        keypoints = self._get_tool_keypoints_w()
        contact_bools, force_norms, tip_forces = self._get_tactile()

        soft_contact_count = self.debug_soft_contact.sum(dim=-1)
        soft_contact_gate = torch.clamp(soft_contact_count / 4.0, 0.0, 1.0)
        r_soft_contact = float(self.cfg.w_soft_contact) * soft_contact_gate

        contact_count = contact_bools.sum(dim=-1)
        thumb_contact = contact_bools[:, 3]

        raw_force_norm = torch.norm(tip_forces, dim=-1)
        tip_to_obj = torch.norm(tip_pos - obj_pos.unsqueeze(1), dim=-1)

        above_table = tip_pos[..., 2] > (origins[:, None, 2] + float(self.cfg.table_height) - 0.002)
        tcp_dist = torch.norm(tcp - obj_pos, dim=-1)

        contact_gate = sigmoid_gate(contact_count - 1.5, sharpness=3.0)
        near_gate = sigmoid_gate(0.32 - tcp_dist, sharpness=18.0)
        near_no_contact = (near_gate > 0.28) & (contact_count < 0.5)

        lift_amount = obj_rel_pos[:, 2] - self._rest_center_height()
        target_lift = self.target_height - self._rest_center_height()
        lift_task_gate = torch.clamp(target_lift / 0.04, 0.0, 1.0)
        actual_lift_gate = sigmoid_gate(lift_amount - 0.035, sharpness=50.0) * contact_gate
        orient_task_gate = (self.curriculum_phase >= 4).float()

        pregrasp = obj_pos + torch.tensor(
            [0.0, 0.0, float(self.cfg.base_pregrasp_height)],
            dtype=torch.float32,
            device=self.device,
        )

        tcp_to_pregrasp = torch.norm(tcp - pregrasp, dim=-1)
        approach_decay_near = torch.clamp(tcp_to_pregrasp / 0.18, 0.0, 1.0)

        r_approach = (
            float(self.cfg.w_approach)
            * torch.exp(-18.0 * tcp_to_pregrasp.square())
            * approach_decay_near
        )
        r_reach_progress = float(self.cfg.w_reach_progress) * torch.clamp(
            self.prev_tcp_dist - tcp_to_pregrasp,
            -0.03,
            0.03,
        )

        finger_to_kp = torch.norm(tip_pos - keypoints, dim=-1)
        r_pregrasp = float(self.cfg.w_pregrasp) * torch.mean(torch.exp(-18.0 * finger_to_kp.square()), dim=-1)

        z_gap = torch.clamp(tcp[:, 2] - obj_pos[:, 2], min=0.0)
        r_descend_to_contact = (
            float(self.cfg.w_descend_to_contact)
            * near_gate
            * (1.0 - torch.clamp(contact_count, 0.0, 1.0))
            * torch.exp(-25.0 * z_gap.square())
        )

        r_contact_count = float(self.cfg.w_contact) * (torch.clamp(contact_count, 0.0, 4.0) / 4.0)
        r_thumb = 0.08 * thumb_contact

        dirs = tip_pos - obj_pos.unsqueeze(1)
        dirs = dirs / torch.clamp(torch.norm(dirs, dim=-1, keepdim=True), min=1e-6)

        pair_scores = []
        for i in range(4):
            for j in range(i + 1, 4):
                pair_scores.append(
                    torch.clamp(-torch.sum(dirs[:, i] * dirs[:, j], dim=-1), 0.0, 1.0)
                    * contact_bools[:, i]
                    * contact_bools[:, j]
                )

        opposing = torch.stack(pair_scores, dim=-1).max(dim=-1)[0]
        r_force_closure = float(self.cfg.w_force_closure) * opposing * (0.5 + 0.5 * thumb_contact)

        obj_to_tcp = torch.norm(obj_pos - tcp, dim=-1)
        rel_vel = torch.norm(obj_lin_vel - self.base_lin_vel, dim=-1)

        r_grip = (
            float(self.cfg.w_grip)
            * contact_gate
            * torch.exp(-25.0 * obj_to_tcp.square())
            * torch.exp(-0.5 * rel_vel.square())
        )

        lift_error = torch.abs(obj_rel_pos[:, 2] - self.target_height)
        r_lift = float(self.cfg.w_lift) * lift_task_gate * contact_gate * torch.exp(-45.0 * lift_error.square())

        r_lift_progress = float(self.cfg.w_lift_progress) * lift_task_gate * contact_gate * torch.clamp(
            lift_amount - self.prev_lift_amount,
            -0.02,
            0.02,
        )

        theta = so3_distance(obj_quat, self.target_quat)
        r_orient = float(self.cfg.w_orient) * orient_task_gate * actual_lift_gate * torch.exp(-1.8 * theta.square())

        obj_axis = quat_rotate(
            obj_quat,
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1),
        )
        target_axis = self._get_target_axis()

        axis_err = torch.acos(torch.clamp(torch.abs(torch.sum(obj_axis * target_axis, dim=-1)), 0.0, 1.0))
        r_axis = float(self.cfg.w_axis) * orient_task_gate * actual_lift_gate * torch.exp(-2.0 * axis_err.square())

        r_stable = float(self.cfg.w_stable) * actual_lift_gate * torch.exp(
            -0.8 * torch.norm(obj_lin_vel, dim=-1).square()
            - 0.25 * torch.norm(obj_ang_vel, dim=-1).square()
        )
        r_hover = float(self.cfg.w_hover) * actual_lift_gate * torch.exp(
            -1.5 * torch.norm(self.base_lin_vel, dim=-1).square()
        )

        p_no_contact_stall = -float(self.cfg.w_no_contact_stall) * near_no_contact.float()
        p_far_from_object = -float(self.cfg.w_far_from_object) * torch.clamp(tcp_dist - 0.35, min=0.0)

        base_center_dist = torch.norm(self.base_pos_rel[:, :2], dim=-1)
        p_base_escape = -float(self.cfg.w_base_escape) * torch.square(base_center_dist)

        p_action_mag = -float(self.cfg.w_action_mag) * torch.sum(raw_actions.square(), dim=-1)
        p_action_rate = -float(self.cfg.w_action_rate) * torch.sum(
            torch.cat([u_hand - self.u_hand_prev, u_base - self.u_base_prev], dim=-1).square(),
            dim=-1,
        )
        p_joint_vel = -float(self.cfg.w_joint_vel) * torch.sum(self.robot.data.joint_vel.square(), dim=-1)
        p_base_rate = -float(self.cfg.w_base_rate) * torch.sum(u_base.square(), dim=-1)
        p_base_jerk = -float(self.cfg.w_base_jerk) * torch.sum(
            (u_base - 2.0 * self.u_base_prev + self.u_base_prev2).square(),
            dim=-1,
        )

        excess_force = torch.clamp(raw_force_norm - float(self.cfg.safe_contact_force), min=0.0)
        p_excess_force = -float(self.cfg.w_excess_force) * torch.sum(excess_force.square(), dim=-1)

        body_z = self.robot.data.body_pos_w[..., 2] - origins[:, None, 2]
        palm_crash = body_z[:, self.palm_index] < (float(self.cfg.table_height) + 0.025)
        any_deep_crash = torch.any(body_z < (float(self.cfg.table_height) - 0.010), dim=-1)
        table_crash = palm_crash | any_deep_crash

        p_table_crash = -float(self.cfg.w_table_crash) * table_crash.float()
        p_workspace = -float(self.cfg.w_workspace_violation) * self.base_was_clamped.float()

        continuous_raw = (
            r_approach
            + r_reach_progress
            + r_pregrasp
            + r_descend_to_contact
            + r_contact_count
            + r_thumb
            + r_force_closure
            + r_grip
            + r_lift
            + r_lift_progress
            + r_orient
            + r_axis
            + r_stable
            + r_hover
            + r_soft_contact
            + p_no_contact_stall
            + p_far_from_object
            + p_base_escape
            + p_action_mag
            + p_action_rate
            + p_joint_vel
            + p_base_rate
            + p_base_jerk
            + p_excess_force
            + p_table_crash
            + p_workspace
        )

        continuous = torch.clamp(
            continuous_raw,
            -float(self.cfg.continuous_reward_clip),
            float(self.cfg.continuous_reward_clip),
        )

        is_drop = obj_rel_pos[:, 2] < (float(self.cfg.table_height) - 0.08)

        table_half = float(self.cfg.table_size_xy) * 0.5
        is_slide_out = (
            (torch.abs(obj_rel_pos[:, 0]) > table_half + 0.03)
            | (torch.abs(obj_rel_pos[:, 1]) > table_half + 0.03)
        )

        orient_ok = theta < 0.25
        height_ok = lift_error < 0.025
        stable_ok = (torch.norm(obj_lin_vel, dim=-1) < 0.25) & (torch.norm(obj_ang_vel, dim=-1) < 0.6)
        contact_ok = contact_count >= 2.0

        success_now = height_ok & stable_ok & contact_ok & ((self.curriculum_phase < 5) | orient_ok)
        self.success_counter = torch.where(success_now, self.success_counter + 1, torch.zeros_like(self.success_counter))
        success = self.success_counter >= int(self.cfg.success_hold_frames)

        event_drop = torch.where(
            is_drop,
            torch.full_like(continuous, float(self.cfg.penalty_drop)),
            torch.zeros_like(continuous),
        )
        event_slide = torch.where(
            is_slide_out,
            torch.full_like(continuous, float(self.cfg.penalty_slide_out)),
            torch.zeros_like(continuous),
        )
        event_success = torch.where(
            success,
            torch.full_like(continuous, float(self.cfg.bonus_success)),
            torch.zeros_like(continuous),
        )

        r_event = event_drop + event_slide + event_success
        reward_raw = continuous + r_event

        projected = self.episode_return + reward_raw
        over_pos = projected > float(self.cfg.episode_return_abs_limit)
        over_neg = projected < -float(self.cfg.episode_return_abs_limit)
        no_event = r_event.abs() < 1e-6

        reward = torch.where(
            over_pos & no_event,
            float(self.cfg.episode_return_abs_limit) - self.episode_return,
            reward_raw,
        )
        reward = torch.where(
            over_neg & no_event,
            -float(self.cfg.episode_return_abs_limit) - self.episode_return,
            reward,
        )

        reward = torch.nan_to_num(reward, nan=0.0, posinf=10.0, neginf=float(self.cfg.penalty_drop))

        self.prev_tcp_dist = tcp_to_pregrasp.clone()
        self.prev_lift_amount = lift_amount.clone()

        success_ids = success.nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel() > 0:
            self._sample_targets(success_ids)
            self.success_counter[success_ids] = 0

        terminated = is_drop | is_slide_out | table_crash

        force_pass = (raw_force_norm > float(self.cfg.contact_force_threshold)).float()
        distance_pass = (tip_to_obj < float(self.cfg.contact_distance_gate)).float()

        total_done_safe = torch.clamp(self.total_done_count, min=1.0)

        info = {
            "reward_components": {
                "R_Approach": r_approach.detach().mean(),
                "R_ReachProgress": r_reach_progress.detach().mean(),
                "R_PreGrasp": r_pregrasp.detach().mean(),
                "R_Descend": r_descend_to_contact.detach().mean(),
                "R_Contact": r_contact_count.detach().mean(),
                "R_Thumb": r_thumb.detach().mean(),
                "R_ForceClosure": r_force_closure.detach().mean(),
                "R_Grip": r_grip.detach().mean(),
                "R_Lift": r_lift.detach().mean(),
                "R_LiftProgress": r_lift_progress.detach().mean(),
                "R_Orient": r_orient.detach().mean(),
                "R_Axis": r_axis.detach().mean(),
                "R_Stable": r_stable.detach().mean(),
                "R_Hover": r_hover.detach().mean(),
                "R_SoftContact": r_soft_contact.detach().mean(),
                "P_NoContact": p_no_contact_stall.detach().mean(),
                "P_Far": p_far_from_object.detach().mean(),
                "P_BaseEscape": p_base_escape.detach().mean(),
                "P_Action": (p_action_mag + p_action_rate).detach().mean(),
                "P_Base": (p_base_rate + p_base_jerk).detach().mean(),
                "P_ExcessForce": p_excess_force.detach().mean(),
                "P_Table": p_table_crash.detach().mean(),
                "P_Workspace": p_workspace.detach().mean(),
                "Continuous": continuous.detach().mean(),
                "Event": r_event.detach().mean(),
                "Total": reward.detach().mean(),
            },
            "events": {
                "Drop": is_drop.float().mean().detach(),
                "SlideOut": is_slide_out.float().mean().detach(),
                "TableCrash": table_crash.float().mean().detach(),
                "Success": success.float().mean().detach(),
                "WorkspaceClamp": self.base_was_clamped.float().mean().detach(),
                "Drop_Count": is_drop.float().sum().detach(),
                "SlideOut_Count": is_slide_out.float().sum().detach(),
                "TableCrash_Count": table_crash.float().sum().detach(),
                "Success_Count": success.float().sum().detach(),
                "Episode_Drop_Total_Rate": self.total_drop_count.detach() / total_done_safe,
                "Episode_Slide_Total_Rate": self.total_slide_count.detach() / total_done_safe,
                "Episode_Crash_Total_Rate": self.total_crash_count.detach() / total_done_safe,
                "Episode_Success_Total_Rate": self.total_success_count.detach() / total_done_safe,
            },
            "telemetry": {
                "Phase": self.curriculum_phase.float().mean().detach(),
                "K": torch.tensor(
                    min(1.0, float(self.global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0)),
                    dtype=torch.float32,
                    device=self.device,
                ),
                "TCP_Dist": tcp_dist.detach().mean(),
                "TCP_Pregrasp_Dist": tcp_to_pregrasp.detach().mean(),
                "Base_H": self.base_pos_rel[:, 2].detach().mean(),
                "Base_XY": base_center_dist.detach().mean(),
                "SoftContact_Count": soft_contact_count.detach().mean(),
                "HardContact_Count": contact_count.detach().mean(),
                "Contact_Count": contact_count.detach().mean(),
                "Thumb_Contact": thumb_contact.detach().mean(),
                "Force_Mean": raw_force_norm.detach().mean(),
                "Force_Max": raw_force_norm.detach().max(dim=-1)[0].mean(),
                "TipToObj_Min": tip_to_obj.detach().min(dim=-1)[0].mean(),
                "TipToObj_Mean": tip_to_obj.detach().mean(),
                "ForcePass": force_pass.detach().mean(),
                "DistancePass": distance_pass.detach().mean(),
                "AboveTable": above_table.float().detach().mean(),
                "NearGate": near_gate.detach().mean(),
                "NearNoContact": near_no_contact.float().detach().mean(),
                "Lift": lift_amount.detach().mean(),
                "SO3_Err": theta.detach().mean(),
                "Obj_H": obj_rel_pos[:, 2].detach().mean(),
                "Obj_Vel": torch.norm(obj_lin_vel, dim=-1).detach().mean(),
                "EpisodeReturn": self.episode_return.detach().mean(),
            },
            "debug": {
                "Actor_Obs_Dim": torch.tensor(float(self.cfg.num_observations), dtype=torch.float32, device=self.device),
                "Privileged_Obs_Dim": torch.tensor(float(self.cfg.num_privileged_obs), dtype=torch.float32, device=self.device),
                "Action_Dim": torch.tensor(float(self.cfg.num_actions), dtype=torch.float32, device=self.device),
                "Continuous_Min": continuous.detach().min(),
                "Continuous_Max": continuous.detach().max(),
                "Reward_Min": reward.detach().min(),
                "Reward_Max": reward.detach().max(),
                "EpRet_Min": self.episode_return.detach().min(),
                "EpRet_Max": self.episode_return.detach().max(),
            },
        }

        return reward, info, terminated


Task3Env = AllegroHandTask3Env
AllegroTask3Env = AllegroHandTask3Env
