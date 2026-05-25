from __future__ import annotations

from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor

from allegro_rl.tasks.task4.task4_config import Task4Config
from allegro_rl.tasks.task4.task4_scene import make_allegro_task4_scene_cfg


# ======================================================================
# Quaternion utilities
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


def quat_from_rotvec(rv: torch.Tensor) -> torch.Tensor:
    angle = torch.norm(rv, dim=-1, keepdim=True)
    axis = rv / torch.clamp(angle, min=1e-8)
    half = angle * 0.5
    return quat_normalize(torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1))


def so3_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = torch.clamp(torch.abs(torch.sum(q1 * q2, dim=-1)), 0.0, 1.0)
    return 2.0 * torch.acos(dot)


# ======================================================================
# Environment
# ======================================================================

class AllegroHandTask4Env(gym.Env):
    """Allegro Hand Task4: Blind Sim2Real / RMA robust reorientation.

    This environment is independent from Task1/Task2/Task3.

    Outputs:
        obs              : blind student obs, 108-D
        teacher_obs      : teacher obs with object state, 139-D
        privileged_obs   : teacher obs + DR vector, 206-D
        history_obs      : 104-D history frame x 50, 5200-D
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task4Config):
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
                "teacher_obs": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(int(cfg.num_teacher_obs),),
                    dtype=np.float32,
                ),
                "privileged_obs": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(int(cfg.num_privileged_obs),),
                    dtype=np.float32,
                ),
                "history_obs": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(int(cfg.history_obs_dim),),
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
                gpu_max_rigid_contact_count=2**24,
                gpu_max_rigid_patch_count=2**23,
            ),
        )

        self.sim = sim_utils.SimulationContext(sim_cfg)

        SceneCfg = make_allegro_task4_scene_cfg(cfg)
        self.scene = InteractiveScene(SceneCfg(num_envs=int(cfg.num_envs)))

        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.cube: RigidObject = self.scene["cube"]
        self.sphere: RigidObject = self.scene["sphere"]
        self.contact_sensor: ContactSensor = self.scene["fingertip_contact"]

        self.num_envs = int(cfg.num_envs)
        self.num_actions = int(cfg.num_actions)
        self.num_observations = int(cfg.num_observations)
        self.num_teacher_obs = int(cfg.num_teacher_obs)
        self.num_privileged_obs = int(cfg.num_privileged_obs)
        self.history_obs_dim = int(cfg.history_obs_dim)

        if self.robot.num_joints != int(cfg.num_joints):
            raise RuntimeError(
                f"Allegro Task4 expects robot.num_joints == {cfg.num_joints}, "
                f"but got {self.robot.num_joints}. joint_names={self.robot.joint_names}"
            )

        self.robot_joint_names = list(self.robot.joint_names)
        self.robot_body_names = list(self.robot.body_names)
        self.contact_body_names = list(self.contact_sensor.body_names)

        n = self.num_envs
        a = self.num_actions
        h = int(cfg.history_obs_dim)

        self.global_steps = 0
        self.episode_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.success_counter = torch.zeros(n, dtype=torch.long, device=self.device)

        self.active_shape = torch.zeros(n, dtype=torch.long, device=self.device)

        self.target_quats = torch.zeros((n, 4), dtype=torch.float32, device=self.device)
        self.target_quats[:, 0] = 1.0
        self.prev_theta = torch.zeros(n, dtype=torch.float32, device=self.device)

        self.raw_action_prev = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.applied_action = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.applied_action_prev = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.joint_cmd_target = torch.zeros((n, a), dtype=torch.float32, device=self.device)

        self.action_delay_buffer = torch.zeros(
            (n, int(cfg.max_action_delay_steps) + 1, a),
            dtype=torch.float32,
            device=self.device,
        )

        self.history_buffer = torch.zeros((n, h), dtype=torch.float32, device=self.device)
        self.prev_q_obs = torch.zeros((n, a), dtype=torch.float32, device=self.device)
        self.prev_qd_obs = torch.zeros((n, a), dtype=torch.float32, device=self.device)

        self.joint_limits = self.robot.data.joint_pos_limits.clone()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.joint_cmd_target[:] = self.default_joint_pos

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

        self._init_dr_buffers()

        self.total_done_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_drop_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_success_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_count = torch.zeros((), dtype=torch.float32, device=self.device)

        if bool(cfg.debug_print_names):
            self._print_debug_names()

        self.reset()

    # ------------------------------------------------------------------
    # Name mapping
    # ------------------------------------------------------------------
    def _print_debug_names(self) -> None:
        print("\n" + "=" * 120)
        print("[AllegroHandTask4Env] selected fingertip mapping")
        print("=" * 120)
        print("[DEBUG][Task4] requested fingertip names:", list(self.cfg.fingertip_body_names))
        print("[DEBUG][Task4] robot fingertip indices:", self.fingertip_indices)
        print("[DEBUG][Task4] contact sensor indices:", self.contact_tip_indices.tolist())
        print("[DEBUG][Task4] selected robot fingertip names:")
        for idx in self.fingertip_indices:
            print(f"  robot[{idx:02d}] = {self.robot_body_names[idx]}")
        print("[DEBUG][Task4] selected contact fingertip names:")
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
        return lower

    def _fallback_find_fingertip(self, requested_name: str, source: list[str]) -> Optional[int]:
        finger = self._finger_keyword(requested_name)

        candidates = []
        for i, name in enumerate(source):
            lower = name.lower()
            if finger not in lower:
                continue

            score = 0
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

    def _names_to_indices(
        self,
        names: list[str],
        source: list[str],
        label: str,
        allow_fallback: bool = True,
    ) -> list[int]:
        out: list[int] = []
        missing: list[str] = []

        for name in names:
            if name in source:
                out.append(source.index(name))
                continue

            fallback_idx = self._fallback_find_fingertip(name, source) if allow_fallback else None
            if fallback_idx is not None:
                out.append(int(fallback_idx))
            else:
                missing.append(name)

        if missing:
            raise RuntimeError(
                f"[Task4] missing {missing} in {label}. Available names:\n{source}"
            )

        if len(set(out)) != len(out):
            raise RuntimeError(
                f"[Task4] duplicated fingertip indices from {label}: {out}. "
                f"Requested names={names}"
            )

        return out

    # ------------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------------
    def _init_dr_buffers(self) -> None:
        n = self.num_envs
        a = self.num_actions
        d = self.device

        self.dr_mass = torch.zeros(n, dtype=torch.float32, device=d)
        self.dr_friction = torch.zeros(n, dtype=torch.float32, device=d)
        self.dr_scale = torch.ones((n, 3), dtype=torch.float32, device=d)
        self.dr_com_offset = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.dr_inertia_diag = torch.ones((n, 3), dtype=torch.float32, device=d)

        self.dr_joint_eff = torch.ones((n, a), dtype=torch.float32, device=d)
        self.dr_joint_stiff = torch.ones((n, a), dtype=torch.float32, device=d)
        self.dr_joint_damp = torch.ones((n, a), dtype=torch.float32, device=d)

        self.dr_action_delay = torch.zeros(n, dtype=torch.long, device=d)
        self.dr_deadzone = torch.zeros(n, dtype=torch.float32, device=d)
        self.dr_action_alpha = torch.full(
            (n,),
            float(self.cfg.default_action_alpha),
            dtype=torch.float32,
            device=d,
        )

        self.dr_q_noise = torch.zeros(n, dtype=torch.float32, device=d)
        self.dr_qd_noise = torch.zeros(n, dtype=torch.float32, device=d)
        self.dr_tactile_noise = torch.zeros(n, dtype=torch.float32, device=d)
        self.dr_tactile_dropout = torch.zeros(n, dtype=torch.float32, device=d)
        self.dr_state_dropout = torch.zeros(n, dtype=torch.float32, device=d)

        self.last_disturbance = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.disturbance_timer = torch.zeros(n, dtype=torch.long, device=d)

    def _env_origins(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        origins = self.scene.env_origins.to(self.device)
        return origins if env_ids is None else origins[env_ids]

    def _k(self) -> float:
        return min(1.0, float(self.global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0))

    @staticmethod
    def _smooth(x: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        return x * x * (3.0 - 2.0 * x)

    def _dr_k(self) -> float:
        k = (self._k() - float(self.cfg.dr_start_k)) / max(
            float(self.cfg.dr_end_k) - float(self.cfg.dr_start_k),
            1e-6,
        )
        return self._smooth(k)

    def _rw_k(self) -> float:
        k = (self._k() - float(self.cfg.reward_start_k)) / max(
            float(self.cfg.reward_end_k) - float(self.cfg.reward_start_k),
            1e-6,
        )
        return self._smooth(k)

    @staticmethod
    def _mix_value(a: float, b: float, k: float) -> float:
        return float(a) + (float(b) - float(a)) * float(k)

    def _range(self, name: str) -> Tuple[float, float]:
        warmup = getattr(self.cfg, f"{name}_warmup")
        full = getattr(self.cfg, f"{name}_full")
        k = self._dr_k()
        return (
            self._mix_value(warmup[0], full[0], k),
            self._mix_value(warmup[1], full[1], k),
        )

    def _irange(self, name: str) -> Tuple[int, int]:
        lo, hi = self._range(name)
        return int(round(lo)), int(round(hi))

    def _w(self, name: str) -> float:
        return self._mix_value(
            getattr(self.cfg, f"{name}_warmup"),
            getattr(self.cfg, f"{name}_full"),
            self._rw_k(),
        )

    def _sample_uniform(self, shape, value_range: Tuple[float, float]) -> torch.Tensor:
        lo, hi = value_range
        return float(lo) + (float(hi) - float(lo)) * torch.rand(shape, dtype=torch.float32, device=self.device)

    def _identity_quat(self, n: int) -> torch.Tensor:
        q = torch.zeros((int(n), 4), dtype=torch.float32, device=self.device)
        q[:, 0] = 1.0
        return q

    def _zero_vel6(self, n: int) -> torch.Tensor:
        return torch.zeros((int(n), 6), dtype=torch.float32, device=self.device)

    def _get_active_object_state(self):
        is_cube = (self.active_shape == 0).unsqueeze(-1)

        pos = torch.where(is_cube, self.cube.data.root_pos_w, self.sphere.data.root_pos_w)
        quat = torch.where(is_cube, self.cube.data.root_quat_w, self.sphere.data.root_quat_w)
        lin = torch.where(is_cube, self.cube.data.root_lin_vel_w, self.sphere.data.root_lin_vel_w)
        ang = torch.where(is_cube, self.cube.data.root_ang_vel_w, self.sphere.data.root_ang_vel_w)

        return pos, quat_normalize(quat), lin, ang

    @staticmethod
    def _limit_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True)
        scale = torch.clamp(float(max_norm) / torch.clamp(norm, min=1e-6), max=1.0)
        return x * scale

    def _write_active_velocity(self, lin: torch.Tensor, ang: torch.Tensor, env_ids: Optional[torch.Tensor] = None) -> None:
        ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids,
            dtype=torch.long,
            device=self.device,
        ).flatten()

        vel = torch.cat([lin[ids], ang[ids]], dim=-1)
        mask = self.active_shape[ids] == 0

        if mask.any():
            self.cube.write_root_velocity_to_sim(vel[mask], env_ids=ids[mask])
        if (~mask).any():
            self.sphere.write_root_velocity_to_sim(vel[~mask], env_ids=ids[~mask])

    def _get_fingertip_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.fingertip_indices, :]

    def _sanitize_sim_state(self) -> None:
        obj_pos, obj_quat, obj_lin, obj_ang = self._get_active_object_state()

        bad = (
            (~torch.isfinite(obj_pos).all(dim=-1))
            | (~torch.isfinite(obj_quat).all(dim=-1))
            | (~torch.isfinite(obj_lin).all(dim=-1))
            | (~torch.isfinite(obj_ang).all(dim=-1))
            | (torch.norm(obj_lin, dim=-1) > float(self.cfg.max_object_linvel) * 3.0)
            | (torch.norm(obj_ang, dim=-1) > float(self.cfg.max_object_angvel) * 3.0)
            | (~torch.isfinite(self.robot.data.joint_pos).all(dim=-1))
            | (~torch.isfinite(self.robot.data.joint_vel).all(dim=-1))
            | (torch.abs(self.robot.data.joint_vel).max(dim=-1)[0] > float(self.cfg.max_joint_vel_abs) * 2.0)
        )

        bad_ids = bad.nonzero(as_tuple=False).squeeze(-1)
        if bad_ids.numel() > 0:
            self.reset(bad_ids)

    # ------------------------------------------------------------------
    # DR / target / tactile / action model
    # ------------------------------------------------------------------
    def _randomize_domain(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        m = int(env_ids.numel())
        c = self.cfg

        self.dr_mass[env_ids] = self._sample_uniform((m,), self._range("mass_range"))
        self.dr_friction[env_ids] = self._sample_uniform((m,), self._range("friction_range"))
        self.dr_scale[env_ids] = self._sample_uniform((m, 3), self._range("scale_range"))
        self.dr_com_offset[env_ids] = self._sample_uniform((m, 3), self._range("com_offset_range"))
        self.dr_inertia_diag[env_ids] = self._sample_uniform((m, 3), self._range("inertia_scale_range"))

        self.dr_joint_eff[env_ids] = self._sample_uniform((m, c.num_actions), self._range("joint_efficiency_range"))
        self.dr_joint_stiff[env_ids] = self._sample_uniform((m, c.num_actions), self._range("joint_stiffness_scale_range"))
        self.dr_joint_damp[env_ids] = self._sample_uniform((m, c.num_actions), self._range("joint_damping_scale_range"))

        delay_lo, delay_hi = self._irange("action_delay_range")
        self.dr_action_delay[env_ids] = torch.randint(
            delay_lo,
            delay_hi + 1,
            (m,),
            dtype=torch.long,
            device=self.device,
        )

        self.dr_deadzone[env_ids] = self._sample_uniform((m,), self._range("actuator_deadzone_range"))
        self.dr_action_alpha[env_ids] = self._sample_uniform((m,), self._range("action_alpha_range"))

        self.dr_q_noise[env_ids] = self._sample_uniform((m,), self._range("joint_pos_noise_range"))
        self.dr_qd_noise[env_ids] = self._sample_uniform((m,), self._range("joint_vel_noise_range"))
        self.dr_tactile_noise[env_ids] = self._sample_uniform((m,), self._range("tactile_noise_range"))
        self.dr_tactile_dropout[env_ids] = self._sample_uniform((m,), self._range("tactile_dropout_range"))
        self.dr_state_dropout[env_ids] = self._sample_uniform((m,), self._range("state_dropout_range"))

        self.last_disturbance[env_ids] = 0.0
        self.disturbance_timer[env_ids] = 0

        if bool(c.enable_physx_mass_com_write):
            self._best_effort_apply_physics_dr(env_ids)

    def _best_effort_apply_physics_dr(self, env_ids: torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        ids_cpu = ids.detach().cpu()

        masses_cpu = self.dr_mass[ids].detach().cpu().view(-1, 1)
        coms_cpu = self.dr_com_offset[ids].detach().cpu().view(-1, 1, 3)

        for obj in [self.cube, self.sphere]:
            view = getattr(obj, "root_physx_view", None)
            if view is None:
                continue

            try:
                if hasattr(view, "set_masses"):
                    view.set_masses(masses_cpu, ids_cpu)
                if hasattr(view, "set_coms"):
                    view.set_coms(coms_cpu, ids_cpu)
            except Exception as exc:
                if bool(self.cfg.debug_physx_dr):
                    print(f"[WARN][Task4 PhysX DR] {type(exc).__name__}: {exc}")

    def _sample_targets(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        max_angle = float(self.cfg.target_min_angle) + self._k() * (
            float(self.cfg.target_max_angle) - float(self.cfg.target_min_angle)
        )

        axis = torch.randn((int(env_ids.numel()), 3), dtype=torch.float32, device=self.device)
        axis = axis / torch.clamp(torch.norm(axis, dim=-1, keepdim=True), min=1e-6)

        angle = torch.rand(int(env_ids.numel()), dtype=torch.float32, device=self.device) * max_angle
        dq = quat_from_rotvec(axis * angle.unsqueeze(-1))

        _, obj_quat, _, _ = self._get_active_object_state()

        self.target_quats[env_ids] = quat_normalize(quat_mul(obj_quat[env_ids], dq))
        self.prev_theta[env_ids] = so3_distance(obj_quat[env_ids], self.target_quats[env_ids])

    def _get_tactile(self, dropout: bool = False):
        forces = self.contact_sensor.data.net_forces_w_history

        if forces.ndim != 4:
            raise RuntimeError(f"[Task4 Contact] expected [N,H,B,3], got {tuple(forces.shape)}")

        hist = forces[:, :, self.contact_tip_indices, :]
        norm_hist = torch.norm(hist, dim=-1)

        peak_ids = torch.argmax(norm_hist, dim=1)
        gather_ids = peak_ids[:, None, :, None].expand(-1, 1, -1, 3)
        peak = torch.gather(hist, dim=1, index=gather_ids).squeeze(1)

        norm = torch.norm(peak, dim=-1)
        contact = (norm > float(self.cfg.contact_force_threshold)).float()
        force = torch.clamp(norm, 0.0, float(self.cfg.contact_force_clip)) / float(self.cfg.contact_force_clip)

        if dropout:
            force = torch.clamp(
                force + torch.randn_like(force) * self.dr_tactile_noise.unsqueeze(-1),
                0.0,
                1.0,
            )

            drop = torch.rand_like(contact) < self.dr_tactile_dropout.unsqueeze(-1)
            contact = torch.where(drop, torch.zeros_like(contact), contact)
            force = torch.where(drop, torch.zeros_like(force), force)

        return contact, force, peak

    def _observe_joint_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.robot.data.joint_pos
        qd = self.robot.data.joint_vel

        q_obs = q + torch.randn_like(q) * self.dr_q_noise.unsqueeze(-1)
        qd_obs = qd + torch.randn_like(qd) * self.dr_qd_noise.unsqueeze(-1)

        drop_q = torch.rand_like(q_obs) < self.dr_state_dropout.unsqueeze(-1)
        drop_qd = torch.rand_like(qd_obs) < self.dr_state_dropout.unsqueeze(-1)

        q_obs = torch.where(drop_q, self.prev_q_obs, q_obs)
        qd_obs = torch.where(drop_qd, self.prev_qd_obs, qd_obs)

        self.prev_q_obs = q_obs.clone()
        self.prev_qd_obs = qd_obs.clone()

        return q_obs, qd_obs

    def _apply_action_model(self, actions: torch.Tensor) -> torch.Tensor:
        n = self.num_envs

        self.action_delay_buffer = torch.roll(self.action_delay_buffer, shifts=1, dims=1)
        self.action_delay_buffer[:, 0, :] = actions

        delayed = self.action_delay_buffer[
            torch.arange(n, dtype=torch.long, device=self.device),
            self.dr_action_delay.clamp(0, int(self.cfg.max_action_delay_steps)),
        ]

        delayed = torch.where(
            torch.abs(delayed - self.applied_action) < self.dr_deadzone.unsqueeze(-1),
            self.applied_action,
            delayed,
        )

        noisy = delayed + float(self.cfg.action_noise_std) * torch.randn_like(delayed)
        raw = noisy * self.dr_joint_eff * self.dr_joint_stiff

        alpha = self.dr_action_alpha.unsqueeze(-1)
        filtered = alpha * raw + (1.0 - alpha) * self.applied_action

        filtered = filtered - 0.003 * (self.dr_joint_damp - 1.0) * self.robot.data.joint_vel

        return torch.clamp(filtered, -0.85, 0.85)

    def _apply_disturbance(self) -> None:
        active = self.episode_steps > int(self.cfg.disturbance_warmup_steps)
        if not active.any():
            return

        obj_pos, _, lin, ang = self._get_active_object_state()
        obj_rel_z = obj_pos[:, 2] - self._env_origins()[:, 2]

        alive = active & (obj_rel_z > float(self.cfg.drop_height))
        if not alive.any():
            return

        k = self._dr_k()

        slip_v = self._mix_value(self.cfg.slip_velocity_scale_warmup, self.cfg.slip_velocity_scale_full, k)
        slip_w = self._mix_value(self.cfg.slip_angular_scale_warmup, self.cfg.slip_angular_scale_full, k)
        push_p = self._mix_value(self.cfg.push_probability_warmup, self.cfg.push_probability_full, k)

        push_lo = self._mix_value(self.cfg.push_delta_v_range_warmup[0], self.cfg.push_delta_v_range_full[0], k)
        push_hi = self._mix_value(self.cfg.push_delta_v_range_warmup[1], self.cfg.push_delta_v_range_full[1], k)

        fr0, fr1 = self._range("friction_range")
        fric = torch.clamp((self.dr_friction - fr0) / max(fr1 - fr0, 1e-6), 0.0, 1.0)
        slip = 1.0 - fric

        new_lin = lin.clone()
        new_ang = ang.clone()

        new_lin[alive] += torch.randn_like(lin[alive]) * slip[alive].unsqueeze(-1) * slip_v
        new_ang[alive] += torch.randn_like(ang[alive]) * slip[alive].unsqueeze(-1) * slip_w

        push_mask = (torch.rand(self.num_envs, dtype=torch.float32, device=self.device) < push_p) & alive
        push = torch.zeros_like(lin)

        if push_mask.any():
            count = int(push_mask.sum().item())

            mag = self._sample_uniform((count, 1), (push_lo, push_hi))
            direction = torch.randn((count, 3), dtype=torch.float32, device=self.device)
            direction = direction / torch.clamp(torch.norm(direction, dim=-1, keepdim=True), min=1e-6)

            push[push_mask] = direction * mag / torch.clamp(self.dr_mass[push_mask].unsqueeze(-1), min=0.05)

            self.last_disturbance[push_mask] = push[push_mask]
            self.disturbance_timer[push_mask] = int(self.cfg.disturbance_recovery_window)

        self.disturbance_timer = torch.clamp(self.disturbance_timer - 1, min=0)

        new_lin = self._limit_norm(new_lin + push, float(self.cfg.max_object_linvel))
        new_ang = self._limit_norm(new_ang, float(self.cfg.max_object_angvel))

        self._write_active_velocity(new_lin, new_ang)

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

        self._randomize_domain(env_ids)

        if bool(self.cfg.use_mixed_shapes):
            self.active_shape[env_ids] = torch.randint(0, 2, (m,), dtype=torch.long, device=self.device)
        else:
            self.active_shape[env_ids] = 0

        pos = origins + torch.cat(
            [
                torch.randn((m, 2), dtype=torch.float32, device=self.device) * 0.015,
                torch.full((m, 1), float(self.cfg.object_spawn_height), dtype=torch.float32, device=self.device),
            ],
            dim=-1,
        )

        quat = self._identity_quat(m)
        vel = self._zero_vel6(m)

        inactive = origins + torch.tensor(
            [0.0, 0.0, float(self.cfg.inactive_object_z)],
            dtype=torch.float32,
            device=self.device,
        )

        is_cube = (self.active_shape[env_ids] == 0).unsqueeze(-1)

        cube_pos = torch.where(is_cube, pos, inactive)
        sphere_pos = torch.where(~is_cube, pos, inactive)

        self.cube.write_root_state_to_sim(torch.cat([cube_pos, quat, vel], dim=-1), env_ids=env_ids)
        self.sphere.write_root_state_to_sim(torch.cat([sphere_pos, quat, vel], dim=-1), env_ids=env_ids)

        q0 = torch.clamp(
            self.default_joint_pos[env_ids] + torch.randn_like(self.default_joint_pos[env_ids]) * 0.08,
            self.joint_limits[env_ids, :, 0],
            self.joint_limits[env_ids, :, 1],
        )

        self.robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), env_ids=env_ids)

        self.robot.reset(env_ids)
        self.cube.reset(env_ids)
        self.sphere.reset(env_ids)

        self.scene.update(dt=0.0)

        self.raw_action_prev[env_ids] = 0.0
        self.applied_action[env_ids] = 0.0
        self.applied_action_prev[env_ids] = 0.0
        self.action_delay_buffer[env_ids] = 0.0

        self.joint_cmd_target[env_ids] = self.robot.data.joint_pos[env_ids]

        self.episode_steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0
        self.success_counter[env_ids] = 0

        self.prev_q_obs[env_ids] = self.robot.data.joint_pos[env_ids]
        self.prev_qd_obs[env_ids] = self.robot.data.joint_vel[env_ids]

        self.last_disturbance[env_ids] = 0.0
        self.disturbance_timer[env_ids] = 0

        self._sample_targets(env_ids)

        return self._compute_obs(update_history=False, fill_history_env_ids=env_ids), {}

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)

        self.applied_action_prev = self.applied_action.clone()
        self.applied_action = self._apply_action_model(actions)

        cmd = torch.clamp(
            self.robot.data.joint_pos + self.applied_action * float(self.cfg.action_scale),
            self.joint_limits[..., 0],
            self.joint_limits[..., 1],
        )

        self.joint_cmd_target = cmd.clone()

        self.robot.set_joint_position_target(cmd)
        self.scene.write_data_to_sim()

        for _ in range(int(self.cfg.decimation)):
            self.sim.step()
            self.scene.update(dt=float(self.cfg.sim_dt))

        self._apply_disturbance()
        self.scene.update(dt=0.0)

        self._sanitize_sim_state()

        self.global_steps += self.num_envs
        self.episode_steps += 1

        reward, info, terminated = self._compute_rewards(actions)
        self.episode_return += reward

        self.raw_action_prev = actions.clone()

        truncated = self.episode_steps >= int(self.cfg.max_episode_length)
        done = terminated | truncated

        if done.any():
            self.total_done_count += done.float().sum()
            self.total_drop_count += terminated.float().sum()
            self.total_timeout_count += truncated.float().sum()

            success_rate = info.get("events", {}).get("Success", 0.0)
            if torch.is_tensor(success_rate):
                self.total_success_count += success_rate.detach().float() * self.num_envs
            else:
                self.total_success_count += float(success_rate) * self.num_envs

        obs = self._compute_obs(update_history=True)

        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            reset_obs, _ = self.reset(reset_ids)
            obs["obs"][reset_ids] = reset_obs["obs"][reset_ids]
            obs["teacher_obs"][reset_ids] = reset_obs["teacher_obs"][reset_ids]
            obs["privileged_obs"][reset_ids] = reset_obs["privileged_obs"][reset_ids]
            obs["history_obs"][reset_ids] = reset_obs["history_obs"][reset_ids]

        return obs, reward, terminated, truncated, info

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _history_frame(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        qerr: torch.Tensor,
        contact: torch.Tensor,
        force: torch.Tensor,
    ) -> torch.Tensor:
        effort = torch.abs(self.joint_cmd_target - self.robot.data.joint_pos) / torch.clamp(self.dr_joint_eff, min=0.2)

        frame = torch.cat(
            [
                q,
                qd,
                qerr,
                self.applied_action,
                self.applied_action - self.applied_action_prev,
                effort,
                contact,
                force,
            ],
            dim=-1,
        )

        if frame.shape[-1] != int(self.cfg.history_frame_dim):
            raise RuntimeError(f"Task4 history frame dim mismatch: got {frame.shape[-1]}")

        return frame

    def _update_history(self, frame: torch.Tensor, fill_ids: Optional[torch.Tensor] = None) -> None:
        d = int(self.cfg.history_frame_dim)

        if fill_ids is not None:
            ids = torch.as_tensor(fill_ids, dtype=torch.long, device=self.device).flatten()
            for i in range(int(self.cfg.history_length)):
                self.history_buffer[ids, i * d : (i + 1) * d] = frame[ids]
        else:
            self.history_buffer[:, :-d] = self.history_buffer[:, d:].clone()
            self.history_buffer[:, -d:] = frame

    def _compute_obs(
        self,
        update_history: bool = True,
        fill_history_env_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        q, qd = self._observe_joint_state()

        qerr = self.joint_cmd_target - q
        contact, force, _ = self._get_tactile(dropout=True)

        obj_pos, obj_quat, obj_lin, obj_ang = self._get_active_object_state()
        origins = self._env_origins()
        obj_rel = obj_pos - origins

        effort = torch.abs(self.joint_cmd_target - self.robot.data.joint_pos) / torch.clamp(self.dr_joint_eff, min=0.2)

        blind = torch.cat(
            [
                q,
                qd,
                qerr,
                self.raw_action_prev,
                self.applied_action,
                effort,
                contact,
                force,
                self.target_quats,
            ],
            dim=-1,
        )

        if blind.shape[-1] != int(self.cfg.num_observations):
            raise RuntimeError(f"Task4 blind obs dim mismatch: got {blind.shape[-1]}, expected {self.cfg.num_observations}")

        blind = torch.nan_to_num(
            torch.clamp(blind, -float(self.cfg.actor_obs_clip), float(self.cfg.actor_obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.actor_obs_clip),
            neginf=-float(self.cfg.actor_obs_clip),
        )

        tip_rel = (self._get_fingertip_pos_w() - obj_pos.unsqueeze(1)).reshape(self.num_envs, -1)

        theta = so3_distance(obj_quat, self.target_quats).unsqueeze(-1)

        qerr_obj = quat_mul(quat_conj(obj_quat), self.target_quats)
        qerr_obj = quat_normalize(qerr_obj)
        axis_err = qerr_obj[:, 1:4] * torch.sign(torch.clamp(qerr_obj[:, 0:1], min=-1.0, max=1.0))

        shape_onehot = torch.nn.functional.one_hot(
            self.active_shape.clamp(0, 1),
            num_classes=2,
        ).float()

        teacher = torch.cat(
            [
                blind,
                obj_rel,
                obj_quat,
                obj_lin,
                obj_ang,
                theta,
                axis_err,
                tip_rel,
                shape_onehot,
            ],
            dim=-1,
        )

        if teacher.shape[-1] != int(self.cfg.num_teacher_obs):
            raise RuntimeError(f"Task4 teacher obs dim mismatch: got {teacher.shape[-1]}, expected {self.cfg.num_teacher_obs}")

        teacher = torch.nan_to_num(
            torch.clamp(teacher, -float(self.cfg.teacher_obs_clip), float(self.cfg.teacher_obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.teacher_obs_clip),
            neginf=-float(self.cfg.teacher_obs_clip),
        )

        dr = torch.cat(
            [
                self.dr_mass.unsqueeze(-1),
                self.dr_friction.unsqueeze(-1),
                self.dr_scale,
                self.dr_com_offset,
                self.dr_inertia_diag,
                self.dr_joint_eff,
                self.dr_joint_damp,
                self.dr_joint_stiff,
                self.dr_action_delay.float().unsqueeze(-1),
                self.dr_deadzone.unsqueeze(-1),
                self.dr_q_noise.unsqueeze(-1),
                self.dr_tactile_dropout.unsqueeze(-1),
                self.dr_action_alpha.unsqueeze(-1),
                self.last_disturbance,
            ],
            dim=-1,
        )

        if dr.shape[-1] != 67:
            raise RuntimeError(f"Task4 DR vector dim mismatch: got {dr.shape[-1]}, expected 67")

        privileged = torch.cat([teacher, dr], dim=-1)

        if privileged.shape[-1] != int(self.cfg.num_privileged_obs):
            raise RuntimeError(
                f"Task4 privileged obs dim mismatch: got {privileged.shape[-1]}, expected {self.cfg.num_privileged_obs}"
            )

        privileged = torch.nan_to_num(
            torch.clamp(privileged, -float(self.cfg.privileged_obs_clip), float(self.cfg.privileged_obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.privileged_obs_clip),
            neginf=-float(self.cfg.privileged_obs_clip),
        )

        frame = self._history_frame(q, qd, qerr, contact, force)

        if fill_history_env_ids is not None:
            self._update_history(frame, fill_history_env_ids)
        elif update_history:
            self._update_history(frame)

        history = torch.nan_to_num(
            torch.clamp(self.history_buffer, -float(self.cfg.history_obs_clip), float(self.cfg.history_obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.history_obs_clip),
            neginf=-float(self.cfg.history_obs_clip),
        )

        return {
            "obs": blind,
            "teacher_obs": teacher,
            "privileged_obs": privileged,
            "history_obs": history.clone(),
        }

    def _compute_states(self) -> torch.Tensor:
        return self._compute_obs()["privileged_obs"]

    def get_privileged_observations(self) -> torch.Tensor:
        return self._compute_obs()["privileged_obs"]

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_rewards(self, raw_actions: torch.Tensor):
        origins = self._env_origins()

        obj_pos, obj_quat, obj_lin, obj_ang = self._get_active_object_state()
        obj_rel = obj_pos - origins

        contact, force, tip_forces = self._get_tactile(dropout=False)

        contact_count = contact.sum(dim=-1)
        contact_gate = torch.clamp(contact_count / 1.0, 0.0, 1.0)

        theta = so3_distance(obj_quat, self.target_quats)
        progress = self.prev_theta - theta

        r_rot = self._w("w_rot") * torch.exp(-2.0 * theta.square())
        r_progress = self._w("w_progress") * torch.clamp(progress, -0.05, 0.05)
        r_contact = self._w("w_contact") * torch.clamp(contact_count, 0.0, 4.0) / 4.0

        palm_dist = torch.norm(obj_rel[:, :2], dim=-1)
        r_center = torch.where(
            palm_dist < 0.05,
            torch.zeros_like(palm_dist),
            -self._w("w_center") * (palm_dist - 0.05),
        )

        r_height = -self._w("w_height") * torch.square(obj_rel[:, 2] - float(self.cfg.target_height))

        r_stable = self._w("w_stable") * contact_gate * torch.exp(
            -0.8 * torch.norm(obj_lin, dim=-1).square()
            - 0.25 * torch.norm(obj_ang, dim=-1).square()
        )

        in_recovery = (self.disturbance_timer > 0).float()

        r_recovery = (
            self._w("w_recovery")
            * in_recovery
            * torch.exp(-2.0 * theta.square())
            * (obj_rel[:, 2] > float(self.cfg.drop_height)).float()
        )

        p_action_rate = -self._w("w_action_rate") * torch.sum(
            (self.applied_action - self.applied_action_prev).square(),
            dim=-1,
        )
        p_action_mag = -self._w("w_action_mag") * torch.sum(raw_actions.square(), dim=-1)
        p_joint_vel = -self._w("w_joint_vel") * torch.sum(self.robot.data.joint_vel.square(), dim=-1)

        p_energy = -self._w("w_energy") * torch.sum(
            torch.abs(self.applied_action * self.robot.data.joint_vel),
            dim=-1,
        )

        torque_proxy = torch.abs(self.joint_cmd_target - self.robot.data.joint_pos) / torch.clamp(
            self.dr_joint_eff,
            min=0.2,
        )

        p_torque = -self._w("w_torque_spike") * torch.sum(
            torch.clamp(torque_proxy - 0.25, min=0.0).square(),
            dim=-1,
        )

        p_force = -self._w("w_contact_force_spike") * torch.sum(
            torch.clamp(torch.norm(tip_forces, dim=-1) - float(self.cfg.safe_contact_force), min=0.0).square(),
            dim=-1,
        )

        continuous_raw = (
            r_rot
            + r_progress
            + r_contact
            + r_center
            + r_height
            + r_stable
            + r_recovery
            + p_action_rate
            + p_action_mag
            + p_joint_vel
            + p_energy
            + p_torque
            + p_force
        )

        continuous = torch.clamp(
            continuous_raw,
            -float(self.cfg.continuous_reward_clip),
            float(self.cfg.continuous_reward_clip),
        )

        is_drop = obj_rel[:, 2] < float(self.cfg.drop_height)

        success_now = (
            (theta < 0.20)
            & (obj_rel[:, 2] > 0.48)
            & (contact_count >= 1.0)
            & (torch.norm(obj_lin, dim=-1) < 0.25)
            & (torch.norm(obj_ang, dim=-1) < 0.60)
        )

        self.success_counter = torch.where(
            success_now,
            self.success_counter + 1,
            torch.zeros_like(self.success_counter),
        )

        success = self.success_counter >= int(self.cfg.success_hold_frames)

        event_drop = torch.where(
            is_drop,
            torch.full_like(theta, self._w("penalty_drop")),
            torch.zeros_like(theta),
        )

        event_success = torch.where(
            success,
            torch.full_like(theta, self._w("bonus_success")),
            torch.zeros_like(theta),
        )

        r_event = event_drop + event_success
        reward_raw = continuous + r_event

        projected = self.episode_return + reward_raw
        no_event = r_event.abs() < 1e-6

        reward = torch.where(
            (projected > float(self.cfg.episode_return_abs_limit)) & no_event,
            float(self.cfg.episode_return_abs_limit) - self.episode_return,
            reward_raw,
        )
        reward = torch.where(
            (projected < -float(self.cfg.episode_return_abs_limit)) & no_event,
            -float(self.cfg.episode_return_abs_limit) - self.episode_return,
            reward,
        )

        reward = torch.nan_to_num(
            reward,
            nan=0.0,
            posinf=10.0,
            neginf=self._w("penalty_drop"),
        )

        success_ids = success.nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel() > 0:
            self._sample_targets(success_ids)
            self.success_counter[success_ids] = 0

        self.prev_theta = theta.clone()

        terminated = is_drop

        total_done_safe = torch.clamp(self.total_done_count, min=1.0)

        info = {
            "reward_components": {
                "R_Rot": r_rot.detach().mean(),
                "R_Progress": r_progress.detach().mean(),
                "R_Contact": r_contact.detach().mean(),
                "R_Center": r_center.detach().mean(),
                "R_Height": r_height.detach().mean(),
                "R_Stable": r_stable.detach().mean(),
                "R_Recovery": r_recovery.detach().mean(),
                "P_ActionRate": p_action_rate.detach().mean(),
                "P_ActionMag": p_action_mag.detach().mean(),
                "P_JointVel": p_joint_vel.detach().mean(),
                "P_Energy": p_energy.detach().mean(),
                "P_TorqueSpike": p_torque.detach().mean(),
                "P_ForceSpike": p_force.detach().mean(),
                "Continuous": continuous.detach().mean(),
                "Event": r_event.detach().mean(),
                "Total": reward.detach().mean(),
            },
            "events": {
                "Drop": is_drop.float().mean().detach(),
                "Success": success.float().mean().detach(),
                "RecoveryMode": in_recovery.mean().detach(),
                "Episode_Drop_Total_Rate": self.total_drop_count.detach() / total_done_safe,
                "Episode_Success_Total_Rate": self.total_success_count.detach() / total_done_safe,
                "Episode_Timeout_Total_Rate": self.total_timeout_count.detach() / total_done_safe,
            },
            "telemetry": {
                "K": torch.tensor(self._k(), dtype=torch.float32, device=self.device),
                "DR_K": torch.tensor(self._dr_k(), dtype=torch.float32, device=self.device),
                "Reward_K": torch.tensor(self._rw_k(), dtype=torch.float32, device=self.device),
                "SO3_Error": theta.detach().mean(),
                "Contact_Count": contact_count.detach().mean(),
                "Object_Height": obj_rel[:, 2].detach().mean(),
                "Object_LinVel": torch.norm(obj_lin, dim=-1).detach().mean(),
                "Object_AngVel": torch.norm(obj_ang, dim=-1).detach().mean(),
                "Mass": self.dr_mass.detach().mean(),
                "Friction": self.dr_friction.detach().mean(),
                "ActionDelay": self.dr_action_delay.float().detach().mean(),
                "Deadzone": self.dr_deadzone.detach().mean(),
                "JointEfficiency": self.dr_joint_eff.detach().mean(),
                "TactileDropout": self.dr_tactile_dropout.detach().mean(),
                "StateDropout": self.dr_state_dropout.detach().mean(),
                "DisturbanceNorm": torch.norm(self.last_disturbance, dim=-1).detach().mean(),
                "EpisodeReturn": self.episode_return.detach().mean(),
            },
            "debug": {
                "Actor_Obs_Dim": torch.tensor(float(self.cfg.num_observations), dtype=torch.float32, device=self.device),
                "Teacher_Obs_Dim": torch.tensor(float(self.cfg.num_teacher_obs), dtype=torch.float32, device=self.device),
                "Privileged_Obs_Dim": torch.tensor(float(self.cfg.num_privileged_obs), dtype=torch.float32, device=self.device),
                "History_Obs_Dim": torch.tensor(float(self.cfg.history_obs_dim), dtype=torch.float32, device=self.device),
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


Task4Env = AllegroHandTask4Env
AllegroTask4Env = AllegroHandTask4Env
