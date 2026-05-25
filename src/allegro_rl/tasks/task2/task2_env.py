from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor

from allegro_rl.tasks.task2.task2_config import Task2Config
from allegro_rl.tasks.task2.task2_scene import make_allegro_task2_scene_cfg


class AllegroHandTask2Env(gym.Env):
    """Allegro Hand Task2: in-hand object reorientation.

    Action:
        16-D joint residual action in [-1, 1].

    Actor observation, 83-D:
        q_t 16
        q_dot_t 16
        last filtered action 16
        object relative position 3
        object quaternion 4
        object linear velocity 3
        object angular velocity 3
        fingertip positions relative to object 12
        fingertip contact bools 4
        active shape one-hot 2
        geodesic error theta 1
        quaternion axis error 3

    Privileged observation, 88-D:
        actor obs 83 + mass proxy 1 + friction proxy 1 + com offset 3.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task2Config):
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

        SceneCfg = make_allegro_task2_scene_cfg(cfg)
        self.scene = InteractiveScene(SceneCfg(num_envs=int(cfg.num_envs)))

        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.cube: RigidObject = self.scene["cube"]
        self.sphere: RigidObject = self.scene["sphere"]
        self.contact_sensor: ContactSensor = self.scene["fingertip_contact"]

        self.num_envs = int(cfg.num_envs)
        self.num_actions = int(cfg.num_actions)
        self.num_observations = int(cfg.num_observations)
        self.num_privileged_obs = int(cfg.num_privileged_obs)

        if self.robot.num_joints != self.num_actions:
            raise RuntimeError(
                f"Allegro Task2 expects robot.num_joints == {self.num_actions}, "
                f"but got {self.robot.num_joints}. joint_names={self.robot.joint_names}"
            )

        self.robot_joint_names = list(self.robot.joint_names)
        self.robot_body_names = list(self.robot.body_names)
        self.contact_body_names = list(self.contact_sensor.body_names)

        self.global_steps = 0
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 0 = cube, 1 = sphere
        self.active_shape = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.target_quats = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.target_quats[:, 0] = 1.0
        self.prev_theta = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.a_t_minus_1 = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.u_t_minus_1 = torch.zeros_like(self.a_t_minus_1)

        self.dr_mass = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.dr_friction = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.dr_com_offset = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self.joint_limits = self.robot.data.joint_pos_limits.clone()
        self.default_pos = self.robot.data.default_joint_pos.clone()
        self.default_vel = torch.zeros_like(self.default_pos)

        self.fingertip_body_names = list(cfg.fingertip_body_names)
        self.fingertip_indices = self._names_to_indices(
            self.fingertip_body_names,
            self.robot_body_names,
            "robot.body_names",
            allow_fallback=True,
        )
        self.contact_sensor_tip_indices = torch.tensor(
            self._names_to_indices(
                self.fingertip_body_names,
                self.contact_body_names,
                "contact_sensor.body_names",
                allow_fallback=True,
            ),
            dtype=torch.long,
            device=self.device,
        )

        self.total_drop_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_success_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_timeout_count = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_done_count = torch.zeros((), dtype=torch.float32, device=self.device)

        if bool(cfg.debug_print_names):
            self._print_debug_names()

        self.reset()

    # ------------------------------------------------------------------
    # Name mapping
    # ------------------------------------------------------------------
    def _print_debug_names(self) -> None:
        print("\n" + "=" * 120)
        print("[AllegroHandTask2Env] selected fingertip mapping")
        print("=" * 120)
        print("[DEBUG] requested fingertip body names:", self.fingertip_body_names)
        print("[DEBUG] robot body count:", len(self.robot_body_names))
        print("[DEBUG] contact body count:", len(self.contact_body_names))
        print("[DEBUG] selected robot fingertip indices:", self.fingertip_indices)
        print("[DEBUG] selected contact sensor indices:", self.contact_sensor_tip_indices.tolist())
        print("[DEBUG] selected robot fingertip names:")
        for idx in self.fingertip_indices:
            print(f"  robot[{idx:02d}] = {self.robot_body_names[idx]}")
        print("[DEBUG] selected contact fingertip names:")
        for idx in self.contact_sensor_tip_indices.tolist():
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
        return candidates[0][1]

    def _names_to_indices(
        self,
        names: list[str],
        source: list[str],
        source_name: str,
        allow_fallback: bool = True,
    ) -> list[int]:
        out: list[int] = []
        missing = []

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
                f"[Name Match Error] {missing} not found in {source_name}.\n"
                f"Available names:\n{source}"
            )

        if len(set(out)) != len(out):
            raise RuntimeError(
                f"[Name Match Error] duplicated fingertip indices from {source_name}: {out}\n"
                f"Requested: {names}\nAvailable: {source}"
            )

        return out

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------
    def _env_origins(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        origins = self.scene.env_origins.to(self.device)
        return origins if env_ids is None else origins[env_ids]

    def _zero_vel6(self, n: int) -> torch.Tensor:
        return torch.zeros((int(n), 6), dtype=torch.float32, device=self.device)

    def _identity_quat(self, n: int) -> torch.Tensor:
        q = torch.zeros((int(n), 4), dtype=torch.float32, device=self.device)
        q[:, 0] = 1.0
        return q

    @staticmethod
    def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
        return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-6)

    def _get_active_object_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        is_cube = (self.active_shape == 0).unsqueeze(-1)

        pos = torch.where(is_cube, self.cube.data.root_pos_w, self.sphere.data.root_pos_w)
        quat = torch.where(is_cube, self.cube.data.root_quat_w, self.sphere.data.root_quat_w)
        lin_vel = torch.where(is_cube, self.cube.data.root_lin_vel_w, self.sphere.data.root_lin_vel_w)
        ang_vel = torch.where(is_cube, self.cube.data.root_ang_vel_w, self.sphere.data.root_ang_vel_w)

        return pos, self._quat_normalize(quat), lin_vel, ang_vel

    def _compute_geodesic_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        q1 = self._quat_normalize(q1)
        q2 = self._quat_normalize(q2)
        dot = torch.clamp(torch.abs(torch.sum(q1 * q2, dim=-1)), 0.0, 1.0)
        return 2.0 * torch.acos(dot)

    def _get_contact_bools(self) -> torch.Tensor:
        forces = self.contact_sensor.data.net_forces_w_history

        if forces.ndim != 4:
            raise RuntimeError(f"[Contact Sensor Error] Expected [N,H,B,3], got {tuple(forces.shape)}")

        tip_forces = forces[:, :, self.contact_sensor_tip_indices, :]
        contact_peak = torch.max(torch.norm(tip_forces, dim=-1), dim=1)[0]

        return (contact_peak > float(self.cfg.contact_force_threshold)).float()

    def curriculum_k(self) -> float:
        return min(1.0, float(self.global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0))

    # ------------------------------------------------------------------
    # Target sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sample_target_quats(self, env_ids: torch.Tensor) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if env_ids.numel() == 0:
            return

        k = self.curriculum_k()
        max_angle = float(self.cfg.min_target_angle) + k * (
            float(self.cfg.max_target_angle) - float(self.cfg.min_target_angle)
        )

        n = int(env_ids.numel())
        axis = torch.randn((n, 3), dtype=torch.float32, device=self.device)
        axis = axis / torch.clamp(torch.norm(axis, dim=-1, keepdim=True), min=1e-6)

        angle = torch.rand(n, dtype=torch.float32, device=self.device) * max_angle
        q_delta = torch.cat(
            [
                torch.cos(angle / 2.0).unsqueeze(-1),
                axis * torch.sin(angle / 2.0).unsqueeze(-1),
            ],
            dim=-1,
        )
        q_delta = self._quat_normalize(q_delta)

        _, obj_quat, _, _ = self._get_active_object_state()
        self.target_quats[env_ids] = self._quat_normalize(math_utils.quat_mul(obj_quat[env_ids], q_delta))
        self.prev_theta[env_ids] = self._compute_geodesic_distance(
            obj_quat[env_ids],
            self.target_quats[env_ids],
        )

    # ------------------------------------------------------------------
    # Gym API
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

        if bool(self.cfg.use_only_cube):
            self.active_shape[env_ids] = 0
        else:
            self.active_shape[env_ids] = torch.randint(0, 2, (m,), dtype=torch.long, device=self.device)

        origins = self._env_origins(env_ids)

        spawn_pos = origins + torch.tensor(
            [0.0, 0.0, float(self.cfg.object_spawn_height)],
            dtype=torch.float32,
            device=self.device,
        )
        spawn_pos[:, :2] += torch.randn((m, 2), dtype=torch.float32, device=self.device) * 0.02

        inactive_pos = origins + torch.tensor(
            [0.0, 0.0, float(self.cfg.inactive_object_z)],
            dtype=torch.float32,
            device=self.device,
        )

        quat = self._identity_quat(m)
        zero_vel = self._zero_vel6(m)

        cube_pos = torch.where((self.active_shape[env_ids] == 0).unsqueeze(-1), spawn_pos, inactive_pos)
        sphere_pos = torch.where((self.active_shape[env_ids] == 1).unsqueeze(-1), spawn_pos, inactive_pos)

        self.cube.write_root_state_to_sim(torch.cat([cube_pos, quat, zero_vel], dim=-1), env_ids=env_ids)
        self.sphere.write_root_state_to_sim(torch.cat([sphere_pos, quat, zero_vel], dim=-1), env_ids=env_ids)

        self.robot.write_joint_state_to_sim(
            self.default_pos[env_ids],
            torch.zeros_like(self.default_pos[env_ids]),
            env_ids=env_ids,
        )

        self.dr_mass[env_ids] = torch.empty(m, dtype=torch.float32, device=self.device).uniform_(
            float(self.cfg.mass_range[0]),
            float(self.cfg.mass_range[1]),
        )
        self.dr_friction[env_ids] = torch.empty(m, dtype=torch.float32, device=self.device).uniform_(
            float(self.cfg.friction_range[0]),
            float(self.cfg.friction_range[1]),
        )
        self.dr_com_offset[env_ids] = torch.randn((m, 3), dtype=torch.float32, device=self.device) * float(
            self.cfg.com_offset_std
        )

        self.cube.reset(env_ids)
        self.sphere.reset(env_ids)
        self.robot.reset(env_ids)

        self.a_t_minus_1[env_ids] = 0.0
        self.u_t_minus_1[env_ids] = 0.0
        self.episode_steps[env_ids] = 0

        self.scene.update(dt=0.0)

        self._sample_target_quats(env_ids)

        return self._compute_obs(), {}

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)

        u_t = float(self.cfg.ema_alpha) * actions + (1.0 - float(self.cfg.ema_alpha)) * self.u_t_minus_1

        cmd_target = torch.clamp(
            self.robot.data.joint_pos + u_t * float(self.cfg.action_scale),
            self.joint_limits[..., 0],
            self.joint_limits[..., 1],
        )

        self.robot.set_joint_position_target(cmd_target)
        self.scene.write_data_to_sim()

        for _ in range(int(self.cfg.decimation)):
            self.sim.step()
            self.scene.update(dt=float(self.cfg.sim_dt))

        self.global_steps += self.num_envs
        self.episode_steps += 1

        obj_pos, obj_quat, obj_lin_vel, obj_ang_vel = self._get_active_object_state()
        obj_rel_pos = obj_pos - self._env_origins()
        theta = self._compute_geodesic_distance(obj_quat, self.target_quats)

        active_contacts = self._get_contact_bools().sum(dim=-1)

        success_mask = (
            (theta < float(self.cfg.success_theta_threshold))
            & (torch.norm(obj_ang_vel, dim=-1) < float(self.cfg.success_ang_vel_threshold))
            & (torch.norm(obj_lin_vel, dim=-1) < float(self.cfg.success_lin_vel_threshold))
            & (obj_rel_pos[:, 2] > float(self.cfg.success_min_height))
            & (torch.norm(obj_rel_pos[:, :2], dim=-1) < float(self.cfg.success_xy_radius))
            & (active_contacts >= float(self.cfg.success_min_contacts))
        )

        rewards, info, is_drop = self._compute_rewards(actions, u_t, success_mask)

        self.a_t_minus_1 = actions.clone()
        self.u_t_minus_1 = u_t.clone()
        self.prev_theta = theta.clone()

        success_ids = success_mask.nonzero(as_tuple=False).squeeze(-1)
        if success_ids.numel() > 0:
            self._sample_target_quats(success_ids)

        terminated = is_drop
        truncated = self.episode_steps >= int(self.cfg.max_episode_length)
        done = terminated | truncated

        if done.any():
            self.total_done_count += done.float().sum()
            self.total_drop_count += terminated.float().sum()
            self.total_timeout_count += truncated.float().sum()
            self.total_success_count += success_mask.float().sum()

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
        q_t = self.robot.data.joint_pos
        q_dot_t = self.robot.data.joint_vel

        obj_pos, obj_quat, obj_lin_vel, obj_ang_vel = self._get_active_object_state()
        obj_rel_pos = obj_pos - self._env_origins()

        fingertip_pos = self.robot.data.body_pos_w[:, self.fingertip_indices, :]
        fingertip_rel_pos = (fingertip_pos - obj_pos.unsqueeze(1)).reshape(self.num_envs, -1)

        theta = self._compute_geodesic_distance(obj_quat, self.target_quats).unsqueeze(-1)
        q_err = math_utils.quat_mul(math_utils.quat_conjugate(obj_quat), self.target_quats)
        q_err = self._quat_normalize(q_err)
        axis_err = q_err[:, 1:4] * torch.sign(torch.clamp(q_err[:, 0:1], min=-1.0, max=1.0))

        actor_obs = torch.cat(
            [
                q_t,
                q_dot_t,
                self.u_t_minus_1,
                obj_rel_pos,
                obj_quat,
                obj_lin_vel,
                obj_ang_vel,
                fingertip_rel_pos,
                self._get_contact_bools(),
                torch.nn.functional.one_hot(self.active_shape, num_classes=2).float(),
                theta,
                axis_err,
            ],
            dim=-1,
        )

        if actor_obs.shape[-1] != int(self.cfg.num_observations):
            raise RuntimeError(
                f"Actor obs dim mismatch: got {actor_obs.shape[-1]}, expected {self.cfg.num_observations}"
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
                self.dr_friction.unsqueeze(-1),
                self.dr_com_offset,
            ],
            dim=-1,
        )

        if privileged_obs.shape[-1] != int(self.cfg.num_privileged_obs):
            raise RuntimeError(
                f"Privileged obs dim mismatch: got {privileged_obs.shape[-1]}, "
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
    def _compute_rewards(self, a_t: torch.Tensor, u_t: torch.Tensor, success_mask: torch.Tensor):
        obj_pos, obj_quat, obj_lin_vel, obj_ang_vel = self._get_active_object_state()
        obj_rel_pos = obj_pos - self._env_origins()
        theta = self._compute_geodesic_distance(obj_quat, self.target_quats)

        r_rot = float(self.cfg.w_rot) * torch.exp(-2.0 * torch.square(theta))

        progress = self.prev_theta - theta
        r_prog = float(self.cfg.w_prog) * torch.clamp(progress, min=-0.05, max=0.05)

        contact_bools = self._get_contact_bools()
        active_contacts = torch.sum(contact_bools, dim=-1)
        r_contact = float(self.cfg.w_contact) * active_contacts

        palm_dist = torch.norm(obj_rel_pos[:, :2], dim=-1)
        r_safe = torch.where(
            palm_dist < 0.05,
            torch.zeros_like(palm_dist),
            -float(self.cfg.w_safe) * (palm_dist - 0.05),
        )

        r_height = -float(self.cfg.w_height) * torch.square(obj_rel_pos[:, 2] - float(self.cfg.target_height))

        p_act_rate = -float(self.cfg.w_act_rate) * torch.sum(torch.square(u_t - self.u_t_minus_1), dim=-1)
        p_joint_vel = -float(self.cfg.w_joint_vel) * torch.sum(torch.square(self.robot.data.joint_vel), dim=-1)
        p_excess_vel = -float(self.cfg.w_excess_vel) * (
            torch.norm(obj_lin_vel, dim=-1) + torch.norm(obj_ang_vel, dim=-1)
        )

        step_reward = torch.clamp(
            r_rot + r_prog + r_contact + r_safe + r_height + p_act_rate + p_joint_vel + p_excess_vel,
            min=float(self.cfg.reward_clip_min),
            max=float(self.cfg.reward_clip_max),
        )

        is_drop = obj_rel_pos[:, 2] < float(self.cfg.drop_height)

        r_event = torch.where(
            is_drop,
            torch.full_like(theta, float(self.cfg.penalty_drop)),
            torch.zeros_like(theta),
        )

        r_event = r_event + success_mask.float() * float(self.cfg.bonus_success)

        r_event = r_event + torch.where(
            (obj_rel_pos[:, 2] >= float(self.cfg.drop_height))
            & (obj_rel_pos[:, 2] < float(self.cfg.pre_drop_height)),
            -float(self.cfg.w_pre_drop) * (float(self.cfg.pre_drop_height) - obj_rel_pos[:, 2]),
            torch.zeros_like(theta),
        )

        total_reward = step_reward + r_event
        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=10.0, neginf=float(self.cfg.penalty_drop))

        total_done_safe = torch.clamp(self.total_done_count, min=1.0)

        info = {
            "reward_components": {
                "R_Rot": r_rot.detach().mean(),
                "R_Prog": r_prog.detach().mean(),
                "R_Contact": r_contact.detach().mean(),
                "R_Safe": r_safe.detach().mean(),
                "R_Height": r_height.detach().mean(),
                "P_Act_Rate": p_act_rate.detach().mean(),
                "P_Joint_Vel": p_joint_vel.detach().mean(),
                "P_Excess_Vel": p_excess_vel.detach().mean(),
                "Event_Bonus_Penalty": r_event.detach().mean(),
                "Total_Reward": total_reward.detach().mean(),
            },
            "events": {
                "Drop_Rate": is_drop.float().mean().detach(),
                "Success_Rate": success_mask.float().mean().detach(),
                "Timeout_Rate": (self.episode_steps >= int(self.cfg.max_episode_length)).float().mean().detach(),
                "Episode_Drop_Total_Rate": self.total_drop_count.detach() / total_done_safe,
                "Episode_Success_Total_Rate": self.total_success_count.detach() / total_done_safe,
                "Episode_Timeout_Total_Rate": self.total_timeout_count.detach() / total_done_safe,
            },
            "telemetry": {
                "Geodesic_Error_Rad": theta.detach().mean(),
                "Geodesic_Error_Max_Rad": theta.detach().max(),
                "Object_Height": obj_rel_pos[:, 2].detach().mean(),
                "Height_Error": torch.abs(obj_rel_pos[:, 2] - float(self.cfg.target_height)).detach().mean(),
                "Object_XY_Distance": torch.norm(obj_rel_pos[:, :2], dim=-1).detach().mean(),
                "Object_Lin_Vel": torch.norm(obj_lin_vel, dim=-1).detach().mean(),
                "Object_Ang_Vel": torch.norm(obj_ang_vel, dim=-1).detach().mean(),
                "Active_Contacts": active_contacts.detach().mean(),
                "Curriculum_Progress_K": torch.tensor(self.curriculum_k(), dtype=torch.float32, device=self.device),
                "Episode_Length": self.episode_steps.float().detach().mean(),
                "Global_Steps": torch.tensor(float(self.global_steps), dtype=torch.float32, device=self.device),
            },
            "debug": {
                "Actor_Obs_Dim": torch.tensor(float(self.cfg.num_observations), dtype=torch.float32, device=self.device),
                "Privileged_Obs_Dim": torch.tensor(float(self.cfg.num_privileged_obs), dtype=torch.float32, device=self.device),
                "Action_Dim": torch.tensor(float(self.cfg.num_actions), dtype=torch.float32, device=self.device),
                "Reward_Min": total_reward.detach().min(),
                "Reward_Max": total_reward.detach().max(),
                "Cube_Active_Ratio": (self.active_shape == 0).float().mean().detach(),
                "Sphere_Active_Ratio": (self.active_shape == 1).float().mean().detach(),
                "DR_Mass_Mean": self.dr_mass.detach().mean(),
                "DR_Friction_Mean": self.dr_friction.detach().mean(),
            },
        }

        return total_reward, info, is_drop


Task2Env = AllegroHandTask2Env
AllegroTask2Env = AllegroHandTask2Env
