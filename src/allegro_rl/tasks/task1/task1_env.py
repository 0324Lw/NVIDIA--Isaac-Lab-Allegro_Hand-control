from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

from allegro_rl.tasks.task1.task1_config import Task1Config


# =============================================================================
# Scene configuration
# =============================================================================
def make_allegro_task1_scene_cfg(cfg: Task1Config):
    """Create an Isaac Lab InteractiveSceneCfg for Allegro Hand Task1.

    This function is intentionally defined as a factory so num_envs and
    env_spacing can come from Task1Config without relying on global state.
    """

    @configclass
    class AllegroHandTask1SceneCfg(InteractiveSceneCfg):
        num_envs: int = int(cfg.num_envs)
        env_spacing: float = float(cfg.env_spacing)

        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )

        robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        def __post_init__(self):
            super().__post_init__()

            self.robot.spawn.fix_base = True
            self.robot.spawn.activate_contact_sensors = False
            self.robot.init_state.pos = (0.0, 0.0, float(cfg.hand_init_height))

    return AllegroHandTask1SceneCfg


# =============================================================================
# Pose dataset
# =============================================================================
class PoseDataset:
    """GPU target-pose dataset manager for Allegro Hand Task1."""

    def __init__(self, data_path: str, device: str):
        self.data_path = str(Path(data_path).expanduser().resolve())
        self.device = device

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"找不到 Allegro Task1 姿态数据集: {self.data_path}\n"
                "请先运行：bash scripts/ubuntu/generate_task1_dataset.sh"
            )

        self.data = torch.load(self.data_path, map_location=device)

        self.ctrl_min = self.data["ctrl_min"].to(device=device, dtype=torch.float32)
        self.ctrl_max = self.data["ctrl_max"].to(device=device, dtype=torch.float32)

        self.pool_easy = self.data["random_easy"].to(device=device, dtype=torch.float32)
        self.pool_hard = self.data["random_hard"].to(device=device, dtype=torch.float32)

        if "semantic_tensor" in self.data:
            self.pool_semantic = self.data["semantic_tensor"].to(device=device, dtype=torch.float32)
            self.semantic_names = list(self.data.get("semantic_names", []))
        else:
            semantic_poses = self.data["semantic_poses"]
            self.semantic_names = list(semantic_poses.keys())
            self.pool_semantic = torch.stack(
                [semantic_poses[name].to(device=device, dtype=torch.float32) for name in self.semantic_names],
                dim=0,
            )

        self.joint_order = list(
            self.data.get(
                "joint_order",
                [
                    "ffj0", "ffj1", "ffj2", "ffj3",
                    "mfj0", "mfj1", "mfj2", "mfj3",
                    "rfj0", "rfj1", "rfj2", "rfj3",
                    "thj0", "thj1", "thj2", "thj3",
                ],
            )
        )

        self.num_actions = int(self.ctrl_min.numel())

        self._validate()

    def _validate(self) -> None:
        assert self.num_actions == 16, f"Allegro Task1 expects 16 DoF, got {self.num_actions}"
        assert self.ctrl_min.shape == (16,)
        assert self.ctrl_max.shape == (16,)
        assert self.pool_easy.ndim == 2 and self.pool_easy.shape[1] == 16
        assert self.pool_hard.ndim == 2 and self.pool_hard.shape[1] == 16
        assert self.pool_semantic.ndim == 2 and self.pool_semantic.shape[1] == 16
        assert torch.all(self.ctrl_min < self.ctrl_max)

    def _probabilities(self, k: float, cfg: Task1Config) -> tuple[float, float, float]:
        """Return dynamic probabilities for semantic / hard / easy pools."""

        k = float(max(0.0, min(1.0, k)))

        easy = float(cfg.easy_prob_start) + k * (float(cfg.easy_prob_end) - float(cfg.easy_prob_start))
        hard = float(cfg.hard_prob_start) + k * (float(cfg.hard_prob_end) - float(cfg.hard_prob_start))
        semantic = float(cfg.semantic_prob_start) + k * (
            float(cfg.semantic_prob_end) - float(cfg.semantic_prob_start)
        )

        total = max(easy + hard + semantic, 1e-6)
        easy /= total
        hard /= total
        semantic /= total

        return semantic, hard, easy

    @torch.no_grad()
    def sample_targets(self, num_samples: int, curriculum_k: float, cfg: Task1Config) -> torch.Tensor:
        """Sample target poses with curriculum-dependent pool probabilities."""

        n = int(num_samples)
        semantic_prob, hard_prob, easy_prob = self._probabilities(curriculum_k, cfg)

        rand = torch.rand(n, device=self.device)

        mask_semantic = rand < semantic_prob
        mask_hard = (rand >= semantic_prob) & (rand < semantic_prob + hard_prob)
        mask_easy = rand >= semantic_prob + hard_prob

        targets = torch.empty((n, 16), dtype=torch.float32, device=self.device)

        if mask_semantic.any():
            idx = torch.randint(
                0,
                self.pool_semantic.shape[0],
                (int(mask_semantic.sum().item()),),
                device=self.device,
            )
            targets[mask_semantic] = self.pool_semantic[idx]

        if mask_hard.any():
            idx = torch.randint(
                0,
                self.pool_hard.shape[0],
                (int(mask_hard.sum().item()),),
                device=self.device,
            )
            targets[mask_hard] = self.pool_hard[idx]

        if mask_easy.any():
            idx = torch.randint(
                0,
                self.pool_easy.shape[0],
                (int(mask_easy.sum().item()),),
                device=self.device,
            )
            targets[mask_easy] = self.pool_easy[idx]

        return torch.clamp(targets, self.ctrl_min, self.ctrl_max)


# =============================================================================
# Environment
# =============================================================================
class AllegroHandTask1Env(gym.Env):
    """Allegro Hand Task1: pure-RL target pose tracking.

    Action:
        16-D residual action in [-1, 1].

    Observation:
        64-D vector:
            target_error      16
            joint_position    16
            joint_velocity    16
            last_filtered_cmd 16

    Reward:
        Tracking + progress + stable bonus - action magnitude - action smoothness
        - soft joint-limit penalty.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Task1Config):
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
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(cfg.num_observations),),
            dtype=np.float32,
        )
        self.state_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(cfg.num_observations),),
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

        SceneCfg = make_allegro_task1_scene_cfg(cfg)
        self.scene = InteractiveScene(SceneCfg(num_envs=int(cfg.num_envs)))

        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]

        self.robot_joint_names = list(self.robot.joint_names)
        self.num_envs = int(cfg.num_envs)
        self.num_actions = int(cfg.num_actions)
        self.num_observations = int(cfg.num_observations)

        if self.robot.num_joints != self.num_actions:
            raise RuntimeError(
                f"Allegro Hand Task1 expects robot.num_joints == {self.num_actions}, "
                f"but got {self.robot.num_joints}. joint_names={self.robot_joint_names}"
            )

        self.dataset = PoseDataset(cfg.dataset_path, self.device)
        self.dataset_to_robot_index = self._build_dataset_to_robot_index()

        self.joint_limits = self.robot.data.joint_pos_limits.clone()
        self.default_pos = self.robot.data.default_joint_pos.clone()
        self.default_vel = torch.zeros_like(self.default_pos)

        dataset_ctrl_min_robot = self._dataset_to_robot_order(self.dataset.ctrl_min)
        dataset_ctrl_max_robot = self._dataset_to_robot_order(self.dataset.ctrl_max)

        self.ctrl_min = torch.maximum(
            self.joint_limits[:, :, 0],
            dataset_ctrl_min_robot.unsqueeze(0).expand(self.num_envs, -1),
        )
        self.ctrl_max = torch.minimum(
            self.joint_limits[:, :, 1],
            dataset_ctrl_max_robot.unsqueeze(0).expand(self.num_envs, -1),
        )

        self.global_steps = 0
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.target_poses = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.current_targets = torch.zeros_like(self.target_poses)

        self.previous_error = torch.zeros_like(self.target_poses)
        self.a_t_minus_1 = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.u_t_minus_1 = torch.zeros_like(self.a_t_minus_1)

        self.total_timeout_episodes = torch.zeros((), dtype=torch.float32, device=self.device)
        self.total_done_episodes = torch.zeros((), dtype=torch.float32, device=self.device)

        if cfg.print_debug_info:
            self._print_debug_info()

        self.reset()


    # ------------------------------------------------------------------
    # Joint order mapping
    # ------------------------------------------------------------------
    def _infer_canonical_joint_key(self, robot_joint_name: str) -> str | None:
        """Infer canonical dataset key from an IsaacLab Allegro joint name.

        Dataset canonical order:
            ffj0..ffj3, mfj0..mfj3, rfj0..rfj3, thj0..thj3

        IsaacLab / USD joint names may use words such as index / middle /
        ring / thumb, or abbreviations ff / mf / rf / th. This function tries
        to recover the canonical key robustly.
        """
        name = robot_joint_name.lower()

        finger = None
        if "index" in name or "ff" in name:
            finger = "ff"
        elif "middle" in name or "mf" in name:
            finger = "mf"
        elif "ring" in name or "rf" in name:
            finger = "rf"
        elif "thumb" in name or "th" in name:
            finger = "th"

        if finger is None:
            return None

        joint_idx = None

        patterns = [
            r"(?:joint|j)[_\-\. ]*([0-3])",
            r"([0-3])$",
            r"([0-3])\D*$",
        ]

        for pattern in patterns:
            m = re.search(pattern, name)
            if m is not None:
                joint_idx = int(m.group(1))
                break

        if joint_idx is None:
            return None

        return f"{finger}j{joint_idx}"

    def _build_dataset_to_robot_index(self) -> torch.Tensor:
        """Build index tensor so dataset[:, idx] becomes robot joint order."""
        canonical_order = list(self.dataset.joint_order)
        canonical_to_idx = {name: i for i, name in enumerate(canonical_order)}

        inferred = []
        missing = []

        for robot_name in self.robot_joint_names:
            key = self._infer_canonical_joint_key(robot_name)
            if key is None or key not in canonical_to_idx:
                inferred.append(None)
                missing.append((robot_name, key))
            else:
                inferred.append(canonical_to_idx[key])

        # If all joints are recognized and unique, use the inferred mapping.
        if all(v is not None for v in inferred) and len(set(inferred)) == len(inferred):
            index = torch.tensor(inferred, dtype=torch.long, device=self.device)
            print("[INFO] Allegro dataset canonical order mapped to robot joint order:")
            for robot_i, data_i in enumerate(inferred):
                print(f"  robot[{robot_i:02d}] {self.robot_joint_names[robot_i]:<32} <- dataset[{data_i:02d}] {canonical_order[data_i]}")
            return index

        # Fallback: identity mapping. This is safe for assets whose joint order
        # already matches the generated dataset order.
        print("[WARN] Could not fully infer Allegro joint order from robot joint names.")
        print("[WARN] Missing / ambiguous joints:")
        for robot_name, key in missing:
            print(f"  robot joint: {robot_name}, inferred key: {key}")
        print("[WARN] Fallback to identity dataset order. Use --print-joints in the test if this is wrong.")

        return torch.arange(16, dtype=torch.long, device=self.device)

    def _dataset_to_robot_order(self, x: torch.Tensor) -> torch.Tensor:
        """Convert dataset canonical pose tensor to IsaacLab robot joint order."""
        return x.index_select(dim=-1, index=self.dataset_to_robot_index)


    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------
    def _print_debug_info(self) -> None:
        print("\n" + "=" * 100)
        print("[AllegroHandTask1Env] initialized")
        print(f"num_envs          : {self.num_envs}")
        print(f"num_actions       : {self.num_actions}")
        print(f"num_observations  : {self.num_observations}")
        print(f"robot.num_joints  : {self.robot.num_joints}")
        print(f"dataset_path      : {self.cfg.dataset_path}")
        print(f"semantic_names    : {self.dataset.semantic_names}")
        print(f"joint_names       : {self.robot_joint_names}")
        print("=" * 100 + "\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def curriculum_k(self) -> float:
        return min(1.0, float(self.global_steps) / max(float(self.cfg.curriculum_total_steps), 1.0))

    def _mean_detached(self, x: torch.Tensor | float) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(float(x), dtype=torch.float32, device=self.device)
        return x.detach().float().mean()

    def _float_tensor(self, x: float) -> torch.Tensor:
        return torch.tensor(float(x), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _sample_initial_joint_positions(self, env_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(targets) * float(self.cfg.init_noise_scale)
        init_pos = targets + noise

        # Blend with default open-ish pose in the early curriculum.
        k = self.curriculum_k()
        default_subset = self.default_pos[env_ids]
        blend = max(0.0, 1.0 - 2.0 * k)
        init_pos = (1.0 - blend) * init_pos + blend * default_subset

        init_pos = torch.clamp(init_pos, self.ctrl_min[env_ids], self.ctrl_max[env_ids])
        return init_pos

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset(
        self,
        env_ids: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        n = int(env_ids.numel())

        if n == 0:
            return self._compute_obs(), {}

        targets = self.dataset.sample_targets(n, self.curriculum_k(), self.cfg)
        targets = self._dataset_to_robot_order(targets)

        # Dataset targets are generated from hand-written Allegro limits.
        # IsaacLab USD joint limits can be slightly more conservative, so the
        # environment must clamp targets again with the real simulated limits.
        targets = torch.clamp(targets, self.ctrl_min[env_ids], self.ctrl_max[env_ids])

        self.target_poses[env_ids] = targets
        self.current_targets[env_ids] = targets

        init_pos = self._sample_initial_joint_positions(env_ids, targets)
        init_vel = torch.zeros_like(init_pos)

        self.robot.write_joint_state_to_sim(init_pos, init_vel, env_ids=env_ids)
        self.robot.reset(env_ids)

        self.scene.update(dt=0.0)

        self.a_t_minus_1[env_ids] = 0.0
        self.u_t_minus_1[env_ids] = 0.0
        self.episode_steps[env_ids] = 0
        self.episode_return[env_ids] = 0.0

        self.previous_error[env_ids] = self.current_targets[env_ids] - self.robot.data.joint_pos[env_ids]

        obs = self._compute_obs()
        return obs, {}

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = torch.clamp(actions, -1.0, 1.0)

        u_t = float(self.cfg.ema_alpha) * actions + (1.0 - float(self.cfg.ema_alpha)) * self.u_t_minus_1

        q_current = self.robot.data.joint_pos
        cmd_target = q_current + u_t * float(self.cfg.action_scale)
        cmd_target = torch.clamp(cmd_target, self.ctrl_min, self.ctrl_max)

        self.robot.set_joint_position_target(cmd_target)
        self.scene.write_data_to_sim()

        for _ in range(int(self.cfg.decimation)):
            self.sim.step()
            self.scene.update(dt=float(self.cfg.sim_dt))

        self.global_steps += self.num_envs
        self.episode_steps += 1

        rewards, info = self._compute_rewards(actions, u_t)
        self.episode_return += rewards

        self.previous_error = self.current_targets - self.robot.data.joint_pos
        self.a_t_minus_1 = actions.clone()
        self.u_t_minus_1 = u_t.clone()

        truncated = self.episode_steps >= int(self.cfg.max_episode_length)
        terminated = torch.zeros_like(truncated, dtype=torch.bool)
        done = truncated | terminated

        if done.any():
            self.total_done_episodes += done.float().sum()
            self.total_timeout_episodes += truncated.float().sum()

        obs = self._compute_obs()

        reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            self.reset(reset_ids)

        return obs, rewards, terminated, truncated, info

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _compute_obs(self) -> torch.Tensor:
        q_t = self.robot.data.joint_pos
        q_dot_t = self.robot.data.joint_vel
        e_t = self.current_targets - q_t

        obs = torch.cat(
            [
                e_t,
                q_t,
                q_dot_t,
                self.u_t_minus_1,
            ],
            dim=-1,
        )

        if obs.shape[-1] != int(self.cfg.num_observations):
            raise RuntimeError(
                f"Allegro Task1 obs dim mismatch: got {obs.shape[-1]}, "
                f"expected {self.cfg.num_observations}"
            )

        obs = torch.nan_to_num(
            torch.clamp(obs, -float(self.cfg.obs_clip), float(self.cfg.obs_clip)),
            nan=0.0,
            posinf=float(self.cfg.obs_clip),
            neginf=-float(self.cfg.obs_clip),
        )

        return obs

    def _compute_states(self) -> torch.Tensor:
        return self._compute_obs()

    def get_privileged_observations(self) -> torch.Tensor:
        return self._compute_obs()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_rewards(self, a_t: torch.Tensor, u_t: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        q_t = self.robot.data.joint_pos
        q_dot_t = self.robot.data.joint_vel
        e_t = self.current_targets - q_t

        sq_error = torch.square(e_t)
        mean_sq_error = torch.mean(sq_error, dim=-1)
        worst_sq_error = torch.max(sq_error, dim=-1).values

        r_track_mean = float(self.cfg.w_track_mean) * torch.exp(-float(self.cfg.track_sigma) * mean_sq_error)
        r_track_worst = float(self.cfg.w_track_worst) * torch.exp(-float(self.cfg.track_sigma) * worst_sq_error)
        r_track = r_track_mean + r_track_worst

        prev_mean_abs_error = torch.mean(torch.abs(self.previous_error), dim=-1)
        curr_mean_abs_error = torch.mean(torch.abs(e_t), dim=-1)
        progress = prev_mean_abs_error - curr_mean_abs_error
        r_progress = float(self.cfg.w_progress) * torch.clamp(progress, min=0.0)

        mean_sq_vel = torch.mean(torch.square(q_dot_t), dim=-1)
        r_stable = (
            float(self.cfg.w_stable)
            * torch.exp(-float(self.cfg.stable_err_sigma) * curr_mean_abs_error)
            * torch.exp(-float(self.cfg.stable_vel_sigma) * mean_sq_vel)
        )

        p_act = -float(self.cfg.w_act_mag) * torch.mean(torch.square(a_t), dim=-1)
        p_smooth = -float(self.cfg.w_act_smooth) * torch.mean(torch.square(u_t - self.u_t_minus_1), dim=-1)

        upper_limit = self.ctrl_max - float(self.cfg.limit_margin)
        lower_limit = self.ctrl_min + float(self.cfg.limit_margin)

        violation_upper = torch.clamp(q_t - upper_limit, min=0.0)
        violation_lower = torch.clamp(lower_limit - q_t, min=0.0)

        p_limit_raw = -float(self.cfg.w_soft_limit) * torch.mean(
            torch.square(violation_upper) + torch.square(violation_lower),
            dim=-1,
        )
        p_limit = torch.clamp(p_limit_raw, min=float(self.cfg.soft_limit_clip))

        total_reward = r_track + r_progress + r_stable + p_act + p_smooth + p_limit
        total_reward = torch.nan_to_num(
            torch.clamp(total_reward, -float(self.cfg.reward_clip_abs), float(self.cfg.reward_clip_abs)),
            nan=0.0,
            posinf=float(self.cfg.reward_clip_abs),
            neginf=-float(self.cfg.reward_clip_abs),
        )

        timeout_rate = (self.episode_steps >= int(self.cfg.max_episode_length)).float()
        total_done_safe = torch.clamp(self.total_done_episodes, min=1.0)

        info = {
            "reward_components": {
                "R_Track_Mean": self._mean_detached(r_track_mean),
                "R_Track_Worst": self._mean_detached(r_track_worst),
                "R_Track": self._mean_detached(r_track),
                "R_Progress": self._mean_detached(r_progress),
                "R_Stable": self._mean_detached(r_stable),
                "P_Action_Mag": self._mean_detached(p_act),
                "P_Action_Smooth": self._mean_detached(p_smooth),
                "P_Soft_Limit": self._mean_detached(p_limit),
                "Total_Reward": self._mean_detached(total_reward),
            },
            "events": {
                "Timeout_Rate": self._mean_detached(timeout_rate),
                "Done_Rate": self._mean_detached(timeout_rate),
                "Episode_Timeout_Total_Rate": self.total_timeout_episodes / total_done_safe,
            },
            "telemetry": {
                "Pose_Error_Rad": self._mean_detached(curr_mean_abs_error),
                "Pose_Error_Max_Rad": self._mean_detached(torch.max(torch.abs(e_t), dim=-1).values),
                "Mean_Sq_Error": self._mean_detached(mean_sq_error),
                "Worst_Sq_Error": self._mean_detached(worst_sq_error),
                "Joint_Velocity": self._mean_detached(torch.mean(torch.abs(q_dot_t), dim=-1)),
                "Action_Mean_Abs": self._mean_detached(torch.mean(torch.abs(a_t), dim=-1)),
                "Filtered_Action_Mean_Abs": self._mean_detached(torch.mean(torch.abs(u_t), dim=-1)),
                "Curriculum_Progress_K": self._float_tensor(self.curriculum_k()),
                "Episode_Length": self._mean_detached(self.episode_steps.float()),
                "Episode_Return": self._mean_detached(self.episode_return),
                "Global_Steps": self._float_tensor(float(self.global_steps)),
            },
            "debug": {
                "Obs_Dim": self._float_tensor(float(self.cfg.num_observations)),
                "Action_Dim": self._float_tensor(float(self.cfg.num_actions)),
                "Reward_Min": total_reward.detach().min(),
                "Reward_Max": total_reward.detach().max(),
                "Ctrl_Min": self._mean_detached(self.ctrl_min),
                "Ctrl_Max": self._mean_detached(self.ctrl_max),
                "Q_Min": q_t.detach().min(),
                "Q_Max": q_t.detach().max(),
            },
        }

        return total_reward, info


Task1Env = AllegroHandTask1Env
AllegroTask1Env = AllegroHandTask1Env
