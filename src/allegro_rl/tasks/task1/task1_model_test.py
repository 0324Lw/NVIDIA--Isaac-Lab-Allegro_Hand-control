from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate Allegro Hand Task1 TRUE skrl PPO model")

parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=1.0)
parser.add_argument("--print-interval", type=int, default=20)
parser.add_argument("--dataset-path", type=str, default=os.environ.get("ALLEGRO_TASK1_DATASET", ""))
parser.add_argument("--frame-stack", type=int, default=5)
parser.add_argument("--visualize", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = not bool(args_cli.visualize)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from allegro_rl.common.info_utils import flat_dict
from allegro_rl.tasks.task1.task1_config import Task1Config
from allegro_rl.tasks.task1.task1_env import AllegroHandTask1Env

from skrl.models.torch import GaussianMixin, Model


class AllegroPolicy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        init_log_std: float = -1.0,
        min_log_std: float = -4.0,
        max_log_std: float = 0.5,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=True,
            clip_log_std=True,
            min_log_std=float(min_log_std),
            max_log_std=float(max_log_std),
            reduction="sum",
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
            nn.Tanh(),
        )

        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(init_log_std)))

    def compute(self, inputs, role):
        states = inputs["states"]
        mean_actions = self.net(states)
        return mean_actions, self.log_std_parameter, {}

    @torch.no_grad()
    def act_deterministic_direct(self, states: torch.Tensor) -> torch.Tensor:
        actions, _, _ = self.compute({"states": states}, role="policy")
        return torch.clamp(actions, -1.0, 1.0)


class EvalFrameStackWrapper(gym.Env):
    def __init__(self, env: AllegroHandTask1Env, n_stack: int = 5):
        super().__init__()

        self.env = env
        self.num_envs = int(env.cfg.num_envs)
        self.device = env.device
        self.n_stack = int(n_stack)

        self.single_obs_dim = int(env.observation_space.shape[0])
        self.stacked_obs_dim = self.single_obs_dim * self.n_stack

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.stacked_obs_dim,),
            dtype=np.float32,
        )
        self.action_space = env.action_space

        self.obs_stack = torch.zeros(
            (self.num_envs, self.stacked_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)

        for i in range(self.n_stack):
            self.obs_stack[:, i * self.single_obs_dim : (i + 1) * self.single_obs_dim] = obs

        return self.obs_stack.clone(), info or {}

    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        self.obs_stack[:, :-self.single_obs_dim] = self.obs_stack[:, self.single_obs_dim :].clone()
        self.obs_stack[:, -self.single_obs_dim :] = obs

        done = terminated | truncated
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            for i in range(self.n_stack):
                self.obs_stack[ids, i * self.single_obs_dim : (i + 1) * self.single_obs_dim] = obs[ids]

        return self.obs_stack.clone(), rewards, terminated, truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def resolve_checkpoint(path: str) -> Path:
    p = Path(path).expanduser().resolve()

    if p.is_file():
        return p

    if p.is_dir():
        candidates = [
            p / "allegro_task1_model.pt",
            p / "final_checkpoint" / "allegro_task1_model.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        pt_files = sorted(p.glob("*.pt"))
        if pt_files:
            return pt_files[0]

    return p


def normalize_with_saved_obs_norm(obs: torch.Tensor, obs_norm: Optional[Dict[str, Any]]) -> torch.Tensor:
    if not obs_norm:
        return obs

    mean = obs_norm.get("mean", None)
    var = obs_norm.get("var", None)
    clip = float(obs_norm.get("clip", 10.0))

    if mean is None or var is None:
        return obs

    mean = mean.to(device=obs.device, dtype=torch.float32)
    var = var.to(device=obs.device, dtype=torch.float32)

    if mean.numel() != obs.shape[-1] or var.numel() != obs.shape[-1]:
        return obs

    return torch.clamp((obs - mean) / torch.sqrt(var + 1e-8), -clip, clip)


def load_policy_checkpoint(ckpt_path: Path, env: EvalFrameStackWrapper):
    ckpt = torch.load(str(ckpt_path), map_location=env.device)

    metadata = ckpt.get("metadata", {})
    args = ckpt.get("args", {})

    if not bool(metadata.get("uses_skrl", False)):
        raise RuntimeError(
            f"Checkpoint is not marked as skrl PPO: {ckpt_path}\n"
            "请使用新版本 task1_train.py 重新训练，旧 custom PPO checkpoint 不再兼容。"
        )

    policy = AllegroPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
        init_log_std=float(args.get("init_log_std", -1.0)),
        min_log_std=float(args.get("min_log_std", -4.0)),
        max_log_std=float(args.get("max_log_std", 0.5)),
    ).to(env.device)

    policy.load_state_dict(ckpt["policy"], strict=True)
    policy.eval()

    obs_norm = ckpt.get("obs_norm", None)
    train_env_steps = int(ckpt.get("env_steps", 0))

    return policy, obs_norm, train_env_steps, metadata


def force_eval_curriculum(env: AllegroHandTask1Env, start_k: float, label: str) -> None:
    k = max(0.0, min(1.0, float(start_k)))
    env.global_steps = int(k * env.cfg.curriculum_total_steps)

    ids = torch.arange(env.cfg.num_envs, dtype=torch.long, device=env.device)
    env.reset(ids)

    print(
        f"[CURRICULUM][{label}] forced start_k={k:.4f}, "
        f"global_steps={env.global_steps:,}",
        flush=True,
    )


def summarize(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}

    keys = sorted({key for row in records for key in row.keys()})
    out: Dict[str, Dict[str, float]] = {}

    for key in keys:
        vals = np.asarray([row[key] for row in records if key in row], dtype=np.float64)
        if vals.size == 0:
            continue

        out[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "max": float(np.max(vals)),
        }

    return out


def print_summary_table(summary: Dict[str, Dict[str, float]]) -> None:
    print("\n" + "=" * 176)
    print("Allegro Hand Task1 TRUE skrl PPO Model Test Summary")
    print("=" * 176)
    print(
        f"{'metric':<78} | {'mean':>12} | {'std':>12} | {'min':>12} | "
        f"{'p25':>12} | {'p50':>12} | {'p75':>12} | {'max':>12}"
    )
    print("-" * 176)

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<78} | "
            f"{row['mean']:>12.6f} | "
            f"{row['std']:>12.6f} | "
            f"{row['min']:>12.6f} | "
            f"{row['p25']:>12.6f} | "
            f"{row['p50']:>12.6f} | "
            f"{row['p75']:>12.6f} | "
            f"{row['max']:>12.6f}"
        )

    print("=" * 176 + "\n")


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    cfg = Task1Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)
    cfg.seed = int(args_cli.seed)

    if args_cli.dataset_path:
        cfg.dataset_path = str(Path(args_cli.dataset_path).expanduser().resolve())

    cfg.validate()

    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(
            f"Task1 dataset not found: {cfg.dataset_path}\n"
            "Please run: bash scripts/ubuntu/generate_task1_dataset.sh"
        )

    base_env = AllegroHandTask1Env(cfg)
    force_eval_curriculum(base_env, args_cli.start_k, "after_env_creation")

    env = EvalFrameStackWrapper(base_env, n_stack=int(args_cli.frame_stack))
    obs, _ = env.reset(seed=int(args_cli.seed))

    force_eval_curriculum(base_env, args_cli.start_k, "after_rollout_reset")
    obs, _ = env.reset(seed=int(args_cli.seed))

    ckpt_path = resolve_checkpoint(args_cli.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    policy, obs_norm, trained_env_steps, metadata = load_policy_checkpoint(ckpt_path, env)

    print("\n" + "=" * 150)
    print("Allegro Hand Task1 TRUE skrl PPO model test started")
    print("=" * 150)
    print(f"checkpoint        : {ckpt_path}")
    print(f"trained_env_steps : {trained_env_steps:,}")
    print(f"num_envs          : {base_env.num_envs}")
    print(f"steps             : {args_cli.steps}")
    print(f"start_k           : {args_cli.start_k}")
    print(f"frame_stack       : {args_cli.frame_stack}")
    print(f"device            : {base_env.device}")
    print(f"dataset_path      : {cfg.dataset_path}")
    print(f"visualize         : {bool(args_cli.visualize)}")
    print("algorithm         : skrl PPO")
    print("note              : deterministic direct policy forward; no agent.act")
    print("=" * 150 + "\n")

    records: List[Dict[str, float]] = []
    total_terminated = 0
    total_truncated = 0
    start_time = time.time()

    try:
        with tqdm(
            total=int(args_cli.steps),
            desc="Allegro Task1 skrl Model Test",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:
            for step in range(int(args_cli.steps)):
                if step < 3:
                    print(f"[DEBUG][eval step {step}] before policy forward", flush=True)

                with torch.no_grad():
                    obs_n = normalize_with_saved_obs_norm(obs, obs_norm)
                    actions = policy.act_deterministic_direct(obs_n)

                if step < 3:
                    print(f"[DEBUG][eval step {step}] after policy forward", flush=True)
                    print(f"[DEBUG][eval step {step}] before env.step", flush=True)

                obs, rewards, terminated, truncated, info = env.step(actions)

                if step < 3:
                    print(f"[DEBUG][eval step {step}] after env.step", flush=True)

                total_terminated += int(terminated.sum().item())
                total_truncated += int(truncated.sum().item())

                if step % max(int(args_cli.print_interval), 1) == 0 or step == int(args_cli.steps) - 1:
                    flat = flat_dict(info)
                    row = {
                        "test/reward_mean": float(rewards.detach().float().mean().cpu().item()),
                        "test/reward_min": float(rewards.detach().float().min().cpu().item()),
                        "test/reward_max": float(rewards.detach().float().max().cpu().item()),
                        "test/terminated_rate": float(terminated.float().mean().cpu().item()),
                        "test/truncated_rate": float(truncated.float().mean().cpu().item()),
                    }
                    row.update(flat)
                    records.append(row)

                    pbar.set_postfix(
                        {
                            "rew": f"{row['test/reward_mean']:+.3f}",
                            "err": f"{flat.get('telemetry/Pose_Error_Rad', 0.0):.3f}",
                            "maxerr": f"{flat.get('telemetry/Pose_Error_Max_Rad', 0.0):.3f}",
                            "vel": f"{flat.get('telemetry/Joint_Velocity', 0.0):.3f}",
                            "k": f"{flat.get('telemetry/Curriculum_Progress_K', 0.0):.2f}",
                        }
                    )

                    if bool(args_cli.visualize):
                        sys.stdout.write(
                            f"\r🤖 K={flat.get('telemetry/Curriculum_Progress_K', 0.0):.2f} | "
                            f"PoseErr={flat.get('telemetry/Pose_Error_Rad', 0.0):.4f} rad | "
                            f"MaxErr={flat.get('telemetry/Pose_Error_Max_Rad', 0.0):.4f} rad | "
                            f"JointVel={flat.get('telemetry/Joint_Velocity', 0.0):.4f} rad/s"
                        )
                        sys.stdout.flush()

                pbar.update(1)

                if bool(args_cli.visualize) and not simulation_app.is_running():
                    print("\n[INFO] Isaac Sim window closed.")
                    break

        elapsed = time.time() - start_time
        env_steps = int(args_cli.steps) * int(base_env.num_envs)
        fps = env_steps / max(elapsed, 1e-6)

        print("\n✅ Allegro Task1 TRUE skrl PPO model test rollout finished")
        print(f"  env steps        : {env_steps:,}")
        print(f"  fps              : {fps:,.2f}")
        print(f"  total terminated : {total_terminated:,}")
        print(f"  total truncated  : {total_truncated:,}")

        print_summary_table(summarize(records))

        print("Allegro Task1 model test checklist:")
        print("1. checkpoint metadata 必须标记 uses_skrl=True。")
        print("2. 默认 start_k=1.0，测试最终姿态采样难度。")
        print("3. 测试脚本不调用 agent.act，避免模型测试卡在 0%。")
        print("4. smoke checkpoint 效果差是正常的，先看推理稳定性和无 NaN/Inf。")
        print("5. 正式效果重点看 Pose_Error_Rad、Pose_Error_Max_Rad、R_Track、R_Stable。")

    finally:
        try:
            env.close()
        except Exception:
            pass

        try:
            simulation_app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
