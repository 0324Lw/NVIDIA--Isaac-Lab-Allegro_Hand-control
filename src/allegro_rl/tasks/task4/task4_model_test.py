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

parser = argparse.ArgumentParser(description="Evaluate Allegro Hand Task4 TRUE skrl Teacher PPO model")

parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=1.0)
parser.add_argument("--print-interval", type=int, default=20)
parser.add_argument("--frame-stack", type=int, default=5)
parser.add_argument("--visualize", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = not bool(args_cli.visualize)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from allegro_rl.common.info_utils import flat_dict
from allegro_rl.tasks.task4.task4_config import Task4Config
from allegro_rl.tasks.task4.task4_env import AllegroHandTask4Env

from skrl.models.torch import GaussianMixin, Model


class Task4TeacherActor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        init_log_std: float = 0.0,
        min_log_std: float = -20.0,
        max_log_std: float = 2.0,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
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
            nn.Linear(observation_space.shape[0], 1024),
            nn.ELU(),
            nn.Linear(1024, 768),
            nn.ELU(),
            nn.Linear(768, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_space.shape[0]),
        )

        self.log_std_parameter = nn.Parameter(torch.full((action_space.shape[0],), float(init_log_std)))

    def compute(self, inputs, role):
        states = inputs.get("observations", inputs.get("states"))
        actions = self.net(states)
        return actions, {"log_std": self.log_std_parameter}

    @torch.no_grad()
    def act_deterministic_direct(self, states: torch.Tensor) -> torch.Tensor:
        actions, _ = self.compute({"states": states}, role="policy")
        return torch.clamp(actions, -1.0, 1.0)


class Task4TeacherEvalFrameStackWrapper(gym.Env):
    def __init__(self, env: AllegroHandTask4Env, n_stack: int = 5):
        super().__init__()

        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = int(env.cfg.num_envs)
        self.device = env.device

        self.teacher_dim = int(env.observation_space["teacher_obs"].shape[0])
        self.priv_dim = int(env.observation_space["privileged_obs"].shape[0])
        self.blind_dim = int(env.observation_space["obs"].shape[0])
        self.history_dim = int(env.observation_space["history_obs"].shape[0])

        self.policy_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.teacher_dim * self.n_stack,),
            dtype=np.float32,
        )
        self.critic_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.priv_dim * self.n_stack,),
            dtype=np.float32,
        )

        self.observation_space = self.policy_space
        self.state_space = self.critic_space
        self.action_space = env.action_space

        self.policy_stack = torch.zeros(
            (self.num_envs, self.teacher_dim * self.n_stack),
            dtype=torch.float32,
            device=self.device,
        )
        self.critic_stack = torch.zeros(
            (self.num_envs, self.priv_dim * self.n_stack),
            dtype=torch.float32,
            device=self.device,
        )

        self.last_blind_obs = torch.zeros(
            (self.num_envs, self.blind_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_history_obs = torch.zeros(
            (self.num_envs, self.history_dim),
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)

        for i in range(self.n_stack):
            self.policy_stack[:, i * self.teacher_dim : (i + 1) * self.teacher_dim] = obs["teacher_obs"]
            self.critic_stack[:, i * self.priv_dim : (i + 1) * self.priv_dim] = obs["privileged_obs"]

        self.last_blind_obs = obs["obs"].clone()
        self.last_history_obs = obs["history_obs"].clone()

        return {"policy": self.policy_stack.clone(), "critic": self.critic_stack.clone()}, info or {}

    def step(self, actions: torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        self.policy_stack[:, :-self.teacher_dim] = self.policy_stack[:, self.teacher_dim :].clone()
        self.critic_stack[:, :-self.priv_dim] = self.critic_stack[:, self.priv_dim :].clone()

        self.policy_stack[:, -self.teacher_dim :] = obs["teacher_obs"]
        self.critic_stack[:, -self.priv_dim :] = obs["privileged_obs"]

        done = terminated | truncated
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            for i in range(self.n_stack):
                self.policy_stack[ids, i * self.teacher_dim : (i + 1) * self.teacher_dim] = obs["teacher_obs"][ids]
                self.critic_stack[ids, i * self.priv_dim : (i + 1) * self.priv_dim] = obs["privileged_obs"][ids]

        self.last_blind_obs = obs["obs"].clone()
        self.last_history_obs = obs["history_obs"].clone()

        return {"policy": self.policy_stack.clone(), "critic": self.critic_stack.clone()}, rewards, terminated, truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


def torch_load_checkpoint(path: Path, device: str):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def resolve_checkpoint(path: str) -> Path:
    p = Path(path).expanduser().resolve()

    if p.is_file():
        return p

    if p.is_dir():
        candidates = [
            p / "allegro_task4_teacher_model.pt",
            p / "final_checkpoint" / "allegro_task4_teacher_model.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        pt_files = sorted(p.glob("*.pt"))
        for pt in pt_files:
            if pt.name == "teacher_model.pt":
                continue
            if pt.name.endswith("_preprocessor.pt"):
                continue
            return pt

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


def load_policy_checkpoint(ckpt_path: Path, env: Task4TeacherEvalFrameStackWrapper):
    ckpt = torch_load_checkpoint(ckpt_path, env.device)

    if not isinstance(ckpt, dict) or "policy" not in ckpt:
        raise RuntimeError(
            f"当前测试脚本需要 task4_train.py 保存的 eval checkpoint: allegro_task4_teacher_model.pt\n"
            f"收到的文件不是 eval checkpoint: {ckpt_path}\n"
            "请传入 final_checkpoint 目录，或传入 final_checkpoint/allegro_task4_teacher_model.pt。"
        )

    metadata = ckpt.get("metadata", {})
    args = ckpt.get("args", {})

    if not bool(metadata.get("uses_skrl", False)):
        raise RuntimeError(
            f"Checkpoint is not marked as skrl PPO: {ckpt_path}\n"
            "请使用当前 TRUE skrl 版本 task4_train.py 重新训练。"
        )

    if not bool(metadata.get("teacher_policy", False)):
        raise RuntimeError(
            f"Checkpoint is not marked as Task4 teacher policy: {ckpt_path}\n"
            "Task4 当前阶段评估的是 Teacher PPO，不是 Student / Adapter。"
        )

    if not bool(metadata.get("rma_ready", False)):
        raise RuntimeError(
            f"Checkpoint is not marked as RMA-ready: {ckpt_path}\\n"
            "Task4 Teacher checkpoint 应保留 blind_obs / history_obs 信息，供后续 Student / Adapter 蒸馏使用。"
        )

    if not bool(metadata.get("asymmetric_actor_critic", False)):
        raise RuntimeError(
            f"Checkpoint is not marked as asymmetric actor-critic: {ckpt_path}\n"
            "Task4 Teacher 需要 teacher_obs actor / privileged_obs critic。"
        )

    expected_actor_dim = int(metadata.get("actor_obs_dim", env.observation_space.shape[0]))
    expected_critic_dim = int(metadata.get("critic_obs_dim", env.state_space.shape[0]))
    expected_action_dim = int(metadata.get("action_dim", env.action_space.shape[0]))

    if expected_actor_dim != env.observation_space.shape[0]:
        raise RuntimeError(
            f"actor obs dim mismatch: checkpoint={expected_actor_dim}, env={env.observation_space.shape[0]}"
        )

    if expected_critic_dim != env.state_space.shape[0]:
        raise RuntimeError(
            f"critic obs dim mismatch: checkpoint={expected_critic_dim}, env={env.state_space.shape[0]}"
        )

    if expected_action_dim != env.action_space.shape[0]:
        raise RuntimeError(
            f"action dim mismatch: checkpoint={expected_action_dim}, env={env.action_space.shape[0]}"
        )

    policy = Task4TeacherActor(
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
        init_log_std=float(args.get("init_log_std", 0.0)),
        min_log_std=float(args.get("min_log_std", -20.0)),
        max_log_std=float(args.get("max_log_std", 2.0)),
    ).to(env.device)

    policy.load_state_dict(ckpt["policy"], strict=True)
    policy.eval()

    actor_obs_norm = ckpt.get("actor_obs_norm", None)
    train_env_steps = int(ckpt.get("env_steps", 0))

    return policy, actor_obs_norm, train_env_steps, metadata


def force_eval_curriculum(env: AllegroHandTask4Env, start_k: float, label: str) -> None:
    k = max(0.0, min(1.0, float(start_k)))
    env.global_steps = int(k * env.cfg.curriculum_total_steps)

    ids = torch.arange(env.cfg.num_envs, dtype=torch.long, device=env.device)
    env.reset(ids)

    print(
        f"[CURRICULUM][{label}] forced start_k={k:.4f}, "
        f"global_steps={env.global_steps:,}, "
        f"DR_K={env._dr_k():.4f}, Reward_K={env._rw_k():.4f}",
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
    print("\n" + "=" * 188)
    print("Allegro Hand Task4 Teacher TRUE skrl PPO Model Test Summary")
    print("=" * 188)
    print(
        f"{'metric':<86} | {'mean':>12} | {'std':>12} | {'min':>12} | "
        f"{'p25':>12} | {'p50':>12} | {'p75':>12} | {'max':>12}"
    )
    print("-" * 188)

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<86} | "
            f"{row['mean']:>12.6f} | "
            f"{row['std']:>12.6f} | "
            f"{row['min']:>12.6f} | "
            f"{row['p25']:>12.6f} | "
            f"{row['p50']:>12.6f} | "
            f"{row['p75']:>12.6f} | "
            f"{row['max']:>12.6f}"
        )

    print("=" * 188 + "\n")


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    cfg = Task4Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(args_cli.device)
    cfg.seed = int(args_cli.seed)
    cfg.debug_print_names = False
    cfg.validate()

    base_env = AllegroHandTask4Env(cfg)
    force_eval_curriculum(base_env, args_cli.start_k, "after_env_creation")

    env = Task4TeacherEvalFrameStackWrapper(base_env, n_stack=int(args_cli.frame_stack))

    obs_dict, _ = env.reset(seed=int(args_cli.seed))
    force_eval_curriculum(base_env, args_cli.start_k, "after_rollout_reset")
    obs_dict, _ = env.reset(seed=int(args_cli.seed))

    ckpt_path = resolve_checkpoint(args_cli.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    policy, actor_obs_norm, trained_env_steps, metadata = load_policy_checkpoint(ckpt_path, env)

    print("\n" + "=" * 150)
    print("Allegro Hand Task4 Teacher TRUE skrl PPO model test started")
    print("=" * 150)
    print(f"checkpoint         : {ckpt_path}")
    print(f"trained_env_steps  : {trained_env_steps:,}")
    print(f"num_envs           : {base_env.num_envs}")
    print(f"steps              : {args_cli.steps}")
    print(f"start_k            : {args_cli.start_k}")
    print(f"DR_K               : {base_env._dr_k():.4f}")
    print(f"Reward_K           : {base_env._rw_k():.4f}")
    print(f"frame_stack        : {args_cli.frame_stack}")
    print(f"device             : {base_env.device}")
    print(f"visualize          : {bool(args_cli.visualize)}")
    print("algorithm          : skrl PPO")
    print("stage              : Task4 Teacher PPO")
    print("actor input        : teacher_obs x frame_stack")
    print("critic input       : privileged_obs x frame_stack")
    print("note               : deterministic direct policy forward; no agent.act")
    print("=" * 150 + "\n")

    records: List[Dict[str, float]] = []
    total_terminated = 0
    total_truncated = 0

    start_time = time.time()

    try:
        with tqdm(
            total=int(args_cli.steps),
            desc="Allegro Task4 Teacher skrl Model Test",
            dynamic_ncols=True,
            mininterval=0.5,
        ) as pbar:
            for step in range(int(args_cli.steps)):
                if step < 3:
                    print(f"[DEBUG][eval step {step}] before policy forward", flush=True)

                with torch.no_grad():
                    teacher_obs = obs_dict["policy"]
                    teacher_obs_n = normalize_with_saved_obs_norm(teacher_obs, actor_obs_norm)
                    actions = policy.act_deterministic_direct(teacher_obs_n)

                if step < 3:
                    print(f"[DEBUG][eval step {step}] after policy forward", flush=True)
                    print(f"[DEBUG][eval step {step}] before env.step", flush=True)

                obs_dict, rewards, terminated, truncated, info = env.step(actions)

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
                        "test/blind_obs_norm": float(torch.norm(env.last_blind_obs, dim=-1).mean().detach().cpu().item()),
                        "test/history_obs_norm": float(torch.norm(env.last_history_obs, dim=-1).mean().detach().cpu().item()),
                    }
                    row.update(flat)
                    records.append(row)

                    pbar.set_postfix(
                        {
                            "rew": f"{row['test/reward_mean']:+.3f}",
                            "K": f"{flat.get('telemetry/K', 0.0):.2f}",
                            "DR": f"{flat.get('telemetry/DR_K', 0.0):.2f}",
                            "SO3": f"{flat.get('telemetry/SO3_Error', 0.0):.3f}",
                            "ct": f"{flat.get('telemetry/Contact_Count', 0.0):.2f}",
                            "h": f"{flat.get('telemetry/Object_Height', 0.0):.3f}",
                            "drop": f"{flat.get('events/Drop', 0.0):.2f}",
                        }
                    )

                    if bool(args_cli.visualize):
                        sys.stdout.write(
                            f"\r🤖 K={flat.get('telemetry/K', 0.0):.2f} | "
                            f"DR={flat.get('telemetry/DR_K', 0.0):.2f} | "
                            f"RW={flat.get('telemetry/Reward_K', 0.0):.2f} | "
                            f"SO3={flat.get('telemetry/SO3_Error', 0.0):.4f} | "
                            f"Contact={flat.get('telemetry/Contact_Count', 0.0):.2f} | "
                            f"Height={flat.get('telemetry/Object_Height', 0.0):.4f} | "
                            f"Delay={flat.get('telemetry/ActionDelay', 0.0):.2f} | "
                            f"Eff={flat.get('telemetry/JointEfficiency', 0.0):.3f} | "
                            f"Hist={row['test/history_obs_norm']:.2f} | "
                            f"R={row['test/reward_mean']:+.3f} | "
                            f"Drop={flat.get('events/Drop', 0.0):.2f} | "
                            f"Success={flat.get('events/Success', 0.0):.2f}"
                        )
                        sys.stdout.flush()

                pbar.update(1)

                if bool(args_cli.visualize) and not simulation_app.is_running():
                    print("\n[INFO] Isaac Sim window closed.")
                    break

        elapsed = time.time() - start_time
        env_steps = int(args_cli.steps) * int(base_env.num_envs)
        fps = env_steps / max(elapsed, 1e-6)

        print("\n✅ Allegro Task4 Teacher TRUE skrl PPO model test rollout finished")
        print(f"  env steps        : {env_steps:,}")
        print(f"  fps              : {fps:,.2f}")
        print(f"  total terminated : {total_terminated:,}")
        print(f"  total truncated  : {total_truncated:,}")

        print_summary_table(summarize(records))

        print("Allegro Task4 Teacher model test checklist:")
        print("1. checkpoint metadata 必须标记 uses_skrl=True。")
        print("2. 当前测试的是 Teacher PPO，不是最终 Student 部署策略。")
        print("3. 默认 start_k=1.0，测试 Full DR / Full Reward 难度。")
        print("4. 测试脚本不调用 agent.act，避免模型测试卡在 0%。")
        print("5. smoke checkpoint 效果差是正常的，先看推理稳定性和无 NaN/Inf。")
        print("6. 正式效果重点看 SO3_Error、Object_Height、Contact_Count、Drop、Success、DR_K、Reward_K。")
        print("7. history_obs_norm 必须大于 0，说明后续 RMA Adapter / Student 所需历史输入正常。")

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
