from __future__ import annotations

import argparse
import dataclasses
import os
import logging
import sys
import math
import time
import traceback
from datetime import datetime
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logging.getLogger("isaaclab.assets.articulation").setLevel(logging.ERROR)
logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Allegro Hand Task1 pose tracking with TRUE skrl PPO")

parser.add_argument("--total-env-steps", type=int, default=1_000_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=20_000_000)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=0.0)

parser.add_argument("--dataset-path", type=str, default=os.environ.get("ALLEGRO_TASK1_DATASET", ""))
parser.add_argument("--resume", type=str, default="", help="Optional skrl checkpoint or final_checkpoint directory")

parser.add_argument("--rollouts", type=int, default=64)
parser.add_argument("--learning-epochs", type=int, default=5)
parser.add_argument("--mini-batches", type=int, default=8)

parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--min-lr", type=float, default=1e-5)
parser.add_argument("--max-lr", type=float, default=5e-4)

parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--clip-range", type=float, default=0.2)
parser.add_argument("--value-clip", type=float, default=0.2)
parser.add_argument("--entropy-coef", type=float, default=0.001)
parser.add_argument("--value-coef", type=float, default=2.0)
parser.add_argument("--grad-clip", type=float, default=1.0)

parser.add_argument("--target-kl", type=float, default=0.015)
parser.add_argument("--hard-kl-stop", type=float, default=0.08)

parser.add_argument("--init-log-std", type=float, default=-1.0)
parser.add_argument("--min-log-std", type=float, default=-4.0)
parser.add_argument("--max-log-std", type=float, default=0.5)

parser.add_argument("--frame-stack", type=int, default=5)
parser.add_argument("--log-root", type=str, default=str(PROJECT_ROOT / "logs" / "task1"))
parser.add_argument("--run-name", type=str, default="")
parser.add_argument("--summary-interval", type=int, default=10)
parser.add_argument("--tb-log-interval-steps", type=int, default=50)
parser.add_argument("--skrl-write-interval", type=int, default=1_000_000)
parser.add_argument("--skrl-checkpoint-interval", type=int, default=0)

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from allegro_rl.tasks.task1.task1_config import Task1Config
from allegro_rl.tasks.task1.task1_env import AllegroHandTask1Env

try:
    from skrl.agents.torch.ppo import PPO, PPO_CFG
except ImportError:
    from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import StepTrainer

try:
    from skrl.envs.wrappers.torch import wrap_env
except Exception:
    from skrl.envs.torch import wrap_env

try:
    from skrl.resources.preprocessors.torch import RunningStandardScaler
except Exception:
    RunningStandardScaler = None

try:
    from skrl.resources.schedulers.torch import KLAdaptiveLR
except Exception:
    KLAdaptiveLR = None


def set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))


class AllegroSkrlFrameStackWrapper(gym.Env):
    """Vectorized Gymnasium-style frame stack wrapper for skrl.

    The wrapped Allegro env already runs many IsaacLab environments in parallel.
    This wrapper only stacks observations and keeps the interface compatible
    with skrl's gymnasium wrapper.
    """

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
        self.state_space = self.observation_space

        self.obs_stack = torch.zeros(
            (self.num_envs, self.stacked_obs_dim),
            dtype=torch.float32,
            device=self.device,
        )

        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)

        for i in range(self.n_stack):
            self.obs_stack[:, i * self.single_obs_dim : (i + 1) * self.single_obs_dim] = obs

        self.last_info = info or {}
        return self.obs_stack.clone(), self.last_info

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

        self.last_info = info or {}
        self.last_reward_mean = float(rewards.detach().float().mean().cpu().item())
        self.last_done_count = int(done.sum().detach().cpu().item())

        return self.obs_stack.clone(), rewards, terminated, truncated, self.last_info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


class AllegroPolicy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        init_log_std: float = -1.0,
        min_log_std: float = -4.0,
        max_log_std: float = 0.5,
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
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.constant_(module.bias, 0.0)

    def compute(self, inputs, role):
        states = inputs["states"]
        mean_actions = self.net(states)
        return mean_actions, self.log_std_parameter, {}


class AllegroValue(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)

        self.net = nn.Sequential(
            nn.Linear(state_space.shape[0], 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.constant_(module.bias, 0.0)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def make_run_name() -> str:
    run_name = args_cli.run_name.strip()
    if run_name:
        return run_name
    return f"allegro_task1_skrl_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def resolve_resume_checkpoint(path: str) -> str:
    if not path:
        return ""

    p = Path(path).expanduser().resolve()

    if p.is_file():
        return str(p)

    if p.is_dir():
        candidates = [
            p / "agent.pt",
            p / "allegro_task1_skrl_agent.pt",
            p / "final_checkpoint" / "agent.pt",
            p / "final_checkpoint" / "allegro_task1_skrl_agent.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    return str(p)


def _find_norm_tensors_from_state_dict(state_dict: Dict[str, Any], obs_dim: int):
    mean = None
    var = None

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        if value.numel() != obs_dim:
            continue

        key_l = key.lower()
        if "mean" in key_l:
            mean = value.detach().cpu()
        if "var" in key_l or "variance" in key_l:
            var = value.detach().cpu()

    if mean is not None and var is not None:
        return {"mean": mean, "var": var, "clip": 10.0}

    return None


def extract_obs_norm(agent: PPO, obs_dim: int):
    """Extract explicit mean/var tensors from skrl preprocessors if present."""

    for attr_name in [
        "_state_preprocessor",
        "_observation_preprocessor",
        "state_preprocessor",
        "observation_preprocessor",
    ]:
        obj = getattr(agent, attr_name, None)
        if obj is None:
            continue

        try:
            state = obj.state_dict()
        except Exception:
            continue

        out = _find_norm_tensors_from_state_dict(state, obs_dim)
        if out is not None:
            out["source_attr"] = attr_name
            return out

    return None


def save_project_checkpoint(
    directory: str,
    agent: PPO,
    models: Dict[str, Model],
    env_cfg: Task1Config,
    env: AllegroSkrlFrameStackWrapper,
    env_steps: int,
    args,
) -> None:
    os.makedirs(directory, exist_ok=True)

    skrl_agent_path = os.path.join(directory, "allegro_task1_skrl_agent.pt")
    eval_model_path = os.path.join(directory, "allegro_task1_model.pt")

    try:
        agent.save(skrl_agent_path)
    except Exception as exc:
        print(f"[WARN] agent.save failed: {type(exc).__name__}: {exc}")

    obs_norm = extract_obs_norm(agent, env.stacked_obs_dim)

    torch.save(
        {
            "policy": models["policy"].state_dict(),
            "value": models["value"].state_dict(),
            "obs_norm": obs_norm,
            "env_steps": int(env_steps),
            "args": vars(args),
            "metadata": {
                "robot": "Allegro Hand",
                "task": "task1_pose_tracking",
                "algorithm": "skrl_PPO",
                "uses_skrl": True,
                "single_obs_dim": int(env_cfg.num_observations),
                "frame_stack": int(args.frame_stack),
                "obs_dim": int(env_cfg.num_observations * int(args.frame_stack)),
                "action_dim": int(env_cfg.num_actions),
                "note": "TRUE skrl PPO checkpoint. Evaluation uses deterministic policy forward, not agent.act.",
            },
        },
        eval_model_path,
    )

    print(f"💾 [Allegro Task1 skrl checkpoint] saved to: {directory}", flush=True)




def _base_ppo_cfg_dict():
    cfg = PPO_CFG()
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return cfg.copy()


def _set_if_supported(cfg: dict, requested: dict) -> None:
    skipped = []
    for key, value in requested.items():
        if key in cfg:
            cfg[key] = value
        else:
            skipped.append(key)

    if skipped:
        print(f"[WARN] 当前 skrl.PPO_CFG 不支持这些字段，已跳过: {skipped}")


def build_skrl_cfg(env, log_dir: str, run_name: str):
    """Build skrl PPO config using the same pattern as Go2/G1.

    Critical rule:
        Start from PPO_CFG(), convert to dict with dataclasses.asdict,
        then only set supported keys. Do not manually invent lambda/lambda_.
    """
    cfg = _base_ppo_cfg_dict()

    requested = {
        "rollouts": int(args_cli.rollouts),
        "learning_epochs": int(args_cli.learning_epochs),
        "mini_batches": int(args_cli.mini_batches),
        "discount_factor": float(args_cli.gamma),
        "gae_lambda": float(args_cli.gae_lambda),
        "learning_rate": float(args_cli.lr),
        "learning_rate_scheduler": KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": {
            "kl_threshold": float(args_cli.target_kl),
            "min_lr": float(args_cli.min_lr),
            "max_lr": float(args_cli.max_lr),
        },
        "observation_preprocessor": RunningStandardScaler,
        "observation_preprocessor_kwargs": {
            "size": env.observation_space,
            "device": env.device,
        },
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {
            "size": env.state_space,
            "device": env.device,
        },
        "value_preprocessor": RunningStandardScaler,
        "value_preprocessor_kwargs": {
            "size": 1,
            "device": env.device,
        },
        "grad_norm_clip": float(args_cli.grad_clip),
        "ratio_clip": float(args_cli.clip_range),
        "value_clip": float(args_cli.value_clip),
        "entropy_loss_scale": float(args_cli.entropy_coef),
        "value_loss_scale": float(args_cli.value_coef),
    }

    _set_if_supported(cfg, requested)

    cfg.setdefault("experiment", {})
    cfg["experiment"].update(
        {
            "directory": log_dir,
            "experiment_name": run_name,
            "write_interval": int(getattr(args_cli, "skrl_write_interval", 1_000_000)),
            "checkpoint_interval": int(getattr(args_cli, "skrl_checkpoint_interval", 0)),
            "store_separately": True,
            "wandb": False,
        }
    )

    return cfg



def main() -> None:
    set_seed(int(args_cli.seed))

    run_name = make_run_name()
    log_dir = os.path.abspath(args_cli.log_root)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 118)
    print("🚀 Allegro Hand Task1 Pose Tracking - TRUE skrl PPO Training")
    print("=" * 118)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_root     = {log_dir}")
    print(f"[INFO] run_name     = {run_name}")
    print("[INFO] This version uses skrl PPO, not a custom PPO loop.")

    env_cfg = Task1Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.device)
    env_cfg.seed = int(args_cli.seed)

    if args_cli.dataset_path:
        env_cfg.dataset_path = str(Path(args_cli.dataset_path).expanduser().resolve())

    env_cfg.validate()

    if not os.path.exists(env_cfg.dataset_path):
        raise FileNotFoundError(
            f"Task1 dataset not found: {env_cfg.dataset_path}\n"
            "Please run: bash scripts/ubuntu/generate_task1_dataset.sh"
        )

    base_env = AllegroHandTask1Env(env_cfg)

    if float(args_cli.start_k) > 0.0:
        base_env.global_steps = int(float(args_cli.start_k) * env_cfg.curriculum_total_steps)
        print(f"[INFO] start_k={args_cli.start_k:.4f}, global_steps={base_env.global_steps:,}")

    local_env = AllegroSkrlFrameStackWrapper(base_env, n_stack=int(args_cli.frame_stack))

    global env_observation_space
    env_observation_space = local_env.observation_space

    env = wrap_env(local_env, wrapper="isaaclab")
    num_envs = getattr(env, "num_envs", local_env.num_envs)

    device = env.device
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    print("\n[DEBUG] Allegro Task1 Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    models = {
        "policy": AllegroPolicy(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            init_log_std=float(args_cli.init_log_std),
            min_log_std=float(args_cli.min_log_std),
            max_log_std=float(args_cli.max_log_std),
        ),
        "value": AllegroValue(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
        ),
    }

    total_env_steps = int(args_cli.total_env_steps)
    total_vector_steps = math.ceil(total_env_steps / num_envs)
    save_freq_env_steps = int(args_cli.save_freq_env_steps)

    cfg = build_skrl_cfg(env, log_dir=log_dir, run_name=run_name)
    update_env_steps = int(cfg["rollouts"]) * int(num_envs)

    memory = RandomMemory(
        memory_size=int(cfg["rollouts"]),
        num_envs=num_envs,
        device=env.device,
    )

    cfg = build_skrl_cfg(env, log_dir=log_dir, run_name=run_name)

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=env.device,
    )

    resume_ckpt = resolve_resume_checkpoint(args_cli.resume)
    if resume_ckpt:
        if os.path.exists(resume_ckpt):
            print(f"[INFO] Loading skrl agent checkpoint: {resume_ckpt}")
            agent.load(resume_ckpt)
        else:
            print(f"[WARN] resume checkpoint not found: {resume_ckpt}")

    total_vector_steps = math.ceil(int(args_cli.total_env_steps) / int(num_envs))

    trainer = StepTrainer(
        cfg={
            "timesteps": int(total_vector_steps),
            "headless": True,
            "disable_progressbar": True,
        },
        env=env,
        agents=agent,
    )

    print("\n[INFO] skrl PPO configuration")
    print(f"  - num_envs            : {local_env.num_envs:,}")
    print(f"  - total_env_steps     : {args_cli.total_env_steps:,}")
    print(f"  - total_vector_steps  : {total_vector_steps:,}")
    print(f"  - single_obs_dim      : {local_env.single_obs_dim}")
    print(f"  - frame_stack         : {local_env.n_stack}")
    print(f"  - obs_dim             : {obs_dim}")
    print(f"  - action_dim          : {action_dim}")
    print(f"  - rollouts            : {args_cli.rollouts}")
    print(f"  - learning_epochs     : {args_cli.learning_epochs}")
    print(f"  - mini_batches        : {args_cli.mini_batches}")
    print(f"  - lr/min/max          : {args_cli.lr} / {args_cli.min_lr} / {args_cli.max_lr}")
    print(f"  - checkpoint_interval : {cfg.get('experiment', {}).get('checkpoint_interval', 0)} vector steps")
    print(f"  - tensorboard         : tensorboard --logdir={log_dir}")
    print("\n🔥 [Allegro Task1 TRUE skrl PPO 已点火]\n")

    last_save = 0
    update_id = 0
    start_time = time.time()

    try:
        trainer.reset()

        with tqdm(
            total=int(args_cli.total_env_steps),
            desc="Allegro Task1 skrl PPO",
            unit="steps",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
        ) as pbar:
            for t in range(total_vector_steps):
                trainer.train(timestep=t, timesteps=total_vector_steps)

                env_steps = min((t + 1) * num_envs, int(args_cli.total_env_steps))
                previous_env_steps = min(t * num_envs, int(args_cli.total_env_steps))
                pbar.update(env_steps - previous_env_steps)

                flat = flat_dict(local_env.last_info)
                elapsed = time.time() - start_time
                fps = env_steps / max(elapsed, 1e-6)

                pbar.set_postfix(
                    {
                        "steps": f"{env_steps:,}",
                        "fps": f"{fps:,.0f}",
                        "rew": f"{local_env.last_reward_mean:+.3f}",
                        "err": f"{flat.get('telemetry/Pose_Error_Rad', 0.0):.3f}",
                        "k": f"{flat.get('telemetry/Curriculum_Progress_K', 0.0):.2f}",
                    }
                )

                if (t + 1) % int(cfg["rollouts"]) == 0:
                    update_id += 1

                    ppo_info = {}
                    for key, value in getattr(agent, "tracking_data", {}).items():
                        try:
                            arr = np.asarray(value, dtype=np.float64)
                            ppo_info[key] = float(np.mean(arr))
                        except Exception:
                            pass

                    lr = None
                    try:
                        lr = float(agent.optimizer.param_groups[0]["lr"])
                    except Exception:
                        lr = float(args_cli.lr)

                    ppo_info["learning_rate"] = lr

                    if update_id % max(int(args_cli.summary_interval), 1) == 0:
                        stat = {
                            "update": float(update_id),
                            "total_env_steps": float(env_steps),
                            "target_env_steps": float(args_cli.total_env_steps),
                            "progress_percent": 100.0 * env_steps / max(int(args_cli.total_env_steps), 1),
                            "num_envs": float(num_envs),
                            "rollouts_per_update": float(cfg["rollouts"]),
                            "fps_env_steps": fps,
                            "learning_rate": lr,
                        }

                        pbar.write(
                            "\n".join(
                                [
                                    "\n" + "=" * 112,
                                    f"📊 [Allegro Task1 skrl PPO 更新 {update_id}] 总步数: {env_steps:,} / {int(args_cli.total_env_steps):,} | "
                                    f"环境 FPS: {fps:,.0f} | LR: {lr:.3e}",
                                    "=" * 112,
                                    make_table("time / progress", stat),
                                    make_table("env info: reward_components + events + telemetry + debug", flat),
                                    make_table("ppo update info", ppo_info),
                                    "=" * 112 + "\n",
                                ]
                            )
                        )

                    try:
                        agent.tracking_data.clear()
                    except Exception:
                        pass

                if env_steps - last_save >= int(args_cli.save_freq_env_steps):
                    last_save = env_steps
                    save_dir = os.path.join(log_dir, run_name, f"checkpoint_{env_steps}")
                    os.makedirs(save_dir, exist_ok=True)
                    try:
                        save_project_checkpoint(
                            save_dir,
                            agent=agent,
                            models=models,
                            env_cfg=env_cfg,
                            env=local_env,
                            env_steps=env_steps,
                            args=args_cli,
                        )
                        pbar.write(f"\n💾 [Allegro Task1 skrl 备份] 总步数: {env_steps:,} | 已保存至: {save_dir}\n")
                    except Exception as exc:
                        pbar.write(f"\n[WARN] checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

    except KeyboardInterrupt:
        print("\n[WARN] 接收到 Ctrl+C，正在保存当前 skrl 模型...")
    except Exception:
        print("\n[ERROR] Allegro Task1 skrl PPO 训练过程中发生真实异常：")
        traceback.print_exc()
        raise
    finally:
        elapsed = time.time() - start_time
        approx_env_steps = int(args_cli.total_env_steps)

        final_dir = os.path.join(log_dir, run_name, "final_checkpoint")
        try:
            save_project_checkpoint(
                final_dir,
                agent=agent,
                models=models,
                env_cfg=env_cfg,
                env=local_env,
                env_steps=approx_env_steps,
                args=args_cli,
            )
            print(f"✅ Allegro Task1 skrl 模型已保存至 {final_dir}")
        except Exception as exc:
            print(f"[WARN] 保存最终 skrl 模型失败: {type(exc).__name__}: {exc}")

        print(f"[INFO] elapsed = {elapsed:.2f}s")
        print(f"[INFO] approx_env_steps = {approx_env_steps:,}")

        try:
            env.close()
        except Exception:
            try:
                local_env.close()
            except Exception:
                pass

        try:
            simulation_app.close()
        except Exception:
            pass

        print("✅ Allegro Task1 TRUE skrl PPO training pipeline safely exited")


if __name__ == "__main__":
    main()
