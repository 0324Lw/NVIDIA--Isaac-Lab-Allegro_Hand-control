from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.getLogger("isaaclab.assets.articulation").setLevel(logging.ERROR)
logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Allegro Hand Task3 dynamic grasping/tool-use with TRUE skrl PPO")

parser.add_argument("--total-env-steps", type=int, default=1_000_000_000)
parser.add_argument("--save-freq-env-steps", type=int, default=20_000_000)
parser.add_argument("--num-envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--start-k", type=float, default=0.0)
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
parser.add_argument("--entropy-coef", type=float, default=0.002)
parser.add_argument("--value-coef", type=float, default=2.0)
parser.add_argument("--grad-clip", type=float, default=0.8)

parser.add_argument("--target-kl", type=float, default=0.015)
parser.add_argument("--hard-kl-stop", type=float, default=0.08)

parser.add_argument("--init-log-std", type=float, default=0.0)
parser.add_argument("--min-log-std", type=float, default=-20.0)
parser.add_argument("--max-log-std", type=float, default=2.0)

parser.add_argument("--frame-stack", type=int, default=5)
parser.add_argument("--log-root", type=str, default=str(PROJECT_ROOT / "logs" / "task3"))
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

from allegro_rl.common.info_utils import flat_dict, make_table, to_float
from allegro_rl.tasks.task3.task3_config import Task3Config
from allegro_rl.tasks.task3.task3_env import AllegroHandTask3Env

from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import StepTrainer
from skrl.utils import set_seed

try:
    from skrl.agents.torch.ppo import PPO, PPO_CFG
except ImportError:
    from skrl.agents.torch.ppo import PPO
    from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG

try:
    from skrl.resources.schedulers.torch import KLAdaptiveLR
except Exception:
    KLAdaptiveLR = None


# ======================================================================
# Logging helpers
# ======================================================================

def write_scalars(writer, data: Dict[str, Any], step: int, prefix: str) -> None:
    if writer is None:
        return

    for key, value in (data or {}).items():
        val = to_float(value)
        if val is None:
            continue
        try:
            writer.add_scalar(f"{prefix}/{key}".replace("//", "/"), val, step)
        except Exception:
            pass


def tracking_mean(agent) -> Dict[str, float]:
    out: Dict[str, float] = {}

    for key, value in getattr(agent, "tracking_data", {}).items():
        if value is None:
            continue
        try:
            if len(value) == 0:
                continue
        except Exception:
            pass

        try:
            arr = np.asarray(value, dtype=np.float64)
            if key.endswith("(min)"):
                out[key] = float(np.min(arr))
            elif key.endswith("(max)"):
                out[key] = float(np.max(arr))
            else:
                out[key] = float(np.mean(arr))
        except Exception:
            val = to_float(value)
            if val is not None:
                out[key] = val

    return out


def current_lr(agent) -> float:
    for obj in [getattr(agent, "optimizer", None), getattr(getattr(agent, "scheduler", None), "optimizer", None)]:
        try:
            if obj is not None:
                return float(obj.param_groups[0]["lr"])
        except Exception:
            pass
    return float("nan")


def make_run_name() -> str:
    run_name = args_cli.run_name.strip()
    if run_name:
        return run_name
    return f"allegro_task3_skrl_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ======================================================================
# Dict observation frame stack wrapper
# ======================================================================

class Task3DictFrameStackAndLogWrapper(gym.Env):
    """Task3 asymmetric observation wrapper.

    Actor:
        stacked actor obs = 147 * frame_stack

    Critic:
        stacked privileged obs = 168 * frame_stack
    """

    def __init__(self, env: AllegroHandTask3Env, log_dir: str, n_stack: int = 5):
        super().__init__()

        self.env = env
        self.n_stack = int(n_stack)
        self.num_envs = int(env.cfg.num_envs)
        self.device = env.device

        self.obs_dim = int(env.observation_space["obs"].shape[0])
        self.priv_dim = int(env.observation_space["privileged_obs"].shape[0])

        self.policy_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim * self.n_stack,),
            dtype=np.float32,
        )
        self.critic_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.priv_dim * self.n_stack,),
            dtype=np.float32,
        )

        self.single_observation_space = gym.spaces.Dict(
            {
                "policy": self.policy_space,
                "critic": self.critic_space,
            }
        )

        self.observation_space = self.policy_space
        self.state_space = self.critic_space
        self.action_space = env.action_space
        self.single_action_space = env.action_space

        self.obs_stack = torch.zeros(
            (self.num_envs, self.obs_dim * self.n_stack),
            dtype=torch.float32,
            device=self.device,
        )
        self.priv_stack = torch.zeros(
            (self.num_envs, self.priv_dim * self.n_stack),
            dtype=torch.float32,
            device=self.device,
        )

        self.writer = SummaryWriter(log_dir)
        self.global_env_steps = 0
        self.last_info: Dict[str, Any] = {}
        self.last_reward_mean = 0.0
        self.last_done_count = 0

    @property
    def unwrapped(self):
        return self

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, **kwargs):
        obs, info = self.env.reset(seed=seed, options=options)

        for i in range(self.n_stack):
            self.obs_stack[:, i * self.obs_dim : (i + 1) * self.obs_dim] = obs["obs"]
            self.priv_stack[:, i * self.priv_dim : (i + 1) * self.priv_dim] = obs["privileged_obs"]

        self.last_info = info or {}

        return {"policy": self.obs_stack.clone(), "critic": self.priv_stack.clone()}, self.last_info

    def step(self, actions: torch.Tensor):
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        self.obs_stack[:, :-self.obs_dim] = self.obs_stack[:, self.obs_dim :].clone()
        self.priv_stack[:, :-self.priv_dim] = self.priv_stack[:, self.priv_dim :].clone()

        self.obs_stack[:, -self.obs_dim :] = obs["obs"]
        self.priv_stack[:, -self.priv_dim :] = obs["privileged_obs"]

        done = terminated | truncated
        if done.any():
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            for i in range(self.n_stack):
                self.obs_stack[ids, i * self.obs_dim : (i + 1) * self.obs_dim] = obs["obs"][ids]
                self.priv_stack[ids, i * self.priv_dim : (i + 1) * self.priv_dim] = obs["privileged_obs"][ids]

        self.global_env_steps += self.num_envs
        self.last_info = info or {}
        self.last_reward_mean = to_float(rewards) or 0.0
        self.last_done_count = int(done.sum().detach().cpu().item())

        write_scalars(self.writer, self.last_info.get("reward_components", {}), self.global_env_steps, "rewards")
        write_scalars(self.writer, self.last_info.get("events", {}), self.global_env_steps, "events")
        write_scalars(self.writer, self.last_info.get("telemetry", {}), self.global_env_steps, "telemetry")
        write_scalars(self.writer, self.last_info.get("debug", {}), self.global_env_steps, "debug")

        try:
            self.writer.add_scalar("rollout/reward_mean_raw", self.last_reward_mean, self.global_env_steps)
            self.writer.add_scalar("rollout/done_count", self.last_done_count, self.global_env_steps)
        except Exception:
            pass

        return {"policy": self.obs_stack.clone(), "critic": self.priv_stack.clone()}, rewards, terminated, truncated, self.last_info

    def close(self):
        try:
            self.writer.flush()
            self.writer.close()
        except Exception:
            pass
        try:
            self.env.close()
        except Exception:
            pass


# ======================================================================
# Asymmetric Actor-Critic
# ======================================================================

class AsymmetricActor(GaussianMixin, Model):
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
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_space.shape[0]),
        )

        self.log_std_parameter = nn.Parameter(torch.full((action_space.shape[0],), float(init_log_std)))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)

    def compute(self, inputs, role):
        states = inputs.get("observations", inputs.get("states"))
        actions = self.net(states)
        return actions, {"log_std": self.log_std_parameter}


class AsymmetricCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False)

        if state_space is None:
            raise RuntimeError("env.state_space is None. Task3 critic requires privileged state space.")

        self.net = nn.Sequential(
            nn.Linear(state_space.shape[0], 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.apply(AsymmetricActor._init_weights)

    def compute(self, inputs, role):
        states = inputs.get("states", None)
        if states is None:
            raise RuntimeError("Critic received no privileged states. Check state_space / wrapper critic output.")
        return self.net(states), {}


# ======================================================================
# skrl config / checkpoint helpers
# ======================================================================

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
    cfg = _base_ppo_cfg_dict()

    requested = {
        "rollouts": int(args_cli.rollouts),
        "learning_epochs": int(args_cli.learning_epochs),
        "mini_batches": int(args_cli.mini_batches),

        "discount_factor": float(args_cli.gamma),
        "gae_lambda": float(args_cli.gae_lambda),

        "learning_rate": float(args_cli.lr),
        "grad_norm_clip": float(args_cli.grad_clip),

        "ratio_clip": float(args_cli.clip_range),
        "value_clip": float(args_cli.value_clip),
        "entropy_loss_scale": float(args_cli.entropy_coef),
        "value_loss_scale": float(args_cli.value_coef),
        "kl_threshold": float(args_cli.hard_kl_stop),

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
    }

    if KLAdaptiveLR is not None:
        requested["learning_rate_scheduler"] = KLAdaptiveLR
        requested["learning_rate_scheduler_kwargs"] = {
            "kl_threshold": float(args_cli.target_kl),
            "min_lr": float(args_cli.min_lr),
            "max_lr": float(args_cli.max_lr),
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


def resolve_resume_checkpoint(path: str) -> str:
    if not path:
        return ""

    p = Path(path).expanduser().resolve()

    if p.is_file():
        return str(p)

    if p.is_dir():
        candidates = [
            p / "allegro_task3_skrl_agent.pt",
            p / "agent.pt",
            p / "final_checkpoint" / "allegro_task3_skrl_agent.pt",
            p / "final_checkpoint" / "agent.pt",
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

        lower = key.lower()
        if "mean" in lower:
            mean = value.detach().cpu()
        if "var" in lower or "variance" in lower:
            var = value.detach().cpu()

    if mean is not None and var is not None:
        return {"mean": mean, "var": var, "clip": 10.0}

    return None


def extract_norm(agent, attr_names, dim: int):
    for attr_name in attr_names:
        obj = getattr(agent, attr_name, None)
        if obj is None:
            continue

        try:
            state = obj.state_dict()
        except Exception:
            continue

        out = _find_norm_tensors_from_state_dict(state, dim)
        if out is not None:
            out["source_attr"] = attr_name
            return out

    return None


def save_project_checkpoint(
    directory: str,
    agent: PPO,
    models: Dict[str, Model],
    env_cfg: Task3Config,
    env: Task3DictFrameStackAndLogWrapper,
    env_steps: int,
    args,
) -> None:
    os.makedirs(directory, exist_ok=True)

    skrl_agent_path = os.path.join(directory, "allegro_task3_skrl_agent.pt")
    eval_model_path = os.path.join(directory, "allegro_task3_model.pt")

    try:
        agent.save(skrl_agent_path)
    except Exception as exc:
        print(f"[WARN] agent.save failed: {type(exc).__name__}: {exc}")

    actor_norm = extract_norm(
        agent,
        ["_observation_preprocessor", "_state_preprocessor", "observation_preprocessor", "state_preprocessor"],
        env.obs_dim * env.n_stack,
    )
    critic_norm = extract_norm(
        agent,
        ["_state_preprocessor", "state_preprocessor"],
        env.priv_dim * env.n_stack,
    )

    torch.save(
        {
            "policy": models["policy"].state_dict(),
            "value": models["value"].state_dict(),
            "actor_obs_norm": actor_norm,
            "critic_obs_norm": critic_norm,
            "env_steps": int(env_steps),
            "args": vars(args),
            "metadata": {
                "robot": "Allegro Hand",
                "task": "task3_dynamic_grasp_tool_use",
                "algorithm": "skrl_PPO",
                "uses_skrl": True,
                "asymmetric_actor_critic": True,
                "actor_single_obs_dim": int(env_cfg.num_observations),
                "critic_single_obs_dim": int(env_cfg.num_privileged_obs),
                "frame_stack": int(args.frame_stack),
                "actor_obs_dim": int(env_cfg.num_observations * int(args.frame_stack)),
                "critic_obs_dim": int(env_cfg.num_privileged_obs * int(args.frame_stack)),
                "action_dim": int(env_cfg.num_actions),
                "note": "TRUE skrl PPO checkpoint. Evaluation uses deterministic policy forward, not agent.act.",
            },
        },
        eval_model_path,
    )

    print(f"💾 [Allegro Task3 skrl checkpoint] saved to: {directory}", flush=True)


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    set_seed(int(args_cli.seed))

    run_name = make_run_name()
    log_dir = os.path.abspath(args_cli.log_root)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 118)
    print("🚀 Allegro Hand Task3 Dynamic Grasp / Tool Use - TRUE skrl PPO Training")
    print("=" * 118)
    print(f"[INFO] PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"[INFO] log_root     = {log_dir}")
    print(f"[INFO] run_name     = {run_name}")
    print("[INFO] This version uses skrl PPO with asymmetric actor-critic.")

    env_cfg = Task3Config()
    env_cfg.num_envs = int(args_cli.num_envs)
    env_cfg.device = str(args_cli.device)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.debug_print_names = False
    env_cfg.validate()

    base_env = AllegroHandTask3Env(env_cfg)

    if float(args_cli.start_k) > 0.0:
        base_env.global_steps = int(float(args_cli.start_k) * env_cfg.curriculum_total_steps)
        print(f"[INFO] start_k={args_cli.start_k:.4f}, global_steps={base_env.global_steps:,}")

    run_dir = os.path.join(log_dir, run_name)
    local_env = Task3DictFrameStackAndLogWrapper(base_env, log_dir=run_dir, n_stack=int(args_cli.frame_stack))

    env = wrap_env(local_env, wrapper="isaaclab")
    num_envs = getattr(env, "num_envs", local_env.num_envs)

    if env.state_space is None:
        raise RuntimeError("env.state_space is None. Task3 requires privileged critic state space.")

    print("\n[DEBUG] Allegro Task3 Spaces")
    print(f"  env.observation_space = {env.observation_space}")
    print(f"  env.state_space       = {env.state_space}")
    print(f"  env.action_space      = {env.action_space}")
    print(f"  policy input dim      = {env.observation_space.shape[0]}")
    print(f"  critic input dim      = {env.state_space.shape[0]}")
    print(f"  action dim            = {env.action_space.shape[0]}")

    models = {
        "policy": AsymmetricActor(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
            init_log_std=float(args_cli.init_log_std),
            min_log_std=float(args_cli.min_log_std),
            max_log_std=float(args_cli.max_log_std),
        ),
        "value": AsymmetricCritic(
            env.observation_space,
            env.state_space,
            env.action_space,
            env.device,
        ),
    }

    cfg = build_skrl_cfg(env, log_dir=log_dir, run_name=run_name)

    memory = RandomMemory(
        memory_size=int(cfg["rollouts"]),
        num_envs=num_envs,
        device=env.device,
    )

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

    total_env_steps = int(args_cli.total_env_steps)
    total_vector_steps = math.ceil(total_env_steps / int(num_envs))
    update_env_steps = int(cfg["rollouts"]) * int(num_envs)

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
    print(f"  - num_envs            : {num_envs:,}")
    print(f"  - total_env_steps     : {total_env_steps:,}")
    print(f"  - total_vector_steps  : {total_vector_steps:,}")
    print(f"  - update_env_steps    : {update_env_steps:,}")
    print(f"  - actor_single_obs    : {local_env.obs_dim}")
    print(f"  - critic_single_obs   : {local_env.priv_dim}")
    print(f"  - frame_stack         : {local_env.n_stack}")
    print(f"  - actor_obs_dim       : {env.observation_space.shape[0]}")
    print(f"  - critic_obs_dim      : {env.state_space.shape[0]}")
    print(f"  - action_dim          : {env.action_space.shape[0]}")
    print(f"  - rollouts            : {cfg['rollouts']}")
    print(f"  - learning_epochs     : {cfg.get('learning_epochs')}")
    print(f"  - mini_batches        : {cfg.get('mini_batches')}")
    print(f"  - lr                  : {cfg.get('learning_rate')}")
    print(f"  - checkpoint_interval : {cfg.get('experiment', {}).get('checkpoint_interval', 0)} vector steps")
    print(f"  - tensorboard         : tensorboard --logdir={log_dir}")
    print("\n🔥 [Allegro Task3 TRUE skrl PPO 已点火]\n")

    last_save = 0
    update_id = 0
    start_time = time.time()

    try:
        trainer.reset()

        with tqdm(
            total=total_env_steps,
            desc="Allegro Task3 skrl PPO",
            unit="steps",
            dynamic_ncols=True,
            mininterval=0.5,
            smoothing=0.05,
        ) as pbar:
            for t in range(total_vector_steps):
                trainer.train(timestep=t, timesteps=total_vector_steps)

                env_steps = min((t + 1) * int(num_envs), total_env_steps)
                previous_env_steps = min(t * int(num_envs), total_env_steps)
                pbar.update(env_steps - previous_env_steps)

                flat = flat_dict(local_env.last_info)
                elapsed = time.time() - start_time
                fps = env_steps / max(elapsed, 1e-6)

                pbar.set_postfix(
                    {
                        "steps": f"{env_steps:,}",
                        "fps": f"{fps:,.0f}",
                        "rew": f"{local_env.last_reward_mean:+.3f}",
                        "done": local_env.last_done_count,
                        "phase": f"{flat.get('telemetry/Phase', 0.0):.1f}",
                        "contact": f"{flat.get('telemetry/Contact_Count', 0.0):.2f}",
                        "lift": f"{flat.get('telemetry/Lift', 0.0):.3f}",
                        "tcp": f"{flat.get('telemetry/TCP_Dist', 0.0):.3f}",
                        "drop": f"{flat.get('events/Drop', 0.0):.2f}",
                    }
                )

                if (t + 1) % int(cfg["rollouts"]) == 0:
                    update_id += 1

                    ppo_info = tracking_mean(agent)
                    lr = current_lr(agent)
                    ppo_info["learning_rate"] = lr

                    writer = getattr(agent, "writer", None)
                    write_scalars(writer, ppo_info, env_steps, "ppo")
                    write_scalars(writer, flat, env_steps, "env_info")

                    if update_id % max(int(args_cli.summary_interval), 1) == 0:
                        stat = {
                            "update": float(update_id),
                            "total_env_steps": float(env_steps),
                            "target_env_steps": float(total_env_steps),
                            "progress_percent": 100.0 * env_steps / max(total_env_steps, 1),
                            "num_envs": float(num_envs),
                            "rollouts_per_update": float(cfg["rollouts"]),
                            "fps_env_steps": float(fps),
                            "learning_rate": float(lr),
                        }

                        pbar.write(
                            "\n".join(
                                [
                                    "\n" + "=" * 118,
                                    f"📊 [Allegro Task3 skrl PPO 更新 {update_id}] "
                                    f"总步数: {env_steps:,} / {total_env_steps:,} | "
                                    f"环境 FPS: {fps:,.0f} | LR: {lr:.3e}",
                                    "=" * 118,
                                    make_table("time / progress", stat),
                                    make_table("env info: reward_components + events + telemetry + debug", flat),
                                    make_table("ppo update info", ppo_info),
                                    "=" * 118 + "\n",
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
                        pbar.write(f"\n💾 [Allegro Task3 skrl 备份] 总步数: {env_steps:,} | 已保存至: {save_dir}\n")
                    except Exception as exc:
                        pbar.write(f"\n[WARN] checkpoint 保存失败: {type(exc).__name__}: {exc}\n")

    except KeyboardInterrupt:
        print("\n[WARN] 接收到 Ctrl+C，正在保存当前 skrl 模型...")
    except Exception:
        print("\n[ERROR] Allegro Task3 skrl PPO 训练过程中发生真实异常：")
        traceback.print_exc()
        raise
    finally:
        final_dir = os.path.join(log_dir, run_name, "final_checkpoint")

        try:
            save_project_checkpoint(
                final_dir,
                agent=agent,
                models=models,
                env_cfg=env_cfg,
                env=local_env,
                env_steps=int(total_env_steps),
                args=args_cli,
            )
            print(f"✅ Allegro Task3 skrl 模型已保存至 {final_dir}")
        except Exception as exc:
            print(f"[WARN] 保存最终 skrl 模型失败: {type(exc).__name__}: {exc}")

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

        print("✅ Allegro Task3 TRUE skrl PPO training pipeline safely exited")


if __name__ == "__main__":
    main()
