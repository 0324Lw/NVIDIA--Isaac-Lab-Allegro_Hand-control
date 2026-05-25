from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Allegro Hand Task4 Blind Sim2Real/RMA env test")
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument("--collect-interval", type=int, default=500)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-names", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

simulation_app = AppLauncher(args_cli).app

from allegro_rl.tasks.task4.task4_config import Task4Config
from allegro_rl.tasks.task4.task4_env import AllegroHandTask4Env, so3_distance


def heading(title: str) -> None:
    print("\n" + "=" * 132)
    print(title)
    print("=" * 132)


def print_ok(msg: str) -> None:
    print(f" ✅ {msg}", flush=True)


def print_warn(msg: str) -> None:
    print(f" ⚠️ {msg}", flush=True)


def to_float(x: Any):
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, np.ndarray):
            return float(np.mean(x))
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
    except Exception:
        return None
    return None


def flatten_info(info: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}

    for key, value in (info or {}).items():
        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flatten_info(value, name))
        else:
            val = to_float(value)
            if val is not None and math.isfinite(val):
                out[name] = val

    return out


def summarize_records(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
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
            "var": float(np.var(vals)),
            "min": float(np.min(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "max": float(np.max(vals)),
        }

    return out


def print_summary_table(summary: Dict[str, Dict[str, float]]) -> None:
    if not summary:
        print_warn("没有收集到有效统计字段")
        return

    print("\n" + "=" * 184)
    print(" " * 54 + "Allegro Hand Task4 Sim2Real / RMA 环境统计报告")
    print("=" * 184)
    print(
        f"{'metric':<74} | {'mean':>11} | {'std':>11} | {'var':>11} | "
        f"{'min':>11} | {'p25':>11} | {'p50':>11} | {'p75':>11} | {'max':>11}"
    )
    print("-" * 184)

    for key in sorted(summary.keys()):
        row = summary[key]
        print(
            f"{key:<74} | "
            f"{row['mean']:>11.5f} | "
            f"{row['std']:>11.5f} | "
            f"{row['var']:>11.5f} | "
            f"{row['min']:>11.5f} | "
            f"{row['p25']:>11.5f} | "
            f"{row['p50']:>11.5f} | "
            f"{row['p75']:>11.5f} | "
            f"{row['max']:>11.5f}"
        )

    print("=" * 184 + "\n")


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    assert torch.is_tensor(tensor), f"{name} 必须是 torch.Tensor，当前为 {type(tensor)}"
    assert torch.isfinite(tensor).all(), f"{name} 出现 NaN 或 Inf"


def check_obs_shapes(cfg: Task4Config, obs: Dict[str, torch.Tensor]) -> None:
    assert isinstance(obs, dict), f"reset()/step() 必须返回 Dict obs，当前为 {type(obs)}"

    expected = {
        "obs": (cfg.num_envs, cfg.num_observations),
        "teacher_obs": (cfg.num_envs, cfg.num_teacher_obs),
        "privileged_obs": (cfg.num_envs, cfg.num_privileged_obs),
        "history_obs": (cfg.num_envs, cfg.history_obs_dim),
    }

    for key, shape in expected.items():
        assert key in obs, f"obs 字典缺少 {key}"
        assert obs[key].shape == shape, f"{key} 维度错误: {obs[key].shape} != {shape}"
        assert_finite_tensor(f"obs['{key}']", obs[key])


def write_active_object_rel_state(env: AllegroHandTask4Env, env_ids, rel_pos, quat=None, vel=None):
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).flatten()
    n = int(env_ids.numel())

    rel_pos = torch.as_tensor(rel_pos, dtype=torch.float32, device=env.device).view(1, 3).repeat(n, 1)

    if quat is None:
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=env.device).view(1, 4).repeat(n, 1)
    else:
        quat = torch.as_tensor(quat, dtype=torch.float32, device=env.device).view(1, 4).repeat(n, 1)

    if vel is None:
        vel = torch.zeros((n, 6), dtype=torch.float32, device=env.device)
    else:
        vel = torch.as_tensor(vel, dtype=torch.float32, device=env.device).view(1, 6).repeat(n, 1)

    root_state = torch.cat([env._env_origins(env_ids) + rel_pos, quat, vel], dim=-1)
    is_cube = env.active_shape[env_ids] == 0

    if is_cube.any():
        env.cube.write_root_state_to_sim(root_state[is_cube], env_ids=env_ids[is_cube])
    if (~is_cube).any():
        env.sphere.write_root_state_to_sim(root_state[~is_cube], env_ids=env_ids[~is_cube])

    env.scene.update(dt=0.0)


def summarize_dr(env: AllegroHandTask4Env):
    return {
        "mass_min": env.dr_mass.min().item(),
        "mass_mean": env.dr_mass.mean().item(),
        "mass_max": env.dr_mass.max().item(),
        "friction_min": env.dr_friction.min().item(),
        "friction_mean": env.dr_friction.mean().item(),
        "friction_max": env.dr_friction.max().item(),
        "delay_min": env.dr_action_delay.float().min().item(),
        "delay_mean": env.dr_action_delay.float().mean().item(),
        "delay_max": env.dr_action_delay.float().max().item(),
        "eff_min": env.dr_joint_eff.min().item(),
        "eff_mean": env.dr_joint_eff.mean().item(),
        "eff_max": env.dr_joint_eff.max().item(),
        "deadzone_mean": env.dr_deadzone.mean().item(),
        "q_noise_mean": env.dr_q_noise.mean().item(),
        "qd_noise_mean": env.dr_qd_noise.mean().item(),
        "tactile_dropout_mean": env.dr_tactile_dropout.mean().item(),
        "state_dropout_mean": env.dr_state_dropout.mean().item(),
        "action_alpha_mean": env.dr_action_alpha.mean().item(),
    }


def check_project_files() -> None:
    heading("[测试 0] Allegro Hand Task4 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task4_blind_sim2real_rma.yaml",
        PROJECT_ROOT / "src" / "allegro_rl" / "tasks" / "task4" / "task4_config.py",
        PROJECT_ROOT / "src" / "allegro_rl" / "tasks" / "task4" / "task4_scene.py",
        PROJECT_ROOT / "src" / "allegro_rl" / "tasks" / "task4" / "task4_env.py",
    ]

    missing = [str(path) for path in required if not path.exists()]
    assert not missing, "缺少 Task4 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))

    print_ok("Allegro Hand Task4 工程文件结构正常")


def check_config() -> None:
    heading("[测试 1] Task4Config 基础配置检测")

    cfg = Task4Config()
    cfg.validate()

    assert cfg.num_actions == 16
    assert cfg.num_observations == 108
    assert cfg.num_teacher_obs == 139
    assert cfg.num_privileged_obs == 206
    assert cfg.history_frame_dim == 104
    assert cfg.history_length == 50
    assert cfg.history_obs_dim == 5200

    print_ok(f"num_actions = {cfg.num_actions}")
    print_ok(f"num_observations = {cfg.num_observations}")
    print_ok(f"num_teacher_obs = {cfg.num_teacher_obs}")
    print_ok(f"num_privileged_obs = {cfg.num_privileged_obs}")
    print_ok(f"history_frame_dim = {cfg.history_frame_dim}")
    print_ok(f"history_length = {cfg.history_length}")
    print_ok(f"history_obs_dim = {cfg.history_obs_dim}")
    print_ok(f"curriculum_total_steps = {cfg.curriculum_total_steps:,}")
    print_ok("Task4Config 基础配置正常")


def check_env_init(cfg: Task4Config) -> AllegroHandTask4Env:
    heading("[测试 2] 环境初始化 / 模型信息 / 名称映射检测")

    env = AllegroHandTask4Env(cfg)

    print_ok(f"device = {env.device}")
    print_ok(f"num_envs = {env.num_envs}")
    print_ok(f"robot.num_joints = {env.robot.num_joints}")
    print_ok(f"num_actions = {env.num_actions}")
    print_ok(f"num_observations = {env.num_observations}")
    print_ok(f"num_teacher_obs = {env.num_teacher_obs}")
    print_ok(f"num_privileged_obs = {env.num_privileged_obs}")
    print_ok(f"history_obs_dim = {env.history_obs_dim}")

    assert env.robot.num_joints == cfg.num_actions
    assert env.num_actions == 16
    assert env.num_observations == 108
    assert env.num_teacher_obs == 139
    assert env.num_privileged_obs == 206
    assert env.history_obs_dim == 5200
    assert len(env.fingertip_indices) == 4
    assert env.contact_tip_indices.numel() == 4

    if args_cli.print_names:
        print("\nrobot.joint_names:")
        for i, name in enumerate(env.robot_joint_names):
            print(f"  {i:02d}: {name}")

        print("\nrobot.body_names:")
        for i, name in enumerate(env.robot_body_names):
            print(f"  {i:02d}: {name}")

        print("\ncontact_sensor.body_names:")
        for i, name in enumerate(env.contact_body_names):
            print(f"  {i:02d}: {name}")

    return env


def check_reset_spaces(env: AllegroHandTask4Env, cfg: Task4Config) -> Dict[str, torch.Tensor]:
    heading("[测试 3] reset / obs / teacher / privileged / history / spaces 检测")

    obs, info = env.reset(seed=args_cli.seed)

    check_obs_shapes(cfg, obs)

    assert env.action_space.shape == (cfg.num_actions,)
    assert env.observation_space["obs"].shape == (cfg.num_observations,)
    assert env.observation_space["teacher_obs"].shape == (cfg.num_teacher_obs,)
    assert env.observation_space["privileged_obs"].shape == (cfg.num_privileged_obs,)
    assert env.observation_space["history_obs"].shape == (cfg.history_obs_dim,)
    assert env.state_space.shape == (cfg.num_privileged_obs,)

    states = env.get_privileged_observations()
    assert states.shape == (cfg.num_envs, cfg.num_privileged_obs)
    assert_finite_tensor("privileged states", states)

    hist_norm = torch.norm(obs["history_obs"], dim=-1).mean().item()
    assert hist_norm > 0.0, "history buffer 没有在 reset 时填充"

    obj_pos, obj_quat, _, _ = env._get_active_object_state()
    obj_rel = obj_pos - env._env_origins()

    assert obj_rel[:, 2].mean().item() > cfg.drop_height
    assert torch.allclose(torch.norm(obj_quat, dim=-1).mean(), torch.tensor(1.0, device=env.device), atol=5e-2)

    print_ok(f"action_space = {env.action_space}")
    print_ok(f"observation_space = {env.observation_space}")
    print_ok(f"state_space = {env.state_space}")
    print_ok(f"blind obs shape = {tuple(obs['obs'].shape)}")
    print_ok(f"teacher obs shape = {tuple(obs['teacher_obs'].shape)}")
    print_ok(f"privileged obs shape = {tuple(obs['privileged_obs'].shape)}")
    print_ok(f"history obs shape = {tuple(obs['history_obs'].shape)}")
    print_ok(f"history norm = {hist_norm:.6f}")
    print_ok(f"object height mean = {obj_rel[:, 2].mean().item():.6f}")
    print_ok("reset / spaces / object state 正常")

    return obs


def check_obs_layout(env: AllegroHandTask4Env, cfg: Task4Config, obs: Dict[str, torch.Tensor]) -> None:
    heading("[测试 4] observation layout 切片检测")

    blind = obs["obs"]
    teacher = obs["teacher_obs"]
    privileged = obs["privileged_obs"]
    history = obs["history_obs"]

    blind_slices = {
        "q": blind[:, 0:16],
        "qd": blind[:, 16:32],
        "qerr": blind[:, 32:48],
        "raw_action_prev": blind[:, 48:64],
        "applied_action": blind[:, 64:80],
        "motor_effort": blind[:, 80:96],
        "contact_bools": blind[:, 96:100],
        "force_norms": blind[:, 100:104],
        "target_quats": blind[:, 104:108],
    }

    for name, value in blind_slices.items():
        assert_finite_tensor(name, value)

    assert torch.all(blind_slices["contact_bools"] >= 0.0) and torch.all(blind_slices["contact_bools"] <= 1.0)
    assert torch.all(blind_slices["force_norms"] >= 0.0) and torch.all(blind_slices["force_norms"] <= 1.0 + 1e-5)

    obj_rel = teacher[:, 108:111]
    obj_quat = teacher[:, 111:115]
    theta = teacher[:, 121:122]
    shape_onehot = teacher[:, 137:139]

    assert_finite_tensor("teacher obj_rel", obj_rel)
    assert_finite_tensor("teacher obj_quat", obj_quat)
    assert_finite_tensor("teacher theta", theta)
    assert torch.all(theta >= -1e-5)
    assert torch.allclose(shape_onehot.sum(dim=-1).mean(), torch.tensor(1.0, device=env.device), atol=1e-5)

    dr = privileged[:, 139:206]
    assert dr.shape[-1] == 67
    assert_finite_tensor("privileged DR vector", dr)

    assert history.shape[-1] == cfg.history_obs_dim
    assert_finite_tensor("history obs", history)

    print_ok("blind obs layout = 108 dim 正常")
    print_ok("teacher obs layout = 139 dim 正常")
    print_ok("privileged obs = teacher 139 + DR 67 dim 正常")
    print_ok("history obs = 104 x 50 = 5200 dim 正常")
    print_ok(f"theta mean = {theta.mean().item():.6f}")
    print_ok(f"contact count mean = {blind_slices['contact_bools'].sum(dim=-1).mean().item():.6f}")


def check_dr_sampling(env: AllegroHandTask4Env, cfg: Task4Config) -> None:
    heading("[测试 5] Sim2Real DR 参数采样范围检测")

    env.global_steps = 0
    obs, _ = env.reset(seed=args_cli.seed)

    dr = summarize_dr(env)

    for k, v in dr.items():
        print(f" - {k:<24}: {v:.6f}")

    assert cfg.mass_range[0] - 1e-6 <= dr["mass_min"] <= cfg.mass_range[1] + 1e-6
    assert cfg.mass_range[0] - 1e-6 <= dr["mass_max"] <= cfg.mass_range[1] + 1e-6
    assert cfg.friction_range[0] - 1e-6 <= dr["friction_min"] <= cfg.friction_range[1] + 1e-6
    assert cfg.friction_range[0] - 1e-6 <= dr["friction_max"] <= cfg.friction_range[1] + 1e-6
    assert 0 <= dr["delay_min"] <= cfg.max_action_delay_steps
    assert 0 <= dr["delay_max"] <= cfg.max_action_delay_steps
    assert dr["eff_min"] > 0.0
    assert 0.0 <= dr["tactile_dropout_mean"] <= 1.0
    assert 0.0 <= dr["state_dropout_mean"] <= 1.0

    print_ok("质量 / 摩擦 / 延迟 / 效率 / 噪声 / dropout DR 参数采样正常")


def check_action_model(env: AllegroHandTask4Env, cfg: Task4Config) -> None:
    heading("[测试 6] action delay / deadzone / motor efficiency 控制链路检测")

    env.reset(seed=args_cli.seed)

    q0 = env.robot.data.joint_pos.clone()
    applied0 = env.applied_action.clone()

    test_actions = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)
    test_actions[:] = 0.7 * (torch.rand((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device) * 2.0 - 1.0)

    latest_rewards = None
    latest_obs = None
    latest_info = {}

    for _ in range(10):
        latest_obs, latest_rewards, terminated, truncated, latest_info = env.step(test_actions)

    q_delta = torch.norm(env.robot.data.joint_pos - q0, dim=-1).mean().item()
    applied_delta = torch.norm(env.applied_action - applied0, dim=-1).mean().item()
    delay_mean = env.dr_action_delay.float().mean().item()
    deadzone_mean = env.dr_deadzone.mean().item()

    assert latest_rewards is not None
    assert_finite_tensor("rewards", latest_rewards)
    assert applied_delta > 1e-6, "动作模型没有产生 applied_action 变化"
    assert q_delta > 1e-6, "动作没有引起手指关节变化"

    check_obs_shapes(cfg, latest_obs)

    print_ok(f"applied_action 平均变化 = {applied_delta:.6f}")
    print_ok(f"手指关节平均位移范数 = {q_delta:.6f}")
    print_ok(f"动作延迟均值 = {delay_mean:.3f} steps")
    print_ok(f"死区均值 = {deadzone_mean:.6f}")
    print_ok("action delay / deadzone / motor efficiency 控制链路正常")


def check_step_reward_info(env: AllegroHandTask4Env, cfg: Task4Config) -> None:
    heading("[测试 7] 向量化 step / reward / info fields 检测")

    env.reset(seed=args_cli.seed)

    actions = torch.rand((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device) * 2.0 - 1.0

    latest_obs = None
    latest_rewards = None
    latest_terminated = None
    latest_truncated = None
    latest_info = {}

    for _ in range(20):
        latest_obs, latest_rewards, latest_terminated, latest_truncated, latest_info = env.step(actions)

    assert latest_rewards is not None
    assert latest_rewards.shape == (cfg.num_envs,)
    assert latest_terminated is not None and latest_terminated.shape == (cfg.num_envs,)
    assert latest_truncated is not None and latest_truncated.shape == (cfg.num_envs,)

    assert_finite_tensor("step rewards", latest_rewards)
    check_obs_shapes(cfg, latest_obs)

    flat = flatten_info(latest_info)

    required_keys = [
        "reward_components/R_Rot",
        "reward_components/R_Progress",
        "reward_components/R_Contact",
        "reward_components/R_Center",
        "reward_components/R_Height",
        "reward_components/R_Stable",
        "reward_components/R_Recovery",
        "reward_components/P_ActionRate",
        "reward_components/P_ActionMag",
        "reward_components/P_JointVel",
        "reward_components/P_Energy",
        "reward_components/P_TorqueSpike",
        "reward_components/P_ForceSpike",
        "reward_components/Continuous",
        "reward_components/Event",
        "reward_components/Total",
        "events/Drop",
        "events/Success",
        "events/RecoveryMode",
        "telemetry/K",
        "telemetry/DR_K",
        "telemetry/Reward_K",
        "telemetry/SO3_Error",
        "telemetry/Contact_Count",
        "telemetry/Object_Height",
        "telemetry/Mass",
        "telemetry/Friction",
        "telemetry/ActionDelay",
        "telemetry/Deadzone",
        "telemetry/JointEfficiency",
        "telemetry/TactileDropout",
        "telemetry/StateDropout",
        "debug/Actor_Obs_Dim",
        "debug/Teacher_Obs_Dim",
        "debug/Privileged_Obs_Dim",
        "debug/History_Obs_Dim",
        "debug/Action_Dim",
    ]

    for key in required_keys:
        assert key in flat, f"info 缺少字段: {key}"

    assert latest_rewards.min().item() >= cfg.penalty_drop_full - 5.0
    assert latest_rewards.max().item() <= cfg.bonus_success_full + cfg.continuous_reward_clip + 5.0

    print_ok(f"奖励范围 = {latest_rewards.min().item():.6f} ~ {latest_rewards.max().item():.6f}")
    print_ok(f"SO3_Error = {flat.get('telemetry/SO3_Error', 0.0):.6f}")
    print_ok(f"Contact_Count = {flat.get('telemetry/Contact_Count', 0.0):.6f}")
    print_ok(f"Object_Height = {flat.get('telemetry/Object_Height', 0.0):.6f}")
    print_ok("step / reward / info fields 正常")


def check_drop_timeout_curriculum(env: AllegroHandTask4Env, cfg: Task4Config) -> None:
    heading("[测试 8] drop / timeout / curriculum target 检测")

    env.reset(seed=args_cli.seed)

    event_env_ids = torch.arange(min(64, cfg.num_envs), dtype=torch.long, device=env.device)

    write_active_object_rel_state(
        env,
        event_env_ids,
        rel_pos=[0.0, 0.0, cfg.drop_height - 0.08],
        vel=[0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    )

    zero_actions = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)

    obs, rewards, terminated, truncated, info = env.step(zero_actions)

    injected_drop_hits = int(terminated[event_env_ids].sum().item())
    assert injected_drop_hits > 0, "强制掉落后没有任何环境触发 terminated"

    print_ok(f"drop terminated count = {injected_drop_hits}/{len(event_env_ids)}")
    print_ok(f"drop reward min among injected = {rewards[event_env_ids].min().item():.6f}")

    env.reset(seed=args_cli.seed)
    env.episode_steps[:] = int(cfg.max_episode_length) - 1

    obs, rewards, terminated, truncated, info = env.step(zero_actions)

    timeout_count = int(truncated.sum().item())
    assert timeout_count > 0, "episode_steps 到达 max_episode_length 后没有触发 truncated"

    check_obs_shapes(cfg, obs)
    print_ok(f"timeout_count = {timeout_count}")

    old_steps = int(env.global_steps)
    ks = [0.0, 0.05, 0.25, 0.50, 0.75, 1.0]
    probe_ids = torch.arange(min(256, cfg.num_envs), dtype=torch.long, device=env.device)

    print(f"{'k':>8} | {'global_steps':>14} | {'theta_mean':>12} | {'theta_max':>12} | {'expected_max':>12}")
    print("-" * 70)

    last_expected = -1.0

    for k in ks:
        env.global_steps = int(k * cfg.curriculum_total_steps)
        env._sample_targets(probe_ids)

        _, obj_quat, _, _ = env._get_active_object_state()
        theta = so3_distance(obj_quat[probe_ids], env.target_quats[probe_ids])

        expected_max = cfg.target_min_angle + min(1.0, k) * (cfg.target_max_angle - cfg.target_min_angle)

        print(
            f"{k:>8.3f} | {env.global_steps:>14,d} | "
            f"{theta.mean().item():>12.6f} | {theta.max().item():>12.6f} | {expected_max:>12.6f}"
        )

        assert expected_max >= last_expected
        assert torch.isfinite(theta).all()
        assert theta.min().item() >= -1e-5
        assert theta.max().item() <= cfg.target_max_angle + 1e-3

        last_expected = expected_max

    env.global_steps = old_steps
    print_ok("drop / timeout / curriculum target 正常")


def check_observation_noise_history(env: AllegroHandTask4Env, cfg: Task4Config) -> None:
    heading("[测试 9] 观测噪声 / 状态丢失 / 触觉丢失 / history buffer 通路检测")

    env.reset(seed=args_cli.seed)

    obs_a = env._compute_obs(update_history=True)
    obs_b = env._compute_obs(update_history=True)

    obs_delta = torch.norm(obs_b["obs"] - obs_a["obs"], dim=-1).mean().item()
    hist_norm = torch.norm(obs_b["history_obs"], dim=-1).mean().item()

    assert_finite_tensor("noisy blind obs", obs_b["obs"])
    assert_finite_tensor("teacher obs", obs_b["teacher_obs"])
    assert_finite_tensor("privileged obs", obs_b["privileged_obs"])
    assert_finite_tensor("history obs", obs_b["history_obs"])

    assert hist_norm > 0.0, "history buffer 没有写入有效数据"

    frame_dim = cfg.history_frame_dim
    last_frame = obs_b["history_obs"][:, -frame_dim:]
    assert last_frame.shape == (cfg.num_envs, cfg.history_frame_dim)
    assert_finite_tensor("last history frame", last_frame)

    print_ok(f"连续两次 blind obs 差异均值 = {obs_delta:.6f}")
    print_ok(f"history buffer norm = {hist_norm:.6f}")
    print_ok("观测噪声 / dropout / history buffer 通路正常")


def random_rollout(env: AllegroHandTask4Env, cfg: Task4Config) -> None:
    heading(f"[测试 10] 随机策略 rollout 稳定性检测：{args_cli.steps} steps x {cfg.num_envs} envs")

    env.global_steps = int(0.80 * cfg.curriculum_total_steps)
    obs, _ = env.reset(seed=args_cli.seed)

    info_history: List[Dict[str, float]] = []

    total_drops = 0
    total_success = 0
    total_timeouts = 0

    start_time = time.time()

    for step in range(int(args_cli.steps)):
        random_actions = torch.rand((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device) * 2.0 - 1.0

        obs, rewards, terminated, truncated, info = env.step(random_actions)

        check_obs_shapes(cfg, obs)
        assert_finite_tensor("rollout rewards", rewards)
        assert_finite_tensor("robot joint_pos", env.robot.data.joint_pos)
        assert_finite_tensor("robot joint_vel", env.robot.data.joint_vel)

        flat = flatten_info(info)

        total_drops += int(terminated.sum().item())
        total_timeouts += int(truncated.sum().item())
        total_success += int(round(flat.get("events/Success", 0.0) * cfg.num_envs))

        if step % max(int(args_cli.collect_interval), 1) == 0 or step == int(args_cli.steps) - 1:
            row = {
                "test/step": float(step),
                "test/reward_mean": float(rewards.detach().float().mean().cpu().item()),
                "test/reward_min": float(rewards.detach().float().min().cpu().item()),
                "test/reward_max": float(rewards.detach().float().max().cpu().item()),
                "test/terminated_count": float(terminated.sum().detach().cpu().item()),
                "test/truncated_count": float(truncated.sum().detach().cpu().item()),
            }
            row.update(flat)
            info_history.append(row)

            print(
                f"step={step + 1:>5}/{args_cli.steps} | "
                f"Reward={row['test/reward_mean']:+8.5f} | "
                f"K={flat.get('telemetry/K', 0.0):.3f} | "
                f"DR={flat.get('telemetry/DR_K', 0.0):.3f} | "
                f"SO3={flat.get('telemetry/SO3_Error', 0.0):.3f} | "
                f"Contact={flat.get('telemetry/Contact_Count', 0.0):.3f} | "
                f"Height={flat.get('telemetry/Object_Height', 0.0):.3f} | "
                f"Delay={flat.get('telemetry/ActionDelay', 0.0):.2f} | "
                f"Drop={flat.get('events/Drop', 0.0):.3f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    env_steps = int(args_cli.steps) * int(cfg.num_envs)
    fps = env_steps / max(elapsed, 1e-6)

    print_ok(f"随机策略 rollout 完成: {args_cli.steps} steps")
    print_ok(f"总 transitions: {env_steps:,}")
    print_ok(f"吞吐约: {fps:,.2f} env steps/s")
    print_ok(f"累计 drops: {total_drops:,}")
    print_ok(f"累计 success estimate: {total_success:,}")
    print_ok(f"累计 timeouts: {total_timeouts:,}")

    heading("[测试 11] 奖励组件 / telemetry 统计分析")
    print_summary_table(summarize_records(info_history))

    print("Allegro Hand Task4 training pre-check guide:")
    print("1. obs=108, teacher_obs=139, privileged_obs=206, history_obs=5200 必须全部正确。")
    print("2. history_obs norm 必须大于 0，说明 RMA 历史输入正常生成。")
    print("3. ActionDelay、Deadzone、JointEfficiency、TactileDropout、StateDropout 应随 DR 采样变化。")
    print("4. 随机策略下 Drop 较高是正常的，但不应出现 NaN/Inf。")
    print("5. reward_components/Continuous 应被裁剪到 [-1, 1] 附近；Event 可突破连续奖励裁剪。")
    print("6. 后续训练重点看 Object_Height、SO3_Error、Contact_Count、Drop、Success、DR_K、Reward_K。")


def run_tests() -> None:
    heading("Allegro Hand Task4 Blind Sim2Real / RMA 环境全量测试启动")

    torch.manual_seed(int(args_cli.seed))
    np.random.seed(int(args_cli.seed))

    if args_cli.test_device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print_warn("CUDA 不可用，自动切换到 CPU")
    else:
        device = args_cli.test_device

    if bool(args_cli.quick):
        args_cli.num_envs = min(int(args_cli.num_envs), 64)
        args_cli.steps = min(int(args_cli.steps), 200)
        args_cli.collect_interval = min(int(args_cli.collect_interval), 50)

    check_project_files()
    check_config()

    cfg = Task4Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.seed = int(args_cli.seed)
    cfg.debug_print_names = bool(args_cli.print_names)
    cfg.validate()

    env: AllegroHandTask4Env | None = None

    try:
        env = check_env_init(cfg)
        obs = check_reset_spaces(env, cfg)
        check_obs_layout(env, cfg, obs)
        check_dr_sampling(env, cfg)
        check_action_model(env, cfg)
        check_step_reward_info(env, cfg)
        check_drop_timeout_curriculum(env, cfg)
        check_observation_noise_history(env, cfg)
        random_rollout(env, cfg)

        heading("Allegro Hand Task4 环境测试全部通过")

    except Exception as exc:
        print("\n❌ Allegro Hand Task4 环境测试失败：")
        print(type(exc).__name__, ":", exc)
        raise

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        run_tests()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
