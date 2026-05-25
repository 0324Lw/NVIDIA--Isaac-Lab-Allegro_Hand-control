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

parser = argparse.ArgumentParser(description="Allegro Hand Task3 dynamic grasping/tool-use env test")
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

from allegro_rl.tasks.task3.task3_config import Task3Config
from allegro_rl.tasks.task3.task3_env import AllegroHandTask3Env


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
    print(" " * 54 + "Allegro Hand Task3 环境统计报告")
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


def assert_finite_tensor(name: str, x: torch.Tensor) -> None:
    assert torch.is_tensor(x), f"{name} must be torch.Tensor, got {type(x)}"
    assert torch.isfinite(x).all(), f"{name} contains NaN/Inf"


def check_obs_dict(cfg: Task3Config, obs: Dict[str, torch.Tensor]) -> None:
    assert isinstance(obs, dict), f"obs must be dict, got {type(obs)}"
    assert "obs" in obs and "privileged_obs" in obs, f"obs dict keys invalid: {obs.keys()}"

    actor = obs["obs"]
    privileged = obs["privileged_obs"]

    assert actor.shape == (cfg.num_envs, cfg.num_observations), (
        f"actor obs shape mismatch: {tuple(actor.shape)} != {(cfg.num_envs, cfg.num_observations)}"
    )
    assert privileged.shape == (cfg.num_envs, cfg.num_privileged_obs), (
        f"privileged obs shape mismatch: {tuple(privileged.shape)} != {(cfg.num_envs, cfg.num_privileged_obs)}"
    )

    assert_finite_tensor("actor obs", actor)
    assert_finite_tensor("privileged obs", privileged)

    assert actor.abs().max().item() <= float(cfg.actor_obs_clip) + 1e-5
    assert privileged.abs().max().item() <= float(cfg.privileged_obs_clip) + 1e-5


def write_active_tool_rel_state(env: AllegroHandTask3Env, env_ids, rel_pos, quat=None, vel=None):
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
    is_pen = env.active_tool[env_ids] == 0

    if is_pen.any():
        env.pen.write_root_state_to_sim(root_state[is_pen], env_ids=env_ids[is_pen])

    if (~is_pen).any():
        env.cup.write_root_state_to_sim(root_state[~is_pen], env_ids=env_ids[~is_pen])

    env.scene.update(dt=0.0)


def check_project_files() -> None:
    heading("[测试 0] Allegro Hand Task3 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task3_dynamic_grasp_tool_use.yaml",
        PROJECT_ROOT / "src" / "allegro_rl" / "tasks" / "task3" / "task3_config.py",
        PROJECT_ROOT / "src" / "allegro_rl" / "tasks" / "task3" / "task3_scene.py",
        PROJECT_ROOT / "src" / "allegro_rl" / "tasks" / "task3" / "task3_env.py",
    ]

    missing = [str(path) for path in required if not path.exists()]
    assert not missing, "缺少 Task3 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))

    print_ok("Allegro Hand Task3 工程文件结构正常")


def check_config() -> None:
    heading("[测试 1] Task3Config 基础配置检测")

    cfg = Task3Config()
    cfg.validate()

    assert cfg.num_hand_actions == 16
    assert cfg.num_base_actions == 6
    assert cfg.num_actions == 22
    assert cfg.num_observations == 147
    assert cfg.num_privileged_obs == 168
    assert cfg.frame_stack == 5

    print_ok(f"num_hand_actions = {cfg.num_hand_actions}")
    print_ok(f"num_base_actions = {cfg.num_base_actions}")
    print_ok(f"num_actions = {cfg.num_actions}")
    print_ok(f"num_observations = {cfg.num_observations}")
    print_ok(f"num_privileged_obs = {cfg.num_privileged_obs}")
    print_ok(f"curriculum_total_steps = {cfg.curriculum_total_steps:,}")
    print_ok("Task3Config 基础配置正常")


def check_env_init(cfg: Task3Config) -> AllegroHandTask3Env:
    heading("[测试 2] 环境初始化 / 模型信息 / 名称映射检测")

    env = AllegroHandTask3Env(cfg)

    print_ok(f"device = {env.device}")
    print_ok(f"num_envs = {env.num_envs}")
    print_ok(f"robot.num_joints = {env.robot.num_joints}")
    print_ok(f"num_actions = {env.num_actions}")
    print_ok(f"num_observations = {env.num_observations}")
    print_ok(f"num_privileged_obs = {env.num_privileged_obs}")

    assert env.robot.num_joints == cfg.num_hand_actions
    assert env.num_actions == 22
    assert env.num_observations == 147
    assert env.num_privileged_obs == 168
    assert len(env.fingertip_indices) == 4
    assert env.contact_tip_indices.numel() == 4
    assert 0 <= env.palm_index < len(env.robot_body_names)

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


def check_reset_spaces(env: AllegroHandTask3Env, cfg: Task3Config) -> Dict[str, torch.Tensor]:
    heading("[测试 3] reset / obs / privileged obs / spaces 检测")

    obs, info = env.reset(seed=args_cli.seed)
    check_obs_dict(cfg, obs)

    assert env.action_space.shape == (cfg.num_actions,)
    assert env.observation_space["obs"].shape == (cfg.num_observations,)
    assert env.observation_space["privileged_obs"].shape == (cfg.num_privileged_obs,)
    assert env.state_space.shape == (cfg.num_privileged_obs,)

    states = env.get_privileged_observations()
    assert states.shape == (cfg.num_envs, cfg.num_privileged_obs)
    assert_finite_tensor("privileged states", states)

    obj_pos, obj_quat, _, _ = env._get_active_tool_state()
    obj_rel = obj_pos - env._env_origins()

    assert obj_rel[:, 2].mean().item() > cfg.table_height - 0.02
    assert torch.allclose(torch.norm(obj_quat, dim=-1).mean(), torch.tensor(1.0, device=env.device), atol=5e-2)

    assert env.target_quat.shape == (cfg.num_envs, 4)
    assert torch.allclose(torch.norm(env.target_quat, dim=-1).mean(), torch.tensor(1.0, device=env.device), atol=5e-2)

    print_ok(f"action_space = {env.action_space}")
    print_ok(f"observation_space = {env.observation_space}")
    print_ok(f"state_space = {env.state_space}")
    print_ok(f"actor obs shape = {tuple(obs['obs'].shape)}")
    print_ok(f"privileged obs shape = {tuple(obs['privileged_obs'].shape)}")
    print_ok(f"object rel height mean = {obj_rel[:, 2].mean().item():.6f}")
    print_ok("reset / spaces / object state 正常")

    return obs


def check_obs_layout(env: AllegroHandTask3Env, cfg: Task3Config, obs: Dict[str, torch.Tensor]) -> None:
    heading("[测试 4] observation layout 切片检测")

    actor = obs["obs"]

    slices = {
        "base_pos_rel": actor[:, 0:3],
        "base_quat": actor[:, 3:7],
        "base_lin_vel": actor[:, 7:10],
        "base_ang_vel": actor[:, 10:13],
        "q": actor[:, 13:29],
        "qdot": actor[:, 29:45],
        "u_hand_prev": actor[:, 45:61],
        "u_base_prev": actor[:, 61:67],
        "obj_rel_pos": actor[:, 67:70],
        "obj_quat": actor[:, 70:74],
        "obj_lin_vel": actor[:, 74:77],
        "obj_ang_vel": actor[:, 77:80],
        "tool_oh": actor[:, 80:82],
        "target_height": actor[:, 82:83],
        "target_quat": actor[:, 83:87],
        "target_axis": actor[:, 87:90],
        "phase_oh": actor[:, 90:97],
        "tcp_to_obj": actor[:, 97:100],
        "fingertip_rel_obj": actor[:, 100:112],
        "fingertip_rel_keypoints": actor[:, 112:124],
        "contact_bools": actor[:, 124:128],
        "force_norms": actor[:, 128:132],
        "keypoints_rel_tcp": actor[:, 132:144],
        "lift_base_tcp": actor[:, 144:147],
    }

    for name, value in slices.items():
        assert_finite_tensor(name, value)

    assert torch.allclose(torch.norm(slices["base_quat"], dim=-1).mean(), torch.tensor(1.0, device=env.device), atol=5e-2)
    assert torch.allclose(torch.norm(slices["obj_quat"], dim=-1).mean(), torch.tensor(1.0, device=env.device), atol=5e-2)
    assert torch.all(slices["contact_bools"] >= -1e-5) and torch.all(slices["contact_bools"] <= 1.0 + 1e-5)
    assert torch.all(slices["force_norms"] >= -1e-5) and torch.all(slices["force_norms"] <= 1.0 + 1e-5)
    assert torch.allclose(slices["tool_oh"].sum(dim=-1).mean(), torch.tensor(1.0, device=env.device), atol=1e-5)
    assert torch.allclose(slices["phase_oh"].sum(dim=-1).mean(), torch.tensor(1.0, device=env.device), atol=1e-5)

    privileged = obs["privileged_obs"]
    dr_mass = privileged[:, 147]
    dr_obj_fric = privileged[:, 148]
    dr_table_fric = privileged[:, 149]
    dr_com = privileged[:, 150:153]
    tip_forces = privileged[:, 153:165]
    inertia_diag = privileged[:, 165:168]

    assert torch.all(dr_mass >= cfg.mass_range[0] - 1e-5)
    assert torch.all(dr_mass <= cfg.mass_range[1] + 1e-5)
    assert torch.all(dr_obj_fric >= cfg.object_friction_range[0] - 1e-5)
    assert torch.all(dr_obj_fric <= cfg.object_friction_range[1] + 1e-5)
    assert torch.all(dr_table_fric >= cfg.table_friction_range[0] - 1e-5)
    assert torch.all(dr_table_fric <= cfg.table_friction_range[1] + 1e-5)
    assert_finite_tensor("dr_com", dr_com)
    assert_finite_tensor("tip_forces", tip_forces)
    assert_finite_tensor("inertia_diag", inertia_diag)

    print_ok("actor obs layout = 147 dim 正常")
    print_ok("privileged obs = actor 147 + DR/contact/inertia 21 dim 正常")
    print_ok(f"object rel z mean = {slices['obj_rel_pos'][:, 2].mean().item():.6f}")
    print_ok(f"base height mean = {slices['base_pos_rel'][:, 2].mean().item():.6f}")
    print_ok(f"contact mean = {slices['contact_bools'].sum(dim=-1).mean().item():.6f}")


def check_control_chain(env: AllegroHandTask3Env, cfg: Task3Config) -> None:
    heading("[测试 5] 22 维动作控制链路检测：hand + floating base")

    env.reset(seed=args_cli.seed)

    q0 = env.robot.data.joint_pos.clone()
    base0 = env.base_pos_rel.clone()

    actions = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)
    actions[:, :16] = 0.65 * (torch.rand((cfg.num_envs, 16), dtype=torch.float32, device=env.device) * 2.0 - 1.0)
    actions[:, 16:19] = torch.tensor([0.5, -0.25, -0.35], dtype=torch.float32, device=env.device)
    actions[:, 19:22] = torch.tensor([0.15, 0.05, -0.10], dtype=torch.float32, device=env.device)

    latest_info = {}
    latest_rewards = None

    for _ in range(8):
        obs, rewards, terminated, truncated, latest_info = env.step(actions)
        latest_rewards = rewards

    q_delta = torch.norm(env.robot.data.joint_pos - q0, dim=-1).mean().item()
    base_delta = torch.norm(env.base_pos_rel - base0, dim=-1).mean().item()

    assert latest_rewards is not None
    assert_finite_tensor("control rewards", latest_rewards)
    assert q_delta > 1e-6, "动作没有引起手指关节变化"
    assert base_delta > 1e-6, "动作没有引起 floating base buffer 变化"

    check_obs_dict(cfg, obs)

    print_ok(f"手指平均位移范数 = {q_delta:.6f}")
    print_ok(f"基座平均位移范数 = {base_delta:.6f}")
    print_ok("hand + floating base 控制链路正常")


def check_tactile(env: AllegroHandTask3Env, cfg: Task3Config) -> None:
    heading("[测试 6] tactile / contact sensor 检测")

    contact_bools, force_norms, tip_forces = env._get_tactile()

    assert contact_bools.shape == (cfg.num_envs, 4)
    assert force_norms.shape == (cfg.num_envs, 4)
    assert tip_forces.shape == (cfg.num_envs, 4, 3)

    assert_finite_tensor("contact_bools", contact_bools)
    assert_finite_tensor("force_norms", force_norms)
    assert_finite_tensor("tip_forces", tip_forces)

    assert torch.all(contact_bools >= 0.0) and torch.all(contact_bools <= 1.0)
    assert torch.all(force_norms >= 0.0) and torch.all(force_norms <= 1.0 + 1e-5)

    print_ok(f"contact_bools shape = {tuple(contact_bools.shape)}")
    print_ok(f"force_norms shape = {tuple(force_norms.shape)}")
    print_ok(f"tip_forces shape = {tuple(tip_forces.shape)}")
    print_ok(f"soft contact mean = {env.debug_soft_contact.mean().item():.6f}")
    print_ok(f"hard contact mean = {contact_bools.mean().item():.6f}")


def check_step_reward_info(env: AllegroHandTask3Env, cfg: Task3Config) -> None:
    heading("[测试 7] action step / reward / info fields 检测")

    env.reset(seed=args_cli.seed)

    actions = torch.rand((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device) * 2.0 - 1.0

    latest_info = {}
    latest_rewards = None
    latest_terminated = None
    latest_truncated = None
    latest_obs = None

    for _ in range(20):
        latest_obs, latest_rewards, latest_terminated, latest_truncated, latest_info = env.step(actions)

    assert latest_rewards is not None
    assert latest_rewards.shape == (cfg.num_envs,)
    assert latest_terminated is not None and latest_terminated.shape == (cfg.num_envs,)
    assert latest_truncated is not None and latest_truncated.shape == (cfg.num_envs,)
    assert_finite_tensor("step rewards", latest_rewards)

    check_obs_dict(cfg, latest_obs)

    flat = flatten_info(latest_info)

    required_keys = [
        "reward_components/R_Approach",
        "reward_components/R_PreGrasp",
        "reward_components/R_Descend",
        "reward_components/R_Contact",
        "reward_components/R_ForceClosure",
        "reward_components/R_Grip",
        "reward_components/R_Lift",
        "reward_components/R_Orient",
        "reward_components/R_SoftContact",
        "reward_components/P_NoContact",
        "reward_components/P_Table",
        "reward_components/P_Workspace",
        "reward_components/Continuous",
        "reward_components/Event",
        "reward_components/Total",
        "events/Drop",
        "events/SlideOut",
        "events/TableCrash",
        "events/Success",
        "events/WorkspaceClamp",
        "telemetry/Phase",
        "telemetry/K",
        "telemetry/TCP_Dist",
        "telemetry/Contact_Count",
        "telemetry/Lift",
        "telemetry/SO3_Err",
        "telemetry/Obj_H",
        "debug/Actor_Obs_Dim",
        "debug/Privileged_Obs_Dim",
        "debug/Action_Dim",
    ]

    for key in required_keys:
        assert key in flat, f"info 缺少字段: {key}"

    print_ok(f"reward range = {latest_rewards.min().item():.6f} ~ {latest_rewards.max().item():.6f}")
    print_ok(f"contact count = {flat.get('telemetry/Contact_Count', 0.0):.6f}")
    print_ok(f"object height = {flat.get('telemetry/Obj_H', 0.0):.6f}")
    print_ok("action step / reward / info fields 正常")


def check_events(env: AllegroHandTask3Env, cfg: Task3Config) -> None:
    heading("[测试 8] drop / slide-out / timeout 事件注入检测")

    zero_actions = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)

    env.reset(seed=args_cli.seed)
    event_env_ids = torch.arange(min(32, cfg.num_envs), dtype=torch.long, device=env.device)

    write_active_tool_rel_state(
        env,
        event_env_ids,
        rel_pos=[0.0, 0.0, cfg.table_height - 0.12],
        vel=[0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    )

    obs, rewards, terminated, truncated, info = env.step(zero_actions)
    drop_hits = int(terminated[event_env_ids].sum().item())

    assert drop_hits > 0, "强制掉落后没有触发 terminated"

    print_ok(f"drop terminated count = {drop_hits}/{len(event_env_ids)}")

    env.reset(seed=args_cli.seed)
    slide_env_ids = torch.arange(min(32, cfg.num_envs), dtype=torch.long, device=env.device)

    write_active_tool_rel_state(
        env,
        slide_env_ids,
        rel_pos=[cfg.table_size_xy * 0.75, 0.0, cfg.table_height + cfg.pen_size[2] * 0.5],
        vel=[0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    obs, rewards, terminated, truncated, info = env.step(zero_actions)
    slide_hits = int(terminated[slide_env_ids].sum().item())

    assert slide_hits > 0, "强制滑出桌面后没有触发 terminated"

    print_ok(f"slide-out terminated count = {slide_hits}/{len(slide_env_ids)}")

    env.reset(seed=args_cli.seed)
    env.episode_steps[:] = int(cfg.max_episode_length) - 1

    obs, rewards, terminated, truncated, info = env.step(zero_actions)
    timeout_count = int(truncated.sum().item())

    assert timeout_count > 0, "episode_steps 到达 max_episode_length 后没有触发 truncated"

    check_obs_dict(cfg, obs)

    print_ok(f"timeout_count = {timeout_count}")
    print_ok("drop / slide-out / timeout 事件正常")


def check_curriculum(env: AllegroHandTask3Env, cfg: Task3Config) -> None:
    heading("[测试 9] curriculum phase / target 检测")

    old_steps = int(env.global_steps)
    probes = [0.0, 0.05, 0.13, 0.26, 0.45, 0.65, 0.85, 1.0]

    last_phase = -1

    print(f"{'k':>8} | {'global_steps':>14} | {'phase':>8} | {'target_lift':>12} | {'orient_max':>12}")
    print("-" * 70)

    for k in probes:
        env.global_steps = int(k * cfg.curriculum_total_steps)
        phase = env._get_phase()

        target_lift = cfg.lift_heights_by_phase[phase]
        orient_max = cfg.orient_max_angle_by_phase[phase]

        print(f"{k:>8.3f} | {env.global_steps:>14,d} | {phase:>8d} | {target_lift:>12.4f} | {orient_max:>12.4f}")

        assert phase >= last_phase, "课程阶段没有随 K 单调递增"
        last_phase = phase

        env.reset(seed=args_cli.seed)
        assert int(env.curriculum_phase.float().mean().item()) == phase

    env.global_steps = old_steps
    print_ok("curriculum phase / lift target / orientation target 正常")


def random_rollout(env: AllegroHandTask3Env, cfg: Task3Config) -> None:
    heading(f"[测试 10] 随机策略 rollout 稳定性检测：{args_cli.steps} steps x {cfg.num_envs} envs")

    env.global_steps = int(0.80 * cfg.curriculum_total_steps)
    obs, _ = env.reset(seed=args_cli.seed)

    info_history: List[Dict[str, float]] = []
    total_terminated = 0
    total_truncated = 0

    start_time = time.time()

    for step in range(int(args_cli.steps)):
        actions = torch.rand((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device) * 2.0 - 1.0

        obs, rewards, terminated, truncated, info = env.step(actions)

        total_terminated += int(terminated.sum().item())
        total_truncated += int(truncated.sum().item())

        check_obs_dict(cfg, obs)
        assert_finite_tensor("rollout rewards", rewards)
        assert_finite_tensor("robot joint_pos", env.robot.data.joint_pos)
        assert_finite_tensor("robot joint_vel", env.robot.data.joint_vel)
        assert_finite_tensor("base_pos_rel", env.base_pos_rel)
        assert_finite_tensor("base_quat", env.base_quat)

        if step % max(int(args_cli.collect_interval), 1) == 0 or step == int(args_cli.steps) - 1:
            flat = flatten_info(info)

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
                f"Phase={flat.get('telemetry/Phase', 0.0):.2f} | "
                f"K={flat.get('telemetry/K', 0.0):.3f} | "
                f"Contact={flat.get('telemetry/Contact_Count', 0.0):.3f} | "
                f"Lift={flat.get('telemetry/Lift', 0.0):.4f} | "
                f"ObjH={flat.get('telemetry/Obj_H', 0.0):.4f} | "
                f"Drop={flat.get('events/Drop', 0.0):.3f} | "
                f"Slide={flat.get('events/SlideOut', 0.0):.3f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    env_steps = int(args_cli.steps) * int(cfg.num_envs)
    fps = env_steps / max(elapsed, 1e-6)

    print_ok(f"随机策略 rollout 完成: {args_cli.steps} steps")
    print_ok(f"总 transitions: {env_steps:,}")
    print_ok(f"吞吐约: {fps:,.2f} env steps/s")
    print_ok(f"累计 terminated: {total_terminated:,}")
    print_ok(f"累计 truncated: {total_truncated:,}")

    heading("[测试 11] 奖励组件 / telemetry 统计分析")
    print_summary_table(summarize_records(info_history))

    print("Allegro Hand Task3 training pre-check guide:")
    print("1. actor obs 必须为 147，privileged obs 必须为 168。")
    print("2. action dim 必须为 22，其中 16 hand + 6 floating base。")
    print("3. 随机策略下 Drop / SlideOut / TableCrash 偏高是正常现象。")
    print("4. Contact_Count 随机策略可以较低，但不能因为传感器错误导致 NaN/Inf。")
    print("5. 后续训练重点看 Obj_H、Contact_Count、Lift、Drop、SlideOut、TableCrash、SO3_Err。")


def run_tests() -> None:
    heading("Allegro Hand Task3 Dynamic Grasping / Tool Use 环境全量测试启动")

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

    cfg = Task3Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.seed = int(args_cli.seed)
    cfg.debug_print_names = bool(args_cli.print_names)
    cfg.validate()

    env: AllegroHandTask3Env | None = None

    try:
        env = check_env_init(cfg)
        obs = check_reset_spaces(env, cfg)
        check_obs_layout(env, cfg, obs)
        check_control_chain(env, cfg)
        check_tactile(env, cfg)
        check_step_reward_info(env, cfg)
        check_events(env, cfg)
        check_curriculum(env, cfg)
        random_rollout(env, cfg)

        heading("Allegro Hand Task3 环境测试全部通过")

    except Exception as exc:
        print("\n❌ Allegro Hand Task3 环境测试失败：")
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
