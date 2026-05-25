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

parser = argparse.ArgumentParser(description="Allegro Hand Task2 in-hand reorientation env test")
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

from allegro_rl.tasks.task2.task2_config import Task2Config
from allegro_rl.tasks.task2.task2_env import AllegroHandTask2Env


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
    print(" " * 54 + "Allegro Hand Task2 环境统计报告")
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


def check_obs_dict(cfg: Task2Config, obs: Dict[str, torch.Tensor]) -> None:
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


def check_project_files() -> None:
    heading("[测试 0] Allegro Hand Task2 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task2_inhand_reorientation.yaml",
        PROJECT_ROOT / "src" / "allegro_rl" / "tasks" / "task2" / "task2_config.py",
        PROJECT_ROOT / "src" / "allegro_rl" / "tasks" / "task2" / "task2_scene.py",
        PROJECT_ROOT / "src" / "allegro_rl" / "tasks" / "task2" / "task2_env.py",
    ]

    missing = [str(path) for path in required if not path.exists()]
    assert not missing, "缺少 Task2 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))

    print_ok("Allegro Hand Task2 工程文件结构正常")


def check_config() -> None:
    heading("[测试 1] Task2Config 基础配置检测")

    cfg = Task2Config()
    cfg.validate()

    assert cfg.num_actions == 16
    assert cfg.num_observations == 83
    assert cfg.num_privileged_obs == 88
    assert cfg.frame_stack == 5
    assert cfg.stacked_actor_obs_dim == 83 * 5

    print_ok(f"num_actions = {cfg.num_actions}")
    print_ok(f"num_observations = {cfg.num_observations}")
    print_ok(f"num_privileged_obs = {cfg.num_privileged_obs}")
    print_ok(f"frame_stack = {cfg.frame_stack}")
    print_ok(f"curriculum_total_steps = {cfg.curriculum_total_steps:,}")
    print_ok("Task2Config 基础配置正常")


def check_env_init(cfg: Task2Config) -> AllegroHandTask2Env:
    heading("[测试 2] 环境初始化 / 模型信息 / 名称映射检测")

    env = AllegroHandTask2Env(cfg)

    print_ok(f"device = {env.device}")
    print_ok(f"num_envs = {env.num_envs}")
    print_ok(f"robot.num_joints = {env.robot.num_joints}")
    print_ok(f"num_actions = {env.num_actions}")
    print_ok(f"num_observations = {env.num_observations}")
    print_ok(f"num_privileged_obs = {env.num_privileged_obs}")

    assert env.robot.num_joints == cfg.num_actions
    assert env.num_actions == 16
    assert env.num_observations == 83
    assert env.num_privileged_obs == 88
    assert len(env.fingertip_indices) == 4
    assert env.contact_sensor_tip_indices.numel() == 4

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


def check_reset_spaces(env: AllegroHandTask2Env, cfg: Task2Config) -> Dict[str, torch.Tensor]:
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

    obj_pos, obj_quat, obj_lin_vel, obj_ang_vel = env._get_active_object_state()
    obj_rel = obj_pos - env._env_origins()

    assert_finite_tensor("obj_pos", obj_pos)
    assert_finite_tensor("obj_quat", obj_quat)
    assert torch.allclose(torch.norm(obj_quat, dim=-1).mean(), torch.tensor(1.0, device=env.device), atol=5e-2)
    assert obj_rel[:, 2].mean().item() > 0.40

    assert env.target_quats.shape == (cfg.num_envs, 4)
    assert torch.allclose(torch.norm(env.target_quats, dim=-1).mean(), torch.tensor(1.0, device=env.device), atol=5e-2)

    print_ok(f"action_space = {env.action_space}")
    print_ok(f"observation_space = {env.observation_space}")
    print_ok(f"state_space = {env.state_space}")
    print_ok(f"actor obs shape = {tuple(obs['obs'].shape)}")
    print_ok(f"privileged obs shape = {tuple(obs['privileged_obs'].shape)}")
    print_ok(f"object height mean = {obj_rel[:, 2].mean().item():.6f}")
    print_ok("reset / spaces / object state 正常")

    return obs


def check_obs_layout(env: AllegroHandTask2Env, cfg: Task2Config, obs: Dict[str, torch.Tensor]) -> None:
    heading("[测试 4] observation layout 切片检测")

    actor = obs["obs"]
    slices = {
        "q": actor[:, 0:16],
        "qdot": actor[:, 16:32],
        "last_u": actor[:, 32:48],
        "obj_rel_pos": actor[:, 48:51],
        "obj_quat": actor[:, 51:55],
        "obj_lin_vel": actor[:, 55:58],
        "obj_ang_vel": actor[:, 58:61],
        "fingertip_rel_pos": actor[:, 61:73],
        "contact_bools": actor[:, 73:77],
        "shape_onehot": actor[:, 77:79],
        "theta": actor[:, 79:80],
        "axis_err": actor[:, 80:83],
    }

    for name, value in slices.items():
        assert_finite_tensor(name, value)

    assert torch.all(slices["contact_bools"] >= -1e-5) and torch.all(slices["contact_bools"] <= 1.0 + 1e-5)
    assert torch.allclose(slices["shape_onehot"].sum(dim=-1).mean(), torch.tensor(1.0, device=env.device), atol=1e-5)
    assert torch.all(slices["theta"] >= -1e-5)

    privileged = obs["privileged_obs"]
    dr_mass = privileged[:, 83]
    dr_friction = privileged[:, 84]
    dr_com = privileged[:, 85:88]

    assert torch.all(dr_mass >= cfg.mass_range[0] - 1e-5)
    assert torch.all(dr_mass <= cfg.mass_range[1] + 1e-5)
    assert torch.all(dr_friction >= cfg.friction_range[0] - 1e-5)
    assert torch.all(dr_friction <= cfg.friction_range[1] + 1e-5)
    assert_finite_tensor("dr_com", dr_com)

    print_ok("actor obs layout = 83 dim 正常")
    print_ok("privileged obs = actor 83 + mass/friction/com 5 dim 正常")
    print_ok(f"theta mean = {slices['theta'].mean().item():.6f}")
    print_ok(f"object rel z mean = {slices['obj_rel_pos'][:, 2].mean().item():.6f}")
    print_ok(f"active contacts mean = {slices['contact_bools'].sum(dim=-1).mean().item():.6f}")


def check_contact_sensor(env: AllegroHandTask2Env, cfg: Task2Config) -> None:
    heading("[测试 5] contact sensor / fingertip mapping 检测")

    contacts = env._get_contact_bools()

    assert contacts.shape == (cfg.num_envs, 4)
    assert_finite_tensor("contact bools", contacts)
    assert torch.all(contacts >= 0.0) and torch.all(contacts <= 1.0)

    print_ok(f"contact bools shape = {tuple(contacts.shape)}")
    print_ok(f"contact mean = {contacts.mean().item():.6f}")
    print_ok(f"selected robot fingertip indices = {env.fingertip_indices}")
    print_ok(f"selected contact sensor indices = {env.contact_sensor_tip_indices.tolist()}")


def check_step_reward(env: AllegroHandTask2Env, cfg: Task2Config) -> Dict[str, Any]:
    heading("[测试 6] action step / reward / info fields 检测")

    obs, _ = env.reset(seed=args_cli.seed)
    q0 = env.robot.data.joint_pos.clone()

    actions = torch.rand((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device) * 2.0 - 1.0

    latest_info: Dict[str, Any] = {}
    latest_rewards = None
    latest_terminated = None
    latest_truncated = None

    for _ in range(20):
        obs, rewards, terminated, truncated, latest_info = env.step(actions)
        latest_rewards = rewards
        latest_terminated = terminated
        latest_truncated = truncated

    q1 = env.robot.data.joint_pos.clone()
    q_delta = torch.norm(q1 - q0, dim=-1).mean().item()

    assert q_delta > 1e-5, "随机动作没有引起 Allegro 关节变化"
    assert latest_rewards is not None
    assert latest_rewards.shape == (cfg.num_envs,)
    assert latest_terminated is not None and latest_terminated.shape == (cfg.num_envs,)
    assert latest_truncated is not None and latest_truncated.shape == (cfg.num_envs,)
    assert_finite_tensor("step rewards", latest_rewards)

    check_obs_dict(cfg, obs)

    flat = flatten_info(latest_info)
    required_keys = [
        "reward_components/R_Rot",
        "reward_components/R_Prog",
        "reward_components/R_Contact",
        "reward_components/R_Safe",
        "reward_components/R_Height",
        "reward_components/P_Act_Rate",
        "reward_components/P_Joint_Vel",
        "reward_components/P_Excess_Vel",
        "reward_components/Event_Bonus_Penalty",
        "reward_components/Total_Reward",
        "events/Drop_Rate",
        "events/Success_Rate",
        "events/Timeout_Rate",
        "telemetry/Geodesic_Error_Rad",
        "telemetry/Object_Height",
        "telemetry/Active_Contacts",
        "telemetry/Curriculum_Progress_K",
        "debug/Actor_Obs_Dim",
        "debug/Privileged_Obs_Dim",
    ]

    for key in required_keys:
        assert key in flat, f"info 缺少字段: {key}"

    print_ok(f"joint 平均位移范数 = {q_delta:.6f}")
    print_ok(f"reward range = {latest_rewards.min().item():.6f} ~ {latest_rewards.max().item():.6f}")
    print_ok(f"drop rate = {flat.get('events/Drop_Rate', 0.0):.6f}")
    print_ok(f"object height = {flat.get('telemetry/Object_Height', 0.0):.6f}")
    print_ok("action step / reward / info fields 正常")

    return latest_info


def check_timeout_event(env: AllegroHandTask2Env, cfg: Task2Config) -> None:
    heading("[测试 7] timeout / reset 事件检测")

    env.reset(seed=args_cli.seed)
    env.episode_steps[:] = int(cfg.max_episode_length) - 1

    actions = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)

    obs, rewards, terminated, truncated, info = env.step(actions)

    timeout_count = int(truncated.sum().item())
    assert timeout_count > 0, "episode_steps 到达 max_episode_length 后没有触发 truncated"

    check_obs_dict(cfg, obs)

    flat = flatten_info(info)
    assert "events/Timeout_Rate" in flat

    print_ok(f"timeout_count = {timeout_count}")
    print_ok(f"Timeout_Rate = {flat.get('events/Timeout_Rate', 0.0):.6f}")
    print_ok("timeout / reset 事件正常")


def check_manual_drop_event(env: AllegroHandTask2Env, cfg: Task2Config) -> None:
    heading("[测试 8] manual drop 事件检测")

    env.reset(seed=args_cli.seed)

    env_ids = torch.arange(min(8, cfg.num_envs), dtype=torch.long, device=env.device)
    m = int(env_ids.numel())
    origins = env._env_origins(env_ids)

    low_pos = origins + torch.tensor([0.0, 0.0, cfg.drop_height - 0.05], dtype=torch.float32, device=env.device)
    quat = env._identity_quat(m)
    zero_vel = env._zero_vel6(m)

    # 当前默认 use_only_cube=True，因此写 cube 即可。
    env.cube.write_root_state_to_sim(torch.cat([low_pos, quat, zero_vel], dim=-1), env_ids=env_ids)
    env.cube.reset(env_ids)
    env.scene.update(dt=0.0)

    actions = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)
    obs, rewards, terminated, truncated, info = env.step(actions)

    drop_count = int(terminated[env_ids].sum().item())
    assert drop_count > 0, "手动降低物体高度后没有触发 drop terminated"

    print_ok(f"manual drop count among first {m} envs = {drop_count}")
    print_ok("manual drop 事件正常")


def check_curriculum(env: AllegroHandTask2Env, cfg: Task2Config) -> None:
    heading("[测试 9] curriculum target quaternion 检测")

    old_steps = int(env.global_steps)
    probes = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"{'k':>8} | {'global_steps':>14} | {'theta_mean':>12} | {'theta_max':>12}")
    print("-" * 58)

    for k in probes:
        env.global_steps = int(k * cfg.curriculum_total_steps)
        env.reset(seed=args_cli.seed)

        obj_pos, obj_quat, _, _ = env._get_active_object_state()
        theta = env._compute_geodesic_distance(obj_quat, env.target_quats)

        print(f"{env.curriculum_k():>8.3f} | {env.global_steps:>14,d} | {theta.mean().item():>12.6f} | {theta.max().item():>12.6f}")

        assert torch.isfinite(theta).all()
        assert theta.min().item() >= -1e-5
        assert theta.max().item() <= math.pi + 1e-4

    env.global_steps = old_steps
    print_ok("curriculum target quaternion 正常")


def random_rollout(env: AllegroHandTask2Env, cfg: Task2Config) -> None:
    heading(f"[测试 10] 随机策略 rollout 稳定性检测：{args_cli.steps} steps x {cfg.num_envs} envs")

    env.global_steps = int(0.80 * cfg.curriculum_total_steps)
    obs, _ = env.reset(seed=args_cli.seed)

    info_history: List[Dict[str, float]] = []
    total_drops = 0
    total_timeouts = 0
    total_success = 0

    start_time = time.time()

    for step in range(int(args_cli.steps)):
        actions = torch.rand((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device) * 2.0 - 1.0

        obs, rewards, terminated, truncated, info = env.step(actions)

        total_drops += int(terminated.sum().item())
        total_timeouts += int(truncated.sum().item())

        flat = flatten_info(info)
        total_success += int(round(flat.get("events/Success_Rate", 0.0) * cfg.num_envs))

        check_obs_dict(cfg, obs)
        assert_finite_tensor("rollout rewards", rewards)
        assert_finite_tensor("robot joint_pos", env.robot.data.joint_pos)
        assert_finite_tensor("robot joint_vel", env.robot.data.joint_vel)

        if step % max(int(args_cli.collect_interval), 1) == 0 or step == int(args_cli.steps) - 1:
            row = {
                "test/step": float(step),
                "test/reward_mean": float(rewards.detach().float().mean().cpu().item()),
                "test/reward_min": float(rewards.detach().float().min().cpu().item()),
                "test/reward_max": float(rewards.detach().float().max().cpu().item()),
                "test/drop_count": float(terminated.sum().detach().cpu().item()),
                "test/truncated_count": float(truncated.sum().detach().cpu().item()),
            }
            row.update(flat)
            info_history.append(row)

            print(
                f"step={step + 1:>5}/{args_cli.steps} | "
                f"Reward={row['test/reward_mean']:+8.5f} | "
                f"Theta={flat.get('telemetry/Geodesic_Error_Rad', 0.0):.5f} | "
                f"Height={flat.get('telemetry/Object_Height', 0.0):.5f} | "
                f"Contacts={flat.get('telemetry/Active_Contacts', 0.0):.3f} | "
                f"Drop={flat.get('events/Drop_Rate', 0.0):.3f} | "
                f"K={flat.get('telemetry/Curriculum_Progress_K', 0.0):.3f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    env_steps = int(args_cli.steps) * int(cfg.num_envs)
    fps = env_steps / max(elapsed, 1e-6)

    print_ok(f"随机策略 rollout 完成: {args_cli.steps} steps")
    print_ok(f"总 transitions: {env_steps:,}")
    print_ok(f"吞吐约: {fps:,.2f} env steps/s")
    print_ok(f"累计 drops: {total_drops:,}")
    print_ok(f"累计 timeouts: {total_timeouts:,}")
    print_ok(f"估算 success events: {total_success:,}")

    heading("[测试 11] 奖励组件 / telemetry 统计分析")
    print_summary_table(summarize_records(info_history))

    print("Allegro Hand Task2 training pre-check guide:")
    print("1. actor obs 必须为 83，privileged obs 必须为 88。")
    print("2. ContactSensor 必须能返回 [num_envs, 4] 的指尖接触布尔。")
    print("3. 随机策略下 Drop_Rate 高是正常的，说明掉落终止链路有效。")
    print("4. Object_Height 不应 NaN/Inf，掉落后应自动 reset。")
    print("5. 后续训练重点看 Object_Height、Drop_Rate、Active_Contacts、Geodesic_Error_Rad。")


def run_tests() -> None:
    heading("Allegro Hand Task2 In-Hand Reorientation 环境全量测试启动")

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

    cfg = Task2Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.seed = int(args_cli.seed)
    cfg.debug_print_names = bool(args_cli.print_names)
    cfg.validate()

    env: AllegroHandTask2Env | None = None

    try:
        env = check_env_init(cfg)
        obs = check_reset_spaces(env, cfg)
        check_obs_layout(env, cfg, obs)
        check_contact_sensor(env, cfg)
        check_step_reward(env, cfg)
        check_timeout_event(env, cfg)
        check_manual_drop_event(env, cfg)
        check_curriculum(env, cfg)
        random_rollout(env, cfg)

        heading("Allegro Hand Task2 环境测试全部通过")

    except Exception as exc:
        print("\n❌ Allegro Hand Task2 环境测试失败：")
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
