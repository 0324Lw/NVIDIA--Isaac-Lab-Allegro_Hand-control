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


parser = argparse.ArgumentParser(description="Allegro Hand Task1 pose tracking env test")
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test-device", type=str, default="cuda:0")
parser.add_argument(
    "--dataset-path",
    type=str,
    default=os.environ.get(
        "ALLEGRO_TASK1_DATASET",
        str(PROJECT_ROOT / "assets" / "motions" / "task1_target_poses.pt"),
    ),
)
parser.add_argument("--collect-interval", type=int, default=500)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--print-joints", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from allegro_rl.tasks.task1.task1_config import Task1Config
from allegro_rl.tasks.task1.task1_env import AllegroHandTask1Env, PoseDataset


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
    print(" " * 56 + "Allegro Hand Task1 环境统计报告")
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


def check_obs_shape_and_values(env: AllegroHandTask1Env, obs: torch.Tensor) -> None:
    expected = (env.cfg.num_envs, env.cfg.num_observations)

    assert torch.is_tensor(obs), f"obs 必须是 torch.Tensor，当前为 {type(obs)}"
    assert tuple(obs.shape) == expected, f"obs shape 错误: {tuple(obs.shape)} != {expected}"
    assert_finite_tensor("obs", obs)

    max_abs = obs.abs().max().item()
    assert max_abs <= float(env.cfg.obs_clip) + 1e-5, f"obs 超出 clamp 范围: {max_abs:.6f}"


def check_project_files() -> None:
    heading("[测试 0] Allegro Hand Task1 工程文件存在性检查")

    required = [
        PROJECT_ROOT / "configs" / "task1_pose_tracking.yaml",
        PROJECT_ROOT / "src" / "allegro_rl" / "data" / "generate_task1_pose_dataset.py",
        PROJECT_ROOT / "src" / "allegro_rl" / "tasks" / "task1" / "task1_config.py",
        PROJECT_ROOT / "src" / "allegro_rl" / "tasks" / "task1" / "task1_env.py",
        PROJECT_ROOT / "assets" / "motions" / "README.md",
    ]

    missing = [str(path) for path in required if not path.exists()]
    assert not missing, "缺少 Task1 必要文件:\n" + "\n".join(missing)

    for path in required:
        print_ok(str(path.relative_to(PROJECT_ROOT)))

    print_ok("Allegro Hand Task1 工程文件结构正常")


def check_config() -> None:
    heading("[测试 1] Task1Config 基础配置检测")

    cfg = Task1Config()
    cfg.validate()

    assert cfg.num_actions == 16
    assert cfg.num_observations == 64
    assert cfg.frame_stack == 1
    assert cfg.stacked_obs_dim == 64
    assert cfg.curriculum_total_steps > 0
    assert 0.0 <= cfg.ema_alpha <= 1.0
    assert cfg.action_scale > 0.0

    print_ok(f"num_actions = {cfg.num_actions}")
    print_ok(f"num_observations = {cfg.num_observations}")
    print_ok(f"stacked_obs_dim = {cfg.stacked_obs_dim}")
    print_ok(f"curriculum_total_steps = {cfg.curriculum_total_steps:,}")
    print_ok(f"ema_alpha = {cfg.ema_alpha}")
    print_ok(f"action_scale = {cfg.action_scale}")
    print_ok("Task1Config 基础配置正常")


def check_dataset(cfg: Task1Config) -> None:
    heading("[测试 2] 姿态数据集完整性 / 采样器检测")

    dataset_path = Path(cfg.dataset_path).expanduser().resolve()
    print(f"dataset_path = {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"数据集不存在: {dataset_path}\n"
            "请先运行：bash scripts/ubuntu/generate_task1_dataset.sh"
        )

    dataset = PoseDataset(str(dataset_path), cfg.device)

    assert dataset.ctrl_min.shape == (cfg.num_actions,)
    assert dataset.ctrl_max.shape == (cfg.num_actions,)
    assert dataset.pool_easy.shape[1] == cfg.num_actions
    assert dataset.pool_hard.shape[1] == cfg.num_actions
    assert dataset.pool_semantic.shape[1] == cfg.num_actions

    for k in [0.0, 0.5, 1.0]:
        samples = dataset.sample_targets(128, k, cfg)
        assert samples.shape == (128, cfg.num_actions)
        assert_finite_tensor(f"dataset samples k={k}", samples)
        assert torch.all(samples >= dataset.ctrl_min.unsqueeze(0) - 1e-6)
        assert torch.all(samples <= dataset.ctrl_max.unsqueeze(0) + 1e-6)

    print_ok(f"ctrl_min shape = {tuple(dataset.ctrl_min.shape)}")
    print_ok(f"ctrl_max shape = {tuple(dataset.ctrl_max.shape)}")
    print_ok(f"semantic_names = {dataset.semantic_names}")
    print_ok(f"pool_easy shape = {tuple(dataset.pool_easy.shape)}")
    print_ok(f"pool_hard shape = {tuple(dataset.pool_hard.shape)}")
    print_ok(f"pool_semantic shape = {tuple(dataset.pool_semantic.shape)}")
    print_ok("姿态数据集完整性 / 采样器正常")


def check_observation_layout(env: AllegroHandTask1Env, obs: torch.Tensor) -> None:
    slices = {
        "target_error": obs[:, 0:16],
        "joint_pos": obs[:, 16:32],
        "joint_vel": obs[:, 32:48],
        "last_filtered_action": obs[:, 48:64],
    }

    for name, value in slices.items():
        assert_finite_tensor(name, value)

    assert torch.all(slices["last_filtered_action"].abs() <= 1.0001), (
        f"last_filtered_action out of range: "
        f"max_abs={slices['last_filtered_action'].abs().max().item():.6f}"
    )
    # Important:
    #   ctrl_min / ctrl_max are target-command safety limits.
    #   Actual joint_pos can slightly exceed those safety limits under PD
    #   dynamics and random actions. For actual robot state, only check the
    #   IsaacLab USD physical joint_pos_limits here. Soft-limit reward will
    #   penalize deviations from ctrl_min / ctrl_max during training.
    physical_lower = env.joint_limits[:, :, 0]
    physical_upper = env.joint_limits[:, :, 1]

    lower_margin = slices["joint_pos"] - physical_lower
    upper_margin = physical_upper - slices["joint_pos"]

    min_lower = lower_margin.min()
    min_upper = upper_margin.min()

    if min_lower.item() < -1e-4:
        flat_idx = int(torch.argmin(lower_margin).item())
        env_id = flat_idx // lower_margin.shape[1]
        joint_id = flat_idx % lower_margin.shape[1]
        joint_name = env.robot_joint_names[joint_id] if joint_id < len(env.robot_joint_names) else str(joint_id)
        raise AssertionError(
            f"joint_pos below physical joint limit: "
            f"env={env_id}, joint={joint_id}({joint_name}), "
            f"q={slices['joint_pos'][env_id, joint_id].item():.6f}, "
            f"physical_min={physical_lower[env_id, joint_id].item():.6f}, "
            f"ctrl_min={env.ctrl_min[env_id, joint_id].item():.6f}, "
            f"margin={lower_margin[env_id, joint_id].item():.6f}"
        )

    if min_upper.item() < -1e-4:
        flat_idx = int(torch.argmin(upper_margin).item())
        env_id = flat_idx // upper_margin.shape[1]
        joint_id = flat_idx % upper_margin.shape[1]
        joint_name = env.robot_joint_names[joint_id] if joint_id < len(env.robot_joint_names) else str(joint_id)
        raise AssertionError(
            f"joint_pos above physical joint limit: "
            f"env={env_id}, joint={joint_id}({joint_name}), "
            f"q={slices['joint_pos'][env_id, joint_id].item():.6f}, "
            f"physical_max={physical_upper[env_id, joint_id].item():.6f}, "
            f"ctrl_max={env.ctrl_max[env_id, joint_id].item():.6f}, "
            f"margin={upper_margin[env_id, joint_id].item():.6f}"
        )

    print_ok("obs layout = target_error 16 + joint_pos 16 + joint_vel 16 + last_filtered_action 16")
    print_ok(f"target_error mean abs = {slices['target_error'].abs().mean().item():.6f}")
    print_ok(f"joint_pos range = {slices['joint_pos'].min().item():.6f} ~ {slices['joint_pos'].max().item():.6f}")
    print_ok(f"joint_vel mean abs = {slices['joint_vel'].abs().mean().item():.6f}")


def check_curriculum(env: AllegroHandTask1Env, cfg: Task1Config) -> None:
    heading("[测试 6] curriculum_k / dataset probability 检测")

    old_steps = int(env.global_steps)
    probes = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"{'k':>8} | {'global_steps':>14} | {'semantic':>10} | {'hard':>10} | {'easy':>10}")
    print("-" * 64)

    last_easy = None
    last_hard = None

    for k in probes:
        env.global_steps = int(k * cfg.curriculum_total_steps)
        ck = env.curriculum_k()
        semantic, hard, easy = env.dataset._probabilities(ck, cfg)

        print(f"{ck:>8.3f} | {env.global_steps:>14,d} | {semantic:>10.4f} | {hard:>10.4f} | {easy:>10.4f}")

        assert 0.0 <= semantic <= 1.0
        assert 0.0 <= hard <= 1.0
        assert 0.0 <= easy <= 1.0
        assert abs((semantic + hard + easy) - 1.0) < 1e-5

        if last_easy is not None:
            assert easy <= last_easy + 1e-6, "easy probability should not increase with curriculum"
            assert hard >= last_hard - 1e-6, "hard probability should not decrease with curriculum"

        last_easy = easy
        last_hard = hard

    env.global_steps = old_steps
    print_ok("curriculum_k / dataset probability 正常")


def check_reset_and_spaces(env: AllegroHandTask1Env, cfg: Task1Config) -> torch.Tensor:
    heading("[测试 4] reset / obs / state / action space 检测")

    obs, info = env.reset(seed=args_cli.seed)

    check_obs_shape_and_values(env, obs)
    check_observation_layout(env, obs)

    state = env.get_privileged_observations()
    check_obs_shape_and_values(env, state)

    print_ok(f"observation_space.shape = {env.observation_space.shape}")
    print_ok(f"state_space.shape = {env.state_space.shape}")
    print_ok(f"action_space.shape = {env.action_space.shape}")

    assert env.observation_space.shape == (cfg.num_observations,), (
        f"observation_space shape mismatch: "
        f"{env.observation_space.shape} != {(cfg.num_observations,)}"
    )
    assert env.state_space.shape == (cfg.num_observations,), (
        f"state_space shape mismatch: "
        f"{env.state_space.shape} != {(cfg.num_observations,)}"
    )
    assert env.action_space.shape == (cfg.num_actions,), (
        f"action_space shape mismatch: "
        f"{env.action_space.shape} != {(cfg.num_actions,)}"
    )

    assert env.target_poses.shape == (cfg.num_envs, cfg.num_actions), (
        f"target_poses shape mismatch: {env.target_poses.shape}"
    )
    assert env.current_targets.shape == (cfg.num_envs, cfg.num_actions), (
        f"current_targets shape mismatch: {env.current_targets.shape}"
    )
    assert env.previous_error.shape == (cfg.num_envs, cfg.num_actions), (
        f"previous_error shape mismatch: {env.previous_error.shape}"
    )
    assert env.a_t_minus_1.shape == (cfg.num_envs, cfg.num_actions), (
        f"a_t_minus_1 shape mismatch: {env.a_t_minus_1.shape}"
    )
    assert env.u_t_minus_1.shape == (cfg.num_envs, cfg.num_actions), (
        f"u_t_minus_1 shape mismatch: {env.u_t_minus_1.shape}"
    )

    assert_finite_tensor("target_poses", env.target_poses)
    assert torch.all(env.target_poses >= env.ctrl_min - 1e-6), (
        f"target_poses below ctrl_min: "
        f"target_min={env.target_poses.min().item():.6f}, "
        f"ctrl_min={env.ctrl_min.min().item():.6f}"
    )
    assert torch.all(env.target_poses <= env.ctrl_max + 1e-6), (
        f"target_poses above ctrl_max: "
        f"target_max={env.target_poses.max().item():.6f}, "
        f"ctrl_max={env.ctrl_max.max().item():.6f}"
    )

    print_ok(f"observation_space = {env.observation_space}")
    print_ok(f"state_space = {env.state_space}")
    print_ok(f"action_space = {env.action_space}")
    print_ok(f"reset obs shape = {tuple(obs.shape)}")
    print_ok(f"target_poses shape = {tuple(env.target_poses.shape)}")
    print_ok("reset / spaces / target buffers 正常")

    return obs


def check_action_step(env: AllegroHandTask1Env, cfg: Task1Config) -> Dict[str, Any]:
    heading("[测试 7] action control / reward / info fields 检测")

    obs, _ = env.reset(seed=args_cli.seed)

    q0 = env.robot.data.joint_pos.clone()

    test_actions = torch.rand((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device) * 2.0 - 1.0

    latest_info: Dict[str, Any] = {}
    latest_rewards = None

    for _ in range(20):
        obs, rewards, terminated, truncated, latest_info = env.step(test_actions)
        latest_rewards = rewards

    q1 = env.robot.data.joint_pos.clone()
    q_delta = torch.norm(q1 - q0, dim=-1).mean().item()

    assert q_delta > 1e-5, "随机动作没有引起 Allegro 关节变化"
    assert latest_rewards is not None
    assert latest_rewards.shape == (cfg.num_envs,)
    assert_finite_tensor("step rewards", latest_rewards)
    check_obs_shape_and_values(env, obs)
    check_observation_layout(env, obs)

    assert terminated.shape == (cfg.num_envs,)
    assert truncated.shape == (cfg.num_envs,)

    flat = flatten_info(latest_info)
    required_keys = [
        "reward_components/R_Track_Mean",
        "reward_components/R_Track_Worst",
        "reward_components/R_Progress",
        "reward_components/R_Stable",
        "reward_components/P_Action_Mag",
        "reward_components/P_Action_Smooth",
        "reward_components/P_Soft_Limit",
        "reward_components/Total_Reward",
        "events/Timeout_Rate",
        "telemetry/Pose_Error_Rad",
        "telemetry/Joint_Velocity",
        "telemetry/Curriculum_Progress_K",
        "debug/Obs_Dim",
        "debug/Action_Dim",
    ]

    for key in required_keys:
        assert key in flat, f"info 缺少字段: {key}"

    print_ok(f"controlled joint 平均位移范数 = {q_delta:.6f}")
    print_ok(f"reward range = {latest_rewards.min().item():.6f} ~ {latest_rewards.max().item():.6f}")
    print_ok("action control / reward / info fields 正常")

    return latest_info


def check_timeout_event(env: AllegroHandTask1Env, cfg: Task1Config) -> None:
    heading("[测试 8] timeout / reset 事件检测")

    env.reset(seed=args_cli.seed)
    env.episode_steps[:] = int(cfg.max_episode_length) - 1

    zero_actions = torch.zeros((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device)

    obs, rewards, terminated, truncated, info = env.step(zero_actions)

    timeout_count = int(truncated.sum().item())
    assert timeout_count > 0, "episode_steps 到达 max_episode_length 后没有触发 truncated"
    assert terminated.sum().item() == 0, "Task1 不应该出现 physical terminated"

    flat = flatten_info(info)
    assert "events/Timeout_Rate" in flat

    print_ok(f"timeout_count = {timeout_count}")
    print_ok(f"Timeout_Rate = {flat.get('events/Timeout_Rate', 0.0):.6f}")
    print_ok("timeout / reset 事件正常")


def random_rollout(env: AllegroHandTask1Env, cfg: Task1Config) -> None:
    heading(f"[测试 9] 随机策略 rollout 稳定性检测：{args_cli.steps} steps x {cfg.num_envs} envs")

    env.global_steps = int(0.80 * cfg.curriculum_total_steps)
    obs, _ = env.reset(seed=args_cli.seed)

    info_history: List[Dict[str, float]] = []
    total_timeouts = 0

    start_time = time.time()

    for step in range(int(args_cli.steps)):
        actions = torch.rand((cfg.num_envs, cfg.num_actions), dtype=torch.float32, device=env.device) * 2.0 - 1.0

        obs, rewards, terminated, truncated, info = env.step(actions)

        total_timeouts += int(truncated.sum().item())

        assert terminated.sum().item() == 0, "Task1 should not early terminate"
        assert_finite_tensor("rollout obs", obs)
        assert_finite_tensor("rollout rewards", rewards)
        assert_finite_tensor("robot joint_pos", env.robot.data.joint_pos)
        assert_finite_tensor("robot joint_vel", env.robot.data.joint_vel)

        if step % max(int(args_cli.collect_interval), 1) == 0 or step == int(args_cli.steps) - 1:
            check_obs_shape_and_values(env, obs)

            flat = flatten_info(info)
            flat["test/step"] = float(step)
            flat["test/reward_mean"] = float(rewards.mean().detach().cpu().item())
            flat["test/reward_min"] = float(rewards.min().detach().cpu().item())
            flat["test/reward_max"] = float(rewards.max().detach().cpu().item())
            flat["test/truncated_count"] = float(truncated.sum().detach().cpu().item())

            info_history.append(flat)

            print(
                f"step={step + 1:>5}/{args_cli.steps} | "
                f"Reward={flat.get('test/reward_mean', 0.0):+8.5f} | "
                f"PoseErr={flat.get('telemetry/Pose_Error_Rad', 0.0):.5f} | "
                f"MaxErr={flat.get('telemetry/Pose_Error_Max_Rad', 0.0):.5f} | "
                f"JointVel={flat.get('telemetry/Joint_Velocity', 0.0):.5f} | "
                f"K={flat.get('telemetry/Curriculum_Progress_K', 0.0):.4f} | "
                f"Timeout={flat.get('events/Timeout_Rate', 0.0):.3f} | "
                f"RTrack={flat.get('reward_components/R_Track', 0.0):+.5f} | "
                f"RProg={flat.get('reward_components/R_Progress', 0.0):+.5f} | "
                f"Smooth={flat.get('reward_components/P_Action_Smooth', 0.0):+.5f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    env_steps = int(args_cli.steps) * int(cfg.num_envs)
    fps = env_steps / max(elapsed, 1e-6)

    print_ok(f"随机策略 rollout 完成: {args_cli.steps} steps")
    print_ok(f"总 transitions: {env_steps:,}")
    print_ok(f"吞吐约: {fps:,.2f} env steps/s")
    print_ok(f"累计 truncated: {total_timeouts:,}")

    heading("[测试 10] 奖励组件 / telemetry 统计分析")
    print_summary_table(summarize_records(info_history))

    print("Allegro Hand Task1 training pre-check guide:")
    print("1. action_dim 应为 16，actor obs_dim 应为 64。")
    print("2. obs layout 应为 error / q / qdot / last_filtered_action。")
    print("3. Task1 没有物理提前终止，只有 timeout truncated。")
    print("4. 随机策略下 Pose_Error_Rad 不需要很低，但不能 NaN/Inf。")
    print("5. R_Progress 在随机策略下可以较小或波动，这是正常现象。")
    print("6. 后续训练重点看 Pose_Error_Rad 是否下降，R_Track / R_Stable 是否上升。")


def run_tests() -> None:
    heading("Allegro Hand Task1 Pose Tracking 环境 / 数据集 / rollout 全量测试启动")

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

    cfg = Task1Config()
    cfg.num_envs = int(args_cli.num_envs)
    cfg.device = str(device)
    cfg.seed = int(args_cli.seed)
    cfg.dataset_path = str(Path(args_cli.dataset_path).expanduser().resolve())
    cfg.print_debug_info = bool(args_cli.print_joints)
    cfg.validate()

    check_dataset(cfg)

    env: AllegroHandTask1Env | None = None

    try:
        heading("[测试 3] 环境初始化 / 模型信息 / 名称映射检测")

        env = AllegroHandTask1Env(cfg)

        print_ok(f"device = {device}")
        print_ok(f"num_envs = {cfg.num_envs}")
        print_ok(f"robot.num_joints = {env.robot.num_joints}")
        print_ok(f"num_actions = {env.num_actions}")
        print_ok(f"num_observations = {cfg.num_observations}")
        print_ok(f"dataset_path = {cfg.dataset_path}")

        assert env.robot.num_joints == cfg.num_actions
        assert env.num_actions == 16
        assert cfg.num_observations == 64
        assert len(env.robot_joint_names) == 16

        if args_cli.print_joints:
            print("\nrobot.joint_names:")
            for i, name in enumerate(env.robot_joint_names):
                print(f"  {i:02d}: {name}")

        obs = check_reset_and_spaces(env, cfg)

        heading("[测试 5] joint limits / target range 检测")

        assert env.joint_limits.shape == (cfg.num_envs, cfg.num_actions, 2)
        assert env.ctrl_min.shape == (cfg.num_envs, cfg.num_actions)
        assert env.ctrl_max.shape == (cfg.num_envs, cfg.num_actions)
        assert torch.all(env.ctrl_min < env.ctrl_max)
        assert torch.all(env.target_poses >= env.ctrl_min - 1e-6)
        assert torch.all(env.target_poses <= env.ctrl_max + 1e-6)

        q = env.robot.data.joint_pos
        physical_lower = env.joint_limits[:, :, 0]
        physical_upper = env.joint_limits[:, :, 1]

        q_lower_margin = q - physical_lower
        q_upper_margin = physical_upper - q

        if q_lower_margin.min().item() < -1e-4:
            flat_idx = int(torch.argmin(q_lower_margin).item())
            env_id = flat_idx // q_lower_margin.shape[1]
            joint_id = flat_idx % q_lower_margin.shape[1]
            joint_name = env.robot_joint_names[joint_id] if joint_id < len(env.robot_joint_names) else str(joint_id)
            raise AssertionError(
                f"reset joint_pos below physical limit: "
                f"env={env_id}, joint={joint_id}({joint_name}), "
                f"q={q[env_id, joint_id].item():.6f}, "
                f"physical_min={physical_lower[env_id, joint_id].item():.6f}, "
                f"ctrl_min={env.ctrl_min[env_id, joint_id].item():.6f}"
            )

        if q_upper_margin.min().item() < -1e-4:
            flat_idx = int(torch.argmin(q_upper_margin).item())
            env_id = flat_idx // q_upper_margin.shape[1]
            joint_id = flat_idx % q_upper_margin.shape[1]
            joint_name = env.robot_joint_names[joint_id] if joint_id < len(env.robot_joint_names) else str(joint_id)
            raise AssertionError(
                f"reset joint_pos above physical limit: "
                f"env={env_id}, joint={joint_id}({joint_name}), "
                f"q={q[env_id, joint_id].item():.6f}, "
                f"physical_max={physical_upper[env_id, joint_id].item():.6f}, "
                f"ctrl_max={env.ctrl_max[env_id, joint_id].item():.6f}"
            )

        print_ok(f"joint_limits shape = {tuple(env.joint_limits.shape)}")
        print_ok(f"ctrl_min mean = {env.ctrl_min.mean().item():.6f}")
        print_ok(f"ctrl_max mean = {env.ctrl_max.mean().item():.6f}")
        print_ok(f"target range = {env.target_poses.min().item():.6f} ~ {env.target_poses.max().item():.6f}")
        print_ok("joint limits / target range 正常")

        check_curriculum(env, cfg)
        check_action_step(env, cfg)
        check_timeout_event(env, cfg)
        random_rollout(env, cfg)

        heading("Allegro Hand Task1 环境测试全部通过")

    except Exception as exc:
        print("\n❌ Allegro Hand Task1 环境测试失败：")
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
