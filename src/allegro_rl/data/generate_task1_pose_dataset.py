from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import torch

from allegro_rl.common.paths import default_asset_path


def allegro_joint_limits() -> tuple[torch.Tensor, torch.Tensor]:
    """Return Allegro Hand 16-DoF joint limits in the original action order.

    Order:
        Index  : ffj0, ffj1, ffj2, ffj3
        Middle : mfj0, mfj1, mfj2, mfj3
        Ring   : rfj0, rfj1, rfj2, rfj3
        Thumb  : thj0, thj1, thj2, thj3
    """
    ctrl_min = torch.tensor(
        [
            -0.470, -0.196, -0.174, -0.227,
            -0.470, -0.196, -0.174, -0.227,
            -0.470, -0.196, -0.174, -0.227,
             0.263, -0.105, -0.189, -0.162,
        ],
        dtype=torch.float32,
    )

    ctrl_max = torch.tensor(
        [
            0.470, 1.610, 1.709, 1.618,
            0.470, 1.610, 1.709, 1.618,
            0.470, 1.610, 1.709, 1.618,
            1.396, 1.163, 1.644, 1.719,
        ],
        dtype=torch.float32,
    )

    return ctrl_min, ctrl_max


def build_semantic_poses(ctrl_min: torch.Tensor, ctrl_max: torch.Tensor) -> Dict[str, torch.Tensor]:
    num_actions = 16
    poses: Dict[str, torch.Tensor] = {}

    pose_open = torch.zeros(num_actions, dtype=torch.float32)
    pose_open[12] = 0.5
    poses["open"] = torch.clamp(pose_open, ctrl_min, ctrl_max)

    pose_fist = pose_open.clone()
    for finger in range(4):
        base = finger * 4
        pose_fist[base + 1] = ctrl_max[base + 1]
        pose_fist[base + 2] = ctrl_max[base + 2]
        pose_fist[base + 3] = ctrl_max[base + 3]
    poses["fist"] = torch.clamp(pose_fist, ctrl_min, ctrl_max)

    pose_point = pose_fist.clone()
    pose_point[0:4] = pose_open[0:4]
    poses["point"] = torch.clamp(pose_point, ctrl_min, ctrl_max)

    pose_v = pose_fist.clone()
    pose_v[0:4] = pose_open[0:4]
    pose_v[4:8] = pose_open[4:8]
    poses["v_sign"] = torch.clamp(pose_v, ctrl_min, ctrl_max)

    pose_pinch = pose_open.clone()
    pose_pinch[1:4] = torch.tensor([0.8, 0.8, 0.8], dtype=torch.float32)
    pose_pinch[13:16] = torch.tensor([0.6, 0.6, 0.6], dtype=torch.float32)
    poses["pinch"] = torch.clamp(pose_pinch, ctrl_min, ctrl_max)

    return poses


def generate_pose_dataset(
    output_path: str,
    num_easy: int = 10_000,
    num_hard: int = 20_000,
    seed: int = 42,
) -> dict:
    torch.manual_seed(int(seed))

    print("\n" + "=" * 80)
    print("Allegro Hand Task1 pose dataset generator")
    print("=" * 80)

    ctrl_min, ctrl_max = allegro_joint_limits()
    semantic_poses = build_semantic_poses(ctrl_min, ctrl_max)

    num_actions = int(ctrl_min.numel())
    pose_open = semantic_poses["open"]

    noise_easy = torch.randn(int(num_easy), num_actions) * 0.1 * (ctrl_max - ctrl_min)
    random_easy = torch.clamp(pose_open + noise_easy, ctrl_min, ctrl_max)

    random_hard = ctrl_min + (ctrl_max - ctrl_min) * torch.rand(int(num_hard), num_actions)

    semantic_tensor = torch.stack([semantic_poses[name] for name in semantic_poses.keys()], dim=0)

    dataset = {
        "ctrl_min": ctrl_min,
        "ctrl_max": ctrl_max,
        "semantic_poses": semantic_poses,
        "semantic_names": list(semantic_poses.keys()),
        "semantic_tensor": semantic_tensor,
        "random_easy": random_easy,
        "random_hard": random_hard,
        "joint_order": [
            "ffj0", "ffj1", "ffj2", "ffj3",
            "mfj0", "mfj1", "mfj2", "mfj3",
            "rfj0", "rfj1", "rfj2", "rfj3",
            "thj0", "thj1", "thj2", "thj3",
        ],
        "metadata": {
            "task": "allegro_task1_pose_tracking",
            "num_actions": num_actions,
            "num_easy": int(num_easy),
            "num_hard": int(num_hard),
            "seed": int(seed),
            "note": "Generated pure-RL target poses. No mocap data required for Task1 baseline.",
        },
    }

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, str(output))

    print(f"  num_actions      : {num_actions}")
    print(f"  semantic poses   : {list(semantic_poses.keys())}")
    print(f"  random_easy      : {tuple(random_easy.shape)}")
    print(f"  random_hard      : {tuple(random_hard.shape)}")
    print(f"  output           : {output}")
    print("=" * 80 + "\n")

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Allegro Hand Task1 target pose dataset")
    parser.add_argument(
        "--output",
        type=str,
        default=default_asset_path("motions/task1_target_poses.pt"),
        help="Output dataset path. Default: assets/motions/task1_target_poses.pt",
    )
    parser.add_argument("--num-easy", type=int, default=10_000)
    parser.add_argument("--num-hard", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_pose_dataset(
        output_path=args.output,
        num_easy=int(args.num_easy),
        num_hard=int(args.num_hard),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
