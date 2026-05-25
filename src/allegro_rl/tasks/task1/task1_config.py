from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

from allegro_rl.common.paths import default_asset_path, path_from_env


@dataclass
class Task1Config:
    """Allegro Hand Task1 pose-tracking config.

    Task objective:
        Track generated target joint poses using pure RL.

    Project positioning:
        This is an educational pure-RL baseline. High-quality dexterous
        manipulation can benefit from mocap / demonstration data, but this
        basic pose-tracking task can be trained with generated targets.
    """

    # ----------------------------- Runtime -----------------------------
    num_envs: int = 4096
    device: str = "cuda:0"
    seed: int = 42

    # ----------------------------- Isaac / sim -----------------------------
    sim_dt: float = 0.005
    decimation: int = 4
    control_dt: float = 0.020
    max_episode_length: int = 100
    env_spacing: float = 1.5
    hand_init_height: float = 0.50

    # ----------------------------- Dataset -----------------------------
    dataset_path: str = field(
        default_factory=lambda: path_from_env(
            "ALLEGRO_TASK1_DATASET",
            default_asset_path("motions/task1_target_poses.pt"),
        )
    )

    # ----------------------------- Spaces -----------------------------
    num_actions: int = 16
    num_observations: int = 64
    frame_stack: int = 1
    stacked_obs_dim: int = 64

    # ----------------------------- Control -----------------------------
    ema_alpha: float = 0.80
    action_scale: float = 0.05

    # ----------------------------- Curriculum -----------------------------
    curriculum_total_steps: int = 150_000_000
    easy_prob_start: float = 0.80
    easy_prob_end: float = 0.15
    hard_prob_start: float = 0.15
    hard_prob_end: float = 0.65
    semantic_prob_start: float = 0.05
    semantic_prob_end: float = 0.20

    init_noise_scale: float = 0.15

    # ----------------------------- Reward weights -----------------------------
    w_track_mean: float = 0.45
    w_track_worst: float = 0.15
    track_sigma: float = 4.0

    w_progress: float = 4.0
    w_stable: float = 0.20
    stable_err_sigma: float = 10.0
    stable_vel_sigma: float = 5.0

    w_act_mag: float = 0.01
    w_act_smooth: float = 0.02
    w_soft_limit: float = 0.20
    limit_margin: float = 0.05
    soft_limit_clip: float = -0.50

    reward_clip_abs: float = 20.0

    # ----------------------------- Safety / debug -----------------------------
    obs_clip: float = 10.0
    print_debug_info: bool = False

    def validate(self) -> None:
        assert self.num_envs > 0
        assert self.device in ("cpu", "cuda", "cuda:0") or self.device.startswith("cuda")
        assert self.sim_dt > 0.0
        assert self.decimation >= 1
        assert self.control_dt > 0.0
        assert self.max_episode_length > 0

        assert self.num_actions == 16
        assert self.num_observations == 64
        assert self.stacked_obs_dim == self.num_observations * self.frame_stack

        assert 0.0 <= self.ema_alpha <= 1.0
        assert self.action_scale > 0.0

        assert self.curriculum_total_steps > 0
        assert 0.0 <= self.easy_prob_start <= 1.0
        assert 0.0 <= self.easy_prob_end <= 1.0
        assert 0.0 <= self.hard_prob_start <= 1.0
        assert 0.0 <= self.hard_prob_end <= 1.0
        assert 0.0 <= self.semantic_prob_start <= 1.0
        assert 0.0 <= self.semantic_prob_end <= 1.0

        assert self.track_sigma > 0.0
        assert self.stable_err_sigma > 0.0
        assert self.stable_vel_sigma > 0.0
        assert self.limit_margin >= 0.0


Task1ConfigAlias = Task1Config
