from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

from allegro_rl.tasks.task3.task3_config import Task3Config


def make_allegro_task3_scene_cfg(cfg: Task3Config):
    """Factory for Allegro Hand Task3 scene config."""

    @configclass
    class AllegroHandTask3SceneCfg(InteractiveSceneCfg):
        num_envs: int = int(cfg.num_envs)
        env_spacing: float = float(cfg.env_spacing)

        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )

        table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.CuboidCfg(
                size=(
                    float(cfg.table_size_xy),
                    float(cfg.table_size_xy),
                    float(cfg.table_thickness),
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.35)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(
                    0.0,
                    0.0,
                    float(cfg.table_height) - float(cfg.table_thickness) * 0.5,
                )
            ),
        )

        robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        pen = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Pen",
            spawn=sim_utils.CuboidCfg(
                size=(
                    float(cfg.pen_size[0]),
                    float(cfg.pen_size[1]),
                    float(cfg.pen_size[2]),
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.10, 0.90)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, float(cfg.inactive_object_z)),
            ),
        )

        cup = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cup",
            spawn=sim_utils.CuboidCfg(
                size=(
                    float(cfg.cup_size[0]),
                    float(cfg.cup_size[1]),
                    float(cfg.cup_size[2]),
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.16),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.20, 0.10)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, float(cfg.inactive_object_z)),
            ),
        )

        hand_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=False,
            update_period=float(cfg.sim_dt),
        )

        def __post_init__(self):
            super().__post_init__()

            self.robot.spawn.activate_contact_sensors = True

            # The hand base is fixed by PhysX articulation, but the environment
            # controls the root pose explicitly to emulate a floating base.
            self.robot.spawn.fix_base = True
            self.robot.init_state.pos = (0.0, 0.0, float(cfg.base_init_z))

    return AllegroHandTask3SceneCfg


AllegroHandTask3SceneCfgFactory = make_allegro_task3_scene_cfg
