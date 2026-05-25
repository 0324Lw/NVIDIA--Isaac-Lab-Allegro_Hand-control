from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

from allegro_rl.tasks.task2.task2_config import Task2Config


def make_allegro_task2_scene_cfg(cfg: Task2Config):
    """Factory for Allegro Hand Task2 scene config.

    A factory is used so that num_envs, env_spacing, cube size, sphere radius,
    and hand spawn height can be controlled by Task2Config.
    """

    @configclass
    class AllegroHandTask2SceneCfg(InteractiveSceneCfg):
        num_envs: int = int(cfg.num_envs)
        env_spacing: float = float(cfg.env_spacing)

        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
        )

        light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0),
        )

        robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(float(cfg.cube_size), float(cfg.cube_size), float(cfg.cube_size)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.10, 0.10)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, float(cfg.inactive_object_z))),
        )

        sphere = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Sphere",
            spawn=sim_utils.SphereCfg(
                radius=float(cfg.sphere_radius),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.10, 0.80)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, float(cfg.inactive_object_z))),
        )

        # Match all robot bodies first, then the environment code selects
        # the four actual fingertip bodies by name. This avoids missing
        # fingertip names such as index_link_3.
        fingertip_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=False,
            update_period=float(cfg.sim_dt),
        )

        def __post_init__(self):
            super().__post_init__()

            self.robot.spawn.activate_contact_sensors = True
            self.robot.spawn.fix_base = True
            self.robot.init_state.pos = (0.0, 0.0, float(cfg.hand_init_height))

    return AllegroHandTask2SceneCfg


AllegroHandTask2SceneCfgFactory = make_allegro_task2_scene_cfg
