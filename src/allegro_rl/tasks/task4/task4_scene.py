from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

from allegro_rl.tasks.task4.task4_config import Task4Config


def make_allegro_task4_scene_cfg(cfg: Task4Config):
    """Factory for Allegro Hand Task4 scene config.

    Task4 keeps the same physical scene style as Task2: a fixed-base Allegro
    hand and two possible in-hand objects. The key difference is not the scene,
    but the Sim2Real actuator/observation/randomization model in task4_env.py.
    """

    @configclass
    class AllegroHandTask4SceneCfg(InteractiveSceneCfg):
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
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, float(cfg.inactive_object_z)),
            ),
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
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, float(cfg.inactive_object_z)),
            ),
        )

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

    return AllegroHandTask4SceneCfg


AllegroHandTask4SceneCfgFactory = make_allegro_task4_scene_cfg
