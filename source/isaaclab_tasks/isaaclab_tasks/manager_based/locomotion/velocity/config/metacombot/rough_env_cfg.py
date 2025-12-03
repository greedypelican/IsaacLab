# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.metacombot.mdp as metacombot_mdp

from isaaclab_assets.robots.metacombotx import METACOMBOTX_FIXED_CFG  # isort: skip

WHEEL_JOINT_EXPR = ".*_wheel_joint"


@configclass
class MetacombotSceneCfg(InteractiveSceneCfg):
    """간단한 바퀴 로봇 환경 씬."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        max_init_terrain_level=2,
        terrain_generator=ROUGH_TERRAINS_CFG,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = METACOMBOTX_FIXED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class MetacombotObservationsCfg:
    """관측 항목."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.005, n_max=0.005))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class MetacombotTerminationsCfg:
    """종료 조건."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class MetacombotEventsCfg:
    """이벤트(리셋 등)."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class MetacombotRewardsCfg:
    """MetaCombOTX용 보상."""

    track_lin_vel_xy_exp = RewTerm(
        func=metacombot_mdp.base_linear_velocity_reward,
        weight=7.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "std": 0.4},
    )
    track_ang_vel_z_exp = RewTerm(
        func=metacombot_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "std": 0.6},
    )
    # heading_alignment = RewTerm(
    #     func=metacombot_mdp.heading_alignment_reward,
    #     weight=2.5,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "base_velocity", "std": 0.2},
    # )
    lin_vel_z_l2 = RewTerm(func=metacombot_mdp.base_motion_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")})
    ang_vel_xy_l2 = RewTerm(
        func=metacombot_mdp.base_orientation_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # action_rate_l2 = RewTerm(func=metacombot_mdp.action_smoothness_penalty, weight=-0.02)


@configclass
class MetacombotActionsCfg:
    """휠 조인트만 제어."""

    wheel_pos = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[WHEEL_JOINT_EXPR],
        scale=10.0,
        use_default_offset=True,
    )


@configclass
class MetacombotCommandsCfg:
    """로봇 로컬 x축 기준 명령."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class MetacombotCurriculumCfg:
    """별도 커리큘럼 없음."""

    terrain_levels = None


@configclass
class MetacombotxVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """MetaCombOTX 휠 기반 velocity 트래킹 환경."""

    scene: MetacombotSceneCfg = MetacombotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: MetacombotObservationsCfg = MetacombotObservationsCfg()
    actions: MetacombotActionsCfg = MetacombotActionsCfg()
    commands: MetacombotCommandsCfg = MetacombotCommandsCfg()
    rewards: MetacombotRewardsCfg = MetacombotRewardsCfg()
    terminations: MetacombotTerminationsCfg = MetacombotTerminationsCfg()
    events: MetacombotEventsCfg = MetacombotEventsCfg()
    curriculum: MetacombotCurriculumCfg = MetacombotCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
            if "boxes" in self.scene.terrain.terrain_generator.sub_terrains:
                self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.01, 0.05)
            if "random_rough" in self.scene.terrain.terrain_generator.sub_terrains:
                self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.005, 0.03)
                self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.005


@configclass
class MetacombotxVelocityRoughEnvCfg_PLAY(MetacombotxVelocityRoughEnvCfg):
    """Play 설정."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 3.0
