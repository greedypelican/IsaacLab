# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Kinova Gen3 N6 6DOF arm-only reach task (no gripper)."""

import math
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg, GroundPlaneCfg
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg, ReachSceneCfg, EventCfg, RewardsCfg, TerminationsCfg, CurriculumCfg
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from . import mdp as kinovagen3n6_mdp
from isaaclab_assets.robots.kinovagen3n6 import KINOVAGEN3N6_REACH_CFG

EPISODE_LENGTH_SEC = 15.0
DECIMATION = 3
FREQUENCY = 50
DELTA_TIME = 1 / (DECIMATION * FREQUENCY)

MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "cuboid": sim_utils.CuboidCfg(
            size=(0.02, 0.02, 0.02),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        ),
    }
)


@configclass
class KinovaGen3N6ArmOnlySceneCfg(ReachSceneCfg):
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_base_link",
        debug_vis=False,
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path = "/Visuals/FrameTransformer",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
                ),
                "connecting_line": sim_utils.CylinderCfg(
                    radius=0.002,
                    height=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), roughness=1.0),
                ),
            }
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_base_link",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.15, 0.0, 0.0],
                ),
            ),
        ],
    )
    # object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0.25, 0.0, 0.0], rot=[0.70711, 0, 0.70711, 0]),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         scale=(0.8, 0.8, 0.8),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #     ),
    # )
    contact_forces_arm = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(arm_base_link|shoulder_link|bicep_link|forearm_link|spherical_wrist_1_link|spherical_wrist_2_link|bracelet_with_vision_link)",
        update_period=DELTA_TIME,
        history_length=DECIMATION,
        debug_vis=False,
    )
    contact_forces_outer_gripper = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(left_outer_finger|right_outer_finger|left_inner_finger|right_inner_finger)",
        update_period=DELTA_TIME,
        history_length=DECIMATION,
        debug_vis=False,
    )
    contact_forces_inner_gripper = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(gripper_base_link|left_inner_knuckle|right_inner_knuckle|left_finger_pad|right_finger_pad)",
        update_period=DELTA_TIME,
        history_length=DECIMATION,
        debug_vis=False,
    )

    def __post_init__(self):
        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            spawn=sim_utils.UsdFileCfg(usd_path="../Assets/pseudo_table/pseudo_table.usd"),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.0, -0.415), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.robot = KINOVAGEN3N6_REACH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class KinovaGen3N6ArmOnlyEventCfg(EventCfg):
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    def __post_init__(self):
        pass


@configclass
class KinovaGen3N6ArmOnlyRewardsCfg(RewardsCfg):
    """Reward configuration for 6DOF arm-only reach task."""

    end_effector_orientation_tracking_fine_grained = RewTerm(
        func=kinovagen3n6_mdp.orientation_command_error_tanh,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["gripper_base_link"]), "std": 0.1, "command_name": "ee_pose"},
    )
    action_penaly = RewTerm(
        func=kinovagen3n6_mdp.action_rate_penalty,
        weight=-0.05,
    )
    velocity_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_velocity_penalty,
        weight=-0.02,
    )

    def __post_init__(self):
        self.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_base_link"]

        self.end_effector_position_tracking_fine_grained.weight = 5.0
        self.end_effector_position_tracking_fine_grained.params["std"] = 0.2
        self.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_base_link"]

        self.end_effector_orientation_tracking.weight = -0.1
        self.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_base_link"]

        self.action_rate = None
        self.joint_vel = None


@configclass
class KinovaGen3N6ArmOnlyTerminationsCfg(TerminationsCfg):
    """Termination configuration for 6DOF arm-only reach task."""

    arm_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_arm"), "threshold": 1.0},
    )
    outer_gripper_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_outer_gripper"), "threshold": 1.0},
    )
    inner_gripper_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_inner_gripper"), "threshold": 1.0},
    )


@configclass
class KinovaGen3N6ArmOnlyCurriculumCfg(CurriculumCfg):
    """Curriculum terms for 6DOF arm-only reach task."""

    # Disable curriculum completely - penalties are unstable
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "action_rate", "weight": -0.01, "num_steps": 20000}
    # )
    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "joint_vel", "weight": -0.0001, "num_steps": 30000}
    # )

    def __post_init__(self):
        pass  # Curriculum disabled



@configclass
class KinovaGen3N6ArmOnlyReachEnvCfg(ReachEnvCfg):
    """Configuration for Kinova Gen3 N6 6DOF arm-only reach task."""

    scene: KinovaGen3N6ArmOnlySceneCfg = KinovaGen3N6ArmOnlySceneCfg(num_envs=4096, env_spacing=2.5)
    events: KinovaGen3N6ArmOnlyEventCfg = KinovaGen3N6ArmOnlyEventCfg()
    rewards: KinovaGen3N6ArmOnlyRewardsCfg = KinovaGen3N6ArmOnlyRewardsCfg()
    terminations: KinovaGen3N6ArmOnlyTerminationsCfg = KinovaGen3N6ArmOnlyTerminationsCfg()
    curriculum = None  # Disable curriculum completely

    def __post_init__(self):
        super().__post_init__()

        # Simulation settings
        self.episode_length_s = EPISODE_LENGTH_SEC
        self.decimation = DECIMATION
        self.sim.dt = DELTA_TIME
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 3.5)


        self.commands.ee_pose.body_name = "gripper_base_link"
        self.commands.ee_pose.resampling_time_range = (5.0, 5.0)
        self.commands.ee_pose.ranges.pos_x = (0.15, 0.45)
        self.commands.ee_pose.ranges.pos_y = (-0.55, 0.55)
        self.commands.ee_pose.ranges.pos_z = (0.2, 0.4)
        self.commands.ee_pose.ranges.roll = (math.pi/2, math.pi/2)
        self.commands.ee_pose.ranges.pitch = (math.pi/2, math.pi/2)
        self.commands.ee_pose.ranges.yaw = (0.0, math.pi)
        # self.commands.ee_pose.goal_pose_visualizer_cfg = MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
        # self.commands.ee_pose.goal_pose_visualizer_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.1, 0.0))


        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            scale=0.5,
            use_default_offset=False
        )
        self.actions.gripper_action = None


@configclass
class KinovaGen3N6ArmOnlyReachEnvCfg_PLAY(KinovaGen3N6ArmOnlyReachEnvCfg):
    """Play configuration for demonstrating 6DOF arm-only reach task."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Reduce number of environments for visualization
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # Increase episode length for demonstration
        self.episode_length_s = 30.0

        # Fixed target for testing arm-only reaching
        self.commands.ee_pose.ranges.pos_x = (0.4, 0.4)  # Fixed position
        self.commands.ee_pose.ranges.pos_y = (0.0, 0.0)  # Centered
        self.commands.ee_pose.ranges.pos_z = (0.3, 0.3)  # Fixed height
        self.commands.ee_pose.resampling_time_range = (30.0, 30.0)  # No resampling

        # Disable curriculum for play mode
        self.curriculum = None

        # Better camera view for observation
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class KinovaGen3N6ArmOnlyReachEnvCfg_DEMO(KinovaGen3N6ArmOnlyReachEnvCfg):
    """Demo configuration for showcasing 6DOF arm-only capabilities."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Single environment for focused demonstration
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0

        # Extended episode length for comprehensive demonstration
        self.episode_length_s = 45.0

        # Diverse targets to show full 6DOF arm workspace
        self.commands.ee_pose.ranges.pos_x = (0.2, 0.6)   # Extended reach
        self.commands.ee_pose.ranges.pos_y = (-0.3, 0.3)  # Side-to-side movement
        self.commands.ee_pose.ranges.pos_z = (0.1, 0.5)   # Vertical range
        self.commands.ee_pose.resampling_time_range = (12.0, 18.0)  # Faster target changes

        # Disable curriculum for demonstration
        self.curriculum = None

        # Optimal camera angle for 6DOF arm demonstration
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.4, 0.0, 0.3)  # Focus on workspace center