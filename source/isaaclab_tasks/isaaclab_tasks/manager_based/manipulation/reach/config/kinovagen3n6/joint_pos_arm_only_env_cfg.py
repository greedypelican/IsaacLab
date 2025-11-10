# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Kinova Gen3 N6 6DOF arm-only reach task (no gripper)."""

import math
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
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

from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg, ReachSceneCfg, EventCfg, ObservationsCfg, RewardsCfg, TerminationsCfg, CurriculumCfg
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from . import mdp as kinovagen3n6_mdp
from isaaclab_assets.robots.kinovagen3n6 import KINOVAGEN3N6_REACH_CFG

EPISODE_LENGTH_SEC = 16.0
DECIMATION = 3
FREQUENCY = 100
DELTA_TIME = 1 / (DECIMATION * FREQUENCY)



@configclass
class KinovaGen3N6ArmOnlySceneCfg(ReachSceneCfg):

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
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.21, 0.0, -0.49), rot=(1.0, 0.0, 0.0, 0.0)),
        )
        self.robot = KINOVAGEN3N6_REACH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class KinovaGen3N6ArmOnlyEventCfg(EventCfg):
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.9, 0.9, 0.9], [1.1, 1.1, 1.1]),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )
    robot_amount_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.75, 1.25),
            "operation": "scale",
        },
    )
    robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_range": {"x": (-0.015, 0.015), "y": (-0.015, 0.015), "z": (-0.015, 0.015)},
        },
    )

    def __post_init__(self):
        pass


@configclass
class KinovaGen3N6ArmOnlyObservationsCfg(ObservationsCfg):
    """Observation configuration for 6DOF arm-only reach task."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        gripper_tip_pose = ObsTerm(
            func=kinovagen3n6_mdp.gripper_tip_pose,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["gripper_tip"])},
        )

        def __post_init__(self):
            super().__post_init__()

    policy: PolicyCfg = PolicyCfg()


@configclass
class KinovaGen3N6ArmOnlyRewardsCfg(RewardsCfg):
    """Reward configuration for 6DOF arm-only reach task."""

    end_effector_position_tracking_negative = RewTerm(
        func=mdp.position_command_error,
        weight=-0.6, #
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["gripper_tip"]), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_positive = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["gripper_tip"]), "std": 0.3, "command_name": "ee_pose"},
    )
    end_effector_position_tracking_positive_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["gripper_tip"]), "std": 0.06, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking_negative = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["gripper_tip"]), "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking_positive = RewTerm(
        func=kinovagen3n6_mdp.orientation_command_error_tanh,
        weight=7.0, #
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["gripper_tip"]), "std": 0.45, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking_positive_fine = RewTerm(
        func=kinovagen3n6_mdp.orientation_command_error_tanh,
        weight=3.6, #
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["gripper_tip"]), "std": 0.09, "command_name": "ee_pose"},
    )

    end_effector_pose_smoothness_penalty = RewTerm(
        func=kinovagen3n6_mdp.end_effector_pose_movement_penalty,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_tip"]),
            "lin_weight": 2.0,
            "ang_weight": 0.05,
        },
        weight=-0.15,
    )
    end_effector_pose_displacement_penalty = RewTerm(
        func=kinovagen3n6_mdp.end_effector_pose_displacement_penalty,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["gripper_tip"]),
            "pos_weight": 200.0,
            "rot_weight": 5.0,
        },
        weight=-0.1, #
    )
    arm1_action_penalty = RewTerm(
        func=kinovagen3n6_mdp.action_rate_penalty, 
        params={"action_type": "arm1"}, 
        weight=-0.05, 
    )
    arm1_velocity_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_velocity_penalty, 
        params={"joint_type": "arm1"}, 
        weight=-0.05,  #
    )
    arm1_acceleration_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_acceleration_penalty, 
        params={"joint_type": "arm1"}, 
        weight=-0.0, 
    )
    arm2_action_penalty = RewTerm(
        func=kinovagen3n6_mdp.action_rate_penalty, 
        params={"action_type": "arm2"}, 
        weight=-0.04, #
    )
    arm2_velocity_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_velocity_penalty, 
        params={"joint_type": "arm2"}, 
        weight=-0.04, #
    )
    arm2_acceleration_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_acceleration_penalty, 
        params={"joint_type": "arm2"}, 
        weight=-0.0, 
    )

    def __post_init__(self):
        self.end_effector_position_tracking = None
        self.end_effector_position_tracking_fine_grained = None
        self.end_effector_orientation_tracking = None

        self.action_rate = None
        self.joint_vel = None


@configclass
class KinovaGen3N6ArmOnlyTerminationsCfg(TerminationsCfg):
    """Termination configuration for 6DOF arm-only reach task."""

    arm_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_arm"), "threshold": 1.0},
    )
    # outer_gripper_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces_outer_gripper"), "threshold": 1.0},
    # )
    # inner_gripper_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces_inner_gripper"), "threshold": 1.0},
    # )


@configclass
class KinovaGen3N6ArmOnlyCurriculumCfg(CurriculumCfg):
    """Curriculum terms for 6DOF arm-only reach task."""

    end_effector_pose_smoothness_penalty = CurrTerm(
        func=kinovagen3n6_mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "end_effector_pose_smoothness_penalty",
            "weight_1": -0.3, #
            "num_steps_1": 25000,
            "weight_2": -0.6, #
            "num_steps_2": 75000,
            }
    )
    end_effector_pose_displacement_penalty = CurrTerm(
        func=kinovagen3n6_mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "end_effector_pose_displacement_penalty",
            "weight_1": -0.15, #
            "num_steps_1": 25000,
            "weight_2": -0.2, #
            "num_steps_2": 75000,
            }
    )

    arm1_action_penalty = CurrTerm(
        func=kinovagen3n6_mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "arm1_action_penalty",
            "weight_1": -0.35, #
            "num_steps_1": 25000,
            "weight_2": -2.5, #
            "num_steps_2": 75000,
            }
    )
    arm1_velocity_penalty = CurrTerm(
        func=kinovagen3n6_mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "arm1_velocity_penalty",
            "weight_1": -0.158, #
            "num_steps_1": 25000,
            "weight_2": -0.5, #
            "num_steps_2": 75000,
            }
    )
    arm1_acceleration_penalty = CurrTerm(
        func=kinovagen3n6_mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "arm1_acceleration_penalty",
            "weight_1": -0.0004, #
            "num_steps_1": 50000,
            "weight_2": -0.0008, #
            "num_steps_2": 100000,
            }
    )
    arm2_action_penalty = CurrTerm(
        func=kinovagen3n6_mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "arm2_action_penalty",
            "weight_1": -0.28, #
            "num_steps_1": 25000,
            "weight_2": -2.0, #
            "num_steps_2": 75000,
            }
    )
    arm2_velocity_penalty = CurrTerm(
        func=kinovagen3n6_mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "arm2_velocity_penalty",
            "weight_1": -0.127, #
            "num_steps_1": 25000,
            "weight_2": -0.4, #
            "num_steps_2": 75000,
            }
    )
    arm2_acceleration_penalty = CurrTerm(
        func=kinovagen3n6_mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "arm2_acceleration_penalty",
            "weight_1": -0.0003, #
            "num_steps_1": 50000,
            "weight_2": -0.0006, #
            "num_steps_2": 100000,
            }
    )
    # arm1_action_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "arm1_action_penalty", "weight": -0.6, "num_steps": 40000}
    # )
    # arm1_velocity_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "arm1_velocity_penalty", "weight": -0.15, "num_steps": 40000}
    # )
    # arm1_acceleration_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "arm1_acceleration_penalty", "weight": -0.0002, "num_steps": 60000}
    # )
    # arm2_action_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "arm2_action_penalty", "weight": -0.3, "num_steps": 40000}
    # )
    # arm2_velocity_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "arm2_velocity_penalty", "weight": -0.06, "num_steps": 40000}
    # )
    # arm2_acceleration_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "arm2_acceleration_penalty", "weight": -0.0001, "num_steps": 60000}
    # )

    def __post_init__(self):
        self.action_rate = None
        self.joint_vel = None



@configclass
class KinovaGen3N6ArmOnlyReachEnvCfg(ReachEnvCfg):
    """Configuration for Kinova Gen3 N6 6DOF arm-only reach task."""

    scene: KinovaGen3N6ArmOnlySceneCfg = KinovaGen3N6ArmOnlySceneCfg(num_envs=4096, env_spacing=2.5)
    events: KinovaGen3N6ArmOnlyEventCfg = KinovaGen3N6ArmOnlyEventCfg()
    observations: KinovaGen3N6ArmOnlyObservationsCfg = KinovaGen3N6ArmOnlyObservationsCfg()
    rewards: KinovaGen3N6ArmOnlyRewardsCfg = KinovaGen3N6ArmOnlyRewardsCfg()
    terminations: KinovaGen3N6ArmOnlyTerminationsCfg = KinovaGen3N6ArmOnlyTerminationsCfg()
    curriculum: KinovaGen3N6ArmOnlyCurriculumCfg = KinovaGen3N6ArmOnlyCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Simulation settings
        self.episode_length_s = EPISODE_LENGTH_SEC
        self.decimation = DECIMATION
        self.sim.dt = DELTA_TIME
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.5, 3.5, 3.5)


        base_ee_pose_cfg = self.commands.ee_pose
        self.commands.ee_pose = kinovagen3n6_mdp.DonutPoseCommandCfg(
            asset_name="robot",
            body_name="gripper_tip",
            resampling_time_range=(4.0, 4.0),
            debug_vis=base_ee_pose_cfg.debug_vis,
            make_quat_unique=base_ee_pose_cfg.make_quat_unique,
            goal_pose_visualizer_cfg=base_ee_pose_cfg.goal_pose_visualizer_cfg,
            current_pose_visualizer_cfg=base_ee_pose_cfg.current_pose_visualizer_cfg,
            ranges=kinovagen3n6_mdp.DonutPoseCommandCfg.Ranges(
                pos_x=(-0.04, 0.46),
                pos_y=(-0.5, 0.5),
                pos_z=(-0.05, 0.25),
                roll=(0.0, 0.0),
                pitch=(math.pi, math.pi),
                yaw=(-math.pi, 0.0),
            ),
            exclusion_pos_x=(-0.04, 0.2),
            exclusion_pos_y=(-0.2, 0.2),
            max_position_delta=0.5,
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            scale=0.5,
            use_default_offset=True
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

        # Disable curriculum for play mode
        self.curriculum = None
        self.observations.policy.enable_corruption = False

        # Better camera view for observation
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
