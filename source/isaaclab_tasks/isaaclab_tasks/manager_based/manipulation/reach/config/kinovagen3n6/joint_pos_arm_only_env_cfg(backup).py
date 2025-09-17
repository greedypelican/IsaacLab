# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Kinova Gen3 N6 6DOF arm-only reach task (no gripper)."""

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from . import mdp as kinovagen3n6_mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg, EventCfg, RewardsCfg, CurriculumCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from isaaclab_assets.robots.kinovagen3n6 import KINOVAGEN3N6_REACH_CFG  # isort: skip


##
# 6DOF Arm-Only Observations Configuration
##

@configclass
class ArmOnlyPolicyCfg(ObsGroup):
    """Observations for 6DOF arm-only policy group."""

    # 6DOF arm joints only
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"])}
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"])}
    )

    # 7DOF pose command (correct)
    pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})

    # 6DOF last action
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


##
# Environment configuration
##

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

    # Use standard action rate penalty for 6DOF actions
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # Use standard joint velocity penalty with much smaller weight
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"])},
    )

    def __post_init__(self):
        # Enhanced position tracking for arm-only task
        self.end_effector_position_tracking_fine_grained.weight = 5.0  # Increased from 3.0
        self.end_effector_position_tracking_fine_grained.params["std"] = 0.2  # Tighter tolerance

        # Set end-effector body names for Kinova Gen3 N6
        self.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_base_link"]
        self.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_base_link"]

        # Enable orientation tracking for 6DOF arm-only task
        self.end_effector_orientation_tracking.weight = -1.0  # Negative weight to penalize orientation error
        self.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_base_link"]

        # Keep the standard action and velocity penalties (they now work with 6DOF)

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

    events: KinovaGen3N6ArmOnlyEventCfg = KinovaGen3N6ArmOnlyEventCfg()
    rewards: KinovaGen3N6ArmOnlyRewardsCfg = KinovaGen3N6ArmOnlyRewardsCfg()
    curriculum = None  # Disable curriculum completely

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Simulation settings
        self.decimation = 3
        self.sim.render_interval = self.decimation
        self.episode_length_s = 8.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 150.0

        # Switch robot to Kinova Gen3 6DOF
        self.scene.robot = KINOVAGEN3N6_REACH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ========================================================================
        # ARM-ONLY ACTIONS (6DOF)
        # ========================================================================
        # Override actions - ONLY arm action (6 joints)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],  # Explicit 6 joints
            scale=0.5,
            use_default_offset=False
        )

        # REMOVE gripper action completely
        self.actions.gripper_action = None

        # ========================================================================
        # OBSERVATIONS (6DOF arm joints only)
        # ========================================================================
        # Replace the default observations with arm-only observations
        self.observations.policy = ArmOnlyPolicyCfg()

        # ========================================================================
        # COMMANDS (7DOF pose - correct!)
        # ========================================================================
        # End-effector commands remain 7DOF (position + orientation)
        self.commands.ee_pose.body_name = "gripper_base_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
        self.commands.ee_pose.ranges.pos_x = (0.25, 0.55)
        self.commands.ee_pose.resampling_time_range = (8.0, 8.0)


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