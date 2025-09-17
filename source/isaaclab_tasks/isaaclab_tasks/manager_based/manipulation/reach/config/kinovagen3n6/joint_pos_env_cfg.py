# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

##
# Pre-defined configs
##
from isaaclab_assets.robots.kinovagen3n6 import KINOVAGEN3N6_REACH_CFG  # isort: skip


##
# Environment configuration
##

@configclass
class KinovaGen3N6EventCfg(EventCfg):
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
        # self.reset_robot_joints = None
        pass

@configclass
class KinovaGen3N6RewardsCfg(RewardsCfg):
    arm_action_penalty = RewTerm(
        func=kinovagen3n6_mdp.action_rate_penalty, 
        params={"action_type": "arm_actions"}, 
        weight=-0.1, 
    )
    arm_velocity_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_velocity_penalty, 
        params={"joint_type": "arm_joints"}, 
        weight=-0.1, 
    )
    gripper_action_penalty = RewTerm(
        func=kinovagen3n6_mdp.action_rate_penalty, 
        params={"action_type": "gripper_actions"}, 
        weight=-0.05, 
    )
    gripper_velocity_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_velocity_penalty, 
        params={"joint_type": "gripper_joints"}, 
        weight=-0.05, 
    )

    def __post_init__(self):
        self.end_effector_position_tracking_fine_grained.weight = 3.0
        self.end_effector_position_tracking_fine_grained.params["std"] = 0.3
        self.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_base_link"]
        self.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_base_link"]
        self.end_effector_orientation_tracking = None
        self.action_rate = None
        self.joint_vel = None

@configclass
class KinovaGen3N6CurriculumCfg(CurriculumCfg):
    """Curriculum terms for the MDP."""

    # Keep curriculum gentle (no further increase in penalty magnitudes)
    arm_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "arm_action_penalty", "weight": -0.3, "num_steps": 8000}
    )
    arm_velocity_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "arm_velocity_penalty", "weight": -0.3, "num_steps": 8000}
    )
    gripper_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "gripper_action_penalty", "weight": -0.15, "num_steps": 8000}
    )
    gripper_velocity_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "gripper_velocity_penalty", "weight": -0.15, "num_steps": 8000}
    )

    def __post_init__(self):
        self.action_rate = None
        self.joint_vel = None


@configclass
class KinovaGen3N6ReachEnvCfg(ReachEnvCfg):
    events: KinovaGen3N6EventCfg = KinovaGen3N6EventCfg()
    rewards: KinovaGen3N6RewardsCfg = KinovaGen3N6RewardsCfg()
    curriculum: KinovaGen3N6CurriculumCfg = KinovaGen3N6CurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.decimation = 3
        self.sim.render_interval = self.decimation
        self.episode_length_s = 8.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 150.0

        # switch robot to Kinova Gen3 6dof
        self.scene.robot = KINOVAGEN3N6_REACH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # limit velocity penalty to arm joints only (exclude gripper/mimic joints)

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_.*"], scale=0.5, use_default_offset=False
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_inner_finger_joint", "right_inner_finger_joint", 
                         "left_outer_finger_joint", "right_outer_finger_joint", 
                         "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint", 
                         "left_outer_knuckle_joint", "right_outer_knuckle_joint"], 
            open_command_expr={"left_inner_finger_joint": 0.0, "right_inner_finger_joint": 0.0, 
                                "left_outer_finger_joint": 0.0, "right_outer_finger_joint": 0.0, 
                                "left_inner_finger_knuckle_joint": 0.0, "right_inner_finger_knuckle_joint": 0.0, 
                                "left_outer_knuckle_joint": 0.0, "right_outer_knuckle_joint": 0.0}, 
            close_command_expr={"left_inner_finger_joint": -math.pi/4, "right_inner_finger_joint": math.pi/4, 
                                "left_outer_finger_joint": 0.0, "right_outer_finger_joint": 0.0, 
                                "left_inner_finger_knuckle_joint": -math.pi/4, "right_inner_finger_knuckle_joint": -math.pi/4, 
                                "left_outer_knuckle_joint": math.pi/4, "right_outer_knuckle_joint": math.pi/4}, 
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "gripper_base_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
        self.commands.ee_pose.ranges.pos_x = (0.25, 0.55)
        self.commands.ee_pose.resampling_time_range = (8.0, 8.0)


@configclass
class KinovaGen3N6ReachEnvCfg_PLAY(KinovaGen3N6ReachEnvCfg):
    """Configuration for playing/demonstration of Kinova Gen3 N6 reach task."""

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()

        # Reduce number of environments for visualization
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5

        # Increase episode length for demonstration
        self.episode_length_s = 30.0

        # Fixed target for testing
        self.commands.ee_pose.ranges.pos_x = (0.4, 0.4)  # Fixed position
        self.commands.ee_pose.ranges.pos_y = (0.0, 0.0)  # Centered
        self.commands.ee_pose.ranges.pos_z = (0.3, 0.3)  # Fixed height
        self.commands.ee_pose.resampling_time_range = (30.0, 30.0)  # No resampling

        # Disable curriculum for play mode
        self.curriculum = None

        # Better camera view for observation
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
