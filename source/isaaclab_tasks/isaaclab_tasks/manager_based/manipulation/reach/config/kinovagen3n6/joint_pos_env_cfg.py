# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from . import mdp as kinovagen3n6_mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg, RewardsCfg, CurriculumCfg
from isaaclab.assets.articulation import ArticulationCfg
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
class KinovaGen3N6EventCfg:
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset", 
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )
    def __post_init__(self):
        pass

@configclass
class KinovaGen3N6RewardsCfg(RewardsCfg):
    arm1_action_penalty = RewTerm(
        func=kinovagen3n6_mdp.action_rate_penalty, 
        params={"action_type": "arm1"}, 
        weight=-0.01, 
    )
    arm1_velocity_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_velocity_penalty, 
        params={"joint_type": "arm1"}, 
        weight=-0.001, 
    )
    arm1_acceleration_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_acceleration_penalty, 
        params={"joint_type": "arm1"}, 
        weight=-0.00001, 
    )
    arm2_action_penalty = RewTerm(
        func=kinovagen3n6_mdp.action_rate_penalty, 
        params={"action_type": "arm2"}, 
        weight=-0.005, 
    )
    arm2_velocity_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_velocity_penalty, 
        params={"joint_type": "arm2"}, 
        weight=-0.0005, 
    )
    arm2_acceleration_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_acceleration_penalty, 
        params={"joint_type": "arm2"}, 
        weight=-0.000005, 
    )
    gripper_action_penalty = RewTerm(
        func=kinovagen3n6_mdp.action_rate_penalty, 
        params={"action_type": "gripper"}, 
        weight=-0.005, 
    )
    gripper_velocity_penalty = RewTerm(
        func=kinovagen3n6_mdp.joint_velocity_penalty, 
        params={"joint_type": "gripper"}, 
        weight=-0.00005, 
    )

    def __post_init__(self):
        self.end_effector_position_tracking_fine_grained.weight = 1.0
        self.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper_base_link"]
        self.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["gripper_base_link"]
        # self.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["gripper_base_link"]
        self.end_effector_orientation_tracking = None
        self.action_rate = None
        self.joint_vel = None
        # self.action_rate.weight = -0.001
        # self.joint_vel.weight = -0.0001
        # self.joint_vel.params["asset_cfg"].joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

@configclass
class KinovaGen3N6CurriculumCfg(CurriculumCfg):
    """Curriculum terms for the MDP."""

    # Keep curriculum gentle (no further increase in penalty magnitudes)
    arm_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "arm1_action_penalty", "weight": -0.02, "num_steps": 10000}
    )
    arm_velocity_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "arm1_velocity_penalty", "weight": -0.002, "num_steps": 10000}
    )
    arm_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "arm2_action_penalty", "weight": -0.01, "num_steps": 10000}
    )
    arm_velocity_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "arm2_velocity_penalty", "weight": -0.001, "num_steps": 10000}
    )
    gripper_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "gripper_action_penalty", "weight": -0.01, "num_steps": 10000}
    )
    gripper_velocity_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "gripper_velocity_penalty", "weight": -0.0001, "num_steps": 10000}
    )

    def __post_init__(self):
        self.action_rate = None
        self.joint_vel = None
        # self.action_rate.params["weight"] = -0.005


@configclass
class KinovaGen3N6ReachEnvCfg(ReachEnvCfg):
    rewards: KinovaGen3N6RewardsCfg = KinovaGen3N6RewardsCfg()
    curriculum: KinovaGen3N6CurriculumCfg = KinovaGen3N6CurriculumCfg()
    events: KinovaGen3N6EventCfg = KinovaGen3N6EventCfg()

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
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
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
