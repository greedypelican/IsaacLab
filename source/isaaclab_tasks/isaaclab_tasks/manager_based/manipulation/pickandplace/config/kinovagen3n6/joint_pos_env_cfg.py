# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

from . import mdp
from isaaclab_tasks.manager_based.manipulation.pickandplace.pickandplace_env_cfg import PickAndPlaceEnvCfg
from isaaclab_assets.robots.kinovagen3n6 import KINOVAGEN3N6_CFG


@configclass
class KinovaGen3N6PickAndPlaceEnvCfg(PickAndPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = KINOVAGEN3N6_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.ee_frame = FrameTransformerCfg(
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
        self.scene.contact_forces_arm = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/(arm_base_link|shoulder_link|bicep_link|forearm_link|spherical_wrist_1_link|spherical_wrist_2_link|bracelet_with_vision_link)",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )
        self.scene.contact_forces_left_finger_pad = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_finger_pad",
            update_period=0.0,
            history_length=1,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]
        )
        self.scene.contact_forces_right_finger_pad = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_finger_pad",
            update_period=0.0,
            history_length=1,
            debug_vis=True,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]
        )


        self.commands.ascend.body_name = "gripper_base_link"
        self.commands.descend.body_name = "gripper_base_link"


        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
            scale=0.5,
            use_default_offset=True
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


        # self.events.robot_physics_material.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["left_finger_pad", "right_finger_pad"])
        # self.events.robot_amount_of_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["arm_base_link", "shoulder_link", "bicep_link",
        #                                                                                            "forearm_link", "spherical_wrist_1_link", "spherical_wrist_2_link",
        #                                                                                            "bracelet_with_vision_link", "gripper_base_link"])
        # self.events.robot_center_of_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["arm_base_link", "shoulder_link", "bicep_link",
        #                                                                                            "forearm_link", "spherical_wrist_1_link", "spherical_wrist_2_link",
        #                                                                                            "bracelet_with_vision_link", "gripper_base_link"])


        self.rewards.goback.params["joint_names"] = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "left_outer_knuckle_joint"]
        self.rewards.ready.params["joint_names"] = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "left_outer_knuckle_joint"]
        self.rewards.ee_alignment_penalty.params["body_name"] = "gripper_base_link"


@configclass
class KinovaGen3N6PickAndPlaceEnvCfg_PLAY(KinovaGen3N6PickAndPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False