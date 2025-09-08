# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import math

from numpy import argmin
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp


EPISODE_LENGTH_SEC = 10.0
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


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.013], rot=[0.70711, 0, 0.70711, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            # usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[1.0, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -0.79]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    contact_forces_arm = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(arm_base_link|shoulder_link|bicep_link|forearm_link|spherical_wrist_1_link|spherical_wrist_2_link|bracelet_with_vision_link)", 
        update_period=0.0, 
        history_length=1, 
        debug_vis=False, 
    )
    contact_forces_outer_gripper = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(left_inner_finger|left_outer_knuckle|left_outer_finger|right_inner_finger|right_outer_knuckle|right_outer_finger)", 
        update_period=0.0, 
        history_length=1, 
        debug_vis=False, 
    )
    contact_forces_inner_gripper = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(left_inner_knuckle|right_inner_knuckle|gripper_base_link)", 
        update_period=0.0, 
        history_length=1, 
        debug_vis=False, 
    )
    contact_forces_left_finger_pad = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_finger_pad", 
        update_period=0.0, 
        history_length=1, 
        debug_vis=True, 
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]
    )
    contact_forces_right_finger_pad = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger_pad", 
        update_period=0.0, 
        history_length=1, 
        debug_vis=True, 
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]
    )


##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ascend = mdp.UniformPoseCommandCfg(
        asset_name="robot", 
        body_name=MISSING, 
        resampling_time_range=(EPISODE_LENGTH_SEC, EPISODE_LENGTH_SEC), 
        debug_vis=True, 
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5), pos_y=(-0.2, 0.2), pos_z=(0.3, 0.3), roll=(0.0, 0.0), pitch=(math.pi/2, math.pi/2), yaw=(0.0, 0.0)  # radian
        ), 
    )
    descend = mdp.UniformPoseCommandCfg(
        asset_name="robot", 
        body_name=MISSING, 
        resampling_time_range=(EPISODE_LENGTH_SEC, EPISODE_LENGTH_SEC), 
        debug_vis=True, 
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5), pos_y=(-0.2, 0.2), pos_z=(0.013, 0.013), roll=(0.0, 0.0), pitch=(math.pi/2, math.pi/2), yaw=(0.0, 0.0)  # radian
        ), 
    )

    def __post_init__(self):
        self.ascend.goal_pose_visualizer_cfg = MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
        self.ascend.goal_pose_visualizer_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.1, 0.0))
        self.descend.goal_pose_visualizer_cfg = MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
        self.descend.goal_pose_visualizer_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 0.6))


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        lift_target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "ascend"})
        place_target_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "descend"})
        current_phase = ObsTerm(func=mdp.current_phase)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(
        func=mdp.reset_scene_to_default, 
        mode="reset", 
    )
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform, 
        mode="reset", 
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)}, 
            "velocity_range": {}, 
            "asset_cfg": SceneEntityCfg("object", body_names="Object"), 
        },
    )
    reset_phase_flags = EventTerm(
        func=mdp.reset_phase_flags, 
        mode="reset", 
    )
    check_and_update_phase_flags = EventTerm(
        func=mdp.check_and_update_phase_flags, 
        mode="interval", 
        interval_range_s=(DELTA_TIME, DELTA_TIME), 
        is_global_time=False, 
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    # Phase-based basic rewards for task progression
    phase = RewTerm(
        func=mdp.phase_complete, 
        weight=1.0, 
    )
    
    distance = RewTerm(
        func=mdp.ee_distance, 
        weight=1.0, 
    )
    contact = RewTerm(
        func=mdp.object_contact, 
        weight=2.0, 
    )
    height = RewTerm(
        func=mdp.object_height, 
        weight=3.0, 
    )
    track = RewTerm(
        func=mdp.object_track, 
        weight=5.0, 
    )
    place = RewTerm(
        func=mdp.object_place, 
        weight=15.0, 
    )
    goback = RewTerm(
        func=mdp.initial_pose_penalty, 
        weight=-1.0, 
    )
    
    ee_alignment_penalty = RewTerm(
        func=mdp.world_ee_z_axis_alignment_penalty, 
        params=MISSING, 
        weight=-1.0, 
    )
    arm_action_penalty = RewTerm(
        func=mdp.action_rate_penalty, 
        params={"action_type": "arm_actions"}, 
        weight=-0.07, 
    )
    arm_velocity_penalty = RewTerm(
        func=mdp.joint_velocity_penalty, 
        params={"joint_type": "arm_joints"}, 
        weight=-0.07, 
    )    
    gripper_action_penalty = RewTerm(
        func=mdp.action_rate_penalty, 
        params={"action_type": "gripper_actions"}, 
        weight=-0.0, 
    )
    gripper_velocity_penalty = RewTerm(
        func=mdp.joint_velocity_penalty, 
        params={"joint_type": "gripper_joints"}, 
        weight=-0.0, 
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.time_out_with_phase_logging, 
        time_out=True, 
    )

    object_dropping = DoneTerm(
        func=mdp.object_drop, 
        params={"object_cfg": SceneEntityCfg("object"), "height_threshold": 0.05, "contact_threshold": 0.01}, 
    )
    object_out_of_bounds = DoneTerm(
        func=mdp.object_out_of_bounds, 
        params={"object_cfg": SceneEntityCfg("object")}, 
    )
    arm_contact = DoneTerm(
        func=mdp.illegal_contact, 
        params={"sensor_cfg": SceneEntityCfg("contact_forces_arm", 
                                  body_names=["arm_base_link", "shoulder_link", "bicep_link", 
                                      "forearm_link", "spherical_wrist_.*", "bracelet_with_vision_link"]), 
                "threshold": 1.0}, 
    )
    object_contact = DoneTerm(
        func=mdp.gripper_contact_after_place, 
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


    arm_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage, 
        params={"term_name": "arm_action_penalty", 
                "num_steps_1": 50000, 
                "num_steps_2": 100000, 
                "weight_1": -0.15, 
                "weight_2": -0.5,}
    )
    arm_velocity_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage, 
        params={"term_name": "arm_velocity_penalty", 
                "num_steps_1": 50000, 
                "num_steps_2": 100000, 
                "weight_1": -0.15, 
                "weight_2": -0.5,}
    )
    gripper_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage, 
        params={"term_name": "gripper_action_penalty", 
                "num_steps_1": 50000, 
                "num_steps_2": 100000, 
                "weight_1": -0.03, 
                "weight_2": -0.1,}
    )
    gripper_velocity_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage, 
        params={"term_name": "gripper_velocity_penalty", 
                "num_steps_1": 50000, 
                "num_steps_2": 100000, 
                "weight_1": -0.02, 
                "weight_2": -0.7,}
    )


##
# Environment configuration
##
@configclass
class PickAndPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        self.episode_length_s = EPISODE_LENGTH_SEC
        self.decimation = DECIMATION
        self.sim.dt = DELTA_TIME
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.scene.contact_forces_arm.update_period = self.sim.dt
        self.scene.contact_forces_outer_gripper.update_period = self.sim.dt
        self.scene.contact_forces_inner_gripper.update_period = self.sim.dt
        self.scene.contact_forces_arm.history_length = self.decimation
        self.scene.contact_forces_outer_gripper.history_length = self.decimation
        self.scene.contact_forces_inner_gripper.history_length = self.decimation