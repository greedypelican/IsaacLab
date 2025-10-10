# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import math

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

from . import mdp


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


##
# Scene definition
##
@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg | DeformableObjectCfg = MISSING
    table: AssetBaseCfg = MISSING
    plane: AssetBaseCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    contact_forces_arm: ContactSensorCfg = MISSING
    contact_forces_left_finger_pad: ContactSensorCfg = MISSING
    contact_forces_right_finger_pad: ContactSensorCfg = MISSING


##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    move = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(EPISODE_LENGTH_SEC, EPISODE_LENGTH_SEC),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5), pos_y=(-0.2, 0.2), pos_z=(0.15, 0.20), roll=(0.0, 0.0), pitch=(math.pi/2, math.pi/2), yaw=(0.0, 0.0)
        ),
    )
    target = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(EPISODE_LENGTH_SEC, EPISODE_LENGTH_SEC),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5), pos_y=(-0.2, 0.2), pos_z=(0.02, 0.02), roll=(0.0, 0.0), pitch=(math.pi/2, math.pi/2), yaw=(0.0, 0.0)
        ),
    )

    def __post_init__(self):
        self.move.goal_pose_visualizer_cfg = MARKER_CFG.replace(prim_path="/Visuals/Command/move_pose")
        self.move.goal_pose_visualizer_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.1, 0.0))
        self.target.goal_pose_visualizer_cfg = MARKER_CFG.replace(prim_path="/Visuals/Command/target_pose")
        self.target.goal_pose_visualizer_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 0.6))


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
        initial_object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, noise=Unoise(n_min=-0.01, n_max=0.01))
        initial_object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_frame_pose = ObsTerm(func=mdp.ee_frame_pose)
        move_command_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "move"})
        target_command_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "target"})
        current_phase = ObsTerm(func=mdp.current_phase)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

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
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0), "roll": (-math.pi, math.pi)},
            "velocity_range": {},
        },
    )
    cache_object_initial_pose = EventTerm(
        func=mdp.cache_object_initial_pose,
        mode="reset",
    )
    cache_ee_initial_pose = EventTerm(
        func=mdp.cache_ee_initial_pose,
        mode="reset",
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
        weight=1.0,
    )
    height = RewTerm(
        func=mdp.object_height,
        weight=1.0,
    )
    track = RewTerm(
        func=mdp.object_track,
        weight=1.0,
    )
    goback = RewTerm(
        func=mdp.initial_pose,
        weight=1.0,
    )
    ready = RewTerm(
        func=mdp.joint_similarity_penalty,
        weight=1.0,
    )

    ee_motion_penalty = RewTerm(
        func=mdp.ee_motion_penalty,
        weight=-0.0,
    )
    ee_alignment_penalty = RewTerm(
        func=mdp.world_ee_z_axis_alignment_penalty,
        params={"body_name": MISSING},
        weight=-0.0, 
    )
    arm1_action_penalty = RewTerm(
        func=mdp.action_rate_penalty,
        params={"action_type": "arm_1"},
        weight=-0.0,
    )
    arm1_velocity_penalty = RewTerm(
        func=mdp.joint_velocity_penalty,
        params={"joint_type": "arm_1"},
        weight=-0.0,
    )
    arm1_acceleration_penalty = RewTerm(
        func=mdp.joint_acceleration_penalty,
        params={"joint_type": "arm_1"},
        weight=-0.0,
    )
    arm2_action_penalty = RewTerm(
        func=mdp.action_rate_penalty,
        params={"action_type": "arm_2"},
        weight=-0.0,
    )
    arm2_velocity_penalty = RewTerm(
        func=mdp.joint_velocity_penalty,
        params={"joint_type": "arm_2"},
        weight=-0.0,
    )
    arm2_acceleration_penalty = RewTerm(
        func=mdp.joint_acceleration_penalty,
        params={"joint_type": "arm_2"},
        weight=-0.0,
    )
    gripper_action_penalty = RewTerm(
        func=mdp.action_rate_penalty,
        params={"action_type": "gripper"},
        weight=-0.0,
    )
    gripper_velocity_penalty = RewTerm(
        func=mdp.joint_velocity_penalty,
        params={"joint_type": "gripper"},
        weight=-0.0,
    )
    gripper_acceleration_penalty = RewTerm(
        func=mdp.joint_acceleration_penalty,
        params={"joint_type": "gripper"},
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
        params={"object_cfg": SceneEntityCfg("object"), "height_threshold": 0.05},
    )
    object_out_of_bounds = DoneTerm(
        func=mdp.object_out_of_bounds,
        params={"object_cfg": SceneEntityCfg("object")},
    )
    arm_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces_arm"), "threshold": 1.0},
    )
    object_move_after_place = DoneTerm(
        func=mdp.object_move_after_place,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    ee_motion_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "ee_motion_penalty",
            "num_steps_1": 100000,
            "weight_1": -0.5,
            "num_steps_2": 200000,
            "weight_2": -1.0,
        }
    )
    ee_alignment_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "ee_alignment_penalty",
            "num_steps_1": 100000,
            "weight_1": -0.5,
            "num_steps_2": 200000,
            "weight_2": -1.0,
        }
    )
    arm1_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "arm1_action_penalty",
            "num_steps_1": 25000,
            "weight_1": -0.05,
            "num_steps_2": 125000,
            "weight_2": -0.25,
        }
    )
    arm1_velocity_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "arm1_velocity_penalty",
            "num_steps_1": 25000,
            "weight_1": -0.02,
            "num_steps_2": 125000,
            "weight_2": -0.05,
        }
    )
    arm1_acceleration_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "arm1_acceleration_penalty",
            "num_steps": 225000,
            "weight": -0.00002,
        }
    )
    arm2_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "arm2_action_penalty",
            "num_steps_1": 25000,
            "weight_1": -0.03,
            "num_steps_2": 125000,
            "weight_2": -0.15,
        }
    )
    arm2_velocity_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "arm2_velocity_penalty",
            "num_steps_1": 25000,
            "weight_1": -0.012,
            "num_steps_2": 125000,
            "weight_2": -0.03,
        }
    )
    arm2_acceleration_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "arm2_acceleration_penalty",
            "num_steps": 225000,
            "weight": -0.000012,
        }
    )
    gripper_action_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "gripper_action_penalty",
            "num_steps_1": 75000,
            "num_steps_2": 175000,
            "weight_1": -0.04,
            "weight_2": -0.2,
        }
    )
    gripper_velocity_penalty = CurrTerm(
        func=mdp.modify_reward_weight_multi_stage,
        params={
            "term_name": "gripper_velocity_penalty",
            "num_steps_1": 75000,
            "num_steps_2": 175000,
            "weight_1": -0.008,
            "weight_2": -0.016,
        }
    )
    gripper_acceleration_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "gripper_acceleration_penalty",
            "num_steps": 275000,
            "weight": -0.000008,
        }
    )


##
# Environment configuration
##
@configclass
class PickAndPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.0)
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
        self.scene.contact_forces_left_finger_pad.update_period = self.sim.dt
        self.scene.contact_forces_right_finger_pad.update_period = self.sim.dt
        self.scene.contact_forces_arm.history_length = self.decimation
        self.scene.contact_forces_left_finger_pad.history_length = self.decimation
        self.scene.contact_forces_right_finger_pad.history_length = self.decimation
