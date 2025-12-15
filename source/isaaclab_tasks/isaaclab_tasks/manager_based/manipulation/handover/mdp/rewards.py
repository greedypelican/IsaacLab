# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils.math import quat_inv
from isaaclab.envs import ManagerBasedRLEnv

from .events import phase_flags, GRASP_THRESHOLD, RELEASE_THRESHOLD, Z_OFFSET


def _phase_states(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Helper function to get current phase states for all environments.
    
    Returns:
        Dictionary containing boolean tensors for each phase state.
    """
    return {
        "pick_phase": ~phase_flags["phase1_complete"],
        "ascend_phase": phase_flags["phase1_complete"] & ~phase_flags["phase2_complete"],
        "descend_phase": phase_flags["phase2_complete"] & ~phase_flags["phase3_complete"],
        "place_phase": phase_flags["phase3_complete"] & ~phase_flags["phase4_complete"],
        "goback_phase": phase_flags["phase4_complete"] & ~phase_flags["phase5_complete"],
        "ready_phase": phase_flags["phase5_complete"],
    }

def _is_grasping(env: ManagerBasedRLEnv, contact_threshold: float = GRASP_THRESHOLD) -> torch.Tensor:
    """Return a boolean tensor per env indicating if both finger pads exceed threshold."""
    left_finger_sensor = env.scene.sensors["contact_forces_left_finger_pad"]
    right_finger_sensor = env.scene.sensors["contact_forces_right_finger_pad"]
    left_sum = torch.sum(left_finger_sensor.data.force_matrix_w, dim=(1, 2))
    right_sum = torch.sum(right_finger_sensor.data.force_matrix_w, dim=(1, 2))
    left_mag = torch.norm(left_sum, dim=-1)
    right_mag = torch.norm(right_sum, dim=-1)
    return (left_mag >= contact_threshold) & (right_mag >= contact_threshold)


def phase_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Basic reward for phase completion to encourage progression through pick and place phases.
    
    Provides base rewards for each phase completion (cumulate):
    - Pick phase complete: 3.0 reward
    - Ascend phase complete: 5.0 reward
    - Descend phase complete: 8.0 reward
    - Place phase complete: 12.0 reward
    - GoBack phase complete: 17.0 reward
    """

    phases = _phase_states(env)
    phase_rewards = torch.zeros(env.num_envs, device=env.device)
    phase_rewards += torch.where(phases["ascend_phase"], 3.0, 0.0)
    phase_rewards += torch.where(phases["descend_phase"], 8.0, 0.0)
    phase_rewards += torch.where(phases["place_phase"], 16.0, 0.0)
    phase_rewards += torch.where(phases["goback_phase"], 28.0, 0.0)
    phase_rewards += torch.where(phases["ready_phase"], 45.0, 0.0)

    return phase_rewards

def ee_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.8,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    phases = _phase_states(env)

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: RigidObject = env.scene["robot"]
    descend_command = env.command_manager.get_command("descend")

    obj_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    obj_pos_b, _ = combine_frame_transforms(torch.zeros_like(robot.data.root_pos_w), quat_inv(robot.data.root_quat_w), obj_pos_w - robot.data.root_pos_w)
    ee_pos_b, _  = combine_frame_transforms(torch.zeros_like(robot.data.root_pos_w), quat_inv(robot.data.root_quat_w), ee_pos_w - robot.data.root_pos_w)

    # Pick phase
    dist_obj_b = torch.norm(obj_pos_b - ee_pos_b, dim=1)
    reach_reward = 1 - torch.tanh(dist_obj_b / std)

    # Place phase
    leave_target_b = descend_command[:, :3].clone()
    leave_target_b[:, 2] += Z_OFFSET
    dist_leave_b = torch.norm(leave_target_b - ee_pos_b, dim=1)
    leave_reward = 1 - torch.tanh(dist_leave_b / std)

    reward = torch.where(phases["pick_phase"], reach_reward * 1.0,
                 torch.where(phases["place_phase"], leave_reward * 5.0, 0.0))
    return reward

def object_contact(
    env: ManagerBasedRLEnv,
    contact_threshold: float = GRASP_THRESHOLD,
) -> torch.Tensor:
    """Linear reward for contacting the object using contact sensor.
    
    Provides linear rewards based on contact forces:
    - Force range: [0.0, 1.0] (clipped)
    - Grasp mode: force=1.0 -> 0.5 reward, force=0.0 -> 0.0 reward
    - Release mode: force=0.0 -> 0.5 reward, force=1.0 -> 0.0 reward
    - Independent rewards for left and right fingers (max total: 1.0)
    
    Args:
        env: The environment.
        contact_threshold: Threshold for contact detection.
    
    Returns:
        Linear reward based on contact forces (0.0 to 1.0).
    """
    phases = _phase_states(env)

    left_finger_sensor = env.scene.sensors["contact_forces_left_finger_pad"]
    right_finger_sensor = env.scene.sensors["contact_forces_right_finger_pad"]
    left_forces = left_finger_sensor.data.force_matrix_w  # filtered
    right_forces = right_finger_sensor.data.force_matrix_w  # filtered
    
    left_forces_sum = torch.sum(left_forces, dim=(1, 2))
    right_forces_sum = torch.sum(right_forces, dim=(1, 2))
    left_force_magnitudes = torch.norm(left_forces_sum, dim=-1)
    right_force_magnitudes = torch.norm(right_forces_sum, dim=-1)
    
    # Clip forces to [0.0, 1.0] range
    left_force_norm = torch.clamp(left_force_magnitudes / contact_threshold, 0.0, 1.0)
    right_force_norm = torch.clamp(right_force_magnitudes / contact_threshold, 0.0, 1.0)
    
    # Contact/release aggregated across both fingers then scaled
    grasp_reward = (torch.square(left_force_norm) + torch.square(right_force_norm)) * 0.5
    release_reward = (torch.square(1.0 - left_force_norm) + torch.square(1.0 - right_force_norm)) * 0.5

    reward = torch.where(phases["pick_phase"], grasp_reward * 1.5,
                 torch.where(phases["ascend_phase"], grasp_reward * 1.0,
                     torch.where(phases["place_phase"], release_reward * 7.0, 0.0)))
    return reward

def object_height(
    env: ManagerBasedRLEnv,
    ascend_threshold: float = 0.0,
    descend_threshold: float = 0.0,
) -> torch.Tensor:
    phases = _phase_states(env)
    robot: RigidObject = env.scene["robot"]
    obj: RigidObject = env.scene["object"]

    obj_pos_w = obj.data.root_pos_w
    obj_pos_b, _ = combine_frame_transforms(torch.zeros_like(robot.data.root_pos_w), quat_inv(robot.data.root_quat_w), obj_pos_w - robot.data.root_pos_w)
    height_b = obj_pos_b[:, 2]

    ascend_reward = torch.where(height_b >= ascend_threshold, 1.0, 0.0)
    descend_reward = torch.where(height_b <= descend_threshold, 1.0, 0.0)
    reward = torch.where(phases["pick_phase"], ascend_reward*0.5, 
                 torch.where(phases["ascend_phase"], ascend_reward*1.0, 
                     torch.where(phases["descend_phase"], descend_reward*1.5,
                         torch.where(phases["place_phase"], descend_reward*0.0, 0.0))))
    return reward

def object_track(
    env: ManagerBasedRLEnv,
    std: float = 0.7,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for tracking the phase-specific waypoints using a tanh kernel."""
    phases = _phase_states(env)

    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene["object"]

    ascend_command = env.command_manager.get_command("ascend")
    descend_command = env.command_manager.get_command("descend")

    obj_pos_w = obj.data.root_pos_w

    # Grasping detection from finger pad contact sensors
    grasping = _is_grasping(env)

    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    object_pos_relative = obj_pos_w - robot_pos_w  # Translate to robot origin
    robot_quat_inv = quat_inv(robot_quat_w)
    object_pos_b, _ = combine_frame_transforms(
        torch.zeros_like(robot_pos_w),  # No translation needed
        robot_quat_inv,  # Use inverse quaternion
        object_pos_relative
    )

    ascend_target = ascend_command[:, :3]
    descend_target = descend_command[:, :3]

    place_target = object_pos_b  # During place phase, hold current object pose (no command target)
    target_pos_b = torch.where(phases["ascend_phase"].unsqueeze(-1), ascend_target,
                       torch.where(phases["descend_phase"].unsqueeze(-1), descend_target, place_target))

    distance = torch.norm(object_pos_b - target_pos_b, dim=1)
    height_diff = torch.abs(object_pos_b[:, 2] - target_pos_b[:, 2])
    command_reward = 1 - torch.tanh((distance + height_diff) / std)

    reward = torch.where(phases["ascend_phase"], command_reward*3.0, 
                 torch.where(phases["descend_phase"], command_reward*4.5, 0.0))
    return reward



def ee_motion_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Penalty proportional to the EE displacement since the previous step."""

    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    ee_pos_relative = ee_pos_w - robot.data.root_pos_w
    robot_quat_inv = quat_inv(robot.data.root_quat_w)
    ee_pos_b, _ = combine_frame_transforms(
        torch.zeros_like(robot.data.root_pos_w),
        robot_quat_inv,
        ee_pos_relative,
    )

    prev_pose = getattr(env, "_prev_ee_pos_b", None)
    if prev_pose is None:
        env._prev_ee_pos_b = ee_pos_b.clone()
        return torch.zeros(env.num_envs, device=env.device)

    delta = ee_pos_b - prev_pose
    env._prev_ee_pos_b = ee_pos_b.clone()

    return torch.linalg.norm(delta, dim=1) * 100

def initial_pose(
    env: ManagerBasedRLEnv,
    std: float = 1.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Reward closeness to the initial joint positions.

    Activates only in the goback phase (after phase4 completion: object placed and released).
    This encourages the robot to return to its initial pose after completing the task.

    Args:
        env: The environment.
        asset_cfg: Asset configuration for the robot.
        joint_names: List of joint names to consider. If None, uses all joints.
        std: Smoothing factor for the reward calculation.

    Returns:
        Reward that is higher when joints are closer to their initial positions (only in goback phase).
    """
    # Only apply reward in goback phase
    phases = _phase_states(env)

    asset: Articulation = env.scene[asset_cfg.name]

    # Get joint IDs
    if joint_names is not None:
        joint_ids, _ = asset.find_joints(joint_names)
        if len(joint_ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
    else:
        joint_ids = slice(None)

    # Get current joint positions and initial joint positions
    current_joint_pos = asset.data.joint_pos[:, joint_ids]
    initial_joint_pos = asset.data.default_joint_pos[:, joint_ids]

    # Calculate deviation from initial pose
    joint_deviation = current_joint_pos - initial_joint_pos

    # Calculate deviation using L2 norm
    deviation_norm = torch.linalg.norm(joint_deviation, dim=1)

    # Convert deviation to reward: small deviation -> high reward (in [0, 1))
    proximity_reward = 1.0 - torch.tanh(deviation_norm / std)

    reward = torch.where(phases["goback_phase"], proximity_reward*15.0, 0.0)
    return reward

def joint_similarity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg  = SceneEntityCfg("robot"),
    joint_names: list[str] | None = None,
    std: float = 0.7,
) -> torch.Tensor:
    """Penalty based on cosine similarity between current and default joint positions.

    Returns a value in [0, 1], where 0 means identical direction (max similarity)
    and 1 means opposite direction (min similarity).
    """
    phases = _phase_states(env)

    asset: Articulation = env.scene[asset_cfg.name]

    # Optionally support joint subsets if provided in cfg
    if joint_names is not None:
        joint_ids, _ = asset.find_joints(joint_names)
        if len(joint_ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
    else:
        joint_ids = slice(None)

    # Get current joint positions and initial joint positions
    current_joint_pos = asset.data.joint_pos[:, joint_ids]
    initial_joint_pos = asset.data.default_joint_pos[:, joint_ids]

    cos_sim = F.cosine_similarity(current_joint_pos, initial_joint_pos, dim=1, eps=1e-8)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    # Tanh-shaped mapping with std: smaller std -> sharper saturation
    # cos=1 -> ~0, cos=0 -> 0.5, cos=-1 -> ~1
    similarity_penalty = 0.5 * (1.0 - torch.tanh(cos_sim / std))
    penalty = torch.where(phases["ready_phase"], similarity_penalty*-100.0, 0.0)
    return penalty


def world_ee_z_axis_alignment_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "gripper_base_link",
) -> torch.Tensor:
    """Penalty for EE -X-axis not being aligned with world +Z-axis.
    
    Args:
        env: The environment.
        asset_cfg: Asset configuration.
        body_name: Name of the body/link to check orientation.
    
    Returns:
        Penalty based on -X-axis misalignment with world +Z-axis (0.0 to 1.0).
    """
    # Extract the asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body index
    body_ids, _ = asset.find_bodies([body_name])
    if len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    body_id = body_ids[0]
    
    # Get body quaternion in world frame
    body_quat_w = asset.data.body_quat_w[:, body_id]  # (num_envs, 4) [w, x, y, z]
    
    # Convert quaternion to rotation matrix
    rot_matrix = matrix_from_quat(body_quat_w)  # (num_envs, 3, 3)
    
    # Extract -X-axis (negative 1st column) from rotation matrix
    local_neg_x_axis = -rot_matrix[:, :, 0]  # (num_envs, 3)
    
    # World Z-axis
    world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)
    world_z_axis = world_z_axis.unsqueeze(0).expand(env.num_envs, -1)  # (num_envs, 3)
    
    # Compute dot product (cosine similarity)
    dot_product = torch.sum(local_neg_x_axis * world_z_axis, dim=1)  # (num_envs,)
    
    # Clamp to [-1, 1] to avoid numerical issues
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Convert to penalty: high penalty when misaligned (dot product != 1)
    # dot_product = 1 (perfect alignment) -> penalty = 0
    # dot_product = -1 (opposite alignment) -> penalty = 1
    # dot_product = 0 (perpendicular) -> penalty = 0.5
    misalignment = 1.0 - dot_product  # Map [1, -1] to [0, 2]
    penalty = misalignment / 2.0  # Normalize to [0, 1]
    penalty = torch.square(penalty)  # Square for sharper penalty
    
    return torch.clamp(penalty, max=1.0, min=1e-6) * 1000


def object_world_z_axis_alignment_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for object's Z-axis not being aligned with world Z-axis.
    
    Args:
        env: The environment.
        object_cfg: Object asset configuration.
    
    Returns:
        Penalty based on object Z-axis misalignment with world Z-axis (0.0 to 1.0).
    """
    # Extract the object
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object quaternion in world frame
    object_quat_w = object.data.root_quat_w  # (num_envs, 4) [w, x, y, z]
    
    # Convert quaternion to rotation matrix
    object_rot_matrix = matrix_from_quat(object_quat_w)  # (num_envs, 3, 3)
    
    # Extract object Z-axis (3rd column) from rotation matrix
    object_z_axis = object_rot_matrix[:, :, 2]  # (num_envs, 3)
    
    # World Z-axis
    world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)
    world_z_axis = world_z_axis.unsqueeze(0).expand(env.num_envs, -1)  # (num_envs, 3)
    
    # Compute dot product (cosine similarity)
    dot_product = torch.sum(object_z_axis * world_z_axis, dim=1)  # (num_envs,)
    
    # Clamp to [-1, 1] to avoid numerical issues
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Convert to penalty: high penalty when misaligned (dot product != 1)
    # dot_product = 1 (perfect alignment) -> penalty = 0
    # dot_product = -1 (opposite alignment) -> penalty = 1
    # dot_product = 0 (perpendicular) -> penalty = 0.5
    misalignment = 1.0 - dot_product  # Map [1, -1] to [0, 2]
    penalty = misalignment / 2.0  # Normalize to [0, 1]
    penalty = torch.square(penalty)  # Square for sharper penalty
    
    return torch.clamp(penalty, max=2.0)



def action_rate_penalty(env: ManagerBasedRLEnv, action_type: str = "all") -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    phases = _phase_states(env)

    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    if action_type == "arm1":
        # Use the first 4 arm joints
        action_diff = current_action[:, :4] - prev_action[:, :4]
    elif action_type == "arm2":
        # Use the 5th and 6th arm joints
        action_diff = current_action[:, 4:6] - prev_action[:, 4:6]
    elif action_type == "gripper":
        # Use the gripper joint (assumed to be the last entry)
        action_diff = current_action[:, -1:] - prev_action[:, -1:]
    elif action_type == "all":
        # 모든 action에 대한 action rate 계산 (기본값)
        action_diff = current_action - prev_action
    else:
        raise ValueError(f"Unknown action_type: {action_type}. Must be 'arm1', 'arm2', 'gripper', or 'all'")

    penalty = torch.mean(torch.square(action_diff), dim=1)
    return torch.where(phases["ready_phase"], penalty*5.0, penalty)

def joint_torques_penalty(env: ManagerBasedRLEnv, joint_type: str = "all") -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    phases = _phase_states(env)

    asset: Articulation = env.scene["robot"]

    if joint_type == "arm1":
        # arm joints에 대한 velocity penalty 계산
        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4"]
    elif joint_type == "arm2":
        # arm joints에 대한 velocity penalty 계산
        joint_names = ["joint_5", "joint_6"]
    elif joint_type == "gripper":
        # gripper joints에 대한 velocity penalty 계산
        joint_names = ["left_outer_knuckle_joint"]
    elif joint_type == "all":
        # 모든 joint에 대한 velocity penalty 계산
        joint_names = ["joint_.*", "left_outer_knuckle_joint"]
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}. Must be 'arm1', 'arm2', 'gripper', or 'all'")

    joint_ids, _ = asset.find_joints(joint_names)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    penalty = torch.mean(torch.square(asset.data.applied_torque[:, joint_ids]), dim=1)
    return torch.where(phases["ready_phase"], penalty*5.0, penalty)

def joint_velocity_penalty(env: ManagerBasedRLEnv, joint_type: str = "all") -> torch.Tensor:
    """Penalize joint velocities on the articulation based on joint type."""
    phases = _phase_states(env)

    asset: Articulation = env.scene["robot"]
    
    if joint_type == "arm1":
        # arm joints에 대한 velocity penalty 계산
        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4"]
    elif joint_type == "arm2":
        # arm joints에 대한 velocity penalty 계산
        joint_names = ["joint_5", "joint_6"]
    elif joint_type == "gripper":
        joint_names = ["left_outer_knuckle_joint"]
    elif joint_type == "all":
        # 모든 joint에 대한 velocity penalty 계산
        joint_names = ["joint_.*", "left_outer_knuckle_joint"]
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}. Must be 'arm1', 'arm2', 'gripper', or 'all'")
    
    joint_ids, _ = asset.find_joints(joint_names)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    penalty = torch.mean(torch.square(asset.data.joint_vel[:, joint_ids]), dim=1)
    return torch.where(phases["ready_phase"], penalty*5.0, penalty)

def joint_acceleration_penalty(env: ManagerBasedRLEnv, joint_type: str = "all") -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    phases = _phase_states(env)

    asset: Articulation = env.scene["robot"]

    if joint_type == "arm1":
        # arm joints에 대한 velocity penalty 계산
        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4"]
    elif joint_type == "arm2":
        # arm joints에 대한 velocity penalty 계산
        joint_names = ["joint_5", "joint_6"]
    elif joint_type == "gripper":
        # gripper joints에 대한 velocity penalty 계산
        joint_names = ["left_outer_knuckle_joint"]
    elif joint_type == "all":
        # 모든 joint에 대한 velocity penalty 계산
        joint_names = ["joint_.*", "left_outer_knuckle_joint"]
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}. Must be 'arm1', 'arm2', 'gripper', or 'all'")

    joint_ids, _ = asset.find_joints(joint_names)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    penalty = torch.mean(torch.square(asset.data.joint_acc[:, joint_ids]), dim=1)
    return torch.where(phases["ready_phase"], penalty*5.0, penalty)
