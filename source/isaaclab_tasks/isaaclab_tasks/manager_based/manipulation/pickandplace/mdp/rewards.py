# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils.math import quat_inv
from isaaclab.envs import ManagerBasedRLEnv

from .events import phase_flags


def get_phase_states(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Helper function to get current phase states for all environments.
    
    Returns:
        Dictionary containing boolean tensors for each phase state.
    """
    return {
        "pick_phase": ~phase_flags["phase1_complete"],
        "ascend_phase": phase_flags["phase1_complete"] & ~phase_flags["phase2_complete"],
        "descend_phase": phase_flags["phase2_complete"] & ~phase_flags["phase3_complete"],
        "place_phase": phase_flags["phase3_complete"] & ~phase_flags["phase4_complete"],
        "goback_phase": phase_flags["phase4_complete"],
    }

def phase_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Basic reward for phase completion to encourage progression through pick and place phases.
    
    Provides base rewards for each phase completion:
    - Pick phase complete: 1.0 reward
    - Ascend phase complete: 5.0 reward
    - Descend phase complete: 7.5 reward
    - Place phase complete: 10.0 reward (final success)
    """
    phases = get_phase_states(env)
    
    # Calculate cumulative phase rewards with different weights
    phase_rewards = torch.zeros(env.num_envs, device=env.device)
    phase_rewards += torch.where(phases["ascend_phase"], 1.0, 0.0)
    phase_rewards += torch.where(phases["descend_phase"], 5.0, 0.0)
    phase_rewards += torch.where(phases["place_phase"], 10, 0.0)
    phase_rewards += torch.where(phases["goback_phase"], 20.0, 0.0)
    return phase_rewards

def ee_distance(
    env: ManagerBasedRLEnv, 
    std: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"), 
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel.
    
    phase_flags에 따라 mode를 동적으로 결정:
    - Pick/Ascend/Descend phase (phase3_complete가 False): reach mode
    - Place phase (phase3_complete가 True): leave mode
    """
    phases = get_phase_states(env)
    
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    
    # 벡터화된 reward 계산
    reach_reward = 1 - torch.tanh(object_ee_distance / std)
    leave_reward = torch.tanh(object_ee_distance / std) - 1
    
    # reward = torch.where(phases["place_phase"] | phases["goback_phase"], leave_reward*3, reach_reward)

    reward = torch.where(phases["goback_phase"], leave_reward, 
                 torch.where(phases["pick_phase"], reach_reward, 0.0))
    return reward

# def object_contact(
#     env: ManagerBasedRLEnv, 
#     contact_threshold: float = 1.0, 
# ) -> torch.Tensor:
#     """Reward the agent for contacting the object using contact sensor.
    
#     phase_flags에 따라 mode를 동적으로 결정:
#     - Pick/Ascend/Descend phase: grasp mode
#     - Place phase: release mode
#     """
#     phases = get_phase_states(env)

#     left_finger_sensor = env.scene.sensors["contact_forces_left_finger_pad"]
#     right_finger_sensor = env.scene.sensors["contact_forces_right_finger_pad"]
#     left_forces = left_finger_sensor.data.force_matrix_w # filtered
#     right_forces = right_finger_sensor.data.force_matrix_w # filtered
    
#     left_forces_sum = torch.sum(left_forces, dim=(1, 2))
#     right_forces_sum = torch.sum(right_forces, dim=(1, 2))
#     left_force_magnitudes = torch.norm(left_forces_sum, dim=-1)
#     right_force_magnitudes = torch.norm(right_forces_sum, dim=-1)
#     left_contact = (left_force_magnitudes > contact_threshold)
#     right_contact = (right_force_magnitudes > contact_threshold)
    
#     # 벡터화된 reward 계산
#     grasp_reward = torch.where(left_contact & right_contact, 1.0, 
#                        torch.where(left_contact | right_contact, 0.5, 0.0))
#     release_reward = torch.where(~(left_contact & right_contact), 1.0, 
#                          torch.where(left_contact | right_contact, 0.5, 0.0))
    
#     reward = torch.where(phases["place_phase"], release_reward, grasp_reward)
#     return reward


def object_contact(
    env: ManagerBasedRLEnv, 
    contact_threshold: float = 1.0, 
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
    phases = get_phase_states(env)

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
    
    # Calculate quadratic rewards for each finger
    # Grasp mode: higher force = higher reward (0.0 to 0.5)
    left_grasp_reward = torch.square(left_force_norm) * 0.5
    right_grasp_reward = torch.square(right_force_norm) * 0.5
    
    # Release mode: lower force = higher reward (0.0 to 0.5)
    left_release_reward = -torch.square(left_force_norm) * 0.5
    right_release_reward = -torch.square(right_force_norm) * 0.5
    
    # Select appropriate rewards based on phase
    left_reward = torch.where(phases["place_phase"] | phases["goback_phase"], left_release_reward*2, left_grasp_reward)
    right_reward = torch.where(phases["place_phase"] | phases["goback_phase"], right_release_reward*2, right_grasp_reward)
    
    # Total reward is sum of both fingers (max 1.0)
    reward = left_reward + right_reward
    
    return reward

def object_height(
    env: ManagerBasedRLEnv, 
    ascend_threshold: float = 0.03, 
    descend_threshold: float = 0.05, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height but below maximal height.
    
    phase_flags에 따라 mode를 동적으로 결정:
    - Pick/Ascend/ phase: ascend mode
    - Descend/Place phase: descend mode
    """
    phases = get_phase_states(env)
    
    object: RigidObject = env.scene[object_cfg.name]
    
    # Pick/Ascend phase: 높은 높이 장려/유지
    ascend_reward = torch.where((object.data.root_pos_w[:, 2] > ascend_threshold) & (object.data.root_pos_w[:, 2] < 0.4), 1.0, 0.0)
    
    # Descend/Place phase: 낮은 높이 장려
    descend_reward = torch.where(object.data.root_pos_w[:, 2] < descend_threshold, 1.0, 0.0)
    
    # phase에 따라 reward 선택
    # reward = torch.where(phases["descend_phase"] | phases["place_phase"] | phases["goback_phase"], descend_reward, ascend_reward)

    reward = torch.where(phases["descend_phase"], descend_reward, 
                 torch.where(phases["ascend_phase"], ascend_reward, 0.0))
    return reward

def object_track(
    env: ManagerBasedRLEnv, 
    std: float, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel.
    
    phase_flags에 따라 mode를 동적으로 결정:
    - Pick/Ascend/ phase: ascend mode
    - Descend/Place phase: descend mode
    """
    phases = get_phase_states(env)
    
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # 각 환경별로 command 가져오기
    ascend_command = env.command_manager.get_command("ascend")
    descend_command = env.command_manager.get_command("descend")
    
    
    # phase에 따라 command 선택
    command = torch.where(phases["descend_phase"].unsqueeze(-1).expand(-1, ascend_command.shape[-1]), 
                         descend_command, ascend_command)
    
    # Target position in robot frame (already in robot frame)
    target_pos_b = command[:, :3]
    
    # Transform object position to robot frame
    object_pos_w = object.data.root_pos_w
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w
    
    # Transform object position to robot frame
    object_pos_relative = object_pos_w - robot_pos_w  # Translate to robot origin
    robot_quat_inv = quat_inv(robot_quat_w)
    object_pos_b, _ = combine_frame_transforms(
        torch.zeros_like(robot_pos_w),  # No translation needed
        robot_quat_inv,  # Use inverse quaternion
        object_pos_relative
    )
    
    # Calculate distance in robot frame
    distance = torch.norm(object_pos_b - target_pos_b, dim=1)
    
    target_height = target_pos_b[:, 2]
    height_diff = torch.abs(object_pos_b[:, 2] - target_height)
    
    command_reward = 1 - torch.tanh((distance + height_diff) / std)

    reward = torch.where(phases["descend_phase"], command_reward*2, 
                 torch.where(phases["ascend_phase"], command_reward, command_reward/2))
    return reward

def object_place(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), 
) -> torch.Tensor:
    """Reward for successfully placing the object at the target location.
    
    Only activates rewards when in the place phase (is_place_phase = True).
    """
    phases = get_phase_states(env)

    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene["robot"]
    place_command = env.command_manager.get_command("descend")
    
    object_pos_w = object.data.root_pos_w
    robot_pos_w = robot.data.root_pos_w
    robot_quat_w = robot.data.root_quat_w

    object_pos_relative = object_pos_w - robot_pos_w  # Translate
    robot_quat_inv = quat_inv(robot_quat_w)
    object_pos_b, _ = combine_frame_transforms(
        torch.zeros_like(robot_pos_w),  # No translation needed
        robot_quat_inv,  # Use inverse quaternion
        object_pos_relative
    )
    
    # Target position in robot frame (already in robot frame)
    target_pos_b = place_command[:, :3]
    
    # check xy-distance to target
    xy_distance = torch.norm(object_pos_b[:, :2] - target_pos_b[:, :2], dim=1)
    
    # check height difference
    height_diff = torch.abs(object_pos_b[:, 2] - target_pos_b[:, 2])
    
    # Calculate placement reward (1.0 if within thresholds, 0.0 otherwise)
    placement_reward = torch.where((xy_distance < 0.03) & (height_diff < 0.03), 1.0, 0.0)
    
    reward = torch.where(phases["place_phase"] | phases["goback_phase"], placement_reward, 0.0)
    return reward

def initial_pose_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    joint_names: list[str] | None = None, 
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize deviation from initial joint positions.
    
    Only activates when in phase5 (after phase4 completion - object placed and released).
    This encourages the robot to return to initial pose after completing the task.
    
    Args:
        env: The environment.
        asset_cfg: Asset configuration for the robot.
        joint_names: List of joint names to penalize. If None, penalizes all joints.
        std: Standard deviation for the penalty calculation.
    
    Returns:
        Penalty based on deviation from initial joint positions (only in phase5).
    """
    # Check if in phase5 (phase4 completed - task finished)
    
    # Only apply penalty in phase5
    phases = get_phase_states(env)

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
    
    # Calculate penalty using L2 norm and tanh kernel for smooth penalty
    deviation_norm = torch.linalg.norm(joint_deviation, dim=1)
    
    # Only apply penalty in phase5 (goback phase)
    if not torch.any(phases["goback_phase"]):
        return torch.zeros(env.num_envs, device=env.device)
    
    penalty = torch.where(phases["goback_phase"], torch.tanh(deviation_norm/std), 0.0)
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
    
    return torch.clamp(penalty, max=2.0)


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



def action_rate_penalty(env: ManagerBasedRLEnv, action_type: str = "all_actions") -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel.
    
    Args:
        env: The environment.
        action_type: Type of action to calculate rate for. Options: "arm_actions", "gripper_actions", "all_actions"
    
    Returns:
        Action rate penalty for the specified action type.
    """
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    
    if action_type == "arm_actions":
        # arm_action에 대한 action rate 계산
        # arm_action은 action vector의 앞부분 (gripper 제외)
        arm_action_dim = current_action.shape[1] - 1  # 마지막 1개는 gripper
        action_diff = current_action[:, :arm_action_dim] - prev_action[:, :arm_action_dim]
        
    elif action_type == "gripper_actions":
        # gripper_action에 대한 action rate 계산
        # gripper_action은 action vector의 마지막 부분
        gripper_action_dim = 1  # BinaryJointPositionActionCfg는 보통 1차원
        action_diff = current_action[:, -gripper_action_dim:] - prev_action[:, -gripper_action_dim:]
        
    elif action_type == "all_actions":
        # 모든 action에 대한 action rate 계산 (기본값)
        action_diff = current_action - prev_action
        
    else:
        raise ValueError(f"Unknown action_type: {action_type}. Must be 'arm_action', 'gripper_action', or 'all_actions'")
    
    return torch.mean(torch.square(action_diff), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, joint_type: str = "all_joints") -> torch.Tensor:
    """Penalize joint velocities on the articulation based on joint type.
    
    Args:
        env: The environment.
        joint_type: Type of joints to penalize. Options: "arm_joints", "gripper_joints", "all_joints"
    
    Returns:
        Velocity penalty for the specified joint type.
    """
    asset: Articulation = env.scene["robot"]
    
    if joint_type == "arm_joints":
        # arm joints에 대한 velocity penalty 계산
        joint_names = ["joint_.*"]  # arm joint 패턴
    elif joint_type == "gripper_joints":
        # gripper joints에 대한 velocity penalty 계산
        joint_names = ["left_outer_knuckle_joint"]  # gripper joint 패턴 (finger, knuckle 포함)
    elif joint_type == "all_joints":
        # 모든 joint에 대한 velocity penalty 계산
        joint_names = ["joint_.*", "left_outer_knuckle_joint"]
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}. Must be 'arm_joints', 'gripper_joints', or 'all_joints'")
    
    joint_ids, _ = asset.find_joints(joint_names)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    return torch.mean(torch.square(asset.data.joint_vel[:, joint_ids]), dim=1)

def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    if asset_cfg.joint_names is not None:
        joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
        if len(joint_ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
    else:
        joint_ids = slice(None)
    
    penalty = torch.linalg.norm(asset.data.joint_acc[:, joint_ids], dim=1)
    return torch.clamp(penalty, max=50.0)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    if asset_cfg.joint_names is not None:
        joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
        if len(joint_ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
    else:
        joint_ids = slice(None)
    
    penalty = torch.linalg.norm(asset.data.applied_torque[:, joint_ids], dim=1)
    
    return torch.clamp(penalty, max=200.0)