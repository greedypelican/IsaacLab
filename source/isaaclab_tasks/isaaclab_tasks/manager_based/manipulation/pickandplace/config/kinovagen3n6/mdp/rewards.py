# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import matrix_from_quat
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    penalty = torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)
    return torch.clamp(penalty, max=10.0)


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


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    if asset_cfg.joint_names is not None:
        joint_ids, _ = asset.find_joints(asset_cfg.joint_names)
        if len(joint_ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
    else:
        joint_ids = slice(None)
    
    penalty = torch.linalg.norm(asset.data.joint_vel[:, joint_ids], dim=1)
    return torch.clamp(penalty, max=30.0)


def world_ee_z_axis_alignment_penalty(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "gripper_base_link"
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


def object_ee_z_axis_alignment_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    body_name: str = "gripper_base_link"
) -> torch.Tensor:
    """Penalty for object's Z-axis not being aligned with end effector's -X-axis.
    
    Args:
        env: The environment.
        asset_cfg: Robot asset configuration.
        object_cfg: Object asset configuration.
        body_name: Name of the end effector body/link.
    
    Returns:
        Penalty based on misalignment between object Z-axis and EE -X-axis (0.0 to 1.0).
    """
    # Extract the robot asset
    robot: Articulation = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get EE body index
    body_ids, _ = robot.find_bodies([body_name])
    if len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    body_id = body_ids[0]
    
    # Get EE quaternion in world frame
    ee_quat_w = robot.data.body_quat_w[:, body_id]  # (num_envs, 4) [w, x, y, z]
    
    # Get object quaternion in world frame
    object_quat_w = object.data.root_quat_w  # (num_envs, 4) [w, x, y, z]
    
    # Convert quaternions to rotation matrices
    ee_rot_matrix = matrix_from_quat(ee_quat_w)  # (num_envs, 3, 3)
    object_rot_matrix = matrix_from_quat(object_quat_w)  # (num_envs, 3, 3)
    
    # Extract EE -X-axis (negative 1st column) and object Z-axis (3rd column)
    ee_neg_x_axis = -ee_rot_matrix[:, :, 0]  # (num_envs, 3) - negative X-axis
    object_z_axis = object_rot_matrix[:, :, 2]  # (num_envs, 3) - Z-axis
    
    # Compute dot product (cosine similarity) between EE -X-axis and object Z-axis
    dot_product = torch.sum(ee_neg_x_axis * object_z_axis, dim=1)  # (num_envs,)
    
    # Clamp to [-1, 1] to avoid numerical issues
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Convert to penalty: high penalty when misaligned
    # We want penalty to be low when axes are aligned or anti-aligned
    # abs_dot_product = 1 (perfect alignment/anti-alignment) -> penalty = 0
    # abs_dot_product = 0 (perpendicular) -> penalty = 1
    abs_dot_product = torch.abs(dot_product)  # Take absolute value for alignment/anti-alignment
    misalignment = 1.0 - abs_dot_product  # Map [1, 0] to [0, 1]
    penalty = torch.square(misalignment)  # Square for sharper penalty
    
    return torch.clamp(penalty, max=2.0)


def object_world_z_axis_alignment_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
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