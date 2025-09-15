# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

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
    
    # Clamp to avoid outliers and ensure numerical stability
    action_diff = torch.clamp(action_diff, -2.0, 2.0)
    penalty = torch.mean(torch.square(action_diff), dim=1)
    return penalty

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
    # Clamp velocities to avoid runaway values from simulation instabilities
    joint_vel = asset.data.joint_vel[:, joint_ids]
    joint_vel = torch.nan_to_num(joint_vel, nan=0.0, posinf=0.0, neginf=0.0)
    joint_vel = torch.clamp(joint_vel, -10.0, 10.0)
    penalty = torch.mean(torch.square(joint_vel), dim=1)
    return penalty

def orientation_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward tracking of the orientation using the tanh kernel.
    
    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1 - torch.tanh(distance / std)