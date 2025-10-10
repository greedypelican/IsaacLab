# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

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

def action_rate_penalty(env: ManagerBasedRLEnv, action_type: str = "all") -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""

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
        raise ValueError(f"Unknown action_type: {action_type}. Must be 'arm1', 'arm_', 'gripper', or 'all'")

    penalty = torch.mean(torch.square(action_diff), dim=1)
    return penalty

def joint_velocity_penalty(env: ManagerBasedRLEnv, joint_type: str = "all") -> torch.Tensor:
    """Penalize joint velocities on the articulation based on joint type."""

    asset: Articulation = env.scene["robot"]
    
    if joint_type == "arm1":
        # arm joints에 대한 velocity penalty 계산
        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4"]
    elif joint_type == "arm2":
        # arm joints에 대한 velocity penalty 계산
        joint_names = ["joint_5", "joint_6"]
    elif joint_type == "all":
        # 모든 joint에 대한 velocity penalty 계산
        joint_names = ["joint_.*", "left_outer_knuckle_joint"]
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}. Must be 'arm1', 'arm2', 'gripper', or 'all'")
    
    joint_ids, _ = asset.find_joints(joint_names)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    penalty = torch.mean(torch.square(asset.data.joint_vel[:, joint_ids]), dim=1)
    return penalty

def joint_acceleration_penalty(env: ManagerBasedRLEnv, joint_type: str = "all") -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""

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
    return penalty