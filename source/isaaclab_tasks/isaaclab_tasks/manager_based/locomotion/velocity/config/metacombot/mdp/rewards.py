# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward helpers tuned for MetaCombOTX."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def base_linear_velocity_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float,
    ramp_at_vel: float = 0.5,
    ramp_rate: float = 0.4,
) -> torch.Tensor:
    """Absolute exponential tracking reward for planar linear velocity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target = command[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


def base_angular_velocity_reward(env, asset_cfg: SceneEntityCfg, command_name: str, std: float) -> torch.Tensor:
    """Absolute exponential tracking reward for yaw angular velocity."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target = command[:, 2]
    ang_vel_error = torch.abs(target - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std)


def base_motion_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize vertical motion and roll/pitch angular velocity for the base."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize deviation from level orientation."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)


def action_smoothness_penalty(env) -> torch.Tensor:
    """Penalize large instantaneous changes in the policy output."""

    return torch.linalg.norm(env.action_manager.action - env.action_manager.prev_action, dim=1)


def joint_position_penalty(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize deviation from default pose when the base should remain still."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos
    joint_pos_default = asset.data.default_joint_pos
    if asset_cfg.joint_names is not None:
        joint_ids = asset.find_joints(asset_cfg.joint_names)[0]
        joint_pos = joint_pos[:, joint_ids]
        joint_pos_default = joint_pos_default[:, joint_ids]
    command = env.command_manager.get_command(command_name)
    cmd_mag = torch.linalg.norm(command[:, :2], dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    error = torch.linalg.norm((joint_pos - joint_pos_default), dim=1)
    return torch.where(torch.logical_or(cmd_mag > 0.0, body_vel > velocity_threshold), error, stand_still_scale * error)


def heading_alignment_reward(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Align base heading with planar command direction."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cmd_xy = command[:, :2]
    cmd_norm = torch.linalg.norm(cmd_xy, dim=1, keepdim=True)
    cmd_dir = torch.zeros_like(cmd_xy)
    valid_cmd = cmd_norm.squeeze(-1) > eps
    cmd_dir = cmd_xy.clone()
    cmd_dir[valid_cmd] = cmd_xy[valid_cmd] / cmd_norm[valid_cmd]

    heading_local = torch.zeros_like(asset.data.root_lin_vel_b)
    heading_local[:, 0] = 1.0
    heading_world = math_utils.quat_apply(asset.data.root_quat_w, heading_local)
    heading_xy = heading_world[:, :2]
    heading_norm = torch.linalg.norm(heading_xy, dim=1, keepdim=True)
    heading_dir = torch.zeros_like(heading_xy)
    valid_heading = heading_norm.squeeze(-1) > eps
    heading_dir[valid_heading] = heading_xy[valid_heading] / heading_norm[valid_heading]

    alignment = torch.sum(heading_dir * cmd_dir, dim=1).clamp(-1.0, 1.0)
    reward = torch.exp(-(1.0 - alignment) / std)
    return torch.where(valid_cmd, reward, torch.ones_like(reward))
