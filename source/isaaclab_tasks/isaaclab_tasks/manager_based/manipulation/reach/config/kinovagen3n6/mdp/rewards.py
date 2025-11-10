# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    euler_xyz_from_quat,
    quat_error_magnitude,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _compute_target_scale(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg | None,
    command_name: str,
    scale_cfg: dict | None,
) -> torch.Tensor | None:
    """Return scaling factor based on proximity to the command target."""

    if scale_cfg is None or asset_cfg is None:
        return None

    gain = float(scale_cfg.get("gain", 0.0))
    radius = float(scale_cfg.get("radius", 0.0))
    max_scale = scale_cfg.get("max_scale", None)
    min_scale = scale_cfg.get("min_scale", 1.0)

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)

    body_ids = asset_cfg.body_ids
    if body_ids is None or len(body_ids) == 0:
        return None

    curr_pos = asset.data.body_pos_w[:, body_ids]
    curr_pos_mean = torch.mean(curr_pos, dim=1)
    dist_to_target = torch.norm(curr_pos_mean - des_pos_w, dim=1)

    if gain != 0.0 and radius > 0.0:
        scale = 1.0 + gain * torch.exp(-torch.square(dist_to_target / radius))
    elif "scale" in scale_cfg:
        desired_scale = float(scale_cfg.get("scale", 5.0))
        pos_threshold = float(scale_cfg.get("pos_threshold", 0.02))
        ori_threshold = float(scale_cfg.get("ori_threshold", 0.05))
        pos_slope = float(scale_cfg.get("pos_slope", max(pos_threshold * 0.5, 1e-6)))
        ori_slope = float(scale_cfg.get("ori_slope", max(ori_threshold * 0.5, 1e-6)))

        pos_gate = torch.sigmoid((pos_threshold - dist_to_target) / pos_slope)

        if ori_threshold > 0.0:
            des_quat_b = command[:, 3:7]
            des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
            curr_quat_w = asset.data.body_quat_w[:, body_ids[0]]

            des_euler = torch.stack(euler_xyz_from_quat(des_quat_w), dim=-1)
            curr_euler = torch.stack(euler_xyz_from_quat(curr_quat_w), dim=-1)
            diff = curr_euler - des_euler
            pi_tensor = torch.tensor(math.pi, device=diff.device, dtype=diff.dtype)
            diff = torch.remainder(diff + pi_tensor, 2 * pi_tensor) - pi_tensor
            ori_dist = torch.max(torch.abs(diff), dim=1).values
            ori_gate = torch.sigmoid((ori_threshold - ori_dist) / ori_slope)
        else:
            ori_gate = torch.ones_like(pos_gate)

        gate = pos_gate * ori_gate
        scale = 1.0 + (desired_scale - 1.0) * gate
    else:
        return None

    if min_scale is not None:
        min_tensor = torch.tensor(float(min_scale), device=env.device, dtype=scale.dtype)
        scale = torch.maximum(scale, min_tensor)
    if max_scale is not None:
        max_tensor = torch.tensor(float(max_scale), device=env.device, dtype=scale.dtype)
        scale = torch.minimum(scale, max_tensor)

    return scale


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward orientation tracking using a tanh kernel."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    distance = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1 - torch.tanh(distance / std)


def action_rate_penalty(
    env: ManagerBasedRLEnv,
    action_type: str = "all",
    asset_cfg: SceneEntityCfg | None = None,
    command_name: str = "ee_pose",
    target_scale: dict | None = None,
) -> torch.Tensor:
    """Penalize rate of change of actions with optional proximity scaling."""

    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    if action_type == "arm1":
        action_diff = current_action[:, :3] - prev_action[:, :3]
    elif action_type == "arm2":
        action_diff = current_action[:, 3:6] - prev_action[:, 3:6]
    elif action_type == "gripper":
        action_diff = current_action[:, -1:] - prev_action[:, -1:]
    elif action_type == "all":
        action_diff = current_action - prev_action
    else:
        raise ValueError(f"Unknown action_type: {action_type}.")

    penalty = torch.mean(torch.square(action_diff), dim=1)
    scale = _compute_target_scale(env, asset_cfg, command_name, target_scale)
    if scale is not None:
        penalty = penalty * scale
    return penalty


def joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    joint_type: str = "all",
    asset_cfg: SceneEntityCfg | None = None,
    command_name: str = "ee_pose",
    target_scale: dict | None = None,
) -> torch.Tensor:
    """Penalize joint velocities for selected joint groups."""

    asset: Articulation = env.scene["robot"]

    if joint_type == "arm1":
        joint_names = ["joint_1", "joint_2", "joint_3"]
    elif joint_type == "arm2":
        joint_names = ["joint_4", "joint_5", "joint_6"]
    elif joint_type == "all":
        joint_names = ["joint_.*", "left_outer_knuckle_joint"]
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}.")

    joint_ids, _ = asset.find_joints(joint_names)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    joint_vel = torch.nan_to_num(asset.data.joint_vel[:, joint_ids], nan=0.0, posinf=0.0, neginf=0.0)
    penalty = torch.mean(torch.square(torch.clamp(joint_vel, -10.0, 10.0)), dim=1)

    scale = _compute_target_scale(env, asset_cfg, command_name, target_scale)
    if scale is not None:
        penalty = penalty * scale
    return penalty


def joint_acceleration_penalty(
    env: ManagerBasedRLEnv,
    joint_type: str = "all",
    asset_cfg: SceneEntityCfg | None = None,
    command_name: str = "ee_pose",
    target_scale: dict | None = None,
) -> torch.Tensor:
    """Penalize joint accelerations for selected joint groups."""

    asset: Articulation = env.scene["robot"]

    if joint_type == "arm1":
        joint_names = ["joint_1", "joint_2", "joint_3"]
    elif joint_type == "arm2":
        joint_names = ["joint_4", "joint_5", "joint_6"]
    elif joint_type == "gripper":
        joint_names = ["left_outer_knuckle_joint"]
    elif joint_type == "all":
        joint_names = ["joint_.*", "left_outer_knuckle_joint"]
    else:
        raise ValueError(f"Unknown joint_type: {joint_type}.")

    joint_ids, _ = asset.find_joints(joint_names)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    joint_acc = torch.nan_to_num(asset.data.joint_acc[:, joint_ids], nan=0.0, posinf=0.0, neginf=0.0)
    penalty = torch.mean(torch.square(torch.clamp(joint_acc, -50.0, 50.0)), dim=1)

    scale = _compute_target_scale(env, asset_cfg, command_name, target_scale)
    if scale is not None:
        penalty = penalty * scale
    return penalty


def end_effector_pose_movement_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    lin_weight: float = 1.0,
    ang_weight: float = 0.1,
    command_name: str = "ee_pose",
    target_scale: dict | None = None,
) -> torch.Tensor:
    """Penalize end-effector velocities to encourage smooth motion."""

    asset: RigidObject = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if body_ids is None or len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    lin_vel = torch.nan_to_num(asset.data.body_lin_vel_w[:, body_ids, :], nan=0.0, posinf=0.0, neginf=0.0)
    ang_vel = torch.nan_to_num(asset.data.body_ang_vel_w[:, body_ids, :], nan=0.0, posinf=0.0, neginf=0.0)

    lin_speed_sq = torch.sum(torch.square(lin_vel), dim=-1)
    ang_speed_sq = torch.sum(torch.square(ang_vel), dim=-1)

    penalty = lin_weight * torch.mean(lin_speed_sq, dim=1) + ang_weight * torch.mean(ang_speed_sq, dim=1)

    scale = _compute_target_scale(env, asset_cfg, command_name, target_scale)
    if scale is not None:
        penalty = penalty * scale

    return penalty


def end_effector_pose_displacement_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_weight: float = 1.0,
    rot_weight: float = 0.0,
    command_name: str = "ee_pose",
    target_scale: dict | None = None,
) -> torch.Tensor:
    """Penalize step-to-step displacement of the end-effector."""

    asset: RigidObject = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if body_ids is None or len(body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    current_pos = torch.nan_to_num(asset.data.body_pos_w[:, body_ids, :], nan=0.0, posinf=0.0, neginf=0.0)
    current_quat = torch.nan_to_num(asset.data.body_quat_w[:, body_ids, :], nan=0.0, posinf=0.0, neginf=0.0)

    body_key = "_".join(asset_cfg.body_names) if asset_cfg.body_names else "_".join(str(bid) for bid in body_ids)
    pos_attr = f"_prev_body_pos_{asset_cfg.name}_{body_key}"
    quat_attr = f"_prev_body_quat_{asset_cfg.name}_{body_key}"

    if not hasattr(env, pos_attr):
        setattr(env, pos_attr, current_pos.clone())
    prev_pos = getattr(env, pos_attr)
    if prev_pos.shape != current_pos.shape:
        prev_pos = current_pos.clone()
        setattr(env, pos_attr, prev_pos)

    if rot_weight > 0.0:
        if not hasattr(env, quat_attr):
            setattr(env, quat_attr, current_quat.clone())
        prev_quat = getattr(env, quat_attr)
        if prev_quat.shape != current_quat.shape:
            prev_quat = current_quat.clone()
            setattr(env, quat_attr, prev_quat)
    else:
        prev_quat = None

    if hasattr(env, "reset_buf"):
        reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_ids.numel() > 0:
            prev_pos[reset_ids] = current_pos[reset_ids]
            if prev_quat is not None:
                prev_quat[reset_ids] = current_quat[reset_ids]

    displacement = current_pos - prev_pos
    pos_dist = torch.linalg.norm(displacement, dim=-1)
    pos_penalty = pos_weight * torch.mean(pos_dist, dim=1)

    rot_penalty = 0.0
    if prev_quat is not None:
        rot_delta = quat_error_magnitude(current_quat, prev_quat)
        rot_penalty = rot_weight * torch.mean(rot_delta, dim=1)

    prev_pos.copy_(current_pos)
    if prev_quat is not None:
        prev_quat.copy_(current_quat)

    penalty = pos_penalty + rot_penalty

    scale = _compute_target_scale(env, asset_cfg, command_name, target_scale)
    if scale is not None:
        penalty = penalty * scale

    return penalty