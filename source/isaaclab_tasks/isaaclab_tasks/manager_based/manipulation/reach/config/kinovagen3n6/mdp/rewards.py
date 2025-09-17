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


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""

    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    action_diff = torch.clamp(current_action - prev_action, -2.0, 2.0)

    penalty = torch.mean(torch.square(action_diff), dim=1)
    return penalty

def joint_velocity_penalty(env: ManagerBasedRLEnv, joint_names=[".*"]) -> torch.Tensor:
    """Penalize joint velocities on the articulation based on joint type."""

    asset: Articulation = env.scene["robot"]

    joint_ids, _ = asset.find_joints(joint_names)
    if len(joint_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    joint_vel = torch.nan_to_num(asset.data.joint_vel[:, joint_ids], nan=0.0, posinf=0.0, neginf=0.0)

    penalty = torch.mean(torch.square(torch.clamp(joint_vel, -10.0, 10.0)), dim=1)
    return penalty

def orientation_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward tracking of the orientation using the tanh kernel."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    distance = quat_error_magnitude(curr_quat_w, des_quat_w)

    penalty = 1 - torch.tanh(distance / std)
    return penalty