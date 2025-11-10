# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation utilities for Kinova Gen3 N6 reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gripper_tip_pose(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["gripper_tip"]),
) -> torch.Tensor:
    """Return gripper tip position and orientation (quat) in world frame."""

    asset = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if body_ids is None or len(body_ids) == 0:
        return torch.zeros(env.num_envs, 7, device=env.device)

    body_id = body_ids[0]
    pos = asset.data.body_pos_w[:, body_id]
    quat = asset.data.body_quat_w[:, body_id]
    return torch.cat((pos, quat), dim=1)

