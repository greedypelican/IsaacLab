# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_mul
from isaaclab.envs import ManagerBasedRLEnv

# Import phase_flags from events module
from .events import phase_flags

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the object in the robot's root frame as quaternion."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # object의 quaternion을 world frame에서 가져오기
    object_quat_w = object.data.root_quat_w
    
    # robot의 root quaternion의 inverse 계산
    robot_quat_w = robot.data.root_quat_w
    robot_quat_w_inv = torch.cat([robot_quat_w[:, 0:1], -robot_quat_w[:, 1:]], dim=1)
    
    # object orientation을 robot root frame으로 변환
    # q_object_b = q_robot_inv * q_object_w * q_robot_inv_conj
    object_quat_b = quat_mul(quat_mul(robot_quat_w_inv, object_quat_w), robot_quat_w)
    
    return object_quat_b


def current_phase(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Return the current phase information for the agent to observe.
    
    Returns:
        Tensor with shape (num_envs, 1) containing phase information:
        0.0: pick phase (아직 phase1_complete가 False)
        1.0: ascend phase (phase1_complete가 True, phase2_complete가 False)
        2.0: descend phase (phase2_complete가 True, phase3_complete가 False)
        3.0: place phase (phase3_complete가 True, phase4_complete가 False)
        4.0: goback phase (phase4_complete가 True)
    """
    # Phase flags가 초기화되지 않았으면 기본값 반환
    if not phase_flags or "phase1_complete" not in phase_flags:
        return torch.zeros(env.num_envs, 1, device=env.device)

    # Phase flags를 기반으로 현재 phase 계산
    phase = torch.zeros(env.num_envs, 1, device=env.device)
    phase[phase_flags["phase1_complete"]] = 1.0
    phase[phase_flags["phase2_complete"]] = 2.0
    phase[phase_flags["phase3_complete"]] = 3.0
    phase[phase_flags["phase4_complete"]] = 4.0
    
    return phase
