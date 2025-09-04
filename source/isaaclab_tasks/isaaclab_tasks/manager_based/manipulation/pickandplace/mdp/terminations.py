# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.envs import ManagerBasedRLEnv

from .events import phase_flags

def time_out_with_phase_logging(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Time out termination with phase metrics logging.
    
    This function performs time out termination and logs phase metrics to extras.
    
    Args:
        env: The environment.
        
    Returns:
        Boolean tensor indicating which environments should terminate (True = terminate).
    """
    # Import here to avoid circular imports
    from .events import phase_flags
    
    # Check time out condition (same as original time_out function)
    time_out_condition = env.episode_length_buf >= env.max_episode_length
    
    # Log phase metrics when time out occurs
    if phase_flags and torch.any(time_out_condition):
        # Ensure extras and log exist
        if not hasattr(env, 'extras'):
            env.extras = {}
        if "log" not in env.extras:
            env.extras["log"] = {}
        
        # Add phase metrics to extras["log"]
        env.extras["log"]["Phases/phase_1_count"] = torch.sum(~phase_flags["phase1_complete"]).item()
        env.extras["log"]["Phases/phase_2_count"] = torch.sum(phase_flags["phase1_complete"] & ~phase_flags["phase2_complete"]).item()
        env.extras["log"]["Phases/phase_3_count"] = torch.sum(phase_flags["phase2_complete"] & ~phase_flags["phase3_complete"]).item()
        env.extras["log"]["Phases/phase_4_count"] = torch.sum(phase_flags["phase3_complete"] & ~phase_flags["phase4_complete"]).item()
        env.extras["log"]["Phases/phase_5_count"] = torch.sum(phase_flags["phase4_complete"]).item()
        
        # Debug print
        # print(f"Phase metrics logged: phase0={env.extras['log']['Metrics/phase_0_count']}, phase1={env.extras['log']['Metrics/phase_1_count']}, phase2={env.extras['log']['Metrics/phase_2_count']}")
    
    # Return the actual time out condition
    return time_out_condition


def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    x_bounds: tuple[float, float] = (-0.168, 0.715),
    y_bounds: tuple[float, float] = (-0.366, 0.366)
) -> torch.Tensor:
    """Termination condition for object going out of bounds relative to env origins.
    
    Args:
        env: The environment.
        object_cfg: Object asset configuration.
        x_bounds: Tuple of (min_x, max_x) bounds for object position relative to env origin.
        y_bounds: Tuple of (min_y, max_y) bounds for object position relative to env origin.
    
    Returns:
        Boolean tensor indicating which environments should terminate (True = terminate).
    """
    # Extract the object
    object: RigidObject = env.scene[object_cfg.name]
    
    # Get object position in world frame
    object_pos_w = object.data.root_pos_w  # (num_envs, 3)
    
    # Get environment origins
    env_origins = env.scene.env_origins  # (num_envs, 3)
    
    # Calculate object position relative to each environment's origin
    object_pos_rel = object_pos_w - env_origins  # (num_envs, 3)
    
    # Check if object is out of bounds relative to env origin
    x_out_of_bounds = (object_pos_rel[:, 0] < x_bounds[0]) | (object_pos_rel[:, 0] > x_bounds[1])
    y_out_of_bounds = (object_pos_rel[:, 1] < y_bounds[0]) | (object_pos_rel[:, 1] > y_bounds[1])
    
    # Terminate if object is out of bounds in either x or y direction
    out_of_bounds = x_out_of_bounds | y_out_of_bounds
    
    return out_of_bounds


def object_drop(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    height_threshold: float = 0.05,
    force_threshold: float = 0.01,
) -> torch.Tensor:
    """Terminate when object is dropped (contact lost while object is above height threshold).
    
    Args:
        env: The environment.
        object_cfg: Object asset configuration.
        height_threshold: Height threshold above which dropping is considered.
        force_threshold: Force threshold for contact detection.
    
    Returns:
        Boolean tensor indicating which environments should terminate (True = terminate).
    """
    # Check if phase_flags is initialized
    if not phase_flags or "phase1_complete" not in phase_flags:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Get object height
    object: RigidObject = env.scene[object_cfg.name]
    object_height = object.data.root_pos_w[:, 2]
    
    # Get contact sensors
    left_finger_sensor = env.scene.sensors["contact_forces_left_finger_pad"]
    right_finger_sensor = env.scene.sensors["contact_forces_right_finger_pad"]
    
    # Calculate contact forces
    left_forces = left_finger_sensor.data.force_matrix_w
    right_forces = right_finger_sensor.data.force_matrix_w
    
    left_forces_sum = torch.sum(left_forces, dim=(1, 2))
    right_forces_sum = torch.sum(right_forces, dim=(1, 2))
    left_force_magnitudes = torch.norm(left_forces_sum, dim=-1)
    right_force_magnitudes = torch.norm(right_forces_sum, dim=-1)
    
    # Check if both fingers are in contact
    left_contact = (left_force_magnitudes > force_threshold)
    right_contact = (right_force_magnitudes > force_threshold)
    both_contact = left_contact & right_contact
    
    # Object is dropped if:
    # 1. phase1_complete is True (object was successfully grasped)
    # 2. Object is above height threshold (was lifted)
    # 3. Both fingers lost contact (grasping failed after success)
    object_drop = phase_flags["phase1_complete"] & (object_height > height_threshold) & ~both_contact
    
    return object_drop