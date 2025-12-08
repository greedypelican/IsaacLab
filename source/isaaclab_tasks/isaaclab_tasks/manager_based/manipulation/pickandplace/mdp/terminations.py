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
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from isaaclab.envs import ManagerBasedRLEnv

from .events import phase_flags, GRASP_THRESHOLD, RELEASE_THRESHOLD

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
        env.extras["log"]["Phases/phase_5_count"] = torch.sum(phase_flags["phase4_complete"] & ~phase_flags["phase5_complete"]).item()
        env.extras["log"]["Phases/phase_6_count"] = torch.sum(phase_flags["phase5_complete"]).item()
        
        # Debug print
        # print(f"Phase metrics logged: phase0={env.extras['log']['Metrics/phase_0_count']}, phase1={env.extras['log']['Metrics/phase_1_count']}, phase2={env.extras['log']['Metrics/phase_2_count']}")
    
    # Return the actual time out condition
    return time_out_condition


def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    x_bounds: tuple[float, float] = (-0.3, 0.3),
    y_bounds: tuple[float, float] = (-0.3, 0.3)
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
    contact_threshold: float = RELEASE_THRESHOLD,
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
    left_contact = (left_force_magnitudes >= contact_threshold)
    right_contact = (right_force_magnitudes >= contact_threshold)
    both_contact = left_contact & right_contact
    
    # Drop only after grasp (phase1 complete) but before phase3 completes (i.e., up to and including descend stage)
    object_drop = (phase_flags["phase1_complete"] & ~phase_flags["phase3_complete"] & (object_height > height_threshold) & ~both_contact)
    return object_drop

# def object_move_after_place(
#     env: ManagerBasedRLEnv,
#     move_threshold: float = 0.2,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Terminate after Phase 3 if object XY is farther than threshold from descend command (robot root frame)."""
#     if not phase_flags or "phase3_complete" not in phase_flags:
#         return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

#     robot = env.scene[robot_cfg.name]
#     obj = env.scene[object_cfg.name]

#     descend_command = env.command_manager.get_command("descend")  # (N, ≥3) in robot frame
#     des_pos_b = descend_command[:, :3]

#     obj_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w[:, :3])

#     dist_xy_b = torch.linalg.norm(obj_pos_b[:, :2] - des_pos_b[:, :2], dim=1)
#     terminate = phase_flags["phase3_complete"] & (dist_xy_b > move_threshold)
#     return terminate

def object_move_after_place(
    env: ManagerBasedRLEnv,
    move_threshold: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),

) -> torch.Tensor:
    """Terminate if, after Phase 5 success, the object moves in XY more than a threshold in the ROBOT ROOT frame.

    Compares current object position in robot root frame vs. the cached position (also in robot root frame)
    captured when Phase 5 first completed.
    """
    # Validate phase flags
    if not phase_flags or "phase5_complete" not in phase_flags:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # If the cache does not exist yet, no termination (nothing to compare against)
    if not hasattr(env, "_object_pos_b_at_phase5"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Assets
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    # Current object position in robot root frame
    obj_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w[:, :3])
    cached_xy = env._object_pos_b_at_phase5[:, :2]
    current_xy = obj_pos_b[:, :2]

    # Displacement in XY in robot frame
    disp_xy = torch.linalg.norm(current_xy - cached_xy, dim=1)

    # Trigger only after Phase 3 is latched
    terminate = phase_flags["phase5_complete"] & (disp_xy > move_threshold)
    return terminate
