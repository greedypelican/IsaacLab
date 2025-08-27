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
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # rewarded if the object is lifted above the threshold
    return distance < threshold


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


def illegal_contact_excluding_bodies(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg,
    exclude_body_names: list[str] = []
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold, excluding specified bodies.
    
    Args:
        env: The environment.
        threshold: Force threshold for termination.
        sensor_cfg: Contact sensor configuration.
        exclude_body_names: List of body names to exclude from contact detection. 
                           Defaults to None (no exclusions).
    
    Returns:
        Boolean tensor indicating which environments should terminate (True = terminate).
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    # If no exclusions specified, use all bodies
    if exclude_body_names is None:
        body_ids = sensor_cfg.body_ids
    else:
        # Get body names from the sensor configuration
        all_body_names = sensor_cfg.body_names
        if all_body_names is None:
            # Fallback to using all body IDs if body_names is not available
            body_ids = sensor_cfg.body_ids
        else:
            # Filter out excluded body names
            filtered_body_names = [name for name in all_body_names if name not in exclude_body_names]
            # Get body IDs for filtered names
            body_ids = [i for i, name in enumerate(all_body_names) if name in filtered_body_names]
    
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
