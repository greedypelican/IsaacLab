#!/usr/bin/env python3
"""
gen3.py
--------------------

Thin wrapper around a pre-trained reach policy for the Kinova Gen3 arm.
Extends `PolicyController` with:

* State update via `update_joint_state()`
* Forward pass (`forward`) that returns a target joint-position command
  every call, computing a new action every ``decimation`` steps.

Change the default paths below or pass them explicitly in the constructor.

Author: Louis Le Lay
"""

from pathlib import Path

import numpy as np

from controllers.policy_controller import PolicyController

class Gen3ReachPolicy(PolicyController):
    """Policy controller for Gen3 Reach using a pre-trained policy model."""

    def __init__(self) -> None:
        """Initialize the URReachPolicy instance."""
        super().__init__()
        self.dof_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]
        # Load the pre-trained policy model and environment configuration
        repo_root = Path(__file__).resolve().parents[3]
        model_dir = repo_root / "pretrained_models" / "reach"
        self.load_policy(
            model_dir / "policy.pt",
            model_dir / "env.yaml",
        )

        self._action_scale = 0.5
        self._previous_action = np.zeros(7)
        self._policy_counter = 0
        self.target_command = np.array([0.5, 0.0, 0.2, 0.7071, 0.0, 0.7071, 0.0])

        self.has_joint_data = False
        self.current_joint_positions = np.zeros(7)
        self.current_joint_velocities = np.zeros(7)

    def update_joint_state(self, position, velocity) -> None:
        """
        Update the current joint state.

        Args:
            position: A list or array of joint positions.
            velocity: A list or array of joint velocities.
        """
        self.current_joint_positions = np.array(position[:self.num_joints], dtype=np.float32)
        self.current_joint_velocities = np.array(velocity[:self.num_joints], dtype=np.float32)
        self.has_joint_data = True

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        """
        Compute the observation vector for the policy network.

        Args:
            command: The target command vector.

        Returns:
            An observation vector if joint data is available, otherwise None.
        """
        if not self.has_joint_data:
            return None
        obs = np.zeros(28)
        obs[:7] = self.current_joint_positions - self.default_pos
        obs[7:14] = self.current_joint_velocities
        obs[14:21] = command
        obs[21:28] = self._previous_action
        return obs

    def forward(self, dt: float, command: np.ndarray) -> np.ndarray:
        """
        Compute the next joint positions based on the policy.

        Args:
            dt: Time step for the forward pass.
            command: The target command vector.

        Returns:
            The computed joint positions if joint data is available, otherwise None.
        """
        if not self.has_joint_data:
            return None

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            if obs is None:
                return None
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

            # Debug Logging (commented out)
            print("\n=== Policy Step ===")
            print(f"{'Command:':<20} {np.round(command, 4)}\n")
            print("--- Observation ---")
            print(f"{'Δ Joint Positions:':<20} {np.round(obs[:6], 4)}")
            print(f"{'Joint Velocities:':<20} {np.round(obs[6:12], 4)}")
            print(f"{'Command:':<20} {np.round(obs[12:19], 4)}")
            print(f"{'Previous Action:':<20} {np.round(obs[19:25], 4)}\n")
            print("--- Action ---")
            print(f"{'Raw Action:':<20} {np.round(self.action, 4)}")
            processed_action = self.default_pos + (self.action * self._action_scale)
            print(f"{'Processed Action:':<20} {np.round(processed_action, 4)}")

        joint_positions = self.default_pos + (self.action * self._action_scale)
        self._policy_counter += 1
        return joint_positions