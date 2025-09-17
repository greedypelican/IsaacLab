#!/usr/bin/env python3
"""
Debug script to check 6DOF default positions and action scaling
"""

import numpy as np

# From env.yaml (6DOF model)
default_pos_from_env = np.array([0.0, 0.2618, -2.0944, 0.0, -0.7854, 1.5708])

# Current robot position (from your output)
current_pos = default_pos_from_env + np.array([-0.1182, 1.0771, -2.8322, 3.2922, -2.9505, 5.2167])

print("=== 6DOF Debug Info ===\n")
print(f"Default pos from env.yaml:")
print(f"  Radians: {np.round(default_pos_from_env, 4)}")
print(f"  Degrees: {np.round(np.rad2deg(default_pos_from_env), 1)}")

print(f"\nCurrent robot position:")
print(f"  Radians: {np.round(current_pos, 4)}")
print(f"  Degrees: {np.round(np.rad2deg(current_pos), 1)}")

print(f"\nDelta (current - default):")
delta = current_pos - default_pos_from_env
print(f"  Radians: {np.round(delta, 4)}")
print(f"  Degrees: {np.round(np.rad2deg(delta), 1)}")

# Problem: The model outputs actions that when scaled become too large
raw_action = np.array([-0.2362, 2.1548, -5.6678, 6.5877, -5.9024, 10.4385])
action_scale = 0.5
processed_action = default_pos_from_env + (raw_action * action_scale)

print(f"\n=== Action Processing ===")
print(f"Raw action: {np.round(raw_action, 4)}")
print(f"Action scale: {action_scale}")
print(f"Processed action (default + raw*scale):")
print(f"  Radians: {np.round(processed_action, 4)}")
print(f"  Degrees: {np.round(np.rad2deg(processed_action), 1)}")

print(f"\n⚠️  PROBLEMS DETECTED:")
print(f"1. Joint 6 processed action: {processed_action[5]:.2f} rad = {np.rad2deg(processed_action[5]):.0f}°")
print(f"   This exceeds 360°! Physical limits are typically ±180° or ±360°")
print(f"2. Joint 3 processed action: {processed_action[2]:.2f} rad = {np.rad2deg(processed_action[2]):.0f}°")
print(f"   This is -282°, likely causing collision")

print(f"\n=== Suggested Fix ===")
print(f"1. Reduce action_scale from 0.5 to something smaller (e.g., 0.1)")
print(f"2. Or clamp the final joint positions to physical limits")
print(f"3. Check if the model was trained with different default positions")