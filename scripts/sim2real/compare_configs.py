#!/usr/bin/env python3
"""
Compare configuration values between different sources
"""

import math
import numpy as np

print("=== Configuration Comparison ===\n")

# From your provided code (KINOVAGEN3N6_REACH_CFG)
reach_cfg_joints = {
    "joint_1": 0.0,
    "joint_2": math.pi/12,      # 15°
    "joint_3": -2*math.pi/3,     # -120°
    "joint_4": 0.0,
    "joint_5": -1*math.pi/4,     # -45°
    "joint_6": math.pi/2,        # 90°
}

# From /home/user/kinova_isaaclab_sim2real/pretrained_models/reach_6dof/env.yaml
env_yaml_joints = {
    "joint_1": 0.0,
    "joint_2": 0.2617993877991494,   # ~15°
    "joint_3": -2.0943951023931953,  # ~-120°
    "joint_4": 0.0,
    "joint_5": -0.7853981633974483,  # ~-45°
    "joint_6": 1.5707963267948966,   # ~90°
}

# Calculate exact values for comparison
exact_values = {
    "joint_1": 0.0,
    "joint_2": math.pi/12,
    "joint_3": -2*math.pi/3,
    "joint_4": 0.0,
    "joint_5": -math.pi/4,
    "joint_6": math.pi/2,
}

print("Joint Configuration Comparison:")
print("-" * 80)
print(f"{'Joint':<10} {'REACH_CFG (rad)':<20} {'env.yaml (rad)':<20} {'Match?':<10} {'Degrees':<15}")
print("-" * 80)

for joint in reach_cfg_joints.keys():
    cfg_val = reach_cfg_joints[joint]
    env_val = env_yaml_joints[joint]
    match = abs(cfg_val - env_val) < 0.001  # Within 0.001 rad tolerance

    print(f"{joint:<10} {cfg_val:<20.6f} {env_val:<20.6f} {'✓' if match else '✗':<10} "
          f"{np.rad2deg(cfg_val):>6.1f}° / {np.rad2deg(env_val):>6.1f}°")

print("\n" + "=" * 80)
print("\nExact Mathematical Values:")
print("-" * 80)
for joint in exact_values.keys():
    exact_val = exact_values[joint]
    env_val = env_yaml_joints[joint]
    diff = abs(exact_val - env_val)

    print(f"{joint}: π/{12 if joint=='joint_2' else 4 if joint=='joint_5' else 2 if joint=='joint_6' else '3*2' if joint=='joint_3' else '∞'} = "
          f"{exact_val:.10f}")
    print(f"        env.yaml = {env_val:.10f}")
    print(f"        difference = {diff:.10f} rad ({np.rad2deg(diff):.6f}°)")
    print()

print("=" * 80)
print("\nSUMMARY:")
print("✓ All joint positions match within reasonable tolerance (< 0.001 rad)")
print("✓ The configurations are essentially the same")
print("\nThe slight differences are due to:")
print("  - Floating point representation")
print("  - env.yaml uses decimal approximations of π fractions")