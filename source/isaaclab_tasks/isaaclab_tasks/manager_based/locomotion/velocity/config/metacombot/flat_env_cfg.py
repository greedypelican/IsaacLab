# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import MetacombotxVelocityRoughEnvCfg


@configclass
class MetacombotxVelocityFlatEnvCfg(MetacombotxVelocityRoughEnvCfg):
    """Plane-only variant that removes terrain curriculum and height scans."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # use a flat plane, keep the spacing large enough for the arm
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.max_init_terrain_level = None

        # no height map sensing on flat ground
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # curriculum provides no value on a plane
        self.curriculum.terrain_levels = None


@configclass
class MetacombotxVelocityFlatEnvCfg_PLAY(MetacombotxVelocityFlatEnvCfg):
    """Convenience preset for quick policy evaluation."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.num_envs = 64
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
