# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command generators for Kinova Gen3 N6 reach task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DonutPoseCommand(UniformPoseCommand):
    """Pose command that samples from a rectangular region while excluding a sub-rectangle."""

    cfg: "DonutPoseCommandCfg"

    def __init__(self, cfg: "DonutPoseCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._previous_positions = torch.zeros(self.num_envs, 3, device=self.device)
        self._has_previous_sample = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        reset_ids = slice(None) if env_ids is None else env_ids
        if isinstance(reset_ids, slice):
            self._previous_positions[reset_ids] = 0.0
            self._has_previous_sample[reset_ids] = False
        else:
            reset_idx = torch.as_tensor(reset_ids, device=self.device, dtype=torch.long)
            if reset_idx.numel() > 0:
                self._previous_positions[reset_idx] = 0.0
                self._has_previous_sample[reset_idx] = False
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        super()._resample_command(env_ids)

        excl_x = self.cfg.exclusion_pos_x
        excl_y = self.cfg.exclusion_pos_y
        max_delta = self.cfg.max_position_delta

        max_iters = max(1, self.cfg.max_resample_iters)
        pending = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        def build_resample_mask(env_idx: torch.Tensor) -> torch.Tensor:
            mask = torch.zeros(env_idx.numel(), dtype=torch.bool, device=self.device)
            if env_idx.numel() == 0:
                return mask

            if excl_x is not None and excl_y is not None:
                positions_xy = self.pose_command_b[env_idx, :2]
                inside = (
                    (positions_xy[:, 0] >= excl_x[0])
                    & (positions_xy[:, 0] <= excl_x[1])
                    & (positions_xy[:, 1] >= excl_y[0])
                    & (positions_xy[:, 1] <= excl_y[1])
                )
                mask |= inside

            if max_delta is not None:
                prev_pos = self._previous_positions[env_idx]
                has_prev = self._has_previous_sample[env_idx]
                deltas = torch.norm(self.pose_command_b[env_idx, :3] - prev_pos, dim=1)
                mask |= has_prev & (deltas >= max_delta)

            return mask

        for _ in range(max_iters):
            if pending.numel() == 0:
                break

            resample_mask = build_resample_mask(pending)
            if not resample_mask.any():
                break

            resample_ids = pending[resample_mask].tolist()
            super()._resample_command(resample_ids)
            pending = pending[resample_mask]

        if pending.numel() == 0:
            self._previous_positions[env_ids] = self.pose_command_b[env_ids, :3]
            self._has_previous_sample[env_ids] = True
            return

        # Fallback: resample remaining environments individually up to max_iters times.
        for env_id in pending.tolist():
            prev_pos = self._previous_positions[env_id]
            has_prev = bool(self._has_previous_sample[env_id])
            for _ in range(max_iters):
                super()._resample_command([env_id])
                pos = self.pose_command_b[env_id, :3]

                inside = False
                if excl_x is not None and excl_y is not None:
                    pos_x = float(pos[0].item())
                    pos_y = float(pos[1].item())
                    inside = (
                        excl_x[0] <= pos_x <= excl_x[1]
                        and excl_y[0] <= pos_y <= excl_y[1]
                    )

                delta_violation = False
                if max_delta is not None and has_prev:
                    delta_violation = torch.norm(pos - prev_pos).item() >= max_delta

                if not inside and not delta_violation:
                    break

        self._previous_positions[env_ids] = self.pose_command_b[env_ids, :3]
        self._has_previous_sample[env_ids] = True


@configclass
class DonutPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for :class:`DonutPoseCommand`."""

    class_type: type = DonutPoseCommand

    exclusion_pos_x: tuple[float, float] | None = None
    """Excluded x-range (in meters)."""

    exclusion_pos_y: tuple[float, float] | None = None
    """Excluded y-range (in meters)."""

    max_position_delta: float | None = None
    """Maximum allowed positional delta (in meters) between consecutive samples before resampling."""

    max_resample_iters: int = 8
    """Maximum number of rejection sampling iterations."""
