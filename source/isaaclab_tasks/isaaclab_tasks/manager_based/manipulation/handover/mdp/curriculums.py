from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class modify_reward_weight_multi_stage(ManagerTermBase):
    """Curriculum that modifies the reward weight based on multiple step-wise schedules."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)
        
        # Multi-stage parameters
        self.num_steps_1 = cfg.params.get("num_steps_1", 0)
        self.num_steps_2 = cfg.params.get("num_steps_2", 0)
        self.weight_1 = cfg.params.get("weight_1", 1.0)
        self.weight_2 = cfg.params.get("weight_2", 1.0)

        # Track current stage to avoid redundant checks
        self._current_stage = 0

    def __call__(
        self, 
        env: ManagerBasedRLEnv, 
        env_ids: Sequence[int], 
        term_name: str, 
        num_steps_1: int, 
        num_steps_2: int, 
        weight_1: float, 
        weight_2: float, 
    ) -> float:
        current_step = env.common_step_counter
        
        if self._current_stage == 0 and current_step > self.num_steps_1:
            self._term_cfg.weight = self.weight_1
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)
            self._current_stage = 1
        
        elif self._current_stage == 1 and current_step > self.num_steps_2:
            self._term_cfg.weight = self.weight_2
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)
            self._current_stage = 2

        return self._term_cfg.weight