# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class MetacombotxRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Default PPO recipe for the MetaCombotX velocity task on rough terrain."""

    num_steps_per_env = 32
    max_iterations = 2000
    save_interval = 50
    experiment_name = "metacombotx_velocity_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MetacombotxFlatPPORunnerCfg(MetacombotxRoughPPORunnerCfg):
    """Lighter PPO hyper-parameters that converge faster on planar terrains."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 600
        self.experiment_name = "metacombotx_velocity_flat"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
