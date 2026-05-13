from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable

from rsl_rl.env import VecEnv
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.storage import RolloutStorage

from dataclasses import dataclass

class Symmetry:
    def __init__(
            self,
            env: VecEnv,
            use_data_augmentation : bool = False,
            use_mirror_loss : bool = False,
            mirror_loss_coeff : float = 0.0,
    ) -> None:
        
        self.env = env
        self.use_data_augmentation = use_data_augmentation
        self.use_mirror_loss = use_mirror_loss
        self.mirror_loss_coeff = mirror_loss_coeff

        if not (use_data_augmentation or use_mirror_loss):
            print("Symmetry not used for learning. We will use it for logging instead.")

    def augment_batch(self, batch: DummyBatch) -> DummyBatch:
        if not self.use_data_augmentation:
            return
        
        original_batch_size = batch.observations.shape[0]

        obs_aug, actions_aug = self.use_data_augmentation(env=self.env, obs=batch.observations, actions=batch.actions)

        num_aug = int(obs_aug[0] / original_batch_size)

        batch.observations = obs_aug
        batch.actions = actions_aug

        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)

    def compute_loss(self, actor:ActorCritic.actor, batch:RolloutStorage.Batch, original_batch_size : int) -> torch.Tensor:
        if not self.use_data_augmentation:
            batch.observations, _ = self.data_augmentation_func(env=self.env, obs=batch.observation, actions=None)

        mean_actions = actor(batch.observations.detach().clone())

        # Mirror the original-slice action means using the augmentation function. We use the action means here rather
        # than the sampled actions in ``batch.actions``, since the symmetry loss is defined on the policy mean.
        _, mean_actions_symm = self.data_augmentation_func(
            env=self.env, obs=None, actions=mean_actions[:original_batch_size]
        )

        # MSE between the actor prediction on mirrored obs and the mirrored actor prediction on the original obs
        symmetry_loss = nn.functional.mse_loss(
            mean_actions[original_batch_size:],
            mean_actions_symm.detach()[original_batch_size:],
        )
        return symmetry_loss if self.use_mirror_loss else symmetry_loss.detach()

def data_augmentation_func(self, env, obs=None, actions=None):
    return obs, actions

@dataclass
class DummyBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_actions_log_prob: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor