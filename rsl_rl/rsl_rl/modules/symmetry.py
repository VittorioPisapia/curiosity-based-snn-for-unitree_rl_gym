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
            return batch
        
        original_batch_size = batch.observations.shape[0]

        obs_aug, actions_aug = self.data_augmentation_func(env=self.env, obs=batch.observations, actions=batch.actions)

        num_aug = int(obs_aug.shape[0] / original_batch_size)

        batch.observations = obs_aug
        batch.actions = actions_aug

        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)
        
        return batch

    def compute_loss(
        self,
        actor,
        batch,
        original_batch_size,
        hidden_states
    ):

        if not self.use_data_augmentation:
            augmented_obs, _ = self.data_augmentation_func(
                env=self.env,
                obs=batch.observations,
                actions=None
            )
        else:
            augmented_obs = batch.observations

        mean_actions, _ = actor(
            augmented_obs.detach(),
            hidden_states=None
        )

        _, mean_actions_symm = self.data_augmentation_func(
            env=self.env,
            obs=None,
            actions=mean_actions[:original_batch_size]
        )

        symmetry_loss = nn.functional.mse_loss(
            mean_actions[original_batch_size:],
            mean_actions_symm.detach()[original_batch_size:]
        )

        return symmetry_loss if self.use_mirror_loss else symmetry_loss.detach()

    def data_augmentation_func(self, env, obs=None, actions=None):
        swap_joint_indices = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        negate_joints_indices = [0, 3, 6, 9] 

        aug_obs = None
        aug_actions = None

        if obs is not None:
            aug_obs = obs.clone()
            
            if obs.shape[1] >= 48:
                # Configurazione a 48 (vale sia per obs standard a 48, sia per privileged_obs a 48)
                aug_obs[:, 1] *= -1.0   # Base Lin Vel Y
                aug_obs[:, 3] *= -1.0   # Base Ang Vel Roll (X)
                aug_obs[:, 5] *= -1.0   # Base Ang Vel Yaw (Z)
                aug_obs[:, 7] *= -1.0   # Projected Gravity Y
                aug_obs[:, 10] *= -1.0  # Command Y
                aug_obs[:, 11] *= -1.0  # Command Yaw
                
                joint_starts = [12, 24, 36]
                h_start = 48

            elif obs.shape[1] >= 45:
                # Configurazione a 45 obs (Actor input nel caso asimmetrico)
                aug_obs[:, 0] *= -1.0   # Base Ang Vel Roll (X)
                aug_obs[:, 2] *= -1.0   # Base Ang Vel Yaw (Z)
                aug_obs[:, 4] *= -1.0   # Projected Gravity Y
                aug_obs[:, 7] *= -1.0   # Command Y
                aug_obs[:, 8] *= -1.0   # Command Yaw
                
                joint_starts = [9, 21, 33]
                h_start = 45

            # --- Joints & Previous Actions ---
            for start_idx in joint_starts:
                leg_data = obs[:, start_idx : start_idx + 12]
                mirrored_legs = leg_data[:, swap_joint_indices].clone()
                mirrored_legs[:, negate_joints_indices] *= -1.0
                aug_obs[:, start_idx : start_idx + 12] = mirrored_legs

            # --- Height Scans ---
            if self.env.cfg.terrain.measure_heights:
                num_rows = 17 # measured_points_x
                num_cols = 11 # measured_points_y
                
                heights = obs[:, h_start : h_start + (num_rows * num_cols)]
                heights_grid = heights.view(-1, num_rows, num_cols)
                flipped_heights = torch.flip(heights_grid, dims=[2]) 
                aug_obs[:, h_start : h_start + (num_rows * num_cols)] = flipped_heights.reshape(obs.shape[0], -1)

        if actions is not None:
            aug_actions = actions.clone()
            aug_actions = aug_actions[:, swap_joint_indices]
            aug_actions[:, negate_joints_indices] *= -1.0

        if obs is not None:
            aug_obs = torch.cat([obs, aug_obs], dim=0)

        if actions is not None:
            aug_actions = torch.cat([actions, aug_actions], dim=0)

        return aug_obs, aug_actions


@dataclass
class DummyBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_actions_log_prob: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor