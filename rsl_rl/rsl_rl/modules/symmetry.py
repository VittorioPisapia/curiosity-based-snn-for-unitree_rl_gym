# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable

from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage


class Symmetry:
    """Symmetry data augmentation and mirror loss.

    The extension supports two (optionally simultaneous) uses of a user-provided symmetry function:

    - :attr:`use_data_augmentation` appends mirrored observation/action pairs to every mini-batch, so that the policy
      and value loss are evaluated on both the original and the mirrored samples.
    - :attr:`use_mirror_loss` adds an auxiliary MSE term that penalizes the policy for disagreeing with itself when
      evaluated on mirrored observations.

    If both flags are disabled the symmetry loss is still computed for logging purposes but detached from the graph.

    References:
        - Mittal et al. "Symmetry Considerations for Learning Task Symmetric Robot Policies." ICRA (2024).
    """

    def __init__(
        self,
        env: VecEnv,
        # data_augmentation_func: str | Callable,
        use_data_augmentation: bool = False,
        use_mirror_loss: bool = False,
        mirror_loss_coeff: float = 0.0,
    ) -> None:
        """Initialize the symmetry extension.

        Args:
            env: Environment object. Passed to the data augmentation function for handling different observation terms.
            data_augmentation_func: Callable that generates mirrored observations / actions. Resolved using
                :func:`~rsl_rl.utils.utils.resolve_callable`.
            use_data_augmentation: Whether to append mirrored samples to every mini-batch.
            use_mirror_loss: Whether to add an auxiliary mirror loss term to the loss function.
            mirror_loss_coeff: Scaling factor applied to the mirror loss when :attr:`use_mirror_loss` is True.
        """
        # Symmetry parameters
        self.env = env
        self.use_data_augmentation = use_data_augmentation
        self.use_mirror_loss = use_mirror_loss
        self.mirror_loss_coeff = mirror_loss_coeff

        # Resolve the augmentation function
        self.data_augmentation_func = go2_data_augmentation

        # Inform the user if symmetry is configured only for logging
        if not (use_data_augmentation or use_mirror_loss):
            print("Symmetry not used for learning. We will use it for logging instead.")

    def augment_batch(self, batch: RolloutStorage.Batch, original_batch_size: int) -> None:
        """Augment the mini-batch in place with mirrored observations and actions.

        After the call ``batch.observations`` and ``batch.actions`` have shape ``[original_batch_size * num_aug, ...]``
        with the original samples in the first slice and the mirrored samples in the remaining slices. The remaining
        rollout tensors (old log probabilities, values, advantages, returns) are repeated to match.

        When :attr:`use_data_augmentation` is False, the batch is left unchanged.
        """
        if not self.use_data_augmentation:
            return
        # Returned shape: [original_batch_size * num_aug, ...]
        batch.observations, batch.actions = self.data_augmentation_func(
            env=self.env,
            obs=batch.observations,
            actions=batch.actions,
        )
        # Repeat the remaining rollout tensors to match the augmented observations/actions
        num_aug = int(batch.observations.shape[0] / original_batch_size)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)

    def compute_loss(self, actor, batch: RolloutStorage.Batch, original_batch_size: int,hid_states: dict = None) -> torch.Tensor:
        """Compute the mirror loss between the actor's action means on original and mirrored observations.

        If :meth:`augment_batch` has not been called for this batch (i.e. :attr:`use_data_augmentation` is False), the
        observations are augmented here first so that the actor is evaluated on both the original and the mirrored
        samples.

        The returned loss is detached when :attr:`use_mirror_loss` is False so that it can be reported for logging
        without contributing to gradients.
        """
        # Augment observations if the batch has not already been augmented
        if not self.use_data_augmentation:
            batch.observations, _ = self.data_augmentation_func(env=self.env, obs=batch.observations, actions=None)

        # Action means predicted by the actor on the augmented observation batch
        if hid_states is not None:
            # Calcoliamo le azioni usando lo stato interno clonato
            mean_actions, _ = actor(batch.observations.detach().clone(), hidden_states=hid_states)
        else:
            res = actor(batch.observations.detach().clone())
            mean_actions = res[0] if isinstance(res, tuple) else res

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


def resolve_symmetry_config(alg_cfg: dict, env: VecEnv) -> dict:
    """Resolve the symmetry configuration.

    Args:
        alg_cfg: Algorithm configuration dictionary.
        env: Environment object.

    Returns:
        The resolved algorithm configuration dictionary.
    """
    # If using symmetry then pass the environment object
    # Note: This is used by the symmetry function for handling different observation terms
    if "symmetry_cfg" in alg_cfg and alg_cfg["symmetry_cfg"] is not None:
        alg_cfg["symmetry_cfg"]["env"] = env
    else:
        alg_cfg["symmetry_cfg"] = None
    return alg_cfg

def go2_data_augmentation(env, obs=None, actions=None):
    full_obs = None
    full_actions = None

    # --- 1. Processo le Osservazioni (se presenti) ---
    if obs is not None:
        obs_sym = obs.clone()
        
        # Mirroring Vy, Roll rate, Yaw rate
        obs_sym[:, 1] *= -1   # vy
        obs_sym[:, 3] *= -1   # roll rate
        obs_sym[:, 5] *= -1   # yaw rate
        obs_sym[:, 7] *= -1   # gravity y
        obs_sym[:, 10] *= -1  # cmd_vy
        obs_sym[:, 11] *= -1  # cmd_yaw
        
        # Swap Gambe (Posizioni e Velocità)
        idx_pos_L = [12, 13, 14, 18, 19, 20] 
        idx_pos_R = [15, 16, 17, 21, 22, 23]
        idx_vel_L = [24, 25, 26, 30, 31, 32]
        idx_vel_R = [27, 28, 29, 33, 34, 35]
        
        obs_sym[:, idx_pos_L], obs_sym[:, idx_pos_R] = obs[:, idx_pos_R].clone(), obs[:, idx_pos_L].clone()
        obs_sym[:, idx_vel_L], obs_sym[:, idx_vel_R] = obs[:, idx_vel_R].clone(), obs[:, idx_vel_L].clone()
        
        # Inversione motori HIP
        hip_indices = [12, 15, 18, 21, 24, 27, 30, 33]
        obs_sym[:, hip_indices] *= -1
        
        # Concateno se necessario (per augment_batch) o restituisco solo lo specchiato (per compute_loss)
        # Il modulo si aspetta che se passiamo obs, restituiamo il batch raddoppiato
        full_obs = torch.cat([obs, obs_sym], dim=0)

    # --- 2. Processo le Azioni (se presenti) ---
    if actions is not None:
        actions_sym = actions.clone()
        # Swap FL/FR e RL/RR
        actions_sym[:, [0,1,2, 6,7,8]], actions_sym[:, [3,4,5, 9,10,11]] = \
            actions[:, [3,4,5, 9,10,11]].clone(), actions[:, [0,1,2, 6,7,8]].clone()
        # Inversione HIP azioni
        actions_sym[:, [0, 3, 6, 9]] *= -1
        
        # Se obs era None, stiamo solo specchiando le azioni per la loss
        # quindi restituiamo [originale, specchiato]
        full_actions = torch.cat([actions, actions_sym], dim=0)

    return full_obs, full_actions