import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories
from .rollout_storage import RolloutStorage

class RolloutStorage_Snn ( RolloutStorage ):
    
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, spike_dim, mem_dim, device='cpu'):

        super().__init__(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device)
        self.snn_m = torch.zeros(num_transitions_per_env, num_envs, mem_dim, device=self.device)
        self.snn_s = torch.zeros(num_transitions_per_env, num_envs, spike_dim, device=self.device)

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None: self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        if transition.hidden_states is not None:
            self.snn_m[self.step].copy_(transition.hidden_states["snn_m"])
            self.snn_s[self.step].copy_(transition.hidden_states["snn_s"])

        #self.hidden_states[self.step].copy(transition.hidden_states)
        #self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        snn_m = self.snn_m.flatten(0,1)
        snn_s = self.snn_s.flatten(0,1)


        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                snn_m_batch = snn_m[batch_idx]
                snn_s_batch = snn_s[batch_idx]
                hidden_states_batch = {
                    "snn_m": snn_m_batch,
                    "snn_s": snn_s_batch
                }

                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hidden_states_batch, None
