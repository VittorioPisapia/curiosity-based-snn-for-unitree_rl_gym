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

        T = self.num_transitions_per_env
        N = self.num_envs

        # chunk temporale (TBPTT window)
        seq_len = T // num_mini_batches

        observations = self.observations  # [T, N, obs]
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations
        else:
            critic_observations = observations

        actions = self.actions
        values = self.values
        returns = self.returns
        old_actions_log_prob = self.actions_log_prob
        advantages = self.advantages
        old_mu = self.mu
        old_sigma = self.sigma

        snn_m = self.snn_m
        snn_s = self.snn_s

        for epoch in range(num_epochs):

            for i in range(num_mini_batches):

                start = i * seq_len
                end = (i + 1) * seq_len

                obs_batch = observations[start:end].reshape(-1, observations.shape[-1])
                critic_obs_batch = critic_observations[start:end].reshape(-1, critic_observations.shape[-1])

                actions_batch = actions[start:end].reshape(-1, actions.shape[-1])
                target_values_batch = values[start:end].reshape(-1, 1)
                returns_batch = returns[start:end].reshape(-1, 1)
                old_actions_log_prob_batch = old_actions_log_prob[start:end].reshape(-1, 1)
                advantages_batch = advantages[start:end].reshape(-1, 1)
                old_mu_batch = old_mu[start:end].reshape(-1, old_mu.shape[-1])
                old_sigma_batch = old_sigma[start:end].reshape(-1, old_sigma.shape[-1])


                snn_m_batch = snn_m[start:end].reshape(-1, snn_m.shape[-1])
                snn_s_batch = snn_s[start:end].reshape(-1, snn_s.shape[-1])

                hidden_states_batch = {
                    "snn_m": snn_m_batch,
                    "snn_s": snn_s_batch
                }

                yield (
                    obs_batch,
                    critic_obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    hidden_states_batch,
                    None
                )
