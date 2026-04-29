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

        envs_per_batch = self.num_envs // num_mini_batches

        for epoch in range(num_epochs):
            env_indices = torch.randperm(self.num_envs, requires_grad=False, device=self.device)

            for i in range(num_mini_batches):
                start = i * envs_per_batch
                end = (i+1) * envs_per_batch
                batch_env_idx = env_indices[start:end]

                obs_batch = self.observations[:, batch_env_idx, :]

                if self.privileged_observations is not None:
                    critic_observations_batch = self.privileged_observations[:, batch_env_idx, :]
                else:
                    critic_observations_batch = obs_batch
                
                actions_batch = self.actions[:, batch_env_idx, :]
                target_values_batch = self.values[:, batch_env_idx, :]
                returns_batch = self.returns[:, batch_env_idx, :]
                old_actions_log_prob_batch = self.actions_log_prob[:, batch_env_idx, :]
                advantages_batch = self.advantages[:, batch_env_idx, :]
                old_mu_batch = self.mu[:, batch_env_idx, :]
                old_sigma_batch = self.sigma[:, batch_env_idx, :]

                snn_m_batch_init = self.snn_m[0, batch_env_idx, :]
                snn_s_batch_init = self.snn_s[0, batch_env_idx, :]  

                hidden_states_batch = {
                "snn_m": snn_m_batch_init,
                "snn_s": snn_s_batch_init
                }

                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hidden_states_batch, None