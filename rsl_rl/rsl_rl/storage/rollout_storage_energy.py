import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class RolloutStorage_Snn_energy:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.energy_target = None
            self.next_observations = None
                    
        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, spike_dim, mem_dim, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.snn_m = torch.zeros(num_transitions_per_env, num_envs, mem_dim, device=self.device)
        self.snn_s = torch.zeros(num_transitions_per_env, num_envs, spike_dim, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env,num_envs,*obs_shape,device=self.device)
        self.energy_target = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

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
        self.next_observations[self.step].copy_(transition.next_observations)
        self.energy_target[self.step].copy_(transition.energy_target)
        if transition.hidden_states is not None:
            self.snn_m[self.step].copy_(transition.hidden_states["snn_m"])
            self.snn_s[self.step].copy_(transition.hidden_states["snn_s"])

        #self.hidden_states[self.step].copy(transition.hidden_states)
        #self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()


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
        energy_target = self.energy_target
        next_observations = self.next_observations

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
                cot_batch = energy_target[start:end].reshape(-1, energy_target.shape[-1])
                next_observations_batch = next_observations[start:end].reshape(-1, next_observations.shape[-1])

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
                    None,
                    cot_batch,
                    next_observations_batch
                )
