import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.actor_critic import ActorCriticSNN
from rsl_rl.storage.rollout_storage_snn import RolloutStorage_Snn

class PPO_Snn:
    actor_critic: ActorCriticSNN
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage_Snn.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage_Snn(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            spike_dim=self.actor_critic.actor.spike_dim,
            mem_dim=self.actor_critic.actor.mem_dim,
            device=self.device
        )

    def act(self, obs, critic_obs):
        prev_hidden_states = None
        if self.actor_critic.hidden_states is not None:
            prev_hidden_states = {
                "snn_m": self.actor_critic.hidden_states["snn_m"].detach().clone(),
                "snn_s": self.actor_critic.hidden_states["snn_s"].detach().clone(),
            }

        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()

        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.hidden_states = prev_hidden_states

        return self.transition.actions
    
