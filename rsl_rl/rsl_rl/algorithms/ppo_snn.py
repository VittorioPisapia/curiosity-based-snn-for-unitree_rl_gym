import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.actor_critic import ActorCriticSNN
from rsl_rl.storage.rollout_storage_snn import RolloutStorage_Snn
from rsl_rl.algorithms.ppo import PPO

class PPO_Snn (PPO):
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
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.update_distribution(obs_batch, hidden_states=hid_states_batch, dones=masks_batch)

                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                actions_log_prob_batch = actions_log_prob_batch.flatten(0, 1)
                value_batch = value_batch.flatten(0, 1)
                mu_batch = mu_batch.flatten(0, 1)
                sigma_batch = sigma_batch.flatten(0, 1)
                entropy_batch = entropy_batch.flatten(0, 1)
                
                old_mu_batch = old_mu_batch.flatten(0, 1)
                old_sigma_batch = old_sigma_batch.flatten(0, 1)
                old_actions_log_prob_batch = old_actions_log_prob_batch.flatten(0, 1)
                advantages_batch = advantages_batch.flatten(0, 1)
                returns_batch = returns_batch.flatten(0, 1)
                target_values_batch = target_values_batch.flatten(0, 1)

                # advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()

                # # --- DEBUG DEI GRADIENTI ---
                # # Calcola la norma totale dei gradienti solo per l'Actor (la SNN)
                # actor_grads = [p.grad.detach() for p in self.actor_critic.actor.parameters() if p.grad is not None]
                # if len(actor_grads) > 0:
                #     grad_norm_pre_clip = torch.norm(torch.stack([torch.norm(g, 2) for g in actor_grads]), 2).item()
                #     # Stampa se il gradiente è sospetto (es. esploso o svanito)
                #     if grad_norm_pre_clip > 50.0 or grad_norm_pre_clip < 1e-4:
                #         print(f"ATTENZIONE! Norma gradiente SNN pre-clip anomala: {grad_norm_pre_clip:.4f}")
                # # ---------------------------


                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss