import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.actor_critic import ActorCriticSNN
from rsl_rl.storage.rollout_storage_energy import RolloutStorage_Snn_energy
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.energy_model import EnergyForwardModel
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.modules.normalization import EmpiricalNormalization, EmpiricalDiscountedVariationNormalization
from rsl_rl.modules.symmetry import DummyBatch, Symmetry

class PPO_energy (PPO):
    actor_critic: ActorCriticSNN
    def __init__(self,
                 env,
                 actor_critic,
                 use_rnd,
                 rnd,
                 use_symmetry,
                 symmetry,
                 use_spike_loss,
                 spike_loss_coeff,
                 spike_rate_target,
                 energy,
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
        self.env = env

        # Energy Model
        self.energy_model = EnergyForwardModel(
                obs_dim=48,      
                action_dim=12,
                device=self.device,
                **energy
            ).to(self.device)
        self.energy_optimizer = optim.Adam(
                self.energy_model.parameters(),
                lr=learning_rate
                )
        
        self.update_counter = 0
        self.warmup_updates = 500   
        self.target_cot_coeff = 0.2

        # Symmetry
        self.use_symmetry = use_symmetry
        self.symmetry_module = Symmetry(env=self.env, **symmetry) if use_symmetry else None
        
        # Spike Loss
        self.use_spike_loss = use_spike_loss
        self.target_rates = spike_rate_target
        self.spike_loss_coeff = spike_loss_coeff

        # RND
        self.use_rnd = use_rnd
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd) if self.use_rnd else None

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage_Snn_energy.Transition()

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
        self.storage = RolloutStorage_Snn_energy(
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
    
    def process_env_step(self, next_obs, rewards, dones, infos, energy_target):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_observations = next_obs.clone()
        self.transition.energy_target = energy_target.clone()
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def update(self):

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_energy_loss = 0
        mean_spike_loss = 0 if self.use_spike_loss else None
        mean_symmetry_loss = 0 if self.use_symmetry else None
        mean_rnd_loss = 0 if self.use_rnd else None

        self.update_counter += 1

        if self.update_counter < self.warmup_updates:
            cot_optimization_coeff = 0.0 
        else:
            rampa = min(1.0, (self.update_counter - self.warmup_updates) / 50)
            cot_optimization_coeff = self.target_cot_coeff * rampa

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, cot_batch, next_obs_batch in generator:

                batch = DummyBatch(
                    observations=obs_batch,
                    actions=actions_batch,
                    old_actions_log_prob=old_actions_log_prob_batch,
                    values=target_values_batch,
                    advantages=advantages_batch,
                    returns=returns_batch
                )

                original_size = obs_batch.shape[0]
                if self.symmetry_module.use_data_augmentation:
                    batch = self.symmetry_module.augment_batch(batch)

                    num_aug = batch.observations.shape[0] // original_size
                    if hid_states_batch is not None:
                        hid_states_batch["snn_m"] = hid_states_batch["snn_m"].repeat(num_aug, 1)
                        hid_states_batch["snn_s"] = hid_states_batch["snn_s"].repeat(num_aug, 1)
                    
                    if critic_obs_batch.shape[0] == original_size:
                         critic_obs_batch = critic_obs_batch.repeat(num_aug, 1)

               
                self.actor_critic.update_distribution(batch.observations, hidden_states=hid_states_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(batch.actions)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean 
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                mu_batch_orig = mu_batch[:original_size]
                sigma_batch_orig = sigma_batch[:original_size]
                entropy_batch_orig = entropy_batch[:original_size]

                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch_orig / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch_orig)) / (2.0 * torch.square(sigma_batch_orig)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                self.energy_optimizer.zero_grad()
                loss_energy = self.energy_model.compute_loss(
                    obs_batch,
                    actions_batch,
                    next_obs_batch,
                    cot_batch
                )
                loss_energy.backward()
                self.energy_optimizer.step()

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                
                # Symmetry loss
                if self.symmetry_module:
                    symmetry_loss = self.symmetry_module.compute_loss(self.actor_critic.actor, batch, original_size, hidden_states=hid_states_batch)
                    if self.symmetry_module.use_mirror_loss:
                        loss = loss + self.symmetry_module.mirror_loss_coeff * symmetry_loss

                if self.use_rnd:
                    obs_lin_vel    = obs_batch[:, 0:3]
                    obs_ang_vel    = obs_batch[:, 3:6]
                    obs_proj_grav  = obs_batch[:, 6:9]
                    obs_commands   = obs_batch[:, 9:12]
                    obs_dof_pos    = obs_batch[:, 12:24]
                    obs_dof_vel    = obs_batch[:, 24:36]
                    obs_last_act   = obs_batch[:, 36:48] 

                    obs_batch_for_rnd = torch.cat((obs_lin_vel,obs_dof_pos,obs_dof_vel), dim=-1)    #+187 add it for height measurements
                    
                    #normalized_obs_batch = self.state_normalizer(obs_batch_for_rnd)
                    
                    rnd_loss = self.rnd.compute_loss(obs_batch_for_rnd)
                else:
                    rnd_loss = None

                if cot_optimization_coeff > 0.0:
                    predicted_cot = self.energy_model.predict_energy(batch.observations, mu_batch)
                    direct_cot_loss = predicted_cot.mean()
                    loss = loss + (direct_cot_loss * cot_optimization_coeff)


                if self.use_spike_loss:
                    spike_rates = self.actor_critic.actor.last_layer_spikes
                    spike_loss = 0.0
                    for l, s in enumerate(spike_rates):
                        rate = s.mean()
                        spike_loss += (rate - self.target_rates[l]) ** 2
                    spike_loss /= len(spike_rates)
                    
                    loss = loss + spike_loss * self.spike_loss_coeff
                

                self.optimizer.zero_grad()
                loss.backward()
                
                if self.use_rnd:
                    self.rnd.optimizer.zero_grad()
                    rnd_loss.backward()

                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.use_rnd:
                    self.rnd.optimizer.step()
            

                # Log metrics
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_energy_loss += loss_energy.item()
                if mean_spike_loss is not None:
                    mean_spike_loss += spike_loss.item()
                if mean_rnd_loss is not None:
                    mean_rnd_loss += rnd_loss.item()
                if mean_symmetry_loss is not None:
                    mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_energy_loss /= num_updates
        if mean_spike_loss is not None:
            mean_spike_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_symmetry_loss, mean_spike_loss, mean_energy_loss, mean_rnd_loss