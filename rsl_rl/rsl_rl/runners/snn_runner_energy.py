import time
import os
from collections import deque
import statistics

import torch

from .on_policy_runner import OnPolicyRunner
from rsl_rl.modules.actor_critic import ActorCriticSNN
from rsl_rl.algorithms.ppo_energy import PPO_energy
from rsl_rl.env import VecEnv
from torch.utils.tensorboard import SummaryWriter

class SnnRunner_energy ( OnPolicyRunner ):
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        #self.use_symmetry = self.alg_cfg["use_symmetry"]
        #self.symmetry_cfg = self.alg_cfg["symmetry"]
#
        #self.use_spike_loss = self.alg_cfg["use_spike_loss"]
        #self.spike_loss_coeff = self.alg_cfg["spike_loss_coeff"]
        #self.spike_rate_target = self.alg_cfg["spike_rate_target"]

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCriticSNN = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO_energy = alg_class(env=self.env, actor_critic=actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        self.alg.actor_critic.hidden_states = None
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        use_energy = self.cfg.get("use_energy", True)

        lookahead_warmup_iters = self.cfg.get("lookahead_warmup_iters", 250)
        lookahead_ramp_iters = self.cfg.get("lookahead_ramp_iters", 100) # Quante iterazioni dura il ramp-up (es. da 250 a 350)
        max_alpha = self.cfg.get("lookahead_max_alpha", 0.2)             # Peso massimo finale del lookahead (20%)
        noise_scale = self.cfg.get("lookahead_noise_scale", 0.2)         # Restringe il raggio di esplorazione attorno alla policy
        lambda_reg = self.cfg.get("lookahead_lambda_reg", 0.5)           # Penalizza azioni troppo distanti dalla policy nominale
        num_candidates = self.cfg.get("lookahead_num_candidates", 5)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            
            if it < lookahead_warmup_iters:
                alpha = 0.0
            elif it < (lookahead_warmup_iters + lookahead_ramp_iters):
                alpha = max_alpha * (it - lookahead_warmup_iters) / lookahead_ramp_iters
            else:
                alpha = max_alpha

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    if use_energy and alpha > 0.0:
                        num_envs = self.env.num_envs
                        num_actions = self.env.num_actions
                        
                        nominal_actions = actions.clone()

                        # Generazione candidati controllata
                        actions_expanded = nominal_actions.unsqueeze(1).repeat(1, num_candidates, 1)
                        action_std = self.alg.actor_critic.std
                        noise = torch.randn_like(actions_expanded) * action_std.unsqueeze(0).unsqueeze(0) * noise_scale
                        candidate_actions = torch.clamp(actions_expanded + noise, -1.0, 1.0)

                        # Predizione dell'energia latente
                        flat_candidates = candidate_actions.view(-1, num_actions)
                        z = self.alg.energy_model.encode(obs)
                        z_flat = z.unsqueeze(1).repeat(1, num_candidates, 1).view(-1, z.shape[-1])

                        x = torch.cat([z_flat, flat_candidates], dim=-1)
                        z_next_pred = self.alg.energy_model.forward_model(x)
                        energy_pred = self.alg.energy_model.energy_head(z_next_pred).view(num_envs, num_candidates)

                        # Costo Bilanciato: Energia + Distanza dall'azione nominale
                        deviation = torch.norm(candidate_actions - nominal_actions.unsqueeze(1), dim=-1)
                        combined_score = energy_pred + lambda_reg * deviation

                        # Selezione dell'azione migliore e blending lineare
                        best_indices = torch.argmin(combined_score, dim=1)
                        best_actions = candidate_actions[torch.arange(num_envs, device=self.device), best_indices]

                        actions = (1.0 - alpha) * nominal_actions + alpha * best_actions

                        # Aggiornamento della transition object
                        self.alg.transition.actions.copy_(actions)

                        if hasattr(self.alg.actor_critic, 'get_actions_log_prob'):
                            new_log_probs = self.alg.actor_critic.get_actions_log_prob(actions)
                            self.alg.transition.actions_log_prob.copy_(new_log_probs.detach())

                        
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    energy_target = self.env.current_cot.unsqueeze(-1)
                    self.alg.process_env_step(obs, rewards, dones, infos, energy_target)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss, mean_symmetry_loss, mean_spike_loss, mean_energy_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        actor = self.alg.actor_critic.actor

        spike_rates = getattr(actor, "last_spike_rates", [])
        membrane_means = getattr(actor, "last_membrane_means", [])
        self.writer.add_scalar(
            'SNN/decay_std',
            actor.last_decay_std,
            locs['it']
        )

        self.writer.add_scalar(
            'SNN/threshold_std',
            actor.last_threshold_std,
            locs['it']
        )
        
        decay_mean = getattr(actor, "last_decay_mean", float("nan"))
        threshold_mean = getattr(actor, "last_threshold_mean", float("nan"))
        membrane_stds = getattr(actor, "last_membrane_stds", [])

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/energy', locs['mean_energy_loss'], locs['it'])
        if locs.get('mean_spike_loss') is not None:
            self.writer.add_scalar('Loss/spike_loss', locs['mean_spike_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        if locs.get('mean_symmetry_loss') is not None:
            self.writer.add_scalar('Loss/symmetry_loss', locs['mean_symmetry_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])

        for layer_idx, rate in enumerate(spike_rates):
            self.writer.add_scalar(
                f'SNN/layer_{layer_idx}_spike_rate',
                rate,
                locs['it']
            )

        for layer_idx, mem in enumerate(membrane_means):
            self.writer.add_scalar(
                f'SNN/layer_{layer_idx}_membrane_mean',
                mem,
                locs['it']
            )

        for layer_idx, mem_std in enumerate(membrane_stds):
            self.writer.add_scalar(
                f'SNN/layer_{layer_idx}_membrane_std',
                mem_std,
                locs['it']
            )
            
        self.writer.add_scalar('SNN/decay_mean', self.alg.actor_critic.actor.last_decay_mean, locs['it'])
        self.writer.add_scalar('SNN/threshold_mean', self.alg.actor_critic.actor.last_threshold_mean, locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        snn_string = ""

        for layer_idx, rate in enumerate(spike_rates):
            snn_string += (
                f"{f'SNN layer {layer_idx} spike rate:':>{pad}} "
                f"{rate:.4f}\n"
            )

        for layer_idx, mem in enumerate(membrane_means):
            snn_string += (
                f"{f'SNN layer {layer_idx} mem mean:':>{pad}} "
                f"{mem:.4f}\n"
            )

        snn_string += (
            f"{f'SNN decay mean:':>{pad}} "
            f"{decay_mean:.4f}\n"
        )

        snn_string += (
            f"{f'SNN decay std:':>{pad}} "
            f"{actor.last_decay_std:.4f}\n"
        )

        snn_string += (
            f"{f'SNN threshold mean:':>{pad}} "
            f"{threshold_mean:.4f}\n"
        )

        snn_string += (
            f"{f'SNN threshold std:':>{pad}} "
            f"{actor.last_threshold_std:.4f}\n"
        )
        
        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
              f"""{str.center(width, ' ')}\n\n"""
              f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
              f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
              f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
              f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
              f"""{snn_string}\n"""
              f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
              f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)
