import time
import os
from collections import deque
import statistics

import torch

from .snn_runner import SnnRunner
from rsl_rl.modules.actor_critic import ActorCriticSNN
from rsl_rl.algorithms.ppo_snn import PPO_Snn
from rsl_rl.env import VecEnv
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.modules.curiosity import ICM, RND

class IcmRunner ( SnnRunner ):
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

        self.use_icm = self.policy_cfg.get("use_icm", True)
        self.use_rnd = self.policy_cfg.get("use_rnd", False)

        self.beta = self.policy_cfg.get("icm_beta", 1)
        self.intrinsic_coeff = self.policy_cfg.get("icm_intrinsic_coeff", 0.01)
        self.icm_reward_clamp = self.policy_cfg.get("icm_reward_clamp", 0.05)
        self.running_std = torch.tensor(1.0, device=self.device)

        self.icm_obs = []
        self.icm_next_obs = []
        self.icm_actions = []

        self.rnd_obs = []
        self.rnd_running_std = torch.tensor(1.0, device=self.device)
        self.rnd_intrinsic_coeff = self.policy_cfg.get("rnd_intrinsic_coeff", 0.005)
        self.rnd_reward_clamp = self.policy_cfg.get("rnd_reward_clamp", 0.05)

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
        self.alg: PPO_Snn = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        if self.use_icm:
            self.icm = ICM(env.num_obs, env.num_actions, hidden_dimension=128, activation="relu").to(self.device)
            self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=1e-4)

        if self.use_rnd:
            self.rnd = RND(env.num_obs, feature_dimension=64, hidden_dimension=128, activation="relu").to(self.device)
            self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor_model.parameters(), lr=1e-4)

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


        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            extrinsic_sum = 0.0
            intrinsic_sum = 0.0
            count = 0
            rnd_sum = 0.0
            intrinsic_reward = torch.zeros(self.env.num_envs, device=self.device)
            rnd_reward = torch.zeros(self.env.num_envs, device=self.device)
            # Rollout
            with torch.no_grad():
                for i in range(self.num_steps_per_env):

                    prev_obs = obs.clone()
                    actions = self.alg.act(obs, critic_obs)

                    clip_actions = self.env.cfg.normalization.clip_actions
                    prev_action = torch.clamp(actions, -clip_actions, clip_actions)

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs

                    prev_obs = prev_obs.to(self.device)
                    prev_action = prev_action.to(self.device)
                    obs = obs.to(self.device)
                    
                    if self.use_icm:
                    # ICM forward
                        encoded_prev = self.icm.compute_encoded(prev_obs)
                        encoded_next = self.icm.compute_encoded(obs)
                        forward_value = self.icm.compute_forward(encoded_prev, prev_action)

                        #ntrinsic_reward = ((encoded_next - forward_value) ** 2).mean(dim=-1)
                        intrinsic_reward = ((encoded_next.detach() - forward_value) ** 2).mean(dim=-1)

                        current_std_icm = intrinsic_reward.std()
                        self.running_std = 0.9 * self.running_std + 0.1 * current_std_icm

                        intrinsic_reward = intrinsic_reward / (self.running_std + 1e-8)
                        intrinsic_reward = torch.clamp(intrinsic_reward, 0.0, self.icm_reward_clamp)

                    if self.use_rnd:
                    # RND TODO
                        rnd_target = self.rnd.target_model(obs).detach()
                        rnd_pred = self.rnd.predictor_model(obs)

                        rnd_error = ((rnd_pred - rnd_target) ** 2).mean(dim=-1)

                        current_std_rnd = rnd_error.std()
                        self.rnd_running_std = 0.9 * self.rnd_running_std + 0.1 * current_std_rnd

                        rnd_reward = rnd_error / (self.rnd_running_std + 1e-8)
                        rnd_reward = torch.clamp(rnd_reward, 0.0, self.rnd_reward_clamp)


                    total_reward = rewards.clone()

                    if self.use_icm:
                        total_reward += self.intrinsic_coeff * intrinsic_reward

                    if self.use_rnd:
                        total_reward += self.rnd_intrinsic_coeff * rnd_reward

                    self.alg.process_env_step(total_reward, dones, infos)

                    if self.use_icm:
                        self.icm_obs.append(prev_obs.detach())
                        self.icm_next_obs.append(obs.detach())
                        self.icm_actions.append(prev_action.detach())
                    if self.use_rnd:
                        self.rnd_obs.append(obs.detach())

                    extrinsic_sum += rewards.mean().item()
                    
                    
                    if self.use_icm:
                        intrinsic_sum += intrinsic_reward.mean().item()
                    if self.use_rnd:
                        rnd_sum += rnd_reward.mean().item()

                    count += 1
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_reward
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

            if self.use_icm:
                self.intrinsic_coeff = max(0.001, self.intrinsic_coeff * 0.999)
            if self.use_rnd:
                self.rnd_intrinsic_coeff = max(0.001, self.rnd_intrinsic_coeff * 0.999)

            self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            if self.use_icm:
                obs_batch = torch.cat(self.icm_obs, dim=0)
                next_obs_batch = torch.cat(self.icm_next_obs, dim=0)
                actions_batch = torch.clamp(torch.cat(self.icm_actions, dim=0), -clip_actions, clip_actions)

                encoded_obs_batch = self.icm.compute_encoded(obs_batch)
                encoded_next_obs_batch = self.icm.compute_encoded(next_obs_batch).detach() 
                
                pred_next = self.icm.compute_forward(encoded_obs_batch, actions_batch)
                forward_loss = ((pred_next - encoded_next_obs_batch)**2).mean()

                pred_action = self.icm.compute_inverse(encoded_obs_batch, encoded_next_obs_batch)
                inverse_loss = ((pred_action - actions_batch)**2).mean()

                icm_loss = forward_loss + self.beta * inverse_loss
                # icm update
                self.icm_optimizer.zero_grad()
                icm_loss.backward()
                self.icm_optimizer.step()

            if self.use_rnd:

                rnd_obs_batch = torch.cat(self.rnd_obs, dim=0)

                rnd_target = self.rnd.target_model(rnd_obs_batch).detach()
                rnd_pred = self.rnd.predictor_model(rnd_obs_batch)

                rnd_loss = ((rnd_pred - rnd_target) ** 2).mean()

                # rnd update
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()
            
            if self.log_dir is not None:
                self.writer.add_scalar("Reward/extrinsic", extrinsic_sum / count, it)
                if self.use_rnd:
                    self.writer.add_scalar("Reward/rnd", rnd_sum / count, it)
                    self.writer.add_scalar("RND/loss", rnd_loss.item(), it)

                if self.use_icm:
                    self.writer.add_scalar("ICM/forward_loss", forward_loss.item(), it)
                    self.writer.add_scalar("ICM/inverse_loss", inverse_loss.item(), it)
                    self.writer.add_scalar("ICM/std", self.running_std, it)
                    self.writer.add_scalar("Reward/icm", intrinsic_sum / count, it)
                    self.writer.add_scalar(
                                "Reward/intrinsic_scaled",
                                self.intrinsic_coeff * (intrinsic_sum / count),
                                it
                            )
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
            self.icm_obs.clear()
            self.icm_next_obs.clear()
            self.icm_actions.clear()
            self.rnd_obs.clear()
        
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
        s1_rate = getattr(actor, "last_s1_rate", float("nan"))
        s2_rate = getattr(actor, "last_s2_rate", float("nan"))
        m1_mean = getattr(actor, "last_m1_mean", float("nan"))
        m2_mean = getattr(actor, "last_m2_mean", float("nan"))
        decay_mean = getattr(actor, "last_decay_mean", float("nan"))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])

        self.writer.add_scalar('SNN/s1_spike_rate', self.alg.actor_critic.actor.last_s1_rate, locs['it'])
        self.writer.add_scalar('SNN/s2_spike_rate', self.alg.actor_critic.actor.last_s2_rate, locs['it'])
        self.writer.add_scalar('SNN/m1_mean', self.alg.actor_critic.actor.last_m1_mean, locs['it'])
        self.writer.add_scalar('SNN/m2_mean', self.alg.actor_critic.actor.last_m2_mean, locs['it'])
        self.writer.add_scalar('SNN/decay_mean', self.alg.actor_critic.actor.last_decay_mean, locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
              f"""{str.center(width, ' ')}\n\n"""
              f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
              f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
              f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
              f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
              f"""{'SNN s1 spike rate:':>{pad}} {s1_rate:.4f}\n"""
              f"""{'SNN s2 spike rate:':>{pad}} {s2_rate:.4f}\n"""
              f"""{'SNN m1 mean:':>{pad}} {m1_mean:.4f}\n"""
              f"""{'SNN m2 mean:':>{pad}} {m2_mean:.4f}\n"""
              f"""{'SNN decay mean:':>{pad}} {decay_mean:.4f}\n"""
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

