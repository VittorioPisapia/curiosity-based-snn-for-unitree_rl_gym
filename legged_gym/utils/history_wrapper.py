import torch

class HistoryWrapper:
    def __init__(self, env):
        self.env = env
        
        # Recupera la configurazione della storia dal file config del robot
        if hasattr(self.env, 'cfg'):
            self.obs_history_length = self.env.cfg.env.num_obs_hist
        else:
            self.obs_history_length = self.env.LeggedRobotCfg.env.num_obs_hist
            
        self.num_obs_single = self.env.num_obs 
        self.num_obs_history = self.obs_history_length * self.num_obs_single
        
        # Sovrascriviamo le proprietà: i Runner leggeranno queste
        self.num_obs = self.num_obs_history
        self.num_privileged_obs = self.env.num_privileged_obs
        
        # Alloca il buffer storico sulla GPU del simulatore
        self.obs_history = torch.zeros(
            self.env.num_envs, 
            self.num_obs_history, 
            dtype=torch.float, 
            device=self.env.device, 
            requires_grad=False
        )

    def step(self, actions):
        obs, privileged_obs, rew, done, info = self.env.step(actions)
        # Shift temporale e inserimento dell'osservazione corrente in coda
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs_single:], obs), dim=-1)


        return self.obs_history, privileged_obs, rew, done, info
    
    def get_observations(self):
        obs = self.env.get_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs_single:], obs), dim=-1)
        return self.obs_history

    def reset_idx(self, env_ids):
        ret = self.env.reset_idx(env_ids)
        # Svuota a 0 solo la storia dei robot resettati
        self.obs_history[env_ids, :] = 0
        return ret
    
    def reset(self):
        obs, privileged_obs = self.env.reset()
        self.obs_history[:, :] = 0
        # Mette il primo frame nello slot più recente
        self.obs_history[:, -self.num_obs_single:] = obs
        return self.obs_history, privileged_obs

    def __getattr__(self, name):
        return getattr(self.env, name)