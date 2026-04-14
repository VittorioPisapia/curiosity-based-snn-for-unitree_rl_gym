from networkx import omega

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.CPG import CPG_RL
import torch
import numpy as np


from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg


class GO2CPGEnv(LeggedRobot):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, sim_device=sim_device, headless=headless)

        self.frequency_high = 1.0 * 2 * np.pi
        self.frequency_low  = 0.5 * 2 * np.pi

        class Go2Kinematics:
            def __init__(self):
                self.hip_link_length = 0.0955
                self.thigh_link_length = 0.213
                self.calf_link_length = 0.213

        self.robot_kinematics = Go2Kinematics()
        self.cpg = CPG_RL(num_envs=self.num_envs, device=self.device, rl_task_string="", time_step=self.dt)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.cpg.reset(env_ids)

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.contact_forces[:, self.feet_indices, 2] > 1.,
                                    (self.cpg.X[:,0,:] - ((self.cpg.mu_up+ self.cpg.mu_low) / 2)) * self.obs_scales.dof_pos,
                                    (self.cpg.X[:,1,:] - np.pi) * 1/np.pi,
                                    self.cpg.X_dot[:,0,:] * 1/30, 
                                    (self.cpg.X_dot[:,1,:] - 15) * 1/30,
                                    ),dim=-1)
        
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def _compute_torques(self, actions):
        """ Compute torques from CPG Signals """
        actions = torch.zeros_like(actions)
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        normal_forces = self.contact_forces[:, self.feet_indices, 2] > 1.

        if "CPG" in control_type:
            xs, ys, zs = self.cpg.get_CPG_RL_actions(actions_scaled, self.frequency_high, self.frequency_low, normal_forces)

            sideSign = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=self.device).view(1, 4)
            y_offset = sideSign * self.robot_kinematics.hip_link_length
            y_final = ys + y_offset 

            self.dof_des_pos = self.cpg.compute_inverse_kinematics(self.robot_kinematics, xs, y_final, zs)
            
            torques = self.p_gains * (self.dof_des_pos - self.dof_pos) - self.d_gains * self.dof_vel 
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        return torch.clip(torques, -self.torque_limits, self.torque_limits)