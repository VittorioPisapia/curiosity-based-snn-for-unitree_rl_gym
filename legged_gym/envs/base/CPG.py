# MIT License
# 
# Copyright (c) 2024 EPFL Biorobotics Laboratory (BioRob). 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from time import time
from warnings import WarningMessage
import numpy as np
import os
import torch

class CPG_RL():
    LEG_INDICES = np.array([1,0,3,2]) #FL, FR, RL, RR 
    def __init__(self,
                 omega_swing = 8*2*np.pi,
                 omega_stance = 2*2*np.pi,
                 gait = "TROT",
                 couple = True,
                 coupling_strength = 1.0,
                 time_step = 0.001,
                 robot_height = 0.28,
                 ground_clearance = 0.07,
                 ground_penetration = 0.01,
                 num_envs=1,
                 device = None,
                 rl_task_string = None,
                 mu_low = 1.0,
                 mu_up = 4.0,
                 max_step_length = 0.15):
        
        self.rl_task_string = rl_task_string
        self._device = device
        self.X = torch.zeros(num_envs, 2, 4, dtype = torch.float, device = device, requires_grad = False)
        self.X_dot = torch.zeros(num_envs, 2, 4, dtype = torch.float, device = device, requires_grad = False)
        self.d2X = torch.zeros(num_envs, 1, 4, dtype = torch.float, device = device, requires_grad = False)
        self.num_envs = num_envs
        self._mu=torch.zeros(num_envs,4, dtype = torch.float, device = device, requires_grad = False)
        if "OFFSETX" in rl_task_string:
            self.offset_x = torch.zeros(num_envs, 4, dtype = torch.float, device = device, requires_grad=False)
            self.offset_z = torch.zeros(num_envs, 4, dtype = torch.float, device = device, requires_grad=False)
        self.y = torch.zeros(num_envs, 4, dtype = torch.float, device = device, requires_grad = False)
        self.mu_low = mu_low
        self.mu_up = mu_up
        self.max_step_length = max_step_length
        self._omega_swing = omega_swing
        self._omega_stance = omega_stance
        self._couple = couple
        self._coupling_strength = coupling_strength
        self._dt = time_step
        self._set_gait(gait)

        self.X[:,0,:] = torch.rand(num_envs, 4, device = self._device) * .1
        self.X[:,1,:] = self.PHI[0,:] #0.0

        self._ground_clearance = ground_clearance
        self._ground_penetration = ground_penetration
        self._robot_height = robot_height

    def reset(self, env_ids):
        self._mu[env_ids,:]= 0
        self.X[env_ids,0, :] = torch.rand(len(env_ids), 4, device = self._device) * .1
        self.X[env_ids,1, :] = self.PHI[0,:] #0.0
        self.X_dot[env_ids,:,:] = 0

    def _set_gait(self, gait):
        trot = torch.tensor([[0, 1, 1, 0],
                             [-1, 0, 0, -1],
                             [-1, 0, 0, -1],
                             [0, 1, 1, 0]],dtype=torch.float, device = self._device, requires_grad=False)
        trot = np.pi * trot
        self.PHI_trot = trot
        if gait == "TROT":
            print("Using TROT gait")
            self.PHI = self.PHI_trot
        else:
            raise NotImplementedError("Gait not implemented")
        
    def update_offset_x(self, offset):
        self._offset_x = offset

    
    def _scale_helper(self, action, lower_lim, upper_lim):
        new_a = lower_lim + 0.5*(action+1) * (upper_lim - lower_lim )
        new_a = torch.clip(new_a, lower_lim, upper_lim)
        return new_a
    
    def get_CPG_RL_actions(self, actions, frequency_high, frequency_low, normal_forces):
        MU_LOW = self.mu_low
        MU_UPP = self.mu_up
        MAX_STEP_LEN = self.max_step_length
        device = self._device
        a = torch.clip(actions, -1.0, 1.0)
        self._mu = self._scale_helper(a[:,:4], MU_LOW**2, MU_UPP **2)
        self._omega_residuals = self._scale_helper(a[:,4:8], frequency_low, frequency_high)
        if "OFFSETX" in self.rl_task_string:
            offset = self._scale_helper(a[:,8:12], -0.07, 0.07)
            self.update_offset_x(offset)
        self.integrate_oscillator_equations()
        x = torch.clip(self.X[:,0,:], MU_LOW, MU_UPP)
        x = MAX_STEP_LEN * (x- MU_LOW) / (MU_UPP - MU_LOW)
        if "OFFSETX" in self.rl_task_string:
            x = -x * torch.cos(self.X[:,1,:]) - self._offset_x 
            y = self.y
        else:
            x = -x * torch.cos(self.X[:,1,:])
            y = self.y
        z = torch.where(torch.sin(self.X[:,1,:]) > 0, 
                        -self._robot_height + self._ground_clearance   * torch.sin(self.X[:,1,:]),
                        -self._robot_height + self._ground_penetration * torch.sin(self.X[:,1,:]))
        return x, y, z
    
    def integrate_oscillator_equations(self):
        device = self._device 
        X_dot = self.X_dot.clone() 
        d2X = self.d2X.clone()
        _a = 150
        dt = 0.001
        for _ in range(int(self._dt/dt)):
            d2X_prev = self.d2X.clone()
            X_dot_prev = self.X_dot.clone()
            X = self.X.clone()
            d2X = (_a * ( _a/4 * (torch.sqrt(self._mu) - X[:,0,:]) - X_dot_prev[:,0,:] )).unsqueeze(1)
            if self._couple:
                for i in range(4):
                    self._omega_residuals[:,i] += torch.sum( X[:,0,:] * self._coupling_strength * torch.sin(X[:,1,:] - torch.remainder(self.X[:,1,i].unsqueeze(-1), (2*np.pi)) - self.PHI[i,:]), dim=1 )
            X_dot[:,1,:] = self._omega_residuals
            X_dot[:,0,:] = X_dot_prev[:,0,:] + (d2X_prev[:,0,:] + d2X[:,0,:]) * dt / 2
            self.X = X + (X_dot_prev + X_dot) * dt / 2 
            self.X_dot = X_dot
            self.d2X = d2X 
            self.X[:,1,:] = torch.remainder(self.X[:,1,:], (2*np.pi))

    def compute_inverse_kinematics(self, robot, x, y, z):
        l1 = robot.hip_link_length
        l2 = robot.thigh_link_length
        l3 = robot.calf_link_length


        sideSign = torch.tensor([-1.0, 1.0, -1.0, 1.0], device=x.device, dtype=torch.float).view(1, 4)

        D = (y**2 + (-z)**2 - l1**2 + (-x)**2 - l2**2 - l3**2) / (2 * l3 * l2)
        D = torch.clip(D, -1.0 + 1e-6, 1.0 - 1e-6) 

        knee_angle = torch.atan2(-torch.sqrt(1 - D**2), D)

        sqrt_component = y**2 + (-z)**2 - l1**2
        sqrt_component = torch.clamp(sqrt_component, min=1e-6)
        
        hip_roll_angle = -1 * (-torch.atan2(z, y) - torch.atan2(
            torch.sqrt(sqrt_component), sideSign * l1))

        hip_thigh_angle = torch.atan2(-x, torch.sqrt(sqrt_component)) - 1 * torch.atan2(
            l3 * torch.sin(knee_angle),
            l2 + l3 * torch.cos(knee_angle)
        )

        output = torch.stack([hip_roll_angle, hip_thigh_angle, knee_angle], dim=-1)
        
        return output.view(-1, 12)
    