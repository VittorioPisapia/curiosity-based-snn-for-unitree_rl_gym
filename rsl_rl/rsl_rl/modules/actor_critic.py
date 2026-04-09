# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import math
from typing import List, Dict, Union, Any, Tuple
from abc import abstractmethod
import torch.nn.functional as F

class Neurons(nn.Module):
    def __init__(
            self,
            hidden_states_names: List[str],
            grad: torch.autograd.Function,
            device: Union[str, torch.device],
        ) -> None:
        super().__init__()

        self.device = torch.device(device) if isinstance(device, str) else device
        self.spike_function = grad.apply
        self.hidden_states_names = hidden_states_names
        self.hidden_states_tensors = {k: None for k in self.hidden_states_names}

    def _set_hidden_states(self, hidden_states: Dict[str, Any], size: Tuple[int, int]):
        """
        size: batch, no neurons
        """
        for name in self.hidden_states_names:
            _hstate = hidden_states.get(name, None)
            if _hstate is None:
                _hstate = torch.zeros(*size, dtype=torch.float32, device=self.device)
            self.hidden_states_tensors[name] = _hstate.clone()
    
    @abstractmethod
    def forward(self, x, hidden_states, spiking_neurons):
        pass

class SpikeFunctionGaussian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_membrane, thresh, lens):
        ctx.save_for_backward(v_membrane)
        #ctx.save_for_backward(v_membrane, thresh)
        ctx.thresh = thresh
        ctx.lens = lens
        return v_membrane.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        v_membrane, = ctx.saved_tensors
        thresh = ctx.thresh
        lens = ctx.lens
        grad_input = grad_output.clone()
        exp = torch.exp(-(v_membrane - thresh)**2 / (2 * lens))
        temp = exp / math.sqrt(2 * math.pi * lens)
        return grad_input * temp.float(), None, None

class LIFGaussian(Neurons):
    def __init__(
            self,
            lens: float,
            device: Union[str, torch.device],
            **kwards,
        ) -> None:
        super().__init__(["snn_s", "snn_m"], SpikeFunctionGaussian, device)
        self.lens = lens

    def forward(self, x, thresholds, decays, hidden_states, spiking_neurons):
        output = {}
        batch_sz, layer_sz = x.shape[0], x.shape[1]

        self._set_hidden_states(hidden_states, (batch_sz, layer_sz))

        # v_prev = self.hidden_states_tensors["snn_m"]  #TODO: try soft-reset of the membrane potential, i.e. reset to v - thresh instead of 0

        # if spiking_neurons:
        #     v_prev = v_prev * -(self.hidden_states_tensors["snn_s"]*thresholds)
        
        # decayed_m = v_prev * decays

        # output["snn_m"] = decayed_m + x
            
        spikes_reset = 1  # if 0 the previous v mem is reset
        if spiking_neurons:
            #spikes_reset = 1 - self.hidden_states_tensors["snn_s"]

            spikes_reset = 0.8 + 0.2 * (1 - self.hidden_states_tensors["snn_s"])
        
        output["snn_m"] = self.hidden_states_tensors["snn_m"] * decays * spikes_reset + x
        if spiking_neurons:
            output["snn_s"] = self.spike_function(output["snn_m"], thresholds, self.lens)
        return output

class SNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, threshold_init=0.3, lens=0.3):
        super().__init__()

        self.device = torch.device(device) if isinstance(device, str) else device
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.fs = LIFGaussian(lens=lens, device=self.device)

        self.spike_dim = 2 * hidden_dim
        self.mem_dim = 2 * hidden_dim

        self.thresholds = nn.Parameter(
            torch.full((self.spike_dim,), threshold_init, device=self.device) 
        )
        self.decays_raw = nn.Parameter(
            torch.full((self.mem_dim,), -0.5, device=self.device)
        )

        self.last_s1_rate = 0.0
        self.last_s2_rate = 0.0
        self.last_m1_mean = 0.0
        self.last_m2_mean = 0.0
        self.last_decay_mean = 0.0

    def _neurons_forward(self, x, hidden_states, start_idx, end_idx, output_spikes=True):
        local_states = {}

        for hname in self.fs.hidden_states_names:
            h = hidden_states.get(hname, None)
            if h is None:
                local_states[hname] = None
            else:
                local_states[hname] = h[:, start_idx:end_idx].clone()

        decays = torch.sigmoid(self.decays_raw[start_idx:end_idx])

        return self.fs(
            x,
            self.thresholds[start_idx:end_idx] if output_spikes else None,
            decays,
            local_states,
            output_spikes
        )

    def forward(self, obs, hidden_states, st=2):
        obs = obs.to(self.device)
        batch_size = obs.shape[0]
        st = int(st)

        if hidden_states is None:
            current_state = {
                "snn_m": torch.zeros(batch_size, self.mem_dim, device=self.device),
                "snn_s": torch.zeros(batch_size, self.spike_dim, device=self.device),
            }
        else:
            current_state = {
                "snn_m": hidden_states["snn_m"].detach().clone(),
                "snn_s": hidden_states["snn_s"].detach().clone(),
            }

        for _ in range(st):
            z1 = self.fc1(obs*1.5) 
            h1 = self._neurons_forward(z1, current_state, 0, self.hidden_dim, True)

            z2 = self.fc2(h1["snn_s"])
            h2 = self._neurons_forward(z2, current_state, self.hidden_dim, 2 * self.hidden_dim, True)

            current_state = {
                "snn_m": torch.cat([h1["snn_m"], h2["snn_m"]], dim=1),
                "snn_s": torch.cat([h1["snn_s"], h2["snn_s"]], dim=1),
            }

            self.last_s1_rate = h1["snn_s"].mean().item()
            self.last_s2_rate = h2["snn_s"].mean().item()
            self.last_m1_mean = h1["snn_m"].mean().item()
            self.last_m2_mean = h2["snn_m"].mean().item()
            self.last_decay_mean = torch.sigmoid(self.decays_raw).mean().item()

        out = self.fc3(h2["snn_m"])
        return out, current_state

class ActorCritic(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        activation = get_activation(activation)

        self.hidden_states = None
        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs

        self.actor = SNN(mlp_input_dim_a, 256, num_actions, device="cuda", threshold_init=kwargs.get('snn_threshold', 0.3), lens=kwargs.get('snn_lens', 0.3))
        self.st = kwargs.get('snn_st', 1)
        
        print(f"Initialized ActorCritic with SNN actor, st={self.st}, snn_threshold={kwargs.get('snn_threshold', 0.3)}, snn_lens={kwargs.get('snn_lens', 0.3)}")

       # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
                
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        if self.hidden_states is None:
            return
        if dones is None:
            return

        done_ids = dones.view(-1).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self.hidden_states["snn_m"][done_ids] = 0.0
            self.hidden_states["snn_s"][done_ids] = 0.0

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hidden_states=None):
        mean, _ = self.actor(observations, hidden_states=hidden_states, st=self.st)
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, observations, **kwargs):
        mean, next_hidden_states = self.actor(observations, hidden_states=self.hidden_states, st=self.st)
        self.distribution = Normal(mean, self.std.expand_as(mean))

        self.hidden_states = {
            "snn_m": next_hidden_states["snn_m"].detach(),
            "snn_s": next_hidden_states["snn_s"].detach()
        }

        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean, next_hidden_states = self.actor(observations, hidden_states=self.hidden_states)

        self.hidden_states = {
            "snn_m" : next_hidden_states["snn_m"].detach(),
            "snn_s" : next_hidden_states["snn_s"].detach()
        }

        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None