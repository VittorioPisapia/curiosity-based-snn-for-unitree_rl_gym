import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Union, Any, Tuple
from abc import abstractmethod

NEURON_TYPE = "Gaussian" # "Gaussian" or "BPTT"

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

        # SOFT RESET  #TODO: try soft-reset of the membrane potential, i.e. reset to v - thresh instead of 0

        # v_prev = self.hidden_states_tensors["snn_m"] 
        # if spiking_neurons:
        #     v_prev = v_prev * -(self.hidden_states_tensors["snn_s"]*thresholds)
        # decayed_m = v_prev * decays
        # output["snn_m"] = decayed_m + x
            
        spikes_reset = 1  # if 0 the previous v mem is reset
        if spiking_neurons:
            spikes_reset = 1 - self.hidden_states_tensors["snn_s"]

            #spikes_reset = 0.8 + 0.2 * (1 - self.hidden_states_tensors["snn_s"])   # Partial reset
        
        output["snn_m"] = self.hidden_states_tensors["snn_m"] * decays * spikes_reset + x
        if spiking_neurons:
            output["snn_s"] = self.spike_function(output["snn_m"], thresholds, self.lens)
        return output
    
class SpikeFunctionBPTT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_scaled, gamma):
        ctx.save_for_backward(v_scaled)
        ctx.gamma = gamma
        z_ = torch.gt(v_scaled, 0.)
        z_ = z_.type(torch.float)
        return z_
    
    @staticmethod
    def backward(ctx, grad_output):
        v_scaled, = ctx.saved_tensors
        gamma = ctx.gamma
        zeros = torch.zeros_like(v_scaled, device=v_scaled.device)
        return torch.maximum(1 - torch.abs(v_scaled), zeros) * gamma * grad_output, None


class LIF_BPTT(Neurons):
    def __init__(
            self,
            #decay: float,
            #threshold: float,
            device: Union[str, torch.device],
            **kwards,
        ) -> None:
        super().__init__(["snn_s", "snn_m"], SpikeFunctionBPTT, device)

    def forward(self, x, thresholds, decays, hidden_states, spiking_neurons):
        output = {}
        batch_sz, layer_sz = x.shape[0], x.shape[1]

        self._set_hidden_states(hidden_states, (batch_sz, layer_sz))

        spikes_reset = 1  # if 0 the previous v mem is reset
        if spiking_neurons:
            spikes_reset = 1 - self.hidden_states_tensors["snn_s"]
        
        output["snn_m"] = self.hidden_states_tensors["snn_m"] * decays * spikes_reset + x
        if spiking_neurons:
            output["snn_s"] = self.spike_function(
                output["snn_m"] - thresholds / thresholds, .3
            )
        return output


class SNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device, threshold_init=0.3, lens=0.3):
        super().__init__()

        self.device = torch.device(device) if isinstance(device, str) else device
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        if NEURON_TYPE == "Gaussian":
            self.fs = LIFGaussian(lens=lens, device=self.device)
        elif NEURON_TYPE == "BPTT":
            self.fs = LIF_BPTT(device=self.device)

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

    def forward(self, obs, hidden_states, st=1):
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
