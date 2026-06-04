import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from typing import List, Dict, Union, Any, Tuple, Optional
from abc import abstractmethod


from typing import List, Dict, Union, Any, Tuple, Optional
import torch
import torch.nn as nn
from abc import abstractmethod

class Neurons(nn.Module):
    hidden_states_names: List[str]
    hidden_states_tensors: Dict[str, torch.Tensor]

    def __init__(
            self,
            hidden_states_names: List[str],
            grad: Any,
            device: Union[str, torch.device],
        ) -> None:
        super().__init__()

        self.device = torch.device(device) if isinstance(device, str) else device
        self.spike_function = grad.apply if grad is not None else None
        
        self.hidden_states_names = hidden_states_names

        self.hidden_states_tensors = {
            name: torch.empty(0, device=self.device) for name in self.hidden_states_names
        }

    def _set_hidden_states(self, hidden_states: Dict[str, torch.Tensor], size: List[int]):
        """
        size: [batch, no_neurons]
        """
        for name in self.hidden_states_names:
            if name in hidden_states:
                _hstate = hidden_states[name]
            else:
                _hstate = torch.zeros(size, dtype=torch.float32, device=self.device)
                
            self.hidden_states_tensors[name] = _hstate.clone()
    
    @abstractmethod
    def forward(self, x: torch.Tensor, hidden_states: Dict[str, torch.Tensor], spiking_neurons: bool):
        pass

class SpikeFunctionGaussian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_membrane, thresh, lens):
        #ctx.save_for_backward(v_membrane)
        ctx.save_for_backward(v_membrane, thresh)
        ctx.thresh = thresh
        ctx.lens = lens
        return v_membrane.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        v_membrane, thresh = ctx.saved_tensors
        lens = ctx.lens
        
        grad_output_clone = grad_output.clone()
        exp = torch.exp(-(v_membrane - thresh)**2 / (2 * lens))
        temp = (exp / math.sqrt(2 * math.pi * lens)).float()
        
        grad_v_membrane = grad_output_clone * temp
        grad_thresh = -grad_v_membrane 

        return grad_v_membrane, grad_thresh, None

class LIFGaussian(Neurons):
    def __init__(
            self,
            lens: float,
            device: Union[str, torch.device],
            **kwards,
        ) -> None:
        super().__init__(["snn_s", "snn_m"], SpikeFunctionGaussian, device)
        self.lens = lens

    def forward(self, x: torch.Tensor, thresholds: torch.Tensor, decays: torch.Tensor, 
                hidden_states: Dict[str, torch.Tensor], spiking_neurons: bool) -> Dict[str, torch.Tensor]:
        
        batch_sz = x.shape[0]
        layer_sz = x.shape[1]

        self._set_hidden_states(hidden_states, [batch_sz, layer_sz])

        #spikes_reset = 1.0
        #if spiking_neurons:
        #    spikes_reset = 1.0 - self.hidden_states_tensors["snn_s"]

        if spiking_neurons:
            spikes_reset = 1.0 - self.hidden_states_tensors["snn_s"]
        else:
            spikes_reset = torch.ones_like(self.hidden_states_tensors["snn_s"])
            
        snn_m_out = self.hidden_states_tensors["snn_m"] * decays * spikes_reset + x
        
        if spiking_neurons:
            if torch.jit.is_scripting():
                snn_s_out = snn_m_out.gt(thresholds).float()
            else:
                snn_s_out = self.spike_function(snn_m_out, thresholds, self.lens)
        else:
            snn_s_out = torch.zeros_like(snn_m_out)
                
        output: Dict[str, torch.Tensor] = {
            "snn_m": snn_m_out,
            "snn_s": snn_s_out
        }
        
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
    def __init__(self, input_dim, num_neurons, output_dim, device, threshold_init=0.5, lens=0.3, neuron_type="Gaussian"):
        super().__init__()
        
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)
        self.total_neurons = sum(num_neurons)

        self.spike_dim = self.total_neurons
        self.mem_dim = self.total_neurons

        self.input_norm = nn.LayerNorm(input_dim)

        # ---- Linear layers ----
        layer_dims = [input_dim] + num_neurons

        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i +1])
            for i in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(num_neurons[-1], output_dim)

        # ---- Neuron model ----
        if neuron_type == "Gaussian":
            self.fs = LIFGaussian(lens=lens, device=self.device)
        elif neuron_type == "BPTT":
            self.fs = LIF_BPTT(device=self.device)
        else:
            raise ValueError(f"Unsupported neuron type: {neuron_type}")

        # ---- State dimensions ----
        self.total_neurons = sum(num_neurons)

        # ---- Decyas and thresholds ----
        self.thresholds_raw = nn.Parameter(
            torch.full((self.total_neurons,), threshold_init, device=self.device) 
        )
        self.decays_raw = nn.Parameter(
            torch.full((self.total_neurons,), -0.5, device=self.device) #-0.5
        )

        # ---- Logging ----
        self.last_spike_rates = [0.0 for _ in range(self.num_layers)]
        self.last_membrane_means = [0.0 for _ in range(self.num_layers)]
        self.last_membrane_stds = [0.0 for _ in range(self.num_layers)]
        self.last_layer_spikes = [None for _ in range(self.num_layers)]
        
        self.last_decay_mean = 0.0
        self.last_decay_std = 0.0

        self.last_threshold_mean = 0.0
        self.last_threshold_std = 0.0

    def _neurons_forward(self, x: torch.Tensor, hidden_states: Dict[str, torch.Tensor], start_idx: int, end_idx: int, output_spikes: bool = True) -> Dict[str, torch.Tensor]:
        
        local_states = torch.jit.annotate(Dict[str, torch.Tensor], {})

        for hname in self.fs.hidden_states_names:
            if hname in hidden_states:
                local_states[hname] = hidden_states[hname][:, start_idx:end_idx].clone()

        decays = torch.sigmoid(self.decays_raw[start_idx:end_idx])
        # thresholds = torch.sigmoid(self.thresholds_raw[start_idx:end_idx])
        thresholds = torch.relu(self.thresholds_raw[start_idx:end_idx]) + 0.1

        return self.fs(
            x,
            thresholds,
            decays,
            local_states,
            output_spikes
        )

    def forward(self, obs: torch.Tensor, hidden_states: Optional[Dict[str, torch.Tensor]] = None, st: int = 1) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        obs = obs.to(self.device)
        obs = self.input_norm(obs)

        batch_size = obs.shape[0]

        if hidden_states is None:
            current_state = {
                "snn_m": torch.zeros(batch_size, self.total_neurons, device=self.device),
                "snn_s": torch.zeros(batch_size, self.total_neurons, device=self.device),
            }
        else:
            current_state = {
                "snn_m": hidden_states["snn_m"].clone(),
                "snn_s": hidden_states["snn_s"].clone(),
            }

        new_mems: List[torch.Tensor] = []
        new_spikes: List[torch.Tensor] = []

        for _ in range(st):

            x = obs
            
            new_mems = torch.jit.annotate(List[torch.Tensor], [])
            new_spikes = torch.jit.annotate(List[torch.Tensor], [])
            start_idx = 0

            for layer_idx, layer in enumerate(self.layers):
                end_idx = start_idx + self.num_neurons[layer_idx]

                z = layer(x)

                h = self._neurons_forward(
                    z,
                    current_state,
                    start_idx,
                    end_idx,
                    True
                )

                x = h["snn_s"]

                new_mems.append(h["snn_m"])
                new_spikes.append(h["snn_s"])
                
                if not torch.jit.is_scripting():
                    self.last_spike_rates[layer_idx] = h["snn_s"].mean().item()
                    self.last_membrane_means[layer_idx] = h["snn_m"].mean().item()
                    self.last_membrane_stds[layer_idx] = h["snn_m"].std().item()

                start_idx = end_idx
            
            current_state = {
                "snn_m" : torch.cat(new_mems, dim=1),
                "snn_s" : torch.cat(new_spikes, dim=1)
            }

            decays = torch.sigmoid(self.decays_raw)
            thresholds = (torch.relu(self.thresholds_raw) + 0.1)

            if not torch.jit.is_scripting():
                self.last_decay_mean = decays.mean().item()
                self.last_decay_std = decays.std().item()
                self.last_threshold_mean = thresholds.mean().item()
                self.last_threshold_std = thresholds.std().item()
                self.last_layer_spikes = new_spikes

        out = self.output_layer(new_mems[-1])    

        return out, current_state