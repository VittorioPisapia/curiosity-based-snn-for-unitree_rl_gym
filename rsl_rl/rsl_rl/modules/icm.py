import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import math
from typing import List, Dict, Union, Any, Tuple
from abc import abstractmethod
import torch.nn.functional as F
from .actor_critic import get_activation

class ICM(nn.Module):
    def __init__(
            self,
            num_obs ,
            num_actions,
            hidden_dimension = 128,
            encoder_output=64,
            activation = "relu"):
            
            super(ICM,self).__init__()

            activation = get_activation(activation)

            forward_layers = []
            forward_layers.append(nn.Linear(encoder_output + num_actions,hidden_dimension))
            forward_layers.append(activation)
            forward_layers.append(nn.Linear(hidden_dimension,encoder_output))

            self.forward_model = nn.Sequential(*forward_layers)

            inverse_layers = []
            inverse_layers.append(nn.Linear(encoder_output*2,hidden_dimension))
            inverse_layers.append(activation)
            inverse_layers.append(nn.Linear(hidden_dimension,num_actions))

            self.inverse_model = nn.Sequential(*inverse_layers)

            encoder_layers = []
            encoder_layers.append(nn.Linear(num_obs,hidden_dimension))
            encoder_layers.append(activation)
            encoder_layers.append(nn.Linear(hidden_dimension,encoder_output))

            self.encoder_model = nn.Sequential(*encoder_layers)

            print("Forward Model: {self.forward_model}")
            print("Inverse Model: {self.inverse_model}")
            print("Encoder Model: {self.encoder_model}")


    def compute_forward(self, prev_obs, prev_action):
        conc = torch.cat([prev_obs, prev_action], dim=-1)
        return self.forward_model(conc)

    def compute_inverse(self, prev_obs, next_obs):
        conc = torch.cat([prev_obs, next_obs], dim=-1)
        return self.inverse_model(conc)
    
    def compute_encoded(self, state):
        encoded_state = self.encoder_model(state)
        return encoded_state



