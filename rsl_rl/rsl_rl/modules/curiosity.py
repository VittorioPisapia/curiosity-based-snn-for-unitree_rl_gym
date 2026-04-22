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
            encoder_output = 64,
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

class RND(nn.Module):
    def __init__(
            self,
            num_obs,
            hidden_dimension = 128,
            feature_dimension = 64,
            activation = "relu"):
        
        super(RND, self).__init__()

        target_layers = []
        target_layers.append(nn.Linear(num_obs, hidden_dimension))
        target_layers.append(get_activation(activation))
        target_layers.append(nn.Linear(hidden_dimension, feature_dimension))

        self.target_model=nn.Sequential(*target_layers)  # FIXED

        predictor_layers = []
        predictor_layers.append(nn.Linear(num_obs, hidden_dimension))
        predictor_layers.append(get_activation(activation))
        predictor_layers.append(nn.Linear(hidden_dimension, feature_dimension))

        self.predictor_model=nn.Sequential(*predictor_layers)  # TRAINABLE

        # Fixing parameters for target model
        for p in self.target_model.parameters():
            p.requires_grad = False

    def forward(self, obs):
        with torch.no_grad():
            target_feat = self.target_model(obs)
        
        pred_feat = self.predictor_model(obs)

        return pred_feat, target_feat