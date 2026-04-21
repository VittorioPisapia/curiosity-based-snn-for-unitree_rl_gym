import torch
import torch.nn as nn
import torch.nn.functional as F
from .actor_critic import get_activation

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