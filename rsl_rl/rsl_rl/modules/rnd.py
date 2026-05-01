import torch
import torch.nn as nn
from typing import Any, NoReturn, List
from .actor_critic import get_activation

from rsl_rl.env import VecEnv

class RandomNetworkDistillation(nn.Module):
    def __init__(
        self,
        num_obs,
        num_outputs,
        predictor_hidden_dims,
        target_hidden_dims,
        weight,
        activation="relu",
        learning_rate=1e-3,
        device="cpu"
    ):

        super().__init__()

        self.num_obs = num_obs
        self.num_outputs = num_outputs
        self.weight = weight
        self.device = device

        self.update_counter = 0

        self.target = build_mlp(num_obs, target_hidden_dims, num_outputs, activation).to(self.device)
        self.target.eval()
        for param in self.target.parameters():
            param.requires_grad = False

        self.predictor = build_mlp(num_obs, predictor_hidden_dims, num_outputs, activation).to(self.device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

    def get_intrinsic_reward(self, obs):

        self.update_counter += 1
        target_embedding = self.target(obs).detach()
        predictor_embedding = self.predictor(obs).detach()

        intrinsic_reward = torch.linalg.norm(target_embedding - predictor_embedding, dim=1)
        intrinsic_reward *= self.weight

        return intrinsic_reward
    
    def compute_loss(self, obs):

        predictor_embedding = self.predictor(obs)
        target_embedding = self.target(obs).detach()

        return nn.functional.mse_loss(predictor_embedding, target_embedding)

    def train(self, mode: bool = True):
        self.predictor.train(mode)
        return self
    
    def eval(self):
        return self.train(False)

def build_mlp(input_size: int, hidden_sizes: List[int], output_size: int, activation: str):
            layers = []
            current_size = input_size
            for h_size in hidden_sizes:
                layers.append(nn.Linear(current_size, h_size))
                layers.append(get_activation(activation))
                current_size = h_size
            # Final output layer (typically no activation applied to the final embedding)
            layers.append(nn.Linear(current_size, output_size))
            return nn.Sequential(*layers)