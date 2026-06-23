import torch
import torch.nn as nn
from typing import List
from .actor_critic import get_activation

class EnergyForwardModel(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        forward_loss_coeff,
        energy_loss_coeff,
        hidden_dims=(256, 256),
        latent_dim=64,
        activation="elu",
        device="cpu",
    ):
        super().__init__()

        self.device = device
        self.forward_loss_coeff = forward_loss_coeff
        self.energy_loss_coeff = energy_loss_coeff

        # ---- Encoder φ(s) ----
        encoder_layers = []
        last_dim = obs_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h))
            encoder_layers.append(get_activation(activation))
            last_dim = h
        encoder_layers.append(nn.Linear(last_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers).to(device)

        #  ---- Forward model: (φ(s), a) → φ(s') ----
        self.forward_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            get_activation(activation),
            nn.Linear(256, latent_dim),
        ).to(device)

        # ---- Energy head: (φ(s), a) → Cot ----
        self.energy_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128), 
            get_activation(activation),
            nn.Linear(128, 1),
        ).to(device)

    def encode(self, obs):
        return self.encoder(obs)

    def predict_next_latent(self, obs, actions):
        z = self.encode(obs)
        x = torch.cat([z, actions], dim=-1)
        return self.forward_model(x)

    # ---- Energy prediction -----
    def predict_energy(self, obs, actions):
        z = self.encode(obs)
        x = torch.cat([z, actions], dim=-1)
        return self.energy_head(x)

    # ---- Loss function ----
    def compute_loss(self, obs, actions, next_obs, energy_target):
        z = self.encode(obs)
        z_next = self.encode(next_obs).detach()

        # forward loss
        z_pred_next = self.forward_model(torch.cat([z, actions], dim=-1))
        forward_loss = torch.mean((z_pred_next - z_next) ** 2)

        # energy loss
        energy_pred = self.predict_energy(obs, actions)
        energy_loss = torch.mean((energy_pred - energy_target) ** 2)

        total_loss = self.forward_loss_coeff * forward_loss + self.energy_loss_coeff * energy_loss

        return total_loss