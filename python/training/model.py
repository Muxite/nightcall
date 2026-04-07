"""
Actor-critic model for nightcall.

Architecture
------------
Shared MLP backbone:  obs(FEATURE_DIM) → 512 → 256 → 128  (ReLU + LayerNorm)
Actor head:           128 → MAX_P1_UNITS × N_UNIT_ACTIONS logits
                      (treated as MAX_P1_UNITS independent categoricals)
Critic head:          128 → 1 scalar value

The actor outputs a (MAX_P1_UNITS, N_UNIT_ACTIONS) logit matrix.
At inference time, each row is sampled independently (MultiCategorical).
For ONNX export only the actor is exported (no Critic, no sampling):
  input  "observation"   shape [1, FEATURE_DIM]
  output "action_logits" shape [1, MAX_P1_UNITS * N_UNIT_ACTIONS]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from env import FEATURE_DIM, MAX_P1_UNITS, N_UNIT_ACTIONS


class NightcallActorCritic(nn.Module):
    """Full actor-critic used during training."""

    def __init__(
        self,
        hidden1: int = 512,
        hidden2: int = 256,
        hidden3: int = 128,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.LayerNorm(hidden3),
            nn.ReLU(),
        )

        # Actor: flat logits for all unit slots concatenated
        self.actor_head  = nn.Linear(hidden3, MAX_P1_UNITS * N_UNIT_ACTIONS)
        # Critic: single value estimate
        self.critic_head = nn.Linear(hidden3, 1)

        self._init_weights()

    def _init_weights(self):
        # Orthogonal init with sqrt(2) gain for backbone layers (good for ReLU),
        # small gain for output heads to start with near-uniform distributions.
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.4142)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: float32 tensor of shape (B, FEATURE_DIM)
        Returns:
            logits: (B, MAX_P1_UNITS, N_UNIT_ACTIONS)
            value:  (B, 1)
        """
        h = self.backbone(obs)
        logits = self.actor_head(h).view(-1, MAX_P1_UNITS, N_UNIT_ACTIONS)
        value  = self.critic_head(h)
        return logits, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ):
        """
        Sample actions and compute log-probs + entropy for PPO.

        Returns:
            action:   (B, MAX_P1_UNITS)  int64
            log_prob: (B,)               sum of per-unit log-probs
            entropy:  (B,)               sum of per-unit entropies
            value:    (B, 1)
        """
        logits, value = self(obs)
        dist = MultiCategorical(logits)

        if action is None:
            action = dist.sample()                      # (B, MAX_P1_UNITS)

        log_prob = dist.log_prob(action)                # (B,)
        entropy  = dist.entropy()                       # (B,)
        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self(obs)
        return value


class MultiCategorical:
    """Batch of independent Categorical distributions for MultiDiscrete."""

    def __init__(self, logits: torch.Tensor):
        # logits: (B, MAX_P1_UNITS, N_UNIT_ACTIONS)
        self._dists = [Categorical(logits=logits[:, i]) for i in range(MAX_P1_UNITS)]

    def sample(self) -> torch.Tensor:
        return torch.stack([d.sample() for d in self._dists], dim=1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: (B, MAX_P1_UNITS)
        lps = [self._dists[i].log_prob(actions[:, i]) for i in range(MAX_P1_UNITS)]
        return torch.stack(lps, dim=1).sum(dim=1)      # (B,)

    def entropy(self) -> torch.Tensor:
        ents = [d.entropy() for d in self._dists]
        return torch.stack(ents, dim=1).sum(dim=1)     # (B,)


# ── Actor-only wrapper for ONNX export ────────────────────────────────────────

class ActorOnly(nn.Module):
    """Thin wrapper that exports only the actor (no Critic, no sampling)."""

    def __init__(self, full_model: NightcallActorCritic):
        super().__init__()
        self.backbone    = full_model.backbone
        self.actor_head  = full_model.actor_head

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:  obs (B, FEATURE_DIM)
        Returns: action_logits (B, MAX_P1_UNITS * N_UNIT_ACTIONS)
        """
        h = self.backbone(obs)
        return self.actor_head(h)
