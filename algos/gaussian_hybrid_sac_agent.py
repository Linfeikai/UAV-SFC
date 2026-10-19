"""Standard SAC mobility with the shared relational categorical placement head."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import SAC


class GaussianHybridSACAgent(SAC):
    """Train Gaussian mobility with SAC and placement with expert CE only."""

    def __init__(
        self,
        *args: Any,
        categorical_bc_coef: float = 1.0,
        categorical_bc_batch_size: int = 128,
        max_grad_norm: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.categorical_bc_coef = categorical_bc_coef
        self.categorical_bc_batch_size = categorical_bc_batch_size
        self.max_grad_norm = max_grad_norm
        self._expert_observations: dict[str, np.ndarray] | None = None
        self._expert_actions: np.ndarray | None = None
        super().__init__(*args, **kwargs)

    def set_expert_data(self, observations, actions: np.ndarray) -> None:
        """Attach the same heuristic transitions used by Diffusion placement."""
        if not observations:
            self._expert_observations = None
            self._expert_actions = None
            return
        self._expert_observations = {
            key: np.stack([obs[key] for obs in observations]).astype(np.float32)
            for key in observations[0]
        }
        self._expert_actions = np.asarray(actions, dtype=np.float32)

    def _placement_ce_step(self) -> float | None:
        if (
            self.categorical_bc_coef <= 0.0
            or self._expert_observations is None
            or self._expert_actions is None
        ):
            return None
        count = len(self._expert_actions)
        indices = np.random.randint(
            0, count, size=self.categorical_bc_batch_size
        )
        observations = {
            key: th.as_tensor(values[indices], device=self.device)
            for key, values in self._expert_observations.items()
        }
        target_actions = th.as_tensor(
            self._expert_actions[indices], device=self.device
        )
        loss = self.actor.placement_bc_loss(observations, target_actions)
        weighted_loss = self.categorical_bc_coef * loss
        self.actor.optimizer.zero_grad()
        weighted_loss.backward()
        th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor.optimizer.step()
        return float(loss.detach().cpu())

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps, batch_size)
        losses = []
        for _ in range(gradient_steps):
            loss = self._placement_ce_step()
            if loss is not None:
                losses.append(loss)
        if losses:
            self.logger.record("train/categorical_bc_loss", np.mean(losses))
