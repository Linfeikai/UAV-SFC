"""Small behavior-cloning utilities for warm-starting off-policy actors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch as th


def mobility_bc_loss(actor, observations, target_actions):
    """Supervise mobility without a costly differentiable diffusion rollout."""
    features = actor.extract_features(observations, actor.features_extractor)
    targets = target_actions[:, :actor.mobility_dim]
    if getattr(actor, "mobility_policy", "diffusion") == "gaussian":
        prediction, _ = actor._sample_gaussian_mobility(features, deterministic=True)
        return (prediction - targets).square().mean()
    clean = th.atanh(targets.clamp(-0.95, 0.95))
    timesteps = th.randint(0, actor.T, (len(targets),), device=targets.device)
    noise = th.randn_like(clean)
    alpha = actor.sqrt_alphas_cumprod[timesteps, None]
    sigma = actor.sqrt_one_minus_alphas_cumprod[timesteps, None]
    prediction = actor._epsilon_net(
        features[:, :actor.core_features_dim], alpha * clean + sigma * noise, timesteps,
    )
    return (prediction - noise).square().mean()


def _observation_batch(
    observations: Sequence[Mapping[str, np.ndarray]],
    indices: np.ndarray,
    device: th.device,
) -> dict[str, th.Tensor]:
    return {
        key: th.as_tensor(
            np.stack([observations[index][key] for index in indices]),
            dtype=th.float32,
            device=device,
        )
        for key in observations[0]
    }


def pretrain_actor(
    model,
    observations: Sequence[Mapping[str, np.ndarray]],
    actions: np.ndarray,
    *,
    gradient_steps: int,
    batch_size: int,
    learning_rate: float,
    mobility_dims: int,
    mobility_weight: float = 0.0,
    seed: int = 0,
) -> dict[str, float]:
    """Fit an SAC or Diffusion actor to expert actions before online RL.

    Placement imitation is the primary objective. Mobility imitation is optional
    because the Diffusion actor applies a safety transform that the raw expert
    action does not naturally parameterize.
    """
    if gradient_steps <= 0 or not observations:
        return {"initial_loss": 0.0, "final_loss": 0.0}
    if len(observations) != len(actions):
        raise ValueError("Expert observation/action counts do not match")

    actor = model.policy.actor
    device = model.device
    optimizer = th.optim.Adam(actor.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    actor.set_training_mode(True)

    for _ in range(gradient_steps):
        indices = rng.integers(0, len(observations), size=batch_size)
        obs_batch = _observation_batch(observations, indices, device)
        target_actions = th.as_tensor(
            actions[indices], dtype=th.float32, device=device
        )

        if getattr(actor, "hybrid_categorical_placement", False):
            placement_loss = actor.placement_bc_loss(obs_batch, target_actions)
            mobility_loss = placement_loss.new_zeros(())
            if mobility_weight > 0.0:
                mobility_targets = target_actions[:, :mobility_dims]
                features = actor.extract_features(
                    obs_batch, actor.features_extractor
                )
                if getattr(actor, "mobility_policy", "diffusion") == "gaussian":
                    predicted_mobility, _ = actor._sample_gaussian_mobility(
                        features, deterministic=True
                    )
                    mobility_loss = (
                        predicted_mobility - mobility_targets
                    ).square().mean()
                else:
                    clean_actions = th.atanh(
                        mobility_targets.clamp(-0.95, 0.95)
                    )
                    timesteps = th.randint(
                        0, actor.T, (batch_size,), device=device
                    )
                    noise = th.randn_like(clean_actions)
                    alpha = actor.sqrt_alphas_cumprod.gather(
                        0, timesteps
                    ).reshape(-1, 1)
                    sigma = actor.sqrt_one_minus_alphas_cumprod.gather(
                        0, timesteps
                    ).reshape(-1, 1)
                    noisy_actions = alpha * clean_actions + sigma * noise
                    predicted_noise = actor._epsilon_net(
                        features[:, : actor.core_features_dim],
                        noisy_actions,
                        timesteps,
                    )
                    mobility_loss = (predicted_noise - noise).square().mean()
            loss = placement_loss + mobility_weight * mobility_loss
        elif hasattr(actor, "_epsilon_net"):
            clean_actions = th.atanh(target_actions.clamp(-0.95, 0.95))
            timesteps = th.randint(0, actor.T, (batch_size,), device=device)
            noise = th.randn_like(clean_actions)
            alpha = actor.sqrt_alphas_cumprod.gather(0, timesteps).reshape(-1, 1)
            sigma = actor.sqrt_one_minus_alphas_cumprod.gather(
                0, timesteps
            ).reshape(-1, 1)
            noisy_actions = alpha * clean_actions + sigma * noise
            features = actor.extract_features(obs_batch, actor.features_extractor)
            predicted_noise = actor._epsilon_net(
                features[:, : actor.core_features_dim], noisy_actions, timesteps
            )
            squared_error = (predicted_noise - noise).square()
            placement_loss = squared_error[:, mobility_dims:].mean()
            mobility_loss = squared_error[:, :mobility_dims].mean()
            loss = placement_loss + mobility_weight * mobility_loss
        else:
            predicted_actions = actor(obs_batch, deterministic=True)
            squared_error = (predicted_actions - target_actions).square()
            placement_loss = squared_error[:, mobility_dims:].mean()
            mobility_loss = squared_error[:, :mobility_dims].mean()
            loss = placement_loss + mobility_weight * mobility_loss

        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    if hasattr(model, "critic_target") and hasattr(model, "critic"):
        model.critic_target.load_state_dict(model.critic.state_dict())

    window = min(100, max(1, len(losses) // 5))
    return {
        "initial_loss": float(np.mean(losses[:window])),
        "final_loss": float(np.mean(losses[-window:])),
    }
