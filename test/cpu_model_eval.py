"""Reproducible CPU evaluation for trained SAC and Diffusion policies."""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algos.diffusion_sac_agent import DiffusionSACAgent
from core.env_config import DEFAULT_CONFIG
from core.sfc_env import SFCEnv
from test.evalu import smart_heuristic_policy


def evaluate(
    model, episodes: int, seed: int, apply_mobility_mask: bool = False
) -> dict[str, float]:
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(
        UAV_TYPES=["gpu", "general", "gpu", "general"],
        NON_GPU_SLOWDOWN=4.0,
        APPLY_MOBILITY_MASK_IN_ENV=apply_mobility_mask,
    )
    env = SFCEnv(config=cfg)
    totals = {
        "reward": 0.0,
        "available": 0,
        "picked": 0,
        "completed": 0,
        "dropped": 0,
        "timeout": 0,
        "energy_j": 0.0,
        "steps": 0,
        "crashed_episodes": 0,
        "charged_steps": 0,
        "harvested_j": 0.0,
        "low_battery_uav_steps": 0,
        "uav_steps": 0,
    }
    minimum_battery_ratio = 1.0

    for episode in range(episodes):
        episode_seed = seed + episode
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        done = False
        while not done:
            if model is None:
                action = smart_heuristic_policy(env)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            totals["reward"] += float(reward)
            totals["available"] += int(info.get("total_available", 0))
            totals["picked"] += int(info.get("actually_picked", 0))
            totals["completed"] += int(info.get("completed_count", 0))
            totals["dropped"] += int(info.get("dropped_count", 0))
            totals["timeout"] += int(info.get("timeout_count", 0))
            totals["energy_j"] += float(info.get("total_energy_J", 0.0))
            totals["steps"] += 1
            ratios = [u.e_battery / u.battery_capacity for u in env.uavs]
            minimum_battery_ratio = min(minimum_battery_ratio, min(ratios))
            totals["low_battery_uav_steps"] += sum(r < 0.25 for r in ratios)
            totals["uav_steps"] += len(ratios)
            totals["charged_steps"] += int(info.get("charge/num_charged", 0) > 0)
            totals["harvested_j"] += float(info.get("charge/total_harvested", 0.0))
            if done and info.get("perf/crashed", 0):
                totals["crashed_episodes"] += 1

    available = max(1, totals["available"])
    picked = max(1, totals["picked"])
    steps = max(1, totals["steps"])
    return {
        "episodes": episodes,
        "mean_episode_reward": totals["reward"] / episodes,
        "success_rate_pct": totals["completed"] / available * 100.0,
        "admission_efficiency_pct": totals["completed"] / picked * 100.0,
        "pick_rate_pct": totals["picked"] / available * 100.0,
        "drop_rate_pct": totals["dropped"] / available * 100.0,
        "timeout_rate_pct": totals["timeout"] / available * 100.0,
        "mean_energy_j_per_step": totals["energy_j"] / steps,
        "mean_episode_steps": totals["steps"] / episodes,
        "crash_episode_rate_pct": totals["crashed_episodes"] / episodes * 100.0,
        "minimum_battery_ratio": minimum_battery_ratio,
        "low_battery_uav_step_rate_pct": (
            totals["low_battery_uav_steps"] / max(1, totals["uav_steps"]) * 100.0
        ),
        "charged_step_rate_pct": totals["charged_steps"] / steps * 100.0,
        "mean_harvested_j_per_episode": totals["harvested_j"] / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", choices=["sac", "diffusion", "heuristic"])
    parser.add_argument("model", nargs="?")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--env-mask", action="store_true")
    args = parser.parse_args()

    if args.algorithm == "heuristic":
        model = None
    else:
        if not args.model:
            parser.error("model path is required for SAC and Diffusion")
        cls = SAC if args.algorithm == "sac" else DiffusionSACAgent
        model = cls.load(args.model, device="cpu")
    result = evaluate(model, args.episodes, args.seed, args.env_mask)
    result["algorithm"] = args.algorithm
    result["model"] = args.model
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
