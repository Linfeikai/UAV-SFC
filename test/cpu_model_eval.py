"""Reproducible CPU evaluation for trained SAC and Diffusion policies."""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from stable_baselines3 import SAC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algos.diffusion_sac_agent import DiffusionSACAgent
from algos.gaussian_hybrid_sac_agent import GaussianHybridSACAgent
from core.env_config import DEFAULT_CONFIG
from core.sfc_env import SFCEnv
from test.evalu import smart_heuristic_policy


def observation_heuristic_policy(
    obs: dict[str, np.ndarray],
    *,
    n_uavs: int,
    m_candidates: int,
    decision_tasks: int,
    max_vnf_len: int,
    grid_res: int,
) -> np.ndarray:
    """Observation-only heuristic with no access to environment internals."""
    state = obs["state"]
    uavs = state[: n_uavs * 10].reshape(n_uavs, 10)
    candidate_offset = n_uavs * 10 + grid_res * grid_res * 3
    candidates = state[
        candidate_offset : candidate_offset + m_candidates * 5
    ].reshape(m_candidates, 5)
    charger_position = state[-2:]
    bounds = obs["mobility_bounds"].reshape(n_uavs, 4)

    primary_position = candidates[0, :2] if candidates[0, 2] > 0 else None
    mobility = np.zeros((n_uavs, 2), dtype=np.float32)
    for uav_id, uav in enumerate(uavs):
        if uav[5] > 0.5:
            continue
        target = charger_position if uav[4] < 0.25 else primary_position
        if target is None:
            continue
        direction = target - uav[:2]
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        direction[0] *= bounds[uav_id, 1 if direction[0] >= 0 else 0]
        direction[1] *= bounds[uav_id, 3 if direction[1] >= 0 else 2]
        mobility[uav_id] = direction

    context = obs["placement_context"].reshape(
        decision_tasks, max_vnf_len, n_uavs + 1
    )
    remaining = np.ones(n_uavs, dtype=np.float32)
    placements = np.full(
        (decision_tasks, max_vnf_len, n_uavs), -0.5, dtype=np.float32
    )
    crashed = uavs[:, 5] > 0.5
    for task_id in range(decision_tasks):
        source = candidates[task_id, :2]
        for vnf_id in range(max_vnf_len):
            requirements = context[task_id, vnf_id, :n_uavs]
            if context[task_id, vnf_id, n_uavs] < 0.5:
                continue
            feasible = (requirements <= remaining + 1e-6) & ~crashed
            candidates_mask = feasible if feasible.any() else ~crashed
            if not candidates_mask.any():
                candidates_mask = np.ones(n_uavs, dtype=bool)
            distances = np.linalg.norm(uavs[:, :2] - source, axis=1)
            positive_requirements = requirements[requirements > 1e-9]
            requirement_ratio = (
                float(positive_requirements.max() / positive_requirements.min())
                if positive_requirements.size
                else 1.0
            )
            if requirement_ratio > 1.5:
                score = requirements + 0.01 * distances
            else:
                score = distances + 0.05 * requirements
            score = np.where(candidates_mask, score, np.inf)
            selected = int(np.argmin(score))
            placements[task_id, vnf_id, selected] = 0.5
            remaining[selected] = max(
                0.0, remaining[selected] - requirements[selected]
            )
            source = uavs[selected, :2]
    return np.concatenate([mobility.ravel(), placements.ravel()])


def _resolve_config_path(model_path: str | None, config_path: str | None) -> str | None:
    if config_path:
        resolved = os.path.abspath(config_path)
        if os.path.isdir(resolved):
            resolved = os.path.join(resolved, ".hydra", "config.yaml")
        return resolved
    if model_path:
        search_dir = os.path.dirname(os.path.abspath(model_path))
        while True:
            candidate = os.path.join(search_dir, ".hydra", "config.yaml")
            if os.path.isfile(candidate):
                return candidate
            parent = os.path.dirname(search_dir)
            if parent == search_dir:
                break
            search_dir = parent
    return None


def load_env_config(config_path: str | None) -> dict:
    """Rebuild the flat environment config used by a Hydra training run."""
    cfg = DEFAULT_CONFIG.copy()
    if config_path is None:
        return cfg
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Hydra config not found: {config_path}")

    hydra_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(hydra_cfg, dict):
        raise TypeError(f"Expected a mapping in {config_path}")
    cfg.update(hydra_cfg.get("env", {}))
    cfg.update({f"UAV_{key}": value for key, value in hydra_cfg.get("uav", {}).items()})
    for key, value in hydra_cfg.get("reward", {}).items():
        cfg[key if key in {"W_ENERGY", "W_CHARGE"} else f"RWD_{key}"] = value
    return cfg


def evaluate(
    model,
    episodes: int,
    seed: int,
    apply_mobility_mask: bool = False,
    env_config: dict | None = None,
    observation_heuristic: bool = False,
    mobility_source: str = "default",
    placement_source: str = "default",
) -> dict[str, float]:
    cfg = (env_config or DEFAULT_CONFIG).copy()
    cfg.setdefault("UAV_TYPES", ["gpu", "general", "gpu", "general"])
    cfg.setdefault("NON_GPU_SLOWDOWN", 4.0)
    if apply_mobility_mask:
        cfg["APPLY_MOBILITY_MASK_IN_ENV"] = True
    env = SFCEnv(config=cfg)
    if mobility_source == "zero" and model is not None:
        raise ValueError("Use ComposedPolicy to recompute learned placement after a motion intervention")
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
        "placement_affinity_sum": 0.0,
        "placement_count": 0,
        "gpu_vnf_count": 0,
        "gpu_vnf_on_gpu": 0,
        "latency_weighted_sum": 0.0,
        "latency_completed_count": 0,
        "mobility_abs_sum": 0.0,
        "mobility_square_sum": 0.0,
        "mobility_component_count": 0,
        "mobility_near_zero_count": 0,
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
            observation_action = None
            if observation_heuristic or mobility_source == "obs_heuristic" or placement_source == "obs_heuristic":
                observation_action = observation_heuristic_policy(
                    obs,
                    n_uavs=env.N,
                    m_candidates=env.M,
                    decision_tasks=env.K,
                    max_vnf_len=env.L,
                    grid_res=env.grid_res,
                )
            if observation_heuristic:
                action = observation_action
            elif model is None:
                action = smart_heuristic_policy(env)
            else:
                learned_action, _ = model.predict(obs, deterministic=True)
                action = learned_action
                mobility_dim = env.N * 2
                if mobility_source == "obs_heuristic":
                    action = action.copy()
                    action[:mobility_dim] = observation_action[:mobility_dim]
                if placement_source == "obs_heuristic":
                    action = action.copy()
                    action[mobility_dim:] = observation_action[mobility_dim:]
            if mobility_source == "zero":
                action = action.copy()
                action[:env.N * 2] = 0.0
            mobility_components = np.asarray(action[: env.N * 2], dtype=np.float64)
            totals["mobility_abs_sum"] += float(np.abs(mobility_components).sum())
            totals["mobility_square_sum"] += float(
                np.square(mobility_components).sum()
            )
            totals["mobility_component_count"] += mobility_components.size
            totals["mobility_near_zero_count"] += int(
                (np.abs(mobility_components) < 0.05).sum()
            )
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
            totals["placement_affinity_sum"] += float(
                info.get("placement_affinity_sum", 0.0)
            )
            totals["placement_count"] += int(info.get("placement_count", 0))
            totals["gpu_vnf_count"] += int(info.get("gpu_vnf_count", 0))
            totals["gpu_vnf_on_gpu"] += int(info.get("gpu_vnf_on_gpu", 0))
            completed_this_step = int(info.get("completed_count", 0))
            totals["latency_weighted_sum"] += (
                float(info.get("avg_latency", 0.0)) * completed_this_step
            )
            totals["latency_completed_count"] += completed_this_step
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
        "mean_placement_affinity": totals["placement_affinity_sum"]
        / max(1, totals["placement_count"]),
        "gpu_vnf_on_gpu_pct": totals["gpu_vnf_on_gpu"]
        / max(1, totals["gpu_vnf_count"])
        * 100.0,
        "mean_completed_sfc_latency": totals["latency_weighted_sum"]
        / max(1, totals["latency_completed_count"]),
        "mean_abs_mobility_action": totals["mobility_abs_sum"]
        / max(1, totals["mobility_component_count"]),
        "rms_mobility_action": np.sqrt(
            totals["mobility_square_sum"]
            / max(1, totals["mobility_component_count"])
        ),
        "near_zero_mobility_component_pct": totals["mobility_near_zero_count"]
        / max(1, totals["mobility_component_count"])
        * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "algorithm",
        choices=["sac", "diffusion", "gaussian", "heuristic", "obs_heuristic"],
    )
    parser.add_argument("model", nargs="?")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for model inference (default: cpu)",
    )
    parser.add_argument("--env-mask", action="store_true")
    parser.add_argument(
        "--config",
        help="Hydra config file or experiment directory; auto-detected from the model",
    )
    parser.add_argument("--placement-affinity-bias", type=float)
    parser.add_argument("--placement-load-bias", type=float)
    parser.add_argument("--placement-distance-bias", type=float)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="Override an environment config value; may be repeated",
    )
    parser.add_argument(
        "--mobility-source",
        choices=["default", "obs_heuristic", "zero"],
        default="default",
    )
    parser.add_argument(
        "--placement-source",
        choices=["default", "obs_heuristic"],
        default="default",
    )
    args = parser.parse_args()

    if args.algorithm in {"heuristic", "obs_heuristic"}:
        model = None
    else:
        if not args.model:
            parser.error("model path is required for SAC and Diffusion")
        classes = {
            "sac": SAC,
            "diffusion": DiffusionSACAgent,
            "gaussian": GaussianHybridSACAgent,
        }
        cls = classes[args.algorithm]
        model = cls.load(args.model, device=args.device)
    config_path = _resolve_config_path(args.model, args.config)
    env_config = load_env_config(config_path)
    for item in args.override:
        if "=" not in item:
            parser.error(f"invalid --override value: {item}")
        key, raw_value = item.split("=", 1)
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        env_config[key] = value
    for arg_name, config_name in (
        ("placement_affinity_bias", "PLACEMENT_AFFINITY_BIAS"),
        ("placement_load_bias", "PLACEMENT_LOAD_BIAS"),
        ("placement_distance_bias", "PLACEMENT_DISTANCE_BIAS"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            env_config[config_name] = value
    result = evaluate(
        model,
        args.episodes,
        args.seed,
        args.env_mask,
        env_config=env_config,
        observation_heuristic=args.algorithm == "obs_heuristic",
        mobility_source=args.mobility_source,
        placement_source=args.placement_source,
    )
    result["algorithm"] = args.algorithm
    result["mobility_source"] = args.mobility_source
    result["placement_source"] = args.placement_source
    result["model"] = args.model
    result["config"] = config_path
    result["reward_success"] = env_config["RWD_SUCCESS"]
    result["reward_drop"] = env_config["RWD_DROP"]
    result["placement_affinity_bias"] = env_config.get(
        "PLACEMENT_AFFINITY_BIAS", 0.0
    )
    result["placement_load_bias"] = env_config.get("PLACEMENT_LOAD_BIAS", 0.0)
    result["placement_distance_bias"] = env_config.get(
        "PLACEMENT_DISTANCE_BIAS", 0.0
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
