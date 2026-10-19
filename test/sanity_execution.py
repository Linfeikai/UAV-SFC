"""Small fixed-seed environment sanity experiment, not a convergence benchmark."""
import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np

from core.sfc_env import SFCEnv
from main import get_flat_config
from test.evalu import smart_heuristic_policy


def run(config, *, episodes=3, steps=64, seed=42):
    results = []
    for name in ("random", "heuristic"):
        episodes_result = []
        for episode in range(episodes):
            episode_seed = seed + episode
            rng = np.random.default_rng(episode_seed)
            env = SFCEnv(config=config)
            env.reset(seed=episode_seed)
            counts = dict(available=0, completed=0, dropped=0, timeout=0, crash_failed=0)
            reward_sum = 0.0
            crashed = False
            try:
                for index in range(steps):
                    action = (rng.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
                              if name == "random" else smart_heuristic_policy(env))
                    obs, reward, terminated, truncated, info = env.step(action)
                    assert np.isfinite(reward)
                    assert all(np.isfinite(value).all() for value in obs.values())
                    available = int(info["total_available"])
                    completed = int(info.get("completed_count", 0))
                    dropped = int(info.get("dropped_count", 0))
                    timeout = int(info.get("timeout_count", 0))
                    crash_failed = int(info.get("crash_failed_count", 0))
                    assert available == completed + dropped + timeout + crash_failed, info
                    assert completed + int(info["failed_count"]) == available, info
                    # No UAV can compute for longer than the slot window.
                    compute_budget = env.N * (env.time_slot - env.dt_fly) * env.config["UAV_FULL_LOAD_POWER"]
                    assert info.get("total_compute_energy", 0) <= compute_budget + 1e-3
                    for key, value in zip(counts, (available, completed, dropped, timeout, crash_failed)):
                        counts[key] += value
                    reward_sum += reward
                    if terminated or truncated:
                        crashed = terminated
                        break
                episodes_result.append(dict(seed=episode_seed, steps=index + 1,
                                            reward=reward_sum, crashed=crashed, **counts))
            finally:
                env.close()
        totals = {key: sum(item[key] for item in episodes_result) for key in counts}
        results.append(dict(
            policy=name, episodes=episodes_result, totals=totals,
            success_rate_pct=100 * totals["completed"] / max(1, totals["available"]),
            mean_reward=float(np.mean([item["reward"] for item in episodes_result])),
        ))
    return dict(seed=seed, episodes=episodes, max_steps=steps, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1:
        parser.error("episodes and steps must be positive")
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "conf")):
        config = get_flat_config(compose(config_name="config"))
    result = run(config, episodes=args.episodes, steps=args.steps, seed=args.seed)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
