"""Screen a training teacher on fresh episodes; this is not a learned policy eval."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--module-dir', help='Temporary diagnostic module directory, if outside the source root')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=13100)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error('episodes must be positive')
    sys.path.insert(0, args.root)
    if args.module_dir:
        sys.path.insert(0, args.module_dir)
        from lookahead_teacher import lookahead_expert_action
    else:
        from algos.lookahead_teacher import lookahead_expert_action
    from test.cpu_model_eval import load_env_config
    from core.sfc_env import SFCEnv
    from test.evalu import smart_heuristic_policy

    cfg = load_env_config(args.config)
    result = dict(config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                  episodes=args.episodes, seed_start=args.seed, screening_only=True,
                  training_performed=False, policies={})
    for label in ('static_heuristic', 'lookahead_teacher'):
        started = time.monotonic()
        env = SFCEnv(config=cfg)
        totals = dict(reward=0., available=0, completed=0, energy=0., steps=0,
                      crashed=0, latency_weighted=0.)
        episodes = []
        try:
            for ep in range(args.episodes):
                seed = args.seed + ep
                random.seed(seed)
                np.random.seed(seed)
                env.reset(seed=seed)
                episode = dict(seed=seed, completed=0, available=0, reward=0.)
                done = False
                while not done:
                    if label == 'lookahead_teacher':
                        action = lookahead_expert_action(env)
                    else:
                        action = smart_heuristic_policy(env)
                        action[:2 * env.N] = 0
                    _, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    completed = info.get('completed_count', 0)
                    available = info.get('total_available', 0)
                    totals['reward'] += reward
                    totals['available'] += available
                    totals['completed'] += completed
                    totals['energy'] += info.get('total_energy_J', 0)
                    totals['steps'] += 1
                    totals['latency_weighted'] += info.get('avg_latency', 0.) * completed
                    episode['completed'] += completed
                    episode['available'] += available
                    episode['reward'] += reward
                totals['crashed'] += int(any(u.is_crashed for u in env.uavs))
                episodes.append(episode)
            result['policies'][label] = dict(
                success_rate_pct=100 * totals['completed'] / max(1, totals['available']),
                crash_episode_rate_pct=100 * totals['crashed'] / args.episodes,
                mean_episode_reward=totals['reward'] / args.episodes,
                mean_energy_j_per_step=totals['energy'] / totals['steps'],
                mean_episode_steps=totals['steps'] / args.episodes,
                mean_completed_sfc_latency=totals['latency_weighted'] / max(1, totals['completed']),
                elapsed_seconds=time.monotonic() - started, episode_records=episodes)
        finally:
            env.close()
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
