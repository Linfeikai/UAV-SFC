"""Compare fixed observed deployment recipes and a simulator teacher upper reference.

All recipe actions use observations only. The oracle uses copied environments
and is explicitly a training diagnostic, never a learned-policy result.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--module-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=17000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.root)
    sys.path.insert(0, args.module_dir)
    from placement_candidates import placement_candidates
    from algos.lookahead_teacher import _brake, _greedy_candidate, RECIPES
    from core.sfc_env import SFCEnv
    from test.cpu_model_eval import load_env_config
    from test.evalu import smart_heuristic_policy
    cfg = load_env_config(args.config)
    assert cfg['ARRIVAL_PROB'] == .5 and cfg['UAV_CPU_FREQ'] == 500000000.
    result = dict(screening_only=True, training_performed=False, seed_start=args.seed,
                  episodes=args.episodes, config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                  policies={}, observation_parity_states=0, observation_parity_mismatches=0)
    for policy in list(range(8)) + ['simulator_oracle']:
        started = time.monotonic()
        env = SFCEnv(config=cfg)
        totals = dict(reward=0., available=0, completed=0, energy=0., steps=0, crashed=0, latency=0.)
        chosen = np.zeros(8, dtype=int)
        try:
            for episode in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + episode)
                done = False
                while not done:
                    proposals = placement_candidates(obs)
                    if policy == 'simulator_oracle':
                        base = smart_heuristic_policy(env)
                        base[:2 * env.N] = _brake(env)
                        reference = np.stack([base] + [_greedy_candidate(env, *r) for r in RECIPES])
                        result['observation_parity_states'] += 1
                        result['observation_parity_mismatches'] += int(not np.allclose(proposals, reference, atol=2e-6))
                        scores = []
                        for action in proposals:
                            branch = copy.deepcopy(env)
                            try:
                                _, reward, _, _, info = branch.step(action)
                                scores.append((float(reward), int(info.get('completed_count', 0)),
                                               -float(info.get('total_energy_J', 0))))
                            finally:
                                branch.close()
                        index = max(range(8), key=lambda i: scores[i])
                    else:
                        index = policy
                    chosen[index] += 1
                    obs, reward, terminated, truncated, info = env.step(proposals[index])
                    done = terminated or truncated
                    completed = info.get('completed_count', 0)
                    totals['reward'] += reward
                    totals['available'] += info.get('total_available', 0)
                    totals['completed'] += completed
                    totals['energy'] += info.get('total_energy_J', 0.)
                    totals['latency'] += info.get('avg_latency', 0.) * completed
                    totals['steps'] += 1
                totals['crashed'] += int(any(u.is_crashed for u in env.uavs))
            result['policies'][str(policy)] = dict(
                success_rate_pct=100 * totals['completed'] / max(1, totals['available']),
                crash_episode_rate_pct=100 * totals['crashed'] / args.episodes,
                mean_episode_reward=totals['reward'] / args.episodes,
                mean_energy_j_per_step=totals['energy'] / totals['steps'],
                mean_episode_steps=totals['steps'] / args.episodes,
                mean_completed_sfc_latency=totals['latency'] / max(1, totals['completed']),
                selected_recipe_counts=chosen.tolist(), elapsed_seconds=time.monotonic() - started)
        finally:
            env.close()
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
