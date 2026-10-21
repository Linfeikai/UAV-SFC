"""Measure teacher imitation and slot regret on shared held-out teacher states.

This is a diagnostic, not an independent-episode model performance estimate.
The teacher drives the live trajectory; model actions run only on copies.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', action='append', required=True, help='label=absolute_model_path')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=16000)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.root)
    from algos.lookahead_teacher import lookahead_expert_action, _brake
    from core.sfc_env import SFCEnv
    from main import ALGO_REGISTRY
    from test.cpu_model_eval import load_env_config
    from test.evalu import smart_heuristic_policy

    torch.set_num_threads(1)
    cfg = load_env_config(args.config)
    assert cfg['ARRIVAL_PROB'] == .5 and cfg['UAV_CPU_FREQ'] == 500000000.0
    cfg['RECORD_DEPLOYMENT'] = True
    env = SFCEnv(config=cfg)
    models = {}
    summary = {}
    for spec in args.model:
        label, path = spec.split('=', 1)
        models[label] = ALGO_REGISTRY['DIFFUSION'].load(path, device='cuda')
        summary[label] = dict(model=path, model_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                              valid_placements=0, matching_placements=0, teacher_changes=0,
                              matched_teacher_changes=0, reward_regret_sum=0.,
                              placement_reward_regret_sum=0., mobility_magnitude_sum=0.)
    started = time.monotonic()
    steps = 0
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            random.seed(seed)
            np.random.seed(seed)
            obs, _ = env.reset(seed=seed)
            done = False
            while not done:
                teacher = lookahead_expert_action(env)
                base = smart_heuristic_policy(env)
                base[:2 * env.N] = _brake(env)
                base_branch = copy.deepcopy(env)
                _, _, _, _, base_info = base_branch.step(base)
                base_branch.close()
                task_ids = [ue for ue, _ in env.current_cand_tasks[:env.K]]
                branch = copy.deepcopy(env)
                _, teacher_reward, _, _, teacher_info = branch.step(teacher)
                branch.close()
                for label, model in models.items():
                    torch.manual_seed(seed * 100 + steps % 40)
                    action, _ = model.predict(obs, deterministic=True)
                    branch = copy.deepcopy(env)
                    _, reward, _, _, info = branch.step(action)
                    branch.close()
                    braked = action.copy()
                    braked[:2 * env.N] = _brake(env)
                    branch = copy.deepcopy(env)
                    _, placement_reward, _, _, _ = branch.step(braked)
                    branch.close()
                    item = summary[label]
                    item['reward_regret_sum'] += teacher_reward - reward
                    item['placement_reward_regret_sum'] += teacher_reward - placement_reward
                    item['mobility_magnitude_sum'] += float(np.abs(action[:2 * env.N]).mean())
                    for ue in task_ids:
                        a = teacher_info['detailed_mappings'][ue]
                        b = info['detailed_mappings'][ue]
                        c = base_info['detailed_mappings'][ue]
                        item['valid_placements'] += len(a)
                        item['matching_placements'] += sum(x == y for x, y in zip(a, b))
                        item['teacher_changes'] += sum(x != z for x, z in zip(a, c))
                        item['matched_teacher_changes'] += sum(x != z and x == y for x, y, z in zip(a, b, c))
                obs, _, terminated, truncated, _ = env.step(teacher)
                done = terminated or truncated
                steps += 1
    finally:
        env.close()
    for item in summary.values():
        item['placement_agreement_pct'] = 100 * item['matching_placements'] / max(1, item['valid_placements'])
        item['changed_placement_agreement_pct'] = 100 * item['matched_teacher_changes'] / max(1, item['teacher_changes'])
        item['mean_current_slot_reward_regret'] = item['reward_regret_sum'] / steps
        item['mean_placement_slot_reward_regret'] = item['placement_reward_regret_sum'] / steps
        item['mean_abs_mobility_action'] = item['mobility_magnitude_sum'] / steps
    result = dict(status='complete', diagnostic_only=True, independent_policy_episode_evaluation=False,
                  episodes=args.episodes, seed_start=args.seed, steps=steps, models=summary,
                  config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic() - started)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
