"""Collect counterfactual current-slot rewards in the immutable original setting."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REFERENCE_SHA = '2ae61b7f8d7f225b283a9a343739f7139388c32d12f86382815d88c9dc7f43f5'
EQUIVALENT_REFERENCE_SHA = '2d38365544d73b9f65b4c6b027f55d2414e9573e5b1ec1ecdb3989effb5a8fb7'


def load_fixed_reference(path):
    """Accept only the original YAML or its audited physical-equivalent backup."""
    actual_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert actual_sha in {REFERENCE_SHA, EQUIVALENT_REFERENCE_SHA}
    from test.cpu_model_eval import load_env_config
    cfg = load_env_config(str(path))
    manifest_path = Path(__file__).resolve().parents[1] / 'experiments/server_20261021/fixed_hybrid_environment_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    normalized = json.loads(json.dumps(cfg, sort_keys=True,
                                      default=lambda value: sorted(value) if isinstance(value, set) else value))
    assert normalized == manifest['full_flat_config'], 'Original physical setting changed'
    return cfg, actual_sha


def collect_chunk(job):
    from algos.placement_candidates import placement_candidates, candidate_features
    from core.sfc_env import SFCEnv
    cfg, seeds = job
    env = SFCEnv(config=cfg)
    rows, rewards, completed, episode_ids = [], [], [], []
    selected_counts = np.zeros(8, dtype=np.int64)
    try:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            obs, _ = env.reset(seed=int(seed))
            done = False
            while not done:
                proposals = placement_candidates(obs)
                slot_rewards, slot_completed = [], []
                for action in proposals:
                    branch = copy.deepcopy(env)
                    try:
                        _, reward, _, _, info = branch.step(action)
                        slot_rewards.append(reward)
                        slot_completed.append(info.get('completed_count', 0))
                    finally:
                        branch.close()
                rows.append(candidate_features(obs, proposals))
                rewards.append(slot_rewards)
                completed.append(slot_completed)
                episode_ids.append(seed)
                # Expose training to alternative trajectories as well as teacher states.
                chosen = int(np.argmax(slot_rewards)) if rng.random() < .8 else int(rng.integers(8))
                selected_counts[chosen] += 1
                obs, _, terminated, truncated, _ = env.step(proposals[chosen])
                done = terminated or truncated
    finally:
        env.close()
    return (np.stack(rows), np.asarray(rewards, dtype=np.float32),
            np.asarray(completed, dtype=np.int16), np.asarray(episode_ids, dtype=np.int64), selected_counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episodes', type=int, default=250)
    parser.add_argument('--seed', type=int, default=620000)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    config = Path(args.config)
    cfg, config_sha = load_fixed_reference(config)
    assert cfg['ARRIVAL_PROB'] == .5 and cfg['UAV_CPU_FREQ'] == 500000000.
    output = Path(args.output)
    assert not output.exists() and args.episodes > 0 and 1 <= args.workers <= 8
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    seeds = np.arange(args.seed, args.seed + args.episodes)
    chunks = [x.tolist() for x in np.array_split(seeds, min(args.workers, args.episodes))]
    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(collect_chunk, [(cfg, chunk) for chunk in chunks]))
    features, rewards, completed, episode_ids = [np.concatenate([r[i] for r in results]) for i in range(4)]
    assert np.isfinite(features).all() and np.isfinite(rewards).all()
    np.savez_compressed(output, features=features, rewards=rewards, completed=completed, episode_ids=episode_ids)
    metadata = dict(status='complete', pid=os.getpid(), config_sha256=config_sha, original_reference_sha256=REFERENCE_SHA,
                    original_environment=True, episodes=args.episodes, seed_start=args.seed,
                    actual_environment_interactions=len(features), counterfactual_simulator_steps=len(features) * 8,
                    feature_shape=list(features.shape), elapsed_seconds=time.monotonic() - started,
                    file_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    selected_recipe_counts=np.stack([r[4] for r in results]).sum(0).tolist())
    output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()
