"""Factorial policy diagnostic with geometry-consistent mobility interventions.

Use one saved model, one device and identical evaluation seeds for all four cells.
Unlike post-hoc action slicing, learned placement is recomputed conditioned on
an intervened mobility action. This does not train or modify the saved model.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.env_config import DEFAULT_CONFIG
from test.cpu_model_eval import (
    _resolve_config_path, evaluate, load_env_config, observation_heuristic_policy,
)


def diagnostic_config(model_path, config_path):
    """Accept the same Hydra run directories as the standard evaluator."""
    resolved = Path(_resolve_config_path(str(model_path), str(config_path)))
    return resolved, load_env_config(str(resolved))


class ComposedPolicy:
    def __init__(self, model, config, mobility='learned', placement='learned'):
        self.model = model
        self.config = {**DEFAULT_CONFIG, **config}
        self.mobility = mobility
        self.placement = placement
        self.dims = int(self.config['NUM_UAVS']) * 2
        if mobility not in {'learned', 'heuristic'} or placement not in {'learned', 'heuristic'}:
            raise ValueError('Unsupported intervention')
        if not np.allclose(model.action_space.low, -1) or not np.allclose(model.action_space.high, 1):
            raise ValueError('Interventions require the symmetric normalized action space')

    def predict(self, obs, deterministic=True):
        heuristic = None
        if 'heuristic' in (self.mobility, self.placement):
            cfg = self.config
            heuristic = observation_heuristic_policy(
                obs, n_uavs=cfg['NUM_UAVS'], m_candidates=cfg['M'],
                decision_tasks=cfg['K'], max_vnf_len=cfg['MAX_VNF_LEN'], grid_res=cfg['GRID_RES'])
        if self.mobility == self.placement == 'heuristic':
            return heuristic, None
        learned, state = self.model.predict(obs, deterministic=deterministic)
        action = np.asarray(learned).copy()
        if self.mobility == 'heuristic':
            action[:self.dims] = heuristic[:self.dims]
            actor = self.model.actor
            if (self.placement == 'learned' and getattr(actor, 'hybrid_categorical_placement', False)
                    and getattr(actor, 'relational_placement', False)
                    and getattr(actor, 'placement_use_post_mobility_geometry', False)):
                obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
                mobility_tensor = torch.as_tensor(action[None, :self.dims],
                                                  dtype=torch.float32, device=self.model.device)
                with torch.no_grad():
                    encoded, _, _ = actor.sample_placement(
                        obs_tensor, mobility_action=mobility_tensor,
                        deterministic=deterministic, straight_through=False)
                action[self.dims:] = encoded[0].cpu().numpy()
        if self.placement == 'heuristic':
            action[self.dims:] = heuristic[self.dims:]
        if not np.isfinite(action).all():
            raise ValueError('Non-finite composed action')
        return action, state


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('algorithm', choices=['diffusion', 'gaussian'])
    parser.add_argument('model', type=Path)
    parser.add_argument('--config', type=Path, required=True,
                        help='Hydra config file or training run directory')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    if args.output.exists():
        parser.error('Refusing to overwrite an existing diagnostic')
    torch.set_num_threads(1)
    from main import ALGO_REGISTRY
    model = ALGO_REGISTRY[args.algorithm.upper()].load(args.model, device=args.device)
    model.policy.set_training_mode(False)
    config_path, config = diagnostic_config(args.model, args.config)
    result = dict(status='running', algorithm=args.algorithm, model=str(args.model),
                  model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  diagnostic_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  config=str(config_path),
                  config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  versions={name: version(name) for name in
                            ('torch', 'numpy', 'stable-baselines3', 'gymnasium')},
                  seed=args.seed, episodes=args.episodes, device=args.device,
                  intervention='recompute learned placement after mobility intervention', cells={})
    for code, mobility, placement in [('DD', 'learned', 'learned'),
                                      ('HD', 'heuristic', 'learned'),
                                      ('DH', 'learned', 'heuristic'),
                                      ('HH', 'heuristic', 'heuristic')]:
        print('Evaluating', code, mobility, placement, flush=True)
        policy = ComposedPolicy(model, config, mobility, placement)
        try:
            metrics = evaluate(policy, args.episodes, args.seed, env_config=config)
            result['cells'][code] = dict(mobility=mobility, placement=placement, **metrics)
            atomic_write(args.output, result)
        except BaseException as error:
            result.update(status='failed', error=repr(error))
            atomic_write(args.output, result)
            raise
    result['status'] = 'complete'
    atomic_write(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
