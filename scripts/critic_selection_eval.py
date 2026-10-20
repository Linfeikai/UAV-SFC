"""Evaluate best-of-K diffusion actions using the minimum of the two learned Qs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test.cpu_model_eval import evaluate, load_env_config, _resolve_config_path
from scripts.experiment_monitor import atomic_json


class CriticSelectionPolicy:
    def __init__(self, model, candidates):
        if candidates < 1:
            raise ValueError('candidates must be positive')
        self.model, self.candidates = model, int(candidates)

    def predict(self, obs, deterministic=True):
        observation, _ = self.model.policy.obs_to_tensor(obs)
        repeated = {key: value.repeat_interleave(self.candidates, dim=0)
                    for key, value in observation.items()}
        with torch.no_grad():
            actions = self.model.actor(repeated, deterministic=deterministic)
            scores = torch.cat(self.model.critic(repeated, actions), dim=1).min(dim=1).values
            if not torch.isfinite(scores).all() or not torch.isfinite(actions).all():
                raise ValueError('Non-finite candidate actions or Q values')
            selected = actions[scores.argmax()].cpu().numpy()
        return self.model.policy.unscale_action(selected), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--config', required=True)
    parser.add_argument('--candidates', nargs='+', type=int, default=[1, 4, 16])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing experiment')
    if args.episodes < 1:
        parser.error('episodes must be positive')
    torch.set_num_threads(1)
    from algos.diffusion_sac_agent import DiffusionSACAgent
    model = DiffusionSACAgent.load(args.model, device=args.device)
    model.policy.set_training_mode(False)
    config_path = Path(_resolve_config_path(str(args.model), args.config))
    config = load_env_config(str(config_path))
    result = dict(status='running', model=str(args.model), device=args.device, seed=args.seed,
                  episodes=args.episodes, model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), runs={})
    atomic_json(args.output, result)
    try:
        for count in args.candidates:
            start = time.perf_counter()
            metrics = evaluate(CriticSelectionPolicy(model, count), args.episodes, args.seed,
                               env_config=config)
            metrics['wall_time_seconds'] = time.perf_counter() - start
            result['runs'][str(count)] = metrics
            atomic_json(args.output, result)
            print(count, json.dumps(metrics), flush=True)
        result['status'] = 'complete'
        atomic_json(args.output, result)
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        atomic_json(args.output, result)
        raise


if __name__ == '__main__':
    main()
