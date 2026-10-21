"""Evaluate a frozen scorer and all fixed-setting comparators on one fresh cohort."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Immutable scorer scientific source root')
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert args.episodes >= 50, 'A fresh qualification cohort needs at least 50 episodes'
    sys.path.insert(0, args.root)
    import torch
    from algos.placement_scorer import ScoredDeploymentPolicy
    from scripts.experiment_monitor import atomic_json
    from scripts.collect_scorer_data import REFERENCE_SHA
    from test.cpu_model_eval import evaluate, load_env_config
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert sha(args.model) == args.expected_sha256
    assert sha(args.config) == REFERENCE_SHA
    torch.set_num_threads(1)
    policy = ScoredDeploymentPolicy.load(args.model)
    config = load_env_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='evaluating', started=datetime.now(timezone.utc).isoformat(),
                 run_id=args.run_id, model=args.model, model_sha256=args.expected_sha256,
                 scientific_source_commit=(Path(args.root) / 'SOURCE_COMMIT').read_text().strip(),
                 evaluator_sha256=sha(__file__), config_sha256=REFERENCE_SHA,
                 seed_start=args.seed, episodes=args.episodes, evaluations={})
    def save():
        state['elapsed_seconds'] = time.monotonic() - started
        atomic_json(output / 'status.json', state)
    save()
    try:
        for name in ('learned', 'static_heuristic', 'heuristic', 'obs_heuristic'):
            tick = time.monotonic()
            with (output / (name + '.stderr.log')).open('x') as err, (output / (name + '.stdout.log')).open('x') as out:
                with redirect_stderr(err), redirect_stdout(out):
                    metrics = evaluate(policy if name == 'learned' else None, args.episodes,
                                       args.seed, env_config=config, observation_heuristic=name == 'obs_heuristic',
                                       mobility_source='zero' if name == 'static_heuristic' else 'default')
            metrics.update(seed_start=args.seed, elapsed_seconds=time.monotonic() - tick)
            state['evaluations'][name] = metrics
            atomic_json(output / (name + '.json'), metrics)
            save()
        assert sha(args.model) == args.expected_sha256, 'Frozen model changed during evaluation'
        strongest = max(('static_heuristic', 'heuristic', 'obs_heuristic'),
                        key=lambda name: state['evaluations'][name]['success_rate_pct'])
        learned = state['evaluations']['learned']
        baseline = state['evaluations'][strongest]
        checks = dict(confirmation_size=args.episodes >= 50,
                      completion_margin=learned['success_rate_pct']-baseline['success_rate_pct'] >= .5,
                      crash_rate=learned['crash_episode_rate_pct'] <= 5,
                      reward=learned['mean_episode_reward'] >= baseline['mean_episode_reward'])
        state.update(status='complete', finished=datetime.now(timezone.utc).isoformat(),
                     comparison=dict(comparator=strongest, checks=checks, passes_this_cohort=all(checks.values()),
                                     completion_margin_percentage_points=learned['success_rate_pct']-baseline['success_rate_pct'],
                                     reward_margin=learned['mean_episode_reward']-baseline['mean_episode_reward']))
        save()
        print(json.dumps({k:state[k] for k in ['run_id','seed_start','episodes','comparison','elapsed_seconds']}),flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        (output / 'error.stderr.log').write_text(traceback.format_exc())
        save()
        raise


if __name__ == '__main__':
    main()
