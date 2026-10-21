"""Run an isolated four-policy confirmation batch; no training or SSH credentials."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='checking', started=stamp(), source_commit=plan['source_commit'],
                 pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                 seed_start=plan['seed_start'], seed_end=plan['seed_end'],
                 episodes_per_policy=plan['episodes_per_policy'], jobs=[], active_process=None)
    def save():
        state['elapsed_seconds'] = time.monotonic() - started
        atomic_json(output / 'status.json', state)
    save()
    try:
        assert plan['episodes_per_policy'] >= 50
        assert plan['seed_end'] == plan['seed_start'] + plan['episodes_per_policy'] - 1
        assert len(plan['jobs']) == 4
        source = Path(plan['source_root'])
        assert (source / 'SOURCE_COMMIT').read_text().strip() == plan['source_commit']
        assert digest(plan['source_checkpoint']) == plan['source_checkpoint_sha256']
        state['model_sha256'] = digest(plan['source_checkpoint'])
        state['config_sha256'] = digest(plan['config'])
        state['source_hashes'] = {file: digest(source / file) for file in plan['source_hashes']}
        assert state['source_hashes'] == plan['source_hashes']
        overrides = (Path(plan['config']).parent / 'overrides.yaml').read_text()
        assert 'env=heterogeneous_hybrid' in overrides
        state['python'] = sys.version
        state['packages'] = {name: importlib.metadata.version(name) for name in
                             ['torch', 'numpy', 'stable-baselines3', 'omegaconf', 'gymnasium', 'hydra-core']}
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                              '--format=csv,noheader'], capture_output=True, text=True, check=True)
        state['gpu'] = gpu.stdout.strip()
        atomic_json(output / 'plan.json', plan)
        state['status'] = 'evaluating'
        save()
        for job in plan['jobs']:
            argv = job['argv']
            assert argv[argv.index('--config') + 1] == plan['config']
            assert argv[argv.index('--seed') + 1] == str(plan['seed_start'])
            assert argv[argv.index('--episodes') + 1] == str(plan['episodes_per_policy'])
            if job['label'] == 'stationary_selected':
                assert argv[3] == 'diffusion' and argv[4] == plan['source_checkpoint']
                assert argv[argv.index('--mobility-source') + 1] == 'default'
            stdout, stderr = Path(job['stdout']), Path(job['stderr'])
            assert stdout.parent == output and stderr.parent == output
            entry = dict(label=job['label'], started=stamp(), status='running', command=argv)
            state['jobs'].append(entry)
            step_started = time.monotonic()
            with stdout.with_suffix('.partial').open('x') as out, stderr.open('x') as err:
                proc = subprocess.Popen(argv, cwd=job['cwd'], stdout=out, stderr=err,
                                        env=dict(os.environ, **plan['environment']))
                state['active_process'] = dict(pid=proc.pid, label=job['label'])
                save()
                code = proc.wait()
            entry['returncode'] = code
            entry['elapsed_seconds'] = time.monotonic() - step_started
            if code:
                raise RuntimeError(f"Evaluator failed for {job['label']}; see {stderr}")
            metrics = json.loads(stdout.with_suffix('.partial').read_text())
            assert metrics['episodes'] == plan['episodes_per_policy']
            for key in plan['decision_gate']['required_metrics']:
                assert key in metrics and math.isfinite(float(metrics[key])), key
            assert digest(plan['config']) == state['config_sha256']
            stdout.with_suffix('.partial').replace(stdout)
            entry.update(status='complete', finished=stamp(), metrics=metrics,
                         output_sha256=digest(stdout))
            state['active_process'] = None
            save()
        learned = next(j['metrics'] for j in state['jobs'] if j['label'] == 'stationary_selected')
        baselines = [j for j in state['jobs'] if j['label'] != 'stationary_selected']
        best = max(baselines, key=lambda j: j['metrics']['success_rate_pct'])
        baseline = best['metrics']
        margin = learned['success_rate_pct'] - baseline['success_rate_pct']
        checks = dict(completion_margin=margin >= 0.5,
                      crash_rate=learned['crash_episode_rate_pct'] <= 5.0,
                      reward=learned['mean_episode_reward'] >= baseline['mean_episode_reward'])
        state['comparison'] = dict(comparator=best['label'], completion_margin_percentage_points=margin,
                                   reward_margin=learned['mean_episode_reward']-baseline['mean_episode_reward'],
                                   checks=checks, passes_this_cohort=all(checks.values()),
                                   final_generalization_and_training_replicates_still_required=True)
        state.update(status='complete', finished=stamp())
        save()
        print(json.dumps(state['comparison'], indent=2))
    except BaseException as error:
        state.update(status='failed', finished=stamp(), error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
