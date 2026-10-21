"""Bounded environment sensitivity screens; transfer evaluation is not retraining."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from omegaconf import OmegaConf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='checking', started=stamp(), pid=os.getpid(),
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                 source_commit=plan['source_commit'], runs=[], active_process=None,
                 screening_only=True, training_performed=False)
    def save():
        state['elapsed_seconds'] = time.monotonic() - started
        write(output / 'status.json', state)
    save()
    try:
        assert (Path(plan['root']) / 'SOURCE_COMMIT').read_text().strip() == plan['source_commit']
        parent = json.loads(Path(plan['parent_status']).read_text())
        assert parent['status'] == 'complete'
        assert 0 < plan['episodes'] <= 20
        for model in plan['models']:
            assert digest(model['path']) == model['sha256']
        state['base_config_sha256'] = digest(plan['base_config'])
        write(output / 'plan.json', plan)
        base = OmegaConf.to_container(OmegaConf.load(plan['base_config']), resolve=True)
        state['status'] = 'evaluating'
        save()
        for setting in plan['settings']:
            folder = output / setting['name']
            folder.mkdir()
            cfg = OmegaConf.to_container(OmegaConf.create(base), resolve=True)
            cfg['env']['ARRIVAL_PROB'] = setting['arrival_probability']
            cfg['uav']['CPU_FREQ'] = setting['cpu_frequency_hz']
            # Preserve reward, batteries, mobility safety, horizon and dimensions.
            assert cfg['reward'] == base['reward']
            for key in ('MAX_STEPS', 'BATTERY_CAPACITY', 'M', 'K', 'APPLY_MOBILITY_MASK_IN_ENV'):
                assert cfg['env'].get(key) == base['env'].get(key)
            config_path = folder / 'config.yaml'
            OmegaConf.save(OmegaConf.create(cfg), config_path)
            policies = [dict(label=m['label'], algorithm='diffusion', model=m['path'], mobility='default')
                        for m in plan['models']]
            policies += [dict(label='static_heuristic', algorithm='heuristic', mobility='zero'),
                         dict(label='heuristic', algorithm='heuristic', mobility='default'),
                         dict(label='obs_heuristic', algorithm='obs_heuristic', mobility='default')]
            for policy in policies:
                command = [sys.executable, '-m', 'test.cpu_model_eval', policy['algorithm']]
                if policy.get('model'):
                    command.append(policy['model'])
                command += ['--config', str(config_path), '--device', 'cuda', '--seed', str(plan['seed']),
                            '--episodes', str(plan['episodes']), '--mobility-source', policy['mobility']]
                record = dict(setting=setting['name'], label=policy['label'], status='running',
                              started=stamp(), command=command, config_sha256=digest(config_path))
                state['runs'].append(record)
                before = time.monotonic()
                result = folder / (policy['label'] + '.json')
                with result.with_suffix('.partial').open('x') as out, result.with_suffix('.stderr.log').open('x') as err:
                    process = subprocess.Popen(command, cwd=plan['root'], stdout=out, stderr=err,
                                               stdin=subprocess.DEVNULL,
                                               env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                    state['active_process'] = dict(pid=process.pid, setting=setting['name'], label=policy['label'])
                    save()
                    returncode = process.wait()
                assert returncode == 0, f'evaluation failed: {setting["name"]}/{policy["label"]}'
                metrics = json.loads(result.with_suffix('.partial').read_text())
                assert metrics['episodes'] == plan['episodes']
                for key in ('success_rate_pct', 'crash_episode_rate_pct', 'mean_episode_reward',
                            'mean_energy_j_per_step', 'mean_episode_steps', 'mean_completed_sfc_latency'):
                    assert math.isfinite(metrics[key]), key
                result.with_suffix('.partial').replace(result)
                record.update(status='complete', finished=stamp(), metrics=metrics,
                              elapsed_seconds=time.monotonic()-before, result_sha256=digest(result))
                state['active_process'] = None
                save()
        state.update(status='complete', finished=stamp(), independent_confirmation_required=True)
        save()
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished=stamp())
        save()
        raise


if __name__ == '__main__':
    main()
