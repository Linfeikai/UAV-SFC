"""Train gated algorithm candidates in one frozen scenario and compare all baselines."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, state):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    decision = json.loads(Path(plan['decision_file']).read_text())
    evidence = Path(decision['remote_evidence'])
    assert sha(evidence) == decision['evidence_sha256'] == plan['launch_gate']['evidence_sha256']
    screen = json.loads(evidence.read_text())
    assert screen['status'] == 'complete' and screen['screening_only']
    assert len(screen['runs']) == 20 and all(r['status'] == 'complete' for r in screen['runs'])
    assert decision['status'] == 'train_balanced_scenario_for_20000_steps'
    assert decision['setting'] == plan['setting']['screen_name']
    selected = [r for r in screen['runs'] if r['setting'] == decision['setting']]
    assert len(selected) == 5
    assert all(r['metrics']['episodes'] == 10 for r in selected)
    parent = json.loads(Path(plan['parent_status']).read_text())
    assert parent['status'] == 'complete' and parent['static_comparator']['episodes'] >= 50
    assert all(all(v['episodes'] >= 50 for v in r['evaluations'].values()) for r in parent['runs'])
    assert (Path(plan['root']) / 'SOURCE_COMMIT').read_text().strip() == plan['source_commit']
    assert sha(plan['source_checkpoint']) == plan['source_checkpoint_sha256']
    assert 0 < len(plan['runs']) <= 2 and plan['max_parallel'] <= 2
    assert not Path(plan['output']).exists()
    assert all(not Path(j['folder']).exists() for j in plan['runs'])
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from main import get_flat_config
    reference = OmegaConf.to_container(OmegaConf.load(plan['screen_config']), resolve=True)
    reference_flat = get_flat_config(reference)
    reference_flat.setdefault("BANDWIDTH_HZ", 2000000.0)
    for job in plan['runs']:
        assert 'env=heterogeneous_balanced' in job['overrides']
        assert 'bc_pretrain_steps=0' in job['overrides']
        assert 'resume_model=' + plan['source_checkpoint'] in job['overrides']
        with initialize_config_dir(config_dir=str(Path(plan['root']) / 'conf'), version_base=None):
            composed = compose(config_name='config', overrides=job['overrides'])
        actual = get_flat_config(composed)
        assert actual == reference_flat, 'Training environment differs from the frozen screened setting'
        assert actual['ARRIVAL_PROB'] == 0.3 and actual['UAV_CPU_FREQ'] == 300000000.0
    output = Path(plan['control_output'])
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status='training', started=stamp(), pid=os.getpid(), source_commit=plan['source_commit'],
                 source_model_sha256=plan['source_checkpoint_sha256'], decision_sha256=sha(plan['decision_file']),
                 boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                 pipeline=plan['output'], active_process=None)
    started = time.monotonic()
    def snapshot():
        state['elapsed_seconds'] = time.monotonic() - started
        save(output / 'status.json', state)
    def run(command, label):
        with (output / (label + '.log')).open('x') as log:
            proc = subprocess.Popen(command, cwd=plan['root'], stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            state['active_process'] = dict(pid=proc.pid, label=label, command=command)
            snapshot()
            code = proc.wait()
        state['active_process'] = None
        snapshot()
        if code:
            raise RuntimeError(f'{label} exited {code}; see its log')
    snapshot()
    try:
        run([sys.executable, 'scripts/motion_search.py', '--plan', str(args.plan.resolve())], 'motion_search')
        pipeline = json.loads((Path(plan['output']) / 'status.json').read_text())
        assert pipeline['status'] == 'complete'
        state['status'] = 'static_comparator'
        snapshot()
        command = plan['additional_confirmation_command']
        with (output / 'static_heuristic.json').open('x') as out, (output / 'static_heuristic.stderr.log').open('x') as err:
            proc = subprocess.Popen(command, cwd=plan['root'], stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            state['active_process'] = dict(pid=proc.pid, label='static_heuristic')
            snapshot()
            code = proc.wait()
        if code:
            raise RuntimeError(f'Static comparator exited {code}')
        baseline = json.loads((output / 'static_heuristic.json').read_text())
        assert baseline['episodes'] >= 50
        state['static_comparator'] = baseline
        state['runs'] = pipeline['runs']
        for record in state['runs']:
            evaluations = record['evaluations']
            learned = evaluations[record['algorithm'].lower()]
            candidates = dict(static_heuristic=baseline, heuristic=evaluations['heuristic'], obs_heuristic=evaluations['obs_heuristic'])
            best = max(candidates, key=lambda name: candidates[name]['success_rate_pct'])
            comparator = candidates[best]
            margin = learned['success_rate_pct'] - comparator['success_rate_pct']
            checks = dict(completion_margin=margin >= 0.5, crash_rate=learned['crash_episode_rate_pct'] <= 5,
                          reward=learned['mean_episode_reward'] >= comparator['mean_episode_reward'])
            record['comparison'] = dict(comparator=best, completion_margin_percentage_points=margin,
                                        reward_margin=learned['mean_episode_reward']-comparator['mean_episode_reward'],
                                        checks=checks, passes_this_cohort=all(checks.values()))
        state.update(status='complete', finished=stamp(), active_process=None,
                     final_generalization_and_training_replicates_still_required=True)
        snapshot()
    except BaseException as error:
        state.update(status='failed', error=str(error), finished=stamp())
        snapshot()
        raise


if __name__ == '__main__':
    main()
