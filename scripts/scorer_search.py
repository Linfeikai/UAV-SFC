"""Gate and run a two-loss, from-scratch complete-deployment scorer experiment."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiment_monitor import atomic_json, process_identity, same_process, validate_run
from scripts.collect_scorer_data import REFERENCE_SHA


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root, out = Path(plan['root']), Path(plan['output'])
    assert (root / 'SOURCE_COMMIT').read_text().strip() == plan['source_commit']
    assert digest(plan['reference_config']) == REFERENCE_SHA
    assert digest(plan['parent_evidence']) == plan['parent_evidence_sha256']
    parent = json.loads(Path(plan['parent_evidence']).read_text())
    assert parent['status'] == 'complete' and parent['episodes'] >= 50
    assert all(not r['comparison']['passes_this_cohort'] for r in parent['runs'])
    assert digest(plan['candidate_evidence']) == plan['candidate_evidence_sha256']
    screen = json.loads(Path(plan['candidate_evidence']).read_text())
    assert screen['screening_only'] and not screen['training_performed']
    assert screen['observation_parity_states'] >= 400 and screen['observation_parity_mismatches'] == 0
    baseline = screen['policies']['0']['success_rate_pct']
    assert screen['policies']['simulator_oracle']['success_rate_pct'] >= baseline + .5
    assert max(screen['policies'][str(i)]['success_rate_pct'] for i in range(8)) <= baseline + .01
    assert plan['confirmation_episodes'] >= 50 and plan['validation_episodes'] >= 10
    assert plan['data_episodes'] >= 50
    assert len(plan['runs']) == plan['max_parallel'] == 2
    assert {r['loss'] for r in plan['runs']} == {'regression', 'ranking'}
    assert not out.exists() and out.resolve().is_relative_to(root.resolve())
    if args.validate_only:
        print(json.dumps({'preflight': 'passed', 'original_config_unchanged': True}))
        return
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='collecting', source_commit=plan['source_commit'], started=stamp(),
                 arguments=dict(epochs=plan['epochs'], eval_episodes=plan['confirmation_episodes']),
                 plan=plan, active_pid=None, runs=[])
    for job in plan['runs']:
        state['runs'].append(dict(algorithm=job['variant'], run_id=job['run_id'], url=job['url'],
                                  model_format='deployment_scorer', directory=str(out / job['variant'].lower()),
                                  status='queued', source_commit=plan['source_commit']))
    def save():
        state['elapsed_seconds'] = time.monotonic() - started
        atomic_json(out / 'status.json', state)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    data = out / ('data_seed' + str(plan['training_seed']) + '.npz')
    command = [sys.executable, 'scripts/collect_scorer_data.py', '--config', plan['reference_config'],
               '--output', str(data), '--episodes', str(plan['data_episodes']),
               '--seed', str(plan['data_seed']), '--workers', '8']
    save()
    try:
        with (out / 'data_collection.log').open('x') as log:
            proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(active_pid=proc.pid, active_command=command)
            save()
            assert proc.wait() == 0, 'Data collection failed; see log'
        state['data'] = json.loads(data.with_suffix('.json').read_text())
        assert state['data']['status'] == 'complete' and digest(data) == state['data']['file_sha256']
        state.update(status='running', active_pid=None, active_command=None)
        workers = []
        for job, record in zip(plan['runs'], state['runs']):
            command = [sys.executable, 'scripts/train_placement_scorer.py', '--data', str(data),
                       '--config', plan['reference_config'], '--output', record['directory'],
                       '--run-id', job['run_id'], '--loss', job['loss'], '--seed', str(plan['training_seed']),
                       '--epochs', str(plan['epochs']), '--validation-seed', str(plan['validation_seed']),
                       '--validation-episodes', str(plan['validation_episodes']),
                       '--confirmation-seed', str(plan['confirmation_seed']),
                       '--confirmation-episodes', str(plan['confirmation_episodes'])]
            with (out / (job['variant'].lower() + '.log')).open('x') as log:
                proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(status='training', identity=process_identity(proc.pid), command=command)
            workers.append((proc, record))
            save()
        while workers:
            remaining = []
            for proc, record in workers:
                code = proc.poll()
                if code is None:
                    remaining.append((proc, record))
                    continue
                assert code == 0, 'Scorer worker failed: ' + record['algorithm']
                worker = json.loads((Path(record['directory']) / 'status.json').read_text())
                assert worker['status'] == 'complete'
                record.update(status='complete', updates=worker['full_optimizer_updates'],
                              training_epochs=len(worker['history']), selected_epoch=worker['selected_epoch'],
                              selected_margin=worker['selected_margin'], model_sha256=worker['model_sha256'],
                              comparison=worker['comparison'], finished=worker['finished'],
                              evaluations={record['algorithm'].lower() if k == 'learned' else k: v
                                           for k, v in worker['confirmation'].items()})
                validate_run(record, state, out)
                save()
            workers = remaining
            if workers:
                handles = []
                try:
                    libc = ctypes.CDLL(None, use_errno=True)
                    for proc, _ in workers:
                        fd = libc.syscall(434, int(proc.pid), 0)
                        if fd >= 0:
                            handles.append(fd)
                    if handles:
                        select.select(handles, [], [], 30)
                    else:
                        time.sleep(10)
                finally:
                    for fd in handles:
                        os.close(fd)
        state.update(status='complete', finished=stamp(), active_pid=None,
                     final_fresh_seeds_and_independent_training_replicates_required=True)
        save()
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished=stamp())
        save()
        raise


if __name__ == '__main__':
    main()
