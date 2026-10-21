"""Replicate the frozen ranking protocol with independent initialization and data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_scorer_data import load_fixed_reference
from scripts.experiment_monitor import atomic_json

FROZEN_SHA = 'a5afe5c2bb2c73d7a626d8df86a6b35666565fd9d140adff7fce47bd7ced6cd8'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_fresh_gate(path):
    state = json.loads(Path(path).read_text())
    assert state['status'] == 'complete' and state['model_sha256'] == FROZEN_SHA
    assert state['episodes'] >= 100 and state['seed_start'] == 90000
    assert state['comparison']['passes_this_cohort']
    evaluations = state['evaluations']
    assert set(evaluations) == {'learned', 'static_heuristic', 'heuristic', 'obs_heuristic'}
    assert all(v['episodes'] == state['episodes'] for v in evaluations.values())
    strongest = max((v for k,v in evaluations.items() if k != 'learned'), key=lambda v:v['success_rate_pct'])
    learned = evaluations['learned']
    assert learned['success_rate_pct'] - strongest['success_rate_pct'] >= .5
    assert learned['crash_episode_rate_pct'] <= 5
    assert learned['mean_episode_reward'] >= strongest['mean_episode_reward']
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--fresh-gate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = (root / 'SOURCE_COMMIT').read_text().strip()
    gate = verify_fresh_gate(args.fresh_gate)
    _, config_sha = load_fixed_reference(args.config)
    assert 1 <= args.workers <= 8
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='collecting', started=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                 source_commit=commit, config_sha256=config_sha, arguments=vars(args),
                 fresh_gate_sha256=sha(args.fresh_gate), frozen_model_sha256=FROZEN_SHA,
                 frozen_protocol=dict(episodes=250, epochs=50, loss='ranking', validation_seed=18000,
                                      validation_episodes=20, confirmation_seed=19000, confirmation_episodes=50,
                                      fresh_seed=90000, fresh_episodes=100), runs={})
    handles = []
    children = []
    def save():
        state['elapsed_seconds'] = time.monotonic() - started
        atomic_json(output / 'status.json', state)
    def launch(label, command):
        out = (output / (label + '.stdout.log')).open('x')
        err = (output / (label + '.stderr.log')).open('x')
        handles.extend([out, err])
        child = subprocess.Popen(command, cwd=root, stdout=out, stderr=err)
        children.append(child)
        state.setdefault('commands', []).append(dict(label=label, argv=command, pid=child.pid,
                                                     started=datetime.now(timezone.utc).isoformat()))
        save()
        return child
    def wait_all(pairs):
        for label, child in pairs:
            code = child.wait()
            state['commands'][next(i for i,r in enumerate(state['commands']) if r['label']==label)]['exit_code'] = code
            save()
            if code:
                raise RuntimeError(f'{label} failed with exit code {code}')
    save()
    try:
        for seed in (43, 44):
            run_id = __import__('uuid').uuid4().hex[:8]
            data_seed = (seed + 20) * 10000
            dataset = output / f'data_seed{seed}.npz'
            state['runs'][str(seed)] = dict(run_id=run_id, training_seed=seed, data_seed=data_seed,
                                            training_from_scratch=True, data=str(dataset))
            save()
            command = [sys.executable, str(root/'scripts/collect_scorer_data.py'), '--config', args.config,
                       '--output', str(dataset), '--episodes', '250', '--seed', str(data_seed),
                       '--workers', str(args.workers)]
            wait_all([(f'data{seed}', launch(f'data{seed}', command))])
            metadata = json.loads(dataset.with_suffix('.json').read_text())
            assert metadata['file_sha256'] == sha(dataset)
            assert metadata['episodes'] == 250 and metadata['seed_start'] == data_seed
            state['runs'][str(seed)]['data_evidence'] = metadata
            save()
        state['status'] = 'training'
        save()
        pairs=[]
        for seed in (43,44):
            run = state['runs'][str(seed)]
            folder = output / f'ranking_seed{seed}'
            run['directory'] = str(folder)
            command = [sys.executable, str(root/'scripts/train_placement_scorer.py'), '--data', run['data'],
                       '--config', args.config, '--output', str(folder), '--run-id', run['run_id'],
                       '--loss', 'ranking', '--seed', str(seed), '--epochs', '50', '--device', args.device,
                       '--validation-seed', '18000', '--validation-episodes', '20',
                       '--confirmation-seed', '19000', '--confirmation-episodes', '50']
            pairs.append((f'train{seed}', launch(f'train{seed}', command)))
        wait_all(pairs)
        state['status'] = 'fresh_evaluation'
        save()
        pairs=[]
        for seed in (43,44):
            run = state['runs'][str(seed)]
            folder = Path(run['directory'])
            result = json.loads((folder/'status.json').read_text())
            model = folder/'selected_model.pt'
            assert result['status'] == 'complete' and result['training_from_scratch']
            assert result['arguments']['seed'] == seed and result['arguments']['loss'] == 'ranking'
            assert len(result['history']) == 50 and result['model_sha256'] == sha(model)
            assert result['data_sha256'] == sha(run['data'])
            run['confirmation'] = result['comparison']
            run['model_sha256'] = result['model_sha256']
            fresh = output / f'fresh90000_seed{seed}'
            command = [sys.executable, str(root/'scripts/evaluate_scorer_cohort.py'), '--root', str(root),
                       '--config', args.config, '--model', str(model), '--expected-sha256', sha(model),
                       '--run-id', run['run_id']+'_fresh', '--seed', '90000', '--episodes', '100',
                       '--output', str(fresh)]
            pairs.append((f'fresh{seed}', launch(f'fresh{seed}', command)))
        wait_all(pairs)
        for seed in (43,44):
            result = json.loads((output/f'fresh90000_seed{seed}'/'status.json').read_text())
            assert result['status'] == 'complete'
            state['runs'][str(seed)]['fresh'] = result['comparison']
        state.update(status='complete', finished=datetime.now(timezone.utc).isoformat(),
                     all_replicates_pass=all(r['confirmation']['passes_this_cohort'] and r['fresh']['passes_this_cohort']
                                             for r in state['runs'].values()))
        save()
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        (output/'controller.error.log').write_text(traceback.format_exc())
        save()
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
