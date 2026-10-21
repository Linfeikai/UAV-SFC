"""Recompute the full qualification audit from saved training and evaluation evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.collect_scorer_data import REFERENCE_SHA, EQUIVALENT_REFERENCE_SHA
from scripts.experiment_monitor import atomic_json

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
NAMES = ('learned', 'static_heuristic', 'heuristic', 'obs_heuristic')


def compare(metrics, minimum):
    assert set(metrics) == set(NAMES)
    assert all(m['episodes'] >= minimum for m in metrics.values())
    assert len({m['episodes'] for m in metrics.values()}) == 1
    comparator = max(NAMES[1:], key=lambda n:metrics[n]['success_rate_pct'])
    learned, baseline = metrics['learned'], metrics[comparator]
    checks = dict(completion_margin=learned['success_rate_pct']-baseline['success_rate_pct'] >= .5,
                  crash_rate=learned['crash_episode_rate_pct'] <= 5,
                  reward=learned['mean_episode_reward'] >= baseline['mean_episode_reward'])
    return dict(comparator=comparator, checks=checks, passes=all(checks.values()),
                completion_margin_percentage_points=learned['success_rate_pct']-baseline['success_rate_pct'],
                reward_margin=learned['mean_episode_reward']-baseline['mean_episode_reward'], metrics=metrics)


def audit_one(seed, folder, data_path, fresh_path, expected_data_seed):
    state = json.loads((folder/'status.json').read_text())
    assert state['status'] == 'complete' and state['training_from_scratch'] and not state['diagnostic_only']
    args = state['arguments']
    assert args['seed'] == seed and args['epochs'] == 50 and args['loss'] == 'ranking'
    assert args['validation_seed'] == 18000 and args['validation_episodes'] == 20
    assert args['confirmation_seed'] == 19000 and args['confirmation_episodes'] == 50
    assert len(state['history']) == 50 and state['full_optimizer_updates'] == 3150
    assert 'resume_model' not in args
    model = folder/'selected_model.pt'
    assert sha(model) == state['model_sha256']
    archive = torch.load(model, map_location='cpu', weights_only=True)
    meta = archive['metadata']
    assert meta['training_seed'] == seed and meta['run_id'] == state['run_id'] and meta['training_from_scratch']
    assert meta['selected_epoch'] == state['selected_epoch'] and archive['margin'] == state['selected_margin']
    assert meta['training_epochs'] == 50 and meta['loss'] == 'ranking'
    assert meta['config_sha256'] in (REFERENCE_SHA,EQUIVALENT_REFERENCE_SHA)
    assert all(torch.isfinite(v).all() for v in archive['state_dict'].values())
    assert torch.isfinite(archive['mean']).all() and torch.isfinite(archive['scale']).all()
    assert torch.all(archive['scale'] >= .05)
    data_sha = sha(data_path)
    assert data_sha == state['data_sha256'] == meta['data_sha256']
    dm = json.loads(data_path.with_suffix('.json').read_text())
    assert dm['status'] == 'complete' and dm['file_sha256'] == data_sha
    assert dm['config_sha256'] in (REFERENCE_SHA,EQUIVALENT_REFERENCE_SHA)
    assert dm['episodes'] == 250 and dm['seed_start'] == expected_data_seed
    assert dm['actual_environment_interactions'] == 10000 and dm['counterfactual_simulator_steps'] == 80000
    with np.load(data_path) as data:
        assert data['features'].shape == (10000,8,450) and data['rewards'].shape == (10000,8)
        assert np.isfinite(data['features']).all() and np.isfinite(data['rewards']).all()
        ids, counts = np.unique(data['episode_ids'],return_counts=True)
        assert np.array_equal(ids,np.arange(expected_data_seed,expected_data_seed+250))
        assert (counts == 40).all()
    confirmation = {name:json.loads((folder/(name+'.json')).read_text()) for name in NAMES}
    assert confirmation == state['confirmation']
    assert all(m['seed_start'] == 19000 for m in confirmation.values())
    fresh = json.loads((fresh_path/'status.json').read_text())
    assert fresh['status'] == 'complete' and fresh['model_sha256'] == sha(model)
    assert fresh['config_sha256'] in (REFERENCE_SHA,EQUIVALENT_REFERENCE_SHA)
    assert fresh['seed_start'] == 90000 and fresh['episodes'] == 100
    fm = {name:json.loads((fresh_path/(name+'.json')).read_text()) for name in NAMES}
    assert fm == fresh['evaluations'] and all(m['seed_start'] == 90000 for m in fm.values())
    c,f = compare(confirmation,50),compare(fm,100)
    assert c['passes'] == state['comparison']['passes_this_cohort']
    assert f['passes'] == fresh['comparison']['passes_this_cohort']
    return dict(training_seed=seed, run_id=state['run_id'], model_sha256=sha(model),data_sha256=data_sha,
                model=str(model.resolve()), training_evidence_sha256=sha(folder/'status.json'),
                fresh_evidence_sha256=sha(fresh_path/'status.json'), selected_epoch=state['selected_epoch'],
                selected_margin=state['selected_margin'], training_runtime=state.get('runtime',{'training_device':'cuda'}),
                data_cost=dm, training_elapsed_seconds=state['elapsed_seconds'],
                confirmation=c, fresh=f, passes=c['passes'] and f['passes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence42',required=True)
    parser.add_argument('--fresh42',required=True)
    parser.add_argument('--replicates',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    base=Path(args.evidence42)
    replicas=Path(args.replicates)
    control=json.loads((replicas/'status.json').read_text())
    assert control['status'] == 'complete'
    rows=[audit_one(42,base/'scorer_ranking',base/'data_seed42.npz',Path(args.fresh42),620000)]
    for seed in (43,44):
        rows.append(audit_one(seed,replicas/f'ranking_seed{seed}',replicas/f'data_seed{seed}.npz',
                              replicas/f'fresh90000_seed{seed}',(seed+20)*10000))
    assert len({r['data_sha256'] for r in rows}) == 3 and len({r['model_sha256'] for r in rows}) == 3
    result=dict(status='complete',audited_at=datetime.now(timezone.utc).isoformat(),
                original_fixed_environment=True,original_reference_sha256=REFERENCE_SHA,
                environment_manifest_sha256=sha(Path(__file__).resolve().parents[1]/'experiments/server_20261021/fixed_hybrid_environment_manifest.json'),
                learned_component='whole-deployment selection from observation only',
                mobility_component='shared analytical velocity-braking controller; no learned mobility claim',
                independent_training_seeds=[42,43,44],all_requirements_pass=all(r['passes'] for r in rows),runs=rows,
                minimum_completion_margin_pp=min(min(r[k]['completion_margin_percentage_points'] for k in ('confirmation','fresh')) for r in rows),
                dataset_cost_total=dict(real_environment_interactions=30000,counterfactual_simulator_steps=240000),
                runtime_caveat='42 trained on server CUDA; 43/44 on explicitly selected local CPU; all fresh cohorts evaluated locally',
                experiment_tracking='Actual local run IDs; W&B publication remains separate and is not claimed here')
    assert not Path(args.output).exists()
    atomic_json(Path(args.output),result)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)


if __name__ == '__main__':
    main()
