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
    assert all(np.isfinite(value) for m in metrics.values() for value in m.values())
    assert len({m['episodes'] for m in metrics.values()}) == 1
    comparator = max(NAMES[1:], key=lambda n:metrics[n]['success_rate_pct'])
    learned, baseline = metrics['learned'], metrics[comparator]
    checks = dict(completion_margin=learned['success_rate_pct']-baseline['success_rate_pct'] >= .5,
                  crash_rate=learned['crash_episode_rate_pct'] <= 5,
                  reward=learned['mean_episode_reward'] >= baseline['mean_episode_reward'])
    return dict(comparator=comparator, checks=checks, passes=all(checks.values()),
                completion_margin_percentage_points=learned['success_rate_pct']-baseline['success_rate_pct'],
                reward_margin=learned['mean_episode_reward']-baseline['mean_episode_reward'], metrics=metrics)


def audit_one(seed, folder, data_path, fresh_path, expected_data_seed, episodes=250, confirmation_seed=19000, fresh_seed=90000):
    state = json.loads((folder/'status.json').read_text())
    assert state['status'] == 'complete' and state['training_from_scratch'] and not state['diagnostic_only']
    args = state['arguments']
    assert args['seed'] == seed and args['epochs'] == 50 and args['loss'] == 'ranking'
    assert args['validation_seed'] == 18000 and args['validation_episodes'] == 20
    assert args['confirmation_seed'] == confirmation_seed and args['confirmation_episodes'] == 50
    assert state['selected_epoch'] == min(state['history'],key=lambda h:(h['holdout_reward_regret'],h['holdout_mse']))['epoch']
    best_margin = max(state['validation'],key=lambda v:(v['metrics']['crash_episode_rate_pct'] <= 5,v['metrics']['success_rate_pct'],v['metrics']['mean_episode_reward']))['margin']
    assert state['selected_margin'] == best_margin
    assert len(state['history']) == 50 and state['full_optimizer_updates'] == ((episodes - max(1, episodes//5))*40 + 127)//128*50
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
    assert dm['episodes'] == episodes and dm['seed_start'] == expected_data_seed
    assert dm['actual_environment_interactions'] == episodes*40 and dm['counterfactual_simulator_steps'] == episodes*40*8
    with np.load(data_path) as data:
        assert data['features'].shape == (episodes*40,8,450) and data['rewards'].shape == (episodes*40,8)
        assert np.isfinite(data['features']).all() and np.isfinite(data['rewards']).all()
        ids, counts = np.unique(data['episode_ids'],return_counts=True)
        assert np.array_equal(ids,np.arange(expected_data_seed,expected_data_seed+episodes))
        assert (counts == 40).all()
    confirmation = {name:json.loads((folder/(name+'.json')).read_text()) for name in NAMES}
    assert confirmation == state['confirmation']
    assert all(m['seed_start'] == confirmation_seed for m in confirmation.values())
    fresh = json.loads((fresh_path/'status.json').read_text())
    assert fresh['status'] == 'complete' and fresh['model_sha256'] == sha(model)
    assert fresh['config_sha256'] in (REFERENCE_SHA,EQUIVALENT_REFERENCE_SHA)
    assert fresh['seed_start'] == fresh_seed and fresh['episodes'] == 100
    fm = {name:json.loads((fresh_path/(name+'.json')).read_text()) for name in NAMES}
    assert fm == fresh['evaluations'] and all(m['seed_start'] == fresh_seed for m in fm.values())
    c,f = compare(confirmation,50),compare(fm,100)
    evaluation_metrics = [v['metrics'] for v in state['validation']] + list(confirmation.values()) + list(fm.values())
    evaluation_cost = dict(episodes=sum(m['episodes'] for m in evaluation_metrics),
                           actual_environment_steps=sum(round(m['episodes']*m['mean_episode_steps']) for m in evaluation_metrics))
    assert c['passes'] == state['comparison']['passes_this_cohort']
    assert f['passes'] == fresh['comparison']['passes_this_cohort']
    return dict(training_seed=seed, run_id=state['run_id'], model_sha256=sha(model),data_sha256=data_sha,
                model=str(model.resolve()), training_evidence_sha256=sha(folder/'status.json'),
                fresh_evidence_sha256=sha(fresh_path/'status.json'), selected_epoch=state['selected_epoch'],
                selected_margin=state['selected_margin'], training_runtime=state.get('runtime',{'training_device':'cuda'}),
                data_cost=dm, evaluation_cost=evaluation_cost, training_elapsed_seconds=state['elapsed_seconds'],
                confirmation=c, fresh=f, passes=c['passes'] and f['passes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence42',required=True)
    parser.add_argument('--fresh42',required=True)
    parser.add_argument('--replicates',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--data-episodes',type=int,default=250)
    parser.add_argument('--data-seeds',type=int,nargs=3,default=[620000,630000,640000])
    parser.add_argument('--confirmation-seed',type=int,default=19000)
    parser.add_argument('--fresh-seed',type=int,default=90000)
    parser.add_argument('--seed42-folder',default='scorer_ranking')
    args=parser.parse_args()
    base=Path(args.evidence42)
    replicas=Path(args.replicates)
    control=json.loads((replicas/'status.json').read_text())
    assert control['status'] == 'complete'
    assert len(set(args.data_seeds)) == 3
    for i,a in enumerate(args.data_seeds):
        assert all(abs(a-b) >= args.data_episodes for b in args.data_seeds[i+1:])
    rows=[audit_one(42,base/args.seed42_folder,base/'data_seed42.npz',Path(args.fresh42),args.data_seeds[0],
                    args.data_episodes,args.confirmation_seed,args.fresh_seed)]
    for seed in (43,44):
        rows.append(audit_one(seed,replicas/f'ranking_seed{seed}',replicas/f'data_seed{seed}.npz',
                              replicas/f'fresh{args.fresh_seed}_seed{seed}',args.data_seeds[seed-42],
                              args.data_episodes,args.confirmation_seed,args.fresh_seed))
    assert len({r['data_sha256'] for r in rows}) == 3 and len({r['model_sha256'] for r in rows}) == 3
    comparator_drifts=[]
    for phase in ('confirmation','fresh'):
        for comparator in NAMES[1:]:
            reference = rows[0][phase]['metrics'][comparator]
            for row in rows[1:]:
                actual = row[phase]['metrics'][comparator]
                for key,value in reference.items():
                    if key != 'elapsed_seconds':
                        discrete_rate = key.endswith('_pct') or key in ('episodes','seed_start','mean_episode_steps')
                        tolerance = 1e-10 if discrete_rate else 1e-6
                        assert np.isclose(value,actual[key],rtol=tolerance,atol=tolerance), (phase,comparator,key)
                        if value != actual[key]:
                            comparator_drifts.append(dict(training_seed=row['training_seed'],phase=phase,
                                                          comparator=comparator,metric=key,
                                                          absolute_difference=abs(value-actual[key])))
    source42=json.loads((base/'status.json').read_text()).get('source_commit')
    result=dict(status='complete',audited_at=datetime.now(timezone.utc).isoformat(),
                original_fixed_environment=True,original_reference_sha256=REFERENCE_SHA,
                environment_manifest_sha256=sha(Path(__file__).resolve().parents[1]/'experiments/server_20261021/fixed_hybrid_environment_manifest.json'),
                learned_component='whole-deployment selection from observation only',
                mobility_component='shared analytical velocity-braking controller; no learned mobility claim',
                source_versions=dict(seed42=source42,replicas=control['source_commit']),
                comparator_metrics_equivalent_across_training_seeds=True, comparator_numerical_drifts=comparator_drifts,
                independent_training_seeds=[42,43,44],all_requirements_pass=all(r['passes'] for r in rows),runs=rows,
                minimum_completion_margin_pp=min(min(r[k]['completion_margin_percentage_points'] for k in ('confirmation','fresh')) for r in rows),
                dataset_cost_total=dict(real_environment_interactions=args.data_episodes*40*3,counterfactual_simulator_steps=args.data_episodes*40*8*3),
                evaluation_cost_total=dict(episodes=sum(r['evaluation_cost']['episodes'] for r in rows),
                                           actual_environment_steps=sum(r['evaluation_cost']['actual_environment_steps'] for r in rows)),
                runtime_caveat='Inspect per-run training_runtime; all fresh cohorts evaluated locally; no claim of cross-runtime bitwise equivalence',
                experiment_tracking='Actual local run IDs; W&B publication remains separate and is not claimed here')
    assert not Path(args.output).exists()
    atomic_json(Path(args.output),result)
    print(json.dumps({k:v for k,v in result.items() if k!='runs'}),flush=True)


if __name__ == '__main__':
    main()
