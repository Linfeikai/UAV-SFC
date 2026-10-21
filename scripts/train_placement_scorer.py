"""Train a deployment scorer from scratch; select using held-out data and validation episodes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from algos.placement_scorer import DeploymentScorer, ScoredDeploymentPolicy
from scripts.experiment_monitor import atomic_json
from test.cpu_model_eval import evaluate, load_env_config
from scripts.collect_scorer_data import REFERENCE_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--loss', choices=['regression', 'ranking'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--validation-seed', type=int, default=18000)
    parser.add_argument('--validation-episodes', type=int, default=20)
    parser.add_argument('--confirmation-seed', type=int, default=19000)
    parser.add_argument('--confirmation-episodes', type=int, default=50)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'No CPU training fallback'
    assert hashlib.sha256(Path(args.config).read_bytes()).hexdigest() == REFERENCE_SHA
    cfg = load_env_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = dict(status='training', started=datetime.now(timezone.utc).isoformat(),
                 pid=os.getpid(), run_id=args.run_id, arguments=vars(args),
                 training_from_scratch=True, diagnostic_only=args.confirmation_episodes < 50,
                 learned_component='whole-deployment scorer',
                 motion_component='explicit shared velocity-braking controller', history=[])
    def save():
        state['elapsed_seconds'] = time.monotonic() - started
        atomic_json(output / 'status.json', state)
    save()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    data = np.load(args.data)
    features = data['features']
    rewards = data['rewards']
    ids = data['episode_ids']
    episodes = np.unique(ids)
    rng.shuffle(episodes)
    holdout = np.isin(ids, episodes[:max(1, len(episodes) // 5)])
    train_indices = np.flatnonzero(~holdout)
    valid_indices = np.flatnonzero(holdout)
    assert len(train_indices) and len(valid_indices)
    mean = features[train_indices].mean((0, 1))
    scale = np.maximum(features[train_indices].std((0, 1)), .05)
    x = torch.as_tensor((features - mean) / scale, device='cuda')
    y = torch.as_tensor((rewards - rewards[:, :1]) / 3., device='cuda')
    valid_x, valid_y = x[valid_indices], y[valid_indices]
    model = DeploymentScorer(features.shape[-1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.00001)
    best = None
    updates = 0
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_indices)
        losses = []
        for offset in range(0, len(order), 128):
            batch = order[offset:offset + 128]
            predicted = model(x[batch])
            loss = F.smooth_l1_loss(predicted, y[batch], beta=.2)
            if args.loss == 'ranking':
                loss = .25 * loss + F.kl_div(F.log_softmax(predicted / .2, dim=-1),
                                             F.softmax(y[batch] / .2, dim=-1), reduction='batchmean')
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            predicted = model(valid_x)
            choices = predicted.argmax(-1)
            realized = valid_y.gather(1, choices[:, None]).squeeze(1)
            regret = float((valid_y.max(-1).values - realized).mean() * 3.)
            mse = float(F.mse_loss(predicted, valid_y))
        record = dict(epoch=epoch + 1, optimizer_updates=updates,
                      mean_training_loss=float(np.mean(losses)), holdout_reward_regret=regret,
                      holdout_mse=mse)
        state['history'].append(record)
        if best is None or (regret, mse) < best:
            best = (regret, mse)
            ScoredDeploymentPolicy(model, mean, scale, device='cuda').save(
                output / 'best_holdout.pt', training_seed=args.seed, epoch=epoch + 1,
                optimizer_updates=updates, run_id=args.run_id, loss=args.loss)
            state['selected_epoch'] = epoch + 1
        save()
    state['status'] = 'validating'
    save()
    policy = ScoredDeploymentPolicy.load(output / 'best_holdout.pt')
    state['validation'] = []
    selected = None
    for margin in [0., .05, .1, .2, .4]:
        policy.margin = margin
        tick = time.monotonic()
        metrics = evaluate(policy, args.validation_episodes, args.validation_seed, env_config=cfg)
        item = dict(margin=margin, seed=args.validation_seed, metrics=metrics,
                    elapsed_seconds=time.monotonic() - tick)
        state['validation'].append(item)
        key = (metrics['crash_episode_rate_pct'] <= 5, metrics['success_rate_pct'], metrics['mean_episode_reward'])
        if selected is None or key > selected[0]:
            selected = key, margin
        save()
    policy.margin = selected[1]
    policy.save(output / 'selected_model.pt', training_seed=args.seed, run_id=args.run_id,
                selected_epoch=state['selected_epoch'], loss=args.loss,
                training_from_scratch=True, training_epochs=args.epochs,
                config_sha256=REFERENCE_SHA, data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest())
    state.update(status='confirming', selected_margin=policy.margin,
                 model_sha256=hashlib.sha256((output / 'selected_model.pt').read_bytes()).hexdigest(),
                 data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                 full_optimizer_updates=updates, confirmation={})
    save()
    for name in ['learned', 'static_heuristic', 'heuristic', 'obs_heuristic']:
        tick = time.monotonic()
        metrics = evaluate(policy if name == 'learned' else None,
                           args.confirmation_episodes, args.confirmation_seed, env_config=cfg,
                           observation_heuristic=name == 'obs_heuristic',
                           mobility_source='zero' if name == 'static_heuristic' else 'default')
        state['confirmation'][name] = dict(metrics, seed_start=args.confirmation_seed,
                                          elapsed_seconds=time.monotonic() - tick)
        atomic_json(output / (name + '.json'), state['confirmation'][name])
        save()
    learned = state['confirmation']['learned']
    baseline_name = max(['static_heuristic', 'heuristic', 'obs_heuristic'],
                        key=lambda name: state['confirmation'][name]['success_rate_pct'])
    baseline = state['confirmation'][baseline_name]
    checks = dict(confirmation_size=args.confirmation_episodes >= 50, completion_margin=learned['success_rate_pct'] - baseline['success_rate_pct'] >= .5,
                  crash_rate=learned['crash_episode_rate_pct'] <= 5,
                  reward=learned['mean_episode_reward'] >= baseline['mean_episode_reward'])
    state.update(status='complete', finished=datetime.now(timezone.utc).isoformat(),
                 comparison=dict(comparator=baseline_name, checks=checks, passes_this_cohort=all(checks.values()),
                                 completion_margin_percentage_points=learned['success_rate_pct']-baseline['success_rate_pct'],
                                 reward_margin=learned['mean_episode_reward']-baseline['mean_episode_reward']),
                 final_fresh_seed_evaluation_and_scratch_replicates_still_required=True)
    save()
    print(json.dumps({k:state[k] for k in ['run_id','selected_epoch','selected_margin','comparison','elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
