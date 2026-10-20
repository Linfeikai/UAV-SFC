"""Run a versioned motion-learning batch, adopting existing jobs without restart.

The plan contains paths, PIDs and Hydra overrides, never account credentials.
At most two training jobs share the existing GPU. Validation selects checkpoints;
separate confirmation seeds compare the selected models with both heuristics.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_monitor import atomic_json, process_identity, same_process
from scripts.training_validation import validation_score


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    import torch
    from main import ALGO_REGISTRY
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    assert torch.cuda.is_available(), 'Refusing CPU fallback'
    torch.set_num_threads(1)
    state = dict(status='running', started=stamp(), source_commit=plan['source_commit'],
                 arguments=dict(steps=plan['steps'], eval_episodes=plan['eval_episodes']),
                 runs=[], active_pid=None, plan=plan)
    workers = {}

    def save():
        atomic_json(out / 'status.json', state)

    def execute(command, root, log, env=None):
        with Path(log).open('w') as handle:
            proc = subprocess.Popen(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True)
            state.update(active_pid=proc.pid, active_command=command)
            save()
            try:
                code = proc.wait(timeout=7200)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
                raise
        if code:
            raise RuntimeError(f'Exit {code}: {log}')
        state['active_pid'] = None
        save()

    def evaluate(job, model_path, policy, filename, seed, episodes):
        command = [sys.executable, '-m', 'test.cpu_model_eval', policy]
        if model_path:
            command.append(str(model_path))
        command += ['--config', job['folder'], '--device', 'cuda', '--seed', str(seed),
                    '--episodes', str(episodes)]
        # Separate stderr: the evaluator's stdout is a machine-readable JSON.
        result = subprocess.run(command, cwd=job['root'], capture_output=True, text=True,
                                timeout=900, env=dict(os.environ, OMP_NUM_THREADS='1'))
        (out / (filename + '.stderr.log')).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr[-3000:])
        metrics = json.loads(result.stdout)
        atomic_json(out / filename, metrics)
        return metrics

    for job in plan['runs']:
        record = dict(algorithm=job['variant'], run_id=job['run_id'], url=job['url'],
                      directory=str(out / job['variant'].lower()), status='queued',
                      source_commit=job['source_commit'], started=stamp())
        state['runs'].append(record)
        log = out / (job['variant'].lower() + '.log')
        if job.get('identity'):
            record['status'] = 'training'
            workers[job['variant']] = dict(identity=job['identity'], job=job, record=record)
            if not log.exists():
                log.symlink_to(job['log'])
    save()
    try:
        while any(r['status'] not in ('complete', 'failed') for r in state['runs']):
            for job, record in zip(plan['runs'], state['runs']):
                if record['status'] != 'queued' or len(workers) >= plan.get('max_parallel', 2):
                    continue
                env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           WANDB_ENTITY='SACtest', WANDB_RUN_ID=job['run_id'],
                           WANDB_RUN_GROUP=out.name, PYTHONUNBUFFERED='1')
                log = out / (job['variant'].lower() + '.log')
                handle = log.open('w')
                command = [sys.executable, 'main.py'] + job['overrides']
                proc = subprocess.Popen(command, cwd=job['root'], env=env, stdout=handle,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
                handle.close()
                record.update(status='training', command=command)
                workers[job['variant']] = dict(identity=process_identity(proc.pid), job=job,
                                               record=record, process=proc)
                save()
            finished = []
            for variant, worker in list(workers.items()):
                identity, job, record = worker['identity'], worker['job'], worker['record']
                if same_process(identity, process_identity(identity['pid'])):
                    continue
                proc = worker.get('process')
                if proc is not None and proc.wait() != 0:
                    raise RuntimeError('Training failed: ' + variant)
                folder = Path(job['folder'])
                final = folder / 'DIFFUSION_final_model.zip'
                if not final.exists():
                    raise RuntimeError('Process exited without a final model: ' + variant)
                record['status'] = 'validating'
                save()
                # Always include the final model, even when training's small
                # validation screen preferred an earlier checkpoint.
                candidates = [final]
                best = folder / 'best_validation_model.zip'
                if best.exists():
                    candidates.append(best)
                options = []
                for index, candidate in enumerate(candidates):
                    metrics = evaluate(job, candidate, 'diffusion',
                                       variant.lower() + f'_selection_{index}.json',
                                       plan['validation_seed'], plan['validation_episodes'])
                    options.append((validation_score(metrics), candidate, metrics))
                _, selected, selection = max(options, key=lambda item: item[0])
                target = Path(record['directory'])
                (target / '.hydra').mkdir(parents=True)
                model_file = target / (variant + '_final_model.zip')
                shutil.copy2(selected, model_file)
                shutil.copy2(folder / '.hydra/config.yaml', target / '.hydra/config.yaml')
                model = ALGO_REGISTRY['DIFFUSION'].load(final, device='cuda')
                assert all(torch.isfinite(p).all() for p in model.policy.parameters())
                record.update(timesteps=int(model.num_timesteps), updates=int(model._n_updates),
                              selected_checkpoint=str(selected), selection=selection,
                              model_sha256=hashlib.sha256(model_file.read_bytes()).hexdigest())
                del model
                count = 0
                for event in (folder / 'tb_logs').rglob('events.out.tfevents.*'):
                    accumulator = EventAccumulator(str(event), size_guidance={'scalars': 0})
                    accumulator.Reload()
                    for tag in accumulator.Tags()['scalars']:
                        values = accumulator.Scalars(tag)
                        assert all(torch.isfinite(torch.tensor(v.value)) for v in values), tag
                        count += len(values)
                record['finite_scalar_count'] = count
                record['evaluations'] = {}
                for policy in ('diffusion', 'heuristic', 'obs_heuristic'):
                    key = variant.lower() if policy == 'diffusion' else policy
                    record['evaluations'][key] = evaluate(
                        job, model_file if policy == 'diffusion' else None, policy,
                        variant.lower() + '_eval_' + key + '.json',
                        plan['confirmation_seed'], plan['eval_episodes'])
                try:
                    import wandb
                    online = wandb.Api(timeout=30).run('SACtest/diffusion_rl/' + job['run_id'])
                    online.summary.update({'motion_search/' + policy + '/' + k: v
                        for policy, values in record['evaluations'].items()
                        for k, v in values.items() if isinstance(v, (float, int))})
                except Exception as error:
                    record['wandb_summary_error'] = repr(error)
                record.update(status='complete', finished=stamp())
                finished.append(variant)
                save()
            for variant in finished:
                workers.pop(variant)
            if workers:
                # Wait on child exits with a bounded health-check interval.
                fds = []
                try:
                    for worker in workers.values():
                        if hasattr(os, 'pidfd_open'):
                            try:
                                fds.append(os.pidfd_open(worker['identity']['pid']))
                            except ProcessLookupError:
                                pass
                    if fds:
                        select.select(fds, [], [], 30)
                    else:
                        time.sleep(30)
                finally:
                    for fd in fds:
                        os.close(fd)
        state.update(status='complete', finished=stamp(), active_pid=None)
        save()
    except BaseException as error:
        state.update(status='failed', error=repr(error), finished=stamp())
        save()
        raise


if __name__ == '__main__':
    main()
