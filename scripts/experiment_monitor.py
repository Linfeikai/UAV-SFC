"""Event-driven Linux experiment watcher and read-only SSH status endpoint.

A completed training process is NOT a completed experiment: all planned models,
finite-metric checks, evaluation JSONs and the controller's terminal exit must agree.
No W&B/SSH credentials are read by this module.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import select
import sys
import zipfile


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def process_identity(pid, proc_root='/proc'):
    """Use boot ID + start ticks, never a naked PID, to avoid PID reuse."""
    root = Path(proc_root)
    try:
        stat = (root / str(int(pid)) / 'stat').read_text()
        tail = stat.rsplit(')', 1)[1].split()
        command = (root / str(int(pid)) / 'cmdline').read_bytes().split(b'\0')
        return dict(pid=int(pid), boot_id=(root / 'sys/kernel/random/boot_id').read_text().strip(),
                    start_ticks=int(tail[19]), state=tail[0],
                    command=[x.decode(errors='replace') for x in command if x],
                    alive=tail[0] not in {'Z', 'X'})
    except (OSError, ValueError, IndexError):
        return dict(pid=int(pid), alive=False)


def same_process(expected, actual):
    return bool(actual.get('alive') and all(
        expected.get(key) == actual.get(key) for key in ('pid', 'boot_id', 'start_ticks')
    ) and (not expected.get('required_arg') or expected['required_arg'] in actual.get('command', [])))


def inside(path, parent):
    resolved, parent = Path(path).resolve(), Path(parent).resolve()
    if not resolved.is_relative_to(parent):
        raise ValueError('Artifact escaped experiment directory')
    return resolved


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return dict(path=str(path), bytes=Path(path).stat().st_size, sha256=h.hexdigest())


def validate_run(run, state, pipeline):
    if run.get('status') != 'complete':
        raise ValueError('Run is not marked complete')
    if run.get('model_format') == 'deployment_scorer':
        return validate_scorer_run(run, state, pipeline)
    args = state['arguments']
    for key, minimum in [('timesteps', int(args['steps'])), ('updates', 1), ('finite_scalar_count', 1)]:
        if int(run.get(key, 0)) < minimum:
            raise ValueError('Incomplete ' + key)
    folder = inside(run['directory'], pipeline)
    model = inside(folder / (run['algorithm'] + '_final_model.zip'), pipeline)
    with zipfile.ZipFile(model) as archive:
        if not {'data', 'policy.pth'}.issubset(archive.namelist()) or archive.testzip() is not None:
            raise ValueError('Model archive missing data or corrupt')
    evidence = {'model': digest(model), 'evaluations': {}}
    key = run['algorithm'].lower()
    for policy in (key, 'heuristic', 'obs_heuristic'):
        path = inside(Path(pipeline) / (key + '_eval_' + policy + '.json'), pipeline)
        result = load_json(path)
        if result != run.get('evaluations', {}).get(policy):
            raise ValueError('Evaluation file / controller summary mismatch: ' + policy)
        if result.get('episodes', 0) < int(args['eval_episodes']):
            raise ValueError('Evaluation episodes incomplete')
        for value in result.values():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError('Non-finite evaluation metric')
        for metric in ('success_rate_pct', 'crash_episode_rate_pct'):
            if not 0 <= result.get(metric, -1) <= 100:
                raise ValueError('Invalid evaluation rate: ' + metric)
        evidence['evaluations'][policy] = digest(path)
    return evidence



def validate_scorer_run(run, state, pipeline):
    """Validate the native scorer artifact without pretending it is an SB3 ZIP."""
    import torch
    folder = inside(run['directory'], pipeline)
    worker = load_json(folder / 'status.json')
    if worker.get('status') != 'complete' or not worker.get('training_from_scratch'):
        raise ValueError('Scorer training did not complete from scratch')
    if len(worker['history']) != int(state['arguments']['epochs']):
        raise ValueError('Incomplete scorer epoch budget')
    if int(worker.get('full_optimizer_updates', 0)) < 1:
        raise ValueError('Scorer has no optimizer updates')
    model = inside(folder / 'selected_model.pt', pipeline)
    model_digest = digest(model)
    if model_digest['sha256'] != worker['model_sha256'] or model_digest['sha256'] != run['model_sha256']:
        raise ValueError('Scorer model hash mismatch')
    saved = torch.load(model, map_location='cpu', weights_only=True)
    if not all(torch.isfinite(value).all() for value in saved['state_dict'].values()):
        raise ValueError('Non-finite scorer weights')
    evidence = dict(model=model_digest, evaluations={})
    for label in ('learned', 'static_heuristic', 'heuristic', 'obs_heuristic'):
        path = inside(folder / (label + '.json'), pipeline)
        result = load_json(path)
        key = run['algorithm'].lower() if label == 'learned' else label
        if result != run['evaluations'][key] or result != worker['confirmation'][label]:
            raise ValueError('Scorer evaluation mismatch: ' + label)
        if result.get('episodes', 0) < int(state['arguments']['eval_episodes']):
            raise ValueError('Insufficient scorer confirmation episodes')
        for value in result.values():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError('Non-finite scorer metric')
        if any(not 0 <= result.get(metric, -1) <= 100
               for metric in ('success_rate_pct', 'crash_episode_rate_pct')):
            raise ValueError('Invalid scorer rate')
        evidence['evaluations'][key] = digest(path)
    return evidence


def read_events(directory):
    path = Path(directory) / 'events.jsonl'
    if not path.exists():
        return []
    # A monitor crash may leave a partial final line; ignore it, never the prior events.
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def append_event(directory, events, key, kind, payload):
    if any(item['key'] == key for item in events):
        return
    item = dict(sequence=max([0] + [e['sequence'] for e in events]) + 1,
                at=now(), key=key, kind=kind, **payload)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'events.jsonl').open('a+b') as handle:
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b'\n':
                handle.write(b'\n')
        handle.write((json.dumps(item, ensure_ascii=False) + '\n').encode('utf-8'))
        handle.flush()
        os.fsync(handle.fileno())
    events.append(item)
    atomic_json(directory / 'latest_event.json', item)


def reconcile(config):
    pipeline, directory = Path(config['pipeline']), Path(config['monitor_dir'])
    state = load_json(pipeline / 'status.json')
    events = read_events(directory)
    live = same_process(config['controller'], process_identity(config['controller']['pid']))
    for run in state.get('runs', []):
        if run.get('status') != 'complete':
            continue
        key = 'run:' + run['run_id'] + ':complete'
        if any(event['key'] == key for event in events):
            continue
        try:
            proof = validate_run(run, state, pipeline)
            append_event(directory, events, key, 'run_complete',
                         dict(algorithm=run['algorithm'], run_id=run['run_id'],
                              url=run['url'], evidence=proof, metrics=run['evaluations']))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            append_event(directory, events, 'invalid:' + run['run_id'], 'validation_failed',
                         dict(algorithm=run['algorithm'], error=str(exc)))
    terminal = None
    if state.get('status') == 'failed':
        terminal = 'pipeline_failed'
    elif not live:
        completed = {e.get('algorithm') for e in events if e['kind'] == 'run_complete'}
        if state.get('status') == 'complete' and completed == set(config['expected_algorithms']):
            terminal = 'pipeline_complete'
        else:
            terminal = 'controller_lost'
    if terminal:
        append_event(directory, events, terminal, terminal,
                     dict(pipeline_status=state.get('status'), controller_alive=live,
                          error=state.get('error'), source_commit=state.get('source_commit')))
    atomic_json(directory / 'watcher_status.json', dict(
        observed_at=now(), controller_alive=live, status=state.get('status'),
        terminal=terminal, watcher=process_identity(os.getpid()),
        last_event_sequence=max([0] + [e['sequence'] for e in events])))
    return terminal


def snapshot(config):
    state = load_json(Path(config['pipeline']) / 'status.json')
    actual = process_identity(config['controller']['pid'])
    controller = dict(expected=config['controller'], actual=actual,
                      identity_matches=same_process(config['controller'], actual))
    monitor = Path(config['monitor_dir'])
    watcher = load_json(monitor / 'watcher_status.json') if (monitor / 'watcher_status.json').exists() else None
    if watcher:
        watcher['live_now'] = same_process(watcher['watcher'], process_identity(watcher['watcher']['pid']))
    active = process_identity(state['active_pid']) if state.get('active_pid') else None
    if active:
        # A reused PID must not be accepted as a live training/evaluation child.
        active['command_matches'] = active.get('command') == state.get('active_command')
    runs = []
    for run in state.get('runs', []):
        item = {k: run[k] for k in ('algorithm', 'run_id', 'status', 'url', 'timesteps',
                                   'updates', 'evaluations', 'finished', 'model_format', 'training_epochs',
                                   'selected_epoch', 'selected_margin', 'comparison') if k in run}
        if run.get('model_format') == 'deployment_scorer':
            worker_path = inside(run['directory'], config['pipeline']) / 'status.json'
            if worker_path.exists():
                item['worker_process'] = process_identity(run['identity']['pid']) if run.get('identity') else None
                progress = load_json(worker_path)
                item.update(worker_phase=progress.get('status'),
                            training_epochs_finished=len(progress.get('history', [])),
                            validation_margins_evaluated=len(progress.get('validation', [])),
                            confirmation_policies_completed=len(progress.get('confirmation', {})))
        log = Path(config['pipeline']) / (run['algorithm'].lower() + '.log')
        if log.exists():
            import re
            with log.open('rb') as handle:
                handle.seek(max(0, log.stat().st_size - 65536))
                tail = handle.read().decode(errors='replace')
            steps = re.findall(r'\|\s+total_timesteps\s+\|\s+(\d+)', tail)
            if steps:
                item['last_logged_timesteps'] = int(steps[-1])
        runs.append(item)
    return dict(schema=1, observed_at=now(), pipeline=str(config['pipeline']),
                source_commit=state.get('source_commit'), status=state.get('status'),
                controller=controller, active_process=active, watcher=watcher,
                events=read_events(monitor), runs=runs, error=state.get('error'))


def watch(config):
    """Block on atomic-status-renames and Linux pidfd exit, not a polling loop."""
    import fcntl
    directory = Path(config['monitor_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    libc = ctypes.CDLL(None, use_errno=True)
    event_fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if event_fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1 failed')
    watch_id = libc.inotify_add_watch(event_fd, os.fsencode(config['pipeline']), 0x80)  # IN_MOVED_TO
    if watch_id < 0:
        os.close(event_fd)
        raise OSError(ctypes.get_errno(), 'inotify_add_watch failed')
    pid_fd = None
    try:
        try:
            if hasattr(os, 'pidfd_open'):
                pid_fd = os.pidfd_open(config['controller']['pid'])
            else:
                # Some Conda builds omit os.pidfd_open despite a capable kernel.
                import platform
                if platform.machine().lower() not in {'x86_64', 'amd64', 'aarch64'}:
                    raise RuntimeError('pidfd syscall fallback unsupported on this architecture')
                pid_fd = libc.syscall(434, int(config['controller']['pid']), 0)
                if pid_fd < 0:
                    raise OSError(ctypes.get_errno(), 'pidfd_open syscall failed')
        except ProcessLookupError:
            pid_fd = None
        while True:
            if reconcile(config):
                return
            handles = [event_fd] + ([pid_fd] if pid_fd is not None else [])
            # Timeout only guards dropped events / watcher health; normally no wakeups.
            ready, _, _ = select.select(handles, [], [], 900)
            if event_fd in ready:
                try:
                    os.read(event_fd, 65536)
                except BlockingIOError:
                    pass
    finally:
        os.close(event_fd)
        if pid_fd is not None:
            os.close(pid_fd)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['watch', 'status', 'once'])
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    if args.mode == 'watch':
        watch(config)
    elif args.mode == 'once':
        print(reconcile(config))
    else:
        print(json.dumps(snapshot(config), ensure_ascii=False))


if __name__ == '__main__':
    main()