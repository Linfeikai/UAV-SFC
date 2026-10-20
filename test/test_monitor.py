"""Regression tests for terminal evidence, PID identity and quiet event deduplication."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import experiment_monitor as monitor
from scripts.check_experiment import fetch


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pipeline = self.root / 'pilot'
        self.pipeline.mkdir()
        self.config = dict(pipeline=str(self.pipeline), monitor_dir=str(self.root / 'monitor'),
                           expected_algorithms=['DIFFUSION', 'GAUSSIAN'],
                           controller=dict(pid=123, boot_id='boot', start_ticks=10))
        self.state = dict(status='training_gaussian', source_commit='fixed',
                          arguments=dict(steps=100, eval_episodes=20), runs=[])
        self.alive = dict(pid=123, boot_id='boot', start_ticks=10, alive=True)

    def add_run(self, algorithm):
        key = algorithm.lower()
        folder = self.pipeline / key
        folder.mkdir()
        with zipfile.ZipFile(folder / (algorithm + '_final_model.zip'), 'w') as archive:
            archive.writestr('data', '{}')
            archive.writestr('policy.pth', b'fixture')
        run = dict(algorithm=algorithm, directory=str(folder), run_id=key, url='https://example.test/' + key,
                   status='complete', timesteps=100, updates=5, finite_scalar_count=40, evaluations={})
        for policy in (key, 'heuristic', 'obs_heuristic'):
            value = dict(episodes=20, success_rate_pct=45.0, crash_episode_rate_pct=5.0)
            monitor.atomic_json(self.pipeline / (key + '_eval_' + policy + '.json'), value)
            run['evaluations'][policy] = value
        self.state['runs'].append(run)
        self.save()
        return run

    def save(self):
        monitor.atomic_json(self.pipeline / 'status.json', self.state)

    def test_pid_reuse_boot_and_zombie_not_live(self):
        self.assertTrue(monitor.same_process(self.config['controller'], self.alive))
        for change in (dict(start_ticks=11), dict(boot_id='another'), dict(alive=False)):
            self.assertFalse(monitor.same_process(self.config['controller'], dict(self.alive, **change)))
        expected = dict(self.config['controller'], required_arg='/correct/output')
        self.assertFalse(monitor.same_process(expected, dict(self.alive, command=['unrelated'])))

    def test_proc_parser_handles_parentheses_in_process_name(self):
        proc = self.root / 'proc'
        (proc / '123').mkdir(parents=True)
        (proc / 'sys/kernel/random').mkdir(parents=True)
        (proc / 'sys/kernel/random/boot_id').write_text('boot\n')
        tail = ['S'] + ['0'] * 18 + ['10']
        (proc / '123/stat').write_text('123 (job with ) bracket) ' + ' '.join(tail))
        (proc / '123/cmdline').write_bytes(b'python\0runner.py\0')
        actual = monitor.process_identity(123, proc)
        self.assertTrue(monitor.same_process(self.config['controller'], actual))
        (proc / '123/stat').write_text('123 (job) ' + ' '.join(['Z'] + tail[1:]))
        self.assertFalse(monitor.process_identity(123, proc)['alive'])

    def test_completed_run_is_emitted_once_while_queue_runs(self):
        self.add_run('DIFFUSION')
        with patch.object(monitor, 'process_identity', return_value=self.alive):
            self.assertIsNone(monitor.reconcile(self.config))
            self.assertIsNone(monitor.reconcile(self.config))
        events = monitor.read_events(self.config['monitor_dir'])
        self.assertEqual([e['kind'] for e in events], ['run_complete'])
        self.assertIn('sha256', events[0]['evidence']['model'])

    def test_partial_event_tail_does_not_swallow_next_event(self):
        directory = Path(self.config['monitor_dir'])
        directory.mkdir()
        (directory / 'events.jsonl').write_text('{"partial":')
        events = monitor.read_events(directory)
        monitor.append_event(directory, events, 'test', 'run_complete', {'algorithm': 'DIFFUSION'})
        saved = monitor.read_events(directory)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['key'], 'test')

    def test_complete_requires_all_planned_algorithms_and_controller_exit(self):
        self.add_run('DIFFUSION')
        self.state['status'] = 'complete'
        self.save()
        with patch.object(monitor, 'process_identity', return_value=self.alive):
            self.assertIsNone(monitor.reconcile(self.config))
        with patch.object(monitor, 'process_identity', return_value=dict(pid=123, alive=False)):
            self.assertEqual(monitor.reconcile(self.config), 'controller_lost')
        self.assertNotIn('pipeline_complete', [e['kind'] for e in monitor.read_events(self.config['monitor_dir'])])

    def test_complete_verified_after_exit(self):
        self.add_run('DIFFUSION')
        self.add_run('GAUSSIAN')
        self.state['status'] = 'complete'
        self.save()
        with patch.object(monitor, 'process_identity', return_value=dict(pid=123, alive=False)):
            self.assertEqual(monitor.reconcile(self.config), 'pipeline_complete')
            monitor.reconcile(self.config)
        self.assertEqual([e['kind'] for e in monitor.read_events(self.config['monitor_dir'])],
                         ['run_complete', 'run_complete', 'pipeline_complete'])

    def test_mismatched_or_incomplete_evaluation_prevents_completion(self):
        run = self.add_run('DIFFUSION')
        path = self.pipeline / 'diffusion_eval_diffusion.json'
        value = dict(run['evaluations']['diffusion'], episodes=19)
        monitor.atomic_json(path, value)
        with self.assertRaises(ValueError):
            monitor.validate_run(run, self.state, self.pipeline)
        run['evaluations']['diffusion'] = value
        with self.assertRaisesRegex(ValueError, 'episodes'):
            monitor.validate_run(run, self.state, self.pipeline)

    def test_nan_or_bad_model_or_escaped_path_rejected(self):
        run = self.add_run('DIFFUSION')
        path = self.pipeline / 'diffusion_eval_diffusion.json'
        run['evaluations']['diffusion']['success_rate_pct'] = float('nan')
        monitor.atomic_json(path, run['evaluations']['diffusion'])
        with self.assertRaises(ValueError):
            monitor.validate_run(run, self.state, self.pipeline)
        with self.assertRaises(ValueError):
            monitor.inside(self.root / 'outside', self.pipeline)
        model = self.pipeline / 'diffusion/DIFFUSION_final_model.zip'
        model.write_bytes(b'truncated')
        with self.assertRaises(zipfile.BadZipFile):
            monitor.validate_run(run, self.state, self.pipeline)

    def test_failure_and_lost_controller_are_not_success(self):
        self.save()
        with patch.object(monitor, 'process_identity', return_value=dict(pid=123, alive=False)):
            self.assertEqual(monitor.reconcile(self.config), 'controller_lost')
        self.state.update(status='failed', error='test failure')
        self.save()
        with patch.object(monitor, 'process_identity', return_value=self.alive):
            self.assertEqual(monitor.reconcile(self.config), 'pipeline_failed')

    def test_client_transport_error_is_observation_failure_not_job_failure(self):
        config = dict(identity_file='key', port=123, user='user', host='host', pipeline='pilot')
        calls = []
        def fake(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 255, '', 'timeout')
        with self.assertRaisesRegex(RuntimeError, 'not evidence of job failure'):
            fetch(config, run=fake)
        self.assertIn('BatchMode=yes', calls[0])
        self.assertIn('StrictHostKeyChecking=yes', calls[0])

    @unittest.skipUnless(sys.platform == 'linux', 'Linux inotify / pidfd integration')
    def test_watcher_wakes_on_process_exit_without_polling(self):
        self.add_run('DIFFUSION')
        self.add_run('GAUSSIAN')
        self.state['status'] = 'complete'
        self.save()
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'])
        self.addCleanup(lambda: process.poll() is None and process.kill())
        self.config['controller'] = monitor.process_identity(process.pid)
        path = self.root / 'config.json'
        monitor.atomic_json(path, self.config)
        watcher = subprocess.Popen([sys.executable, str(Path(monitor.__file__).resolve()),
                                    'watch', '--config', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.addCleanup(lambda: watcher.poll() is None and watcher.kill())
        _, err = watcher.communicate(timeout=10)
        self.assertEqual(watcher.returncode, 0, err.decode())
        self.assertEqual(monitor.read_events(self.config['monitor_dir'])[-1]['kind'], 'pipeline_complete')
        process.wait(timeout=1)


if __name__ == '__main__':
    unittest.main()