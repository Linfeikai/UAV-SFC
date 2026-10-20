"""Validation checkpoint selection without consuming training random streams."""
from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


@contextmanager
def preserve_training_state(model):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    training = model.policy.training
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        model.policy.set_training_mode(training)


def validation_score(metrics, crash_limit=5.0):
    return (metrics['crash_episode_rate_pct'] <= crash_limit,
            metrics['success_rate_pct'], metrics['mean_episode_reward'])


class SFCValidationCallback(BaseCallback):
    def __init__(self, config, directory, frequency=5000, episodes=5, seed=2000):
        super().__init__()
        self.config, self.directory = config, Path(directory)
        self.frequency, self.episodes, self.seed = int(frequency), int(episodes), int(seed)
        if min(self.frequency, self.episodes) <= 0:
            raise ValueError('Validation frequency and episodes must be positive')
        self.next_step, self.best_score, self.records = 0, None, []

    def _evaluate(self):
        from test.cpu_model_eval import evaluate
        with preserve_training_state(self.model):
            metrics = evaluate(self.model, self.episodes, self.seed, env_config=self.config)
        score = validation_score(metrics)
        improved = self.best_score is None or score > self.best_score
        if improved:
            self.model.save(self.directory / 'best_validation_model')
            self.best_score = score
        record = dict(timesteps=self.num_timesteps, seed=self.seed, episodes=self.episodes,
                      improved=improved, metrics=metrics)
        self.records.append(record)
        path = self.directory / 'validation_history.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.records, indent=2) + '\n', encoding='utf-8')
        temp.replace(path)
        for name, value in metrics.items():
            self.logger.record('validation/' + name, value)
        print('[Validation]', self.num_timesteps, json.dumps(metrics), flush=True)

    def _on_training_start(self):
        self.model.save(self.directory / 'after_bc_model')
        self._evaluate()
        self.next_step = self.num_timesteps + self.frequency

    def _on_step(self):
        if self.num_timesteps >= self.next_step:
            self._evaluate()
            self.next_step = self.num_timesteps + self.frequency
        return True
