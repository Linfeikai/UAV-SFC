"""Whole-proposal selection, native serialization, and completion-proof coverage."""
import hashlib
import json
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from algos.placement_candidates import placement_candidates, candidate_features
from algos.placement_scorer import DeploymentScorer, ScoredDeploymentPolicy
from core.sfc_env import SFCEnv
from main import get_flat_config
from scripts.experiment_monitor import atomic_json, validate_run


class FixedScores(torch.nn.Module):
    def forward(self, x):
        values = x.new_zeros(x.shape[0])
        values[2] = 1.
        return values


class PlacementScorerTests(unittest.TestCase):
    def observation(self):
        with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
            cfg = get_flat_config(compose(config_name='config', overrides=[
                'hybrid=relational_stationary', 'env=heterogeneous_hybrid']))
        env = SFCEnv(config=cfg)
        self.addCleanup(env.close)
        return env.reset(seed=81)[0]

    def test_selection_uses_one_whole_proposal_and_margin_falls_back_to_base(self):
        obs = self.observation()
        before = pickle.dumps(obs)
        proposals = placement_candidates(obs)
        features = candidate_features(obs, proposals)
        self.assertTrue(np.isfinite(features).all())
        policy = ScoredDeploymentPolicy(FixedScores(), np.zeros(features.shape[-1]),
                                        np.ones(features.shape[-1]))
        action, _ = policy.predict(obs)
        np.testing.assert_array_equal(action, proposals[2])
        policy.margin = 4.
        action, _ = policy.predict(obs)
        np.testing.assert_array_equal(action, proposals[0])
        self.assertEqual(pickle.dumps(obs), before)

    def test_native_round_trip_keeps_decision_and_monitor_rejects_changed_metrics(self):
        obs = self.observation()
        features = candidate_features(obs, placement_candidates(obs))
        torch.manual_seed(91)
        model = DeploymentScorer(features.shape[-1])
        policy = ScoredDeploymentPolicy(model, features.mean(0), np.maximum(features.std(0), .05))
        expected, _ = policy.predict(obs)
        with tempfile.TemporaryDirectory() as temp:
            pipeline = Path(temp)
            folder = pipeline / 'scorer'
            folder.mkdir()
            model_path = folder / 'selected_model.pt'
            policy.save(model_path, run_id='native', selected_epoch=2, training_seed=42)
            loaded = ScoredDeploymentPolicy.load(model_path)
            actual, _ = loaded.predict(obs)
            np.testing.assert_array_equal(actual, expected)
            sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
            metric = dict(episodes=50, success_rate_pct=55., crash_episode_rate_pct=0., mean_episode_reward=680.)
            confirmation = {}
            evaluations = {}
            for name in ['learned', 'static_heuristic', 'heuristic', 'obs_heuristic']:
                confirmation[name] = metric.copy()
                evaluations['scorer' if name == 'learned' else name] = metric.copy()
                atomic_json(folder / (name + '.json'), metric)
            worker = dict(status='complete', training_from_scratch=True, history=[{}, {}],
                          full_optimizer_updates=3, model_sha256=sha, confirmation=confirmation)
            atomic_json(folder / 'status.json', worker)
            run = dict(status='complete', model_format='deployment_scorer', algorithm='SCORER',
                       directory=str(folder), model_sha256=sha, evaluations=evaluations)
            state = dict(arguments=dict(epochs=2, eval_episodes=50))
            proof = validate_run(run, state, pipeline)
            self.assertIn('model', proof)
            atomic_json(folder / 'learned.json', dict(metric, episodes=5))
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                validate_run(run, state, pipeline)


if __name__ == '__main__':
    unittest.main()
