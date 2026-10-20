"""Ensure counterfactual placement sees the actual intervened motion."""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from scripts.placement_diagnostic import ComposedPolicy, diagnostic_config


class DiagnosticTests(unittest.TestCase):
    def test_config_file_and_run_directory_restore_same_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.hydra').mkdir()
            config = root / '.hydra' / 'config.yaml'
            config.write_text('env:\n  MAX_STEPS: 7\nreward:\n  CRASH: -123\n')
            file_path, file_cfg = diagnostic_config(root / 'model.zip', config)
            run_path, run_cfg = diagnostic_config(root / 'model.zip', root)
            self.assertEqual(file_path, run_path)
            self.assertEqual(file_cfg, run_cfg)
            self.assertEqual(run_cfg['MAX_STEPS'], 7)
            self.assertEqual(run_cfg['RWD_CRASH'], -123)

    def setUp(self):
        actor = SimpleNamespace(hybrid_categorical_placement=True, relational_placement=True,
                                placement_use_post_mobility_geometry=True,
                                sample_placement=Mock(return_value=(torch.tensor([[0.7]]), None, None)))
        self.model = SimpleNamespace(actor=actor, device='cpu',
            action_space=SimpleNamespace(low=-np.ones(3), high=np.ones(3)),
            policy=SimpleNamespace(obs_to_tensor=lambda obs: (obs, False)),
            predict=Mock(return_value=(np.array([0.8, 0.4, -0.8], dtype=np.float32), None)))
        self.config = dict(NUM_UAVS=1, M=1, K=1, MAX_VNF_LEN=1, GRID_RES=1)
        self.heuristic = np.array([-0.3, 0.2, 0.9], dtype=np.float32)
        self.patch = patch('scripts.placement_diagnostic.observation_heuristic_policy', return_value=self.heuristic)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_learned_learned_is_unchanged(self):
        action, _ = ComposedPolicy(self.model, self.config).predict({})
        np.testing.assert_allclose(action, [0.8, 0.4, -0.8])
        self.model.actor.sample_placement.assert_not_called()

    def test_intervened_mobility_recomputes_conditional_placement(self):
        action, _ = ComposedPolicy(self.model, self.config, 'heuristic', 'learned').predict({})
        np.testing.assert_allclose(action, [-0.3, 0.2, 0.7])
        kwargs = self.model.actor.sample_placement.call_args.kwargs
        np.testing.assert_allclose(kwargs['mobility_action'].numpy(), [[-0.3, 0.2]])
        self.assertTrue(kwargs['deterministic'])
        self.assertFalse(kwargs['straight_through'])

    def test_placement_intervention_keeps_learned_mobility(self):
        action, _ = ComposedPolicy(self.model, self.config, 'learned', 'heuristic').predict({})
        np.testing.assert_allclose(action, [0.8, 0.4, 0.9])
        self.model.actor.sample_placement.assert_not_called()

    def test_heuristic_heuristic_does_not_sample_model(self):
        action, _ = ComposedPolicy(self.model, self.config, 'heuristic', 'heuristic').predict({})
        np.testing.assert_allclose(action, self.heuristic)
        self.model.predict.assert_not_called()


if __name__ == '__main__':
    unittest.main()
