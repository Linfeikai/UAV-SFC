"""Exercise mobility regularization and validation isolation through real agents."""
import random
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from algos.diffusion_sac_agent import DiffusionSACAgent
from algos.gaussian_hybrid_sac_agent import GaussianHybridSACAgent
from algos.diffusion_sac_policy import DiffusionSACPolicy
from algos.diffusion_extractor import SFCFeaturesExtractor
from core.sfc_env import SFCEnv
from scripts.training_validation import preserve_training_state, validation_score
from test.evalu import smart_heuristic_policy


class MotionLearningTests(unittest.TestCase):
    def test_validation_restores_random_streams_and_mode_even_on_error(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        expected = (random.random(), np.random.random(), torch.rand(3))
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        model = SimpleNamespace(policy=torch.nn.Linear(1, 1))
        model.policy.set_training_mode = model.policy.train
        torch.manual_seed(17)  # Linear construction consumed Torch's stream.
        with self.assertRaises(RuntimeError):
            with preserve_training_state(model):
                model.policy.train(False)
                random.seed(99)
                np.random.seed(99)
                torch.manual_seed(99)
                raise RuntimeError('evaluation failed')
        self.assertTrue(model.policy.training)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2])

    def test_unsafe_high_success_does_not_replace_safe_checkpoint(self):
        safe = dict(crash_episode_rate_pct=0, success_rate_pct=49, mean_episode_reward=600)
        unsafe = dict(crash_episode_rate_pct=20, success_rate_pct=55, mean_episode_reward=650)
        self.assertGreater(validation_score(safe), validation_score(unsafe))

    def test_online_mobility_updates_are_finite_and_survive_reload(self):
        torch.set_num_threads(1)
        for cls, kind in ((DiffusionSACAgent, 'diffusion'), (GaussianHybridSACAgent, 'gaussian')):
            with self.subTest(policy=kind), tempfile.TemporaryDirectory() as directory:
                env = SFCEnv(config={'EXPOSE_PLACEMENT_CONTEXT': True})
                self.addCleanup(env.close)
                obs, _ = env.reset(seed=14)
                action = smart_heuristic_policy(env)
                options = dict(qne_k_samples=2) if kind == 'diffusion' else {}
                model = cls(DiffusionSACPolicy, env, seed=14, device='cpu',
                    buffer_size=32, batch_size=4, learning_starts=0,
                    mobility_bc_coef=1, categorical_bc_coef=1,
                    policy_kwargs=dict(net_arch=[32, 32], T=5,
                        features_extractor_class=SFCFeaturesExtractor,
                        features_extractor_kwargs=dict(n_uavs=env.N, m_candidates=env.M,
                                                       grid_res=env.grid_res),
                        hybrid_categorical_placement=True, relational_placement=True,
                        mobility_policy=kind, detach_placement_for_rl=(kind == 'gaussian')),
                    **options)
                model.set_expert_data([obs], np.array([action]))
                model.learn(total_timesteps=8)
                self.assertGreater(model._n_updates, 0)
                self.assertTrue(np.isfinite(model.logger.name_to_value['train/mobility_bc_loss']))
                self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
                path = Path(directory) / 'model'
                model.save(path)
                loaded = cls.load(path, device='cpu')
                self.assertEqual(loaded.mobility_bc_coef, 1)
                prediction, _ = loaded.predict(obs)
                self.assertTrue(np.isfinite(prediction).all())


if __name__ == '__main__':
    unittest.main()
