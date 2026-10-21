"""Guard teacher state isolation and the deployment labels' executed meaning."""
import copy
import pickle
import random
import unittest

import numpy as np

from algos.lookahead_teacher import lookahead_expert_action, _brake
from core.sfc_env import SFCEnv
from test.evalu import smart_heuristic_policy


class LookaheadTeacherTests(unittest.TestCase):
    def environment(self):
        # The complete original fixed environment, including reward/physical settings.
        from hydra import compose, initialize_config_dir
        from main import get_flat_config
        from pathlib import Path
        with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
            cfg = get_flat_config(compose(config_name='config', overrides=[
                'hybrid=relational_stationary', 'env=heterogeneous_hybrid']))
        env = SFCEnv(config=dict(cfg, RECORD_DEPLOYMENT=True))
        self.addCleanup(env.close)
        env.reset(seed=29)
        return env

    def test_query_preserves_live_state_and_random_streams(self):
        env = self.environment()
        # Exercise a moving state as well as the initial static state.
        env.step(env.action_space.sample())
        before = pickle.dumps(env, protocol=5)
        random.seed(37)
        np.random.seed(37)
        expected = random.random(), np.random.random()
        random.seed(37)
        np.random.seed(37)
        action = lookahead_expert_action(env)
        self.assertEqual(pickle.dumps(env, protocol=5), before)
        self.assertEqual((random.random(), np.random.random()), expected)
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue(env.action_space.contains(action))

    def test_encoded_labels_keep_selected_outcome_and_dominate_base(self):
        env = self.environment()
        for step in range(20):
            baseline = smart_heuristic_policy(env)
            baseline[:2 * env.N] = _brake(env)
            base_branch = copy.deepcopy(env)
            _, base_reward, _, _, _ = base_branch.step(baseline)
            action = lookahead_expert_action(env)
            # Candidate budget 1 must be exactly equivalent to the baseline,
            # including any canonicalization of executed assignment labels.
            encoded_base = lookahead_expert_action(env, candidate_budget=1)
            encoded_branch = copy.deepcopy(env)
            _, encoded_reward, _, _, _ = encoded_branch.step(encoded_base)
            self.assertAlmostEqual(encoded_reward, base_reward, places=9)
            branch = copy.deepcopy(env)
            _, branch_reward, _, _, branch_info = branch.step(action)
            _, reward, terminated, truncated, info = env.step(action)
            self.assertAlmostEqual(reward, branch_reward, places=9)
            self.assertGreaterEqual(reward + 1e-9, base_reward)
            self.assertEqual(info['detailed_mappings'], branch_info['detailed_mappings'])
            # The preferred labels must reproduce the chosen executed mapping.
            preferred = action[2 * env.N:].reshape(env.K, env.L, env.N).argmax(-1)
            for k, (_, mapping) in enumerate(branch_info['detailed_mappings'].items()):
                np.testing.assert_array_equal(preferred[k, :len(mapping)], mapping)
            base_branch.close()
            encoded_branch.close()
            branch.close()
            if terminated or truncated:
                break


if __name__ == '__main__':
    unittest.main()
