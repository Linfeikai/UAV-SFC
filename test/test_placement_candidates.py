"""Verify observation-only proposals against existing full-state recipes."""
import pickle
import unittest
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from algos.lookahead_teacher import _brake, _greedy_candidate, RECIPES
from algos.placement_candidates import placement_candidates
from core.sfc_env import SFCEnv
from main import get_flat_config
from test.evalu import smart_heuristic_policy


class PlacementCandidateTests(unittest.TestCase):
    def test_proposals_match_full_state_recipes_without_mutating_observations(self):
        with initialize_config_dir(config_dir=str(Path('conf').resolve()), version_base=None):
            cfg = get_flat_config(compose(config_name='config', overrides=[
                'hybrid=relational_stationary', 'env=heterogeneous_hybrid']))
        env = SFCEnv(config=cfg)
        self.addCleanup(env.close)
        for seed in (31, 32, 33):
            obs, _ = env.reset(seed=seed)
            for step in range(15):
                before = pickle.dumps(obs)
                candidates = placement_candidates(obs)
                self.assertEqual(pickle.dumps(obs), before)
                base = smart_heuristic_policy(env)
                base[:2 * env.N] = _brake(env)
                reference = [base] + [_greedy_candidate(env, *recipe) for recipe in RECIPES]
                np.testing.assert_allclose(candidates, np.stack(reference), atol=2e-6)
                # Random mobility exercises braking from nonzero observed velocity.
                action = candidates[step % len(candidates)].copy()
                action[:2 * env.N] = env.action_space.sample()[:2 * env.N]
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break


if __name__ == '__main__':
    unittest.main()
