from types import SimpleNamespace
import unittest
import numpy as np
import torch

from scripts.critic_selection_eval import CriticSelectionPolicy


class CriticSelectionTests(unittest.TestCase):
    def test_selects_pessimistic_double_q_and_uses_policy_action_scaling(self):
        actions = torch.tensor([[0.1], [0.5], [-0.5]])
        model = SimpleNamespace(
            policy=SimpleNamespace(
                obs_to_tensor=lambda obs: ({'state': torch.tensor([[1.]])}, False),
                unscale_action=lambda action: action * 2 + 3),
            actor=lambda obs, deterministic: actions,
            critic=lambda obs, action: (torch.tensor([[100.], [4.], [3.]]),
                                        torch.tensor([[-1.], [4.], [3.]])),
        )
        selected, _ = CriticSelectionPolicy(model, 3).predict({})
        np.testing.assert_allclose(selected, [4.])


if __name__ == '__main__':
    unittest.main()
