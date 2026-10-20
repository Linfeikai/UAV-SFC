import unittest
import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from algos.diffusion_policy_critic import ContinuousCritic


class Extractor(BaseFeaturesExtractor):
    def __init__(self, space):
        super().__init__(space, 4)
        self.layer = torch.nn.Linear(3, 4)
        self.was_grad_enabled = None

    def forward(self, obs):
        self.was_grad_enabled = torch.is_grad_enabled()
        return self.layer(obs['state'])


class CriticRepresentationTests(unittest.TestCase):
    def make(self, share=False):
        space = gym.spaces.Dict(dict(
            state=gym.spaces.Box(-1, 1, (3,)),
            mobility_bounds=gym.spaces.Box(0, 1, (2, 4)),
            placement_context=gym.spaces.Box(0, 1, (6,))))
        extractor = Extractor(space)
        critic = ContinuousCritic(space, gym.spaces.Box(-1, 1, (8,)), [8],
                                  extractor, 4, share_features_extractor=share,
                                  categorical_n_uavs=2)
        obs = dict(state=torch.tensor([[.2, .3, .4]]),
                   mobility_bounds=torch.ones(1, 2, 4),
                   placement_context=torch.tensor([[.1, .2, 1., 0, 0, 0]]))
        return critic, extractor, obs

    def test_q_ignores_equivalent_logit_scales_and_inactive_slots(self):
        critic, _, obs = self.make()
        first = torch.tensor([[0., 0, 0, 0, -.5, .5, .8, -.8]], requires_grad=True)
        second = torch.tensor([[0., 0, 0, 0, -.25, .25, -.3, .2]])
        torch.testing.assert_close(critic(obs, first)[0], critic(obs, second)[0])
        critic(obs, first)[0].sum().backward()
        torch.testing.assert_close(first.grad[:, -2:], torch.zeros(1, 2))

    def test_independent_encoder_learns_and_no_grad_remains_disabled(self):
        for share in (False, True):
            critic, extractor, obs = self.make(share)
            critic(obs, torch.zeros(1, 8))[0].sum().backward()
            self.assertEqual(extractor.layer.weight.grad is None, share)
            with torch.no_grad():
                critic(obs, torch.zeros(1, 8))
            self.assertFalse(extractor.was_grad_enabled)


if __name__ == '__main__':
    unittest.main()
