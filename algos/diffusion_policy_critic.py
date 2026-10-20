# =====================================================================================
# 描述: 实现了与 DiffusionPolicyActor 配套的 Critic 网络。
#       它接收状态和单一的连续动作向量作为输入，来预测Q值。
# =====================================================================================

import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Tuple, List, Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.type_aliases import PyTorchObs


class ContinuousCritic(BaseModel):
    """
    为扩散策略配套的Critic网络 (Q值函数)。
    它接收状态(state)和单一的连续动作向量(action)作为输入。

    :param observation_space: 观察空间
    :param action_space: 动作空间 (必须是 spaces.Box)
    :param net_arch: 网络架构
    :param features_extractor: 特征提取器
    :param features_dim: 特征维度
    :param activation_fn: 激活函数
    :param normalize_images: 是否归一化图像
    :param n_critics: 要创建的Critic网络数量
    :param share_features_extractor: 是否共享特征提取器
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,  # --- 动作空间现在必须是 Box ---
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        categorical_n_uavs: int = 0,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.features_dim = features_dim
        self.action_dim = get_action_dim(action_space)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.categorical_n_uavs = int(categorical_n_uavs)
        self.q_networks = nn.ModuleList()
        self.net_arch = net_arch
        self.activation_fn = activation_fn

        # Q 网络输入维度 = 特征 (273) + 动作 (62) = 335
        q_net_input_dim = self.features_dim + self.action_dim

        # 创建指定数量的Critic网络

        for _ in range(n_critics):
            q_net = nn.Sequential(
                *create_mlp(q_net_input_dim, 1, net_arch, activation_fn)
            )
            self.q_networks.append(q_net)

    def _canonical_action(self, obs, action):
        """Preserve placement ranking, removing irrelevant logit magnitudes."""
        n = self.categorical_n_uavs
        if not n:
            return action
        logits = action[:, n * 2:].reshape(len(action), -1, n)
        order = logits.argsort(dim=-1, stable=True)
        ranks = torch.zeros_like(logits).scatter(-1, order,
            torch.linspace(-1, 1, n, device=action.device, dtype=action.dtype).expand_as(logits))
        active = obs['placement_context'].reshape(len(action), -1, n + 1)[..., -1:]
        # The forward path is rank invariant; retain the categorical actor's
        # straight-through gradient when an experiment enables placement Q loss.
        encoded = (logits + (ranks - logits).detach()) * active
        return torch.cat([action[:, :n * 2], encoded.flatten(1)], dim=1)

    def forward(
        self, obs: PyTorchObs, action: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        :param obs: 观察, shape: (batch_size, obs_dim)
        :param action: 动作, shape: (batch_size, action_dim) <-- 注意：不再是Tuple
        :return: 一个包含每个Critic网络输出的Q值的元组
        """
        assert self.features_extractor is not None, (
            "Features extractor must be initialized"
        )
        # 💡 【核心修复】：如果是共享特征，切断 Critic 的梯度回传！
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)

        qvalue_input = torch.cat([features, self._canonical_action(obs, action)], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: PyTorchObs, action: torch.Tensor) -> torch.Tensor:
        assert self.features_extractor is not None, (
            "Features extractor must be initialized"
        )
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = torch.cat([features, self._canonical_action(obs, action)], dim=1)
        return self.q_networks[0](qvalue_input)

    def q2_forward(self, obs: PyTorchObs, action: torch.Tensor) -> torch.Tensor:
        assert self.features_extractor is not None, (
            "Features extractor must be initialized"
        )
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = torch.cat([features, self._canonical_action(obs, action)], dim=1)
        return self.q_networks[1](qvalue_input)

    def _get_constructor_parameters(self) -> dict:
        """
        返回创建此Critic所需的参数，用于模型保存和加载。
        """
        data = super()._get_constructor_parameters()
        data.update(
            net_arch=self.net_arch,  # ✅ 直接给列表
            n_critics=self.n_critics,
            activation_fn=self.activation_fn,  # ✅ 直接给类
            features_extractor=self.features_extractor,
            features_dim=self.features_dim,
            share_features_extractor=self.share_features_extractor,
            categorical_n_uavs=self.categorical_n_uavs,
        )
        return data
