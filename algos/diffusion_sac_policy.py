# =====================================================================================
# 文件名: diffusion_sac_policy.py (最终修复版本)
# 描述: 通过重写_build方法，完全掌控Actor和Critic的创建过程，彻底解决参数传递问题。
# =====================================================================================

import torch as th
import torch.nn as nn
from gymnasium import spaces
from typing import Dict, Any, List, Type, Optional, Union
import numpy as np

# 导入我们新定义的Actor和Critic
from .diffusion_policy_actor import DiffusionPolicyActor
from .diffusion_policy_critic import ContinuousCritic

# 从SB3导入必要的基类和类型提示
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    get_actor_critic_arch,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule, PyTorchObs


# =====================================================================================
# 新策略: DiffusionSACPolicy
# 描述: 这是一个专门为 DiffusionSACAgent 设计的策略。
#       它将 DiffusionPolicyActor 和 ContinuousCritic 组合在一起。
#       这个策略只处理连续动作空间 (Box)。
# =====================================================================================
class DiffusionSACPolicy(BasePolicy):
    """
    基于扩散模型的SAC策略 (仅支持连续动作空间).
    """

    actor: DiffusionPolicyActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,  # <-- 明确指定为Box
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        # --- Diffusion Actor 特有的参数 ---
        T: int = 5,
        beta_schedule: str = "linear",
        n_uavs: int = 4,  # 用于构建 Actor 的掩码参数维度
        m_candidates: int = 6,  # 用于构建 Actor 的掩码参数维度
        core_features_dim: int = 256,  # 用于构建 Actor 的核心特征维度
        **kwargs: Any,
    ):
        # print(
        #     "DiffusionSACPolicy kwargs:", kwargs
        # )  # 检查是否包含 lr_actor_schedule 和 lr_critic_schedule

        lr_actor_schedule = kwargs.pop("lr_actor_schedule", lr_schedule)
        lr_critic_schedule = kwargs.pop("lr_critic_schedule", lr_schedule)
        self.lr_actor_schedule = lr_actor_schedule
        self.lr_critic_schedule = lr_critic_schedule
        # 调用父类构造函数，但只传递它能安全处理的核心参数
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
        )

        # --- 手动设置所有其他必要的属性 ---
        # 绕过在super().__init__中传递它们时可能引发的TypeError
        self.normalize_images = normalize_images
        self.share_features_extractor = share_features_extractor

        # SAC/DiffusionSAC 策略需要将动作压缩到 [-1, 1] 范围
        # 'squash_output' 是一个只读属性，所以我们必须设置底层的私有变量
        self._squash_output = True

        if net_arch is None:
            net_arch = [256, 256]
        self.activation_fn = activation_fn
        self.n_critics = n_critics

        # -- 保存 Diffusion Actor 的参数 --
        self.T = T
        self.beta_schedule = beta_schedule
        self.n_uavs = n_uavs
        self.m_candidates = m_candidates
        self.core_features_dim = core_features_dim

        # actor和critic共同的参数
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "activation_fn": self.activation_fn,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()

        # 从 net_arch 中分离 actor 和 critic 的网络结构
        # get_actor_critic_arch 是一个辅助函数，如果 net_arch 是列表，则两者共享；
        # 如果是字典 {'pi': [...], 'vf': [...]} (或 'actor', 'critic')，则分别使用。
        actor_arch, critic_arch = get_actor_critic_arch(net_arch)
        # 更新 actor 和 critic 各自的参数
        self.actor_kwargs.update(
            {
                "net_arch": actor_arch,
                "T_steps": self.T,
                "n_uavs": self.n_uavs,
                "m_candidates": self.m_candidates,
                "core_features_dim": self.core_features_dim,
            }
        )
        self.critic_kwargs.update(
            {
                "net_arch": critic_arch,
            }
        )

        self._build(lr_actor_schedule, lr_critic_schedule)

    def _build(self, lr_actor_schedule: Schedule, lr_critic_schedule: Schedule) -> None:
        """
        创建actor, critic, 和它们的优化器.
        """
        # 1. 先创建 Actor (传入默认的 None，底层会自动实例化 SFCFeaturesExtractor)
        # 注意：这里调用的是 self.make_actor() 而不是传入 self.features_extractor
        self.actor = self.make_actor()
        # 2. 根据共享标志创建 Critic
        if self.share_features_extractor:
            # 共享模式：直接从 Actor 那里“借用”刚造好的特征提取器实例
            self.critic = self.make_critic(
                features_extractor=self.actor.features_extractor
            )
        else:
            # 独立模式：传入 None，底层会为 Critic 再造一个全新的实例
            self.critic = self.make_critic(features_extractor=None)

        # 3. 创建 Critic Target (永远独立，不与 Critic 共享提取器的内存引用)
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.set_training_mode(False)  # Freeze target network

        # 4.Setup optimizers
        # 💡 【修复警告】：将 lr 塞进 kwargs 字典中，绕过 Pylance 的函数签名检查
        actor_optim_kwargs = self.optimizer_kwargs.copy()
        actor_optim_kwargs["lr"] = lr_actor_schedule(1)

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(), **actor_optim_kwargs
        )

        # 💡 【关键修复】：如果共享特征，必须从 Critic 优化器中剔除 features_extractor
        if self.share_features_extractor:
            critic_parameters = [
                param
                for name, param in self.critic.named_parameters()
                if "features_extractor" not in name
            ]
        else:
            critic_parameters = list(self.critic.parameters())

        critic_optim_kwargs = self.optimizer_kwargs.copy()
        critic_optim_kwargs["lr"] = lr_critic_schedule(1)

        self.critic.optimizer = self.optimizer_class(
            critic_parameters, **critic_optim_kwargs
        )

        self.to(self.device)

    def make_actor(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> DiffusionPolicyActor:
        """
        创建 Diffusion Actor.
        """
        # 这里就是把actor_kwargs输入_update_features_extractor函数，是创建一个新的features_extractor（如果传入的参数是None），
        # 并把这个features_extractor和它的输出维度features_dim更新到actor_kwargs这个字典里。
        # 最后再用更新后的actor_kwargs来实例化DiffusionPolicyActor。
        # 更新后actor_kwargs里就包含了正确的features_extractor实例和features_dim值，确保DiffusionPolicyActor能正确构建。
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor
        )
        return DiffusionPolicyActor(**actor_kwargs).to(self.device)

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> ContinuousCritic:
        """
        创建 Critic 网络.
        """
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        前向传播.
        """
        return self._predict(obs, deterministic=deterministic)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> th.Tensor:
        """
        获取动作. 这是 `predict()` 的核心.
        """
        return self.actor(observation, deterministic)

    def set_training_mode(self, mode: bool) -> None:
        """
        将策略及其子模块设置为训练模式或评估模式.
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters()
