# =====================================================================================
# 描述: 实现了基于扩散模型的Actor。
#       新增了 approximate_log_prob 方法，用于精确计算 log π(a|s)。
# =====================================================================================

import torch
import torch.nn as nn
import math
from gymnasium import spaces
from typing import Tuple, Dict, Any, List
from torch import Tensor

# 从SB3导入必要的基类和类型提示
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs

# -------------------------------------------------------------------------------------
# 辅助模块和函数
# -------------------------------------------------------------------------------------


def _precompute_diffusion_schedule(
    T_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> Dict[str, Tensor]:
    """预计算扩散过程的所有beta/alpha相关参数"""
    betas = torch.linspace(beta_start, beta_end, T_steps, dtype=torch.float32)
    alphas = 1.0 - betas  # 保留原始 alpha_t
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8)
    )  # 防止除0

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "alphas_cumprod_prev": alphas_cumprod_prev,
    }


class _SinusoidalTimestepEmbedding(nn.Module):
    """
    将标量时间步 t 编码成一个高维向量。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        device = timestep.device
        half_dim = self.dim // 2
        # 修正：使用更稳健的广播计算
        freqs = torch.exp(
            torch.arange(half_dim, device=device)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        # 无论输入是 [B] 还是 [B, 1]，都强制转为 [B, 1] 进行外积
        args = timestep.float().view(-1, 1) * freqs.view(1, -1)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


# -------------------------------------------------------------------------------------
# 核心网络：EpsilonNet
# -------------------------------------------------------------------------------------
class _EpsilonNet(nn.Module):
    """
    噪声预测网络的具体实现。作为 DiffusionPolicyActor 的一个内部模块。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        time_embedding_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # 时间步编码器
        self.time_encoder = _SinusoidalTimestepEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 状态和动作的编码器
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Mish())
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.Mish()
        )

        # 融合信息的主干网络
        combined_dim = hidden_dim + hidden_dim + hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, state: torch.Tensor, noisy_action: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        # 分别编码
        time_features = self.time_mlp(self.time_encoder(timestep))
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(noisy_action)

        # 融合并处理
        combined_features = torch.cat(
            [state_features, action_features, time_features], dim=-1
        )
        backbone_output = self.backbone(combined_features)

        # 输出预测的噪声
        return self.output_layer(backbone_output)


# =====================================================================================
# 主类：DiffusionPolicyActor (替换原有的 HybridActor)
# =====================================================================================
class DiffusionPolicyActor(BasePolicy):
    """
    基于扩散模型的Actor。
    它将动作空间视为一个单一的、可通过去噪过程生成的多模态连续空间。
    """

    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    alphas_cumprod_prev: torch.Tensor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,  # 动作空间必须是 Box
        net_arch: List[int],  # 这里的 net_arch 可以用于 EpsilonNet 的 hidden_dim
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        n_uavs: int,  # 用于构建 Actor 的掩码参数维度
        m_candidates: int,  # 用于构建 Actor 的掩码参数维度
        core_features_dim: int,  # 用于构建 Actor 的核心特征维度
        # 扩散模型特定超参数
        T_steps: int = 20,  # 扩散总步数 T_total
        log_prob_n_steps: int = 10,  # 数值积分步数 T_log_prob
        log_prob_n_samples: int = 10,  # 蒙特卡洛采样数 N
        normalize_images: bool = True,
        entropy_scale: float = 1.0,  # 用于熵代理的缩放系数
        t_min: float = 1e-3,  # 积分区间的下界
        t_max: float = 1.0 - 1e-3,  # 积分区间的上界
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if not isinstance(action_space, spaces.Box):
            raise ValueError("DiffusionPolicyActor 要求动作空间为 spaces.Box。")

        self.action_dim = action_space.shape[0]
        self.features_dim = features_dim
        self.n_uavs = n_uavs
        self.m_candidates = m_candidates
        self.core_features_dim = core_features_dim
        self.T = T_steps
        self.T_log_prob = log_prob_n_steps
        self.N_log_prob = log_prob_n_samples
        self.entropy_scale = entropy_scale
        self.t_min = t_min
        self.t_max = t_max
        self.net_arch = net_arch  # <-- 新增这行，存下来

        hidden_dim = net_arch[0] if net_arch else 256
        self._epsilon_net = _EpsilonNet(
            state_dim=self.core_features_dim,  # 只把核心特征传给 epsilon_net
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
        )

        diffusion_schedule = _precompute_diffusion_schedule(self.T)
        for key, value in diffusion_schedule.items():
            self.register_buffer(key, value)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            net_arch=self.net_arch,  # <-- 改成直接读取属性，更优雅
            features_dim=self.features_dim,  # <-- 【必须增加】保存特征维度
            # --- 【新增】确保模型保存/加载后依然能正确执行 Ray Mask ---
            n_uavs=self.n_uavs,
            m_candidates=self.m_candidates,
            core_features_dim=self.core_features_dim,
            # -------------------------------------------------------
            T_steps=self.T,
            log_prob_n_steps=self.T_log_prob,
            log_prob_n_samples=self.N_log_prob,
            entropy_scale=self.entropy_scale,
            t_min=self.t_min,
            t_max=self.t_max,
        )
        return data

    def _apply_ray_mask(
        self, pred_x0: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        分段非对称射线遮罩 (Segmented Asymmetric Ray Mask)
        """
        """修正版：解决 Inplace 梯度破坏与 Clamp Tensor 兼容性"""
        # 1. 克隆以保护梯度
        new_x0 = pred_x0.clone()
        B = new_x0.shape[0]

        # 2. 提取掩码参数 [Batch, n_uavs, 4] -> [L, R, B, T]
        masks_start = self.core_features_dim
        mob_masks = features[:, masks_start : masks_start + self.n_uavs * 4].view(
            B, self.n_uavs, 4
        )
        pick_limit = features[
            :, masks_start + self.n_uavs * 4 : masks_start + self.n_uavs * 4 + 1
        ]

        # --- A. 向量化处理 Mobility (0-7 维) ---
        # Step 1: 将 8 维展成 [Batch, n_uavs, 2] -> 分离出 (vx, vy)
        mobility_part = new_x0[:, : self.n_uavs * 2].reshape(B, self.n_uavs, 2).clone()
        vx = mobility_part[:, :, 0]
        vy = mobility_part[:, :, 1]

        # Step 2: 向量化选择缩放系数
        # X轴：vx >= 0 ? Right(索引1) : Left(索引0)
        mask_x = torch.where(vx >= 0, mob_masks[:, :, 1], mob_masks[:, :, 0])
        # Y轴：vy >= 0 ? Top(索引3) : Bottom(索引2)
        mask_y = torch.where(vy >= 0, mob_masks[:, :, 3], mob_masks[:, :, 2])

        # Step 3: 并行应用缩放
        # 利用 view 的内存共享机制，直接修改 new_x0 的前 8 位
        mobility_part[:, :, 0] = vx * mask_x
        mobility_part[:, :, 1] = vy * mask_y

        # 将修改后的部分刷回 new_x0 (reshape 会保持内存视图一致)
        new_x0[:, : self.n_uavs * 2] = mobility_part.reshape(B, -1)
        # --- B. Pick 处理 (8-13 维) ---
        p_start = 2 * self.n_uavs
        p_end = p_start + self.m_candidates
        picks = new_x0[:, p_start:p_end]
        # 修正：使用 minimum/maximum 解决 Tensor 边界报错
        picks = torch.maximum(picks, torch.tensor(-1.0, device=picks.device))
        new_x0[:, p_start:p_end] = torch.minimum(picks, pick_limit)

        return new_x0

    def _sample_from_noise(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
        inference_steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """修正版：统一 DDIM 数学公式与 None 比较逻辑"""
        batch_size = features.shape[0]
        action_t = torch.randn((batch_size, self.action_dim), device=self.device)
        step_indices = (
            torch.linspace(self.T - 1, 0, inference_steps, device=self.device)
            .round()
            .long()
        )

        last_noise = torch.zeros_like(action_t)

        for i in range(len(step_indices)):
            t_curr = step_indices[i]
            t_prev = step_indices[i + 1] if i < len(step_indices) - 1 else None
            t_tensor = torch.full(
                (batch_size,), int(t_curr.item()), device=self.device, dtype=torch.long
            )

            core_features = features[:, : self.core_features_dim]
            predicted_noise = self._epsilon_net(
                core_features, action_t, t_tensor.unsqueeze(1)
            )

            if i == len(step_indices) - 1:
                last_noise = predicted_noise

            alpha_bar_t = self.alphas_cumprod[t_curr]
            alpha_bar_prev = (
                self.alphas_cumprod[t_prev]
                if t_prev is not None
                else torch.tensor(1.0, device=self.device)
            )

            pred_x0 = (
                action_t - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)
            pred_x0 = self._apply_ray_mask(pred_x0, features)

            # --- 二、数学实现统一 (DDIM 分支) ---
            if not deterministic and t_prev is not None:
                eta = 1.0  # 对应 DDPM 强度
                var = (
                    (1.0 - alpha_bar_prev)
                    / (1.0 - alpha_bar_t)
                    * (1.0 - alpha_bar_t / alpha_bar_prev)
                )
                sigma_t = eta * torch.sqrt(torch.clamp(var, min=1e-8))
            else:
                sigma_t = 0.0

            # 严格遵循 DDIM 公式：sqrt(1 - alpha - sigma^2)
            dir_coeff = torch.sqrt(
                torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)
            )
            dir_xt_prev = dir_coeff * predicted_noise
            noise_term = sigma_t * torch.randn_like(action_t)
            action_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt_prev + noise_term

        return action_t, last_noise

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        unbounded_action, _ = self._sample_from_noise(features, deterministic)
        return torch.tanh(unbounded_action)

    def _predict(
        self, observation: PyTorchObs, deterministic: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(observation, deterministic=deterministic)

    def action_log_prob(
        self,
        obs: PyTorchObs,
        use_entropy_proxy: bool = True,  # <-- 直接在这里设为 True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成动作并计算其对数概率。
        """
        features = self.extract_features(obs, self.features_extractor)
        unbounded_action, last_noise = self._sample_from_noise(
            features, deterministic=False
        )
        policy_action = torch.tanh(unbounded_action)
        # 2. 使用建议的噪声范数作为代理对数概率

        if use_entropy_proxy:
            # 1. 代理熵：用负 L2 范数近似无界空间的对数概率 log p(u)
            log_prob_u = (
                -torch.sum(unbounded_action**2, dim=-1, keepdim=True)
                * self.entropy_scale
            )
        else:
            # 1. 精确积分：计算无界空间的对数概率 log p(u) (速度极慢)
            log_prob_u = self.approximate_log_prob(features, policy_action)

        # 2. 【无论走哪个分支，都必须执行】Tanh 雅可比修正
        # log p(a) = log p(u) - sum(log(1 - tanh(u)^2))
        squash_correction = torch.sum(
            torch.log(1 - policy_action**2 + 1e-6), dim=-1, keepdim=True
        )

        # 3. 最终的对数概率
        log_prob = log_prob_u - squash_correction

        return policy_action, log_prob

    def approximate_log_prob(
        self, features: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        根据论文公式21，使用数值积分精确地近似log_prob。
        """
        # 0. 准备参数
        B, A = action.shape  # Batch size, Action dimension

        # 将动作从 [-1, 1] 映射回无界空间，因为扩散过程定义在无界空间
        # 注意：这里假设 forward 采样后，action是tanh()的结果
        unbounded_action = torch.atanh(action.clamp(-1 + 1e-6, 1 - 1e-6))

        # 1. 准备积分所需的时间步和alpha_bar值
        # 在 [t_min, t_max] 之间均匀取 T_log_prob+1 个点来定义 T_log_prob 个积分区间
        time_steps = torch.linspace(
            self.t_min, self.t_max, self.T_log_prob + 1, device=self.device
        )

        # 将连续时间 t (0-1) 映射到离散的 schedule 索引 (0 to T-1)
        discrete_indices = torch.clamp((time_steps * self.T).long(), max=self.T - 1)
        # 获取 alpha_bar_{t_{i-1}} 和 alpha_bar_{t_i}
        # alphas_cumprod 是预先计算好的 alpha_bar schedule
        alpha_hat_t_minus_1 = self.alphas_cumprod[discrete_indices[:-1]]
        alpha_hat_t = self.alphas_cumprod[discrete_indices[1:]]

        # 2. 向量化计算所有时间步的误差
        # 扩展维度以进行批处理计算: [B, A] -> [B, N, T, A]
        action_expanded = (
            unbounded_action.unsqueeze(1)
            .unsqueeze(1)
            .expand(B, self.N_log_prob, self.T_log_prob, A)
        )
        features_expanded = (
            features.unsqueeze(1)
            .unsqueeze(1)
            .expand(B, self.N_log_prob, self.T_log_prob, -1)
        )

        # 扩展时间步相关参数: [T] -> [1, 1, T, 1]
        alpha_hats_expanded = alpha_hat_t.view(1, 1, self.T_log_prob, 1)

        # 采样N次噪声: [B, N, T, A]
        epsilon = torch.randn_like(action_expanded)

        # 计算加噪动作 a_t
        noisy_actions = (
            torch.sqrt(alpha_hats_expanded) * action_expanded
            + torch.sqrt(1.0 - alpha_hats_expanded) * epsilon
        )

        # 扩展离散时间步用于epsilon-net输入: [T] -> [1, 1, T, 1]
        discrete_t_expanded = (
            discrete_indices[1:]
            .view(1, 1, self.T_log_prob, 1)
            .expand(B, self.N_log_prob, -1, -1)
        )
        core_features_for_net = features_expanded[
            :, :, :, : self.core_features_dim
        ]  # 只取核心特征输入 epsilon_net
        # 预测噪声 ε_φ
        # reshape for batch matmul: [B*N*T, Dim]
        predicted_noise = self._epsilon_net(
            features_expanded[:, :, :, : self.core_features_dim].reshape(
                -1, self.core_features_dim
            ),
            noisy_actions.reshape(-1, action.shape[1]),
            discrete_t_expanded.reshape(-1),  # 改为一维
        ).reshape(
            features_expanded.shape[0],
            self.N_log_prob,
            self.T_log_prob,
            action.shape[1],
        )

        # 计算平均MSE误差 ε̃φ
        error_mse = torch.sum(
            (epsilon - predicted_noise) ** 2, dim=-1
        )  # Sum over action dim: [B, N, T]
        average_mse = torch.mean(error_mse, dim=1)  # Mean over N samples: [B, T]

        # 3. 计算积分权重 w_ti
        # [T] -> [1, T] for broadcasting
        alpha_hats_t_reshaped = alpha_hat_t.view(1, -1)
        alpha_hat_t_minus_1_reshaped = alpha_hat_t_minus_1.view(1, -1)

        # VP SDE 下 σ(α) = α, σ(-α) = 1-α
        # 这里使用论文中的公式
        sigma_at = alpha_hats_t_reshaped
        sigma_neg_at = 1.0 - alpha_hats_t_reshaped
        sigma_at_prev = alpha_hat_t_minus_1_reshaped

        # w_ti = (σ(α_ti-1) - σ(α_ti)) / (σ(α_ti) * σ(-α_ti))
        integration_weight = (sigma_at_prev - sigma_at) / (
            sigma_at * sigma_neg_at + 1e-8
        )

        # 4. 计算求和项 (d * σ(α_ti) - ε̃φ)
        term_inside_sum = A * sigma_at - average_mse

        # 5. 最终求和并加上常数 c'
        # [B, T] * [1, T] -> [B, T], then sum over T dim
        weighted_term = term_inside_sum * integration_weight
        sum_term = torch.sum(weighted_term, dim=-1)  # Result shape: [B]

        # c' = -d/2 * log(2πε)
        c_prime = -0.5 * A * math.log(2 * math.pi * math.e)

        # log p ≈ c' + 1/2 * Σ(...)
        log_prob = c_prime + 0.5 * sum_term

        return log_prob.unsqueeze(-1)  # Ensure shape is [B, 1]
