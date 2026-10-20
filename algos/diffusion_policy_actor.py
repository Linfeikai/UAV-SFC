# =====================================================================================
# 描述: 实现了基于扩散模型的Actor。
#       新增了 approximate_log_prob 方法，用于精确计算 log π(a|s)。
# =====================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from core import uav as flight_physics
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
    T_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02,
    beta_schedule: str = "linear",
) -> Dict[str, Tensor]:
    """预计算扩散过程的所有beta/alpha相关参数"""
    if T_steps < 2:
        raise ValueError("Diffusion requires at least two timesteps")
    if beta_schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T_steps, dtype=torch.float32)
    elif beta_schedule == "cosine":
        # Nichol & Dhariwal's discretization: near-zero terminal signal even
        # with a short action diffusion chain. Keep legacy linear checkpoints.
        times = torch.linspace(0, 1, T_steps + 1, dtype=torch.float64)
        cumulative = torch.cos((times + 0.008) / 1.008 * math.pi / 2).square()
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999).float()
    else:
        raise ValueError(f"Unknown beta schedule: {beta_schedule}")
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
        m_candidates: int,  # 候选任务池大小 M
        decision_tasks: int,  # 每步最多决策任务数 K
        core_features_dim: int,  # 用于构建 Actor 的核心特征维度
        # 扩散模型特定超参数
        T_steps: int = 20,  # 扩散总步数 T_total
        beta_schedule: str = "linear",
        clip_denoised: float | None = None,
        log_prob_n_steps: int = 10,  # 数值积分步数 T_log_prob
        log_prob_n_samples: int = 10,  # 蒙特卡洛采样数 N
        normalize_images: bool = True,
        entropy_scale: float = 1.0,  # 用于熵代理的缩放系数
        t_min: float = 1e-3,  # 积分区间的下界
        t_max: float = 1.0 - 1e-3,  # 积分区间的上界
        hybrid_categorical_placement: bool = False,
        categorical_temperature: float = 0.7,
        placement_prior_scale: float = 1.0,
        relational_placement: bool = False,
        placement_use_post_mobility_geometry: bool = True,
        mobility_scale_x: float = 0.04,
        mobility_scale_y: float = 0.04,
        mobility_dynamics: dict | None = None,
        placement_embedding_dim: int = 16,
        mobility_policy: str = "diffusion",
        detach_placement_for_rl: bool = False,
        gaussian_log_std_min: float = -5.0,
        gaussian_log_std_max: float = 2.0,
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
        self.mobility_dim = n_uavs * 2
        self.placement_dim = self.action_dim - self.mobility_dim
        self.features_dim = features_dim
        self.n_uavs = n_uavs
        self.m_candidates = m_candidates
        self.decision_tasks = decision_tasks
        self.core_features_dim = core_features_dim
        self.T = T_steps
        self.T_log_prob = log_prob_n_steps
        self.N_log_prob = log_prob_n_samples
        self.entropy_scale = entropy_scale
        self.t_min = t_min
        self.t_max = t_max
        self.net_arch = net_arch  # <-- 新增这行，存下来
        self.hybrid_categorical_placement = hybrid_categorical_placement
        self.categorical_temperature = categorical_temperature
        self.placement_prior_scale = placement_prior_scale
        self.relational_placement = relational_placement
        self.placement_use_post_mobility_geometry = (
            placement_use_post_mobility_geometry
        )
        self.mobility_scale_x = mobility_scale_x
        self.mobility_scale_y = mobility_scale_y
        self.mobility_dynamics = dict(
            max_speed=20.0, acceleration=(5.0, 5.0), dt=1.0,
            min_safe_distance=10.0, battery_capacity=30000.0, env_mask=False,
        )
        self.mobility_dynamics.update(mobility_dynamics or {})
        self.placement_embedding_dim = placement_embedding_dim
        self.mobility_policy = mobility_policy.lower()
        self.detach_placement_for_rl = detach_placement_for_rl
        self.gaussian_log_std_min = gaussian_log_std_min
        self.gaussian_log_std_max = gaussian_log_std_max
        if self.mobility_policy not in {"diffusion", "gaussian"}:
            raise ValueError(f"Unsupported mobility policy: {mobility_policy}")

        placement_slots = self.decision_tasks * self.n_uavs
        if self.placement_dim % placement_slots != 0:
            raise ValueError("Placement action dimension is not divisible by K*N")
        self.max_vnf_len = self.placement_dim // placement_slots

        hidden_dim = net_arch[0] if net_arch else 256
        if self.mobility_policy == "diffusion":
            self._epsilon_net = _EpsilonNet(
                state_dim=self.core_features_dim,
                action_dim=(
                    self.mobility_dim
                    if hybrid_categorical_placement
                    else self.action_dim
                ),
                hidden_dim=hidden_dim,
            )
        else:
            if not hybrid_categorical_placement:
                raise ValueError("Gaussian mobility requires categorical placement")
            self.gaussian_mobility_net = nn.Sequential(
                nn.Linear(self.core_features_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, 2 * self.mobility_dim),
            )
        if self.hybrid_categorical_placement:
            placement_input_dim = self.core_features_dim + 3 * self.n_uavs + 1
            if self.relational_placement:
                placement_input_dim += (
                    self.n_uavs + 2 * self.placement_embedding_dim
                )
                self.task_embedding = nn.Embedding(
                    self.decision_tasks, self.placement_embedding_dim
                )
                self.vnf_embedding = nn.Embedding(
                    self.max_vnf_len, self.placement_embedding_dim
                )
            self.placement_net = nn.Sequential(
                nn.Linear(placement_input_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, self.n_uavs),
            )

        self.beta_schedule = beta_schedule
        self.clip_denoised = clip_denoised
        if clip_denoised is not None and clip_denoised <= 0:
            raise ValueError("clip_denoised must be positive")
        diffusion_schedule = _precompute_diffusion_schedule(self.T, beta_schedule=beta_schedule)
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
            decision_tasks=self.decision_tasks,
            core_features_dim=self.core_features_dim,
            # -------------------------------------------------------
            T_steps=self.T,
            beta_schedule=self.beta_schedule,
            clip_denoised=self.clip_denoised,
            log_prob_n_steps=self.T_log_prob,
            log_prob_n_samples=self.N_log_prob,
            entropy_scale=self.entropy_scale,
            t_min=self.t_min,
            t_max=self.t_max,
            hybrid_categorical_placement=self.hybrid_categorical_placement,
            categorical_temperature=self.categorical_temperature,
            placement_prior_scale=self.placement_prior_scale,
            relational_placement=self.relational_placement,
            placement_use_post_mobility_geometry=(
                self.placement_use_post_mobility_geometry
            ),
            mobility_scale_x=self.mobility_scale_x,
            mobility_scale_y=self.mobility_scale_y,
            mobility_dynamics=self.mobility_dynamics,
            placement_embedding_dim=self.placement_embedding_dim,
            mobility_policy=self.mobility_policy,
            detach_placement_for_rl=self.detach_placement_for_rl,
            gaussian_log_std_min=self.gaussian_log_std_min,
            gaussian_log_std_max=self.gaussian_log_std_max,
        )
        return data

    def _relational_geometry(
        self, obs: PyTorchObs, mobility_action: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict post-action UAV positions and read candidate UE positions."""
        state = obs["state"]
        batch_size = state.shape[0]
        uav_state = state[:, : self.n_uavs * 10].reshape(
            batch_size, self.n_uavs, 10
        )
        uav_positions = uav_state[:, :, :2]
        # None means explicitly pre-mobility geometry, whereas a zero action
        # still integrates the UAV's existing velocity.
        predicted_positions = uav_positions
        if mobility_action is not None:
            dynamics = self.mobility_dynamics
            dt = dynamics["dt"]
            max_speed = dynamics["max_speed"]
            mobility = mobility_action[:, : self.mobility_dim].reshape(
                batch_size, self.n_uavs, 2
            )
            if dynamics["env_mask"]:
                bounds = obs["mobility_bounds"].reshape(batch_size, self.n_uavs, 4)
                mask = torch.stack((
                    torch.where(mobility[:, :, 0] >= 0, bounds[:, :, 1], bounds[:, :, 0]),
                    torch.where(mobility[:, :, 1] >= 0, bounds[:, :, 3], bounds[:, :, 2]),
                ), dim=-1)
                mobility = mobility * mask
            initial_velocity = uav_state[:, :, 2:4] * max_speed
            velocity = initial_velocity + mobility * state.new_tensor(
                dynamics["acceleration"]
            ) * dt
            speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
            velocity = velocity * (max_speed / speed.clamp_min(1e-9)).clamp(max=1.0)
            average_velocity = (initial_velocity + velocity) / 2.0

            # Mirror the environment's low-energy hover fallback. The discrete
            # gate is intentional; gradients still flow through feasible moves.
            v = torch.linalg.vector_norm(average_velocity, dim=-1)
            power = (
                flight_physics.P0 * (1 + 3 * v.square() / flight_physics.U_tip**2)
                + flight_physics.P1 * (
                    1 + v**4 / (4 * flight_physics.v_0**4)
                    - v.square() / (2 * flight_physics.v_0**2)
                ).clamp_min(0).sqrt()
                + 0.5 * flight_physics.d_0 * flight_physics.rho
                * flight_physics.s * flight_physics.A * v**3
            )
            battery = uav_state[:, :, 4] * dynamics["battery_capacity"]
            hover_energy = (flight_physics.P0 + flight_physics.P1) * dt
            hover = (battery < power * dt) & (battery >= hover_energy)
            average_velocity = torch.where(
                hover.unsqueeze(-1), torch.zeros_like(average_velocity), average_velocity
            )
            scale = state.new_tensor([self.mobility_scale_x, self.mobility_scale_y])
            predicted_positions = (
                uav_positions + average_velocity * dt * scale / max_speed
            ).clamp(0.0, 1.0)
            # The environment returns before any motion when even hovering fails.
            critical = (battery < hover_energy).any(dim=1)
            predicted_positions = torch.where(
                critical[:, None, None], uav_positions, predicted_positions
            )
            # Match the environment's ordered pairwise collision/freeze rule.
            map_size = max_speed / scale
            for i in range(self.n_uavs):
                for j in range(i + 1, self.n_uavs):
                    distance = torch.linalg.vector_norm(
                        (predicted_positions[:, i] - predicted_positions[:, j]) * map_size,
                        dim=-1,
                    )
                    collision = distance < dynamics["min_safe_distance"]
                    pair = torch.zeros(self.n_uavs, device=state.device, dtype=torch.bool)
                    pair[i] = pair[j] = True
                    predicted_positions = torch.where(
                        collision[:, None, None] & pair[None, :, None],
                        uav_positions, predicted_positions,
                    )

        candidate_offset = state.shape[1] - 3 - self.m_candidates * 5
        candidate_positions = state[
            :, candidate_offset : candidate_offset + self.m_candidates * 5
        ].reshape(batch_size, self.m_candidates, 5)[:, :, :2]
        return predicted_positions, candidate_positions

    def _placement_network_input(
        self,
        core: torch.Tensor,
        requirements: torch.Tensor,
        remaining: torch.Tensor,
        previous: torch.Tensor,
        valid: torch.Tensor,
        source_position: torch.Tensor | None,
        predicted_positions: torch.Tensor | None,
        task_index: int,
        vnf_index: int,
    ) -> torch.Tensor:
        parts = [core, requirements, remaining, previous, valid]
        if self.relational_placement:
            if source_position is None or predicted_positions is None:
                raise ValueError("Relational placement requires predicted geometry")
            distances = torch.linalg.vector_norm(
                predicted_positions - source_position.unsqueeze(1), dim=2
            ) / math.sqrt(2.0)
            batch_size = core.shape[0]
            task_ids = torch.full(
                (batch_size,), task_index, device=core.device, dtype=torch.long
            )
            vnf_ids = torch.full(
                (batch_size,), vnf_index, device=core.device, dtype=torch.long
            )
            parts.extend(
                [distances, self.task_embedding(task_ids), self.vnf_embedding(vnf_ids)]
            )
        return torch.cat(parts, dim=1)

    def sample_placement(
        self,
        obs: PyTorchObs,
        features: torch.Tensor | None = None,
        mobility_action: torch.Tensor | None = None,
        deterministic: bool = False,
        straight_through: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate sequential feasible categorical VNF placements.

        The returned Box-compatible scores are +1 for the selected UAV and -1
        for alternatives. Gradients use the straight-through Gumbel estimator.
        """
        if not self.hybrid_categorical_placement:
            raise RuntimeError("Categorical placement is disabled for this actor")
        if not isinstance(obs, dict) or "placement_context" not in obs:
            raise ValueError("Hybrid placement requires placement_context observations")
        if features is None:
            features = self.extract_features(obs, self.features_extractor)

        batch_size = features.shape[0]
        core = features[:, : self.core_features_dim]
        context = obs["placement_context"].reshape(
            batch_size,
            self.decision_tasks,
            self.max_vnf_len,
            self.n_uavs + 1,
        )
        state = obs["state"]
        crashed = torch.stack(
            [state[:, uav_id * 10 + 5] > 0.5 for uav_id in range(self.n_uavs)],
            dim=1,
        )
        remaining = torch.ones(
            (batch_size, self.n_uavs), device=features.device, dtype=features.dtype
        )
        previous = torch.zeros_like(remaining)
        encoded_actions = []
        log_probs = []
        entropies = []
        predicted_positions = None
        candidate_positions = None
        if self.relational_placement:
            predicted_positions, candidate_positions = self._relational_geometry(
                obs,
                (
                    mobility_action
                    if self.placement_use_post_mobility_geometry
                    else None
                ),
            )

        for k in range(self.decision_tasks):
            previous = torch.zeros_like(remaining)
            source_position = (
                candidate_positions[:, k] if candidate_positions is not None else None
            )
            for l in range(self.max_vnf_len):
                requirements = context[:, k, l, : self.n_uavs]
                valid = context[:, k, l, self.n_uavs : self.n_uavs + 1]
                network_input = self._placement_network_input(
                    core,
                    requirements,
                    remaining,
                    previous,
                    valid,
                    source_position,
                    predicted_positions,
                    k,
                    l,
                )
                logits = self.placement_net(network_input)
                logits = logits - self.placement_prior_scale * torch.log(
                    requirements.clamp_min(1e-6)
                )

                feasible = (requirements <= remaining + 1e-6) & ~crashed
                fallback = ~crashed
                no_feasible = ~feasible.any(dim=1, keepdim=True)
                feasible = torch.where(no_feasible, fallback, feasible)
                no_live_uav = ~feasible.any(dim=1, keepdim=True)
                feasible = torch.where(no_live_uav, torch.ones_like(feasible), feasible)
                masked_logits = logits.masked_fill(~feasible, -1e4)
                distribution = torch.distributions.Categorical(logits=masked_logits)

                if deterministic:
                    indices = masked_logits.argmax(dim=1)
                    selected = F.one_hot(indices, self.n_uavs).to(features.dtype)
                elif straight_through:
                    selected = F.gumbel_softmax(
                        masked_logits,
                        tau=self.categorical_temperature,
                        hard=True,
                    )
                    indices = selected.argmax(dim=1)
                else:
                    indices = distribution.sample()
                    selected = F.one_hot(indices, self.n_uavs).to(features.dtype)

                active = valid.clamp(0.0, 1.0)
                selected = selected * active
                encoded_actions.append((2.0 * selected - 1.0) * active)
                log_probs.append(distribution.log_prob(indices).unsqueeze(1) * active)
                entropies.append(distribution.entropy().unsqueeze(1) * active)
                remaining = torch.clamp(
                    remaining - selected * requirements, min=0.0
                )
                previous = selected
                if predicted_positions is not None:
                    source_position = torch.sum(
                        selected.unsqueeze(2) * predicted_positions, dim=1
                    )

        actions = torch.cat(encoded_actions, dim=1)
        valid_count = context[..., self.n_uavs].sum(dim=(1, 2)).clamp_min(1.0)
        log_prob = torch.cat(log_probs, dim=1).sum(dim=1, keepdim=True)
        entropy = (
            torch.cat(entropies, dim=1).sum(dim=1, keepdim=True)
            / valid_count.unsqueeze(1)
        )
        return actions, log_prob, entropy

    def placement_bc_loss(
        self,
        obs: PyTorchObs,
        target_actions: torch.Tensor,
        mobility_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced cross entropy for expert categorical placements."""
        features = self.extract_features(obs, self.features_extractor)
        batch_size = features.shape[0]
        core = features[:, : self.core_features_dim]
        context = obs["placement_context"].reshape(
            batch_size,
            self.decision_tasks,
            self.max_vnf_len,
            self.n_uavs + 1,
        )
        targets = target_actions[:, self.mobility_dim :].reshape(
            batch_size, self.decision_tasks, self.max_vnf_len, self.n_uavs
        ).argmax(dim=-1)
        state = obs["state"]
        crashed = torch.stack(
            [state[:, uav_id * 10 + 5] > 0.5 for uav_id in range(self.n_uavs)], dim=1
        )
        remaining = torch.ones_like(context[:, 0, 0, : self.n_uavs])
        previous = torch.zeros_like(remaining)
        losses = []
        predicted_positions = None
        candidate_positions = None
        if self.relational_placement:
            geometry_mobility = (
                target_actions[:, : self.mobility_dim]
                if mobility_action is None
                else mobility_action[:, : self.mobility_dim]
            )
            if not self.placement_use_post_mobility_geometry:
                geometry_mobility = None
            predicted_positions, candidate_positions = self._relational_geometry(
                obs, geometry_mobility
            )
        for k in range(self.decision_tasks):
            previous = torch.zeros_like(remaining)
            source_position = (
                candidate_positions[:, k] if candidate_positions is not None else None
            )
            for l in range(self.max_vnf_len):
                requirements = context[:, k, l, : self.n_uavs]
                valid = context[:, k, l, self.n_uavs] > 0.5
                logits = self.placement_net(
                    self._placement_network_input(
                        core,
                        requirements,
                        remaining,
                        previous,
                        valid.float().unsqueeze(1),
                        source_position,
                        predicted_positions,
                        k,
                        l,
                    )
                )
                logits = logits - self.placement_prior_scale * torch.log(
                    requirements.clamp_min(1e-6)
                )
                feasible = (requirements <= remaining + 1e-6) & ~crashed
                fallback = ~crashed
                feasible = torch.where(
                    ~feasible.any(dim=1, keepdim=True), fallback, feasible
                )
                logits = logits.masked_fill(~feasible, -1e4)
                if valid.any():
                    losses.append(F.cross_entropy(logits[valid], targets[:, k, l][valid]))
                selected = F.one_hot(targets[:, k, l], self.n_uavs).to(features.dtype)
                selected = selected * valid.float().unsqueeze(1)
                remaining = torch.clamp(remaining - selected * requirements, min=0.0)
                previous = selected
                if predicted_positions is not None:
                    source_position = torch.sum(
                        selected.unsqueeze(2) * predicted_positions, dim=1
                    )
        return torch.stack(losses).mean() if losses else core.sum() * 0.0

    def _apply_ray_mask(
        self, pred_x0: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        分段非对称射线遮罩 (Segmented Asymmetric Ray Mask)
        """
        """修正版：解决 Inplace 梯度破坏与 Clamp Tensor 兼容性"""
        B = pred_x0.shape[0]

        # 2. 提取掩码参数 [Batch, n_uavs, 4] -> [L, R, B, T]
        masks_start = self.core_features_dim
        mob_masks = features[:, masks_start : masks_start + self.n_uavs * 4].view(
            B, self.n_uavs, 4
        )

        # --- A. 向量化处理 Mobility (0-7 维) ---
        # Step 1: 将 8 维展成 [Batch, n_uavs, 2] -> 分离出 (vx, vy)
        mobility_part = pred_x0[:, : self.mobility_dim].reshape(
            B, self.n_uavs, 2
        )
        vx = mobility_part[:, :, 0]
        vy = mobility_part[:, :, 1]

        # Step 2: 向量化选择缩放系数
        # X轴：vx >= 0 ? Right(索引1) : Left(索引0)
        mask_x = torch.where(vx >= 0, mob_masks[:, :, 1], mob_masks[:, :, 0])
        # Y轴：vy >= 0 ? Top(索引3) : Bottom(索引2)
        mask_y = torch.where(vy >= 0, mob_masks[:, :, 3], mob_masks[:, :, 2])

        # Keep this transform functional: candidate actions now participate in
        # the categorical Q-loss graph, so view assignment would invalidate
        # tensors saved by autograd.
        masked_mobility = torch.stack([vx * mask_x, vy * mask_y], dim=2).reshape(
            B, self.mobility_dim
        )
        if pred_x0.shape[1] == self.mobility_dim:
            return masked_mobility
        return torch.cat([masked_mobility, pred_x0[:, self.mobility_dim :]], dim=1)

    def _sample_from_noise(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
        inference_steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """修正版：统一 DDIM 数学公式与 None 比较逻辑"""
        batch_size = features.shape[0]
        # DDIM is deterministic conditional on its initial latent. Keep that
        # latent sampled so evaluation measures the learned multimodal policy;
        # callers obtain reproducibility by setting the Torch seed per episode.
        diffusion_dim = (
            self.mobility_dim if self.hybrid_categorical_placement else self.action_dim
        )
        action_t = torch.randn((batch_size, diffusion_dim), device=self.device)
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
            if self.clip_denoised is not None:
                pred_x0 = pred_x0.clamp(-self.clip_denoised, self.clip_denoised)
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

    def _sample_gaussian_mobility(
        self, features: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tanh-Gaussian mobility and return its corrected log density."""
        parameters = self.gaussian_mobility_net(
            features[:, : self.core_features_dim]
        )
        mean, log_std = parameters.chunk(2, dim=1)
        log_std = log_std.clamp(
            self.gaussian_log_std_min, self.gaussian_log_std_max
        )
        distribution = torch.distributions.Normal(mean, log_std.exp())
        unbounded = mean if deterministic else distribution.rsample()
        raw_action = torch.tanh(unbounded)
        # Ray bounds live in normalized action space. Applying them before tanh
        # can violate the bound when an unconstrained Gaussian mean grows large.
        action = self._apply_ray_mask(raw_action, features)
        batch_size = features.shape[0]
        mobility_bounds = features[
            :,
            self.core_features_dim : self.core_features_dim + self.n_uavs * 4,
        ].reshape(batch_size, self.n_uavs, 4)
        raw_xy = raw_action.reshape(batch_size, self.n_uavs, 2)
        scale_x = torch.where(
            raw_xy[:, :, 0] >= 0,
            mobility_bounds[:, :, 1],
            mobility_bounds[:, :, 0],
        )
        scale_y = torch.where(
            raw_xy[:, :, 1] >= 0,
            mobility_bounds[:, :, 3],
            mobility_bounds[:, :, 2],
        )
        ray_scale = torch.stack([scale_x, scale_y], dim=2).reshape(
            batch_size, self.mobility_dim
        )
        component_log_prob = distribution.log_prob(unbounded)
        component_log_prob -= torch.log(1.0 - raw_action.square() + 1e-6)
        movable = ray_scale > 1e-6
        component_log_prob = torch.where(
            movable,
            component_log_prob - torch.log(ray_scale.clamp_min(1e-6)),
            torch.zeros_like(component_log_prob),
        )
        log_prob = component_log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> torch.Tensor:
        features = self.extract_features(obs, self.features_extractor)
        if self.mobility_policy == "gaussian":
            mobility_action, _ = self._sample_gaussian_mobility(
                features, deterministic
            )
        else:
            unbounded_action, _ = self._sample_from_noise(features, deterministic)
            mobility_action = torch.tanh(unbounded_action)
        if not self.hybrid_categorical_placement:
            return mobility_action
        placement_action, _, _ = self.sample_placement(
            obs,
            features=features,
            mobility_action=mobility_action,
            deterministic=deterministic,
        )
        return torch.cat([mobility_action, placement_action], dim=1)

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
        if self.mobility_policy == "gaussian":
            policy_action, log_prob = self._sample_gaussian_mobility(features)
            placement_action, placement_log_prob, _ = self.sample_placement(
                obs,
                features=features,
                mobility_action=policy_action,
                deterministic=False,
            )
            if self.detach_placement_for_rl:
                placement_action = placement_action.detach()
            else:
                log_prob = log_prob + placement_log_prob
            return torch.cat([policy_action, placement_action], dim=1), log_prob

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

        if self.hybrid_categorical_placement:
            placement_action, placement_log_prob, _ = self.sample_placement(
                obs,
                features=features,
                mobility_action=policy_action,
                deterministic=False,
            )
            policy_action = torch.cat([policy_action, placement_action], dim=1)
            log_prob = log_prob + placement_log_prob

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
        # 预测噪声 ε_φ，只取核心特征输入 epsilon_net
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
