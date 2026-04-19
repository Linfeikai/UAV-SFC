# =====================================================================================
# 文件名: diffusion_sac_agent.py
# 描述: 实现了基于扩散策略的SAC Agent (MaxEntDP)。
#       核心修改在于 train 方法，使用 QNE 来更新 Actor。
# =====================================================================================

import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from typing import Any, ClassVar, Optional, Type, TypeVar, Union, Dict, List, Tuple

# 修改后 (推荐):
from torch.amp.autocast_mode import autocast
from torch.amp import GradScaler

# 导入我们新定义的 Policy
from .diffusion_sac_policy import DiffusionSACPolicy

# 导入我们新定义的 Actor 和 Critic
from .diffusion_policy_actor import DiffusionPolicyActor
from .diffusion_policy_critic import ContinuousCritic

# 导入标准的 ReplayBuffer 和 SB3 的核心组件
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.buffers import DictReplayBuffer  # 导入 DictReplayBuffer

SelfDiffusionSACAgent = TypeVar("SelfDiffusionSACAgent", bound="DiffusionSACAgent")


class DiffusionSACAgent(OffPolicyAlgorithm):
    """
    基于扩散策略的软演员-评论家 (Soft Actor-Critic) 算法。
    它使用Q加权噪声估计(QNE)来训练一个扩散模型作为策略。
    """

    # --- 指定新的默认策略、Actor和Critic类型 ---
    policy: DiffusionSACPolicy
    actor: DiffusionPolicyActor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    # 定义策略名称到类的映射
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "DiffusionSACPolicy": DiffusionSACPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DiffusionSACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-5,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10000,
        batch_size: int = 256,
        tau: float = 0.0005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        # --- 移除了 HybridSAC 特有的、不再需要的参数 ---
        replay_buffer_class: Optional[Type[ReplayBuffer]] = DictReplayBuffer,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = 0.0,
        target_update_interval: int = 1,
        # target_entropy: Union[str, float] = "auto",
        # --- 新增的扩散模型特定参数 ---
        qne_k_samples: int = 32,  # QNE中的K值，即"头脑风暴"的样本数
        policy_kwargs: Optional[Dict[str, Any]] = None,
        qne_temperature: float = 0.5,  # QNE的温度参数，用于控制Softmax的平滑度
        max_grad_norm: float = 1.0,  # <-- 确保有这个参数
        # --- 其他标准参数 ---
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        self.qne_k_samples = qne_k_samples

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=None,  # SAC不使用外部噪声
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=100,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            supported_action_spaces=(spaces.Box,),  # <-- 关键：现在只支持Box空间
            support_multi_env=True,
        )

        # self.target_entropy = target_entropy 在MAX_ENT中不需要目标熵了
        self.log_ent_coef: Optional[th.Tensor] = None
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.qne_temperature = qne_temperature
        self.max_grad_norm = max_grad_norm
        self.critic_updates_per_step = 4  # 现在没有使用到
        self.qne_entropy_lambda = 0.001

        if _init_setup_model:
            self._setup_model()

        # === 在这里添加AMP相关的初始化 ===
        # 仅当设备为CUDA时才创建Scaler
        self.scaler = GradScaler("cuda") if self.device.type == "cuda" else None

    def _setup_model(self) -> None:
        # P-TODO: 移除 "use_sde"，因为它不被 DiffusionSACPolicy 支持
        if "use_sde" in self.policy_kwargs:
            self.policy_kwargs.pop("use_sde")

        super()._setup_model()
        self._create_aliases()
        # 【修改】移除所有关于 target_entropy 的计算
        # 【修改】移除所有关于 log_ent_coef 和 ent_coef_optimizer 的创建

        # 【新增】将 ent_coef 转换为一个固定的张量
        if isinstance(self.ent_coef, str):
            # 如果是 "auto"，给一个默认值。更好的做法是在__init__中就处理好
            # 为简单起见，这里我们假设它总是一个 float
            raise ValueError("ent_coef must be a float for MaxEntDP, not 'auto'")

        # 将 ent_coef (β) 设置为一个不可训练的张量
        self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    # === NEW: 显式的动作缩放/反缩放，确保策略空间([-1,1])与环境空间一致 ===
    # 它的核心作用是将环境输出的原始物理动作映射到神经网络喜欢的 [-1, 1] 标准空间。
    def _to_policy_space(self, env_actions: th.Tensor) -> th.Tensor:
        assert isinstance(self.action_space, spaces.Box)
        low = th.as_tensor(self.action_space.low, device=self.device)
        high = th.as_tensor(self.action_space.high, device=self.device)
        return 2.0 * (env_actions - low) / (high - low) - 1.0

    def _to_env_space(self, policy_actions: th.Tensor) -> th.Tensor:
        assert isinstance(self.action_space, spaces.Box)
        low = th.as_tensor(self.action_space.low, device=self.device)
        high = th.as_tensor(self.action_space.high, device=self.device)
        return (policy_actions + 1.0) * 0.5 * (high - low) + low

    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        """
        训练循环。这是算法的核心，Actor的更新逻辑将在这里被彻底改变。
        """
        self.policy.set_training_mode(True)
        optimizers = [self.critic.optimizer, self.actor.optimizer]
        # if self.ent_coef_optimizer is not None:
        #     optimizers.append(self.ent_coef_optimizer)

        self._update_learning_rate(optimizers)

        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            assert self.replay_buffer is not None, (
                "Replay buffer must be initialized before training."
            )
            # 1. 从Replay Buffer采样
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            obs = replay_data.observations

            # if isinstance(obs, dict):
            #     for v in obs.values():
            #         if th.isnan(v).any() or th.isinf(v).any():
            #             raise ValueError("NaN or Inf detected in observations!")
            # else:
            #     if th.isnan(obs).any() or th.isinf(obs).any():
            #         raise ValueError("NaN or Inf detected in observations!")
            # if (
            #     th.isinf(replay_data.actions).any()
            #     or th.isnan(replay_data.actions).any()
            # ):
            #     raise ValueError("NaN or Inf detected in actions!")
            # if (
            #     th.isinf(replay_data.rewards).any()
            #     or th.isnan(replay_data.rewards).any()
            # ):
            #     raise ValueError("NaN or Inf detected in rewards!")

            # 【修改】直接使用 self.ent_coef_tensor，不再需要复杂的if-else
            ent_coef_tensor = self.ent_coef_tensor

            with autocast(
                device_type=self.device.type,
                dtype=th.float16,
                enabled=(self.scaler is not None),
            ):
                # 2. --- Critic Loss 计算 (与标准SAC非常相似) ---
                with th.no_grad():
                    # 使用 Actor 生成下一状态的动作及其对数概率
                    # 大小：[Batch_Size, Action_Dim]
                    next_actions, next_log_prob = self.actor.action_log_prob(
                        replay_data.next_observations
                    )

                    # -------------------------------------------------------------
                    # 【核心插入】：Wandb 动作流形与越界监控 (白盒化诊断)
                    # -------------------------------------------------------------
                    N_UAV = getattr(self.env, "N", 4) if hasattr(self.env, "N") else 4
                    K_PICK = getattr(self.env, "K", 2) if hasattr(self.env, "K") else 2

                    dim_mob = N_UAV * 2
                    dim_pick = K_PICK

                    # 切片分解动作 (注意此时 next_actions 已经是 [-1, 1] 范围)
                    mob_acts = next_actions[:, :dim_mob]
                    pick_acts = next_actions[:, dim_mob : dim_mob + dim_pick]
                    place_acts = next_actions[:, dim_mob + dim_pick :]

                    # 计算越界率
                    mob_hit_rate = (mob_acts.abs() > 0.95).float().mean().item()
                    pick_hit_rate = (pick_acts.abs() > 0.95).float().mean().item()
                    place_hit_rate = (place_acts.abs() > 0.95).float().mean().item()

                    # 💡 直接使用 SB3 的 logger Wandb 会自动把它们抓取上去
                    self.logger.record("Action_Boundary/Mobility_HitRate", mob_hit_rate)
                    self.logger.record("Action_Boundary/Pick_HitRate", pick_hit_rate)
                    self.logger.record("Action_Boundary/Place_HitRate", place_hit_rate)

                    # 2. 直方图属于 Wandb 独有高级对象，必须单独发送
                    import wandb

                    if wandb.run is not None and self._n_updates % 500 == 0:
                        wandb.log(
                            {
                                "Action_Dist/1_Mobility": wandb.Histogram(
                                    mob_acts.cpu().numpy()
                                ),
                                "Action_Dist/2_Pick": wandb.Histogram(
                                    pick_acts.cpu().numpy()
                                ),
                                "Action_Dist/3_Place": wandb.Histogram(
                                    place_acts.cpu().numpy()
                                ),
                                "global_step": self.num_timesteps,
                            },
                            step=self.num_timesteps,
                        )
                    # -------------------------------------------------------------

                    # 用目标Critic网络评估下一状态-动作对的Q值
                    qf_next_target = th.cat(
                        self.critic_target(replay_data.next_observations, next_actions),
                        dim=1,
                    )
                    min_qf_next_target, _ = th.min(qf_next_target, dim=1, keepdim=True)

                    # 在这里，可以诊断目标Q网络给出的分数
                    self.logger.record(
                        "train/qf_next_target_mean",
                        th.mean(min_qf_next_target).item(),
                    )

                    # 加上熵项，计算最终的目标Q值
                    # 现在是0.0，所以这个项不会对结果产生影响
                    entropy_bonus = ent_coef_tensor * next_log_prob

                    # 从目标Q值中减去熵项
                    min_qf_next_target -= entropy_bonus

                    next_q_value = (
                        replay_data.rewards
                        + (1 - replay_data.dones) * self.gamma * min_qf_next_target
                    )
                    self.logger.record(
                        "debug/target_q_value_mean", th.mean(next_q_value).item()
                    )

                # 计算当前Critic的Q值
                qf_values = self.critic(replay_data.observations, replay_data.actions)
                # 计算 TD 误差
                # 注意：qf_values 返回两个 Q 值（例如，SAC 中有两个 Critic 网络），
                # 你需要选择一个来计算 TD 误差，或者计算每个 Q 值的 TD 误差的平均值或范数。
                # 假设你希望记录第一个 Critic 的 TD 误差，或者所有 Critic 误差的平均值
                td_error_per_qf = [
                    q - next_q_value for q in qf_values
                ]  # 这是一个列表，包含每个Q头的TD误差

                # 取所有 Critic 头的 TD 误差的平均绝对值作为示例：
                # 先将所有批次样本和所有Q头的误差压平
                td_errors_flat = th.cat([error.flatten() for error in td_error_per_qf])
                # 计算平均绝对误差
                mean_abs_td_error = th.mean(th.abs(td_errors_flat)).item()

                # 记录 TD 误差
                self.logger.record("debug/td_error_mean_abs", mean_abs_td_error)

                # 计算Critic的MSE损失
                critic_loss = 0.5 * sum(F.mse_loss(q, next_q_value) for q in qf_values)
                assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(float(critic_loss))

            # 优化Critic
            self.critic.optimizer.zero_grad()
            # 【修改】使用 scaler.scale() 来缩放loss
            if self.scaler is not None:
                self.scaler.scale(critic_loss).backward()
                self.scaler.unscale_(self.critic.optimizer)

            else:
                critic_loss.backward()
            total_norm = th.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm or 1.0
            )
            self.logger.record("train/critic_grad_norm_raw", total_norm.item())

            if self.scaler is not None:
                self.scaler.step(self.critic.optimizer)
            else:
                self.critic.optimizer.step()

            # --- 3. Actor Loss 计算 (QNE核心逻辑) ---
            if (self._n_updates + 1) % 2 == 0:
                # (a) 准备计算Actor Loss所需的数据
                # 从replay_data中获取干净的动作`a` (即 `replay_data.actions`)
                # 和对应的状态`s` (即 `replay_data.observations`)
                with autocast(
                    device_type=self.device.type,
                    dtype=th.float16,
                    enabled=(self.scaler is not None),
                ):
                    EPS = 1e-6

                    policy_actions_from_buffer = self._to_policy_space(
                        replay_data.actions
                    )
                    policy_actions_from_buffer = policy_actions_from_buffer.clamp(
                        -1.0 + EPS, 1.0 - EPS
                    )
                    unbounded_clean_actions = th.atanh(policy_actions_from_buffer)

                    states_from_buffer = replay_data.observations

                    # (b) 随机采样扩散时间步t和真实噪声epsilon
                    # 随机生成大小为 (batch_size, 1) 的时间步t
                    t = th.randint(0, self.actor.T, (batch_size,), device=self.device)
                    epsilon = th.randn_like(unbounded_clean_actions)

                    # (c) 根据公式创建加噪动作 a_t
                    sqrt_alpha_bar = self.actor.sqrt_alphas_cumprod.gather(
                        0, t
                    ).reshape(-1, 1)
                    sqrt_one_minus_alpha_bar = (
                        self.actor.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(
                            -1, 1
                        )
                    )
                    noisy_actions_t = (
                        sqrt_alpha_bar * unbounded_clean_actions
                        + sqrt_one_minus_alpha_bar * epsilon
                    )
                    # 加噪动作=从buffer里采样的干净动作*一个系数+噪声*一个系数

                    # (d) 通过QNE计算"目标噪声"epsilon*
                    #    i. "头脑风暴" K 个候选动作
                    k_candidate_actions = []
                    k_noises = th.randn(
                        batch_size,
                        self.qne_k_samples,
                        self.actor.action_dim,
                        device=self.device,
                    )  # shape: [B, K, A_dim]

                    # 使用去噪公式反向生成K个候选的干净动作
                    # 这是一个批处理操作，效率很高
                    a_t_expanded = noisy_actions_t.unsqueeze(1).expand(
                        -1, self.qne_k_samples, -1
                    )  # [B, 1, A_dim] -> [B, K, A_dim]
                    sqrt_alpha_bar_exp = sqrt_alpha_bar.unsqueeze(1).expand(
                        -1, self.qne_k_samples, -1
                    )
                    sqrt_one_minus_alpha_bar_exp = sqrt_one_minus_alpha_bar.unsqueeze(
                        1
                    ).expand(-1, self.qne_k_samples, -1)
                    k_candidate_actions = (
                        a_t_expanded - sqrt_one_minus_alpha_bar_exp * k_noises
                    ) / sqrt_alpha_bar_exp  # [B, K, A_dim]
                    policy_k_candidate_actions = th.tanh(k_candidate_actions)
                    self.logger.record(
                        "debug/qne_candidate_actions_mean",
                        th.mean(policy_k_candidate_actions).item(),
                    )
                    self.logger.record(
                        "debug/qne_candidate_actions_std",
                        th.std(policy_k_candidate_actions).item(),
                    )

                    acts_reshaped = policy_k_candidate_actions.reshape(
                        batch_size * self.qne_k_samples, -1
                    )

                    #    ii. Critic打分
                    if isinstance(states_from_buffer, dict):
                        # 如果是字典（DictReplayBuffer），对每个键值对分别展开
                        states_expanded = {
                            key: val.unsqueeze(1).expand(
                                -1, self.qne_k_samples, *([-1] * (val.dim() - 1))
                            )
                            for key, val in states_from_buffer.items()
                        }
                        # 变形为 [B*K, Dim]
                        states_reshaped = {
                            key: val.reshape(
                                batch_size * self.qne_k_samples, *val.shape[2:]
                            )
                            for key, val in states_expanded.items()
                        }
                    else:
                        # 如果是普通张量（ReplayBuffer）
                        states_expanded = states_from_buffer.unsqueeze(1).expand(
                            -1, self.qne_k_samples, -1
                        )
                        states_reshaped = states_expanded.reshape(
                            batch_size * self.qne_k_samples, -1
                        )

                    q1 = self.critic.q1_forward(states_reshaped, acts_reshaped)
                    q2 = self.critic.q2_forward(states_reshaped, acts_reshaped)
                    q_values_k = th.min(q1, q2).reshape(
                        batch_size, self.qne_k_samples, 1
                    )
                    with th.no_grad():  # 在no_grad环境下计算，以防影响梯度
                        # 计算这 K*B 个Q值的均值和标准差
                        q_values_k_mean = th.mean(q_values_k).item()
                        q_values_k_std = th.std(q_values_k).item()

                        # 使用 SB3 的 logger 记录下来
                        # 可以在 TensorBoard 中看到名为 "train/qne_q_mean" 和 "train/qne_q_std" 的图表
                        self.logger.record("train/qne_q_mean", q_values_k_mean)
                        self.logger.record("train/qne_q_std", q_values_k_std)

                        # 计算当前批次中所有K个候选Q值的均值和标准差
                        # [B, K, 1] -> 按 K 维做均值/方差
                        q_mean = q_values_k.mean(dim=1, keepdim=True)
                        q_std = (
                            q_values_k.std(dim=1, keepdim=True, unbiased=False) + 1e-6
                        )
                        normalized_q_values = (
                            q_values_k - q_mean
                        ) / q_std  # 形状仍是 [B, K, 1]

                        # 在TensorBoard中监控归一化后的Q值，它们应该稳定得多
                        self.logger.record(
                            "train/qne_q_normalized_mean",
                            th.mean(normalized_q_values).item(),
                        )
                        self.logger.record(
                            "train/qne_q_normalized_std",
                            th.std(normalized_q_values).item(),
                        )

                        # 3. 在归一化的Q值上应用温度系数和Softmax
                        softmax_weights = F.softmax(
                            normalized_q_values / self.qne_temperature, dim=1
                        )
                        self.logger.record(
                            "debug/softmax_weights_std", th.std(softmax_weights).item()
                        )
                    # Target_Noise 是对那K个随机噪声的加权和，可以诊断Softmax的输出
                    target_noise = th.sum(
                        softmax_weights * k_noises, dim=1
                    )  # [B, A_dim]
                    self.logger.record(
                        "debug/target_noise_std", th.std(target_noise).item()
                    )
                    # --- 添加结束 ---

                    # (e) Actor进行噪声预测
                    # 1. 提取总特征 [Batch, 273]
                    features_from_buffer = self.actor.extract_features(
                        states_from_buffer, self.actor.features_extractor
                    )
                    # 2. 【必须切片】：只取前 core_features_dim (256) 维进大脑
                    core_feats = features_from_buffer[:, : self.actor.core_features_dim]
                    # 3. 噪声预测
                    predicted_noise = self.actor._epsilon_net(
                        core_feats, noisy_actions_t, t
                    )

                    # (f) 计算最终的Actor Loss (MSE)
                    actor_loss = F.mse_loss(
                        predicted_noise, target_noise.detach()
                    )  # target_noise不反向传播
                    if self.qne_entropy_lambda > 0:
                        entropy = (
                            -(softmax_weights * (softmax_weights + 1e-8).log())
                            .sum(dim=1)
                            .mean()
                        )
                        actor_loss += self.qne_entropy_lambda * entropy
                        self.logger.record("debug/qne_entropy", entropy.item())

                actor_losses.append(actor_loss.item())

                # --- 4. 优化Actor ---
                self.actor.optimizer.zero_grad()
                # 使用 scaler
                if self.scaler is not None:
                    self.scaler.scale(actor_loss).backward()
                    self.scaler.unscale_(self.actor.optimizer)

                else:
                    actor_loss.backward()
                # 梯度裁剪 + 获取真实梯度范数（一行搞定）
                total_norm = th.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=self.max_grad_norm or 1.0,  # 防御 None
                )
                self.logger.record("train/actor_grad_norm_raw", total_norm.item())
                if self.scaler is not None:
                    self.scaler.step(self.actor.optimizer)
                else:
                    self.actor.optimizer.step()

            # 【新增】在所有优化器步骤之后，更新scaler
            if self.scaler is not None:
                self.scaler.update()

            # --- 6. 更新目标网络 ---
            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(), self.critic_target.parameters(), self.tau
                )

            self._n_updates += 1
        actor_lr = self.actor.optimizer.param_groups[0]["lr"]
        critic_lr = self.critic.optimizer.param_groups[0]["lr"]
        # 使用 logger.record 将它们记录下来，以便在 WandB 中显示
        self.logger.record("train/actor_lr", actor_lr)
        self.logger.record("train/critic_lr", critic_lr)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        # self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        # 【新增】记录 ent_coef (β) 的固定值，方便在实验中追踪
        self.logger.record("train/ent_coef", self.ent_coef_tensor.item())

    def learn(self: SelfDiffusionSACAgent, **kwargs) -> SelfDiffusionSACAgent:
        # 重写learn方法，主要是为了类型提示
        return super().learn(**kwargs)

    # _excluded_save_params 和 _get_torch_save_params 可以从父类继承或根据需要微调
    # 由于我们现在使用标准的Actor/Critic，父类的实现可能已经足够

    def _update_learning_rate(self, optimizers: List[th.optim.Optimizer]) -> None:
        """
        重写此方法以支持 Actor 和 Critic 的独立学习率。
        """
        # 1. 计算当前的训练进度
        progress = self._current_progress_remaining

        # 2. 从 Policy 中获取各自的 schedule，并计算新的学习率
        new_actor_lr = self.policy.lr_actor_schedule(progress)
        new_critic_lr = self.policy.lr_critic_schedule(progress)

        # 3. 手动为 Actor 和 Critic 的优化器设置新的学习率
        self.actor.optimizer.param_groups[0]["lr"] = new_actor_lr
        self.critic.optimizer.param_groups[0]["lr"] = new_critic_lr

        # 4. (可选但推荐) 同时更新熵系数优化器的学习率
        # 它通常使用 Agent 的主学习率
        if self.ent_coef_optimizer is not None:
            new_ent_lr = self.lr_schedule(progress)
            self.ent_coef_optimizer.param_groups[0]["lr"] = new_ent_lr
