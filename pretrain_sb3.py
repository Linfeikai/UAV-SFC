"""
📌 预训练脚本: pretrain_sb3.py
🔥 目标: 使用 Stable Baselines3 的 SAC 和 PPO 实现行为克隆 (BC) 预训练，基于启发式专家策略采集数据。
最后将预训练好的 Actor 权重保存为 .pth 文件，供后续在线微调使用。
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import hydra
from tqdm import tqdm
from omegaconf import DictConfig

# 引入 Stable Baselines3 的标准算法
from stable_baselines3 import SAC, PPO

# 导入你的环境和特征提取器
from core.sfc_env import SFCEnv
from algos.diffusion_extractor import SFCFeaturesExtractor
from test.evalu import smart_heuristic_policy


def get_flat_config(cfg: DictConfig) -> dict:
    flat_config = {}
    if "env" in cfg:
        for k, v in cfg["env"].items():
            flat_config[k] = v
    if "uav" in cfg:
        for k, v in cfg["uav"].items():
            flat_config[f"UAV_{k}"] = v
    if "reward" in cfg:
        for k, v in cfg["reward"].items():
            if k in ["W_ENERGY", "W_CHARGE"]:
                flat_config[k] = v
            else:
                flat_config[f"RWD_{k}"] = v
    return flat_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [基准对齐] 正在启动 SAC & PPO 的行为克隆 (BC) 预训练...")

    # 1. 环境初始化
    flat_config = get_flat_config(cfg)
    env = SFCEnv(config=flat_config)

    # --- 为 SAC 准备的干净参数 ---
    sac_policy_kwargs = dict(
        features_extractor_class=SFCFeaturesExtractor,
        features_extractor_kwargs=dict(
            n_uavs=env.config["NUM_UAVS"],
            m_candidates=env.config["M"],
            grid_res=env.config["GRID_RES"],
        ),
        share_features_extractor=True,
        net_arch=dict(
            pi=[512, 512, 512], qf=[512, 512, 512]
        ),  # SAC 使用 qf (Q-function)
    )

    # --- 为 PPO 准备的干净参数 ---
    ppo_policy_kwargs = dict(
        features_extractor_class=SFCFeaturesExtractor,
        features_extractor_kwargs=dict(
            n_uavs=env.config["NUM_UAVS"],
            m_candidates=env.config["M"],
            grid_res=env.config["GRID_RES"],
        ),
        share_features_extractor=True,
        net_arch=dict(
            pi=[512, 512, 512], vf=[512, 512, 512]
        ),  # PPO 使用 vf (Value-function)
    )

    # 2. 实例化 Agent
    print("🤖 正在初始化 SAC 和 PPO 模型...")
    sac_agent = SAC(
        "MultiInputPolicy", env, policy_kwargs=sac_policy_kwargs, device=device
    )
    ppo_agent = PPO(
        "MultiInputPolicy", env, policy_kwargs=ppo_policy_kwargs, device=device
    )

    # 3. 采集专家数据
    expert_data = []
    print("📥 正在采集启发式专家轨迹 (100 Episodes)...")
    for _ in tqdm(range(100)):
        obs, _ = env.reset()
        done = False
        while not done:
            action = smart_heuristic_policy(env)
            expert_data.append((obs, action))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    num_samples = len(expert_data)
    print(f"✅ 采集完成，总样本数: {num_samples}")

    # 4. 准备优化器
    # 提取 SAC 的 Actor 和 PPO 的 Policy
    sac_actor = sac_agent.policy.actor
    ppo_policy = ppo_agent.policy

    sac_actor.train()
    ppo_policy.train()

    optimizer_sac = torch.optim.AdamW(sac_actor.parameters(), lr=1e-4)
    optimizer_ppo = torch.optim.AdamW(ppo_policy.parameters(), lr=1e-4)

    # 5. 开始同步训练
    print("🧠 开始同步拟合专家动作 (100 Epochs)...")

    for epoch in range(100):
        np.random.shuffle(expert_data)
        sac_epoch_loss = 0
        ppo_epoch_loss = 0

        for i in range(0, num_samples, 256):
            batch = expert_data[i : i + 256]
            if len(batch) < 64:
                continue

            # 构造输入张量
            states = torch.tensor(
                np.array([b[0]["state"] for b in batch]),
                dtype=torch.float32,
                device=device,
            )
            m_bounds = torch.tensor(
                np.array([b[0]["mobility_bounds"] for b in batch]),
                dtype=torch.float32,
                device=device,
            )
            p_limits = torch.tensor(
                np.array([b[0]["pick_limit"] for b in batch]),
                dtype=torch.float32,
                device=device,
            )

            b_obs = {
                "state": states,
                "mobility_bounds": m_bounds,
                "pick_limit": p_limits,
            }

            # 目标动作 (专家物理动作，环境空间已经在 [-1, 1] 之间)
            target_actions = torch.tensor(
                np.array([b[1] for b in batch]), dtype=torch.float32, device=device
            )

            # ==========================================
            # A. 训练 SAC Actor
            # SAC Actor 直接输出 squashed (经过 tanh) 的确定性动作
            # ==========================================
            sac_pred_actions = sac_actor(b_obs, deterministic=True)
            sac_loss = F.mse_loss(sac_pred_actions, target_actions)

            optimizer_sac.zero_grad()
            sac_loss.backward()
            optimizer_sac.step()
            sac_epoch_loss += sac_loss.item()

            # ==========================================
            # B. 训练 PPO Actor
            # PPO Policy 通过 get_distribution 获取分布，模式 (mode) 即均值
            # ==========================================
            ppo_dist = ppo_policy.get_distribution(b_obs)
            ppo_pred_actions = ppo_dist.mode()  # 获取确定性输出
            ppo_loss = F.mse_loss(ppo_pred_actions, target_actions)

            optimizer_ppo.zero_grad()
            ppo_loss.backward()
            optimizer_ppo.step()
            ppo_epoch_loss += ppo_loss.item()

        # 每 10 个 Epoch 打印一次验证结果
        if epoch % 10 == 0:
            sac_actor.eval()
            ppo_policy.eval()
            with torch.no_grad():
                # 用前 10 个样本做符号准确率测试
                test_obs = {
                    "state": states[:10],
                    "mobility_bounds": m_bounds[:10],
                    "pick_limit": p_limits[:10],
                }
                test_targets = target_actions[:10]

                sac_test_preds = sac_actor(test_obs, deterministic=True)
                sac_correct = ((sac_test_preds * test_targets) > 0).float().mean()

                ppo_test_preds = ppo_policy.get_distribution(test_obs).mode()
                ppo_correct = ((ppo_test_preds * test_targets) > 0).float().mean()

                print(
                    f"Epoch {epoch:02d} | SAC Loss: {sac_epoch_loss:.4f} (准确率: {sac_correct * 100:.1f}%) | PPO Loss: {ppo_epoch_loss:.4f} (准确率: {ppo_correct * 100:.1f}%)"
                )

            sac_actor.train()
            ppo_policy.train()

    # 6. 保存基线模型
    sac_save_path = os.path.join(
        hydra.utils.get_original_cwd(), "pretrained_sac_actor.pth"
    )
    ppo_save_path = os.path.join(
        hydra.utils.get_original_cwd(), "pretrained_ppo_actor.pth"
    )

    # SB3 推荐保存整个 policy 的 state_dict
    torch.save(sac_agent.policy.state_dict(), sac_save_path)
    torch.save(ppo_agent.policy.state_dict(), ppo_save_path)

    print(f"\n✅ SAC 基线权重已保存至: {sac_save_path}")
    print(f"✅ PPO 基线权重已保存至: {ppo_save_path}")


if __name__ == "__main__":
    main()
