import os
import torch
import torch.nn.functional as F
import numpy as np
import hydra
from tqdm import tqdm
from omegaconf import DictConfig

from core.sfc_env import SFCEnv
from algos.diffusion_sac_agent import DiffusionSACAgent
from algos.diffusion_sac_policy import DiffusionSACPolicy
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
    print(f"🚀 [终极排查模式] 正在启动...")

    flat_config = get_flat_config(cfg)
    env = SFCEnv(config=flat_config)

    # 打印动作空间信息进行核对
    print(
        f"DEBUG: Action Space Low: {env.action_space.low[:2]}... High: {env.action_space.high[:2]}..."
    )

    policy_kwargs = dict(
        features_extractor_class=SFCFeaturesExtractor,
        features_extractor_kwargs=dict(
            n_uavs=env.config["NUM_UAVS"],
            m_candidates=env.config["M"],
            grid_res=env.config["GRID_RES"],
        ),
        n_uavs=env.config["NUM_UAVS"],
        m_candidates=env.config["M"],
        decision_tasks=env.config["K"],
        core_features_dim=256,
        share_features_extractor=True,
        T=20,
        net_arch=[512, 512, 512],
    )

    agent = DiffusionSACAgent(
        policy=DiffusionSACPolicy, env=env, policy_kwargs=policy_kwargs, device=device
    )

    # 1. 采集数据并强制校验
    expert_data = []
    print("📥 采集专家轨迹中...")
    for _ in tqdm(range(100)):
        obs, _ = env.reset()
        done = False
        while not done:
            action = smart_heuristic_policy(env)  # 拿到原始物理动作
            expert_data.append((obs, action))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    # 2. 核心修正：手动执行归一化校验
    # 你的环境 action_space 是 [-1, 1]，但你的启发式函数返回的是物理坐标！
    # 这里我们模拟 agent._to_policy_space 的逻辑，并打印前5条检查
    def manual_normalize(act):
        # 假设启发式返回的是环境物理动作，我们需要将其缩放到 [-1, 1]
        # 注意：这里的 low/high 必须手动对应你的真实物理边界(如 0-500)
        # 如果你的 action_space 定义已经是 [-1, 1]，说明你的启发式函数写错了，它应该输出归一化后的值
        return act  # 暂时保持，等打印结果后再改

    # 3. 训练循环
    optimizer = torch.optim.AdamW(agent.actor.parameters(), lr=1e-4)
    num_samples = len(expert_data)

    for epoch in range(100):
        np.random.shuffle(expert_data)
        epoch_loss = 0

        for i in range(0, num_samples, 256):
            batch = expert_data[i : i + 256]
            if len(batch) < 64:
                continue

            # 构造输入
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

            # 提取动作并进行 atanh 预处理
            raw_acts = np.array([b[1] for b in batch])
            acts_norm = agent._to_policy_space(
                torch.tensor(raw_acts, dtype=torch.float32, device=device)
            )

            # 💡 诊断打印：如果这里的数值不在 [-1, 1]，说明 low/high 设置全错了
            if epoch == 0 and i == 0:
                print(
                    f"DEBUG: 专家动作(PolicySpace) 首行前5维: {acts_norm[0, :5].cpu().numpy()}"
                )
                print(
                    f"DEBUG: 专家动作(PolicySpace) 最大值: {acts_norm.max():.2f}, 最小值: {acts_norm.min():.2f}"
                )

            unbounded_act = torch.atanh(acts_norm.clamp(-0.99, 0.99))

            # 扩散去噪学习
            t = torch.randint(0, agent.actor.T, (len(batch),), device=device)
            noise = torch.randn_like(unbounded_act)
            sqrt_alpha_bar = agent.actor.sqrt_alphas_cumprod.gather(0, t).view(-1, 1)
            sqrt_one_minus_alpha_bar = agent.actor.sqrt_one_minus_alphas_cumprod.gather(
                0, t
            ).view(-1, 1)
            noisy_act = (
                sqrt_alpha_bar * unbounded_act + sqrt_one_minus_alpha_bar * noise
            )

            # 预测
            b_obs = {
                "state": states,
                "mobility_bounds": m_bounds,
                "pick_limit": p_limits,
            }
            feats = agent.actor.extract_features(b_obs, agent.actor.features_extractor)
            pred_noise = agent.actor._epsilon_net(feats[:, :256], noisy_act, t)

            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 验证
        if epoch % 5 == 0:
            with torch.no_grad():
                print("\n🔍 [一致性诊断] 正在核对专家决策与状态是否匹配...")
                # 找到一个电量低或有任务的样本进行肉眼比对
                found_test_case = False
                for idx in range(min(50, len(batch))):
                    state_vec = batch[idx][0]["state"]
                    # 假设：第 5 维是 UAV0 的电池 (索引 4)
                    # 假设：第 1, 2 维是 UAV0 的 x, y 坐标
                    uav0_batt = state_vec[4]
                    uav0_loc = state_vec[:2]

                    # 专家动作的前 2 维 (UAV0 的移动指令)
                    expert_move = raw_acts[idx, :2]

                    if uav0_batt < 0.4:  # 找一个低电量的样本
                        charger_dir = np.array([250.0, 250.0]) / 500.0 - uav0_loc
                        print(f"  -> 检测到低电量样本 (Batt: {uav0_batt:.2f})")
                        print(f"  -> 专家移动指令: {expert_move}")
                        print(f"  -> 理论充电方向: {charger_dir}")
                        # 如果专家移动方向和理论充电方向相反，说明采集时的 obs 和 action 记反了！
                        found_test_case = True
                        break
            if not found_test_case:
                print("  -> 本批次未发现低电量样本，跳过诊断。")
            agent.actor.eval()
            with torch.no_grad():
                test_obs = {
                    "state": states[:5],
                    "mobility_bounds": m_bounds[:5],
                    "pick_limit": p_limits[:5],
                }
                pred_act = agent.actor(test_obs, deterministic=True)
                true_act = acts_norm[:5]
                correct = ((pred_act * true_act) > 0).float().mean()
                print(
                    f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | 方向准确率: {correct * 100:.2f}%"
                )
            agent.actor.train()


if __name__ == "__main__":
    main()
