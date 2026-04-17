"""
📌 测试脚本: test_trained_actor.py
🔥 目标: 加载之前预训练好的 Actor 权重，直接在环境中评估其性能表现。这个脚本不进行任何训练，
只是纯粹的评测，帮助我们验证预训练权重的有效性。"""

import os
import torch
import hydra
import numpy as np
from omegaconf import DictConfig

from stable_baselines3 import SAC, PPO

# 导入你的核心组件
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


def run_evaluate(
    env,
    policy_fn=None,
    agent=None,
    agent_type="diffusion",
    device=None,
    policy_name="Unknown",
    num_episodes=5,
):
    """支持三大流派(Heuristic, Diffusion, SB3)的通用评估函数"""
    print(f"🚀 正在测试: [{policy_name}] ...")

    ep_rewards = []
    ep_success_rates = []
    ep_completed = []
    crash_count = 0

    for i in range(num_episodes):
        current_seed = 42 + i
        obs, _ = env.reset(seed=current_seed)
        done = False

        ep_total_reward = 0.0
        ep_total_completed = 0
        ep_total_generated = 0
        ep_crashed = False

        while not done:
            if agent_type == "heuristic":
                action = policy_fn(env)
            elif agent_type == "diffusion":
                with torch.no_grad():
                    b_obs = {
                        "state": torch.tensor(
                            obs["state"], dtype=torch.float32, device=device
                        ).unsqueeze(0),
                        "mobility_bounds": torch.tensor(
                            obs["mobility_bounds"], dtype=torch.float32, device=device
                        ).unsqueeze(0),
                        "pick_limit": torch.tensor(
                            obs["pick_limit"], dtype=torch.float32, device=device
                        ).unsqueeze(0),
                    }
                    policy_action = agent.actor(b_obs, deterministic=True)
                    action = agent._to_env_space(policy_action).squeeze(0).cpu().numpy()
            elif agent_type == "sb3":
                # SB3 的 predict 可以直接接受 numpy 的 dict obs
                action, _ = agent.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_total_reward += reward
            ep_total_completed += info.get("completed_count", 0)
            ep_total_generated += info.get("total_available", 0)
            if info.get("perf/crashed", 0) > 0:
                ep_crashed = True

        # 计算该 Episode 成功率
        ep_success_rate = (
            ep_total_completed / max(1, info.get("total_available", 1))
        ) * 100

        ep_rewards.append(ep_total_reward)
        ep_success_rates.append(ep_success_rate)
        ep_completed.append(ep_total_completed)
        if ep_crashed:
            crash_count += 1

    return {
        "name": policy_name,
        "avg_reward": np.mean(ep_rewards),
        "std_reward": np.std(ep_rewards),
        "avg_success": np.mean(ep_success_rates),
        "std_success": np.std(ep_success_rates),
        "avg_completed": np.mean(ep_completed),
        "std_completed": np.std(ep_completed),
        "crash_count": crash_count,
        "num_episodes": num_episodes,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 [科研级 七方基准对比] 启动 (Device: {device})...")

    NUM_EVAL_EPISODES = 5
    flat_config = get_flat_config(cfg)
    flat_config["IS_OVERFIT_TEST"] = False
    env = SFCEnv(config=flat_config)

    all_stats = []

    # ================= 1. Heuristic =================
    all_stats.append(
        run_evaluate(
            env,
            policy_fn=smart_heuristic_policy,
            agent_type="heuristic",
            policy_name="Heuristic (专家)",
            num_episodes=NUM_EVAL_EPISODES,
        )
    )

    # ================= 架构参数准备 =================
    base_kwargs = dict(
        features_extractor_class=SFCFeaturesExtractor,
        features_extractor_kwargs=dict(
            n_uavs=env.config["NUM_UAVS"],
            m_candidates=env.config["M"],
            grid_res=env.config["GRID_RES"],
        ),
        share_features_extractor=True,
    )
    diff_kwargs = {
        **base_kwargs,
        "n_uavs": env.config["NUM_UAVS"],
        "m_candidates": env.config["M"],
        "core_features_dim": 256,
        "T": 20,
        "net_arch": [512, 512, 512],
    }
    sac_kwargs = {
        **base_kwargs,
        "net_arch": dict(pi=[512, 512, 512], qf=[512, 512, 512]),
    }
    ppo_kwargs = {
        **base_kwargs,
        "net_arch": dict(pi=[512, 512, 512], vf=[512, 512, 512]),
    }

    cwd = hydra.utils.get_original_cwd()

    # ================= 2 & 3. Diffusion =================
    # 纯随机
    diff_rand = DiffusionSACAgent(
        policy=DiffusionSACPolicy, env=env, policy_kwargs=diff_kwargs, device=device
    )
    diff_rand.actor.eval()
    all_stats.append(
        run_evaluate(
            env,
            agent=diff_rand,
            agent_type="diffusion",
            device=device,
            policy_name="Diffusion (纯随机)",
            num_episodes=NUM_EVAL_EPISODES,
        )
    )

    # 预训练
    diff_pre = DiffusionSACAgent(
        policy=DiffusionSACPolicy, env=env, policy_kwargs=diff_kwargs, device=device
    )
    diff_path = os.path.join(cwd, "pretrained_actor.pth")
    if os.path.exists(diff_path):
        diff_pre.actor.load_state_dict(torch.load(diff_path, map_location=device))
        diff_pre.actor.eval()
        all_stats.append(
            run_evaluate(
                env,
                agent=diff_pre,
                agent_type="diffusion",
                device=device,
                policy_name="Diffusion (预训练)",
                num_episodes=NUM_EVAL_EPISODES,
            )
        )

    # ================= 4 & 5. SAC =================
    # 纯随机
    sac_rand = SAC("MultiInputPolicy", env, policy_kwargs=sac_kwargs, device=device)
    all_stats.append(
        run_evaluate(
            env,
            agent=sac_rand,
            agent_type="sb3",
            policy_name="SAC (纯随机)",
            num_episodes=NUM_EVAL_EPISODES,
        )
    )

    # 预训练
    sac_pre = SAC("MultiInputPolicy", env, policy_kwargs=sac_kwargs, device=device)
    sac_path = os.path.join(cwd, "pretrained_sac_actor.pth")
    if os.path.exists(sac_path):
        sac_pre.policy.load_state_dict(torch.load(sac_path, map_location=device))
        all_stats.append(
            run_evaluate(
                env,
                agent=sac_pre,
                agent_type="sb3",
                policy_name="SAC (预训练)",
                num_episodes=NUM_EVAL_EPISODES,
            )
        )

    # ================= 6 & 7. PPO =================
    # 纯随机
    ppo_rand = PPO("MultiInputPolicy", env, policy_kwargs=ppo_kwargs, device=device)
    all_stats.append(
        run_evaluate(
            env,
            agent=ppo_rand,
            agent_type="sb3",
            policy_name="PPO (纯随机)",
            num_episodes=NUM_EVAL_EPISODES,
        )
    )

    # 预训练
    ppo_pre = PPO("MultiInputPolicy", env, policy_kwargs=ppo_kwargs, device=device)
    ppo_path = os.path.join(cwd, "pretrained_ppo_actor.pth")
    if os.path.exists(ppo_path):
        ppo_pre.policy.load_state_dict(torch.load(ppo_path, map_location=device))
        all_stats.append(
            run_evaluate(
                env,
                agent=ppo_pre,
                agent_type="sb3",
                policy_name="PPO (预训练)",
                num_episodes=NUM_EVAL_EPISODES,
            )
        )

    # ================= 生成科研表格 =================
    print("\n" + "=" * 105)
    print(
        f"🏆 终极大乱斗：七方基准性能对比报告 (共测 {NUM_EVAL_EPISODES} 轮, 固定序列种子)"
    )
    print("=" * 105)
    print(
        f"{'策略模型':<22} | {'总奖励 (Reward)':<22} | {'完成任务数':<18} | {'成功率 (%)':<18} | {'坠毁/总轮数':<12}"
    )
    print("-" * 105)

    def fmt(s, key):
        return f"{s['avg_' + key]:.2f} ± {s['std_' + key]:.2f}"

    for s in all_stats:
        crash_str = f"{s['crash_count']} / {s['num_episodes']}"
        # 加点视觉区分：如果是预训练模型，稍微标记一下
        name_str = (
            f"🌟 {s['name']}"
            if "预训练" in s["name"] or "专家" in s["name"]
            else f"   {s['name']}"
        )
        print(
            f"{name_str:<22} | {fmt(s, 'reward'):<22} | {fmt(s, 'completed'):<18} | {fmt(s, 'success'):<18} | {crash_str:<12}"
        )
    print("=" * 105)


if __name__ == "__main__":
    main()
