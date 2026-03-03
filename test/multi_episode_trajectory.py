import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import sys
import os
from omegaconf import OmegaConf


# 将当前文件的父目录（即项目根目录）加入到搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在可以正常导入了
from core.sfc_env import SFCEnv


# ==========================================
# 配置拍扁工具 (同步 main.py 逻辑)
# ==========================================
def flatten_hydra_cfg(cfg):
    flat = {}
    for k, v in cfg.env.items():
        flat[k] = v
    for k, v in cfg.uav.items():
        flat[f"UAV_{k}"] = v
    for k, v in cfg.reward.items():
        if k in ["W_ENERGY", "W_CHARGE"]:
            flat[k] = v
        else:
            flat[f"RWD_{k}"] = v
    return flat


# 这个脚本用于运行多个 episode，并绘制 UAV 在每个 episode 中的移动轨迹。
# 这个是用于展示单独一个模型在多个 episode 下的轨迹表现差异。
def run_multi_eval(run_folder_path, n_episodes=10):
    # 1. 拼接配置文件和模型文件的完整路径
    cfg_path = os.path.join(run_folder_path, ".hydra/config.yaml")
    model_path = os.path.join(run_folder_path, "SAC_final_model.zip")

    # 加载配置与模型
    cfg = OmegaConf.load(cfg_path)
    flat_cfg = flatten_hydra_cfg(cfg)

    env = SFCEnv(config=flat_cfg)
    model = SAC.load(model_path)

    # 2. 设置绘图网格 (假设 n=10, 2行5列)
    cols = 5
    rows = (n_episodes + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(25, 5 * rows))
    axs = axs.flatten() if n_episodes > 1 else [axs]

    print(f"正在进行 {n_episodes} 次评估运行...")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        uav_paths = [[] for _ in range(env.N)]
        charger_loc = env.chargers[0].loc

        # 记录初始位置
        for i in range(env.N):
            uav_paths[i].append(env.uavs[i].loc.copy())

        total_reward = 0
        step_count = 0

        # 运行一个 Episode
        for _ in range(env.config["MAX_STEPS"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            for i in range(env.N):
                uav_paths[i].append(env.uavs[i].loc.copy())

            total_reward += reward
            step_count += 1
            if terminated or truncated:
                break

        # --- 在子图中绘制轨迹 ---
        ax = axs[ep]

        # 画 3x3 观测网格线 (灰色背景)
        for i in range(1, 3):
            ax.axhline(i * (500 / 3), color="gray", linestyle=":", alpha=0.2)
            ax.axvline(i * (500 / 3), color="gray", linestyle=":", alpha=0.2)

        # 画所有 UE (淡蓝色背景)
        ue_locs = np.array([ue.loc for ue in env.ues])
        ax.scatter(ue_locs[:, 0], ue_locs[:, 1], c="blue", alpha=0.05, s=10)

        # 画充电桩 (绿色方块)
        ax.scatter(
            charger_loc[0], charger_loc[1], c="green", marker="s", s=80, alpha=0.8
        )

        # 画 4 台 UAV 的轨迹
        colors = ["red", "orange", "purple", "blue"]
        for i in range(env.N):
            path = np.array(uav_paths[i])
            ax.plot(path[:, 0], path[:, 1], color=colors[i], alpha=0.7, linewidth=1.5)
            ax.scatter(path[0, 0], path[0, 1], c=colors[i], marker="o", s=20)  # 起点
            ax.scatter(path[-1, 0], path[-1, 1], c=colors[i], marker="x", s=40)  # 终点

        ax.set_title(f"EP {ep + 1} | Rew: {total_reward:.0f} | Steps: {step_count}")
        ax.set_xlim(0, 500)
        ax.set_ylim(0, 500)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("multi_episode_trajectories.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    # 填入你训练好的模型路径
    run_multi_eval("multirun/2026-02-03/16-11-25/1", n_episodes=10)
""
