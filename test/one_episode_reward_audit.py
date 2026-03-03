import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import SAC
import sys
from omegaconf import OmegaConf


# 将当前文件的父目录（即项目根目录）加入到搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在可以正常导入了
from core.sfc_env import SFCEnv
# 这个脚本用于对单个episode的奖励进行详细审计，绘制各分项奖励随时间的变化曲线，并记录在文本日志中。


# ==========================================
# 2. 配置拍扁工具 (同步 main.py 逻辑)
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


def run_reward_audit(model_path, n_episodes=5):
    # 1. 获取cfg配置与模型路径
    cfg_path = os.path.join(model_path, ".hydra/config.yaml")
    model_path = os.path.join(model_path, "SAC_final_model.zip")

    # 加载配置与模型
    cfg = OmegaConf.load(cfg_path)
    flat_cfg = flatten_hydra_cfg(cfg)
    # 2. 显式覆盖或添加参数
    flat_cfg.update({"RECORD_DEPLOYMENT": True})
    env = SFCEnv(config=flat_cfg)
    try:
        model = SAC.load(model_path)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 准备保存目录
    os.makedirs("results/audit", exist_ok=True)
    log_file = open("results/audit/reward_audit_log.txt", "w")

    # 1. 对应 SFCEnv._calculate_reward 中的 reward_info 键值
    # 注意：r_survival 在环境里是通过 step() 的终止逻辑触发的，这里我们手动追踪它
    reward_keys = ["r_task", "r_energy", "r_charge", "r_collision", "r_survival"]
    colors = [
        "#2ecc71",  # 绿色: 任务
        "#e74c3c",  # 红色: 能耗
        "#f1c40f",  # 黄色: 充电
        "#e67e22",  # 橙色: 碰撞
        "#9b59b6",  # 紫色: 生存/坠毁
    ]

    fig, axs = plt.subplots(n_episodes, 1, figsize=(12, 4 * n_episodes), sharex=True)
    if n_episodes == 1:
        axs = [axs]

    for ep in range(n_episodes):
        obs, _ = env.reset()
        history = {key: [] for key in reward_keys}
        history["total"] = []

        step_count = 0
        for step in range(env.config["MAX_STEPS"]):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # 2. 填充奖励明细
            # 从 info 中获取各分项奖励
            for key in reward_keys:
                if key == "r_survival":
                    # 如果坠毁(terminated)，且不是因为步数截止(truncated)，则记录坠毁惩罚
                    history[key].append(
                        env.config["RWD_CRASH"] if terminated and not truncated else 0.0
                    )
                else:
                    history[key].append(info.get(key, 0.0))

            history["total"].append(reward)

            step_count += 1
            if terminated or truncated:
                if terminated and not truncated:
                    log_file.write(f"EP {ep + 1}: 坠毁(Crashed) at Step {step}\n")
                break

        # --- 绘图逻辑 ---
        ax = axs[ep]
        x = range(len(history["total"]))

        for i, key in enumerate(reward_keys):
            # 将列表转为 numpy 数组方便处理
            vals = np.array(history[key])
            ax.plot(x, vals, label=f"{key}", color=colors[i], linewidth=2)
            # 填充颜色（对负向惩罚也很直观）
            ax.fill_between(x, vals, 0, color=colors[i], alpha=0.15)

        # 画总奖励线
        ax.plot(
            x,
            history["total"],
            label="Step Total",
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_title(f"Episode {ep + 1} Reward Breakdown (Steps: {step_count})")
        ax.set_ylabel("Reward Value")
        ax.grid(True, alpha=0.3)
        if ep == 0:
            ax.legend(loc="upper right", ncol=3)

        # --- 写入文本日志 ---
        log_file.write(f"--- Episode {ep + 1} Summary ---\n")
        for key in reward_keys:
            log_file.write(f"{key}: {sum(history[key]):.2f} | ")
        log_file.write(f"TOTAL: {sum(history['total']):.2f}\n")
        # 补充统计：成功率和任务丢弃率
        log_file.write(
            f"Completed: {info.get('count/completed', 0)} | Dropped: {info.get('count/dropped', 0)}\n\n"
        )

    plt.xlabel("Step")
    plt.tight_layout()
    plt.savefig("results/audit/reward_breakdown_all.png")
    log_file.close()
    print(
        f"审计完成！图表保存至 results/audit/reward_breakdown_all.png，日志详见 reward_audit_log.txt"
    )
    plt.show()


if __name__ == "__main__":
    # 请确保路径指向你最新的模型文件
    run_reward_audit("multirun/2026-02-03/16-11-25/1", n_episodes=5)
