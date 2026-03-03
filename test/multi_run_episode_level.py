import os
import sys
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from stable_baselines3 import SAC, PPO

# 确保能找到 core 文件夹
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sfc_env import SFCEnv


# ==========================================
# 1. 策略评估器类 (自包含版)
# ==========================================
# 这个类统计的是重复多个episode，统计episode级的数据。
# 算出来这个agent在多个episode下的平均表现。
# 他会找multi_run下面的zip模型，然后加载对应的配置，跑50个episode，算平均值。
class PolicyEvaluator:
    def __init__(self, env):
        self.env = env

    def evaluate(self, model, n_episodes=50):
        results = []
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            ep_stats = {
                "succ": 0,
                "gen": 0,
                "pick": 0,
                "drop": 0,
                "time": 0,
                "steps": 0,
                "jains": [],
            }

            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = self.env.step(action)

                # 累加指标
                ep_stats["succ"] += info.get("completed_count", 0)
                ep_stats["gen"] += info.get("total_available", 0)
                ep_stats["pick"] += info.get("actually_picked", 0)
                ep_stats["drop"] += info.get("dropped_count", 0)
                ep_stats["time"] += info.get("timeout_count", 0)
                ep_stats["steps"] += 1

                # 计算负载均衡 (Jain's Index)
                loads = self.env.uav_load_status
                if np.sum(loads) > 0.01:
                    j_idx = (np.sum(loads) ** 2) / (
                        self.env.N * np.sum(loads**2) + 1e-9
                    )
                    ep_stats["jains"].append(j_idx)

                done = term or trunc

            # 汇总单局数据
            ep_summary = {
                "Success_Rate": (ep_stats["succ"] / max(1, ep_stats["gen"])) * 100,
                "Pick_Rate": (ep_stats["pick"] / max(1, ep_stats["gen"])) * 100,
                "Admission_Eff": (ep_stats["succ"] / max(1, ep_stats["pick"])) * 100,
                "Fairness": np.mean(ep_stats["jains"]) if ep_stats["jains"] else 0,
                "Survival": ep_stats["steps"],
            }
            results.append(ep_summary)

        return pd.DataFrame(results).mean().to_dict()


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


# ==========================================
# 新增：自动识别模型类型并加载
# ==========================================
def load_model_by_filename(model_path):
    """
    根据模型文件名自动识别并加载SAC/PPO模型

    Args:
        model_path: 模型zip文件的完整路径

    Returns:
        加载好的模型对象
    """
    # 获取文件名（全小写，避免大小写敏感问题）
    file_name = os.path.basename(model_path).lower()

    # 判断模型类型
    if "sac" in file_name:
        print(f"识别到SAC模型，使用SAC加载: {model_path}")
        return SAC.load(model_path)
    elif "ppo" in file_name:
        print(f"识别到PPO模型，使用PPO加载: {model_path}")
        return PPO.load(model_path)
    else:
        # 未识别类型时抛出明确异常
        raise ValueError(
            f"无法识别模型类型！文件名 {os.path.basename(model_path)} 中未包含SAC/PPO关键词"
        )


# ==========================================
# 3. 自动化扫描与对比逻辑
# ==========================================
def run_comparison(multirun_dir):
    # 查找所有数字命名的文件夹 (0, 1, 2...)
    subfolders = [f for f in os.listdir(multirun_dir) if f.isdigit()]
    comparison_list = []

    print(f"检测到 {len(subfolders)} 组实验，开始批量评估...")

    for folder in subfolders:
        run_path = os.path.join(multirun_dir, folder)
        cfg_path = os.path.join(run_path, ".hydra/config.yaml")
        # 【修改1】查找文件夹下所有zip文件（替代固定文件名）
        zip_files = [f for f in os.listdir(run_path) if f.lower().endswith(".zip")]
        if not zip_files:
            print(f"跳过 {folder}: 找不到任何zip模型文件")
            continue
        # 取第一个zip文件（如果有多个，优先取第一个）
        model_filename = zip_files[0]
        model_path = os.path.join(run_path, model_filename)

        if not os.path.exists(model_path):
            print(f"跳过 {folder}: 找不到模型文件")
            continue

        # 加载配置与模型
        cfg = OmegaConf.load(cfg_path)
        flat_cfg = flatten_hydra_cfg(cfg)

        env = SFCEnv(config=flat_cfg)
        # 【修改2】自动识别并加载模型
        try:
            model = load_model_by_filename(model_path)
        except ValueError as e:
            print(f"跳过 {folder}: {e}")
            continue
        except Exception as e:
            print(f"跳过 {folder}: 加载模型失败 - {e}")
            continue

        # 执行评估
        print(f"正在评估: {folder} (UAV主频: {flat_cfg['UAV_CPU_FREQ']})")
        evaluator = PolicyEvaluator(env)
        metrics = evaluator.evaluate(model, n_episodes=50)

        # 记录关键变量
        metrics["UAV_FREQ"] = flat_cfg["UAV_CPU_FREQ"]
        metrics["Folder"] = folder
        comparison_list.append(metrics)

    # 结果汇总与排序
    df = pd.DataFrame(comparison_list)
    df = df.sort_values(by="UAV_FREQ")

    print("\n" + "=" * 90)
    print(
        f"{'Folder':<8} | {'UAV_FREQ':<10} | {'Success%':<10} | {'Pick%':<10} | {'Adm_Eff%':<10} | {'Fairness':<10}"
    )
    print("-" * 90)
    for _, row in df.iterrows():
        print(
            f"{row['Folder']:<8} | {row['UAV_FREQ']:<10.1e} | {row['Success_Rate']:<10.2f} | "
            f"{row['Pick_Rate']:<10.2f} | {row['Admission_Eff']:<10.2f} | {row['Fairness']:<10.3f}"
        )
    print("=" * 90)

    # 保存结果
    df.to_csv(os.path.join(multirun_dir, "comparison_results.csv"), index=False)
    print(f"对比报告已保存至: {multirun_dir}/comparison_results.csv")


if __name__ == "__main__":
    # ！！！请修改这里为你实际的文件夹路径 ！！！
    # 比如: "multirun/2024-01-23/15-30-00"
    PATH = "multirun/2026-02-03/16-11-25"
    run_comparison(PATH)
