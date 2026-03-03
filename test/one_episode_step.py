import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import sys
import os

# 将当前文件的父目录（即项目根目录）加入到搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 现在可以正常导入了
from core.sfc_env import SFCEnv
import os
import sys
from omegaconf import OmegaConf


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


def run_comprehensive_audit(model_path):
    # 1. 初始化环境与模型 (强制开启记录模式)
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

    # 2. 准备审计结果目录
    output_dir = "results/audit_final"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "detailed_audit_log.txt")

    # 3. 数据记录容器
    uav_locs = [[] for _ in range(env.N)]
    uav_batts = [[] for _ in range(env.N)]
    audit_data = {
        "steps": [],
        "env_supply": [],
        "agent_picked": [],
        "actual_completed": [],
        "jains_index": [],
        "rewards": [],
    }

    obs, _ = env.reset()
    colors = ["red", "green", "purple", "brown", "orange", "blue"]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write("        无人机 SFC 调度全流程深度审计日志 (终极融合版)        \n")
        f.write("============================================================\n\n")

        total_reward = 0
        total_succ_ever = 0

        for step in range(env.config["MAX_STEPS"]):
            # A. 决策前快照
            active_buffer_ids = [ue.node_id for ue in env.ues if ue.task_buffer]

            # B. 决策与执行
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # C. 数据采集
            total_reward += reward
            total_succ_ever += info.get("completed_count", 0)
            loads = env.uav_load_status
            j_index = (np.sum(loads) ** 2) / (env.N * np.sum(loads**2) + 1e-9)

            audit_data["steps"].append(step)
            audit_data["env_supply"].append(info.get("total_available", 0))
            audit_data["agent_picked"].append(info.get("actually_picked", 0))
            audit_data["actual_completed"].append(info.get("completed_count", 0))
            audit_data["jains_index"].append(j_index)
            audit_data["rewards"].append(reward)

            for i in range(env.N):
                uav_locs[i].append(env.uavs[i].loc.copy())
                uav_batts[i].append(env.uavs[i].e_battery)

            # D. 日志写入
            f.write(f"【Step {step:02d}】\n")
            f.write(
                f"  [供需] 桌面任务: {info.get('total_available', 0)} | Agent接单: {info.get('actually_picked', 0)}\n"
            )
            f.write(
                f"  [物理] 负载: {np.round(loads, 3)} | 奖励: {reward:.2f} | 下步新增: {env.current_step_gen}\n"
            )
            f.write("-" * 50 + "\n")

            if terminated or truncated:
                f.write(f"\n>>>> 结束原因: {'坠毁' if terminated else '步数上限'}\n")
                f.write(f"最终成功: {total_succ_ever} | 总分: {total_reward:.2f}\n")
                break

    # --- 4. 综合绘图 (3子图布局) ---
    fig = plt.figure(figsize=(20, 12))
    steps_arr = audit_data["steps"]

    # 子图 1: 空间轨迹图 (融合 test_model.py 的精华)
    ax1 = fig.add_subplot(221)
    ue_locs = np.array([ue.loc for ue in env.ues])
    ax1.scatter(ue_locs[:, 0], ue_locs[:, 1], c="blue", alpha=0.1, s=20, label="UEs")
    for i in range(env.N):
        path = np.array(uav_locs[i])
        ax1.plot(
            path[:, 0], path[:, 1], color=colors[i % 6], label=f"UAV {i}", linewidth=1.5
        )
        ax1.scatter(path[-1, 0], path[-1, 1], c=colors[i % 6], marker="X", s=80)
    ax1.scatter(250, 250, c="darkgreen", marker="s", s=100, label="Charger")
    ax1.set_title("Flight Trajectories")
    ax1.legend(loc="lower right", ncol=2, fontsize="small")

    # 子图 2: 供需矛盾分析图
    ax2 = fig.add_subplot(222)
    ax2.plot(
        steps_arr,
        audit_data["env_supply"],
        "g--",
        label="Total Avail (Supply)",
        alpha=0.5,
    )
    ax2.plot(
        steps_arr, audit_data["agent_picked"], "b-o", label="Agent Picked (Demand)"
    )
    ax2.plot(
        steps_arr, audit_data["actual_completed"], "r-x", label="Completed (Output)"
    )
    ax2.set_title("SFC Supply-Demand-Output Chain")
    ax2.legend()

    # 子图 3: 资源健康与电量 (双轴)
    ax3 = fig.add_subplot(223)
    ax3_twin = ax3.twinx()
    for i in range(env.N):
        ax3.plot(steps_arr, uav_batts[i], color=colors[i % 6], alpha=0.6, linestyle=":")
    ax3_twin.plot(
        steps_arr,
        audit_data["jains_index"],
        color="purple",
        linewidth=2,
        label="Jain's Index",
    )
    ax3.set_ylabel("Battery (Joules)")
    ax3_twin.set_ylabel("Fairness Index")
    ax3.set_title("Battery Levels & Load Fairness")

    # 子图 4: 奖励曲线
    ax4 = fig.add_subplot(224)
    ax4.plot(
        steps_arr,
        np.cumsum(audit_data["rewards"]),
        color="orange",
        label="Cumulative Reward",
    )
    ax4.set_title("Cumulative Total Reward")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "full_audit_visualization.png"))
    print(f"✅ 审计报告已保存至: {output_dir}")


def smart_heuristic_policy(env):
    """
    智能启发式规则 (生成与 SAC 格式一致的动作向量 [-1, 1]):
    - Mobility: 能量低去充电，能量足追逐最紧急任务
    - Pick: 选择最紧急的 K 个任务
    - Place: 简单的负载均衡策略
    """
    K, L, M, N = env.K, env.L, env.M, env.N

    # --- A. 移动逻辑 ---
    mobility = []
    for uav in env.uavs:
        # 简单逻辑：电量低于30%回充，否则去最近的候选任务点
        target = (
            env.chargers[0].loc
            if uav.e_battery < uav.battery_capacity * 0.3
            else uav.loc
        )
        if uav.e_battery >= uav.battery_capacity * 0.3 and env.current_cand_tasks:
            target = env.ues[env.current_cand_tasks[0][0]].loc

        diff = target - uav.loc
        dist = np.linalg.norm(diff) + 1e-9
        # 映射到 [-1, 1] 范围
        action_move = np.clip(diff / dist, -1.0, 1.0)
        mobility.extend(action_move)

    # --- B. 选人逻辑 ---
    # 启发式直接按顺序选前 K 个，映射为连续动作空间的值
    pick_actions = [(k / (M + 1)) * 2 - 1 + 0.01 for k in range(K)]

    # --- C. 部署逻辑 (负载均衡) ---
    place_actions = []
    # 模拟预测负载
    uav_busy_cycles = np.zeros(N)
    for k in range(K):
        if k < len(env.current_cand_tasks):
            # 找到当前累积预估负载最低的 UAV
            best_uav_id = np.argmin(uav_busy_cycles)
            uav_busy_cycles[best_uav_id] += env.current_cand_tasks[k][1].total_cycles
            # 映射到 [-1, 1]
            val = (best_uav_id / N) * 2 - 1 + 0.01
            place_actions.extend([val] * L)
        else:
            place_actions.extend([-1.0] * L)  # 不选

    return np.concatenate([mobility, pick_actions, place_actions])


def run_final_heuristic_audit(model_dir=None, strategy_name="Heuristic_Balanced"):
    """
    终极融合规格审计函数
    model_dir: 可选。如果提供，则从 Hydra 目录加载配置；否则使用默认环境配置。
    """
    # 1. 配置加载逻辑
    if model_dir and os.path.exists(model_dir):
        print(f"检测到目录，正在从 {model_dir} 加载实验配置...")
        cfg_path = os.path.join(model_dir, ".hydra/config.yaml")
        cfg = OmegaConf.load(cfg_path)
        flat_cfg = dict(cfg)
    else:
        print("未提供 model_dir 或目录不存在，将使用系统默认环境配置运行。")
        flat_cfg = {}  # 使用 SFCEnv 的内部默认配置

    # 强制开启详细记录
    flat_cfg.update({"RECORD_DEPLOYMENT": True})

    # 初始化环境
    env = SFCEnv(config=flat_cfg)

    # 2. 准备审计结果目录
    output_dir = f"results/audit_{strategy_name}"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"detailed_{strategy_name}_log.txt")

    # 3. 数据记录容器
    uav_locs = [[] for _ in range(env.N)]
    uav_batts = [[] for _ in range(env.N)]
    audit_data = {
        "steps": [],
        "env_supply": [],
        "agent_picked": [],
        "actual_completed": [],
        "jains_index": [],
        "rewards": [],
    }

    obs, _ = env.reset()
    colors = ["red", "green", "purple", "brown", "orange", "blue"]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write(f"        无人机 SFC 调度全流程审计报告 ({strategy_name})        \n")
        f.write("============================================================\n\n")

        total_reward = 0
        total_succ_ever = 0

        for step in range(env.config["MAX_STEPS"]):
            active_buffer_ids = [ue.node_id for ue in env.ues if ue.task_buffer]
            action = smart_heuristic_policy(env)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            total_succ_ever += info.get("completed_count", 0)
            loads = env.uav_load_status
            j_index = (np.sum(loads) ** 2) / (env.N * np.sum(loads**2) + 1e-9)

            audit_data["steps"].append(step)
            audit_data["env_supply"].append(info.get("total_available", 0))
            audit_data["agent_picked"].append(info.get("actually_picked", 0))
            audit_data["actual_completed"].append(info.get("completed_count", 0))
            audit_data["jains_index"].append(j_index)
            audit_data["rewards"].append(reward)

            for i in range(env.N):
                uav_locs[i].append(env.uavs[i].loc.copy())
                uav_batts[i].append(env.uavs[i].e_battery)

            f.write(f"【Step {step:02d}】\n")
            f.write(
                f"  [供需] 桌面任务: {info.get('total_available', 0)} | 认领: {info.get('actually_picked', 0)}\n"
            )
            f.write(
                f"  [物理] 负载: {np.round(loads, 3)} | 奖励: {reward:.2f} | 新增: {env.current_step_gen}\n"
            )
            f.write("-" * 50 + "\n")

            if terminated or truncated:
                f.write(f"\n>>>> 结束原因: {'坠毁' if terminated else '步数上限'}\n")
                f.write(f"最终成功: {total_succ_ever} | 总分: {total_reward:.2f}\n")
                break

    # --- 4. 综合绘图 ---
    fig = plt.figure(figsize=(20, 12))
    steps_arr = audit_data["steps"]

    # 飞行轨迹
    ax1 = fig.add_subplot(221)
    ue_locs = np.array([ue.loc for ue in env.ues])
    ax1.scatter(ue_locs[:, 0], ue_locs[:, 1], c="blue", alpha=0.1, s=20, label="UEs")
    for i in range(env.N):
        path = np.array(uav_locs[i])
        ax1.plot(path[:, 0], path[:, 1], color=colors[i % 6], label=f"UAV {i}")
        ax1.scatter(path[-1, 0], path[-1, 1], c=colors[i % 6], marker="X")
    ax1.scatter(250, 250, c="darkgreen", marker="s", s=100, label="Charger")
    ax1.set_title("Trajectories")
    ax1.legend(loc="lower right", ncol=2)

    # 供需链
    ax2 = fig.add_subplot(222)
    ax2.plot(steps_arr, audit_data["env_supply"], "g--", label="Supply", alpha=0.5)
    ax2.plot(steps_arr, audit_data["agent_picked"], "b-o", label="Picked")
    ax2.plot(steps_arr, audit_data["actual_completed"], "r-x", label="Completed")
    ax2.set_title("Supply-Demand Chain")
    ax2.legend()

    # 资源健康
    ax3 = fig.add_subplot(223)
    ax3_twin = ax3.twinx()
    for i in range(env.N):
        ax3.plot(steps_arr, uav_batts[i], color=colors[i % 6], alpha=0.4, linestyle=":")
    ax3_twin.plot(
        steps_arr, audit_data["jains_index"], color="purple", label="Fairness"
    )
    ax3.set_title("Battery & Fairness")

    # 累计分
    ax4 = fig.add_subplot(224)
    ax4.plot(steps_arr, np.cumsum(audit_data["rewards"]), color="orange")
    ax4.set_title("Cumulative Reward")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heuristic_audit_viz.png"))
    print(f"✅ 基准审计完成。报告存至: {output_dir}")


if __name__ == "__main__":
    # run_step_by_step_audit(
    #     "experiments/uav_sfc_run_0123_163238/sac_sfc_model"
    # )  # 替换为你的模型路径")
    # run_detailed_debug()
    # run_comprehensive_audit("multirun/2026-02-03/16-11-25/1")  # 替换为你的模型路径")
    run_final_heuristic_audit(model_dir=None, strategy_name="Heuristic_Balanced")
