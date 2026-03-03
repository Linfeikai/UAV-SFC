import numpy as np
import pandas as pd
import os
import sys

# --- 1. 路径处理 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sfc_env import SFCEnv

# 这个脚本用于评估一个简单的多UAV多任务规则策略，在多个episode下的表现。
# 给出的表现是episiode级别的、有均值和标准差。
# 这个评估所处的环境全是环境的默认参数


def smart_heuristic_policy(env):
    """
    针对 V1 (M选K) 架构的智能规则策略：
    - Pick: 选候选池里最急的前 K 个
    - Place: 寻找 [不超载] 且 [离UE最近] 的 UAV
    """
    K, L, M, N = env.K, env.L, env.M, env.N

    # --- A. 移动逻辑 (生存优先 + 热点追逐) ---
    mobility_actions = []
    charger_loc = env.chargers[0].loc
    for i, uav in enumerate(env.uavs):
        if uav.e_battery < uav.battery_capacity * 0.25:
            target = charger_loc
        elif env.current_cand_tasks:
            # 追随最紧急任务的 UE 位置
            target = env.ues[env.current_cand_tasks[0][0]].loc
        else:
            target = uav.loc
        diff = target - uav.loc
        dist = np.linalg.norm(diff) + 1e-9
        mobility_actions.extend(diff / dist)

    # --- B. 选人逻辑 (Pick 最急的 K 个) ---
    # 将索引映射回 [-1, 1] 供环境解析
    pick_actions = []
    for k in range(K):
        if k < len(env.current_cand_tasks):
            # 选第 k 个候选人
            raw_pick = (k / (M + 1)) * 2 - 1 + 0.001
        else:
            # 选 M (空位)
            raw_pick = (M / (M + 1)) * 2 - 1 + 0.001
        pick_actions.append(raw_pick)

    # --- C. 部署逻辑 (Place: 负载均衡 + 就近) ---
    place_actions = []
    uav_capacity_used = np.zeros(N)
    dt_compute = env.time_slot - env.dt_fly
    uav_caps = np.array([u.cpu_freq * dt_compute for u in env.uavs])

    for k in range(K):
        if k < len(env.current_cand_tasks):
            ue_id, sfc = env.current_cand_tasks[k]
            ue_loc = env.ues[ue_id].loc

            best_uav = -1
            min_dist = float("inf")
            # 筛选：不仅要近，还要能放下这整个 SFC
            for u_id in range(N):
                if uav_capacity_used[u_id] + sfc.total_cycles < uav_caps[u_id]:
                    d = np.linalg.norm(env.uavs[u_id].loc - ue_loc)
                    if d < min_dist:
                        min_dist = d
                        best_uav = u_id

            # 如果都超载了，就给最闲的那个
            if best_uav == -1:
                best_uav = np.argmin(uav_capacity_used)

            uav_capacity_used[best_uav] += sfc.total_cycles
            raw_place = (best_uav / N) * 2 - 1 + 0.001
            place_actions.extend([raw_place] * L)
        else:
            place_actions.extend([0.0] * L)

    return np.concatenate([mobility_actions, pick_actions, place_actions])


def bulk_evaluate(n_episodes=100):
    env = SFCEnv()
    all_ep_results = []

    print(f"开始 {n_episodes} 轮批量基准评估...")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        stats = {
            "gen": 0,
            "pick": 0,
            "succ": 0,
            "drop": 0,
            "timeout": 0,
            "reward": 0,
            "len": 0,
            "jains": [],
        }

        done = False
        while not done:
            action = smart_heuristic_policy(env)
            obs, reward, term, trunc, info = env.step(action)

            # 累加指标
            stats["gen"] += info.get("total_available", 0)  # 使用桌面可见总数作为分母
            stats["pick"] += info.get("actually_picked", 0)
            stats["succ"] += info.get("completed_count", 0)
            stats["drop"] += info.get("dropped_count", 0)  # 准入拒绝
            stats["timeout"] += info.get("timeout_count", 0)
            stats["reward"] += reward
            stats["len"] += 1

            # 计算 Jain's Index (负载均衡)
            loads = env.uav_load_status
            if np.sum(loads) > 0.01:
                j = (np.sum(loads) ** 2) / (env.N * np.sum(loads**2) + 1e-9)
                stats["jains"].append(j)

            done = term or trunc

        # 结算该 Episode
        ep_res = {
            "Total_Success_Rate (%)": (stats["succ"] / max(1, stats["gen"])) * 100,
            "Pick_Rate (%)": (stats["pick"] / max(1, stats["gen"])) * 100,
            "Admission_Efficiency (%)": (stats["succ"] / max(1, stats["pick"])) * 100,
            "Avg_Fairness": np.mean(stats["jains"]) if stats["jains"] else 0,
            "Survival_Steps": stats["len"],
            "Total_Reward": stats["reward"],
        }
        all_ep_results.append(ep_res)

    # 汇总计算
    df = pd.DataFrame(all_ep_results)
    summary = df.agg(["mean", "std"]).T

    print("\n" + "=" * 60)
    print(f"{'论文指标 (Metric)':<30} | {'均值 (Mean)':<12} | {'标准差 (Std)'}")
    print("-" * 60)
    for idx, row in summary.iterrows():
        print(f"{idx:<30} | {row['mean']:<12.4f} | {row['std']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    bulk_evaluate(100)
