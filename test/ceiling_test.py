import numpy as np
import sys
import os

# 路径处理
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sfc_env import SFCEnv

# 这个test文件用于测试“智能负载均衡策略”在不同带宽条件下的表现差异。
# 跑出来差距不大，说明带宽不影响这个策略的核心效果。


def smart_load_balanced_policy(env, bandwidth_cheat=False):
    """
    智能规则策略：
    1. 移动：依然追逐最紧急任务。
    2. Pick：选最急的 K 个。
    3. Place (核心优化)：不再无脑给最近的，而是给 [能跑完] 且 [目前最闲] 的 UAV。
    """
    K, L, M, N = env.K, env.L, env.M, env.N

    # --- 临时作弊逻辑：如果开启，强制覆盖环境带宽 ---
    if bandwidth_cheat:
        original_bw = env.config["BANDWIDTH_HZ"]
        env.config["BANDWIDTH_HZ"] = original_bw * 10

    # 1. 移动动作 (维持现状)
    mobility = []
    for uav in env.uavs:
        if env.current_cand_tasks:
            target = env.ues[env.current_cand_tasks[0][0]].loc
            diff = target - uav.loc
            mobility.extend(diff / (np.linalg.norm(diff) + 1e-9))
        else:
            mobility.extend([0, 0])

    # 2. 选人 (按顺序选最急的 K 个)
    pick_actions = [(k / (M + 1)) * 2 - 1 + 0.01 for k in range(K)]

    # 3. 部署动作 (算力均衡优化)
    place_actions = []
    # 模拟一个本轮的“预估负载表”
    uav_estimated_cycles = np.zeros(N)
    dt_compute = env.time_slot - env.dt_fly
    cap_per_uav = np.array([u.cpu_freq * dt_compute for u in env.uavs])

    for k in range(K):
        if k < len(env.current_cand_tasks):
            ue_id, sfc = env.current_cand_tasks[k]
            ue_loc = env.ues[ue_id].loc

            best_uav_id = -1
            min_score = float("inf")

            # 遍历所有 UAV，找一个“最合适”的
            for u_id in range(N):
                # 检查加上这个 SFC 后会不会超载
                if uav_estimated_cycles[u_id] + sfc.total_cycles < cap_per_uav[u_id]:
                    # 计算距离得分（越近越好）
                    dist = np.linalg.norm(env.uavs[u_id].loc - ue_loc)
                    # 综合得分 = 距离 + 负载压力 (这里是一个简单的启发式公式)
                    score = (
                        dist + (uav_estimated_cycles[u_id] / cap_per_uav[u_id]) * 200
                    )

                    if score < min_score:
                        min_score = score
                        best_uav_id = u_id

            # 如果没找到不超载的，就给最闲的那个（死马当活马医）
            if best_uav_id == -1:
                best_uav_id = np.argmin(uav_estimated_cycles)

            # 更新预估负载，供下一个任务参考
            uav_estimated_cycles[best_uav_id] += sfc.total_cycles
            place_actions.extend([(best_uav_id / N) * 2 - 1 + 0.01] * L)
        else:
            place_actions.extend([0.0] * L)

    # 还原带宽（不影响环境原始配置）
    if bandwidth_cheat:
        env.config["BANDWIDTH_HZ"] = original_bw

    return np.concatenate([mobility, pick_actions, place_actions])


def run_test():
    env = SFCEnv()

    # 测试 A：普通带宽 + 算力分配优化
    print(">>> 测试 A: 负载均衡策略 (Normal BW)...")
    res_a = run_episodes(env, cheat=False)

    # 测试 B：作弊带宽 (10x) + 算力分配优化
    print("\n>>> 测试 B: 负载均衡策略 + 10x 带宽 (Cheat BW)...")
    res_b = run_episodes(env, cheat=True)


def run_episodes(env, cheat, n=10):
    total_succ_across_episodes = []

    for _ in range(n):
        env.reset()
        done = False
        episode_completed_sum = 0  # 增加一个累加器，记录本局的总成功数

        while not done:
            act = smart_load_balanced_policy(env, bandwidth_cheat=cheat)
            _, _, term, trunc, info = env.step(act)

            # --- 核心修正：累加每一步的成功数 ---
            episode_completed_sum += info.get("completed_count", 0)

            done = term or trunc

        # 这一局跑完后，用【本局总成功】除以【本局总生成】
        # 其中本局总成功是每一步的info的completed_count累加而来，本局总生成直接用环境的total_gen_tasks属性
        actual_succ_rate = episode_completed_sum / max(1, env.total_gen_tasks)
        total_succ_across_episodes.append(actual_succ_rate)

    avg_succ = np.mean(total_succ_across_episodes)
    print(f"平均成功率: {avg_succ:.2%}")
    return avg_succ


if __name__ == "__main__":
    run_test()
