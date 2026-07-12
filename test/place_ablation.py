"""
Place 策略消融实验 (A/B/C 对照)
=================================================
目的：验证"选哪架 UAV 部署"这个决策在当前环境里到底重不重要。

控制变量法：移动逻辑、Pick 逻辑三种策略完全相同，
唯一变量是 Place 策略：
    A. nearest   —— 每个 VNF 指给离 UE 最近的 UAV（你现在 place_intent 最近邻映射的等价物）
    B. balance   —— 优先指给"放得下且最近"的 UAV，放不下就给最闲的（负载均衡）
    C. random    —— 随机指一架 UAV（下界基准）

如果 B 显著 > A，说明"选 UAV"重要且"选近的"是次优 → place 表现形式值得改。
如果 B ≈ A ≈ C，说明这个决策对结果影响不大 → 现有最近邻映射够用。

不训练、不碰算法，纯启发式。
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sfc_env import SFCEnv


# ---------------------------------------------------------------------------
# 公共部分：移动 + Pick（三种策略完全一致，保证唯一变量是 Place）
# ---------------------------------------------------------------------------
def _mobility_actions(env):
    acts = []
    charger_loc = env.chargers[0].loc
    for uav in env.uavs:
        if uav.e_battery < uav.battery_capacity * 0.25:
            target = charger_loc
        elif env.current_cand_tasks:
            target = env.ues[env.current_cand_tasks[0][0]].loc
        else:
            target = uav.loc
        diff = target - uav.loc
        dist = np.linalg.norm(diff) + 1e-9
        acts.extend(diff / dist)
    return acts


def _pick_actions(env):
    K, M = env.K, env.M
    acts = []
    for k in range(K):
        if k < len(env.current_cand_tasks):
            raw = (k / (M + 1)) * 2 - 1 + 0.001
        else:
            raw = (M / (M + 1)) * 2 - 1 + 0.001
        acts.append(raw)
    return acts


def _uav_id_to_raw(uav_id, N):
    """把离散 UAV id 编码回 place_intent 期望的 [-1,1]（占位，实际用 _build_action 直连）"""
    return (uav_id / N) * 2 - 1 + 0.001


# ---------------------------------------------------------------------------
# 三种 Place 策略：返回一个 [K, L] 的 uav_id 矩阵
# ---------------------------------------------------------------------------
def place_nearest(env):
    """A. 每个 VNF 指给离 UE 最近的 UAV（无视负载）"""
    K, L, N = env.K, env.L, env.N
    place = np.zeros((K, L), dtype=np.int32)
    for k in range(K):
        if k < len(env.current_cand_tasks):
            ue_id, _sfc = env.current_cand_tasks[k]
            ue_loc = env.ues[ue_id].loc
            dists = [np.linalg.norm(env.uavs[u].loc - ue_loc) for u in range(N)]
            best = int(np.argmin(dists))
            place[k, :] = best
    return place


def place_balance(env):
    """B. 负载均衡：优先选放得下且最近的，放不下给最闲的"""
    K, L, N = env.K, env.L, env.N
    place = np.zeros((K, L), dtype=np.int32)
    dt_compute = env.time_slot - env.dt_fly
    uav_caps = np.array([u.cpu_freq * dt_compute for u in env.uavs])
    used = np.zeros(N)
    for k in range(K):
        if k < len(env.current_cand_tasks):
            ue_id, sfc = env.current_cand_tasks[k]
            ue_loc = env.ues[ue_id].loc
            best, min_d = -1, float("inf")
            for u in range(N):
                if used[u] + sfc.total_cycles < uav_caps[u]:
                    d = np.linalg.norm(env.uavs[u].loc - ue_loc)
                    if d < min_d:
                        min_d, best = d, u
            if best == -1:
                best = int(np.argmin(used))
            used[best] += sfc.total_cycles
            place[k, :] = best
    return place


def place_random(env, rng):
    """C. 随机指派（下界基准）"""
    K, L, N = env.K, env.L, env.N
    place = np.zeros((K, L), dtype=np.int32)
    for k in range(K):
        if k < len(env.current_cand_tasks):
            place[k, :] = rng.integers(0, N)
    return place


PLACE_FNS = {
    "A_nearest": lambda env, rng: place_nearest(env),
    "B_balance": lambda env, rng: place_balance(env),
    "C_random": lambda env, rng: place_random(env, rng),
}


# ---------------------------------------------------------------------------
# 用 place_intent 的坐标通道，把"目标 UAV"精确编码进动作
# 做法：把意图坐标直接设成目标 UAV 的当前坐标 -> 最近邻必然选中它
# 这样就绕过了 (uav_id/N) 这种脆弱的索引编码，和真实 step 解码完全对齐。
# ---------------------------------------------------------------------------
def _build_action(env, place_matrix):
    N, K, L = env.N, env.K, env.L
    W, H = env.config["GROUND_WIDTH"], env.config["GROUND_HEIGHT"]

    mob = _mobility_actions(env)
    pick = _pick_actions(env)

    place_intent = np.zeros((K, L, 2), dtype=np.float32)
    for k in range(K):
        for l in range(L):
            uav_id = place_matrix[k, l]
            ux, uy = env.uavs[uav_id].loc
            # 把 UAV 坐标反归一化回 [-1,1]，使 step 的最近邻映射精确命中该 UAV
            place_intent[k, l, 0] = (ux / W) * 2 - 1
            place_intent[k, l, 1] = (uy / H) * 2 - 1

    return np.concatenate([mob, pick, place_intent.flatten()]).astype(np.float32)


# ---------------------------------------------------------------------------
# 评估单个策略
# ---------------------------------------------------------------------------
def evaluate(place_key, n_episodes, seed):
    env = SFCEnv()
    rng = np.random.default_rng(seed)
    place_fn = PLACE_FNS[place_key]
    rows = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        gen = pick = succ = drop = timeout = 0
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            place_matrix = place_fn(env, rng)
            action = _build_action(env, place_matrix)
            obs, reward, term, trunc, info = env.step(action)

            gen += info.get("total_available", 0)
            pick += info.get("actually_picked", 0)
            succ += info.get("completed_count", 0)
            drop += info.get("dropped_count", 0)
            timeout += info.get("timeout_count", 0)
            total_reward += reward
            steps += 1
            done = term or trunc

        rows.append({
            "SuccessRate(%)": succ / max(1, gen) * 100,
            "AdmissionEff(%)": succ / max(1, pick) * 100,
            "Dropped": drop,
            "Timeout": timeout,
            "Reward": total_reward,
            "SurvivalSteps": steps,
        })
    return pd.DataFrame(rows)


def main(n_episodes=100, seed=42):
    print(f"Place 策略消融实验 | {n_episodes} episodes/策略 | seed={seed}\n")
    summaries = {}
    for key in PLACE_FNS:
        df = evaluate(key, n_episodes, seed)
        summaries[key] = df.mean()

    result = pd.DataFrame(summaries).T
    pd.set_option("display.float_format", lambda x: f"{x:8.3f}")
    print(result.to_string())
    print("\n判读：")
    a, b, c = summaries["A_nearest"], summaries["B_balance"], summaries["C_random"]
    sr_a, sr_b, sr_c = a["SuccessRate(%)"], b["SuccessRate(%)"], c["SuccessRate(%)"]
    print(f"  B(负载均衡) 完成率 {sr_b:.2f}%  vs  A(最近) {sr_a:.2f}%  vs  C(随机) {sr_c:.2f}%")
    if sr_b - sr_a > 3:
        print(f"  → B 比 A 高 {sr_b - sr_a:.2f} 个百分点：选 UAV 重要，且'选近的'是次优。place 表现形式值得改。")
    else:
        print(f"  → B 与 A 差距仅 {sr_b - sr_a:.2f} 个百分点：当前 setting 下选 UAV 影响不大。")


if __name__ == "__main__":
    main(100)
