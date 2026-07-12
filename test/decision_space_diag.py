"""
决策空间诊断：量化 A(最近邻) 与 B(负载均衡) 在【同一环境状态】下
给出的部署方案到底多相似。

方法：用 A 驱动轨迹(A=现状 place_intent 的等价物)，但每一步同时计算
A 和 B 的 place_matrix，只在【实际被 pick 的槽位】上比较二者选的 UAV。
两策略看到完全相同的状态，所以重合率 = 纯粹的决策分化度。

关键指标：
  slot_agree%     —— 逐槽位(每个被pick的VNF)A、B选同一UAV的比例
  step_identical% —— 整步部署矩阵完全一致的比例
  nearest_is_freest% —— A选的最近UAV，恰好也是"当前负载最低可行"的比例
                        (若很高，说明'最近'天然≈'均衡'，B无腾挪空间)
  overload_slots% —— A的部署里，目标UAV已超载的槽位比例(B本可救而A救不了的空间)
"""
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sfc_env import SFCEnv
from place_ablation import place_nearest, place_balance, _build_action, _mobility_actions, _pick_actions


def picked_slots(env):
    """返回本步实际被 pick(且有效)的 k 索引列表，及对应 sfc。"""
    K, M = env.K, env.M
    out = []
    seen = set()
    for k in range(K):
        if k < len(env.current_cand_tasks):
            ue_id, sfc = env.current_cand_tasks[k]
            if ue_id not in seen:
                seen.add(ue_id)
                out.append((k, sfc))
    return out


def diagnose(cfg, n_ep=40, seed=42):
    env = SFCEnv(config=cfg)
    N = env.N

    agree_slots = tot_slots = 0
    identical_steps = tot_steps = 0
    nearest_is_freest = freest_denom = 0
    overload_slots = 0

    dt_compute = env.time_slot - env.dt_fly

    for ep in range(n_ep):
        env.reset(seed=seed + ep)
        done = False
        while not done:
            pa = place_nearest(env)   # [K,L]
            pb = place_balance(env)   # [K,L]
            caps = np.array([u.cpu_freq * dt_compute for u in env.uavs])

            slots = picked_slots(env)
            step_all_agree = True
            # 模拟 A 的累计负载，用于判断"最近是否也最空"
            used_a = np.zeros(N)
            for (k, sfc) in slots:
                L = len(sfc.vnf_chain)
                for l in range(L):
                    ua, ub = pa[k, l], pb[k, l]
                    tot_slots += 1
                    if ua == ub:
                        agree_slots += 1
                    else:
                        step_all_agree = False

                # A 把整条链放在 pa[k,0]（现状策略同一UAV），判断它是否已超载
                target = pa[k, 0]
                freest_denom += 1
                # "最空可行"：在放本任务前，负载最低且放得下的UAV
                feasible = [u for u in range(N) if used_a[u] + sfc.total_cycles < caps[u]]
                if feasible:
                    freest = min(feasible, key=lambda u: used_a[u])
                    if target == freest:
                        nearest_is_freest += 1
                if used_a[target] + sfc.total_cycles >= caps[target]:
                    overload_slots += 1
                used_a[target] += sfc.total_cycles

            if slots:
                tot_steps += 1
                if step_all_agree:
                    identical_steps += 1

            action = _build_action(env, pa)
            _, _, term, trunc, _ = env.step(action)
            done = term or trunc

    return {
        "slot_agree%": agree_slots / max(1, tot_slots) * 100,
        "step_identical%": identical_steps / max(1, tot_steps) * 100,
        "nearest_is_freest%": nearest_is_freest / max(1, freest_denom) * 100,
        "overload_slots%": overload_slots / max(1, freest_denom) * 100,
        "n_slots": tot_slots,
    }


def main():
    cases = [(0.3, 4), (0.4, 5), (0.5, 6)]
    print(f"{'case':<12}{'slot_agree%':>12}{'step_ident%':>12}{'near=freest%':>14}{'overload%':>11}{'n':>8}")
    for ap, k in cases:
        for hard in [True, False]:
            cfg = {"ARRIVAL_PROB": ap, "K": k, "USE_HARD_CAP": hard}
            r = diagnose(cfg)
            mech = "hard" if hard else "soft"
            print(f"ap{ap}K{k} {mech:<5}{r['slot_agree%']:12.1f}{r['step_identical%']:12.1f}"
                  f"{r['nearest_is_freest%']:14.1f}{r['overload_slots%']:11.1f}{r['n_slots']:8d}")
    print("\n判读：slot_agree% 越高 → A、B决策越雷同 → place不分化 → 换UAV没意义。")
    print("      near=freest% 越高 → '最近'天然≈'最空' → B无腾挪空间。")
    print("      overload% 越高 → A确实频繁放到超载UAV → B本应有救的空间(若B真救得动)。")


if __name__ == "__main__":
    main()
