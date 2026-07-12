"""
Place 决策"甜区"二维参数扫描
=================================================
在 (ARRIVAL_PROB, K) 网格上，对比 A(最近邻) 与 B(负载均衡) 两种 place 策略。
目标：找出 B - A 完成率差距最大的参数组合 —— 那就是"精细 place 决策最值钱"
      的甜区，也是最能凸显扩散策略多模态优势的环境 setting。

理论依据：
  - ARRIVAL_PROB 控制"总需求水位" -> 决定完成率是否被总量锁死。
  - K 控制"瞬时并发挤压"        -> 决定是否存在"分配艺术"的空间。
  - 两者交叉的甜区：总量吃紧 + 并发充分 -> 笨策略踩超载、聪明策略靠腾挪胜出。

复用 place_ablation.py 的策略与动作构造，唯一变量仍是 place。
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sfc_env import SFCEnv
from place_ablation import place_nearest, place_balance, _build_action


def run_one(config_override, place_fn, n_episodes, seed):
    env = SFCEnv(config=config_override)
    succ_rates, rewards, drops = [], [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        gen = succ = drop = 0
        total_reward = 0.0
        done = False
        while not done:
            place_matrix = place_fn(env)
            action = _build_action(env, place_matrix)
            _, reward, term, trunc, info = env.step(action)
            gen += info.get("total_available", 0)
            succ += info.get("completed_count", 0)
            drop += info.get("dropped_count", 0)
            total_reward += reward
            done = term or trunc
        succ_rates.append(succ / max(1, gen) * 100)
        rewards.append(total_reward)
        drops.append(drop)
    return np.mean(succ_rates), np.mean(rewards), np.mean(drops)


def main(n_episodes=40, seed=42):
    arrival_grid = [0.2, 0.3, 0.4, 0.5]
    k_grid = [3, 4, 5, 6]

    print("Place 甜区扫描 (A=nearest, B=balance)")
    print(f"每格 {n_episodes} episodes | seed={seed}")
    print("供给上限 ~5.6e9 cycles/slot (N=4 x 2e8 x 7s)\n")

    records = []
    best = None
    for ap in arrival_grid:
        for k in k_grid:
            cfg = {"ARRIVAL_PROB": ap, "K": k}
            sr_a, rw_a, dr_a = run_one(cfg, place_nearest, n_episodes, seed)
            sr_b, rw_b, dr_b = run_one(cfg, place_balance, n_episodes, seed)
            gap = sr_b - sr_a
            records.append({
                "arrival": ap, "K": k,
                "A_succ%": sr_a, "B_succ%": sr_b, "gap(B-A)": gap,
                "A_reward": rw_a, "B_reward": rw_b,
            })
            tag = ""
            if best is None or gap > best["gap(B-A)"]:
                best = records[-1]
            print(f"arrival={ap:.1f} K={k} | A={sr_a:5.1f}%  B={sr_b:5.1f}%  "
                  f"gap={gap:+5.2f}  (reward A={rw_a:6.1f} B={rw_b:6.1f})")

    df = pd.DataFrame(records)
    print("\n=== gap(B-A) 透视表：行=arrival 列=K ===")
    pivot = df.pivot(index="arrival", columns="K", values="gap(B-A)")
    pd.set_option("display.float_format", lambda x: f"{x:6.2f}")
    print(pivot.to_string())

    print("\n=== B_succ% 透视表（完成率水位，用于判断是否在甜区）===")
    pivot_b = df.pivot(index="arrival", columns="K", values="B_succ%")
    print(pivot_b.to_string())

    print(f"\n>>> 甜区（B-A 最大）: arrival={best['arrival']}, K={best['K']}, "
          f"gap={best['gap(B-A)']:+.2f} 个百分点 "
          f"(A={best['A_succ%']:.1f}% -> B={best['B_succ%']:.1f}%)")
    print("    解读：该组合下'选对 UAV'最值钱，最适合作为凸显扩散优势的实验 setting。")


if __name__ == "__main__":
    main()
