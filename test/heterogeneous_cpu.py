"""
异构算力实验：能否打破"守恒"，让 place 决策产生真实价差？
=================================================
前三个实验结论：均质算力下，A(最近邻) 与 B(负载均衡) 决策虽分化，
但把任务从一个坑挪到另一个坑总损失守恒 -> B 赢不了 A。

假设：给 UAV 异构算力后，"把重任务放到强 UAV"是净赚的、不守恒。
     A 只看距离、对算力盲视 -> 系统性放错；
     B 判断可行性时读 u.cpu_freq、算力可见 -> 净胜。
若 B-A 拉开到两位数，说明 place 决策价差被激活。

对照：总算力守恒(4 架加起来都是 8e8)，只改分布均匀度。
  homo:   [2.0, 2.0, 2.0, 2.0] e8   (均质基线)
  mild:   [1.4, 1.8, 2.2, 2.6] e8   (轻度异构)
  strong: [0.8, 1.4, 2.6, 3.2] e8   (强异构)
  extreme:[0.5, 1.0, 2.5, 4.0] e8   (极端异构)
"""
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sfc_env import SFCEnv
from place_ablation import place_nearest, place_balance, _build_action

E8 = 1e8
PROFILES = {
    "homo":    [2.0, 2.0, 2.0, 2.0],
    "mild":    [1.4, 1.8, 2.2, 2.6],
    "strong":  [0.8, 1.4, 2.6, 3.2],
    "extreme": [0.5, 1.0, 2.5, 4.0],
}


def place_strongest(env):
    """C. 优先选'最强且放得下'的 UAV(价差上限参照)。"""
    K, L, N = env.K, env.L, env.N
    place = np.zeros((K, L), dtype=np.int32)
    dt = env.time_slot - env.dt_fly
    caps = np.array([u.cpu_freq * dt for u in env.uavs])
    used = np.zeros(N)
    order = np.argsort(-np.array([u.cpu_freq for u in env.uavs]))  # 强->弱
    for k in range(K):
        if k < len(env.current_cand_tasks):
            _, sfc = env.current_cand_tasks[k]
            chosen = order[0]
            for u in order:
                if used[u] + sfc.total_cycles < caps[u]:
                    chosen = u
                    break
            used[chosen] += sfc.total_cycles
            place[k, :] = chosen
    return place


def run(cfg, place_fn, n_ep, seed):
    env = SFCEnv(config=cfg)
    srs, rws = [], []
    for ep in range(n_ep):
        env.reset(seed=seed + ep)
        gen = succ = 0
        rw = 0.0
        done = False
        while not done:
            a = _build_action(env, place_fn(env))
            _, r, term, trunc, info = env.step(a)
            gen += info.get("total_available", 0)
            succ += info.get("completed_count", 0)
            rw += r
            done = term or trunc
        srs.append(succ / max(1, gen) * 100)
        rws.append(rw)
    return np.mean(srs), np.mean(rws)


def main(n_ep=40, seed=42, arrival=0.4, K=5):
    print(f"异构算力实验 | arrival={arrival} K={K} | {n_ep}ep seed={seed}")
    print("总算力守恒(sum=8e8)，仅改分布\n")
    print(f"{'profile':<10}{'A_succ':>8}{'B_succ':>8}{'C_succ':>8}{'B-A':>7}{'C-A':>7}{'A_rw':>8}{'B_rw':>8}{'C_rw':>8}")
    for name, prof in PROFILES.items():
        freqs = [f * E8 for f in prof]
        cfg = {"ARRIVAL_PROB": arrival, "K": K, "UAV_CPU_FREQS": freqs, "USE_HARD_CAP": False}
        sa, ra = run(cfg, place_nearest, n_ep, seed)
        sb, rb = run(cfg, place_balance, n_ep, seed)
        sc, rc = run(cfg, place_strongest, n_ep, seed)
        print(f"{name:<10}{sa:8.1f}{sb:8.1f}{sc:8.1f}{sb-sa:+7.2f}{sc-sa:+7.2f}{ra:8.1f}{rb:8.1f}{rc:8.1f}")
    print("\n判读：homo 行 B-A 应≈0(复现旧结论)。若异构行 B-A / C-A 随异构程度显著增大，")
    print("      说明'选对UAV'产生真实价差 -> place 决策变得值钱 -> 扩散有优势可争。")


if __name__ == "__main__":
    main()
