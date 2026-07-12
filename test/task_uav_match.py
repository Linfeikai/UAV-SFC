"""
任务异构 × UAV异构 组合实验
=================================================
洞察：任务本身天然异构 —— Video链(ObjectDetection, ~225 cyc/bit)是"重任务"，
     IoT/Security链轻得多。上个实验价差偏小(2.7%)，原因是 UAV 强机太少一填就满。
本实验：加大 UAV 异构到"2强2弱"，并扫 arrival 找"强机够用不过剩"的甜区，
       同时量化【匹配质量】：重任务是否被送到强机。

策略：A=最近邻(算力盲视)  B=负载均衡(算力可见)  M=匹配感知(重任务优先给强机)
匹配指标 match%：被pick任务里，total_cycles 前50%的"重任务"，其部署目标UAV
              的 cpu_freq 也在前50%(强机) 的比例。越高=匹配越好。
"""
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sfc_env import SFCEnv
from place_ablation import place_nearest, place_balance, _build_action

E8 = 1e8
# 2强2弱，总算力守恒 = 8e8
PROFILES = {
    "homo":     [2.0, 2.0, 2.0, 2.0],
    "2s2w":     [1.0, 1.0, 3.0, 3.0],
    "2s2w_ext": [0.6, 0.6, 3.4, 3.4],
}


def place_match(env):
    """M. 匹配感知：按任务从重到轻，依次分配给'当前最空的强机优先'。"""
    K, L, N = env.K, env.L, env.N
    dt = env.time_slot - env.dt_fly
    caps = np.array([u.cpu_freq * dt for u in env.uavs])
    freqs = np.array([u.cpu_freq for u in env.uavs])
    used = np.zeros(N)
    place = np.zeros((K, L), dtype=np.int32)

    tasks = [(k, env.current_cand_tasks[k][1]) for k in range(K)
             if k < len(env.current_cand_tasks)]
    # 重任务优先决策
    tasks.sort(key=lambda t: -t[1].total_cycles)
    for k, sfc in tasks:
        # 评分：能放下的里，freq 高且剩余多者优先
        best, best_score = 0, -1e18
        for u in range(N):
            slack = caps[u] - used[u] - sfc.total_cycles
            score = freqs[u] + slack * 1e-9  # 强机优先，其次看余量
            if slack > 0 and score > best_score:
                best_score, best = score, u
        used[best] += sfc.total_cycles
        place[k, :] = best
    return place


def run(cfg, place_fn, n_ep, seed):
    env = SFCEnv(config=cfg)
    N = env.N
    srs, rws = [], []
    heavy_hit = heavy_tot = 0
    freqs = np.array([f for f in cfg["UAV_CPU_FREQS"]])
    strong_thresh = np.median(freqs)
    for ep in range(n_ep):
        env.reset(seed=seed + ep)
        gen = succ = 0
        rw = 0.0
        done = False
        while not done:
            pm = place_fn(env)
            # 匹配质量：本步 pick 任务的重/轻 vs 目标机强/弱
            picked = [(k, env.current_cand_tasks[k][1]) for k in range(env.K)
                      if k < len(env.current_cand_tasks)]
            if picked:
                cyc = np.array([s.total_cycles for _, s in picked])
                cyc_med = np.median(cyc)
                for (k, s) in picked:
                    if s.total_cycles >= cyc_med:  # 重任务
                        heavy_tot += 1
                        if freqs[pm[k, 0]] >= strong_thresh:
                            heavy_hit += 1
            a = _build_action(env, pm)
            _, r, term, trunc, info = env.step(a)
            gen += info.get("total_available", 0)
            succ += info.get("completed_count", 0)
            rw += r
            done = term or trunc
        srs.append(succ / max(1, gen) * 100)
        rws.append(rw)
    match = heavy_hit / max(1, heavy_tot) * 100
    return np.mean(srs), np.mean(rws), match


def main(n_ep=40, seed=42):
    for arrival, K in [(0.3, 4), (0.4, 5)]:
        print(f"\n=== arrival={arrival} K={K} | {n_ep}ep ===")
        print(f"{'profile':<10}{'A':>6}{'B':>6}{'M':>6}{'B-A':>7}{'M-A':>7}"
              f"{'A_mt%':>7}{'B_mt%':>7}{'M_mt%':>7}")
        for name, prof in PROFILES.items():
            freqs = [f * E8 for f in prof]
            cfg = {"ARRIVAL_PROB": arrival, "K": K,
                   "UAV_CPU_FREQS": freqs, "USE_HARD_CAP": False}
            sa, ra, ma_ = run(cfg, place_nearest, n_ep, seed)
            sb, rb, mb_ = run(cfg, place_balance, n_ep, seed)
            sm, rm, mm_ = run(cfg, place_match, n_ep, seed)
            print(f"{name:<10}{sa:6.1f}{sb:6.1f}{sm:6.1f}{sb-sa:+7.2f}{sm-sa:+7.2f}"
                  f"{ma_:7.1f}{mb_:7.1f}{mm_:7.1f}")
    print("\n判读：mt%=重任务送到强机的比例。若 M(匹配感知) 的成功率和 mt% 明显高于 A，")
    print("      且价差随异构增大 -> '重任务↔强机'匹配是真正值钱的决策 -> 扩散主场。")


if __name__ == "__main__":
    main()
