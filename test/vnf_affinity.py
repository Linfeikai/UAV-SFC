"""
VNF↔UAV 亲和实验：place 作为核心变量能否产生大价差？
=================================================
新机制：UAV 分 gpu/general 两型。重计算 VNF(GPU_VNFS)在 general 机上慢
NON_GPU_SLOWDOWN 倍。这让"一条链的每个 VNF 放哪"成为真正的高维决策：
重 VNF 该送 GPU 机，但 GPU 机少且可能离 UE 远 -> 多目标冲突，无贪心通吃。

策略（全部支持逐 VNF 拆分部署）：
  A_nearest  整条链 -> 最近 UAV（亲和盲视，= 现状 place_intent 等价物）
  B_balance  整条链 -> 最近且放得下（亲和盲视 + 负载感知，上个实验的赢家）
  M_split    逐 VNF 拆分：重 VNF->最空 GPU 机，轻 VNF->就近，负载均衡（亲和感知+拆分）

若 M_split 大幅超过 A/B，证明 VNF 拆分部署在异构下真正值钱 -> place 可作核心变量。
"""
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sfc_env import SFCEnv
from place_ablation import place_nearest, place_balance, _build_action

E8 = 1e8


def make_split_policy(gpu_vnfs, uav_types):
    gpu_ids = [i for i, t in enumerate(uav_types) if t == "gpu"]
    gen_ids = [i for i, t in enumerate(uav_types) if t != "gpu"]

    def policy(env):
        K, N = env.K, env.N
        dt = env.time_slot - env.dt_fly
        caps = np.array([u.cpu_freq * dt for u in env.uavs])
        used = np.zeros(N)
        place = np.zeros((K, env.L), dtype=np.int32)
        for k in range(K):
            if k >= len(env.current_cand_tasks):
                continue
            ue_id, sfc = env.current_cand_tasks[k]
            ue_loc = env.ues[ue_id].loc
            for l, vnf in enumerate(sfc.vnf_chain):
                heavy = vnf.vnf_type in gpu_vnfs
                cand = gpu_ids if heavy else list(range(N))
                # 重 VNF：最空 GPU 机；轻 VNF：就近且不太满
                best, best_score = cand[0], -1e18
                for u in cand:
                    slack = caps[u] - used[u]
                    dist = np.linalg.norm(env.uavs[u].loc - ue_loc)
                    # 重任务重余量，轻任务重距离
                    score = (slack * 1e-9) - (0.0 if heavy else dist * 0.01)
                    if score > best_score:
                        best_score, best = score, u
                used[best] += vnf.required_cycles / (1.0 if (heavy and best in gpu_ids) or not heavy else 4.0)
                place[k, l] = best
        return place
    return policy


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


def main(n_ep=40, seed=42):
    uav_types = ["gpu", "general", "gpu", "general"]  # 2 GPU 2 通用
    from core.env_config import DEFAULT_CONFIG
    gpu_vnfs = DEFAULT_CONFIG["GPU_VNFS"]
    split_fn = make_split_policy(gpu_vnfs, uav_types)

    for slowdown in [1.0, 4.0, 8.0]:
        print(f"\n=== NON_GPU_SLOWDOWN={slowdown} (1.0=均质对照) | arrival=0.4 K=5 ===")
        print(f"{'策略':<12}{'succ%':>8}{'reward':>9}")
        cfg_base = {"ARRIVAL_PROB": 0.4, "K": 5, "USE_HARD_CAP": False,
                    "UAV_TYPES": uav_types, "NON_GPU_SLOWDOWN": slowdown}
        results = {}
        for name, fn in [("A_nearest", place_nearest),
                         ("B_balance", place_balance),
                         ("M_split", split_fn)]:
            s, r = run(cfg_base, fn, n_ep, seed)
            results[name] = s
            print(f"{name:<12}{s:8.1f}{r:9.1f}")
        print(f"  M_split - A = {results['M_split']-results['A_nearest']:+.2f}  "
              f"M_split - B = {results['M_split']-results['B_balance']:+.2f}")
    print("\n判读：均质(slowdown=1)行三者应接近；随 slowdown 增大，若 M_split 明显甩开 A/B，")
    print("      说明异构亲和下'VNF拆分部署'产生大价差 -> place 值得作为核心决策变量。")


if __name__ == "__main__":
    main()
