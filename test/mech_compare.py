"""
机制敏感性对比：硬砍(USE_HARD_CAP=True) vs 软竞争(False)
在若干代表性 (arrival, K) 上，看 place 决策(A vs B)的差距是否被放大。
"""
import numpy as np
import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.sfc_env import SFCEnv
from place_ablation import place_nearest, place_balance, _build_action


def run(cfg, place_fn, n_ep, seed):
    env = SFCEnv(config=cfg)
    srs, rws, crashes = [], [], 0
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
            if info.get("perf/crashed"):
                crashes += 1
            done = term or trunc
        srs.append(succ / max(1, gen) * 100)
        rws.append(rw)
    return np.mean(srs), np.mean(rws), crashes


def main(n_ep=40, seed=42):
    cases = [(0.3, 4), (0.4, 5), (0.5, 6)]
    print(f"{n_ep} ep/格 seed={seed}\n")
    print(f"{'case':<12}{'mech':<8}{'A_succ':>8}{'B_succ':>8}{'gap':>7}{'A_rw':>8}{'B_rw':>8}{'crashA':>8}")
    for ap, k in cases:
        for mech, hard in [("hard", True), ("soft", False)]:
            cfg = {"ARRIVAL_PROB": ap, "K": k, "USE_HARD_CAP": hard}
            sa, ra, ca = run(cfg, place_nearest, n_ep, seed)
            sb, rb, cb = run(cfg, place_balance, n_ep, seed)
            print(f"ap{ap} K{k}  {mech:<8}{sa:8.1f}{sb:8.1f}{sb-sa:+7.2f}{ra:8.1f}{rb:8.1f}{ca:8d}")
        print()


if __name__ == "__main__":
    main()
