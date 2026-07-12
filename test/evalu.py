import os
import hydra
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.logger import configure

# 导入你的环境和组件
from core.sfc_env import SFCEnv


def smart_heuristic_policy(env):
    """
    亲和感知 + 逐 VNF 拆分 的强启发式策略（M_split 级别）。
    - 移动：低电回充电桩，否则追最紧急任务的 UE。
    - Pick：选候选池最紧急的前 K 个。
    - Place：逐 VNF 拆分部署——重计算 VNF(GPU_VNFS) 送最空的 GPU 机，
             轻 VNF 就近且避免过载。异构下这是能规避"放错加速器"惩罚的关键。

    直接读 env 内部状态（current_cand_tasks / vnf_chain / _vnf_affinity），
    动作用"目标 UAV 当前坐标"编码，经 step 的最近邻映射精确命中该 UAV。
    维度: N*2(移动) + K(挑选) + K*L*2(逐VNF部署意图)。
    """
    K, L, M, N = env.K, env.L, env.M, env.N
    W = env.config["GROUND_WIDTH"]
    H = env.config["GROUND_HEIGHT"]
    gpu_vnfs = env.config.get("GPU_VNFS", set())
    uav_types = env.config.get("UAV_TYPES", None)
    gpu_ids = [i for i in range(N) if uav_types is not None and uav_types[i] == "gpu"]
    charger_loc = env.chargers[0].loc

    # --- A. 移动 ---
    mobility_actions = []
    primary_target = None
    if env.current_cand_tasks:
        primary_target = env.ues[env.current_cand_tasks[0][0]].loc
    for uav in env.uavs:
        if uav.is_crashed:
            target = uav.loc
        elif uav.e_battery < uav.battery_capacity * 0.25:
            target = charger_loc
        elif primary_target is not None:
            target = primary_target
        else:
            target = uav.loc
        diff = np.asarray(target, dtype=np.float64) - uav.loc
        dist = np.linalg.norm(diff) + 1e-9
        mobility_actions.extend(diff / dist)

    # --- B. Pick：最紧急的前 K 个 ---
    num_valid = len(env.current_cand_tasks)
    pick_actions = []
    for k in range(K):
        if k < num_valid:
            raw_pick = (k / (M + 1)) * 2 - 1 + 0.001
        else:
            raw_pick = (M / (M + 1)) * 2 - 1 + 0.001
        pick_actions.append(raw_pick)

    # --- C. Place：逐 VNF 拆分 + 亲和感知 ---
    dt = env.time_slot - env.dt_fly
    caps = np.array([u.cpu_freq * dt for u in env.uavs], dtype=np.float64)
    used = np.zeros(N)
    place_intent_actions = []
    for k in range(K):
        if k >= num_valid:
            place_intent_actions.extend([0.0, 0.0] * L)
            continue
        ue_id, sfc = env.current_cand_tasks[k]
        ue_loc = env.ues[ue_id].loc
        for l in range(L):
            if l < len(sfc.vnf_chain):
                vnf = sfc.vnf_chain[l]
                heavy = vnf.vnf_type in gpu_vnfs
                cand = gpu_ids if (heavy and gpu_ids) else list(range(N))
                cand = [u for u in cand if not env.uavs[u].is_crashed] or list(range(N))
                # 有效占用(放错加速器则占用更多)
                best, best_score = cand[0], -1e18
                for u in cand:
                    slack = caps[u] - used[u]
                    dist = np.linalg.norm(env.uavs[u].loc - ue_loc)
                    # 重任务重余量，轻任务重距离
                    score = slack * 1e-9 - (0.0 if heavy else dist * 0.01)
                    if score > best_score:
                        best_score, best = score, u
                aff = env._vnf_affinity(best, vnf)
                used[best] += vnf.required_cycles / max(aff, 1e-9)
                tx = env.uavs[best].loc
                place_intent_actions.extend(
                    [(tx[0] / W) * 2 - 1, (tx[1] / H) * 2 - 1]
                )
            else:
                place_intent_actions.extend([0.0, 0.0])

    return np.concatenate(
        [mobility_actions, pick_actions, place_intent_actions]
    ).astype(np.float32)


class HeuristicEvaluator:
    def __init__(self, env, log_path):
        self.env = env
        # 直接初始化并配置 Logger
        self.logger = configure(log_path, ["stdout", "tensorboard"])

        # 初始化统计累加器 (完全复用你 Callback 里的逻辑)
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            "completed": 0,
            "dropped": 0,
            "timeout": 0,
            "unpicked": 0,
            "available": 0,
            "picked": 0,
        }

    def evaluate(self, total_steps=100000):
        print(f"开始评估启发式策略，总步数: {total_steps}...")
        obs, _ = self.env.reset()

        # 用于存储每个【完整】Episode 最终指标的容器
        all_ep_success_rates = []
        all_ep_admission_effs = []
        all_ep_rewards = []
        
        current_episode_reward = 0  

        for step in range(1, total_steps + 1):
            # 1. 获取启发式动作
            action = smart_heuristic_policy(self.env)

            # 2. 环境交互
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            current_episode_reward += reward

            # 3. 累加基础数据
            self.stats["completed"] += info.get("completed_count", 0)
            self.stats["dropped"] += info.get("dropped_count", 0)
            self.stats["timeout"] += info.get("timeout_count", 0)
            self.stats["unpicked"] += info.get("unpicked_count", 0)
            self.stats["available"] += info.get("total_available", 0)
            self.stats["picked"] += info.get("actually_picked", 0)

            if done:
                # --- 计算并记录科研指标 ---
                total_gen = max(1, self.stats["available"])
                total_pick = max(1, self.stats["picked"])

                success_rate = (self.stats["completed"] / total_gen) * 100
                admission_eff = (self.stats["completed"] / total_pick) * 100
                pick_rate = (self.stats["picked"] / total_gen) * 100

                # 记录到 TensorBoard
                self.logger.record("sfc/success_rate_pct", success_rate)
                self.logger.record("sfc/admission_efficiency_pct", admission_eff)
                self.logger.record("sfc/pick_rate_pct", pick_rate)
                self.logger.record("sfc/completed_total", self.stats["completed"])
                self.logger.record("sfc/dropped_admission", self.stats["dropped"])

                all_ep_success_rates.append(success_rate)
                all_ep_admission_effs.append(admission_eff)
                all_ep_rewards.append(current_episode_reward)

                # 记录总分
                self.logger.record(
                    "rollout/ep_rew_mean", np.mean(all_ep_rewards[-100:])
                )

                # 强制写入
                self.logger.dump(step)

                # 重置
                self.reset_stats()
                obs, _ = self.env.reset()
                current_episode_reward = 0
            else:
                obs = next_obs

        # =======================================================
        # 循环结束：只对【完整】的 Episode 取平均
        # =======================================================
        if len(all_ep_rewards) > 0:
            final_mean_sr = np.mean(all_ep_success_rates)
            final_mean_ae = np.mean(all_ep_admission_effs)
            final_mean_rew = np.mean(all_ep_rewards)

            print(f"\n📊 评估总结 (共完成 {len(all_ep_rewards)} 个完整 Episode):")
            print(f"平均成功率: {final_mean_sr:.2f}% | 平均奖励: {final_mean_rew:.2f}")

            # 写入 TensorBoard 最终对比项
            self.logger.record("final_avg/success_rate", final_mean_sr)
            self.logger.record("final_avg/admission_efficiency", final_mean_ae)
            self.logger.record("final_avg/reward", final_mean_rew)
            self.logger.dump(total_steps)
        else:
            print("⚠️ 警告：设定的 total_steps 太短，未能完成任何一个完整的 Episode！")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # --- 扁平化配置逻辑 ---
    flat_config = {}
    for k, v in cfg.env.items():
        flat_config[k] = v
    for k, v in cfg.uav.items():
        flat_config[f"UAV_{k}"] = v
    for k, v in cfg.reward.items():
        if k in ["W_ENERGY", "W_CHARGE"]:
            flat_config[k] = v
        else:
            flat_config[f"RWD_{k}"] = v

    # 设置对比基准参数
    flat_config["RWD_CRASH"] = -300.0
    flat_config["RWD_SUCCESS"] = 20.0

    env = SFCEnv(config=flat_config)

    # 路径处理
    base_path = hydra.utils.get_original_cwd()
    log_dir = os.path.join(
        base_path, "experiments", "heuristic_baseline", "crash300_succ20"
    )

    evaluator = HeuristicEvaluator(env, log_dir)
    evaluator.evaluate(total_steps=cfg.total_timesteps)


if __name__ == "__main__":
    main()
