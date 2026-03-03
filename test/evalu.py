import os
import hydra
import numpy as np
from omegaconf import DictConfig
from stable_baselines3.common.logger import configure

# 导入你的环境和组件
from core.sfc_env import SFCEnv
from main import SFCStatsCallback


def smart_heuristic_policy(env):
    """
    适配 62 维语义意图空间的智能规则策略
    维度分布: 8 (移动) + 6 (挑选) + 48 (部署意图: 6任务 * 4VNF * 2坐标)
    """
    K, L, M, N = env.K, env.L, env.M, env.N
    width = env.config["GROUND_WIDTH"]
    height = env.config["GROUND_HEIGHT"]

    # --- A. 移动逻辑 (保持原样，输出 8 维) ---
    mobility_actions = []
    charger_loc = env.chargers[0].loc
    for i, uav in enumerate(env.uavs):
        if uav.e_battery < uav.battery_capacity * 0.25:
            target = charger_loc
        elif env.current_cand_tasks:
            target = env.ues[env.current_cand_tasks[0][0]].loc
        else:
            target = uav.loc
        diff = target - uav.loc
        dist = np.linalg.norm(diff) + 1e-9
        mobility_actions.extend(diff / dist)

    # --- B. 选人逻辑 (保持原样，输出 6 维) ---
    pick_actions = []
    for k in range(K):
        if k < len(env.current_cand_tasks):
            raw_pick = (k / (M + 1)) * 2 - 1 + 0.001
        else:
            raw_pick = (M / (M + 1)) * 2 - 1 + 0.001
        pick_actions.append(raw_pick)

    # --- C. 部署逻辑 (核心修改：输出 48 维坐标意图) ---
    place_intent_actions = []
    uav_capacity_used = np.zeros(N)
    dt_compute = env.time_slot - env.dt_fly
    uav_caps = np.array([u.cpu_freq * dt_compute for u in env.uavs])

    for k in range(K):
        if k < len(env.current_cand_tasks):
            ue_id, sfc = env.current_cand_tasks[k]
            ue_loc = env.ues[ue_id].loc

            # 寻找最佳 UAV (逻辑不变：不超载且离 UE 最近)
            best_uav_idx = -1
            min_dist = float("inf")
            for u_id in range(N):
                if (
                    not env.uavs[u_id].is_crashed
                    and uav_capacity_used[u_id] + sfc.total_cycles < uav_caps[u_id]
                ):
                    d = np.linalg.norm(env.uavs[u_id].loc - ue_loc)
                    if d < min_dist:
                        min_dist = d
                        best_uav_idx = u_id

            if best_uav_idx == -1:
                best_uav_idx = np.argmin(uav_capacity_used)

            uav_capacity_used[best_uav_idx] += sfc.total_cycles

            # --- 关键：将该 UAV 的物理位置映射到 [-1, 1] 意图空间 ---
            target_uav = env.uavs[best_uav_idx]
            norm_x = (target_uav.loc[0] / width) * 2 - 1
            norm_y = (target_uav.loc[1] / height) * 2 - 1

            # 为该任务的 L=4 个 VNF 生成同样的意图坐标 (共 4*2=8 个值)
            for _ in range(L):
                place_intent_actions.extend([norm_x, norm_y])
        else:
            # 填充位：如果不选任务，输出 [0, 0] 坐标意图
            place_intent_actions.extend([0.0, 0.0] * L)

    # 最终返回 8 + 6 + 48 = 62 维向量
    return np.concatenate([mobility_actions, pick_actions, place_intent_actions])


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

        episode_rewards = []
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

                # 记录总分
                episode_rewards.append(current_episode_reward)
                self.logger.record(
                    "rollout/ep_rew_mean", np.mean(episode_rewards[-100:])
                )

                # 强制写入
                self.logger.dump(step)

                # 重置
                self.reset_stats()
                obs, _ = self.env.reset()
                current_episode_reward = 0
            else:
                obs = next_obs

            if step % 10000 == 0:
                print(
                    f"进度: {step}/{total_steps} | 平均奖励: {np.mean(episode_rewards[-10:] if episode_rewards else 0):.2f}"
                )


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
