import os
import torch
import hydra
import numpy as np
from omegaconf import DictConfig

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback

# 导入你的环境和特征提取器
from core.sfc_env import SFCEnv
from algos.diffusion_extractor import SFCFeaturesExtractor


def get_flat_config(cfg: DictConfig) -> dict:
    flat_config = {}
    if "env" in cfg:
        for k, v in cfg["env"].items():
            flat_config[k] = v
    if "uav" in cfg:
        for k, v in cfg["uav"].items():
            flat_config[f"UAV_{k}"] = v
    if "reward" in cfg:
        for k, v in cfg["reward"].items():
            if k in ["W_ENERGY", "W_CHARGE"]:
                flat_config[k] = v
            else:
                flat_config[f"RWD_{k}"] = v
    return flat_config


# ==========================================
# 注入你的自定义回调函数 (SFCStatsCallback)
# ==========================================
class SFCStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # 初始化 Episode 累加器
        self.stats = {
            "completed": 0,
            "dropped": 0,
            "timeout": 0,
            "unpicked": 0,
            "available": 0,
            "picked": 0,
        }

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]

        # 1. 累加基础数据
        self.stats["completed"] += info.get("completed_count", 0)
        self.stats["dropped"] += info.get("dropped_count", 0)
        self.stats["timeout"] += info.get("timeout_count", 0)
        self.stats["unpicked"] += info.get("unpicked_count", 0)
        self.stats["available"] += info.get("total_available", 0)
        self.stats["picked"] += info.get("actually_picked", 0)

        # 2. 当 Episode 结束时计算比例并写入 TB
        if self.locals["dones"][0]:
            # 计算核心科研指标
            total_gen = max(1, self.stats["available"])
            total_pick = max(1, self.stats["picked"])

            # 最终成功率 (成功数 / 总供应)
            success_rate = (self.stats["completed"] / total_gen) * 100
            # 准入转化率 (成功数 / 认领数) -> 反应 Place 动作好坏
            admission_eff = (self.stats["completed"] / total_pick) * 100
            # 认领率 (认领数 / 总供应) -> 反应 Pick 动作好坏
            pick_rate = (self.stats["picked"] / total_gen) * 100

            # 写入 TensorBoard
            self.logger.record("sfc/success_rate_pct", success_rate)
            self.logger.record("sfc/admission_efficiency_pct", admission_eff)
            self.logger.record("sfc/pick_rate_pct", pick_rate)
            self.logger.record("sfc/completed_total", self.stats["completed"])
            self.logger.record("sfc/dropped_admission", self.stats["dropped"])

            # 重置累加器
            for k in self.stats:
                self.stats[k] = 0

        return True


# ==========================================
# 主训练逻辑
# ==========================================
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [SB3 微调] 正在启动 SAC & PPO 在线 RL 训练 (Device: {device})")

    TOTAL_TIMESTEPS = 200_000
    TB_LOG_DIR = "./tensorboard_logs_sb3_finetune"

    flat_config = get_flat_config(cfg)
    flat_config["IS_OVERFIT_TEST"] = False
    env = SFCEnv(config=flat_config)

    # 架构配置字典 (保持绝对干净，防止参数污染)
    base_kwargs = dict(
        features_extractor_class=SFCFeaturesExtractor,
        features_extractor_kwargs=dict(
            n_uavs=env.config["NUM_UAVS"],
            m_candidates=env.config["M"],
            grid_res=env.config["GRID_RES"],
        ),
        share_features_extractor=True,
    )
    sac_kwargs = {
        **base_kwargs,
        "net_arch": dict(pi=[512, 512, 512], qf=[512, 512, 512]),
    }
    ppo_kwargs = {
        **base_kwargs,
        "net_arch": dict(pi=[512, 512, 512], vf=[512, 512, 512]),
    }

    cwd = hydra.utils.get_original_cwd()
    algorithms = ["PPO", "SAC"]

    for algo in algorithms:
        print(f"\n" + "=" * 50)
        print(f"🔥 正在启动 [{algo}] 的在线微调...")
        print("=" * 50)
    # ==========================================
    # 核心修改 1：通过 policy_kwargs 压死 PPO 的初始噪声
    # ==========================================
    base_kwargs = dict(
        features_extractor_class=SFCFeaturesExtractor,
        features_extractor_kwargs=dict(
            n_uavs=env.config["NUM_UAVS"],
            m_candidates=env.config["M"],
            grid_res=env.config["GRID_RES"],
        ),
        share_features_extractor=True,
    )

    sac_kwargs = {
        **base_kwargs,
        "net_arch": dict(pi=[512, 512, 512], qf=[512, 512, 512]),
    }

    ppo_kwargs = {
        **base_kwargs,
        "net_arch": dict(pi=[512, 512, 512], vf=[512, 512, 512]),
        # 💡 救命参数：log_std_init=-3.0 会让初始动作的标准差降到极低 (exp(-3) ≈ 0.05)
        # 这意味着 PPO 几乎会 100% 贴着预训练的专家动作走，不会乱飞。
        "log_std_init": -3.0,
    }

    cwd = hydra.utils.get_original_cwd()
    algorithms = ["PPO", "SAC"]
    for algo in algorithms:
        print(f"\n" + "=" * 50)
        print(f"🔥 正在启动 [{algo}] 的在线微调...")
        print("=" * 50)

        if algo == "PPO":
            # ==========================================
            # 核心修改 2：降学习率，关掉额外探索 (ent_coef)
            # ==========================================
            model = PPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=ppo_kwargs,
                learning_rate=1e-5,  # 💡 极低学习率，防止剧烈震荡
                ent_coef=0.0,  # 💡 彻底关闭 PPO 的额外熵奖励 (不需要探索了)
                clip_range=0.1,  # 💡 收紧 PPO 的更新步幅，强制信任预训练权重
                tensorboard_log=TB_LOG_DIR,
                device=device,
            )

            weight_path = os.path.join(cwd, "pretrained_ppo_actor.pth")
            if os.path.exists(weight_path):
                model.policy.load_state_dict(
                    torch.load(weight_path, map_location=device)
                )
                print("✅ 已成功注入 PPO 预训练权重！(探索噪声已压至最低)")

        elif algo == "SAC":
            # ==========================================
            # 核心修改 3：锁死 SAC 的熵系数 (ent_coef)
            # ==========================================
            model = SAC(
                "MultiInputPolicy",
                env,
                policy_kwargs=sac_kwargs,
                learning_rate=1e-5,  # 💡 极低学习率
                ent_coef=0.001,  # 💡 固定为一个极小值，不要用 "auto" (会引发剧烈变动)
                learning_starts=5000,  # 让 Replay Buffer 填入一些高质量的预训练轨迹
                tensorboard_log=TB_LOG_DIR,
                device=device,
            )

            weight_path = os.path.join(cwd, "pretrained_sac_actor.pth")
            if os.path.exists(weight_path):
                model.policy.load_state_dict(
                    torch.load(weight_path, map_location=device)
                )
                print("✅ 已成功注入 SAC 预训练权重！(熵系数已锁死)")
        print(f"📈 开始与环境交互，目标步数: {TOTAL_TIMESTEPS} ...")
        # 挂载你的统计回调
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=SFCStatsCallback(),
            tb_log_name=f"Finetune_Pretrained_{algo}",
        )

        print(f"🏁 [{algo}] 训练完毕！")


if __name__ == "__main__":
    main()
