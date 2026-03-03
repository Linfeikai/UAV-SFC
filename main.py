import hydra
from omegaconf import DictConfig
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback

# 导入环境
from core.sfc_env import SFCEnv

# 导入新定义的策略和特征提取器
from algos.diffusion_sac_agent import DiffusionSACAgent
from algos.diffusion_sac_policy import DiffusionSACPolicy
from algos.diffusion_extractor import SFCFeaturesExtractor

# --- 算法仓库 (Registry) ---
# 只要在这里注册，main 函数逻辑就永远不用改
ALGO_REGISTRY = {
    "PPO": PPO,
    "SAC": SAC,
    "DIFFUSION": DiffusionSACAgent,  # <--- 注册你的新算法
}


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


# --- 2. 训练主函数 ---
# 这里的 config_path 和 config_name 必须对应你 conf 文件夹的文件名
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # --- A. 核心缝合逻辑：将层级配置转为扁平配置 ---
    # 这一步是为了让你 YAML 里的 uav.CPU_FREQ 变成环境认的 UAV_CPU_FREQ
    flat_config = {}

    # 合并环境参数
    for k, v in cfg.env.items():
        flat_config[k] = v

    # 合并并重命名 UAV 参数
    for k, v in cfg.uav.items():
        flat_config[f"UAV_{k}"] = v

    # 合并并重命名奖励权重
    for k, v in cfg.reward.items():
        if k in ["W_ENERGY", "W_CHARGE"]:
            flat_config[k] = v
        else:
            flat_config[f"RWD_{k}"] = v

    # --- B. 初始化环境 ---
    # 将缝合好的扁平字典传给环境的 init
    env = SFCEnv(config=flat_config)

    # --- B. 算法动态获取 ---
    algo_str = cfg.algo_name.upper()

    if algo_str not in ALGO_REGISTRY:
        raise ValueError(f"算法 {algo_str} 还没在 ALGO_REGISTRY 里注册呢！")
    # --- C. 初始化算法 ---
    # 注意：tensorboard_log 使用相对路径，Hydra 会自动在 outputs 文件夹下创建它
    # 获取算法类
    AlgoClass = ALGO_REGISTRY[algo_str]
    # 获取对应算法的超参数，并将其转换为普通的 Python 字典
    algo_params = dict(cfg.get(algo_str.lower(), {}))
    if algo_str == "DIFFUSION":
        # 如果是扩散模型，使用专属的 Policy 和 特征提取器图纸
        policy_class = DiffusionSACPolicy

        # 动态读取 Hydra 配置里的环境参数来构建 kwargs
        policy_kwargs = dict(
            features_extractor_class=SFCFeaturesExtractor,
            features_extractor_kwargs=dict(
                n_uavs=env.config["NUM_UAVS"],  # <--- 改成 env.config
                m_candidates=env.config["M"],  # <--- 改成 env.config
                grid_res=env.config["GRID_RES"],  # <--- 改成 env.config
            ),
            share_features_extractor=True,
            T=20,  # Actor 的扩散步数
            net_arch=[256, 256],
        )
    else:
        # 如果是原生 SAC/PPO，回退到默认设置
        policy_class = "MlpPolicy"
        policy_kwargs = {}
    # --- D. 统一实例化 ---
    model = AlgoClass(
        policy=policy_class,  # 动态传入 Policy
        env=env,
        verbose=1,
        seed=cfg.seed,
        tensorboard_log="./tb_logs/",
        policy_kwargs=policy_kwargs,  # 动态传入图纸
        **algo_params,  # 自动填入 learning_rate, batch_size 等
    )
    # --- E. 训练 ---
    print(f"当前模式: {algo_str} | 总步数: {cfg.total_timesteps}")
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=SFCStatsCallback(),
        tb_log_name=f"{cfg.exp_name}_{algo_str}",
    )

    model.save(f"{algo_str}_final_model")


# --- 3. 程序入口 ---
if __name__ == "__main__":
    # ！！！注意：这里 main() 括号里不能写任何东西 ！！！
    # Hydra 会自动把 conf 文件夹里的内容填进 cfg 参数里
    main()
