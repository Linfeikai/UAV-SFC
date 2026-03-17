import os
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback


# 导入环境
from core.sfc_env import SFCEnv

# 导入新定义的策略和特征提取器
from algos.diffusion_sac_agent import DiffusionSACAgent
from algos.diffusion_sac_policy import DiffusionSACPolicy
from algos.diffusion_extractor import SFCFeaturesExtractor

from hydra.core.hydra_config import HydraConfig

# 导入启发式
from test.evalu import HeuristicEvaluator

os.environ["OMP_NUM_THREADS"] = "1"

# --- 算法仓库 (Registry) ---
# 只要在这里注册，main 函数逻辑就永远不用改
ALGO_REGISTRY = {
    "PPO": PPO,
    "SAC": SAC,
    "DIFFUSION": DiffusionSACAgent,  # <--- 注册你的新算法
}


def find_latest_exp(algo_name: str, root_dir: str = "experiments") -> str:
    """
    根据算法名找最新的实验目录（基于脚本绝对路径推导，避免cwd影响）
    :param algo_name: 算法名（如DIFFUSION/SAC/PPO）
    :param root_dir: 实验根目录（默认是experiments）
    :return: 最新实验目录的绝对路径
    """
    print(f"🔍 当前运行脚本的工作目录（cwd）：{os.getcwd()}")
    print(f"🔍 main.py 脚本的绝对路径：{os.path.abspath(__file__)}")

    # 获取脚本所在目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接实验根目录的绝对路径，彻底摆脱相对路径依赖
    root_abs_dir = os.path.join(script_dir, root_dir)
    # 拼接算法目录的绝对路径
    algo_dir = os.path.join(root_abs_dir, algo_name.upper())
    if not os.path.exists(algo_dir):
        # 优化报错信息，打印绝对路径方便排查
        raise FileNotFoundError(
            f"未找到算法{algo_name}的实验目录！\n"
            f"期望绝对路径：{algo_dir}\n"
            f"请检查路径是否存在或算法名是否正确"
        )

    # 遍历算法目录下所有实验子目录，过滤掉非目录文件
    exp_dirs = [
        d for d in os.listdir(algo_dir) if os.path.isdir(os.path.join(algo_dir, d))
    ]
    if not exp_dirs:
        raise ValueError(f"算法{algo_name}目录下无实验记录：{algo_dir}")

    # 按目录的修改时间排序（最新的在最后）
    exp_dirs_with_mtime = [
        (d, os.path.getmtime(os.path.join(algo_dir, d))) for d in exp_dirs
    ]
    exp_dirs_with_mtime.sort(key=lambda x: x[1])
    latest_exp_name = exp_dirs_with_mtime[-1][0]

    # 返回绝对路径
    return os.path.join(algo_dir, latest_exp_name)


def load_hydra_config(exp_dir: str) -> dict:
    """
    从实验目录加载Hydra保存的config.yaml配置
    :param exp_dir: 实验目录路径（如sfc_uav_diffusion_full_level2_0304_233551）
    :return: 解析后的配置字典
    """
    config_path = os.path.join(exp_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"该实验目录下未找到配置文件：{config_path}")

    # 用OmegaConf读取Hydra的yaml（兼容DictConfig格式）
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = OmegaConf.load(f)

    # 转成基础的 Python 容器类型（resolve=True 解析变量插值）
    container = OmegaConf.to_container(cfg, resolve=True)

    # 强制类型断言：告诉类型检查器这一定是个 dict，并在运行时做安全兜底
    if not isinstance(container, dict):
        raise TypeError(
            f"解析配置失败：期望得到字典(dict)，但实际得到了 {type(container)}。请检查 yaml 文件内容。"
        )

    return container


def get_flat_config(cfg: dict | DictConfig) -> dict:
    """
    把Hydra配置转换成原代码需要的扁平配置（和训练代码逻辑一致）
    :param cfg: 从config.yaml读取的原始配置字典/DictConfig
    :return: 扁平化后的环境参数字典
    """
    flat_config = {}

    # 合并环境参数
    if "env" in cfg:
        for k, v in cfg["env"].items():
            flat_config[k] = v

    # 合并并重命名UAV参数
    if "uav" in cfg:
        for k, v in cfg["uav"].items():
            flat_config[f"UAV_{k}"] = v

    # 合并并重命名奖励权重
    if "reward" in cfg:
        for k, v in cfg["reward"].items():
            if k in ["W_ENERGY", "W_CHARGE"]:
                flat_config[k] = v
            else:
                flat_config[f"RWD_{k}"] = v

    return flat_config


def run_heuristic_with_prev_config(input_str: str, custom_log_dir: str):
    """复用历史参数运行启发式算法"""
    try:
        # 判断输入是路径还是算法名
        if os.path.isdir(input_str):
            exp_dir = input_str
            print(f"✅ 检测到输入为实验路径，读取：{exp_dir}")
        else:
            algo_name = input_str
            # ========== 关键改动3：调用修复后的find_latest_exp（绝对路径） ==========
            exp_dir = find_latest_exp(algo_name)
            print(f"✅ 检测到输入为算法名{algo_name}，最新实验目录：{exp_dir}")

        # 加载配置并扁平化
        # 1. 加载配置（这里保持从原 exp_dir 读取 .hydra/config.yaml）
        cfg_old = load_hydra_config(exp_dir)

        # 2. 创建你想要的子目录结构
        # 创建：experiments/DIFFUSION/xxx/heuristic_run/heuristic_logs/
        log_dir = os.path.join(custom_log_dir, "heuristic_logs")
        os.makedirs(log_dir, exist_ok=True)

        # 3. (可选) 手动归档本次运行的参数，实现你要求的 .hydra 子文件夹
        param_dir = os.path.join(custom_log_dir, ".hydra")
        os.makedirs(param_dir, exist_ok=True)
        # 将当前的配置写进去，方便以后回溯
        with open(os.path.join(param_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg_old))

        # 4. 运行评估
        env = SFCEnv(config=get_flat_config(cfg_old))
        evaluator = HeuristicEvaluator(env, log_dir)
        evaluator.evaluate(total_steps=20000)

    except Exception as e:
        print(f"❌ 运行失败：{str(e)}")
        raise


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
# 这个装饰器把 main 函数变成 Hydra 的入口，自动从 conf/config.yaml 读取配置，封装成 cfg（DictConfig 类型）传给函数。
# 这里的 config_path 和 config_name 必须对应你 conf 文件夹的文件名
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    if cfg.get(
        "run_heuristic_reuse", False
    ):  # 说明是想复用rl算法配置来走一遍启发式算法
        input_str = cfg.get(
            "heuristic_reuse_input"
        )  # 这个是需要找到的参数，既可以是算法名（如DIFFUSION/SAC/PPO），也可以是具体的实验路径（如experiments/DIFFUSION/DIFF_level2_0305_1710/）
        # 这里的 exp_dir 就会找到那个：experiments/DIFFUSION/DIFF_level2_0305_1710/
        exp_dir = find_latest_exp(input_str)

        # 定义你想要的子目录路径
        # experiments/DIFFUSION/xxx/heuristic_run_0306_1400
        import time

        sub_dir_name = f"heuristic_run_{time.strftime('%m%d_%H%M')}"
        heuristic_output_dir = os.path.join(exp_dir, sub_dir_name)

        # 执行启发式逻辑，并传入这个子目录
        run_heuristic_with_prev_config(input_str, custom_log_dir=heuristic_output_dir)
        return

    # ==========================================
    # 获取正常的 Hydra 运行目录 (模式2 和 模式3 共享)
    # ==========================================
    hydra_exp_dir = HydraConfig.get().run.dir
    os.makedirs(hydra_exp_dir, exist_ok=True)
    flat_config = get_flat_config(cfg)
    algo_str = cfg.algo_name.upper()

    # --- A. 核心缝合逻辑：将层级配置转为扁平配置 ---
    if algo_str == "HEURISTIC":
        print(
            f"\n🚀 [独立模式] 仅运行启发式基准测试 (总步数: {cfg.total_timesteps})..."
        )
        env_heuristic = SFCEnv(config=flat_config)
        # 日志保存在当前生成的 Hydra 实验目录下的 tb_logs 里
        heu_log_dir = os.path.join(hydra_exp_dir, "tb_logs")
        os.makedirs(heu_log_dir, exist_ok=True)
        evaluator = HeuristicEvaluator(env_heuristic, heu_log_dir)
        evaluator.evaluate(total_steps=cfg.total_timesteps)
        return  # <--- 提前 return，跳过后面的 RL 训练

    if cfg.get("run_heuristic_baseline", True):
        print(f"\n🚀 [前置测试] 在训练前先跑一小段基准线作为对比...")
        env_heuristic = SFCEnv(config=flat_config)
        heu_log_dir = os.path.join(hydra_exp_dir, "tb_logs", "HEURISTIC_BASELINE")
        os.makedirs(heu_log_dir, exist_ok=True)
        evaluator = HeuristicEvaluator(env_heuristic, heu_log_dir)
        evaluator.evaluate(total_steps=cfg.get("heuristic_timesteps", 20000))

    # --- 算法初始化时的tensorboard_log路径 ---
    # 原路径：./tb_logs/
    # 新路径：hydra_exp_dir/tb_logs/
    tensorboard_log_dir = os.path.join(hydra_exp_dir, "tb_logs")
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    # --- B. 初始化环境 ---
    # 将缝合好的扁平字典传给环境的 init
    if algo_str not in ALGO_REGISTRY:
        raise ValueError(f"算法 {algo_str} 还没在 ALGO_REGISTRY 里注册")

    env = SFCEnv(config=flat_config)

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
            # --- 【新增】：直接传给 Policy，用于实例化带 Mask 的 Actor ---
            n_uavs=env.config["NUM_UAVS"],  # 确定 Mobility 掩码切片位置
            m_candidates=env.config["M"],  # 确定 Pick 掩码切片位置 [cite: 1]
            core_features_dim=256,  # 对应特征提取器中语义特征的维度
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
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=policy_kwargs,  # 动态传入图纸
        **algo_params,  # 自动填入 learning_rate, batch_size 等
    )
    # --- E. 训练 ---
    print(f"当前模式: {algo_str} | 总步数: {cfg.total_timesteps}")
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=SFCStatsCallback(),
        tb_log_name=f"{cfg.exp_name}",
    )

    model_save_path = os.path.join(hydra_exp_dir, f"{algo_str}_final_model")
    model.save(model_save_path)
    print(f"🎉 训练完成！模型已保存至：{model_save_path}")


# --- 3. 程序入口 ---
if __name__ == "__main__":
    # ！！！注意：这里 main() 括号里不能写任何东西 ！！！
    # Hydra 会自动把 conf 文件夹里的内容填进 cfg 参数里
    main()
