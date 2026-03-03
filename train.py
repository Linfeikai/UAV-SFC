from stable_baselines3 import SAC  # 假设你使用的是 SAC 或其他 SB3 算法
from algos.diffusion_sac_agent import DiffusionSACAgent
from algos.diffusion_sac_policy import DiffusionSACPolicy  # 你的策略类
from algos.diffusion_extractor import SFCFeaturesExtractor  # 你的策略类

# 导入你的环境
from core.sfc_env import SFCEnv

# 1. 实例化环境
env = SFCEnv()


def test_pipeline():
    # 1. 初始化环境
    env = SFCEnv()

    # 2. 定制图纸 (包含环境的真实参数)
    policy_kwargs = dict(
        features_extractor_class=SFCFeaturesExtractor,
        features_extractor_kwargs=dict(n_uavs=4, m_candidates=6, grid_res=3),
        share_features_extractor=True,  # 开启我们刚修好的共享魔法！
        T=20,  # 传给 Actor 的扩散步数
        net_arch=[256, 256],
    )

    model = DiffusionSACAgent(
        policy=DiffusionSACPolicy, env=env, policy_kwargs=policy_kwargs
    )
    print("模型初始化成功！开始点火测试...")
    # 4. 只跑 100 步，验证环境交互、特征提取、Actor生动动作、Critic算Q值、梯度回传是否畅通
    model.learn(total_timesteps=100)
    print("点火测试完美通过！没有 Shape Mismatch 报错！")


if __name__ == "__main__":
    test_pipeline()
