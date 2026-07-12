# import wandb
# from wandb.integration.sb3 import WandbCallback  # Wandb 官方提供的 SB3 回调
# from stable_baselines3.common.logger import configure

# # 1. 初始化 Wandb
# run = wandb.init(
#     project="UAV-Diffusion-SAC",  # 你的项目名字
#     name="Finetune_QNE_Expert_Noise",  # 这次实验的名字
#     sync_tensorboard=True,  # 💡 核心魔法：自动同步 SB3 的 TensorBoard 输出！
#     monitor_gym=True,  # 自动记录环境的视频（如果你的 env 支持 render）
#     save_code=True,  # 备份你的代码，方便日后复现
# )

# # 2. 设置 SB3 的输出路径 (Wandb 会监听这个路径)
# tensorboard_log_dir = f"runs/{run.id}"

# # 3. 实例化你的 Agent
# model = DiffusionSACAgent(
#     env=env,
#     tensorboard_log=tensorboard_log_dir,  # 必须指定这个路径
#     # ... 其他参数
# )

# # 4. 创建 WandbCallback (它会自动帮你记录网络权重、梯度分布等硬核数据)
# wandb_callback = WandbCallback(
#     gradient_save_freq=1000,
#     model_save_path=f"models/{run.id}",
#     verbose=2,
# )

# # 5. 把 WandbCallback 和你的 Freeze 回调一起塞进去
# model.learn(
#     total_timesteps=100000,
#     callback=[wandb_callback, freeze_callback],  # 挂载回调
# )

# run.finish()

# import wandb

# wandb.login()

# # Project that the run is recorded to
# project = "my-awesome-project"

# # Dictionary with hyperparameters
# config = {"epochs": 10, "lr": 0.01}

# with wandb.init(project=project, config=config) as run:
#     # Training code here
#     # Log values to W&B with run.log()
#     run.log({"accuracy": 0.9, "loss": 0.1})

import pandas as pd

# 替换成你下载的文件实际路径
file_path = "episode_000000.parquet"

# 读取数据
df = pd.read_parquet(file_path)

# 1. 查看一共有哪些列（这是适配 config.py 最关键的信息）
print("所有列名：", df.columns.tolist())

# 2. 查看前几行数据
print("\n数据预览：")
print(df.head())

# 3. 查看动作维度（检查是不是 14 维）
# 假设列名叫 'action'
if "action" in df.columns:
    sample_action = df["action"].iloc[0]
    print(f"\n动作维度: {len(sample_action)}")
    print(f"动作内容: {sample_action}")
