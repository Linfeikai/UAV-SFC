# Stationary 续训与确认：2026-10-21

用户已授权迁移至 connect.westb.seetacloud.com:14379 并继续实验。原数据盘、模型和 Python 环境均保留，GPU 为 NVIDIA GeForce RTX 4090（49140 MiB），驱动 595.71.05。当前 runtime 的 Torch 2.12.1+cu130、NumPy 2.4.6、OmegaConf 2.3.1、SB3 2.7.1 与原服务器保存的 runtime lock 一致。认证材料仅保存在本机 SSH 身份文件和服务器原有凭据存储，未写入实验产物。

## 续训前 50 集确认

四策略均使用原训练 `.hydra/config.yaml`，不覆盖奖励、环境或物理参数。种子为 4000–4049，每策略 50 个 episode；各策略按相同种子分别重置 Python、NumPy、Torch 与环境。学习策略使用默认学习运动头；只有静态启发式设 `mobility_source=zero`。所选源模型 SHA256 为 `2106f25d3f9b2cfcbdb775d2549fb3322aae72ea45ed322adb0053c083720fa2`。Hydra 配置 SHA256 为 `2ae61b7f8d7f225b283a9a343739f7139388c32d12f86382815d88c9dc7f43f5`。

| 策略 | 完成率 % | 坠毁率 % | 平均回报 | 能耗 J/step | 存活步数 | 已完成 SFC 平均时延（环境时间单位） |
|---|---:|---:|---:|---:|---:|---:|
| 学习模型（20k 所选 checkpoint） | 52.7273 | 0.0000 | 643.8442 | 765.2367 | 40.0000 | 2.4574 |
| 静态全状态零移动启发式 | 53.1241 | 0.0000 | 648.9748 | 769.0846 | 40.0000 | 2.4536 |
| 全状态启发式 | 48.9906 | 12.0000 | 584.1754 | 2438.3557 | 39.6800 | 2.5009 |
| 仅观测启发式 | 47.8608 | 16.0000 | 564.2583 | 2607.1432 | 39.5000 | 2.4365 |

学习模型比最强静态对照低 0.3968 个百分点和 5.1306 平均回报，仍未达到完成率至少高 0.5 个百分点、坠毁率 ≤5%、平均回报不低于对照的联合目标。50 集零坠毁是本批观察，最终仍需要新种子与独立训练复现。保存结果在 `experiments/server_20261021/stationary_confirmation_20261021_02/`，完整确认耗时约 82.5 秒。

基于安全性、40 步存活和与最强对照接近的差距，决策文件 `experiments/server_20261021/continuation_decision.json` 允许原定有界 30k 续训。初次预检查因 Windows Git 归档输出 CRLF、预期 Git blob 为 LF 而停止，尚未开始任何 episode；已核对规范化归档与 Git 原文完全一致，并保留原始与规范化哈希。该预检查记录单独保存，不算作训练失败。

## 当前续训

- 科学源码独立导出版本 `c5bd80a`，根目录 `/root/autodl-tmp/UAV-SFC-eval-c5bd80a`。未覆盖原 `e150a29` 科学源和 checkpoint；main、算法、环境与原版本一致，新评估器补充静态对照并禁止对学习模型强制零运动。
- W&B run `f3b92a71` 已真实启动；日志确认加载 20000 步源模型、重新填充 5000 步 replay warmup，`bc_pretrain_steps=0`。
- 计划 `experiments/server_20261021/resume_plan.json`，训练目录 `server_runs/stationary_resume_50k_seed42`，外层控制器目录 `server_runs/stationary_resume_control_20261021`，底层管线目录 `server_runs/stationary_resume_pipeline_20261021`，均位于新源码根目录。
- SB3 `learn` 默认重置 `num_timesteps`；最终文件的 30000 步是新增训练步，加源 20000 步约为 50k。新 warmup 另产生 5000 步环境交互，旧 replay 未恢复。
- 训练保留原 Stationary 配置、`env=heterogeneous_hybrid`、奖励、物理限制、8 个并行环境；检查点及5集训练验证每5000步运行。最终候选用5000–5009选择，再用6000–6049进行50集确认，外层自动补齐同种子静态零移动对照。
- 不将底层管线完成等同于整个实验完成：必须等待外层静态对照、联合目标比较和证据归档；本次更新时尚在训练，不声称达标。
- 一次只读检查：`C:/Users/30395/.conda/envs/sfc_env/python.exe scripts/check_experiment.py --config experiments/server_20261021/monitor_config.json`。新只读端点和inotify watcher已验证。连接失败只表示观察不可用，不构成重启训练的依据。

## 后续

先分析约50k续训与同种子最强对照的结果；若仍不满足目标，按机制继续迭代。任何新科学代码或新训练须使用独立版本、目录和模型记录。最终需独立新种子确认，并用训练种子43/44复现；从seed42权重改随机种子续训不能算独立训练复现。最多两组GPU训练同时运行。
