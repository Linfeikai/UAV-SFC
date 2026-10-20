# 运动策略持续实验（2026-10-20，进行中）

用户要求持续使用现有按小时租用的服务器迭代算法，直到在既定环境下获得良好表现。本文件记录后续实验，不能把一批实验完成解释为目标已达成。

## 固定口径

- 环境始终为 `env=heterogeneous_hybrid`，保留原物理约束和奖励。
- 使用现有 RTX 4090；服务器 CPU 配额约 20 核，最多同时运行两组训练。
- 第一批训练种子 42，每组 20000 步，5000 步 warmup；先筛选，再延长有进展的配置。
- 训练中每 5000 步用 seeds 2000–2004 做小样本检查；结束后在 2000–2009 上比较最终模型和训练中选出的模型。
- 选择后在 seeds 3000–3019 上确认，并重跑全状态、仅观测两种启发式。后续迭代若继续参考这些成绩，它们就属于开发阶段的评估，最终结论需另留新种子。
- 初步筛选目标：完成率超过较强启发式至少 0.5 个百分点，坠毁率不高于 5%，平均回报不低于启发式；达到后再扩至至少 50 个新 episode，并补训练种子 43、44。

## 本批配置

| 变体 | 机制 | 科学代码 | W&B |
|---|---|---|---|
| BC | 原线性日程；4000 步运动 BC 预训练，在线不锚定 | e8d2772 | [md5360b0](https://wandb.ai/SACtest/diffusion_rl/runs/md5360b0) |
| Anchor | 同样预训练，在线运动去噪 BC 系数 1 | cd3c87a | [ac41c13b](https://wandb.ai/SACtest/diffusion_rl/runs/ac41c13b) |
| Cosine | 在线 BC 系数 1；余弦日程及去噪动作裁剪 2.5 | 68a4e8e | [c42e212b](https://wandb.ai/SACtest/diffusion_rl/runs/c42e212b) |

三组使用相同 warmup/BC 数量、batch 256、QNE 候选数 8、8 个训练环境。余弦方案同时包含数值裁剪，因此结果应解释为这组实现改动的效果；若有效，还需分离日程和裁剪的贡献。

原代码在 20 个扩散步上直接使用 beta 0.0001–0.02，末端 `alpha_bar` 约为 0.817，推理却从标准高斯初始化。这是一个可检验的训练／采样分布问题。新增可选余弦日程，使短链末端接近纯噪声，并保留旧线性日程的兼容行为。日程参考 [Improved Diffusion 的官方实现](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py)。此项机制分析不等于已经证实成绩改善。

## 已得到的结果

BC 变体已完成 20000 步。在 20 个确认 episode 上：

| 策略 | 完成率 | 坠毁率 | 平均 episode 回报 |
|---|---:|---:|---:|
| BC 最终模型 | 43.82% | 5% | 517.0 |
| 全状态启发式 | 50.03% | 15% | 587.8 |
| 仅观测启发式 | 49.57% | 25% | 561.4 |

此配置没有达到目标，不能因此停止迭代。Anchor 的初步 5-episode 验证中，预训练模型优于 5000-step 模型，训练期间已保存早期候选，等待完整确认。

## 调度与持续检查

- 控制器：`/root/autodl-tmp/UAV-SFC-motion-68a4e8e/scripts/motion_search.py`。
- 计划：`/root/autodl-tmp/motion_plan.json`；控制器日志：`/root/autodl-tmp/motion_search.log`。
- 结果目录：`/root/autodl-tmp/UAV-SFC-motion-68a4e8e/server_runs/motion_search_20261020`。
- PID 7139 仅为启动快照；判断存活需结合 boot ID、start ticks 和命令参数。
- 控制器接管已启动任务，不重启；空出训练槽后继续下一组。每组保存配置、模型哈希、完整验证与确认结果，并回写 W&B。
- 本地一次检查：`python scripts/check_experiment.py --config experiments/server_20261020/motion_monitor_config.json`。
- 只读 SSH 端点已切换到新批次；旧 `monitor_config.json` 对应的原始 100k 证据已缓存在本地，不应再把旧配置当作新任务检查入口。
- 本聊天已启用每 10 分钟一次的持续实验自动化 `uav-sfc-2`。批次完成但目标未达到时，应分析并继续下一轮；状态没有实质变化时保持安静。

## 验证记录

- 新增在线运动 BC；验证前后恢复 Python、NumPy、Torch 随机状态和策略模式，避免评估改变训练随机流。
- 支持预训练后模型、验证最优模型保存和 `resume_model` 续训。续训会重新填充 replay buffer，不能声称恢复了未保存的历史 replay。
- 本地 44 项测试：43 通过、1 项 Linux 专属测试跳过。早一版本在服务器 43 项全部通过；余弦版本 GPU 实际短训练、验证与模型保存成功。
- 新训练源码使用独立版本目录，未覆盖正在运行的科学代码。
