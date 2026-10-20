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
| Anchor 验证所选模型 | 47.96% | 20% | 559.9 |
| Cosine 验证所选模型 | 48.73% | 45% | 541.0 |
| 全状态启发式 | 50.03% | 15% | 587.8 |
| 仅观测启发式 | 49.57% | 25% | 561.4 |

三组都没有达到目标，因此没有延长它们。第一批完整记录保存在 `motion_round1_2026-10-20.json`。训练时小样本验证中的安全性没有稳定泛化到确认种子，不能把 0/5 的坠毁数当成已经解决安全问题。

### 后续诊断与第二批

在 seeds 2000–2009 上，对原始 100k Diffusion 最终模型做两种诊断：

- 用两个已学 Critic 的较小 Q 值从 K 个采样动作中选最优：K=1、4、16 的完成率分别为 45.14%、37.53%、34.78%，均未坠毁。当前 Critic 的动作排名不可靠，不能靠增加候选数提升性能。
- 把运动改为零加速度，并按该动作重新计算模型部署：完成率 51.23%、坠毁率 0%、回报 639.60、能耗 762.86 J/step。同一验证集的全状态启发式为 49.88%、0%、620.80。零加速度只用于诊断，不能算模型独立生成动作的成绩。

据此继续两组训练，均未改变环境或奖励：

| 变体 | 机制 | 代码 | W&B |
|---|---|---|---|
| X0 | 直接预测干净动作；cosine；在线运动 BC 系数 4，保留原追踪教师 | 2115f3b | [xc5102db](https://wandb.ai/SACtest/diffusion_rl/runs/xc5102db) |
| Stationary | 在 X0 基础上用速度反馈制动教师、独立 Critic 编码器和排名规范化动作表示 | e150a29 | [s61ea975](https://wandb.ai/SACtest/diffusion_rl/runs/s61ea975) |

X0 已完成。其验证所选模型在 confirmation seeds 3000–3019 上完成率 48.49%、坠毁率 35%；同一批全状态与仅观测启发式分别为 50.03%/15% 和 49.57%/25%。X0 未达筛选目标，不再延长。此前同一确认集上的全状态启发式零移动对照为 53.345% 完成率、0% 坠毁、平均回报 643.16、能耗 767.043 J/step；这是额外的静态启发式对照，不是学习算法成绩。

Stationary 的 20k 组已完成（seeds 3000–3019）：模型完成率 52.849%、坠毁率 0%、平均回报 636.388、能耗 764.475 J/step、平均存活 40 步；普通全状态启发式为 50.026%/15%/587.820/2727.372 J/step/39.65 步，仅观测启发式为 49.572%/25%/561.354/2748.981 J/step/38.6 步。模型运动幅度均值为 0.1341，显示其成绩主要伴随低移动成本。虽然它超过普通两种启发式并达到零坠毁，但相比更强的全状态零移动对照，完成率低 0.496 个百分点、回报低 6.77，因此尚未达到预设目标。完成事件序号 1–2 已保存并确认处理；本地指标和模型/评估哈希在 `experiments/server_20261020/stationary_round2_status.json`。

该结果足以支持有界续训与更大确认集：从版本化输出中验证所选 checkpoint，以 `resume_model` 继续训练、重新 warmup、`bc_pretrain_steps=0`，优先扩到约 50k 总训练步数；用独立至少 50 个确认 episode 检查能否在保持安全和奖励的同时超过 53.345% 静态对照。不要把 20 集的 0% 坠毁当作安全泛化结论。当前服务端监控端点只提供状态读取；续训需在拿到可写 SSH 会话后执行，不能以端点路径不匹配来猜测模型或启动任务。

Stationary 的教师只用于 warmup 和训练标签：`a = clip(-v / (a_max * dt), -1, 1)`。在线推理完全使用学习到的运动头和部署头，不强制置零动作。随机 warmup 会产生非零速度，所以教师需要制动，不能简单对所有状态都贴零加速度标签。

Critic 的新表示保留完整部署排名和容量不足时的备选顺序，消除教师 ±0.5 与 Actor ±1 等等价部署的幅值差异；忽略无效 VNF 槽。该模式要求环境的部署附加 bias 为零，主入口会校验。

这些是机制组合筛选，若有效，后续需要分别消融干净动作预测、制动标签、独立 Critic 和动作规范化的贡献。

## 调度与持续检查

- 控制器：`/root/autodl-tmp/UAV-SFC-motion-68a4e8e/scripts/motion_search.py`。
- 计划：`/root/autodl-tmp/motion_plan.json`；控制器日志：`/root/autodl-tmp/motion_search.log`。
- 结果目录：`/root/autodl-tmp/UAV-SFC-motion-68a4e8e/server_runs/motion_search_20261020`。
- PID 7139 仅为启动快照；判断存活需结合 boot ID、start ticks 和命令参数。
- 控制器接管已启动任务，不重启；空出训练槽后继续下一组。每组保存配置、模型哈希、完整验证与确认结果，并回写 W&B。
- 本地一次检查：`python scripts/check_experiment.py --config experiments/server_20261020/motion_monitor_config.json`。
- 只读 SSH 端点已切换到新批次；旧 `monitor_config.json` 对应的原始 100k 证据已缓存在本地，不应再把旧配置当作新任务检查入口。
- 本聊天已启用每 10 分钟一次的持续实验自动化 `uav-sfc-2`。批次完成但目标未达到时，应分析并继续下一轮；状态没有实质变化时保持安静。

第二批活动路径：

- X0 控制器启动 PID 7721，已完成并确认未达目标：`/root/autodl-tmp/UAV-SFC-motion-2115f3b/server_runs/x0_search_20261020/status.json`；训练验证：同根目录 `server_runs/x0_bc4_20k_seed42/validation_history.json`。
- Stationary 控制器启动 PID 8143：`/root/autodl-tmp/UAV-SFC-motion-e150a29/scripts/motion_search.py --plan /root/autodl-tmp/stationary_plan.json`。
- Stationary 状态：`/root/autodl-tmp/UAV-SFC-motion-e150a29/server_runs/stationary_search_20261020/status.json`；训练验证：同根目录 `server_runs/stationary_20k_seed42/validation_history.json`。
- 2026-10-20 11:07 UTC 的只读快照确认 Stationary 的完整结果、模型 archive 和评估文件有效；事件序号 1–2 已在本地保存后确认处理。监控端点固定读取 Stationary；对该端点用 X0 路径查询会返回路径不匹配，不代表 X0 失败。
- Stationary 验证所选模型输出：`/root/autodl-tmp/UAV-SFC-motion-e150a29/server_runs/stationary_search_20261020/diffusion_stationary/DIFFUSION_STATIONARY_final_model.zip`，SHA-256 `2106f25d3f9b2cfcbdb775d2549fb3322aae72ea45ed322adb0053c083720fa2`。

## 验证记录

- 新增在线运动 BC；验证前后恢复 Python、NumPy、Torch 随机状态和策略模式，避免评估改变训练随机流。
- 支持预训练后模型、验证最优模型保存和 `resume_model` 续训。续训会重新填充 replay buffer，不能声称恢复了未保存的历史 replay。
- 本地 44 项测试：43 通过、1 项 Linux 专属测试跳过。早一版本在服务器 43 项全部通过；余弦版本 GPU 实际短训练、验证与模型保存成功。
- 新训练源码使用独立版本目录，未覆盖正在运行的科学代码。
- 新版累计本地 48 项测试：47 通过、1 项 Linux 专属测试跳过。X0 和 Stationary 均完成真实 GPU 短训练、验证和保存；独立 Critic 编码器梯度、动作排名不变性、推理 Q 排名、零加速度几何都已有覆盖。
