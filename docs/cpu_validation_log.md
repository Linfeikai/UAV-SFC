# CPU Validation Log

本文件持续记录 UAV-SFC 环境与算法在本地 CPU 上的验证过程。所有性能数字仅在对应配置、种子和训练预算下解释；smoke test 只用于验证代码健康，不作为算法优劣结论。

## 2026-07-22：接口修复与首次训练冒烟

### 发现并修复的问题

1. `SFCEnv.action_space` 已改为 `mobility + K*L*N placement logits`，但 `step()` 仍访问旧的 `raw_pick` 和 `raw_place_intent`，导致首次步进 `NameError`。
2. 恢复逐 VNF logits 解码；候选任务按紧急度选择；容量兜底使用 `required_cycles / affinity`，使 general UAV 执行 GPU VNF 时正确占用更多有效算力。
3. 启发式 warmup 和 place 实验动作构造统一到 104 维新格式。
4. Hydra 默认配置引用了不存在的 `env: level2`，已移除；`get_flat_config()` 以 `DEFAULT_CONFIG` 为基础，确保 policy 构造可获得环境维度。
5. SB3 的 SAC/PPO 面对 Dict observation 不能使用 `MlpPolicy`，已改为 `MultiInputPolicy`。
6. Diffusion Actor 的 ray mask 仍切分已删除的 Pick 动作，导致张量维度错误；已仅对 mobility 应用 ray mask，并同步删除失效的 Pick 日志切片。
7. Critic loss 日志改为 `critic_loss.detach().item()`，避免 PyTorch 的 requires-grad 转标量警告。

### 环境 smoke test

- 动作维度：`(104,)`。
- 均质与异构配置均连续随机步进 5 次，无异常、NaN 或提前崩溃。
- Python `compileall` 与 `git diff --check` 通过。
- Hydra `python main.py --cfg job` 可以解析。

### 启发式快速趋势测试

配置：2 episodes/策略，`arrival=0.4`、`K=5`、2 GPU + 2 general。

| slowdown | nearest success | balance success | M_split success | M_split - nearest |
|---:|---:|---:|---:|---:|
| 1 | 39.0% | 41.6% | 39.9% | +0.96pp |
| 4 | 36.8% | 37.0% | 38.9% | +2.08pp |
| 8 | 31.5% | 34.0% | 39.2% | +7.74pp |

样本很小，只能说明新动作接口下仍存在“异构越强，亲和拆分越有价值”的预期趋势。

### CPU 训练 smoke test

| 算法 | timesteps | warmup | batch | gradient steps | CPU 时间 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| SAC | 1000 | 100 | 默认 | 默认 | 约 24s | 完成、保存、重新加载成功 |
| Diffusion | 300 | 50 | 32 | 1 | 约 40s | 完成、保存、重新加载成功 |

Diffusion 在 300 步时的健康指标：actor loss `0.253`、critic loss `16.6`、QNE Q std `2.22`、归一化 Q std `1.0`，未出现 NaN。两份模型重新加载后均输出 `(104,)` 动作。

> 注意：两次训练预算不同，以上数字不能用于比较 SAC 与 Diffusion 性能。

## 2026-07-22：可复现性、压力测试与公平短训练

### 可复现性修复

SB3 `check_env()` 通过，但两个环境执行 `reset(seed=123)` 后状态不同。根因是每个 `UENode` 在构造时创建了独立、未由 Gymnasium seed 派生的 `default_rng()`，因此任务到达和数据大小不可复现。

修复方式：每次环境 reset 时，从 `self.np_random` 为每个 UE 派生 RNG，再生成初始任务。修复后验证：

- 相同配置、相同 seed 的初始 observation 完全相等；
- 输入相同固定动作后，next observation、reward、terminated、truncated 完全相等；
- `stable_baselines3.common.env_checker.check_env()` 通过。

### 100 episodes/策略异构稳定性实验

配置：`arrival=0.4`、`K=5`、2 GPU + 2 general、seed=42。

| slowdown | nearest success | balance success | M_split success | M_split-nearest | M_split-balance |
|---:|---:|---:|---:|---:|---:|
| 1 | 42.5% | 42.2% | 42.2% | -0.33pp | -0.03pp |
| 4 | 40.1% | 39.8% | 41.9% | +1.81pp | +2.07pp |
| 8 | 37.0% | 36.7% | 41.8% | +4.77pp | +5.07pp |

结论：均质退化检查通过；异构增强时，亲和感知逐 VNF 拆分的优势稳定扩大。slowdown=4 的信号真实但不大，适合作为主设定；slowdown=8 更适合作为压力测试。

### 环境随机压力测试

组合：`USE_HARD_CAP ∈ {false,true}` × `slowdown ∈ {1,4,8}`，每组 50 episodes，共执行 11,995 个环境 step。

- 每一步 observation 都通过 `observation_space.contains()`；
- observation 和 reward 全部为有限值；
- 无异常、维度错误或 NaN；
- 每组 0～1 个随机策略坠毁，属于环境终止机制而非执行错误。

### 公平 CPU 短训练

共同配置：异构 `[gpu,general,gpu,general]`、slowdown=4、1000 training steps、200 warmup steps、batch=32、gradient steps=1、buffer=5000、n_envs=1、seed=42、关闭 W&B。

| 算法 | CPU 时间 | 训练健康情况 |
|---|---:|---|
| SAC | 约 33s | 完成并保存；loss 有限，但短训练 rollout reward 后段下降 |
| Diffusion | 约 110s | 完成并保存；actor loss 约 0.24～0.29，QNE 归一化稳定，无 NaN |

Diffusion 末段诊断：QNE Q std `7.25`、normalized Q std `1.0`、softmax weight std `0.0782`、actor grad norm `0.0604`。Critic 原始梯度最高约 `137`，但已执行梯度裁剪。

### 独立固定种子评估

使用训练之外的 seeds 2000～2049，各评估 50 episodes；SAC 使用 deterministic action，Diffusion 每个 episode 固定 Python/NumPy/Torch seed。

| 指标 | SAC | Diffusion |
|---|---:|---:|
| success rate | 15.54% | 21.07% |
| admission efficiency | 27.31% | 36.87% |
| drop rate | 81.17% | 74.64% |
| timeout rate | 3.31% | 4.29% |
| mean episode reward | -106.54 | -34.25 |
| mean energy/step | 2944.06 J | 1315.31 J |
| mean episode length | 33.88 | 40.00 |
| crash episode rate | 100% | 0% |

这里不能得出“Diffusion 已最终优于 SAC”：1000 steps 太短，SAC 的主要失败模式是尚未学会能量/充电控制，确定性策略在所有评估 episode 中约第 34 步耗尽 UAV 电量；单个案例显示 UAV 0、1 同时坠毁，非碰撞。Diffusion 当前动作能耗更低并完整存活 40 步，因此短预算下表现更好。该结果证明训练与评估链路有效，同时暴露了能量规划是主要学习难点。

评估脚本：`test/cpu_model_eval.py`。

### SAC 生存问题的 CPU 消融

所有独立评估继续使用 seeds 2000～2049，共 50 episodes。

| 训练配置 | steps | success | crash episodes | mean episode steps | mean energy/step |
|---|---:|---:|---:|---:|---:|
| 默认奖励 | 1000 | 15.54% | 100% | 33.88 | 2944 J |
| 默认奖励 | 5000 | 22.65% | 100% | 31.22 | 3753 J |
| CRASH=-200 | 3000 | 14.92% | 100% | 31.18 | 3574 J |
| CRASH=-200, W_ENERGY=20 | 3000 | 17.86% | 100% | 31.98 | 3630 J |

观察：增加训练步数让 SAC 完成更多任务，但策略更加激进、能耗更高、坠毁更早；单独提高终止惩罚或同时提高稠密能耗权重，在 3000 步预算内均未学会充电/生存。`CRASH=-200` 实验模型在评估时仍使用默认环境计算展示用 reward，因此表中重点比较行为指标而不是跨奖励配置的 mean reward。

当前解释：能量/充电是一个长时程信用分配问题。任务成功奖励每步即时出现，而电量耗尽约在第 31～34 步才终止。短预算 SAC 容易学习到“高速服务更多任务，然后坠毁”。这说明正式 GPU 训练前仍需在 CPU 上检查奖励尺度、充电可达性、启发式生存率，以及是否需要电量安全 mask 或更明确的低电量 shaping；目前不应直接开始大规模算法排名。

### 充电机制可解性检查

亲和感知启发式在相同异构 slowdown=4 环境上运行 100 episodes（seeds 7000～7099）：

- success rate：`41.20%`；
- crash episodes：`1/100`；
- mean episode length：`40.0`；
- mean energy/step：`2079 J`。

因此环境并非必然坠毁，低电量回充逻辑能够维持完整 episode。SAC 的 100% crash 更可能是短训练与奖励信用分配问题，而不是充电桩不可达或环境物理机制完全无解。

### Mobility mask 公平性审计

发现 Diffusion Actor 的 `_apply_ray_mask()` 会实际缩放 mobility 动作，mask 同时包含低电量限速、边界限制和近距离避碰；vanilla SAC 虽然在 Dict observation 中看到 `mobility_bounds`，但 `MultiInputPolicy` 不会自动执行该 mask。因此此前“Diffusion 0% crash vs SAC 100% crash”不是纯算法差异。

为构造公平安全基线，新增默认关闭的 `APPLY_MOBILITY_MASK_IN_ENV`。开启后，环境按动作方向应用与 Diffusion 相同的 bounds，使 vanilla SAC 也能获得同类安全变换。顺便修复了零速度时 `np.where` 仍计算除零分支产生的 RuntimeWarning。

SAC + environment mobility mask，3000 training steps，50 个独立评估 episodes：

- success rate：`15.62%`；
- crash episode rate：`92%`（未加 mask 为 100%）；
- mean episode length：`38.0`（未加 mask 的同预算实验约 31～32）；
- mean energy/step：`3069 J`。

结论：mask 显著延迟了坠毁，但不能代替回充策略；SAC 仍需学会主动到达充电区域。正式算法对比必须明确区分 `SAC`、`SAC+mask`、`Diffusion+mask`，不能把 mask 带来的全部收益归因于 diffusion policy。

## 2026-07-22：电池轨迹、充电记账与回充学习诊断

### 评估指标扩展

`test/cpu_model_eval.py` 新增：minimum battery ratio、低于 25% 电量的 UAV-step 比例、实际充电 step 比例、每 episode 实际 harvested energy，以及 heuristic 评估模式。

同 seeds 3000～3049 的 50 episodes 对照（修正充电记账前采集，策略行为结论仍有效）：

| 指标 | Heuristic | SAC+mask | Diffusion |
|---|---:|---:|---:|
| success | 40.77% | 14.00% | 20.24% |
| crash episodes | 8% | 84% | 0% |
| energy/step | 2224 J | 3058 J | 1255 J |
| low-battery UAV-step | 2.78% | 20.16% | 0.43% |
| charged steps | 63.04% | 29.81% | 29.90% |

关键结论：Diffusion 并没有比 SAC 更频繁充电，而是输出更节能的 mobility/placement 行为，因此几乎不进入低电量区。SAC 的问题同时包含高能耗和回充不足。

### 充电记账 bug 修复

`LaserCharger.charge()` 原先调用 `receive_energy(E_harvest)` 后仍返回理论照射能量 `E_harvest`。当电池接近满电时，实际存入能量会被 capacity 截断，但 reward 和日志仍按理论能量计算，可能奖励“满电驻留充电区”。现已返回 `receive_energy()` 的实际入电量；`charge/num_charged` 也只在实际 harvested energy 大于 0 时计数。

修复后用新 seeds 4000～4019 做 20 episodes 行为复核：

| 指标 | Heuristic | SAC+mask | Diffusion |
|---|---:|---:|---:|
| success | 38.90% | 12.56% | 20.72% |
| crash episodes | 0% | 70% | 0% |
| energy/step | 2010 J | 3043 J | 1311 J |
| low-battery UAV-step | 1.28% | 22.04% | 0.59% |
| actual charged steps | 71.00% | 29.12% | 34.63% |

### 低电量距离 shaping 消融

新增默认关闭的 `RWD_LOW_BATTERY_DISTANCE`：仅当电量低于 30% 时，根据电量缺口与到最近充电桩的归一化距离产生稠密负奖励。默认值为 0，因此不改变旧实验。

`weight=5 + SAC+mask`、3000 training steps 的 50-episode 评估：success `19.30%`、crash `100%`、charged steps `17.39%`、energy/step `3481 J`。结果比无 shaping 更差，说明这一直接距离惩罚在短预算下没有形成正确返航行为，暂不推荐启用。

### 启发式 warmup 覆盖消融

将 SAC+mask 的 heuristic warmup 从 200 提高到 2000 steps，正式训练仍为 3000 steps。50 episodes（seeds 3000～3049）：

- success：`22.98%`；
- charged steps：`51.20%`；
- crash episodes：`94%`；
- mean episode length：`35.12`；
- energy/step：`3262 J`。

更多示范显著提升了充电频率和任务成功率，但仍未解决最终坠毁，说明 vanilla SAC 仅通过 replay 中的启发式轨迹很难学到稳定的长时程返航策略。后续若要利用示范，更合理的是显式 behavior cloning / offline pretraining，而不是无限增加 replay warmup。

### 本地最长 Diffusion 趋势训练

在修复充电记账后运行：3000 training steps、2000 heuristic warmup、batch=32、gradient steps=1、buffer=10000、n_envs=1、异构 slowdown=4。CPU 用时 `367s`，训练吞吐约 `8 FPS`。

训练全程无 NaN 或异常，模型保存/推理正常。后段健康指标：actor loss `0.223`、actor grad norm `0.0436`、critic loss `18.1`、QNE normalized Q std `1.0`。但 QNE Q mean 从早期约 `16` 上升到后段约 `109`，Q std 到 `16.2`；数值仍有限，但长训练必须监控 critic 过估计。

50 episodes（seeds 3000～3049）独立评估：

| 指标 | Diffusion 1k | Diffusion 3k |
|---|---:|---:|
| success | 20.24% | 22.02% |
| admission efficiency | 34.84% | 37.89% |
| crash episodes | 0% | 2% |
| energy/step | 1255 J | 1505 J |
| low-battery UAV-step | 0.43% | 0.88% |
| charged steps | 29.90% | 42.35% |
| mean episode reward | -38.51 | -28.85 |

3k 相比 1k 有小幅任务性能提升，也学到更多充电行为，但能耗和 critic Q 同时上升。CPU 证据支持“算法可训练且有趋势”，尚不足以声称收敛或显著优于所有公平 baseline。

## CPU 阶段结论与 GPU 阶段待办

### 已由 CPU 证实

1. 环境 Gymnasium/SB3 API、104 维动作、逐 VNF logits 解码、异构 affinity、soft/hard overload、模型保存加载可运行。
2. reset seed 和任务生成可复现；约 1.2 万随机 steps 无 NaN/维度错误。
3. slowdown=1 的均质退化检查通过；异构增强时 M_split 优势单调扩大。
4. 充电机制可解；启发式在 100 episodes 中约 41% success、约 1% crash。
5. SAC、SAC+mask、Diffusion 均能完成 CPU 训练；Diffusion 的 QNE/actor/critic 更新链路有效。
6. mobility mask 是重要混杂因素，正式对比必须向 baseline 明确提供同类 mask 或分别报告。
7. 充电 theoretical-vs-actual energy 记账已修复。

### 不应再用单次 CPU 短跑回答的问题

- Diffusion 是否统计显著优于 SAC/TD3；
- 10万～50万步后的最终收敛性能；
- Q 值上升是否最终稳定；
- 多训练 seeds 的方差与置信区间；
- 大 batch、更多 QNE samples、更多 diffusion steps 的收益。

这些问题需要至少 5 seeds × 多算法 × 10万级 steps。按本地实测 Diffusion 约 8 FPS，单个 100k run 纯训练至少约 3.5 小时，完整矩阵需要数十小时 CPU，因此应转 GPU/服务器。

### GPU 正式实验建议

主环境已经写入 `conf/env/heterogeneous.yaml` 并设为 Hydra 默认：2 GPU + 2 general、slowdown=4、soft overload。slowdown=1 用作退化对照，slowdown=8 用作压力测试。

第一阶段建议：

- 算法：`SAC`、`SAC+env mobility mask`、`Diffusion+actor mask`；有实现后加入 TD3。
- seeds：`0,1,2,3,4`。
- 训练：先 100k steps，通过稳定性门槛后再 500k。
- 相同 replay size、batch、gradient steps、warmup 数据量和评估 seeds。
- 每 5k～10k steps 保存 checkpoint，并独立 deterministic/fixed-randomness 评估。
- 重点监控：success、crash、energy、charge、low-battery exposure、Q mean/std、TD error、actor/critic grad norm。

GPU 启动前门槛：任何配置若出现 NaN、Q 持续无界增长、100% crash 或模型无法 reload，应停止扩展 seeds，先修复稳定性。
