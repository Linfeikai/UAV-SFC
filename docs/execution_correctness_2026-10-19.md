# 主训练链路正确性修复与验证（2026-10-19）

## 范围与同步

本次集中修改 `main.py`、`core` 和主 Diffusion / Gaussian 策略链路，不调整旧训练入口和论文文件。
已有本地工作先保存为 `7ad6d6a`，随后以 `3797325` 合入远端 `origin/main`（`42e4dc4`）。修复在已有 `agent/uav-sfc-validation` 分支上提交，未覆盖 main、未强制推送。

## 修复内容

1. **CPU 竞争与任务结算**：新增 `core/execution.py`，替换整 slot 总需求近似。仅已完成输入传输且前序 VNF 完成的就绪 VNF 参与同机等份 CPU；多个任务不能分别独占同一 CPU。计算速率保留异构 VNF affinity。
2. **截止时间与能耗**：事件推进到下一次就绪、完成、deadline 或 slot 边界；超时取消释放 CPU，恰好在截止点完成有效。只收取实际 CPU busy time 和已经开始的跨机传输能耗，传输按剩余截止窗口截断。不会给未执行到的后继 VNF 收全额传输费。
3. **运动状态**：动作是加速度，位置按初末平均速度积分，但持久化末速度；悬停、碰撞和边界相应清零速度。
4. **部署位置预测**：关系型 placement 根据当前速度、加速度、速度上限、飞行时间、边界、低电量悬停与碰撞冻结预测移动后位置，并由入口传入环境参数；不再把动作当位移。
5. **训练预热**：PPO 等无 replay buffer 算法跳过手工 replay warmup；手动采样保留 `TimeLimit.truncated`，使时间截断仍能 bootstrap。多环境日志按 transition 而非 buffer 行数统计。
6. **坠毁与统计**：结算阶段坠毁只使涉及该 UAV 的路径失败，其余路径仍完成结算；终止步统一统计剩余任务并清空 buffer。保留原 crash reward 覆盖规则，增加分量对账项。

## 模型边界

- 每 slot 先飞行，再在 `TIME_SLOT_DURATION - DT_FLY` 内执行，执行超时任务不跨 slot 隐式续跑。
- 无线链路继续使用独立、恒定速率模型；未新增通信带宽竞争。
- 电池仍在 slot 内聚合结算，不是连续时间耗尽仿真；若结算发现某 UAV 电量不足，其路径任务保守判失败，不能解释为精确重建坠毁发生时刻。
- 负载观测仍是分配需求 / slot 容量的诊断量，并非实际 CPU busy 比例。
- 修复改变了环境执行语义。正式比较需在同一修复版本重新训练、重新运行 baseline，不能直接混用旧结果。

## 自动回归

环境：Windows / Python 3.10 / `sfc_env` / CPU。

```powershell
$env:PYTHONIOENCODING='utf-8'
$py = 'C:/Users/30395/.conda/envs/sfc_env/python.exe'
& $py -m unittest test.test_regressions -v
& $py -m test.sanity_execution --output docs/execution_sanity_2026-10-19.json
```

24 项 unittest 全部通过，包括 CPU 容量反例、分机并行、截止点完成、上行与链依赖、超时释放、能耗截断、affinity、坠毁后存活任务结算、终止统计、运动与预测一致性、有限梯度、seed 可复现、SB3 `check_env` 和真实 replay buffer 的超时 bootstrap。
主修改文件另通过 Python 编译检查。

## 四组短程训练 smoke

以下只验证 BC / 采样 / 梯度更新 / 保存和重载链路，不是收敛实验。`hydra.run.dir` 指定到系统 TEMP，模型不提交 Git；CPU 即可复跑。

```powershell
& $py main.py algo_name=DIFFUSION n_envs=1 warmup_steps=8 total_timesteps=8 diffusion.batch_size=4 diffusion.buffer_size=64 diffusion.gradient_steps=1 use_wandb=false run_heuristic_baseline=false hydra.run.dir=C:/Users/30395/AppData/Local/Temp/uav-sfc-fixed-diffusion

& $py main.py hybrid=relational env=heterogeneous_hybrid n_envs=2 warmup_steps=8 bc_pretrain_steps=2 bc_batch_size=4 total_timesteps=16 checkpoint_freq=0 diffusion.batch_size=4 diffusion.buffer_size=64 diffusion.gradient_steps=1 diffusion.qne_k_samples=2 diffusion.categorical_bc_batch_size=4 use_wandb=false run_heuristic_baseline=false hydra.run.dir=C:/Users/30395/AppData/Local/Temp/uav-sfc-fixed-relational

& $py main.py hybrid=gaussian_relational env=heterogeneous_hybrid n_envs=1 warmup_steps=8 bc_pretrain_steps=2 bc_batch_size=4 total_timesteps=16 checkpoint_freq=0 gaussian.batch_size=4 gaussian.buffer_size=64 gaussian.categorical_bc_batch_size=4 use_wandb=false run_heuristic_baseline=false hydra.run.dir=C:/Users/30395/AppData/Local/Temp/uav-sfc-fixed-gaussian

& $py main.py algo_name=PPO n_envs=1 total_timesteps=8 ppo.n_steps=8 ppo.batch_size=4 use_wandb=false run_heuristic_baseline=false hydra.run.dir=C:/Users/30395/AppData/Local/Temp/uav-sfc-fixed-ppo
```

| 配置 | 结果 | 保存模型 `_n_updates` |
|---|---|---:|
| 默认 Diffusion | 短训练、保存、重载通过 | 8 |
| relational Diffusion，2 环境 | BC + RL、保存、重载通过 | 8 |
| relational Gaussian | BC + RL、保存、重载通过 | 16 |
| PPO，保留默认 warmup_steps=20000 | 正确跳过 replay warmup、训练、保存、重载通过 | 10 |

重载使用各自 `.hydra/config.yaml` 重建环境（含 context）；检查 reset seed=42 后的预测动作有限且属于 action space，policy 参数均有限。

## 固定种子环境 sanity 对照

脚本：`test/sanity_execution.py`；原始结果：`docs/execution_sanity_2026-10-19.json`。
使用主入口默认配置，seeds=42/43/44，最多 64 步，但正常环境 episode 在第 40 步截断。
每步检查观测 / reward 有限、任务分类对账和计算能耗上限。

| 策略 | 总任务 | 完成 | 丢弃 | 超时 | 坠毁失败 | 完成率 | 坠毁 episode |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 1206 | 300 | 847 | 59 | 0 | 24.88% | 0/3 |
| 现有 smart heuristic | 1174 | 583 | 546 | 31 | 14 | 49.66% | 2/3 |

完成率为全部 episode 的总完成数 / 总任务数。启发式存在明显安全问题，不能只比较完成率而忽略提前终止；本次没有重新设计启发式控制器。
这三组种子不能证明算法显著提升，更不代表 Diffusion、Gaussian 或 PPO 的收敛性能。尚未启动正式多种子长训练或 GPU 实验。
