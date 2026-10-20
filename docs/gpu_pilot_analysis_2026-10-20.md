# GPU 稳定性筛选与运动／部署诊断（2026-10-20）

## 本轮结论

修复版训练通路稳定，两种算法均完成 100000 步，但最终策略的主要短板在运动控制。Gaussian 的高坠毁率不能靠替换部署头解决。Diffusion 在这一训练种子上更省电、更安全，但任务完成率仍低于启发式，尚不能据此宣称算法优越性。

## 完成证据与实验口径

- 科学代码：`e8d277274b3b0207a923f7bfc2347d9df5a46407`。
- RTX 4090；训练种子 42，8 环境，每种算法 100000 步，另有 5000 步 replay warmup 和 2000 步 BC。
- 测试 seeds 1000–1019，各 20 episode。BC 默认只模仿部署（`bc_mobility_weight=0`）；Gaussian 的 `ent_coef=0`。
- 服务器控制器已退出；两组最终模型、评估文件及其哈希通过服务器监控校验。完成事件序号为 1、2、3，最后一个为 `pipeline_complete`。这比仅看 W&B 的 finished 状态更完整。
- 服务器状态快照保存在本地忽略目录 `experiments/server_20261020/pilot_status.json`，监控证据在同目录 `monitor_cache/last_snapshot.json`。
- [Diffusion W&B](https://wandb.ai/SACtest/diffusion_rl/runs/e595b8fd)；[Gaussian W&B](https://wandb.ai/SACtest/diffusion_rl/runs/07335b93)。

## 最终模型的原始 CUDA 评估

| 策略 | 任务完成率 | 坠毁 episode | 平均能耗 J/step | 平均 episode 回报 | 完成任务平均时延 s |
|---|---:|---:|---:|---:|---:|
| Diffusion | 45.42% | 5%（1/20） | 1916.72 | 551.08 | 2.543 |
| Gaussian | 38.27% | 90%（18/20） | 3496.71 | 363.01 | 2.612 |
| 全状态启发式 | 50.05% | 5%（1/20） | 2423.17 | 611.68 | 2.459 |
| 仅观测启发式 | 48.82% | 0%（0/20） | 2606.45 | 597.72 | 2.393 |

任务完成率为累计完成任务数／累计可用任务数；坠毁率按 episode 计。Gaussian 提前终止较多，平均每个 episode 35.3 步，其他策略约 40 步，因此不能把其完成率直接当作相同时间窗的吞吐率。平均时延只统计成功任务，低时延也不能抵消失败。

## 同设备四格替换诊断

以下均在服务器 CPU 推理，使用同一最终模型、同一环境配置和相同测试种子。每个 episode 重置 Python、NumPy、Torch 随机种子。替换运动动作时，关系型部署头会依据替换后的动作重新预测位置并重新选择部署，避免把旧运动对应的部署动作直接拼接过去。

代码标记中，第一位表示运动来源，第二位表示部署来源；D 为模型，H 为仅观测启发式。每个模型运行 4 × 20 episode，无梯度更新；本轮补齐了 Gaussian 的全部四格。

| 模型 | 组合 | 完成率 | 坠毁率 | J/step | 平均回报 |
|---|---|---:|---:|---:|---:|
| Diffusion | DD：模型运动＋模型部署 | 43.25% | 5% | 1931.31 | 522.62 |
| Diffusion | HD：启发式运动＋模型部署 | 49.33% | 0% | 2602.69 | 606.69 |
| Diffusion | DH：模型运动＋启发式部署 | 44.97% | 5% | 1937.33 | 543.42 |
| Diffusion | HH：全部启发式 | 48.82% | 0% | 2606.45 | 597.72 |
| Gaussian | DD：模型运动＋模型部署 | 38.27% | 90% | 3496.71 | 363.01 |
| Gaussian | HD：启发式运动＋模型部署 | 49.49% | 0% | 2602.43 | 608.72 |
| Gaussian | DH：模型运动＋启发式部署 | 39.72% | 85% | 3519.41 | 377.01 |
| Gaussian | HH：全部启发式 | 48.82% | 0% | 2606.45 | 597.72 |

原始 JSON 为 `diagnostic_diffusion_2026-10-20.json` 与 `diagnostic_gaussian_2026-10-20.json`，均位于本目录。

两个文件的诊断脚本哈希相同：`1b2ad0e30bdd393853c8338ab85cd319304388cafdead8d950f211e1990c6b59`。

模型 SHA256：

- Diffusion：`8402cdb4d5e5709e91f9906fd1544517ec76eb2aeb9e85e0c0b2e72c34614605`
- Gaussian：`dd160e376c47e42938d861e750b7f7907f7ba12af5f53982ae653925556c815e`

Diffusion 的 deterministic 推理仍采样初始 latent，因此 CPU 和 CUDA 的种子相同也不保证产生同一动作轨迹。原始 CUDA 完成率 45.42% 与 CPU 的 43.25% 不应混入同一替换效应；所有替换效应均在上述 CPU 表内计算。

## 能支持的解释

1. Gaussian 仅替换运动，完成率增加约 11.22 个百分点、坠毁率从 90% 降至 0%；仅替换部署，坠毁率仍为 85%。这支持优先修进运动学习。
2. Diffusion 仅替换运动，完成率增加约 6.07 个百分点；仅替换部署增加约 1.71 个百分点。这同样指向运动侧，但不排除运动与部署的交互影响。
3. 模型运动时每 episode 充电收获约为 Diffusion 6787 J、Gaussian 10537 J；替换运动后均约为 24628 J。Gaussian 的低电量 UAV-step 比例在原始评估中达 36.54%。充电导航和能量管理值得优先排查；这还不能证明其训练目标或某行代码就是根因。
4. HD 比 HH 略高的完成率不足以证明部署头更优：只有一个训练种子、20 个测试 episode，尚无多训练种子置信区间。

## 下一轮实施依据

先固定环境、奖励和部署设置，单独检查运动策略的学习过程。现有代码支持 `bc_mobility_weight`，可在短预检中检查运动模仿损失和动作分布，再开展有预算的对照。若验证 Gaussian 探索不足，再单独比较非零熵系数；不要同时更改奖励、熵与 BC 后把收益归因到其中一项。

正式多种子实验应同时报告完成率、episode 生存步数、坠毁率、充电量、能耗和时延；最终模型与 checkpoint 的筛选应使用独立验证种子，保留本轮测试种子的评估用途。

本轮没有启动新长训练。已完成训练与诊断的汇总、脚本修复和验证；下一轮运动学习改进仍需实际实验验证。

## 工具修复与验证

- 修复四格诊断 `--config` 传入训练目录时找不到 Hydra 文件的问题，与标准评估入口共用目录解析。
- 新诊断结果会补充配置文件 SHA256 和推理环境版本；已有结果保持原样，避免把新脚本哈希追记为旧实验来源。
- 本地核心回归、监控及诊断测试共运行 40 项，39 项通过、1 项跳过（Windows 不具备 Linux watcher 运行环境）；服务器复测 40 项全部通过，包含 Linux inotify / pidfd 集成测试。Gaussian 实际 80 episode 诊断成功完成。
- 诊断与监控脚本不含 SSH 密码或 W&B Key，监控配置及模型留在忽略的实验目录。
- 完成事件 1–3 已确认处理；原聊天的半小时监控已暂停，没有新训练或诊断进程待跟进。

服务器上复现诊断（项目根目录，输出文件必须不存在）：

```bash
/root/autodl-tmp/uav-sfc-venv/bin/python scripts/placement_diagnostic.py gaussian \
  server_runs/gpu_pilot_100k_seed42_20261020/gaussian/GAUSSIAN_final_model.zip \
  --config server_runs/gpu_pilot_100k_seed42_20261020/gaussian \
  --episodes 20 --seed 1000 --device cpu --output server_runs/diagnostic_repeat/result.json
```
