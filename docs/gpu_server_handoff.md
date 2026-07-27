# GPU Server Handoff Plan

本文档用于把 UAV-SFC 项目从本地 CPU 验证阶段交接到 GPU 服务器。服务器执行者应先阅读 `docs/cpu_validation_log.md`，不要重新采用旧的 Pick/坐标意图动作接口。

## 1. 当前可信状态

- 主环境：4 UAV、20 UE、500m × 500m、40 steps/episode、每步 1s 飞行 + 7s 计算。
- 默认研究环境：`conf/env/heterogeneous.yaml`。
  - UAV 类型：`gpu, general, gpu, general`
  - `NON_GPU_SLOWDOWN=4`
  - soft overload：`USE_HARD_CAP=false`
- 动作维度：104。
  - mobility：`4 × 2 = 8`
  - placement logits：`K × L × N = 6 × 4 × 4 = 96`
- observation：Dict，包括 130 维 state 与 16 维 mobility bounds。
- Diffusion Actor 使用 mobility ray mask；vanilla SAC 不会自动执行，需要用 `env.APPLY_MOBILITY_MASK_IN_ENV=true` 构造公平的 SAC+mask baseline。
- 本地 CPU 已通过：环境 API、seed 可复现、随机压力测试、SAC/Diffusion 训练、模型保存/加载、固定种子评估。

## 2. 同步代码时必须包含的文件

当前工作树包含尚未提交的关键修复。上传服务器时不能只使用旧 commit，至少确认下列文件与本地一致：

- `core/sfc_env.py`
- `core/env_config.py`
- `core/laser_charger.py`
- `main.py`
- `algos/diffusion_policy_actor.py`
- `algos/diffusion_sac_agent.py`
- `test/evalu.py`
- `test/place_ablation.py`
- `test/cpu_model_eval.py`
- `conf/config.yaml`
- `conf/env/heterogeneous.yaml`
- `docs/cpu_validation_log.md`
- 本文档

不要上传本地的 `experiments/`、`wandb/`、TensorBoard 大目录或 `.zip` 模型，除非需要保留 CPU checkpoint 作参考。

## 3. 服务器环境初始化

建议创建独立 Conda 环境，不要直接污染 base：

```bash
conda env create -f environment.yml
conda activate sfc_env
```

如果 `environment.yml` 中的 PyTorch 不是 CUDA 构建，应按服务器 CUDA 版本重新安装 PyTorch，再安装其余依赖。初始化后执行：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
assert torch.cuda.is_available()
PY
```

然后确认配置和代码语法：

```bash
python main.py --cfg job
python -m compileall -q core algos test main.py
```

`--cfg job` 输出必须包含 `env.UAV_TYPES` 和 `NON_GPU_SLOWDOWN: 4.0`。

## 4. GPU 上的第一道门：短 smoke test

先关闭 W&B，避免代码错误产生无效云端 run。

### SAC baseline（无 mask）

```bash
python main.py \
  algo_name=SAC total_timesteps=1000 warmup_steps=200 n_envs=1 \
  use_wandb=false run_heuristic_baseline=false \
  +sac.batch_size=32 +sac.gradient_steps=1 sac.buffer_size=5000 \
  exp_name=gpu_smoke_sac
```

### SAC + 公平 mobility mask

```bash
python main.py \
  algo_name=SAC total_timesteps=1000 warmup_steps=200 n_envs=1 \
  use_wandb=false run_heuristic_baseline=false \
  env.APPLY_MOBILITY_MASK_IN_ENV=true \
  +sac.batch_size=32 +sac.gradient_steps=1 sac.buffer_size=5000 \
  exp_name=gpu_smoke_sac_mask
```

### Diffusion + Actor ray mask

```bash
python main.py \
  algo_name=DIFFUSION total_timesteps=1000 warmup_steps=200 n_envs=1 \
  use_wandb=false run_heuristic_baseline=false \
  diffusion.batch_size=32 diffusion.buffer_size=5000 \
  diffusion.gradient_steps=1 \
  exp_name=gpu_smoke_diffusion
```

Smoke test 通过标准：

- 日志显示 `Using cuda device`，而不是 CPU。
- replay buffer 正常增长。
- actor/critic loss、Q mean/std、TD error 全部有限。
- 模型成功保存。
- 保存后的模型可以重新加载并输出 `(104,)` 动作。

任意一项失败都不要启动正式多种子实验。

## 5. 第一阶段：100k 稳定性筛选

目标不是立即写最终结论，而是筛掉不稳定配置。

### 共同设置

- seeds：`0,1,2,3,4`
- `total_timesteps=100000`
- `warmup_steps=20000`
- `n_envs=4`；若显存或进程不稳定，先降为 1
- `batch_size=256`
- `gradient_steps=1` 起步；稳定后再测试 2
- `use_wandb=true`
- 相同环境、相同评估 seeds、相同 checkpoint 周期

### 必跑矩阵

| ID | 算法 | Mobility mask | 作用 |
|---|---|---|---|
| A | SAC | 无 | vanilla baseline |
| B | SAC | environment mask | 与 Diffusion 安全约束公平对照 |
| C | Diffusion | actor ray mask | 当前主算法 |

每个 seed 单独设置 `exp_name`，例如：

```bash
python main.py \
  algo_name=DIFFUSION seed=0 total_timesteps=100000 warmup_steps=20000 \
  n_envs=4 use_wandb=true run_heuristic_baseline=false \
  exp_name=stage1_diffusion_seed0
```

SAC+mask 额外添加：

```text
env.APPLY_MOBILITY_MASK_IN_ENV=true
```

不要给 Diffusion 同时开启 environment mask，否则会形成双重 mask。

## 6. 训练期间必须监控

### 环境性能

- success rate
- admission efficiency
- drop / timeout rate
- crash episode rate
- episode length
- mean energy per step
- charged step rate / harvested energy
- low-battery UAV-step rate

### 算法稳定性

- actor loss / critic loss
- actor / critic gradient norm
- `qne_q_mean`、`qne_q_std`
- normalized Q mean/std
- target Q mean
- TD error
- QNE entropy / softmax weight std
- mobility/place boundary hit rate

### 立即停止条件

- NaN 或 Inf
- Q mean 持续指数增长且无平台趋势
- critic loss/gradient 连续爆炸
- 100% crash 持续多个评估窗口
- success 长期为零
- checkpoint 不能加载

CPU 3k Diffusion 的 Q mean 从约 16 上升到约 109，因此服务器长跑必须特别关注 Q 过估计。

## 7. 独立评估要求

训练 rollout 不能替代正式评估。每个 checkpoint 至少使用 50 个固定 seeds：

```bash
python test/cpu_model_eval.py diffusion \
  experiments/.../DIFFUSION_final_model.zip \
  --episodes 50 --seed 10000
```

SAC+mask：

```bash
python test/cpu_model_eval.py sac \
  experiments/.../SAC_final_model.zip \
  --episodes 50 --seed 10000 --env-mask
```

正式汇总建议使用 seeds 10000～10099，至少 100 episodes/模型。Diffusion 内部仍含采样噪声，评估脚本会为每个 episode 固定 Python、NumPy 和 Torch seed。

## 8. 进入 500k 正式阶段的门槛

只有满足以下条件才将 100k 扩展到 500k：

- 5 个训练 seeds 中至少 4 个无数值崩溃。
- checkpoint 全部可加载。
- Q mean/std 出现平台或缓慢增长，而非无界增长。
- crash rate 与 success rate 明显优于随机策略。
- SAC+mask 与 Diffusion 的比较使用相同环境约束。
- 100k 多种子结果显示值得继续投入计算。

若 Diffusion 在 100k 前出现 Q 过估计，优先测试：降低 critic learning rate、减少 gradient steps、增大 target 平滑、检查 reward scale；不要直接启动 500k。

## 9. 第二阶段消融

主对比稳定后再做，避免同时改变多个变量：

1. slowdown：`1,4,8`
2. overload：soft vs hard cap
3. mask：无 mask / environment mask / actor ray mask
4. warmup：0 vs 20k heuristic
5. QNE temperature：建议 `0.25,0.5,1.0`
6. QNE candidates：默认 32，可测试 8/16/32
7. diffusion steps：默认 20，可测试 5/10/20

slowdown=1 是退化正确性检查；slowdown=4 是主结果；slowdown=8 只作为压力测试，不应成为唯一报告配置。

## 10. 最终应产出的结果

- 每个算法 5 seeds 的学习曲线，带均值与 95% CI。
- 独立评估表：success、crash、energy、latency、drop、timeout、charge。
- 训练耗时、GPU 型号、显存峰值和推理耗时。
- SAC / SAC+mask / Diffusion+mask 公平对比。
- slowdown 与 mask 消融。
- 失败 seed 和异常 checkpoint 也要保留并解释，不能只挑最好 seed。

服务器每完成一个阶段，都应把命令、commit hash、环境版本、结果和结论追加到 `docs/cpu_validation_log.md`，或新建 `docs/gpu_validation_log.md`，确保实验可复现。
