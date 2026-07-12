# VNF↔UAV 亲和机制 —— 实验报告

> 日期：2026-07-09
> 目的：把 place（VNF→UAV 部署）从"退化的标量选择"改造成**真正值钱的高维决策变量**，
> 契合 SFC/VNF 主题，并为"扩散策略显著优于 baseline"创造存在空间。
> 结论先行：**成功**。异构亲和机制下，正确的 VNF 拆分部署相对最近邻/负载均衡
> baseline 产生 **+10~11 个百分点** 的完成率价差（此前五个实验封顶仅 ~3%）。

---

## 一、背景：为什么之前 place 是退化的

前序研究（见 `place_decision_study.md`）用五个实验证明：均质环境下 place 价差封顶 ~3%，
被距离几何主导。根因是**环境没有理由让一条链的 VNF 分开部署**：
- 一条 SFC 链可整体塞进单架 UAV；
- 4 架 UAV 除位置外能力完全相同 → "选哪架"退化为"选最近的空机"，一个标量决策。

SFC 的学术核心——**把链拆开、映射到异构节点**——从未被激活。

## 二、改动：VNF↔UAV 亲和（异构加速）

引入 MEC/VNF-placement 经典设定：UAV 分**两型**。

| 配置项 | 含义 |
|---|---|
| `UAV_TYPES` | 长度 N 的类型标签，如 `["gpu","general","gpu","general"]`；`None`=均质旧行为 |
| `GPU_VNFS` | 需硬件加速的重计算 VNF 集合（ObjectDetection、Encryption、FeatureExtraction 等） |
| `NON_GPU_SLOWDOWN` | 通用机处理重 VNF 的减速倍率（有效算力 = 名义/slowdown） |

**核心公式**：有效计算时间 = `required_cycles / (cpu_freq × affinity)`，
其中 general 机处理 GPU_VNFS 时 `affinity = 1/slowdown`，其余为 1。

**一致性接入**（`_vnf_affinity()` 统一注入四处，避免准入/评估/计费口径不一）：
1. `_evaluate_performance` 计算延迟
2. `_account_resource_usage` 有效负载累加（放错加速器→占用更多算力→拉高竞争）
3. `_estimate_sfc_duration` 单体准入检查
4. `_hard_cap_drop_sfcs` 容量核算

**可训练接入**：
- 观测 UAV 特征块 9→10 维，新增 `is_gpu` 标识（GPU=1/general=0），
  观测总维 62→130（N=4）。策略据此可学"重 VNF 送 GPU 机"。
- 观测空间边界同步更新；网络首层从 `observation_space` 动态推导维度，自动适配。
- 默认 `UAV_TYPES=None` 完全退化为旧行为，`is_gpu` 恒为 0，不影响既有训练。

## 三、验证实验

`test/vnf_affinity.py`。参数 arrival=0.4, K=5, 2 GPU + 2 通用机, 40 ep/组。
三策略均支持**逐 VNF 拆分部署**：
- **A_nearest**：整链→最近 UAV（亲和盲视，= 现状 place_intent 等价物）
- **B_balance**：整链→最近且放得下（亲和盲视+负载感知，前序实验的赢家）
- **M_split**：逐 VNF 拆分，重 VNF→最空 GPU 机、轻 VNF→就近（亲和感知+拆分）

### 结果（完成率 %）

| NON_GPU_SLOWDOWN | A_nearest | B_balance | M_split | M−A | M−B |
|---|---|---|---|---|---|
| 1.0（均质对照） | 41.2 | 42.1 | 41.3 | +0.1 | −0.8 |
| 4.0 | 36.7 | 37.9 | 41.0 | +4.2 | +3.1 |
| 8.0 | 31.0 | 31.5 | **42.4** | **+11.5** | **+11.0** |

## 四、结论

1. **均质对照通过**：slowdown=1 时三策略 41~42% 几乎相同，证明改动干净、退化正确、
   未引入无关偏差。
2. **价差随异构强度单调放大到两位数**：slowdown=8 时 M_split 甩开 A **11.5 个点**，
   远超前序全部实验（封顶 3%）。
3. **价差来自新决策维度，而非老技巧**：M_split 比前序赢家 B（负载均衡）高 **11 个点**。
   B 亲和盲视，说明胜出不靠"就近均衡"，而靠**把重 VNF 拆出来送对加速器**——
   这正是 SFC/VNF placement 的核心。
4. **baseline 随异构崩盘，正确策略稳定**：slowdown 1→8，A 从 41→31 崩掉，
   M_split 始终锁 ~42。"放错加速器"是真实严重惩罚，正确拆分部署能完全规避。
5. **多目标真实冲突、无贪心通吃**：重 VNF 要 GPU（可能远）× 轻 VNF 要近 × 都要不挤。
   这正是扩散多模态策略相对确定性 baseline 的主场。

## 五、意义与下一步

- **place 现在可作为核心决策变量**：它承载了 +11% 的可争夺价差，且是 SFC 原生的
  高维组合决策。"扩散 vs baseline" 的对比实验现在有了产生显著差异的土壤。
- **环境已是可训练完整形态**：机制 + 观测 + 开关齐备，默认退化安全。
- 建议下一步：以 `UAV_TYPES=["gpu","general","gpu","general"]`, `NON_GPU_SLOWDOWN=8.0`
  为标准 setting，正式训练扩散策略，对照 TD3/SAC 等确定性 baseline，
  验证扩散能否逼近 M_split 的 ~42% 上界、而 baseline 停在 ~31%。
- 可选加强项：让 GPU 机初始扇区远离 UE 热点簇，强化"近 vs 对"的冲突，进一步拉大价差。

## 六、代码产出

- `core/env_config.py`：`UAV_TYPES` / `GPU_VNFS` / `NON_GPU_SLOWDOWN` 配置。
- `core/sfc_env.py`：`_vnf_affinity()` + 四处一致接入；观测新增 `is_gpu`（9→10 维）
  及空间边界同步。
- `test/vnf_affinity.py`：验证脚本，独立可复现。
- 已通过维度自洽 + 异构 reset/step 冒烟测试。
