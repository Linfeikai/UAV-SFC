import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import copy
from .env_config import DEFAULT_CONFIG, Nodetype, DATA_SIZE_RANGES
from .sfc_config import SFC_DEADLINES, VNF_TEMPLATES, SFC_STRUCTURES
from .uav import UAVNode, calculate_uav_power
from .sfc import SFC
from .uenode import UENode
from .laser_charger import LaserCharger
# from .charger import Charger


class SFCEnv(gym.Env):
    """
    A custom Gym environment for Service Function Chaining (SFC) tasks.
    """

    def __init__(self, render_mode=None, config=None):
        # 1. 复制一份默认配置
        self.config = copy.deepcopy(DEFAULT_CONFIG)

        # 2. 如果外部传了新参数（做实验时），覆盖默认值
        if config is not None:
            self.config.update(config)

        self.N = self.config["NUM_UAVS"]
        self.ue_num = self.config["NUM_UES"]
        self.L = self.config["MAX_VNF_LEN"]
        self.M = self.config["M"]  # 候选池大小
        self.K = self.config["K"]  # 候选池中选K个任务
        self.grid_res = self.config["GRID_RES"]  # 3x3 网格
        self.current_step = 0

        # 统一使用 self.uavs / self.ues
        self.uavs: list[UAVNode] = []
        self.ues: list[UENode] = []
        self.chargers: list[LaserCharger] = []  # 初始化 充电桩 列表
        self._create_entities()
        self.time_slot = self.config["TIME_SLOT_DURATION"]
        self.dt_fly = 1.0  # 假设每次飞行时间为 1s
        self.current_time = 0.0
        self.current_cand_tasks = []  # 当前 Slot 候选任务列表
        # 注意：_create_entities 已经创建并填充 self.uavs / self.ues 列表

        # 定义动作空间 (Box 代表连续空间)
        # 动作包含两部分：移动 + 部署
        # 为了方便 Diffusion，通常把它们展平成一个大向量，或者作为 Dict
        # 这里假设展平成一个大的一维向量，进环境再 reshape

        dim_mobility = self.N * 2
        dim_pick = self.K
        dim_place = self.K * self.L * 2  # 注意这里乘了 2
        total_dim = dim_mobility + dim_pick + dim_place

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_dim,), dtype=np.float32
        )

        # ⭐ 核心步骤 1：计算标量悬停功率
        # 传入 np.array([0.0]) 以便向量化函数正确运行
        P_hover_array = calculate_uav_power(np.array([0.0]))

        # 提取标量值（P_hover_array[0]）
        P_hover_scalar = P_hover_array[0]

        # 计算标量悬停能耗
        E_hover_scalar = P_hover_scalar * self.dt_fly

        # ⭐ 核心步骤 2：生成 N 份能耗数组
        # 使用 np.full() 创建一个 N 维数组，所有值都等于 E_hover_scalar
        self.hover_energy = np.full(self.N, E_hover_scalar, dtype=np.float32)

        self.uav_load_status = np.zeros(self.N, dtype=np.float32)  # UAV 计算负载状态

        # Precompute maxima for normalization in observations
        # MAX_DATA: take the maximum 'max_bits' across DATA_SIZE_RANGES entries
        try:
            self._max_data = float(max(v[1] for v in DATA_SIZE_RANGES.values()))
        except Exception:
            # Fallback: if DATA_SIZE_RANGES is not structured as expected
            self._max_data = 1.0

        # Maximum cycles per bit among all VNF templates (complexity is cycles/bit)
        try:
            max_complexity = max(float(t["complexity"]) for t in VNF_TEMPLATES.values())
        except Exception:
            max_complexity = 1.0

        # Longest chain length
        try:
            max_chain_len = max(len(c) for c in SFC_STRUCTURES.values())
        except Exception:
            max_chain_len = 1

        # Upper bounds for normalization
        self._max_cycles_per_vnf = max_complexity * self._max_data
        self._max_total_cycles = max_complexity * self._max_data * float(max_chain_len)

        # ==========================================
        # 4. 观测空间 (Observation Space)
        # ==========================================
        # --- 原有的物理状态 (62维) ---
        # (1) UAV: N * 9 (6基础 + 2相对充电桩 + 1负载)
        # (2) Grid: grid_res * grid_res * 3 = 27
        # (3) Candidates: M * 5 (x, y, data, cycles, time)
        # (4) Global: 3 (step_norm, charger_x, charger_y)

        # --- A. UAV Bounds (9维) ---
        # [x, y, vx, vy, batt, crashed, dx, dy, load]
        uav_low = np.array([0, 0, -1, -1, 0, 0, -1, -1, 0], dtype=np.float32)
        uav_high = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)

        # --- B. Grid Bounds (27维) ---
        # 密度、重量、热度的累加值，给 10.0 作为安全上限
        grid_low = np.zeros(self.grid_res**2 * 3, dtype=np.float32)
        grid_high = np.full(self.grid_res**2 * 3, 1.2, dtype=np.float32)

        # --- C. Candidate Bounds (5维) ---
        # [x, y, data, cycles, time_left]
        cand_low = np.array([0, 0, 0, 0, -1], dtype=np.float32)
        cand_high = np.array([1, 1, 1, 1, 1], dtype=np.float32)

        # --- D. Global Bounds (3维) ---
        # [step_norm, charger_x, charger_y]
        global_low = np.array([0, 0, 0], dtype=np.float32)
        global_high = np.array([1, 1, 1], dtype=np.float32)

        low_state = np.concatenate(
            [np.tile(uav_low, self.N), grid_low, np.tile(cand_low, self.M), global_low]
        )
        high_state = np.concatenate(
            [
                np.tile(uav_high, self.N),
                grid_high,
                np.tile(cand_high, self.M),
                global_high,
            ]
        )

        # --- B. 动作遮罩空间 (针对 Ray Mask 算法) ---
        # 1. Mobility Bounds [N, 4]: 对应每架 UAV 的 [左, 右, 下, 上] 缩放比例
        # 取值 0.0 代表该方向完全封死，1.0 代表全速 [cite: 139]
        mob_low = np.zeros((self.N, 4), dtype=np.float32)
        mob_high = np.ones((self.N, 4), dtype=np.float32)

        # 2. Pick Limit [1]: 定义 1D 索引投影的有效上限 [cite: 199]
        pick_low = np.array([-1.0], dtype=np.float32)
        pick_high = np.array([1.0], dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=low_state, high=high_state, dtype=np.float32),
                "mobility_bounds": spaces.Box(
                    low=0.0, high=1.0, shape=(self.N * 4,), dtype=np.float32
                ),
                "pick_limit": spaces.Box(
                    low=pick_low, high=pick_high, dtype=np.float32
                ),
            }
        )

    def _create_entities(self):
        """创建 UAV、UE 和 充电桩 实例"""
        # 创建 UAV 实例
        for i in range(self.N):
            uav = UAVNode(
                node_id=i,
                cpu_freq=self.config["UAV_CPU_FREQ"],
                max_speed=self.config["UAV_MAX_SPEED"],
                battery_capacity=self.config["BATTERY_CAPACITY"],
                charging_power=self.config["UAV_CHARGING_POWER"],
            )
            self.uavs.append(uav)

        # 创建 UE 实例
        # 1. 计算数量
        target_counts = {}
        total_assigned = 0
        distribution = self.config.get(
            "NODE_TYPE_DISTRIBUTION",
            {
                Nodetype.Video: 0.4,
                Nodetype.Security: 0.4,
                Nodetype.IoT: 0.2,
            },
        )

        for n_type, ratio in distribution.items():
            count = int(self.ue_num * ratio)
            target_counts[n_type] = count
            total_assigned += count

        # 2. 补齐余数 (给 IoT)
        remainder = self.ue_num - total_assigned
        if remainder > 0:
            target_counts[Nodetype.IoT] += remainder

        # 3. 生成固定列表 (比如 [V, V, S, S, I])
        self.fixed_type_list = []
        for n_type, count in target_counts.items():
            self.fixed_type_list.extend([n_type] * count)

        # 确保预计算无误
        assert len(self.fixed_type_list) == self.ue_num, "Type list length mismatch!"

        # 4. 创建 UE 实例时，按顺序分配类型

        for j in range(self.ue_num):
            ue = UENode(
                node_id=j,
                nodetype=self.fixed_type_list[j],
                deadline_factor=self.config["DEADLINE_FACTOR"],
            )
            ue.arrival_prob = self.config["ARRIVAL_PROB"]
            self.ues.append(ue)

        # 创建 充电桩 实例
        charger = LaserCharger(
            node_id=0,
            loc=np.array([250.0, 250.0]),  # 假设放在地图中心
            effective_charge_radius=self.config["CHG_RADIUS"],  # 充电有效半径
            k_charge_scale=self.config["CHG_K_SCALE"],  # 充电功率缩放系数
        )
        self.chargers.append(charger)
        # 兼容旧代码：提供单一 charger 快速访问
        self.charger = charger

    def _handle_mobility_and_energy(self, mobility_act):
        """
        处理 飞行、能量、边界、碰撞、充电 的耦合逻辑
        遵循：能量预判 -> 动作修正 -> 物理移动 -> 状态更新
        处理物理移动、能量消耗、碰撞检测。
        Returns:
            flight_info (dict): 包含本次飞行的统计信息，用于计算 Reward 和 Info。
                - energy_cost: float (所有 UAV 总耗能)
                - collision_count: int
                - is_crashed: bool (是否发生坠毁)
                - out_of_bound_count: int (可选，越界次数)
        Args:
            mobility_act (np.ndarray): 形状 (num_uavs, 2) 的数组，表示每个 UAV 在 x 和 y 方向上的速度指令，范围 [-1, 1]。
        """
        # 0. 准备数据
        flight_info = {
            "energy_cost": 0.0,
            "collision_count": 0,
            "is_crashed": False,
            "crashed_uavs": [],
        }
        # 1. 动作解析 (Action Parsing)
        # 将 [-1, 1] 映射到 [-MAX_SPEED, MAX_SPEED]
        # a.获取uav最开始的速度 [4,2]
        initial_velocities = np.array([u.velocity for u in self.uavs])
        # b.计算速度变化率 [4,2]
        delta_vs = UAVNode.MAX_ACCELERATION * mobility_act * self.dt_fly
        # c.计算期望速度 [4,2]
        simulated_velocities = initial_velocities + delta_vs
        # c-1.限制速度
        max_speed_sq = self.config["UAV_MAX_SPEED"] ** 2
        # # 计算当前合速度的平方 (N,)
        # simulated_velocities[:, 0] 是所有 UAV 的 Vx 分量
        # simulated_velocities[:, 1] 是所有 UAV 的 Vy 分量
        current_speed_sq = (
            simulated_velocities[:, 0] ** 2 + simulated_velocities[:, 1] ** 2
        )
        # 找出缩放因子： V_max / V_current
        # 如果 V_current > V_max，则 scale < 1
        scale_factors = np.where(
            current_speed_sq > 0, np.sqrt(max_speed_sq / current_speed_sq), 1.0
        )
        # 确保只缩减超速的，未超速的保持 scale=1.0
        scale_factors = np.minimum(scale_factors, 1.0)
        # 重塑为 (N, 1) 形状以便与 (N, 2) 的速度数组进行广播
        scale_factors = scale_factors[:, np.newaxis]
        # 对超速的 UAVs 进行缩放 (钳制)
        # 这一步将 simulated_velocities 限制在了 max_speed 范围内
        simulated_velocities = simulated_velocities * scale_factors
        # d.计算平均速度 [N, 2]
        # 用于计算该时间步长内的位移和能量消耗
        average_velocities = (initial_velocities + simulated_velocities) / 2.0
        # d-1 计算平均合速度：
        V_avg_magnitude = np.sqrt(
            average_velocities[:, 0] ** 2 + average_velocities[:, 1] ** 2
        )

        # 2. 能量预计算 (Pre-calculation)
        # 计算 Agent "想飞" 的动作需要多少能量
        powers_needed = calculate_uav_power(V_avg_magnitude)  # 功率数组 [N,]
        energies_needed = powers_needed * self.dt_fly  # 能

        current_batteries = np.array([u.e_battery for u in self.uavs])
        # 这是一个布尔数组
        insufficient_energy_for_flight = current_batteries < energies_needed

        # 2. 对电量不足的 UAVs，检查他们是否足以支撑“悬停能耗”
        # 这一步只关心那些 insufficient_energy_for_flight 为 True 的 UAV
        # 创建一个掩码，标记那些“应该悬停”的 UAV
        # 条件是：(电池不够飞 AND 电池 >= 悬停所需能耗)
        should_hover_mask = np.logical_and(
            insufficient_energy_for_flight, current_batteries >= self.hover_energy
        )

        # 3. 对那些“应该悬停”的 UAVs，强制它们的速度为 0

        # 向量化更新 V_avg_magnitude (合速度)
        # 将所有应该悬停的位置设置为 0
        # 当您将一个布尔掩码放在方括号 [] 内对 NumPy 数组进行索引时，NumPy 会执行以下操作：
        # 筛选（Filtering）： NumPy 遍历 should_hover_mask 数组的每一个元素。
        # 提取（Extraction）： 它只提取出原始数组 V_avg_magnitude 中，对应位置为 True 的那些元素。

        V_avg_magnitude[should_hover_mask] = 0.0
        # 向量化更新 average_velocities (速度向量)
        # 将所有应该悬停的位置设置为 [0.0, 0.0]
        # 注意：需要将 (N,) 的布尔掩码扩展为 (N, 2) 才能应用于 (N, 2) 数组
        average_velocities[should_hover_mask] = 0.0

        # 4. 对电量不足以悬停的 UAVs，留空处理

        # 找出电量不足以悬停的 UAVs
        critical_energy_mask = current_batteries < self.hover_energy
        critical_uav_indices = np.where(critical_energy_mask)[0]
        if critical_uav_indices.size > 0:
            crashed = list(critical_uav_indices)  # 转为 Python int 列表
            for idx in crashed:
                self.uavs[idx].is_crashed = True
            # 记录到 flight_info 以便外部使用（step() 会检查并提前返回）
            flight_info["is_crashed"] = True
            flight_info["crashed_uavs"].extend(crashed)

            return flight_info

        # 3. 边界判断
        uav_current_locs = np.array([u.loc for u in self.uavs])
        displacement = average_velocities * self.dt_fly  # 保留方向信息
        proposed_locs = uav_current_locs + displacement  # (N,2)
        clamped_locs = np.clip(proposed_locs, 0, self.config["GROUND_WIDTH"])
        # 4.碰撞检测：如果两者相撞，都回到原来的位置 not moved
        # 这里要增加penalty！！告诉agent你不能让两个uav碰撞
        collision_mask = np.zeros(self.N, dtype=bool)
        final_locs = clamped_locs.copy()

        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist = np.linalg.norm(final_locs[i] - final_locs[j])
                if dist < self.config["UAV_MIN_SAFE_DISTANCE"]:
                    collision_mask[i] = True
                    collision_mask[j] = True
                    # Count this pairwise collision event
                    flight_info["collision_count"] += 1
                    final_locs[i] = uav_current_locs[i]  # Freeze
                    final_locs[j] = uav_current_locs[j]  # Freeze
                    average_velocities[i] = 0.0
                    average_velocities[j] = 0.0
                    V_avg_magnitude[i] = 0.0
                    V_avg_magnitude[j] = 0.0

        # 经过这两个判断之后，应该现在的 final_locs 就是合法的了
        # 需要更新位置、然后计算能量消耗
        for i in range(self.N):
            energy_consumed = self.uavs[i].moveto(
                final_locs[i], average_velocities[i], V_avg_magnitude[i], self.dt_fly
            )
            flight_info["energy_cost"] += energy_consumed
            if self.uavs[i].is_crashed:
                flight_info["is_crashed"] = True
                flight_info["crashed_uavs"].append(i)

        # # 经过这个函数后，现在我们就假设uav已经飞到本episode的固定位置，能量已经扣除了。我们假设这里每个uav都是活着的。
        return flight_info

    def _handle_charging(self):
        """
        Arbitration and execution of charging for UAVs within charger range.

        - Resets per-UAV `harvested_energy_last_step`.
        - For each charger, finds UAVs in range (excluding crashed ones),
          selects the lowest-battery UAV (tie-break by distance) and
          calls `charger.charge(...)` for the remaining compute window.
        - Writes returned energy into `uav.harvested_energy_last_step`.
        """
        charged = []
        total = 0.0
        seconds_used = 0.0
        charge_duration = max(0.0, self.time_slot - self.dt_fly)

        # 清零上一步记录（以防止历史累积）
        for u in self.uavs:
            u.clear_harvested_energy()

        for charger in self.chargers:
            candidates = []  # list of (uav_index, battery, distance)
            for i, uav in enumerate(self.uavs):
                if uav.is_crashed:
                    continue
                dx = charger.loc[0] - uav.loc[0]
                dy = charger.loc[1] - uav.loc[1]
                dh = getattr(uav, "flying_height", 0.0)
                dist_3d = math.sqrt(float(dx) ** 2 + float(dy) ** 2 + float(dh) ** 2)
                if charger._is_uav_in_range(dist_3d):
                    candidates.append((i, float(uav.e_battery), dist_3d))

            if not candidates:
                continue

            # 选电量最低的 UAV，距离作为次级排序以打破平局  先比较电量，再比较距离，不返回新列表，修改原列表
            candidates.sort(key=lambda x: (x[1], x[2]))
            winner_idx = candidates[0][0]
            charged.append(winner_idx)
            winner_uav = self.uavs[winner_idx]

            # 执行充电并写回 UAV 实例（通过封装方法）
            try:
                harvested = charger.charge(winner_uav, charge_duration)
            except Exception:
                harvested = 0.0
            total += float(harvested)

            winner_uav.record_harvested_energy(harvested, apply=False)
            if harvested > 0:
                seconds_used += charge_duration
        # 充电桩利用率计算
        util = seconds_used / max(len(self.chargers) * charge_duration, 1e-9)

        return {
            "charge/num_charged": len(charged),  # 本次充电的uav数量 目前只可能是0/1
            "charge/total_harvested": total,  # 本次充电总能量
            # "charge/charged_uavs": charged, #本次充电的uav列表
            "charge/charger_utilization": util,
        }

    def _trim_mapping(self, sfc: SFC, mapping: np.ndarray) -> np.ndarray:
        chain_len = len(getattr(sfc, "vnf_chain"))
        m = np.asarray(mapping, dtype=np.int64)
        return m[:chain_len]

    def _hard_cap_drop_sfcs(
        self, candidate_tasks: list[tuple[int, SFC, np.ndarray]]
    ) -> set[int]:
        """
        群体容量检查：基于 UAV 物理算力上限，判定哪些任务因竞争失败需丢弃。

        Args:
            candidate_tasks: 列表，每个元素为 (ue_id, sfc_instance, vnf_uav_map)
                            - ue_id: 来源 UE 的索引 (int)
                            - sfc_instance: SFC 类的实例，包含 vnf_chain
                            - vnf_uav_map: 该 SFC 对应的 UAV 分配方案 (np.array)

        Returns:
            dropped_ue_ids: 需要被丢弃的 UE ID 集合 (set[int])
        """
        # --- 1. 初始化容量与状态 ---

        dt_compute = float(self.time_slot - self.dt_fly)  # 可用计算时间窗口
        cap_cycles = np.array(
            [float(u.cpu_freq) * dt_compute for u in self.uavs],
            dtype=np.float32,
        )

        dropped_ue_ids = set()

        # --- 2. 迭代检查负载 (考虑丢弃后的链式反应) ---
        while True:
            # 统计当前“幸存”任务对每个 UAV 造成的总负载
            current_load = np.zeros(self.N, dtype=np.float32)

            for ue_id, sfc, mapping in candidate_tasks:
                if ue_id in dropped_ue_ids:
                    continue

                # 累加该 SFC 的所有 VNF 到对应的 UAV
                for vnf_idx, uav_id in enumerate(mapping):
                    current_load[uav_id] += sfc.vnf_chain[vnf_idx].required_cycles

            # 检查哪些 UAV 依然超载 (容差 1.0 cycle 避免浮点误差)
            overloaded_uavs = np.where(current_load > cap_cycles + 1e-1)[0]

            if overloaded_uavs.size == 0:
                break  # 所有 UAV 均在负荷内，退出

            # --- 3. 丢弃决策 (FIFO 策略) ---
            changed = False
            # 从列表末尾（即最后生成的任务）开始向前搜索，寻找占用超载 UAV 的“倒霉蛋”
            for uav_id in overloaded_uavs:
                for ue_id, sfc, mapping in reversed(candidate_tasks):
                    if ue_id in dropped_ue_ids:
                        continue

                    # 如果这个任务在这个超载的 UAV 上分配了计算任务
                    if uav_id in mapping:
                        dropped_ue_ids.add(ue_id)
                        changed = True
                        break  # 丢弃该任务后，重新计算全局负载 (因为一个任务可能占用多个 UAV)

                if changed:
                    break  # 重新进入 while True 循环进行下一轮审计

            if not changed:
                break  # 防止极端情况下的死循环

        return dropped_ue_ids

    def _estimate_sfc_duration(self, sfc, vnf_uav_map, ue_loc):
        """
        估算 SFC 在给定映射下的总耗时（上行 + 计算 + 传输）
        用于单体 7s 检查
        """
        # 传输功率
        P_TX_CROSS = self.config["UAV_P_TX_CROSS"]

        # Uplink time estimate
        first_uav = self.uavs[vnf_uav_map[0]]
        rate_up = self._calculate_rate(
            ue_loc, first_uav.loc, self.config["P_UPLINK"], link_type="UAV_UE"
        )
        rate_up = max(float(rate_up), 1e-9)
        t_up = sfc.vnf_chain[0].data_in / rate_up

        # Compute time estimate (sum of required cycles / cpu_freq)
        # 假设每个uav都是一样的cpu频率
        comp_time_sum = 0.0
        for vnf_idx, uav_id in enumerate(vnf_uav_map):
            vnf = sfc.vnf_chain[vnf_idx]
            comp_time_sum += vnf.required_cycles / self.uavs[uav_id].cpu_freq

        # Transmission time estimate (conservative)
        tx_time_sum = 0.0
        for vnf_idx in range(len(vnf_uav_map) - 1):
            uav_id = vnf_uav_map[vnf_idx]
            next_uav_id = vnf_uav_map[vnf_idx + 1]
            if uav_id != next_uav_id:
                uav_curr = self.uavs[uav_id]
                uav_next = self.uavs[next_uav_id]
                rate = self._calculate_rate(
                    uav_curr.loc, uav_next.loc, p_tx=P_TX_CROSS, link_type="UAV_UAV"
                )
                rate = max(float(rate), 1e-9)
                tx_time_sum += sfc.vnf_chain[vnf_idx].data_out / rate

        estimated_total = t_up + comp_time_sum + tx_time_sum
        return estimated_total

    def _task_admission_stage(self, chosen_tasks):
        """
        负责把不合格的任务刷掉
        """
        # rewards = np.zeros(self.ue_num)  # 每个 UE 的奖励
        # failed_count = 0  # 本 slot 失败的任务数
        # 初始化统计
        stage_rewards = np.zeros(self.ue_num, dtype=np.float32)
        dropped_count = 0
        pre_eligible_tasks = []

        dt_compute_window = self.time_slot - self.dt_fly

        for ue_id, sfc, vnf_uav_map in chosen_tasks:
            # A. Zombie Check
            if self.current_time >= sfc.deadline:
                self.ues[ue_id].task_buffer.popleft()
                stage_rewards[ue_id] += self.config["RWD_DROP"]
                dropped_count += 1
                continue

            # B. 单体可行性检查 (7s物理窗口)
            estimated_total = self._estimate_sfc_duration(
                sfc, vnf_uav_map, self.ues[ue_id].loc
            )
            if estimated_total > dt_compute_window:
                self.ues[ue_id].task_buffer.popleft()
                stage_rewards[ue_id] += self.config["RWD_DROP"]
                dropped_count += 1
                continue

            pre_eligible_tasks.append((ue_id, sfc, vnf_uav_map))

        # --- B. 群体硬上限检查 ---
        dropped_by_cap = self._hard_cap_drop_sfcs(pre_eligible_tasks)

        final_tasks = []
        for ue_id, sfc, mapping in pre_eligible_tasks:
            if ue_id in dropped_by_cap:
                self.ues[ue_id].task_buffer.popleft()
                # 记录惩罚
                stage_rewards[ue_id] += self.config["RWD_DROP"]
                dropped_count += 1
                continue
            final_tasks.append((ue_id, sfc, mapping))

        # 返回一个“结果包”
        return {
            "eligible_tasks": final_tasks,
            "stage_rewards": stage_rewards,
            "dropped_count": dropped_count,
        }

    def _account_resource_usage(
        self, eligible_tasks: list[tuple[int, SFC, np.ndarray]]
    ):
        """
        资源账单统计：计算每个 UAV 承载的总计算负载和总传输能耗需求。

        Args:
            eligible_tasks: 通过准入控制的合格任务列表 [(ue_id, sfc, mapping), ...]

        Returns:
            usage_stats: 字典，包含：
                - uav_total_cycles: np.array (N,) 每个 UAV 被分配的总周期
                - uav_tx_energy_bill: np.array (N,) 每个 UAV 需支付的传输能耗 (J)
                - task_mappings: dict {ue_id: mapping} 方便后续阶段查询
        """
        # 1. 初始化统计容器
        uav_total_cycles = np.zeros(self.N, dtype=np.float32)
        uav_tx_energy_bill = np.zeros(self.N, dtype=np.float32)
        task_mappings = {}

        # 获取传输功率配置 (0.5W)
        P_TX_CROSS = self.config["UAV_P_TX_CROSS"]

        # 2. 遍历每一个合格任务
        for ue_id, sfc, vnf_uav_map in eligible_tasks:
            task_mappings[ue_id] = vnf_uav_map

            # 遍历 SFC 中的每一个 VNF
            for vnf_idx, uav_id in enumerate(vnf_uav_map):
                vnf = sfc.vnf_chain[vnf_idx]

                # --- A. 累加计算负载 ---
                uav_total_cycles[uav_id] += vnf.required_cycles

                # --- B. 累加传输能耗 (如果涉及跨机传输) ---
                # 只有当不是最后一个 VNF，且下一跳在不同 UAV 上时，才产生 Backhaul 能耗
                if vnf_idx < len(vnf_uav_map) - 1:
                    next_uav_id = vnf_uav_map[vnf_idx + 1]

                    if uav_id != next_uav_id:
                        # 获取当前 UAV 和下一跳 UAV
                        uav_curr = self.uavs[uav_id]
                        uav_next = self.uavs[next_uav_id]

                        # 计算跨机传输速率 (UAV-to-UAV)
                        rate = self._calculate_rate(
                            uav_curr.loc,
                            uav_next.loc,
                            p_tx=P_TX_CROSS,
                            link_type="UAV_UAV",
                        )

                        # 防御性：避免速率过低导致能耗计算爆炸
                        rate = max(float(rate), 1e-9)

                        # 传输时间 = 数据量(bits) / 速率(bps)
                        t_tx = vnf.data_out / rate
                        # 传输能耗 = 功率(W) * 时间(s)
                        e_tx = P_TX_CROSS * t_tx

                        # 记在发送方 (uav_id) 的账上
                        uav_tx_energy_bill[uav_id] += e_tx

        return {
            "uav_total_cycles": uav_total_cycles,
            "uav_tx_energy_bill": uav_tx_energy_bill,
            "task_mappings": task_mappings,
        }

    def _bill_uav_energy(self, usage_stats):
        """
        能量结算：执行电量扣除。

        Args:
            usage_stats: 包含 uav_total_cycles, uav_tx_energy_bill 的字典

        Returns:
            billing_report: 字典，包含：
                - uav_compute_energy: np.array (N,) 每个 UAV 消耗的计算能耗
                - uav_crashed: bool 是否有 UAV 在此阶段坠毁
                - crashed_uav_ids: list 坠毁的 UAV 索引
        """
        # 1. 基础参数准备
        uav_total_cycles = usage_stats["uav_total_cycles"]
        uav_tx_energy_bill = usage_stats["uav_tx_energy_bill"]

        dt_compute = self.time_slot - self.dt_fly
        FULL_LOAD_POWER = self.config["UAV_FULL_LOAD_POWER"]
        cpu_freqs = np.array([u.cpu_freq for u in self.uavs])
        cap_cycles = cpu_freqs * dt_compute

        # --- 【关键步骤】：更新负载状态 ---
        # 计算负载率 (0.0 ~ 1.0)
        load_ratio = usage_stats["uav_total_cycles"] / (cap_cycles + 1e-9)
        # 存入实例变量，供下一轮 _get_obs 读取
        self.uav_load_status = np.clip(load_ratio, 0.0, 1.0)

        # 3. 能量扣除与统计
        uav_compute_energy = np.zeros(self.N, dtype=np.float32)
        crashed_uav_ids = []
        uav_crashed_flag = False

        for uav_id, uav in enumerate(self.uavs):
            if uav.is_crashed:
                continue

            # A. 计算计算能耗 (E = P * t)
            # 实际忙碌时间 = 总周期 / 频率
            busy_time = uav_total_cycles[uav_id] / uav.cpu_freq
            e_comp = FULL_LOAD_POWER * busy_time
            uav_compute_energy[uav_id] = e_comp

            # B. 汇总总账单 (计算 + 传输)
            total_energy_needed = e_comp + uav_tx_energy_bill[uav_id]

            # C. 扣除电量并检查坠毁
            if total_energy_needed > 0:
                is_dead = uav.consume_energy(total_energy_needed)
                if is_dead:
                    uav_crashed_flag = True
                    crashed_uav_ids.append(uav_id)

        return {
            "uav_compute_energy": uav_compute_energy,
            "uav_crashed": uav_crashed_flag,
            "crashed_uav_ids": crashed_uav_ids,
        }

    def _handle_billing_crash(self, billing_info, eligible_tasks):
        """
        结算坠毁处理：当 UAV 在计算/传输过程中耗尽电量时，清理受影响的任务并给予惩罚。

        Args:
            billing_report: 包含 crashed_uav_ids, uav_compute_energy 的字典
            eligible_tasks: 准入阶段通过的候选任务列表

        Returns:
            与 _process_sfc_tasks 结构一致的结算字典
        """
        rewards = np.zeros(self.ue_num, dtype=np.float32)
        # 专门统计因坠毁导致的失败数
        crash_failed_count = 0
        crashed_uav_ids = set(billing_info["crashed_uav_ids"])

        for ue_id, sfc, mapping in eligible_tasks:
            # 检查任务路径中是否有坠毁的 UAV
            if not set(mapping).isdisjoint(crashed_uav_ids):
                # 【关键修改】：使用专门的链路中断惩罚
                rewards[ue_id] += self.config["RWD_LINK_BROKEN"]
                crash_failed_count += 1

                if self.ues[ue_id].task_buffer:
                    self.ues[ue_id].task_buffer.popleft()
                    sfc.status = -2  # 状态码：因坠毁中断

        total_compute = float(np.sum(billing_info["uav_compute_energy"]))

        return {
            "rewards": rewards,
            "completed_count": 0,
            "dropped_count": 0,  # 此阶段不产生准入丢弃
            "timeout_count": 0,  # 此阶段不产生普通超时
            "crash_failed_count": crash_failed_count,  # 新增统计项
            "failed_count": crash_failed_count,
            "avg_latency": 0.0,
            "total_compute_energy": total_compute,
            "total_tx_energy": 0.0,
            "uav_crashed": True,
            "crashed_uavs": list(crashed_uav_ids),
        }

    def _evaluate_performance(self, eligible_tasks, usage_stats, billing_report):
        """
        性能与奖励评估：计算每个 SFC 的实际延迟，判定成功/失败，并生成奖励。

        Args:
            eligible_tasks: 通过准入的候选任务列表
            usage_stats: 包含 task_mappings 等统计信息
            billing_report: 包含 结算信息

        Returns:
            sfc_results: 包含 rewards(array), completed_count, failed_count, avg_latency 等
        """
        # 1. 基础参数准备
        rewards = np.zeros(self.ue_num, dtype=np.float32)
        completed_count = 0
        timeout_count = 0
        total_latency_sum = 0.0

        P_TX_CROSS = self.config["UAV_P_TX_CROSS"]
        P_UPLINK = self.config["P_UPLINK"]

        # 2. 逐一评估任务 (只有通过了准入和容量检查的任务才会到这里)
        for ue_id, sfc, mapping in eligible_tasks:
            ue = self.ues[ue_id]
            current_delay = 0.0

            # --- A. 上行延迟 (UE -> 第一个 UAV) ---
            first_uav_id = mapping[0]
            first_uav = self.uavs[first_uav_id]
            rate_up = self._calculate_rate(
                ue.loc, first_uav.loc, P_UPLINK, link_type="UAV_UE"
            )
            rate_up = max(float(rate_up), 1e-9)
            t_up = sfc.vnf_chain[0].data_in / rate_up
            current_delay += t_up

            # --- B. 链式处理与传输延迟 ---
            for vnf_idx, uav_id in enumerate(mapping):
                vnf = sfc.vnf_chain[vnf_idx]

                # 1. 计算处理延迟
                t_comp = vnf.required_cycles / self.uavs[uav_id].cpu_freq
                current_delay += t_comp

                # 2. 跨 UAV 传输延迟 (如果有下一跳)
                if vnf_idx < len(mapping) - 1:
                    next_uav_id = mapping[vnf_idx + 1]
                    if uav_id != next_uav_id:
                        uav_curr = self.uavs[uav_id]
                        uav_next = self.uavs[next_uav_id]
                        rate_cross = self._calculate_rate(
                            uav_curr.loc, uav_next.loc, P_TX_CROSS, link_type="UAV_UAV"
                        )
                        rate_cross = max(float(rate_cross), 1e-9)
                        t_trans = vnf.data_out / rate_cross
                        current_delay += t_trans

            # --- C. 结果结算 ---
            # 实际完成绝对时间 = 当前 Slot 开始时间(已加飞行1s) + 任务处理总延迟
            finish_time = self.current_time + current_delay

            # 判定是否满足 Deadline
            if finish_time <= sfc.deadline:
                # 成功！
                rewards[ue_id] += self.config["RWD_SUCCESS"]

                # 计算时延奖励 (Latency Bonus)
                actual_duration = finish_time - sfc.arrival_time
                max_allowed_duration = sfc.deadline - sfc.arrival_time
                ratio = np.clip(actual_duration / max_allowed_duration, 0.0, 1.0)

                bonus_weight = self.config.get("RWD_LATENCY_BONUS", 5.0)
                rewards[ue_id] += bonus_weight * (1.0 - ratio)

                completed_count += 1
                total_latency_sum += actual_duration
                sfc.status = 1  # Success
            else:
                # 这里的超时归因于：上行/跨机传输太慢，或者估计误差（不再是排队）
                rewards[ue_id] += self.config["RWD_TIMEOUT"]
                timeout_count += 1
                sfc.status = -1  # Failed

            # 任务处理结束，移出缓冲区
            ue.task_buffer.popleft()

        # 3. 统计汇总
        avg_latency = (
            total_latency_sum / completed_count if completed_count > 0 else 0.0
        )

        return {
            "rewards": rewards,
            "completed_count": completed_count,
            "timeout_count": timeout_count,
            "avg_latency": float(avg_latency),
        }

    def _process_sfc_tasks(self, chosen_tasks):
        res = {
            "rewards": np.zeros(self.ue_num, dtype=np.float32),  # 每个 UE 的奖励
            "completed_count": 0,  # 本 slot 成功完成的SFC数
            "dropped_count": 0,  # 本 slot 被准入控制丢弃的SFC数
            "timeout_count": 0,  # 本 slot 超时失败的SFC数
            "crash_failed_count": 0,  # 本 slot 因 UAV 坠毁失败的SFC数
            "failed_count": 0,  # 本 slot 失败的SFC数（含超时和坠毁）
            "avg_latency": 0.0,  # 成功完成的 SFC 平均时延
            "total_compute_energy": 0.0,  # 本 slot UAV 计算能耗总和
            "total_tx_energy": 0.0,  # 本 slot UAV 传输能耗总和
            "uav_crashed": False,  # 本 slot 是否有 UAV 坠毁
            "crashed_uavs": [],  # 本 slot 坠毁的 UAV 列表
        }

        # 1. 准入控制 (Admission Control)
        # 过滤掉已经过期的、以及单机 7s 窗口内绝对算不完的任务
        admission_res = self._task_admission_stage(chosen_tasks)
        # 将准入阶段的奖励和计数直接更新进模版
        res["rewards"] += admission_res["stage_rewards"]
        res["dropped_count"] = admission_res["dropped_count"]
        res["failed_count"] += admission_res["dropped_count"]

        eligible_tasks = admission_res["eligible_tasks"]

        # 2. 资源统计 (Resource Accounting)
        # 统计每个 UAV 承载的总计算量和传输能耗需求
        usage_stats = self._account_resource_usage(eligible_tasks)

        # 3. 能量结算 (Energy Billing & Crash Check)
        # 真正扣除 UAV 电池，处理因计算/传输导致的坠毁
        billing_info = self._bill_uav_energy(usage_stats)
        res["total_compute_energy"] = float(np.sum(billing_info["uav_compute_energy"]))
        res["total_tx_energy"] = float(np.sum(usage_stats["uav_tx_energy_bill"]))

        # 如果结算阶段发现有 UAV 坠毁，提前中止（Short-circuit）
        if billing_info["uav_crashed"]:
            # 这里失败是电量没规划好，计算+传输电量不够了
            crash_res = self._handle_billing_crash(billing_info, eligible_tasks)
            res["rewards"] += crash_res["rewards"]
            res["crash_failed_count"] = crash_res["crash_failed_count"]
            res["failed_count"] += crash_res["crash_failed_count"]
            res["uav_crashed"] = True
            res["crashed_uavs"] = billing_info["crashed_uav_ids"]
        else:
            # 正常评估
            eval_res = self._evaluate_performance(
                eligible_tasks, usage_stats, billing_info
            )
            res["rewards"] += eval_res["rewards"]
            res["completed_count"] = eval_res["completed_count"]
            res["timeout_count"] = eval_res["timeout_count"]
            res["failed_count"] += eval_res["timeout_count"]
            res["avg_latency"] = eval_res["avg_latency"]

        return res

    def _calculate_reward(self, sfc_results, flight_info):
        # 1. 任务奖励 (主奖励)
        # 这里包含了 Success, Drop, Timeout, Link_Broken 的汇总
        r_task = np.sum(sfc_results.get("rewards", np.zeros(self.ue_num)))

        # --- 2. 能耗惩罚 (关键修改点) ---
        # A. 提取总能耗: sum(E_i)
        # 包含：计算能耗 + 传输能耗 + 飞行能耗 这里是总体之和
        e_compute = float(sfc_results.get("total_compute_energy", 0.0))
        e_tx = float(sfc_results.get("total_tx_energy", 0.0))
        e_flight = float(flight_info.get("energy_cost", 0.0))

        sum_E_i = e_compute + e_tx + e_flight

        # B. 准备公式分母: N * Battery_Cap
        N = self.N
        battery_cap = self.config["BATTERY_CAPACITY"]
        system_total_capacity = N * battery_cap

        # C. 应用公式计算 r_energy
        # W_ENERGY 建议设为 10.0 ~ 20.0，以匹配任务奖励的量级
        w_energy = float(self.config.get("W_ENERGY", 10.0))

        # 核心公式实现
        r_energy = -w_energy * (sum_E_i / system_total_capacity)

        # 3. 充电激励 (Shaping Reward)
        r_charge = 0.0
        w_charge = float(self.config.get("W_CHARGE", 1.0))
        for uav in self.uavs:
            harvested = getattr(uav, "harvested_energy_last_step", 0.0)
            if harvested > 0:
                # 电池越空，充电奖励越高 (Shaping)
                bat_ratio = uav.e_battery / max(1.0, uav.battery_capacity)
                # 同样对充电能量进行归一化
                r_charge += (
                    w_charge * (harvested / uav.battery_capacity) * (1.0 - bat_ratio)
                )

        # 4. 碰撞惩罚
        collision_count = int(flight_info.get("collision_count", 0))
        r_collision_weight = float(self.config.get("RWD_COLLISION", -5.0))
        r_collision = r_collision_weight * collision_count

        # 5. 总分汇总
        total_reward = r_task + r_energy + r_charge + r_collision

        # 【优化点：更详细的监控字典】
        reward_info = {
            "r_task": float(r_task),
            "r_energy": float(r_energy),
            "r_charge": float(r_charge),
            "r_collision": float(r_collision),
            "total_energy_J": float(sum_E_i),
            # 方便在 TensorBoard 查看不同失败类型的占比
            "count/completed": int(sfc_results.get("completed_count", 0)),
            "count/dropped": int(sfc_results.get("dropped_count", 0)),
            "count/timeout": int(sfc_results.get("timeout_count", 0)),
            "count/crash_fail": int(sfc_results.get("crash_failed_count", 0)),
        }

        return float(total_reward), reward_info

    def _update_state(self):
        # 要重新生成任务
        self.current_step_gen = 0

        for ue in self.ues:
            is_gen = ue.generate_task(self.current_time)
            self.current_step_gen += is_gen

        self.total_gen_tasks += self.current_step_gen  # 累加到全局

    def step(self, action):
        """
        根据 V1 决策方案（M 选 K）执行一个时间步
        """

        self.current_step += 1
        # --- 新增：初始化本步统计占位符，防止早退时 info 缺失 ---
        step_stats = {"completed_count": 0, "dropped_count": 0, "timeout_count": 0}

        truncated = False
        terminated = False
        # K 是每个 Slot 允许处理的最大任务数，L 是最大链长
        K, L, M, N = self.K, self.L, self.M, self.N
        X = 2  # 意图特征维度 (x, y)

        # 在执行动作前，统计目前 Buffer 里到底有多少活等着被处理
        total_tasks_on_table = sum(len(ue.task_buffer) for ue in self.ues)

        # --- 1. 动作切片 (根据 62 维重新分配) ---
        # Mobility: 0~7 (8维)
        raw_mobility = action[: N * 2]
        # Pick: 8~13 (6维)
        raw_pick = action[N * 2 : N * 2 + K]
        # Place Intent: 14~61 (48维)
        raw_place_intent = action[N * 2 + K :].reshape(K, L, X)

        # 2. 拆解动作并执行移动
        mobility_act = raw_mobility.reshape(self.N, 2)
        # 执行完这个函数后，uav的位置和能量都更新好了
        flight_info = self._handle_mobility_and_energy(mobility_act)
        self.current_time += self.dt_fly  # 更新时间
        if flight_info["is_crashed"]:
            # 给予巨大惩罚
            reward = self.config["RWD_CRASH"]
            terminated = True
            truncated = False

            # 填充 info (可以把 flight_info 放进去)
            info = {**step_stats, "perf/crashed": 1.0, **flight_info}

            return self._get_obs(), reward, terminated, truncated, info

        # --- 3. 任务筛选 (Pick) ---
        # 注意：这里依然保留了 floor，因为任务池 M 是离散的
        pick_indices = np.floor((raw_pick + 1) / 2 * (M + 1)).astype(np.int32)
        pick_indices = np.clip(pick_indices, 0, M)
        # --- 4. 意图解码部署 (Place Intent -> UAV ID) ---
        # 我们将 K*L*X 的意图矩阵转化为环境需要的 K*L 索引矩阵
        place_matrix = np.zeros((K, L), dtype=np.int32)

        width = self.config["GROUND_WIDTH"]
        height = self.config["GROUND_HEIGHT"]

        for k in range(K):
            for l in range(L):
                # 将 [-1, 1] 映射到地图坐标 [0, 500]
                intent_x = (raw_place_intent[k, l, 0] + 1) / 2 * width
                intent_y = (raw_place_intent[k, l, 1] + 1) / 2 * height
                intent_loc = np.array([intent_x, intent_y])

                # 寻找距离该意图坐标最近的 UAV
                best_uav_id = 0
                min_dist = float("inf")
                for uav in self.uavs:
                    if uav.is_crashed:
                        continue
                    dist = np.linalg.norm(uav.loc - intent_loc)
                    if dist < min_dist:
                        min_dist = dist
                        best_uav_id = uav.node_id

                place_matrix[k, l] = best_uav_id

        chosen_tasks_with_map = []
        picked_ue_ids = set()  # 用于去重，防止 Agent 重复选同一个候选人
        # 注意：这里依然保留了 floor，因为任务池 M 是离散的
        pick_indices = np.floor((raw_pick + 1) / 2 * (M + 1)).astype(np.int32)
        pick_indices = np.clip(pick_indices, 0, M)
        for k in range(K):
            cand_idx = pick_indices[k]
            # 如果索引在 [0, M-1] 范围内，说明 Pick 了一个真实候选人
            if 0 <= cand_idx < len(self.current_cand_tasks):
                ue_id, sfc = self.current_cand_tasks[cand_idx]

                if ue_id not in picked_ue_ids:
                    # 获取该任务对应的 UAV 映射方案 (取 place_matrix 的第 k 行)
                    vnf_uav_map = self._trim_mapping(sfc, place_matrix[k])
                    chosen_tasks_with_map.append((ue_id, sfc, vnf_uav_map))
                    picked_ue_ids.add(ue_id)

        num_picked = len(chosen_tasks_with_map)  # Agent 真正“领走”的任务

        # 5.进行充电桩判定 根据本time 1s后飞行到的位置和充电桩位置判定谁能充电
        charge_info = self._handle_charging()

        # 4. 执行部署 (Placement with Masking)
        sfc_results = self._process_sfc_tasks(chosen_tasks_with_map)

        # If a UAV died during billing, _process_sfc_tasks will short-circuit
        # and set 'uav_crashed'. Handle termination and crash penalty here
        # centrally in `step()` so episode control remains in one place.
        if sfc_results.get("uav_crashed", False):
            reward = self.config["RWD_CRASH"]
            terminated = True
            truncated = False
            info = {**sfc_results, **charge_info, "perf/crashed": 1.0}
            return self._get_obs(), reward, terminated, truncated, info
        # 清理未选中的候选任务 1/22 15：34 ：先不做惩罚了，只清理任务
        num_unpicked = 0
        unpicked_penalty = 0.0  # 💡3/4 23:04 新增：累计不接单的惩罚
        for ue in self.ues:
            if ue.task_buffer:
                # 无论这个 UE 是否被 Pick 了，剩下的所有任务在 Slot 结束时都会失效
                # (如果被 Pick 了，头名任务已经在 _process_sfc_tasks 里处理并 popleft 了)
                # (如果没被 Pick，那么整个队列都是要被丢弃的)
                tasks_ignored = len(ue.task_buffer)
                num_unpicked += tasks_ignored

                # 💡 新增：不接单等同于被丢弃，必须严惩！
                unpicked_penalty += tasks_ignored * self.config["RWD_DROP"]
                ue.task_buffer.clear()
        # --- 5. 环境状态更新 (为下一回合生成新的任务) ---
        self.current_time += self.time_slot - self.dt_fly
        self._update_state()

        # --- 6. 【数据对齐】：构建你需要的“当季报表” ---
        # 我们把所有数据统一进 sfc_results 返回给 info
        sfc_results["total_available"] = total_tasks_on_table  # 1. 桌面上一共有多少活
        sfc_results["actually_picked"] = num_picked  # 2. Agent 接了几个

        # 3. 总丢弃数 = 准入杀掉的 + 根本没 Pick 的
        # 这能让你看出是“能力问题”还是“名额不够”
        sfc_results["unpicked_count"] = num_unpicked
        sfc_results["dropped_count"] += num_unpicked
        # 修正总失败数：确保涵盖了所有没成功的任务 failed = 准入拒绝 + 坠毁失败 + 正常超时 + 没认领
        sfc_results["failed_count"] += num_unpicked

        reward, reward_info = self._calculate_reward(sfc_results, flight_info)
        # 💡 新增：将躺平惩罚算入本步总奖励
        reward += unpicked_penalty
        reward_info["r_unpicked_penalty"] = float(unpicked_penalty)  # 方便在TB里看
        # 3. 为下一回合准备候选人 (刷新 self.current_cand_tasks)
        all_active = [
            (i, ue.task_buffer[0]) for i, ue in enumerate(self.ues) if ue.task_buffer
        ]
        all_active.sort(key=lambda x: x[1].deadline)
        self.current_cand_tasks = all_active[: self.M]  # 更新

        # 1. 判断 Terminated (自然终止：坠毁)
        if any(u.is_crashed for u in self.uavs):
            terminated = True
            # 给一个大大的惩罚 (如果还没给的话)
            # reward -= 100.0

        # 2. 判断 Truncated (时间截断)
        truncated = False
        if self.current_step >= self.config["MAX_STEPS"]:
            truncated = True
        sfc_results["picked_ue_ids"] = list(picked_ue_ids)

        # 如果是评估模式 (RECORD_DEPLOYMENT)，记录更详细的映射表
        if self.config.get("RECORD_DEPLOYMENT", False):
            # 记录格式: {ue_id: [uav_id_vnf1, uav_id_vnf2, ...]}
            detailed_map = {
                int(ue_id): mapping.tolist()
                for ue_id, sfc, mapping in chosen_tasks_with_map
            }
            sfc_results["detailed_mappings"] = detailed_map

        info = {**sfc_results, **charge_info, **reward_info}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode="human"):
        """
        Render the environment to the screen.
        """
        pass  # Implement rendering logic if needed

    def _calculate_rate(self, loc1, loc2, p_tx, link_type="UAV_UE"):
        """
        Calculates the Shannon data rate based on probabilistic LoS/NLoS path loss models.

        Ref: Al-Hourani et al., "Optimal LAP Altitude for Maximum Coverage," IEEE WCL, 2014.

        Args:
            loc1 (np.array): Position of node 1 [x, y]
            loc2 (np.array): Position of node 2 [x, y]
            p_tx (float): Transmission power in Watts
            link_type (str): 'UAV_UE' for Air-to-Ground, 'UAV_UAV' for Air-to-Air

        Returns:
            float: Data rate in bits/second (bps)
        """
        # --- 1. Constants & Configuration (Align with IEEE standard values) ---
        FC = 2.4e9  # Carrier frequency: 2 GHz
        C = 3e8  # Speed of light: 3e8 m/s
        # Noise Power = -174 dBm/Hz + 10log(B) + NoiseFigure(10dB)
        # For B=2MHz, Noise ~ -100 dBm = 1e-13 Watts
        NOISE_FIGURE = 10  # 噪声系数(dB)，通信设备典型值
        NOISE_POWER_W = (
            10
            ** (
                (
                    -174
                    + 10 * np.log10(self.config.get("BANDWIDTH_HZ", 2e6))
                    + NOISE_FIGURE
                )
                / 10
            )
            / 1000
        )
        BANDWIDTH = self.config.get("BANDWIDTH_HZ", 2e6)  # 2 MHz

        # Urban Environment Parameters (Dense Urban / Suburban can vary these)
        # a, b are S-curve parameters
        # --- 2. 环境参数修正（对齐原论文+城市/郊区场景） ---
        # S曲线参数（原论文值，无需改）
        ENV_A = 9.61
        ENV_B = 0.16
        # 附加损耗修正（原1.0/20.0不合理，改为学术界通用值）
        ETA_LOS = 6.0  # 普通城区LoS附加损耗(dB)
        ETA_NLOS = 22.0  # 普通城区NLoS附加损耗(dB)
        # 路径损耗指数（补充通用模型的γ，适配非自由空间）
        PATH_LOSS_EXP = 2.8  # 普通城区γ值（自由空间=2）

        # --- 2. Geometry Calculation ---
        d_2d = np.linalg.norm(loc1 - loc2)
        d_2d = max(d_2d, 1.0)  # Avoid singularity at d=0

        # Height Setup
        # Assuming UAVs are at fixed height H, UEs are at ground (H=0)
        h_uav = self.config.get("UAV_HEIGHT", 100.0)  # Default 100m

        if link_type == "UAV_UE":
            delta_h = h_uav  # |H_uav - H_ue| = 100 - 0
        else:  # UAV_UAV
            delta_h = 0.0  # Assuming same altitude for all UAVs

        d_3d = np.sqrt(d_2d**2 + delta_h**2)
        d_3d = max(d_3d, 1e-6)

        # --- 3. Path Loss Calculation (in dB) ---
        # --- 4. 路径损耗计算（修正为原论文通用模型） ---
        # 替换原纯自由空间公式，改为原论文的10γlog10形式（更贴合原论文）
        # L_fspl = 10*γ*log10(4πfc d / c) （拆分计算更易读）
        fspl_linear = (4 * np.pi * FC * d_3d) / C
        L_fspl = 10 * PATH_LOSS_EXP * np.log10(fspl_linear)

        if link_type == "UAV_UE":
            # === Air-to-Ground (Probabilistic LoS/NLoS) ===

            # Calculate Elevation Angle (theta) in degrees
            theta_rad = np.arcsin(delta_h / d_3d)
            theta_deg = np.degrees(theta_rad)

            # Probability of Line-of-Sight (P_LoS)
            p_los = 1.0 / (1.0 + ENV_A * np.exp(-ENV_B * (theta_deg - ENV_A)))

            # Average Path Loss (Probabilistic weighted sum)
            # L_avg = L_fspl + P_LoS * eta_LoS + (1 - P_LoS) * eta_NLoS
            L_avg_db = L_fspl + p_los * ETA_LOS + (1.0 - p_los) * ETA_NLOS

        else:
            # === Air-to-Air (Mostly LoS) ===
            # Typically modeled as FSPL with minor fading margin
            L_fspl_aa = 10 * 2 * np.log10((4 * np.pi * FC * d_3d) / C)
            L_avg_db = L_fspl_aa + 1.0  # 小衰落余量

        # --- 4. Shannon Capacity ---

        if p_tx <= 0:
            return 0.0

        # 发射功率转换（W → dBm，原逻辑正确）,dBm 是 “以 1 毫瓦为基准的分贝值”
        # p_tx是以w为单位的发射功率，根据A2G和A2A链路的不同，p_tx一般在0.1~1W之间
        p_tx_dbm = 10 * np.log10(p_tx * 1000)
        # 接收功率（dBm）= 发射功率（dBm） - 路径损耗（dB）
        p_rx_dbm = p_tx_dbm - L_avg_db
        # 接收功率转换：dBm → W  转换回瓦（添加保护，避免对数负数）
        p_rx = max((10 ** (p_rx_dbm / 10.0)) / 1000.0, 1e-20)
        # SNR计算（用精准噪声功率）
        snr = p_rx / NOISE_POWER_W
        # 香农速率（添加SNR保护，避免log2(1+0)）
        rate = BANDWIDTH * np.log2(1.0 + max(snr, 1e-10))

        return rate

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 检查是否是过拟合测试模式
        is_overfit = self.config.get("IS_OVERFIT_TEST", False)
        # 1. 重置 UAV 物理状态
        for i, uav in enumerate(self.uavs):
            base_loc = np.array(self.config["SECTOR_CENTER"][i], dtype=np.float32)
            if not is_overfit:
                jitter = self.np_random.uniform(-20, 20, size=2)
            else:
                jitter = np.zeros(2)  # 过拟合测试取消抖动
            start_loc = np.clip(base_loc + jitter, 0, self.config["GROUND_WIDTH"])
            uav.reset(start_loc)
        # 2. 重置 UE (类型洗牌与随机位置)
        current_type_list = self.fixed_type_list.copy()
        self.np_random.shuffle(current_type_list)

        # --- 【创新点：生成任务热点中心】 ---
        num_clusters = 2  # 设置 2 个热点区域
        cluster_centers = self.np_random.uniform(100, 400, size=(num_clusters, 2))
        cluster_std = 60.0  # 簇的松散程度 (60米标准差)

        for i, ue in enumerate(self.ues):
            assigned_type = current_type_list[i]

            if is_overfit:
                # 过拟合测试：固定 UE 在地图中心偏右上一点
                loc = np.array([300.0, 300.0])
            else:
                # 原有的随机逻辑保持不变
                if self.np_random.random() < 0.8:
                    center = cluster_centers[self.np_random.integers(0, num_clusters)]
                    loc = center + self.np_random.normal(0, cluster_std, size=2)
                else:
                    loc = self.np_random.uniform(
                        20, self.config["GROUND_WIDTH"] - 20, size=2
                    )

            new_loc = np.clip(loc, 20, self.config["GROUND_WIDTH"] - 20).astype(
                np.float32
            )
            ue.reset(new_nodetype=assigned_type, new_loc=new_loc)
        # 3. 重置环境全局元数据
        self.current_time = 0.0
        self.current_step = 0
        # --- 【同步统计】：重置时各 UE 已经生成了第一波任务，需要计入总数 ---
        initial_gen_count = sum(1 for ue in self.ues if ue.task_buffer)
        self.total_gen_tasks = initial_gen_count
        self.current_step_gen = initial_gen_count

        # --- 创新特征初始化 ---
        # A. 初始化负载状态为全 0
        self.uav_load_status = np.zeros(self.N, dtype=np.float32)

        # C. 【关键创新点】：构建初始候选任务池 (M 选 K 逻辑的起点)
        # 扫描所有 UE，提取此时刚生成的初始任务
        all_active = []
        for idx, ue in enumerate(self.ues):
            if ue.task_buffer:
                all_active.append((idx, ue.task_buffer[0]))

        # 按照紧急程度(Deadline)排序，取前 M 个
        all_active.sort(key=lambda x: x[1].deadline)
        self.current_cand_tasks = all_active[: self.M]

        # 4. 返回初始观测
        return self._get_obs(), {}

    def get_action_mask_params(self):
        """
        高稳定性动作遮罩计算 (生产级)
        集成了：电量、边界、避障、非对称缩放及数值稳定性保护。
        """
        # 1. 初始化 4 方向限制：[左, 右, 下, 上] (范围 0.0~1.0)
        mobility_bounds = np.ones((self.N, 4), dtype=np.float32)

        width = self.config["GROUND_WIDTH"]
        height = self.config["GROUND_HEIGHT"]
        max_step_dist = self.config["UAV_MAX_SPEED"] * self.dt_fly
        safe_dist = self.config["UAV_MIN_SAFE_DISTANCE"]
        # 建议 5：警戒距离 = 安全距离 + 一个步长的提前量
        warning_dist = safe_dist + max_step_dist

        for i, uav_i in enumerate(self.uavs):
            # --- A. 电量约束 (全局缩放) ---
            battery_ratio = uav_i.e_battery / uav_i.battery_capacity
            # 线性减速，电量越低，允许的最大速度越慢。
            speed_limit = np.clip((battery_ratio - 0.05) / 0.15, 0.0, 1.0)
            mobility_bounds[i, :] *= speed_limit

            # --- B. 边界约束 (非对称，防止越界产生的负值) ---
            # 建议 2：使用 max(0.0, ...) 彻底杜绝负数 Bound 导致训练崩溃
            mobility_bounds[i, 0] = min(
                mobility_bounds[i, 0], max(0.0, uav_i.loc[0] / max_step_dist)
            )  # Left
            mobility_bounds[i, 1] = min(
                mobility_bounds[i, 1], max(0.0, (width - uav_i.loc[0]) / max_step_dist)
            )  # Right
            mobility_bounds[i, 2] = min(
                mobility_bounds[i, 2], max(0.0, uav_i.loc[1] / max_step_dist)
            )  # Bottom
            mobility_bounds[i, 3] = min(
                mobility_bounds[i, 3], max(0.0, (height - uav_i.loc[1]) / max_step_dist)
            )  # Top

            # --- C. 避障约束 (带优先级的非对称缩放) ---
            for j, uav_j in enumerate(self.uavs):
                if i == j or uav_j.is_crashed:
                    continue

                rel_pos = uav_j.loc - uav_i.loc
                # 建议 1：防止 dist=0 导致的数值异常
                dist = max(np.linalg.norm(rel_pos), 1e-6)

                if dist < warning_dist:
                    # 💡【Trick：优先级避障】如果 i > j，i 让路，j 照飞。防止两机对冲时全部锁死。
                    if i > j:
                        safe_move = max(0.0, (dist - safe_dist) / max_step_dist)

                        # 建议 3 & 4：根据相对位置，只限制“靠近”对方的方向
                        if rel_pos[0] > 0:  # j 在 i 右边
                            mobility_bounds[i, 1] = min(
                                mobility_bounds[i, 1], safe_move
                            )
                        else:  # j 在 i 左边
                            mobility_bounds[i, 0] = min(
                                mobility_bounds[i, 0], safe_move
                            )

                        if rel_pos[1] > 0:  # j 在 i 上方
                            mobility_bounds[i, 3] = min(
                                mobility_bounds[i, 3], safe_move
                            )
                        else:  # j 在 i 下方
                            mobility_bounds[i, 2] = min(
                                mobility_bounds[i, 2], safe_move
                            )

        # --- D. Pick 任务选择遮罩 (逻辑验证正确) ---
        num_tasks = len(self.current_cand_tasks)
        pick_limit = (2.0 * num_tasks / (self.M + 1)) - 1.0 - 1e-5

        return {
            "mobility_bounds": mobility_bounds,
            "pick_limit": np.array([np.clip(pick_limit, -1.0, 1.0)], dtype=np.float32),
        }

    def _get_obs(self):
        obs_list = []
        width = self.config["GROUND_WIDTH"]
        height = self.config["GROUND_HEIGHT"]
        charger_loc = self.chargers[0].loc

        # --- 1. UAV 物理流形 (N * 9 维) ---
        max_spd = self.config["UAV_MAX_SPEED"]
        # 获取上一步记录的 UAV 负载率 (0.0~1.0)
        uav_loads = getattr(self, "uav_load_status", np.zeros(self.N))

        for i, uav in enumerate(self.uavs):
            # 基础物理量 (6维)
            obs_list.extend(
                [
                    uav.loc[0] / width,
                    uav.loc[1] / height,
                    uav.velocity[0] / max_spd,
                    uav.velocity[1] / max_spd,
                    uav.e_battery / uav.battery_capacity,
                    1.0 if uav.is_crashed else 0.0,
                ]
            )
            # 几何增强：相对充电桩位移 (2维) - 引导“向心力”
            obs_list.append((uav.loc[0] - charger_loc[0]) / width)
            obs_list.append((uav.loc[1] - charger_loc[1]) / height)
            # 状态增强：实时负载 (1维) - 引导“负载均衡”
            obs_list.append(uav_loads[i])

        # --- 2. 宏观需求流形 (3x3 Grid = 27 维) ---
        # 对应 ICLR 2025 流形压缩：不关心具体哪个 UE，关心哪个区域“红”了
        demand_grid = np.zeros((self.grid_res, self.grid_res, 3), dtype=np.float32)
        MAX_DATA = self._max_data
        MAX_TIME = max(SFC_DEADLINES.values()) * self.config["DEADLINE_FACTOR"]

        for ue in self.ues:
            if ue.task_buffer:
                task = ue.task_buffer[0]
                gx = int(min(ue.loc[0] // (width / self.grid_res), self.grid_res - 1))
                gy = int(min(ue.loc[1] // (height / self.grid_res), self.grid_res - 1))

                # 特征 0: 区域任务数累加
                # --- 修改归一化方式 ---
                # 全部除以 ue_num，确保单格数值在物理上绝不会超过 1.0
                demand_grid[gx, gy, 0] += 1.0 / self.ue_num
                demand_grid[gx, gy, 1] += (task.total_data_in / MAX_DATA) / self.ue_num

                time_left = task.deadline - self.current_time
                urgency = 1.0 - np.clip(time_left / MAX_TIME, 0, 1)
                demand_grid[gx, gy, 2] += urgency / self.ue_num

        obs_list.extend(demand_grid.flatten())

        # --- 3. 微观候选任务包 (M * 5 维) ---
        # 专门辅助 V1 动作决策 (M选K)
        # 预先在 step 开始时选好候选人存入 self.current_cand_tasks
        cand_tasks = getattr(self, "current_cand_tasks", [])

        for i in range(self.M):
            if i < len(cand_tasks):
                ue_id, sfc = cand_tasks[i]
                ue = self.ues[ue_id]
                # 候选特征：位置(2), 数据量(1), 总周期(1), 剩余时间(1)
                obs_list.extend(
                    [
                        ue.loc[0] / width,
                        ue.loc[1] / height,
                        sfc.total_data_in / MAX_DATA,
                        sfc.total_cycles / self._max_total_cycles,
                        np.clip((sfc.deadline - self.current_time) / MAX_TIME, -1, 1),
                    ]
                )
            else:
                # 填充位 (PAD)
                obs_list.extend([0.0] * 5)

        # --- 4. 全局趋势 (3 维) ---
        # 剩余时间步, 充电桩坐标
        obs_list.append(
            (self.config["MAX_STEPS"] - self.current_step) / self.config["MAX_STEPS"]
        )
        obs_list.extend([charger_loc[0] / width, charger_loc[1] / height])
        # --- 返回字典格式 ---
        state_vector = np.array(obs_list, dtype=np.float32)

        # 获取实时动作遮罩参数
        mask_params = self.get_action_mask_params()

        return {
            "state": state_vector,
            "mobility_bounds": mask_params[
                "mobility_bounds"
            ].flatten(),  # 展平为 16 维 (4 UAVs * 4)
            "pick_limit": mask_params["pick_limit"],  # 已经是 1 维
        }
