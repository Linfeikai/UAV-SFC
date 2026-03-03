from dataclasses import dataclass, field
from collections import deque
import numpy as np
from numpy.random import Generator

from .sfc import SFC, VNF
from .sfc_config import VNF_TEMPLATES, SFC_STRUCTURES, SFC_DEADLINES
from .env_config import GROUND_WIDTH, DATA_SIZE_RANGES, Nodetype


@dataclass
class UENode:
    node_id: int
    nodetype: Nodetype
    # 只需要位置，不需要 cache_capacity 了
    loc: np.ndarray = field(default_factory=lambda: np.array([0, 0], dtype=np.float32))

    # 这里的 queue 只是为了存一下当前 slot 没处理完的任务（如果有排队机制）
    # 如果你的设定是“即时服务”，这个 queue甚至可以是临时的 list
    task_buffer: deque = field(default_factory=deque)
    arrival_prob: float = 0.2  # 默认任务到达概率，可以在外部配置覆盖
    deadline_factor: float = 1.0  # 默认截止时间因子，可以在外部配置覆盖

    rng: Generator = None

    def __post_init__(self):
        # 初始化随机生成器
        if self.rng is None:
            self.rng = np.random.default_rng()

    def generate_task(self, current_time):
        """
        生成逻辑：每个 Slot 最多生成 1 个 SFC
        current_time: 当前环境时间 (float) 应该是绝对时间
        """
        # 1. 定义到达概率 p (这就对应了之前的 lambda 概念)
        # 这个 p_arrival 应该从外部配置读入，用于调节难度
        # 这里的 self.arrival_prob 现在代表泊松分布的均值 λ (lambda)
        # 即：预期平均每个 Slot 生成多少个任务
        lam = self.arrival_prob  # 比如 0.3

        # 从泊松分布中采样本次产生的任务数量
        num_new_tasks = self.rng.poisson(lam)

        # 如果采样结果是 0，直接结束
        if num_new_tasks == 0:
            return 0
        for _ in range(num_new_tasks):
            chain_structure = SFC_STRUCTURES[self.nodetype]
            deadline_duration = SFC_DEADLINES[self.nodetype] * self.deadline_factor

            initial_data_size = self.rng.integers(
                int(DATA_SIZE_RANGES[self.nodetype][0]),
                int(DATA_SIZE_RANGES[self.nodetype][1]),
            )

            # 构建 VNF 链
            vnf_list = []
            current_data_in = initial_data_size

            for idx, vnf_name in enumerate(chain_structure):
                template = VNF_TEMPLATES[vnf_name]

                scaling = template["scaling_factor"]
                complexity = template["complexity"]

                # 修改后 (强制转为原生 int)
                data_out = int(
                    current_data_in * scaling
                )  # scaling是这个data经过该VNF后的缩放比例
                cpu_cycles = float(
                    current_data_in
                    * complexity  # complexity是一个bit要几个cpu cycle来计算
                )  # cycles 通常用 float 比较好

                vnf = VNF(
                    vnf_id=idx,
                    vnf_type=vnf_name,
                    data_in=int(
                        current_data_in
                    ),  # 注意：对于第一个 VNF，data_in 就是需要从 UE 传上来的数据量
                    data_out=data_out,
                    required_cycles=cpu_cycles,
                )
                vnf_list.append(vnf)
                current_data_in = data_out

            new_sfc = SFC(
                sfc_id=f"UE{self.node_id}_T{current_time:.1f}_{idx}",
                ue_id=self.node_id,
                arrival_time=current_time,
                deadline=current_time + deadline_duration,
                vnf_chain=vnf_list,
                status=0,  # 0: Pending
            )

            # 直接放入缓冲区，不做溢出检查（除非你想模拟 buffer overflow）
            self.task_buffer.append(new_sfc)
        return num_new_tasks

    def reset(self, new_nodetype=None, new_loc=None):
        """
        重置 UE 节点状态。
        """
        self.task_buffer.clear()
        if new_nodetype is not None:
            self.nodetype = new_nodetype

        if new_loc is not None:
            self.loc = new_loc.astype(np.float32)
        else:
            self.loc = self.rng.uniform(20, GROUND_WIDTH - 20, 2).astype(np.float32)
        self.generate_task(current_time=0.0)  # 初始化时生成任务

    def move_to(self, new_loc: np.ndarray):
        """
        将 UE 移动到新的位置。
        """
        self.loc = new_loc.astype(np.float32)

    # 删除了 local_offloading
    # 删除了 partial_offloading
    # UE 现在非常干净，就是个发任务的。
