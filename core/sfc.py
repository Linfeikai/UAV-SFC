from dataclasses import dataclass, field
from typing import List


@dataclass
class VNF:
    vnf_id: int  # VNF 在链中的序号 (0, 1, 2...)
    vnf_type: str  # 类型名称，如 "Firewall", "Transcoder"
    data_in: int  # 输入数据大小 (bit)
    data_out: int  # 输出数据大小 (bit) -> 决定了传给下一个 VNF 的数据量
    required_cycles: float  # 计算所需 CPU 周期 (cycles)

    # 状态追踪
    processed: bool = False  # 是否已完成


@dataclass
class SFC:
    sfc_id: str  # 唯一标识符 (e.g., "UE1_Slot5_0")
    ue_id: int  # 来源 UE
    arrival_time: float  # 到达时间
    deadline: float  # 截止时间

    # 核心结构：有序的 VNF 列表
    vnf_chain: List[VNF] = field(default_factory=list)

    # 状态追踪
    current_vnf_index: int = 0  # 当前处理到第几个 VNF 了
    status: int = 0  # 0:未完成, 1:成功, 2:失败/丢弃
    finished_time: float = 0.0

    @property
    def total_data_in(self):
        """整个链的初始输入数据量"""
        return self.vnf_chain[0].data_in if self.vnf_chain else 0

    @property
    def total_cycles(self):
        """整个链的总计算负载"""
        return sum(v.required_cycles for v in self.vnf_chain)

    @property
    def is_completed(self):
        return self.current_vnf_index >= len(self.vnf_chain)

    def get_current_vnf(self):
        if self.current_vnf_index < len(self.vnf_chain):
            return self.vnf_chain[self.current_vnf_index]
        return None
