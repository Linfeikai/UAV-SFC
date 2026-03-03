from .sfc_config import SFC_STRUCTURES
from .node_types import Nodetype

# ==========================================
# 1. 难度与动态参数 (Difficulty Tuning)
# ==========================================
# 修改此变量来一键调整训练难度
_DIFFICULTY_LEVEL = 2

# 基础概率分布 (通常不随难度改变)
_NODE_TYPE_DISTRIBUTION = {
    Nodetype.Video: 0.4,
    Nodetype.Security: 0.4,
    Nodetype.IoT: 0.2,
}

if _DIFFICULTY_LEVEL == 1:
    _ARRIVAL_PROB = 0.1
    _BATTERY_CAPACITY = 50000.0
    _DEADLINE_FACTOR = 3.0
elif _DIFFICULTY_LEVEL == 2:
    _ARRIVAL_PROB = 0.5
    _BATTERY_CAPACITY = 30000.0  # 提高到3万，保证能撑到充电逻辑被触发
    _DEADLINE_FACTOR = 1.2
else:
    _ARRIVAL_PROB = 0.6
    _BATTERY_CAPACITY = 7000.0
    _DEADLINE_FACTOR = 1.1

# ==========================================
# 2. 物理与环境常量 (Physical Constants)
# ==========================================
GROUND_WIDTH = 500.0
GROUND_HEIGHT = 500.0
TIME_SLOT_DURATION = 8.0
MAX_STEPS = 40

# 通信模型参数
BANDWIDTH_HZ = 2.0 * 10**6
P_UPLINK = 0.2
NOISE_POWER = 1e-13
ALPHA0 = 1e-5

# 初始扇区中心 (无人机的出生点)
# 【在这里定义变量，供后面 reset 和 config 引用】
SECTOR_CENTER = [
    [125, 125],  # 左下
    [375, 125],  # 右下
    [125, 375],  # 左上
    [375, 375],  # 右上
]

# ==========================================
# 3. 实体特征参数 (Entity Specs)
# ==========================================
NUM_UAVS = 4
NUM_UES = 20
NUM_CHARGERS = 1

# UAV 硬件参数 (已调低 CPU 频率以增加负载感)
UAV_CPU_FREQ = 2.0e8  # 200 MHz (从 2.4GHz 降下来，让负载率更真实)
# 让一台 UAV 同时处理两个任务时必然超载。这样“负载均衡”的策略好坏会产生巨大的分数差异。
UAV_MAX_SPEED = 20.0
UAV_HEIGHT = 100.0
UAV_MIN_SAFE_DISTANCE = 10.0
UAV_CHARGING_POWER = 1000.0
UAV_P_TX_CROSS = 0.5  # UAV 横向传输功率
UAV_FULL_LOAD_POWER = 20.0  # UAV 满载计算功耗

# 充电桩参数
CHG_RADIUS = 150.0
CHG_K_SCALE = 0.5

# ==========================================
# 4. 奖励函数权重 (Reward Weights)
# ==========================================
RWD_TIMEOUT = -10.0
RWD_DROP = -8.0
RWD_COLLISION = -5.0
RWD_SUCCESS = 20.0  # 提高成功奖励，激励 Agent 克服能耗去干活
RWD_LINK_BROKEN = -10.0
RWD_LATENCY_BONUS = 5.0
RWD_CRASH = -1000.0  # 巨额惩罚，强制学习生存
RWD_W_ENERGY = 10.0  # 能耗惩罚权重
RWD_W_CHARGE = 2.0  # 充电激励权重

# ==========================================
# 5. 决策模型参数 (Decision Specs)
# ==========================================
GRID_RES = 3  # 3x3 观测网格
CANDIDATE_POOL_SIZE = 12  # M
DECISION_VARIABLE_TASKS = 6  # K (适当调高，增加并发)

# 自动计算
MAX_VNF_LEN = max(len(chain) for chain in SFC_STRUCTURES.values())

# 数据范围 (保持字典，因为它是一个查找表)
DATA_SIZE_RANGES = {
    Nodetype.Video: (0.5 * 10**6, 4 * 10**6),  # 0.5MB - 4MB
    Nodetype.Security: (0.2 * 10**6, 1 * 10**6),  # 0.2MB - 1MB
    Nodetype.IoT: (40 * 10**3, 400 * 10**3),  # 40KB - 400KB
}

# ==========================================
# 6. 默认扁平配置字典 (Flattened Configuration)
# ==========================================
DEFAULT_CONFIG = {
    # 实验元数据
    "NUM_UAVS": NUM_UAVS,
    "NUM_UES": NUM_UES,
    "MAX_STEPS": MAX_STEPS,
    "GROUND_WIDTH": GROUND_WIDTH,
    "GROUND_HEIGHT": GROUND_HEIGHT,
    "TIME_SLOT_DURATION": TIME_SLOT_DURATION,
    "SECTOR_CENTER": SECTOR_CENTER,  # 成功引用上方的变量
    # 动态难度
    "ARRIVAL_PROB": _ARRIVAL_PROB,
    "BATTERY_CAPACITY": _BATTERY_CAPACITY,
    "DEADLINE_FACTOR": _DEADLINE_FACTOR,
    "NODE_TYPE_DISTRIBUTION": _NODE_TYPE_DISTRIBUTION,
    "DATA_SIZE_RANGES": DATA_SIZE_RANGES,
    # 决策模型
    "M": CANDIDATE_POOL_SIZE,
    "K": DECISION_VARIABLE_TASKS,
    "GRID_RES": GRID_RES,
    "MAX_VNF_LEN": MAX_VNF_LEN,
    "RECORD_DEPLOYMENT": False,
    # 通信
    "BANDWIDTH_HZ": BANDWIDTH_HZ,
    "P_UPLINK": P_UPLINK,
    "NOISE_POWER": NOISE_POWER,
    "ALPHA0": ALPHA0,
    # UAV 物理 (拍平)
    "UAV_CPU_FREQ": UAV_CPU_FREQ,
    "UAV_MAX_SPEED": UAV_MAX_SPEED,
    "UAV_HEIGHT": UAV_HEIGHT,
    "UAV_MIN_SAFE_DISTANCE": UAV_MIN_SAFE_DISTANCE,
    "UAV_CHARGING_POWER": UAV_CHARGING_POWER,
    "UAV_P_TX_CROSS": UAV_P_TX_CROSS,
    "UAV_FULL_LOAD_POWER": UAV_FULL_LOAD_POWER,
    # 充电桩 (拍平)
    "CHG_RADIUS": CHG_RADIUS,
    "CHG_K_SCALE": CHG_K_SCALE,
    # 奖励 (拍平)
    "RWD_TIMEOUT": RWD_TIMEOUT,
    "RWD_DROP": RWD_DROP,
    "RWD_COLLISION": RWD_COLLISION,
    "RWD_SUCCESS": RWD_SUCCESS,
    "RWD_LINK_BROKEN": RWD_LINK_BROKEN,
    "RWD_LATENCY_BONUS": RWD_LATENCY_BONUS,
    "RWD_CRASH": RWD_CRASH,
    "W_ENERGY": RWD_W_ENERGY,
    "W_CHARGE": RWD_W_CHARGE,
}
