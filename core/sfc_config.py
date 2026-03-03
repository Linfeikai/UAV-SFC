from .node_types import Nodetype


# ==========================================
# 1. 辅助工具
# ==========================================
def avg_cycles_per_bit(min_c, max_c):
    """
    将 cycles/byte 范围取平均值，并转换为 cycles/bit
    1 Byte = 8 bits
    """
    avg_byte = (min_c + max_c) / 2.0
    return avg_byte / 8.0 * 3.0


# ==========================================
# 2. VNF 模板库 (VNF Templates)
# ==========================================
# 依据你的笔记表格进行定义
# complexity 单位: cycles/bit
VNF_TEMPLATES = {
    # --- 1️⃣ Security SFC Components ---
    "Firewall": {
        "scaling_factor": 0.7,  # 过滤掉 30% 数据 (Input↓)
        "complexity": avg_cycles_per_bit(50, 100),  # 低计算
    },
    "Encryption": {
        "scaling_factor": 1.1,  # 加密头部开销 (Input↑)
        "complexity": avg_cycles_per_bit(200, 400),  # 高计算
    },
    "IntrusionDetection": {
        "scaling_factor": 0.6,  # 提取特征后变小 (Input↓)
        "complexity": avg_cycles_per_bit(300, 500),  # 中高计算
    },
    "LogAnalyzer": {
        "scaling_factor": 0.2,  # 聚合统计，数据大幅减小
        "complexity": avg_cycles_per_bit(50, 100),
    },
    # --- 2️⃣ Video SFC Components ---
    "Decoder": {
        "scaling_factor": 1.5,  # 解码后变成原始帧，大幅膨胀 (Input↑)
        "complexity": avg_cycles_per_bit(100, 200),  # 中计算
    },
    "ObjectDetection": {
        "scaling_factor": 0.1,  # 只输出检测框结果 (Input↓↓)
        "complexity": avg_cycles_per_bit(400, 800),  # 极高计算 (瓶颈)
    },
    "Encoder": {
        "scaling_factor": 0.2,  # 重新压缩传输
        "complexity": avg_cycles_per_bit(300, 600),  # 高计算
    },
    "Streaming": {
        "scaling_factor": 1.0,  # 转发
        "complexity": avg_cycles_per_bit(20, 50),  # 极低
    },
    # --- 3️⃣ IoT SFC Components ---
    "DataFilter": {
        "scaling_factor": 0.5,  # 去除无效样本
        "complexity": avg_cycles_per_bit(80, 150),
    },
    "FeatureExtraction": {
        "scaling_factor": 0.8,
        "complexity": avg_cycles_per_bit(200, 400),
    },
    "AnomalyDetection": {
        "scaling_factor": 0.1,  # 仅输出异常报告
        "complexity": avg_cycles_per_bit(300, 500),
    },
}

# ==========================================
# 3. SFC 结构定义 (Chains)
# ==========================================
SFC_STRUCTURES = {
    # 1️⃣ 安全与入侵检测链
    Nodetype.Security: ["Firewall", "Encryption", "IntrusionDetection", "LogAnalyzer"],
    # 2️⃣ 视频内容处理链 (高难度 Benchmark)
    Nodetype.Video: ["Decoder", "ObjectDetection", "Encoder", "Streaming"],
    # 3️⃣ IoT 数据分析链
    Nodetype.IoT: ["DataFilter", "FeatureExtraction", "AnomalyDetection"],
}

# ==========================================
# 4. 截止时间 (Deadlines)
# ==========================================
# 相对时间 (seconds)
SFC_DEADLINES = {
    Nodetype.Video: 5.0,  # 极低延迟要求
    Nodetype.Security: 5.0,
    Nodetype.IoT: 5.0,  # 宽松
}
