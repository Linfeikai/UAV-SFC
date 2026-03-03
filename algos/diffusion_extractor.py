import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class SFCFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        n_uavs: int = 4,
        m_candidates: int = 6,
        grid_res: int = 3,
    ):
        # 初始化基类
        super().__init__(observation_space, features_dim)

        self.n_uavs = n_uavs
        self.m_candidates = m_candidates
        self.grid_res = grid_res

        # 计算各个部分的切片节点
        self.uav_dim = self.n_uavs * 9
        self.grid_dim = (self.grid_res**2) * 3
        self.cand_dim = self.m_candidates * 5
        self.global_dim = 3

        # 1. 针对 3x3 需求网格的 CNN (提取空间特征)
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 3x3 经过 kernel=2, stride=1 的卷积后变成 2x2，Flatten 后维度为 16 * 2 * 2 = 64
        cnn_output_dim = 16 * 2 * 2

        # 2. 针对候选任务的自注意力/全连接 (处理无序集合)
        self.cand_mlp = nn.Sequential(nn.Linear(self.cand_dim, 64), nn.ReLU())

        # 3. 特征融合主干网络
        total_concat_dim = self.uav_dim + cnn_output_dim + 64 + self.global_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(total_concat_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, features_dim),
            nn.Mish(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # --- 1. 拆解扁平化的观测向量 ---
        uav_obs = observations[:, : self.uav_dim]

        grid_start = self.uav_dim
        grid_end = grid_start + self.grid_dim
        # 还原成 [Batch, Channel, Height, Width] 给 CNN
        grid_obs = observations[:, grid_start:grid_end].view(
            -1, 3, self.grid_res, self.grid_res
        )

        cand_start = grid_end
        cand_end = cand_start + self.cand_dim
        cand_obs = observations[:, cand_start:cand_end]

        global_obs = observations[:, cand_end:]

        # --- 2. 局部特征提取 ---
        grid_features = self.grid_cnn(grid_obs)
        cand_features = self.cand_mlp(cand_obs)

        # --- 3. 拼接与融合 ---
        combined_features = torch.cat(
            [uav_obs, grid_features, cand_features, global_obs], dim=1
        )

        return self.fusion_net(combined_features)
