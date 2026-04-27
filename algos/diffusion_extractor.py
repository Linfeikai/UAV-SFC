import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SFCFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,  # 修改为 Dict 空间
        features_dim: int = 256,
        n_uavs: int = 4,
        m_candidates: int = 6,
        grid_res: int = 3,
    ):
        self.mask_dim = n_uavs * 4
        total_output_dim = features_dim + self.mask_dim
        assert isinstance(observation_space, gym.spaces.Dict), "必须使用 Dict 观察空间"
        assert "state" in observation_space.spaces
        assert "mobility_bounds" in observation_space.spaces

        super().__init__(observation_space, total_output_dim)
        self._features_dim = total_output_dim

        self.n_uavs = n_uavs
        self.m_candidates = m_candidates
        self.grid_res = grid_res

        self.uav_dim = self.n_uavs * 9
        self.grid_dim = (self.grid_res**2) * 3
        self.cand_dim = self.m_candidates * 5
        # self.global_dim = 3

        # 1. 针对 3x3 需求网格的 CNN
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=2, stride=1),
            nn.Mish(),
            nn.Flatten(),
        )
        cnn_output_dim = 16 * 2 * 2  # 64

        # 2. 针对候选任务的 MLP
        self.cand_mlp = nn.Sequential(nn.Linear(self.cand_dim, 64), nn.Mish())

        # 3. 特征融合主干 (仅处理 96 维物理状态)
        total_concat_dim = self.uav_dim + cnn_output_dim + 64 + 3  # 3是global_dim
        self.state_fusion_net = nn.Sequential(
            nn.Linear(total_concat_dim, 256),
            nn.LayerNorm(256),
            nn.Mish(),
            nn.Linear(256, features_dim),
            nn.Mish(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        """
        observations 现在是一个字典，包含 'state', 'mobility_bounds', 'pick_limit'
        """
        # --- 1. 提取物理状态向量 (96维) ---
        state = observations["state"]

        # 像以前一样拆解状态
        uav_obs = state[:, : self.uav_dim]
        grid_obs = (
            state[:, self.uav_dim : self.uav_dim + self.grid_dim]
            .reshape(-1, self.grid_res, self.grid_res, 3)
            .permute(0, 3, 1, 2)
        )
        cand_obs = state[
            :,
            self.uav_dim + self.grid_dim : self.uav_dim + self.grid_dim + self.cand_dim,
        ]
        global_obs = state[:, self.uav_dim + self.grid_dim + self.cand_dim :]

        # --- 2. 局部特征提取 ---
        grid_features = self.grid_cnn(grid_obs)
        cand_features = self.cand_mlp(cand_obs)

        # --- 3. 融合核心物理特征 (得到 256 维) ---
        combined_phys_features = torch.cat(
            [uav_obs, grid_features, cand_features, global_obs], dim=1
        )
        core_features = self.state_fusion_net(combined_phys_features)

        # --- 4. 【关键】拼接原始掩码参数 ---
        # 展平 mobility_bounds [Batch, N, 4] -> [Batch, N*4]
        mob_masks = observations["mobility_bounds"].reshape(core_features.size(0), -1)

        # 最终输出 = [256 维核心特征 | 16 维移动边界]
        return torch.cat([core_features, mob_masks], dim=1)
