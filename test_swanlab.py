import wandb
import numpy as np
import torch
from torch import Tensor
from typing import Dict

# wandb.init(project="test-dist")


# for step in range(50):
#     data = np.random.normal(step / 10, 2, 1000)
#     wandb.log({"dist": wandb.Histogram(data)})  # 🔥 直接用
def _precompute_diffusion_schedule(
    T_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02
) -> Dict[str, Tensor]:
    """预计算扩散过程的所有beta/alpha相关参数"""
    betas = torch.linspace(beta_start, beta_end, T_steps, dtype=torch.float32)
    alphas = 1.0 - betas  # 保留原始 alpha_t
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # \bar{alpha}_t
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8)
    )  # 防止除0

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "alphas_cumprod_prev": alphas_cumprod_prev,
    }


results = _precompute_diffusion_schedule(T_steps=10)
for key, value in results.items():
    print(f"{key}: {value}")
