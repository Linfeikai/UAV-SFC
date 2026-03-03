# sac_tanh_logprob_check.py
import torch, torch.nn.functional as F, math

torch.manual_seed(0)


def stable_tanh_logdet_jacobian(u):
    # log(1 - tanh(u)^2) 的稳定写法：2*(log(2) - u - softplus(-2u))
    return (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(dim=-1)


def sample_squashed_gauss(mu, log_std):
    std = log_std.exp().clamp_min(1e-6)
    base = torch.distributions.Normal(mu, std)
    u = base.rsample()
    a = torch.tanh(u)
    # 正确：base.log_prob(u) - log|det d tanh(u)/du|
    logp = base.log_prob(u).sum(dim=-1) - stable_tanh_logdet_jacobian(u)
    return a, logp


def sample_wrong(mu, log_std):
    std = log_std.exp()
    base = torch.distributions.Normal(mu, std)
    u = base.rsample()
    a = torch.tanh(u)
    # 错误：漏掉雅可比修正
    logp = base.log_prob(u).sum(dim=-1)
    return a, logp


D, N = 3, 4096
mu = torch.zeros(N, D)
log_std = torch.zeros(N, D)

a1, lp1 = sample_squashed_gauss(mu, log_std)
a2, lp2 = sample_wrong(mu, log_std)

print("mean|a| correct/ wrong:", a1.abs().mean().item(), a2.abs().mean().item())
print("mean logp correct/ wrong:", lp1.mean().item(), lp2.mean().item())
print("diff logp (should not be ~0):", (lp1 - lp2).abs().mean().item())
