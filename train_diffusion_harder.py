import os

import hydra
import numpy as np
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from algos.diffusion_extractor import SFCFeaturesExtractor
from algos.diffusion_sac_agent import DiffusionSACAgent
from algos.diffusion_sac_policy import DiffusionSACPolicy
from core_harder.sfc_env import SFCEnv
from test.evalu import HeuristicEvaluator, smart_heuristic_policy

os.environ["OMP_NUM_THREADS"] = "1"


def get_flat_config(cfg: dict | DictConfig) -> dict:
    flat_config = {}

    if "env" in cfg:
        for k, v in cfg["env"].items():
            flat_config[k] = v

    if "uav" in cfg:
        for k, v in cfg["uav"].items():
            flat_config[f"UAV_{k}"] = v

    if "reward" in cfg:
        for k, v in cfg["reward"].items():
            if k in ["W_ENERGY", "W_CHARGE"]:
                flat_config[k] = v
            else:
                flat_config[f"RWD_{k}"] = v

    return flat_config


def make_env(rank: int, seed: int, config: dict):
    def _init():
        env = SFCEnv(config=config)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


class SFCStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.n_envs = 0
        self.per_env_stats = []

    def _on_training_start(self) -> None:
        self.n_envs = self.training_env.num_envs
        self.per_env_stats = [
            {
                "completed": 0,
                "dropped": 0,
                "timeout": 0,
                "unpicked": 0,
                "available": 0,
                "picked": 0,
                "crash": 0,
            }
            for _ in range(self.n_envs)
        ]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i in range(self.n_envs):
            info = infos[i]
            env_stats = self.per_env_stats[i]

            env_stats["completed"] += info.get("completed_count", 0)
            env_stats["dropped"] += info.get("dropped_count", 0)
            env_stats["timeout"] += info.get("timeout_count", 0)
            env_stats["unpicked"] += info.get("unpicked_count", 0)
            env_stats["available"] += info.get("total_available", 0)
            env_stats["picked"] += info.get("actually_picked", 0)
            env_stats["crash"] += info.get("perf/crashed", 0)

            if dones[i]:
                total_gen = max(1, env_stats["available"])
                total_pick = max(1, env_stats["picked"])

                success_rate = (env_stats["completed"] / total_gen) * 100
                admission_eff = (env_stats["completed"] / total_pick) * 100
                pick_rate = (env_stats["picked"] / total_gen) * 100

                self.logger.record("sfc_harder/success_rate_pct", success_rate)
                self.logger.record("sfc_harder/admission_efficiency_pct", admission_eff)
                self.logger.record("sfc_harder/pick_rate_pct", pick_rate)
                self.logger.record("sfc_harder/completed_total", env_stats["completed"])
                self.logger.record("sfc_harder/dropped_admission", env_stats["dropped"])
                self.logger.record("sfc_harder/timeout_total", env_stats["timeout"])
                self.logger.record("sfc_harder/crash_total", env_stats["crash"])

                for k in env_stats:
                    env_stats[k] = 0

        return True


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    hydra_exp_dir = HydraConfig.get().run.dir
    os.makedirs(hydra_exp_dir, exist_ok=True)

    flat_config = get_flat_config(cfg)
    probe_env = SFCEnv(config=flat_config)
    flat_config = dict(probe_env.config)
    probe_env.close()

    n_envs = cfg.get("n_envs", 4)
    use_wandb = cfg.get("use_wandb", False)

    if cfg.get("run_heuristic_baseline", True):
        print("\n🚀 [core_harder 前置测试] 先跑 observation-only heuristic 作为对比...")
        env_heuristic = SFCEnv(config=flat_config)
        heu_log_dir = os.path.join(hydra_exp_dir, "tb_logs", "HEURISTIC_BASELINE_HARDER")
        os.makedirs(heu_log_dir, exist_ok=True)
        evaluator = HeuristicEvaluator(env_heuristic, heu_log_dir)
        evaluator.evaluate(total_steps=cfg.get("heuristic_timesteps", 5000))
        env_heuristic.close()

    tensorboard_log_dir = os.path.join(hydra_exp_dir, "tb_logs")
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    env = SubprocVecEnv([make_env(i, cfg.seed, flat_config) for i in range(n_envs)])
    env = VecMonitor(env)

    if use_wandb:
        wandb.init(
            project="diffusion_rl_harder",
            name=f"{cfg.exp_name}_DIFFUSION_HARDER",
            dir=hydra_exp_dir,
            sync_tensorboard=True,
            config=OmegaConf.to_container(cfg, resolve=True),
            save_code=True,
        )

    algo_params = dict(cfg.get("diffusion", {}))
    policy_kwargs = dict(
        features_extractor_class=SFCFeaturesExtractor,
        features_extractor_kwargs=dict(
            n_uavs=flat_config["NUM_UAVS"],
            m_candidates=flat_config["M"],
            grid_res=flat_config["GRID_RES"],
        ),
        n_uavs=flat_config["NUM_UAVS"],
        m_candidates=flat_config["M"],
        decision_tasks=flat_config["K"],
        core_features_dim=256,
        share_features_extractor=True,
        T=20,
        net_arch=[256, 256],
    )

    model = DiffusionSACAgent(
        policy=DiffusionSACPolicy,
        env=env,
        verbose=1,
        seed=cfg.seed,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=policy_kwargs,
        **algo_params,
    )

    warmup_steps = cfg.get("warmup_steps", 0)
    if warmup_steps > 0:
        print(f"\n🔥 [core_harder 预热] 填充 {warmup_steps} 步 replay buffer...")
        temp_env = SFCEnv(config=flat_config)
        obs, _ = temp_env.reset(seed=cfg.seed + 999)

        for i in range(warmup_steps // n_envs):
            if np.random.random() > 0.2:
                action = smart_heuristic_policy(temp_env)
            else:
                action = temp_env.action_space.sample()

            next_obs, reward, terminated, truncated, info = temp_env.step(action)
            done = terminated or truncated

            obs_vec = {k: np.tile(v, (n_envs, 1)) for k, v in obs.items()}
            next_obs_vec = {k: np.tile(v, (n_envs, 1)) for k, v in next_obs.items()}
            action_vec = np.tile(action, (n_envs, 1))
            reward_vec = np.tile(reward, (n_envs,))
            done_vec = np.tile(done, (n_envs,))
            info_vec = [info] * n_envs

            model.replay_buffer.add(
                obs_vec, next_obs_vec, action_vec, reward_vec, done_vec, info_vec
            )

            obs = next_obs
            if done:
                obs, _ = temp_env.reset()

            report_every = max(1, (warmup_steps // n_envs) // 5)
            if (i + 1) % report_every == 0:
                print(f"   已填入约 {(i + 1) * n_envs}/{warmup_steps} 步")

        temp_env.close()
        print(f"✅ 预热完成，buffer size: {model.replay_buffer.size()}")

    print(f"\n当前模式: DIFFUSION_HARDER | n_envs={n_envs} | 总步数: {cfg.total_timesteps}")
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=SFCStatsCallback(),
        tb_log_name=f"{cfg.exp_name}_harder",
    )

    model_save_path = os.path.join(hydra_exp_dir, "DIFFUSION_HARDER_final_model")
    model.save(model_save_path)
    print(f"🎉 core_harder diffusion 训练完成，模型已保存至：{model_save_path}")

    env.close()
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
