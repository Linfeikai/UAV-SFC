"""Regression tests for the primary training environment (stdlib unittest).

Run from the repository root: python -m unittest test.test_regressions -v
"""
from collections import deque
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

from core.execution import simulate_execution
from core.sfc import SFC, VNF
from core.sfc_env import SFCEnv
from algos.diffusion_extractor import SFCFeaturesExtractor
from algos.diffusion_policy_actor import DiffusionPolicyActor


def task(ue_id=0, work=(1.0,), deadline=10.0, data_in=0.0, data_out=0.0):
    return SFC(str(ue_id), ue_id, 0.0, deadline, [
        VNF(i, "Firewall", data_in if i == 0 else data_out, data_out, cycles)
        for i, cycles in enumerate(work)
    ])


def scheduler_env(window=7.0):
    return SimpleNamespace(
        N=2, time_slot=window + 1, dt_fly=1.0, current_time=0.0,
        uavs=[SimpleNamespace(loc=np.zeros(2), cpu_freq=1.0) for _ in range(2)],
        ues=[SimpleNamespace(loc=np.zeros(2)) for _ in range(3)],
        config={"P_UPLINK": 1.0, "UAV_P_TX_CROSS": 2.0},
        _calculate_rate=lambda *args, **kwargs: 1.0,
        _vnf_affinity=lambda *args: 1.0,
    )


class ExecutionTests(unittest.TestCase):
    def test_contention_cannot_invent_cpu_capacity(self):
        env = scheduler_env()
        tasks = [(i, task(i, (2.5,), deadline=3), [0]) for i in range(2)]
        result = simulate_execution(env, tasks)
        self.assertTrue(np.isinf(result["finish_times"]).all())
        np.testing.assert_allclose(result["uav_busy_time"], [3, 0])

    def test_separate_cpus_can_finish(self):
        tasks = [(i, task(i, (2.5,), deadline=3), [i]) for i in range(2)]
        result = simulate_execution(scheduler_env(), tasks)
        np.testing.assert_allclose(result["finish_times"], [2.5, 2.5])
        np.testing.assert_allclose(result["uav_busy_time"], [2.5, 2.5])

    def test_exact_deadline_completion(self):
        sfc = task(work=(3.0,), deadline=3)
        result = simulate_execution(scheduler_env(), [(0, sfc, [0])])
        self.assertEqual(result["finish_times"][0], 3)
        self.assertTrue(sfc.is_completed)
        self.assertTrue(sfc.vnf_chain[0].processed)

    def test_uplink_and_backhaul_obey_chain_precedence(self):
        sfc = task(work=(1, 1), data_in=1, data_out=2)
        result = simulate_execution(scheduler_env(), [(0, sfc, [0, 1])])
        np.testing.assert_allclose(result["finish_times"], [5])
        np.testing.assert_allclose(result["uav_busy_time"], [1, 1])
        np.testing.assert_allclose(result["uav_tx_energy_bill"], [4, 0])

    def test_expiry_releases_shared_cpu(self):
        tasks = [(0, task(0, (9,), deadline=1), [0]),
                 (1, task(1, (2,), deadline=3), [0])]
        result = simulate_execution(scheduler_env(), tasks)
        self.assertTrue(np.isinf(result["finish_times"][0]))
        self.assertEqual(result["finish_times"][1], 2.5)
        self.assertEqual(result["uav_busy_time"][0], 2.5)

    def test_slot_cutoff_cancels_work(self):
        result = simulate_execution(scheduler_env(window=2), [(0, task(work=(3,)), [0])])
        self.assertTrue(np.isinf(result["finish_times"][0]))
        self.assertEqual(result["uav_busy_time"][0], 2)

    def test_transfer_energy_clipped_at_deadline(self):
        sfc = task(work=(1, 1), deadline=2, data_out=5)
        result = simulate_execution(scheduler_env(), [(0, sfc, [0, 1])])
        np.testing.assert_allclose(result["uav_tx_energy_bill"], [2, 0])
        np.testing.assert_allclose(result["uav_busy_time"], [1, 0])
        self.assertEqual(sfc.current_vnf_index, 1)
        self.assertFalse(sfc.vnf_chain[1].processed)

    def test_unreached_vnf_does_not_consume_energy(self):
        sfc = task(work=(9, 1), deadline=2, data_out=5)
        result = simulate_execution(scheduler_env(), [(0, sfc, [0, 1])])
        np.testing.assert_allclose(result["uav_tx_energy_bill"], [0, 0])
        np.testing.assert_allclose(result["uav_busy_time"], [2, 0])

    def test_affinity_affects_effective_cpu_work(self):
        env = scheduler_env()
        env._vnf_affinity = lambda *args: 0.25
        result = simulate_execution(env, [(0, task(work=(1,)), [0])])
        self.assertEqual(result["finish_times"][0], 4)

    def test_zero_work_and_empty_slot(self):
        result = simulate_execution(scheduler_env(), [(0, task(work=(0, 0)), [0, 1])])
        self.assertEqual(result["finish_times"][0], 0)
        self.assertEqual(simulate_execution(scheduler_env(), [])["finish_times"].size, 0)

    def test_uplink_after_deadline_uses_no_cpu(self):
        result = simulate_execution(scheduler_env(), [
            (0, task(work=(1,), deadline=2, data_in=5), [0])])
        np.testing.assert_allclose(result["uav_busy_time"], [0, 0])
        self.assertTrue(np.isinf(result["finish_times"][0]))


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = SFCEnv(config={"K": 2, "M": 3})
        self.env.reset(seed=17)
        for ue in self.env.ues:
            ue.task_buffer.clear()
        self.env._calculate_rate = lambda *args, **kwargs: 1e12
        self.env._handle_charging = lambda: {}
        for i, uav in enumerate(self.env.uavs):
            uav.loc = np.array([100 + 100 * (i % 2), 100 + 100 * (i // 2)], dtype=np.float32)
            uav.velocity[:] = 0

    def tearDown(self):
        self.env.close()

    def install_tasks(self, count=2):
        tasks = []
        for i in range(count):
            sfc = task(i, (self.env.uavs[i % self.env.N].cpu_freq * 0.2,), deadline=5)
            self.env.ues[i].task_buffer.append(sfc)
            tasks.append((i, sfc, np.array([i % self.env.N])))
        self.env.current_cand_tasks = [(i, sfc) for i, sfc, _ in tasks]
        return tasks

    def test_cpu_capacity_regression_through_environment(self):
        tasks = self.install_tasks()
        self.env.current_time = 1
        for _, sfc, mapping in tasks:
            sfc.vnf_chain[0].required_cycles = 5e8
            sfc.deadline = 4
            mapping[:] = 0
        result = self.env._process_sfc_tasks(tasks)
        self.assertEqual(result["completed_count"], 0)
        self.assertEqual(result["timeout_count"], 2)
        self.assertAlmostEqual(result["total_compute_energy"], 3 * self.env.config["UAV_FULL_LOAD_POWER"])

    def test_billing_crash_still_settles_survivor(self):
        tasks = self.install_tasks()
        self.env.uavs[0].e_battery = 0.001
        result = self.env._process_sfc_tasks(tasks)
        self.assertTrue(result["uav_crashed"])
        self.assertEqual(result["completed_count"], 1)
        self.assertEqual(result["crash_failed_count"], 1)
        self.assertEqual(tasks[0][1].status, -2)
        self.assertEqual(tasks[1][1].status, 1)
        self.assertTrue(all(not ue.task_buffer for ue in self.env.ues))

    def test_billing_terminal_counts_all_tasks(self):
        self.install_tasks(3)
        self.env.uavs[0].e_battery = float(self.env.hover_energy[0]) + 0.001
        action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        logits = action[2 * self.env.N:].reshape(self.env.K, self.env.L, self.env.N)
        logits[0, :, 0] = 1
        logits[1, :, 1] = 1
        _, reward, terminated, truncated, info = self.env.step(action)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["total_available"], 3)
        self.assertEqual(info["completed_count"], 1)
        self.assertEqual(info["crash_failed_count"], 2)
        self.assertEqual(info["failed_count"] + info["completed_count"], 3)
        self.assertEqual(self.env.current_cand_tasks, [])
        self.assertTrue(all(not ue.task_buffer for ue in self.env.ues))
        self.assertEqual(reward, self.env.config["RWD_CRASH"])
        self.assertAlmostEqual(reward, sum(v for k, v in info.items() if k.startswith("r_")))

    def test_flight_terminal_counts_all_tasks(self):
        self.install_tasks(3)
        self.env.uavs[0].e_battery = 0
        _, _, terminated, _, info = self.env.step(np.zeros(self.env.action_space.shape))
        self.assertTrue(terminated)
        self.assertEqual(info["crash_failed_count"], info["total_available"])
        self.assertEqual(info["total_available"], 3)
        self.assertEqual(self.env.current_cand_tasks, [])

    def test_motion_stores_terminal_not_average_velocity(self):
        action = np.zeros((self.env.N, 2))
        action[0, 0] = 1
        start = self.env.uavs[0].loc.copy()
        self.env._handle_mobility_and_energy(action)
        np.testing.assert_allclose(self.env.uavs[0].loc - start, [2.5, 0])
        np.testing.assert_allclose(self.env.uavs[0].velocity, [5, 0])
        self.env._handle_mobility_and_energy(np.zeros_like(action))
        np.testing.assert_allclose(self.env.uavs[0].loc - start, [7.5, 0])

    def test_speed_boundary_and_collision(self):
        self.env.uavs[0].loc[:] = [499, 100]
        self.env.uavs[0].velocity[:] = [20, 0]
        self.env._handle_mobility_and_energy(np.ones((self.env.N, 2)))
        self.assertEqual(self.env.uavs[0].loc[0], 500)
        self.assertEqual(self.env.uavs[0].velocity[0], 0)
        self.assertLessEqual(np.linalg.norm(self.env.uavs[1].velocity), 20.00001)
        self.env.uavs[0].loc[:] = [100, 100]
        self.env.uavs[1].loc[:] = [111, 100]
        self.env.uavs[0].velocity[:] = 0
        self.env.uavs[1].velocity[:] = 0
        action = np.zeros((self.env.N, 2)); action[0, 0] = 1; action[1, 0] = -1
        result = self.env._handle_mobility_and_energy(action)
        self.assertGreater(result["collision_count"], 0)
        np.testing.assert_allclose(self.env.uavs[0].velocity, [0, 0])
        np.testing.assert_allclose(self.env.uavs[1].loc, [111, 100])

    def test_seed_reproducibility_and_sb3_contract(self):
        first, _ = self.env.reset(seed=81)
        second, _ = self.env.reset(seed=81)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(self.env.total_gen_tasks, sum(len(ue.task_buffer) for ue in self.env.ues))
        check_env(SFCEnv(), warn=True)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env = SFCEnv()
        self.env.reset(seed=11)
        for i, uav in enumerate(self.env.uavs):
            uav.loc = np.array([100 + 100 * (i % 2), 100 + 100 * (i // 2)], dtype=np.float32)
            uav.velocity = np.array([3.0, -2.0], dtype=np.float32)
        extractor = SFCFeaturesExtractor(self.env.observation_space, n_uavs=self.env.N,
                                         m_candidates=self.env.M, grid_res=self.env.grid_res)
        self.actor = DiffusionPolicyActor(
            self.env.observation_space, self.env.action_space, [32, 32], extractor,
            extractor.features_dim, self.env.N, self.env.M, self.env.K, 256,
            hybrid_categorical_placement=True, relational_placement=True,
        )

    def tearDown(self):
        self.env.close()

    def obs(self):
        return {k: torch.tensor(v).unsqueeze(0) for k, v in self.env._get_obs().items()}

    def assert_geometry_matches(self, action):
        predicted, _ = self.actor._relational_geometry(self.obs(), torch.tensor(action).reshape(1, -1))
        self.env._handle_mobility_and_energy(action)
        actual = np.array([u.loc for u in self.env.uavs]) / [500, 500]
        np.testing.assert_allclose(predicted.detach().numpy()[0], actual, atol=1e-6)

    def test_acceleration_and_inertia_geometry(self):
        self.assert_geometry_matches(np.ones((self.env.N, 2), dtype=np.float32))
        self.assert_geometry_matches(np.zeros((self.env.N, 2), dtype=np.float32))

    def test_speed_limit_boundary_and_collision_geometry(self):
        self.env.uavs[0].loc[:] = [499, 100]
        self.env.uavs[0].velocity[:] = [20, 0]
        self.env.uavs[1].loc[:] = [110, 200]
        self.env.uavs[2].loc[:] = [100, 200]
        self.assert_geometry_matches(np.ones((self.env.N, 2), dtype=np.float32))

    def test_low_energy_hover_and_critical_geometry(self):
        self.env.uavs[0].velocity[:] = [20, 0]
        self.env.uavs[0].e_battery = float(self.env.hover_energy[0]) + 1
        self.assert_geometry_matches(np.zeros((self.env.N, 2), dtype=np.float32))
        self.env.uavs[0].e_battery = 0
        self.assert_geometry_matches(np.zeros((self.env.N, 2), dtype=np.float32))

    def test_none_is_pre_move_and_gradients_are_finite(self):
        obs = self.obs()
        predicted, _ = self.actor._relational_geometry(obs, None)
        np.testing.assert_allclose(predicted.detach()[0], np.array([u.loc for u in self.env.uavs]) / 500)
        action = torch.zeros((1, self.env.N * 2), requires_grad=True)
        predicted, _ = self.actor._relational_geometry(obs, action)
        predicted.sum().backward()
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertGreater(action.grad.abs().sum().item(), 0)

    def test_custom_dynamics_and_environment_mask(self):
        self.env.dt_fly = 0.5
        self.env.hover_energy *= 0.5
        self.env.config["APPLY_MOBILITY_MASK_IN_ENV"] = True
        self.env.uavs[0].e_battery = 0.1 * self.env.uavs[0].battery_capacity
        self.actor.mobility_dynamics.update(dt=0.5, env_mask=True)
        self.assert_geometry_matches(np.ones((self.env.N, 2), dtype=np.float32))


class WarmupTests(unittest.TestCase):
    def test_main_warmup_preserves_time_limit_bootstrapping(self):
        # Run the actual main warmup loop, replacing subprocesses and training
        # only. Inspect the real SB3 replay buffer rather than a duplicate loop.
        import main as entry
        from omegaconf import OmegaConf
        from stable_baselines3 import SAC
        captured = []
        def make_model(**kwargs):
            model = SAC(**kwargs)
            captured.append(model)
            model.learn = lambda **kwargs: model
            model.save = lambda *args: None
            return model
        cfg = OmegaConf.create(dict(
            algo_name="SAC", seed=7, n_envs=1, warmup_steps=3,
            warmup_random_fraction=1.0, total_timesteps=0, use_wandb=False,
            run_heuristic_baseline=False, exp_name="regression", env={"MAX_STEPS": 1},
            sac={"buffer_size": 16, "batch_size": 2},
        ))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(entry, "SubprocVecEnv", side_effect=DummyVecEnv), \
             patch.object(entry.HydraConfig, "get", return_value=SimpleNamespace(run=SimpleNamespace(dir=tmp))), \
             patch.dict(entry.ALGO_REGISTRY, {"SAC": make_model}):
            entry.main.__wrapped__(cfg)
        model = captured[0]
        buffer = model.replay_buffer
        self.assertEqual(buffer.size(), 3)
        np.testing.assert_array_equal(buffer.dones[:3], 1)
        np.testing.assert_array_equal(buffer.timeouts[:3], 1)
        self.assertTrue(torch.all(buffer.sample(8).dones == 0))
        model.env.close()


if __name__ == "__main__":
    unittest.main()
