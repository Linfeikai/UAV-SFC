import numpy as np
import time
from core.sfc_env import SFCEnv


def check_obs_bounds(obs, space, step_idx):
    """检查观测值是否在定义的 Box 范围内"""
    is_within = space.contains(obs)
    if not is_within:
        print(f"\n[警告] Step {step_idx}: 观测值超出 Space 定义范围!")
        # 找出哪些维度超了
        out_low = obs < space.low
        out_high = obs > space.high
        if np.any(out_low):
            print(f"  -> 以下索引过小: {np.where(out_low)[0]}, 最小值: {obs[out_low]}")
        if np.any(out_high):
            print(
                f"  -> 以下索引过大: {np.where(out_high)[0]}, 最大值: {obs[out_high]}"
            )


def test_diagnostic():
    print("=== 开始 SFCEnv 深度诊断测试 ===\n")

    # 1. 初始化测试
    try:
        env = SFCEnv()
        print(f"[OK] 环境初始化成功")
        print(f"动作空间: {env.action_space}")
        print(f"观测空间维度: {env.observation_space.shape[0]}")
    except Exception as e:
        print(f"[FAILED] 初始化失败: {e}")
        return

    # 2. Reset 测试
    print("\n--- 执行 Reset 测试 ---")
    obs, info = env.reset()
    print(f"Reset 返回观测形状: {obs.shape}")
    if obs.shape[0] != env.observation_space.shape[0]:
        print(f"[错误] Reset 返回维度与定义不符！")

    # 3. 动作解析与运行测试
    print("\n--- 执行 100 步随机动作应力测试 ---")
    start_time = time.time()
    total_reward = 0

    for i in range(100):
        action = env.action_space.sample()

        # 捕获 step 中的潜在错误
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # 每一秒打印一次关键指标，观察逻辑是否闭环
            if i % 20 == 0:
                print(
                    f"Step {i:03d} | Reward: {reward:6.2f} | 成功: {info.get('completed_count', 0)} "
                    f"| 丢弃: {info.get('dropped_count', 0)} | 超时: {info.get('timeout_count', 0)}"
                )

            # 实时检查观测界限
            check_obs_bounds(obs, env.observation_space, i)

            if terminated or truncated:
                print(
                    f"Episode 结束 (Terminated: {terminated}, Truncated: {truncated})"
                )
                obs, info = env.reset()

        except Exception as e:
            print(f"\n[CRITICAL] Step {i} 崩溃! 错误信息: {e}")
            import traceback

            traceback.print_exc()
            break

    duration = time.time() - start_time
    print(f"\n测试完成! 平均每步耗时: {duration / 100:.4f}s (越小代表训练吞吐越高)")
    print(f"总奖励: {total_reward:.2f}")

    # 4. 逻辑一致性专项检查
    print("\n--- 专项逻辑检查 ---")
    # 检查是否成功刷洗了候选池
    if hasattr(env, "current_cand_tasks"):
        print(f"[OK] 候选池状态正常，当前UE数: {len(env.current_cand_tasks)}")
    else:
        print("[错误] 环境没有 current_cand_tasks 属性")

    # 检查负载状态更新
    if hasattr(env, "uav_load_status"):
        print(f"[OK] UAV 负载率监控正常: {env.uav_load_status}")
    else:
        print("[错误] 环境没有 uav_load_status 属性")


import numpy as np
from core.sfc_env import SFCEnv


def run_heuristic_eval(strategy="static", n_episodes=5):
    env = SFCEnv()
    all_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        for _ in range(env.config["MAX_STEPS"]):
            # --- 核心逻辑：手动构造不同策略的动作 ---
            if strategy == "static":
                # 移动设为 0，Pick 随机选，Place 随机分
                mobility = np.zeros(env.N * 2)
                pick_place = np.random.uniform(-1, 1, env.K + env.K * env.L)
                action = np.concatenate([mobility, pick_place])

            elif strategy == "greedy":
                # 1. 移动：寻找最近的活跃 UE
                mobility_list = []
                for uav in env.uavs:
                    if env.current_cand_tasks:
                        # 找到最近的任务位置
                        target_ue_id = env.current_cand_tasks[0][
                            0
                        ]  # 简化：追最紧急的任务
                        target_loc = env.ues[target_ue_id].loc
                        diff = target_loc - uav.loc
                        norm_diff = diff / (np.linalg.norm(diff) + 1e-9)
                        mobility_list.extend(norm_diff)
                    else:
                        mobility_list.extend([0, 0])
                mobility = np.array(mobility_list)
                # 2. Pick/Place 依然保持随机或简单分配，排除算法干扰
                pick_place = np.random.uniform(-1, 1, env.K + env.K * env.L)
                action = np.concatenate([mobility, pick_place])

            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if term or trunc:
                break

        all_rewards.append(ep_reward)

    print(f"策略 {strategy} | 平均奖励: {np.mean(all_rewards):.2f}")


if __name__ == "__main__":
    run_heuristic_eval(strategy="static", n_episodes=5)
    run_heuristic_eval(strategy="greedy", n_episodes=5)
