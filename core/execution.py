"""Slot-bounded, equal-share CPU execution for sequential SFCs.

Radio links retain the environment's independent constant-rate link model.
Only ready VNFs share a UAV's CPU. Expired work is cancelled, and no work
or energy is carried implicitly beyond the current compute window.
"""

import numpy as np


def simulate_execution(env, eligible_tasks):
    count = len(eligible_tasks)
    window = max(0.0, float(env.time_slot - env.dt_fly))
    finish_times = np.full(count, np.inf, dtype=np.float64)
    busy_time = np.zeros(env.N, dtype=np.float64)
    tx_energy = np.zeros(env.N, dtype=np.float64)
    # waiting -> running -> waiting (next VNF) or finished/expired
    phases = ["waiting"] * count
    indices = np.zeros(count, dtype=np.int64)
    remaining = np.zeros(count, dtype=np.float64)  # full-CPU seconds
    ready_at = np.zeros(count, dtype=np.float64)
    cutoffs = np.array([
        min(window, max(0.0, float(sfc.deadline - env.current_time)))
        for _, sfc, _ in eligible_tasks
    ])
    eps = 1e-10

    for task_id, (ue_id, sfc, mapping) in enumerate(eligible_tasks):
        rate = max(float(env._calculate_rate(
            env.ues[ue_id].loc, env.uavs[mapping[0]].loc,
            env.config["P_UPLINK"], link_type="UAV_UE",
        )), 1e-9)
        ready_at[task_id] = sfc.vnf_chain[0].data_in / rate

    now = 0.0
    while True:
        # Drain simultaneous/zero-duration transitions before advancing time.
        changed = True
        while changed:
            changed = False
            for task_id, (_, sfc, mapping) in enumerate(eligible_tasks):
                if phases[task_id] == "running" and remaining[task_id] <= eps:
                    vnf_id = int(indices[task_id])
                    sfc.vnf_chain[vnf_id].processed = True
                    sfc.current_vnf_index = vnf_id + 1
                    indices[task_id] += 1
                    changed = True
                    if indices[task_id] == len(sfc.vnf_chain):
                        phases[task_id] = "finished"
                        finish_times[task_id] = now
                        continue
                    sender = int(mapping[vnf_id])
                    receiver = int(mapping[vnf_id + 1])
                    transfer = 0.0
                    if sender != receiver:
                        rate = max(float(env._calculate_rate(
                            env.uavs[sender].loc, env.uavs[receiver].loc,
                            env.config["UAV_P_TX_CROSS"], link_type="UAV_UAV",
                        )), 1e-9)
                        transfer = sfc.vnf_chain[vnf_id].data_out / rate
                        tx_energy[sender] += env.config["UAV_P_TX_CROSS"] * min(
                            transfer, max(0.0, cutoffs[task_id] - now)
                        )
                    ready_at[task_id] = now + transfer
                    phases[task_id] = "waiting"
                if (phases[task_id] == "waiting"
                        and ready_at[task_id] <= now + eps):
                    vnf = sfc.vnf_chain[int(indices[task_id])]
                    uav_id = int(mapping[int(indices[task_id])])
                    remaining[task_id] = vnf.required_cycles / (
                        env.uavs[uav_id].cpu_freq * env._vnf_affinity(uav_id, vnf)
                    )
                    phases[task_id] = "running"
                    changed = True

        for task_id in range(count):
            # Completion exactly at a deadline/window boundary is valid.
            if (phases[task_id] not in {"finished", "expired"}
                    and now >= cutoffs[task_id] - eps):
                phases[task_id] = "expired"
        live = [i for i in range(count)
                if phases[i] not in {"finished", "expired"}]
        if not live or now >= window - eps:
            break

        running = [[] for _ in range(env.N)]
        next_time = min(window, min(cutoffs[i] for i in live))
        for task_id in live:
            if phases[task_id] == "waiting":
                next_time = min(next_time, ready_at[task_id])
            else:
                mapping = eligible_tasks[task_id][2]
                running[int(mapping[int(indices[task_id])])].append(task_id)
        for tasks in running:
            if tasks:
                next_time = min(next_time, now + min(
                    remaining[i] * len(tasks) for i in tasks
                ))
        elapsed = next_time - now
        if elapsed <= 0.0:
            raise RuntimeError("SFC execution clock failed to advance")
        for uav_id, tasks in enumerate(running):
            if tasks:
                busy_time[uav_id] += elapsed
                remaining[tasks] -= elapsed / len(tasks)
        now = next_time

    return {
        "finish_times": finish_times,
        "uav_busy_time": busy_time,
        "uav_tx_energy_bill": tx_energy,
    }
