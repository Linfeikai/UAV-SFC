"""Training-only current-slot placement teacher; inference uses the learned actor."""
from contextlib import contextmanager
import copy
import random

import numpy as np

from core.uav import UAVNode
from test.evalu import smart_heuristic_policy


RECIPES = ((1, 0, 0), (1, .25, 0), (1, .75, 0), (1, 1.5, 0),
           (1, 0, .25), (1, .5, .5), (0, 1, .5))


@contextmanager
def _preserve_global_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _brake(env):
    velocity = np.asarray([u.velocity for u in env.uavs])
    return np.clip(-velocity / (UAVNode.MAX_ACCELERATION * env.dt_fly), -1, 1).reshape(-1)


def _greedy_candidate(env, load_weight, distance_weight, stay_weight):
    """Rank feasible placements from current demand, geometry and affinity."""
    n, k, l = env.N, env.K, env.L
    capacity = np.array([u.cpu_freq * (env.time_slot - env.dt_fly) for u in env.uavs])
    used = np.zeros(n)
    scores = np.full((k, l, n), -.5, dtype=np.float32)
    types, heavy = env.config.get('UAV_TYPES'), env.config.get('GPU_VNFS', set())
    for task, (ue, sfc) in enumerate(env.current_cand_tasks[:k]):
        previous = None
        for vnf_id, vnf in enumerate(sfc.vnf_chain[:l]):
            candidates = [u for u in range(n) if not env.uavs[u].is_crashed]
            preferred = ([u for u in candidates if types and types[u] == 'gpu']
                         if vnf.vnf_type in heavy else candidates)
            preferred = preferred or candidates or list(range(n))
            demand = np.array([vnf.required_cycles / max(env._vnf_affinity(u, vnf), 1e-9)
                               for u in range(n)])
            feasible = [u for u in preferred if used[u] + demand[u] <= capacity[u]]
            choices = feasible or preferred
            source = env.ues[ue].loc if previous is None else env.uavs[previous].loc
            cost = {u: load_weight * (used[u] + demand[u]) / capacity[u]
                       + distance_weight * np.linalg.norm(env.uavs[u].loc - source) / 500.0
                       + stay_weight * float(previous is not None and u != previous)
                    for u in choices}
            chosen = min(choices, key=lambda u: (cost[u], u))
            scores[task, vnf_id, chosen] = .5
            used[chosen] += demand[chosen]
            previous = chosen
    return np.concatenate([_brake(env), scores.flatten()]).astype(np.float32)


def _executed_action(env, action, info, task_ids):
    """Encode executed assignments, including the capacity decoder's fallback."""
    out = action.copy()
    values = out[2 * env.N:].reshape(env.K, env.L, env.N)
    for k, ue in enumerate(task_ids):
        mapping = info.get('detailed_mappings', {}).get(ue)
        if mapping is None:
            continue
        for l, u in enumerate(mapping):
            values[k, l] = -.5
            values[k, l, u] = .5
    return out


def lookahead_expert_action(env, candidate_budget=8):
    """Compare current-slot rewards on copies; never read future observations.

    This full-state simulator teacher is only used to collect training labels.
    Including the existing heuristic protects its immediate objective. Each
    branch copies environment-owned RNGs; global streams are preserved as well.
    """
    if not 1 <= candidate_budget <= 1 + len(RECIPES):
        raise ValueError('candidate_budget must be between 1 and 8')
    with _preserve_global_rng():
        base = smart_heuristic_policy(env)
        base[:2 * env.N] = _brake(env)
        candidates = [base] + [_greedy_candidate(env, *recipe)
                               for recipe in RECIPES[:candidate_budget - 1]]
        task_ids = [ue for ue, _ in env.current_cand_tasks[:env.K]]
        best, best_score = base, (-np.inf,)
        for action in candidates:
            branch = copy.deepcopy(env)
            branch.config = dict(branch.config, RECORD_DEPLOYMENT=True)
            try:
                _, reward, _, _, info = branch.step(action)
                score = (float(reward), int(info.get('completed_count', 0)),
                         -float(info.get('total_energy_J', 0)))
                if score > best_score:
                    best_score = score
                    best = _executed_action(env, action, info, task_ids)
            finally:
                branch.close()
        return best
