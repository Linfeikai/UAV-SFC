"""Observation-only deployment proposals for a learned whole-action scorer.

No environment object or simulator is available to this module. Dynamics and
normalization constants are public policy metadata, frozen with the setting.
"""
from dataclasses import dataclass

import numpy as np


RECIPES = ((1, 0, 0), (1, .25, 0), (1, .75, 0), (1, 1.5, 0),
           (1, 0, .25), (1, .5, .5), (0, 1, .5))


@dataclass(frozen=True)
class CandidateGeometry:
    n_uavs: int = 4
    m_candidates: int = 12
    decision_tasks: int = 6
    max_vnf_len: int = 4
    grid_res: int = 3
    cpu_hz: float = 500000000.
    slot_duration: float = 8.
    dt_fly: float = 1.
    max_speed: float = 20.
    max_acceleration: float = 5.
    ground_width: float = 500.
    ground_height: float = 500.


def placement_candidates(obs, geometry=CandidateGeometry()):
    """Return the base placement and seven recipes using existing observation fields.

    Motion is the same explicit velocity-braking controller for every proposal.
    A scorer learns which complete deployment to execute; no per-VNF action
    averaging or online simulator scoring is performed here.
    """
    g = geometry
    state = np.asarray(obs['state'])
    uavs = state[:g.n_uavs * 10].reshape(g.n_uavs, 10)
    offset = g.n_uavs * 10 + g.grid_res ** 2 * 3
    tasks = state[offset:offset + g.m_candidates * 5].reshape(g.m_candidates, 5)
    context = np.asarray(obs['placement_context']).reshape(
        g.decision_tasks, g.max_vnf_len, g.n_uavs + 1)
    map_scale = np.array([g.ground_width, g.ground_height])
    positions = uavs[:, :2].astype(np.float64) * map_scale
    task_positions = tasks[:, :2].astype(np.float64) * map_scale
    brake = np.clip(-uavs[:, 2:4] * g.max_speed /
                    (g.max_acceleration * g.dt_fly), -1, 1).ravel()
    crashed = uavs[:, 5] > .5
    gpu = uavs[:, 9] > .5
    capacity = g.cpu_hz * (g.slot_duration - g.dt_fly)
    proposals = []
    for recipe in (None,) + RECIPES:
        used = np.zeros(g.n_uavs, dtype=np.float64)
        scores = np.full((g.decision_tasks, g.max_vnf_len, g.n_uavs), -.5, dtype=np.float32)
        for task in range(g.decision_tasks):
            previous = None
            for vnf in range(g.max_vnf_len):
                if context[task, vnf, -1] < .5:
                    continue
                demand = context[task, vnf, :g.n_uavs].astype(np.float64)
                # In this frozen equal-CPU heterogeneous setting, affinity is
                # directly exposed by GPU vs general effective demand ratios.
                heavy = bool(gpu.any() and (~gpu).any() and
                             demand[~gpu].min() > 1.5 * demand[gpu].max())
                available = [u for u in range(g.n_uavs) if not crashed[u]]
                preferred = [u for u in available if gpu[u]] if heavy else available
                preferred = preferred or available or list(range(g.n_uavs))
                if recipe is None:
                    distances = np.linalg.norm(positions - task_positions[task], axis=1)
                    utility = capacity * (1 - used) * 1e-9 - (0 if heavy else .01 * distances)
                    chosen = max(preferred, key=lambda u: (utility[u], -u))
                else:
                    feasible = [u for u in preferred if used[u] + demand[u] <= 1.]
                    choices = feasible or preferred
                    source = task_positions[task] if previous is None else positions[previous]
                    load, distance, stay = recipe
                    costs = {u: load * (used[u] + demand[u])
                                + distance * np.linalg.norm(positions[u] - source) / 500.
                                + stay * float(previous is not None and u != previous)
                             for u in choices}
                    chosen = min(choices, key=lambda u: (costs[u], u))
                scores[task, vnf, chosen] = .5
                used[chosen] += demand[chosen]
                previous = chosen
        proposals.append(np.concatenate([brake, scores.ravel()]).astype(np.float32))
    return np.stack(proposals)


def candidate_features(obs, proposals, geometry=CandidateGeometry()):
    """Encode each complete proposal with observed demand, geometry and load.

    These features never use achieved rewards, future observations, task object
    fields, or simulator state. Labels are collected separately during training.
    """
    g = geometry
    state = np.asarray(obs['state'], dtype=np.float32)
    context = np.asarray(obs['placement_context'], dtype=np.float32).reshape(
        g.decision_tasks, g.max_vnf_len, g.n_uavs + 1)
    uavs = state[:g.n_uavs * 10].reshape(g.n_uavs, 10)
    offset = g.n_uavs * 10 + g.grid_res ** 2 * 3
    tasks = state[offset:offset + g.m_candidates * 5].reshape(g.m_candidates, 5)
    shared = np.concatenate([state, np.asarray(obs['mobility_bounds']), context.ravel()])
    valid = context[..., -1]
    rows = []
    for proposal in proposals:
        choices = proposal[2 * g.n_uavs:].reshape(g.decision_tasks, g.max_vnf_len, g.n_uavs).argmax(-1)
        assigned = np.eye(g.n_uavs, dtype=np.float32)[choices] * valid[..., None]
        demand = (assigned * context[..., :g.n_uavs]).sum(-1)
        load = (assigned * demand[..., None]).sum((0, 1))
        locations = uavs[choices, :2]
        uplink = np.linalg.norm(locations[:, 0] - tasks[:g.decision_tasks, :2], axis=-1)
        hops = np.zeros_like(valid)
        hops[:, 1:] = np.linalg.norm(locations[:, 1:] - locations[:, :-1], axis=-1) * valid[:, 1:]
        # Sum of full-CPU work per chain and terminal competition distinguish
        # coherent deployments that individual placement imitation can mix up.
        chain_work = demand.sum(-1)
        competition = (assigned * load[None, None, :]).sum(-1) * valid
        rows.append(np.concatenate([shared, assigned.ravel(), demand.ravel(), load,
                                    uplink, hops.ravel(), chain_work, competition.ravel()]))
    return np.asarray(rows, dtype=np.float32)
