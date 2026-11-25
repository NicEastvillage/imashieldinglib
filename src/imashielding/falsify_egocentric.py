"""Falsification testing of the ego-centric abstraction against concrete runs.

For concrete states ``s`` reached by simulation (before anybody violated their
clause), every agent ``i`` and every action ``a`` of ``i``, we check the
p_i-simulation conditions (Definition 12 in the paper) for the abstraction
function ``alpha = system.normalize(., i)``:

* labels: if ``alpha(s)`` is SAFE then ``s`` satisfies p_i; if it is
  SAFE_RELEASE then ``s`` satisfies p_i and somebody else violated theirs;
* successors: for every concrete successor ``s'`` of ``s`` under ``a`` (all
  actions of the other agents, all outcomes):

  - if ``s'`` violates p_i, some abstract successor of ``(alpha(s), a)`` is UNSAFE,
  - if somebody else violated their clause in ``s'``, nothing is required (the
    release condition holds, the trace is done),
  - otherwise ``alpha(s')`` must be an abstract successor of ``(alpha(s), a)``,
    unless ``a`` is abstractly unsafe (has an UNSAFE abstract successor): such an
    action is never allowed by the shield, so its successors do not matter
    (and the abstraction only returns one unsafe successor for it).

Successors after a release state are not needed: the release property is
satisfied as soon as the release condition holds (with p_i holding there).

Run as a script for a quick check of the bundled scenarios::

    python -m imashielding.falsify_egocentric
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple

from imashielding.roads import CarState, RoadNetwork, RoadNetworkState
from imashielding.roads_egocentric import EgoCentricRoadSystem, EgoCentricState
from imashielding.roads_reduction import road_safety
from imashielding.systems import StateSafety


@dataclass
class FalsificationReport:
    checked_states: int = 0
    checked_transitions: int = 0
    violations: List[str] = field(default_factory=list)
    stats: Counter = field(default_factory=Counter)

    @property
    def ok(self) -> bool:
        return not self.violations

    def __str__(self) -> str:
        lines = [f'states={self.checked_states} transitions={self.checked_transitions} '
                 f'violations={len(self.violations)}']
        for k, v in sorted(self.stats.items()):
            lines.append(f'  {k}: {v}')
        for v in self.violations[:10]:
            lines.append('  VIOLATION ' + v)
        return '\n'.join(lines)


def ego_view(state: RoadNetworkState, agent: int) -> RoadNetworkState:
    """Reindex the cars such that ``agent`` has id 0 (as ``road_safety`` expects)."""
    ego = state.cars[agent]
    others = [c for c in state.cars if c.id != agent]
    cars = (CarState(0, ego.road, ego.pos, ego.vel, ego.just_merged, False),) + tuple(
        CarState(i + 1, c.road, c.pos, c.vel, c.just_merged, False) for i, c in enumerate(others)
    )
    return RoadNetworkState(cars)


def falsify(
    network: RoadNetwork,
    system: EgoCentricRoadSystem,
    *,
    steps: int = 2000,
    trace_length: int = 200,
    seed: int = 0,
    policy: Optional[Callable[[RoadNetworkState, random.Random], Tuple[int, ...]]] = None,
    max_violations: int = 20,
    verbose: bool = False,
) -> FalsificationReport:
    rng = random.Random(seed)
    safety = road_safety(network, min_dist=system.min_dist, merge_space=system.merge_space)
    n = len(network.cars)
    n_actions = len(network.accel_actions)
    report = FalsificationReport()
    succ_cache: Dict[Tuple[EgoCentricState, int], frozenset] = {}

    def abstract_succs(t: EgoCentricState, a: int) -> frozenset:
        key = (t, a)
        res = succ_cache.get(key)
        if res is None:
            res = frozenset(system.get_abstract_successor_states(t, a))
            succ_cache[key] = res
        return res

    def violation(msg: str) -> None:
        report.violations.append(msg)
        if verbose:
            print('VIOLATION', msg)

    state = network.get_init_state()
    t_len = 0
    start = time.time()
    for step in range(steps):
        if len(report.violations) >= max_violations:
            break
        if safety(state) != StateSafety.SAFE or t_len >= trace_length:
            state = network.get_init_state()
            t_len = 0
            report.stats['restarts'] += 1
        report.checked_states += 1

        for i in range(n):
            t = system.normalize(state, i)
            label = system.label(t)
            if label == StateSafety.UNSAFE:
                report.stats['abstract UNSAFE but concrete SAFE (conservative)'] += 1
                continue    # not explored by synthesis; nothing to simulate
            if label == StateSafety.SAFE_RELEASE:
                violation(f'abstract release, concrete safe: agent {i} {state} -> {t}')
                continue

            for a in range(n_actions):
                succs = abstract_succs(t, a)
                any_unsafe = any(system.label(x) == StateSafety.UNSAFE for x in succs)
                other_ranges = [range(n_actions)] * (n - 1)
                for combo in product(*other_ranges):
                    actions = list(combo)
                    actions.insert(i, a)
                    for cs in network.get_successors(state, tuple(actions)):
                        report.checked_transitions += 1
                        s2 = cs.state
                        status = safety(ego_view(s2, i))
                        t2 = system.normalize(s2, i)
                        if status == StateSafety.UNSAFE:
                            report.stats['concrete unsafe successors'] += 1
                            if not any_unsafe:
                                violation(f'unsafe concrete successor not matched: agent {i} action {a}\n'
                                          f'    s={state}\n    s2={s2}\n    t={t}\n    t2={t2}')
                            elif t2 not in succs:
                                report.stats['unsafe successor matched only weakly'] += 1
                            if system.label(t2) != StateSafety.UNSAFE:
                                violation(f'concrete unsafe but abstract {system.label(t2)}: agent {i}\n'
                                          f'    s2={s2}\n    t2={t2}')
                            continue
                        if status == StateSafety.SAFE_RELEASE:
                            report.stats['concrete release successors'] += 1
                            continue
                        if any_unsafe:
                            # The action is unsafe in the abstraction, so the shield never
                            # allows it; its other successors are not generated.
                            report.stats['successors of abstractly unsafe actions'] += 1
                            continue
                        if t2 not in succs:
                            violation(f'missing abstract successor: agent {i} action {a}\n'
                                      f'    s={state}\n    s2={s2}\n    t={t}\n    t2={t2}')
                        lbl2 = system.label(t2)
                        if lbl2 == StateSafety.SAFE_RELEASE:
                            violation(f'abstract release, concrete safe: agent {i}\n    s2={s2}\n    t2={t2}')
                        elif lbl2 == StateSafety.UNSAFE:
                            report.stats['abstract UNSAFE but concrete SAFE (conservative)'] += 1

        # Advance the concrete run.
        actions = policy(state, rng) if policy is not None else tuple(rng.randrange(n_actions) for _ in range(n))
        outcomes = network.get_successors(state, actions)
        r = rng.random()
        acc = 0.0
        for cs in outcomes:
            acc += cs.chance
            if r <= acc:
                state = cs.state
                break
        else:
            state = outcomes[-1].state
        state = RoadNetworkState(tuple(CarState(c.id, c.road, c.pos, c.vel, c.just_merged, False) for c in state.cars))
        t_len += 1

    report.stats['seconds'] = int(time.time() - start)
    report.stats['abstract states touched'] = len({k[0] for k in succ_cache})
    return report


def cautious_random_policy(network: RoadNetwork, min_dist: int, merge_space: int, brake_bias: float = 0.4):
    """A random policy that brakes more often, to get longer runs without crashes."""
    def policy(state: RoadNetworkState, rng: random.Random) -> Tuple[int, ...]:
        acts = []
        for c in state.cars:
            r = rng.random()
            if r < brake_bias:
                acts.append(0)
            else:
                acts.append(rng.randrange(len(network.accel_actions)))
        return tuple(acts)
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    scenarios = {
        'roundabout': lambda: RoadNetwork.create_roundabout_scenario(car_count=4, road_per_circ=40),
        'crossing_roads': lambda: RoadNetwork.create_crossing_roads_scenario(car_count=3, max_vel=2),
        'crossing_roads_v4': lambda: RoadNetwork.create_crossing_roads_scenario(car_count=3, max_vel=4),
        'double_roundabout': lambda: RoadNetwork.create_double_roundabout_scenario(car_count=3),
    }
    for name, make in scenarios.items():
        net = make()
        system = EgoCentricRoadSystem(net)
        report = falsify(net, system, steps=args.steps, seed=args.seed,
                         policy=cautious_random_policy(net, 3, 8))
        print(f'== {name}\n{report}')


if __name__ == '__main__':
    main()
