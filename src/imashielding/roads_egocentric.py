"""Ego-centric, range-limited abstraction for arbitrary road networks.

The abstraction keeps only what the ego car needs to uphold its own clause of
``road_safety`` (see ``roads_reduction.py``):

* the ego car itself (road, position, velocity, ``just_merged``), where the
  position is collapsed to a canonical "far" position whenever the ego is far
  from both ends of its road (nothing interesting can be seen from there);
* the *view cone*: all cells within forward distance ``view_range`` of the ego
  along any path (splits branch the cone).  Each road segment in the cone is a
  "slot" whose nearest car is tracked.  A segment is *shadowed* if every path
  from the ego to it passes a tracked car; shadowed segments are not tracked
  (the car in between must be passed first, and that is a crash);
* every car within ``min_dist`` ahead (only ever more than one car in states
  where somebody already violated their clause; needed to label those states);
* for every merge within ``merge_range`` where the ego has to yield, the car
  on the priority road that is nearest to the merge point, if it is within
  ``approach_range`` of it;
* right after the ego crossed a yield line: the cars behind it within
  ``merge_space`` (only needed for labelling);
* cars within ``min_dist`` behind the ego (they only exist after somebody
  violated their clause, but are needed to label those states exactly).

Everything else is *hidden*.  The successor generator moves the ego and all
tracked cars with the concrete dynamics (``RoadNetwork._advance``) and
then adds, non-deterministically, every hidden car that could have become
visible during the step ("reveals"): cars entering the cone because the ego
moved forward, cars that were shadowed and are exposed because the car in
front turned off at a split, cars merging in from un-tracked yield roads, cars
entering an approach zone, and so on.  A reveal is only generated from cells
that were hidden before the step, and only with a previous velocity under
which the hidden car would not already have violated a clause together with a
tracked car (in that case the concrete state was a release state and need not
be simulated).

Safety labelling is done by running the concrete ``road_safety`` on the
tracked cars.  Since the tracked cars are a subset of the concrete cars with
exact relative positions, an abstract ``UNSAFE``/``SAFE_RELEASE`` verdict is
also a concrete one.  Concrete violations of the ego caused by an untracked
car are, before anybody else fails, only possible through the cars the
abstraction tracks; ``falsify_egocentric`` checks this (and the rest of the
simulation relation) empirically on concrete runs.

Two shortcuts keep synthesis cheap without affecting the shield: when an
action has an UNSAFE successor only that one is returned (the action is unsafe
anyway), and successors where another car violated first (SAFE_RELEASE) are
left out (they constrain nothing).

Nothing in an abstract state depends on the number of cars, so one shield
serves any number of cars on the same network (as long as their initial views
are covered).

Assumptions: ``min_vel == 0``, two-way splits and merges only, roads ending
in a merge do not split, no road loops onto itself (``road_safety`` does not
measure across such a loop), and every cycle is longer than ``view_range``.
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from imashielding.roads import CarState, RoadNetwork, RoadNetworkState
from imashielding.roads_reduction import _network_cache_key, road_safety
from imashielding.systems import LabelledTransitionSystem, StateSafety

Cell = Tuple[int, int]   # (road, pos)

# Segment keys of the view cone
SEG_AHEAD = (-1, 0)   # the ego's own road, ahead of the ego
SEG_WRAP = (-2, 0)    # the ego's own road, reached again by driving around a loop
# Other roads are keyed (road, 1)


@dataclass(frozen=True, slots=True)
class EgoCentricState:
    """An abstract state.

    ``cars[0]`` is the ego car; the other cars are the tracked cars, sorted and
    re-indexed.  If ``far`` is set, the ego is somewhere in the far-zone of its
    road and has been moved (together with the tracked cars on its road) to the
    canonical far position of that road.
    """
    far: bool
    cars: Tuple[CarState, ...]
    # States are dictionary keys all over synthesis; hashing the nested dataclasses
    # every time dominates the run time, so the hash is computed once.
    _hash: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, '_hash', hash((self.far, tuple(_key(c) for c in self.cars))))

    def __hash__(self) -> int:
        return self._hash

    def __getstate__(self):
        return (self.far, self.cars)

    def __setstate__(self, state):
        if len(state) == 2 and isinstance(state[1], dict):     # default slots pickling
            state = (state[1]['far'], state[1]['cars'])
        object.__setattr__(self, 'far', state[0])
        object.__setattr__(self, 'cars', state[1])
        self.__post_init__()

    def as_road_network_state(self) -> RoadNetworkState:
        return RoadNetworkState(self.cars)


@dataclass
class _Geometry:
    """View geometry of an ego position.  Independent of the other cars."""
    cell_dist: Dict[Cell, int] = field(default_factory=dict)      # cone cells -> forward distance
    cell_seg: Dict[Cell, Tuple[int, int]] = field(default_factory=dict)
    seg_order: List[Tuple[int, int]] = field(default_factory=list)  # increasing distance
    seg_preds: Dict[Tuple[int, int], Tuple[Tuple[int, int], ...]] = field(default_factory=dict)
    zones: Dict[int, int] = field(default_factory=dict)            # priority road -> first pos in zone
    behind: FrozenSet[Cell] = frozenset()                         # cells behind a just-merged ego
    close_behind: FrozenSet[Cell] = frozenset()                   # cells within min_dist behind the ego
    slot_cells: FrozenSet[Cell] = frozenset()                     # every cell something can be tracked in

    def in_zone(self, road: int, pos: int) -> bool:
        qmin = self.zones.get(road)
        return qmin is not None and pos >= qmin


class EgoCentricRoadSystem(LabelledTransitionSystem[EgoCentricState]):
    def __init__(
        self,
        network: RoadNetwork,
        min_dist: int = 3,
        merge_space: int = 8,
        view_range: Optional[int] = None,
        merge_range: Optional[int] = None,
        approach_range: Optional[int] = None,
        shadowing: bool = True,
        drop_release_successors: bool = True,
    ):
        net = network
        self._system = net
        self.min_dist = min_dist
        self.merge_space = merge_space
        self.shadowing = shadowing
        self.drop_release_successors = drop_release_successors
        V = net.max_vel
        assert net.min_vel == 0, 'the ego-centric abstraction assumes min_vel == 0'

        # Worst-case stopping distance of the ego (braking with -1 per step).
        stop = V * (V - 1) // 2
        self.view_range = view_range if view_range is not None else min_dist + stop + V
        self.merge_range = merge_range if merge_range is not None else stop + 2 * V
        self.approach_range = approach_range if approach_range is not None else merge_space + V * (V + 1)
        assert self.view_range >= min_dist
        assert self.approach_range >= merge_space

        # The safety property is the concrete one, evaluated on the tracked cars.
        self._safety = road_safety(net, min_dist=min_dist, merge_space=merge_space)

        roads = net.roads
        for r in roads:
            if r.id in (r.end_left, r.end_right):
                raise ValueError(
                    f'road {r.id} loops onto itself; road_safety does not measure distances '
                    f'across such a loop (use two segments, or the roundabout abstraction)')
        self._preds: Tuple[Tuple[int, ...], ...] = tuple(
            tuple(w.id for w in roads if r.id in (w.end_left, w.end_right)) for r in roads
        )
        # Merges: priority road of a merge road and the yield roads (with their merge road).
        self._yield_into: Dict[int, int] = {}
        for w in roads:
            if w.end_left == w.end_right and roads[w.end_left].start != w.id:
                self._yield_into[w.id] = w.end_left
        for r in roads:
            assert len(self._preds[r.id]) <= 2, 'only two-way merges are supported'
            for w in self._preds[r.id]:
                if len(self._preds[r.id]) == 2:
                    assert roads[w].end_left == roads[w].end_right, \
                        'roads ending in a merge must not split'

        # Far zone: positions where the view neither reaches the end nor the start of the road.
        self._back = max(V, min_dist)
        self._front = max(self.view_range, self.merge_range) + V + 1
        self._far_pos: Tuple[Optional[int], ...] = tuple(
            (r.length - self._front - 1) if r.length - self._front - 1 >= self._back else None
            for r in roads
        )

        # Velocities reachable from v in one step, for any action and outcome.
        self._vel_succ: Tuple[FrozenSet[int], ...] = tuple(
            frozenset(
                min(max(v + o.accel, net.min_vel), net.max_vel)
                for outcomes in net.accel_actions for o in outcomes
            )
            for v in range(V + 1)
        )

        self._geom_cache: Dict[Tuple[int, int, bool], _Geometry] = {}
        self._hidden_cache: Dict[EgoCentricState, Tuple[CarState, ...]] = {}
        self._pair_cache: Dict[Tuple[Tuple, Tuple], bool] = {}
        self._label_cache: Dict[EgoCentricState, StateSafety] = {}
        self._abstract_cache: Dict[Tuple, EgoCentricState] = {}

    # ------------------------------------------------------------------ interface

    def get_cache_key(self) -> str:
        params = (self.min_dist, self.merge_space, self.view_range, self.merge_range,
                  self.approach_range, self.shadowing)
        digest = hashlib.sha256(repr(params).encode()).hexdigest()[:16]
        return f'{type(self).__qualname__}(v2):{_network_cache_key(self._system)}:{digest}'

    def get_init_states(self) -> List[EgoCentricState]:
        s = self._system.get_init_state()
        return list(dict.fromkeys(self.normalize(s, c.id) for c in self._system.cars))

    def get_action_count(self, state: EgoCentricState) -> int:
        return len(self._system.accel_actions)

    def normalize(self, state: RoadNetworkState, agent: int) -> EgoCentricState:
        ego = state.cars[agent]
        others = [c for c in state.cars if c.id != agent]
        return self._abstract(ego, others)

    def safety(self) -> Callable[[EgoCentricState], StateSafety]:
        """The safety labelling of abstract states (``road_safety`` on the tracked cars)."""
        def func(state: EgoCentricState) -> StateSafety:
            return self.label(state)
        func.shield_cache_key = f'egocentric_road_safety(v2,min_dist={self.min_dist},merge_space={self.merge_space})'
        return func

    def label(self, state: EgoCentricState) -> StateSafety:
        verdict = self._label_cache.get(state)
        if verdict is None:
            verdict = self._safety(state.as_road_network_state())
            self._label_cache[state] = verdict
        return verdict

    # ------------------------------------------------------------------ geometry

    def _geometry(self, road: int, pos: int, just_merged: bool) -> _Geometry:
        key = (road, pos, just_merged)
        g = self._geom_cache.get(key)
        if g is not None:
            return g

        roads = self._system.roads
        R = self.view_range
        g = _Geometry()
        ego_len = roads[road].length

        # Distances to the start of every road (Dijkstra over road ends).
        dstart = [None] * len(roads)
        pq = []
        for end in {roads[road].end_left, roads[road].end_right}:
            heapq.heappush(pq, (ego_len - pos, end))
        while pq:
            d, r = heapq.heappop(pq)
            if dstart[r] is not None:
                continue
            dstart[r] = d
            seg = roads[r]
            for end in {seg.end_left, seg.end_right}:
                if dstart[end] is None:
                    heapq.heappush(pq, (d + seg.length, end))

        segs: List[Tuple[int, Tuple[int, int]]] = []
        for q in range(pos, min(ego_len, pos + R + 1)):
            g.cell_dist[(road, q)] = q - pos
            g.cell_seg[(road, q)] = SEG_AHEAD
        segs.append((0, SEG_AHEAD))
        for r, d in enumerate(dstart):
            if d is None or d > R:
                continue
            if r == road:
                key_r = SEG_WRAP
                upper = min(pos, R - d + 1)
            else:
                key_r = (r, 1)
                upper = min(roads[r].length, R - d + 1)
            if upper <= 0:
                continue
            for q in range(upper):
                g.cell_dist[(r, q)] = d + q
                g.cell_seg[(r, q)] = key_r
            segs.append((d, key_r))
        segs.sort()
        g.seg_order = [k for _, k in segs]

        for _, key_r in segs:
            if key_r == SEG_AHEAD:
                continue
            r = road if key_r == SEG_WRAP else key_r[0]
            preds = []
            for w in self._preds[r]:
                end_cell = (w, roads[w].length - 1)
                if end_cell in g.cell_seg:
                    preds.append(g.cell_seg[end_cell])
            g.seg_preds[key_r] = tuple(preds)

        # Approach zones of merges where the ego has to yield.
        for y, m in self._yield_into.items():
            if y == road:
                line = ego_len - pos
            elif dstart[y] is not None:
                line = dstart[y] + roads[y].length
            else:
                continue
            if line <= self.merge_range:
                p = roads[m].start
                g.zones[p] = max(0, roads[p].length - self.approach_range)
        if just_merged and len(self._preds[road]) == 2:
            p = roads[road].start
            g.zones[p] = max(0, roads[p].length - self.approach_range)

        if just_merged:
            g.behind = frozenset((road, q) for q in range(max(0, pos - self.merge_space + 1), pos))
        # Cars within min_dist behind the ego only exist after somebody violated their clause
        # (or the ego passed through a car); they are tracked to label those states exactly.
        # Cars on a yield road behind the ego are separated by the yield line and never count.
        close_behind: Set[Cell] = {(road, q) for q in range(max(0, pos - self.min_dist + 1), pos)}
        for w in self._preds[road]:
            if self._yield_into.get(w) == road:
                continue
            w_len = roads[w].length
            close_behind.update((w, w_len - j) for j in range(1, self.min_dist - pos) if w_len - j >= 0)
        g.close_behind = frozenset(close_behind)

        slot = set(g.cell_dist)
        for p, qmin in g.zones.items():
            slot.update((p, q) for q in range(qmin, roads[p].length))
        slot.update(g.behind)
        g.slot_cells = frozenset(slot)

        self._geom_cache[key] = g
        return g

    def _shadowed(self, g: _Geometry, blocked: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        shadowed: Set[Tuple[int, int]] = set()
        if not self.shadowing:
            return shadowed
        for key_r in g.seg_order:
            if key_r == SEG_AHEAD:
                continue
            preds = g.seg_preds[key_r]
            if preds and all(p in blocked or p in shadowed for p in preds):
                shadowed.add(key_r)
        return shadowed

    # ------------------------------------------------------------------ abstraction

    def _select(self, ego: CarState, others: Iterable[CarState], g: _Geometry) -> List[CarState]:
        """The tracked cars among ``others`` (concrete positions, ego frame unchanged)."""
        md = self.min_dist
        keep: Dict[Tuple, CarState] = {}
        nearest: Dict[Tuple[int, int], Tuple[int, List[CarState]]] = {}
        zone_front: Dict[int, Tuple[int, List[CarState]]] = {}
        for c in others:
            cell = (c.road, c.pos)
            d = g.cell_dist.get(cell)
            if d is not None:
                if d < md:
                    keep[_key(c)] = c
                seg = g.cell_seg[cell]
                cur = nearest.get(seg)
                if cur is None or d < cur[0]:
                    nearest[seg] = (d, [c])
                elif d == cur[0]:
                    cur[1].append(c)
            if g.in_zone(c.road, c.pos):
                cur = zone_front.get(c.road)
                if cur is None or c.pos > cur[0]:
                    zone_front[c.road] = (c.pos, [c])
                elif c.pos == cur[0]:
                    cur[1].append(c)
            if cell in g.behind or cell in g.close_behind:
                keep[_key(c)] = c
        shadowed = self._shadowed(g, set(nearest))
        for seg, (_, cs) in nearest.items():
            if seg not in shadowed:
                for c in cs:
                    keep[_key(c)] = c
        for _, cs in zone_front.values():
            for c in cs:
                keep[_key(c)] = c
        return list(keep.values())

    def _abstract(self, ego: CarState, others: Iterable[CarState]) -> EgoCentricState:
        others = list(others)
        memo_key = (ego.road, ego.pos, ego.vel, ego.just_merged, frozenset(others))
        res = self._abstract_cache.get(memo_key)
        if res is None:
            res = self._abstract_uncached(ego, others)
            if len(self._abstract_cache) > 300_000:
                self._abstract_cache.clear()
            self._abstract_cache[memo_key] = res
        return res

    def _abstract_uncached(self, ego: CarState, others: List[CarState]) -> EgoCentricState:
        g = self._geometry(ego.road, ego.pos, ego.just_merged)
        tracked = self._select(ego, others, g)
        roads = self._system.roads

        far = False
        far_pos = self._far_pos[ego.road]
        ego_pos = ego.pos
        shift = 0
        if far_pos is not None and self._back <= ego.pos <= far_pos:
            far = True
            shift = far_pos - ego.pos
            ego_pos = far_pos

        sym = roads[ego.road].symmetric_to
        use_sym = sym is not None and sym < ego.road

        def conv(c: CarState, i: int) -> CarState:
            pos = c.pos + shift if c.road == ego.road else c.pos
            road = roads[c.road].symmetric_to if use_sym else c.road
            return CarState(i, road, pos, c.vel, c.just_merged, False)

        abs_ego = conv(CarState(0, ego.road, ego.pos, ego.vel, ego.just_merged, False), 0)
        abs_ego = CarState(0, abs_ego.road, ego_pos, ego.vel, ego.just_merged, False)
        rest = sorted((conv(c, 0) for c in tracked), key=_key)
        cars = (abs_ego,) + tuple(
            CarState(i + 1, c.road, c.pos, c.vel, c.just_merged, False) for i, c in enumerate(rest)
        )
        return EgoCentricState(far, cars)

    # ------------------------------------------------------------------ hidden cars

    def _pair_legal(self, a: CarState, b: CarState) -> bool:
        """Whether two cars can coexist without either violating its clause."""
        key = (_key(a), _key(b))
        res = self._pair_cache.get(key)
        if res is None:
            s = RoadNetworkState((CarState(0, a.road, a.pos, a.vel, a.just_merged, False),
                                  CarState(1, b.road, b.pos, b.vel, b.just_merged, False)))
            res = self._safety(s) == StateSafety.SAFE
            self._pair_cache[key] = res
        return res

    def _back_cells(self, cell: Cell, steps: int, out: Set[Cell]) -> None:
        """All cells from which ``cell`` is reachable by moving 0..steps cells forward."""
        road, pos = cell
        out.add(cell)
        for j in range(1, steps + 1):
            if pos - j >= 0:
                out.add((road, pos - j))
            else:
                rem = steps - j
                for w in self._preds[road]:
                    self._back_cells((w, self._system.roads[w].length - 1), rem, out)
                return

    def _hidden_moves(self, state: EgoCentricState) -> Tuple[CarState, ...]:
        """Every (post-step) placement of a hidden car that may become visible this step.

        Returned as CarStates (id -1) with the post-step road, position, velocity and
        ``just_merged`` flag, in the frame of ``state`` (before any far shifting).
        """
        # Synthesis asks for all actions of a state in a row, so remembering the last
        # few states is enough (and keeps memory flat).
        cached = self._hidden_cache.get(state)
        if cached is not None:
            return cached
        if len(self._hidden_cache) >= 64:
            self._hidden_cache.clear()

        net = self._system
        roads = net.roads
        V = net.max_vel
        ego = state.cars[0]
        tracked = state.cars[1:]
        g = self._geometry(ego.road, ego.pos, ego.just_merged)

        # Cells where we know there is no hidden car.
        known: Set[Cell] = {(c.road, c.pos) for c in state.cars}
        md = self.min_dist
        nearest: Dict[Tuple[int, int], int] = {}
        for c in tracked:
            d = g.cell_dist.get((c.road, c.pos))
            if d is not None:
                seg = g.cell_seg[(c.road, c.pos)]
                if seg not in nearest or d < nearest[seg]:
                    nearest[seg] = d
        shadowed = self._shadowed(g, set(nearest))
        for cell, d in g.cell_dist.items():
            seg = g.cell_seg[cell]
            if d < md:
                known.add(cell)
            elif seg not in shadowed and d < nearest.get(seg, d + 1):
                known.add(cell)
        for p, qmin in g.zones.items():
            front = max((c.pos for c in tracked if c.road == p and c.pos >= qmin), default=qmin - 1)
            known.update((p, q) for q in range(front + 1, roads[p].length))
        known.update(g.behind)
        known.update(g.close_behind)

        # Cells that could be visible after the step.
        targets: Set[Cell] = set()
        for v in range(V + 1):
            for _, r, p, jm in net._advance(ego.road, ego.pos, v):
                targets |= self._geometry(r, p, jm).slot_cells
        sources: Set[Cell] = set()
        for cell in targets:
            self._back_cells(cell, V, sources)

        moves: Dict[Tuple, CarState] = {}
        for (r, p) in sources:
            if (r, p) in known:
                continue
            jm_options = (False, True) if (p < V and len(self._preds[r]) == 2) else (False,)
            for v_pre in range(V + 1):
                for jm in jm_options:
                    hidden = CarState(-1, r, p, v_pre, jm, False)
                    if not all(self._pair_legal(c, hidden) for c in state.cars):
                        continue
                    for v_post in self._vel_succ[v_pre]:
                        for _, r2, p2, jm2 in net._advance(r, p, v_post):
                            c2 = CarState(-1, r2, p2, v_post, jm2, False)
                            moves[_key(c2)] = c2
        result = tuple(moves.values())
        self._hidden_cache[state] = result
        return result

    # ------------------------------------------------------------------ successors

    def _car_outcomes(self, c: CarState, actions: Iterable[int]) -> Tuple[CarState, ...]:
        """Every next state of a single car under the given actions (all outcomes, all branches)."""
        net = self._system
        out: Dict[Tuple, CarState] = {}
        for a in actions:
            for o in net.accel_actions[a]:
                vel = min(max(c.vel + o.accel, net.min_vel), net.max_vel)
                for _, road, pos, jm in net._advance(c.road, c.pos, vel):
                    c2 = CarState(c.id, road, pos, vel, jm, False)
                    out[_key(c2)] = c2
        return tuple(out.values())

    def get_abstract_successor_states(self, state: EgoCentricState, action: int) -> Tuple[EgoCentricState, ...]:
        """Abstract successors of ``state`` under ego ``action``.

        If the action can lead to a state where the ego violated its clause, only one
        such (UNSAFE) successor is returned: a single unsafe successor already makes
        the action unsafe, so the shield never uses the others.
        """
        n_actions = len(self._system.accel_actions)
        ego_outcomes = self._car_outcomes(state.cars[0], (action,))
        other_outcomes = list(product(*(self._car_outcomes(c, range(n_actions)) for c in state.cars[1:])))

        # Cheap pass: without reveals.  Ego violations are monotone in the set of cars.
        for ego in ego_outcomes:
            for moved in other_outcomes:
                for b in self._abstract_post(state, ego, list(moved)):
                    if self.label(b) == StateSafety.UNSAFE:
                        return (b,)

        hidden = self._hidden_moves(state)
        results: Set[EgoCentricState] = set()
        for ego in ego_outcomes:
            g = self._geometry(ego.road, ego.pos, ego.just_merged)
            cands = self._classify_reveals(ego, hidden)
            # Post-states that only differ in cars that are not tracked afterwards have
            # the same expansions; expand each tracked subset once.
            done: Set[FrozenSet[CarState]] = set()
            for moved in other_outcomes:
                tracked = frozenset(self._select(ego, moved, g))
                if tracked in done:
                    continue
                done.add(tracked)
                unsafe = self._expand_post(state, ego, tuple(tracked), cands, results)
                if unsafe is not None:
                    return (unsafe,)
        return tuple(results)

    def _abstract_post(self, state: EgoCentricState, ego: CarState, others: List[CarState]) -> List[EgoCentricState]:
        if state.far:
            return self._far_variants(state, ego, others)
        return [self._abstract(ego, others)]

    def _classify_reveals(self, ego: CarState, hidden: Tuple[CarState, ...]) -> List[Tuple]:
        """Hidden placements that land in a slot of the ego's view, with their slot data:
        ``(car, cell, dist or None, segment or None, zone road or None, in behind slot)``."""
        g = self._geometry(ego.road, ego.pos, ego.just_merged)
        out = []
        for h in hidden:
            cell = (h.road, h.pos)
            if cell not in g.slot_cells:
                continue
            d = g.cell_dist.get(cell)
            seg = g.cell_seg[cell] if d is not None else None
            qmin = g.zones.get(h.road)
            zone = h.road if qmin is not None and h.pos >= qmin else None
            out.append((h, cell, d, seg, zone, cell in g.behind))
        return out

    def _expand_post(self, state: EgoCentricState, ego: CarState, moved: Tuple[CarState, ...],
                     cands: List[Tuple], results: Set[EgoCentricState]) -> Optional[EgoCentricState]:
        """Add the abstractions of the post-state with every combination of reveals to ``results``.

        Returns an UNSAFE successor as soon as one is found.
        """
        g = self._geometry(ego.road, ego.pos, ego.just_merged)
        md = self.min_dist

        # Current nearest per segment / frontmost per zone among the moved tracked cars.
        nearest: Dict[Tuple[int, int], int] = {}
        zone_front: Dict[int, int] = {}
        occupied = {(ego.road, ego.pos)}
        for c in moved:
            cell = (c.road, c.pos)
            occupied.add(cell)
            d = g.cell_dist.get(cell)
            if d is not None:
                seg = g.cell_seg[cell]
                if seg not in nearest or d < nearest[seg]:
                    nearest[seg] = d
            qmin = g.zones.get(c.road)
            if qmin is not None and c.pos >= qmin and c.pos > zone_front.get(c.road, -1):
                zone_front[c.road] = c.pos
        shadowed = self._shadowed(g, set(nearest)) if nearest else set()

        # Group reveal candidates by the slot they would occupy.
        groups: Dict[Tuple, List[CarState]] = {}
        for h, cell, d, seg, zone, behind in cands:
            if cell in occupied:
                continue
            if d is not None:
                if d < md:
                    groups.setdefault(('close',), []).append(h)
                elif seg not in shadowed and d < nearest.get(seg, d + 1):
                    groups.setdefault(seg, []).append(h)
            if zone is not None and h.pos > zone_front.get(zone, -1):
                groups.setdefault(('zone', zone), []).append(h)
            if behind:
                groups.setdefault(('behind',), []).append(h)

        moved = list(moved)
        options = [[None] + c for c in groups.values()]
        for choice in product(*options):
            extra = [h for h in choice if h is not None]
            for succ in self._abstract_post(state, ego, moved + extra):
                if succ not in results:
                    verdict = self.label(succ)
                    if verdict == StateSafety.UNSAFE:
                        return succ
                    if verdict == StateSafety.SAFE_RELEASE and self.drop_release_successors:
                        # Somebody else violated their clause first: the release property
                        # holds on every trace through this state, so it constrains nothing.
                        continue
                    results.add(succ)
        return None

    def _far_variants(self, state: EgoCentricState, ego: CarState, others: List[CarState]) -> List[EgoCentricState]:
        """The ego was anywhere in the far zone; it may now be anywhere in [far_pos, ego.pos]."""
        start = state.cars[0].pos
        assert ego.road == state.cars[0].road
        out = []
        for back in range(0, ego.pos - start + 1):
            e = CarState(0, ego.road, ego.pos - back, ego.vel, ego.just_merged, False)
            os = [CarState(c.id, c.road, c.pos - back, c.vel, c.just_merged, False) if c.road == ego.road else c
                  for c in others]
            out.append(self._abstract(e, os))
        return out


def _key(c: CarState) -> Tuple[int, int, int, bool]:
    return (c.road, c.pos, c.vel, c.just_merged)
