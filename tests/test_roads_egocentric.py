"""Tests for the ego-centric abstraction (``roads_egocentric.py``).

The main check is falsification: on concrete runs, every concrete successor must
be matched by an abstract successor (see ``falsify_egocentric.py``).  Run from
``imashieldinglib`` with::

    python -m unittest tests.test_roads_egocentric
"""

import contextlib
import io
import random
import tempfile
import unittest

from imashielding.falsify_egocentric import cautious_random_policy, falsify
from imashielding.roads import CarState, RoadNetwork, RoadNetworkState
from imashielding.roads_egocentric import EgoCentricRoadSystem, EgoCentricState
from imashielding.roads_reduction import road_safety
from imashielding.shield import compute_shield
from imashielding.systems import StateSafety


def car(id, road, pos, vel=0, just_merged=False):
    return CarState(id, road, pos, vel, just_merged, False)


class MergeRuleTest(unittest.TestCase):
    """The merge rule of road_safety (fixed 2026-11-24)."""

    def setUp(self):
        # Road 0 (length 6) is entered from road 1 (priority) and road 2 (yield).
        self.net = RoadNetwork.create_crossing_roads_scenario(car_count=2, road_per_circ=36, max_vel=4)
        self.safety = road_safety(self.net, min_dist=3, merge_space=8)

    def test_merging_car_checks_priority_road_beyond_min_dist(self):
        # Landed at pos 3 (>= min_dist), priority car 4 cells behind: too close.
        state = RoadNetworkState((car(0, 0, 3, 4, True), car(1, 1, 29, 4)))
        self.assertIs(self.safety(state), StateSafety.UNSAFE)
        # 9 cells behind is fine.
        state = RoadNetworkState((car(0, 0, 3, 4, True), car(1, 1, 24, 4)))
        self.assertIs(self.safety(state), StateSafety.SAFE)

    def test_car_merging_in_the_same_step_is_not_the_front_cars_fault(self):
        # Both crossed the yield line in this step; the rear car is 4 cells behind.
        state = RoadNetworkState((car(0, 0, 4, 2, True), car(1, 0, 0, 4, True)))
        self.assertIs(self.safety(state), StateSafety.SAFE)

    def test_faster_car_is_at_fault_when_both_merged_too_close(self):
        # The car that came from behind moved further (faster) and closed the gap.
        for rear_pos in (0, 1):    # just behind, or caught up to the same cell
            front_view = RoadNetworkState((car(0, 0, 1, 1, True), car(1, 0, rear_pos, 4, True)))
            rear_view = RoadNetworkState((car(0, 0, rear_pos, 4, True), car(1, 0, 1, 1, True)))
            self.assertIs(self.safety(front_view), StateSafety.SAFE_RELEASE)
            self.assertIs(self.safety(rear_view), StateSafety.UNSAFE)


class NormalizeTest(unittest.TestCase):
    def setUp(self):
        self.net = RoadNetwork.create_crossing_roads_scenario(car_count=3, road_per_circ=36, max_vel=4)
        self.system = EgoCentricRoadSystem(self.net)

    def test_far_cars_are_dropped(self):
        # Road 1 has length 30; with view range 13 the car 20 cells ahead is invisible.
        state = RoadNetworkState((car(0, 1, 0), car(1, 1, 20), car(2, 2, 5)))
        abstract = self.system.normalize(state, 0)
        self.assertEqual(len(abstract.cars), 1)

    def test_nearest_car_ahead_is_kept(self):
        state = RoadNetworkState((car(0, 1, 0, 2), car(1, 1, 8, 1), car(2, 1, 12, 3)))
        abstract = self.system.normalize(state, 0)
        # Car 2 is shadowed by car 1 (same road, behind it from the ego's view).
        self.assertEqual([(c.road, c.pos, c.vel) for c in abstract.cars[1:]], [(1, 8, 1)])

    def test_cars_within_min_dist_are_always_kept(self):
        state = RoadNetworkState((car(0, 1, 0, 2), car(1, 1, 1, 4), car(2, 1, 2, 0)))
        abstract = self.system.normalize(state, 0)
        self.assertEqual(len(abstract.cars), 3)

    def test_priority_car_is_tracked_when_approaching_yield_line(self):
        # Road 2 yields to road 1 when merging into road 0.
        state = RoadNetworkState((car(0, 2, 20, 1), car(1, 1, 25, 3), car(2, 1, 10, 3)))
        abstract = self.system.normalize(state, 0)
        self.assertIn((1, 25, 3), [(c.road, c.pos, c.vel) for c in abstract.cars[1:]])
        self.assertNotIn((1, 10, 3), [(c.road, c.pos, c.vel) for c in abstract.cars[1:]])

    def test_labels_agree_with_concrete_safety(self):
        safety = road_safety(self.net)
        unsafe = RoadNetworkState((car(0, 1, 5, 3), car(1, 1, 7, 1)))
        self.assertIs(safety(unsafe), StateSafety.UNSAFE)
        self.assertIs(self.system.label(self.system.normalize(unsafe, 0)), StateSafety.UNSAFE)

    def test_init_states_cover_every_agent(self):
        init = self.system.get_init_states()
        s = self.net.get_init_state()
        for c in s.cars:
            self.assertIn(self.system.normalize(s, c.id), init)


class FalsificationTest(unittest.TestCase):
    def check(self, net, steps):
        system = EgoCentricRoadSystem(net)
        report = falsify(net, system, steps=steps, seed=0,
                         policy=cautious_random_policy(net, 3, 8))
        self.assertTrue(report.ok, str(report))

    def test_crossing_roads_v2(self):
        self.check(RoadNetwork.create_crossing_roads_scenario(car_count=3, road_per_circ=36, max_vel=2), 150)

    def test_crossing_roads_v4(self):
        self.check(RoadNetwork.create_crossing_roads_scenario(car_count=4, road_per_circ=36, max_vel=4), 40)

    def test_double_roundabout_v3(self):
        self.check(RoadNetwork.create_double_roundabout_scenario(car_count=3, road_per_circ=32, max_vel=3), 80)

    def test_default_scenarios(self):
        self.check(RoadNetwork.create_crossing_roads_scenario(), 25)
        self.check(RoadNetwork.create_double_roundabout_scenario(), 25)

    def test_full_point_of_view_normalizes_like_the_concrete_state(self):
        # BenchMARL queries the shield with AbstractRoadNetworkSystem points of view.
        from imashielding.roads_reduction import AbstractRoadNetworkSystem
        net = RoadNetwork.create_double_roundabout_scenario()
        full, ego = AbstractRoadNetworkSystem(net), EgoCentricRoadSystem(net)
        rng = random.Random(0)
        state = net.get_init_state()
        for _ in range(200):
            for c in state.cars:
                self.assertEqual(ego.normalize(state, c.id), ego.normalize(full.normalize(state, c.id), 0))
            outcomes = net.get_successors(state, tuple(rng.choice((0, 1, 2)) for _ in state.cars))
            state = rng.choices([o.state for o in outcomes], [o.chance for o in outcomes])[0]


class SynthesisTest(unittest.TestCase):
    def test_shielded_run_is_safe_for_any_car_count(self):
        net2 = RoadNetwork.create_crossing_roads_scenario(car_count=2, road_per_circ=36, max_vel=2)
        system = EgoCentricRoadSystem(net2)
        with tempfile.TemporaryDirectory() as cache, contextlib.redirect_stdout(io.StringIO()):
            shield = compute_shield(system, system.safety(), cache_dir=cache)

        # The abstraction does not depend on the number of cars, so the same shield
        # works for a network with more cars (as long as their initial views are covered).
        net4 = RoadNetwork.create_crossing_roads_scenario(car_count=4, road_per_circ=36, max_vel=2)
        system4 = EgoCentricRoadSystem(net4)
        safety = road_safety(net4)
        rng = random.Random(0)
        state = net4.get_init_state()
        for _ in range(400):
            actions = []
            for c in state.cars:
                abstract = system4.normalize(state, c.id)
                self.assertTrue(shield.contains(abstract), abstract)
                safe = shield.get_safe_actions(abstract)
                self.assertTrue(safe)
                actions.append(rng.choice(safe))
            outcomes = net4.get_successors(state, tuple(actions))
            state = rng.choices([o.state for o in outcomes], [o.chance for o in outcomes])[0]
            self.assertIs(safety(state), StateSafety.SAFE)


if __name__ == '__main__':
    unittest.main()
