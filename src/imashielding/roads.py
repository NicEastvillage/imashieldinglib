import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from imashielding.systems import ChancedState, ConcurrentGame
from imashielding.shapes import Shape, Arc, Straight


@dataclass(frozen=True, slots=True)
class CarState:
    id: int
    road: int
    pos: int
    vel: int
    just_merged: bool
    braking: bool


@dataclass(frozen=True, slots=True)
class RoadNetworkState:
    cars: Tuple[CarState, ...]


@dataclass(frozen=True, slots=True)
class CarDef:
    id: int
    init_road: int
    init_pos: int
    init_vel: int = 0


@dataclass(frozen=True, slots=True)
class RoadSegment:
    id: int
    length: int
    end_left: int
    end_right: int
    start: int
    symmetric_to: Optional[int] = None
    shape: Optional[Shape] = None


@dataclass(frozen=True, slots=True)
class ChancedOutcomeAccel:
    chance: float
    accel: int


def _accel_actions(high_acceleration_chance: float) -> Tuple[Tuple[ChancedOutcomeAccel, ...], ...]:
    """Build the acceleration outcomes with a chance of doubled accel/decel."""
    assert 0.0 <= high_acceleration_chance <= 1.0
    normal_chance = 1.0 - high_acceleration_chance
    outcomes = (
        (
            (high_acceleration_chance, -2),
            (normal_chance, -1),
        ),
        ((1.0, 0),),
        (
            (normal_chance, 1),
            (high_acceleration_chance, 2),
        ),
    )
    # Drop impossible outcomes so they cannot affect synthesis or sampling.
    return tuple(
        tuple(ChancedOutcomeAccel(chance, accel) for chance, accel in actions if chance > 0.0)
        for actions in outcomes
    )


class RoadNetwork(ConcurrentGame[RoadNetworkState]):
    def __init__(self, roads: Tuple[RoadSegment, ...], cars: Tuple[CarDef, ...], max_vel: int, min_vel: int, accel_actions: Tuple[Tuple[ChancedOutcomeAccel, ...], ...]):
        self.roads: Tuple[RoadSegment, ...] = roads
        self.cars: Tuple[CarDef, ...] = cars
        self.max_vel = max_vel
        self.min_vel = min_vel
        self.accel_actions = accel_actions   # Indexes are actions; Entries are tuples of weighted outcome accelerations

        # Validation
        assert min_vel <= max_vel

        assert len(roads) > 0
        for i, road in enumerate(roads):
            assert road.id == i
            assert 0 < road.length
            assert 0 <= road.start < len(roads)
            assert 0 <= road.end_left < len(roads)
            assert 0 <= road.end_right < len(roads)

        assert len(cars) > 0
        for i, car in enumerate(cars):
            assert car.id == i
            assert 0 <= car.init_road < len(roads)
            assert 0 <= car.init_pos < roads[car.init_road].length
            assert min_vel <= car.init_vel <= max_vel

    @staticmethod
    def create_roundabout_scenario(car_count: int = 6, road_per_circ: int = 32, max_vel: int = 4, high_acceleration_chance: float = 0.2) -> "RoadNetwork":
        road = RoadSegment(0, road_per_circ, 0, 0, 0, None, Arc((0.0, 0.0), 1.0, -math.pi / 2, 2 * math.pi))
        init_dist = road_per_circ / car_count
        cars = tuple(CarDef(i, 0, int(init_dist * i)) for i in range(car_count))
        accel_actions = _accel_actions(high_acceleration_chance)
        return RoadNetwork((road,), cars, max_vel, 0, accel_actions)

    @staticmethod
    def create_double_roundabout_scenario(car_count: int = 5, road_per_circ: int = 50, max_vel: int = 3, high_acceleration_chance: float = 0.2) -> "RoadNetwork":
        """Two roundabouts joined by two connecting roads.

        ``road_per_circ`` is the number of positions in one roundabout, split
        across its two segments.
        """
        road_dist = 2.4
        straight_radius = 3.0
        straight_y_offset = math.sqrt(straight_radius**2 - (road_dist / 2)**2)
        straight_sweep = math.acos(straight_y_offset / straight_radius)
        roads = (
            RoadSegment(0, round(road_per_circ / 2), 1, 1, 1, 3, Arc((0.0, 0.0), 1.0, -math.pi / 2, math.pi)),
            RoadSegment(1, round(road_per_circ / 2), 0, 5, 0, 2, Arc((0.0, 0.0), 1.0, math.pi / 2, math.pi)),
            RoadSegment(2, round(road_per_circ / 2), 3, 4, 3, 1, Arc((road_dist, 0.0), 1.0, -math.pi / 2, math.pi)),
            RoadSegment(3, round(road_per_circ / 2), 2, 2, 2, 0, Arc((road_dist, 0.0), 1.0, math.pi / 2, math.pi)),
            RoadSegment(4, round(road_dist * road_per_circ / math.pi), 1, 1, 2, 5, Arc((road_dist / 2, -straight_y_offset + 1.0), straight_radius, math.pi / 2 - straight_sweep, 2 * straight_sweep)),
            RoadSegment(5, round(road_dist * road_per_circ / math.pi), 2, 2, 1, 4, Arc((road_dist / 2, straight_y_offset - 1.0), straight_radius, -math.pi / 2 - straight_sweep, 2 * straight_sweep)),
        )
        cars = tuple(CarDef(i, i % len(roads), 2 + int(road_per_circ * i / len(roads) / 4)) for i in range(car_count))
        accel_actions = _accel_actions(high_acceleration_chance)
        return RoadNetwork(roads, cars, max_vel, 0, accel_actions)

    @staticmethod
    def create_crossing_roads_scenario(
        car_count: int = 5,
        road_per_circ: int = 50,
        max_vel: int = 3,
        high_acceleration_chance: float = 0.2,
        right_road_scale: float = 0.8,
    ) -> "RoadNetwork":
        """Two looping roads joined by a straight middle road.

        ``road_per_circ`` is the number of positions in the left loop when
        travelled end to end: the left arc plus the shared middle road.
        The right-hand arc is ``right_road_scale`` times as long as the
        left-hand arc.
        """
        assert 0.0 < right_road_scale <= 1.0

        angle = math.pi / 6.0
        radius = round(road_per_circ / math.tau)
        road_length_middle = round(2 * math.sin(angle) * radius)
        road_length_arc = road_per_circ - road_length_middle
        road_length_arc_right = max(1, round(road_length_arc * right_road_scale))

        vis_straight_offset = math.sin(angle)
        vis_arc_center_offset = math.cos(angle)
        left_arc_sweep = math.tau - 2 * angle

        # Both arcs connect the endpoints (0, -vis_straight_offset) and
        # (0, vis_straight_offset).  A circle with radius r through those
        # points covers a major arc of length r * (tau - 2 * asin(offset / r)).
        # Shrink the right circle until its drawn arc is shorter by the same
        # factor as its discrete road length, keeping both junctions fixed.
        right_road_ratio = road_length_arc_right / road_length_arc
        target_right_arc_length = right_road_ratio * left_arc_sweep  # left radius is 1
        assert target_right_arc_length > vis_straight_offset * math.pi  # feasible for r > offset

        lo, hi = vis_straight_offset, 1.0
        for _ in range(64):
            mid = (lo + hi) / 2.0
            arc_length = mid * (math.tau - 2 * math.asin(vis_straight_offset / mid))
            if arc_length < target_right_arc_length:
                lo = mid
            else:
                hi = mid
        right_arc_radius = (lo + hi) / 2.0
        right_arc_center_offset = math.sqrt(right_arc_radius ** 2 - vis_straight_offset ** 2)
        right_arc_start = math.pi - math.asin(vis_straight_offset / right_arc_radius)
        right_arc_sweep = -(math.tau - 2 * math.asin(vis_straight_offset / right_arc_radius))

        roads = (
            RoadSegment(0, road_length_middle, 1, 2, 1, 0, Straight((0.0, -vis_straight_offset), (0.0, vis_straight_offset))),
            RoadSegment(1, road_length_arc, 0, 0, 0, None, Arc((-vis_arc_center_offset, 0.0), 1.0, angle, left_arc_sweep)),
            RoadSegment(2, road_length_arc_right, 0, 0, 0, None, Arc((right_arc_center_offset, 0.0), right_arc_radius, right_arc_start, right_arc_sweep)),
        )
        cars = tuple(
            CarDef(
                i,
                1 + i % 2,
                int((road_length_arc if i % 2 == 0 else road_length_arc_right) * i / float(car_count)),
            )
            for i in range(car_count)
        )
        accel_actions = _accel_actions(high_acceleration_chance)
        return RoadNetwork(roads, cars, max_vel, 0, accel_actions)


    def get_init_state(self) -> RoadNetworkState:
        cars = tuple(CarState(c.id, c.init_road, c.init_pos, c.init_vel, False, False) for c in self.cars)
        return RoadNetworkState(cars)

    def get_successors(self, state: RoadNetworkState, actions: Tuple[int, ...]) -> Tuple[ChancedState[RoadNetworkState], ...]:
        assert len(state.cars) == len(actions)

        # Partial joint successors; cars [0..i) have already been moved
        partials: List[Tuple[float, Tuple[CarState, ...]]] = [(1.0, tuple(state.cars))]
        for i, action in enumerate(actions):
            expanded: Dict[Tuple[CarState, ...], float] = {}
            for chance, cars in partials:
                car = cars[i]
                for outcome in self.accel_actions[action]:
                    vel = min(max(car.vel + outcome.accel, self.min_vel), self.max_vel)
                    for move_chance, road, pos, just_merged in self._advance(car.road, car.pos, vel):
                        new_cars = cars[:i] + (CarState(i, road, pos, vel, just_merged, outcome.accel < 0),) + cars[i + 1:]
                        expanded[new_cars] = expanded.get(new_cars, 0.0) + chance * outcome.chance * move_chance
            partials = [(chance, cars) for cars, chance in expanded.items()]
        return tuple(ChancedState[RoadNetworkState](chance, RoadNetworkState(tuple(cars))) for chance, cars in partials)

    def _advance(self, road: int, pos: int, vel: int) -> Iterator[Tuple[float, int, int, bool]]:
        """Yields (chance, road, pos, just_merged) placements after moving `vel` cells start from (road, pos).

        Completing a segment moves the car onto the right of left exit at random.
        """
        segment = self.roads[road]
        new_pos = pos + vel
        if new_pos < segment.length:
            yield 1.0, road, new_pos, False
            return
        remaining = new_pos - segment.length
        next_segs = ((segment.end_left, 1.0),) if segment.end_left == segment.end_right else ((segment.end_left, 0.5), (segment.end_right, 0.5))
        for next_seg, chance in next_segs:
            just_merged = segment.end_left == segment.end_right and segment.id != self.roads[next_seg].start
            for sub_chance, sub_road, sub_pos, jm in self._advance(next_seg, 0, remaining):
                yield chance * sub_chance, sub_road, sub_pos, just_merged

    def get_agent_action_count(self, state: RoadNetworkState, agent: int) -> int:
        return len(self.accel_actions)
