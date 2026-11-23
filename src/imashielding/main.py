import argparse
from dataclasses import dataclass
from typing import Any, Callable

from imashielding.roads import RoadNetwork
from imashielding.roads_reduction import (
    AbstractRoadNetworkSystem,
    AbstractRoundaboutSystem,
    road_safety,
    roundabout_safety,
)
from imashielding.shield import compute_shield
from imashielding.visualize import visualize


@dataclass(frozen=True)
class Scenario:
    create_network: Callable[[], RoadNetwork]
    create_abstraction: Callable[[RoadNetwork], Any]
    create_safety: Callable[[RoadNetwork], Any]
    fps: int


SCENARIOS = {
    "roundabout": Scenario(
        create_network=lambda: RoadNetwork.create_roundabout_scenario(road_per_circ=200),
        create_abstraction=AbstractRoundaboutSystem,
        create_safety=lambda net: roundabout_safety(6),
        fps=24,
    ),
    "crossing_roads": Scenario(
        create_network=lambda: RoadNetwork.create_crossing_roads_scenario(car_count=2, max_vel=2),
        create_abstraction=AbstractRoadNetworkSystem,
        create_safety=road_safety,
        fps=16,
    ),
    "double_roundabout": Scenario(
        create_network=lambda: RoadNetwork.create_double_roundabout_scenario(car_count=2),
        create_abstraction=AbstractRoadNetworkSystem,
        create_safety=road_safety,
        fps=12,
    ),
}


# Synthesis statistics for the crossing_roads scenario:
# 2 cars: ~36,500 safe states, ~9 s
# 3 cars: ~3,030,000 safe states, ~3,819 s


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a road network scenario, optionally with shields."
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        default="roundabout",
        choices=SCENARIOS.keys(),
        help="Scenario to visualize (default: roundabout).",
    )
    parser.add_argument(
        '--resynthesize', action='store_true',
        help='Ignore the shield cache and synthesize again.'
    )
    parser.add_argument(
        '--no-shield', action='store_true',
        help='Skip shield synthesis and visualize without a shield.'
    )
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    net = scenario.create_network()

    if args.no_shield:
        shields = {}
    else:
        absnet = scenario.create_abstraction(net)
        dg_shield = compute_shield(
            absnet, scenario.create_safety(net), force_recompute=args.resynthesize
        )
        shields = {
            agent: dg_shield.with_normalizer(lambda state, a=agent: absnet.normalize(state, a))
            for agent in range(len(net.cars))
        }

    visualize(net, steps=0, shields=shields, fps=scenario.fps)


if __name__ == '__main__':
    main()
