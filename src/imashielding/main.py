import argparse
from dataclasses import dataclass
from typing import Any, Callable, Optional

from imashielding.roads import RoadNetwork
from imashielding.roads_reduction import (
    AbstractRoadNetworkSystem,
    AbstractRoundaboutSystem,
    road_safety,
    roundabout_safety,
)
from imashielding.roads_egocentric import EgoCentricRoadSystem
from imashielding.shield import compute_shield
from imashielding.visualize import visualize


@dataclass(frozen=True)
class Scenario:
    create_network: Callable[[], RoadNetwork]
    # Returns (abstraction, safety property over its states).
    create_abstraction: Callable[[RoadNetwork], Any]
    fps: int
    # The full point-of-view abstraction (all cars), for --full-abstraction.
    create_full_abstraction: Optional[Callable[[RoadNetwork], Any]] = None


def _egocentric(net: RoadNetwork):
    system = EgoCentricRoadSystem(net)
    return system, system.safety()


def _full(net: RoadNetwork):
    return AbstractRoadNetworkSystem(net), road_safety(net)


SCENARIOS = {
    "roundabout": Scenario(
        create_network=lambda: RoadNetwork.create_roundabout_scenario(road_per_circ=200),
        create_abstraction=lambda net: (AbstractRoundaboutSystem(net), roundabout_safety(6)),
        fps=24,
    ),
    "crossing_roads": Scenario(
        create_network=lambda: RoadNetwork.create_crossing_roads_scenario(),
        create_abstraction=_egocentric,
        create_full_abstraction=_full,
        fps=16,
    ),
    "double_roundabout": Scenario(
        create_network=lambda: RoadNetwork.create_double_roundabout_scenario(),
        create_abstraction=_egocentric,
        create_full_abstraction=_full,
        fps=12,
    ),
}


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
    parser.add_argument(
        '--full-abstraction', action='store_true',
        help='Use the full point-of-view abstraction (AbstractRoadNetworkSystem) instead of the '
             'ego-centric one. Synthesis scales exponentially with the number of cars.'
    )
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    net = scenario.create_network()

    if args.no_shield:
        shields = {}
    else:
        if args.full_abstraction:
            if scenario.create_full_abstraction is None:
                parser.error(f'--full-abstraction is not available for {args.scenario}')
            absnet, safety = scenario.create_full_abstraction(net)
        else:
            absnet, safety = scenario.create_abstraction(net)
        dg_shield = compute_shield(absnet, safety, force_recompute=args.resynthesize)
        shields = {
            agent: dg_shield.with_normalizer(lambda state, a=agent: absnet.normalize(state, a))
            for agent in range(len(net.cars))
        }

    visualize(net, steps=0, shields=shields, fps=scenario.fps)


if __name__ == '__main__':
    main()
