"""
mobility_demo.py

Console smoke-test / demo for Phase 4 (MANET Mobility).

Builds the existing static district network, wraps it in a NetworkState,
spins up the default disaster-response mobile fleet (ambulances, fire
trucks, police vehicles, rescue teams, medical teams, drones), and then
manually drives N simulation ticks -- printing each mobile node's position,
battery, mission, and current temporary MANET connections per tick.

This script does not modify, subclass, or monkey-patch any existing
Phase 1-3 file. It only *reads* the static topology (via NetworkState's
public accessors) to anchor and route the mobile fleet.

Integrating with SimulationClock
---------------------------------
Phase 4 mobility is designed to be driven either standalone (as shown
below) or wired into the existing SimulationClock without touching
simulation_clock.py, via its public hook-registration API:

    from simulation.simulation_clock import SimulationClock

    clock = SimulationClock(
        network_state=state,
        event_scheduler=scheduler,
        disaster_engine=disaster_engine,   # existing Phase 3 component
        network_updater=updater,           # existing Phase 3 component
    )
    clock.register_hook("after_tick", mobility_manager.update)
    clock.run()

SimulationClock's hook dispatcher calls
`callback(network_state, current_tick=<tick>)`, which exactly matches
`MobilityManager.update`'s signature -- no adapter code required.

Author: Simulation Framework Designer
License: MIT
"""

import logging
import sys

from communication.graph import build_network
from simulation.network_state import NetworkState, SimulationConfig
from simulation.Mobility.mobility_manager import build_default_fleet, FleetConfig
from simulation.Mobility.mobile_node import MobileNodeStatus

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

TOTAL_DEMO_TICKS = 5
SEPARATOR = "-" * 60


def print_tick_report(tick: int, manager) -> None:
    print(f"\nTick {tick}")
    print(SEPARATOR)
    for node in manager.get_all_nodes():
        pos_str = f"({node.x:.1f}, {node.y:.1f})"
        dest = node.destination_position
        dest_str = f"({dest[0]:.1f}, {dest[1]:.1f})" if dest else "None"
        connections = sorted(node.connected_nodes) or ["-"]
        towers = sorted(node.connected_towers) or ["-"]

        print(f"{node.name} [{node.id}] ({node.node_type.value})")
        print(f"  Position          : {pos_str}")
        print(f"  Destination       : {dest_str}")
        print(f"  Battery           : {node.battery_level:.1f}%")
        print(f"  Status            : {node.status.value}")
        print(f"  Mission           : {node.mission}")
        print(f"  Connected Nodes   : {', '.join(connections)}")
        if node.node_type.value == "Drone":
            print(f"  Connected Towers  : {', '.join(towers)}")
        print(SEPARATOR)


def main() -> int:
    graph = build_network()
    state = NetworkState(graph, config=SimulationConfig())

    manager = build_default_fleet(state, config=FleetConfig())

    print(f"Phase 4 MANET Mobility Demo -> {len(manager)} mobile nodes registered.")

    for tick in range(1, TOTAL_DEMO_TICKS + 1):
        manager.update(network_state=state, current_tick=tick)
        print_tick_report(tick, manager)

    offline_count = sum(1 for n in manager.get_all_nodes() if n.status == MobileNodeStatus.OFFLINE)
    print(f"\nDemo complete -> {TOTAL_DEMO_TICKS} tick(s) simulated, {offline_count} node(s) offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())