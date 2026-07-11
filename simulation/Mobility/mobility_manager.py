"""
mobility_manager.py

Provides the central coordination layer (`MobilityManager`) for Phase 4
(MANET Mobility) of the AI-Assisted QoS-Aware MANET Framework for Disaster
Communication.

Responsibilities:
    * Register / remove MobileNode instances (see mobile_node.py).
    * Drive per-tick movement using each node's assigned mobility model
      (see mobility_models.py).
    * Recompute a *temporary* MANET connectivity graph every tick, based
      purely on communication_range vs. distance -- both between mobile
      nodes, and between mobile nodes and the existing static district
      infrastructure (towers, hospitals, HQs, relief camps, villages).
    * Expose read-only snapshots for visualization/telemetry consumers.

Design guarantees (per Phase 4 spec):
    * NO routing, QoS optimization, packet transmission, traffic generation,
      congestion prediction, or AI logic lives here -- only mobility and
      distance-based temporary connectivity.
    * The permanent district topology (communication/graph.py,
      simulation/network_state.py) is NEVER mutated by this module. Static
      node coordinates are only *read* (via NetworkState's public,
      read-only accessors) so mobile fleets can be anchored at real
      infrastructure (hospitals, fire stations, towers, relief camps) and
      so temporary links to that infrastructure can be evaluated. The
      temporary MANET graph is a wholly separate, independent structure.

Author: Simulation Framework Designer
License: MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx

from .mobile_node import (
    MobileNode,
    MobileNodeStatus,
    MobileNodeType,
)

from .mobility_models import (
    CircularMobility,
    MissionBasedMobility,
    MobilityModel,
    PatrolMobility,
    RandomWaypointMobility,
    StationaryMobility,
)

# NetworkState is only used for its public, read-only accessors
# (node_exists / get_node) to source static infrastructure coordinates.
# This import creates no coupling to graph mutation -- MobilityManager
# never calls any NetworkState mutator.
try:
    from simulation.network_state import NetworkState
except ImportError:  # pragma: no cover - defensive: mobility must not hard-fail
    NetworkState = Any  # type: ignore

logger = logging.getLogger("DisasterMANET.MobilityManager")

Coordinate = Tuple[float, float]


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class MobilityManagerException(Exception):
    """Base exception for all MobilityManager runtime anomalies."""
    pass


class DuplicateMobileNodeError(MobilityManagerException):
    """Raised when attempting to register a mobile node id that already exists."""
    pass


class MobileNodeNotFoundError(MobilityManagerException):
    """Raised when an operation references an unregistered mobile node id."""
    pass


class StaticAnchorNotFoundError(MobilityManagerException):
    """Raised when default-fleet construction cannot locate a required static anchor node."""
    pass


# ============================================================================
# FLEET CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class FleetConfig:
    """Declarative counts for the default disaster-response mobile fleet."""
    ambulances: int = 5
    fire_trucks: int = 3
    police_vehicles: int = 4
    rescue_teams: int = 6
    medical_teams: int = 4
    drones: int = 3


DEFAULT_SPEED_BY_TYPE: Dict[MobileNodeType, float] = {
    MobileNodeType.AMBULANCE: 8.0,
    MobileNodeType.FIRE_TRUCK: 7.0,
    MobileNodeType.POLICE_VEHICLE: 9.0,
    MobileNodeType.RESCUE_TEAM: 5.0,
    MobileNodeType.MEDICAL_TEAM: 6.0,
    MobileNodeType.DRONE: 14.0,
    MobileNodeType.VOLUNTEER_VEHICLE: 6.0,
}

DEFAULT_RANGE_BY_TYPE: Dict[MobileNodeType, float] = {
    MobileNodeType.AMBULANCE: 100.0,
    MobileNodeType.FIRE_TRUCK: 100.0,
    MobileNodeType.POLICE_VEHICLE: 110.0,
    MobileNodeType.RESCUE_TEAM: 90.0,
    MobileNodeType.MEDICAL_TEAM: 90.0,
    MobileNodeType.DRONE: 160.0,
    MobileNodeType.VOLUNTEER_VEHICLE: 80.0,
}


# ============================================================================
# CENTRAL MOBILITY MANAGER
# ============================================================================

class MobilityManager:
    """
    Owns and advances the full population of MobileNode instances, and
    maintains a temporary, independent MANET connectivity graph rebuilt
    from scratch every tick.
    """

    def __init__(self, network_state: Optional["NetworkState"] = None) -> None:
        """
        Args:
            network_state: Optional reference to the simulation's
                NetworkState. When provided, MobilityManager will (a) read
                static infrastructure coordinates via its public accessors
                to build/anchor default fleets and evaluate mobile-to-static
                connectivity, and (b) never call any of its mutating
                methods. May be left as None to run mobility fully
                standalone (e.g. in unit tests).
        """
        self._network_state = network_state
        self._nodes: Dict[str, MobileNode] = {}
        # Independent, temporary MANET topology -- never merged into the
        # permanent district graph.
        self._manet_graph: nx.Graph = nx.Graph()
        self._tick_count: int = 0

        logger.info("MobilityManager initialized (network_state attached=%s).", network_state is not None)

    # ------------------------------------------------------------------------
    # REGISTRATION
    # ------------------------------------------------------------------------

    def register_node(self, node: MobileNode) -> None:
        """Registers a new mobile node for tracking and per-tick updates."""
        if node.id in self._nodes:
            raise DuplicateMobileNodeError(f"MobileNode id '{node.id}' is already registered.")
        self._nodes[node.id] = node
        self._manet_graph.add_node(node.id, kind="mobile")
        logger.info(f"MobileNode registered -> '{node.id}' ({node.node_type.value})")

    def remove_node(self, node_id: str) -> MobileNode:
        """Deregisters and returns a mobile node, purging it from the MANET graph."""
        if node_id not in self._nodes:
            raise MobileNodeNotFoundError(f"Cannot remove unknown MobileNode id '{node_id}'.")
        node = self._nodes.pop(node_id)
        if self._manet_graph.has_node(node_id):
            self._manet_graph.remove_node(node_id)
        logger.info(f"MobileNode removed -> '{node_id}'")
        return node

    def get_node(self, node_id: str) -> MobileNode:
        """Fetches a registered mobile node by id."""
        if node_id not in self._nodes:
            raise MobileNodeNotFoundError(f"MobileNode id '{node_id}' is not registered.")
        return self._nodes[node_id]

    def get_all_nodes(self) -> List[MobileNode]:
        """Returns all currently registered mobile nodes."""
        return list(self._nodes.values())

    def __len__(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------------
    # PER-TICK UPDATE PIPELINE
    # ------------------------------------------------------------------------

    def update(self, network_state: Optional["NetworkState"] = None, current_tick: Optional[int] = None, dt: float = 1.0) -> None:
        """
        Advances every registered mobile node by one simulation step and
        recomputes temporary MANET connectivity.

        This method's signature intentionally matches SimulationClock's
        hook-callback convention (`callback(network_state, current_tick=...)`)
        so it can be wired in externally via:

            clock.register_hook("after_tick", mobility_manager.update)

        without requiring any modification to simulation_clock.py itself.

        Args:
            network_state: Optional NetworkState snapshot passed by the
                SimulationClock hook dispatcher; falls back to the instance
                bound at construction time if omitted.
            current_tick: The current absolute simulation tick, for logging.
            dt: Time delta (in ticks) to advance; defaults to a single tick.
        """
        state = network_state if network_state is not None else self._network_state
        self._tick_count += 1
        tick_label = current_tick if current_tick is not None else self._tick_count

        try:
            self._move_all_nodes(dt)
            self._recompute_connectivity(state)
        except Exception as exc:  # fail-soft: mobility must never crash the clock loop
            logger.error(f"MobilityManager.update failed at tick {tick_label}: {exc}", exc_info=True)
            return

        logger.debug(f"MobilityManager tick {tick_label} complete -> {len(self._nodes)} mobile node(s) processed.")

    def _move_all_nodes(self, dt: float) -> None:
        """Advances every registered node's position and battery for this tick."""
        for node in self._nodes.values():
            node.move(dt)

    def _recompute_connectivity(self, state: Optional["NetworkState"]) -> None:
        """
        Rebuilds the temporary MANET graph from scratch for this tick:
        mobile-to-mobile links and mobile-to-static-infrastructure links,
        purely on a distance-vs-communication_range basis.
        """
        # Rebuild fresh -- links are strictly temporary/per-tick, never
        # accumulated or merged into the permanent topology.
        self._manet_graph = nx.Graph()
        for node in self._nodes.values():
            self._manet_graph.add_node(node.id, kind="mobile")
            node.connected_nodes.clear()
            node.connected_towers.clear()

        node_list = list(self._nodes.values())

        # Mobile <-> Mobile connectivity
        for i, node_a in enumerate(node_list):
            if node_a.status == MobileNodeStatus.OFFLINE:
                continue
            for node_b in node_list[i + 1:]:
                if node_b.status == MobileNodeStatus.OFFLINE:
                    continue
                distance = node_a.distance_to(node_b)
                if distance <= min(node_a.communication_range, node_b.communication_range):
                    node_a.connect(node_b.id)
                    node_b.connect(node_a.id)
                    self._manet_graph.add_edge(node_a.id, node_b.id, distance=distance, kind="mobile-mobile")

        # Mobile <-> Static infrastructure connectivity (read-only lookups)
        if state is not None:
            static_positions = self._get_static_node_positions(state)
            for node in node_list:
                if node.status == MobileNodeStatus.OFFLINE:
                    continue
                for static_id, (sx, sy) in static_positions.items():
                    distance = math.dist(node.current_position, (sx, sy))
                    if distance <= node.communication_range:
                        is_tower = static_id.startswith("T")
                        node.connect(static_id, is_tower=is_tower)
                        if not self._manet_graph.has_node(static_id):
                            self._manet_graph.add_node(static_id, kind="static")
                        self._manet_graph.add_edge(node.id, static_id, distance=distance, kind="mobile-static")

    @staticmethod
    def _get_static_node_positions(state: "NetworkState") -> Dict[str, Coordinate]:
        """
        Reads (never mutates) coordinates for every static district node
        via NetworkState's public graph accessor.
        """
        positions: Dict[str, Coordinate] = {}
        try:
            graph = state.graph
            for node_id, attrs in graph.nodes(data=True):
                x = attrs.get("x")
                y = attrs.get("y")
                if x is not None and y is not None:
                    positions[node_id] = (float(x), float(y))
        except Exception as exc:  # defensive: never let a bad state break mobility
            logger.warning(f"Unable to read static node positions from NetworkState: {exc}")
        return positions

    # ------------------------------------------------------------------------
    # VISUALIZATION / TELEMETRY SUPPORT
    # ------------------------------------------------------------------------

    def get_manet_graph_copy(self) -> nx.Graph:
        """Returns a deep-copied snapshot of the current temporary MANET graph."""
        return self._manet_graph.copy()

    def get_snapshot(self) -> List[Dict[str, Any]]:
        """Returns a plain-dict snapshot of every mobile node, for dashboards/consoles."""
        return [node.as_dict() for node in self._nodes.values()]

    def get_nodes_by_type(self, node_type: MobileNodeType) -> List[MobileNode]:
        """Filters registered mobile nodes by their MobileNodeType."""
        return [n for n in self._nodes.values() if n.node_type == node_type]


# ============================================================================
# DEFAULT FLEET FACTORY
# ============================================================================

def _require_static_position(state: "NetworkState", node_id: str) -> Coordinate:
    """Reads a required static anchor's (x, y) position, raising if missing."""
    if not state.node_exists(node_id):
        raise StaticAnchorNotFoundError(f"Required static anchor node '{node_id}' not found in NetworkState.")
    data = state.get_node(node_id)
    x, y = data.get("x"), data.get("y")
    if x is None or y is None:
        raise StaticAnchorNotFoundError(f"Static anchor node '{node_id}' is missing x/y coordinates.")
    return (float(x), float(y))


def build_default_fleet(
    network_state: "NetworkState",
    config: Optional[FleetConfig] = None,
) -> MobilityManager:
    """
    Constructs a MobilityManager pre-populated with the default
    disaster-response fleet, anchored at real static infrastructure read
    from `network_state` (hospitals, fire station, police HQ, relief
    camps, towers), with mission destinations drawn from the village nodes.

    Args:
        network_state: The simulation's NetworkState, used read-only to
            source anchor and destination coordinates.
        config: Optional FleetConfig overriding default fleet counts.

    Returns:
        A fully populated MobilityManager, ready to be driven by
        SimulationClock via `clock.register_hook("after_tick", manager.update)`.
    """
    cfg = config or FleetConfig()
    manager = MobilityManager(network_state=network_state)

    hospitals = ["H1", "H2"]
    fire_station = "F1"
    police_hq = "P1"
    relief_camps = ["R1", "R2"]
    towers = ["T1", "T2", "T3", "T4", "T5"]
    villages = ["V1", "V2", "V3", "V4", "V5", "V6"]

    village_positions = [_require_static_position(network_state, v) for v in villages]

    def _pick_village(idx: int) -> Coordinate:
        return village_positions[idx % len(village_positions)]

    # --- Ambulances: Hospital -> Village -> Hospital -------------------------
    for i in range(cfg.ambulances):
        home = hospitals[i % len(hospitals)]
        home_pos = _require_static_position(network_state, home)
        model = MissionBasedMobility(
            waypoints=[home_pos, _pick_village(i)],
            mission_labels=[f"Staged at {home}", "Responding to village casualty"],
        )
        node = MobileNode(
            id=f"A{i + 1}",
            name=f"Ambulance A{i + 1}",
            node_type=MobileNodeType.AMBULANCE,
            current_position=home_pos,
            speed=DEFAULT_SPEED_BY_TYPE[MobileNodeType.AMBULANCE],
            communication_range=DEFAULT_RANGE_BY_TYPE[MobileNodeType.AMBULANCE],
            mobility_model=model,
            assigned_zone=home,
        )
        manager.register_node(node)

    # --- Fire Trucks: Fire Station -> Incident -> Fire Station --------------
    station_pos = _require_static_position(network_state, fire_station)
    for i in range(cfg.fire_trucks):
        model = MissionBasedMobility(
            waypoints=[station_pos, _pick_village(i + 1)],
            mission_labels=[f"Staged at {fire_station}", "Responding to incident"],
        )
        node = MobileNode(
            id=f"FT{i + 1}",
            name=f"Fire Truck FT{i + 1}",
            node_type=MobileNodeType.FIRE_TRUCK,
            current_position=station_pos,
            speed=DEFAULT_SPEED_BY_TYPE[MobileNodeType.FIRE_TRUCK],
            communication_range=DEFAULT_RANGE_BY_TYPE[MobileNodeType.FIRE_TRUCK],
            mobility_model=model,
            assigned_zone=fire_station,
        )
        manager.register_node(node)

    # --- Police Vehicles: Police HQ patrol loop through villages ------------
    hq_pos = _require_static_position(network_state, police_hq)
    for i in range(cfg.police_vehicles):
        patrol_route = [hq_pos] + village_positions[i % len(village_positions):] + village_positions[:i % len(village_positions)]
        model = PatrolMobility(patrol_points=patrol_route if len(patrol_route) >= 2 else [hq_pos, village_positions[0]])
        node = MobileNode(
            id=f"PV{i + 1}",
            name=f"Police Vehicle PV{i + 1}",
            node_type=MobileNodeType.POLICE_VEHICLE,
            current_position=hq_pos,
            speed=DEFAULT_SPEED_BY_TYPE[MobileNodeType.POLICE_VEHICLE],
            communication_range=DEFAULT_RANGE_BY_TYPE[MobileNodeType.POLICE_VEHICLE],
            mobility_model=model,
            assigned_zone=police_hq,
        )
        manager.register_node(node)

    # --- Rescue Teams: Relief Camp -> Village -> Relief Camp ----------------
    for i in range(cfg.rescue_teams):
        camp = relief_camps[i % len(relief_camps)]
        camp_pos = _require_static_position(network_state, camp)
        model = MissionBasedMobility(
            waypoints=[camp_pos, _pick_village(i + 2)],
            mission_labels=[f"Staged at {camp}", "Rescue operation in village"],
        )
        node = MobileNode(
            id=f"RT{i + 1}",
            name=f"Rescue Team RT{i + 1}",
            node_type=MobileNodeType.RESCUE_TEAM,
            current_position=camp_pos,
            speed=DEFAULT_SPEED_BY_TYPE[MobileNodeType.RESCUE_TEAM],
            communication_range=DEFAULT_RANGE_BY_TYPE[MobileNodeType.RESCUE_TEAM],
            mobility_model=model,
            assigned_zone=camp,
        )
        manager.register_node(node)

    # --- Medical Teams: Hospital -> Village -> Hospital ---------------------
    for i in range(cfg.medical_teams):
        home = hospitals[i % len(hospitals)]
        home_pos = _require_static_position(network_state, home)
        model = MissionBasedMobility(
            waypoints=[home_pos, _pick_village(i + 3)],
            mission_labels=[f"Staged at {home}", "Field triage in village"],
        )
        node = MobileNode(
            id=f"MT{i + 1}",
            name=f"Medical Team MT{i + 1}",
            node_type=MobileNodeType.MEDICAL_TEAM,
            current_position=home_pos,
            speed=DEFAULT_SPEED_BY_TYPE[MobileNodeType.MEDICAL_TEAM],
            communication_range=DEFAULT_RANGE_BY_TYPE[MobileNodeType.MEDICAL_TEAM],
            mobility_model=model,
            assigned_zone=home,
        )
        manager.register_node(node)

    # --- Drones: Tower -> Village -> Tower -----------------------------------
    for i in range(cfg.drones):
        tower = towers[i % len(towers)]
        tower_pos = _require_static_position(network_state, tower)
        model = MissionBasedMobility(
            waypoints=[tower_pos, _pick_village(i + 4)],
            mission_labels=[f"Docked at {tower}", "Aerial survey over village"],
        )
        node = MobileNode(
            id=f"D{i + 1}",
            name=f"Drone D{i + 1}",
            node_type=MobileNodeType.DRONE,
            current_position=tower_pos,
            speed=DEFAULT_SPEED_BY_TYPE[MobileNodeType.DRONE],
            communication_range=DEFAULT_RANGE_BY_TYPE[MobileNodeType.DRONE],
            battery_drain_per_tick=0.15,
            mobility_model=model,
            assigned_zone=tower,
        )
        manager.register_node(node)

    logger.info(
        f"Default fleet constructed -> {cfg.ambulances} Ambulance(s), {cfg.fire_trucks} Fire Truck(s), "
        f"{cfg.police_vehicles} Police Vehicle(s), {cfg.rescue_teams} Rescue Team(s), "
        f"{cfg.medical_teams} Medical Team(s), {cfg.drones} Drone(s) -> total {len(manager)} mobile nodes."
    )
    return manager