"""
mobile_node.py

Provides the `MobileNode` entity model for Phase 4 (MANET Mobility) of the
AI-Assisted QoS-Aware MANET Framework for Disaster Communication.

A MobileNode represents any moving physical asset participating in the
disaster-response mesh: an ambulance, fire truck, police vehicle, rescue
team, medical team, drone, or volunteer vehicle. This module owns exactly
one responsibility: describing what a mobile node *is* and how it moves,
connects, and depletes power at a given instant.

This module is intentionally independent from the static district topology
(communication/graph.py, communication/nodes.py) and from the disaster
engine (simulation/disaster_engine.py, simulation/network_state.py,
simulation/network_updater.py). It does not import or mutate them. Higher
layers (mobility_manager.py) are responsible for bridging MobileNode
instances with the rest of the simulation.

Author: Simulation Framework Designer
License: MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for static type checking; avoids a hard runtime
    # dependency / import cycle between mobile_node.py and mobility_models.py.
    from .mobility_models import MobilityModel

logger = logging.getLogger("DisasterMANET.MobileNode")

Coordinate = Tuple[float, float]


# ============================================================================
# ENUMS
# ============================================================================

@unique
class MobileNodeType(str, Enum):
    """Enumerates every category of mobile MANET asset supported in Phase 4."""
    AMBULANCE = "Ambulance"
    FIRE_TRUCK = "FireTruck"
    POLICE_VEHICLE = "PoliceVehicle"
    RESCUE_TEAM = "RescueTeam"
    MEDICAL_TEAM = "MedicalTeam"
    DRONE = "Drone"
    VOLUNTEER_VEHICLE = "VolunteerVehicle"


@unique
class MobileNodeStatus(str, Enum):
    """Defines the discrete operational lifecycle status of a mobile node."""
    IDLE = "Idle"
    EN_ROUTE = "EnRoute"
    ARRIVED = "Arrived"
    ON_MISSION = "OnMission"
    RETURNING = "Returning"
    OFFLINE = "Offline"


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class MobileNodeException(Exception):
    """Base exception for all MobileNode-related runtime anomalies."""
    pass


class InvalidMobilityParameterError(MobileNodeException):
    """Raised when a MobileNode is configured with an illegal parameter value."""
    pass


# ============================================================================
# MOBILE NODE ENTITY
# ============================================================================

@dataclass
class MobileNode:
    """
    Represents a single mobile MANET-participating asset.

    Attributes:
        id: Unique identifier (e.g. "A1", "D2", "RT3").
        name: Human-readable display name (e.g. "Ambulance A1").
        node_type: Category of mobile asset (see MobileNodeType).
        current_position: Live (x, y) coordinate in the same coordinate
            space used by communication/nodes.py.
        destination_position: Target (x, y) coordinate the node is currently
            travelling toward, or None if idle/stationary.
        speed: Distance units travelled per simulation tick.
        communication_range: Maximum radius (same units as coordinates)
            within which this node can form a MANET communication link.
        battery_level: Remaining battery percentage in [0.0, 100.0].
        status: Current lifecycle status (see MobileNodeStatus).
        mission: Free-text description of the current mission/task, if any.
        assigned_zone: Optional named zone/sector this node is assigned to.
        connected_nodes: Set of node ids (static or mobile) this node is
            currently linked to via a temporary MANET connection.
        connected_towers: Subset-style tracking of connected static Tower
            node ids, kept distinct for quick "do I have backhaul?" checks.
        movement_history: Ordered trail of previously visited coordinates.
        mobility_model: Pluggable strategy object (see mobility_models.py)
            responsible for computing this node's next position.
        battery_drain_per_tick: Percentage points of battery consumed per
            tick while moving. Idle/stationary nodes drain at a lower rate.
        mobility_state: Free-form scratch dictionary that mobility models
            may use to store any per-node bookkeeping they need (e.g. a
            waypoint index or patrol angle) without polluting this class
            with model-specific fields.
    """

    id: str
    name: str
    node_type: MobileNodeType
    current_position: Coordinate
    destination_position: Optional[Coordinate] = None
    speed: float = 5.0
    communication_range: float = 120.0
    battery_level: float = 100.0
    status: MobileNodeStatus = MobileNodeStatus.IDLE
    mission: Optional[str] = None
    assigned_zone: Optional[str] = None
    connected_nodes: Set[str] = field(default_factory=set)
    connected_towers: Set[str] = field(default_factory=set)
    movement_history: List[Coordinate] = field(default_factory=list)
    mobility_model: Optional["MobilityModel"] = None
    battery_drain_per_tick: float = 0.05
    idle_battery_drain_per_tick: float = 0.01
    mobility_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.speed < 0:
            raise InvalidMobilityParameterError(f"MobileNode '{self.id}': speed must be non-negative.")
        if self.communication_range < 0:
            raise InvalidMobilityParameterError(f"MobileNode '{self.id}': communication_range must be non-negative.")
        if not (0.0 <= self.battery_level <= 100.0):
            raise InvalidMobilityParameterError(f"MobileNode '{self.id}': battery_level must fall within [0, 100].")
        self.movement_history.append(self.current_position)
        logger.debug(f"MobileNode created -> {self.id} ({self.node_type.value}) @ {self.current_position}")

    # ------------------------------------------------------------------------
    # MOVEMENT
    # ------------------------------------------------------------------------

    def move(self, dt: float = 1.0) -> Coordinate:
        """
        Advances this node by one simulation step using its assigned mobility
        model, then records the resulting position in the movement trail and
        depletes battery accordingly.

        Args:
            dt: Time delta (in ticks) to advance. Defaults to a single tick.

        Returns:
            The node's new current_position.
        """
        if self.status == MobileNodeStatus.OFFLINE:
            logger.debug(f"MobileNode '{self.id}' is OFFLINE; move() skipped.")
            return self.current_position

        previous_position = self.current_position

        if self.mobility_model is not None:
            new_position = self.mobility_model.next_position(self, dt)
        else:
            # No mobility model assigned -> the node is effectively stationary.
            new_position = self.current_position

        self.current_position = new_position
        self.movement_history.append(new_position)

        moved_distance = math.dist(previous_position, new_position)
        if moved_distance > 1e-9:
            if self.status not in (MobileNodeStatus.ON_MISSION,):
                self.status = MobileNodeStatus.EN_ROUTE
            self.update_battery(self.battery_drain_per_tick)
        else:
            self.update_battery(self.idle_battery_drain_per_tick)

        if self.destination_position is not None:
            if math.dist(self.current_position, self.destination_position) <= 1e-6:
                self.arrive_destination()

        return self.current_position

    def distance_to(self, other: "MobileNode | Coordinate") -> float:
        """
        Computes the Euclidean distance from this node to another MobileNode
        or to a raw (x, y) coordinate.
        """
        if isinstance(other, MobileNode):
            target = other.current_position
        else:
            target = other
        return math.dist(self.current_position, target)

    # ------------------------------------------------------------------------
    # CONNECTIVITY
    # ------------------------------------------------------------------------

    def is_in_range(self, other_position: Coordinate, other_range: Optional[float] = None) -> bool:
        """
        Determines whether a target coordinate falls within communication
        reach of this node.

        Args:
            other_position: The (x, y) coordinate of the candidate peer.
            other_range: Optional communication range of the peer. When
                provided, the effective link range is the minimum of the two
                ranges (a link is only as good as its weaker radio); when
                omitted, only this node's own range is considered.
        """
        effective_range = self.communication_range
        if other_range is not None:
            effective_range = min(effective_range, other_range)
        return math.dist(self.current_position, other_position) <= effective_range

    def connect(self, node_id: str, is_tower: bool = False) -> None:
        """Registers a temporary MANET communication link to another node id."""
        self.connected_nodes.add(node_id)
        if is_tower:
            self.connected_towers.add(node_id)
        logger.debug(f"MobileNode '{self.id}' connected -> '{node_id}' (tower={is_tower})")

    def disconnect(self, node_id: str) -> None:
        """Tears down a previously established temporary MANET communication link."""
        self.connected_nodes.discard(node_id)
        self.connected_towers.discard(node_id)
        logger.debug(f"MobileNode '{self.id}' disconnected -> '{node_id}'")

    # ------------------------------------------------------------------------
    # BATTERY
    # ------------------------------------------------------------------------

    def update_battery(self, delta: Optional[float] = None) -> None:
        """
        Depletes (or, with a negative delta, recharges) the battery level.

        Args:
            delta: Percentage points to subtract. Defaults to this node's
                configured `battery_drain_per_tick` if not provided.
        """
        drain = self.battery_drain_per_tick if delta is None else delta
        self.battery_level = max(0.0, min(100.0, self.battery_level - drain))

        if self.battery_level <= 0.0 and self.status != MobileNodeStatus.OFFLINE:
            self.status = MobileNodeStatus.OFFLINE
            self.connected_nodes.clear()
            self.connected_towers.clear()
            logger.warning(f"MobileNode '{self.id}' battery depleted -> transitioned to OFFLINE.")

    # ------------------------------------------------------------------------
    # MISSION LIFECYCLE
    # ------------------------------------------------------------------------

    def assign_mission(
        self,
        mission: str,
        destination: Coordinate,
        assigned_zone: Optional[str] = None,
    ) -> None:
        """
        Assigns a new mission and destination to this node, transitioning it
        out of IDLE/ARRIVED status into active movement.
        """
        self.mission = mission
        self.destination_position = destination
        if assigned_zone is not None:
            self.assigned_zone = assigned_zone
        if self.status != MobileNodeStatus.OFFLINE:
            self.status = MobileNodeStatus.EN_ROUTE
        logger.info(f"MobileNode '{self.id}' assigned mission '{mission}' -> destination {destination}")

    def arrive_destination(self) -> None:
        """
        Marks the node as having reached its current destination. Mobility
        models that manage multi-leg missions (see MissionBasedMobility)
        typically inspect this transition on the next tick to assign the
        next leg of the journey.
        """
        if self.status == MobileNodeStatus.OFFLINE:
            return
        self.status = MobileNodeStatus.ARRIVED
        logger.info(f"MobileNode '{self.id}' arrived at destination {self.current_position} (mission='{self.mission}')")

    # ------------------------------------------------------------------------
    # VISUALIZATION SUPPORT
    # ------------------------------------------------------------------------

    @property
    def x(self) -> float:
        return self.current_position[0]

    @property
    def y(self) -> float:
        return self.current_position[1]

    def as_dict(self) -> Dict[str, Any]:
        """Flattens this node into a plain dict for visualization/telemetry consumers."""
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "x": self.x,
            "y": self.y,
            "current_position": self.current_position,
            "destination_position": self.destination_position,
            "speed": self.speed,
            "communication_range": self.communication_range,
            "battery_level": round(self.battery_level, 2),
            "status": self.status.value,
            "mission": self.mission,
            "assigned_zone": self.assigned_zone,
            "connected_nodes": sorted(self.connected_nodes),
            "connected_towers": sorted(self.connected_towers),
            "movement_trail": list(self.movement_history),
        }