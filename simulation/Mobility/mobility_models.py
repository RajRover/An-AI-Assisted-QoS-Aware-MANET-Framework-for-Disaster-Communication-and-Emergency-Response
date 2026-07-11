"""
mobility_models.py

Provides reusable, pluggable mobility model strategies for Phase 4 (MANET
Mobility) of the AI-Assisted QoS-Aware MANET Framework for Disaster
Communication.

Each mobility model implements a single method, `next_position(node, dt)`,
following the Strategy pattern: a `MobileNode` (see mobile_node.py) is
handed a mobility model instance and delegates all "where do I go next?"
decisions to it. This keeps movement logic decoupled from the node entity
itself and lets mobility_manager.py assign different behaviors to different
fleets without special-casing node types.

No routing, QoS, traffic, or AI logic lives here -- only geometric motion.

Author: Simulation Framework Designer
License: MIT
"""

from __future__ import annotations

import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .mobile_node import MobileNode

logger = logging.getLogger("DisasterMANET.MobilityModels")

Coordinate = Tuple[float, float]


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class MobilityModelException(Exception):
    """Base exception for all mobility model configuration/runtime anomalies."""
    pass


class InvalidMobilityModelConfigError(MobilityModelException):
    """Raised when a mobility model is constructed with illegal parameters."""
    pass


# ============================================================================
# SHARED GEOMETRIC HELPERS
# ============================================================================

def _step_towards(current: Coordinate, target: Coordinate, max_step: float) -> Coordinate:
    """
    Computes the point reached after moving from `current` toward `target`
    by at most `max_step` distance units. Snaps exactly onto `target` if
    the remaining distance is smaller than the requested step.
    """
    remaining = math.dist(current, target)
    if remaining <= max_step or remaining <= 1e-9:
        return target

    ratio = max_step / remaining
    new_x = current[0] + (target[0] - current[0]) * ratio
    new_y = current[1] + (target[1] - current[1]) * ratio
    return (new_x, new_y)


# ============================================================================
# ABSTRACT BASE MOBILITY MODEL
# ============================================================================

class MobilityModel(ABC):
    """
    Abstract base for all mobility strategies. Concrete subclasses must
    implement `next_position`, which is called exactly once per tick per
    node by `MobileNode.move()`.
    """

    @abstractmethod
    def next_position(self, node: "MobileNode", dt: float = 1.0) -> Coordinate:
        """
        Computes the node's next (x, y) coordinate for this tick.

        Args:
            node: The MobileNode being advanced. Implementations may read
                and write `node.mobility_state` for per-node bookkeeping
                (e.g. a waypoint index), and may read `node.speed`,
                `node.current_position`, and `node.destination_position`.
            dt: Time delta (in ticks) to advance.

        Returns:
            The computed (x, y) coordinate for this node after this tick.
        """
        raise NotImplementedError


# ============================================================================
# RANDOM WAYPOINT MOBILITY
# ============================================================================

@dataclass
class RandomWaypointMobility(MobilityModel):
    """
    Classic random waypoint model: whenever a node has no destination (or
    has just arrived at one), a new random destination is picked uniformly
    within the configured rectangular bounds, and the node moves toward it
    at its own speed.
    """

    bounds_min: Coordinate = (0.0, 0.0)
    bounds_max: Coordinate = (500.0, 500.0)
    pause_ticks_at_waypoint: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        if self.bounds_min[0] >= self.bounds_max[0] or self.bounds_min[1] >= self.bounds_max[1]:
            raise InvalidMobilityModelConfigError("RandomWaypointMobility: bounds_min must be strictly less than bounds_max.")

    def _random_point(self) -> Coordinate:
        return (
            self.rng.uniform(self.bounds_min[0], self.bounds_max[0]),
            self.rng.uniform(self.bounds_min[1], self.bounds_max[1]),
        )

    def next_position(self, node: "MobileNode", dt: float = 1.0) -> Coordinate:
        pause_remaining = node.mobility_state.get("pause_remaining", 0)
        if pause_remaining > 0:
            node.mobility_state["pause_remaining"] = pause_remaining - 1
            return node.current_position

        if node.destination_position is None:
            node.destination_position = self._random_point()

        step = node.speed * dt
        new_position = _step_towards(node.current_position, node.destination_position, step)

        if new_position == node.destination_position:
            node.mobility_state["pause_remaining"] = self.pause_ticks_at_waypoint
            node.destination_position = None

        return new_position


# ============================================================================
# MISSION-BASED MOBILITY
# ============================================================================

@dataclass
class MissionBasedMobility(MobilityModel):
    """
    Drives a node through an ordered sequence of mission waypoints, looping
    back to the start once the final waypoint is reached. This is the
    primary model for emergency-response fleets, e.g.:

        Ambulance:   Hospital -> Village -> Hospital
        Fire Truck:  Fire Station -> Incident -> Fire Station
        Drone:       Tower -> Village -> Tower
        Rescue Team: Relief Camp -> Village -> Relief Camp

    Waypoints may optionally carry a mission label, surfaced onto the node
    via `assign_mission()` whenever a new leg begins.
    """

    waypoints: List[Coordinate]
    mission_labels: Optional[List[str]] = None
    loop: bool = True

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise InvalidMobilityModelConfigError("MissionBasedMobility requires at least 2 waypoints.")
        if self.mission_labels is not None and len(self.mission_labels) != len(self.waypoints):
            raise InvalidMobilityModelConfigError("mission_labels length must match waypoints length.")

    def next_position(self, node: "MobileNode", dt: float = 1.0) -> Coordinate:
        leg_index = node.mobility_state.get("leg_index", 0)

        if node.destination_position is None or node.status.value == "Arrived":
            leg_index = (leg_index + 1) % len(self.waypoints) if node.mobility_state.get("started", False) else leg_index
            node.mobility_state["started"] = True
            node.mobility_state["leg_index"] = leg_index

            destination = self.waypoints[leg_index]
            label = self.mission_labels[leg_index] if self.mission_labels else f"Leg {leg_index + 1}/{len(self.waypoints)}"
            node.assign_mission(mission=label, destination=destination)

            if not self.loop and leg_index == len(self.waypoints) - 1:
                logger.debug(f"MobileNode '{node.id}' reached final non-looping mission leg.")

        step = node.speed * dt
        return _step_towards(node.current_position, node.destination_position, step)


# ============================================================================
# PATROL MOBILITY
# ============================================================================

@dataclass
class PatrolMobility(MobilityModel):
    """
    Continuously cycles a node through a fixed loop of patrol points, with
    no mission-completion semantics (unlike MissionBasedMobility). Suited
    for police vehicles or perimeter-watch drones.
    """

    patrol_points: List[Coordinate]

    def __post_init__(self) -> None:
        if len(self.patrol_points) < 2:
            raise InvalidMobilityModelConfigError("PatrolMobility requires at least 2 patrol points.")

    def next_position(self, node: "MobileNode", dt: float = 1.0) -> Coordinate:
        point_index = node.mobility_state.get("patrol_index", 0)

        if node.destination_position is None:
            node.destination_position = self.patrol_points[point_index]

        step = node.speed * dt
        new_position = _step_towards(node.current_position, node.destination_position, step)

        if new_position == node.destination_position:
            point_index = (point_index + 1) % len(self.patrol_points)
            node.mobility_state["patrol_index"] = point_index
            node.destination_position = self.patrol_points[point_index]

        return new_position


# ============================================================================
# CIRCULAR MOBILITY
# ============================================================================

@dataclass
class CircularMobility(MobilityModel):
    """
    Moves a node along a fixed circular orbit around a center point, at a
    constant angular speed. Useful for surveillance drones holding station
    over a village or incident zone.
    """

    center: Coordinate
    radius: float
    angular_speed_deg_per_tick: float = 10.0

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise InvalidMobilityModelConfigError("CircularMobility: radius must be non-negative.")

    def next_position(self, node: "MobileNode", dt: float = 1.0) -> Coordinate:
        angle_deg = node.mobility_state.get("orbit_angle_deg", 0.0)
        angle_deg = (angle_deg + self.angular_speed_deg_per_tick * dt) % 360.0
        node.mobility_state["orbit_angle_deg"] = angle_deg

        angle_rad = math.radians(angle_deg)
        new_x = self.center[0] + self.radius * math.cos(angle_rad)
        new_y = self.center[1] + self.radius * math.sin(angle_rad)
        return (new_x, new_y)


# ============================================================================
# STATIONARY MOBILITY
# ============================================================================

@dataclass
class StationaryMobility(MobilityModel):
    """
    A no-op mobility model: the node holds its current position indefinitely.
    Useful for a mobile asset that is temporarily parked/standing by
    (e.g. an ambulance staged at a relief camp awaiting dispatch).
    """

    def next_position(self, node: "MobileNode", dt: float = 1.0) -> Coordinate:
        return node.current_position