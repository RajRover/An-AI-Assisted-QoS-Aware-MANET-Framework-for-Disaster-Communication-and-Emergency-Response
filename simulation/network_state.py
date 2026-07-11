"""
network_state.py

Provides the centralized state manager (`NetworkState`) for the AI-Assisted QoS-Aware
MANET Framework for Disaster Communication. This module implements structural state-tracking,
discrete-event boundaries, historical logging registries, and transactional snapshots.

Author: Simulation Framework Designer
License: MIT
"""

import copy
import logging
import heapq
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx

# Configure Module Logger
logger = logging.getLogger("DisasterMANET.NetworkState")


# ============================================================================
# ENUMS & CUSTOM CONFIGURATION DATACLASSES
# ============================================================================

class SimulationStatus(Enum):
    """Defines explicit states for the simulation control loop boundary."""
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"


@dataclass
class SimulationConfig:
    """Encapsulates static baseline simulation configurations and structural constraints."""
    total_duration_ticks: int = 1000
    tick_delta_seconds: float = 1.0
    max_nodes: int = 100
    coverage_grid_bounds: Tuple[float, float] = (1000.0, 1000.0)
    qos_monitoring_interval: int = 5
    ai_prediction_interval: int = 10
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CUSTOM SIMULATION EXCEPTIONS
# ============================================================================

class NetworkStateException(Exception):
    """Base exception for all state management anomalies."""
    pass


class NodeNotFoundError(NetworkStateException):
    """Raised when an operation references a missing node."""
    pass


class EdgeNotFoundError(NetworkStateException):
    """Raised when an operation references a missing edge."""
    pass


class InvalidStateOperationError(NetworkStateException):
    """Raised when a state modification breaches logical runtime state constraints."""
    pass


# ============================================================================
# DATACLASS DEFINITIONS FOR SUPPORT STRUCTURES
# ============================================================================

@dataclass(order=True)
class SimulationEvent:
    """Represents a scheduled discrete event within the execution queue timeline."""
    scheduled_tick: int = field(compare=True)                # Absolute tick when the event executes
    event_type: str = field(compare=False)                   # Identifier for engine parsing
    priority: int = field(default=10)                       # Tie-breaker priority (lower = higher priority)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)  # Extensible parameter dictionary


@dataclass
class DisasterToken:
    """Stores metadata regarding an active, resolved, or historic disaster track."""
    disaster_id: str
    name: str
    disaster_type: str                                       # e.g., "Flood", "Earthquake"
    epicenter_coords: Tuple[float, float]                    # (x, y) coordinates
    impact_radius: float                                    # Operational range of effect
    severity: float                                         # Normalized impact vector [0.0, 1.0]
    affected_nodes: List[str] = field(default_factory=list)  # Tracking identifiers for nodes inside the impact zone
    affected_links: List[Tuple[str, str]] = field(default_factory=list)  # Tracking links degraded/severed by disaster
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# CENTRAL NETWORK STATE MANAGER
# ============================================================================

class NetworkState:
    """
    The Single Source of Truth managing the runtime state of the MANET framework.
    Encapsulates topological baselines, dynamic states, discrete events, and 
    system telemetry while enforcing isolated access control hooks.
    """

    def __init__(self, initial_graph: nx.Graph, config: Optional[SimulationConfig] = None):
        """
        Initializes the state manager, establishing the baseline network topology.

        Args:
            initial_graph (nx.Graph): Network structure derived from build_network().
            config (Optional[SimulationConfig]): Global structured configuration dataclass.
        """
        if not isinstance(initial_graph, nx.Graph):
            raise TypeError("Initial graph must be an instance of networkx.Graph")

        # Isolation Layer: Immutable Base Graph vs. Dynamic Live Graph
        self._base_graph: nx.Graph = copy.deepcopy(initial_graph)
        self._graph: nx.Graph = copy.deepcopy(initial_graph)

        # Simulation Clock & State Control
        self._current_tick: int = 0
        self._simulation_seconds: float = 0.0
        self._status: SimulationStatus = SimulationStatus.STOPPED

        # Enhanced State Registries
        self._active_disasters: Dict[str, DisasterToken] = {}
        self._resolved_disasters: Dict[str, DisasterToken] = {}
        self._disaster_history: List[Dict[str, Any]] = []
        self._mobile_nodes: Dict[str, Dict[str, Any]] = {}
        self._event_queue: List[SimulationEvent] = []

        # Advanced Analytical/AI Predictive Telemetry Registries
        self._statistics: Dict[str, Any] = {
            "network_metrics": {
                "average_latency": 0.0,
                "average_bandwidth": 0.0,
                "network_load": 0.0,
                "average_packet_loss": 0.0,
                "network_utilization": 0.0,
                "congested_links": [],
                "failed_nodes": [],
                "failed_links": [],
                "connected_components": 0,
                "isolated_nodes": [],
                "active_mobile_nodes": 0,
                "active_routes": [],
                "average_hops": 0.0
            },
            "simulation_metrics": {
                "events_processed": 0,
                "ticks_advanced": 0,
                "total_failures_logged": 0
            }
        }

        # Snapshot Registers
        self._snapshots: Dict[str, Dict[str, Any]] = {}

        # Configuration Dataclass Registry
        self._configuration: SimulationConfig = config if config is not None else SimulationConfig()
        logger.info("NetworkState container successfully initialized.")

    # ------------------------------------------------------------------------
    # SAFE GRAPH EXPOSURE INTERFACES (IMMUTABILITY HOOKS)
    # ------------------------------------------------------------------------

    @property
    def graph(self) -> nx.Graph:
        """
        Provides a direct reference to the active NetworkX graph instance.
        Modules may use this to run performance analytics sweeps or metrics extractions.
        """
        return self._graph

    def get_graph_copy(self) -> nx.Graph:
        """Returns an isolated, deep copied duplicate of the live network topology."""
        return copy.deepcopy(self._graph)

    @property
    def current_tick(self) -> int:
        return self._current_tick

    @property
    def simulation_seconds(self) -> float:
        return self._simulation_seconds

    @property
    def status(self) -> SimulationStatus:
        return self._status

    @property
    def mobile_nodes(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._mobile_nodes)

    @property
    def configuration(self) -> SimulationConfig:
        return self._configuration

    @property
    def global_metrics(self) -> Dict[str, Any]:
        """Exposes the internal network performance metrics."""
        metrics = self._statistics["network_metrics"]
        if "hospital_utilization_avg" in metrics:
            metrics["hospital_utilization_rate"] = metrics["hospital_utilization_avg"]
        return metrics

    # ------------------------------------------------------------------------
    # NODE OPERATIONS
    # ------------------------------------------------------------------------

    def get_node(self, node_id: str) -> Dict[str, Any]:
        if not self._graph.has_node(node_id):
            raise NodeNotFoundError(f"Node '{node_id}' does not exist.")
        return dict(self._graph.nodes[node_id])

    def update_node(self, node_id: str, **attributes) -> None:
        if not self._graph.has_node(node_id):
            raise NodeNotFoundError(f"Node '{node_id}' missing; cannot update.")
        
        self._graph.nodes[node_id].update(attributes)
        
        # Keep tracking registry dynamically synchronized for routing/mobility phases
        if attributes.get("is_mobile") or attributes.get("type") in ["Rescue", "Drone", "MobileNode"]:
            if node_id not in self._mobile_nodes:
                self._mobile_nodes[node_id] = {"x": 0.0, "y": 0.0, "speed": 0.0, "battery": 100.0, "mission": "Standby"}
            self._mobile_nodes[node_id].update(attributes)
            self._statistics["network_metrics"]["active_mobile_nodes"] = len(self._mobile_nodes)

    def remove_node(self, node_id: str) -> Dict[str, Any]:
        if not self._graph.has_node(node_id):
            raise NodeNotFoundError(f"Cannot extract node '{node_id}'; missing from state matrix.")
        
        node_data = dict(self._graph.nodes[node_id])
        self._graph.remove_node(node_id)
        self._mobile_nodes.pop(node_id, None)
        
        self._statistics["simulation_metrics"]["total_failures_logged"] += 1
        if node_id not in self._statistics["network_metrics"]["failed_nodes"]:
            self._statistics["network_metrics"]["failed_nodes"].append(node_id)
        self._statistics["network_metrics"]["active_mobile_nodes"] = len(self._mobile_nodes)
            
        logger.warning(f"Node '{node_id}' removed from live topology runtime.")
        return node_data

    def restore_node(self, node_id: str, node_data: Optional[Dict[str, Any]] = None) -> None:
        """Restores a node, defaulting to pristine values from _base_graph if none are passed."""
        if node_data is None:
            if not self._base_graph.has_node(node_id):
                raise NodeNotFoundError(f"Node '{node_id}' not found in configuration baseline.")
            node_data = dict(self._base_graph.nodes[node_id])
            
        self._graph.add_node(node_id, **node_data)
        if node_id in self._statistics["network_metrics"]["failed_nodes"]:
            self._statistics["network_metrics"]["failed_nodes"].remove(node_id)
        logger.info(f"Node '{node_id}' successfully restored to live framework topology.")

    # ------------------------------------------------------------------------
    # EDGE OPERATIONS
    # ------------------------------------------------------------------------

    def get_edge(self, node1: str, node2: str) -> Dict[str, Any]:
        if not self._graph.has_edge(node1, node2):
            raise EdgeNotFoundError(f"No direct link discovered joining '{node1}' and '{node2}'.")
        return dict(self._graph.edges[node1, node2])

    def update_edge(self, node1: str, node2: str, **attributes) -> None:
        if not self._graph.has_edge(node1, node2):
            raise EdgeNotFoundError(f"Link ('{node1}', '{node2}') does not exist; cannot mutate.")
        self._graph.edges[node1, node2].update(attributes)

    def remove_link(self, node1: str, node2: str) -> Dict[str, Any]:
        if not self._graph.has_edge(node1, node2):
            raise EdgeNotFoundError(f"Link ('{node1}', '{node2}') does not exist.")
        edge_data = dict(self._graph.edges[node1, node2])
        self._graph.remove_edge(node1, node2)
        
        link_tuple = (node1, node2)
        if link_tuple not in self._statistics["network_metrics"]["failed_links"]:
            self._statistics["network_metrics"]["failed_links"].append(link_tuple)
            
        return edge_data

    def restore_link(self, node1: str, node2: str, edge_data: Optional[Dict[str, Any]] = None) -> None:
        """Restores an edge, defaulting to pristine parameters from _base_graph if none are provided."""
        if not (self._graph.has_node(node1) and self._graph.has_node(node2)):
            raise NodeNotFoundError("Link registration error: Vertices must exist to establish connection.")
        
        if edge_data is None:
            if not self._base_graph.has_edge(node1, node2):
                raise EdgeNotFoundError(f"Edge ('{node1}', '{node2}') missing from baseline schema configuration.")
            edge_data = dict(self._base_graph.edges[node1, node2])
            
        self._graph.add_edge(node1, node2, **edge_data)
        
        for edge_entry in [tuple((node1, node2)), tuple((node2, node1))]:
            if edge_entry in self._statistics["network_metrics"]["failed_links"]:
                self._statistics["network_metrics"]["failed_links"].remove(edge_entry)

    # ------------------------------------------------------------------------
    # GRAPH METADATA QUERYING OPERATIONS
    # ------------------------------------------------------------------------

    def get_neighbors(self, node: str) -> List[str]:
        if not self._graph.has_node(node):
            raise NodeNotFoundError(f"Node '{node}' missing from topology reference maps.")
        return list(self._graph.neighbors(node))

    def node_exists(self, node: str) -> bool:
        return self._graph.has_node(node)

    def edge_exists(self, node1: str, node2: str) -> bool:
        return self._graph.has_edge(node1, node2)

    def find_shortest_path(self, source: str, destination: str, weight: Optional[str] = None) -> List[str]:
        if not (self.node_exists(source) and self.node_exists(destination)):
            raise NodeNotFoundError("Topological coordinates incomplete; validation failed.")
        try:
            return nx.shortest_path(self._graph, source=source, target=destination, weight=weight)
        except nx.NetworkXNoPath:
            return []

    # Maps DestinationType enum names  →  NodeType.value strings stored on graph nodes.
    # DestinationType lives in services/message.py; NodeType lives in communication/nodes.py.
    # They use different naming conventions, so we normalise here at the boundary.
    _DESTINATION_TYPE_MAP: Dict[str, List[str]] = {
        "CONTROL_CENTER":  ["ControlCentre"],
        "HOSPITAL":        ["Hospital"],
        "POLICE_STATION":  ["PoliceHQ"],
        "FIRE_STATION":    ["FireStation"],
        "RELIEF_CAMP":     ["ReliefCamp"],
        "FIELD_TEAM":      ["Village", "Tower"],   # closest field-level proxies
    }

    def get_nodes_by_type(self, type_str: str) -> List[str]:
        """Returns all node IDs whose 'type' attribute matches the given DestinationType name.

        Translates DestinationType enum names (e.g. CONTROL_CENTER) to the
        NodeType.value strings actually stored on graph nodes (e.g. ControlCentre).
        Falls back to a direct case-insensitive comparison if the key is unknown.
        """
        # Resolve canonical NodeType values for the requested destination type
        target_values = self._DESTINATION_TYPE_MAP.get(type_str.upper())

        result = []
        for node_id, attrs in self._graph.nodes(data=True):
            node_type = attrs.get("type", "")
            if not isinstance(node_type, str):
                continue
            if target_values:
                if node_type in target_values:
                    result.append(node_id)
            else:
                # Unknown type_str — fall back to case-insensitive direct match
                if node_type.upper() == type_str.upper():
                    result.append(node_id)

        if not result:
            available = set(d.get("type", "") for _, d in self._graph.nodes(data=True))
            logger.debug(
                f"get_nodes_by_type: no nodes matched type='{type_str}' "
                f"(resolved targets={target_values}). Available types: {available}"
            )
        return result

    def is_node_operational(self, node: str) -> bool:
        """Returns True if the node exists in the live graph and is not in the failed_nodes list."""
        if not self._graph.has_node(node):
            return False
        return node not in self._statistics["network_metrics"]["failed_nodes"]

    def compute_shortest_path_distance(self, source: str, target: str) -> float:
        """Returns the hop-count distance between two nodes, or infinity if no path exists."""
        if not (self._graph.has_node(source) and self._graph.has_node(target)):
            return float('inf')
        try:
            return float(nx.shortest_path_length(self._graph, source=source, target=target))
        except nx.NetworkXNoPath:
            return float('inf')

    def get_edge_metrics(self, u: str, v: str) -> Dict[str, float]:
        """Returns QoS edge attributes (latency, packet_loss, jitter, bandwidth) with safe defaults."""
        if not self._graph.has_edge(u, v):
            logger.debug(f"get_edge_metrics: edge ({u}, {v}) not found; returning defaults.")
            return {"latency": 1.0, "packet_loss": 0.0, "jitter": 0.0, "bandwidth": 100.0}
        raw = dict(self._graph.edges[u, v])
        return {
            "latency":     float(raw.get("latency",      raw.get("weight", 1.0))),
            "packet_loss": float(raw.get("packet_loss",  raw.get("loss",   0.0))),
            "jitter":      float(raw.get("jitter",       0.0)),
            "bandwidth":   float(raw.get("bandwidth",    raw.get("capacity", 100.0))),
        }

    # ------------------------------------------------------------------------
    # SIMULATION CONTROL INTERFACES
    # ------------------------------------------------------------------------

    def start(self) -> None:
        if self._status == SimulationStatus.RUNNING:
            raise InvalidStateOperationError("Lifecycle error: Simulation is already running.")
        self._status = SimulationStatus.RUNNING
        logger.info("Simulation engine transitioned status: RUNNING.")

    def pause(self) -> None:
        if self._status != SimulationStatus.RUNNING:
            raise InvalidStateOperationError("Lifecycle error: Can only pause an actively running simulation engine.")
        self._status = SimulationStatus.PAUSED
        logger.info("Simulation engine transitioned status: PAUSED.")

    def resume(self) -> None:
        if self._status != SimulationStatus.PAUSED:
            return
        self._status = SimulationStatus.RUNNING
        logger.info("Simulation engine transitioned status: RESUMED.")

    def stop(self) -> None:
        self._status = SimulationStatus.STOPPED
        logger.info("Simulation engine transitioned status: STOPPED.")

    def reset(self) -> None:
        """Completely restores pristine baseline configurations and clears transient data registries."""
        self._graph = copy.deepcopy(self._base_graph)
        self._current_tick = 0
        self._simulation_seconds = 0.0
        self._status = SimulationStatus.STOPPED
        
        self._active_disasters.clear()
        self._resolved_disasters.clear()
        self._disaster_history.clear()
        self._mobile_nodes.clear()
        self._event_queue.clear()
        self._snapshots.clear()
        
        # Reset telemetry metrics blocks to pristine structures
        self._statistics["simulation_metrics"] = {"events_processed": 0, "ticks_advanced": 0, "total_failures_logged": 0}
        self._statistics["network_metrics"] = {
            "average_latency": 0.0, "average_bandwidth": 0.0, "network_load": 0.0, "average_packet_loss": 0.0,
            "network_utilization": 0.0, "congested_links": [], "failed_nodes": [], "failed_links": [],
            "connected_components": 0, "isolated_nodes": [], "active_mobile_nodes": 0, "active_routes": [], "average_hops": 0.0
        }
        
        logger.warning("Simulation space completely cleared and rolled back to baseline configuration.")

    def advance_tick(self, dynamic_step_seconds: Optional[float] = None) -> int:
        if self._status != SimulationStatus.RUNNING:
            raise InvalidStateOperationError("Clock cycle shift rejected: State processing framework is not active.")
        
        step = dynamic_step_seconds if dynamic_step_seconds is not None else self._configuration.tick_delta_seconds
        self._current_tick += 1
        self._simulation_seconds += step
        self._statistics["simulation_metrics"]["ticks_advanced"] += 1
        return self._current_tick

    # ------------------------------------------------------------------------
    # DISASTER STATE SUPPORT
    # ------------------------------------------------------------------------

    def add_disaster(self, disaster: DisasterToken) -> None:
        if disaster.disaster_id in self._active_disasters:
            raise InvalidStateOperationError(f"Collision error: Disaster identification code '{disaster.disaster_id}' exists.")
        self._active_disasters[disaster.disaster_id] = disaster
        
        self._disaster_history.append({
            "disaster_id": disaster.disaster_id,
            "type": disaster.disaster_type,
            "start_tick": self._current_tick,
            "end_tick": None,
            "status": "Impacted",
            "initial_affected_nodes": list(disaster.affected_nodes),
            "initial_affected_links": list(disaster.affected_links)
        })

    def remove_disaster(self, disaster_id: str) -> None:
        """Resolves an active disaster boundary tracking token contextually."""
        if disaster_id not in self._active_disasters:
            raise InvalidStateOperationError(f"Extraction execution failed; identifier code '{disaster_id}' missing.")
        
        token = self._active_disasters.pop(disaster_id)
        self._resolved_disasters[disaster_id] = token
        
        # Log resolution lifecycle boundaries inside historic analysis lists
        for record in self._disaster_history:
            if record["disaster_id"] == disaster_id and record["end_tick"] is None:
                record["end_tick"] = self._current_tick
                record["status"] = "Resolved"
                record["final_affected_nodes"] = list(token.affected_nodes)
                record["final_affected_links"] = list(token.affected_links)
                break

    def get_active_disasters(self) -> Dict[str, DisasterToken]:
        return dict(self._active_disasters)

    def get_resolved_disasters(self) -> Dict[str, DisasterToken]:
        return dict(self._resolved_disasters)

    def get_disaster_history(self) -> List[Dict[str, Any]]:
        return list(self._disaster_history)

    # ------------------------------------------------------------------------
    # DISCRETE EVENT SCHEDULER SUPPORT
    # ------------------------------------------------------------------------

    def add_event(self, event: SimulationEvent) -> None:
        if event.scheduled_tick < self._current_tick:
            raise InvalidStateOperationError("Scheduling conflict: Target event window timestamp exists in past history.")
        heapq.heappush(self._event_queue, event)

    def pop_next_event(self) -> Optional[SimulationEvent]:
        if not self._event_queue:
            return None
        event = heapq.heappop(self._event_queue)
        self._statistics["simulation_metrics"]["events_processed"] += 1
        return event

    def get_due_events(self) -> List[SimulationEvent]:
        """Extracts and drains all simulation items from queue whose schedule window matches current timeline step constraints."""
        due_events = []
        while self._event_queue and self._event_queue[0].scheduled_tick <= self._current_tick:
            event = heapq.heappop(self._event_queue)
            self._statistics["simulation_metrics"]["events_processed"] += 1
            due_events.append(event)
        return due_events

    def clear_events(self) -> None:
        self._event_queue.clear()

    # ------------------------------------------------------------------------
    # STATISTICS LAYER HOOKS
    # ------------------------------------------------------------------------

    def update_network_statistics(self, stats: Dict[str, Any]) -> None:
        self._statistics["network_metrics"].update(stats)

    def get_statistics(self) -> Dict[str, Any]:
        return dict(self._statistics)

    # ------------------------------------------------------------------------
    # CORE TRANSACTUAL SNAPSHOT INTERFACES
    # ------------------------------------------------------------------------

    def create_snapshot(self, snapshot_id: str, reason: str = "Unspecified Simulation Event") -> None:
        """Deep-freezes and structuralizes the total state memory register map."""
        snapshot_payload = {
            "metadata": {
                "snapshot_id": snapshot_id,
                "reason": reason,
                "tick": self._current_tick,
                "simulation_seconds": self._simulation_seconds
            },
            "graph": copy.deepcopy(self._graph),
            "status": self._status,
            "active_disasters": copy.deepcopy(self._active_disasters),
            "resolved_disasters": copy.deepcopy(self._resolved_disasters),
            "disaster_history": copy.deepcopy(self._disaster_history),
            "mobile_nodes": copy.deepcopy(self._mobile_nodes),
            "event_queue": copy.deepcopy(self._event_queue),
            "statistics": copy.deepcopy(self._statistics)
        }
        self._snapshots[snapshot_id] = snapshot_payload
        logger.info(f"State snapshot successfully captured: ID='{snapshot_id}' | Reason='{reason}' [Tick {self._current_tick}]")

    def restore_snapshot(self, snapshot_id: str) -> None:
        """Restores the complete processing timeline topology state back to matching parameters."""
        if snapshot_id not in self._snapshots:
            raise InvalidStateOperationError(f"Snapshot restore failed; record lookup token '{snapshot_id}' not found.")
        
        source = self._snapshots[snapshot_id]
        
        self._graph = copy.deepcopy(source["graph"])
        self._current_tick = source["metadata"]["tick"]
        self._simulation_seconds = source["metadata"]["simulation_seconds"]
        self._status = source["status"]
        self._active_disasters = copy.deepcopy(source["active_disasters"])
        self._resolved_disasters = copy.deepcopy(source["resolved_disasters"])
        self._disaster_history = copy.deepcopy(source["disaster_history"])
        self._mobile_nodes = copy.deepcopy(source["mobile_nodes"])
        self._event_queue = copy.deepcopy(source["event_queue"])
        self._statistics = copy.deepcopy(source["statistics"])
        
        logger.warning(f"State memory space reverted back to checkpoint: '{snapshot_id}' (Reason: {source['metadata']['reason']}).")

    # ------------------------------------------------------------------------
    # FUTURE EXPERIMENTAL STATE EXTERNAL HOOK LOGGERS
    # ------------------------------------------------------------------------

    def log_state(self) -> None:
        """
        Hook placeholder designed to stream internal state details out 
        to dedicated filesystem engines, CSV writers, or monitoring datasets.
        """
        pass