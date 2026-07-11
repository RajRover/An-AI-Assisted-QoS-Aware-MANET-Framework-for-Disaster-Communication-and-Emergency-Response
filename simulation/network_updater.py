"""
network_updater.py

Provides the centralized mutation engine (`NetworkUpdater`) for the AI-Assisted
QoS-Aware MANET Framework for Disaster Communication. Enforces atomic attribute 
modifications, computes ongoing telemetry analytics, and executes structural recovery 
transitions using baseline graph templates.

Author: Simulation Framework Designer & Software Architect
License: MIT
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

# Import cross-module dependency structures for typing validation
from simulation.network_state import NetworkState
from simulation.disaster_profiles import DisasterProfile

# Configure Module Logger
logger = logging.getLogger("DisasterMANET.NetworkUpdater")


# ============================================================================
# CUSTOM UPDATER EXCEPTIONS
# ============================================================================

class NetworkUpdaterException(Exception):
    """Base exception for all architectural network updater anomalies."""
    pass


class TargetValidationConcreteError(NetworkUpdaterException):
    """Raised when an update payload breaches domain boundaries or logical telemetry limits."""
    pass


# ============================================================================
# CENTRAL NETWORK UPDATER LAYER
# ============================================================================

class NetworkUpdater:
    """
    The exclusive transactional entity authorized to alter the active topology,
    QoS constraints, and infrastructure metrics held within the central NetworkState.
    """

    def __init__(self, network_state: NetworkState) -> None:
        """
        Initializes the updater context bound to an execution state container.

        Args:
            network_state (NetworkState): Single source of truth simulation state.
        """
        if network_state is None:
            raise TypeError("NetworkUpdater initialization aborted: NetworkState container cannot be None.")
        self._state: NetworkState = network_state

    # ------------------------------------------------------------------------
    # VALIDATION HELPERS
    # ------------------------------------------------------------------------

    def _assert_node_exists(self, node_id: str) -> None:
        """Validates node registration within the state wrapper."""
        if not self._state.node_exists(node_id):
            raise TargetValidationConcreteError(f"Node operation failure: Node '{node_id}' missing from active topology.")

    def _assert_edge_exists(self, node1: str, node2: str) -> None:
        """Validates bidirectional edge registration within the state wrapper."""
        if not self._state.edge_exists(node1, node2):
            raise TargetValidationConcreteError(f"Edge operation failure: Link joining ('{node1}', '{node2}') does not exist.")

    def _validate_qos_bounds(self, key: str, value: float) -> None:
        """Enforces structural domain sanity checks on inbound network communication primitives."""
        if key in ["current_bandwidth", "current_latency", "current_jitter"] and value <= 0:
            raise TargetValidationConcreteError(f"QoS Violation: Attribute '{key}' value {value} must be strictly positive (> 0).")
        if key == "current_packet_loss" and not (0.0 <= value <= 100.0):
            raise TargetValidationConcreteError(f"QoS Violation: Packet loss percentage {value} must fall within range [0, 100].")
        if key == "current_reliability" and not (0.0 <= value <= 1.0):
            raise TargetValidationConcreteError(f"QoS Violation: Reliability factor {value} must fall within range [0.0, 1.0].")
        if key == "current_signal_strength" and not (-120.0 <= value <= 0.0):
            # Enforce standard wireless decibel-milliwatts (dBm) scaling metrics
            raise TargetValidationConcreteError(f"QoS Violation: Signal Strength {value} dBm out of realistic operational boundaries.")

    # ------------------------------------------------------------------------
    # NODE OPERATIONS
    # ------------------------------------------------------------------------

    def update_node_status(self, node_id: str, status: str) -> None:
        """Mutates the core structural lifecycle operational status of a vertex."""
        self._assert_node_exists(node_id)
        self._state.update_node(node_id, status=status)
        logger.debug(f"Node Updated -> ID: '{node_id}' | Status changed to: {status}")
        self.recompute_global_telemetry()

    def update_node_load(self, node_id: str, load_percentage: float) -> None:
        """Alters the dynamic computational utilization metrics on a target node."""
        self._assert_node_exists(node_id)
        if not (0.0 <= load_percentage <= 100.0):
            raise TargetValidationConcreteError(f"Load factor {load_percentage}% exceeds operational parameters.")
        self._state.update_node(node_id, load=load_percentage)
        logger.debug(f"Node Updated -> ID: '{node_id}' | Dynamic Load: {load_percentage}%")

    def update_hospital_load(self, node_id: str, load_increment: int) -> None:
        """Increments active tracking occupancy metrics on healthcare infrastructure nodes."""
        self._assert_node_exists(node_id)
        node_data = self._state.get_node(node_id)
        if node_data.get("type") != "Hospital":
            raise TargetValidationConcreteError(f"Infrastructure Type Mismatch: Node '{node_id}' is not a Hospital entity.")
        
        current_load = node_data.get("hospital_utilization", 0)
        updated_load = max(0, current_load + load_increment)
        self._state.update_node(node_id, hospital_utilization=updated_load)
        logger.debug(f"Node Updated -> Hospital: '{node_id}' | Utilization adjusted to: {updated_load}")
        self.recompute_global_telemetry()

    def update_relief_camp(self, node_id: str, current_occupancy: int, max_capacity: Optional[int] = None) -> None:
        """Modifies baseline capacity or refugee populations inside shelter camps."""
        self._assert_node_exists(node_id)
        node_data = self._state.get_node(node_id)
        if node_data.get("type") != "ReliefCamp":
            raise TargetValidationConcreteError(f"Infrastructure Type Mismatch: Node '{node_id}' is not a ReliefCamp.")
        
        updates: Dict[str, Any] = {"camp_occupancy": max(0, current_occupancy)}
        if max_capacity is not None:
            if max_capacity <= 0:
                raise TargetValidationConcreteError("Camp structural max limits must be strictly positive.")
            updates["camp_max_capacity"] = max_capacity
            
        self._state.update_node(node_id, **updates)
        logger.debug(f"Node Updated -> ReliefCamp: '{node_id}' | Metrics set to: {updates}")

    def update_tower(self, node_id: str, health: float, status: str) -> None:
        """Alters structural health indices and status signatures on telecommunication structures."""
        self._assert_node_exists(node_id)
        if not (0.0 <= health <= 1.0):
            raise TargetValidationConcreteError("Physical tower health index factor must fall within range [0.0, 1.0].")
        self._state.update_node(node_id, tower_health=health, status=status)
        logger.debug(f"Node Updated -> Tower: '{node_id}' | Health: {health} | Status: {status}")
        self.recompute_global_telemetry()

    def update_utility(self, node_id: str, grid_status: str, health: float) -> None:
        """Saves systemic operational conditions for critical power or logistics hubs."""
        self._assert_node_exists(node_id)
        if not (0.0 <= health <= 1.0):
            raise TargetValidationConcreteError("Utility structural health index must be a factor within [0.0, 1.0].")
        self._state.update_node(node_id, utility_status=grid_status, utility_health=health)
        logger.debug(f"Node Updated -> Utility: '{node_id}' | Grid: {grid_status} | Health: {health}")

    def update_village(self, node_id: str, population_affected: int, isolated: bool) -> None:
        """Saves humanitarian demographics changes inside target destination nodes."""
        self._assert_node_exists(node_id)
        self._state.update_node(node_id, population_affected=max(0, population_affected), is_isolated=isolated)
        logger.debug(f"Node Updated -> Settlement: '{node_id}' | Affected Pop: {population_affected} | Isolated: {isolated}")

    def isolate_node(self, node_id: str) -> None:
        """Sets node status to dead and severs all connection links."""
        self._assert_node_exists(node_id)
        self._state.update_node(node_id, status="OFFLINE", dynamic_load=0.0)
        
        # Enforce structural topological decoupling across neighbors
        neighbors = self._state.get_neighbors(node_id)
        for target in neighbors:
            self.remove_link(node_id, target)
            
        logger.warning(f"Node Isolated -> ID: '{node_id}' transitioned to OFFLINE state; incident links purged.")
        self.recompute_global_telemetry()

    def remove_node(self, node_id: str) -> Dict[str, Any]:
        """Purges node tracking records from active simulation contexts directly."""
        # Delegated directly via state layer transaction handlers
        data = self._state.remove_node(node_id)
        logger.warning(f"Node Isolated -> Structural execution complete: Node '{node_id}' removed from topology graph.")
        self.recompute_global_telemetry()
        return data

    def restore_node(self, node_id: str, node_data: Optional[Dict[str, Any]] = None) -> None:
        """Re-introduces a dead or removed node back to active state layers."""
        self._state.restore_node(node_id, node_data)
        logger.info(f"Node Updated -> Restoration complete for target identifier: '{node_id}'.")
        self.recompute_global_telemetry()

    # ------------------------------------------------------------------------
    # EDGE OPERATIONS
    # ------------------------------------------------------------------------

    def update_edge_status(self, node1: str, node2: str, status: str) -> None:
        """Mutates the physical availability status tracking on an active edge."""
        self._assert_edge_exists(node1, node2)
        self._state.update_edge(node1, node2, status=status)
        logger.debug(f"Edge Updated -> Link ('{node1}', '{node2}') status changed to: {status}")
        self.recompute_global_telemetry()

    def update_latency(self, node1: str, node2: str, current_latency: float) -> None:
        self._assert_edge_exists(node1, node2)
        self._validate_qos_bounds("current_latency", current_latency)
        self._state.update_edge(node1, node2, current_latency=current_latency)
        logger.debug(f"QoS Updated -> Link ('{node1}', '{node2}') Latency: {current_latency} ms")

    def update_bandwidth(self, node1: str, node2: str, current_bandwidth: float) -> None:
        self._assert_edge_exists(node1, node2)
        self._validate_qos_bounds("current_bandwidth", current_bandwidth)
        self._state.update_edge(node1, node2, current_bandwidth=current_bandwidth)
        logger.debug(f"QoS Updated -> Link ('{node1}', '{node2}') Bandwidth: {current_bandwidth} Mbps")

    def update_packet_loss(self, node1: str, node2: str, current_packet_loss: float) -> None:
        self._assert_edge_exists(node1, node2)
        self._validate_qos_bounds("current_packet_loss", current_packet_loss)
        self._state.update_edge(node1, node2, current_packet_loss=current_packet_loss)
        logger.debug(f"QoS Updated -> Link ('{node1}', '{node2}') Loss: {current_packet_loss}%")

    def update_jitter(self, node1: str, node2: str, current_jitter: float) -> None:
        self._assert_edge_exists(node1, node2)
        self._validate_qos_bounds("current_jitter", current_jitter)
        self._state.update_edge(node1, node2, current_jitter=current_jitter)

    def update_signal_strength(self, node1: str, node2: str, current_signal_strength: float) -> None:
        self._assert_edge_exists(node1, node2)
        self._validate_qos_bounds("current_signal_strength", current_signal_strength)
        self._state.update_edge(node1, node2, current_signal_strength=current_signal_strength)

    def update_reliability(self, node1: str, node2: str, current_reliability: float) -> None:
        self._assert_edge_exists(node1, node2)
        self._validate_qos_bounds("current_reliability", current_reliability)
        self._state.update_edge(node1, node2, current_reliability=current_reliability)

    def remove_link(self, node1: str, node2: str) -> Dict[str, Any]:
        """Drops a communication edge between two nodes and logs the drop."""
        data = self._state.remove_link(node1, node2)
        logger.warning(f"Link Removed -> Edge channel dropped between ('{node1}', '{node2}').")
        self.recompute_global_telemetry()
        return data

    def restore_link(self, node1: str, node2: str, edge_data: Optional[Dict[str, Any]] = None) -> None:
        """Reconstructs a connection link using previous properties."""
        self._state.restore_link(node1, node2, edge_data)
        logger.info(f"Link Restored -> Spatial link bridge established between ('{node1}', '{node2}').")
        self.recompute_global_telemetry()

    # ------------------------------------------------------------------------
    # DISASTER PROFILE SUPPORT
    # ------------------------------------------------------------------------

    def apply_disaster_effects(self, profile: DisasterProfile, target_nodes: List[str]) -> None:
        """
        Scales and injects localized physical and communication performance drops
        across target assets based on a disaster profile.

        Args:
            profile (DisasterProfile): The configuration parameters profile.
            target_nodes (List[str]): Unique string labels of nodes inside the hazard zone.
        """
        if not isinstance(profile, DisasterProfile):
            raise TypeError("Disaster effects application requires a valid DisasterProfile reference instance.")
        
        logger.info(f"Applying Disaster Effects -> Profile: [{profile.disaster_type.name}] across {len(target_nodes)} vertices.")

        node_set: Set[str] = set(target_nodes)

        # Step 1: Update Node States and Infrastructure
        for node_id in target_nodes:
            if not self._state.node_exists(node_id):
                continue
            
            node_data = self._state.get_node(node_id)
            node_type = node_data.get("type", "Standard")

            # Apply explicit component-level drops
            if node_type == "Tower":
                current_health = node_data.get("tower_health", 1.0)
                new_health = max(0.0, current_health - (profile.tower_failure_probability * profile.default_severity))
                status = "DEGRADED" if new_health > 0.3 else "OFFLINE"
                self.update_tower(node_id, health=new_health, status=status)
                
            elif node_type == "Hospital":
                self.update_hospital_load(node_id, profile.hospital_load_increment)
                
            elif node_type == "ReliefCamp":
                current_occ = node_data.get("camp_occupancy", 0)
                self.update_relief_camp(node_id, current_occupancy=current_occ + profile.relief_camp_load_increment)
                
            elif node_type == "Utility":
                current_health = node_data.get("utility_health", 1.0)
                new_health = max(0.0, current_health - (profile.utility_failure_probability * profile.default_severity))
                status = "OPERATIONAL" if new_health > 0.5 else "CRITICAL"
                self.update_utility(node_id, grid_status=status, health=new_health)

            # Apply general node load degradation factors
            current_load = node_data.get("load", 10.0)
            new_load = min(100.0, current_load + (profile.emergency_request_multiplier * 10.0))
            self._state.update_node(node_id, load=new_load)

        # Step 2: Update Impacted Edge Communication States
        live_graph = self._state.graph
        for u, v in list(live_graph.edges):
            if u in node_set or v in node_set:
                edge_data = live_graph.edges[u, v]
                
                # Fetch baseline graph values safely to prevent recursive inflation drops
                base_lat = edge_data.get("base_latency", 5.0)
                base_bw = edge_data.get("base_bandwidth", 54.0)
                base_loss = edge_data.get("base_packet_loss", 0.5)
                base_jitter = edge_data.get("base_jitter", 1.0)
                base_sig = edge_data.get("base_signal_strength", -45.0)
                base_rel = edge_data.get("base_reliability", 0.99)

                # Compute current metrics using profile multipliers scaled by severity
                severity_factor = profile.default_severity
                
                computed_lat = base_lat * (1.0 + (profile.latency_multiplier - 1.0) * severity_factor)
                computed_bw = max(1.0, base_bw * (1.0 - (1.0 - profile.bandwidth_multiplier) * severity_factor))
                computed_loss = min(100.0, base_loss + (profile.packet_loss_multiplier * 10.0 * severity_factor))
                computed_jitter = base_jitter * (1.0 + (profile.jitter_multiplier - 1.0) * severity_factor)
                
                # Attenuate dBm signal values down toward the -120 dBm noise floor
                computed_sig = max(-120.0, base_sig - (abs(base_sig) * (1.0 - profile.signal_strength_multiplier) * severity_factor))
                computed_rel = max(0.01, base_rel * (1.0 - profile.link_failure_probability * severity_factor))

                # Apply mutations safely using validation interfaces
                self.update_latency(u, v, computed_lat)
                self.update_bandwidth(u, v, computed_bw)
                self.update_packet_loss(u, v, computed_loss)
                self.update_jitter(u, v, computed_jitter)
                self.update_signal_strength(u, v, computed_sig)
                self.update_reliability(u, v, computed_rel)

        logger.info("Disaster Applied -> Regional topology attributes successfully updated.")
        self.recompute_global_telemetry()

    # ------------------------------------------------------------------------
    # RECOVERY SUPPORT OPERATIONS
    # ------------------------------------------------------------------------

    def restore_node_defaults(self, node_id: str) -> None:
        """Rolls a single node back to pristine configuration parameters."""
        self._assert_node_exists(node_id)
        base_graph = self._state.get_graph_copy() # Deep isolated backup reference pass
        if base_graph.has_node(node_id):
            pristine_attrs = base_graph.nodes[node_id]
            self._state.update_node(node_id, **pristine_attrs)
            logger.info(f"Recovery Applied -> Node '{node_id}' reset to baseline default settings.")
            self.recompute_global_telemetry()

    def restore_edge_defaults(self, node1: str, node2: str) -> None:
        """Rolls an edge back to pristine configuration parameters."""
        self._assert_edge_exists(node1, node2)
        # Fetch initial reference variables mapping fields back
        edge_data = self._state.get_edge(node1, node2)
        
        updates = {
            "current_latency": edge_data.get("base_latency", 5.0),
            "current_bandwidth": edge_data.get("base_bandwidth", 100.0),
            "current_packet_loss": edge_data.get("base_packet_loss", 0.0),
            "current_jitter": edge_data.get("base_jitter", 1.0),
            "current_signal_strength": edge_data.get("base_signal_strength", -50.0),
            "current_reliability": edge_data.get("base_reliability", 1.0),
            "status": "OPERATIONAL"
        }
        self._state.update_edge(node1, node2, **updates)
        logger.info(f"Recovery Applied -> Link ('{node1}', '{node2}') QoS metrics reset to pristine limits.")

    def restore_network(self) -> None:
        """Iterates over all topological elements, resetting metrics to pristine baseline settings."""
        logger.warning("Recovery Applied -> Structural full network restoration triggered.")
        live_graph = self._state.graph
        
        # Restore edges
        for u, v in list(live_graph.edges):
            self.restore_edge_defaults(u, v)
            
        # Restore nodes via sequential base updates
        for node_id in list(live_graph.nodes):
            self._state.update_node(node_id, status="OPERATIONAL", load=10.0, tower_health=1.0, hospital_utilization=0, camp_occupancy=0)
            
        logger.info("Recovery Applied -> Complete structural rollback completed successfully.")
        self.recompute_global_telemetry()

    # ------------------------------------------------------------------------
    # TELEMETRY ANALYSIS AND STATS HOOKS
    # ------------------------------------------------------------------------

    def recompute_global_telemetry(self) -> None:
        """
        Executes a real-time tracking pass across the dynamic NetworkX graph view,
        computing performance summaries to feed the central metrics payload registries.
        """
        live_graph = self._state.graph
        total_edges = live_graph.number_of_edges()
        total_nodes = live_graph.number_of_nodes()

        # Fallback handling for empty topologies
        if total_nodes == 0:
            return

        running_latency = 0.0
        running_bw = 0.0
        congested_links: List[Tuple[str, str]] = []
        offline_towers = 0
        running_hospital_load = 0
        hospital_count = 0

        # Scan edges for communications metrics
        for u, v, data in live_graph.edges(data=True):
            running_latency += data.get("current_latency", 0.0)
            bw = data.get("current_bandwidth", 1.0)
            running_bw += bw
            
            # Identify congested links where current bandwidth drops below 20% of baseline capacity
            if bw < (data.get("base_bandwidth", 100.0) * 0.2):
                congested_links.append((u, v))

        # Scan nodes for infrastructure state metrics
        for node_id, data in live_graph.nodes(data=True):
            if data.get("type") == "Tower" and data.get("status") == "OFFLINE":
                offline_towers += 1
            elif data.get("type") == "Hospital":
                hospital_count += 1
                running_hospital_load += data.get("hospital_utilization", 0)

        # Calculate graph connectivity features using NetworkX algorithms
        connected_components = nx.number_connected_components(live_graph) if not live_graph.is_directed() else 1
        isolated_nodes = list(nx.isolates(live_graph))

        # Compute averages safely
        avg_latency = running_latency / total_edges if total_edges > 0 else 0.0
        avg_bandwidth = running_bw / total_edges if total_edges > 0 else 0.0
        avg_hospital_util = running_hospital_load / hospital_count if hospital_count > 0 else 0.0
        network_availability = ((total_nodes - len(isolated_nodes)) / total_nodes) * 100.0

        # Package statistics into structured data blocks
        telemetry_payload = {
            "average_latency": avg_latency,
            "average_bandwidth": avg_bandwidth,
            "congested_links": congested_links,
            "offline_towers_count": offline_towers,
            "hospital_utilization_avg": avg_hospital_util,
            "connected_components": connected_components,
            "isolated_nodes": isolated_nodes,
            "network_utilization": network_availability
        }

        # Update central registers securely via the state interface hook
        self._state.update_network_statistics(telemetry_payload)
        logger.debug("Telemetry Refresh -> Global structural performance indices recomputed and pushed.")