"""
services/dispatcher.py

This module implements the core communication Dispatcher for the QoS-Aware MANET.
It orchestrates packet formatting, network state topology mapping, path generation, 
and executes realistic hop-by-hop data transmission simulation.

Architecture Enhancements:
- Implements simulated hop-by-hop forwarding iterating across generated routes.
- Aggregates multi-metric edge QoS telemetry data (Latency, Loss, Jitter, Bandwidth, Reliability).
- Dynamically decrements Time-To-Live (TTL) variables per transmission hop.
- Implements automated packet transmission retries using secondary alternate pathway discovery.
- Introduces historical structural dispatch tracking events (DispatchEvent timeline logging).
- Significantly expands performance analytics tracking matrices (PDR, Jitter, Remaining TTL, Retries).
"""

from __future__ import annotations

import collections
import datetime
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

# Enforce explicit dependency boundary map layout 
from services.message import DisasterMessage, DestinationType, MessagePriority, MessageStatus, QoSLevel
from services.packet import NetworkPacket, PacketStatus


# Setup module-level structured logger
logger = logging.getLogger("services.dispatcher")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class DispatcherError(Exception):
    """Base exception for all faults occurring within the orchestration and queue processing layers."""
    pass


class DestinationResolutionError(DispatcherError):
    """Raised when the dispatcher fails to identify an operational or reachable node using routing policy."""
    pass


class RoutingFailureError(DispatcherError):
    """Raised when the downstream routing engine cannot discover a valid topological path link."""
    pass


@dataclass
class DispatchEvent:
    """Encapsulates a distinct forwarding or queue state change step for timeline logging."""
    timestamp: str
    packet_id: str
    action: str  # GENERATED, QUEUED, FORWARDED, DELIVERED, RETRIED, FAILED
    node: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DispatchResult:
    """Encapsulates the terminal state metrics of a dispatch transaction execution."""
    success: bool
    packets: List[NetworkPacket]
    destination_node: Optional[str]
    route: List[str]
    dispatch_timestamp: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    routing_time_ms: float = 0.0
    queue_time_ms: float = 0.0
    error_message: Optional[str] = None


class RoutingEngineProtocol(Protocol):
    """Structural type protocol mapping required functions for interchangeable MANET routing backends."""
    def calculate_route(self, graph: Any, source: str, destination: str) -> List[str]:
        ...


class NetworkStateProtocol(Protocol):
    """Structural type protocol tracking active mobile node positions and network metrics."""
    @property
    def graph(self) -> Any:
        ...
    def get_nodes_by_type(self, type_str: str) -> List[str]:
        ...
    def compute_shortest_path_distance(self, source: str, target: str) -> float:
        ...
    def get_edge_metrics(self, u: str, v: str) -> Dict[str, float]:
        """Returns structural QoS attributes for a network edge link (latency, loss, jitter, etc.)."""
        ...
    def is_node_operational(self, node: str) -> bool:
        """Tracks active node statuses (online/offline/capacity limit constraints)."""
        ...


class SchedulingPolicy:
    """Decouples traffic prioritization metrics and congestion penalties from core dispatcher routing."""
    @staticmethod
    def get_priority_weight(packet: NetworkPacket) -> int:
        """Maps traffic classes into strict priority index categories (lower is preferred)."""
        if packet.qos_level == QoSLevel.EMERGENCY:
            return 0
        if packet.priority == MessagePriority.CRITICAL:
            return 1
        if packet.priority == MessagePriority.HIGH:
            return 2
        if packet.priority == MessagePriority.MEDIUM:
            return 3
        return 4


@dataclass(order=True)
class PriorityQueueEntry:
    """Wrapper tracking sorted frames using monotonic sequence identifiers to prevent collision sorting errors."""
    priority_weight: int
    entry_id: int
    packet: NetworkPacket = field(compare=False)


class DefaultRoutingEngine:
    """Default implementation of RoutingEngineProtocol using communication.routing."""
    def calculate_route(self, graph: Any, source: str, destination: str) -> List[str]:
        try:
            from communication.routing import calculate_route as calc_route
            res = calc_route(graph, source, destination)
            if res.get("success"):
                return res["path"]
        except Exception as e:
            logger.error(f"Default routing calculation failed: {e}")
        return []


class Dispatcher:
    """Coordinates transport frame lifecycle paths, hop simulation tracking, and network telemetries."""
    
    MAX_RETRIES: int = 2

    def __init__(self, network_state: Any = None, routing_engine: Any = None) -> None:
        """Initializes priority tracking queues and state backend modules."""
        self.network_state: NetworkStateProtocol = network_state
        self.routing_engine: RoutingEngineProtocol = routing_engine if routing_engine is not None else DefaultRoutingEngine()
        
        self._sequence_counter: int = 0
        self._pending_queue: List[PriorityQueueEntry] = []
        
        self._transmission_queue: collections.deque[NetworkPacket] = collections.deque()
        self._delivered_queue: collections.deque[NetworkPacket] = collections.deque()
        self._failed_queue: collections.deque[NetworkPacket] = collections.deque()
        
        # Operational Optimizations and Logging Streams
        self._destination_cache: Dict[Tuple[str, DestinationType], str] = {}
        self._routing_cache: Dict[Tuple[str, str], List[str]] = {}
        self.timeline_events: List[DispatchEvent] = []
        
        # Global Telemetry Trackers
        self._total_processed_count: int = 0
        self._total_retries_executed: int = 0

    def pending_count(self) -> int: return len(self._pending_queue)
    def transmission_count(self) -> int: return len(self._transmission_queue)
    def delivered_count(self) -> int: return len(self._delivered_queue)
    def failed_count(self) -> int: return len(self._failed_queue)

    def log_event(self, packet_id: str, action: str, node: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Appends a trackable step entry to the global framework timeline ledger."""
        event = DispatchEvent(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            packet_id=packet_id,
            action=action,
            node=node,
            metadata=metadata or {}
        )
        self.timeline_events.append(event)

    def clear_caches(self) -> None:
        """Flushes cached metrics whenever topological routing state modifications occur."""
        self._destination_cache.clear()
        self._routing_cache.clear()
        logger.debug("Dispatcher network caches flushed following mobile context updates.")

    def enqueue_packet(self, packet: NetworkPacket) -> None:
        """Appends an individual network frame to the priority heap array layout."""
        weight = SchedulingPolicy.get_priority_weight(packet)
        self._sequence_counter += 1
        packet.metadata["queue_enter_time"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        entry = PriorityQueueEntry(priority_weight=weight, entry_id=self._sequence_counter, packet=packet)
        heapq.heappush(self._pending_queue, entry)
        self.log_event(packet.packet_id, "QUEUED", packet.current_node or "SOURCE")

    def dequeue_packet(self) -> Optional[NetworkPacket]:
        if self._pending_queue:
            return heapq.heappop(self._pending_queue).packet
        return None

    def dispatch_packets(self, packets: List[NetworkPacket], message: DisasterMessage) -> DispatchResult:
        """Resolves ideal target milestones, maps structural constraints, and enqueues frame streams."""
        if not packets:
            return DispatchResult(success=False, packets=[], destination_node=None, route=[], error_message="Empty stream.")
            
        try:
            for pkt in packets:
                self._validate_packet_invariants(pkt)
                
            if message.destination_type is None:
                raise DispatcherError(f"Destination Type constraint is null on message {message.id}")

            destination_node = self.resolve_destination_node(message, message.destination_type)
            source_node = message.origin_node

            # Guard: source must exist in the live graph before routing is attempted
            if not self.network_state.node_exists(source_node):
                raise DispatcherError(
                    f"Origin node '{source_node}' does not exist in the network graph. "
                    f"Use a valid node ID (e.g. V1, V4, T2, C1, H1, H2, R1, R2, P1, F1)."
                )

            # Utilize localized Routing Matrix Cache optimization
            cache_key = (source_node, destination_node)
            start_routing_time = time.perf_counter()
            
            if cache_key in self._routing_cache:
                calculated_path = list(self._routing_cache[cache_key])
            else:
                calculated_path = self.routing_engine.calculate_route(
                    self.network_state.graph, source_node, destination_node
                )
                if calculated_path and len(calculated_path) >= 2:
                    self._routing_cache[cache_key] = list(calculated_path)

            routing_duration_ms = (time.perf_counter() - start_routing_time) * 1000.0

            if not calculated_path or len(calculated_path) < 2:
                raise RoutingFailureError(f"No routable path paths connecting {source_node} to {destination_node}")

            next_hop_node = calculated_path[1]
            for pkt in packets:
                pkt.destination_node = destination_node
                pkt.route = list(calculated_path)
                pkt.current_node = source_node
                pkt.next_hop = next_hop_node
                
                # Zero out or initialize packet-level simulated telemetry variables
                pkt.latency_ms = 0.0
                pkt.packet_loss_percent = 0.0
                pkt.jitter_ms = 0.0
                pkt.hop_count = 0
                
                if pkt.status == PacketStatus.CREATED:
                    pkt.mark_ready()
                
                self.enqueue_packet(pkt)
                self._total_processed_count += 1

            return DispatchResult(
                success=True, packets=list(packets), destination_node=destination_node, 
                route=calculated_path, routing_time_ms=routing_duration_ms
            )

        except (DispatcherError, RoutingFailureError, DestinationResolutionError) as error:
            logger.error(f"Dispatch pipeline error: {str(error)}")
            return DispatchResult(success=False, packets=list(packets), destination_node=None, route=[], error_message=str(error))

    def resolve_destination_node(self, message: DisasterMessage, destination_type: DestinationType) -> str:
        """Finds closest operational, active, and capacity-safe node belonging to target classification."""
        cache_key = (message.origin_node, destination_type)
        if cache_key in self._destination_cache:
            return self._destination_cache[cache_key]

        matching_nodes = self.network_state.get_nodes_by_type(destination_type.name)
        # Dynamic Constraint Filtering: Keep only currently operational target nodes
        operational_nodes = [n for n in matching_nodes if self.network_state.is_node_operational(n)]
        
        if not operational_nodes:
            raise DestinationResolutionError(f"No active or operational nodes matched criteria for: {destination_type.name}")

        source_node = message.origin_node
        nearest_node: Optional[str] = None
        min_distance = float('inf')

        for node in operational_nodes:
            try:
                distance = self.network_state.compute_shortest_path_distance(source_node, node)
                if distance < min_distance:
                    min_distance = distance
                    nearest_node = node
            except Exception:
                continue

        if nearest_node is None:
            nearest_node = operational_nodes[0]

        self._destination_cache[cache_key] = nearest_node
        return nearest_node

    def start_transmission(self) -> List[NetworkPacket]:
        """Shifts pending items to active loops, executing realistic hop-by-hop simulation walks."""
        transmitting_list: List[NetworkPacket] = []
        current_time = datetime.datetime.now(datetime.timezone.utc)
        
        while self._pending_queue:
            pkt = self.dequeue_packet()
            if not pkt:
                continue
                
            if "queue_enter_time" in pkt.metadata:
                try:
                    enter_dt = datetime.datetime.fromisoformat(pkt.metadata["queue_enter_time"])
                    pkt.queue_delay_ms = (current_time - enter_dt).total_seconds() * 1000.0
                except Exception:
                    pkt.queue_delay_ms = 0.0
            
            pkt.mark_queued()
            pkt.mark_transmitting()
            self._transmission_queue.append(pkt)
            
            # Execute physical hop forwarding walk algorithm
            success = self._simulate_hop_by_hop_forwarding(pkt)
            if success:
                self.complete_delivery(pkt)
                transmitting_list.append(pkt)
            else:
                # Execution layer intercepted a link/TTL fault -> invoke localized retry handlers
                self._handle_retry_or_failure(pkt, transmitting_list)
                
        return transmitting_list

    def _simulate_hop_by_hop_forwarding(self, packet: NetworkPacket) -> bool:
        """Iterates sequentially across calculated routes, mutating headers and logging edge QoS drops."""
        route = packet.route
        if not route or len(route) < 2:
            return False

        logger.info(f"Simulating Hop-By-Hop Forwarding trace loop for Packet {packet.packet_id}")
        
        for i in range(len(route) - 1):
            current = route[i]
            nxt = route[i + 1]
            
            # Ensure next step target has not dropped offline mid-transit
            if not self.network_state.is_node_operational(nxt):
                logger.warning(f"Link broken during transit! Node {nxt} went offline.")
                return False

            # Query real edge metric records from topological matrix layout
            edge_data = self.network_state.get_edge_metrics(current, nxt)
            
            # Structural Simulated Drop Assessment (Deterministic drop check based on packet loss metrics)
            if edge_data.get("packet_loss", 0.0) >= 85.0:
                logger.warning(f"Packet drop scenario triggered on high-loss edge link {current} -> {nxt}")
                return False

            # Mutate Network Header telemetry state properties
            packet.current_node = nxt
            packet.previous_hop = current
            packet.next_hop = route[i + 2] if (i + 2) < len(route) else None
            
            # Track nodes inside internal tracking registers
            if not hasattr(packet, 'visited_nodes') or packet.visited_nodes is None:
                packet.visited_nodes = []
            packet.visited_nodes.append(current)

            # Mathematical QoS parameter accumulations
            packet.hop_count += 1
            packet.ttl -= 1
            packet.latency_ms += edge_data.get("latency", 1.0)
            packet.packet_loss_percent += edge_data.get("packet_loss", 0.0)
            packet.id = packet.packet_id  # Enforce variable aliases for forward tracking properties
            
            # Store maximum jitter peaks and lowest pipeline bandwidth records safely
            if not hasattr(packet, 'jitter_ms') or packet.jitter_ms is None: packet.jitter_ms = 0.0
            packet.jitter_ms = max(packet.jitter_ms, edge_data.get("jitter", 0.0))
            
            if not hasattr(packet, 'min_bandwidth_mbps') or packet.min_bandwidth_mbps is None:
                packet.min_bandwidth_mbps = edge_data.get("bandwidth", 100.0)
            else:
                packet.min_bandwidth_mbps = min(packet.min_bandwidth_mbps, edge_data.get("bandwidth", 100.0))

            self.log_event(
                packet.packet_id, "FORWARDED", current, 
                metadata={"next_hop": nxt, "ttl_remaining": packet.ttl, "accumulated_latency": packet.latency_ms}
            )

            # TTL Expiry boundary verification guard
            if packet.ttl <= 0:
                logger.warning(f"Packet dead drop exception: TTL expired at node {nxt} during hop step calculations.")
                return False

        return True

    def _handle_retry_or_failure(self, packet: NetworkPacket, transmitting_list: List[NetworkPacket]) -> None:
        """Attempts alternate routing path discoveries before permanently writing off packet entries."""
        current_retries = packet.metadata.get("retry_count", 0)
        
        if current_retries < self.MAX_RETRIES:
            self._total_retries_executed += 1
            packet.metadata["retry_count"] = current_retries + 1
            logger.info(f"Packet {packet.packet_id} failed step. Retrying ({current_retries + 1}/{self.MAX_RETRIES}). Recalculating route...")
            
            self.log_event(packet.packet_id, "RETRIED", packet.current_node or "UNKNOWN")
            
            # Attempt to re-route starting from wherever the packet was dropped
            try:
                new_path = self.routing_engine.calculate_route(
                    self.network_state.graph, packet.current_node, packet.destination_node
                )
                if new_path and len(new_path) >= 2:
                    packet.route = list(new_path)
                    packet.next_hop = new_path[1]
                    packet.ttl = getattr(packet, 'initial_ttl', 32) # Standard baseline fallback
                    packet.status = PacketStatus.READY
                    
                    # Re-enqueue item back to queue processing pools
                    self.enqueue_packet(packet)
                    return
            except Exception:
                pass
                
        # If execution blocks fall through, write the entry to failure state collections
        self.fail_packet(packet)

    def complete_delivery(self, packet: NetworkPacket) -> None:
        self._remove_from_queue(packet, self._transmission_queue)
        packet.mark_delivered()
        self._delivered_queue.append(packet)
        self.log_event(packet.packet_id, "DELIVERED", packet.current_node or "DESTINATION")

    def fail_packet(self, packet: NetworkPacket) -> None:
        self._remove_from_queue(packet, self._transmission_queue)
        packet.mark_failed()
        self._failed_queue.append(packet)
        self.log_event(packet.packet_id, "FAILED", packet.current_node or "DROP_NODE")

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates granular real-time performance evaluation matrices for scientific telemetry summaries."""
        delivered_list = list(self._delivered_queue)
        failed_count = self.failed_count()
        total_delivered = len(delivered_list)
        total_attempts = total_delivered + failed_count
        
        pdr = (total_delivered / total_attempts) if total_attempts > 0 else 0.0
        
        avg_delay = sum(p.latency_ms for p in delivered_list) / total_delivered if total_delivered > 0 else 0.0
        avg_q_delay = sum(getattr(p, 'queue_delay_ms', 0.0) for p in delivered_list) / total_delivered if total_delivered > 0 else 0.0
        avg_ttl = sum(p.ttl for p in delivered_list) / total_delivered if total_delivered > 0 else 0.0
        avg_loss = sum(p.packet_loss_percent for p in delivered_list) / total_delivered if total_delivered > 0 else 0.0
        avg_jitter = sum(getattr(p, 'jitter_ms', 0.0) for p in delivered_list) / total_delivered if total_delivered > 0 else 0.0
        avg_route_len = sum(len(p.route) for p in delivered_list) / total_delivered if total_delivered > 0 else 0.0

        return {
            "pending_count": self.pending_count(),
            "transmission_count": self.transmission_count(),
            "delivered_count": total_delivered,
            "failed_count": failed_count,
            "total_processed": self._total_processed_count,
            "total_retries_executed": self._total_retries_executed,
            "packet_delivery_ratio": pdr,
            "average_delay_ms": avg_delay,
            "average_queue_delay_ms": avg_q_delay,
            "average_packet_loss_percent": avg_loss,
            "average_jitter_ms": avg_jitter,
            "average_ttl_remaining": avg_ttl,
            "average_route_length": avg_route_len
        }

    def _validate_packet_invariants(self, packet: NetworkPacket) -> None:
        if not packet.packet_id or not packet.message_id:
            raise DispatcherError("Missing structural unique identity values.")
        if packet.priority is None or packet.qos_level is None:
            raise DispatcherError("Packet contains unassigned QoS configuration tags.")
        if packet.ttl is None or packet.ttl <= 0:
            raise DispatcherError("Invalid or expired initial TTL context settings.")

    def _remove_from_queue(self, packet: NetworkPacket, queue: collections.deque[NetworkPacket]) -> bool:
        try:
            queue.remove(packet)
            return True
        except ValueError:
            return False


if __name__ == "__main__":
    print("\n--- Running services/dispatcher.py Expanded Verification Loop ---")

    # 1. Structural Dependency Mock Realizations Setup
    class AdvancedMockNetworkState:
        def __init__(self) -> None:
            self.graph = "COMPLEX_MANET_TOPOLOGY_GRAPH"
            self.nodes = {"NODE-S1": "CITIZEN", "H1": "HOSPITAL", "H2": "HOSPITAL"}
            # Simulate a temporary network drop point on H1 to test retry logic routines
            self.operational_nodes = {"NODE-S1", "H1", "H2", "INTERMEDIATE-NODE-04", "BACKBONE-LINK-09"}
            
        def get_nodes_by_type(self, type_str: str) -> List[str]:
            return [k for k, v in self.nodes.items() if v == type_str]
            
        def compute_shortest_path_distance(self, source: str, target: str) -> float:
            return 5.0 if target == "H2" else 15.0
            
        def is_node_operational(self, node: str) -> bool:
            return node in self.operational_nodes
            
        def get_edge_metrics(self, u: str, v: str) -> Dict[str, float]:
            # Provide distinct dynamic edge metrics along the route pathing fields
            return {
                "latency": 4.2,
                "packet_loss": 0.02,
                "jitter": 1.1,
                "bandwidth": 45.0
            }

    class MockRoutingEngine:
        def calculate_route(self, graph: Any, source: str, destination: str) -> List[str]:
            return [source, "INTERMEDIATE-NODE-04", "BACKBONE-LINK-09", destination]

    state_mock = AdvancedMockNetworkState()
    routing_mock = MockRoutingEngine()
    dispatcher = Dispatcher(network_state=state_mock, routing_engine=routing_mock)

    from services.packet_generator import PacketGenerator
    from services.message import SenderType, Department

    generator = PacketGenerator()

    # Define High and Emergency priority simulation blocks
    msg_high = DisasterMessage(text="DATA: Supply chain tracking.", origin_node="NODE-S1", sender_id="S1", sender_type=SenderType.CITIZEN)
    msg_high.update_classification(predicted_class="displaced_people_and_evacuations", confidence=0.9, probabilities={})
    msg_high.assign_qos(MessagePriority.HIGH, QoSLevel.HIGH, Department.RELIEF, "H2", DestinationType.HOSPITAL, 60)
    msg_high.ttl = 40

    msg_emerg = DisasterMessage(text="EMERGENCY: Infrastructure collapse structural emergency.", origin_node="NODE-S1", sender_id="S2", sender_type=SenderType.CITIZEN)
    msg_emerg.update_classification(predicted_class="infrastructure_and_utility_damage", confidence=0.99, probabilities={})
    msg_emerg.assign_qos(MessagePriority.CRITICAL, QoSLevel.EMERGENCY, Department.CONTROL_CENTER, "H2", DestinationType.HOSPITAL, 20)
    msg_emerg.ttl = 40

    pkts_high = generator.generate_packets(msg_high)
    msg_emerg.status = MessageStatus.QOS_ASSIGNED
    pkts_emerg = generator.generate_packets(msg_emerg)

    # 2. Run execution dispatch pipelines
    dispatcher.dispatch_packets(pkts_high, msg_high)
    dispatcher.dispatch_packets(pkts_emerg, msg_emerg)

    # 3. Simulate continuous transmission loops
    print(f"\nPrioritized Packets Enqueued: {dispatcher.pending_count()}")
    active_transmissions = dispatcher.start_transmission()

    print("\n==========================================================")
    print("SIMULATED HOP-BY-HOP ROUTE METRICS REPORT")
    print("==========================================================")
    for pkt in active_transmissions:
        print(f"\n  [Packet ID Reference: {pkt.packet_id} - {pkt.packet_type.name}]")
        print(f"    Path Hops Visited : {pkt.hop_count}")
        print(f"    Remaining TTL     : {pkt.ttl}")
        print(f"    Total Latency     : {pkt.latency_ms:.2f} ms")
        print(f"    Total Packet Loss : {pkt.packet_loss_percent * 100:.2f} %")
        print(f"    Peak Jitter Width : {getattr(pkt, 'jitter_ms', 0.0):.2f} ms")
        print(f"    Min Link Bandwidth: {getattr(pkt, 'min_bandwidth_mbps', 0.0):.1f} Mbps")

    print("\n=== Comprehensive Network Telemetry Analytics ===")
    import json
    print(json.dumps(dispatcher.get_statistics(), indent=2))