"""
services/packet.py

This module contains the network packet abstraction layer for the QoS-Aware MANET.
It encapsulates binary data framing, fragmentation metrics, link-layer metrics,
delay tracking, and state-machine transitions across a packet's transmission lifecycle.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, ClassVar, Dict, List, Optional

# Enforce strict import boundary restriction
from .message import MessagePriority, QoSLevel

# Setup module-level structured logger
logger = logging.getLogger("services.packet")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class PacketStatus(IntEnum):
    """Lifecycle states of an active communication packet travelling within the MANET fabric."""
    CREATED = 1
    READY = 2
    QUEUED = 3
    TRANSMITTING = 4
    FORWARDED = 5
    DELIVERED = 6
    DROPPED = 7
    EXPIRED = 8
    FAILED = 9


class PacketType(IntEnum):
    """Categorized protocols used to handle differentiated traffic processing constraints."""
    DATA = 1
    CONTROL = 2
    ACK = 3
    HEARTBEAT = 4
    EMERGENCY = 5


class IllegalPacketStateTransition(Exception):
    """Raised when an invalid state change is requested on a NetworkPacket instance."""
    pass


class PacketValidationError(Exception):
    """Raised when validation criteria are violated during packet initialization or mutation."""
    pass


@dataclass
class NetworkPacket:
    """
    Represents the primary unit of communication data traveling across nodes in the MANET.
    Supports binary payload streams, fragmentation tracking, explicit link telemetry, and delay metrics.
    """
    packet_id: str
    message_id: str
    packet_type: PacketType
    sequence_number: int
    total_packets: int
    fragment_offset: int
    source_node: str
    priority: MessagePriority
    qos_level: QoSLevel
    ttl: int
    delivery_deadline_ms: int
    
    # Internal payload fields managed using properties for explicit side-effect triggers
    _payload: bytes = field(repr=False)
    
    # Structural Network Invariants (Class-wide constraints)
    MAX_PACKET_SIZE: ClassVar[int] = 1024
    
    # Path & Topology State tracking
    destination_node: Optional[str] = None
    previous_hop: Optional[str] = None
    current_node: str = field(default="")
    next_hop: Optional[str] = None
    route: List[str] = field(default_factory=list)
    visited_nodes: List[str] = field(default_factory=list)
    hop_count: int = 0
    retry_count: int = 0
    
    # Performance Link Telemetry Metrics
    latency_ms: float = 0.0
    packet_loss_percent: float = 0.0
    jitter_ms: float = 0.0
    received_signal_strength_dbm: float = 0.0
    transmission_time_ms: float = 0.0
    
    # Core Delay Profiling (Analytical Research Parameters)
    queue_delay_ms: float = 0.0
    transmission_delay_ms: float = 0.0
    processing_delay_ms: float = 0.0
    propagation_delay_ms: float = 0.0
    
    # Checksum tracking variables auto-calculated upon mutations
    checksum: str = field(init=False)
    payload_size_bytes: int = field(init=False)
    
    # Timestamps tracking metrics performance boundary markers
    creation_timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    transmission_timestamp: Optional[datetime.datetime] = None
    delivery_timestamp: Optional[datetime.datetime] = None
    
    # Initial Baseline State tracking
    status: PacketStatus = PacketStatus.CREATED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        packet_id: str,
        message_id: str,
        packet_type: PacketType,
        sequence_number: int,
        total_packets: int,
        fragment_offset: int,
        payload: bytes,
        source_node: str,
        priority: MessagePriority,
        qos_level: QoSLevel,
        ttl: int,
        delivery_deadline_ms: int,
        destination_node: Optional[str] = None,
        previous_hop: Optional[str] = None,
        current_node: Optional[str] = None,
        next_hop: Optional[str] = None,
        route: Optional[List[str]] = None,
        visited_nodes: Optional[List[str]] = None,
        hop_count: int = 0,
        retry_count: int = 0,
        latency_ms: float = 0.0,
        packet_loss_percent: float = 0.0,
        jitter_ms: float = 0.0,
        received_signal_strength_dbm: float = 0.0,
        transmission_time_ms: float = 0.0,
        queue_delay_ms: float = 0.0,
        transmission_delay_ms: float = 0.0,
        processing_delay_ms: float = 0.0,
        propagation_delay_ms: float = 0.0,
        creation_timestamp: Optional[datetime.datetime] = None,
        transmission_timestamp: Optional[datetime.datetime] = None,
        delivery_timestamp: Optional[datetime.datetime] = None,
        status: PacketStatus = PacketStatus.CREATED,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.packet_id = packet_id
        self.message_id = message_id
        self.packet_type = packet_type
        self.sequence_number = sequence_number
        self.total_packets = total_packets
        self.fragment_offset = fragment_offset
        self.source_node = source_node
        self.priority = priority
        self.qos_level = qos_level
        self.ttl = ttl
        self.delivery_deadline_ms = delivery_deadline_ms
        
        self.destination_node = destination_node
        self.previous_hop = previous_hop
        self.current_node = current_node if current_node is not None else source_node
        self.next_hop = next_hop
        self.route = route if route is not None else []
        self.visited_nodes = visited_nodes if visited_nodes is not None else [self.current_node]
        self.hop_count = hop_count
        self.retry_count = retry_count
        
        self.latency_ms = latency_ms
        self.packet_loss_percent = packet_loss_percent
        self.jitter_ms = jitter_ms
        self.received_signal_strength_dbm = received_signal_strength_dbm
        self.transmission_time_ms = transmission_time_ms
        
        self.queue_delay_ms = queue_delay_ms
        self.transmission_delay_ms = transmission_delay_ms
        self.processing_delay_ms = processing_delay_ms
        self.propagation_delay_ms = propagation_delay_ms
        
        if creation_timestamp is not None:
            self.creation_timestamp = creation_timestamp
        else:
            self.creation_timestamp = datetime.datetime.now(datetime.timezone.utc)
            
        self.transmission_timestamp = transmission_timestamp
        self.delivery_timestamp = delivery_timestamp
        self.status = status
        self.metadata = metadata if metadata is not None else {}
        
        # Binary bytes execution triggers automated size, validation checks, and SHA256 binding
        self.payload = payload
        self.validate()

    @property
    def payload(self) -> bytes:
        """Retrieves raw binary payload stream bytes."""
        return self._payload

    @payload.setter
    def payload(self, val: bytes) -> None:
        """Sets binary payload updating internal size metrics and integrity checks."""
        if not isinstance(val, bytes):
            raise PacketValidationError("Payload parameter must explicitly consist of binary bytes datatype.")
        if len(val) == 0:
            raise PacketValidationError(f"Packet initialization failed: empty payload detected on {self.packet_id}")
        if len(val) > self.MAX_PACKET_SIZE:
            raise PacketValidationError(
                f"Payload size {len(val)} bytes exceeds maximum structural packet transmission capacity boundary limit of {self.MAX_PACKET_SIZE} bytes."
            )
        self._payload = val
        self.payload_size_bytes = len(val)
        self.checksum = self.compute_checksum()

    @property
    def end_to_end_delay_ms(self) -> float:
        """Computes Total End-to-End network delay for research simulation evaluations."""
        return self.queue_delay_ms + self.transmission_delay_ms + self.processing_delay_ms + self.propagation_delay_ms

    @property
    def age_ms(self) -> float:
        """Calculates the time elapsed since packet generation in milliseconds."""
        return (datetime.datetime.now(datetime.timezone.utc) - self.creation_timestamp).total_seconds() * 1000.0

    @property
    def is_successful(self) -> bool:
        """Returns True if the packet reached its intended destination node successfully."""
        return self.status == PacketStatus.DELIVERED

    def validate(self) -> None:
        """Enforces structural data type limitations and protocol invariant safety limits."""
        if self.ttl < 0:
            raise PacketValidationError(f"Negative Time-To-Live metric bound: {self.ttl}")
        if self.sequence_number < 0 or self.sequence_number >= self.total_packets:
            raise PacketValidationError(
                f"Invalid segment numbering layout: sequence={self.sequence_number} out of max {self.total_packets}"
            )
        if self.fragment_offset < 0:
            raise PacketValidationError(f"Invalid fragment offset tracking: {self.fragment_offset}")
        if self.retry_count < 0:
            raise PacketValidationError(f"Negative retry initialization count metric context: {self.retry_count}")

    def compute_checksum(self) -> str:
        """Generates SHA256 boundary signatures confirming structural segment consistency."""
        hasher = hashlib.sha256()
        hasher.update(self._payload)
        return hasher.hexdigest()

    def verify_checksum(self) -> bool:
        """Verifies received payloads against recorded cryptosignatures."""
        is_valid = self.compute_checksum() == self.checksum
        if not is_valid:
            logger.error(f"Checksum Failure identified on packet unit layer frame verification instance: {self.packet_id}")
        return is_valid

    def is_corrupted(self) -> bool:
        """Evaluates whether the frame has been modified or broken across transmission channels."""
        return not self.verify_checksum()

    def increment_retry(self) -> None:
        """Increments link transmission attempt counters using clear encapsulation wrappers."""
        self.retry_count += 1
        logger.debug(f"Packet {self.packet_id} retry count incremented to {self.retry_count}")

    def summary(self) -> str:
        """Generates clean structural trace logging summaries."""
        dest = self.destination_node if self.destination_node else "?"
        return (
            f"[{self.packet_id}] | MSG_REF: {self.message_id} | TYPE: {self.packet_type.name} | "
            f"HOP: {self.previous_hop or 'None'} -> {self.current_node} -> {self.next_hop or '?'} | "
            f"PRIORITY: {self.priority.name} | STATUS: {self.status.name}"
        )

    def _assert_transition(self, target: PacketStatus, allowed_origins: List[PacketStatus]) -> None:
        """Enforces clean lifecycle state machine transition progression validation mappings."""
        if self.status not in allowed_origins:
            raise IllegalPacketStateTransition(
                f"Forbidden pipeline tracking shift: cannot advance '{self.packet_id}' to status '{target.name}' from '{self.status.name}'"
            )

    def mark_ready(self) -> None:
        """Advances packet state from CREATED to READY state."""
        self._assert_transition(PacketStatus.READY, [PacketStatus.CREATED])
        self.status = PacketStatus.READY
        logger.debug(f"Packet Ready for transmission scheduling: {self.packet_id}")

    def mark_queued(self) -> None:
        """Places packet into transmission priority queues."""
        self._assert_transition(PacketStatus.QUEUED, [PacketStatus.READY])
        self.status = PacketStatus.QUEUED
        logger.info(f"Packet Queued: {self.packet_id}")

    def mark_transmitting(self) -> None:
        """Marks packet as actively occupying link layer channels."""
        self._assert_transition(PacketStatus.TRANSMITTING, [PacketStatus.QUEUED, PacketStatus.FORWARDED])
        self.status = PacketStatus.TRANSMITTING
        if self.transmission_timestamp is None:
            self.transmission_timestamp = datetime.datetime.now(datetime.timezone.utc)

    def mark_forwarded(self, next_node: str) -> None:
        """Advances packet frame steps to sequential intermediate network hops."""
        self._assert_transition(PacketStatus.FORWARDED, [PacketStatus.TRANSMITTING])
        
        self.route.append(self.current_node)
        self.previous_hop = self.current_node
        self.current_node = next_node
        self.next_hop = None  # Cleared out for next downstream routing evaluation step
        self.visited_nodes.append(next_node)
        self.hop_count += 1
        self.status = PacketStatus.FORWARDED
        logger.info(f"Packet Forwarded: {self.packet_id} reached node {next_node} via hop {self.previous_hop}")

    def mark_delivered(self) -> None:
        """Finalizes transmission tracker pipelines with success statuses."""
        self._assert_transition(PacketStatus.DELIVERED, [PacketStatus.TRANSMITTING])
        self.status = PacketStatus.DELIVERED
        self.delivery_timestamp = datetime.datetime.now(datetime.timezone.utc)
        if self.current_node not in self.route:
            self.route.append(self.current_node)
        logger.info(f"Packet Delivered: {self.packet_id} reached final destination in {self.hop_count} hops.")

    def mark_failed(self) -> None:
        """Flags transport execution pipelines as broken."""
        self.status = PacketStatus.FAILED
        logger.warning(f"Packet Transmission Failure logged: {self.packet_id}")

    def mark_dropped(self) -> None:
        """Drops packet from memory contexts due to buffer limits or path drop-outs."""
        self.status = PacketStatus.DROPPED
        logger.warning(f"Packet Dropped: {self.packet_id}")

    def mark_expired(self) -> None:
        """Applies expired status markers when tracking timers run out."""
        self.status = PacketStatus.EXPIRED
        logger.warning(f"Packet Expired: {self.packet_id} exceeded structural time limits.")

    def decrement_ttl(self) -> None:
        """Reduces tracking TTL hops. Automatically triggers drops upon exhaustion events."""
        if self.status in [PacketStatus.DELIVERED, PacketStatus.DROPPED, PacketStatus.EXPIRED, PacketStatus.FAILED]:
            return
            
        self.ttl -= 1
        if self.ttl <= 0:
            self.mark_expired()
            self.mark_dropped()

    def to_dict(self) -> Dict[str, Any]:
        """Serializes current structural frame contexts into key-value data primitives."""
        return {
            "packet_id": self.packet_id,
            "message_id": self.message_id,
            "packet_type": self.packet_type.name,
            "sequence_number": self.sequence_number,
            "total_packets": self.total_packets,
            "fragment_offset": self.fragment_offset,
            "payload": self.payload.hex(),  # Convert binary stream into transportable hex string representation
            "payload_size_bytes": self.payload_size_bytes,
            "source_node": self.source_node,
            "destination_node": self.destination_node,
            "previous_hop": self.previous_hop,
            "current_node": self.current_node,
            "next_hop": self.next_hop,
            "route": list(self.route),
            "visited_nodes": list(self.visited_nodes),
            "priority": self.priority.name,
            "qos_level": self.qos_level.name,
            "ttl": self.ttl,
            "delivery_deadline_ms": self.delivery_deadline_ms,
            "hop_count": self.hop_count,
            "retry_count": self.retry_count,
            "latency_ms": self.latency_ms,
            "packet_loss_percent": self.packet_loss_percent,
            "jitter_ms": self.jitter_ms,
            "received_signal_strength_dbm": self.received_signal_strength_dbm,
            "transmission_time_ms": self.transmission_time_ms,
            "queue_delay_ms": self.queue_delay_ms,
            "transmission_delay_ms": self.transmission_delay_ms,
            "processing_delay_ms": self.processing_delay_ms,
            "propagation_delay_ms": self.propagation_delay_ms,
            "end_to_end_delay_ms": self.end_to_end_delay_ms,
            "age_ms": self.age_ms,
            "is_successful": self.is_successful,
            "checksum": self.checksum,
            "creation_timestamp": self.creation_timestamp.isoformat(),
            "transmission_timestamp": self.transmission_timestamp.isoformat() if self.transmission_timestamp else None,
            "delivery_timestamp": self.delivery_timestamp.isoformat() if self.delivery_timestamp else None,
            "status": self.status.name,
            "metadata": dict(self.metadata)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NetworkPacket:
        """Reconstructs proper object instances from structured dictionaries."""
        return cls(
            packet_id=data["packet_id"],
            message_id=data["message_id"],
            packet_type=PacketType[data["packet_type"]],
            sequence_number=data["sequence_number"],
            total_packets=data["total_packets"],
            fragment_offset=data["fragment_offset"],
            payload=bytes.fromhex(data["payload"]),
            source_node=data["source_node"],
            priority=MessagePriority[data["priority"]],
            qos_level=QoSLevel[data["qos_level"]],
            ttl=data["ttl"],
            delivery_deadline_ms=data["delivery_deadline_ms"],
            destination_node=data.get("destination_node"),
            previous_hop=data.get("previous_hop"),
            current_node=data.get("current_node"),
            next_hop=data.get("next_hop"),
            route=data.get("route"),
            visited_nodes=data.get("visited_nodes"),
            hop_count=data.get("hop_count", 0),
            retry_count=data.get("retry_count", 0),
            latency_ms=data.get("latency_ms", 0.0),
            packet_loss_percent=data.get("packet_loss_percent", 0.0),
            jitter_ms=data.get("jitter_ms", 0.0),
            received_signal_strength_dbm=data.get("received_signal_strength_dbm", 0.0),
            transmission_time_ms=data.get("transmission_time_ms", 0.0),
            queue_delay_ms=data.get("queue_delay_ms", 0.0),
            transmission_delay_ms=data.get("transmission_delay_ms", 0.0),
            processing_delay_ms=data.get("processing_delay_ms", 0.0),
            propagation_delay_ms=data.get("propagation_delay_ms", 0.0),
            creation_timestamp=datetime.datetime.fromisoformat(data["creation_timestamp"]),
            transmission_timestamp=datetime.datetime.fromisoformat(data["transmission_timestamp"]) if data.get("transmission_timestamp") else None,
            delivery_timestamp=datetime.datetime.fromisoformat(data["delivery_timestamp"]) if data.get("delivery_timestamp") else None,
            status=PacketStatus[data["status"]],
            metadata=data.get("metadata")
        )


if __name__ == "__main__":
    print("\n--- Running services/packet.py Refined Production Verification ---")

    # Fixed Framework Type imports & Initialization mapping 
    from services.message import DisasterMessage, SenderType, MessageStatus, Department, DestinationType
    
    mock_msg = DisasterMessage(
        text="Critical: Flash flood boundary breached near coordinates sector 4. Evacuate.",
        origin_node="V1",
        sender_id="S_04",
        sender_type=SenderType.FIELD_SENSOR
    )
    mock_msg.status = MessageStatus.NEW
    
    # Corrected update_classification argument signatures
    mock_msg.update_classification(
        predicted_class="displaced_people_and_evacuations",
        confidence=0.96,
        probabilities={
            "displaced_people_and_evacuations": 0.96
        }
    )   
    
    # Corrected object runtime attribute injection 
    mock_msg.ttl = 40
    mock_msg.required_bandwidth_mbps = 20.0
    mock_msg.maximum_latency_ms = 40.0
    mock_msg.maximum_packet_loss_percent = 4.0
    mock_msg.minimum_reliability_percent = 95.0
    
    # Corrected Enums injection across pipeline boundaries
    mock_msg.assign_qos(
        priority=MessagePriority.HIGH,
        qos_level=QoSLevel.HIGH,
        department=Department.RELIEF,
        destination_node="H1",
        destination_type=DestinationType.RELIEF_CAMP,
        delivery_deadline_ms=40
    )

    # 2. Network Packet Instantiation Mapping Frame
    payload_chunk = mock_msg.text.encode("utf-8")
    pkt = NetworkPacket(
        packet_id="PKT-000001",
        message_id=mock_msg.id,
        packet_type=PacketType.DATA,
        sequence_number=0,
        total_packets=1,
        fragment_offset=0,
        payload=payload_chunk,
        source_node=mock_msg.origin_node,
        priority=mock_msg.priority,
        qos_level=mock_msg.qos_level,
        ttl=mock_msg.ttl,
        delivery_deadline_ms=mock_msg.delivery_deadline_ms
    )

    print(f"\nInitial State Trace : {pkt.summary()}")
    print(f"Computed Byte Size  : {pkt.payload_size_bytes} bytes (Class Max: {NetworkPacket.MAX_PACKET_SIZE} bytes)")
    assert pkt.is_corrupted() is False, "Checksum initialization mismatch."

    # 3. Running State Machine Lifecycle Verifications
    pkt.mark_ready()
    pkt.mark_queued()
    
    if pkt.status > PacketStatus.READY:
        print(f"✓ IntEnum Success: State check verified that packet state ({pkt.status.name}) is past scheduling phases.")

    pkt.mark_transmitting()
    pkt.destination_node = "H1"
    
    pkt.next_hop = "T2"
    pkt.mark_forwarded(next_node="T2")
    pkt.increment_retry()
    
    pkt.mark_transmitting()
    pkt.next_hop = "R1"
    pkt.mark_forwarded(next_node="R1")
    
    pkt.mark_transmitting()
    pkt.next_hop = "H1"
    pkt.mark_forwarded(next_node="H1")
    
    pkt.mark_transmitting()
    
    pkt.queue_delay_ms = 1.2
    pkt.transmission_delay_ms = 4.5
    pkt.processing_delay_ms = 0.3
    pkt.propagation_delay_ms = 2.1
    
    pkt.mark_delivered()
    
    print(f"\nFinal State Trace   : {pkt.summary()}")
    print(f"Total Traveled Path : {' -> '.join(pkt.visited_nodes)}")
    print(f"Calculated End-Delay: {pkt.end_to_end_delay_ms:.2f} ms")
    print(f"Current Packet Age  : {pkt.age_ms:.4f} ms")
    print(f"Delivery Successful : {pkt.is_successful}")
    
    print("\n=== Serialized Network Packet Object JSON Structure ===")
    import json
    print(json.dumps(pkt.to_dict(), indent=2))