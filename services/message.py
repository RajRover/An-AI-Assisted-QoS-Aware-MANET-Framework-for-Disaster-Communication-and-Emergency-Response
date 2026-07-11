"""
services/message.py

Production-grade data model representing a disaster communication message before 
and after AI-assisted classification and QoS mapping. Follows strict state machine 
transitions, implements PEP8/SOLID principles, and remains fully decoupled from 
routing or packet-generation runtimes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum, IntEnum, auto
import itertools
from typing import Any, Dict, List, Optional

# Setup logger
logger = logging.getLogger("services.message")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class MessageStatus(Enum):
    """Lifecycle states of a disaster communication pipeline."""
    NEW = auto()
    CLASSIFIED = auto()
    QOS_ASSIGNED = auto()
    PACKET_CREATED = auto()
    QUEUED = auto()
    TRANSMITTING = auto()
    DELIVERED = auto()
    FAILED = auto()
    EXPIRED = auto()


class MessagePriority(IntEnum):
    """Urgency level assigned by QoS/AI classifiers.

    IntEnum enables direct logical sorting and comparison operators in network queues.
    """
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class QoSLevel(IntEnum):
    """Network-level operational priority classes.

    IntEnum enables direct threshold comparisons across downstream routers.
    """
    BEST_EFFORT = 1
    NORMAL = 2
    HIGH = 3
    EMERGENCY = 4


class MessageType(Enum):
    """The architectural form-factor of the message payload data."""
    TEXT = auto()
    SENSOR = auto()
    IMAGE = auto()
    VIDEO = auto()
    ALERT = auto()


class SenderType(Enum):
    CITIZEN = auto()
    FIELD_SENSOR = auto()
    DRONE = auto()
    AMBULANCE = auto()
    FIRE_TRUCK = auto()
    POLICE = auto()
    RESCUE_TEAM = auto()
    MEDICAL_TEAM = auto()
    COMMAND_CENTER = auto()
    HOSPITAL = auto()
    RELIEF_TEAM = auto()


class DestinationType(Enum):
    """Strict types representing target receiving environments."""
    HOSPITAL = auto()
    POLICE_STATION = auto()
    FIRE_STATION = auto()
    CONTROL_CENTER = auto()
    RELIEF_CAMP = auto()
    FIELD_TEAM = auto()


class Department(Enum):
    """Responding agencies mapping to localized field incidents."""
    MEDICAL = auto()
    FIRE = auto()
    POLICE = auto()
    RELIEF = auto()
    CONTROL_CENTER = auto()
    SEARCH_AND_RESCUE = auto()


@dataclass
class DisasterMessage:
    """
    Pure Data Model capturing text payloads, routing targets, mobile originators,
    and runtime performance metrics for disaster response MANET topologies.
    """
    text: str
    origin_node: str               # Physical static/mobile anchor where message generated (e.g., 'V1')
    sender_id: str                 # Unique id of sender entity (e.g., 'A1', 'RT3')
    sender_type: SenderType        # Structured archetype enum avoiding typos
    
    # Optional / Downstream Pipeline Parameters
    message_type: MessageType = MessageType.TEXT
    destination_node: Optional[str] = None
    destination_type: Optional[DestinationType] = None
    
    # Internal Global & Log-Friendly Tracking IDs
    id: str = field(default_factory=lambda: f"MSG-{next(DisasterMessage._id_generator):06d}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    # AI Classifier Outputs
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    classifier_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # QoS Metadata Framework
    priority: Optional[MessagePriority] = None
    qos_level: Optional[QoSLevel] = None
    department: Optional[Department] = None
    
    # Dispatch & Routing Topology Details
    ttl: int = 30
    delivery_deadline_ms: Optional[int] = None
    assigned_vehicle: Optional[str] = None
    route: List[str] = field(default_factory=list)
    
    # Transmission Network Metrics
    latency_ms: float = 0.0
    packet_loss: float = 0.0
    bandwidth: float = 0.0
    jitter: float = 0.0
    
    # Dynamic properties injected dynamically by Mapper layer
    required_bandwidth_mbps: float = 0.0
    maximum_packet_loss_percent: float = 0.0
    minimum_reliability_percent: float = 0.0
    maximum_latency_ms: float = 0.0
    
    # Internal Lifecycle & Retry Parameters
    max_retries: int = 3
    _delivery_attempts: int = 0
    status: MessageStatus = MessageStatus.NEW
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Class-level counter for human-readable logging IDs
    _id_generator = itertools.count(1)

    def __post_init__(self) -> None:
        """Enforces field constraints immediately after dataclass instantiation."""
        if not self.text or not self.text.strip():
            raise ValueError("Message text cannot be empty, None, or whitespace-only.")

    @property
    def hop_count(self) -> int:
        """Dynamically derived value calculated exclusively from the route payload footprint."""
        return max(0, len(self.route) - 1)

    @property
    def delivery_attempts(self) -> int:
        """Read-only exposure of self-contained verification state counter."""
        return self._delivery_attempts

    def increment_delivery_attempt(self) -> None:
        """Encapsulates and mutates delivery effort increments internally."""
        self._delivery_attempts += 1
        logger.debug(f"Message {self.id} transmission attempt incremented to {self._delivery_attempts}")

    def can_retry(self) -> bool:
        """Determines whether the message framework is allowed to schedule further packet retries."""
        return self._delivery_attempts < self.max_retries

    def summary(self) -> str:
        """Returns a highly scannable, single-line log summary."""
        cls_name = self.predicted_class if self.predicted_class else "Unclassified"
        prio_name = self.priority.name if self.priority else "None"
        dest_name = self.destination_node if self.destination_node else "None"
        dest_type_name = self.destination_type.name if self.destination_type else "None"
        return f"{self.id} | {cls_name} | Priority {prio_name} | Origin {self.origin_node} -> Destination {dest_name} ({dest_type_name})"

    def _validate_transition(self, allowed_previous_states: List[MessageStatus], target_state: MessageStatus) -> None:
        """Strict structural integrity validation for state progression."""
        if self.status not in allowed_previous_states:
            raise IllegalStateTransitionError(
                f"Invalid transition for {self.id}: Cannot progress from {self.status.name} to {target_state.name}."
            )

    def update_classification(self, predicted_class: str, confidence: float, probabilities: Dict[str, float]) -> None:
        """Updates metadata populated by the AI Classifier layer."""
        self._validate_transition([MessageStatus.NEW], MessageStatus.CLASSIFIED)
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.classifier_probabilities = probabilities
        self.status = MessageStatus.CLASSIFIED
        logger.info(f"Classification Updated: ID={self.id} | Class={predicted_class} | Conf={confidence:.2f}")

    def assign_qos(self, priority: MessagePriority, qos_level: QoSLevel, department: Department, 
                   destination_node: str, destination_type: DestinationType, delivery_deadline_ms: Optional[int] = None) -> None:
        """Transforms classification boundaries into precise network constraints."""
        self._validate_transition([MessageStatus.CLASSIFIED], MessageStatus.QOS_ASSIGNED)
        self.priority = priority
        self.qos_level = qos_level
        self.department = department
        self.destination_node = destination_node
        self.destination_type = destination_type
        self.delivery_deadline_ms = delivery_deadline_ms
        self.status = MessageStatus.QOS_ASSIGNED
        logger.info(f"QoS Assigned: ID={self.id} | Priority={priority.name} | QoS={qos_level.name} | Dept={department.name}")

    def mark_packet_created(self) -> None:
        """Signals that separate network frames have been abstractly built around this message wrapper."""
        self._validate_transition([MessageStatus.QOS_ASSIGNED], MessageStatus.PACKET_CREATED)
        self.status = MessageStatus.PACKET_CREATED
        logger.info(f"Packet Created: ID={self.id}")

    def mark_queued(self) -> None:
        """Signals that packet chunks have entered the priority queue buffers."""
        self._validate_transition([MessageStatus.PACKET_CREATED], MessageStatus.QUEUED)
        self.status = MessageStatus.QUEUED

    def mark_transmitting(self) -> None:
        """Signals that payload fragments have left the queue layers into the ether."""
        self._validate_transition([MessageStatus.QUEUED, MessageStatus.TRANSMITTING], MessageStatus.TRANSMITTING)
        self.status = MessageStatus.TRANSMITTING

    def mark_delivered(self, final_metrics: Optional[Dict[str, float]] = None) -> None:
        """Confirms successful framing termination at the destination node network interface."""
        self._validate_transition([MessageStatus.TRANSMITTING], MessageStatus.DELIVERED)
        if final_metrics:
            self.latency_ms = final_metrics.get("latency_ms", self.latency_ms)
            self.packet_loss = final_metrics.get("packet_loss", self.packet_loss)
            self.bandwidth = final_metrics.get("bandwidth", self.bandwidth)
            self.jitter = final_metrics.get("jitter", self.jitter)
            
        self.status = MessageStatus.DELIVERED
        logger.info(f"Delivered: ID={self.id} reached target {self.destination_node} in {self.hop_count} hops")

    def mark_failed(self) -> None:
        """Flags transmission catastrophic failure or path dropoff limits."""
        self.status = MessageStatus.FAILED
        logger.warning(f"Failed: ID={self.id} transmission dropped or unroutable.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes structure into JSON-compliant dictionary elements."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "origin_node": self.origin_node,
            "destination_node": self.destination_node,
            "destination_type": self.destination_type.name if self.destination_type else None,
            "sender_id": self.sender_id,
            "sender_type": self.sender_type.name,
            "text": self.text,
            "message_type": self.message_type.name,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "classifier_probabilities": self.classifier_probabilities,
            "priority": self.priority.name if self.priority else None,
            "qos_level": self.qos_level.name if self.qos_level else None,
            "department": self.department.name if self.department else None,
            "ttl": self.ttl,
            "delivery_deadline_ms": self.delivery_deadline_ms,
            "assigned_vehicle": self.assigned_vehicle,
            "route": self.route,
            "metrics": {
                "latency_ms": self.latency_ms,
                "packet_loss": self.packet_loss,
                "bandwidth": self.bandwidth,
                "jitter": self.jitter,
                "hop_count": self.hop_count,
            },
            "required_bandwidth_mbps": self.required_bandwidth_mbps,
            "maximum_packet_loss_percent": self.maximum_packet_loss_percent,
            "minimum_reliability_percent": self.minimum_reliability_percent,
            "maximum_latency_ms": self.maximum_latency_ms,
            "max_retries": self.max_retries,
            "delivery_attempts": self.delivery_attempts,
            "status": self.status.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisasterMessage":
        """Reconstructs full operational entities using native serialized formats."""
        metrics = data.get("metrics", {})
        msg = cls(
            text=data["text"],
            origin_node=data["origin_node"],
            sender_id=data["sender_id"],
            sender_type=SenderType[data["sender_type"]],
            message_type=MessageType[data.get("message_type", "TEXT")],
            destination_node=data.get("destination_node"),
            destination_type=DestinationType[data["destination_type"]] if data.get("destination_type") else None,
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            predicted_class=data.get("predicted_class"),
            confidence=data.get("confidence"),
            classifier_probabilities=data.get("classifier_probabilities", {}),
            priority=MessagePriority[data["priority"]] if data.get("priority") else None,
            qos_level=QoSLevel[data["qos_level"]] if data.get("qos_level") else None,
            department=Department[data["department"]] if data.get("department") else None,
            ttl=data.get("ttl", 30),
            delivery_deadline_ms=data.get("delivery_deadline_ms"),
            assigned_vehicle=data.get("assigned_vehicle"),
            route=data.get("route", []),
            latency_ms=metrics.get("latency_ms", 0.0),
            packet_loss=metrics.get("packet_loss", 0.0),
            bandwidth=metrics.get("bandwidth", 0.0),
            jitter=metrics.get("jitter", 0.0),
            required_bandwidth_mbps=data.get("required_bandwidth_mbps", 0.0),
            maximum_packet_loss_percent=data.get("maximum_packet_loss_percent", 0.0),
            minimum_reliability_percent=data.get("minimum_reliability_percent", 0.0),
            maximum_latency_ms=data.get("maximum_latency_ms", 0.0),
            max_retries=data.get("max_retries", 3),
            status=MessageStatus[data["status"]],
            metadata=data.get("metadata", {}),
        )
        msg._delivery_attempts = data.get("delivery_attempts", 0)
        return msg


class IllegalStateTransitionError(Exception):
    """Exception raised when an invalid state transition is requested in the message lifecycle."""
    pass


if __name__ == "__main__":
    print("\n--- Running services/message.py High-Fidelity Invariance Smoke Test ---\n")
    msg = DisasterMessage(
        text="Critical infrastructure broken near flooded area, 3 ambulances required immediately!",
        origin_node="V1",
        sender_id="D2",
        sender_type=SenderType.DRONE
    )
    print(f"Initial Phase Summary: {msg.summary()}")