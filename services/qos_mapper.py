"""
services/qos_mapper.py

This module acts as the mapping layer translating high-level AI classification
outputs into concrete network communication requirements and traffic constraints
for the MANET routing engine. It modifies message structures in-place.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Final, Optional

# Import the standalone data model types
from .message import (
    Department,
    DestinationType,
    DisasterMessage,
    MessagePriority,
    MessageStatus,
    QoSLevel,
    SenderType,
)

# Setup logger
logger = logging.getLogger("services.qos_mapper")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class QoSMappingError(Exception):
    """Base exception for mapping failures within the QoS Translation engine."""
    pass


class UnknownDisasterClassError(QoSMappingError):
    """Raised when the message contains a class prediction missing from the static profiles."""
    pass


@dataclass(frozen=True)
class QoSProfile:
    """
    Immutable specification mapping an operational threat/incident class
    to strict MANET layer transport policies and service level thresholds.
    """
    priority: MessagePriority
    qos_level: QoSLevel
    department: Department
    destination_type: DestinationType
    delivery_deadline_ms: int
    ttl: int
    required_bandwidth_mbps: float
    maximum_packet_loss_percent: float
    minimum_reliability_percent: float
    maximum_latency_ms: float
    retry_limit: int = 3


# Authority Profile Configuration Table Mapping Classifier Output Classes directly to Quality of Service Enforcements
CLASS_TO_QOS: Final[Dict[str, QoSProfile]] = {
    "injured_or_dead_people": QoSProfile(
        priority=MessagePriority.CRITICAL,
        qos_level=QoSLevel.EMERGENCY,
        department=Department.MEDICAL,
        destination_type=DestinationType.HOSPITAL,
        delivery_deadline_ms=20,
        ttl=30,
        required_bandwidth_mbps=50.0,
        maximum_packet_loss_percent=1.0,
        minimum_reliability_percent=99.0,
        maximum_latency_ms=20.0,
        retry_limit=5
    ),
    "requests_or_urgent_needs": QoSProfile(
        priority=MessagePriority.CRITICAL,
        qos_level=QoSLevel.EMERGENCY,
        department=Department.RELIEF,
        destination_type=DestinationType.RELIEF_CAMP,
        delivery_deadline_ms=30,
        ttl=30,
        required_bandwidth_mbps=30.0,
        maximum_packet_loss_percent=2.0,
        minimum_reliability_percent=98.0,
        maximum_latency_ms=30.0,
        retry_limit=5
    ),
    "infrastructure_and_utility_damage": QoSProfile(
        priority=MessagePriority.HIGH,
        qos_level=QoSLevel.HIGH,
        department=Department.CONTROL_CENTER,
        destination_type=DestinationType.CONTROL_CENTER,
        delivery_deadline_ms=50,
        ttl=40,
        required_bandwidth_mbps=25.0,
        maximum_packet_loss_percent=3.0,
        minimum_reliability_percent=95.0,
        maximum_latency_ms=50.0,
        retry_limit=3
    ),
    "missing_or_found_people": QoSProfile(
        priority=MessagePriority.HIGH,
        qos_level=QoSLevel.HIGH,
        department=Department.POLICE,
        destination_type=DestinationType.POLICE_STATION,
        delivery_deadline_ms=40,
        ttl=40,
        required_bandwidth_mbps=20.0,
        maximum_packet_loss_percent=4.0,
        minimum_reliability_percent=95.0,
        maximum_latency_ms=40.0,
        retry_limit=3
    ),
    "displaced_people_and_evacuations": QoSProfile(
        priority=MessagePriority.HIGH,
        qos_level=QoSLevel.HIGH,
        department=Department.RELIEF,
        destination_type=DestinationType.RELIEF_CAMP,
        delivery_deadline_ms=40,
        ttl=40,
        required_bandwidth_mbps=20.0,
        maximum_packet_loss_percent=4.0,
        minimum_reliability_percent=95.0,
        maximum_latency_ms=40.0,
        retry_limit=3
    ),
    "rescue_volunteering_or_donation_effort": QoSProfile(
        priority=MessagePriority.MEDIUM,
        qos_level=QoSLevel.NORMAL,
        department=Department.RELIEF,
        destination_type=DestinationType.RELIEF_CAMP,
        delivery_deadline_ms=100,
        ttl=60,
        required_bandwidth_mbps=15.0,
        maximum_packet_loss_percent=5.0,
        minimum_reliability_percent=90.0,
        maximum_latency_ms=100.0,
        retry_limit=3
    ),
    "caution_and_advice": QoSProfile(
        priority=MessagePriority.MEDIUM,
        qos_level=QoSLevel.NORMAL,
        department=Department.CONTROL_CENTER,
        destination_type=DestinationType.CONTROL_CENTER,
        delivery_deadline_ms=100,
        ttl=60,
        required_bandwidth_mbps=10.0,
        maximum_packet_loss_percent=5.0,
        minimum_reliability_percent=90.0,
        maximum_latency_ms=100.0,
        retry_limit=3
    ),
    "sympathy_and_support": QoSProfile(
        priority=MessagePriority.LOW,
        qos_level=QoSLevel.BEST_EFFORT,
        department=Department.RELIEF,
        destination_type=DestinationType.RELIEF_CAMP,
        delivery_deadline_ms=300,
        ttl=120,
        required_bandwidth_mbps=5.0,
        maximum_packet_loss_percent=10.0,
        minimum_reliability_percent=80.0,
        maximum_latency_ms=300.0,
        retry_limit=2
    ),
    "other_relevant_information": QoSProfile(
        priority=MessagePriority.LOW,
        qos_level=QoSLevel.BEST_EFFORT,
        department=Department.CONTROL_CENTER,
        destination_type=DestinationType.CONTROL_CENTER,
        delivery_deadline_ms=300,
        ttl=120,
        required_bandwidth_mbps=5.0,
        maximum_packet_loss_percent=10.0,
        minimum_reliability_percent=80.0,
        maximum_latency_ms=300.0,
        retry_limit=2
    ),
    "not_humanitarian": QoSProfile(
        priority=MessagePriority.LOW,
        qos_level=QoSLevel.BEST_EFFORT,
        department=Department.CONTROL_CENTER,
        destination_type=DestinationType.CONTROL_CENTER,
        delivery_deadline_ms=500,
        ttl=120,
        required_bandwidth_mbps=1.0,
        maximum_packet_loss_percent=15.0,
        minimum_reliability_percent=70.0,
        maximum_latency_ms=500.0,
        retry_limit=1
    ),
}


class QoSMapper:
    """Decision Engine translating predictive class labels into network policies."""

    @staticmethod
    def get_profile(class_name: str) -> QoSProfile:
        """
        Retrieves the immutable QoS profile associated with a disaster classification.
        Useful for downstream routing optimization, analytics, and testing.
        """
        if class_name not in CLASS_TO_QOS:
            raise UnknownDisasterClassError(
                f"No QoS Profile registered for incident classification: '{class_name}'"
            )
        return CLASS_TO_QOS[class_name]

    @staticmethod
    def map(message: DisasterMessage) -> DisasterMessage:
        """Alias for map_message to support pipeline integration contracts."""
        return QoSMapper.map_message(message)

    @staticmethod
    def map_message(message: DisasterMessage) -> DisasterMessage:
        """
        Validates the message status, extracts the predicted class mapping,
        and modifies the existing DisasterMessage wrapper with network operational constraints.
        """
        logger.info(f"QoS Mapping Started: ID={message.id}")

        if message.status != MessageStatus.CLASSIFIED or message.predicted_class is None:
            logger.error(f"Mapping Error: Message {message.id} is unclassified.")
            raise QoSMappingError(
                f"Cannot map QoS policies for message {message.id}: status must be CLASSIFIED."
            )

        # Retrieve profile safely using the encapsulated getter
        try:
            profile: QoSProfile = QoSMapper.get_profile(message.predicted_class)
        except UnknownDisasterClassError as e:
            logger.error(f"Mapping Error: {e}")
            raise

        logger.info(f"Profile Selected: Class={message.predicted_class} -> Priority={profile.priority.name}")

        # Bind link parameters directly into first-class message fields
        message.ttl = profile.ttl
        message.max_retries = profile.retry_limit
        message.required_bandwidth_mbps = profile.required_bandwidth_mbps
        message.maximum_packet_loss_percent = profile.maximum_packet_loss_percent
        message.minimum_reliability_percent = profile.minimum_reliability_percent
        message.maximum_latency_ms = profile.maximum_latency_ms

        # Transition the state machine. Left destination_node as None (or its pre-assigned value) 
        # so that downstream Dispatchers can evaluate the appropriate localized target mapping (e.g., nearest H1).
        message.assign_qos(
            priority=profile.priority,
            qos_level=profile.qos_level,
            department=profile.department,
            destination_node=message.destination_node,  # Retains clean separation of concerns (None)
            destination_type=profile.destination_type,
            delivery_deadline_ms=profile.delivery_deadline_ms
        )

        logger.info(f"QoS Assigned: ID={message.id} | Status={message.status.name}")
        return message


if __name__ == "__main__":
    print("\n--- Running services/qos_mapper.py Production Verification ---\n")

    # 1. Instantiating a clean new message instance
    test_msg = DisasterMessage(
        text="Casualties reported following apartment structural failures. Dispatch medical units!",
        origin_node="V1",
        sender_id="RT_03",
        sender_type=SenderType.RESCUE_TEAM
    )
    
    # 2. Mocking Classifier execution output step
    test_msg.status = MessageStatus.NEW  # explicit baseline
    test_msg.update_classification(
        predicted_class="injured_or_dead_people",
        confidence=0.985,
        probabilities={"injured_or_dead_people": 0.985, "requests_or_urgent_needs": 0.015}
    )

    # 3. Executing the QoS Mapper Engine
    try:
        updated_msg = QoSMapper.map_message(test_msg)
        
        print("\n=== Extracted Mapping Verification ===")
        print(f"Priority Level       : {updated_msg.priority} (Value: {updated_msg.priority.value})")
        print(f"QoS Level Channel    : {updated_msg.qos_level} (Value: {updated_msg.qos_level.value})")
        print(f"Target Department    : {updated_msg.department}")
        print(f"Destination Type     : {updated_msg.destination_type}")
        print(f"Destination Node ID  : {updated_msg.destination_node} (Handled cleanly by Dispatcher/Routing)")
        print(f"Delivery Deadline    : {updated_msg.delivery_deadline_ms} ms")
        print(f"Time To Live (TTL)   : {updated_msg.ttl} ticks")
        print(f"Retry Threshold Count: {updated_msg.max_retries} attempts")
        print(f"Required Bandwidth   : {updated_msg.required_bandwidth_mbps} Mbps")
        print(f"Max Latency Target   : {updated_msg.maximum_latency_ms} ms")
        print(f"Max Packet Loss Spec : {updated_msg.maximum_packet_loss_percent}%")
        print(f"Min Reliability Spec : {updated_msg.minimum_reliability_percent}%")
        
        print("\n=== Final Serialized Message Object ===")
        import json
        print(json.dumps(updated_msg.to_dict(), indent=2))
        
    except QoSMappingError as error:
        print(f"❌ Verification Failure: Unexpected Exception encountered: {error}")