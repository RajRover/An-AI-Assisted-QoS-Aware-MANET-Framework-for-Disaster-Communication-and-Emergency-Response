"""
services/packet_generator.py

This module implements the PacketGenerator for the QoS-Aware MANET Framework.
It translates a validated, QoS-mapped DisasterMessage into a sequence of segmented
or single-frame NetworkPacket instances based on layer capacity constraints.
"""

from __future__ import annotations

import itertools
import logging
from typing import ClassVar, List, Tuple

# Enforce explicit structural dependency boundaries
from services.message import DisasterMessage, MessagePriority, MessageStatus, QoSLevel
from services.packet import NetworkPacket, PacketType


# Setup module-level structured logger
logger = logging.getLogger("services.packet_generator")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class PacketGenerationError(Exception):
    """Raised when validation criteria or lifecycle state checks fail during network packet framing."""
    pass


class PacketGenerator:
    """
    Architectural Factory class responsible for validating a DisasterMessage, encoding 
    its textual payload, executing chunk-offset segmentation, and generating a 
    sequence of bound NetworkPacket objects.
    """
    DEFAULT_ENCODING: ClassVar[str] = "utf-8"
    PACKET_ID_PREFIX: ClassVar[str] = "PKT"

    # Thread-safe global counter across all instances to guarantee globally unique Packet IDs
    _global_counter = itertools.count(1)

    def generate_packets(self, message: DisasterMessage) -> List[NetworkPacket]:
        """
        Translates a fully QoS-assigned DisasterMessage into a list of fragmented or
        single-frame NetworkPacket instances, updating the source message's state machine.
        """
        logger.info(f"Packet Generation Started for message ID: {message.id}")
        
        # 1. State and Dependency Integrity Validation
        self._validate_message(message)
        
        # 2. Byte Stream Encoding Conversion
        try:
            payload_bytes = message.text.encode(self.DEFAULT_ENCODING)
        except Exception as e:
            logger.error(f"Failed to encode message text stream context: {str(e)}")
            raise PacketGenerationError(f"Text encoding violation using {self.DEFAULT_ENCODING}: {str(e)}") from e

        max_chunk_size = NetworkPacket.MAX_PACKET_SIZE
        
        # 3. Payload Fragmentation with Zip-Offset Mapping
        if len(payload_bytes) > max_chunk_size:
            logger.info(f"Fragmentation Started: Payload size {len(payload_bytes)} bytes exceeds max boundary of {max_chunk_size} bytes.")
        chunks_with_offsets = self._fragment_payload(payload_bytes, max_chunk_size)
        total_packets = len(chunks_with_offsets)
        
        # 4. Packet Instantiation Sequence
        packets: List[NetworkPacket] = []
        for sequence_number, (chunk, fragment_offset) in enumerate(chunks_with_offsets):
            packet = self._create_packet(
                message=message,
                chunk=chunk,
                sequence_number=sequence_number,
                total_packets=total_packets,
                fragment_offset=fragment_offset
            )
            packets.append(packet)
            
        # 5. Automated Lifecycle Lifecycle Update
        message.mark_packet_created()
        
        logger.info(f"Packets Created successfully. Total fragments generated: {total_packets}")
        logger.info(f"Packet Generation Finished for message ID: {message.id}. Message status updated to PACKET_CREATED.")
        return packets

    def _validate_message(self, message: DisasterMessage) -> None:
        """Enforces field-level validation invariants and lifecycle rules before framing data."""
        # Standardize set lookup to guard against non-ordered Enum constraints safely
        allowed_states = {
            MessageStatus.QOS_ASSIGNED,
            MessageStatus.PACKET_CREATED,
        }
        
        if message.status not in allowed_states:
            logger.error(f"Validation failure: Message {message.id} is in illegal state '{message.status.name}' for packet conversion.")
            raise PacketGenerationError(f"Precondition Failure: Message status must be QOS_ASSIGNED or PACKET_CREATED. Current state: {message.status.name}")
            
        if message.predicted_class is None:
            logger.error(f"Validation failure: Message {message.id} missing AI classification taxonomy records.")
            raise PacketGenerationError("Cannot generate packets for a message without a valid predicted class.")

        # Validate that QoS attributes are explicit and sound
        if (
            message.priority is None or 
            message.qos_level is None or 
            message.ttl is None or 
            message.delivery_deadline_ms is None or 
            message.destination_type is None
        ):
            logger.error(f"Validation failure: Message {message.id} contains unassigned QoS metadata parameters.")
            raise PacketGenerationError("Precondition Failure: Missing explicit priority, qos_level, ttl, deadline, or destination_type assignments.")

    def _fragment_payload(self, payload: bytes, max_size: int) -> List[Tuple[bytes, int]]:
        """
        Slices raw binary payloads into clean chunk blocks paired with their relative 
        byte offsets within the overall payload structure.
        """
        fragments = []
        for i in range(0, len(payload), max_size):
            chunk = payload[i : i + max_size]
            fragments.append((chunk, i))
        return fragments

    def _generate_packet_id(self) -> str:
        """Returns a globally unique monotonic packet identification sequence string."""
        return f"{self.PACKET_ID_PREFIX}-{next(self._global_counter):06d}"

    def _create_packet(
        self, 
        message: DisasterMessage, 
        chunk: bytes, 
        sequence_number: int, 
        total_packets: int, 
        fragment_offset: int
    ) -> NetworkPacket:
        """Constructs an individual structured and isolated NetworkPacket framework model."""
        packet_id = self._generate_packet_id()
        
        # Multi-tiered classification for forward compatibility handles
        if message.qos_level == QoSLevel.EMERGENCY:
            packet_type = PacketType.EMERGENCY
        elif message.priority == MessagePriority.LOW:
            packet_type = PacketType.CONTROL
        else:
            packet_type = PacketType.DATA
        
        return NetworkPacket(
            packet_id=packet_id,
            message_id=message.id,
            packet_type=packet_type,
            sequence_number=sequence_number,
            total_packets=total_packets,
            fragment_offset=fragment_offset,
            payload=chunk,
            source_node=message.origin_node,
            priority=message.priority,
            qos_level=message.qos_level,
            ttl=message.ttl,
            delivery_deadline_ms=message.delivery_deadline_ms,
            destination_node=message.destination_node
        )


if __name__ == "__main__":
    print("\n--- Running services/packet_generator.py Production Verification ---")
    
    from services.message import SenderType, DestinationType, Department
    
    # 1. Synthesize a mock DisasterMessage layout mirroring latest parameters
    large_payload_text = (
        "INCIDENT REPORT SEQUENCE: " + ("B" * 1200) + " [END TELEMETERED FRAME SECTOR 9]"
    )
    
    msg_mock = DisasterMessage(
        text=large_payload_text,
        origin_node="NODE-HQ-01",
        sender_id="S_OPERATOR_2",
        sender_type=SenderType.COMMAND_CENTER
    )
    
    # 2. Advance the state machine through the workflow pipelines using correct API definitions
    msg_mock.update_classification(
        predicted_class="infrastructure_and_utility_damage", 
        confidence=0.98,
        probabilities={"infrastructure_and_utility_damage": 0.98}
    )
    
    msg_mock.assign_qos(
        priority=MessagePriority.CRITICAL,
        qos_level=QoSLevel.EMERGENCY,
        department=Department.CONTROL_CENTER,
        destination_node="HQ-DESK",
        destination_type=DestinationType.CONTROL_CENTER,
        delivery_deadline_ms=30
    )
    msg_mock.ttl = 32
    
    print(f"Message configured. Status: {msg_mock.status.name}, Length: {len(msg_mock.text)} chars.")
    
    # 3. Process packet generation across distinct factory invocations to verify unique sequence IDs
    gen_alpha = PacketGenerator()
    gen_beta = PacketGenerator()
    
    print(f"\n--- Processing generation via Generator Alpha ---")
    packet_stream_1 = gen_alpha.generate_packets(msg_mock)
    
    print(f"\n--- Processing generation via Generator Beta (Simulated concurrent context) ---")
    # Shift status manually to bypass transition state engine limits on the same reuse context mock
    msg_mock.status = MessageStatus.QOS_ASSIGNED 
    packet_stream_2 = gen_beta.generate_packets(msg_mock)
    
    # 4. Display Unified Verification Reports
    all_generated_packets = packet_stream_1 + packet_stream_2
    print("\n==========================================================")
    print("GLOBAL TRACKING FRAME VERIFICATION REPORT")
    print("==========================================================")
    print(f"Total Packet Objects Evaluated: {len(all_generated_packets)}")
    print(f"Final Underlying Message State : {msg_mock.status.name}")
    
    for pkt in all_generated_packets:
        print(f"\n  [Packet Frame Ref: {pkt.packet_id}]")
        print(f"    Parent Msg ID : {pkt.message_id}")
        print(f"    Protocol Type : {pkt.packet_type.name}")
        print(f"    Seq Context   : {pkt.sequence_number + 1} of {pkt.total_packets}")
        print(f"    Frag Offset   : {pkt.fragment_offset} bytes")
        print(f"    Chunk Size    : {pkt.payload_size_bytes} bytes")
        print(f"    Source Node   : {pkt.source_node}")
        print(f"    Age Telemetry : {pkt.age_ms:.4f} ms")
        print(f"    Summary Line  : {pkt.summary()}")