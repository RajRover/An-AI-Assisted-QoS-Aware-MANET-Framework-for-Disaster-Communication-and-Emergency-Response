"""
integration/communication_pipeline.py

Central orchestrator for the AI-Assisted QoS-Aware MANET Framework.
Coordinates the end-to-end communication lifecycle from message entry to physical delivery.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Strict relative/direct imports per project specification
from services.message import (
    DisasterMessage,
    MessagePriority,
    QoSLevel,
    MessageStatus,
)
from services.qos_mapper import QoSMapper
from services.packet_generator import PacketGenerator
from services.dispatcher import Dispatcher, DispatchResult
from services.packet import NetworkPacket, PacketStatus
from Disaster_Prediction.classifier import DisasterClassifier

logger = logging.getLogger(__name__)


class CommunicationPipelineError(Exception):
    """Raised when an error occurs during any stage of the communication pipeline orchestration."""
    pass


@dataclass
class CommunicationResult:
    """Encapsulates the complete end-to-end context and outcomes of a pipeline execution pass."""
    message: DisasterMessage
    predicted_class: str
    confidence: float
    priority: MessagePriority
    qos_level: QoSLevel
    department: str
    destination_node: Optional[str]
    route: List[str]
    generated_packets: List[NetworkPacket]
    packet_count: int
    delivery_success: bool
    hop_count: int
    latency_ms: float
    packet_loss_percent: float
    jitter_ms: float
    bandwidth_mbps: float
    dispatcher_result: DispatchResult
    
    # Granular profiling metrics
    classification_time_ms: float
    qos_time_ms: float
    packet_generation_time_ms: float
    dispatch_time_ms: float
    total_processing_time_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineStatistics:
    """Maintains running historical statistics across multiple processed text payloads."""
    messages_processed: int = 0
    messages_delivered: int = 0
    messages_failed: int = 0
    delivery_rate: float = 0.0
    average_latency: float = 0.0
    average_hops: float = 0.0
    average_packet_loss: float = 0.0
    average_bandwidth: float = 0.0
    average_jitter: float = 0.0
    average_processing_time: float = 0.0
    average_packets_per_message: float = 0.0


class CommunicationPipeline:
    """Orchestrates the lifecycle of a disaster message through the MANET architectural stack."""

    def __init__(
        self,
        classifier: Optional[DisasterClassifier] = None,
        qos_mapper: Optional[QoSMapper] = None,
        packet_generator: Optional[PacketGenerator] = None,
        dispatcher: Optional[Dispatcher] = None,
        network_state: Optional[Any] = None,
        **kwargs
    ):
        """Pure Dependency Injection constructor with default fallbacks. NetworkState is fully encapsulated inside Dispatcher."""
        self.classifier = classifier if classifier is not None else DisasterClassifier()
        self.qos_mapper = qos_mapper if qos_mapper is not None else QoSMapper()
        self.packet_generator = packet_generator if packet_generator is not None else PacketGenerator()
        
        ns = network_state or kwargs.get("network_state")
        if dispatcher is not None:
            self.dispatcher = dispatcher
            if getattr(self.dispatcher, "network_state", None) is None and ns is not None:
                self.dispatcher.network_state = ns
        else:
            self.dispatcher = Dispatcher(network_state=ns)
        
        # Internal running statistics metrics block
        self._stats = PipelineStatistics()

    def process_message(
        self,
        text: Any,
        source_node: Optional[str] = None,
        sender_type: Optional[Any] = None,
        sender_id: Optional[str] = None
    ) -> CommunicationResult:
        """
        Executes the full pipeline step-by-step:
        Classification -> QoS Mapping -> Packetization -> Target Resolution & Dispatching.
        """
        logger.info("Pipeline started for message parsing.")
        start_pipeline = time.perf_counter()

        # STEP 1: Message Initialization / Resolution
        if isinstance(text, DisasterMessage):
            message = text
            message_text = message.text
        else:
            message_text = text
            try:
                from services.message import SenderType
                if isinstance(sender_type, str):
                    try:
                        s_type = SenderType[sender_type.upper()]
                    except (KeyError, ValueError):
                        s_type = SenderType.CITIZEN
                else:
                    s_type = sender_type or SenderType.CITIZEN

                message = DisasterMessage(
                    text=message_text,
                    origin_node=source_node,
                    sender_type=s_type,
                    sender_id=sender_id
                )
            except Exception as e:
                logger.error(f"Failed inside Message Construction setup: {e}")
                raise CommunicationPipelineError("Message construction step encountered failures.") from e

        # STEP 2: AI Classification Stage
        classification_time = 0.0
        if message.status == MessageStatus.NEW:
            try:
                logger.info("Running AI Classification model optimization...")
                t0 = time.perf_counter()
                classification_res = self.classifier.classify(message_text)
                
                if isinstance(classification_res, dict):
                    pred_class = classification_res["predicted_class"]
                    confidence = classification_res["confidence"]
                    probs = classification_res.get("probabilities", {})
                else:
                    pred_class = classification_res.predicted_class
                    confidence = classification_res.confidence
                    probs = classification_res.probabilities

                message.update_classification(pred_class, confidence, probs)
                classification_time = (time.perf_counter() - t0) * 1000.0
                logger.debug(f"Classifier completed. Assigned: {pred_class} ({confidence*100:.1f}%)")
            except Exception as e:
                logger.error(f"Pipeline failed at Classification stage: {e}")
                raise CommunicationPipelineError("Classification step failed.") from e
        else:
            logger.info("Skipping AI Classification as message is already classified.")

        # STEP 3: QoS Mapping Stage
        qos_time = 0.0
        if message.status == MessageStatus.CLASSIFIED:
            try:
                logger.info("Executing QoS Mapper policy extraction...")
                t0 = time.perf_counter()
                self.qos_mapper.map_message(message)  # Modifies parameters context in place
                qos_time = (time.perf_counter() - t0) * 1000.0
                logger.debug(f"QoS assigned -> Priority: {message.priority}, Dept: {message.department}")
            except Exception as e:
                logger.error(f"Pipeline failed at QoS Mapping stage: {e}")
                raise CommunicationPipelineError("QoS Mapping step failed.") from e
        else:
            logger.info("Skipping QoS Mapping as message already has QoS assigned.")

        # STEP 4: Packet Generation Stage
        try:
            logger.info("Fragmenting data structures via PacketGenerator...")
            t0 = time.perf_counter()
            packets: List[NetworkPacket] = self.packet_generator.generate_packets(message)
            packet_gen_time = (time.perf_counter() - t0) * 1000.0
            logger.debug(f"Packets generated. Sequence Count: {len(packets)}")
        except Exception as e:
            logger.error(f"Pipeline failed at Packet Generation stage: {e}")
            raise CommunicationPipelineError("Packet generation step failed.") from e

        # STEP 5: Dispatcher Handling Destination Target Rerouting & Node Graph Transmit
        try:
            logger.info("Invoking Dispatcher layer topology checks...")
            t0 = time.perf_counter()
            dispatch_result: DispatchResult = self.dispatcher.dispatch_packets(packets, message)
            self.dispatcher.start_transmission()
            dispatch_time = (time.perf_counter() - t0) * 1000.0
            logger.info("Dispatcher execution loop completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline failed at Dispatch stage: {e}")
            raise CommunicationPipelineError("Network packet dispatch step failed.") from e

        total_time_ms = (time.perf_counter() - start_pipeline) * 1000.0

        # Collect metrics from packets
        hop_count = 0
        latency_ms = 0.0
        packet_loss_percent = 0.0
        jitter_ms = 0.0
        bandwidth_mbps = 0.0
        
        if packets:
            hop_count = max((getattr(pkt, "hop_count", 0) for pkt in packets), default=0)
            latency_ms = sum(getattr(pkt, "latency_ms", 0.0) for pkt in packets) / len(packets)
            packet_loss_percent = sum(getattr(pkt, "packet_loss_percent", 0.0) for pkt in packets) / len(packets)
            jitter_ms = max((getattr(pkt, "jitter_ms", 0.0) for pkt in packets), default=0.0)
            bandwidth_mbps = min((getattr(pkt, "min_bandwidth_mbps", 100.0) for pkt in packets), default=100.0)

        # STEP 6: Collect Metrics & Encapsulate into Result Model
        result = CommunicationResult(
            message=message,
            predicted_class=message.predicted_class,
            confidence=message.confidence,
            priority=message.priority,
            qos_level=message.qos_level,
            department=message.department,
            destination_node=dispatch_result.destination_node,
            route=dispatch_result.route,
            generated_packets=packets,
            packet_count=len(packets),
            delivery_success=dispatch_result.success and len(packets) > 0 and all(pkt.status == PacketStatus.DELIVERED for pkt in packets),
            hop_count=hop_count,
            latency_ms=latency_ms,
            packet_loss_percent=packet_loss_percent,
            jitter_ms=jitter_ms,
            bandwidth_mbps=bandwidth_mbps,
            dispatcher_result=dispatch_result,
            classification_time_ms=classification_time,
            qos_time_ms=qos_time,
            packet_generation_time_ms=packet_gen_time,
            dispatch_time_ms=dispatch_time,
            total_processing_time_ms=total_time_ms
        )
        
        # Update pipeline execution counters
        self._update_cumulative_stats(result)
        
        # Refined single-line tracing for quick production telemetry analysis
        logger.info(
            "Pipeline complete | MessageID=%s | Delivered=%s | Route=%d hops | Total=%.2f ms",
            message.id,
            result.delivery_success,
            result.hop_count,
            result.total_processing_time_ms
        )

        return result

    def process_batch(self, batch_inputs: List[tuple]) -> List[CommunicationResult]:
        """
        Processes an aggregated batch of data packets sequentially.
        Maintains resilience by logging individual errors instead of aborting the entire run.
        """
        results = []
        for index, item in enumerate(batch_inputs):
            try:
                text, source_node, sender_type, sender_id = item
                res = self.process_message(text, source_node, sender_type, sender_id)
                results.append(res)
            except Exception as e:
                logger.error(
                    f"Fault Recovery: Skipping corrupted input entry record reference "
                    f"at batch index #{index} due to processing failure: {e}"
                )
                # Continue loop securely to process surrounding crisis messages
                continue
        return results

    def get_statistics(self) -> PipelineStatistics:
        """Retrieves read-only runtime statistics generated across framework operation sessions."""
        return self._stats

    def reset_statistics(self) -> None:
        """Flushes execution records, resetting tracking parameters to clear state baselines."""
        self._stats = PipelineStatistics()
        logger.info("Pipeline performance accumulation registers successfully cleared.")

    def _update_cumulative_stats(self, res: CommunicationResult) -> None:
        """Calculates running averages cleanly while maintaining a hard-calculated delivery_rate."""
        s = self._stats
        n = s.messages_processed
        new_n = n + 1
        
        s.messages_processed = new_n
        if res.delivery_success:
            s.messages_delivered += 1
        else:
            s.messages_failed += 1

        s.delivery_rate = s.messages_delivered / s.messages_processed

        def moving_avg(current_avg, new_val):
            return ((current_avg * n) + new_val) / new_n

        s.average_latency = moving_avg(s.average_latency, res.latency_ms)
        s.average_hops = moving_avg(s.average_hops, res.hop_count)
        s.average_packet_loss = moving_avg(s.average_packet_loss, res.packet_loss_percent)
        s.average_bandwidth = moving_avg(s.average_bandwidth, res.bandwidth_mbps)
        s.average_jitter = moving_avg(s.average_jitter, res.jitter_ms)
        s.average_processing_time = moving_avg(s.average_processing_time, res.total_processing_time_ms)
        s.average_packets_per_message = moving_avg(s.average_packets_per_message, res.packet_count)


# ==============================================================================
# INTEGRATION TEST BENCHMARK RUNNER
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    from communication.graph import build_network
    from simulation.network_state import NetworkState, SimulationConfig
    
    logger.info("Initializing high-fidelity integration suite with full hardware dependency injections...")
    
    try:
        # Build live production-grade system topologies 
        graph = build_network()
        network_state = NetworkState(graph, config=SimulationConfig())
        
        # Instantiate real operational sub-modules directly
        classifier = DisasterClassifier()
        qos_mapper = QoSMapper()
        packet_generator = PacketGenerator()
        dispatcher = Dispatcher(network_state=network_state)
        
        pipeline = CommunicationPipeline(
            classifier=classifier,
            qos_mapper=qos_mapper,
            packet_generator=packet_generator,
            dispatcher=dispatcher
        )
        
        # NOTE: origin_node must be an existing node ID in the graph (see communication/nodes.py).
        # SenderType values must match the SenderType enum in services/message.py.
        sample_scenarios = [
            ("There are injured people trapped inside Building A.", "V1", "CITIZEN", "C_091"),
            ("We need food and drinking water in Village V4.", "V4", "RESCUE_TEAM", "R_202"),
            ("Bridge collapsed after flooding near Highway.", "T2", "DRONE", "D_007"),
        ]
        
        print("\n" + "="*80)
        print("          LIVE PRODUCTION MANET PIPELINE INTEGRATION RUN")
        print("="*80 + "\n")
        
        batch_results = pipeline.process_batch(sample_scenarios)
        
        for idx, output in enumerate(batch_results, 1):
            print(f"--- Processing Real Scenario Message #{idx} ---")
            print(f"  Raw Text        : \"{output.message.text}\"")
            print(f"  Classification  : {output.predicted_class} (Confidence: {output.confidence*100:.1f}%)")
            print(f"  Priority Level  : {output.priority.name} | QoS Level: {output.qos_level.name}")
            print(f"  Destination Node: {output.destination_node}")
            print(f"  Selected Route  : {' -> '.join(output.route)}")
            print(f"  Total Overhead  : {output.total_processing_time_ms:.3f} ms")
            print(f"  Delivery Status : {'SUCCESS' if output.delivery_success else 'FAILED'}\n")
            
        print("="*80)
        print("                        FINAL AGGREGATED STATISTICS")
        print("="*80)
        stats = pipeline.get_statistics()
        print(f"  Total Messages Processed  : {stats.messages_processed}")
        print(f"  Delivery Success Rate     : {stats.delivery_rate * 100:.1f}%")
        print(f"  Mean Latency Profile      : {stats.average_latency:.2f} ms")
        print(f"  Mean Path Hop Constraints : {stats.average_hops:.2f} hops")
        print("="*80 + "\n")

    except Exception as exc:
        logger.error(f"Integration initialization halted: missing concrete dependencies profile. Error: {exc}")