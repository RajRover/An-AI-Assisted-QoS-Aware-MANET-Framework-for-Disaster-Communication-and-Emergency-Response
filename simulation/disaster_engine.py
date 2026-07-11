"""
disaster_engine.py

Provides the declarative lifecycle state tracking, spatio-temporal boundary scaling,
and high-fidelity discrete event emission structures (`DisasterEngine`) for the 
AI-Assisted QoS-Aware MANET Framework for Disaster Communication.

Author: Simulation Framework Designer & Software Architect
License: MIT
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Set, Tuple

# Import cross-module dependency configuration layers
from simulation.disaster_profiles import DisasterProfile, DisasterType, DisasterProfileManager
from simulation.event_scheduler import EventScheduler, Event, EventPriority, EventType
from simulation.network_state import NetworkState

# Configure Module Logger
logger = logging.getLogger("DisasterMANET.DisasterEngine")


# ============================================================================
# ENUMS & EXCEPTIONS
# ============================================================================

@unique
class DisasterStage(Enum):
    """Defines the discrete tracking phases of a dynamic disaster lifecycle."""
    CREATED = "CREATED"
    STARTING = "STARTING"
    ACTIVE = "ACTIVE"
    PEAK = "PEAK"
    RECOVERING = "RECOVERING"
    FINISHED = "FINISHED"


class DisasterEngineException(Exception):
    """Base exception class for all architectural disaster tracking anomalies."""
    pass


class DisasterNotFoundError(DisasterEngineException):
    """Raised when performing operations on an untracked or unregistered incident ID."""
    pass


class InvalidDisasterStateConfigurationError(DisasterEngineException):
    """Raised when a baseline severity scale or duration violates domain boundaries."""
    pass


# ============================================================================
# DISASTER INSTANCE DATA STRUCTURE
# ============================================================================

@dataclass
class DisasterInstance:
    """
    Tracks state, geographical bounds, severity scales, and event registries
    for an ongoing localized disaster scenario.
    """
    disaster_id: str
    profile: DisasterProfile
    severity_level: int                         # Normalized integer tier scale [1, 5]
    affected_zones: List[str]
    start_tick: int
    duration_ticks: int
    affected_nodes: Set[str] = field(default_factory=set)
    affected_links: Set[Tuple[str, str]] = field(default_factory=set)
    current_stage: DisasterStage = DisasterStage.CREATED
    active: bool = False
    end_tick: int = field(init=False)

    def __post_init__(self) -> None:
        """Enforces operational constraint bounds validation rules."""
        if not (1 <= self.severity_level <= 5):
            raise InvalidDisasterStateConfigurationError(
                f"Severity level {self.severity_level} out of bounds. Must be an integer tier in [1, 5]."
            )
        if self.start_tick < 0 or self.duration_ticks <= 0:
            raise InvalidDisasterStateConfigurationError("Simulation baseline timeline values must be positive integers.")
        self.end_tick = self.start_tick + self.duration_ticks


# ============================================================================
# MAIN DISASTER LIFECYCLE ENGINE CONTROLLER
# ============================================================================

class DisasterEngine:
    """
    The orchestrator module responsible for creating disaster instances, calculating
    dynamic topological footprints, and scheduling events to execute lifecycle phases.
    """

    def __init__(
        self,
        network_state: NetworkState,
        event_scheduler: EventScheduler,
        network_updater: Any,  # Kept generic to prevent direct type circularity
        profile_manager: DisasterProfileManager
    ) -> None:
        self._state: NetworkState = network_state
        self._scheduler: EventScheduler = event_scheduler
        self._updater: Any = network_updater
        self._profiles: DisasterProfileManager = profile_manager
        
        self._active_incidents: Dict[str, DisasterInstance] = {}
        logger.info("Disaster Lifecycle Engine successfully initialized and bound to operational layers.")

    # ------------------------------------------------------------------------
    # API LIFECYCLE CONTROLLERS
    # ------------------------------------------------------------------------

    def create_disaster(
        self,
        disaster_id: str,
        disaster_type: DisasterType,
        severity_level: int,
        affected_zones: List[str],
        start_tick: int,
        duration_ticks: int
    ) -> DisasterInstance:
        """Constructs and registers a new disaster instance, scheduling its initial activation."""
        if disaster_id in self._active_incidents:
            raise InvalidDisasterStateConfigurationError(f"Incident ID collision: '{disaster_id}' already tracks an active event.")
        
        profile = self._profiles.get_profile(disaster_type)
        self._validate_zones(affected_zones)

        instance = DisasterInstance(
            disaster_id=disaster_id,
            profile=profile,
            severity_level=severity_level,
            affected_zones=affected_zones,
            start_tick=start_tick,
            duration_ticks=duration_ticks
        )

        self._active_incidents[disaster_id] = instance
        logger.info(f"Disaster Created -> ID: '{disaster_id}' | Profile: {disaster_type.name} | Targeted Zones: {affected_zones}")

        # Schedule discrete execution boundaries via explicit signature callbacks
        self._schedule_lifecycle_milestone(disaster_id, start_tick, self.start_disaster, "STARTING")
        
        return instance

    def start_disaster(self, disaster_id: str) -> None:
        """Transitions an incident to the STARTING stage and determines its footprint."""
        instance = self._get_tracked_instance(disaster_id)
        instance.current_stage = DisasterStage.STARTING
        instance.active = True
        logger.warning(f"Disaster Started -> Incident: '{disaster_id}' is now entering active tracking space.")

        # Resolve spatial topology targeting across regional attributes
        self._resolve_topological_footprint(instance)

        # Emit foundational high-priority structural degradation tasks
        self._emit_degradation_events(instance)

        # Schedule subsequent lifecycle state machine expansions
        expansion_tick = instance.start_tick + max(1, int(instance.duration_ticks * 0.15))
        self._schedule_lifecycle_milestone(disaster_id, expansion_tick, self.expand_disaster, "ACTIVE")

    def expand_disaster(self, disaster_id: str) -> None:
        """Models the spatial propagation or intensification of the disaster."""
        instance = self._get_tracked_instance(disaster_id)
        if not instance.active:
            return
            
        instance.current_stage = DisasterStage.ACTIVE
        logger.warning(f"Disaster Expanded -> Incident: '{disaster_id}' propagating through zone fields.")

        scaled_profile = self._generate_scaled_profile(instance, severity_modifier=1.2)
        
        self._scheduler.schedule_event(Event(
            event_id=f"{disaster_id}_expansion_apply",
            event_name=f"Apply Expanded Impact: {disaster_id}",
            scheduled_tick=instance.start_tick,
            priority=EventPriority.HIGH,
            event_type=EventType.DISASTER,
            callback=self._updater.apply_disaster_effects,
            parameters={"profile": scaled_profile, "target_nodes": list(instance.affected_nodes)}
        ))

        peak_tick = instance.start_tick + int(instance.duration_ticks * 0.45)
        self._schedule_lifecycle_milestone(disaster_id, peak_tick, self.peak_disaster, "PEAK")

    def peak_disaster(self, disaster_id: str) -> None:
        """Triggers peak structural disruption across the target zone footprint."""
        instance = self._get_tracked_instance(disaster_id)
        if not instance.active:
            return

        instance.current_stage = DisasterStage.PEAK
        logger.error(f"Disaster Peak -> Incident: '{disaster_id}' has reached peak structural disruption metrics.")

        scaled_profile = self._generate_scaled_profile(instance, severity_modifier=1.5)
        
        self._scheduler.schedule_event(Event(
            event_id=f"{disaster_id}_peak_apply",
            event_name=f"Apply Peak Structural Degradation: {disaster_id}",
            scheduled_tick=instance.start_tick,
            priority=EventPriority.CRITICAL,
            event_type=EventType.DISASTER,
            callback=self._updater.apply_disaster_effects,
            parameters={"profile": scaled_profile, "target_nodes": list(instance.affected_nodes)}
        ))

        recovery_start_tick = instance.start_tick + int(instance.duration_ticks * 0.70)
        self._schedule_lifecycle_milestone(disaster_id, recovery_start_tick, self.recover_disaster, "RECOVERING")

    def recover_disaster(self, disaster_id: str) -> None:
        """Initiates a gradual phased recovery sequence across the asset registers."""
        instance = self._get_tracked_instance(disaster_id)
        if not instance.active:
            return

        instance.current_stage = DisasterStage.RECOVERING
        logger.warning(f"Recovery Started -> Phased network restoration initiated for incident: '{disaster_id}'.")

        ticks_remaining = instance.end_tick - (instance.start_tick + int(instance.duration_ticks * 0.70))
        step_interval = max(1, int(ticks_remaining / 3))

        for idx, target_tick in enumerate(range(instance.end_tick - ticks_remaining, instance.end_tick, step_interval)):
            self._scheduler.schedule_event(Event(
                event_id=f"{disaster_id}_recovery_step_{idx}",
                event_name=f"Phased Recovery Step {idx} | Incident: {disaster_id}",
                scheduled_tick=target_tick,
                priority=EventPriority.NORMAL,
                event_type=EventType.SYSTEM,
                callback=self._execute_stepped_recovery_callback,
                parameters={"disaster_id": disaster_id, "step_index": idx}
            ))

        self._schedule_lifecycle_milestone(disaster_id, instance.end_tick, self.finish_disaster, "FINISHED")

    def finish_disaster(self, disaster_id: str) -> None:
        """Concludes tracking parameters and executes a complete structural rollback to baseline defaults."""
        instance = self._get_tracked_instance(disaster_id)
        instance.current_stage = DisasterStage.FINISHED
        instance.active = False
        
        logger.info(f"Recovery Finished -> Infrastructure restoration verification complete for: '{disaster_id}'.")

        # Fixed signature mapping compatibility call for restore_network
        self._scheduler.schedule_event(Event(
            event_id=f"{disaster_id}_final_restitution",
            event_name=f"Full Network Restitution Rollback: {disaster_id}",
            scheduled_tick=instance.end_tick,
            priority=EventPriority.HIGH,
            event_type=EventType.SYSTEM,
            callback=self._updater.restore_network,
            parameters={}
        ))

    def cancel_disaster(self, disaster_id: str) -> None:
        """Terminates an active tracking block immediately, bypassing ongoing scheduling pipelines."""
        instance = self._get_tracked_instance(disaster_id)
        instance.active = False
        logger.warning(f"Disaster Cancelled -> Core execution thread for tracking block '{disaster_id}' revoked manually.")

    def get_active_disasters(self) -> List[DisasterInstance]:
        return [inst for inst in self._active_incidents.values() if inst.active]

    def get_disaster(self, disaster_id: str) -> DisasterInstance:
        return self._get_tracked_instance(disaster_id)

    # ------------------------------------------------------------------------
    # PRIVATE STRATEGY HELPERS & EVENT EMITTERS
    # ------------------------------------------------------------------------

    def _get_tracked_instance(self, disaster_id: str) -> DisasterInstance:
        if disaster_id not in self._active_incidents:
            raise DisasterNotFoundError(f"Operation failed: Incident target token '{disaster_id}' is not registered.")
        return self._active_incidents[disaster_id]

    def _validate_zones(self, zones: List[str]) -> None:
        if not zones:
            raise InvalidDisasterStateConfigurationError("Disaster targeting profiles must include at least one zone target.")

    def _resolve_topological_footprint(self, instance: DisasterInstance) -> None:
        live_graph = self._state.graph
        target_zones = set(instance.affected_zones)

        for node_id, data in live_graph.nodes(data=True):
            if data.get("zone") in target_zones:
                instance.affected_nodes.add(node_id)

        for u, v, data in live_graph.edges(data=True):
            u_zone = live_graph.nodes[u].get("zone")
            v_zone = live_graph.nodes[v].get("zone")
            if u_zone in target_zones or v_zone in target_zones:
                instance.affected_links.add((u, v))

        logger.info(
            f"Footprint Resolved -> Incident: '{instance.disaster_id}' mapping covers "
            f"{len(instance.affected_nodes)} nodes and {len(instance.affected_links)} edges."
        )

    def _emit_degradation_events(self, instance: DisasterInstance) -> None:
        scaled_profile = self._generate_scaled_profile(instance, severity_modifier=1.0)
        base_tick = instance.start_tick

        self._scheduler.schedule_event(Event(
            event_id=f"{instance.disaster_id}_initial_degrade",
            event_name=f"Apply Initial Shock Profile: {instance.disaster_id}",
            scheduled_tick=base_tick,
            priority=EventPriority.HIGH,
            event_type=EventType.DISASTER,
            callback=self._updater.apply_disaster_effects,
            parameters={"profile": scaled_profile, "target_nodes": list(instance.affected_nodes)}
        ))

        for node_id in instance.affected_nodes:
            node_data = self._state.get_node(node_id)
            node_type = node_data.get("type", "Standard")

            if node_type == "Tower" and scaled_profile.tower_failure_probability > 0.3:
                self._scheduler.schedule_event(Event(
                    event_id=f"{instance.disaster_id}_tower_fail_{node_id}",
                    event_name=f"Tower Structural Outage Edge Check: {node_id}",
                    scheduled_tick=base_tick + 1,
                    priority=EventPriority.HIGH,
                    event_type=EventType.NODE,
                    callback=self._updater.isolate_node,
                    parameters={"node_id": node_id}
                ))

    def _generate_scaled_profile(self, instance: DisasterInstance, severity_modifier: float) -> DisasterProfile:
        base = instance.profile
        tier_multiplier = (instance.severity_level / 3.0) * severity_modifier

        return DisasterProfile(
            name=f"{base.name} [Scaled Tier {instance.severity_level}]",
            description=base.description,
            disaster_type=base.disaster_type,
            default_severity=min(1.0, base.default_severity * tier_multiplier),
            min_severity=base.min_severity,
            max_severity=base.max_severity,
            default_duration=instance.duration_ticks,
            growth_rate=base.growth_rate,
            latency_multiplier=base.latency_multiplier * tier_multiplier,
            bandwidth_multiplier=max(0.05, base.bandwidth_multiplier / tier_multiplier),
            packet_loss_multiplier=min(10.0, base.packet_loss_multiplier * tier_multiplier),
            jitter_multiplier=base.jitter_multiplier * tier_multiplier,
            signal_strength_multiplier=max(0.1, base.signal_strength_multiplier / tier_multiplier),
            link_failure_probability=min(1.0, base.link_failure_probability * tier_multiplier),
            tower_failure_probability=min(1.0, base.tower_failure_probability * tier_multiplier),
            road_failure_probability=min(1.0, base.road_failure_probability * tier_multiplier),
            bridge_failure_probability=min(1.0, base.bridge_failure_probability * tier_multiplier),
            utility_failure_probability=min(1.0, base.utility_failure_probability * tier_multiplier),
            hospital_load_increment=int(base.hospital_load_increment * tier_multiplier),
            relief_camp_load_increment=int(base.relief_camp_load_increment * tier_multiplier),
            node_failure_probability=min(1.0, base.node_failure_probability * tier_multiplier),
            emergency_request_multiplier=base.emergency_request_multiplier * tier_multiplier,
            population_affected_percentage=min(1.0, base.population_affected_percentage * tier_multiplier),
            recovery_rate=base.recovery_rate,
            repair_probability=base.repair_probability
        )

    def _schedule_lifecycle_milestone(self, disaster_id: str, tick: int, method: Any, stage_label: str) -> None:
        self._scheduler.schedule_event(Event(
            event_id=f"{disaster_id}_lifecycle_{stage_label.lower()}",
            event_name=f"Transition Lifecycle Stage -> {stage_label} | Incident: {disaster_id}",
            scheduled_tick=tick,
            priority=EventPriority.CRITICAL,
            event_type=EventType.DISASTER,
            callback=method,
            parameters={"disaster_id": disaster_id}
        ))

    def _execute_stepped_recovery_callback(self, disaster_id: str, step_index: int) -> None:
        instance = self._get_tracked_instance(disaster_id)
        if not instance.active or instance.current_stage != DisasterStage.RECOVERING:
            return

        logger.debug(f"Executing Recovery Event Step {step_index} for incident context: '{disaster_id}'.")
        
        for node_id in instance.affected_nodes:
            if not self._state.node_exists(node_id):
                continue
            
            node_data = self._state.get_node(node_id)
            current_load = node_data.get("load", 10.0)
            recovered_load = max(10.0, current_load - (20.0 * instance.profile.recovery_rate))
            self._updater.update_node_load(node_id, recovered_load)

            if node_data.get("type") == "Hospital":
                self._updater.update_hospital_load(node_id, -int(instance.profile.hospital_load_increment * 0.3))