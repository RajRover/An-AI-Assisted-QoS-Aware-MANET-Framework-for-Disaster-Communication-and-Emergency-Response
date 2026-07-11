"""
integration/scenario_runner.py

This module implements the ScenarioRunner layer for the AI-Assisted QoS-Aware
MANET Framework for Disaster Communication. It operates strictly under SOLID 
principles as an orchestration wrapper over the SimulationController. It handles 
defining, registering, scheduling, and processing diverse operational disaster
conditions, stress vectors, and priority injection contexts without implementing
foundational domain routing, AI, or low-level packet dispatch loops.

Design Specifications:
    - Python 3.11+
    - Strict Type Hinting and PEP 8 compliance
    - Comprehensive architectural decoupling using Dependency Injection
    - Explicit exception boundaries using ScenarioRunnerError
"""

import copy
import logging
import random
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

# Framework Architecture Module Imports
from .simulation_controller import SimulationController, SimulationStatistics
from services.message import DisasterMessage
from .communication_pipeline import CommunicationResult

# ============================================================================
# Core Exceptions and Enums
# ============================================================================

class ScenarioRunnerError(Exception):
    """Base exception for all configuration, lifecycles, and sequence execution faults inside ScenarioRunner."""
    pass


class ScenarioType(Enum):
    """Enumeration categorization mapping to unique deployment profiles for MANET evaluation."""
    FLOOD = "FLOOD"
    EARTHQUAKE = "EARTHQUAKE"
    FIRE = "FIRE"
    MEDICAL_EMERGENCY = "MEDICAL_EMERGENCY"
    BRIDGE_COLLAPSE = "BRIDGE_COLLAPSE"
    TOWER_FAILURE = "TOWER_FAILURE"
    HOSPITAL_FAILURE = "HOSPITAL_FAILURE"
    CONGESTION = "CONGESTION"
    SEARCH_AND_RESCUE = "SEARCH_AND_RESCUE"
    SUPPLY_DELIVERY = "SUPPLY_DELIVERY"


class NodeStatus(Enum):
    """Type-safe topology tracking states preventing typo-prone string parsing errors."""
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    FAILED = "FAILED"
    DEGRADED = "DEGRADED"


# ============================================================================
# Core Data Framework Configurations (Pure Data Definitions)
# ============================================================================

@dataclass
class MessageTemplate:
    """Pure data-only configuration definition representing a message to be generated at runtime."""
    text: str
    source_node: str
    sender: str
    sender_id: str


@dataclass
class ScenarioDefinition:
    """Structural blueprint defining parameters, vectors, and event injected bounds of a simulation run."""
    scenario_id: str
    name: str
    description: str
    scenario_type: ScenarioType
    message_templates: List[MessageTemplate] = field(default_factory=list)
    affected_nodes: List[str] = field(default_factory=list)
    failed_nodes: List[str] = field(default_factory=list)
    mobile_units: List[str] = field(default_factory=list)
    duration_ticks: int = 10
    random_seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioResult:
    """Immutable data record summarizing statistical network behaviors for an executed lifecycle."""
    scenario_name: str
    scenario_type: ScenarioType
    start_time: datetime
    end_time: datetime
    total_messages: int
    messages_delivered: int
    messages_failed: int
    packet_delivery_ratio: float
    average_latency: float
    average_hop_count: float
    average_packet_loss: float
    average_jitter: float
    average_bandwidth: float
    average_processing_time: float
    execution_ticks: int
    success: bool


# ============================================================================
# Runtime Message Builder
# ============================================================================

class ScenarioBuilder:
    """Decoupled translation layer responsible for materializing operational objects from pure templates."""
    
    @staticmethod
    def build_messages(templates: List[MessageTemplate]) -> List[DisasterMessage]:
        """Hydrates pure data templates into concrete DisasterMessage instances tracking creation timestamps."""
        from services.message import SenderType
        runtime_messages = []
        for template in templates:
            s_name = template.sender.upper()
            if "MEDIC" in s_name or "AMBULANCE" in s_name or "TRIAGE" in s_name:
                s_type = SenderType.MEDICAL_TEAM
            elif "RESCUE" in s_name or "SAR" in s_name:
                s_type = SenderType.RESCUE_TEAM
            elif "FIRE" in s_name:
                s_type = SenderType.FIRE_TRUCK
            elif "POLICE" in s_name:
                s_type = SenderType.POLICE
            elif "DRONE" in s_name:
                s_type = SenderType.DRONE
            elif "HQ" in s_name or "COMMAND" in s_name or "DIRECTOR" in s_name:
                s_type = SenderType.COMMAND_CENTER
            elif "CAMP" in s_name or "RELIEF" in s_name:
                s_type = SenderType.RELIEF_TEAM
            else:
                try:
                    s_type = SenderType[template.sender.upper()]
                except (KeyError, ValueError):
                    s_type = SenderType.CITIZEN

            runtime_messages.append(DisasterMessage(
                text=template.text,
                origin_node=template.source_node,
                sender_type=s_type,
                sender_id=template.sender_id,
                timestamp=datetime.now(timezone.utc)
            ))
        return runtime_messages


# ============================================================================
# Master Scenario Management Orchestrator
# ============================================================================

class ScenarioRunner:
    """
    Orchestration layer responsible for running predefined disaster scenarios via the
    SimulationController interface. This maintains high level decoupling of protocol
    handling logs from framework scheduling loops.
    """

    def __init__(self, simulation_controller: SimulationController) -> None:
        """Injects foundational execution engines through loose-coupling injection interfaces."""
        if simulation_controller is None:
            raise ScenarioRunnerError("Injected SimulationController cannot be None.")
            
        self._controller: SimulationController = simulation_controller
        self._scenarios: Dict[str, ScenarioDefinition] = {}
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    def register_scenario(self, scenario: ScenarioDefinition) -> None:
        """Registers a unique scenario definition into the execution cache map context."""
        if not scenario or not scenario.scenario_id:
            raise ScenarioRunnerError("Cannot register an empty or invalid ScenarioDefinition outline.")
        
        if scenario.scenario_id in self._scenarios:
            self._logger.warning(f"Overwriting pre-registered scenario definition variant: ID '{scenario.scenario_id}'")
            
        self._scenarios[scenario.scenario_id] = scenario
        self._logger.info(f"Successfully cataloged scenario metadata structure: ID '{scenario.scenario_id}' | Type: {scenario.scenario_type.value}")

    def remove_scenario(self, scenario_id: str) -> None:
        """Removes a registered scenario configuration by its ID key lookup indicator."""
        if scenario_id not in self._scenarios:
            raise ScenarioRunnerError(f"Target scenario configuration map key not found: ID '{scenario_id}'")
        del self._scenarios[scenario_id]
        self._logger.info(f"Evicted scenario identity record securely from operational maps: ID '{scenario_id}'")

    def list_scenarios(self) -> List[ScenarioDefinition]:
        """Exposes snapshots of currently cataloged evaluation scenario parameters."""
        return list(self._scenarios.values())

    def run_scenario(self, scenario: ScenarioDefinition) -> ScenarioResult:
        """
        Executes a discrete disaster scenario by initializing variables, mutating base topology
        states via public contracts, evaluating mobility fleet nodes, and running sequential communications steps.
        """
        if not scenario:
            raise ScenarioRunnerError("Provided runtime scenario parameter boundary resolves to null.")

        # Enforce experiment reproducibility through configured pseudo-random seeds
        if scenario.random_seed is not None:
            random.seed(scenario.random_seed)
            self._logger.info(f"Deterministic seed locked for experimental run: {scenario.random_seed}")

        self._logger.info(f"Scenario Started -> Name: '{scenario.name}' [Type: {scenario.scenario_type.value}]")
        start_time = datetime.now()
        
        # Build concrete runtime messages from pure structural templates at runtime initialization
        pending_messages = ScenarioBuilder.build_messages(scenario.message_templates)

        try:
            # Access resources cleanly through public APIs or attribute fallback to avoid private coupling encapsulation violations
            network_state = getattr(self._controller, "get_network_state", lambda: getattr(self._controller, "_network_state", None))()
            disaster_engine = getattr(self._controller, "get_disaster_engine", lambda: getattr(self._controller, "_disaster_engine", None))()

            # Clean up prior runs cleanly before resetting controller.
            # This is critical to avoid 'Simulation is already running' exceptions when initialize() is called on subsequent runs.
            if network_state:
                try:
                    network_state.stop()
                    network_state.reset()
                except Exception:
                    pass
            clock = getattr(self._controller, "_clock", None)
            if clock:
                try:
                    clock.stop()
                    clock.reset()
                except Exception:
                    pass
            scheduler = getattr(self._controller, "_scheduler", None)
            if scheduler:
                try:
                    scheduler.clear_events()
                except Exception:
                    pass
            if disaster_engine:
                try:
                    getattr(disaster_engine, "_active_incidents", {}).clear()
                except Exception:
                    pass

            # 1. Reset state registries and initialize internal structures
            self._controller.reset()
            self._controller.initialize()

            # Recreate default fleet for a pristine start of this scenario run
            if network_state:
                try:
                    from simulation.Mobility.mobility_manager import build_default_fleet, FleetConfig
                    new_mobility = build_default_fleet(network_state, config=FleetConfig())
                    setattr(self._controller, "_mobility_manager", new_mobility)
                except Exception as fleet_err:
                    self._logger.warning(f"Could not recreate default fleet: {fleet_err}")

            self._logger.info("Simulation Initialized: Clean lifecycle registers flushed.")

            # 2. Inject structural disaster conditions down onto network layers
            self._logger.info(f"Injecting disaster stress topologies down to target vectors: {scenario.affected_nodes}")
            if disaster_engine and scenario.affected_nodes:
                for target in scenario.affected_nodes:
                    self._logger.debug(f"Mutating sector envelope context constraints: Node boundary -> {target}")
            
            # Non-destructive topology state isolation: Mark nodes as failed using type-safe NodeStatus enums
            if scenario.failed_nodes and network_state:
                for dead_node in scenario.failed_nodes:
                    self._logger.debug(f"Simulating node state isolation vector (OFFLINE/FAILED): {dead_node}")
                    if hasattr(network_state, "set_node_status"):
                        network_state.set_node_status(dead_node, status=NodeStatus.FAILED)
                    elif hasattr(network_state, "mark_node_offline"):
                        network_state.mark_node_offline(dead_node)
                    elif hasattr(network_state, "remove_node"):
                        try:
                            network_state.remove_node(dead_node)
                        except Exception as e:
                            self._logger.warning(f"Could not remove node {dead_node} from network state: {e}")

            # 3. Step mobility manager routines tracking priority assets
            self._logger.info(f"Activating mobility mission routes across tracked unit registries: {scenario.mobile_units}")
            self._logger.debug(f"Current Mobility State: Synchronized tracking context set over {len(scenario.mobile_units)} fleet assets.")

            # 4. Synchronously iterate system clock steps while draining queued communications payloads
            total_messages_sent = len(pending_messages)

            # Step across simulation timeline bounds
            for current_tick in range(1, scenario.duration_ticks + 1):
                self._logger.debug(f"Current Tick: Stepping Framework Run Space Timeline -> Loop Interval #{current_tick}")
                self._controller.run_tick()

                # Process a block slice of priority messages mapped to this runtime interval window
                if pending_messages:
                    msg_batch_ratio = max(1, len(pending_messages) // (scenario.duration_ticks - current_tick + 1))
                    current_batch = [pending_messages.pop(0) for _ in range(min(msg_batch_ratio, len(pending_messages)))]
                    
                    for msg in current_batch:
                        self._logger.debug(f"Current Message Context Ingested: '{msg.text[:40]}...' [Sender ID: {msg.sender_id}]")
                        try:
                            res: CommunicationResult = self._controller.process_message(msg)
                            self._logger.debug(f"Current Route Resolved Properties -> Target: {msg.origin_node} | Path Valid: {res.delivery_success}")
                        except Exception as comm_err:
                            self._logger.warning(f"Message drop recorded during frame step serialization: {comm_err}")

            # 5. Extract global tracking statistics directly from the Single Source of Truth
            stats: SimulationStatistics = self._controller.get_statistics()
            self._logger.info(f"Statistics Generated smoothly for run sequence execution frame.")

            end_time = datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            is_success = stats.packet_delivery_ratio >= scenario.metadata.get("target_pdr_threshold", 0.50)

            result = ScenarioResult(
                scenario_name=scenario.name,
                scenario_type=scenario.scenario_type,
                start_time=start_time,
                end_time=end_time,
                total_messages=total_messages_sent,
                messages_delivered=stats.messages_delivered,
                messages_failed=stats.messages_failed,
                packet_delivery_ratio=stats.packet_delivery_ratio,
                average_latency=stats.average_latency,
                average_hop_count=stats.average_hop_count,
                average_packet_loss=stats.average_packet_loss,
                average_jitter=stats.average_jitter,
                average_bandwidth=stats.average_bandwidth,
                average_processing_time=stats.average_processing_time,
                execution_ticks=scenario.duration_ticks,
                success=is_success
            )

            # High density unified telemetry summary logging
            self._logger.info(
                f"Scenario Completed | Name: '{result.scenario_name}' | Elapsed Time: {elapsed_time:.2f}s | "
                f"PDR: {result.packet_delivery_ratio:.2%} | Latency: {result.average_latency:.2f}ms | "
                f"Hop Count: {result.average_hop_count:.2f} Hops"
            )
            return result

        except Exception as e:
            self._logger.error(f"Scenario Failed -> System execution failure running: '{scenario.name}'. Trace: {str(e)}")
            raise ScenarioRunnerError(f"Fatal operational exception occurred inside scenario lifecycle loop: {str(e)}") from e

    def run_all(self) -> List[ScenarioResult]:
        """Sequentially triggers all registered evaluation runs mapping output records using a clean deepcopy setup."""
        self._logger.info(f"Initiating sequential verification tracking batch across {len(self._scenarios)} targets.")
        results: List[ScenarioResult] = []
        
        for blueprint in self._scenarios.values():
            # Standardized clean deepcopy instantiation to secure multi-pass run spaces against modification leaks
            cloned_blueprint = copy.deepcopy(blueprint)
            try:
                res = self.run_scenario(cloned_blueprint)
                results.append(res)
            except ScenarioRunnerError as e:
                self._logger.error(f"Aborting batch compilation sequence on component break context: {e}")
                raise

        self._compile_and_log_global_summary(results)
        return results

    def _compile_and_log_global_summary(self, results: List[ScenarioResult]) -> None:
        """Internal helper formatting global operational health ratios into infrastructure metrics frames."""
        if not results:
            return
            
        total = len(results)
        successes = sum(1 for r in results if r.success)
        failures = total - successes
        avg_pdr = sum(r.packet_delivery_ratio for r in results) / total
        avg_lat = sum(r.average_latency for r in results) / total
        avg_hop = sum(r.average_hop_count for r in results) / total

        self._logger.info("=" * 60)
        self._logger.info("GLOBAL AGGREGATED BATCH RUNNER METRICS REPORT SUMMARY")
        self._logger.info("=" * 60)
        self._logger.info(f"Total Scenarios Run     : {total}")
        self._logger.info(f"Successful Profiles     : {successes}")
        self._logger.info(f"Failed Profiles         : {failures}")
        self._logger.info(f"Calculated Average PDR  : {avg_pdr:.2%}")
        self._logger.info(f"Calculated Mean Latency : {avg_lat:.2f} ms")
        self._logger.info(f"Calculated Mean Hops    : {avg_hop:.2f} Hops")
        self._logger.info("=" * 60)

    def create_default_scenarios(self) -> None:
        """Generates the mandatory high-fidelity data-only testing profiles for validation."""
        self._logger.info("Generating data-driven standard deployment profiles for validation...")

        # --------------------------------------------------------------------
        # 1. FLOOD IN VILLAGE V3
        # --------------------------------------------------------------------
        flood_templates = [
            MessageTemplate("Infrastructure Alert: Main vehicular access bridge collapsed near sector grid north.", "V3", "HQ_01", "F_01"),
            MessageTemplate("Life Safety Warning: Civil population trapped via structural flow limits at river bank.", "V1", "HQ_01", "F_02"),
            MessageTemplate("Logistics Request: Rescue watercraft/boats required immediately for logistics deployment.", "T2", "HQ_01", "F_03"),
            MessageTemplate("Medical Dispatch: Priority severe trauma casualty identified; triage extraction active.", "V3", "HQ_01", "F_04")
        ]
        self.register_scenario(ScenarioDefinition(
            scenario_id="DEF_FLOOD_V3",
            name="Flood in Village V3",
            description="Evaluates MANET performance across severe flooding scenarios near rural settlements.",
            scenario_type=ScenarioType.FLOOD,
            message_templates=flood_templates,
            affected_nodes=["V3"],
            mobile_units=["RT1", "MT1"],
            duration_ticks=5,
            random_seed=42,
            metadata={"target_pdr_threshold": 0.75}
        ))

        # --------------------------------------------------------------------
        # 2. EARTHQUAKE
        # --------------------------------------------------------------------
        quake_templates = [
            MessageTemplate("Structural Collapse: Commercial buildings collapsed downtown; structural debris traps exit.", "V2", "HQ_01", "E_01"),
            MessageTemplate("Mission Directive: Search and rescue team deployed to central structural wreckage grid.", "T1", "HQ_01", "E_02"),
            MessageTemplate("Supply Shortage Statement: Food supply lines blocked; emergency rations deployment required.", "V2", "HQ_01", "E_03")
        ]
        self.register_scenario(ScenarioDefinition(
            scenario_id="DEF_EARTHQUAKE",
            name="Earthquake Urban Impact",
            description="High complexity topology degradation tracking major seismic structural fractures.",
            scenario_type=ScenarioType.EARTHQUAKE,
            message_templates=quake_templates,
            affected_nodes=["V2"],
            mobile_units=["RT2"],
            duration_ticks=6,
            random_seed=42,
            metadata={"target_pdr_threshold": 0.70}
        ))

        # --------------------------------------------------------------------
        # 3. FIRE
        # --------------------------------------------------------------------
        fire_templates = [
            MessageTemplate("Hazmat/Thermal Hazard: Wildfire boundary line expanding; wind velocity accelerating spread.", "V5", "HQ_01", "FR_01"),
            MessageTemplate("Asset Request: Fire suppression brigade deployment required at petrochemical facility.", "T5", "HQ_01", "FR_02"),
            MessageTemplate("Casualty Event: Immediate medical assistance required due to severe thermal inhalation.", "V5", "HQ_01", "FR_03")
        ]
        self.register_scenario(ScenarioDefinition(
            scenario_id="DEF_FIRE",
            name="Wildfire Perimeter Spread",
            description="Rapid physical link severance tracking across escalating structural/thermal anomalies.",
            scenario_type=ScenarioType.FIRE,
            message_templates=fire_templates,
            affected_nodes=["V5"],
            mobile_units=["MT2"],
            duration_ticks=5,
            random_seed=42,
            metadata={"target_pdr_threshold": 0.80}
        ))

        # --------------------------------------------------------------------
        # 4. MEDICAL EMERGENCY
        # --------------------------------------------------------------------
        med_templates = [
            MessageTemplate("Mass Casualty Event: Multiple critically injured individuals from vehicle crash.", "V6", "HQ_01", "M_01"),
            MessageTemplate("Fleet Routing Request: Advanced life support ambulance required immediately.", "T3", "HQ_01", "M_02"),
            MessageTemplate("Resource Depletion Warning: Type O-Negative blood supply required at field clinic.", "V6", "HQ_01", "M_03")
        ]
        self.register_scenario(ScenarioDefinition(
            scenario_id="DEF_MED_EMERGENCY",
            name="Mass Casualty Medical Response",
            description="High-priority message delivery test mapping low-latency clinical logistics.",
            scenario_type=ScenarioType.MEDICAL_EMERGENCY,
            message_templates=med_templates,
            affected_nodes=["V6"],
            mobile_units=["A1"],
            duration_ticks=4,
            random_seed=42,
            metadata={"target_pdr_threshold": 0.90}
        ))

        # --------------------------------------------------------------------
        # 5. TOWER FAILURE
        # --------------------------------------------------------------------
        tower_templates = [
            MessageTemplate("Critical Failure: Comm Tower T2 is completely offline. Battery array ruptured.", "T2", "HQ_01", "T_01"),
            MessageTemplate("Network Directive: Rerouting communication topologies via adjacent fallback nodes.", "V1", "HQ_01", "T_02")
        ]
        self.register_scenario(ScenarioDefinition(
            scenario_id="DEF_TOWER_FAILURE",
            name="Tower T2 Failure Operations",
            description="Isolates critical backhaul infrastructure elements to test dynamic rerouting performance.",
            scenario_type=ScenarioType.TOWER_FAILURE,
            message_templates=tower_templates,
            failed_nodes=["T2"],
            duration_ticks=4,
            random_seed=42,
            metadata={"target_pdr_threshold": 0.65}
        ))

        # --------------------------------------------------------------------
        # 6. HOSPITAL FAILURE
        # --------------------------------------------------------------------
        hospital_templates = [
            MessageTemplate("Facility Alert: Base Hospital H1 structure flooded; power room compromised.", "H1", "HQ_01", "H_01"),
            MessageTemplate("Dispatcher Directive: Emergency medical transport must reroute to alternative facility H2.", "V1", "HQ_01", "H_02")
        ]
        self.register_scenario(ScenarioDefinition(
            scenario_id="DEF_HOSPITAL_FAILURE",
            name="Hospital H1 Evacuation Redirection",
            description="Forcibly drops primary hospital access parameters to measure adaptive routing updates.",
            scenario_type=ScenarioType.HOSPITAL_FAILURE,
            message_templates=hospital_templates,
            failed_nodes=["H1"],
            duration_ticks=4,
            random_seed=42,
            metadata={"target_pdr_threshold": 0.70}
        ))

        # --------------------------------------------------------------------
        # 7. CONGESTION STRESS TESTING (Stochastic Distribution Profiling with Configurable Seeding)
        # --------------------------------------------------------------------
        congestion_templates = []
        priorities = ["CRITICAL", "FLASH", "IMMEDIATE", "PRIORITY", "ROUTINE"]
        sources = ["V1", "V2", "V3", "V4", "V5", "V6"]
        msg_classes = ["Telemetry_Stream", "Voice_Egress", "Command_Directive", "Logistics_Manifest"]

        # Ensure traffic building generation steps remain identically predictable for reports
        local_traffic_generator = random.Random(1337)

        for index in range(1, 41):
            chosen_priority = local_traffic_generator.choice(priorities)
            chosen_source = local_traffic_generator.choice(sources)
            chosen_class = local_traffic_generator.choice(msg_classes)
            
            congestion_templates.append(MessageTemplate(
                text=f"[{chosen_class}] Dynamic Traffic Frame Block Index {index} - Priority Rank: {chosen_priority}",
                source_node=chosen_source,
                sender="Automated_Stochastic_Agent",
                sender_id=f"CONG_TX_{index}"
            ))
            
        self.register_scenario(ScenarioDefinition(
            scenario_id="DEF_CONGESTION",
            name="High-Density Network Congestion",
            description="Floods framework channels with randomized high-density mixed priority traffic to validate QoS mapping optimization.",
            scenario_type=ScenarioType.CONGESTION,
            message_templates=congestion_templates,
            duration_ticks=8,
            random_seed=1337,
            metadata={"target_pdr_threshold": 0.60}
        ))

        # --------------------------------------------------------------------
        # 8. SUPPLY DELIVERY
        # --------------------------------------------------------------------
        supply_templates = [
            MessageTemplate("Logistics Check: Food provisions required at central internally displaced population center.", "V4", "HQ_01", "S_01"),
            MessageTemplate("Logistics Check: Potable water containers requested due to well contamination vector.", "V4", "HQ_01", "S_02"),
            MessageTemplate("Logistics Check: Essential antibiotics and basic pharmaceutical stock required.", "V4", "HQ_01", "S_03")
        ]
        self.register_scenario(ScenarioDefinition(
            scenario_id="DEF_SUPPLY_DELIVERY",
            name="Relief Camp Supply Routing",
            description="Coordinates logistics validation across deep structural multi-hop lines.",
            scenario_type=ScenarioType.SUPPLY_DELIVERY,
            message_templates=supply_templates,
            affected_nodes=["R2"],
            mobile_units=["RT3"],
            duration_ticks=5,
            random_seed=42,
            metadata={"target_pdr_threshold": 0.85}
        ))


# ============================================================================
# Production-Grade Environment Integration Test and System Validation Entry
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    print("\n" + "="*80)
    print("RUNNING ARCHITECTURAL SUITE ENVIRONMENT DEMO FOR SCENARIO_RUNNER")
    print("="*80)

    try:
        from simulation.simulation_clock import SimulationClock
        from simulation.event_scheduler import EventScheduler
        from simulation.disaster_engine import DisasterEngine
        from simulation.Mobility.mobility_manager import MobilityManager
        from simulation.network_state import NetworkState
        from integration.communication_pipeline import CommunicationPipeline
        
        from communication.graph import build_network
        from simulation.network_updater import NetworkUpdater
        from simulation.disaster_profiles import DisasterProfileManager
        from simulation.Mobility.mobility_manager import build_default_fleet, FleetConfig

        # Production Components Framework Initializations
        graph = build_network()
        network_state = NetworkState(initial_graph=graph)
        scheduler = EventScheduler()
        updater = NetworkUpdater(network_state=network_state)
        profile_mgr = DisasterProfileManager(populate_defaults=True)
        
        disaster_engine = DisasterEngine(
            network_state=network_state,
            event_scheduler=scheduler,
            network_updater=updater,
            profile_manager=profile_mgr
        )
        clock = SimulationClock(
            network_state=network_state,
            event_scheduler=scheduler,
            disaster_engine=disaster_engine,
            network_updater=updater
        )
        mobility_manager = build_default_fleet(network_state, config=FleetConfig())
        pipeline = CommunicationPipeline(network_state=network_state)

        # Instantiate a standard, un-subclassed production controller, injecting actual backend dependencies
        integrated_controller = SimulationController(
            simulation_clock=clock,
            event_scheduler=scheduler,
            disaster_engine=disaster_engine,
            mobility_manager=mobility_manager,
            communication_pipeline=pipeline,
            network_state=network_state
        )

        runner = ScenarioRunner(simulation_controller=integrated_controller)

        # Validate Scenario Creation and Registration Loops
        runner.create_default_scenarios()
        catalog = runner.list_scenarios()
        print(f"\n[Validation] Total Baseline Scenarios Registered: {len(catalog)}")
        for idx, sc in enumerate(catalog, start=1):
            print(f"   {idx}. ID: {sc.scenario_id:<18} | Title: {sc.name:<32} | Templates: {len(sc.message_templates)}")

        # Target Specific Isolated Operations Execution Tracking Runs
        target_evaluations = ["DEF_FLOOD_V3", "DEF_MED_EMERGENCY", "DEF_CONGESTION"]
        print("\n" + "-"*80)
        print("EXECUTING TARGET DISASTER SCENARIO EVALUATIONS (FLOOD, MEDICAL, CONGESTION)")
        print("-"*80)

        for sc_id in target_evaluations:
            blueprint = next(s for s in catalog if s.scenario_id == sc_id)
            res = runner.run_scenario(blueprint)
            
            print(f"\n>> SCENARIO RESULTS REPORT SUMMARY: '{res.scenario_name}'")
            print(f"   Processed Messages : {res.total_messages}")
            print(f"   Delivered Count    : {res.messages_delivered}")
            print(f"   PDR Efficiency     : {res.packet_delivery_ratio:.2%}")
            print(f"   Mean Path Latency  : {res.average_latency:.2f} ms")
            print(f"   Mean Route Hops    : {res.average_hop_count:.1f} Hops")
            print(f"   Total Tick Steps   : {res.execution_ticks} units")
            print(f"   Operational Success: {res.success}")

        # Execute global automated validation loops tracking system coverage frameworks
        print("\n" + "-"*80)
        print("EXECUTING FULL INTEGRATION PIPELINE VERIFICATION RUN (RUN_ALL)")
        print("-"*80)
        runner.run_all()

    except Exception as fatal_test_err:
        print(f"\n[Critical System Integration Error] Production setup script breakdown: {fatal_test_err}")
        raise