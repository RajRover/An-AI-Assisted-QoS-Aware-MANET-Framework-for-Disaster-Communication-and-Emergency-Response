"""
dashboard/utils.py
-------------------
Presentation-layer glue code for the AI-Assisted QoS-Aware MANET Dashboard.

This module owns EXACTLY one responsibility: wiring together the existing,
untouched backend packages (communication/, services/, simulation/,
integration/, Disaster_Prediction/) into a single object graph that the
Streamlit pages can read from and drive.

No routing, QoS, disaster, mobility, or classification algorithms are
implemented here -- every computation is delegated to the real backend
modules. This file only constructs objects, manages Streamlit session
state, and provides small, presentation-only formatting helpers.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────
# Path bootstrap: dashboard/ lives as a sibling of communication/, services/,
# simulation/, integration/, Disaster_Prediction/ at the project root.
# ─────────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ─────────────────────────────────────────────────────────────────────────
# Backend imports -- existing modules ONLY, nothing reimplemented.
# ─────────────────────────────────────────────────────────────────────────
from communication.graph import build_network, NetworkValidationError
from communication.network_stats import generate_network_report

from simulation.network_state import NetworkState, SimulationConfig
from simulation.event_scheduler import EventScheduler
from simulation.network_updater import NetworkUpdater
from simulation.disaster_profiles import DisasterProfileManager, DisasterType
from simulation.disaster_engine import DisasterEngine, DisasterStage, DisasterEngineException
from simulation.simulation_clock import SimulationClock
from simulation.Mobility.mobility_manager import MobilityManager, build_default_fleet, FleetConfig
from simulation.Mobility.mobile_node import MobileNodeStatus, MobileNodeType

from services.message import (
    DisasterMessage,
    MessagePriority,
    QoSLevel,
    MessageStatus,
    MessageType,
    SenderType,
    DestinationType,
    Department,
)
from services.qos_mapper import QoSMapper
from services.packet_generator import PacketGenerator
from services.dispatcher import Dispatcher
from services.packet import NetworkPacket, PacketStatus, PacketType

from Disaster_Prediction.classifier import DisasterClassifier

from integration.communication_pipeline import (
    CommunicationPipeline,
    CommunicationResult,
    CommunicationPipelineError,
)
from integration.simulation_controller import (
    SimulationController,
    SimulationStatus,
    SimulationSnapshot,
    SimulationStatistics,
    SimulationControllerError,
)
from integration.scenario_runner import (
    ScenarioRunner,
    ScenarioDefinition,
    ScenarioType,
    MessageTemplate,
    ScenarioResult,
    ScenarioRunnerError,
)

# ─────────────────────────────────────────────────────────────────────────
# Dashboard-level logging (separate from, and additive to, backend logging)
# ─────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("MANET.Dashboard")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────
# Reference constants (mirrors precedent already established in the
# backend's own reference scripts, e.g. tests/simulation_runner.py)
# ─────────────────────────────────────────────────────────────────────────
KNOWN_ZONES: List[str] = ["North Zone", "South Zone", "Central Zone"]
DISASTER_TYPE_OPTIONS: List[str] = [d.name for d in DisasterType]
SENDER_TYPE_OPTIONS: List[str] = [s.name for s in SenderType]


# ============================================================================
# BACKEND CONTEXT
# ============================================================================

@dataclass
class BackendContext:
    """Holds one live instance of every wired backend subsystem for this session."""
    graph: Any
    network_state: NetworkState
    profile_manager: DisasterProfileManager
    scheduler: EventScheduler
    updater: NetworkUpdater
    disaster_engine: DisasterEngine
    mobility_manager: MobilityManager
    clock: SimulationClock
    classifier: DisasterClassifier
    qos_mapper: QoSMapper
    packet_generator: PacketGenerator
    dispatcher: Dispatcher
    pipeline: CommunicationPipeline
    controller: SimulationController
    scenario_runner: ScenarioRunner
    disaster_counter: int = 0


@st.cache_resource(show_spinner="Loading AI disaster classification model...")
def _load_classifier() -> DisasterClassifier:
    """Loads the transformer classification model exactly once per server process.

    The classifier performs inference only and holds no per-session mutable
    state, so sharing one loaded instance across sessions is safe and avoids
    reloading multi-hundred-MB model weights on every browser session.
    """
    return DisasterClassifier()


def _build_backend() -> BackendContext:
    """Constructs a fresh, fully-wired backend object graph.

    Mirrors the exact wiring sequence demonstrated in the backend's own
    reference entry points (integration/simulation_controller.py and
    integration/scenario_runner.py `__main__` blocks).
    """
    classifier = _load_classifier()

    graph = build_network()
    config = SimulationConfig()
    network_state = NetworkState(graph, config=config)

    profile_manager = DisasterProfileManager(populate_defaults=True)
    scheduler = EventScheduler()
    updater = NetworkUpdater(network_state=network_state)
    disaster_engine = DisasterEngine(
        network_state=network_state,
        event_scheduler=scheduler,
        network_updater=updater,
        profile_manager=profile_manager,
    )
    clock = SimulationClock(
        network_state=network_state,
        event_scheduler=scheduler,
        disaster_engine=disaster_engine,
        network_updater=updater,
        tick_duration_ms=100.0,
        max_ticks=10000,
        real_time_mode=False,
        auto_stop=False,
    )
    mobility_manager = build_default_fleet(network_state, config=FleetConfig())

    qos_mapper = QoSMapper()
    packet_generator = PacketGenerator()
    dispatcher = Dispatcher(network_state=network_state)
    pipeline = CommunicationPipeline(
        classifier=classifier,
        qos_mapper=qos_mapper,
        packet_generator=packet_generator,
        dispatcher=dispatcher,
        network_state=network_state,
    )

    controller = SimulationController(
        simulation_clock=clock,
        event_scheduler=scheduler,
        disaster_engine=disaster_engine,
        mobility_manager=mobility_manager,
        communication_pipeline=pipeline,
        network_state=network_state,
    )
    controller.initialize()

    scenario_runner = ScenarioRunner(simulation_controller=controller)

    logger.info("Backend context constructed: %d nodes, %d mobile units.",
                graph.number_of_nodes(), len(mobility_manager))

    return BackendContext(
        graph=graph,
        network_state=network_state,
        profile_manager=profile_manager,
        scheduler=scheduler,
        updater=updater,
        disaster_engine=disaster_engine,
        mobility_manager=mobility_manager,
        clock=clock,
        classifier=classifier,
        qos_mapper=qos_mapper,
        packet_generator=packet_generator,
        dispatcher=dispatcher,
        pipeline=pipeline,
        controller=controller,
        scenario_runner=scenario_runner,
    )


def ensure_backend() -> BackendContext:
    """Idempotently builds (once per browser session) and returns the BackendContext.

    Every page module calls this first, since Streamlit executes each
    pages/*.py file as an independent script run and only shares state via
    st.session_state.
    """
    if "backend" not in st.session_state:
        with st.spinner("Initializing MANET framework backend (network, AI model, simulation engine)..."):
            try:
                st.session_state.backend = _build_backend()
            except (NetworkValidationError, Exception) as exc:  # surface, never crash silently
                logger.error("Backend initialization failed: %s", exc, exc_info=True)
                st.error(f"Fatal backend initialization error: {exc}")
                st.stop()
        st.session_state.packet_log = []          # List[NetworkPacket]
        st.session_state.message_history = []      # List[Dict[str, Any]]
        st.session_state.last_route = []            # List[str]
        st.session_state.last_snapshot = None
        st.session_state.scenario_results = []      # List[ScenarioResult]
        st.session_state.default_scenarios_loaded = False
    return st.session_state.backend


def get_ctx() -> BackendContext:
    """Fetches the already-initialized BackendContext (call ensure_backend() first)."""
    return st.session_state.backend


def full_reset() -> None:
    """Performs a complete lifecycle reset across every subsystem.

    Mirrors the exact reset sequence the backend's own ScenarioRunner uses
    before every scenario run (integration/scenario_runner.py `run_scenario`),
    reusing only public/established reset entry points.
    """
    ctx = get_ctx()

    try:
        ctx.network_state.stop()
    except Exception:
        pass
    ctx.network_state.reset()

    try:
        ctx.clock.stop()
    except Exception:
        pass
    ctx.clock.reset()

    ctx.scheduler.clear_events()

    try:
        ctx.disaster_engine._active_incidents.clear()  # same pattern used by ScenarioRunner
    except Exception:
        pass

    ctx.controller.reset()
    ctx.controller.initialize()

    try:
        new_mobility = build_default_fleet(ctx.network_state, config=FleetConfig())
        ctx.controller._mobility_manager = new_mobility  # mirrors ScenarioRunner precedent
        ctx.mobility_manager = new_mobility
    except Exception as exc:
        logger.warning("Could not rebuild default mobility fleet on reset: %s", exc)

    ctx.pipeline.reset_statistics()
    ctx.disaster_counter = 0

    st.session_state.packet_log = []
    st.session_state.message_history = []
    st.session_state.last_route = []
    st.session_state.last_snapshot = None
    st.session_state.scenario_results = []
    logger.info("Full system reset completed.")


def next_disaster_id(prefix: str = "DS") -> str:
    ctx = get_ctx()
    ctx.disaster_counter += 1
    return f"{prefix}-{ctx.disaster_counter:04d}-{uuid.uuid4().hex[:4].upper()}"


# ============================================================================
# FORMATTING / DISPLAY HELPERS (pure presentation, no business logic)
# ============================================================================

def enum_name(value: Any) -> str:
    """Safely renders an Enum member (or plain value) as a display string."""
    if value is None:
        return "—"
    return value.name if hasattr(value, "name") else str(value)


def fmt_ms(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f} ms"


def fmt_pct(value: Optional[float], already_fraction: bool = True, decimals: int = 1) -> str:
    if value is None:
        return "—"
    pct = value * 100.0 if already_fraction else value
    return f"{pct:.{decimals}f}%"


def fmt_num(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}"


NODE_TYPE_ICONS: Dict[str, str] = {
    "ControlCentre": "🏛️",
    "PoliceHQ": "🚓",
    "FireStation": "🚒",
    "Hospital": "🏥",
    "ReliefCamp": "⛺",
    "Tower": "📡",
    "Village": "🏘️",
    "Utility": "⚡",
}

STATUS_COLORS: Dict[str, str] = {
    "Healthy": "#2ecc71",
    "Active": "#2ecc71",
    "Operational": "#2ecc71",
    "Stable": "#2ecc71",
    "Congested": "#f1c40f",
    "Degraded": "#f39c12",
    "Weak": "#e67e22",
    "Low Battery": "#e67e22",
    "Evacuating": "#f1c40f",
    "Broken": "#e74c3c",
    "Offline": "#e74c3c",
    "Critical": "#e74c3c",
    "Failed": "#e74c3c",
    # MobileNodeStatus values (simulation/Mobility/mobile_node.py)
    "Idle": "#95a5a6",
    "EnRoute": "#3498db",
    "Arrived": "#2ecc71",
    "OnMission": "#9b59b6",
    "Returning": "#f39c12",
}


def status_color(status: str) -> str:
    return STATUS_COLORS.get(status, "#95a5a6")


def get_network_report(ctx: BackendContext) -> Dict[str, Any]:
    """Delegates to the backend's own communication.network_stats module for
    whole-graph structural/QoS health metrics (no computation performed here)."""
    return generate_network_report(ctx.network_state.graph)


def get_static_node_ids(ctx: BackendContext) -> List[str]:
    return sorted(ctx.network_state.graph.nodes())


def sender_type_from_name(name: str) -> SenderType:
    return SenderType[name]


def disaster_type_from_name(name: str) -> DisasterType:
    return DisasterType[name]