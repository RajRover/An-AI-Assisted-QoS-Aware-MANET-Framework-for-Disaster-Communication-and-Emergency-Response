"""
simulation_clock.py

Provides the central scheduling heartbeat and execution pipeline runner 
(`SimulationClock`) for the AI-Assisted QoS-Aware MANET Framework for Disaster 
Communication. Implements discrete-time state progression, precise runtime tracking, 
and generic hook registrations.

Author: Simulation Framework Designer & Software Architect
License: MIT
"""

import time
import logging
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Callable, Dict, List, Optional, Set

# Core Architecture Interface References
from simulation.network_state import NetworkState
from simulation.event_scheduler import EventScheduler
from simulation.disaster_engine import DisasterEngine
from simulation.network_updater import NetworkUpdater

# Configure Module Logger
logger = logging.getLogger("DisasterMANET.SimulationClock")


# ============================================================================
# ENUMS & EXCEPTIONS
# ============================================================================

@unique
class SimulationClockStatus(Enum):
    """Defines the discrete runtime lifecycle operational phases of the master clock."""
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"


class SimulationClockException(Exception):
    """Base exception class for all time progression and lifecycle runtime synchronization anomalies."""
    pass


class SimulationAlreadyRunningError(SimulationClockException):
    """Raised when an illegal lifecycle transition is requested while the clock runner is active."""
    pass


class SimulationNotRunningError(SimulationClockException):
    """Raised when operations requiring active execution frames are invoked while stopped."""
    pass


class SimulationPausedError(SimulationClockException):
    """Raised when timeline execution commands violate an ongoing tracking pause condition."""
    pass


class InvalidTickError(SimulationClockException):
    """Raised when time step variables fail configuration boundaries or logical range limits."""
    pass


# ============================================================================
# RUNTIME METRICS DATACLASS
# ============================================================================

@dataclass
class SimulationTelemetryMetrics:
    """Encapsulates low-overhead architectural execution analytics and timing statistics."""
    current_tick: int = 0
    elapsed_simulation_time_ms: float = 0.0
    total_events_executed: int = 0
    average_tick_duration_seconds: float = 0.0
    total_processing_time_seconds: float = 0.0


# ============================================================================
# SIMULATION MASTER HEARTBEAT RUNNER
# ============================================================================

class SimulationClock:
    """
    The main execution orchestrator for the simulation workspace. Controls step progression
    and invokes modular update cascades at designated temporal intervals.
    """

    def __init__(
        self,
        network_state: NetworkState,
        event_scheduler: EventScheduler,
        disaster_engine: DisasterEngine,
        network_updater: NetworkUpdater,
        tick_duration_ms: float = 100.0,
        max_ticks: int = 10000,
        real_time_mode: bool = False,
        auto_stop: bool = True
    ) -> None:
        """
        Initializes the simulation clock wrapper with its structural component dependencies.

        Args:
            network_state (NetworkState): Single source of truth context layer.
            event_scheduler (EventScheduler): High-performance core priority event queue.
            disaster_engine (DisasterEngine): Incidents coordinator controller.
            network_updater (NetworkUpdater): Exclusive graph transaction layer.
            tick_duration_ms (float): Real-world temporal mapping assigned to each individual tick.
            max_ticks (int): Upper bound limit tracking threshold for the total run duration.
            real_time_mode (bool): Emulates true real-time wall-clock step limits if enabled.
            auto_stop (bool): Halts running thread states if work queues empty out early.
        """
        self._state: NetworkState = network_state
        self._scheduler: EventScheduler = event_scheduler
        self._disaster_engine: DisasterEngine = disaster_engine
        self._updater: NetworkUpdater = network_updater

        # Configuration Parameters
        if tick_duration_ms <= 0:
            raise InvalidTickError("Tick duration settings must be strictly greater than zero.")
        if max_ticks <= 0:
            raise InvalidTickError("Simulation max execution bounds must be greater than zero.")

        self.tick_duration_ms: float = tick_duration_ms
        self.max_ticks: int = max_ticks
        self.real_time_mode: bool = real_time_mode
        self.auto_stop: bool = auto_stop

        # Internal State Registers
        self._status: SimulationClockStatus = SimulationClockStatus.STOPPED
        self.metrics: SimulationTelemetryMetrics = SimulationTelemetryMetrics()

        # Zero-Cost Hook Notification Registries
        self._hooks: Dict[str, List[Callable[..., None]]] = {
            "before_tick": [],
            "after_tick": [],
            "before_event": [],
            "after_event": [],
            "before_update": [],
            "after_update": []
        }

        logger.info(
            f"Simulation Clock Engine context initialized. [Max Ticks: {self.max_ticks} | "
            f"Tick Duration: {self.tick_duration_ms}ms | Real-Time: {self.real_time_mode}]"
        )

    # ------------------------------------------------------------------------
    # LIFECYCLE MANAGEMENT API
    # ------------------------------------------------------------------------

    def start(self) -> None:
        """Transitions the engine into an active running state."""
        if self._status == SimulationClockStatus.RUNNING:
            raise SimulationAlreadyRunningError("Clock execution requested while simulation is already active.")
        
        self._status = SimulationClockStatus.RUNNING
        logger.info("Simulation Started -> Heartbeat time series loop activated.")

    def pause(self) -> None:
        """Suspends ongoing time progression loops safely without clearing tracking states."""
        if self._status != SimulationClockStatus.RUNNING:
            raise SimulationNotRunningError("Pause rejected: Simulation framework runner is not actively executing.")
        
        self._status = SimulationClockStatus.PAUSED
        logger.warning("Simulation Paused -> Temporal step progression suspended.")

    def resume(self) -> None:
        """Restores a paused tracking framework session back to an active runner status."""
        if self._status != SimulationClockStatus.PAUSED:
            raise SimulationClockException("Resume rejected: Clock layer does not hold an active pause latch state.")
        
        self._status = SimulationClockStatus.RUNNING
        logger.info("Simulation Resumed -> Time series processing pipeline restored.")

    def stop(self) -> None:
        """Terminates active simulation loops, marking the runtime as finished."""
        if self._status == SimulationClockStatus.STOPPED:
            return
            
        self._status = SimulationClockStatus.FINISHED
        logger.warning(f"Simulation Finished -> Execution halted at absolute tick threshold: {self.metrics.current_tick}")

    def reset(self) -> None:
        """Restores pristine initialization defaults across internal tracking fields."""
        if self._status == SimulationClockStatus.RUNNING:
            raise SimulationAlreadyRunningError("Cannot reset core orchestration contexts while loop execution is active.")
            
        self._status = SimulationClockStatus.STOPPED
        self.metrics = SimulationTelemetryMetrics()
        logger.info("Simulation Reset -> Temporal registers and analytical telemetry flushes complete.")

    # ------------------------------------------------------------------------
    # TIMELINE ADVANCEMENT DRIVERS
    # ------------------------------------------------------------------------

    def step(self) -> bool:
        """
        Executes exactly one discrete step through the simulation tracking pipeline.

        Returns:
            bool: True if execution can proceed; False if boundary conditions necessitate a halt.
        """
        if self._status == SimulationClockStatus.PAUSED:
            raise SimulationPausedError("Step execution blocked: Simulation clock context remains paused.")
        if self._status in [SimulationClockStatus.STOPPED, SimulationClockStatus.FINISHED]:
            self.start()

        # Enforce terminal bounds check rules
        if self.metrics.current_tick >= self.max_ticks:
            self.stop()
            return False

        # Validate conditions for auto-stop optimization
        if self.auto_stop and self.metrics.current_tick > 0:
            active_disasters = self._disaster_engine.get_active_disasters()
            pending_events = self._scheduler.get_pending_events()
            if not active_disasters and not pending_events:
                logger.info("Auto-Stop Condition Met -> Work registers clear. Halting execution pipeline.")
                self.stop()
                return False

        # Drive sequential tick advancement
        self.advance_tick()
        return True

    def run(self) -> None:
        """Continuously steps through the execution timeline until a terminal state is reached."""
        self.start()
        
        while self._status == SimulationClockStatus.RUNNING:
            tick_start_wall = time.perf_counter()
            
            should_continue = self.step()
            if not should_continue:
                break

            # Enforce true real-world wall clock frame updates if configured
            if self.real_time_mode:
                elapsed_wall = time.perf_counter() - tick_start_wall
                target_sleep = (self.tick_duration_ms / 1000.0) - elapsed_wall
                if target_sleep > 0:
                    time.sleep(target_sleep)

    def advance_tick(self) -> None:
        """
        Executes the mandatory structural coordination sequence for a single tick.
        """
        tick_start_timestamp = time.perf_counter()
        
        # 1. Advance simulation time variables
        self.metrics.current_tick += 1
        self.metrics.elapsed_simulation_time_ms += self.tick_duration_ms
        
        logger.debug(f"Tick Started -> Processing milestone index: {self.metrics.current_tick}")

        # 2-3. Trigger Extensibility Hooks & Flush Due Events from Scheduler
        self._trigger_hooks("before_tick")
        
        self._trigger_hooks("before_event")
        events_processed = self._scheduler.execute_due_events(self.metrics.current_tick, self._state)
        self.metrics.total_events_executed += events_processed
        self._trigger_hooks("after_event")

        # 4. Allow DisasterEngine to update incident life-cycles and emit fresh actions
        # Note: The DisasterEngine internally handles stage checking based on tick timelines
        # via automated lifecycle events scheduled during its creation context.

        # 5-6. Allow NetworkUpdater to calculate topological telemetry arrays and save results
        self._trigger_hooks("before_update")
        self._updater.recompute_global_telemetry()
        self._trigger_hooks("after_update")

        # 7. Notify future modules via standard notification hooks
        self._trigger_hooks("after_tick")

        # Compile background execution profiles
        tick_processing_time = time.perf_counter() - tick_start_timestamp
        self.metrics.total_processing_time_seconds += tick_processing_time
        self.metrics.average_tick_duration_seconds = (
            self.metrics.total_processing_time_seconds / self.metrics.current_tick
        )

        logger.debug(
            f"Tick Finished -> Processing complete for milestone: {self.metrics.current_tick} | "
            f"Wall Execution Time: {tick_processing_time:.5f}s"
        )

    # ------------------------------------------------------------------------
    # EXTENSIBILITY AND HOOK SYSTEM INTERFACES
    # ------------------------------------------------------------------------

    def register_hook(self, hook_point: str, callback: Callable[..., None]) -> None:
        """
        Registers an external component callback into the time tracking engine pipeline.

        Args:
            hook_point (str): Targeted sequence boundary phase key.
            callback (Callable): Target execution function reference signature.
        """
        if hook_point not in self._hooks:
            raise KeyError(f"Hook registration rejected: Target phase identifier '{hook_point}' is invalid.")
        if not callable(callback):
            raise TypeError("Hook execution callback parameter is not callable.")
            
        self._hooks[hook_point].append(callback)
        logger.info(f"Extension Hook Registered successfully onto tracking milestone point: '{hook_point}'")

    def _trigger_hooks(self, hook_point: str) -> None:
        """Dispatches ongoing network state arrays down to active external overrides safely."""
        for callback in self._hooks[hook_point]:
            try:
                callback(self._state, current_tick=self.metrics.current_tick)
            except Exception as exc:
                logger.error(
                    f"Hook Execution Failure inside phase registration zone '{hook_point}': {str(exc)}",
                    exc_info=True
                )

    # ------------------------------------------------------------------------
    # ACCESSOR GETTERS
    # ------------------------------------------------------------------------

    @property
    def current_tick(self) -> int:
        """Returns the current absolute simulation tick index milestone."""
        return self.metrics.current_tick

    @property
    def current_time(self) -> float:
        """Returns the total elapsed virtual operational simulation time in milliseconds."""
        return self.metrics.elapsed_simulation_time_ms

    @property
    def status(self) -> SimulationClockStatus:
        """Exposes the internal operational phase signature of the clock runner."""
        return self._status