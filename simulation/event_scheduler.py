"""
event_scheduler.py

Provides the centralized, generic discrete-event scheduling mechanics for the 
AI-Assisted QoS-Aware MANET Framework for Disaster Communication. Operates via a 
high-efficiency priority queue (heapq) to register, lifecycle-manage, and execute 
time-sensitive component updates via zero-knowledge callbacks.

Author: Simulation Framework Designer & Software Architect
License: MIT
"""

import heapq
import logging
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Callable, Dict, List, Optional

# Configure Module Logger
logger = logging.getLogger("DisasterMANET.EventScheduler")


# ============================================================================
# ENUMS
# ============================================================================

@unique
class EventPriority(Enum):
    """
    Defines structural precedence for concurrent events scheduled at the exact same tick.
    Lower numerical values indicate higher execution priority.
    """
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3

    def __lt__(self, other: "EventPriority") -> bool:
        if not isinstance(other, EventPriority):
            return NotImplemented
        return self.value < other.value


@unique
class EventType(Enum):
    """Categorizes the subsystem origin or behavioral domain of an event."""
    DISASTER = "DISASTER"
    NETWORK = "NETWORK"
    NODE = "NODE"
    EDGE = "EDGE"
    MOBILITY = "MOBILITY"
    TRAFFIC = "TRAFFIC"
    SYSTEM = "SYSTEM"
    CUSTOM = "CUSTOM"


# ============================================================================
# CUSTOM SCHEDULER EXCEPTIONS
# ============================================================================

class SchedulerException(Exception):
    """Base exception class for all scheduler runtime anomalies."""
    pass


class EventAlreadyExistsError(SchedulerException):
    """Raised when an attempt is made to schedule an event using a duplicate unique ID."""
    pass


class EventNotFoundError(SchedulerException):
    """Raised when looking up or altering a target event ID that does not exist."""
    pass


class InvalidEventError(SchedulerException):
    """Raised when an event definition parameters fail safety validations."""
    pass


# ============================================================================
# EVENT DEFINITION DATACLASS
# ============================================================================

@dataclass
class Event:
    """
    Encapsulates standard telemetry variables, execution tracking configurations,
    and the zero-knowledge execution callback signature for an operational simulation unit.
    """
    event_id: str
    event_name: str
    scheduled_tick: int
    priority: EventPriority
    event_type: EventType
    callback: Callable[..., Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    repeat: bool = False
    repeat_interval: int = 0
    enabled: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        """Executes structural safety validation assertions."""
        if self.scheduled_tick < 0:
            raise InvalidEventError(f"Event '{self.event_id}' cannot be scheduled in a negative tick space.")
        if self.repeat and self.repeat_interval <= 0:
            raise InvalidEventError(f"Event '{self.event_id}' marked as repeating must declare a positive interval.")
        if not callable(self.callback):
            raise InvalidEventError(f"Event '{self.event_id}' register failed: callback parameter is not callable.")

    def __lt__(self, other: "Event") -> bool:
        """
        Structural sorting comparator for the heapq min-heap optimization.
        Sorts primarily by tick milestone, then resolves temporal race conditions with priority variables.
        """
        if not isinstance(other, Event):
            return NotImplemented
        if self.scheduled_tick != other.scheduled_tick:
            return self.scheduled_tick < other.scheduled_tick
        return self.priority.value < other.priority.value


# ============================================================================
# EVENT SCHEDULER CORE FRAMEWORK
# ============================================================================

class EventScheduler:
    """
    Central state-isolated discrete event coordinator. Owns tracking queues,
    enforces strict transaction boundaries during callbacks, and facilitates 
    decoupled component integration without storing structural domain context.
    """

    def __init__(self) -> None:
        """Initializes pristine execution tracking structures."""
        self._heap: List[Event] = []
        self._registry: Dict[str, Event] = {}
        self.failed_events: int = 0
        logger.info("Generic Discrete-Event Scheduler engine initialized successfully.")

    def schedule_event(self, event: Event) -> None:
        """
        Registers an event instance inside the operational tracking queue.

        Args:
            event (Event): Fully configured baseline event block.
        """
        if event.event_id in self._registry:
            raise EventAlreadyExistsError(f"Event ID registration collision: '{event.event_id}' already active.")

        self._registry[event.event_id] = event
        heapq.heappush(self._heap, event)
        logger.debug(
            f"Event Scheduled -> ID: {event.event_id} | Name: '{event.event_name}' | "
            f"Type: {event.event_type.name} | Tick Target: {event.scheduled_tick} | Priority: {event.priority.name}"
        )

    def cancel_event(self, event_id: str) -> None:
        """
        Soft-disables an active execution context inside the heap lookup tracking maps.

        Args:
            event_id (str): Unique structural key assigned to the target event.
        """
        if event_id not in self._registry:
            raise EventNotFoundError(f"Cancellation rejected: Event target ID '{event_id}' does not exist.")
        
        event = self._registry[event_id]
        event.enabled = False
        logger.warning(f"Event Cancelled -> ID: {event_id} has been disabled within execution registries.")

    def reschedule_event(self, event_id: str, new_tick: int) -> None:
        """
        Modifies target timeline ticks of existing tracking blocks safely.

        Args:
            event_id (str): Target tracking module identity key.
            new_tick (int): Target tick milestone inside the simulation timeline.
        """
        if event_id not in self._registry:
            raise EventNotFoundError(f"Reschedule rejected: Event target ID '{event_id}' does not exist.")
        if new_tick < 0:
            raise InvalidEventError(f"Reschedule target bound error: '{new_tick}' cannot be a negative tick space.")

        old_event = self._registry.pop(event_id)
        
        # Build fresh context tracking block maintaining core parameters with updated execution limits
        new_event = Event(
            event_id=old_event.event_id,
            event_name=old_event.event_name,
            scheduled_tick=new_tick,
            priority=old_event.priority,
            event_type=old_event.event_type,
            callback=old_event.callback,
            parameters=old_event.parameters,
            repeat=old_event.repeat,
            repeat_interval=old_event.repeat_interval,
            enabled=old_event.enabled,
            description=old_event.description
        )
        
        # Soft-disable the old reference in heap, then register the restructured tracking block
        old_event.enabled = False
        self.schedule_event(new_event)
        logger.info(f"Event Rescheduled -> ID: {event_id} | Shifted from Tick {old_event.scheduled_tick} to Tick {new_tick}.")

    def clear_events(self) -> None:
        """Flushes storage structures and internal state registers completely."""
        self._heap.clear()
        self._registry.clear()
        logger.warning("Event execution priority queue registries completely cleared.")

    def peek_next_event(self) -> Optional[Event]:
        """
        Inspects the next valid enabled element from the heap without removing it.

        Returns:
            Optional[Event]: Next valid execution event block, or None if empty.
        """
        # Clean stale/disabled headers from queue top to guarantee deterministic peeks
        while self._heap and not self._heap[0].enabled:
            popped = heapq.heappop(self._heap)
            self._registry.pop(popped.event_id, None)

        return self._heap[0] if self._heap else None

    def get_pending_events(self) -> List[Event]:
        """Returns a filtered list of all currently active enabled events."""
        return [evt for evt in self._registry.values() if evt.enabled]

    def event_exists(self, event_id: str) -> bool:
        """Validates structural tracking registration parameters."""
        return event_id in self._registry and self._registry[event_id].enabled

    def get_due_events(self, current_tick: int) -> List[Event]:
        """
        Extracts and registers all actionable operations up to the specified timeline boundary limit.

        Args:
            current_tick (int): Bound constraint limit up to which items should be processed.
        Returns:
            List[Event]: Sorted collection of valid sequential events.
        """
        due_events: List[Event] = []
        
        while True:
            next_event = self.peek_next_event()
            if not next_event or next_event.scheduled_tick > current_tick:
                break
            
            # Dequeue item from execution queue structures safely
            event = heapq.heappop(self._heap)
            self._registry.pop(event.event_id, None)
            
            if event.enabled:
                due_events.append(event)
                
        return due_events

    def execute_due_events(self, current_tick: int, network_state: Any) -> int:
        """
        Drains the execution queues up to the designated absolute tracking tick, passing the
        central state payload directly to external module hooks. Handles recurrence sequencing transparently.

        Args:
            current_tick (int): Timeline step target constraint boundary tracking point.
            network_state (Any): Unmanaged generic container payload passed directly into runtime handlers.

        Returns:
            int: The total count of active operational callback loops executed.
        """
        due_events = self.get_due_events(current_tick)
        execution_count = 0

        for event in due_events:
            # Re-verify runtime enabled flags to prevent race condition regressions
            if not event.enabled:
                continue

            try:
                logger.debug(f"Executing Event context -> ID: {event.event_id} | Name: '{event.event_name}'")
                
                # Zero-knowledge handoff execution barrier
                event.callback(**event.parameters)
                
                execution_count += 1
                logger.debug(f"Event Executed successfully -> ID: {event.event_id}")

                # Manage recurring lifecycle chains safely without altering the reference frame parameters
                if event.repeat and event.enabled:
                    next_tick = current_tick + event.repeat_interval
                    recurring_event = Event(
                        event_id=event.event_id,
                        event_name=event.event_name,
                        scheduled_tick=next_tick,
                        priority=event.priority,
                        event_type=event.event_type,
                        callback=event.callback,
                        parameters=event.parameters,
                        repeat=True,
                        repeat_interval=event.repeat_interval,
                        enabled=True,
                        description=event.description
                    )
                    self.schedule_event(recurring_event)

            except Exception as exc:
                self.failed_events += 1
                logger.error(
                    f"Event Failed tracking break -> ID: {event.event_id} | "
                    f"Exception trace context crash: {str(exc)}", 
                    exc_info=True
                )
                # Fail-soft strategy: log execution trace breakdowns, isolate context to avoid core thread termination.

        return execution_count