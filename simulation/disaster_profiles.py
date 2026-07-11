"""
disaster_profiles.py

Provides the declarative configuration layer for disaster profiles within the 
AI-Assisted QoS-Aware MANET Framework for Disaster Communication. Enforces strict
structural validation rules using Python dataclasses and extensible validation descriptors.

Author: Simulation Framework Designer
License: MIT
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Configure Module Logger
logger = logging.getLogger("DisasterMANET.DisasterProfiles")


# ============================================================================
# ENUMS & CUSTOM EXCEPTIONS
# ============================================================================

class DisasterType(Enum):
    """Supported disaster variants for topological and operational categorization."""
    FLOOD = "FLOOD"
    CYCLONE = "CYCLONE"
    EARTHQUAKE = "EARTHQUAKE"
    FIRE = "FIRE"
    LANDSLIDE = "LANDSLIDE"


class DisasterProfileException(Exception):
    """Base exception for profile schema and registration anomalies."""
    pass


class ProfileValidationError(DisasterProfileException):
    """Raised when a disaster profile fails structural validation constraints."""
    pass


class ProfileNotFoundError(DisasterProfileException):
    """Raised when querying a profile that is missing from the registry."""
    pass


class DuplicateProfileError(DisasterProfileException):
    """Raised when attempting to add an already registered disaster profile type."""
    pass


# ============================================================================
# DISASTER PROFILE DEFINITION
# ============================================================================

@dataclass
class DisasterProfile:
    """
    Encapsulates structural multipliers, damage probabilities, and baseline tracking 
    constants mapping a specific disaster's impact onto the MANET architecture.
    """
    # General Information
    name: str
    description: str
    disaster_type: DisasterType
    default_severity: float                      # Normalized scale [0.0, 1.0]
    min_severity: float                          # Lower bounds [0.0, 1.0]
    max_severity: float                          # Upper bounds [0.0, 1.0]
    default_duration: int                        # Measured in simulation ticks
    growth_rate: float                           # Extent of expansion per tick

    # Communication Impact (QoS Multipliers)
    latency_multiplier: float                    # Impact on propagation time (values > 1.0 degrade QoS)
    bandwidth_multiplier: float                  # Impact on bandwidth (values < 1.0 degrade capacity)
    packet_loss_multiplier: float                # Multiplier tracking link drop increases
    jitter_multiplier: float                     # Variation factor for latency variance
    signal_strength_multiplier: float            # Decibel attenuation factor on wireless transceivers
    link_failure_probability: float              # Random baseline drop chance [0.0, 1.0]

    # Infrastructure Impact (Failure Probabilities)
    tower_failure_probability: float             # Physical telecommunication base degradation risk
    road_failure_probability: float              # Land corridor network disruption probability
    bridge_failure_probability: float            # Choke point river crossway severance risk
    utility_failure_probability: float           # Grid power or supply failure risk
    hospital_load_increment: int                 # Absolute surge factor for bed request tracking
    relief_camp_load_increment: int              # Absolute surge factor for refugee population mapping

    # Node Impact
    node_failure_probability: float              # Transceiver unit hardware destruction probability
    emergency_request_multiplier: float          # Surge tracking multiplier for regional alerts
    population_affected_percentage: float        # Demographics density factor inside impact radius

    # Recovery Parameters
    recovery_rate: float                         # Automated or assistance reduction rate per tick
    repair_probability: float                    # Link or infrastructure regeneration success rate

    def validate(self) -> None:
        """
        Executes structural assertions ensuring compliance with the framework validation specification.
        Raises ProfileValidationError on any semantic or value range breach.
        """
        # Validate Probabilities and Percentages [0.0, 1.0]
        prob_fields = {
            "default_severity": self.default_severity,
            "min_severity": self.min_severity,
            "max_severity": self.max_severity,
            "link_failure_probability": self.link_failure_probability,
            "tower_failure_probability": self.tower_failure_probability,
            "road_failure_probability": self.road_failure_probability,
            "bridge_failure_probability": self.bridge_failure_probability,
            "utility_failure_probability": self.utility_failure_probability,
            "node_failure_probability": self.node_failure_probability,
            "population_affected_percentage": self.population_affected_percentage,
            "recovery_rate": self.recovery_rate,
            "repair_probability": self.repair_probability
        }
        for name, value in prob_fields.items():
            if not (0.0 <= value <= 1.0):
                raise ProfileValidationError(f"Constraint Violation: '{name}' value {value} must be between 0.0 and 1.0.")

        # Logical Boundary Validation for Severity Ranges
        if self.min_severity > self.max_severity:
            raise ProfileValidationError(
                f"Logical Error: min_severity ({self.min_severity}) cannot exceed max_severity ({self.max_severity})."
            )
        if not (self.min_severity <= self.default_severity <= self.max_severity):
            raise ProfileValidationError(
                f"Logical Error: default_severity ({self.default_severity}) must fall within range "
                f"[{self.min_severity}, {self.max_severity}]."
            )

        # Validate Strict Positives (> 0)
        positive_fields = {
            "default_duration": self.default_duration,
            "latency_multiplier": self.latency_multiplier,
            "bandwidth_multiplier": self.bandwidth_multiplier,
            "packet_loss_multiplier": self.packet_loss_multiplier,
            "jitter_multiplier": self.jitter_multiplier,
            "signal_strength_multiplier": self.signal_strength_multiplier,
            "emergency_request_multiplier": self.emergency_request_multiplier
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ProfileValidationError(f"Constraint Violation: '{name}' value {value} must be strictly greater than 0.")

        # Validate Structural Floor Constraints (>= 0)
        if self.growth_rate < 0:
            raise ProfileValidationError(f"Constraint Violation: growth_rate ({self.growth_rate}) cannot be negative.")
        if self.hospital_load_increment < 0:
            raise ProfileValidationError(f"Constraint Violation: hospital_load_increment cannot be negative.")
        if self.relief_camp_load_increment < 0:
            raise ProfileValidationError(f"Constraint Violation: relief_camp_load_increment cannot be negative.")


# ============================================================================
# DISASTER PROFILE MANAGER
# ============================================================================

class DisasterProfileManager:
    """
    Central storage context and registration authority handling lifecycle mutations,
    validation compliance, and structural queries for disaster profiles.
    """

    def __init__(self, populate_defaults: bool = True):
        """Initializes an empty registry and injects reference baselines if flagged."""
        self._profiles: Dict[DisasterType, DisasterProfile] = {}
        
        if populate_defaults:
            self._load_default_profiles()

    def get_profile(self, disaster_type: DisasterType) -> DisasterProfile:
        """Retrieves a read-only copy of the target disaster specification profile."""
        if not self.profile_exists(disaster_type):
            raise ProfileNotFoundError(f"Lookup failure: No profile registered for disaster type '{disaster_type.name}'.")
        return self._profiles[disaster_type]

    def list_profiles(self) -> List[DisasterProfile]:
        """Lists all registered disaster profiles within the active context layer."""
        return list(self._profiles.values())

    def add_profile(self, profile: DisasterProfile) -> None:
        """Validates and registers a new disaster configuration profile."""
        if not isinstance(profile, DisasterProfile):
            raise TypeError("Supplied profile parameter must be a valid instance of DisasterProfile.")
        
        if self.profile_exists(profile.disaster_type):
            raise DuplicateProfileError(
                f"Registration collision: Profile type '{profile.disaster_type.name}' already exists. Use remove_profile first."
            )
            
        profile.validate()
        self._profiles[profile.disaster_type] = profile
        logger.info(f"Successfully validated and registered disaster profile: [{profile.disaster_type.name}] - {profile.name}")

    def remove_profile(self, disaster_type: DisasterType) -> DisasterProfile:
        """Removes a profile from the registration ledger context and returns it."""
        if not self.profile_exists(disaster_type):
            raise ProfileNotFoundError(f"Removal failed: No profile matching '{disaster_type.name}' registered.")
        
        removed_profile = self._profiles.pop(disaster_type)
        logger.warning(f"Disaster profile registration revoked: [{disaster_type.name}]")
        return removed_profile

    def profile_exists(self, disaster_type: DisasterType) -> bool:
        """Validates if the structural signature type is loaded inside the runtime context ledger."""
        return disaster_type in self._profiles

    # ------------------------------------------------------------------------
    # FACTORY REFERENCE DEFAULT IMPLEMENTATIONS
    # ------------------------------------------------------------------------

    def _load_default_profiles(self) -> None:
        """Injects highly specialized profiles matching targeted disaster communication dynamics."""
        defaults = [
            # 1. FLOOD PROFILE
            DisasterProfile(
                name="Severe Regional Inundation",
                description=" caractérisé par un engorgement de zone entraînant des pannes de lignes terrestres importantes.",
                disaster_type=DisasterType.FLOOD,
                default_severity=0.65,
                min_severity=0.20,
                max_severity=1.00,
                default_duration=120,
                growth_rate=0.03,
                latency_multiplier=1.8,
                bandwidth_multiplier=0.45,       # Medium bandwidth degradation (reduced to 45%)
                packet_loss_multiplier=3.0,
                jitter_multiplier=2.5,
                signal_strength_multiplier=0.75,
                link_failure_probability=0.35,
                tower_failure_probability=0.25,     # Moderate tower failures
                road_failure_probability=0.70,      # High road failures
                bridge_failure_probability=0.60,
                utility_failure_probability=0.50,
                hospital_load_increment=35,
                relief_camp_load_increment=120,
                node_failure_probability=0.15,
                emergency_request_multiplier=4.5,   # Large increase in emergency requests
                population_affected_percentage=0.60,
                recovery_rate=0.02,
                repair_probability=0.40
            ),
            # 2. CYCLONE PROFILE
            DisasterProfile(
                name="Tropical Cyclone Impact",
                description="High velocity winds causing severe transceiver link attenuation and physical tower collapse.",
                disaster_type=DisasterType.CYCLONE,
                default_severity=0.80,
                min_severity=0.40,
                max_severity=1.00,
                default_duration=60,
                growth_rate=0.05,
                latency_multiplier=2.5,
                bandwidth_multiplier=0.30,
                packet_loss_multiplier=5.0,
                jitter_multiplier=4.0,
                signal_strength_multiplier=0.20,   # Severe RF attenuation (reduced to 20% strength)
                link_failure_probability=0.65,
                tower_failure_probability=0.60,     # High tower failures
                road_failure_probability=0.40,      # Moderate road failures
                bridge_failure_probability=0.25,
                utility_failure_probability=0.80,
                hospital_load_increment=50,
                relief_camp_load_increment=200,
                node_failure_probability=0.40,
                emergency_request_multiplier=5.0,
                population_affected_percentage=0.85,
                recovery_rate=0.01,
                repair_probability=0.20
            ),
            # 3. EARTHQUAKE PROFILE
            DisasterProfile(
                name="High-Magnitude Seismic Event",
                description="Catastrophic infrastructure collapse resulting in severed core links and critical hospital surges.",
                disaster_type=DisasterType.EARTHQUAKE,
                default_severity=0.85,
                min_severity=0.50,
                max_severity=1.00,
                default_duration=30,
                growth_rate=0.00,                  # Instantaneous burst impact, no continuous spatial growth
                latency_multiplier=1.5,
                bandwidth_multiplier=0.50,
                packet_loss_multiplier=2.5,
                jitter_multiplier=2.0,
                signal_strength_multiplier=0.80,
                link_failure_probability=0.50,
                tower_failure_probability=0.50,
                road_failure_probability=0.65,
                bridge_failure_probability=0.85,    # High bridge failures
                utility_failure_probability=0.75,   # High infrastructure failures
                hospital_load_increment=150,        # Very high hospital load surge
                relief_camp_load_increment=180,
                node_failure_probability=0.45,
                emergency_request_multiplier=6.0,
                population_affected_percentage=0.70,
                recovery_rate=0.005,
                repair_probability=0.15
            ),
            # 4. FIRE PROFILE
            DisasterProfile(
                name="Localized Flash Fire / Wildfire",
                description="Thermal hazards causing localized tracking node losses and mild communication adjustments.",
                disaster_type=DisasterType.FIRE,
                default_severity=0.40,
                min_severity=0.10,
                max_severity=0.80,
                default_duration=90,
                growth_rate=0.08,                  # Rapid localized expansion
                latency_multiplier=1.2,
                bandwidth_multiplier=0.80,       # Moderate communication degradation
                packet_loss_multiplier=1.5,
                jitter_multiplier=1.3,
                signal_strength_multiplier=0.90,
                link_failure_probability=0.20,
                tower_failure_probability=0.20,
                road_failure_probability=0.30,
                bridge_failure_probability=0.05,
                utility_failure_probability=0.40,
                hospital_load_increment=20,
                relief_camp_load_increment=40,
                node_failure_probability=0.60,      # High local hardware node failure risk
                emergency_request_multiplier=3.0,
                population_affected_percentage=0.15, # Localized footprint
                recovery_rate=0.05,
                repair_probability=0.50
            ),
            # 5. LANDSLIDE PROFILE
            DisasterProfile(
                name="Terrain Failure / Landslide",
                description="Localized structural slope failure, blocking transport corridors and isolating rural villages.",
                disaster_type=DisasterType.LANDSLIDE,
                default_severity=0.50,
                min_severity=0.20,
                max_severity=0.75,
                default_duration=45,
                growth_rate=0.01,
                latency_multiplier=1.4,
                bandwidth_multiplier=0.70,
                packet_loss_multiplier=2.0,
                jitter_multiplier=1.8,
                signal_strength_multiplier=0.85,
                link_failure_probability=0.40,
                tower_failure_probability=0.35,     # Moderate tower failures
                road_failure_probability=0.80,      # High localized road blockages
                bridge_failure_probability=0.40,
                utility_failure_probability=0.30,
                hospital_load_increment=15,
                relief_camp_load_increment=50,
                node_failure_probability=0.25,
                emergency_request_multiplier=2.5,
                population_affected_percentage=0.10, # Isolated mountain settlements affected
                recovery_rate=0.03,
                repair_probability=0.30
            )
        ]

        for p in defaults:
            self.add_profile(p)