"""
nodes.py
--------
Node data model for the AI-Assisted QoS-Aware MANET Framework for
Disaster Communication.

This module is intentionally decoupled from graph construction (graph.py),
edge/QoS computation (edges.py) and rendering (visualization.py). It owns
one responsibility only: describing *what a node is* at a given instant.

Because a live disaster-response engine and a MANET mobility model will
mutate these attributes at runtime (battery drain, load changes, tower
failures, evacuee counts, etc.), every node is represented first as a
dataclass (so the rest of the codebase gets validation, defaults and
autocomplete) and then flattened to a plain dict via `as_dict()`. Down-
stream modules (graph.py, edges.py, routing.py, ...) consume the plain
dict form so nothing else in the framework needs to know dataclasses
exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List


class NodeType(str, Enum):
    """Enumerates every physical/logical entity that can sit on the graph."""
    CONTROL_CENTRE = "ControlCentre"
    POLICE_HQ = "PoliceHQ"
    FIRE_STATION = "FireStation"
    HOSPITAL = "Hospital"
    RELIEF_CAMP = "ReliefCamp"
    TOWER = "Tower"
    VILLAGE = "Village"
    UTILITY = "Utility"


@dataclass
class BaseNode:
    """Common fields shared by every node in the network, regardless of type."""

    id: str
    name: str
    type: NodeType
    x: float
    y: float
    status: str = "Active"
    capacity: int = 100
    current_load: int = 0
    network_status: str = "Healthy"
    zone: str = "Central Zone"
    sector: str = "Central Sector"
    district: str = "Central"

    def as_dict(self) -> Dict[str, Any]:
        """Flatten this node (and any subclass fields) to a plain dict.

        `dataclasses.asdict` walks the full MRO of a dataclass instance,
        so subclasses automatically include their extra fields here.
        The NodeType enum is stringified so the rest of the codebase
        (which predates this refactor) can keep treating `type` as a str.
        """
        raw = asdict(self)
        raw["type"] = self.type.value if isinstance(self.type, NodeType) else self.type
        return raw


@dataclass
class ControlCentreNode(BaseNode):
    """District Control Centre - the root of the command hierarchy."""
    command_capacity: int = 0


@dataclass
class HQNode(BaseNode):
    """Generic emergency-services HQ (Police HQ, Fire Station, Disaster
    Management Office, ...). No extra type-specific attributes are
    mandated by spec beyond the common BaseNode fields, but the subclass
    exists so these entities are still individually addressable and
    extensible later (e.g. `fleet_size`, `on_duty_officers`)."""
    pass


@dataclass
class TowerNode(BaseNode):
    """A MANET relay tower providing wireless coverage to villages."""
    connected_nodes: int = 0
    coverage_radius: float = 0.0
    battery_level: int = 100


@dataclass
class HospitalNode(BaseNode):
    """A medical facility with bed and emergency-intake capacity."""
    beds: int = 0
    available_beds: int = 0
    emergency_capacity: int = 0


@dataclass
class VillageNode(BaseNode):
    """A population centre relying on a single connected tower for uplink."""
    population: int = 0
    emergency_requests: int = 0
    connected_tower: str = ""


@dataclass
class ReliefCampNode(BaseNode):
    """A temporary shelter/relief camp with finite occupancy."""
    occupancy: int = 0
    max_capacity: int = 0


@dataclass
class UtilityNode(BaseNode):
    """Critical infrastructure such as power substations or water plants."""
    utility_type: str = "Generic"


def _build_catalog() -> List[BaseNode]:
    """Construct the canonical node catalog for the district network.

    Coordinates and topology mirror the reference network map (Sahyadri
    Nagar district): a District Control Centre hub, emergency-services
    HQs, two hospitals, two relief camps, five MANET towers, six
    flood/landslide-affected villages, and two utility sites.
    """
    return [
        ControlCentreNode(
            id="C1", name="District Control Centre", type=NodeType.CONTROL_CENTRE,
            x=300.0, y=300.0, status="Active", capacity=500, current_load=40,
            network_status="Healthy", command_capacity=1000,
            zone="Central Zone", sector="Central-East", district="Central",
        ),
        HQNode(
            id="P1", name="Police HQ", type=NodeType.POLICE_HQ,
            x=281.0, y=350.4, status="Active", capacity=200, current_load=20,
            network_status="Healthy",
            zone="North Zone", sector="North-West", district="Central",
        ),
        HQNode(
            id="F1", name="Fire Station", type=NodeType.FIRE_STATION,
            x=350.6, y=333.9, status="Active", capacity=200, current_load=15,
            network_status="Healthy",
            zone="North Zone", sector="North-West", district="Central",
        ),
        HospitalNode(
            id="H1", name="Civil Hospital", type=NodeType.HOSPITAL,
            x=381.1, y=171.9, status="Active", capacity=120, current_load=48,
            network_status="Healthy", beds=120, available_beds=72, emergency_capacity=30,
            zone="North Zone", sector="North-East", district="Central",
        ),
        HospitalNode(
            id="H2", name="Community Hospital", type=NodeType.HOSPITAL,
            x=454.2, y=193.2, status="Active", capacity=100, current_load=30,
            network_status="Healthy", beds=100, available_beds=70, emergency_capacity=20,
            zone="South Zone", sector="South-East", district="Central",
        ),
        ReliefCampNode(
            id="R1", name="Relief Camp A", type=NodeType.RELIEF_CAMP,
            x=78.5, y=443.1, status="Active", capacity=300, current_load=50,
            network_status="Healthy", occupancy=150, max_capacity=300,
            zone="North Zone", sector="North-West", district="Central",
        ),
        ReliefCampNode(
            id="R2", name="Relief Camp B", type=NodeType.RELIEF_CAMP,
            x=487.6, y=412.3, status="Active", capacity=300, current_load=65,
            network_status="Healthy", occupancy=195, max_capacity=300,
            zone="South Zone", sector="South-East", district="Central",
        ),
        TowerNode(
            id="T1", name="Tower 1", type=NodeType.TOWER,
            x=236.4, y=230.1, status="Active", capacity=100, current_load=30,
            network_status="Healthy", connected_nodes=2, coverage_radius=120.0, battery_level=88,
            zone="Central Zone", sector="Central-West", district="Central",
        ),
        TowerNode(
            id="T2", name="Tower 2", type=NodeType.TOWER,
            x=178.6, y=308.2, status="Low Battery", capacity=100, current_load=45,
            network_status="Healthy", connected_nodes=4, coverage_radius=120.0, battery_level=22,
            zone="North Zone", sector="North-West", district="Central",
        ),
        TowerNode(
            id="T3", name="Tower 3", type=NodeType.TOWER,
            x=325.2, y=419.0, status="Active", capacity=100, current_load=20,
            network_status="Healthy", connected_nodes=3, coverage_radius=120.0, battery_level=95,
            zone="North Zone", sector="North-West", district="Central",
        ),
        TowerNode(
            id="T4", name="Tower 4", type=NodeType.TOWER,
            x=410.2, y=306.4, status="Active", capacity=100, current_load=35,
            network_status="Healthy", connected_nodes=3, coverage_radius=120.0, battery_level=80,
            zone="North Zone", sector="North-East", district="Central",
        ),
        TowerNode(
            id="T5", name="Tower 5", type=NodeType.TOWER,
            x=421.1, y=102.6, status="Active", capacity=100, current_load=10,
            network_status="Healthy", connected_nodes=2, coverage_radius=120.0, battery_level=97,
            zone="South Zone", sector="South-East", district="Central",
        ),
        VillageNode(
            id="V1", name="Ramnagar", type=NodeType.VILLAGE,
            x=109.9, y=283.0, status="Evacuating", capacity=100, current_load=0,
            network_status="Healthy", population=5400, emergency_requests=0, connected_tower="T2",
            zone="North Zone", sector="North-West", district="Central",
        ),
        VillageNode(
            id="V2", name="Shivapur", type=NodeType.VILLAGE,
            x=178.1, y=168.4, status="Stable", capacity=100, current_load=0,
            network_status="Healthy", population=3200, emergency_requests=0, connected_tower="T1",
            zone="Central Zone", sector="Central-West", district="Central",
        ),
        VillageNode(
            id="V3", name="Hanumanwadi", type=NodeType.VILLAGE,
            x=260.1, y=472.9, status="Stable", capacity=100, current_load=0,
            network_status="Healthy", population=4100, emergency_requests=0, connected_tower="T3",
            zone="North Zone", sector="North-West", district="Central",
        ),
        VillageNode(
            id="V4", name="Lakshmipur", type=NodeType.VILLAGE,
            x=482.4, y=298.1, status="Stable", capacity=100, current_load=0,
            network_status="Healthy", population=2800, emergency_requests=0, connected_tower="T4",
            zone="South Zone", sector="South-East", district="Central",
        ),
        VillageNode(
            id="V5", name="River Colony", type=NodeType.VILLAGE,
            x=501.5, y=99.4, status="Stable", capacity=100, current_load=0,
            network_status="Healthy", population=1500, emergency_requests=0, connected_tower="T5",
            zone="South Zone", sector="South-East", district="Central",
        ),
        VillageNode(
            id="V6", name="Hill View", type=NodeType.VILLAGE,
            x=230.0, y=380.0, status="Offline", capacity=100, current_load=0,
            network_status="Weak", population=1900, emergency_requests=5, connected_tower="T2",
            zone="North Zone", sector="North-West", district="Central",
        ),
        UtilityNode(
            id="U1", name="Power Substation", type=NodeType.UTILITY,
            x=240.2, y=227.1, status="Active", capacity=150, current_load=60,
            network_status="Healthy", utility_type="Power",
            zone="Central Zone", sector="Central-West", district="Central",
        ),
        UtilityNode(
            id="U2", name="Water Plant", type=NodeType.UTILITY,
            x=361.3, y=251.7, status="Active", capacity=150, current_load=40,
            network_status="Healthy", utility_type="Water",
            zone="Central Zone", sector="Central-East", district="Central",
        ),
    ]


def get_nodes_data() -> Dict[str, Dict[str, Any]]:
    """Return the canonical node catalog as a dict keyed by node id.

    Each value is a flattened plain dict (not a dataclass instance) so
    that graph.py, edges.py and every other consumer can keep treating
    nodes as ordinary dicts, e.g. `nodes["T2"]["battery_level"]`.
    """
    return {node.id: node.as_dict() for node in _build_catalog()}


if __name__ == "__main__":
    nodes = get_nodes_data()
    print(f"Nodes loaded: {len(nodes)}")
    for node_id, attrs in nodes.items():
        print(f"  {node_id}: {attrs['name']} ({attrs['type']})")