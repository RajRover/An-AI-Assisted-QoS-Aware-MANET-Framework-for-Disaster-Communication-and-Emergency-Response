"""
edges.py
--------
Communication-link (edge) model for the QoS-Aware MANET Framework.

Every link between two nodes is assigned realistic, *derived* QoS
metrics rather than fixed constants. Two things drive the numbers:

1.  The **link category** (which kinds of node it connects), which sets
    baseline expectations - a Tower<->Tower backbone link is provisioned
    very differently from a Village<->Tower access link.
2.  The **Euclidean distance** between the two endpoints, which degrades
    bandwidth/signal and inflates latency/loss/jitter the further apart
    two radios are.

All magic numbers live in named constants at the top of the file so the
model can be re-tuned without touching the calculation logic.
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple

Node = Dict[str, Any]
Edge = Tuple[str, str, Dict[str, Any]]


# --------------------------------------------------------------------------- #
# Topology: which node pairs are physically linked.
# --------------------------------------------------------------------------- #
LINK_LIST: List[Tuple[str, str]] = [
    ("C1", "P1"), ("C1", "F1"), ("P1", "F1"),
    ("C1", "T1"), ("C1", "T2"), ("C1", "T3"), ("C1", "T4"), ("C1", "T5"),
    ("C1", "H1"), ("H1", "H2"),
    ("C1", "U1"), ("C1", "U2"),
    ("V1", "T2"), ("V2", "T1"),
    ("V3", "T3"), ("V6", "T2"), ("V6", "T3"),
    ("V4", "T4"), ("V5", "T5"),
    ("R1", "T2"), ("R2", "T4"), ("R2", "H2"),
]


# --------------------------------------------------------------------------- #
# Baseline QoS profiles per link category.
# Format: (base_bandwidth_mbps, base_latency_ms, base_reliability_pct,
#          base_packet_loss_pct, base_jitter_ms)
# --------------------------------------------------------------------------- #
_TOWER = "Tower"
_HOSPITAL = "Hospital"
_CONTROL = "ControlCentre"
_VILLAGE = "Village"
_UTILITY = "Utility"

QOS_PROFILES: Dict[str, Tuple[float, float, float, float, float]] = {
    "tower_tower":    (150.0, 4.0, 99.5, 0.2, 1.5),   # backbone microwave/mesh link
    "control_tower":  (200.0, 3.0, 99.8, 0.1, 1.0),   # control centre uplink, top priority
    "hospital_tower": (120.0, 5.0, 99.9, 0.1, 1.0),   # highest reliability, medical priority
    "village_tower":  (60.0, 10.0, 97.0, 1.5, 4.0),   # last-mile access link
    "utility_tower":  (90.0, 7.0, 98.5, 0.8, 2.5),    # SCADA/telemetry link
    "generic":        (80.0, 6.0, 98.0, 0.5, 2.0),    # HQ-HQ, HQ-Hospital, HQ-Utility, etc.
}

# Distance normalisation: beyond this many map-units, degradation saturates.
MAX_EFFECTIVE_DISTANCE = 400.0

# How strongly distance is allowed to erode/inflate each metric.
BANDWIDTH_DISTANCE_PENALTY = 0.45   # up to 45% bandwidth loss at MAX_EFFECTIVE_DISTANCE
LATENCY_DISTANCE_FACTOR = 0.05      # ms added per map-unit of distance (propagation + relay)
PACKET_LOSS_DISTANCE_PENALTY = 3.0  # extra loss percentage points at MAX_EFFECTIVE_DISTANCE
JITTER_DISTANCE_PENALTY = 3.0       # extra jitter ms at MAX_EFFECTIVE_DISTANCE
SIGNAL_DISTANCE_PENALTY = 55.0      # signal_strength points lost at MAX_EFFECTIVE_DISTANCE
RELIABILITY_DISTANCE_PENALTY = 2.5  # reliability points lost at MAX_EFFECTIVE_DISTANCE


def _euclidean_distance(a: Node, b: Node) -> float:
    """Straight-line distance between two nodes' map coordinates."""
    return math.sqrt((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2)


def _distance_ratio(distance: float) -> float:
    """Normalise distance to [0, 1] against MAX_EFFECTIVE_DISTANCE."""
    return min(distance / MAX_EFFECTIVE_DISTANCE, 1.0)


def _classify_link(type_u: str, type_v: str) -> str:
    """Pick the QoS profile key for a pair of node types (order-independent)."""
    pair = {type_u, type_v}
    if pair == {_TOWER}:
        return "tower_tower"
    if _CONTROL in pair and _TOWER in pair:
        return "control_tower"
    if _HOSPITAL in pair and _TOWER in pair:
        return "hospital_tower"
    if _VILLAGE in pair and _TOWER in pair:
        return "village_tower"
    if _UTILITY in pair and _TOWER in pair:
        return "utility_tower"
    return "generic"


def _status_from_metrics(packet_loss: float, reliability: float) -> str:
    """Derive a human-readable link status from computed QoS numbers."""
    if reliability < 90.0 or packet_loss > 5.0:
        return "Broken"
    if reliability < 96.0 or packet_loss > 2.5:
        return "Weak"
    if reliability < 99.0 or packet_loss > 1.0:
        return "Congested"
    return "Healthy"


def _compute_edge_attributes(u: Node, v: Node) -> Dict[str, Any]:
    """Derive full QoS attribute set for a single edge between two nodes."""
    distance = round(_euclidean_distance(u, v), 2)
    ratio = _distance_ratio(distance)

    category = _classify_link(u["type"], v["type"])
    base_bw, base_latency, base_reliability, base_loss, base_jitter = QOS_PROFILES[category]

    bandwidth = round(base_bw * (1 - BANDWIDTH_DISTANCE_PENALTY * ratio), 2)
    latency = round(base_latency + LATENCY_DISTANCE_FACTOR * distance, 2)
    packet_loss = round(base_loss + PACKET_LOSS_DISTANCE_PENALTY * ratio, 2)
    jitter = round(base_jitter + JITTER_DISTANCE_PENALTY * ratio, 2)
    signal_strength = round(max(100.0 - SIGNAL_DISTANCE_PENALTY * ratio, 5.0), 2)
    reliability = round(max(base_reliability - RELIABILITY_DISTANCE_PENALTY * ratio, 50.0), 2)

    # Capacity/utilisation: derive current usage from the two endpoints'
    # own load so busier nodes naturally saturate their links first.
    load_u = u.get("current_load", 0) / max(u.get("capacity", 1), 1)
    load_v = v.get("current_load", 0) / max(v.get("capacity", 1), 1)
    utilisation = min((load_u + load_v) / 2, 1.0)
    capacity = round(bandwidth, 2)
    current_usage = round(bandwidth * utilisation, 2)

    status = _status_from_metrics(packet_loss, reliability)

    return {
        "distance": distance,
        "weight": distance,  # kept for backward compatibility with routing weights
        "latency": latency,
        "bandwidth": bandwidth,
        "packet_loss": packet_loss,
        "jitter": jitter,
        "signal_strength": signal_strength,
        "reliability": reliability,
        "capacity": capacity,
        "current_usage": current_usage,
        "status": status,
        "category": category,
    }


def get_edges_data(nodes_dict: Dict[str, Node]) -> List[Edge]:
    """Build the full list of (u, v, attrs) edges for the given node set.

    Silently skips any link in LINK_LIST that references a node id not
    present in `nodes_dict`, so the topology and node catalog can evolve
    independently without one hard-crashing the other.
    """
    detailed_edges: List[Edge] = []
    for u_id, v_id in LINK_LIST:
        if u_id not in nodes_dict or v_id not in nodes_dict:
            continue
        attrs = _compute_edge_attributes(nodes_dict[u_id], nodes_dict[v_id])
        detailed_edges.append((u_id, v_id, attrs))
    return detailed_edges


if __name__ == "__main__":
    try:
        from .nodes import get_nodes_data
    except (ImportError, ValueError):
        from nodes import get_nodes_data

    nodes = get_nodes_data()
    edges = get_edges_data(nodes)
    print(f"Nodes loaded: {len(nodes)}")
    print(f"Edges generated: {len(edges)}")
    for u, v, attrs in edges[:5]:
        print(f"  {u}-{v}: {attrs}")