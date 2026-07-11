"""
routing.py
----------
Route computation and QoS-aware path metrics for the MANET framework.

Public functions:
    calculate_route(G, source, target)   - shortest path + full metrics
    calculate_route_metrics(G, path)      - metrics for an already-known path
    validate_route(G, path)                - confirm a path is actually walkable

All functions return structured dictionaries (never raise on "expected"
failure modes like a missing node or no path) so callers - e.g. a live
dashboard - can render an error state instead of crashing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import networkx as nx

logger = logging.getLogger(__name__)


def validate_route(graph: nx.Graph, path: List[str]) -> Dict[str, Any]:
    """Confirm that `path` is a sequence of existing nodes joined by real edges.

    Returns {"valid": bool, "error": Optional[str]}.
    """
    if not path:
        return {"valid": False, "error": "Path is empty."}

    for node in path:
        if node not in graph:
            return {"valid": False, "error": f"Node '{node}' does not exist in the network."}

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not graph.has_edge(u, v):
            return {"valid": False, "error": f"No direct link between '{u}' and '{v}'."}

    return {"valid": True, "error": None}


def calculate_route_metrics(graph: nx.Graph, path: List[str]) -> Dict[str, Any]:
    """Compute aggregate QoS metrics for an already-determined path.

    Assumes `path` has already been validated with validate_route();
    calling this on an invalid path will raise a KeyError, since at that
    point it's a programming error rather than an expected runtime one.
    """
    hop_count = len(path) - 1

    total_latency = 0.0
    min_bandwidth = float("inf")
    total_packet_loss = 0.0
    total_jitter = 0.0
    reliability_product = 1.0

    for i in range(hop_count):
        u, v = path[i], path[i + 1]
        edge = graph[u][v]

        total_latency += edge.get("latency", 0.0)
        min_bandwidth = min(min_bandwidth, edge.get("bandwidth", float("inf")))
        total_packet_loss += edge.get("packet_loss", 0.0)
        total_jitter += edge.get("jitter", 0.0)
        # Reliability is per-hop probability of success; the path's overall
        # reliability is the product across hops, not a sum/average.
        reliability_product *= edge.get("reliability", 100.0) / 100.0

    average_jitter = round(total_jitter / hop_count, 2) if hop_count else 0.0

    return {
        "total_latency_ms": round(total_latency, 2),
        "bottleneck_bandwidth_mbps": round(min_bandwidth, 2) if hop_count else 0.0,
        "total_packet_loss_percent": round(total_packet_loss, 2),
        "average_jitter_ms": average_jitter,
        "hop_count": hop_count,
        "overall_reliability_percent": round(reliability_product * 100, 2),
    }


def calculate_route(graph: nx.Graph, source: str, target: str) -> Dict[str, Any]:
    """Find the shortest path from `source` to `target` and return its metrics.

    Uses edge 'weight' (Euclidean distance) as the shortest-path cost, so
    the "shortest" route is the geographically shortest one; QoS metrics
    for that path are then computed separately and returned alongside it.
    """
    if source not in graph:
        return {"success": False, "error": f"Source node '{source}' not found in the network."}
    if target not in graph:
        return {"success": False, "error": f"Target node '{target}' not found in the network."}

    try:
        path = nx.shortest_path(graph, source=source, target=target, weight="weight")
    except nx.NetworkXNoPath:
        return {"success": False, "error": f"No route exists between '{source}' and '{target}'."}
    except nx.NodeNotFound as exc:
        return {"success": False, "error": str(exc)}

    validation = validate_route(graph, path)
    if not validation["valid"]:
        # Should not normally happen since nx.shortest_path only returns
        # real edges, but guards against a corrupted graph.
        logger.error("Computed path failed validation: %s", validation["error"])
        return {"success": False, "error": validation["error"]}

    metrics = calculate_route_metrics(graph, path)

    return {
        "success": True,
        "path": path,
        "metrics": metrics,
    }


if __name__ == "__main__":
    try:
        from .graph import build_network
    except (ImportError, ValueError):
        from graph import build_network

    G = build_network()

    print("Testing valid route (C1 -> H2):")
    print(calculate_route(G, "C1", "H2"))

    print("\nTesting missing node route (C3 -> H2):")
    print(calculate_route(G, "C3", "H2"))

    print("\nTesting validate_route on a manual path (C1 -> T4 -> R2):")
    print(validate_route(G, ["C1", "T4", "R2"]))