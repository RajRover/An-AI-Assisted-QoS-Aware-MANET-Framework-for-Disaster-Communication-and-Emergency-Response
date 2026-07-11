"""
network_stats.py
-----------------
Whole-network health metrics for the QoS-Aware MANET Framework.

generate_network_report(G) returns a single dictionary covering
connectivity, topology and QoS-relevant graph-theoretic measures.
Every metric that is undefined for disconnected or empty graphs is
handled explicitly rather than allowed to raise, since a partially
destroyed disaster network is exactly the situation this framework
needs to describe correctly.
"""

from __future__ import annotations

from typing import Any, Dict

import networkx as nx


def _empty_report() -> Dict[str, Any]:
    """Report shape returned for a graph with zero nodes."""
    return {
        "total_nodes": 0,
        "total_links": 0,
        "average_degree": 0.0,
        "graph_density": 0.0,
        "is_connected": False,
        "connected_components": 0,
        "diameter": 0,
        "average_path_length": 0.0,
        "network_efficiency": 0.0,
        "average_clustering_coefficient": 0.0,
    }


def generate_network_report(graph: nx.Graph) -> Dict[str, Any]:
    """Compute a full structural/QoS health report for the network.

    Metrics:
        total_nodes, total_links      - basic size
        average_degree                 - mean connections per node
        graph_density                  - fraction of all possible edges present
        is_connected                   - whether every node can reach every other
        connected_components           - count of disjoint sub-networks
        diameter                        - longest shortest-path; on a
                                          disconnected graph this is computed
                                          on the largest connected component
                                          and labelled accordingly
        average_path_length            - mean shortest-path length (same
                                          largest-component fallback as diameter)
        network_efficiency              - NetworkX global efficiency: how
                                          close the network is to an ideal
                                          fully-connected mesh (1.0 = ideal)
        average_clustering_coefficient - mean local clustering, i.e. how
                                          much neighbors of a node are
                                          themselves interconnected
    """
    if graph.number_of_nodes() == 0:
        return _empty_report()

    num_nodes = graph.number_of_nodes()
    is_connected = nx.is_connected(graph)
    num_components = nx.number_connected_components(graph)

    stats: Dict[str, Any] = {
        "total_nodes": num_nodes,
        "total_links": graph.number_of_edges(),
        "average_degree": round(sum(dict(graph.degree()).values()) / num_nodes, 2),
        "graph_density": round(nx.density(graph), 4),
        "is_connected": is_connected,
        "connected_components": num_components,
        # Global efficiency is well-defined even for disconnected graphs
        # (unreachable pairs simply contribute 0 to the average), so no
        # fallback is needed here.
        "network_efficiency": round(nx.global_efficiency(graph), 4),
        "average_clustering_coefficient": round(nx.average_clustering(graph), 4),
    }

    if is_connected:
        stats["diameter"] = nx.diameter(graph)
        stats["average_path_length"] = round(nx.average_shortest_path_length(graph), 2)
    else:
        largest_cc_nodes = max(nx.connected_components(graph), key=len)
        largest_cc = graph.subgraph(largest_cc_nodes)
        stats["diameter"] = f"{nx.diameter(largest_cc)} (Largest Component)"
        stats["average_path_length"] = (
            f"{round(nx.average_shortest_path_length(largest_cc), 2)} (Largest Component)"
        )

    return stats


if __name__ == "__main__":
    try:
        from .graph import build_network
    except (ImportError, ValueError):
        from graph import build_network

    G = build_network()
    print("Network Report:")
    report = generate_network_report(G)
    for k, v in report.items():
        print(f"  {k}: {v}")

    print("\nEmpty Network Report:")
    empty_G = nx.Graph()
    empty_report = generate_network_report(empty_G)
    for k, v in empty_report.items():
        print(f"  {k}: {v}")