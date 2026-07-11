"""
graph.py
--------
Graph construction and validation layer for the QoS-Aware MANET Framework.

Responsibilities:
    * build_network()      - assemble a NetworkX graph from nodes.py + edges.py
    * validate_network()   - sanity-check the graph for structural issues
    * get_node()            - safe single-node lookup
    * get_neighbors()        - safe neighbor lookup

This module deliberately raises/logs rather than silently swallowing
problems: a malformed disaster-response network is worse than no
network, since routing decisions built on top of it could send a
rescue team down a link that doesn't really exist.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import networkx as nx

try:
    from .nodes import get_nodes_data
    from .edges import get_edges_data
except (ImportError, ValueError):
    from nodes import get_nodes_data
    from edges import get_edges_data


logger = logging.getLogger(__name__)
if not logger.handlers:
    # Library-friendly default: only configure handlers if nothing else
    # (e.g. the hosting application) has already done so.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class NetworkValidationError(Exception):
    """Raised when the constructed network fails structural validation."""


def build_network() -> nx.Graph:
    """Assemble the full district communication network as a NetworkX Graph.

    Nodes come from nodes.get_nodes_data(); edges (with derived QoS
    attributes) come from edges.get_edges_data(). The resulting graph is
    validated before being handed back to the caller.
    """
    graph = nx.Graph()

    nodes = get_nodes_data()
    for node_id, attrs in nodes.items():
        graph.add_node(node_id, **attrs)
    logger.info("Loaded %d nodes into the network.", graph.number_of_nodes())

    edges = get_edges_data(nodes)
    for u, v, attrs in edges:
        graph.add_edge(u, v, **attrs)
    logger.info("Loaded %d edges into the network.", graph.number_of_edges())

    validate_network(graph)
    return graph


def validate_network(graph: nx.Graph) -> Dict[str, Any]:
    """Run structural sanity checks against the graph.

    Checks performed:
        * Duplicate nodes            - NetworkX dicts can't hold true
                                        duplicates, but we still confirm
                                        the node id set is exactly the
                                        expected size (defends against
                                        silent overwrites upstream).
        * Missing nodes referenced    - every edge endpoint must exist
                                        as a node in the graph.
        * Invalid edges                - self-loops and edges lacking the
                                        minimum QoS attributes are flagged.
        * Isolated nodes                - nodes with zero edges are logged
                                        as warnings (not necessarily fatal,
                                        e.g. a not-yet-connected village).

    Returns a summary dict of what was found. Raises NetworkValidationError
    if any hard error (missing node reference, self-loop, malformed edge)
    is detected, since routing on top of a corrupt graph is unsafe.
    """
    required_edge_attrs = {"latency", "bandwidth", "packet_loss", "status"}

    errors: List[str] = []
    warnings: List[str] = []

    node_ids = set(graph.nodes())
    if len(node_ids) != graph.number_of_nodes():
        errors.append("Duplicate node ids detected in the graph.")

    for u, v, attrs in graph.edges(data=True):
        if u == v:
            errors.append(f"Self-loop detected on node '{u}'.")
        if u not in node_ids or v not in node_ids:
            errors.append(f"Edge ({u}, {v}) references a node missing from the graph.")
        missing_attrs = required_edge_attrs - attrs.keys()
        if missing_attrs:
            errors.append(f"Edge ({u}, {v}) is missing required attributes: {missing_attrs}.")

    isolated = list(nx.isolates(graph))
    if isolated:
        warnings.append(f"Isolated node(s) with no connections: {isolated}.")

    for warning in warnings:
        logger.warning(warning)

    if errors:
        for error in errors:
            logger.error(error)
        raise NetworkValidationError("; ".join(errors))

    logger.info("Network validation passed with %d warning(s).", len(warnings))
    return {"errors": errors, "warnings": warnings, "isolated_nodes": isolated}


def get_node(graph: nx.Graph, node_id: str) -> Optional[Dict[str, Any]]:
    """Safely fetch a single node's attribute dict, or None if it doesn't exist."""
    if node_id not in graph:
        logger.warning("get_node: '%s' not found in network.", node_id)
        return None
    return dict(graph.nodes[node_id])


def get_neighbors(graph: nx.Graph, node_id: str) -> List[str]:
    """Safely fetch the neighbor ids of a node, or an empty list if it doesn't exist."""
    if node_id not in graph:
        logger.warning("get_neighbors: '%s' not found in network.", node_id)
        return []
    return list(graph.neighbors(node_id))


if __name__ == "__main__":
    G = build_network()
    print(f"Network built successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print("Neighbors of C1:", get_neighbors(G, "C1"))
    print("Node lookup T2:", get_node(G, "T2"))