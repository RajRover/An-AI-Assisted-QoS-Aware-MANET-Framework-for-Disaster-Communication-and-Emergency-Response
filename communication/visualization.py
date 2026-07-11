"""
visualization.py
-----------------
Rendering layer for the QoS-Aware MANET Framework.

draw_qos_network(G, ax=None) draws the full network:
    * node color encodes node TYPE (Village, Hospital, Tower, Control
      Centre, Relief Camp, Utility, Police, Fire Station)
    * node border encodes node STATUS (Active / Offline / Low Battery / Evacuating)
    * edge color encodes link QoS STATUS (Healthy / Congested / Weak / Broken)
    * edge labels show latency / bandwidth / packet loss

The function accepts an optional matplotlib Axes (`ax`). When omitted it
creates its own figure and calls plt.show(); when provided (e.g. by a
FuncAnimation callback that clears and redraws the same Axes every
frame), it draws into that Axes only and leaves show()/redraw control to
the caller. This is what makes the module animation-ready without any
changes needed later.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx


# --------------------------------------------------------------------------- #
# Color legends
# --------------------------------------------------------------------------- #
NODE_TYPE_COLORS: Dict[str, str] = {
    "ControlCentre": "#1abc9c",
    "PoliceHQ": "#34495e",
    "FireStation": "#e67e22",
    "Hospital": "#e74c3c",
    "ReliefCamp": "#9b59b6",
    "Tower": "#3498db",
    "Village": "#2ecc71",
    "Utility": "#7f8c8d",
}

NODE_STATUS_BORDER_COLORS: Dict[str, str] = {
    "Offline": "#ff0000",
    "Low Battery": "#ffa500",
    "Evacuating": "#ffcc00",
    "Active": "black",
    "Stable": "black",
}

EDGE_STATUS_COLORS: Dict[str, str] = {
    "Healthy": "#2ecc71",     # green
    "Congested": "#f1c40f",   # yellow
    "Weak": "#e67e22",        # orange
    "Broken": "#e74c3c",      # red
}
DEFAULT_EDGE_COLOR = "#95a5a6"


def _node_fill_colors(graph: nx.Graph) -> list:
    """Resolve each node's fill color from its `type` attribute."""
    return [
        NODE_TYPE_COLORS.get(data.get("type", ""), "#bdc3c7")
        for _, data in graph.nodes(data=True)
    ]


def _node_border_colors(graph: nx.Graph) -> list:
    """Resolve each node's border color from its `status` attribute."""
    return [
        NODE_STATUS_BORDER_COLORS.get(data.get("status", "Active"), "black")
        for _, data in graph.nodes(data=True)
    ]


def _edge_colors(graph: nx.Graph) -> list:
    """Resolve each edge's color from its QoS `status` attribute."""
    return [
        EDGE_STATUS_COLORS.get(data.get("status", ""), DEFAULT_EDGE_COLOR)
        for _, _, data in graph.edges(data=True)
    ]


def _build_legend_handles() -> list:
    """Construct combined node-type and edge-status legend handles."""
    handles = []
    for label, color in NODE_TYPE_COLORS.items():
        handles.append(
            mlines.Line2D([], [], marker="o", linestyle="None", markersize=10,
                          markerfacecolor=color, markeredgecolor="black", label=label)
        )
    for label, color in EDGE_STATUS_COLORS.items():
        handles.append(
            mlines.Line2D([], [], color=color, linewidth=3, label=f"Link: {label}")
        )
    return handles


def draw_qos_network(graph: nx.Graph, ax: Optional[plt.Axes] = None,
                      show: Optional[bool] = None) -> plt.Axes:
    """Draw the QoS-annotated communication network.

    Parameters
    ----------
    graph : nx.Graph
        The network to render (as built by graph.build_network()).
    ax : matplotlib.axes.Axes, optional
        Axes to draw into. If None, a new figure/axes is created. Pass
        an existing Axes (and clear it beforehand) when animating frame
        by frame with matplotlib.animation.FuncAnimation.
    show : bool, optional
        Whether to call plt.show() at the end. Defaults to True only
        when `ax` was not supplied (i.e. standalone single-shot use);
        defaults to False when drawing into a caller-supplied Axes,
        since an animation loop controls its own rendering.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the network was drawn on, so callers building
        animations can keep a handle to it across frames.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(13, 11))
    if show is None:
        show = standalone

    pos = {node: (data["x"], data["y"]) for node, data in graph.nodes(data=True)}

    # --- Edges, colored by QoS status ---
    nx.draw_networkx_edges(
        graph, pos, ax=ax, width=2.2, edge_color=_edge_colors(graph)
    )

    # --- Nodes, filled by type, bordered by status ---
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=950,
        node_color=_node_fill_colors(graph),
        edgecolors=_node_border_colors(graph),
        linewidths=2.2,
    )

    # --- Node labels: id, type, and status ---
    node_labels = {
        node: f"{node}\n{data['type']}\n[{data.get('status', 'Active')}]"
        for node, data in graph.nodes(data=True)
    }
    nx.draw_networkx_labels(graph, pos, labels=node_labels, ax=ax, font_size=7, font_weight="bold")

    # --- Edge labels: latency / bandwidth / packet loss ---
    edge_labels = {
        (u, v): f"L:{d.get('latency', 0)}ms\nB:{d.get('bandwidth', 0)}M\nP:{d.get('packet_loss', 0)}%"
        for u, v, d in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, ax=ax, font_size=6.5, font_color="#2c3e50"
    )

    ax.set_title(
        "Disaster Response Network Map - QoS Link Layer",
        fontsize=14, fontweight="bold",
    )
    ax.legend(handles=_build_legend_handles(), loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True, title="Legend")
    ax.axis("off")

    if standalone:
        plt.tight_layout()

    if show:
        plt.show()

    return ax


if __name__ == "__main__":
    try:
        from .graph import build_network
    except (ImportError, ValueError):
        from graph import build_network

    G = build_network()
    print("Drawing the network. Close the window to continue.")
    draw_qos_network(G)