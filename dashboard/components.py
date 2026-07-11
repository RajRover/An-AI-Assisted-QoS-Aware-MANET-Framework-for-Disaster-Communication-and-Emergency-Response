"""
dashboard/components.py
------------------------
Reusable, presentation-only Streamlit UI building blocks shared across pages.

Nothing in this module computes QoS, routing, disaster, or classification
results -- it only renders data already produced by the backend.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

# ─────────────────────────────────────────────────────────────────────────
# Path bootstrap: must happen BEFORE importing anything from the backend
# packages (communication/, services/, etc.). This mirrors utils.py's own
# bootstrap so components.py works regardless of which module gets
# imported first (app.py and every page import `components` before
# `utils`, so components.py cannot rely on utils.py having already set
# up sys.path).
# ─────────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from communication.visualization import draw_qos_network

import utils


# ============================================================================
# PAGE CHROME
# ============================================================================

def configure_page(title: str, icon: str = "🛰️") -> None:
    """Sets Streamlit page config once and injects the shared stylesheet."""
    try:
        st.set_page_config(page_title=f"{title} · MANET Dashboard", page_icon=icon, layout="wide")
    except Exception:
        pass  # already configured earlier in this script run
    _inject_css()


def _inject_css() -> None:
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as fh:
            st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)


def render_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="manet-header"><h1>{title}</h1>'
                f'<p>{subtitle}</p></div>', unsafe_allow_html=True)


def render_sidebar_status(ctx: "utils.BackendContext") -> None:
    """Compact always-visible system status panel shown in the sidebar of every page."""
    with st.sidebar:
        st.markdown("### 🛰️ System Status")
        status = ctx.controller.status
        st.markdown(
            f'<span class="status-pill" style="background:{controller_status_color(status)}">'
            f'{status.value}</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Nodes: {ctx.network_state.graph.number_of_nodes()} · "
                   f"Mobile units: {len(ctx.mobility_manager)} · "
                   f"Active disasters: {len(ctx.disaster_engine.get_active_disasters())}")
        st.divider()


def controller_status_color(status) -> str:
    mapping = {
        "RUNNING": "#2ecc71",
        "READY": "#3498db",
        "PAUSED": "#f1c40f",
        "STOPPED": "#e74c3c",
        "COMPLETED": "#9b59b6",
        "FAILED": "#e74c3c",
        "INITIALIZED": "#95a5a6",
    }
    return mapping.get(status.value, "#95a5a6")


# ============================================================================
# METRIC CARDS
# ============================================================================

def metric_row(items: Sequence[tuple]) -> None:
    """Renders a row of st.metric cards. Each item: (label, value, delta_or_None)."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value = item[0], item[1]
        delta = item[2] if len(item) > 2 else None
        with col:
            st.metric(label, value, delta=delta)


def status_pill(text: str, color: Optional[str] = None) -> str:
    color = color or utils.status_color(text)
    return f'<span class="status-pill" style="background:{color}">{text}</span>'


# ============================================================================
# NETWORK GRAPH VISUALIZATION (matplotlib, delegates drawing to backend)
# ============================================================================

def render_network_figure(
    graph,
    mobile_nodes: Optional[List[Dict[str, Any]]] = None,
    manet_graph=None,
    highlight_route: Optional[List[str]] = None,
    figsize: tuple = (12, 9),
):
    """Builds the network figure by delegating the core draw to the backend's
    own communication.visualization.draw_qos_network(), then layering
    dashboard-only overlays (mobile units, MANET links, highlighted route)
    on top -- no QoS/topology computation happens here.
    """
    fig, ax = plt.subplots(figsize=figsize)
    draw_qos_network(graph, ax=ax, show=False)

    pos = {n: (d["x"], d["y"]) for n, d in graph.nodes(data=True)}

    # Overlay: highlighted route (already computed by the backend's routing engine)
    if highlight_route and len(highlight_route) >= 2:
        xs = [pos[n][0] for n in highlight_route if n in pos]
        ys = [pos[n][1] for n in highlight_route if n in pos]
        ax.plot(xs, ys, color="#ff00ff", linewidth=4.0, alpha=0.85, zorder=5,
                solid_capstyle="round", label="Active Route")
        ax.scatter(xs, ys, s=180, facecolors="none", edgecolors="#ff00ff", linewidths=2.5, zorder=6)

    # Overlay: mobile nodes (already computed positions from MobilityManager)
    if mobile_nodes:
        mx = [m["x"] for m in mobile_nodes]
        my = [m["y"] for m in mobile_nodes]
        colors = [utils.status_color(m.get("status", "")) for m in mobile_nodes]
        ax.scatter(mx, my, s=140, marker="^", c=colors, edgecolors="black",
                   linewidths=1.2, zorder=7, label="Mobile Units")
        for m in mobile_nodes:
            ax.annotate(m["id"], (m["x"], m["y"]), fontsize=6.5, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points", zorder=8)

    # Overlay: temporary MANET connectivity edges
    if manet_graph is not None and manet_graph.number_of_edges() > 0:
        static_pos = pos
        mobile_pos = {m["id"]: (m["x"], m["y"]) for m in (mobile_nodes or [])}
        combined_pos = {**static_pos, **mobile_pos}
        for u, v in manet_graph.edges():
            if u in combined_pos and v in combined_pos:
                x1, y1 = combined_pos[u]
                x2, y2 = combined_pos[v]
                ax.plot([x1, x2], [y1, y2], color="#00bcd4", linewidth=1.0,
                        linestyle="--", alpha=0.6, zorder=3)

    ax.set_title("Disaster Response Network Map — Live View", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ============================================================================
# PACKET TABLE
# ============================================================================

PACKET_COLUMNS = [
    "packet_id", "message_id", "priority", "status", "source_node",
    "destination_node", "current_node", "hop_count", "latency_ms",
    "ttl", "age_ms", "packet_type",
]

PACKET_COLUMN_LABELS = {
    "packet_id": "Packet ID",
    "message_id": "Message ID",
    "priority": "Priority",
    "status": "Status",
    "source_node": "Source",
    "destination_node": "Destination",
    "current_node": "Current Node",
    "hop_count": "Hop Count",
    "latency_ms": "Latency (ms)",
    "ttl": "TTL",
    "age_ms": "Age (ms)",
    "packet_type": "Packet Type",
}


def packets_dataframe(packets: List[Any]) -> pd.DataFrame:
    """Converts live NetworkPacket objects into the Packet Monitor table.

    Reads .to_dict() fresh on every call so age_ms / status / hop_count
    reflect the packet's current live state.
    """
    if not packets:
        return pd.DataFrame(columns=[PACKET_COLUMN_LABELS[c] for c in PACKET_COLUMNS])

    rows = []
    for pkt in packets:
        d = pkt.to_dict()
        rows.append({
            "packet_id": d["packet_id"],
            "message_id": d["message_id"],
            "priority": d["priority"],
            "status": d["status"],
            "source_node": d["source_node"],
            "destination_node": d.get("destination_node") or "—",
            "current_node": d["current_node"],
            "hop_count": d["hop_count"],
            "latency_ms": round(d.get("end_to_end_delay_ms") or d.get("latency_ms", 0.0), 3),
            "ttl": d["ttl"],
            "age_ms": round(d["age_ms"], 1),
            "packet_type": d["packet_type"],
        })
    df = pd.DataFrame(rows)[PACKET_COLUMNS]
    return df.rename(columns=PACKET_COLUMN_LABELS)


# ============================================================================
# DISASTER CARDS
# ============================================================================

def disaster_card(instance) -> None:
    color = utils.status_color("Broken" if instance.current_stage.value in ("PEAK", "ACTIVE") else "Congested")
    st.markdown(
        f"""
        <div class="disaster-card">
            <div class="disaster-card-title">{instance.profile.name}
                <span class="status-pill" style="background:{color}">{instance.current_stage.value}</span>
            </div>
            <div class="disaster-card-meta">
                ID: {instance.disaster_id} &nbsp;|&nbsp;
                Type: {instance.profile.disaster_type.name} &nbsp;|&nbsp;
                Severity Tier: {instance.severity_level}/5 &nbsp;|&nbsp;
                Affected Nodes: {len(instance.affected_nodes)} &nbsp;|&nbsp;
                Window: tick {instance.start_tick} → {instance.end_tick}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# AUTO REFRESH (lightweight, no external dependency)
# ============================================================================

def auto_refresh_control(key_prefix: str, default_interval: int = 5) -> None:
    """Renders an auto-refresh toggle + interval slider and reruns the page
    on a timer when enabled. Kept dependency-free (no streamlit-autorefresh)."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        enabled = st.checkbox("🔄 Auto-refresh", key=f"{key_prefix}_autorefresh")
    with col2:
        interval = st.slider("Interval (seconds)", 2, 30, default_interval,
                              key=f"{key_prefix}_interval", disabled=not enabled)
    with col3:
        st.button("Refresh now", key=f"{key_prefix}_refresh_btn")

    if enabled:
        import time
        time.sleep(interval)
        st.rerun()