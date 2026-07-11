#!/usr/bin/env python3
"""
streamlit_app.py

Streamlit web front-end for the AI-Assisted QoS-Aware MANET Framework
for Disaster Communication.

This mirrors simulation_runner.py's SimulationRunner exactly — same
subsystem construction, same function/method calls into the
`communication` and `simulation` packages — but renders through
Streamlit widgets instead of a text menu / input() loop.

Calls ONLY functions/methods that simulation_runner.py already calls.
No routing algorithms, disaster logic, or graph generation implemented here.

Run with:
    streamlit run streamlit_app.py
"""

import os
import sys
import logging

import streamlit as st
import pandas as pd
import networkx as nx

# ── Path bootstrap: add project root so imports work regardless of cwd ────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

# ── Communication modules ──────────────────────────────────────────────────
try:
    from communication.graph import build_network
    from communication.routing import (
        calculate_route,
        calculate_route_metrics,
        validate_route,
    )
    from communication.network_stats import generate_network_report
    from communication.visualization import draw_qos_network
except ImportError as e:
    st.error(f"Failed to import communication modules: {e}")
    st.stop()

# ── Simulation modules ─────────────────────────────────────────────────────
try:
    from simulation.network_state import NetworkState
    from simulation.disaster_profiles import DisasterProfileManager, DisasterType
    from simulation.event_scheduler import EventScheduler
    from simulation.network_updater import NetworkUpdater
    from simulation.disaster_engine import DisasterEngine, DisasterStage
    from simulation.simulation_clock import SimulationClock, SimulationClockStatus
except ImportError as e:
    st.error(f"Failed to import simulation modules: {e}")
    st.stop()

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StreamlitSimulationRunner")

_DISASTER_TYPES = {
    "Flood": DisasterType.FLOOD,
    "Cyclone": DisasterType.CYCLONE,
    "Earthquake": DisasterType.EARTHQUAKE,
    "Fire": DisasterType.FIRE,
    "Landslide": DisasterType.LANDSLIDE,
}
_KNOWN_ZONES = ["North Zone", "South Zone", "Central Zone"]

MENU_OPTIONS = [
    "District Network Summary",
    "Display All Nodes",
    "Find Best Communication Route",
    "Show ALL Possible Communication Routes",
    "Compare Routes",
    "Start Disaster Simulation",
    "Advance One Tick",
    "Run Simulation Until Completion",
    "Display Network Statistics",
    "Display Active Disasters",
    "Display Pending Events",
    "Reset Simulation",
]

st.set_page_config(page_title="QoS-Aware MANET Framework", layout="wide")


# ══════════════════════════════════════════════════════════════════════════
# Small utilities (identical logic to simulation_runner.py)
# ══════════════════════════════════════════════════════════════════════════

def _qos_score(metrics: dict) -> float:
    """Heuristic QoS score: higher = better."""
    lat = max(metrics.get("total_latency_ms", 1.0), 0.01)
    bw = max(metrics.get("bottleneck_bandwidth_mbps", 1.0), 0.01)
    loss = max(metrics.get("total_packet_loss_percent", 0.0), 0.0)
    rel = max(metrics.get("overall_reliability_percent", 1.0), 0.01)
    return round((rel * bw) / (lat * (1.0 + loss)), 4)


def _route_status(metrics: dict) -> str:
    rel = metrics.get("overall_reliability_percent", 100.0)
    loss = metrics.get("total_packet_loss_percent", 0.0)
    if rel < 80.0 or loss > 10.0:
        return "CRITICAL"
    if rel < 92.0 or loss > 5.0:
        return "DEGRADED"
    return "HEALTHY"


# ══════════════════════════════════════════════════════════════════════════
# Subsystem initialization — same wiring as SimulationRunner.initialize_simulation
# ══════════════════════════════════════════════════════════════════════════

def initialize_simulation():
    """Build the graph and wire every subsystem. Stores everything in
    st.session_state. Returns (True, None) on success or (False, error)."""
    try:
        graph = build_network()
        state = NetworkState(initial_graph=graph)
        profile_mgr = DisasterProfileManager(populate_defaults=True)
        scheduler = EventScheduler()
        updater = NetworkUpdater(network_state=state)
        disaster_eng = DisasterEngine(
            network_state=state,
            event_scheduler=scheduler,
            network_updater=updater,
            profile_manager=profile_mgr,
        )
        clock = SimulationClock(
            network_state=state,
            event_scheduler=scheduler,
            disaster_engine=disaster_eng,
            network_updater=updater,
            tick_duration_ms=100.0,
            max_ticks=10000,
            real_time_mode=False,
            auto_stop=False,
        )
        updater.recompute_global_telemetry()

        st.session_state.graph = graph
        st.session_state.state = state
        st.session_state.profile_mgr = profile_mgr
        st.session_state.scheduler = scheduler
        st.session_state.updater = updater
        st.session_state.disaster_eng = disaster_eng
        st.session_state.clock = clock
        st.session_state.run_history = []
        st.session_state.initialized = True
        return True, None
    except Exception as exc:
        logger.error("Initialization error: %s", exc, exc_info=True)
        return False, str(exc)


if "initialized" not in st.session_state:
    ok, err = initialize_simulation()
    if not ok:
        st.error(f"Initialization failed: {err}")
        st.stop()


# ══════════════════════════════════════════════════════════════════════════
# Shared helper: node ID list
# ══════════════════════════════════════════════════════════════════════════

def node_ids():
    return sorted(st.session_state.state.graph.nodes())


def tick_summary_row():
    """Same fields as SimulationRunner._display_tick_summary, as a dict."""
    clock = st.session_state.clock
    state = st.session_state.state
    disaster_eng = st.session_state.disaster_eng
    g = state.graph

    tick = clock.current_tick
    metrics = state.global_metrics
    active = disaster_eng.get_active_disasters()
    stage = active[0].current_stage.name if active else "NONE"

    off_nodes = sum(1 for _, d in g.nodes(data=True) if str(d.get("status", "")).upper() == "OFFLINE")
    off_towers = sum(
        1 for _, d in g.nodes(data=True)
        if d.get("type") == "Tower" and str(d.get("status", "")).upper() == "OFFLINE"
    )
    h_load, h_cnt = 0, 0
    for _, d in g.nodes(data=True):
        if d.get("type") == "Hospital":
            h_cnt += 1
            h_load += d.get("hospital_utilization", 0)
    avg_h = h_load / h_cnt if h_cnt else 0

    return {
        "Tick": tick,
        "Stage": stage,
        "Events Executed": clock.metrics.total_events_executed,
        "Avg Latency (ms)": round(metrics.get("average_latency", 0.0), 2),
        "Avg Bandwidth (Mbps)": round(metrics.get("average_bandwidth", 0.0), 2),
        "Avg Packet Loss": round(metrics.get("average_packet_loss", 0.0), 4),
        "Offline Nodes": off_nodes,
        "Offline Towers": off_towers,
        "Hospital Load": round(avg_h, 1),
    }


def render_route_visualization(path, title):
    """Mirrors _trigger_route_visualization / _trigger_disaster_visualization,
    rendered via st.pyplot instead of plt.show()."""
    import matplotlib.pyplot as plt

    g = st.session_state.state.graph
    pos = {n: (g.nodes[n]["x"], g.nodes[n]["y"]) for n in g.nodes()}
    fig, ax = plt.subplots(figsize=(11, 9))

    draw_qos_network(g, ax=ax, show=False)

    if path:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(g, pos, edgelist=path_edges, ax=ax, width=5, edge_color="#e74c3c", alpha=0.9)
        nx.draw_networkx_nodes(g, pos, nodelist=[path[0]], ax=ax, node_size=1200,
                                node_color="#27ae60", edgecolors="black", linewidths=3)
        nx.draw_networkx_nodes(g, pos, nodelist=[path[-1]], ax=ax, node_size=1200,
                                node_color="#c0392b", edgecolors="black", linewidths=3)
        if len(path) > 2:
            nx.draw_networkx_nodes(g, pos, nodelist=path[1:-1], ax=ax, node_size=1100,
                                    node_color="#f39c12", edgecolors="black", linewidths=2.5)

    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_damage_visualization():
    import matplotlib.pyplot as plt

    g = st.session_state.state.graph
    pos = {n: (g.nodes[n]["x"], g.nodes[n]["y"]) for n in g.nodes()}
    fig, ax = plt.subplots(figsize=(11, 9))

    draw_qos_network(g, ax=ax, show=False)

    offline_towers = [n for n, d in g.nodes(data=True)
                       if d.get("type") == "Tower" and str(d.get("status", "")).upper() == "OFFLINE"]
    if offline_towers:
        nx.draw_networkx_nodes(g, pos, nodelist=offline_towers, ax=ax, node_size=1300,
                                node_color="#e74c3c", node_shape="X", edgecolors="black", linewidths=3)

    hospitals = [n for n, d in g.nodes(data=True) if d.get("type") == "Hospital"]
    if hospitals:
        nx.draw_networkx_nodes(g, pos, nodelist=hospitals, ax=ax, node_size=1100,
                                node_color="#e74c3c", edgecolors="#c0392b", linewidths=3)

    camps = [n for n, d in g.nodes(data=True) if d.get("type") == "ReliefCamp"]
    if camps:
        nx.draw_networkx_nodes(g, pos, nodelist=camps, ax=ax, node_size=1100,
                                node_color="#9b59b6", edgecolors="#8e44ad", linewidths=3)

    villages = [n for n, d in g.nodes(data=True) if d.get("type") == "Village"]
    if villages:
        nx.draw_networkx_nodes(g, pos, nodelist=villages, ax=ax, node_size=1000,
                                node_color="#2ecc71", edgecolors="#27ae60", linewidths=2.5)

    damaged = [(u, v) for u, v, d in g.edges(data=True)
               if d.get("status", "Healthy") not in ("Healthy", "OPERATIONAL")]
    if damaged:
        nx.draw_networkx_edges(g, pos, edgelist=damaged, ax=ax, width=4,
                                edge_color="#e74c3c", style="dashed", alpha=0.8)

    ax.set_title("Damaged Network Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# Page renderers — one per original CLI menu option
# ══════════════════════════════════════════════════════════════════════════

def page_network_summary():
    st.header("District Network Summary")
    g = st.session_state.state.graph
    report = generate_network_report(g)

    lats, bws = [], []
    for _, _, d in g.edges(data=True):
        lats.append(d.get("latency", 0.0))
        bws.append(d.get("bandwidth", 0.0))
    avg_lat = sum(lats) / len(lats) if lats else 0.0
    avg_bw = sum(bws) / len(bws) if bws else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Nodes", report["total_nodes"])
    c1.metric("Total Links", report["total_links"])
    c2.metric("Connected Components", report["connected_components"])
    c2.metric("Average Degree", f"{report['average_degree']:.2f}")
    c3.metric("Diameter", report["diameter"])
    c3.metric("Average Latency", f"{avg_lat:.2f} ms")
    st.metric("Average Bandwidth", f"{avg_bw:.2f} Mbps")


def page_display_nodes():
    st.header("All Network Nodes")
    g = st.session_state.state.graph
    rows = []
    for node_id, data in g.nodes(data=True):
        rows.append({
            "Node ID": node_id,
            "Name": data.get("name", "-"),
            "Type": data.get("type", "-"),
            "Status": data.get("status", "-"),
            "Capacity": data.get("capacity", "-"),
            "Load": data.get("current_load", data.get("load", "-")),
            "Zone": data.get("zone", "-"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_best_route():
    st.header("Find Best Communication Route")
    ids = node_ids()
    col1, col2 = st.columns(2)
    src = col1.selectbox("Source Node", ids, key="best_route_src")
    dst = col2.selectbox("Destination Node", ids, key="best_route_dst")

    if st.button("Find Route", type="primary"):
        result = calculate_route(st.session_state.state.graph, src, dst)
        if not result.get("success"):
            st.error(result.get("error", "No route found."))
            return

        path = result["path"]
        metrics = result["metrics"]
        qos = _qos_score(metrics)
        st.session_state["last_best_route_path"] = path

        st.success(" -> ".join(path))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Hop Count", metrics["hop_count"])
        c2.metric("Latency", f"{metrics['total_latency_ms']:.2f} ms")
        c3.metric("Bandwidth", f"{metrics['bottleneck_bandwidth_mbps']:.2f} Mbps")
        c4.metric("Packet Loss", f"{metrics['total_packet_loss_percent']:.2f} %")
        c5.metric("Reliability", f"{metrics['overall_reliability_percent']:.2f} %")
        st.metric("Estimated QoS Score", qos)

    if st.session_state.get("last_best_route_path") and st.checkbox("Visualize this route"):
        path = st.session_state["last_best_route_path"]
        render_route_visualization(path, f"Route: {' -> '.join(path)}")


def page_all_possible_routes():
    st.header("Show ALL Possible Communication Routes")
    ids = node_ids()
    col1, col2 = st.columns(2)
    src = col1.selectbox("Source Node", ids, key="all_routes_src")
    dst = col2.selectbox("Destination Node", ids, key="all_routes_dst")

    if st.button("Compute All Routes", type="primary"):
        g = st.session_state.state.graph
        try:
            all_paths = list(nx.all_simple_paths(g, source=src, target=dst, cutoff=8))
        except nx.NodeNotFound as exc:
            st.error(str(exc))
            return

        if not all_paths:
            st.warning(f"No reachable paths between '{src}' and '{dst}'.")
            return

        routes = []
        for idx, path in enumerate(all_paths, 1):
            m = calculate_route_metrics(g, path)
            qos = _qos_score(m)
            routes.append({"number": idx, "path": path, "metrics": m, "qos": qos})

        recommended = max(routes, key=lambda r: r["qos"])
        st.session_state["last_recommended_path"] = recommended["path"]

        table_rows = []
        for r in routes:
            m = r["metrics"]
            table_rows.append({
                "Route": f"#{r['number']}" + (" (RECOMMENDED)" if r["number"] == recommended["number"] else ""),
                "Path": " -> ".join(r["path"]),
                "Hops": m["hop_count"],
                "Latency (ms)": round(m["total_latency_ms"], 2),
                "Bandwidth (Mbps)": round(m["bottleneck_bandwidth_mbps"], 2),
                "Packet Loss (%)": round(m["total_packet_loss_percent"], 2),
                "Reliability (%)": round(m["overall_reliability_percent"], 2),
                "QoS": r["qos"],
            })

        st.success(f"Found {len(routes)} path(s).")
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.subheader("Sorted Views")
        tabs = st.tabs(["Lowest Latency", "Highest Bandwidth", "Highest Reliability", "Lowest Packet Loss"])
        with tabs[0]:
            st.dataframe(pd.DataFrame(table_rows).sort_values("Latency (ms)"), use_container_width=True, hide_index=True)
        with tabs[1]:
            st.dataframe(pd.DataFrame(table_rows).sort_values("Bandwidth (Mbps)", ascending=False), use_container_width=True, hide_index=True)
        with tabs[2]:
            st.dataframe(pd.DataFrame(table_rows).sort_values("Reliability (%)", ascending=False), use_container_width=True, hide_index=True)
        with tabs[3]:
            st.dataframe(pd.DataFrame(table_rows).sort_values("Packet Loss (%)"), use_container_width=True, hide_index=True)

    if st.session_state.get("last_recommended_path") and st.checkbox("Visualize recommended route"):
        path = st.session_state["last_recommended_path"]
        render_route_visualization(path, f"Recommended Route: {' -> '.join(path)}")


def page_compare_routes():
    st.header("Compare Routes")
    ids = node_ids()
    col1, col2 = st.columns(2)
    src = col1.selectbox("Source Node", ids, key="cmp_src")
    dst = col2.selectbox("Destination Node", ids, key="cmp_dst")

    if st.button("Compare", type="primary"):
        g = st.session_state.state.graph
        try:
            all_paths = list(nx.all_simple_paths(g, source=src, target=dst, cutoff=8))
        except nx.NodeNotFound as exc:
            st.error(str(exc))
            return
        if not all_paths:
            st.warning(f"No routes found between '{src}' and '{dst}'.")
            return

        rows = []
        for idx, path in enumerate(all_paths, 1):
            m = calculate_route_metrics(g, path)
            qos = _qos_score(m)
            rows.append({
                "label": f"Route {idx}",
                "hops": m["hop_count"],
                "latency": m["total_latency_ms"],
                "bandwidth": m["bottleneck_bandwidth_mbps"],
                "loss": m["total_packet_loss_percent"],
                "reliability": m["overall_reliability_percent"],
                "qos": qos,
                "status": _route_status(m),
            })

        df = pd.DataFrame([{
            "Route": r["label"], "Hops": r["hops"],
            "Latency (ms)": round(r["latency"], 2),
            "Bandwidth (Mbps)": round(r["bandwidth"], 2),
            "Loss (%)": round(r["loss"], 2),
            "Reliability (%)": round(r["reliability"], 2),
            "QoS": r["qos"], "Status": r["status"],
        } for r in rows])
        st.dataframe(df, use_container_width=True, hide_index=True)

        fastest = min(rows, key=lambda r: r["latency"])
        reliable = max(rows, key=lambda r: r["reliability"])
        emergency = max(rows, key=lambda r: r["qos"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Fastest", fastest["label"], f"{fastest['latency']:.2f} ms")
        c2.metric("Most Reliable", reliable["label"], f"{reliable['reliability']:.2f} %")
        c3.metric("Best for Emergency", emergency["label"], f"QoS {emergency['qos']:.4f}")


def page_start_disaster():
    st.header("Start Disaster Simulation")
    with st.form("start_disaster_form"):
        dtype_label = st.selectbox("Disaster Type", list(_DISASTER_TYPES.keys()))
        severity = st.slider("Severity", 1, 5, 3)
        duration = st.number_input("Duration (ticks)", min_value=1, value=50, step=1)
        zone = st.selectbox("Affected Zone", _KNOWN_ZONES)
        submitted = st.form_submit_button("Create Disaster", type="primary")

    if submitted:
        dtype = _DISASTER_TYPES[dtype_label]
        clock = st.session_state.clock
        disaster_id = f"DISASTER_{dtype.name}_{clock.current_tick}"
        start_tick = clock.current_tick + 1
        try:
            instance = st.session_state.disaster_eng.create_disaster(
                disaster_id=disaster_id,
                disaster_type=dtype,
                severity_level=severity,
                affected_zones=[zone],
                start_tick=start_tick,
                duration_ticks=int(duration),
            )
            st.success(f"Disaster '{disaster_id}' registered.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Type", dtype.name)
            c2.metric("Severity", f"{severity}/5")
            c3.metric("Starts at", f"Tick {instance.start_tick}")
            c4.metric("Ends at", f"Tick {instance.end_tick}")
        except Exception as exc:
            st.error(f"Could not create disaster: {exc}")
            logger.error("create_disaster failed: %s", exc, exc_info=True)


def page_advance_tick():
    st.header("Advance One Tick")
    if st.button("Advance 1 Tick", type="primary"):
        continued = st.session_state.clock.step()
        row = tick_summary_row()
        st.session_state.run_history.append(row)

        cols = st.columns(4)
        cols[0].metric("Tick", row["Tick"])
        cols[1].metric("Stage", row["Stage"])
        cols[2].metric("Avg Latency", f"{row['Avg Latency (ms)']} ms")
        cols[3].metric("Avg Bandwidth", f"{row['Avg Bandwidth (Mbps)']} Mbps")
        st.json(row)
        if not continued:
            st.info("Simulation reached a terminal state.")


def page_run_until_completion():
    st.header("Run Simulation Until Completion")
    if st.button("Run Until Completion", type="primary"):
        clock = st.session_state.clock
        disaster_eng = st.session_state.disaster_eng
        scheduler = st.session_state.scheduler

        history = []
        limit = 10000
        status_box = st.empty()
        for _ in range(limit):
            continued = clock.step()
            row = tick_summary_row()
            history.append(row)
            status_box.write(f"Tick {row['Tick']} — Stage: {row['Stage']}")

            active = disaster_eng.get_active_disasters()
            pending = scheduler.get_pending_events()

            if not continued:
                st.info("Simulation clock reached terminal state.")
                break
            if row["Tick"] > 1 and not active and not pending:
                st.info("All disasters resolved and event queue empty.")
                clock.stop()
                break

        st.session_state.run_history = history
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)

    if st.session_state.run_history and st.checkbox("Visualize damaged network"):
        render_damage_visualization()


def page_display_stats():
    st.header("Complete Network Statistics")
    g = st.session_state.state.graph
    report = generate_network_report(g)
    stats = st.session_state.state.get_statistics()

    st.subheader("Topology")
    st.table(pd.DataFrame(
        [(k.replace("_", " ").title(), v) for k, v in report.items()],
        columns=["Metric", "Value"],
    ))

    st.subheader("Network Metrics")
    nm = stats.get("network_metrics", {})
    st.table(pd.DataFrame(
        [(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else v) for k, v in nm.items()],
        columns=["Metric", "Value"],
    ))

    st.subheader("Simulation Metrics")
    sm = stats.get("simulation_metrics", {})
    st.table(pd.DataFrame(
        [(k.replace("_", " ").title(), v) for k, v in sm.items()],
        columns=["Metric", "Value"],
    ))


def page_active_disasters():
    st.header("Active Disasters")
    active = st.session_state.disaster_eng.get_active_disasters()
    if not active:
        st.info("No active disasters at this time.")
        return

    clock = st.session_state.clock
    for inst in active:
        remaining = max(0, inst.end_tick - clock.current_tick)
        with st.expander(f"{inst.disaster_id}  —  {inst.current_stage.name}", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Severity", f"{inst.severity_level} / 5")
            c2.metric("Current Stage", inst.current_stage.name)
            c3.metric("Remaining Duration", f"{remaining} ticks")
            st.write(f"**Affected Nodes ({len(inst.affected_nodes)}):** {sorted(inst.affected_nodes)}")
            st.write(f"**Affected Links ({len(inst.affected_links)}):**")
            for u, v in sorted(inst.affected_links):
                st.write(f"- {u} <-> {v}")


def page_pending_events():
    st.header("Event Scheduler Priority Queue")
    events = st.session_state.scheduler.get_pending_events()
    if not events:
        st.info("Event queue is currently empty.")
        return

    sorted_events = sorted(events, key=lambda e: (e.scheduled_tick, e.priority.value))
    df = pd.DataFrame([{
        "Tick": e.scheduled_tick,
        "Priority": e.priority.name,
        "Description": e.event_name,
    } for e in sorted_events])
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_reset_simulation():
    st.header("Reset Simulation")
    st.warning("This will reset the network, clock, and event queue to their initial state.")
    if st.button("Reset to Initial State", type="primary"):
        try:
            if st.session_state.clock.status == SimulationClockStatus.RUNNING:
                st.session_state.clock.stop()
            st.session_state.clock.reset()
        except Exception:
            pass

        st.session_state.state.reset()
        st.session_state.scheduler.clear_events()
        ok, err = initialize_simulation()
        if ok:
            st.success("Simulation reset to initial state.")
        else:
            st.error(f"Reset failed during re-initialization: {err}")


# ══════════════════════════════════════════════════════════════════════════
# App shell
# ══════════════════════════════════════════════════════════════════════════

st.title("AI-Assisted QoS-Aware MANET Framework")
st.caption("Disaster Communication Simulation — Streamlit Interface")

with st.sidebar:
    st.header("Menu")
    choice = st.radio("Select an option", MENU_OPTIONS, label_visibility="collapsed")
    st.divider()
    st.caption(f"Current Tick: {st.session_state.clock.current_tick}")
    active_now = st.session_state.disaster_eng.get_active_disasters()
    st.caption(f"Active Disasters: {len(active_now)}")

PAGE_DISPATCH = {
    "District Network Summary": page_network_summary,
    "Display All Nodes": page_display_nodes,
    "Find Best Communication Route": page_best_route,
    "Show ALL Possible Communication Routes": page_all_possible_routes,
    "Compare Routes": page_compare_routes,
    "Start Disaster Simulation": page_start_disaster,
    "Advance One Tick": page_advance_tick,
    "Run Simulation Until Completion": page_run_until_completion,
    "Display Network Statistics": page_display_stats,
    "Display Active Disasters": page_active_disasters,
    "Display Pending Events": page_pending_events,
    "Reset Simulation": page_reset_simulation,
}

PAGE_DISPATCH[choice]()