"""dashboard/pages/6_Analytics.py — Live charts over simulation tick history
and the session's packet log."""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DASHBOARD_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

import pandas as pd
import streamlit as st

import components
import utils

components.configure_page("Analytics", icon="📊")
ctx = utils.ensure_backend()
components.render_sidebar_status(ctx)

components.render_header("Analytics", "Network & traffic performance over the current session")

# ── Tick-level time series (from SimulationController.snapshots) ─────────
st.subheader("Simulation Tick Trends")
snapshots = ctx.controller.snapshots
if not snapshots:
    st.info("No tick history yet. Advance the simulation on the Simulation Control page.")
else:
    tick_df = pd.DataFrame([{
        "Tick": s.current_tick,
        "Packet Delivery Ratio": s.packet_delivery_ratio,
        "Average Latency (ms)": s.average_latency,
        "Network Health": s.network_health,
        "Active Disasters": s.active_disasters,
        "Active Mobile Nodes": s.active_mobile_nodes,
    } for s in snapshots]).set_index("Tick")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Packet Delivery Ratio**")
        st.line_chart(tick_df[["Packet Delivery Ratio"]])
        st.markdown("**Network Health**")
        st.line_chart(tick_df[["Network Health"]])
    with c2:
        st.markdown("**Average Latency**")
        st.line_chart(tick_df[["Average Latency (ms)"]])
        st.markdown("**Active Disasters / Mobile Nodes**")
        st.line_chart(tick_df[["Active Disasters", "Active Mobile Nodes"]])

st.divider()

# ── Packet-level analytics (from this session's packet log) ──────────────
st.subheader("Packet-Level Traffic Analytics")
packets = st.session_state.get("packet_log", [])

if not packets:
    st.info("No packets generated yet. Submit messages on the Message Console "
            "or run a Disaster Scenario to populate this view.")
else:
    rows = []
    for p in packets:
        d = p.to_dict()
        rows.append({
            "Packet ID": d["packet_id"],
            "Priority": d["priority"],
            "QoS Level": d["qos_level"],
            "Status": d["status"],
            "Hop Count": d["hop_count"],
            "Latency (ms)": d.get("end_to_end_delay_ms") or d.get("latency_ms", 0.0),
            "Packet Loss (%)": d["packet_loss_percent"],
            "Jitter (ms)": d["jitter_ms"],
            "Bandwidth (Mbps)": getattr(p, "min_bandwidth_mbps", None),
        })
    pkt_df = pd.DataFrame(rows)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Packet Loss (%) over packet sequence**")
        st.line_chart(pkt_df[["Packet Loss (%)"]])
        st.markdown("**Jitter (ms) over packet sequence**")
        st.line_chart(pkt_df[["Jitter (ms)"]])
        st.markdown("**Hop Count over packet sequence**")
        st.bar_chart(pkt_df[["Hop Count"]])
    with c2:
        st.markdown("**Bandwidth (Mbps) over packet sequence**")
        bw = pkt_df.dropna(subset=["Bandwidth (Mbps)"])
        if not bw.empty:
            st.line_chart(bw[["Bandwidth (Mbps)"]])
        else:
            st.caption("No bandwidth telemetry recorded yet (packets have not begun hop-by-hop transmission).")

        st.markdown("**Message Priority Distribution**")
        st.bar_chart(pkt_df["Priority"].value_counts())

    st.markdown("**QoS Level Distribution**")
    st.bar_chart(pkt_df["QoS Level"].value_counts())

st.divider()
components.auto_refresh_control("analytics")