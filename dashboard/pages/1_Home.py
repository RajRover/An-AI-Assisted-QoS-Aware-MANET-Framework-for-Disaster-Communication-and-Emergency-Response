"""dashboard/pages/1_Home.py — System overview and simulation lifecycle controls."""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DASHBOARD_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

import streamlit as st

import components
import utils
from integration.simulation_controller import SimulationStatus, SimulationControllerError

components.configure_page("Home", icon="🛰️")
ctx = utils.ensure_backend()
components.render_sidebar_status(ctx)

components.render_header(
    "AI-Assisted QoS-Aware MANET Framework",
    "Disaster Communication Simulation & Monitoring Dashboard",
)

# ── Top-level metrics ───────────────────────────────────────────────────
snapshot = st.session_state.get("last_snapshot")
active_disasters = ctx.disaster_engine.get_active_disasters()
active_disaster_label = (
    ", ".join(sorted({d.profile.disaster_type.name.title() for d in active_disasters}))
    if active_disasters else "None"
)
network_report = utils.get_network_report(ctx)
if snapshot is not None:
    network_health = utils.fmt_pct(snapshot.network_health)
else:
    network_health = utils.fmt_pct(network_report["network_efficiency"], already_fraction=True)
pipeline_stats = ctx.pipeline.get_statistics()

components.metric_row([
    ("Simulation Status", ctx.controller.status.value),
    ("Active Disaster(s)", active_disaster_label),
    ("Network Health", network_health),
    ("Total Nodes", ctx.network_state.graph.number_of_nodes()),
    ("Mobile Nodes", len(ctx.mobility_manager)),
    ("Messages Processed", pipeline_stats.messages_processed),
])

st.divider()

# ── Simulation lifecycle controls ───────────────────────────────────────
st.subheader("Simulation Controls")
status = ctx.controller.status
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    if st.button("▶️ Start", use_container_width=True,
                 disabled=status not in (SimulationStatus.INITIALIZED, SimulationStatus.READY)):
        try:
            if ctx.controller.status == SimulationStatus.INITIALIZED:
                ctx.controller.initialize()
            snap = ctx.controller.run_tick()
            st.session_state.last_snapshot = snap
            st.toast("Simulation started — tick 1 processed.")
        except SimulationControllerError as exc:
            st.error(f"Could not start simulation: {exc}")
        st.rerun()

with c2:
    if st.button("⏸️ Pause", use_container_width=True,
                 disabled=status != SimulationStatus.RUNNING):
        try:
            ctx.controller.pause()
            st.toast("Simulation paused.")
        except SimulationControllerError as exc:
            st.error(f"Could not pause: {exc}")
        st.rerun()

with c3:
    if st.button("⏵ Resume", use_container_width=True,
                 disabled=status != SimulationStatus.PAUSED):
        try:
            ctx.controller.resume()
            st.toast("Simulation resumed.")
        except SimulationControllerError as exc:
            st.error(f"Could not resume: {exc}")
        st.rerun()

with c4:
    if st.button("⏹️ Stop", use_container_width=True,
                 disabled=status in (SimulationStatus.STOPPED, SimulationStatus.COMPLETED)):
        try:
            ctx.controller.stop()
            st.toast("Simulation stopped.")
        except SimulationControllerError as exc:
            st.error(f"Could not stop: {exc}")
        st.rerun()

with c5:
    if st.button("🔁 Reset", use_container_width=True, type="secondary"):
        utils.full_reset()
        st.toast("Full system reset complete.")
        st.rerun()

st.divider()

# ── Latest tick snapshot ────────────────────────────────────────────────
st.subheader("Latest Tick Snapshot")
if snapshot is None:
    st.info("No ticks have been executed yet. Press **Start** or visit the "
            "Simulation Control page to advance the simulation.")
else:
    components.metric_row([
        ("Tick", snapshot.current_tick),
        ("Sim. Time (ms)", f"{snapshot.simulation_time:,.0f}"),
        ("Packet Delivery Ratio", utils.fmt_pct(snapshot.packet_delivery_ratio)),
        ("Avg. Latency", utils.fmt_ms(snapshot.average_latency)),
        ("Active Mobile Nodes", snapshot.active_mobile_nodes),
    ])

st.divider()

# ── Active disasters ─────────────────────────────────────────────────────
st.subheader(f"Active Disasters ({len(active_disasters)})")
if not active_disasters:
    st.caption("No active disaster incidents. Use the Disaster Scenarios page to trigger one.")
else:
    for instance in active_disasters:
        components.disaster_card(instance)