"""dashboard/pages/5_Simulation.py — Direct control over the SimulationController
lifecycle: initialize, tick stepping, and run statistics."""

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
from integration.simulation_controller import SimulationStatus, SimulationControllerError

components.configure_page("Simulation Control", icon="⚙️")
ctx = utils.ensure_backend()
components.render_sidebar_status(ctx)

components.render_header("Simulation Control", "Drive the SimulationController's tick-by-tick execution loop")

status = ctx.controller.status
st.markdown(components.status_pill(status.value, components.controller_status_color(status)),
            unsafe_allow_html=True)
st.write("")

r1c1, r1c2, r1c3, r1c4 = st.columns(4)
with r1c1:
    if st.button("🧩 Initialize", use_container_width=True,
                 disabled=status != SimulationStatus.INITIALIZED):
        try:
            ctx.controller.initialize()
            st.toast("Controller initialized → READY.")
        except SimulationControllerError as exc:
            st.error(str(exc))
        st.rerun()

with r1c2:
    if st.button("⏭️ Run Tick", use_container_width=True,
                 disabled=status not in (SimulationStatus.READY, SimulationStatus.RUNNING)):
        try:
            snap = ctx.controller.run_tick()
            st.session_state.last_snapshot = snap
            st.toast(f"Tick {snap.current_tick} complete.")
        except SimulationControllerError as exc:
            st.error(str(exc))
        st.rerun()

with r1c3:
    if st.button("⏭️⏭️ Run 10 Ticks", use_container_width=True,
                 disabled=status not in (SimulationStatus.READY, SimulationStatus.RUNNING)):
        try:
            for _ in range(10):
                snap = ctx.controller.run_tick()
            st.session_state.last_snapshot = snap
            st.toast("10 ticks executed.")
        except SimulationControllerError as exc:
            st.error(str(exc))
        st.rerun()

with r1c4:
    n_ticks = st.number_input("Custom N", min_value=1, max_value=500, value=25, step=5, label_visibility="collapsed")

r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
with r2c1:
    if st.button("Run N Ticks", use_container_width=True,
                 disabled=status not in (SimulationStatus.READY, SimulationStatus.RUNNING)):
        try:
            for _ in range(int(n_ticks)):
                snap = ctx.controller.run_tick()
            st.session_state.last_snapshot = snap
            st.toast(f"{int(n_ticks)} ticks executed.")
        except SimulationControllerError as exc:
            st.error(str(exc))
        st.rerun()
with r2c2:
    if st.button("▶️ Start", use_container_width=True,
                 disabled=status not in (SimulationStatus.INITIALIZED, SimulationStatus.READY)):
        if ctx.controller.status == SimulationStatus.INITIALIZED:
            ctx.controller.initialize()
        st.session_state.last_snapshot = ctx.controller.run_tick()
        st.rerun()
with r2c3:
    if st.button("⏸️ Pause", use_container_width=True, disabled=status != SimulationStatus.RUNNING):
        ctx.controller.pause()
        st.rerun()
with r2c4:
    if st.button("⏵ Resume", use_container_width=True, disabled=status != SimulationStatus.PAUSED):
        ctx.controller.resume()
        st.rerun()
with r2c5:
    if st.button("⏹️ Stop", use_container_width=True,
                 disabled=status in (SimulationStatus.STOPPED, SimulationStatus.COMPLETED)):
        ctx.controller.stop()
        st.rerun()

st.write("")
if st.button("🔁 Full Reset", type="secondary"):
    utils.full_reset()
    st.toast("Simulation fully reset.")
    st.rerun()

st.divider()

st.subheader("Cumulative Statistics")
stats = ctx.controller.get_statistics()
components.metric_row([
    ("Total Ticks", stats.total_ticks),
    ("Messages Processed", stats.messages_processed),
    ("Messages Delivered", stats.messages_delivered),
    ("Messages Failed", stats.messages_failed),
])
components.metric_row([
    ("Packet Delivery Ratio", utils.fmt_pct(stats.packet_delivery_ratio)),
    ("Avg. Latency", utils.fmt_ms(stats.average_latency)),
    ("Avg. Hop Count", utils.fmt_num(stats.average_hop_count)),
    ("Avg. Packet Loss", utils.fmt_pct(stats.average_packet_loss, already_fraction=False)),
])

st.divider()

st.subheader("Tick History")
snapshots = ctx.controller.snapshots
if snapshots:
    df = pd.DataFrame([{
        "Tick": s.current_tick,
        "Sim Time (ms)": s.simulation_time,
        "Active Disasters": s.active_disasters,
        "Active Mobile Nodes": s.active_mobile_nodes,
        "Messages Processed": s.messages_processed,
        "Messages Delivered": s.messages_delivered,
        "Network Health": s.network_health,
        "Avg Latency (ms)": s.average_latency,
        "PDR": s.packet_delivery_ratio,
    } for s in snapshots])
    st.dataframe(df.tail(50), use_container_width=True, hide_index=True)
else:
    st.caption("No ticks executed yet.")

st.divider()
components.auto_refresh_control("simulation_control")