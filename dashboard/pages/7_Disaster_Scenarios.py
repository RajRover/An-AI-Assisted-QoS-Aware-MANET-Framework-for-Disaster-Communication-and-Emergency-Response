"""dashboard/pages/7_Disaster_Scenarios.py — Trigger disasters (Flood,
Earthquake, Fire, Cyclone, Landslide) via the live DisasterEngine, and
optionally run the backend's predefined scenario catalog."""

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
from simulation.disaster_engine import DisasterEngineException
from integration.scenario_runner import ScenarioRunnerError

components.configure_page("Disaster Scenarios", icon="🌪️")
ctx = utils.ensure_backend()
components.render_sidebar_status(ctx)

components.render_header("Disaster Scenarios", "Inject disaster incidents into the live network topology")

# ── Trigger a disaster ────────────────────────────────────────────────────
st.subheader("Trigger a Disaster")
with st.form("trigger_disaster_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        disaster_type_name = st.selectbox("Disaster Type", utils.DISASTER_TYPE_OPTIONS)
    with c2:
        severity = st.slider("Severity Tier", 1, 5, 3)
    with c3:
        duration_ticks = st.number_input("Duration (ticks)", min_value=5, max_value=500, value=60, step=5)

    zones = st.multiselect("Affected Zones", utils.KNOWN_ZONES, default=[utils.KNOWN_ZONES[0]])
    current_tick = ctx.controller.snapshots[-1].current_tick if ctx.controller.snapshots else 0
    start_tick = st.number_input(
        "Start Tick", min_value=current_tick, value=current_tick,
        help="Disaster activates once the simulation reaches this tick. "
             "Defaults to the current tick so it fires on the next Run Tick.",
    )

    launch = st.form_submit_button("🌪️ Trigger Disaster", type="primary", use_container_width=True)

if launch:
    if not zones:
        st.warning("Select at least one affected zone.")
    else:
        try:
            disaster_id = utils.next_disaster_id(prefix=disaster_type_name[:3])
            ctx.disaster_engine.create_disaster(
                disaster_id=disaster_id,
                disaster_type=utils.disaster_type_from_name(disaster_type_name),
                severity_level=int(severity),
                affected_zones=zones,
                start_tick=int(start_tick),
                duration_ticks=int(duration_ticks),
            )
            st.success(f"Disaster '{disaster_id}' ({disaster_type_name}) scheduled at tick {start_tick}. "
                       f"Advance the simulation (Simulation Control page) to activate it.")
        except DisasterEngineException as exc:
            st.error(f"Could not create disaster: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

st.divider()

# ── Currently tracked disasters ───────────────────────────────────────────
st.subheader("Active Disasters")
active = ctx.disaster_engine.get_active_disasters()
if not active:
    st.caption("No active disasters.")
else:
    for instance in active:
        components.disaster_card(instance)

st.divider()

# ── Predefined scenario catalog (bonus: exercises ScenarioRunner) ────────
st.subheader("Predefined Scenario Catalog")
st.caption("Optional: runs the backend's built-in ScenarioRunner definitions "
           "(message traffic + disaster injection combined) end-to-end.")

if not st.session_state.get("default_scenarios_loaded"):
    if st.button("Load Default Scenario Catalog"):
        ctx.scenario_runner.create_default_scenarios()
        st.session_state.default_scenarios_loaded = True
        st.rerun()
else:
    catalog = ctx.scenario_runner.list_scenarios()
    scenario_ids = [s.scenario_id for s in catalog]
    chosen_id = st.selectbox("Choose a scenario to run", scenario_ids)
    chosen = next(s for s in catalog if s.scenario_id == chosen_id)
    st.caption(chosen.description)

    if st.button(f"▶️ Run '{chosen.name}'", type="primary"):
        try:
            with st.spinner(f"Running scenario '{chosen.name}'..."):
                result = ctx.scenario_runner.run_scenario(chosen)
            st.session_state.scenario_results.insert(0, result)
            st.session_state.last_snapshot = ctx.controller.snapshots[-1] if ctx.controller.snapshots else None
            st.success(f"Scenario complete — PDR {utils.fmt_pct(result.packet_delivery_ratio)}")
        except ScenarioRunnerError as exc:
            st.error(f"Scenario execution failed: {exc}")
        st.rerun()

    st.caption("Note: running a scenario resets and re-initializes the SimulationController "
               "(same lifecycle ScenarioRunner uses internally), so tick history on the "
               "Simulation Control / Analytics pages restarts from tick 0 afterward. "
               "Aggregated scenario metrics are captured below regardless.")

    results = st.session_state.get("scenario_results", [])
    if results:
        st.markdown("**Recent Scenario Results**")
        df = pd.DataFrame([{
            "Scenario": r.scenario_name,
            "Type": r.scenario_type.value,
            "Messages": r.total_messages,
            "Delivered": r.messages_delivered,
            "PDR": r.packet_delivery_ratio,
            "Avg Latency (ms)": r.average_latency,
            "Avg Hops": r.average_hop_count,
            "Success": r.success,
        } for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True)