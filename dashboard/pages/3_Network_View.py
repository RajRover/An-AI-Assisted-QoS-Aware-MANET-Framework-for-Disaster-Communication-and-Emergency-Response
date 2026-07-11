"""dashboard/pages/3_Network_View.py — Live NetworkX topology visualization
with mobile-unit overlay and packet-route highlighting."""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DASHBOARD_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

import streamlit as st

import components
import utils

components.configure_page("Network View", icon="🗺️")
ctx = utils.ensure_backend()
components.render_sidebar_status(ctx)

components.render_header("Network View", "Villages, towers, hospitals, relief camps, control center & mobile units")

col_a, col_b, col_c = st.columns(3)
with col_a:
    show_mobile = st.checkbox("Show mobile units", value=True)
with col_b:
    show_manet_links = st.checkbox("Show MANET connectivity", value=True)
with col_c:
    show_route = st.checkbox("Highlight last packet route", value=True)

mobile_snapshot = ctx.mobility_manager.get_snapshot() if show_mobile else None
manet_graph = ctx.mobility_manager.get_manet_graph_copy() if show_manet_links else None
highlight_route = st.session_state.get("last_route") if show_route else None

fig = components.render_network_figure(
    ctx.network_state.graph,
    mobile_nodes=mobile_snapshot,
    manet_graph=manet_graph,
    highlight_route=highlight_route,
)
st.pyplot(fig, use_container_width=True)

if highlight_route:
    st.caption(f"Highlighted route: {' → '.join(highlight_route)}")

st.divider()
components.auto_refresh_control("network_view")

st.divider()
st.subheader("Network Statistics")
report = utils.get_network_report(ctx)
components.metric_row([
    ("Total Nodes", report["total_nodes"]),
    ("Total Links", report["total_links"]),
    ("Connected", "Yes" if report["is_connected"] else "No"),
    ("Connected Components", report["connected_components"]),
])
components.metric_row([
    ("Avg. Degree", utils.fmt_num(report["average_degree"])),
    ("Graph Density", utils.fmt_num(report["graph_density"], 4)),
    ("Network Efficiency", utils.fmt_pct(report["network_efficiency"])),
    ("Avg. Clustering Coeff.", utils.fmt_num(report["average_clustering_coefficient"], 4)),
])
st.caption(f"Diameter: {report['diameter']}  ·  Average Path Length: {report['average_path_length']}")

st.divider()
st.subheader("Node Legend")
legend_cols = st.columns(4)
node_types = list(utils.NODE_TYPE_ICONS.items())
for i, (ntype, icon) in enumerate(node_types):
    with legend_cols[i % 4]:
        st.markdown(f"{icon} **{ntype}**")