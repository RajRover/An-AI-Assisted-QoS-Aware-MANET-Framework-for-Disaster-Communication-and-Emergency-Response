"""
dashboard/app.py
------------------
Entry point for the AI-Assisted QoS-Aware MANET Framework Dashboard.

Run with:
    streamlit run dashboard/app.py

This file only bootstraps the shared backend context and hands off to the
"Home" page. All page content lives under dashboard/pages/, following
Streamlit's native multipage-app convention (auto-discovered sidebar nav).
"""

import components
import utils

components.configure_page("Home", icon="🛰️")
ctx = utils.ensure_backend()
components.render_sidebar_status(ctx)

# Hand off to the Home page. st.switch_page is available on modern Streamlit
# (>=1.31); fall back to inline guidance for older versions.
try:
    import streamlit as st
    st.switch_page("pages/1_Home.py")
except Exception:
    import streamlit as st
    components.render_header(
        "AI-Assisted QoS-Aware MANET Framework",
        "Disaster Communication Simulation & Monitoring Dashboard",
    )
    st.info("👈 Select **Home** from the sidebar to open the dashboard.")