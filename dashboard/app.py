import os
import sys

# 1. Dynamically download the model from Hugging Face if it's missing (Streamlit Cloud)
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Disaster_Prediction", "models", "multi_class_disaster_model"))

if not os.path.exists(MODEL_DIR):
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading model to {MODEL_DIR}...")
        snapshot_download(
            repo_id="rajvishwakarmaNIT/disaster-message-classifier",
            local_dir=MODEL_DIR
        )
    except Exception as e:
        print(f"Failed to auto-download model: {e}")

# 2. Continue with your normal bootstrap
import components
import utils

components.configure_page("Home", icon="🛰️")
ctx = utils.ensure_backend()
components.render_sidebar_status(ctx)

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
