import streamlit as st
import torch
import json
import os
import random
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Emergency Operations Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLES
# =========================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0rem;
    }
    .sub-header {
        color: #9aa0a6;
        margin-bottom: 1.2rem;
    }
    .panel-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 1.6rem;
        margin-bottom: 0.6rem;
    }
    .card {
        background-color: rgba(120,120,120,0.08);
        border: 1px solid rgba(120,120,120,0.2);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }
    .priority-critical { color:#ffffff; background:#d32f2f; padding:4px 12px; border-radius:20px; font-weight:700; }
    .priority-high { color:#ffffff; background:#f57c00; padding:4px 12px; border-radius:20px; font-weight:700; }
    .priority-medium { color:#000000; background:#fbc02d; padding:4px 12px; border-radius:20px; font-weight:700; }
    .priority-low { color:#ffffff; background:#43a047; padding:4px 12px; border-radius:20px; font-weight:700; }
    .priority-none { color:#ffffff; background:#757575; padding:4px 12px; border-radius:20px; font-weight:700; }

    .route-box {
        border: 2px solid #4f8bf9;
        border-radius: 10px;
        padding: 10px 16px;
        text-align: center;
        font-weight: 600;
        background: rgba(79,139,249,0.08);
        margin: 0 auto;
        width: 260px;
    }
    .route-arrow {
        text-align: center;
        font-size: 1.4rem;
        color: #4f8bf9;
        margin: 2px 0;
    }
    .summary-label { color:#9aa0a6; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.5px; }
    .summary-value { font-size:1.25rem; font-weight:700; margin-bottom:0.8rem; }
</style>
""", unsafe_allow_html=True)

PRIORITY_CLASS = {
    "Critical": "priority-critical",
    "High": "priority-high",
    "Medium": "priority-medium",
    "Low": "priority-low",
    "None": "priority-none",
}

# Rough estimated response time by priority, purely illustrative
RESPONSE_TIME_MIN = {
    "Critical": (5, 10),
    "High": (12, 20),
    "Medium": (25, 40),
    "Low": (45, 90),
    "None": (0, 0),
}

# =========================================================
# 1. LOAD MODEL, TOKENIZER, CLASS MAP
# =========================================================
@st.cache_resource
def load_model_components():
    model_path = "./models/multi_class_disaster_model"

    if not os.path.exists(model_path):
        st.error(f"Model directory not found at `{model_path}`. Unzip 'multi_class_disaster_model.zip' next to app.py.")
        st.stop()

    try:
        loaded_tokenizer = AutoTokenizer.from_pretrained(model_path)
        loaded_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        loaded_model.eval()
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.stop()

    class_map_path = "./models/class_map.json"
    if not os.path.exists(class_map_path):
        st.error(f"`class_map.json` not found next to app.py.")
        st.stop()

    try:
        with open(class_map_path, "r") as f:
            class_map = json.load(f)
    except Exception as e:
        st.error(f"Error loading class_map.json: {e}")
        st.stop()

    id_to_class = {v: k for k, v in class_map.items()}
    return loaded_model, loaded_tokenizer, class_map, id_to_class


loaded_model, loaded_tokenizer, class_map, id_to_class = load_model_components()

# =========================================================
# 2. CATEGORY INFO (decision knowledge base)
# =========================================================
CATEGORY_INFO = {
    "not_humanitarian": {
        "description": "This message is not related to an active humanitarian disaster.",
        "priority": "None",
        "department": "None",
        "qos": "None",
        "recommended_action": "No emergency action required.",
        "routing_strategy": "Ignore",
    },
    "other_relevant_information": {
        "description": "General information that may assist disaster situational awareness.",
        "priority": "Low",
        "department": "District Control Centre",
        "qos": "Low",
        "recommended_action": "Store and forward to Control Centre for monitoring.",
        "routing_strategy": "Normal Routing",
    },
    "sympathy_and_support": {
        "description": "Messages expressing sympathy, emotional support, or encouragement.",
        "priority": "Low",
        "department": "Archive",
        "qos": "Low",
        "recommended_action": "Archive message. No operational response required.",
        "routing_strategy": "Low Priority Queue",
    },
    "caution_and_advice": {
        "description": "Safety alerts, evacuation instructions, or precautionary advice.",
        "priority": "Medium",
        "department": "Public Information Cell",
        "qos": "Medium",
        "recommended_action": "Broadcast advisory to affected regions.",
        "routing_strategy": "Reliable Broadcast",
    },
    "rescue_volunteering_or_donation_effort": {
        "description": "Volunteer registrations or donation offers.",
        "priority": "Medium",
        "department": "Relief Coordination Centre",
        "qos": "Medium",
        "recommended_action": "Forward to NGO and Relief Coordination Team.",
        "routing_strategy": "Balanced Routing",
    },
    "displaced_people_and_evacuations": {
        "description": "People displaced or requiring evacuation.",
        "priority": "High",
        "department": "Evacuation & Relief Team",
        "qos": "High",
        "recommended_action": "Dispatch evacuation resources and assign nearest relief camp.",
        "routing_strategy": "Reliable Low-Latency Route",
    },
    "infrastructure_and_utility_damage": {
        "description": "Damage to roads, bridges, power lines, water supply or communication infrastructure.",
        "priority": "High",
        "department": "Utility Restoration Team",
        "qos": "High",
        "recommended_action": "Notify utility department and district control centre.",
        "routing_strategy": "Reliable Route",
    },
    "requests_or_urgent_needs": {
        "description": "Immediate requests for food, water, shelter, rescue or medical assistance.",
        "priority": "Critical",
        "department": "Emergency Response Team",
        "qos": "Highest",
        "recommended_action": "Dispatch nearest rescue team immediately.",
        "routing_strategy": "Lowest Latency Route",
    },
    "missing_or_found_people": {
        "description": "Reports regarding missing or located individuals.",
        "priority": "High",
        "department": "Police & Search Team",
        "qos": "High",
        "recommended_action": "Initiate search and verification process.",
        "routing_strategy": "Reliable Route",
    },
    "injured_or_dead_people": {
        "description": "Reports of injuries or fatalities.",
        "priority": "Critical",
        "department": "Medical Emergency Unit",
        "qos": "Highest",
        "recommended_action": "Dispatch ambulance and nearest medical rescue team immediately.",
        "routing_strategy": "Lowest Latency Route",
    },
}

# =========================================================
# 3. PREDICTION FUNCTION
# =========================================================
def predict_sentence(sentence: str):
    inputs = loaded_tokenizer(
        sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=128
    )
    device = torch.device("cpu")
    loaded_model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = loaded_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    predicted_class_label = id_to_class.get(predicted_class_id, "Unknown")

    return predicted_class_label, probabilities.squeeze().tolist()


# =========================================================
# 4. FAKE NETWORK TELEMETRY (placeholder until simulator is wired in)
# =========================================================
def get_network_snapshot(priority: str):
    """Randomized placeholder network stats.
    Replace this function body with a call to your teammate's simulator API
    once it's available -- keep the same return keys.
    """
    # Slightly stress the network more for higher priority events, just for demo realism
    stress = {"Critical": 0.6, "High": 0.4, "Medium": 0.2, "Low": 0.05, "None": 0.0}.get(priority, 0.1)

    latency_ms = round(20 + stress * 150 + random.uniform(-5, 15), 1)
    packet_loss = round(stress * 4 + random.uniform(0, 1.5), 2)
    bandwidth_mbps = round(max(2, 100 - stress * 60 + random.uniform(-5, 5)), 1)
    hospital_load = min(100, round(30 + stress * 60 + random.uniform(-5, 10)))
    congestion = "High" if stress > 0.45 else ("Moderate" if stress > 0.2 else "Low")

    if latency_ms < 60 and packet_loss < 2:
        health = "Healthy"
    elif latency_ms < 150 and packet_loss < 4:
        health = "Degraded"
    else:
        health = "Critical"

    return {
        "latency_ms": latency_ms,
        "packet_loss": packet_loss,
        "bandwidth_mbps": bandwidth_mbps,
        "hospital_load": hospital_load,
        "network_health": health,
        "congestion": congestion,
    }


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    st.caption("Network telemetry is currently simulated. Swap `get_network_snapshot()` for your teammate's live feed when it's ready.")
    st.divider()
    st.markdown("### 📚 Category Legend")
    for cat, info in CATEGORY_INFO.items():
        badge = PRIORITY_CLASS.get(info["priority"], "priority-none")
        st.markdown(
            f"<span class='{badge}'>{info['priority']}</span> &nbsp; {cat.replace('_',' ').title()}",
            unsafe_allow_html=True,
        )

# =========================================================
# HEADER
# =========================================================
st.markdown("<div class='main-header'>🚨 AI Emergency Operations Dashboard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Classify incoming disaster messages and get an instant AI-driven response plan.</div>",
    unsafe_allow_html=True,
)

user_input = st.text_area(
    "**Incoming message**",
    "The power lines are down and many homes are without electricity due to the storm.",
    height=130,
)

col_a, col_b = st.columns([1, 5])
with col_a:
    classify_clicked = st.button("🔍 Classify Message", use_container_width=True, type="primary")

# =========================================================
# MAIN LOGIC
# =========================================================
if classify_clicked:
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    else:
        with st.spinner("Analyzing message and routing decision..."):
            predicted_label, probs = predict_sentence(user_input)
            time.sleep(0.3)  # small UX pause

        info = CATEGORY_INFO.get(predicted_label, {
            "description": "No information available for this category.",
            "priority": "None",
            "department": "Unknown",
            "qos": "Unknown",
            "recommended_action": "N/A",
            "routing_strategy": "N/A",
        })

        net = get_network_snapshot(info["priority"])

        # ---------------------------------------------------
        # 🚨 AI EMERGENCY DECISION PANEL
        # ---------------------------------------------------
        st.markdown("<div class='panel-title'>🚨 AI Emergency Decision Panel</div>", unsafe_allow_html=True)

        badge_class = PRIORITY_CLASS.get(info["priority"], "priority-none")
        st.markdown(f"""
        <div class="card">
            <b>Predicted Category:</b> {predicted_label.replace('_', ' ').title()}<br>
            <b>Priority:</b> <span class="{badge_class}">{info['priority']}</span><br><br>
            <b>Description:</b> {info['description']}
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Responsible Department", info["department"])
        c2.metric("QoS Level", info["qos"])
        c3.metric("Routing Strategy", info["routing_strategy"])

        st.info(f"**Recommended Action:** {info['recommended_action']}")

        # ---------------------------------------------------
        # 📡 COMMUNICATION NETWORK PANEL
        # ---------------------------------------------------
        st.markdown("<div class='panel-title'>📡 Communication Network Panel</div>", unsafe_allow_html=True)
        st.caption("Placeholder telemetry — will be replaced by the live network simulator.")

        n1, n2, n3, n4, n5, n6 = st.columns(6)
        n1.metric("Latency", f"{net['latency_ms']} ms")
        n2.metric("Packet Loss", f"{net['packet_loss']} %")
        n3.metric("Bandwidth", f"{net['bandwidth_mbps']} Mbps")
        n4.metric("Hospital Load", f"{net['hospital_load']} %")
        n5.metric("Network Health", net["network_health"])
        n6.metric("Congestion", net["congestion"])

        # ---------------------------------------------------
        # 🛣 RECOMMENDED ROUTE
        # ---------------------------------------------------
        st.markdown("<div class='panel-title'>🛣️ Recommended Route</div>", unsafe_allow_html=True)

        dept = info["department"] if info["department"] != "None" else "Control Centre"
        route_steps = ["Village A", "Rescue Team A", "Control Centre"]
        if dept not in route_steps:
            route_steps.append(dept)
        route_steps.append("Hospital A")

        route_html = ""
        for i, step in enumerate(route_steps):
            route_html += f"<div class='route-box'>{step}</div>"
            if i < len(route_steps) - 1:
                route_html += "<div class='route-arrow'>↓</div>"
        st.markdown(route_html, unsafe_allow_html=True)

        # ---------------------------------------------------
        # 📊 CONFIDENCE GRAPH
        # ---------------------------------------------------
        st.markdown("<div class='panel-title'>📊 Model Confidence</div>", unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "Category": [id_to_class.get(i, f"Class {i}").replace("_", " ").title() for i in range(len(probs))],
            "Confidence": [p * 100 for p in probs],
        }).sort_values("Confidence", ascending=False).set_index("Category")

        st.bar_chart(prob_df, horizontal=True)

        with st.expander("See raw probabilities"):
            st.dataframe(
                prob_df.rename(columns={"Confidence": "Confidence (%)"}).round(2),
                use_container_width=True,
            )

        # ---------------------------------------------------
        # 🏥 AI DECISION SUMMARY
        # ---------------------------------------------------
        st.markdown("<div class='panel-title'>🏥 AI Decision Summary</div>", unsafe_allow_html=True)

        lo, hi = RESPONSE_TIME_MIN.get(info["priority"], (0, 0))
        eta = f"{lo}–{hi} minutes" if hi else "N/A"

        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown(f"<div class='summary-label'>Priority</div><div class='summary-value'><span class='{badge_class}'>{info['priority']}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='summary-label'>Department</div><div class='summary-value'>{info['department']}</div>", unsafe_allow_html=True)
        with s2:
            st.markdown(f"<div class='summary-label'>Estimated Response</div><div class='summary-value'>{eta}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='summary-label'>QoS</div><div class='summary-value'>{info['qos']}</div>", unsafe_allow_html=True)
        with s3:
            st.markdown(f"<div class='summary-label'>Recommended Routing</div><div class='summary-value'>{info['routing_strategy']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='summary-label'>Current Network</div><div class='summary-value'>{net['network_health']}</div>", unsafe_allow_html=True)

else:
    st.info("Enter a message above and click **Classify Message** to generate the emergency response plan.")

st.divider()
st.caption("Built with ❤️ using Streamlit and Hugging Face Transformers · AI Emergency Operations Dashboard")