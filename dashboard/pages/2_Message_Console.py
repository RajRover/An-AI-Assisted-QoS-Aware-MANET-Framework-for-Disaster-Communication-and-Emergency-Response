"""dashboard/pages/2_Message_Console.py — Submit messages through the live
CommunicationPipeline (AI classification → QoS mapping → packetization →
dispatch) and inspect the full result."""

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
from integration.communication_pipeline import CommunicationPipelineError

components.configure_page("Message Console", icon="✉️")
ctx = utils.ensure_backend()
components.render_sidebar_status(ctx)

components.render_header("Message Console", "Submit a disaster-response message through the live AI + QoS pipeline")

node_ids = utils.get_static_node_ids(ctx)

with st.form("message_console_form", clear_on_submit=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        source_node = st.selectbox("Source Node", node_ids, index=node_ids.index("V1") if "V1" in node_ids else 0)
    with c2:
        sender_id = st.text_input("Sender ID", value="OPS-001")
    with c3:
        sender_type_name = st.selectbox("Sender Type", utils.SENDER_TYPE_OPTIONS,
                                         index=utils.SENDER_TYPE_OPTIONS.index("CITIZEN"))

    message_text = st.text_area(
        "Message Text",
        placeholder="e.g. There are injured people trapped near the collapsed bridge, send an ambulance immediately.",
        height=110,
    )
    submitted = st.form_submit_button("🚀 Submit Message", type="primary", use_container_width=True)

if submitted:
    if not message_text or not message_text.strip():
        st.warning("Please enter a non-empty message before submitting.")
    else:
        try:
            with st.spinner("Running classification → QoS mapping → packetization → dispatch..."):
                result = ctx.pipeline.process_message(
                    text=message_text,
                    source_node=source_node,
                    sender_type=sender_type_name,
                    sender_id=sender_id,
                )

            # Feed shared session state consumed by other pages
            st.session_state.packet_log.extend(result.generated_packets)
            st.session_state.last_route = list(result.route)
            st.session_state.message_history.insert(0, {
                "Time": result.message.timestamp.strftime("%H:%M:%S"),
                "Text": result.message.text,
                "Class": result.predicted_class,
                "Confidence": result.confidence,
                "Priority": utils.enum_name(result.priority),
                "QoS": utils.enum_name(result.qos_level),
                "Destination": result.destination_node or "—",
                "Hops": result.hop_count,
                "Delivered": result.delivery_success,
            })

            st.success("Message processed successfully.")

            st.subheader("Classification & QoS Result")
            components.metric_row([
                ("Predicted Disaster Class", result.predicted_class),
                ("Confidence", utils.fmt_pct(result.confidence)),
                ("QoS Level", utils.enum_name(result.qos_level)),
                ("Priority", utils.enum_name(result.priority)),
            ])
            components.metric_row([
                ("Department", utils.enum_name(result.department)),
                ("Destination Node", result.destination_node or "—"),
                ("Hop Count", result.hop_count),
                ("Delivery Status",
                 "✅ Delivered" if result.delivery_success else "❌ Failed"),
            ])

            st.markdown("**Selected Route**")
            if result.route:
                st.code(" → ".join(result.route), language=None)
            else:
                st.warning(f"No route could be established. "
                            f"{result.dispatcher_result.error_message or ''}")

            with st.expander("Class Probability Distribution"):
                probs = result.message.classifier_probabilities
                if probs:
                    prob_df = pd.DataFrame(
                        sorted(probs.items(), key=lambda kv: kv[1], reverse=True),
                        columns=["Class", "Probability"],
                    )
                    st.bar_chart(prob_df.set_index("Class"))

            with st.expander(f"Generated Packets ({result.packet_count})"):
                st.dataframe(components.packets_dataframe(result.generated_packets),
                             use_container_width=True, hide_index=True)

        except CommunicationPipelineError as exc:
            st.error(f"Pipeline error: {exc}")
        except Exception as exc:  # defensive: never let the console crash the app
            st.error(f"Unexpected error while processing message: {exc}")

st.divider()
st.subheader("Recent Messages")
if st.session_state.message_history:
    hist_df = pd.DataFrame(st.session_state.message_history[:25])
    hist_df["Confidence"] = hist_df["Confidence"].apply(lambda v: utils.fmt_pct(v))
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
else:
    st.caption("No messages submitted yet in this session.")