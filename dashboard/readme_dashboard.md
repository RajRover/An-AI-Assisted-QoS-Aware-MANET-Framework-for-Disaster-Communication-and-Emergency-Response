# MANET Framework — Streamlit Dashboard

Presentation-only Streamlit front end for the **AI-Assisted QoS-Aware MANET
Framework for Disaster Communication**. This dashboard does not implement any
routing, QoS, disaster, mobility, or classification logic — it only calls the
existing backend packages (`communication/`, `services/`, `simulation/`,
`integration/`, `Disaster_Prediction/`) and renders their output.

## 1. Placement

Copy (or leave) this `dashboard/` folder as a **sibling** of the existing
backend packages, e.g.:

```
AI_assisted_QOS_based_MANET/
├── communication/
├── services/
├── simulation/
├── integration/
├── Disaster_Prediction/
├── Datasets/
├── tests/
└── dashboard/            ← this folder
    ├── app.py
    ├── components.py
    ├── utils.py
    ├── styles.css
    └── pages/
        ├── 1_Home.py
        ├── 2_Message_Console.py
        ├── 3_Network_View.py
        ├── 4_Packet_Monitor.py
        ├── 5_Simulation.py
        ├── 6_Analytics.py
        └── 7_Disaster_Scenarios.py
```

## 2. Dependencies

The dashboard adds only two new dependencies on top of what the backend
already requires (`networkx`, `matplotlib`, `torch`, `transformers`, ...):

```
pip install streamlit pandas
```

## 3. Run

From the project root:

```
streamlit run dashboard/app.py
```

The first page load will build the full backend object graph (network
topology, `NetworkState`, `SimulationController`, mobility fleet, and load
the AI disaster classifier model) — this can take a few seconds, mostly for
model loading. The loaded classifier is cached process-wide
(`st.cache_resource`) so subsequent sessions start instantly; all other
simulation state is per-browser-session (`st.session_state`), so multiple
users/tabs each get an independent simulation.

## 4. Pages

| Page | Purpose |
|---|---|
| **Home** | System overview + Start/Pause/Resume/Stop/Reset lifecycle controls |
| **Message Console** | Submit a message through the live `CommunicationPipeline` (AI classification → QoS mapping → packetization → dispatch) |
| **Network View** | Live topology map (`communication.visualization.draw_qos_network`) with mobile-unit and packet-route overlays |
| **Packet Monitor** | Table of every packet generated this session, sourced from `Dispatcher` |
| **Simulation Control** | Direct `SimulationController` tick stepping and statistics |
| **Analytics** | Live charts: PDR, latency, packet loss, jitter, bandwidth, hop count, priority/QoS distribution |
| **Disaster Scenarios** | Trigger Flood / Earthquake / Fire / Cyclone / Landslide incidents via `DisasterEngine`, or run the backend's predefined `ScenarioRunner` catalog |

## 5. Notes on design choices

- All charts use Streamlit's built-in `st.line_chart` / `st.bar_chart`
  (backed by `pandas`) rather than an extra plotting dependency.
- The network map reuses the backend's own
  `communication.visualization.draw_qos_network()` for the base render;
  the dashboard only *overlays* mobile units, MANET links, and the
  highlighted route on top — no topology or QoS computation is duplicated.
- Auto-refresh is implemented with a lightweight `time.sleep` + `st.rerun()`
  toggle to avoid a hard dependency on `streamlit-autorefresh`.
- Packets are accumulated into `st.session_state.packet_log` as they are
  returned by `CommunicationResult.generated_packets`, since the `Dispatcher`
  only exposes aggregate counts/statistics publicly, not a full packet list.
- Running a predefined scenario (`ScenarioRunner.run_scenario`) resets and
  re-initializes the `SimulationController`, mirroring the exact lifecycle
  the backend's own `ScenarioRunner` uses internally — this is called out
  in the UI so it isn't mistaken for a bug.