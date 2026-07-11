# AI-Assisted QoS-Aware MANET Framework for Disaster Communication

> An intelligent disaster communication framework that integrates AI-based disaster message classification, QoS-aware routing, packet prioritization, and MANET simulation to enable resilient communication during disaster scenarios.

---

## 📌 Overview

Communication infrastructure often becomes unavailable during natural disasters, making coordination among rescue teams difficult. This project proposes an AI-Assisted QoS-Aware Mobile Ad Hoc Network (MANET) framework capable of intelligently processing disaster messages and delivering them through a dynamic wireless network.

The framework combines:

- AI-based disaster message classification
- QoS-aware message prioritization
- Packet fragmentation and transmission
- Dynamic MANET routing
- Disaster simulation
- Mobile rescue unit simulation
- Interactive dashboard for monitoring

---

## ✨ Features

### 🤖 AI Disaster Message Classification

- Fine-tuned DistilBERT classifier
- Supports 10 humanitarian disaster classes
- Confidence score generation
- Batch and single inference

Supported classes include:

- Injured or Dead People
- Requests or Urgent Needs
- Infrastructure and Utility Damage
- Missing or Found People
- Displaced People
- Rescue / Donation Efforts
- Caution and Advice
- Sympathy and Support
- Other Relevant Information
- Not Humanitarian

---

### 📡 QoS-Aware Communication

Automatically assigns

- Priority
- QoS Level
- Destination Department
- Target Node

based on AI classification.

---

### 📦 Intelligent Packet Generation

- Packet fragmentation
- Packet metadata generation
- Sequence numbering
- TTL assignment
- Checksum generation
- Delivery tracking

---

### 🚀 Priority Dispatcher

Implements

- Priority Queue scheduling
- Route assignment
- Packet forwarding
- Delivery confirmation

---

### 🌍 MANET Network Simulation

Supports

- Dynamic network topology
- Static infrastructure nodes
- Mobile rescue units
- Routing engine
- Link monitoring
- Network health estimation

---

### 🚑 Mobility Simulation

Simulated mobile units include

- Ambulances
- Fire Trucks
- Police Vehicles
- Rescue Teams
- Medical Teams
- Drones

---

### 🌪 Disaster Simulation

Supports multiple disaster scenarios

- Flood
- Earthquake
- Cyclone
- Fire
- Landslide

---

### 📈 Simulation Engine

Includes

- Event Scheduler
- Disaster Engine
- Simulation Clock
- Simulation Controller
- Scenario Runner

---

## 🏗 Project Architecture

```
                    Disaster Message
                            │
                            ▼
               DistilBERT Disaster Classifier
                            │
                            ▼
                    QoS Mapper Service
                            │
                            ▼
                 Disaster Message Object
                            │
                            ▼
                  Packet Generator Service
                            │
                            ▼
                     Network Packets
                            │
                            ▼
                    Priority Dispatcher
                            │
                            ▼
                     Routing Engine
                            │
                            ▼
                MANET Communication Layer
                            │
                            ▼
                  Destination Department
```

---

## 📂 Project Structure

```
AI_assisted_QOS_based_MANET/

├── communication/
│   ├── graph.py
│   ├── routing.py
│   ├── nodes.py
│   ├── edges.py
│   └── visualization.py
│
├── Disaster_Prediction/
│   ├── classifier.py
│   ├── app.py
│   └── models/
│
├── services/
│   ├── message.py
│   ├── qos_mapper.py
│   ├── packet.py
│   ├── packet_generator.py
│   └── dispatcher.py
│
├── simulation/
│   ├── disaster_engine.py
│   ├── event_scheduler.py
│   ├── network_state.py
│   ├── simulation_clock.py
│   └── Mobility/
│
├── integration/
│   ├── communication_pipeline.py
│   ├── simulation_controller.py
│   ├── scenario_runner.py
│   └── demo.py
│
├── dashboard/
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

## 🧠 Technology Stack

### Programming

- Python

### Artificial Intelligence

- PyTorch
- Transformers
- DistilBERT

### Data Processing

- NumPy
- Pandas
- Scikit-Learn

### Networking

- NetworkX

### Visualization

- Streamlit
- Matplotlib
- Plotly

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/<your_username>/AI-Assisted-QoS-Aware-MANET.git

cd AI-Assisted-QoS-Aware-MANET
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Run the classifier

```bash
python Disaster_Prediction/app.py
```

---

### Run communication pipeline

```bash
python -m integration.communication_pipeline
```

---

### Run simulation controller

```bash
python -m integration.simulation_controller
```

---

### Run scenario simulation

```bash
python -m integration.scenario_runner
```

---

### Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📊 Performance

### Disaster Classification

- 10 disaster categories
- DistilBERT-based inference
- Batch inference supported

### Communication

- QoS-aware packet routing
- Priority scheduling
- Packet fragmentation
- Route computation
- Delivery tracking

### Simulation

- Dynamic MANET topology
- Mobile node support
- Disaster lifecycle simulation
- Event-driven execution

---

## 📸 Screenshots

> Screenshots and dashboard visualizations will be added after dashboard implementation.

---

## 🔮 Future Work

- Congestion-aware routing
- Dynamic rerouting
- Reinforcement Learning-based routing
- FastAPI backend
- Real-time GIS map integration
- Multi-user dashboard
- Live communication monitoring

---

## 📖 Publications

This project is being developed as part of research on AI-Assisted QoS-Aware MANET communication for disaster management.

---

## 👨‍💻 Author

**Raj Vishwakarma**

Electronics and Communication Engineering

National Institute of Technology Andhra Pradesh

---

## ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.
