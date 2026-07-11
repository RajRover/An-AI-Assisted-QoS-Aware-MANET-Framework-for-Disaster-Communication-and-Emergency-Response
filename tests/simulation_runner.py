#!/usr/bin/env python3
"""
simulation_runner.py

Orchestration script and main interactive CLI entry point for the
AI-Assisted QoS-Aware MANET Framework for Disaster Communication.

Calls ONLY functions that exist in the imported modules.
No routing algorithms, disaster logic, or graph generation implemented here.
"""

import logging
import os
import sys

# ── Path bootstrap: add project root so imports work from tests/ or anywhere ──
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import networkx as nx

# ── Communication modules ──────────────────────────────────────────────────
try:
    from communication.graph import build_network
    from communication.routing import (
        calculate_route,
        calculate_route_metrics,
        validate_route,
    )
    from communication.network_stats import generate_network_report
    from communication.visualization import draw_qos_network
except ImportError as e:
    print(f"[Error] Failed to import communication modules: {e}")
    sys.exit(1)

# ── Simulation modules ─────────────────────────────────────────────────────
try:
    from simulation.network_state import NetworkState
    from simulation.disaster_profiles import DisasterProfileManager, DisasterType
    from simulation.event_scheduler import EventScheduler
    from simulation.network_updater import NetworkUpdater
    from simulation.disaster_engine import DisasterEngine, DisasterStage
    from simulation.simulation_clock import SimulationClock, SimulationClockStatus
except ImportError as e:
    print(f"[Error] Failed to import simulation modules: {e}")
    sys.exit(1)

# ── Logging: WARNING level keeps the console clean ────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SimulationRunner")

# ──────────────────────────────────────────────────────────────────────────
# Disaster type menu
# ──────────────────────────────────────────────────────────────────────────
_DISASTER_TYPES = {
    "1": DisasterType.FLOOD,
    "2": DisasterType.CYCLONE,
    "3": DisasterType.EARTHQUAKE,
    "4": DisasterType.FIRE,
    "5": DisasterType.LANDSLIDE,
}

_KNOWN_ZONES = ["North Zone", "South Zone", "Central Zone"]


# ──────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────

def _qos_score(metrics: dict) -> float:
    """Heuristic QoS score: higher = better."""
    lat  = max(metrics.get("total_latency_ms", 1.0), 0.01)
    bw   = max(metrics.get("bottleneck_bandwidth_mbps", 1.0), 0.01)
    loss = max(metrics.get("total_packet_loss_percent", 0.0), 0.0)
    rel  = max(metrics.get("overall_reliability_percent", 1.0), 0.01)
    return round((rel * bw) / (lat * (1.0 + loss)), 4)


def _route_status(metrics: dict) -> str:
    rel  = metrics.get("overall_reliability_percent", 100.0)
    loss = metrics.get("total_packet_loss_percent", 0.0)
    if rel < 80.0 or loss > 10.0:
        return "CRITICAL"
    if rel < 92.0 or loss > 5.0:
        return "DEGRADED"
    return "HEALTHY"


def _safe_input(prompt: str, default: str = "") -> str:
    try:
        val = input(prompt).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


# ══════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER
# ══════════════════════════════════════════════════════════════════════════

class SimulationRunner:

    def __init__(self):
        self._graph          = None
        self._state          = None
        self._profile_mgr    = None
        self._scheduler      = None
        self._updater        = None
        self._disaster_eng   = None
        self._clock          = None
        # Plug-in registry for future modules
        self._extensions: dict = {}

    # ── Initialization ─────────────────────────────────────────────────

    def initialize_simulation(self) -> bool:
        """Build the graph and wire every subsystem. Returns True on success."""
        try:
            print("\n  Initializing subsystems…")

            # 1. Graph (communication layer)
            self._graph = build_network()
            print(f"  [OK] Graph built  ({self._graph.number_of_nodes()} nodes, "
                  f"{self._graph.number_of_edges()} edges)")

            # 2. NetworkState  — constructor: NetworkState(initial_graph)
            self._state = NetworkState(initial_graph=self._graph)
            print("  [OK] NetworkState created")

            # 3. DisasterProfileManager
            self._profile_mgr = DisasterProfileManager(populate_defaults=True)
            print("  [OK] Disaster profiles loaded")

            # 4. EventScheduler
            self._scheduler = EventScheduler()
            print("  [OK] Event scheduler ready")

            # 5. NetworkUpdater — constructor: NetworkUpdater(network_state)
            self._updater = NetworkUpdater(network_state=self._state)
            print("  [OK] Network updater ready")

            # 6. DisasterEngine — constructor: DisasterEngine(state, scheduler, updater, profile_mgr)
            self._disaster_eng = DisasterEngine(
                network_state=self._state,
                event_scheduler=self._scheduler,
                network_updater=self._updater,
                profile_manager=self._profile_mgr,
            )
            print("  [OK] Disaster engine ready")

            # 7. SimulationClock — constructor: SimulationClock(state, scheduler, engine, updater, …)
            self._clock = SimulationClock(
                network_state=self._state,
                event_scheduler=self._scheduler,
                disaster_engine=self._disaster_eng,
                network_updater=self._updater,
                tick_duration_ms=100.0,
                max_ticks=10000,
                real_time_mode=False,
                auto_stop=False,
            )
            print("  [OK] Simulation clock ready")

            # Compute initial telemetry baseline
            self._updater.recompute_global_telemetry()
            print("  [OK] Baseline telemetry computed")
            return True

        except Exception as exc:
            logger.error("Initialization error: %s", exc, exc_info=True)
            print(f"\n  [ERROR] Initialization failed: {exc}")
            return False

    # ── Main entry point ───────────────────────────────────────────────

    def run(self):
        print("\n" + "=" * 51)
        print("   AI-Assisted QoS-Aware MANET Framework")
        print("=" * 51)

        if not self.initialize_simulation():
            print("  Aborting: fix errors above and retry.")
            sys.exit(1)

        self._menu_loop()

    # ── Menu ───────────────────────────────────────────────────────────

    def display_main_menu(self):
        print("\n" + "=" * 51)
        print("   AI-Assisted QoS-Aware MANET Framework")
        print("=" * 51)
        print(" 1.  Display District Network Summary")
        print(" 2.  Display All Nodes")
        print(" 3.  Find Best Communication Route")
        print(" 4.  Show ALL Possible Communication Routes")
        print(" 5.  Compare Routes")
        print(" 6.  Start Disaster Simulation")
        print(" 7.  Advance One Tick")
        print(" 8.  Run Simulation Until Completion")
        print(" 9.  Display Network Statistics")
        print("10.  Display Active Disasters")
        print("11.  Display Pending Events")
        print("12.  Reset Simulation")
        print("13.  Exit")
        print("=" * 51)

    def _menu_loop(self):
        dispatch = {
            "1":  self.opt_network_summary,
            "2":  self.opt_display_nodes,
            "3":  self.opt_best_route,
            "4":  self.opt_all_possible_routes,
            "5":  self.opt_compare_routes,
            "6":  self.opt_start_disaster,
            "7":  self.opt_advance_tick,
            "8":  self.opt_run_until_completion,
            "9":  self.opt_display_stats,
            "10": self.opt_active_disasters,
            "11": self.opt_pending_events,
            "12": self.opt_reset_simulation,
            "13": self._opt_exit,
        }
        while True:
            try:
                self.display_main_menu()
                choice = _safe_input("\nSelect an option (1-13): ")
                handler = dispatch.get(choice)
                if handler is None:
                    print("[Warning] Selection out of bounds. Choose a valid menu entry (1-13).")
                else:
                    handler()
            except KeyboardInterrupt:
                print("\nGracefully terminating. Goodbye.")
                break
            except Exception as exc:
                logger.error("Menu error: %s", exc, exc_info=True)
                print(f"[Error] {exc}")

    # ══════════════════════════════════════════════════════════════════
    # OPTION 1 — District Network Summary
    # ══════════════════════════════════════════════════════════════════

    def opt_network_summary(self):
        print("\n--- District Network Summary ---")
        g = self._state.graph                          # live nx.Graph from NetworkState.graph property
        report = generate_network_report(g)            # generate_network_report(graph) -> dict

        # Average latency / bandwidth from edge attributes
        lats, bws = [], []
        for _, _, d in g.edges(data=True):
            lats.append(d.get("latency", 0.0))
            bws.append(d.get("bandwidth", 0.0))
        avg_lat = sum(lats) / len(lats) if lats else 0.0
        avg_bw  = sum(bws)  / len(bws)  if bws  else 0.0

        print(f"Total Nodes:          {report['total_nodes']}")
        print(f"Total Links:          {report['total_links']}")
        print(f"Connected Components: {report['connected_components']}")
        print(f"Average Degree:       {report['average_degree']:.2f}")
        print(f"Diameter:             {report['diameter']}")
        print(f"Average Latency:      {avg_lat:.2f} ms")
        print(f"Average Bandwidth:    {avg_bw:.2f} Mbps")

    # ══════════════════════════════════════════════════════════════════
    # OPTION 2 — Display All Nodes
    # ══════════════════════════════════════════════════════════════════

    def opt_display_nodes(self):
        print("\n--- All Network Nodes ---")
        g = self._state.graph
        print(f"{'Node ID':<10} {'Name':<25} {'Type':<20} {'Status':<12} "
              f"{'Capacity':<10} {'Load':<8} {'Zone'}")
        print("-" * 95)
        for node_id, data in g.nodes(data=True):
            print(
                f"{str(node_id):<10} "
                f"{str(data.get('name',       '-')):<25} "
                f"{str(data.get('type',       '-')):<20} "
                f"{str(data.get('status',     '-')):<12} "
                f"{str(data.get('capacity',   '-')):<10} "
                f"{str(data.get('current_load', data.get('load', '-'))):<8} "
                f"{str(data.get('zone', '-'))}"
            )

    # ══════════════════════════════════════════════════════════════════
    # OPTION 3 — Find Best Communication Route
    # ══════════════════════════════════════════════════════════════════

    def opt_best_route(self):
        print("\n--- Find Best Communication Route ---")
        self._print_node_ids()
        src = _safe_input("Enter Source Node ID:      ")
        dst = _safe_input("Enter Destination Node ID: ")
        if not src or not dst:
            print("[Error] Both source and destination are required.")
            return

        # calculate_route(graph, source, target) -> {"success", "path", "metrics"} or {"success": False, "error"}
        result = calculate_route(self._state.graph, src, dst)
        if not result.get("success"):
            print(f"[Error] {result.get('error', 'No route found.')}")
            return

        path    = result["path"]
        metrics = result["metrics"]
        qos     = _qos_score(metrics)

        print("\nOptimal Communication Path Details:")
        print(f"  Shortest Path:        {' -> '.join(path)}")
        print(f"  Hop Count:            {metrics['hop_count']}")
        print(f"  Latency:              {metrics['total_latency_ms']:.2f} ms")
        print(f"  Bandwidth:            {metrics['bottleneck_bandwidth_mbps']:.2f} Mbps")
        print(f"  Packet Loss:          {metrics['total_packet_loss_percent']:.2f} %")
        print(f"  Reliability:          {metrics['overall_reliability_percent']:.2f} %")
        print(f"  Estimated QoS Score:  {qos}")

        self._trigger_route_visualization(path)

    # ══════════════════════════════════════════════════════════════════
    # OPTION 4 — Show ALL Possible Communication Routes
    # ══════════════════════════════════════════════════════════════════

    def opt_all_possible_routes(self):
        print("\n--- All Possible Communication Routes ---")
        self._print_node_ids()
        src = _safe_input("Enter Source Node ID:      ")
        dst = _safe_input("Enter Destination Node ID: ")
        if not src or not dst:
            print("[Error] Both nodes required.")
            return

        g = self._state.graph
        if src not in g or dst not in g:
            print("[Error] One or both node IDs not found.")
            return

        try:
            all_paths = list(nx.all_simple_paths(g, source=src, target=dst, cutoff=8))
        except nx.NodeNotFound as exc:
            print(f"[Error] {exc}")
            return

        if not all_paths:
            print(f"No reachable paths between '{src}' and '{dst}'.")
            return

        # Compute metrics for every path using calculate_route_metrics(graph, path) -> dict
        routes = []
        for idx, path in enumerate(all_paths, 1):
            m   = calculate_route_metrics(g, path)
            qos = _qos_score(m)
            routes.append({"number": idx, "path": path, "metrics": m, "qos": qos})

        # Sort variants
        by_latency     = sorted(routes, key=lambda r: r["metrics"]["total_latency_ms"])
        by_bandwidth   = sorted(routes, key=lambda r: -r["metrics"]["bottleneck_bandwidth_mbps"])
        by_reliability = sorted(routes, key=lambda r: -r["metrics"]["overall_reliability_percent"])
        by_loss        = sorted(routes, key=lambda r: r["metrics"]["total_packet_loss_percent"])
        recommended    = max(routes, key=lambda r: r["qos"])

        print(f"\nFound {len(routes)} path(s):\n")
        for r in routes:
            m   = r["metrics"]
            tag = "  * RECOMMENDED" if r["number"] == recommended["number"] else ""
            print(f"Route #{r['number']}{tag}")
            print(f"  Path:         {' -> '.join(r['path'])}")
            print(f"  Hop Count:    {m['hop_count']}")
            print(f"  Latency:      {m['total_latency_ms']:.2f} ms")
            print(f"  Bandwidth:    {m['bottleneck_bandwidth_mbps']:.2f} Mbps")
            print(f"  Packet Loss:  {m['total_packet_loss_percent']:.2f} %")
            print(f"  Reliability:  {m['overall_reliability_percent']:.2f} %")
            print()

        print("── Sorted by Lowest Latency ──")
        for r in by_latency:
            print(f"  Route #{r['number']:>2}  {r['metrics']['total_latency_ms']:.2f} ms")

        print("\n── Sorted by Highest Bandwidth ──")
        for r in by_bandwidth:
            print(f"  Route #{r['number']:>2}  {r['metrics']['bottleneck_bandwidth_mbps']:.2f} Mbps")

        print("\n── Sorted by Highest Reliability ──")
        for r in by_reliability:
            print(f"  Route #{r['number']:>2}  {r['metrics']['overall_reliability_percent']:.2f} %")

        print("\n── Sorted by Lowest Packet Loss ──")
        for r in by_loss:
            print(f"  Route #{r['number']:>2}  {r['metrics']['total_packet_loss_percent']:.2f} %")

        self._trigger_route_visualization(recommended["path"])

    # ══════════════════════════════════════════════════════════════════
    # OPTION 5 — Compare Routes
    # ══════════════════════════════════════════════════════════════════

    def opt_compare_routes(self):
        print("\n--- Compare Routes ---")
        self._print_node_ids()
        src = _safe_input("Enter Source Node ID:      ")
        dst = _safe_input("Enter Destination Node ID: ")
        if not src or not dst:
            print("[Error] Both nodes required.")
            return

        g = self._state.graph
        if src not in g or dst not in g:
            print("[Error] One or both node IDs not found.")
            return

        try:
            all_paths = list(nx.all_simple_paths(g, source=src, target=dst, cutoff=8))
        except nx.NodeNotFound as exc:
            print(f"[Error] {exc}")
            return

        if not all_paths:
            print(f"No routes found between '{src}' and '{dst}'.")
            return

        rows = []
        for idx, path in enumerate(all_paths, 1):
            m   = calculate_route_metrics(g, path)
            qos = _qos_score(m)
            rows.append({
                "label":       f"Route {idx}",
                "hops":        m["hop_count"],
                "latency":     m["total_latency_ms"],
                "bandwidth":   m["bottleneck_bandwidth_mbps"],
                "loss":        m["total_packet_loss_percent"],
                "reliability": m["overall_reliability_percent"],
                "qos":         qos,
                "status":      _route_status(m),
            })

        # Table
        print("\n" + "-" * 100)
        print(f"{'Route':<10} {'Hops':>5} {'Latency(ms)':>12} {'BW(Mbps)':>10} "
              f"{'Loss(%)':>9} {'Reliab.(%)':>11} {'QoS':>8} {'Status'}")
        print("-" * 100)
        for r in rows:
            print(
                f"{r['label']:<10} {r['hops']:>5} "
                f"{r['latency']:>12.2f} {r['bandwidth']:>10.2f} "
                f"{r['loss']:>9.2f} {r['reliability']:>11.2f} "
                f"{r['qos']:>8.4f} {r['status']}"
            )
        print("-" * 100)

        fastest   = min(rows, key=lambda r: r["latency"])
        reliable  = max(rows, key=lambda r: r["reliability"])
        emergency = max(rows, key=lambda r: r["qos"])

        print(f"  [FASTEST]  {fastest['label']}  ({fastest['latency']:.2f} ms)")
        print(f"  [RELIABLE] {reliable['label']}  ({reliable['reliability']:.2f} %)")
        print(f"  [EMERGENCY]{emergency['label']}  (QoS {emergency['qos']:.4f})")

    # ==================================================================
    # OPTION 6 - Start Disaster Simulation
    # ==================================================================

    def opt_start_disaster(self):
        print("\n--- Start Disaster Simulation ---")
        print("  Disaster Types:")
        print("    1. Flood")
        print("    2. Cyclone")
        print("    3. Earthquake")
        print("    4. Fire")
        print("    5. Landslide")

        choice = _safe_input("\n  Select type [1-5]: ", "1")
        dtype  = _DISASTER_TYPES.get(choice)
        if dtype is None:
            print("[Error] Invalid choice. Please enter 1–5.")
            return

        try:
            severity = int(_safe_input("  Severity [1-5]: ", "3"))
            if not 1 <= severity <= 5:
                raise ValueError
        except ValueError:
            print("[Error] Severity must be an integer 1–5.")
            return

        try:
            duration = int(_safe_input("  Duration (ticks): ", "50"))
            if duration <= 0:
                raise ValueError
        except ValueError:
            print("[Error] Duration must be a positive integer.")
            return

        print(f"  Known zones: {', '.join(_KNOWN_ZONES)}")
        zone = _safe_input("  Affected Zone: ", _KNOWN_ZONES[0])

        disaster_id = f"DISASTER_{dtype.name}_{self._clock.current_tick}"
        start_tick  = self._clock.current_tick + 1

        try:
            # create_disaster(disaster_id, disaster_type, severity_level, affected_zones, start_tick, duration_ticks)
            instance = self._disaster_eng.create_disaster(
                disaster_id=disaster_id,
                disaster_type=dtype,
                severity_level=severity,
                affected_zones=[zone],
                start_tick=start_tick,
                duration_ticks=duration,
            )
            print(f"[OK] Disaster '{disaster_id}' registered.")
            print(f"  Type      : {dtype.name}")
            print(f"  Severity  : {severity}/5")
            print(f"  Duration  : {duration} ticks")
            print(f"  Zone      : {zone}")
            print(f"  Starts at : Tick {instance.start_tick}")
            print(f"  Ends at   : Tick {instance.end_tick}")
        except Exception as exc:
            print(f"[Error] Could not create disaster: {exc}")
            logger.error("create_disaster failed: %s", exc, exc_info=True)

    # ==================================================================
    # OPTION 7 - Advance One Tick
    # ==================================================================

    def opt_advance_tick(self):
        print("\nAdvancing simulation by exactly 1 tick...")
        # SimulationClock.step() -> bool
        continued = self._clock.step()
        self._display_tick_summary()
        if not continued:
            print("[Info] Simulation reached a terminal state.")

    # ==================================================================
    # OPTION 8 - Run Simulation Until Completion
    # ==================================================================

    def opt_run_until_completion(self):
        print("\nRunning simulation until all disasters resolve...\n")
        header = (f"{'Tick':>6}  {'Stage':<12} {'Lat(ms)':>9} {'BW(M)':>7} "
                  f"{'Loss':>6} {'OffTwr':>7} {'HospLd':>8} {'Events':>7}")
        print(header)
        print("-" * len(header))

        limit = 10000
        for _ in range(limit):
            continued = self._clock.step()
            tick    = self._clock.current_tick
            metrics = self._state.global_metrics        # NetworkState.global_metrics property
            active  = self._disaster_eng.get_active_disasters()   # -> List[DisasterInstance]
            pending = self._scheduler.get_pending_events()         # -> List[Event]
            stage   = active[0].current_stage.name if active else "NONE"

            off_twr = sum(
                1 for _, d in self._state.graph.nodes(data=True)
                if d.get("type") == "Tower" and str(d.get("status","")).upper() == "OFFLINE"
            )
            h_load, h_cnt = 0, 0
            for _, d in self._state.graph.nodes(data=True):
                if d.get("type") == "Hospital":
                    h_cnt  += 1
                    h_load += d.get("hospital_utilization", 0)
            avg_h = h_load / h_cnt if h_cnt else 0

            print(
                f"{tick:>6}  {stage:<12} "
                f"{metrics.get('average_latency',0.0):>9.2f} "
                f"{metrics.get('average_bandwidth',0.0):>7.2f} "
                f"{metrics.get('average_packet_loss',0.0):>6.2f} "
                f"{off_twr:>7} {avg_h:>8.1f} "
                f"{self._clock.metrics.total_events_executed:>7}"
            )

            if not continued:
                print("\n[Info] Simulation clock reached terminal state.")
                break
            if tick > 1 and not active and not pending:
                print("\n[Info] All disasters resolved and event queue empty.")
                self._clock.stop()
                break

        self._trigger_disaster_visualization()

    # ==================================================================
    # OPTION 9 - Display Network Statistics
    # ==================================================================

    def opt_display_stats(self):
        print("\n--- Complete Network Statistics ---")
        g      = self._state.graph
        report = generate_network_report(g)          # generate_network_report(graph) -> dict
        stats  = self._state.get_statistics()        # NetworkState.get_statistics() -> dict

        print("\n  -- Topology --")
        for key, val in report.items():
            print(f"  {key.replace('_',' ').title():<35} : {val}")

        print("\n  -- Network Metrics --")
        for key, val in stats.get("network_metrics", {}).items():
            fmt = f"{val:.4f}" if isinstance(val, float) else str(val)
            print(f"  {key.replace('_',' ').title():<35} : {fmt}")

        print("\n  -- Simulation Metrics --")
        for key, val in stats.get("simulation_metrics", {}).items():
            print(f"  {key.replace('_',' ').title():<35} : {val}")

    # ==================================================================
    # OPTION 10 - Display Active Disasters
    # ==================================================================

    def opt_active_disasters(self):
        print("\n--- Active Disasters ---")
        # get_active_disasters() -> List[DisasterInstance]
        active = self._disaster_eng.get_active_disasters()
        if not active:
            print("No active disasters at this time.")
            return

        for inst in active:
            remaining = max(0, inst.end_tick - self._clock.current_tick)
            print(f"\n  Disaster Name     : {inst.disaster_id}")
            print(f"  Severity          : {inst.severity_level} / 5")
            print(f"  Current Stage     : {inst.current_stage.name}")
            print(f"  Remaining Duration: {remaining} ticks")
            print(f"  Affected Nodes    : {len(inst.affected_nodes)}  {sorted(inst.affected_nodes)}")
            print(f"  Affected Links    : {len(inst.affected_links)}")
            for u, v in sorted(inst.affected_links):
                print(f"                      {u} <-> {v}")

    # ==================================================================
    # OPTION 11 - Display Pending Events
    # ==================================================================

    def opt_pending_events(self):
        print("\n--- Event Scheduler Priority Queue ---")
        # get_pending_events() -> List[Event]
        events = self._scheduler.get_pending_events()
        if not events:
            print("Event queue is currently empty.")
            return

        sorted_events = sorted(events, key=lambda e: (e.scheduled_tick, e.priority.value))
        print(f"\n  {'Tick':>6}  {'Priority':<10} Description")
        print("  " + "-" * 65)
        for evt in sorted_events:
            print(f"  {evt.scheduled_tick:>6}  {evt.priority.name:<10} {evt.event_name}")

    # ==================================================================
    # OPTION 12 - Reset Simulation
    # ==================================================================

    def opt_reset_simulation(self):
        confirm = _safe_input("Reset to initial state? (Y/N): ", "N")
        if confirm.upper() != "Y":
            print("Reset cancelled.")
            return

        print("\nResetting simulation...")
        # Stop clock if running
        try:
            if self._clock.status == SimulationClockStatus.RUNNING:
                self._clock.stop()
            self._clock.reset()                   # SimulationClock.reset()
        except Exception:
            pass

        self._state.reset()                        # NetworkState.reset()
        self._scheduler.clear_events()             # EventScheduler.clear_events()
        self.initialize_simulation()
        print("  [OK] Simulation reset to initial state.\n")

    # ==================================================================
    # OPTION 13 - Exit
    # ==================================================================

    def _opt_exit(self):
        print("\nExiting Simulation Framework. Goodbye.\n")
        sys.exit(0)

    # -- Helpers --------------------------------------------------------

    def _print_node_ids(self):
        ids = sorted(self._state.graph.nodes())
        print(f"  Available nodes: {', '.join(ids)}\n")

    def _display_tick_summary(self):
        """Print a one-liner tick summary after advancing the clock."""
        tick    = self._clock.current_tick
        metrics = self._state.global_metrics
        active  = self._disaster_eng.get_active_disasters()
        stage   = active[0].current_stage.name if active else "NONE"

        off_nodes  = sum(1 for _, d in self._state.graph.nodes(data=True)
                         if str(d.get("status","")).upper() == "OFFLINE")
        off_towers = sum(1 for _, d in self._state.graph.nodes(data=True)
                         if d.get("type") == "Tower"
                         and str(d.get("status","")).upper() == "OFFLINE")
        h_load, h_cnt = 0, 0
        for _, d in self._state.graph.nodes(data=True):
            if d.get("type") == "Hospital":
                h_cnt  += 1
                h_load += d.get("hospital_utilization", 0)
        avg_h = h_load / h_cnt if h_cnt else 0

        print(f"  -- Tick {tick} ------------------------------------------")
        print(f"  Events Executed : {self._clock.metrics.total_events_executed}")
        print(f"  Disaster Stage  : {stage}")
        print(f"  Avg Latency     : {metrics.get('average_latency',  0.0):.2f} ms")
        print(f"  Avg Bandwidth   : {metrics.get('average_bandwidth', 0.0):.2f} Mbps")
        print(f"  Avg Packet Loss : {metrics.get('average_packet_loss', 0.0):.4f}")
        print(f"  Offline Nodes   : {off_nodes}")
        print(f"  Offline Towers  : {off_towers}")
        print(f"  Hospital Load   : {avg_h:.1f}")

    def _trigger_route_visualization(self, path: list):
        """Ask to visualize a route using draw_qos_network with path overlay."""
        ans = _safe_input("\nWould you like to visualize the route? (Y/N): ", "N")
        if ans.upper() != "Y":
            return
        try:
            import matplotlib.pyplot as plt
            g   = self._state.graph
            pos = {n: (g.nodes[n]["x"], g.nodes[n]["y"]) for n in g.nodes()}
            fig, ax = plt.subplots(figsize=(13, 11))

            # draw_qos_network(graph, ax, show) draws the full network
            draw_qos_network(g, ax=ax, show=False)

            # Overlay path edges
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(g, pos, edgelist=path_edges, ax=ax,
                                   width=5, edge_color="#e74c3c", alpha=0.9)
            # Source / destination / intermediate highlights
            nx.draw_networkx_nodes(g, pos, nodelist=[path[0]], ax=ax,
                                   node_size=1200, node_color="#27ae60",
                                   edgecolors="black", linewidths=3)
            nx.draw_networkx_nodes(g, pos, nodelist=[path[-1]], ax=ax,
                                   node_size=1200, node_color="#c0392b",
                                   edgecolors="black", linewidths=3)
            if len(path) > 2:
                nx.draw_networkx_nodes(g, pos, nodelist=path[1:-1], ax=ax,
                                       node_size=1100, node_color="#f39c12",
                                       edgecolors="black", linewidths=2.5)
            ax.set_title(f"Route: {' -> '.join(path)}", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            print(f"[Error] Visualization failed: {exc}")
            logger.error("Route visualization error: %s", exc, exc_info=True)

    def _trigger_disaster_visualization(self):
        """Ask to visualize the damaged network after simulation."""
        ans = _safe_input("\nVisualize damaged network? (Y/N): ", "N")
        if ans.upper() != "Y":
            return
        try:
            import matplotlib.pyplot as plt
            g   = self._state.graph
            pos = {n: (g.nodes[n]["x"], g.nodes[n]["y"]) for n in g.nodes()}
            fig, ax = plt.subplots(figsize=(13, 11))

            draw_qos_network(g, ax=ax, show=False)

            # Highlight offline towers
            offline_towers = [n for n, d in g.nodes(data=True)
                              if d.get("type") == "Tower"
                              and str(d.get("status","")).upper() == "OFFLINE"]
            if offline_towers:
                nx.draw_networkx_nodes(g, pos, nodelist=offline_towers, ax=ax,
                                       node_size=1300, node_color="#e74c3c",
                                       node_shape="X", edgecolors="black", linewidths=3)

            # Highlight hospitals
            hospitals = [n for n, d in g.nodes(data=True) if d.get("type") == "Hospital"]
            if hospitals:
                nx.draw_networkx_nodes(g, pos, nodelist=hospitals, ax=ax,
                                       node_size=1100, node_color="#e74c3c",
                                       edgecolors="#c0392b", linewidths=3)

            # Highlight relief camps
            camps = [n for n, d in g.nodes(data=True) if d.get("type") == "ReliefCamp"]
            if camps:
                nx.draw_networkx_nodes(g, pos, nodelist=camps, ax=ax,
                                       node_size=1100, node_color="#9b59b6",
                                       edgecolors="#8e44ad", linewidths=3)

            # Highlight affected villages
            villages = [n for n, d in g.nodes(data=True) if d.get("type") == "Village"]
            if villages:
                nx.draw_networkx_nodes(g, pos, nodelist=villages, ax=ax,
                                       node_size=1000, node_color="#2ecc71",
                                       edgecolors="#27ae60", linewidths=2.5)

            # Highlight damaged links
            damaged = [(u, v) for u, v, d in g.edges(data=True)
                       if d.get("status", "Healthy") not in ("Healthy", "OPERATIONAL")]
            if damaged:
                nx.draw_networkx_edges(g, pos, edgelist=damaged, ax=ax,
                                       width=4, edge_color="#e74c3c",
                                       style="dashed", alpha=0.8)

            ax.set_title("Damaged Network Visualization", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            print(f"[Error] Visualization failed: {exc}")
            logger.error("Disaster visualization error: %s", exc, exc_info=True)

    # ── Extension plug-in hooks for future modules ─────────────────────

    def register_extension(self, name: str, module) -> None:
        """Register a future module (MANET Mobility, Traffic Generator, etc.)."""
        self._extensions[name] = module
        logger.info("Extension '%s' registered.", name)

    def get_extension(self, name: str):
        return self._extensions.get(name)


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        runner = SimulationRunner()
        runner.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting.\n")
        sys.exit(0)