# OMNeT++ / Veins / SUMO Integration Guide
# Dual-Loop Predictive VANET Framework

---

## 1. Overview

This document describes how to connect the **Python Transformer backend** to your existing
**OMNeT++ + Veins + SUMO** simulation so the AI model can influence live MAC-layer 
decisions during the simulation and so you can evaluate the results rigorously.

There are two operational modes:

| Mode | Description | When to use |
|---|---|---|
| **Static Mode** | Run simulation → export CSV → run Python → compare metrics | Quick validation, thesis results |
| **Live Mode** | Python server runs *alongside* OMNeT++ and controls it in real-time | Full dual-loop demonstration |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SUMO (Traffic Physics)                      │
│   most.highway.flows.xml / myMap.net.xml / mySumoConfig.sumocfg    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ TraCI protocol (vehicle positions)
┌──────────────────────────────▼──────────────────────────────────────┐
│                    OMNeT++ + Veins (Network Physics)                │
│   DataCollectorApp.cc  ←→  TrajSafetyMessage.msg                   │
│   omnetpp.ini          ←→  simulations/config.xml                  │
└──────────────┬──────────────────────────────────┬───────────────────┘
               │                                  │
        CSV data export                    ZeroMQ socket
        (Static Mode)                     (Live Mode)
               │                                  │
┌──────────────▼──────────────────────────────────▼───────────────────┐
│                  Python Transformer Backend                          │
│                                                                      │
│   main.py ──► src/clustering/run.py ──► src/reachability_analysis/  │
│                                                                      │
│   src/loops/loop_a.py  (beacon Hz control)                         │
│   src/loops/loop_b.py  (MAC backoff timer)                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Static Mode (Post-Hoc Evaluation)

This is the simplest path to results. The simulation runs completely, exports CSV files,
and the Python pipeline analyses them afterwards.

### Step 1 — Run the simulation normally

```bash
# From your OMNeT++ project root
cd /path/to/your/omnetpp_project
./simulations/run
```

This generates per-vehicle CSV files (e.g. `data_car_32_t18003.csv`) in the output
directory you configured in `DataCollectorApp.cc`.

### Step 2 — Copy/symlink the CSV files into the Python project

```bash
# Point the Python project at the new dataset folder
ln -s /path/to/omnetpp_output \
  /home/massi/myStuff/dev/VANET_Trajectory_Pred_With_Transformers/resources/VANET_data/raw/latest_run
```

### Step 3 — Run the full Python pipeline (4 commands)

```bash
cd /home/massi/myStuff/dev/VANET_Trajectory_Pred_With_Transformers

# Phase 1 — Train the Transformer (first time only, or retrain on new data)
python main.py \
  --data_dir=resources/VANET_data/raw/latest_run/ \
  --pattern=data_car_ \
  --config=best_config/configuration.json \
  --exclude_feats=49,50 \
  --name=static_eval_run

# Phase 2 — Cluster the embeddings and build the AnnoyModel memory
python src/clustering/run.py \
  --folder=experiments \
  --model_file=static_eval_run_XXXX   # replace with folder name from Step 1

# Phase 3 — Run reachability simulation
python src/reachability_analysis/simulation.py \
  --folder=experiments \
  --model_file=static_eval_run_XXXX

# Phase 3b — Evaluate inclusion accuracy (how often real car was inside safety polygon)
python src/reachability_analysis/inclusion_accuracy.py \
  --folder=experiments \
  --model_file=static_eval_run_XXXX
```

### Step 4 — Feed the model decisions back into a second simulation run

This is the critical step. You **cannot** measure network improvement from Python alone
because OMNeT++ is a closed physics engine — the radio collisions already happened.
You must re-run the simulation with the model's decisions applied.

**How it works:**

After running the Python pipeline on the exported CSVs, generate a decision lookup file
that maps every vehicle and every timestep to the Loop A/B output:

```bash
# The Python pipeline writes this automatically after fine-tuning
python src/deploy/export_decisions.py \
  --folder=experiments \
  --model_file=static_eval_run_XXXX \
  --output=decisions.json
```

The file looks like this:
```json
{
  "car_32": {
    "18003.1": {"flag": 0, "mac_wait_ms": 100.0},
    "18003.2": {"flag": 1, "mac_wait_ms": 1.0}
  },
  "car_211": {
    "18133.4": {"flag": 0, "mac_wait_ms": 100.0}
  }
}
```

**Run 2:** Re-run the **exact same SUMO scenario** (same seed, same route files:
`most.highway.flows.xml`, `mySumoConfig.sumocfg`) with `DataCollectorApp.cc`
modified to read `decisions.json` at startup instead of beaconing at a flat rate.

Because SUMO is **fully deterministic** with the same seed, vehicle positions in Run 2
are **bit-for-bit identical** to Run 1. The only variable that changes is how and when
vehicles transmit. Any difference in PDR, delay, or collision rate is caused **purely
by the model's decisions** — that is your thesis evidence.

```
Run 1 (Baseline):  flat 10Hz beaconing, standard CSMA/CA  →  PDR_baseline, Delay_baseline
                                    ↓
                         export CSVs → Python pipeline → decisions.json
                                    ↓
Run 2 (With Model): AI-controlled Hz + MAC backoff         →  PDR_model,    Delay_model

Thesis claim:  PDR_model > PDR_baseline  AND  Delay_model < Delay_baseline
```

Extract both sets of metrics from `results/General-#0.sca`:

```bash
opp_scavetool export -f "name =~ *packetLoss* OR name =~ *delay*" \
  results/General-#0.sca -o baseline_metrics.csv
```

---

## 4. Live Mode (Real-Time Co-Simulation)

In live mode, a Python gRPC/ZeroMQ server runs alongside OMNeT++. Every time a vehicle
in OMNeT++ is about to transmit a packet, it asks the Python server: 
*"What beacon rate and MAC backoff timer should I use?"*

### 4.1  Install the Python server dependency

```bash
cd /home/massi/myStuff/dev/VANET_Trajectory_Pred_With_Transformers
.venv/bin/pip install pyzmq
```

### 4.2  Create the Python inference server

Create `src/deploy/server.py` (new file), which:
1. Loads the trained Transformer model and AnnoyModel index at startup.
2. Listens on `tcp://localhost:5555` for incoming vehicle state windows.
3. Runs Loop A (`DiscrepancyMonitor`) and Loop B (`StabilityScorer` + `MACBiasMapper`).
4. Replies with `{"beacon_hz": 2.0, "mac_wait_ms": 1.0}`.

```python
# src/deploy/server.py  (skeleton)
import zmq
import json
import numpy as np
import torch

from src.loops.loop_a import DiscrepancyMonitor
from src.loops.loop_b import StabilityScorer, MACBiasMapper
# ... load model, annoy index ...

monitor = DiscrepancyMonitor(epsilon=0.5)
scorer  = StabilityScorer()
mapper  = MACBiasMapper(min_wait_ms=1.0, max_wait_ms=100.0)

ctx    = zmq.Context()
socket = ctx.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("[Python Server] Listening on port 5555 ...")
while True:
    msg = socket.recv_json()                        # receive from C++
    window    = np.array(msg["window"])             # (seq_len, 51)
    pred      = np.array(msg["prediction"])         # (seq_len, 51)
    intent    = torch.tensor(msg["intent_logits"])  # (4,)

    loop_a    = monitor.check(pred, window)
    stability = scorer.score(intent)
    wait_ms   = mapper.map(stability.P_stable)

    socket.send_json({
        "beacon_hz":  loop_a.beacon_hz,
        "mac_wait_ms": float(wait_ms),
        "flag":        loop_a.flag,
    })
```

Run it before launching OMNeT++:

```bash
cd /home/massi/myStuff/dev/VANET_Trajectory_Pred_With_Transformers
.venv/bin/python src/deploy/server.py
```

### 4.3  Modify `DataCollectorApp.cc` to call Python

In your simulation's `src/trajectories/DataCollectorApp.cc`, add a ZeroMQ client call
inside the packet-transmission handler. Below is the minimal patch pattern:

```cpp
// At top of DataCollectorApp.cc — add includes
#include <zmq.hpp>
#include <nlohmann/json.hpp>   // or any JSON lib

// In DataCollectorApp::initialize() — connect socket once
zmq::context_t zmq_ctx(1);
zmq::socket_t  zmq_sock(zmq_ctx, zmq::socket_type::req);
zmq_sock.connect("tcp://localhost:5555");

// In DataCollectorApp::handleSelfMsg() — BEFORE sendDown(wsm)
if (window_buffer.size() >= SEQ_LEN) {           // once you have enough history
    nlohmann::json req;
    req["window"]       = serialize_window();     // (seq_len, 51) as nested array
    req["prediction"]   = last_prediction;        // cached from last inference
    req["intent_logits"]= last_intent_logits;     // (4,) softmax output

    std::string payload = req.dump();
    zmq::message_t request(payload.size());
    memcpy(request.data(), payload.c_str(), payload.size());
    zmq_sock.send(request, zmq::send_flags::none);

    zmq::message_t reply;
    zmq_sock.recv(reply, zmq::recv_flags::none);
    auto response     = nlohmann::json::parse(reply.to_string_view());

    double beacon_hz  = response["beacon_hz"];
    double mac_wait   = response["mac_wait_ms"];

    // Apply Loop A — adjust beacon interval
    double new_interval_s = 1.0 / beacon_hz;
    scheduleAt(simTime() + new_interval_s, sendBeaconEvt);

    // Apply Loop B — bias MAC backoff (Veins-specific)
    mac->par("contentionWindow") = (int)(mac_wait / 0.015625); // slots
}
```

> [!NOTE]
> The exact Veins MAC API call depends on your version. For Veins 5.x the parameter
> is accessible via `FindModule<Mac1609_4*>::findSubModule(getParentModule())`.

---

## 5. Evaluation Pipeline: What to Measure

### 5.1 Network Metrics (from OMNeT++ results)

| Metric | Baseline | With Dual-Loop | Source |
|---|---|---|---|
| Packet Delivery Ratio (PDR) | Measured | Expected +15–25% | `results/raw` |
| Average Message Delay | Measured | Expected −30% | `results/raw` |
| Channel Busy Ratio (CBR) | Measured | Expected −20% | OMNeT++ vector output |
| Beacon Collision Rate | Measured | Expected −40% | MAC layer stats |

### 5.2 AI Model Metrics (from Python pipeline)

| Metric | Script | What it tells you |
|---|---|---|
| Reconstruction Loss (MSE) | `main.py` training output | How well the Transformer predicts motion |
| Silhouette Score | `src/clustering/run.py` | Quality of HDBSCAN behavioral clusters |
| Inclusion Accuracy (%) | `src/reachability_analysis/inclusion_accuracy.py` | How often real car was inside safety polygon |
| Intent Classification Accuracy | `main.py` fine-tuning output | How well the 4-class intent head performs |

### 5.3 Combined Evaluation (Thesis Evidence)

The strongest thesis claim is a **side-by-side comparison**:

```
Run 1: OMNeT++ simulation, NO Python model → record PDR, delay, collision count
Run 2: OMNeT++ simulation + Python Live Server → record same metrics
```

Then show:
- **PDR improved** from X% → Y% under broadcast storm conditions (32% PLR dataset)
- **Safety polygon accuracy** > 85% (reachability inclusion accuracy)
- **Beacon suppression worked**: Low-entropy cars reduced transmission to 2Hz
- **MAC biasing worked**: Stable cars relayed first, reducing redundant retransmissions

---

## 6. Quick Reference: File Mapping

| Python file | OMNeT++ counterpart | Role |
|---|---|---|
| `src/loops/loop_a.py` | `DataCollectorApp.cc` (beacon timer) | Controls BSM frequency |
| `src/loops/loop_b.py` | `DataCollectorApp.cc` (MAC send) | Controls relay backoff |
| `src/reachability_analysis/simulation.py` | N/A (offline only) | Draws safety polygons |
| `src/clustering/run.py` | N/A (offline only) | Builds Annoy memory index |
| `src/deploy/server.py` | ZeroMQ endpoint in `DataCollectorApp.cc` | Live inference bridge |
| `resources/VANET_data/raw/` | OMNeT++ CSV output directory | Shared data store |
