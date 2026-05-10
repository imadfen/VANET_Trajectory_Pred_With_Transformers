# Understanding the VANET Framework — Full Pipeline

This guide explains the complete architecture in simple, conceptual terms — from raw GPS data to live network decisions.

The framework has three phases that build on each other:

> **Phase 1** → Train a Transformer AI to understand vehicle motion  
> **Phase 2** → Use that AI to discover natural driving behavior patterns (clusters)  
> **Phase 3** → Use those clusters in real time to make smarter network decisions

---

## Phase 1 — Trajectory Encoding & Dual-Loop Deployment

### The Core Idea

Originally, the Transformer was trained to look at a vehicle's past movements and predict its future trajectory. Phase 1 is exactly that — but with two additions that make it useful for the VANET network:

1. The Transformer is trained through **masked autoregression** (intentionally hiding parts of the trajectory, then asking it to reconstruct them). This simulates packet loss — the AI learns to reconstruct a neighbor's path even when BSMs are missing.
2. After training, the model is deployed with a **Dual-Loop** that uses its predictions to actively control network behavior.

### What the Transformer Produces

Every 100ms the car feeds a window of the last 50 time steps into the encoder. That window is a matrix of shape `[50, 23]` — 50 moments in time, each with 23 numbers (position, speed, heading, lane info, 3 neighbor relative states, network stats).

The encoder's **Dual Head** produces two outputs simultaneously:
1. **Future trajectory** `[50, 23]` — where the Transformer thinks the car will go next
2. **Intent logits** `[4]` — confidence scores for `[MaintainLane, Turn, Exit, Brake]`

**Where it lives:** `src/transformer_model/encoder.py` + `src/transformer_model/model.py`

---

### Loop A — Adaptive Awareness (Don't spam the network)

**The Problem:** If every car broadcasts at 10 Hz blindly, the wireless channel gets congested and critical safety alerts get lost.

**The Solution:** Use the Transformer's prediction to decide *when* broadcasting is actually necessary.

**How it works:**
1. The car compares the Transformer's predicted trajectory against its *actual* GPS position — measuring the "residual" (how wrong the prediction was).
2. **Small residual (Low Entropy):** The car is driving exactly as expected → drop to **2 Hz**. Neighbors can predict its position themselves. Frees up 80% of bandwidth.
3. **Large residual (High Entropy):** The car did something unexpected (pothole dodge, emergency brake) → jump to **10 Hz**. Everyone needs to know its real position immediately.

**Input / Output:**
- **In:** predicted `[50, 23]` vs actual `[50, 23]`
- **Out:** `flag = 0` (2 Hz) or `flag = 1` (10 Hz) + `beacon_hz` value

**Where it lives:** `src/loops/loop_a.py` → `DiscrepancyMonitor`

**Example:**
```python
from src.loops.loop_a import DiscrepancyMonitor
import numpy as np

monitor = DiscrepancyMonitor(epsilon=0.5)
prediction = np.zeros((50, 23))
reality    = np.zeros((50, 23)) + 1.0  # reality was very different

decision = monitor.check(prediction, reality)
print(decision.flag)       # 1 (High Entropy)
print(decision.beacon_hz)  # 10.0 Hz
```

---

### Loop B — Selective Forwarding (Only the best driver relays)

**The Problem:** When an emergency alert arrives, every car tries to re-broadcast it simultaneously → signals collide (Broadcast Storm) → nobody hears it.

**The Solution:** Use the Transformer's intent output to rank cars by stability. The most stable car relays first; the others cancel.

**How it works:**
1. The Transformer's 4 intent logits are converted into a **Stability Score** ($P_{stable}$) — just the softmax probability of the `MaintainLane` class.
2. When an emergency arrives, every car starts a wait timer before relaying.
3. **Loop B biases that timer:**
   - Stable car (P_stable ≈ 1.0) → wait **1 ms** → relays first ✓
   - Unstable car (P_stable ≈ 0.0) → wait **100 ms** → hears the relay → cancels ✗
4. Storm is prevented without any extra protocol.

**Input / Output:**
- **In:** 4 intent logits e.g. `[10.0, -1.0, -1.0, -1.0]`
- **Intermediate:** $P_{stable} \approx 0.99$
- **Out:** `T_wait_ms = 1.0 ms` (stable) or `T_wait_ms = 100.0 ms` (unstable)

**Where it lives:** `src/loops/loop_b.py` → `StabilityScorer` + `MACBiasMapper`

**Example:**
```python
import torch
from src.loops.loop_b import StabilityScorer, MACBiasMapper

scorer = StabilityScorer()
mapper = MACBiasMapper(min_wait_ms=1.0, max_wait_ms=100.0)

intent_logits = torch.tensor([-5.0, -5.0, -5.0, 10.0])  # braking hard
result = scorer.score(intent_logits)
print(result.P_stable)              # practically 0.0
print(mapper.map(result.P_stable))  # 100.0 ms
```

---

### How to Run Phase 1

**Smoke test (1 epoch):**
```bash
python main.py \
  --data_dir=resources/VANET_data/raw/ \
  --data_class=sind \
  --pattern=data_car_ \
  --data_normalization=standardization \
  --epochs=1 --batch_size=32 \
  --pos_encoding=learnable \
  --num_intents=4 --intent_weight=0.0 \
  --name=dual_loop_smoke
```

**Full unsupervised training:**
```bash
python main.py \
  --data_dir=resources/VANET_data/raw/ \
  --data_class=sind \
  --pattern=data_car_ \
  --data_normalization=standardization \
  --epochs=500 --batch_size=256 \
  --pos_encoding=learnable \
  --num_intents=4 --intent_weight=0.0 \
  --harden \
  --name=dual_loop_pretraining
```

> `--intent_weight=0.0` during pre-training because we have no intent labels yet. The dual-head architecture is built and ready — the intent head will be trained in a supervised fine-tuning step later using cluster-derived labels.

**Loop tuning flags:**

| Flag | Default | Meaning |
|------|---------|---------|
| `--entropy_threshold` | `0.5` | Residual cutoff for 2Hz vs 10Hz |
| `--beacon_hz_low` | `2.0` | Beacon rate when stable |
| `--beacon_hz_high` | `10.0` | Beacon rate when unstable |
| `--relay_constant` | `0.1` | Scaling factor for Loop B timer |
| `--relay_min_wait_ms` | `1.0` | Fastest possible relay (most stable car) |
| `--relay_max_wait_ms` | `100.0` | Slowest possible relay (most unstable car) |

---

## Phase 2 — Behavior Clustering ("Building the Memory")

Phase 1 produces a trained model that predicts trajectories and classifies intent. But it doesn't yet *know* what kind of driver it is looking at. It just generates numbers. 

**Phase 2 gives the AI a memory of driving behaviors — without any human labels.**

Think of it like watching 10,000 hours of driving footage and naturally grouping what you see: *"that's a lane-changer", "that's a steady cruiser", "that's a panic-braker"*. You created mental buckets just by observing patterns. Phase 2 does the same thing, automatically.

### What happens

1. The trained encoder from Phase 1 runs over the **entire dataset once** in eval mode (`--eval_only`) — no learning, just extracting.
2. For every 50-step driving window, it produces a compact **embedding** (~128 numbers) that captures the "flavor" of that window's driving behavior.
3. All embeddings are fed into **HDBSCAN** — a clustering algorithm that finds natural groups without being told how many groups there are.
4. Result: a saved **Behavior Cluster Index** — e.g., Cluster 0 = stable cruise, Cluster 3 = sharp turn, Cluster 7 = erratic stop-start.
5. An **AnnoyModel** (fast Nearest Neighbor index) is built from the cluster centroids and saved to disk so Phase 3 can use it in real time.

### Where it lives

- `src/clustering/Clusters.py` — HDBSCAN logic
- `src/clustering/run.py` — orchestration: load model → extract embeddings → cluster → save index
- `src/clustering/NearestNeighbor.py` — builds and saves the AnnoyModel

---

## Phase 3 — Real-Time Reachability & Loop Feedback ("Acting on the Memory")

Phase 2 gave the AI a memory of behavior types. Phase 3 uses that memory in real time to make Loop A and Loop B even smarter by adding a **neighbor awareness** layer.

Imagine you see a car in your rear-view mirror behaving aggressively. You don't just know where it is — you know roughly where it *will* be in the next 3 seconds based on how aggressive drivers like that tend to move. That predicted bounding zone is the **Reachable Set**.

### What happens

1. A car receives a BSM from a neighbor vehicle.
2. It encodes the last 50 time steps of that neighbor's data → 128-number embedding.
3. A fast **Nearest Neighbor lookup** (`AnnoyModel`) instantly matches the embedding to the closest saved behavior cluster.
4. The system retrieves the **statistical distribution of trajectories** in that cluster.
5. This distribution is used to compute the **Reachable Set** — the zone the neighbor is statistically likely to occupy within 3 seconds.
6. The Reachable Set feeds back into both loops:
   - **Loop A:** if a neighbor's Reachable Set overlaps your predicted path → raise your beacon rate to 10 Hz even if your own residual is low.
   - **Loop B:** the neighbor's cluster acts as an "Interaction Factor" — a stable car with an aggressive neighbor nearby should relay emergency messages *faster*, because the situation is riskier than it looks from just your own state.

### Where it lives

- `src/clustering/NearestNeighbor.py` — ANN lookup (`AnnoyModel`)
- `src/reachability_analysis/operations.py` — zonotope set operations
- `src/reachability_analysis/zonotope.py` — bounding region data structure
- `src/reachability_analysis/simulation.py` — kinematic propagation
- `src/reachability_analysis/reachability.py` — high-level runner

---

## The Full Pipeline

```
════════════════════════════════════════════════════════════════
 OFFLINE — done once before deployment
════════════════════════════════════════════════════════════════

  Raw vehicle CSV data  (resources/VANET_data/raw/)
            │
            ▼
  PHASE 1: Train Transformer — unsupervised masking (packet loss simulation)
           Dual-Head encoder: trajectory head + intent head
            │  src/transformer_model/encoder.py + model.py
            ▼
  Save  model_best.pth
            │
            ▼
  PHASE 2: Run encoder over full dataset (eval_only)
           Extract embeddings → HDBSCAN clustering
           Build AnnoyModel nearest-neighbor index
            │  src/clustering/run.py + Clusters.py + NearestNeighbor.py
            ▼
  Save  cluster index (.pkl) + AnnoyModel (.ann)

════════════════════════════════════════════════════════════════
 ONLINE — runs inside the car every 100 ms
════════════════════════════════════════════════════════════════

  New BSM arrives from neighbor vehicle
            │
            ▼
  PHASE 3: Encode neighbor BSMs → ANN lookup → Closest behavior cluster
           → Compute Reachable Set (3-second statistical zone)
            │  src/clustering/NearestNeighbor.py
            │  src/reachability_analysis/
            │
            ├──► Neighbor's Reachable Set overlaps ego path?
            │         YES → raise Loop A entropy threshold
            │
            └──► Neighbor in aggressive cluster?
                      YES → increase Loop B relay urgency
            │
            ▼
  LOOP A  (src/loops/loop_a.py)
    ego predicted trajectory  vs  actual GPS
    → Low Entropy  →  beacon at  2 Hz  (predictable driving)
    → High Entropy →  beacon at 10 Hz  (unexpected maneuver)
            │
  LOOP B  (src/loops/loop_b.py)
    ego intent logits → P_stable
    → Stable car    →  wait   1 ms  →  relays first  ✓
    → Unstable car  →  wait 100 ms  →  hears relay, cancels  ✗
```

**The Transformer is the brain. The clusters are the memory. The Reachable Set is the judgment.**  
Together they turn raw AI predictions into safe, bandwidth-efficient, network-aware decisions — without any extra protocol overhead.

---

## 4. How to See It In Action (The Execution Pipeline)

If you want to run the pipeline sequentially from start to finish to see the algorithms dynamically calculate safety polygons:

### Step 1: Train the "Brain" (Phase 1)
Pass your tuned hyperparameters and raw VANET data into `main.py` to train the Transformer.
```bash
python main.py \
  --data_dir=resources/VANET_data/raw/ \
  --pattern=data_car_ \
  --config=config_after_finetuning/configuration.json \
  --name=dual_loop_run
```
* **Result:** It finishes validation and outputs the golden memory file containing embeddings: `experiments/dual_loop_run_XXX/output_data.pt`.

### Step 2: Build the "Memory" & Labels (Phase 2)
Hook into the memory file generated above to build the cluster logic.
```bash
python src/clustering/run.py \
  --folder=experiments \
  --model_file=dual_loop_run_XXX 
```
* **Result:** Runs classical HDBSCAN, prints the Silhouette Score, drops fast Nearest-Neighbor index trees (`.pkl` files) into an `experiments/dual_loop_run_XXX/clusters/` directory, and **generates intent labels** out of those clusters.

### Step 2.5: The Fine-Tuning Loop (Train the Intent Head)
*(This is the crucial step to wake up the dual-loop logic!)* 
Now that Phase 2 has algorithmically generated the "Intent Labels" (Maintain, Turn, Exit, Brake) out of the raw data, you merge those labels back into your dataset and run `main.py` ONE more time.
* This time, you pass `--load_model=experiments/dual_loop_run_XXX/model_best.pth` and ensure `"intent_weight": 1.0` is set in your configuration file.
* **Result:** The Transformer's Intent Head officially learns what the clusters mean, and can now spit out the `intent_logits` needed by Loop B!

### Step 3: See the Polygons Live (Phase 3)
Run the physical reachability logic using the trees we just grouped.
```bash
python src/reachability_analysis/simulation.py \
  --folder=experiments \
  --model_file=dual_loop_run_XXX 
```
* **Result:** A Matplotlib window pops up drawing the 2D Zonotopes (safety bounding polygons) calculating physical clearance based on a vehicle max speed of 50m/s. Saves outcomes to `reachable_sets.pkl`.

### Step 4: Prove It Worked
Grade the system by seeing how often cars actually stayed inside those bounds.
```bash
python src/reachability_analysis/inclusion_accuracy.py \
  --folder=experiments \
  --model_file=dual_loop_run_XXX
```
* **Result:** Computes exact percentage of times the ground-truth car remained inside the predictive bounds!

---

## 5. OMNeT++ / Veins Integration (The Simulation Goal)

While the Python backend calculates everything, showing physical network improvement requires a Network+Mobility co-simulator like **Veins** (OMNeT++ glued to SUMO).

The ultimate architecture looks like a **Fast Live Negotiation**:
1. You run a Python backend server that loads the Transformer and `src/loops/`.
2. You run an OMNeT++ simulation full of cars slamming their brakes and fighting for radio frequency.
3. Every MAC transmission inside `C++` sends the car's 23-feature historic trajectory bouncing string over a **ZeroMQ/TCP socket** to the Python script in milliseconds.
4. Python runs Phase 1 & 2 & 3, computes Loop A and Loop B formulas, and responds over the pipe: `{"beacon_hz": 2.0, "mac_wait_ms": 100.0}`.
5. OMNeT++ alters the 802.11p Contention Window and Physical TX layers with those variables.
6. **The Result:** The Packet Delivery Ratio (PDR) graph stays near 98% because aggressive, braking cars get clear airwaves, while predictable cars go silently into the background. Your Python code physically controls the simulated radio wave.
