# Understanding the Dual-Loop VANET Framework

This guide explains the newly implemented **Dual-Loop Predictive Framework** in simple, conceptual terms.
## The Core Idea: Smart Vehicles
Originally, this project trained an AI (a "Transformer") to look at a vehicle's past movements and predict its future trajectory. 

The **Dual-Loop Framework** takes that AI and uses it to actively control how the vehicle talks to other vehicles on the road (the "VANET" — Vehicular Ad-hoc Network). Instead of screaming "Here I am!" at 10 times a second blindly, vehicles now use AI to decide *when* to broadcast their position and *who* should act as a relay during emergencies.

---

## 1. Loop A: "Adaptive Awareness" (Don't spam the network)

**The Problem:** If every car broadcasts its GPS coordinates 10 times a second (10 Hz), the wireless channel gets congested. The network crashes, and critical safety alerts get lost in the noise.
**The Solution:** Most of the time, cars travel in a straight line or follow a predictable curve. If a car behaves predictably, its neighbors can *guess* where it is using their own AI.

**How Loop A works:**
1. The ego vehicle (our car) uses its AI to predict where it *should* be in the next few seconds.
2. It compares this prediction to its *actual* GPS position (measuring the "discrepancy" or "residual").
3. **If the difference is small (Low Entropy):** The car says, "I'm driving exactly as expected." It lowers its broadcast rate to **2 Hz**. This frees up 80% of the network bandwidth.
4. **If the difference is large (High Entropy):** The car says, "I did something unexpected (like dodging a pothole)!" It instantly increases its broadcast rate to **10 Hz** so everyone around it knows its true position.

#### What does the input and output actually look like?
* **Input:** A numpy array (matrix) representing the past 50 time steps of the car's state (Position X, Position Y, Speed, Acceleration, etc. = 23 features). Size: `[50, 23]`.
* **The prediction:** Another matrix size `[50, 23]` representing where the Transformer *thought* it would go.
* **The reality:** Another matrix size `[50, 23]` representing where the car *actually* went (measured by its GPS right now).
* **The Output:** A single simple decision: `flag = 0` (Low Entropy / 2 Hz) or `flag = 1` (High Entropy / 10 Hz).

---

## 2. Loop B: "Selective Forwarding" (Only the best driver relays the message)

**The Problem:** When an emergency happens (e.g., a hard brake), a warning message is sent. Every car that hears it tries to re-broadcast it to cars further back. If 10 cars re-broadcast at the exact same millisecond, the signals crash into each other (a "Broadcast Storm"), and nobody hears anything.
**The Solution:** Only the *most stable* car should relay the message.

**How Loop B works:**
1. The AI has been upgraded with a "Dual Head". While predicting the trajectory, it also classifies the vehicle's intent (e.g., *Maintain Lane*, *Turn*, *Exit*, *Brake*).
2. The AI outputs a **Stability Score** ($P_{stable}$), which is simply how confident it is that the car is just smoothly maintaining its lane.
3. When a group of cars receives an emergency alert, they all start a random wait timer before they relay the message.
4. **Loop B alters this timer:** 
    - A highly stable car (e.g., cruising happily in the middle lane) gets a very short timer (e.g., **1 millisecond**). 
    - An unstable car (e.g., currently swerving or braking) gets a long timer (e.g., **100 milliseconds**).
5. The stable car's timer hits zero first! It relays the message.
6. The unstable cars hear the stable car's relay before their own timers finish, so they cancel their own broadcasts. The storm is averted!

#### What does the input and output actually look like?
* **Input:** The Transformer's raw output probabilities (called "logits") for the 4 intent classes: `[MaintainLane, Turn, Exit, Brake]`. For example: `[10.0, -1.0, -1.0, -1.0]` (very confident it represents `MaintainLane`).
* **Intermediate Step:** This gets turned into a $P_{stable}$ probability. In this case, $P_{stable} \approx 0.99$.
* **The Output:** A strict millisecond timer value. Since $P_{stable}$ is very high (0.99), the output is `T_wait_ms = 1.0 ms`. If the car was braking hard, $P_{stable}$ might be $0.01$, resulting in `T_wait_ms = 100.0 ms`.

---

## 3. The New Files: `loop_a.py` and `loop_b.py`

Behind the scenes, we haven't just glued this logic straight into the giant training loops. We've built entirely independent Python modules inside a new folder called `src/loops/`.

1. **`src/loops/loop_a.py`**: Contains the `DiscrepancyMonitor` class. It's essentially a calculator that takes predicted arrays and actual arrays, calculates the mathematical difference (L2 residual), and spits out the recommended beacon frequency.
2. **`src/loops/loop_b.py`**: Contains the `StabilityScorer` and `MACBiasMapper` classes. It takes the intent outputs from the Transformer and does the mathematical mapping to turn them into safe millisecond timers.

**Why is this cool?** Because these files are completely disconnected from the heavy PyTorch training codebase. If you want to take this logic and deploy it onto a tiny Raspberry Pi inside a real car, or drag it into a network simulator like OMNeT++, you can import these two files directly without loading massive neural networks!

---

## How to Run the New Architecture

The new architecture introduces several command-line flags to control these loops. 

### 1. Basic Smoke Test
To verify the system is working (runs just 1 epoch on a tiny batch of data):

```bash
python main.py \
  --data_dir=resources/VANET_data/raw/ \
  --data_class=sind \
  --pattern=data_car_ \
  --data_normalization=standardization \
  --epochs=1 \
  --batch_size=32 \
  --pos_encoding=learnable \
  --num_intents=4 \
  --intent_weight=0.0 \
  --name=dual_loop_smoke
```

### 2. Full Unsupervised Training
To train the Transformer from scratch to act as the AI brain for these loops:

```bash
python main.py \
  --data_dir=resources/VANET_data/raw/ \
  --data_class=sind \
  --pattern=data_car_ \
  --data_normalization=standardization \
  --epochs=500 \
  --batch_size=256 \
  --pos_encoding=learnable \
  --num_intents=4 \
  --intent_weight=0.0 \
  --harden \
  --name=dual_loop_pretraining
```
*(Note: `--intent_weight=0.0` is used during unsupervised pre-training because we don't have intent labels yet, but the dual-head architecture is still built and prepared).*

### Understanding the Loop Parameters
You can fine-tune how the loops behave using these new flags (they have sensible defaults so you don't *need* to provide them):

* **Loop A Limits:**
  * `--entropy_threshold 0.5` : The discrepancy limit. Below this = 2 Hz, above this = 10 Hz.
  * `--beacon_hz_low 2.0` : The relaxed beacon rate.
  * `--beacon_hz_high 10.0` : The alert beacon rate.
* **Loop B Limits:**
  * `--relay_constant 0.1` : Defines the scaling curve.
  * `--relay_min_wait_ms 1.0` : The shortest wait time (for stable cars).
  * `--relay_max_wait_ms 100.0` : The longest wait time (for unstable cars).

You can pass these to `main.py` just like any other argument!

### 3. Running the Loops by Themselves!

You do not need to run the massive `main.py` to use Loop A or Loop B. You can use their logic in your own scripts using regular data. 

**Example: Testing Loop A in a simple Python script:**
```python
import numpy as np
from src.loops.loop_a import DiscrepancyMonitor

# 1. Create the monitor
monitor = DiscrepancyMonitor(epsilon=0.5)

# 2. Fake some data: predictions vs reality.
# Say the array is [50 time steps, 23 features]
prediction = np.zeros((50, 23)) 
reality    = np.zeros((50, 23)) + 1.0  # Reality was vastly different!

# 3. Ask Loop A what to do!
decision = monitor.check(prediction, reality)

print(f"Flag: {decision.flag}")                 # Outputs: 1 (High Entropy)
print(f"Broadcast rate: {decision.beacon_hz}")  # Outputs: 10.0 (Hz)
```

**Example: Testing Loop B in a simple Python script:**
```python
import torch
from src.loops.loop_b import StabilityScorer, MACBiasMapper

scorer = StabilityScorer()
mapper = MACBiasMapper(min_wait_ms=1.0, max_wait_ms=100.0)

# 1. The Transformer outputs intent logits: [Maintain, Turn, Exit, Brake]
# Let's pretend the car is braking hard:
intent_logits = torch.tensor([-5.0, -5.0, -5.0, 10.0]) 

# 2. Get the Stability Score
scored_state = scorer.score(intent_logits)
print(f"Probability of being stable: {scored_state.P_stable}") # Practically 0.0

# 3. Get the MAC timer!
timer_ms = mapper.map(scored_state.P_stable)
print(f"You must wait {timer_ms} ms before relaying!") # Outputs 100.0 ms
```

---

## 4. Model Prediction Workflow: Step-By-Step

If you sit inside the car and hit "Start", this is the exact flow of data through the files inside this project, from raw GPS coordinates to a final network decision:

### Step 1: Receiving Raw Sensor Data
**Where it happens:** Normally via real vehicle sensors, but here we read it from `resources/VANET_data/raw/data_car_*.csv`.
**What goes in:** A new row of data arrives every 100 milliseconds. This row contains 23 raw numbers representing the car's current state (X and Y coordinates, Speed, Acceleration, Heading, distance to lane edges, and relative distances/speeds of 3 neighboring cars).

### Step 2: Preparing the Matrix
**Where it happens:** `src/datasets/data.py` (specifically the `SINDData` class) and `src/utils/load_data.py`.
**What happens:** 
1. The AI cannot predict the future based on a single snapshot in time. It needs *history*. 
2. The code takes the newest row of 23 numbers and appends it to the last 49 rows to create a "Time Window" array of size `[50, 23]`. 
3. *Crucial fix:* It passes this array through the `Normalizer` to mathematically shrink the giant, crazy GPS numbers down to small, standardized values (roughly between -3 and +3) so the AI doesn't crash mathematically.

### Step 3: Into the "Brain" 
**Where it happens:** `src/transformer_model/encoder.py`.
**What happens:** The `[50, 23]` standardized matrix is passed into the `TSTransformerEncoder`.
**What comes out:** The AI's "Dual Head" kicks in, producing two separate outputs simultaneously from that identical input:
1. **The Future Trajectory:** A `[50, 23]` matrix that represents where the car thinks it will go next. 
2. **The Intent Probabilities:** A small array of just 4 numbers (logits) representing the probability that the car is about to *[Maintain, Turn, Exit, Brake]*.

### Step 4: The Network Decisions (The Loops)
**Where it happens:** `src/loops/loop_a.py` and `src/loops/loop_b.py`.
**What happens:** 
* **For Loop A:** The predicted trajectory `[50, 23]` is compared against where the car mathematically *actually* went at the end of the window. `DiscrepancyMonitor.check()` calculates the physical difference. 
   - **Output:** A command telling the car's Wi-Fi router to transmit a beacon at **2 Hz** or **10 Hz**.
* **For Loop B:** If an emergency alert arrives, the 4 intent logits are grabbed instantly and passed into `StabilityScorer` and `MACBiasMapper`.
  - **Output:** A strict network timer command (e.g., **1.5 ms** or **98.2 ms**) telling the car's router exactly how long to wait before relaying the emergency message over the air.
