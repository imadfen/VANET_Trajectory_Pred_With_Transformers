# Simulation and Implementation Plan: Dual-Loop Predictive Framework

This plan outlines the necessary modifications to the existing Transformer architecture to support the **Dual-Loop System** for Adaptive Awareness (Loop A) and Selective Forwarding (Loop B).


## 1. Loop A: Adaptive Awareness (Ego-Vehicle Suppression)

**Objective**: Reduce channel congestion by only broadcasting when the driver's behavior deviates from the Transformer's stable prediction.

### Required Changes:

- **Discrepancy Monitoring Module**:
  - Implement a real-time comparison between the **Transformer's Predicted Future ($\hat{T}$)** and the **Actual Sensor Path ($T$)**.
  - Define a threshold $\epsilon$.
  - **Logic**:
    - If $||\hat{T} - T || < \epsilon$: Set state to **Low Entropy**.Returns **Flag=0** and **Residual Error**.(Reduce Beacon frequency (2Hz))
    - If $||\hat{T} - T || \geq \epsilon$: Set state to **High Entropy**. Returns **Flag=1** and **Residual Error**.(Trigger Alert Mode (10Hz))
- **Hallucination Engine**:
  - Build a module for receiving neighbors to generate "Hallucinated Trajectories" using their local Transformers to fill the gaps during the ego-vehicle's 2Hz suppression periods.

## 2. Loop B: Selective Forwarding (Relay Meritocracy)

**Objective**: Use AI confidence to bias the 802.11p MAC layer backoff timers, ensuring the most stable vehicle rebroadcasts emergency alerts first.

### Required Changes:

- **Stability Score Calculation ($S_{relay}$)**:
  - Update `src/transformer_model/model.py` to output the **Softmax Probability ($P_{stable}$)** of a maneuver (instead of just X, Y coordinates).
  - Implement the weighting formula: $S_{relay} = (P_{stable} \times \omega_{dist}) + \text{Interaction Factor}$.
- **Predictive MAC Biasing**:
  - Create a mapping function that converts $P_{stable}$ into a **Transmission Delay ($T_{wait}$)**.
  - **Backoff Logic**:
    - $T_{wait} = \frac{Constant}{P_{stable}}$
    - Stable Vehicles ($P_{stable} \approx 1.0$) $\rightarrow$ $1-5ms$ delay.
    - Unstable Vehicles ($P_{stable} \approx 0.2$) $\rightarrow$ $50-100ms$ delay.
                       |

## 3. Structural Model Modifications

To support the "Mental Map" and "Multi-Agent" requirements, the following architectural shifts are needed:

### Multi-Agent Context

- **Input Transformation**: The model must move from single-vehicle history to a joint tensor containing the ego-vehicle + $K$ nearest neighbors.
- **Attention Map Extraction**: Modify the `encoder.py` to export the **Self-Attention Maps**. These will be used for the "Interaction Factor" to identify if a vehicle's stability is threatened by a braking car ahead.

### Intent Classification Head

- **Dual-Output Head**: Add a classification layer to the Transformer that predicts **Discrete Intents** (Maintain Lane, Turn, Exit, Brake) alongside the **Coordinate Regressor** (X, Y points).


## 4. The Execution Architecture (The 3 Phases)

The Dual-Loop logic sits on top of a three-phase software system that translates raw neural network math into physical and behavioral meaning. 

### Phase 1: The Brain (Behavioral Modeling)
- **Objective:** Train the physical Transformer architecture.
- **Execution:** Offline pre-training on historical VANET data using the unsupervised reconstruction head (ignoring the intent head initially).
- **Output:** Behavioral `embeddings` (dense multi-dimensional representations of trajectory physics) saved to an `output_data.pt` artifact.

### Phase 2: The Memory (Unsupervised Clustering)
- **Objective:** Form natural behavioral groupings to artificially generate "Intent Labels" without expensive human annotation.
- **Execution:** Feed Phase 1 `embeddings` into the **HDBSCAN** algorithm to naturally separate standard cruising from aggressive lane-changing or braking.
- **Output:** Fast Nearest-Neighbor **AnnoyModel** index trees (`.pkl` files) which link embeddings to discrete Intent intents. *(These synthesized labels are then fed back into Phase 1 to train the Intent Classification Head!)*

### Phase 3: The Judgment (Real-Time Reachability)
- **Objective:** Draw physical collision safety polygons in live 2D space.
- **Execution:** Live network BSMs/CAMs are matched instantly against the Phase 2 memory bank (`get_cluster`), generating physical bounds based on maximum vehicle kinematics (e.g., 50 m/s).
- **Output:** Mathematical Reachable Sets (**Zonotopes**) mapped around the car to calculate physical intersections.


## 5. The Predictive Pipeline: Multi-Agent Operational Flow

Follow a single emergency event involving **Vehicle 42** through the dual-loop architectural flow.

### Step 1: The "Joint Context" Phase (Data Preprocessing)

- **What we have**: A cluster of $1$ Ego-vehicle (V42) and its $K$ nearest neighbors.
- **What we do**:
  1. **Multi-Agent Windowing**: We package the historical sequences for all $K+1$ vehicles into a shared tensor.
  2. **Network Statistics Injection**: We append V2X features like `PacketLossRate` and `AvgMsgDelay` to the motion data.
- **The Input**: Tensors of shape `[Batch, K+1, Seq_Len, 23_Features]`.
- **The Output**: A synchronized "Social Snapshot" of the intersection.

### Step 2: The "Interaction Mapping" Phase (Encoder)

- **What we have**: The multi-agent social snapshot.
- **What we do**: The model runs **Multi-Head Self-Attention** not just across time, but across _vehicles_.
- **The Result**: An **Attention Map** showing how much V42 is being influenced by its neighbors (e.g., "V42 is highly attentive to the car ahead slowing down").
- **The Goal**: Extract a **Latent Vector** that captures "Social Intent" (The flavor of the maneuver within the group).

### Step 3: The "Judgment" Phase (Reachability & Safety Bounds)

- **What we have**: V42's generated `embedding` from the Transformer Encoder.
- **What we do**: V42 queries its local `AnnoyModel` (Memory) to map the embedding to an intent, then mathematically constructs 2D **Reachable Sets** (Zonotopes) using its physical velocity and heading constraints.
- **The Result**: A series of continuous collision-boundary polygons spanning the road ahead of V42.
- **The Goal**: Instantly determine if V42 intersects with any neighbor's reachable boundary, triggering emergency braking protocols physically before manipulating the network.

### Step 4: Loop A - The "Predicted vs. Actual" Check (Awareness)

- **What we have**: The Transformer's stable-state prediction for V42.
- **What we do**: V42 compares its _real_ 100ms movement against the Transformer's prediction.
- **The Result**: A binary **Trigger Signal** (Control Flag) and a **Residual Error Scalar**.
  - **Entropy Low (Flag=0)**: Prediction matches reality within threshold $\epsilon$.
  - **Entropy High (Flag=1)**: V42 starts an unexpected turn or maneuver.
- **The Execution**: This Trigger Signal is passed to the OBU's (On-Board Unit) Network Interface to manipulate the BSM beaconing frequency (2Hz vs 10Hz).
- **The Goal**: Maintain background awareness while clearing the 5.9GHz channel.

### Step 5: Loop B - The "Meritocratic Selection" (Relay)

- **What we have**: An emergency "Braking Alert" received by neighbors.
- **What we do**: Each neighbor feeds the scenario into their local Transformer.
- **The Result**: The model outputs a **Stability Probability ($P_{stable}$)**.
  - **Stable Relay**: $P_{stable} = 0.98$ (Predicted to stay in lane).
  - **Unstable Relay**: $P_{stable} = 0.30$ (Predicted to exit the road).
- **The Goal**: Calculate local merit without extra network traffic.

### Step 6: The "Timed Shout" Phase (MAC Layer Biasing)

- **What we have**: The $P_{stable}$ score from Step 5.
- **What we do**: The neighbor's networking hardware calculates a **Transmission Delay**: $T_{wait} = Constant / P_{stable}$.
- **The Result**:
  - **Stable Vehicle**: Waits **2ms** and broadcasts.
  - **Unstable Vehicle**: Waits **80ms**. Sensing the first broadcast, it kills its own timer.
- **The Goal**: Eliminate the "Broadcast Storm" by ensuring the most predictable vehicle shouts first.
