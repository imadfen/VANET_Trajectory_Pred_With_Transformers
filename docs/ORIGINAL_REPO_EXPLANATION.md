# VANET Trajectory Prediction with Transformers

This project focuses on predicting vehicle and pedestrian trajectories in Vehicular Ad-hoc Networks (VANETs) using a **Transformer-based architecture**, combined with **HDBSCAN clustering** and **Zonotope-based reachability analysis** for formal safety verification.

---

## Project Architecture

The repository is structured to separate data handling, neural modeling, and geometric verification.

### 1. Core Logic & Model (`src/transformer_model/`)

- **`model.py`**: Manages training loops, loss functions, and evaluation logic.
- **`encoder.py`**: Implementation of Transformer layers and Multi-Head Self-Attention.
- **`main.py`**: Primary entry point for training and testing.

### 2. Clustering & Retrieval (`src/clustering/`)

- **`run.py`**: Extracts latent embeddings and performs HDBSCAN clustering.
- **`NearestNeighbor.py`**: Uses **Annoy** for $O(\log n)$ nearest-neighbor lookups.

### 3. Data Processing (`src/datasets/`)

- **`data.py`**: Handles loading raw SinD (State-in-Drones) files and converting them to standardized sequences.
- **`masked_datasets.py`**: Manages data masking and windowing for the Transformer training.
- **`datasplit.py`**: Splits the data into training, validation, and testing sets.

### 4. Safety Verification (`src/reachability_analysis/`)

- **`zonotope.py`**: Geometric and mathematical definitions for Zonotope operations.
- **`simulation.py`**: Visualizes reachable sets and safety envelopes.
- **`inclusion_accuracy.py`**: Checks if ground-truth points fall within predicted bounds.

### 5. Ground Truth & Persistence (`resources/`)

- **`test/`**: High-fidelity smoothed tracks categorized by specific behavioral scenarios (e.g., `cross_illegal`, `cross_now`) used for safety envelope verification.
- **`VANET_data/`**: Local bridge for the 23-feature high-dimensional datasets.
- **`clusters/`**: Persistent storage for pre-computed HDBSCAN weights and Annoy search indices, enabling rapid retrieval of driving "fingerprints" without re-computation.

---

## The Master Pipeline Breakdown

Follow a single sample of **Vehicle 42** through the system logic and technical data flow.

### Step 1: The "Looking Back" Phase (Preprocessing)

- **What we have**: A messy list of GPS coordinates from a drone.
- **What we do**:
  1. **Filtering**: We remove noisy "jitter" from the drone sensors using smoothing.
  2. **Sliding Window**: We chop the long trajectories into 3-second historical "clips" + 5-second future "targets."
  3. **Normalization**: We scale the numbers (X, Y coordinates and Speeds) so they are small and consistent (e.g., between 0 and 1).
- **The Input**: Raw SinD CSV data (x, y, speed, heading).
- **The Output**: A standardized sequence tensor ready for the Transformer.
- **The Goal**: Clean and package the data so the model can read it easily.
- **Technical Detail**:
  - **Shape**: `[Batch, Seq_Len, Features]` $\rightarrow$ `[1, 30, 4]` (30 time steps, 4 features like x, y, vx, vy).
  - **Module**: `src/datasets/`

### Step 2: The "Pattern Recognition" Phase (Encoder)

- **What we have**: The 3-second clip.
- **What we do**: The model looks at points and asks, _"Is this vehicle turning?"_ It calculates attention weights to focus on critical frames (e.g., frames 25-30).
- **The Result**: A single string of 128 numbers (the **Latent Vector**) that represents the _flavor_ of the driving.
- **The Goal**: Turn a sequence of movement into a single representative "ID."
- **Technical Detail**: Uses **Positional Encoding** and **Multi-Head Self-Attention**.
- **Module**: `src/transformer_model/encoder.py`

### Step 3: The "Contextual Retrieval" Phase (Clustering)

- **What we have**: That "flavor" ID (Latent Vector) from Step 2.
- **What we do**:
  1. The Latent Vector acts as a **fingerprint** for the current maneuver.
  2. We query the **Annoy Index**, which contains thousands of fingerprints from historical data (e.g., recorded lane changes, stops, turns).
  3. We ask: _"Which 10 drivers in my history behaved most like this vehicle is behaving right now?"_
- **The Result**: We identify the 10 closest historical neighbors. A "98% match" means the current movement is mathematically almost identical to a specific cluster of historical trajectories.
- **The Goal**: Ground the prediction in reality by seeing how similar drivers finished their maneuvers.
- **Module**: `src/clustering/run.py`

### Step 4: The "Crystal Ball" Phase (Decoder)

- **What we have**: The "flavor" ID and historical context.
- **What we do**: The model uses a Feed-Forward Network to draw a line 5 seconds into the future.
- **The Result**: A list of coordinates (X, Y) where we _think_ the car will be (`[1, 50, 2]`).
- **The Goal**: Guess the future path based on current behavior.
- **Module**: `src/transformer_model/model.py`

### Step 5: The "Safety Net" Phase (Reachability)

- **What we have**: Our guessed future path and the model's known error margin (MSE).
- **What we do**: We draw a geometric "bubble" around our guessed path using Minkowski sums.
- **The Result**: A **Zonotope** that says "The car is almost certainly inside this area."
- **The Goal**: Provide a safety margin instead of just a single line.
- **Technical Detail**: $Z = c + \sum G_i$ where $c$ is the center and $G$ are generators.
- **Module**: `src/reachability_analysis/zonotope.py`

### Step 6: The "Reality Check" Phase (Verification)

- **What we have**: Our "safety bubble" and the _actual_ real-world drone data.
- **What we do**: We check if the real car actually stayed inside our bubble using the inequality $|A(P_{actual} - c)| \leq 1$.
- **The Result**: A score (e.g., "95% Safe").
- **The Goal**: Prove that our math is reliable enough for real-world driving.
- **Module**: `src/reachability_analysis/inclusion_accuracy.py`

---

## Live Scenario: Intersection Walkthrough

**Scenario**: Vehicle 42 is approaching a signalized intersection in the SinD dataset.

1.  **Observation**: `src/datasets` pulls 30 frames: `[[x1, y1, v1], ..., [x30, y30, v30]]`.
2.  **Inference**: `encoder.py` identifies a velocity drop and leftward yaw.
3.  **Clustering**: `NearestNeighbor.py` finds this is a 98% match for "Cluster 5: Left Turn."
4.  **Envelope**: `simulation.py` generates a Zonotope (box-like polygon) around the predicted $(x, y)$ path.
5.  **Outcome**: If the real-world sensor data stays inside the box, the prediction is verified as **Formally Safe**.
