

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from src.clustering.Labels import (
    VEHICLE_LABELS,
    assign_intent_labels,
    label_trajectories,
)
from src.reachability_analysis.input_state import VEHICLE_LABELS as VL_input

print("[Test 1] VEHICLE_LABELS contents ...")
assert VEHICLE_LABELS == {0: "MaintainLane", 1: "Turn", 2: "Exit", 3: "Brake"}
print(f"  PASS  {VEHICLE_LABELS}")

print("[Test 2] VEHICLE_LABELS consistent between Labels.py and input_state.py ...")
assert VEHICLE_LABELS == VL_input, f"Mismatch:\n  Labels: {VEHICLE_LABELS}\n  input_state: {VL_input}"
print("  PASS")

print("[Test 3] assign_intent_labels ...")
logits = np.array([
    [10.0, -1.0, -1.0, -1.0],   # → 0 MaintainLane
    [-1.0, 10.0, -1.0, -1.0],   # → 1 Turn
    [-1.0, -1.0, 10.0, -1.0],   # → 2 Exit
    [-1.0, -1.0, -1.0, 10.0],   # → 3 Brake
])
labels = assign_intent_labels(logits)
assert list(labels) == [0, 1, 2, 3], f"Expected [0,1,2,3], got {list(labels)}"
print(f"  PASS  labels = {list(labels)} → {[VEHICLE_LABELS[l] for l in labels]}")

print("[Test 4] label_trajectories grouping ...")
N, T, F = 8, 50, 51
data     = np.random.randn(N, T, F)
padding  = np.ones((N, T), dtype=bool)
clusters = np.array([0, 1, 0, 2, 1, 0, 3, 2])
lbl      = np.array([0, 0, 1, 1, 2, 2, 3, 3])

trajs, pads, clusts = label_trajectories(data, padding, clusters, lbl)

assert set(trajs.keys()) == {0, 1, 2, 3}, f"Expected keys {{0,1,2,3}}, got {set(trajs.keys())}"
assert len(trajs[0]) == 2   # lbl==0 appears at indices 0,1
assert len(trajs[1]) == 2   # lbl==1 appears at indices 2,3
assert trajs[0][0].shape == (T, F), f"Trajectory shape wrong: {trajs[0][0].shape}"
groups_str = {k: len(v) for k, v in trajs.items()}
print(f"  PASS  groups = {groups_str}")
