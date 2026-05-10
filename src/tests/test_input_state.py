"""
test_input_state.py — verify VANET state vector extraction and I/O state creation.

Run from project root:
    python src/tests/test_input_state.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from src.reachability_analysis.input_state import (
    IDX_X, IDX_Y, IDX_SPEED, IDX_HEADING,
    filter_paddings,
    separate_data_to_class,
    structure_input_data,
    structure_input_data_for_clusters,
)
from src.reachability_analysis.simulation import get_initial_conditions

# ── 1. Feature indices ─────────────────────────────────────────────────────
print("[Test 1] Feature indices ...")
assert IDX_X == 0,       f"Expected IDX_X=0,       got {IDX_X}"
assert IDX_Y == 1,       f"Expected IDX_Y=1,       got {IDX_Y}"
assert IDX_SPEED == 2,   f"Expected IDX_SPEED=2,   got {IDX_SPEED}"
assert IDX_HEADING == 4, f"Expected IDX_HEADING=4, got {IDX_HEADING}"
print(f"  PASS  X={IDX_X}  Y={IDX_Y}  Speed={IDX_SPEED}  Heading={IDX_HEADING}")

# ── 2. filter_paddings ─────────────────────────────────────────────────────
print("[Test 2] filter_paddings ...")
data    = np.arange(24).reshape(4, 2, 3).astype(float)   # 4 chunks
padding = np.array([[True, True], [True, False], [True, True], [False, True]])
filtered = filter_paddings(data, padding)
assert filtered.shape[0] == 2, f"Expected 2 unpadded chunks, got {filtered.shape[0]}"
print(f"  PASS  4 chunks → {filtered.shape[0]} fully unpadded")

# ── 3. separate_data_to_class ─────────────────────────────────────────────
print("[Test 3] separate_data_to_class ...")
N, T, F = 12, 50, 23
data   = np.random.randn(N, T, F)
labels = np.array([0, 0, 1, 1, 2, 0, 3, 1, 2, 3, 0, 2])
grouped = separate_data_to_class(data, labels, size=4)
assert set(grouped.keys()) == {0, 1, 2, 3}
assert grouped[0].shape[0] == 4   # label 0 appears 4 times
assert grouped[1].shape[0] == 3   # label 1 appears 3 times
assert grouped[2].shape[0] == 3   # label 2 appears 3 times
assert grouped[3].shape[0] == 2   # label 3 appears 2 times
counts_str = {k: v.shape[0] for k, v in grouped.items()}
print(f"  PASS  counts = {counts_str}")

# ── 4. structure_input_data (equalise classes) ────────────────────────────
print("[Test 4] structure_input_data (balanced classes) ...")
new_d, new_l = structure_input_data(data, labels)
unique, counts = np.unique(new_l, return_counts=True)
assert len(set(counts)) == 1, f"Classes not balanced: {dict(zip(unique, counts))}"
print(f"  PASS  balanced at {counts[0]} samples per class")

# ── 5. structure_input_data_for_clusters (cap at max_data) ────────────────
print("[Test 5] structure_input_data_for_clusters (cap) ...")
capped_d, capped_l = structure_input_data_for_clusters(data, labels, max_data=2)
for lbl_id in np.unique(capped_l):
    n = (capped_l == lbl_id).sum()
    assert n <= 2, f"Class {lbl_id} has {n} samples > max_data=2"
print(f"  PASS  all classes capped at ≤2 samples")

# ── 6. get_initial_conditions: pos and vel from VANET chunk ──────────────
print("[Test 6] get_initial_conditions from VANET chunk ...")
chunk = np.zeros((50, 23))
chunk[:, IDX_X]       = np.linspace(0, 100, 50)   # X from 0→100 m
chunk[:, IDX_Y]       = 5.0                        # constant lane Y
chunk[:, IDX_SPEED]   = 20.0                       # 20 m/s
chunk[:, IDX_HEADING] = np.pi / 4                  # 45° heading

pos, vel = get_initial_conditions(chunk)

expected_vx = 20.0 * np.cos(np.pi / 4)
expected_vy = 20.0 * np.sin(np.pi / 4)

assert abs(pos[0] - 100.0)    < 0.01, f"pos X wrong: {pos[0]}"
assert abs(pos[1] - 5.0)      < 0.01, f"pos Y wrong: {pos[1]}"
assert abs(vel[0] - expected_vx) < 0.01, f"vel x wrong: {vel[0]}"
assert abs(vel[1] - expected_vy) < 0.01, f"vel y wrong: {vel[1]}"
print(f"  PASS  pos={np.round(pos,2)}  vel={np.round(vel,2)}")

print()
print("All input_state tests PASSED.")
