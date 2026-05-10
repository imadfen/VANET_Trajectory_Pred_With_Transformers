"""
test_inclusion_accuracy.py — verify the vehicle reachable set inclusion metric.

Run from project root:
    python src/tests/test_inclusion_accuracy.py
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from src.reachability_analysis.inclusion_accuracy import compute_inclusion_accuracy
from src.reachability_analysis.zonotope import zonotope

# Helper — build a simple 2×2 square zonotope centred at c with half-side r
def square_zonotope(cx, cy, r):
    z = zonotope(
        np.array([cx, cy]),
        np.array([[r, 0.0], [0.0, r]]),
    )
    z.color = [0.3, 0.6, 0.9, 0.4]
    return z

# ── 1. All points inside → 100% ────────────────────────────────────────────
print("[Test 1] All ground-truth points INSIDE → 100% accuracy ...")
z = square_zonotope(0.0, 0.0, 5.0)
mock_ra = {"veh_0": [z, z, z]}
mock_gt = {"veh_0": np.array([[0.1, 0.1], [1.0, 1.0], [4.5, 0.0]])}

results = compute_inclusion_accuracy(mock_ra, mock_gt)
acc = results["accuracy_per_step"]
assert (acc == 100.0).all(), f"Expected 100%, got {acc}"
print(f"  PASS  accuracy = {acc}%")

# ── 2. All points outside → 0% ────────────────────────────────────────────
print("[Test 2] All ground-truth points OUTSIDE → 0% accuracy ...")
mock_gt_out = {"veh_0": np.array([[50.0, 50.0], [50.0, 50.0], [50.0, 50.0]])}
results_out = compute_inclusion_accuracy(mock_ra, mock_gt_out)
acc_out = results_out["accuracy_per_step"]
assert (acc_out == 0.0).all(), f"Expected 0%, got {acc_out}"
print(f"  PASS  accuracy = {acc_out}%")

# ── 3. Mixed: 2 vehicles, 1 inside 1 outside → 50% ───────────────────────
print("[Test 3] Mixed vehicles → 50% accuracy ...")
mock_ra_mix = {
    "veh_0": [z],          # inside
    "veh_1": [z],          # outside
}
mock_gt_mix = {
    "veh_0": np.array([[0.5, 0.5]]),    # inside
    "veh_1": np.array([[99.0, 99.0]]),  # outside
}
results_mix = compute_inclusion_accuracy(mock_ra_mix, mock_gt_mix)
acc_mix = results_mix["accuracy_per_step"]
assert abs(acc_mix[0] - 50.0) < 0.01, f"Expected 50%, got {acc_mix[0]}"
print(f"  PASS  accuracy = {acc_mix[0]:.1f}%")

# ── 4. Degenerate zonotope (huge area) is excluded ────────────────────────
print("[Test 4] Degenerate (huge) zonotope is excluded from counting ...")
z_huge = square_zonotope(0.0, 0.0, 200.0)  # area >> max_area threshold (5000 m²)
mock_ra_degen = {"veh_0": [z_huge]}
mock_gt_degen = {"veh_0": np.array([[0.0, 0.0]])}
results_degen = compute_inclusion_accuracy(mock_ra_degen, mock_gt_degen, max_area=100.0)
n = results_degen["n_evaluated"]
assert n == 0, f"Expected 0 evaluations (all excluded), got {n}"
print(f"  PASS  degenerate zonotope excluded (n_evaluated={n})")

# ── 5. Missing vehicle in ground_truth is skipped ─────────────────────────
print("[Test 5] Vehicle missing from ground_truth is skipped ...")
mock_ra_miss = {"veh_0": [z], "veh_ghost": [z]}
mock_gt_miss = {"veh_0": np.array([[0.5, 0.5]])}   # veh_ghost not in GT
results_miss = compute_inclusion_accuracy(mock_ra_miss, mock_gt_miss)
assert results_miss["n_evaluated"] == 1, \
    f"Expected 1 evaluated (veh_ghost skipped), got {results_miss['n_evaluated']}"
print(f"  PASS  veh_ghost skipped, n_evaluated={results_miss['n_evaluated']}")

print()
print("All inclusion_accuracy tests PASSED.")
