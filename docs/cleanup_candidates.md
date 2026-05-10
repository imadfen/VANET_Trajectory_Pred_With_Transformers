# Codebase Cleanup Candidates (Final Revised)

> [!IMPORTANT]
> **Phases 2 and 3 of the thesis architecture (Clustering + Reachability) are NOT legacy pedestrian code.**
> They are core components of the proposed VANET system. However, they were written 
> for the SinD pedestrian dataset and are **currently broken/unadapted** for vehicle data.
> This document records what is safe to delete vs. what must be rewritten.

---

## The 3-Phase Architecture (Thesis Sections 3.4.1–3.4.3)

```
Phase 1 ──────────────► Phase 2 ───────────────────────► Phase 3
Transformer Encoder      HDBSCAN Behavior Clustering       ANN Cluster Match
(src/transformer_model)  (src/clustering/)                 + Reachability Analysis
       ✅ Done            ❌ Broken — needs rewriting       (src/reachability_analysis/)
                                                            ❌ Broken — needs rewriting
                                                                    ↓
                                                            Reachable Set → Dual-Loop feedback
```

---

## 🔴 Safe to Delete Right Now — Genuinely Pedestrian-Specific

Files with **zero reusable logic** for vehicles. Hardcoded to the SinD Tianjin intersection OSM map, pedestrian polygon geometry, and pedestrian crossing intent labels.

| File | Why |
|------|-----|
| `src/datasets/plot.py` | Renders the Tianjin intersection OSM map. Hardcoded path to `Pedestrian_Project/resources/intersection.jpg`. All plot functions use `pedestrian_data={}` parameter naming and overlay crosswalk/sidewalk polygons. Zero vehicle concept. |
| `src/utils/poly_process.py` | Cuts the SinD crosswalk polygon into 4 sub-zones (West/North/East/South) using hardcoded coordinates from the Tianjin map. Only consumed by `labeling_oracle.py`. |
| `src/reachability_analysis/labeling_oracle.py` | Labels **pedestrian** crossing intents: `cross_left`, `cross_right`, `cross_straight`, `cross_illegal`, `cross_now`, `not_cross`, `unknown`. Uses crosswalk polygon intersections from `SinDMap`. Completely inapplicable to vehicles on a road. This labeler must be **rebuilt** as a vehicle-intent labeler (MaintainLane / Turn / Exit / Brake) to power the intent head added in the Dual-Loop. |
| `labeling_oracle_Soderlund.ipynb` | Interactive notebook that runs the pedestrian labeling oracle. |

---

## ❌ Currently Broken — Core Thesis Phases That Need Rewriting

### Why they are broken right now

**`src/clustering/run.py`**
- Imports `SinDMap` and `LabelingOracleSINDData` — both pedestrian-only
- `get_original_data()` only reads 6 features: `["x", "y", "vx", "vy", "ax", "ay"]` → your vehicle data has **23 VANET features**
- Calls `encoder(X, mask)` returning **2 values** → crashes because the dual-loop encoder now returns **4**: `(recon_output, intent_logits, attn_maps, embeddings)`
- Hardcoded default model path: `SINDDataset_pretrained_2024-04-27_00-11-45_KIP`
- Plot calls pass `pedestrian_data={}` to `SinDMap.plot_dataset()`

**`src/clustering/Labels.py`**
- Line 1: `from src.reachability_analysis.labeling_oracle import LabelingOracleSINDData, LABELS` — hardcoded pedestrian labels
- Line 51: reshapes data to shape `(-1, 6)` — your vehicle data is `(-1, 23)`
- All plotting goes through `labeling_oracle.map.plot_dataset(pedestrian_data=...)` using the SinD map

**`src/reachability_analysis/` (most files)**
- `input_state.py`: Defines pedestrian position/velocity state vector (x, y, vx, vy, walking speed)
- `simulation.py`: Simulates pedestrian kinematic motion — walking speeds, not vehicle dynamics
- `inclusion_accuracy.py`: Checks if a pedestrian falls within a reachable zonotope computed from pedestrian kinematics
- None of the state vectors reference LaneID, AvgMsgDelay, PacketLossRate, neighbour relative fields, or any VANET signal

### What CAN be reused as-is (no changes)
| File | Status |
|------|--------|
| `src/clustering/Clusters.py` | `HDBSCANCluster` operates on numpy embedding arrays — geometry-agnostic ✅ |
| `src/clustering/NearestNeighbor.py` | `AnnoyModel` is a generic nearest-neighbour lookup — geometry-agnostic ✅ |
| `src/reachability_analysis/operations.py` | Zonotope/polygon union operations — pure geometry ✅ |
| `src/reachability_analysis/zonotope.py` | Data structure — pure geometry ✅ |
| `src/reachability_analysis/utils.py` | Generic helpers ✅ |

---

## 🟢 Keep — Vehicle-Ready, No Changes Needed

| File | Status |
|------|--------|
| `src/datasets/data.py` (`SINDData` class) | Name is legacy; it already loads 23-feature VANET CSV files. Works correctly. |
| `src/datasets/datasplit.py` | Generic train/val split. No pedestrian coupling. |
| `src/datasets/masked_datasets.py` | Generic imputation masking. No pedestrian coupling. |
| `src/transformer_model/` | Dual-loop encoder fully implemented. |
| `src/loops/` | Loop A and Loop B implemented. |
| `src/utils/load_data.py` | Vehicle-ready. |
| `src/utils/model_helpers.py` | Generic. |
| `src/utils/record_data.py` | Generic. |

---

## Recommended Action Order

| Step | Action |
|------|--------|
| **1. Now** | Delete `src/datasets/plot.py`, `src/utils/poly_process.py`, `src/reachability_analysis/labeling_oracle.py`, `labeling_oracle_Soderlund.ipynb` |
| **2. Next** | Rewrite `clustering/run.py` per instructions in `dual_loop_instructions.md` §6 |
| **3. After** | Rewrite `clustering/Labels.py` with vehicle driving-behavior labels (see `dual_loop_instructions.md` §7) |
| **4. Last** | Adapt `reachability_analysis/input_state.py` + `simulation.py` to use vehicle kinematics (see `dual_loop_instructions.md` §8) |
