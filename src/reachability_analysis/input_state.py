"""
Vehicle state vector utilities for Phase 3 reachability analysis.

State vector: [X, Y, Speed, Heading]  — indices (0, 1, 2, 4) from the 23-feature VANET vector.
Velocity is derived as (Speed * cos(Heading), Speed * sin(Heading)).
"""

import pypolycontain as pp
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from typing import List, Union

# Vehicle intent labels (consistent with encoder intent_head)
VEHICLE_LABELS = {
    0: "MaintainLane",
    1: "Turn",
    2: "Exit",
    3: "Brake",
}

# Feature indices inside the 23-dim VANET vector
IDX_X       = 0
IDX_Y       = 1
IDX_SPEED   = 2
IDX_HEADING = 4   # radians


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def filter_paddings(dataset: np.ndarray, padded_batches: np.ndarray) -> np.ndarray:
    """Keep only fully unpadded trajectory chunks."""
    unpadded = np.all(padded_batches, axis=1)
    return dataset[unpadded]


# ---------------------------------------------------------------------------
# Data organisation
# ---------------------------------------------------------------------------

def separate_data_to_class(
    data: np.ndarray, classification: np.ndarray, size: int
) -> dict:
    """Separate dataset into a dict keyed by class label.

    Parameters
    ----------
    data : np.ndarray  shape (N, seq_len, feat_dim)
    classification : np.ndarray  shape (N,)
    size : int  number of classes

    Returns
    -------
    dict { class_id -> np.ndarray (M, seq_len, feat_dim) }
    """
    _class = {i: [] for i in range(size)}
    for i, traj in enumerate(data):
        _class[int(classification[i])].append(traj)
    return {k: np.array(v) for k, v in _class.items()}


def structure_input_data(data: np.ndarray, labels: np.ndarray):
    """Equalise class sizes by random sub-sampling to the smallest class count."""
    _d = {l: [] for l in np.unique(labels)}
    for i, l in enumerate(labels):
        _d[l].append(data[i])
    _min = min(len(v) for v in _d.values())
    new_d, new_l = [], []
    for l, v in _d.items():
        v = np.array(v)
        ids = np.random.randint(0, len(v), size=_min)
        new_d.extend(v[ids])
        new_l.extend([l] * _min)
    return np.array(new_d), np.array(new_l)


def structure_input_data_for_clusters(
    data: np.ndarray, labels: np.ndarray, max_data: int = 100
):
    """Cap each class at max_data samples for efficient reachability computation."""
    _d = {l: [] for l in np.unique(labels)}
    for i, l in enumerate(labels):
        _d[l].append(data[i])
    new_d, new_l = [], []
    for l, v in _d.items():
        v = np.array(v)
        ids = np.random.randint(0, len(v), size=min(max_data, len(v)))
        new_d.extend(v[ids])
        new_l.extend([l] * len(ids))
    return np.array(new_d), np.array(new_l)


# ---------------------------------------------------------------------------
# I/O state construction for reachability
# ---------------------------------------------------------------------------

def _vel_from_row(row: np.ndarray) -> np.ndarray:
    """Derive 2-D velocity vector from Speed and Heading (vehicle kinematics)."""
    speed   = row[IDX_SPEED]
    heading = row[IDX_HEADING]
    return np.array([speed * np.cos(heading), speed * np.sin(heading)])


def create_io_state(
    data: dict,
    measurement: pp.zonotope,
    vel: np.ndarray,
    classification: Union[int, List[int]],
    drop_equal: bool = True,
    angle_filter: bool = True,
    method: str = "",
    data_statistics: dict = None,
    clustering: bool = False,
) -> List[np.ndarray]:
    """Build D = (U_d, X+, X-, U) for the LTI reachability algorithm.

    State: (X, Y)   Input: (vx, vy) derived from Speed and Heading.

    Parameters
    ----------
    data : dict of { class_id -> (M, seq_len, 23) }
    measurement : pp.zonotope  – initial position zonotope (2-D)
    vel : np.ndarray  – current velocity (2,)
    classification : int or list[int]
    """
    if isinstance(classification, list):
        _data = np.concatenate(
            [data[cls] for cls in classification if cls in data], axis=0
        )
    else:
        _data = data.get(classification, np.array([]))

    if _data.size == 0:
        return None

    X_m, X_p, U = np.array([]), np.array([]), np.array([])
    _pos_poly = Polygon(pp.to_V(measurement))

    _angle_set = (
        np.array([
            np.arctan2(vel[1], vel[0]) - (np.pi / 8),
            np.arctan2(vel[1], vel[0]) + (np.pi / 8),
        ])
        if angle_filter
        else False
    )

    for traj in _data:
        # traj: (seq_len, 23)  — extract X, Y and derived vx, vy
        _x = traj[:, IDX_X]
        _y = traj[:, IDX_Y]
        vel_vectors = np.array([_vel_from_row(row) for row in traj])  # (seq_len, 2)
        _vx, _vy = vel_vectors[:, 0], vel_vectors[:, 1]

        _line = LineString(list(zip(_x, _y)))
        _X = np.array([_x, _y])          # (2, seq_len)
        _X_p = _X[:, 1:]                 # X+
        _X_m = _X[:, :-1]               # X-
        _U   = np.array([_vx, _vy])[:, :-1]  # inputs

        init_heading = np.arctan2(_vy[0], _vx[0])
        if (
            _line.intersects(_pos_poly)
            and _in_between(init_heading, _angle_set)
        ) or clustering:
            X_p = np.hstack([X_p, _X_p]) if X_p.size else _X_p
            X_m = np.hstack([X_m, _X_m]) if X_m.size else _X_m
            U   = np.hstack([U,   _U])   if U.size   else _U

    if X_p.size == 0:
        if data_statistics is not None:
            data_statistics.setdefault(method, {}).setdefault("data_constraint", 0)
            data_statistics[method]["data_constraint"] += 1
        return None

    # Safety guard against memory blow-up
    if X_p.shape[1] / max(_data.shape[0], 1) > 300:
        if data_statistics is not None:
            data_statistics.setdefault(method, {}).setdefault("memory_constraint", 0)
            data_statistics[method]["memory_constraint"] += 1
        return None

    return [U, X_p, X_m, U]


# ---------------------------------------------------------------------------
# Angle helper
# ---------------------------------------------------------------------------

def _in_between(val: float, angle_range) -> bool:
    """True if val lies within the half-open angular range."""
    if angle_range is False:
        return True
    a_min, a_max = float(angle_range[0]), float(angle_range[1])
    if a_min <= a_max:
        return a_min <= val <= a_max
    return val >= a_min or val <= a_max


# keep old name for backward compatibility
__in_between = _in_between


# ---------------------------------------------------------------------------
# Trajectory split for LTI data-driven reachability
# ---------------------------------------------------------------------------

def split_io_to_trajs(
    X_p: np.ndarray,
    X_m: np.ndarray,
    U: np.ndarray,
    threshold: float = 5.0,
    dropped: bool = True,
    N: int = 30,
):
    """Split stacked IO arrays back into individual trajectory segments."""
    _X_p, _X_m, _U = [], [], []
    if dropped:
        x_prev = X_p[:, 0]
        i_prev = 0
        for i, x in enumerate(X_p[:, 1:].T):
            if np.linalg.norm(x - x_prev) > threshold:
                _X_p.append(X_p[:, i_prev:i + 1])
                _X_m.append(X_m[:, i_prev:i + 1])
                _U.append(U[:, i_prev:i + 1])
                i_prev = i + 1
            x_prev = x
    else:
        for i in range(N, U.shape[1] + 1, N):
            _U.append(U[:, i - N:i])

    if not _U:
        _X_p.append(X_p)
        _X_m.append(X_m)
        _U.append(U)

    return _X_p, _X_m, _U
