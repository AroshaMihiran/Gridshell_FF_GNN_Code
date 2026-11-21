"""Framework-neutral CSV graph loading; no training or fitted preprocessing.

Copied from Dataset_FFT_GNN_15000_2026-11-16/src/training_io.py on 2026-11-17.
The parser implementation is unchanged; kept local for portable training.

All returned features retain their declared physical units. Rows are sorted by
integer entity ID and connectivity is mapped explicitly; IDs need not be
contiguous or zero based. Only member.csv creates physical structural edges.
Panel incidences are directed panel-to-joint relations, never invented bars.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np


LPE_COLUMNS = tuple(f"lap_pe_{i:02d}" for i in range(1, 9))
NODE_FEATURE_NAMES = ("x0_m", "y0_m") + LPE_COLUMNS + ("support",)
HOMOGENEOUS_LOAD_FEATURE_NAMES = ("PD0_kN", "PW0_kN")
MEMBER_FEATURE_NAMES = ("E_kPa", "A_m2", "L0_m", "EA_over_L0_kN_per_m")
PANEL_FEATURE_NAMES = ("qD_kPa", "qW_kPa", "initial_area_m2")
TARGET_COLUMNS = ("ux_ff_m", "uy_ff_m", "uz_ff_m")
SUPPORT_DISPLACEMENT_TOLERANCE_M = 1e-10


def _read_csv(path, required):
    if not path.is_file():
        raise ValueError(f"Missing graph CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            columns = [s.strip() for s in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"{path.name} is empty.") from exc
        if len(columns) != len(set(columns)) or "" in columns:
            raise ValueError(f"{path.name} has duplicate or empty column names.")
        missing = sorted(set(required) - set(columns))
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")
        rows = []
        for line, values in enumerate(reader, 2):
            if not values or all(not v.strip() for v in values):
                continue
            if len(values) != len(columns):
                raise ValueError(f"{path.name} line {line} has {len(values)} fields; expected {len(columns)}.")
            rows.append(dict(zip(columns, (v.strip() for v in values))))
    if not rows:
        raise ValueError(f"{path.name} contains no entities.")
    return rows, set(columns)


def _integer(value, context):
    # Reject floats, booleans and silently truncated IDs; integral CSV tokens
    # and JSON integer values are the schema's supported identifiers.
    if isinstance(value, (bool, np.bool_)) or not re.fullmatch(r"[+-]?\d+", str(value)):
        raise ValueError(f"{context} must be an integer, got {value!r}.")
    number = int(value)
    if not np.iinfo(np.int64).min <= number <= np.iinfo(np.int64).max:
        raise ValueError(f"{context} is outside the supported 64-bit integer range.")
    return number


def _sort_entities(rows, id_column, filename):
    by_id = {}
    for row in rows:
        entity_id = _integer(row[id_column], f"{filename} {id_column}")
        if entity_id in by_id:
            raise ValueError(f"{filename} has duplicate {id_column} {entity_id}.")
        by_id[entity_id] = row
    ids = np.array(sorted(by_id), dtype=np.int64)
    return [by_id[int(i)] for i in ids], ids, {int(v): i for i, v in enumerate(ids)}


def _numbers(rows, columns, filename):
    result = np.empty((len(rows), len(columns)), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, name in enumerate(columns):
            try:
                value = float(row[name])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{filename} entity row {i + 1}: {name} must contain a finite number, got {row[name]!r}.") from exc
            if not np.isfinite(value):
                raise ValueError(f"{filename} entity row {i + 1}: {name} must be finite.")
            result[i, j] = value
    return result


def _joint_index(value, joint_map, context):
    entity_id = _integer(value, context)
    if entity_id not in joint_map:
        raise ValueError(f"{context} references missing joint_id {entity_id}.")
    return joint_map[entity_id]


def _targets(rows, columns, coords, supports, require_targets):
    present = [c in columns for c in TARGET_COLUMNS]
    if any(present) and not all(present):
        raise ValueError("joint.csv has only some FFT target columns; supply ux_ff_m, uy_ff_m and uz_ff_m together.")
    filled = np.array([[bool(row[c]) for c in TARGET_COLUMNS] for row in rows]) if all(present) else np.zeros((len(rows), 3), bool)
    if np.any(filled) and not np.all(filled):
        raise ValueError("joint.csv contains partially filled FFT targets. Every displacement component at every joint must be present, including zero values at supports.")
    if not np.any(filled):
        if require_targets:
            raise ValueError("FFT displacement targets are absent or blank. Run and validate FFT before training, or use load_graph(..., require_targets=False) to inspect inputs only.")
        return None, None, "input_only"
    displacement = _numbers(rows, TARGET_COLUMNS, "joint.csv")
    if np.any(np.abs(displacement[supports]) > SUPPORT_DISPLACEMENT_TOLERANCE_M):
        raise ValueError(f"Supported-joint FFT displacements must be zero within {SUPPORT_DISPLACEMENT_TOLERANCE_M:g} m; supports fix all translations.")
    target = coords + displacement
    if not np.all(np.isfinite(target)):
        raise ValueError("Final coordinate targets overflowed when adding FFT displacements.")
    return displacement, target, "labelled"


def load_graph(sample_dir, require_targets=True):
    """Load joint.csv, member.csv and panel.csv as a NumPy graph dictionary.

    ``node_features`` has columns x0, y0, eight stored Laplacian coordinates,
    support (11 columns). ``homogeneous_load_features`` contains the separate
    dead/signed-wind initial nodal forces; ``homogeneous_node_features`` appends
    these two columns. Initial z is retained in ``initial_coords`` but, as in
    the agreed flat-layout model, is not an input feature.

    ``physical_edge_index`` is 2 x M in member-ID order. The first M columns
    of ``structural_edge_index`` follow those directions and the second M
    reverse them. ``structural_edge_features`` and ``structural_edge_to_member``
    use the same order. ``panel_incidence_index`` is 2 x K with panel rows
    first and joint rows second, preserving each panel's cyclic corner order.

    ``target`` and ``target_coords`` are the final coordinates (initial + FFT
    displacement). Fully absent/blank targets are permitted only when
    require_targets=False, returning None and status='input_only'. Complete
    targets are loaded in either mode; partial targets always fail. No blank
    value is converted to zero. ``free_mask`` selects joints for free-DOF loss;
    support labels are checked against a 1e-10 m displacement tolerance.

    Additional CSV columns are permitted. IDs, required numerical values,
    connectivity, feature dimensions, pressure signs and initial-load sign
    consistency are validated. This routine does not certify FFT convergence.
    """
    if not isinstance(require_targets, (bool, np.bool_)):
        raise ValueError("require_targets must be True or False.")
    directory = Path(sample_dir)
    joint_required = ("joint_id", "x0_m", "y0_m", "z0_m", "support") + LPE_COLUMNS + (
        "PD0_kN", "PW0_kN", "Fx0_kN", "Fy0_kN", "Fz0_kN")
    member_required = ("member_id", "start_joint_id", "end_joint_id") + MEMBER_FEATURE_NAMES
    panel_required = ("panel_id", "corner_joint_ids", "n_corners", "panel_type") + PANEL_FEATURE_NAMES
    joints, joint_columns = _read_csv(directory / "joint.csv", joint_required)
    members, _ = _read_csv(directory / "member.csv", member_required)
    panels, _ = _read_csv(directory / "panel.csv", panel_required)
    joints, joint_ids, joint_map = _sort_entities(joints, "joint_id", "joint.csv")
    members, member_ids, member_map = _sort_entities(members, "member_id", "member.csv")
    panels, panel_ids, panel_map = _sort_entities(panels, "panel_id", "panel.csv")

    coords = _numbers(joints, ("x0_m", "y0_m", "z0_m"), "joint.csv")
    node_features = _numbers(joints, NODE_FEATURE_NAMES, "joint.csv")
    support_values = node_features[:, -1]
    if np.any((support_values != 0) & (support_values != 1)):
        raise ValueError("joint.csv support must be 0 or 1; 1 fixes all three translations.")
    supports = support_values.astype(bool)
    nodal_pressure_loads = _numbers(joints, HOMOGENEOUS_LOAD_FEATURE_NAMES, "joint.csv")
    initial_forces = _numbers(joints, ("Fx0_kN", "Fy0_kN", "Fz0_kN"), "joint.csv")
    if np.any(nodal_pressure_loads[:, 0] < 0) or np.any(nodal_pressure_loads[:, 1] > 0):
        raise ValueError("joint.csv requires PD0_kN >= 0 (dead) and PW0_kN <= 0 (suction).")
    if not np.allclose(initial_forces[:, :2], 0, rtol=0, atol=1e-10):
        raise ValueError("joint.csv initial panel loads must be vertical (Fx0_kN=Fy0_kN=0).")
    if not np.allclose(initial_forces[:, 2], -nodal_pressure_loads.sum(axis=1), rtol=1e-9, atol=1e-10):
        raise ValueError("joint.csv initial force convention requires Fz0_kN = -(PD0_kN + PW0_kN).")

    member_features = _numbers(members, MEMBER_FEATURE_NAMES, "member.csv")
    if np.any(member_features <= 0):
        raise ValueError("member.csv E, A, L0 and EA/L0 must be strictly positive.")
    edges = []
    seen_pairs = set()
    for member in members:
        u = _joint_index(member["start_joint_id"], joint_map, "member.csv start_joint_id")
        v = _joint_index(member["end_joint_id"], joint_map, "member.csv end_joint_id")
        if u == v:
            raise ValueError("member.csv contains a self-loop member.")
        pair = tuple(sorted((u, v)))
        if pair in seen_pairs:
            raise ValueError("member.csv repeats a physical member endpoint pair.")
        seen_pairs.add(pair)
        edges.append((u, v))
    physical_edges = np.asarray(edges, dtype=np.int64).T
    directed_edges = np.concatenate((physical_edges, physical_edges[::-1]), axis=1)

    panel_features = _numbers(panels, PANEL_FEATURE_NAMES, "panel.csv")
    if np.any(panel_features[:, 0] < 0) or np.any(panel_features[:, 1] > 0) or np.any(panel_features[:, 2] <= 0):
        raise ValueError("panel.csv requires qD_kPa >= 0, qW_kPa <= 0, and initial_area_m2 > 0.")
    panel_corners, incidence_panels, incidence_joints, panel_types = [], [], [], []
    for p, panel in enumerate(panels):
        try:
            corner_ids = json.loads(panel["corner_joint_ids"])
        except json.JSONDecodeError as exc:
            raise ValueError("panel.csv corner_joint_ids must be a JSON array of integer joint IDs.") from exc
        if not isinstance(corner_ids, list):
            raise ValueError("panel.csv corner_joint_ids must be a JSON array.")
        count = _integer(panel["n_corners"], "panel.csv n_corners")
        if count < 3 or count != len(corner_ids):
            raise ValueError("panel.csv n_corners must match the JSON array and be at least three.")
        corners = np.array([_joint_index(value, joint_map, "panel.csv corner_joint_ids") for value in corner_ids], dtype=np.int64)
        if len(np.unique(corners)) != count:
            raise ValueError("panel.csv has repeated panel corners; do not repeat the first corner at the end.")
        panel_type = panel["panel_type"]
        if not panel_type:
            raise ValueError("panel.csv panel_type must not be blank.")
        if (panel_type.upper() == "T3" and count != 3) or (panel_type.upper() == "Q4" and count != 4):
            raise ValueError("panel.csv panel_type and n_corners disagree.")
        panel_types.append(panel_type)
        panel_corners.append(corners)
        incidence_panels.extend([p] * count)
        incidence_joints.extend(corners.tolist())

    displacement, target, status = _targets(joints, joint_columns, coords, supports, require_targets)
    metadata = {
        "schema_version": "fft-gnn-csv-graph-1",
        "feature_scaling": "raw; no normalization fitted or applied",
        "normalization_policy": "Fit any preprocessing on the training partition only.",
        "id_order": "ascending integer IDs; explicit maps, no contiguous-ID assumption",
        "joint_feature_names": NODE_FEATURE_NAMES,
        "homogeneous_load_feature_names": HOMOGENEOUS_LOAD_FEATURE_NAMES,
        "member_feature_names": MEMBER_FEATURE_NAMES,
        "panel_feature_names": PANEL_FEATURE_NAMES,
        "structural_edge_order": "one forward copy per sorted member, then one reverse copy",
        "panel_incidence_direction": "panel_to_joint_only",
        "support_displacement_tolerance_m": SUPPORT_DISPLACEMENT_TOLERANCE_M,
        "targets_are_convergence_certification": False,
        "warnings": [
            "Stored Laplacian eigenvector signs and bases of repeated eigenspaces are conventions; training-time sign/eigenspace augmentation must be decided later.",
            "Laplacian encoding is topological and does not replace the raw geometry coordinates.",
            "No training, augmentation, normalization fit, or FFT convergence validation is performed by this loader.",
        ],
    }
    return {
        "status": status, "sample_dir": str(directory.resolve()),
        "joint_ids": joint_ids, "member_ids": member_ids, "panel_ids": panel_ids,
        "joint_id_to_index": joint_map, "member_id_to_index": member_map, "panel_id_to_index": panel_map,
        "initial_coords": coords, "node_features": node_features,
        "homogeneous_load_features": nodal_pressure_loads,
        "homogeneous_node_features": np.column_stack((node_features, nodal_pressure_loads)),
        "member_features": member_features, "panel_features": panel_features,
        "physical_edge_index": physical_edges, "structural_edge_index": directed_edges,
        "structural_edge_features": np.concatenate((member_features, member_features), axis=0),
        "structural_edge_to_member": np.tile(np.arange(len(members), dtype=np.int64), 2),
        "panel_incidence_index": np.array([incidence_panels, incidence_joints], dtype=np.int64),
        "panel_corners": panel_corners, "panel_types": tuple(panel_types),
        "support_mask": supports, "free_mask": ~supports,
        "initial_forces_kN": initial_forces,
        "PD0_kN": nodal_pressure_loads[:, 0].copy(), "PW0_kN": nodal_pressure_loads[:, 1].copy(),
        "target_displacements": displacement, "target": target, "target_coords": target,
        "metadata": metadata,
    }
