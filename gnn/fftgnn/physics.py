"""Differentiable FFT mechanics, in metres, kN and kPa (= kN/m²).

The fixed initial lengths, current-surface dead pressure and horizontal-
projection signed wind pressure reproduce the dataset's MATLAB formulation.
Only physical members carry stiffness. Panel incidence links never do.
T3 and Q4 panels are supported; other polygons fail explicitly.

Raw geometry and properties should remain float64. Computation is promoted
to float64 here without detaching, so gradients reach the displacement head.
This module does not run FFT, size sections or certify structural stability.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


def graph_mean(values, graph_index, num_graphs):
    """Mean of scalar values in each graph; fail if a graph has no entries."""
    graph_index = graph_index.long()
    sums = values.new_zeros(num_graphs).index_add(0, graph_index, values)
    counts = values.new_zeros(num_graphs).index_add(0, graph_index, torch.ones_like(values))
    if bool((counts == 0).any()):
        raise ValueError("Every graph must contain at least one free joint.")
    return sums / counts


@lru_cache(maxsize=12)
def _numpy_q4_rule(order):
    if isinstance(order, bool) or not isinstance(order, int) or not 2 <= order <= 128:
        raise ValueError("Q4 integration order must be an integer from 2 through 128.")
    g, w = np.polynomial.legendre.leggauss(order)
    a, b = np.meshgrid(g, g, indexing="ij")
    sx, sy = np.array([-1., 1., 1., -1.]), np.array([-1., -1., 1., 1.])
    a, b = a.ravel()[:, None], b.ravel()[:, None]
    n = (1 + a * sx) * (1 + b * sy) / 4
    d1, d2 = sx * (1 + b * sy) / 4, sy * (1 + a * sx) / 4
    return n, d1, d2, np.outer(w, w).ravel()


def _q4_rule(order, reference):
    return tuple(torch.as_tensor(a, dtype=reference.dtype, device=reference.device) for a in _numpy_q4_rule(order))


def _panel_vectors(vertices, d1, d2):
    # Centering improves translation invariance of finite precision arithmetic.
    v = vertices - vertices.mean(dim=1, keepdim=True)
    return torch.linalg.cross(torch.einsum("qc,pcd->pqd", d1, v),
                              torch.einsum("qc,pcd->pqd", d2, v), dim=-1)


def _invalid_graph_mask(invalid, graph_ids, graph_count):
    count = torch.zeros(graph_count, dtype=torch.int64, device=graph_ids.device)
    count.index_add_(0, graph_ids, invalid.to(torch.int64))
    return count > 0


def physics_state(coords, batch, order=8):
    """Assemble forces and report geometric validity for a packed graph batch.

    ``coords`` is N×3 predicted coordinates, not normalized displacements.
    The returned ``valid_graph`` detects collapsed members/panels, reversed
    roof projection, Q4 folds (including corners) and nonfinite values. A bad
    graph is *not* dropped from the force residual or physics loss. Small
    denominator guards only avoid division by zero during early predictions.
    Nonfinite input remains nonfinite and should stop the optimizer.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (number_of_joints, 3).")
    xyz = coords.to(dtype=torch.float64)
    x0 = batch["x0"].to(device=xyz.device, dtype=xyz.dtype)
    edge = batch["edge_index"].to(xyz.device).long()
    material = batch["member_raw"].to(device=xyz.device, dtype=xyz.dtype)
    panel = batch["panel_raw"].to(device=xyz.device, dtype=xyz.dtype)
    node_batch = batch["node_batch"].to(xyz.device).long()
    n_graphs = int(batch["num_graphs"])
    n_nodes = len(xyz)
    if x0.shape != xyz.shape or edge.shape[0] != 2 or len(material) != edge.shape[1]:
        raise ValueError("Packed geometry/member arrays have inconsistent dimensions.")
    if bool((material[:, :3] <= 0).any()):
        raise ValueError("E, A and original L0 must be positive.")
    if bool((panel[:, 0] < 0).any()) or bool((panel[:, 1] > 0).any()):
        raise ValueError("This dataset requires dead pressure >=0 and upward wind pressure <=0.")
    tri = batch["tri_index"].to(xyz.device).long().reshape(-1, 3)
    quad = batch["quad_index"].to(xyz.device).long().reshape(-1, 4)
    tp = batch["tri_panel"].to(xyz.device).long()
    qp = batch["quad_panel"].to(xyz.device).long()
    if len(tp) != len(tri) or len(qp) != len(quad) or len(tp) + len(qp) != len(panel):
        raise ValueError("Every panel must occur exactly once as T3 or Q4; other polygons are unsupported.")
    if len(panel) and not torch.equal(torch.cat((tp, qp)).sort().values, torch.arange(len(panel), device=xyz.device)):
        raise ValueError("T3/Q4 panel indices must partition all panels without duplicates.")

    delta = xyz[edge[1]] - xyz[edge[0]]
    lengths = torch.linalg.vector_norm(delta, dim=-1)
    e, a, l0 = material[:, 0], material[:, 1], material[:, 2]
    minimum_length = 1e-12 * l0.clamp_min(1.)
    axial = e * a * (lengths - l0) / l0
    end_force = delta * (axial / lengths.clamp_min(minimum_length))[:, None]
    internal = torch.zeros_like(xyz).index_add(0, edge[0], -end_force).index_add(0, edge[1], end_force)
    invalid = _invalid_graph_mask(~torch.isfinite(xyz).all(dim=1), node_batch, n_graphs)
    invalid = invalid | _invalid_graph_mask((lengths <= minimum_length) | ~torch.isfinite(lengths), node_batch[edge[0]], n_graphs)
    pd = xyz.new_zeros(n_nodes)
    pw = xyz.new_zeros(n_nodes)
    surface_area = xyz.new_zeros(len(panel))
    projected_area = xyz.new_zeros(len(panel))

    if len(tri):
        v, vr = xyz[tri], x0[tri]
        av = torch.linalg.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1)
        ref_av = torch.linalg.cross(vr[:, 1] - vr[:, 0], vr[:, 2] - vr[:, 0], dim=-1)
        js = torch.linalg.vector_norm(av, dim=-1)
        jh = av[:, 2].abs()
        areas, projected = js / 2, jh / 2
        ds, ws = areas[:, None].expand(-1, 3) / 3, projected[:, None].expand(-1, 3) / 3
        pd = pd.index_add(0, tri.reshape(-1), (ds * panel[tp, 0, None]).reshape(-1))
        pw = pw.index_add(0, tri.reshape(-1), (ws * panel[tp, 1, None]).reshape(-1))
        surface_area = surface_area.index_add(0, tp, areas)
        projected_area = projected_area.index_add(0, tp, projected)
        bad = (av[:, 2] * torch.sign(ref_av[:, 2]) <= 1e-10 * js) | (js <= 0) | ~torch.isfinite(av).all(dim=1)
        invalid = invalid | _invalid_graph_mask(bad, node_batch[tri[:, 0]], n_graphs)

    if len(quad):
        v, vr = xyz[quad], x0[quad]
        shape, d1, d2, weights = _q4_rule(order, xyz)
        av = _panel_vectors(v, d1, d2)
        js = torch.linalg.vector_norm(av, dim=-1)
        jh = av[..., 2].abs()
        surface_weights = torch.einsum("qc,pq,q->pc", shape, js, weights)
        projected_weights = torch.einsum("qc,pq,q->pc", shape, jh, weights)
        pd = pd.index_add(0, quad.reshape(-1), (surface_weights * panel[qp, 0, None]).reshape(-1))
        pw = pw.index_add(0, quad.reshape(-1), (projected_weights * panel[qp, 1, None]).reshape(-1))
        surface_area = surface_area.index_add(0, qp, (js * weights).sum(dim=1))
        projected_area = projected_area.index_add(0, qp, (jh * weights).sum(dim=1))
        # A bilinear map's projected Jacobian is affine on its square. Check
        # the corners as well as Gauss points to catch folds near the edge.
        sx = xyz.new_tensor([-1., 1., 1., -1.])
        sy = xyz.new_tensor([-1., -1., 1., 1.])
        cd1 = sx[None, :] * (1 + sy[:, None] * sy[None, :]) / 4
        cd2 = sy[None, :] * (1 + sx[:, None] * sx[None, :]) / 4
        cav = _panel_vectors(v, cd1, cd2)
        all_av = torch.cat((av, cav), dim=1)
        all_js = torch.linalg.vector_norm(all_av, dim=-1)
        ref_sign = torch.sign(torch.linalg.cross(vr[:, 1] - vr[:, 0], vr[:, 3] - vr[:, 0], dim=-1)[:, 2])
        bad = (all_av[..., 2] * ref_sign[:, None] <= 1e-10 * all_js).any(dim=1)
        bad = bad | (~torch.isfinite(all_av).all(dim=2)).any(dim=1)
        vc = v - v.mean(dim=1, keepdim=True)
        normal = torch.linalg.cross(vc, torch.roll(vc, -1, dims=1), dim=-1).sum(dim=1)
        nlen = torch.linalg.vector_norm(normal, dim=1)
        unit_normal = normal / nlen.clamp_min(1e-30)[:, None]
        orient = (all_av * unit_normal[:, None, :]).sum(dim=-1)
        edges = torch.roll(v, -1, dims=1) - v
        turns = (torch.linalg.cross(edges, torch.roll(edges, -1, dims=1), dim=-1) * unit_normal[:, None, :]).sum(dim=-1)
        scale2 = vc.square().sum(dim=-1).max(dim=1).values
        bad = bad | (nlen <= 1e-10 * scale2) | (orient <= 1e-10 * all_js).any(dim=1)
        bad = bad | (turns < -1e-10 * scale2[:, None]).any(dim=1)
        bad = bad | (torch.linalg.vector_norm(edges, dim=-1) <= 1e-10 * scale2.sqrt()[:, None]).any(dim=1)
        invalid = invalid | _invalid_graph_mask(bad, node_batch[quad[:, 0]], n_graphs)

    external = torch.stack((torch.zeros_like(pd), torch.zeros_like(pd), -(pd + pw)), dim=-1)
    residual = internal - external
    invalid = invalid | _invalid_graph_mask(~torch.isfinite(residual).all(dim=1), node_batch, n_graphs)
    return {"internal": internal, "external": external, "residual": residual,
            "axial": axial, "lengths": lengths, "PD": pd, "PW": pw,
            "panel_surface_area": surface_area, "panel_projected_area": projected_area,
            "valid_graph": ~invalid}


def physics_loss(coords, batch, force_scale, order=8):
    """One dimensionless mean squared free-DOF residual per graph.

    Average this vector over graphs for the training loss. Supported nodes'
    reactions are deliberately excluded. ``force_scale`` is fitted once on
    the training partition; it is not recomputed on predictions or targets.
    """
    if not np.isfinite(float(force_scale)) or float(force_scale) <= 0:
        raise ValueError("force_scale must be finite and positive.")
    state = physics_state(coords, batch, order)
    free = batch["free_mask"].to(coords.device).bool()
    values = (state["residual"][free] / force_scale).square().mean(dim=1)
    return graph_mean(values, batch["node_batch"].to(coords.device)[free], int(batch["num_graphs"]))


def physics_loss_terms(coords, batch, force_scale, order=8):
    """Separate free equilibrium and supervised support-reaction errors.

    R_hat = f_internal(X_hat)-f_external(X_hat) at supported joint IDs.
    R_target is a fixed, verified FFT label. Subtracting R_hat from itself
    would be a tautology and is deliberately never used as a loss.
    Geometry is not clamped at supports in learned-support mode.
    """
    if not np.isfinite(float(force_scale)) or float(force_scale) <= 0:
        raise ValueError("force_scale must be finite and positive")
    if "reaction_target" not in batch:
        raise ValueError("Verified FFT support-reaction labels are required")
    target = batch["reaction_target"].to(device=coords.device, dtype=torch.float64).detach()
    free = batch["free_mask"].to(coords.device).bool()
    if target.shape != coords.shape or not bool(torch.isfinite(target).all()):
        raise ValueError("Reaction labels must be finite and aligned with joint coordinates")
    if bool((target[free] != 0).any()):
        raise ValueError("Free-node residuals must not be treated as support reactions")
    state = physics_state(coords, batch, order)
    graph = batch["node_batch"].to(coords.device)
    n = int(batch["num_graphs"])
    free_loss = graph_mean((state["residual"][free]/force_scale).square().mean(-1),graph[free],n)
    reaction_error = state["residual"][~free] - target[~free]
    reaction = graph_mean((reaction_error/force_scale).square().mean(-1),graph[~free],n)
    return {"free_equilibrium":free_loss,"reaction":reaction,"state":state}


def _read_rows(path, id_name):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return sorted(rows, key=lambda r: int(r[id_name]))


def load_case_for_verification(case_dir):
    """Read one accepted MATLAB case independently of the learning loader."""
    case = Path(case_dir)
    folder = case / "training"
    j = _read_rows(folder / "joint.csv", "joint_id")
    m = _read_rows(folder / "member.csv", "member_id")
    p = _read_rows(folder / "panel.csv", "panel_id")
    ids = {int(r["joint_id"]): i for i, r in enumerate(j)}
    def numeric(rows, names):
        return torch.tensor([[float(r[n]) for n in names] for r in rows], dtype=torch.float64)
    corners = [[ids[int(i)] for i in json.loads(r["corner_joint_ids"])] for r in p]
    if any(len(c) not in (3, 4) for c in corners):
        raise NotImplementedError("Target verification supports the dataset's T3/Q4 panels only.")
    tri_panel = [i for i, c in enumerate(corners) if len(c) == 3]
    quad_panel = [i for i, c in enumerate(corners) if len(c) == 4]
    batch = {"x0": numeric(j, ("x0_m", "y0_m", "z0_m")),
             "target_u": numeric(j, ("ux_ff_m", "uy_ff_m", "uz_ff_m")),
             "free_mask": torch.tensor([r["support"] == "0" for r in j]),
             "edge_index": torch.tensor([[ids[int(r["start_joint_id"])], ids[int(r["end_joint_id"])]] for r in m], dtype=torch.long).T,
             "member_raw": numeric(m, ("E_kPa", "A_m2", "L0_m", "EA_over_L0_kN_per_m")),
             "panel_raw": numeric(p, ("qD_kPa", "qW_kPa", "initial_area_m2")),
             "tri_index": torch.tensor([corners[i] for i in tri_panel], dtype=torch.long).reshape(-1, 3),
             "tri_panel": torch.tensor(tri_panel, dtype=torch.long),
             "quad_index": torch.tensor([corners[i] for i in quad_panel], dtype=torch.long).reshape(-1, 4),
             "quad_panel": torch.tensor(quad_panel, dtype=torch.long),
             "node_batch": torch.zeros(len(j), dtype=torch.long), "num_graphs": 1}
    return batch


def verify_fft_case(case_dir, order=8, comparison_atol_kN=5e-9, comparison_rtol=1e-9):
    """Reconstruct target forces and compare with saved MATLAB exports.

    The force comparison tolerance covers 15-significant-digit CSV export;
    it is not a training or FFT convergence tolerance. The actual residual
    is also tested against the original case threshold with a 5e-9 kN
    round-trip allowance. Order 16 independently rechecks integration.
    """
    case = Path(case_dir)
    b = load_case_for_verification(case)
    coords = b["x0"] + b["target_u"]
    with torch.no_grad():
        s = physics_state(coords, b, order)
        fine = physics_state(coords, b, 2 * order)
    j = _read_rows(case / "final_joint.csv", "joint_id")
    m = _read_rows(case / "final_member.csv", "member_id")
    summary = json.loads((case / "summary.json").read_text(encoding="utf-8-sig"))
    def expected(rows, fields):
        return np.array([[float(r[f]) for f in fields] for r in rows])
    comparisons = {
        "external": (s["external"].numpy(), expected(j, ("Fx_final_kN", "Fy_final_kN", "Fz_final_kN"))),
        "internal": (s["internal"].numpy(), expected(j, ("Fx_internal_kN", "Fy_internal_kN", "Fz_internal_kN"))),
        "axial": (s["axial"].numpy(), expected(m, ("axial_force_kN",)).ravel()),
    }
    errors = {k: float(np.max(np.abs(a - e))) for k, (a, e) in comparisons.items()}
    passes = {k: bool(np.allclose(a, e, atol=comparison_atol_kN, rtol=comparison_rtol)) for k, (a, e) in comparisons.items()}
    free = b["free_mask"]
    threshold = float(summary["force_acceptance_threshold_kN"])
    residual = float(s["residual"][free].abs().max())
    fine_residual = float(fine["residual"][free].abs().max())
    quadrature_error = float((s["external"] - fine["external"]).abs().max())
    passes.update(valid_geometry=bool(s["valid_graph"].all() and fine["valid_graph"].all()),
                  final_residual=residual <= threshold + comparison_atol_kN,
                  fine_residual=fine_residual <= threshold + comparison_atol_kN,
                  quadrature=quadrature_error <= .1 * threshold + comparison_atol_kN)
    return {"sample_id": summary["sample_id"], "passed": all(passes.values()), "checks": passes,
            "max_force_difference_kN": errors, "free_residual_kN": residual,
            "fine_free_residual_kN": fine_residual, "original_force_threshold_kN": threshold,
            "quadrature_difference_kN": quadrature_error, "n_joints": len(coords),
            "triangles": len(b["tri_index"]), "quadrilaterals": len(b["quad_index"]),
            "comparison_atol_kN": comparison_atol_kN, "comparison_rtol": comparison_rtol}
