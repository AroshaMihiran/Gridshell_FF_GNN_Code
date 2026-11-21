"""Held-out evaluation, physically scaled errors and transparent examples."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data import collate_graphs
from .models import FormFindingGNN, predict_coordinates
from .physics import physics_state
from .training import move_batch, select_device
from .experiment import save_json
from .checkpoint_selection import evaluation_checkpoint, checkpoint_status


def regression_metrics(target, prediction):
    target, prediction = np.asarray(target, float), np.asarray(prediction, float)
    if target.shape != prediction.shape or not target.size or not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require matching, finite, nonempty arrays")
    error = prediction - target
    ss = float(np.square(target-target.mean()).sum())
    return dict(rmse=float(np.sqrt(np.square(error).mean())), mae=float(np.abs(error).mean()),
                r2=float(1-np.square(error).sum()/ss) if ss > 0 else None, n=int(target.size))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reaction_diagnostics(initial, predicted_u, target_u, external, residual,
                         reaction_target, free_mask, force_scale, support_tolerance):
    """Evaluate learned supports without projecting them back onto restraints.

    R_pred is the support part of internal-external force, not another output
    head. R_target is the fixed verified FFT label and is never recalculated
    on the prediction. Reaction-derived global force balance is dependent on
    the free residual; it is not a second independent equilibrium proof.
    Moments are about the GLOBAL origin. Both predicted and prescribed
    support lever arms are reported because learned supports can drift.
    """
    arrays = [np.asarray(a, dtype=float) for a in
              (initial, predicted_u, target_u, external, residual, reaction_target)]
    x0, pred_u, truth_u, force, res, label = arrays
    free = np.asarray(free_mask, dtype=bool)
    if x0.ndim != 2 or x0.shape[1] != 3 or any(a.shape != x0.shape for a in arrays):
        raise ValueError("Reaction diagnostics require matching N-by-3 arrays")
    if free.shape != (len(x0),) or not (~free).any():
        raise ValueError("Reaction diagnostics require a supported-joint mask")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Reaction diagnostics cannot silently discard nonfinite values")
    if not np.isfinite(force_scale) or force_scale <= 0 or not np.isfinite(support_tolerance) or support_tolerance <= 0:
        raise ValueError("Reaction force scale and support tolerance must be positive")
    if np.any(label[free] != 0):
        raise ValueError("Reaction targets must be exactly zero at free joints")
    supported = ~free
    coords = x0 + pred_u
    reaction = np.zeros_like(res)
    reaction[supported] = res[supported]
    difference = reaction[supported] - label[supported]
    support_difference = pred_u[supported] - truth_u[supported]
    support_norm = np.linalg.norm(support_difference, axis=1)
    row = dict(supported_joints=int(supported.sum()),
               support_rmse_m=float(np.sqrt(np.square(support_difference).mean())),
               support_error_m=float(np.abs(support_difference).max()),
               support_max_joint_error_m=float(support_norm.max()),
               support_constraints_satisfied=bool(support_norm.max() <= support_tolerance),
               support_displacement_tolerance_m=float(support_tolerance),
               all_joint_rmse_m=float(np.sqrt(np.square(pred_u-truth_u).mean())),
               reaction_rmse_kN=float(np.sqrt(np.square(difference).mean())),
               reaction_max_component_error_kN=float(np.abs(difference).max()),
               reaction_max_joint_error_kN=float(np.linalg.norm(difference, axis=1).max()),
               reaction_loss=float(np.square(difference/force_scale).mean()))
    for i, axis in enumerate("xyz"):
        row[f"reaction_rmse_{axis}_kN"] = float(np.sqrt(np.square(difference[:, i]).mean()))
        row[f"support_rmse_{axis}_m"] = float(np.sqrt(np.square(support_difference[:, i]).mean()))

    def record_vector(prefix, vector, unit):
        for i, axis in enumerate("xyz"):
            row[f"{prefix}_{axis}_{unit}"] = float(vector[i])
        row[f"{prefix}_norm_{unit}"] = float(np.linalg.norm(vector))

    for name, value in (("computed_reaction", reaction), ("label_reaction", label)):
        imbalance = (force+value).sum(axis=0)
        moment_predicted = np.cross(coords, force+value).sum(axis=0)
        # External pressure forces act on predicted nodes; reactions are also
        # evaluated at the original prescribed support locations for this
        # separate physical-boundary diagnostic.
        moment_prescribed = (np.cross(coords, force).sum(axis=0)
                             + np.cross(x0[supported], value[supported]).sum(axis=0))
        record_vector(f"{name}_balance_force", imbalance, "kN")
        record_vector(f"{name}_balance_predicted_arms_moment", moment_predicted, "kNm")
        record_vector(f"{name}_balance_prescribed_support_arms_moment", moment_prescribed, "kNm")
        scale = float(np.linalg.norm(force, axis=1).sum()+np.linalg.norm(value[supported], axis=1).sum())
        row[f"{name}_balance_relative_force"] = float(np.linalg.norm(imbalance)/max(scale, force_scale))
    return row, reaction


def evaluate_model(checkpoint, dataset, cfg, normalizer, run_fingerprint, output_dir):
    """Never fit/tune on test data. Invalid predictions stay in the statistics."""
    if not len(dataset):
        raise ValueError("Evaluation requires a nonempty held-out partition")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck["fingerprint"] != run_fingerprint:
        raise ValueError("Evaluation checkpoint does not match the frozen run")
    mid, seed = ck["model_id"], ck["seed"]
    status = checkpoint_status(ck)
    if status["diagnostic_only"] and not cfg.get("evaluate_incomplete_m4", False):
        raise ValueError("M4 comparison requires a checkpoint trained at full physics weight; enable evaluate_incomplete_m4 for explicitly labelled diagnostic plots")
    plot_label = status["model_label"]
    device = select_device(cfg)
    model = FormFindingGNN(mid, cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    pred_dir = folder / "predictions"
    if cfg["save_node_predictions"]:
        pred_dir.mkdir(exist_ok=True)
    reaction_dir = folder / "support_reactions"
    reaction_dir.mkdir(exist_ok=True)
    rows, predictions, truths, true_coordinates, predicted_coordinates = [], [], [], [], []
    reaction_predictions, reaction_truths = [], []
    started = time.perf_counter()
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_graphs)
    warm = False
    with torch.no_grad():
        for cpu_batch in loader:
            b = move_batch(cpu_batch, device)
            if not warm:
                for _ in range(3):
                    model(b)
                _sync(device)
                warm = True
            tick = time.perf_counter()
            coords = predict_coordinates(model, b, normalizer.length_scale)
            _sync(device)
            prediction_s = time.perf_counter()-tick
            tick = time.perf_counter()
            state = physics_state(coords, b, order=cfg["evaluation_integration_order"])
            _sync(device)
            physics_s = time.perf_counter()-tick
            tick = time.perf_counter()
            target_coords = b["x0"] + b["target_u"]
            target_state = physics_state(target_coords, b, order=cfg["evaluation_integration_order"])
            _sync(device)
            target_physics_s = time.perf_counter()-tick
            if not bool(target_state["valid_graph"].all()):
                raise ValueError("An accepted FFT target fails the evaluation geometry check; inspect the data/operator before comparing models")
            u = (coords-b["x0"]).cpu().numpy()
            for g,sample in enumerate(cpu_batch["samples"]):
                start, end = map(int, cpu_batch["ptr"][g:g+2])
                em0, em1 = map(int, cpu_batch["member_ptr"][g:g+2])
                free = sample["free_mask"].numpy()
                t, p = sample["target_u"].numpy(), u[start:end]
                xyz0 = sample["x0"].numpy()
                residual_all = state["residual"][start:end].cpu().numpy()
                external = state["external"][start:end].cpu().numpy()
                residual = residual_all[free]
                reaction_target = sample["reaction_target"].cpu().numpy()
                truth_residual = target_state["residual"][start:end].cpu().numpy()[free]
                n_pred = state["axial"][em0:em1].cpu().numpy()
                n_true = target_state["axial"][em0:em1].cpu().numpy()
                finite = bool(np.isfinite(p).all() and np.isfinite(residual_all).all())
                valid = bool(state["valid_graph"][g]) and finite
                if not finite:
                    raise FloatingPointError(f"Nonfinite prediction at {sample['sample_id']}; evaluation stopped rather than dropping it")
                d = p[free]-t[free]
                row = dict(sample_id=sample["sample_id"], base_layout_id=sample["base_layout_id"],
                    family=sample.get("family", "unknown"), split=sample["split"], model=mid, seed=seed,
                    free_joints=int(free.sum()), valid_geometry=valid,
                    rmse_m=float(np.sqrt(np.square(d).mean())),
                    max_joint_error_m=float(np.linalg.norm(d, axis=1).max()),
                    physics_loss=float(np.square(residual/normalizer.force_scale).mean()),
                    residual_rms_kN=float(np.sqrt(np.square(residual).mean())),
                    residual_max_kN=float(np.abs(residual).max()),
                    target_residual_rms_kN=float(np.sqrt(np.square(truth_residual).mean())),
                    axial_rmse_kN=float(np.sqrt(np.square(n_pred-n_true).mean())),
                    inference_ms_amortized=prediction_s*1000/b["num_graphs"],
                    physics_ms_amortized=physics_s*1000/b["num_graphs"],
                    target_physics_ms_amortized=target_physics_s*1000/b["num_graphs"])
                for c,component in enumerate("xyz"):
                    metric = regression_metrics(t[free,c], p[free,c])
                    row[f"rmse_{component}_m"], row[f"r2_{component}"] = metric["rmse"], metric["r2"]
                support_metrics, reaction_pred = reaction_diagnostics(
                    xyz0, p, t, external, residual_all, reaction_target, free,
                    normalizer.force_scale, cfg["support_displacement_tolerance_m"])
                row.update(support_metrics)
                row["combined_physics_loss"] = (cfg["free_residual_weight"]*row["physics_loss"]
                                                 + cfg["reaction_weight"]*row["reaction_loss"])
                rows.append(row)
                truths.append(t[free]); predictions.append(p[free])
                true_coordinates.append((xyz0+t)[free]); predicted_coordinates.append((xyz0+p)[free])
                reaction_truths.append(reaction_target[~free]); reaction_predictions.append(reaction_pred[~free])
                reaction_table = {"joint_id":sample["joint_ids"].numpy()[~free]}
                for c,axis in enumerate("xyz"):
                    reaction_table[f"{axis}_prescribed_m"] = xyz0[~free,c]
                    reaction_table[f"{axis}_predicted_m"] = (xyz0+p)[~free,c]
                    reaction_table[f"u{axis}_target_m"] = t[~free,c]
                    reaction_table[f"u{axis}_predicted_m"] = p[~free,c]
                    reaction_table[f"R{axis}_label_kN"] = reaction_target[~free,c]
                    reaction_table[f"R{axis}_predicted_kN"] = reaction_pred[~free,c]
                    reaction_table[f"R{axis}_error_kN"] = reaction_pred[~free,c]-reaction_target[~free,c]
                pd.DataFrame(reaction_table).to_csv(reaction_dir/f"{sample['sample_id']}.csv",index=False)
                if cfg["save_node_predictions"]:
                    np.savez_compressed(pred_dir/f"{sample['sample_id']}.npz", joint_ids=sample["joint_ids"].numpy(),
                        initial=xyz0, target_u=t, predicted_u=p, free_mask=free, edge_index=sample["edge_index"].numpy(),
                        residual_kN=residual_all, external_pred_kN=external,
                        reaction_pred_kN=reaction_pred, reaction_target_kN=reaction_target,
                        axial_pred_kN=n_pred, axial_target_kN=n_true)
    metrics = pd.DataFrame(rows)
    for key, value in status.items():
        metrics[key] = value
    metrics.to_csv(folder/"per_case_metrics.csv", index=False)
    target, pred = np.concatenate(truths), np.concatenate(predictions)
    tc, pc = np.concatenate(true_coordinates), np.concatenate(predicted_coordinates)
    reaction_target, reaction_prediction = np.concatenate(reaction_truths), np.concatenate(reaction_predictions)
    summary = dict(model=mid, seed=seed, checkpoint_epoch=ck["epoch"], cases=len(rows),
                   invalid_geometry_count=int((~metrics.valid_geometry).sum()),
                   support_constraint_violation_count=int((~metrics.support_constraints_satisfied).sum()),
                   support_displacement_tolerance_m=cfg["support_displacement_tolerance_m"],
                   maximum_support_displacement_error_m=float(metrics.support_max_joint_error_m.max()),
                   mean_support_rmse_m=float(metrics.support_rmse_m.mean()),
                   mean_reaction_rmse_kN=float(metrics.reaction_rmse_kN.mean()),
                   macro_case_rmse_m=float(metrics.rmse_m.mean()),
                   median_case_rmse_m=float(metrics.rmse_m.median()),
                   invocation_wall_s=time.perf_counter()-started,
                   physics_weight_at_checkpoint=float(ck["history"][-1]["physics_weight"]),
                   displacement_metrics={c:regression_metrics(target[:,i],pred[:,i]) for i,c in enumerate("xyz")},
                   coordinate_metrics={c:regression_metrics(tc[:,i],pc[:,i]) for i,c in enumerate("xyz")},
                   reaction_metrics={c:regression_metrics(reaction_target[:,i],reaction_prediction[:,i]) for i,c in enumerate("xyz")},
                   metric_note="Displacement/coordinate R² is pooled over free joints; reaction R² is pooled over supports. Support displacement errors remain explicit and predictions are not clamped. Reaction labels come from verified FFT exports, not the predicted geometry. Global moment diagnostics use the global origin and distinguish predicted support locations from prescribed initial locations. Computed-reaction balance depends on the free residual and is not an independent equilibrium proof. Timing is batch-amortized; file I/O excluded from inference.")
    summary.update(status)
    summary["checkpoint_path"] = str(Path(checkpoint).resolve())
    save_json(folder/"metrics.json", summary)
    parity_plot(target, pred, folder/"displacement_parity.png", title=f"{plot_label}: held-out displacements", labels=("ux", "uy", "uz"))
    parity_plot(tc, pc, folder/"coordinate_parity.png", title=f"{plot_label}: held-out coordinates", labels=("x", "y", "z"))
    parity_plot(reaction_target, reaction_prediction, folder/"reaction_parity.png",
                title=f"{plot_label}: supported-joint reactions", labels=("Rx", "Ry", "Rz"), unit="kN")
    return metrics, summary


def parity_plot(target, predicted, path, title, labels, unit="m"):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    # Sampling is only for drawing; R²/RMSE use every free joint.
    ids = np.random.default_rng(0).choice(len(target), min(18000, len(target)), replace=False)
    for c, ax in enumerate(axes):
        metric = regression_metrics(target[:,c], predicted[:,c])
        ax.scatter(target[ids,c], predicted[ids,c], s=3, alpha=.25, color="#315a7d", rasterized=True)
        lo = float(min(target[:,c].min(), predicted[:,c].min())); hi = float(max(target[:,c].max(), predicted[:,c].max()))
        if hi == lo:
            lo, hi = lo-.5, hi+.5
        ax.plot([lo,hi], [lo,hi], color="#7b7b7b", linewidth=1)
        r2 = "undefined" if metric["r2"] is None else f"{metric['r2']:.4f}"
        ax.set(title=f"{labels[c]} · R²={r2}", xlabel=f"FFT target ({unit})", ylabel=f"GNN prediction ({unit})")
        ax.grid(alpha=.2)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def comparison_plots(metrics, output_dir):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in ("M1", "M2", "M3", "M4") if m in set(metrics.model)]
    for column, ylabel, name, log in (
        ("rmse_m", "Free-joint coordinate RMSE per case (m)", "rmse_boxplot", False),
        ("physics_loss", "Unweighted normalized free-force residual MSE", "physics_loss_boxplot", True),
        ("residual_rms_kN", "Free-force residual RMS per case (kN)", "force_residual_boxplot", True),
        ("axial_rmse_kN", "Member axial-force RMSE per case (kN)", "axial_force_boxplot", True),
        ("support_rmse_m", "Supported-joint displacement RMSE per case (m)", "support_displacement_boxplot", True),
        ("reaction_rmse_kN", "Supported-joint reaction RMSE per case (kN)", "reaction_rmse_boxplot", True),
        ("reaction_loss", "Normalized support-reaction matching MSE", "reaction_loss_boxplot", True),
        ("combined_physics_loss", "Weighted free-residual + reaction MSE (before λ)", "combined_physics_boxplot", True)):
        if column not in metrics:
            continue
        fig, ax = plt.subplots(figsize=(7,4.2), constrained_layout=True)
        values = [metrics.loc[metrics.model==m,column].to_numpy() for m in models]
        labels = ["\n".join(metrics.loc[metrics.model==m,"model_label"].unique()) if "model_label" in metrics else m for m in models]
        ax.boxplot(values, tick_labels=labels, showfliers=True, flierprops=dict(markersize=2, alpha=.2),
                   medianprops=dict(color="#b75b36"))
        if log and all((v>0).all() for v in values):
            ax.set_yscale("log")
        ax.set(ylabel=ylabel, xlabel="Model", title="Held-out cases; invalid finite predictions retained")
        ax.grid(axis="y", alpha=.2)
        fig.savefig(output_dir/f"{name}.png", dpi=180)
        plt.close(fig)


def rank_progression(metrics, top_k=6, minimum_relative_gap=0.):
    """Return naturally occurring strict M4<M3<M2<M1 pairs, separately by seed.

    Result indices are (seed, sample_id) and (seed, base_layout_id). Seed means
    cannot be used to claim that an individual plotted seed has this ordering.
    Every variant must have all four predictions before its layout is ranked.
    Invalid geometry or support-constraint violations remain in aggregate
    metrics; only the illustrative ranking excludes them. This descriptive
    search does not tune the models.
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not np.isfinite(minimum_relative_gap) or not 0 <= minimum_relative_gap < 1:
        raise ValueError("minimum_relative_gap must lie in [0, 1)")
    models = ["M1", "M2", "M3", "M4"]
    if "diagnostic_only" in metrics:
        metrics = metrics.loc[~metrics.diagnostic_only.astype(bool)]
    if not set(models).issubset(set(metrics.model)):
        return pd.DataFrame(), pd.DataFrame(), {"available":False,"reason":"All four completed models are required; incomplete M4 diagnostics are shown separately"}
    work = metrics.loc[metrics.model.isin(models)].copy()
    if "support_constraints_satisfied" not in work:
        raise ValueError("Illustrative ranking requires explicit support-constraint checks")
    keys = ["seed", "sample_id", "model"]
    if work.duplicated(keys).any():
        raise ValueError("Ranking requires one prediction per seed, sample and model")
    if not np.isfinite(work.rmse_m).all() or (work.rmse_m < 0).any():
        raise ValueError("Ranking requires finite, nonnegative RMSE values")
    if (work.groupby(["seed", "sample_id"]).base_layout_id.nunique() != 1).any():
        raise ValueError("Models disagree on a sample's base layout")
    wide_all = work.pivot(index=["seed", "sample_id"], columns="model", values="rmse_m").reindex(columns=models)
    complete = wide_all.notna().all(axis=1)
    wide = wide_all.loc[complete]
    geometry_valid = work.groupby(["seed", "sample_id"]).valid_geometry.all()
    supports_valid = work.groupby(["seed", "sample_id"]).support_constraints_satisfied.all()
    valid = geometry_valid & supports_valid
    mapping = work.drop_duplicates(["seed", "sample_id"]).set_index(["seed", "sample_id"]).base_layout_id
    eligible = wide.loc[valid.reindex(wide.index).fillna(False)].copy()
    def keep(frame):
        mask = np.ones(len(frame), bool)
        for a,b in zip(models[:-1],models[1:]):
            mask &= frame[b].to_numpy() < frame[a].to_numpy() * (1-minimum_relative_gap)
        out = frame.loc[mask].copy()
        out["relative_improvement_M4_over_M1"] = 1-out.M4/out.M1.clip(lower=1e-15)
        return out.sort_values("relative_improvement_M4_over_M1", ascending=False)
    cases = keep(eligible)
    # Layout means use all cases, including nonmonotonic ones. Incomplete or
    # invalid variants disqualify the whole layout from illustrative ranking.
    status = pd.DataFrame({"complete": complete, "valid": valid, "base_layout_id": mapping})
    layout_status = status.reset_index().groupby(["seed", "base_layout_id"])[["complete", "valid"]].all()
    all_layout = wide.assign(base_layout_id=mapping).reset_index().groupby(["seed", "base_layout_id"])[models].mean()
    layout_ok = layout_status.reindex(all_layout.index).fillna(False).all(axis=1)
    layouts = keep(all_layout.loc[layout_ok])
    report = dict(available=True, paired_cases=len(wide), all_models_valid_cases=len(eligible),
                  all_models_geometry_valid_cases=int(geometry_valid.reindex(wide.index).fillna(False).sum()),
                  all_models_support_constraints_satisfied_cases=int(supports_valid.reindex(wide.index).fillna(False).sum()),
                  strict_order_cases=len(cases), compared_layouts=len(all_layout), strict_order_layouts=len(layouts),
                  complete_valid_layouts=int(layout_ok.sum()),
                  unmatched_seed_cases=int((~complete).sum()),
                  minimum_relative_gap=minimum_relative_gap,
                  note="Descriptive held-out examples ranked separately by seed; model order is not guaranteed. Illustrations require valid geometry and the declared support-displacement tolerance in every model. Layout means include all variants. Full metrics, including invalid geometry and displaced supports, remain exported.")
    return cases.head(top_k), layouts.head(top_k), report


def plot_ranked_cases(ranking, run_dir, seed, split="test", top_k=3, minimum_relative_gap=0., support_tolerance=1e-3):
    """Same target, scale, camera and color range for M1--M4."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    run_dir = Path(run_dir)
    if not isinstance(ranking.index, pd.MultiIndex) or "seed" not in ranking.index.names:
        raise ValueError("Case ranking must identify the individual seed being plotted")
    if seed not in ranking.index.get_level_values("seed"):
        return []
    selected = ranking.xs(seed, level="seed")
    written = []
    for sid in selected.index[:top_k]:
        paths = [run_dir/f"{m}_seed{seed}"/f"evaluation_{split}"/"predictions"/f"{sid}.npz" for m in ("M1","M2","M3","M4")]
        if not all(p.is_file() for p in paths):
            continue
        records = []
        for path in paths:
            with np.load(path) as archive:
                records.append({k: archive[k] for k in archive.files})
        for record in records[1:]:
            for key in ("joint_ids", "initial", "target_u", "free_mask", "edge_index"):
                if not np.array_equal(record[key], records[0][key]):
                    raise ValueError(f"Saved predictions do not share an identical target/graph: {sid}, {key}")
        for record in records:
            supported = ~record["free_mask"]
            if supported.any() and np.linalg.norm(record["predicted_u"][supported]-record["target_u"][supported],axis=1).max() > support_tolerance:
                raise ValueError(f"Saved predictions exceed the support displacement tolerance for {sid}")
        rmse = np.array([np.sqrt(np.square(r["predicted_u"][r["free_mask"]]-r["target_u"][r["free_mask"]]).mean()) for r in records])
        if not np.isfinite(rmse).all() or not np.all(rmse[1:] < rmse[:-1] * (1-minimum_relative_gap)):
            raise ValueError(f"Saved seed {seed} predictions do not have the claimed strict ordering for {sid}")
        target = records[0]["initial"] + records[0]["target_u"]
        errors = [np.linalg.norm(r["predicted_u"]-r["target_u"],axis=1) for r in records]
        v_max = max(float(e.max()) for e in errors) or 1.
        coords = [r["initial"]+r["predicted_u"] for r in records]
        allcoords = np.vstack([target,*coords]); lower, upper = allcoords.min(0), allcoords.max(0)
        widths = np.maximum(upper-lower, .05)
        fig = plt.figure(figsize=(15,4), constrained_layout=True)
        axes=[]
        for i,(r,x,err,model_id) in enumerate(zip(records,coords,errors,("M1","M2","M3","M4"))):
            ax = fig.add_subplot(1,4,i+1,projection="3d"); axes.append(ax)
            for u,v in r["edge_index"].T:
                ax.plot(*target[[u,v]].T, color="#aaa",alpha=.4,linewidth=.45)
                ax.plot(*x[[u,v]].T,color="#315a7d",linewidth=.65)
            ax.scatter(*x.T,c=err,cmap="magma",vmin=0,vmax=v_max,s=7)
            ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),zlim=(lower[2],upper[2]),
                   xlabel="x (m)",ylabel="y (m)",zlabel="z (m)",title=f"{model_id} · RMSE {rmse[i]:.3g} m")
            ax.set_box_aspect(widths); ax.view_init(25,-55)
        fig.colorbar(ScalarMappable(norm=Normalize(0,v_max),cmap="magma"),ax=axes,shrink=.65,label="Joint position error (m)")
        fig.suptitle(f"{sid} · FFT target in grey · seed {seed}")
        destination = run_dir/f"comparison_{sid}_seed{seed}.png"
        fig.savefig(destination,dpi=180)
        written.append(str(destination))
        plt.close(fig)
    return written


def evaluate_registry(registry, datasets, cfg, normalizer, run_dir, run_fingerprint):
    frames, summaries = [], []
    split = cfg["evaluation_split"]
    for item in registry:
        checkpoint = evaluation_checkpoint(item, cfg.get("evaluate_incomplete_m4", False))
        if checkpoint is None:
            print(f"Skipping {item['model_id']}: no eligible checkpoint; see training_summary.json")
            continue
        folder = Path(item.get("evaluation_directory", Path(item["directory"])/f"evaluation_{split}"))
        frame, summary = evaluate_model(checkpoint, datasets[split], cfg, normalizer, run_fingerprint, folder)
        item.update(evaluation_directory=str(folder), evaluation_checkpoint=str(checkpoint),
                    physics_status=summary.get("physics_status", "full" if item["model_id"] == "M4" else "not_applicable"),
                    checkpoint_epoch=summary.get("checkpoint_epoch"))
        if summary.get("diagnostic_only"):
            print(f"{summary['model_label']}: plotting epoch {summary['checkpoint_epoch']}; this is not a completed physics-trained M4 result.", flush=True)
        frames.append(frame); summaries.append(summary)
    if not frames:
        raise ValueError("No eligible model checkpoint is available for evaluation")
    metrics = pd.concat(frames, ignore_index=True)
    save_json(Path(run_dir)/"evaluation_registry.json", registry)
    metrics.to_csv(Path(run_dir)/f"all_models_{split}_metrics.csv",index=False)
    save_json(Path(run_dir)/f"all_models_{split}_summary.json",summaries)
    comparison_plots(metrics,run_dir)
    cases,layouts,report=rank_progression(metrics,cfg["comparison_top_k"],cfg["comparison_min_relative_gap"])
    cases.to_csv(Path(run_dir)/"ordered_accuracy_cases.csv")
    layouts.to_csv(Path(run_dir)/"ordered_accuracy_layouts.csv")
    save_json(Path(run_dir)/"ordered_accuracy_report.json",report)
    if len(cases) and cfg["save_node_predictions"]:
        for seed in cfg["seeds"]:
            plot_ranked_cases(cases,run_dir,seed,split,top_k=min(3,cfg["comparison_top_k"]),minimum_relative_gap=cfg["comparison_min_relative_gap"],support_tolerance=cfg["support_displacement_tolerance_m"])
    metrics.groupby(["model","seed","family"]).agg(cases=("sample_id","count"),mean_rmse_m=("rmse_m","mean"),
                mean_physics_loss=("physics_loss","mean"),valid_fraction=("valid_geometry","mean")).to_csv(Path(run_dir)/"metrics_by_family.csv")
    print(report)
    return metrics, summaries, cases, layouts
