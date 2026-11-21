"""Sequential model training, live plots and epoch-boundary resume."""
import math
import os
import random
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .data import collate_graphs
from .models import FormFindingGNN, augment_lpe, segment_sum
from .physics import physics_loss_terms
from .schedule import PhysicsSchedule
from .experiment import atomic_torch_save, save_json


def select_device(cfg):
    requested = cfg["device"]
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this notebook kernel; select cpu or the Pytorch kernel")
    return torch.device(requested)


def seed_everything(seed, cfg):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(cfg["deterministic"])
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False


def move_batch(batch, device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch.items()}


def data_loss_terms(normalized_u, batch, length_scale, component_weights, support_weight=1.):
    """Learn free displacements and zero supported displacements separately.

    Each group has its own per-graph mean, so a large support count cannot
    dilute the free-node accuracy. No prediction is clamped in this loss.
    """
    squared = (normalized_u.double() - batch["target_u"] / length_scale).square()
    groups={}
    for name,mask in (("free",batch["free_mask"]),("support",~batch["free_mask"])):
        graph=batch["node_batch"][mask]
        counts=torch.bincount(graph,minlength=batch["num_graphs"])
        if bool((counts==0).any()):
            raise ValueError("Every graph must contain free and supported joints")
        groups[name]=segment_sum(squared[mask],graph,batch["num_graphs"])/counts[:,None]
    weights=squared.new_tensor(component_weights)
    lf=(groups["free"]*weights).mean(-1)
    ls=(groups["support"]*weights).mean(-1)
    return {"total":lf+support_weight*ls,"free":lf,"support":ls,
            "free_components":groups["free"],"support_components":groups["support"]}


def data_loss(normalized_u, batch, length_scale, component_weights, support_weight=1.):
    terms=data_loss_terms(normalized_u,batch,length_scale,component_weights,support_weight)
    return terms["total"],terms["free_components"]


def _rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0], n[1].tolist(), n[2], n[3], n[4]],
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])


def _metric_text(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.6g}"


def format_epoch_line(model_id, seed, row, total_epochs):
    """Format recorded losses; unevaluated physics is explicitly shown as n/a."""
    def pair(name, key):
        return (f"{name} train={_metric_text(row.get('train_' + key))} "
                f"val={_metric_text(row.get('validation_' + key))}")

    fields = [f"[{model_id} seed={seed}] epoch {row['epoch']}/{total_epochs}",
              pair("data", "data"), pair("objective", "objective")]
    if "train_support_data" in row:
        fields.extend([pair("free-data", "free_data"), pair("support-data", "support_data")])
    physics_key = "free_physics" if "train_free_physics" in row else "physics"
    fields.append(pair("free-physics", physics_key))
    if "train_reaction" in row:
        fields.append(pair("reaction", "reaction"))
    fields.extend([f"lambda={_metric_text(row.get('physics_weight'))}",
                   f"lr={_metric_text(row.get('learning_rate'))}",
                   f"epoch_s={_metric_text(row.get('epoch_s'))}"])
    return " | ".join(fields)


def report_epoch(model_id, seed, row, total_epochs, folder, log_every=1, force=False):
    """Append every completed epoch and flush selected progress lines to stdout."""
    line = format_epoch_line(model_id, seed, row, total_epochs)
    with (Path(folder) / "epochs.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    if force or row["epoch"] % max(1, int(log_every)) == 0 or row["epoch"] == total_epochs:
        print(line, flush=True)
    return line


def _loss_scale(ax):
    """Keep exact zero errors visible; log only strictly positive finite losses."""
    values = np.concatenate([np.asarray(line.get_ydata(), dtype=float).ravel() for line in ax.lines])
    finite = values[np.isfinite(values)]
    if finite.size and np.all(finite > 0):
        ax.set_yscale("log")
    else:
        ax.set_yscale("linear")
        if not finite.size:
            ax.text(.5, .5, "Not evaluated yet", transform=ax.transAxes,
                    ha="center", va="center", color="#dddddd", fontsize=9)


def make_training_figure(history, model_id, seed):
    """Create a local dark-styled figure without changing Matplotlib defaults."""
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    support_terms = "train_support_data" in h
    rows = 3 if support_terms else 2
    fig, axs = plt.subplots(rows, 3, figsize=(13, 10 if support_terms else 7),
                            constrained_layout=True, facecolor="black")
    train_color, val_color = "#56c8ff", "#ffb454"

    def pair(ax, key, title, ylabel, root=False):
        for prefix, color, label in (("train", train_color, "Train"),
                                     ("validation", val_color, "Validation")):
            values = np.sqrt(h[f"{prefix}_{key}"]) if root else h[f"{prefix}_{key}"]
            measured = np.isfinite(values)
            if not measured.any():
                label += " (not evaluated)"
            ax.plot(h.epoch[measured], values[measured], color=color, label=label,
                    marker="o" if measured.sum() < len(values) else None, markersize=3)
        ax.set(title=title, ylabel=ylabel)
        _loss_scale(ax)

    pair(axs[0, 0], "data", "Supervised data loss", "Dimensionless weighted MSE")
    for c, color in zip("xyz", (train_color, val_color, "#8bd17c")):
        axs[0, 1].plot(h.epoch, np.sqrt(h[f"validation_{c}"]), color=color, label=c)
    axs[0, 1].set(title="Validation displacement error", ylabel="RMSE / training length scale")
    _loss_scale(axs[0, 1])
    if support_terms:
        pair(axs[0, 2], "support_data", "Supported-joint displacement error",
             "Weighted RMSE / training length scale", root=True)
        pair(axs[1, 0], "free_physics", "Free-joint force balance", "Normalized residual MSE")
        pair(axs[1, 1], "reaction", "Support reaction matching", "Normalized reaction error MSE")
        pair(axs[1, 2], "objective", "Combined objective", "Data + weighted physics")
    else:
        pair(axs[0, 2], "physics", "Free-joint force balance", "Normalized residual MSE")
    axs[-1, 0].plot(h.epoch, h.physics_weight, color="#cba6f7")
    axs[-1, 0].set(title="Physics coefficient", ylabel="λ")
    axs[-1, 1].plot(h.epoch, h.learning_rate, color=train_color)
    axs[-1, 1].set(title="Learning rate", ylabel="AdamW learning rate")
    _loss_scale(axs[-1, 1])
    axs[-1, 2].plot(h.epoch, h.epoch_s, color="#8bd17c")
    axs[-1, 2].set(title="Elapsed time per epoch", ylabel="Seconds")
    for ax in axs.flat:
        ax.set_facecolor("black")
        ax.set_xlabel("Epoch")
        ax.title.set_color("#eeeeee")
        ax.xaxis.label.set_color("#dddddd")
        ax.yaxis.label.set_color("#dddddd")
        ax.xaxis.get_offset_text().set_color("#dddddd")
        ax.yaxis.get_offset_text().set_color("#dddddd")
        ax.tick_params(colors="#dddddd", which="both")
        for spine in ax.spines.values():
            spine.set_color("#777777")
        ax.grid(color="#444444", alpha=.6, linewidth=.6)
        if ax.get_legend_handles_labels()[0]:
            legend = ax.legend(fontsize=8, facecolor="black", edgecolor="#777777", framealpha=1.)
            for text in legend.get_texts():
                text.set_color("#eeeeee")
    fig.suptitle(f"{model_id} · seed {seed}", color="#eeeeee")
    return fig


class LiveTraining:
    def __init__(self):
        self.handle = None

    def update(self, history, model_id, seed, path=None):
        import matplotlib.pyplot as plt
        from IPython.display import display
        if not history:
            return
        fig = make_training_figure(history, model_id, seed)
        if path:
            fig.savefig(path, dpi=150, facecolor="black", edgecolor="none", bbox_inches="tight")
        if self.handle is None:
            self.handle = display(fig, display_id=True)
        else:
            self.handle.update(fig)
        plt.close(fig)


def _epoch(model, dataset, optimizer, cfg, normalizer, device, seed, epoch, physics_weight, training, compute_physics):
    generator = torch.Generator().manual_seed(seed + epoch * 104729)
    # num_workers=0 avoids Windows notebook process-spawn/cell-pickling issues.
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=training,
                        collate_fn=collate_graphs, num_workers=0, generator=generator)
    model.train(training)
    total = dict(data=0., free_data=0., support_data=0., physics=0., free_physics=0., reaction=0.,
                 objective=0., components=np.zeros(3), n=0, physics_n=0)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for cpu_batch in loader:
            b = move_batch(cpu_batch, device)
            if training and cfg["lpe_sign_augmentation"]:
                b = augment_lpe(b, cfg["lpe_dim"], normalizer)
            if training:
                optimizer.zero_grad(set_to_none=True)
            u = model(b)
            dterms=data_loss_terms(u,b,normalizer.length_scale,cfg["component_weights"],cfg["support_displacement_weight"])
            ld = dterms["total"].mean()
            components=dterms["free_components"]
            lp = ld.new_zeros(())
            lf=ld.new_zeros(()); lr=ld.new_zeros(())
            if compute_physics:
                coords = b["x0"] + u.double() * normalizer.length_scale
                pterms=physics_loss_terms(coords,b,normalizer.force_scale,order=cfg["integration_order"])
                lf=pterms["free_equilibrium"].mean(); lr=pterms["reaction"].mean()
                lp=cfg["free_residual_weight"]*lf+cfg["reaction_weight"]*lr
            objective = cfg["data_weight"] * ld + physics_weight * lp
            if not torch.isfinite(objective):
                ids = [s["sample_id"] for s in cpu_batch["samples"]]
                raise FloatingPointError(f"Nonfinite loss; samples={ids}; no silent sample skipping")
            if training:
                objective.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
            n = b["num_graphs"]
            total["data"] += float(ld.detach()) * n
            total["free_data"] += float(dterms["free"].detach().sum())
            total["support_data"] += float(dterms["support"].detach().sum())
            total["objective"] += float(objective.detach()) * n
            total["components"] += components.detach().sum(0).cpu().numpy()
            total["n"] += n
            if compute_physics:
                total["physics"] += float(lp.detach()) * n
                total["free_physics"] += float(lf.detach()) * n
                total["reaction"] += float(lr.detach()) * n
                total["physics_n"] += n
    if total["n"] == 0:
        raise ValueError("Empty dataset partition")
    return dict(data=total["data"]/total["n"], objective=total["objective"]/total["n"],
                free_data=total["free_data"]/total["n"],support_data=total["support_data"]/total["n"],
                physics=total["physics"]/total["physics_n"] if total["physics_n"] else math.nan,
                free_physics=total["free_physics"]/total["physics_n"] if total["physics_n"] else math.nan,
                reaction=total["reaction"]/total["physics_n"] if total["physics_n"] else math.nan,
                components=(total["components"]/total["n"]).tolist())


def measure_initial_validation(model, dataset, cfg, normalizer, device, seed):
    """Measure the full initial validation data loss without advancing training RNG."""
    state, was_training = _rng_state(), model.training
    try:
        return _epoch(model, dataset, None, cfg, normalizer, device, seed, 0,
                      physics_weight=0., training=False, compute_physics=False)
    finally:
        _restore_rng(state)
        model.train(was_training)


def train_one(model_id, seed, cfg, datasets, normalizer, run_dir, run_fingerprint, live=True):
    seed_everything(seed, cfg)
    device = select_device(cfg)
    model = FormFindingGNN(model_id, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg["scheduler_factor"],
                  patience=cfg["scheduler_patience"], min_lr=cfg["minimum_lr"])
    physics = PhysicsSchedule(cfg)
    folder = Path(run_dir) / f"{model_id}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    last_path = folder / "last.pt"
    history, best, best_full, last_improvement, start_epoch = [], math.inf, math.inf, 0, 1
    startup_validation = None
    started = time.perf_counter()
    if cfg["resume"] and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=True)
        if ck["fingerprint"] != run_fingerprint or ck["model_id"] != model_id or ck["seed"] != seed:
            raise ValueError("Checkpoint belongs to a different experiment")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        physics.load_state_dict(ck["physics_schedule"])
        history, best, best_full = ck["history"], ck["best"], ck["best_full"]
        last_improvement, start_epoch = ck["last_improvement"], ck["epoch"] + 1
        startup_validation = ck.get("startup_validation")
        _restore_rng(ck["rng"])
    elif last_path.exists():
        raise FileExistsError(f"Refusing to replace {last_path}")
    else:
        baseline_started = time.perf_counter()
        baseline = measure_initial_validation(model, datasets["validation"], cfg, normalizer, device, seed)
        physics.bind_initial_validation(baseline["data"])
        startup_validation = dict(model_id=model_id, seed=seed,
            startup_validation_data=baseline["data"],
            startup_validation_free_data=baseline["free_data"],
            startup_validation_support_data=baseline["support_data"],
            startup_validation_components=baseline["components"],
            physics_start_mode=physics.config["physics_start_mode"],
            physics_effective_threshold=physics.effective_threshold,
            physics_enabled=(model_id == "M4"), schedule_config=dict(physics.config),
            measurement="Initial model in evaluation mode, full validation partition, no physics loss",
            calibration_wall_s=time.perf_counter()-baseline_started)
        save_json(folder/"startup_validation.json", startup_validation)
    display = LiveTraining() if live else None
    print(f"{model_id}: {sum(p.numel() for p in model.parameters()):,} parameters · {device} · seed {seed}", flush=True)
    if model_id == "M4":
        if physics.config["physics_start_mode"] == "immediate":
            print(f"Physics starts at epoch 1: first weight={physics.weight(1):.6g}, "
                  f"full weight={cfg['physics_max_weight']:.6g} at epoch {cfg['physics_ramp_epochs']}.", flush=True)
        elif physics.config["physics_start_mode"] == "relative_initial_validation":
            print(f"Physics gate: initial validation data loss={physics.initial_validation_data:.6g}; "
                  f"activation threshold={physics.effective_threshold:.6g} "
                  f"({physics.config['physics_initial_loss_ratio']:.3g} of the initial loss).", flush=True)
    try:
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            tick = time.perf_counter()
            weight = physics.weight(epoch) if model_id == "M4" else 0.
            check_physics = weight > 0 or epoch % cfg["validation_physics_every"] == 0
            tr = _epoch(model, datasets["train"], optimizer, cfg, normalizer, device, seed, epoch, weight, True, weight > 0)
            va = _epoch(model, datasets["validation"], optimizer, cfg, normalizer, device, seed, epoch, weight, False, check_physics)
            # Scheduler/selection use the same validation DATA objective,
            # avoiding an artificial jump when the physics weight changes.
            scheduler.step(va["data"])
            monitored = tr["data"] if cfg["physics_monitor"] == "train" else va["data"]
            if physics.observe(epoch, monitored) and model_id == "M4":
                print(f"Physics threshold met at epoch {epoch}; ramp starts at epoch {epoch+1}.")
            row = dict(epoch=epoch, train_data=tr["data"], validation_data=va["data"],
                       train_free_data=tr["free_data"],validation_free_data=va["free_data"],
                       train_support_data=tr["support_data"],validation_support_data=va["support_data"],
                       train_objective=tr["objective"], validation_objective=va["objective"],
                       train_physics=tr["physics"], validation_physics=va["physics"],
                       train_free_physics=tr["free_physics"],validation_free_physics=va["free_physics"],
                       train_reaction=tr["reaction"],validation_reaction=va["reaction"],
                       physics_weight=weight, learning_rate=optimizer.param_groups[0]["lr"],
                       epoch_s=time.perf_counter()-tick)
            row.update({f"validation_{c}": v for c,v in zip("xyz", va["components"])})
            history.append(row)
            improved = va["data"] < best
            full_eligible = model_id != "M4" or (physics.full_weight(epoch) and weight > 0)
            improved_full = full_eligible and va["data"] < best_full
            if improved:
                best = va["data"]
            if improved_full:
                best_full, last_improvement = va["data"], epoch
            payload = dict(version=1, fingerprint=run_fingerprint, model_id=model_id, seed=seed,
                           epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                           scheduler=scheduler.state_dict(), physics_schedule=physics.state_dict(),
                           history=history, best=best, best_full=best_full, last_improvement=last_improvement,
                           rng=_rng_state(), config=cfg, physics_active=weight > 0,
                           startup_validation=startup_validation,
                           startup_validation_data=physics.initial_validation_data,
                           physics_start_mode=physics.config["physics_start_mode"],
                           physics_effective_threshold=physics.effective_threshold,
                           full_physics_weight=(model_id == "M4" and full_eligible))
            if improved:
                atomic_torch_save(folder/"best_data.pt", payload)
            if improved_full:
                atomic_torch_save(folder/"best.pt", payload)
            if epoch % cfg["checkpoint_every"] == 0 or epoch == cfg["epochs"]:
                atomic_torch_save(last_path, payload)
            pd.DataFrame(history).to_csv(folder/"history.csv", index=False)
            will_stop = (full_eligible and epoch >= cfg["minimum_epochs"]
                         and epoch-last_improvement >= cfg["early_stopping_patience"])
            report_epoch(model_id, seed, row, cfg["epochs"], folder,
                         log_every=cfg.get("log_every", 1), force=(epoch == start_epoch or will_stop))
            update_curves = epoch % cfg["plot_every"] == 0 or epoch == cfg["epochs"] or will_stop
            if update_curves and display:
                display.update(history, model_id, seed, folder/"learning_curves.png")
            elif update_curves and cfg.get("save_curves_without_display", False):
                import matplotlib.pyplot as plt
                fig = make_training_figure(history, model_id, seed)
                try:
                    fig.savefig(folder/"learning_curves.png", dpi=150, facecolor="black",
                                edgecolor="none", bbox_inches="tight")
                finally:
                    plt.close(fig)
            if will_stop:
                atomic_torch_save(last_path, payload)
                print(f"Early stopping at epoch {epoch}; validation-selected checkpoint retained.")
                break
    except KeyboardInterrupt:
        print(f"Interrupted. Last completed checkpoint: {last_path}. Resume this run from that epoch.")
        raise
    elapsed = time.perf_counter() - started
    ready = (folder/"best.pt").exists()
    if model_id == "M4" and not ready:
        warnings.warn("M4 has no checkpoint trained at full physics weight. Inspect threshold/ramp; best_data.pt is a warm-up result and is NOT reported as a completed M4 experiment.")
    result = dict(model_id=model_id, seed=seed, checkpoint=str(folder/"best.pt") if ready else None,
                  support_mode=cfg["support_mode"],
                  hidden_dim=model.hidden_dim,
                  startup_validation_data=physics.initial_validation_data,
                  physics_start_mode=physics.config["physics_start_mode"],
                  physics_effective_threshold=physics.effective_threshold,
                  completed_epochs=history[-1]["epoch"] if history else 0,
                  physics_start_epoch=physics.start_epoch if model_id == "M4" else None,
                  full_physics_checkpoint_available=ready if model_id == "M4" else None,
                  invocation_wall_s=elapsed, directory=str(folder),
                  parameters=sum(p.numel() for p in model.parameters()))
    save_json(folder/"training_summary.json", result)
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def train_models(cfg, datasets, normalizer, run_dir, run_fingerprint, live=True):
    results = []
    for seed in cfg["seeds"]:
        for model_id in cfg["models"]:
            results.append(train_one(model_id, seed, cfg, datasets, normalizer, run_dir, run_fingerprint, live))
    save_json(Path(run_dir)/"training_registry.json", results)
    return results
