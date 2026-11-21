"""Experiment configuration. Units: m, kN, kPa (= kN/m²)."""
from pathlib import Path
import math
import warnings


def default_config(project_root=None):
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    return dict(
        dataset_root=str(root.parent / "dataset"),
        output_root=str(root / "runs"), run_name=None, resume=False,
        models=["M1", "M2", "M3", "M4"], seeds=[42],
        hidden_dim=128, hidden_dims_by_model={}, layers=6, mlp_depth=2, activation="silu", dropout=0.0,
        epochs=300, batch_size=8, learning_rate=1e-3, weight_decay=1e-5,
        gradient_clip=1.0, scheduler_factor=0.5, scheduler_patience=15,
        minimum_lr=1e-6, early_stopping_patience=60, minimum_epochs=60,
        split_mode="existing", split_fractions=[0.8, 0.1, 0.1], split_seed=42,
        max_layouts=None, max_variants_per_layout=None, lpe_dim=8,
        lpe_sign_augmentation=True, cache_size=12000, verify_hashes=True,
        device="auto", deterministic=True, cpu_threads=4,
        data_weight=1.0, component_weights=[1.0, 1.0, 1.0],
        support_mode="learned", support_displacement_weight=1.0,
        free_residual_weight=1.0, reaction_weight=1.0,
        support_displacement_tolerance_m=1e-3,
        physics_max_weight=0.01, physics_min_epoch=30,
        physics_data_threshold=0.02, physics_consecutive_epochs=3,
        physics_ramp_epochs=40, physics_monitor="validation",
        physics_start_mode="absolute_threshold", physics_initial_loss_ratio=0.1,
        integration_order=8, evaluation_integration_order=16,
        plot_every=1, log_every=1, checkpoint_every=1, validation_physics_every=5,
        save_node_predictions=True, evaluation_split="test", evaluate_incomplete_m4=True,
        comparison_top_k=6, comparison_min_relative_gap=0.0,
    )


def validate_config(cfg):
    cfg.setdefault("log_every", 1)
    cfg.setdefault("evaluate_incomplete_m4", True)
    cfg.setdefault("hidden_dims_by_model", {})
    cfg.setdefault("physics_start_mode", "absolute_threshold")
    cfg.setdefault("physics_initial_loss_ratio", 0.1)
    if not cfg["models"] or len(set(cfg["models"])) != len(cfg["models"]) or set(cfg["models"]) - {"M1", "M2", "M3", "M4"}:
        raise ValueError("models must be a nonempty, unique subset of M1, M2, M3, M4")
    for key in ("hidden_dim", "layers", "mlp_depth", "epochs", "batch_size", "cpu_threads", "physics_min_epoch", "physics_consecutive_epochs", "physics_ramp_epochs", "plot_every", "log_every", "checkpoint_every", "validation_physics_every"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    widths = cfg["hidden_dims_by_model"]
    if (not isinstance(widths, dict) or set(widths) - {"M1", "M2", "M3", "M4"}
            or any(type(value) is not int or value < 1 for value in widths.values())):
        raise ValueError("hidden_dims_by_model must map model IDs to positive integer widths")
    if cfg["physics_start_mode"] not in ("absolute_threshold", "immediate", "relative_initial_validation"):
        raise ValueError("Invalid physics_start_mode")
    ratio = cfg["physics_initial_loss_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("physics_initial_loss_ratio must lie in (0, 1]")
    if cfg["physics_start_mode"] == "relative_initial_validation" and cfg["physics_monitor"] != "validation":
        raise ValueError("relative_initial_validation requires physics_monitor='validation'")
    if type(cfg["lpe_dim"]) is not int or not 1 <= cfg["lpe_dim"] <= 8:
        raise ValueError("The CSVs contain eight stored Laplacian modes; choose lpe_dim from 1 to 8")
    if not 0 <= cfg["dropout"] < 1 or cfg["activation"] not in ("silu", "relu", "gelu"):
        raise ValueError("Invalid dropout or activation")
    if cfg["support_mode"] not in ("learned", "hard"):
        raise ValueError("support_mode must be learned or hard")
    for key in ("support_displacement_weight", "free_residual_weight", "reaction_weight", "support_displacement_tolerance_m"):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("learning_rate", "data_weight", "physics_data_threshold", "gradient_clip", "minimum_lr"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(cfg["physics_max_weight"]) or cfg["physics_max_weight"] < 0:
        raise ValueError("physics_max_weight must be finite and nonnegative")
    if "M4" in cfg["models"] and cfg["physics_max_weight"] == 0:
        raise ValueError("M4 requires a positive physics weight; use M3 for the data-only counterpart")
    if not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not math.isfinite(cfg["scheduler_factor"]) or not 0 < cfg["scheduler_factor"] < 1:
        raise ValueError("scheduler_factor must lie strictly between zero and one")
    for key in ("scheduler_patience", "minimum_epochs", "cache_size"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("early_stopping_patience", "comparison_top_k"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("resume", "deterministic", "verify_hashes", "save_node_predictions", "lpe_sign_augmentation", "evaluate_incomplete_m4"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be True or False")
    gap=cfg["comparison_min_relative_gap"]
    if not math.isfinite(gap) or not 0 <= gap < 1:
        raise ValueError("comparison_min_relative_gap must be finite and lie in [0, 1)")
    if len(cfg["component_weights"]) != 3 or any(not math.isfinite(w) or w <= 0 for w in cfg["component_weights"]):
        raise ValueError("component_weights must contain three positive finite weights")
    if cfg["physics_monitor"] not in ("train", "validation"):
        raise ValueError("physics_monitor must be train or validation")
    if cfg["evaluation_split"] not in ("validation", "test"):
        raise ValueError("evaluation_split must be validation or test")
    if type(cfg["integration_order"]) is not int or cfg["integration_order"] != 8 or type(cfg["evaluation_integration_order"]) is not int or not 8 <= cfg["evaluation_integration_order"] <= 128:
        raise ValueError("Training must retain FFT's order 8; evaluation order must be an integer from 8 to 128")
    if not cfg["seeds"] or any(type(s) is not int or s < 0 for s in cfg["seeds"]) or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("seeds must be unique nonnegative integers")
    earliest_full = (cfg["physics_ramp_epochs"] if cfg["physics_start_mode"] == "immediate" else
                     cfg["physics_min_epoch"] + cfg["physics_consecutive_epochs"] + cfg["physics_ramp_epochs"] - 1)
    if "M4" in cfg["models"] and cfg["epochs"] < earliest_full:
        warnings.warn(f"M4 cannot finish its physics ramp within {cfg['epochs']} epochs; earliest possible full-weight epoch is {earliest_full}.")
    return cfg
