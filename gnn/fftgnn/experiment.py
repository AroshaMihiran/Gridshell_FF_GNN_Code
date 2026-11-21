"""Run provenance, atomic saves and reproducible preprocessing."""
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
import torch
from .config import validate_config


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def prepare_run(cfg, selected, normalizer):
    """Freeze selections/statistics before training; never change dataset files."""
    validate_config(cfg)
    module_dir = Path(__file__).parent
    source_hashes = {p.name: sha_file(p) for p in sorted(module_dir.glob("*.py")) if p.name != "viewer.py"}
    records = selected.sort_values("sample_id")[["sample_id", "base_layout_id", "normalized_geometry_group", "split"]].to_dict("records")
    label_hashes = {r.sample_id: sha_file(Path(r.training_directory)/"label_manifest.json") for r in selected.itertuples()}
    # Increasing the epoch budget or changing display/device does not change
    # the experiment definition. Architecture, losses and splits do.
    operational = {"epochs", "resume", "run_name", "output_root", "dataset_root", "device", "cpu_threads", "plot_every", "log_every", "evaluate_incomplete_m4", "checkpoint_every", "save_node_predictions", "save_curves_without_display", "evaluation_split", "comparison_top_k", "comparison_min_relative_gap", "cache_size", "validation_physics_every"}
    definition = dict(config={k:v for k,v in cfg.items() if k not in operational},
                      selection=records, normalization=normalizer.to_dict(), label_manifest_sha256=label_hashes,
                      target_definition={
                          "displacements": "FFT joint displacements at all joints; supported targets are zero",
                          "support_mode": cfg["support_mode"],
                          "reaction_source": "final_joint.csv: Rx_support_kN, Ry_support_kN, Rz_support_kN",
                          "reaction_convention": "Support force on the structure; R_s=f_internal_s-f_external_s; free entries zero",
                          "reaction_verification": "Mandatory label identity, source hashes and force/moment balance checks before graph loading",
                          "reaction_integrity": "The saved label_manifest_sha256 covers result_sha256.final_joint.csv and summary.json",
                          "reaction_usage": "Additional supervised M4 target and evaluation reference; excluded from graph features and scaler fitting",
                      },
                      source_sha256=source_hashes)
    digest = fingerprint(definition)
    name = cfg["run_name"]
    if cfg["resume"] and not name:
        raise ValueError("For resume=True, set run_name to the existing run folder name")
    if name is None:
        name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + digest[:8]
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("run_name must be a single folder name")
    folder = Path(cfg["output_root"]).resolve() / name
    meta_path = folder / "experiment.json"
    if folder.exists() and not cfg["resume"]:
        raise FileExistsError(f"Run already exists: {folder}; choose another name or resume=True")
    if cfg["resume"]:
        if not meta_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {meta_path}")
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        if saved["fingerprint"] != digest:
            raise ValueError("Run code/data/split/scaling/configuration changed; use a new run name")
    else:
        folder.mkdir(parents=True)
        save_json(meta_path, dict(fingerprint=digest, definition=definition,
                  created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"))
        save_json(folder / "normalizer.json", normalizer.to_dict())
        selected.to_csv(folder / "selected_cases.csv", index=False)
    save_json(folder / "latest_configuration.json", cfg)
    return folder, digest
