"""Separate diagnostic predictions from completed physics-training evidence."""
from pathlib import Path


def evaluation_checkpoint(item, allow_incomplete=False):
    """Never replace the official training checkpoint with a warm-up checkpoint."""
    if item.get("checkpoint"):
        return item["checkpoint"]
    if item["model_id"] == "M4" and allow_incomplete:
        candidate = Path(item.get("source_directory", item["directory"])) / "best_data.pt"
        if candidate.is_file():
            return str(candidate)
    return None


def checkpoint_status(checkpoint):
    mid = checkpoint["model_id"]
    weight = float(checkpoint["history"][-1]["physics_weight"])
    full = bool(checkpoint.get("full_physics_weight", False))
    status = "not_applicable" if mid != "M4" else ("full" if full else "ramping" if weight > 0 else "data_only")
    label = {"data_only": "M4 (data-only warm-up)", "ramping": "M4 (incomplete physics ramp)",
             "full": "M4", "not_applicable": mid}[status]
    return dict(physics_status=status, model_label=label,
                completed_physics_training=(mid == "M4" and full),
                diagnostic_only=(mid == "M4" and not full),
                physics_weight_at_checkpoint=weight)
