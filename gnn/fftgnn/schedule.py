"""Latched, resumable delayed physics weighting, inspired by the forward notebook."""
import math


class PhysicsSchedule:
    def __init__(self, cfg):
        self.config = {k: cfg[k] for k in ("physics_max_weight", "physics_min_epoch", "physics_data_threshold", "physics_consecutive_epochs", "physics_ramp_epochs", "physics_monitor")}
        self.config.update(physics_start_mode=cfg.get("physics_start_mode", "absolute_threshold"),
                           physics_initial_loss_ratio=cfg.get("physics_initial_loss_ratio", 0.1))
        self.initial_validation_data = None
        self.start_epoch = 1 if self.config["physics_start_mode"] == "immediate" else None
        self.streak, self.last_epoch = 0, 0

    @property
    def effective_threshold(self):
        if self.config["physics_start_mode"] == "immediate":
            return None
        if self.config["physics_start_mode"] == "relative_initial_validation":
            if self.initial_validation_data is None:
                return None
            return self.config["physics_initial_loss_ratio"] * self.initial_validation_data
        return self.config["physics_data_threshold"]

    def bind_initial_validation(self, data_loss):
        """Bind the unchanged initial model's validation loss exactly once."""
        if self.initial_validation_data is not None or self.last_epoch:
            raise ValueError("Initial validation calibration is immutable; resume its saved state")
        if not math.isfinite(data_loss) or data_loss < 0:
            raise ValueError("Initial validation data loss must be finite and nonnegative")
        self.initial_validation_data = float(data_loss)

    def weight(self, epoch):
        if self.start_epoch is None or epoch < self.start_epoch:
            return 0.
        return self.config["physics_max_weight"] * min(1., (epoch-self.start_epoch+1)/self.config["physics_ramp_epochs"])

    def observe(self, epoch, data_loss):
        if epoch != self.last_epoch + 1:
            raise ValueError("Observe every completed epoch exactly once")
        if self.config["physics_start_mode"] == "relative_initial_validation" and self.initial_validation_data is None:
            raise ValueError("Bind initial validation loss before observing a relative schedule")
        self.last_epoch = epoch
        if self.start_epoch is not None:
            return False
        good = (epoch >= self.config["physics_min_epoch"] and math.isfinite(data_loss)
                and data_loss <= self.effective_threshold)
        self.streak = self.streak + 1 if good else 0
        if self.streak >= self.config["physics_consecutive_epochs"]:
            self.start_epoch = epoch + 1
            return True
        return False

    def full_weight(self, epoch):
        return self.start_epoch is not None and epoch >= self.start_epoch+self.config["physics_ramp_epochs"]-1

    def state_dict(self):
        return dict(config=dict(self.config), start_epoch=self.start_epoch, streak=self.streak,
                    last_epoch=self.last_epoch, initial_validation_data=self.initial_validation_data,
                    effective_threshold=self.effective_threshold)

    def load_state_dict(self, state):
        saved_config = dict(state["config"])
        saved_config.setdefault("physics_start_mode", "absolute_threshold")
        saved_config.setdefault("physics_initial_loss_ratio", 0.1)
        if saved_config != self.config:
            raise ValueError("Physics schedule changed; start a new run")
        if state["last_epoch"] < 0 or state["streak"] < 0:
            raise ValueError("Invalid schedule state")
        baseline = state.get("initial_validation_data")
        if baseline is not None and (not math.isfinite(baseline) or baseline < 0):
            raise ValueError("Invalid initial validation calibration")
        if self.config["physics_start_mode"] == "relative_initial_validation" and baseline is None:
            raise ValueError("Relative schedule checkpoint lacks its initial validation calibration")
        self.initial_validation_data = baseline
        if "effective_threshold" in state and state["effective_threshold"] != self.effective_threshold:
            raise ValueError("Saved relative calibration threshold is inconsistent")
        self.start_epoch, self.streak, self.last_epoch = state["start_epoch"], state["streak"], state["last_epoch"]
