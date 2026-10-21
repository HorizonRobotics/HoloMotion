"""Early stopping observes validation only and survives checkpoint restart."""

import math


class ValidationStop:
    def __init__(self, patience=20, min_relative_improvement=0.001):
        if type(patience) is not int or patience < 0:
            raise ValueError("Patience must be a nonnegative validation count")
        if (
            not math.isfinite(min_relative_improvement)
            or not 0 <= min_relative_improvement < 1
        ):
            raise ValueError("Relative improvement must lie in [0, 1)")
        self.patience = patience
        self.min_relative_improvement = min_relative_improvement
        self.reference = None
        self.bad_checks = 0
        self.checks = 0

    def observe(self, metric):
        if not math.isfinite(metric) or metric < 0:
            raise ValueError(
                "Validation metric must be finite and nonnegative"
            )
        self.checks += 1
        if self.reference is None or (
            metric < self.reference
            and (self.reference - metric) / max(abs(self.reference), 1e-12)
            >= self.min_relative_improvement
        ):
            self.reference, self.bad_checks = metric, 0
        else:
            self.bad_checks += 1
        return self.stopped

    @property
    def stopped(self):
        return bool(self.patience and self.bad_checks >= self.patience)

    def state_dict(self):
        return dict(vars(self))

    def load_state_dict(self, state):
        if state.keys() != self.state_dict().keys() or any(
            state[k] != getattr(self, k)
            for k in ("patience", "min_relative_improvement")
        ):
            raise ValueError("Early-stopping recipe differs")
        if (
            any(
                type(state[k]) is not int or state[k] < 0
                for k in ("checks", "bad_checks")
            )
            or state["bad_checks"] > state["checks"]
        ):
            raise ValueError("Invalid validation progress")
        if state["reference"] is not None and (
            not math.isfinite(state["reference"]) or state["reference"] < 0
        ):
            raise ValueError("Invalid saved validation reference")
        if (state["reference"] is None) != (state["checks"] == 0):
            raise ValueError(
                "Validation reference differs from saved check count"
            )
        self.reference, self.bad_checks, self.checks = (
            state[k] for k in ("reference", "bad_checks", "checks")
        )
