"""Shared dynamic threshold from the modified LeRobot evaluation entry."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RobustThresholdState:
    window_size: int = 15
    base_noise_floor: float = 0.25
    bootstrap_initial_threshold: float = 10.0
    bootstrap_queue_size: int = 15
    bootstrap_sigma_scale: float = 1.0
    ewma_alpha: float = 0.35
    k_on: float = 3.0
    k_off: float = 2.0
    trigger_margin: float = 0.02
    jump_trigger: float = 0.06
    trigger_consecutive_steps: int = 5
    reset_consecutive_steps: int = 5
    cooldown_steps: int = 0
    hard_retrigger_margin: float = 0.08
    hard_retrigger_consecutive_steps: int = 3

    bootstrap_et_queue: list[float] = field(default_factory=list)
    bootstrapped: bool = False
    clean_history: list[float] = field(default_factory=list)
    dynamic_threshold: list[float] = field(default_factory=list)
    dynamic_release_threshold: list[float] = field(default_factory=list)
    ewma_prev: float | None = None
    alert_active: bool = False
    over_count: int = 0
    under_count: int = 0
    hard_over_count: int = 0
    cooldown_remaining: int = 0


def robust_threshold_step(current_et: float, state: RobustThresholdState) -> tuple[float, bool]:
    if not np.isfinite(current_et):
        state.dynamic_threshold.append(float("nan"))
        state.dynamic_release_threshold.append(float("nan"))
        return float("nan"), False

    if state.ewma_prev is None or not np.isfinite(state.ewma_prev):
        ebar = float(current_et)
        jump = 0.0
    else:
        alpha = float(np.clip(state.ewma_alpha, 1e-6, 1.0))
        ebar = alpha * float(current_et) + (1.0 - alpha) * float(state.ewma_prev)
        jump = abs(ebar - float(state.ewma_prev))
    state.ewma_prev = float(ebar)

    if not state.bootstrapped:
        thresh = float(state.bootstrap_initial_threshold)
        state.bootstrap_et_queue.append(float(ebar))
        if len(state.bootstrap_et_queue) > int(state.bootstrap_queue_size):
            state.bootstrap_et_queue = state.bootstrap_et_queue[-int(state.bootstrap_queue_size) :]

        if len(state.bootstrap_et_queue) >= int(state.bootstrap_queue_size):
            q = np.asarray(state.bootstrap_et_queue, dtype=np.float32)
            med = float(np.median(q))
            mad = float(np.median(np.abs(q - med)))
            mad = max(mad, 1e-6)
            ton = max(state.base_noise_floor, med + float(state.k_on) * mad)
            toff = max(state.base_noise_floor, med + float(state.k_off) * mad)
            if toff > ton:
                toff = ton
            thresh = ton
            state.bootstrapped = True
            state.clean_history = list(float(x) for x in q.tolist())  # noqa: C400
            state.dynamic_release_threshold.append(float(toff))
        else:
            state.dynamic_release_threshold.append(float(thresh))

        state.dynamic_threshold.append(thresh)
        return thresh, False

    recent_history = state.clean_history[-max(2, int(state.window_size)) :]
    rh = np.asarray(recent_history, dtype=np.float32)
    med = float(np.median(rh))
    mad = float(np.median(np.abs(rh - med)))
    mad = max(mad, 1e-6)
    ton = max(state.base_noise_floor, med + float(state.k_on) * mad)
    toff = max(state.base_noise_floor, med + float(state.k_off) * mad)
    if toff > ton:
        toff = ton

    state.dynamic_threshold.append(float(ton))
    state.dynamic_release_threshold.append(float(toff))

    cond_over = bool(ebar > (ton + float(state.trigger_margin)))
    cond_jump = bool(jump > float(state.jump_trigger) and ebar > ton)
    cond_hard = bool(ebar > (ton + float(state.hard_retrigger_margin)))

    state.over_count = state.over_count + 1 if cond_over else 0
    state.hard_over_count = state.hard_over_count + 1 if cond_hard else 0
    state.under_count = state.under_count + 1 if ebar < toff else 0

    if state.alert_active and state.under_count >= int(state.reset_consecutive_steps):
        state.alert_active = False
        state.over_count = 0
        state.hard_over_count = 0

    should_trigger = False
    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1
        if state.hard_over_count >= int(state.hard_retrigger_consecutive_steps):
            should_trigger = True
    else:
        if cond_jump or state.over_count >= int(state.trigger_consecutive_steps):
            should_trigger = True

    if should_trigger:
        state.alert_active = True
        state.cooldown_remaining = max(0, int(state.cooldown_steps))
        state.over_count = 0
        state.hard_over_count = 0
        state.under_count = 0

    if state.alert_active:
        state.clean_history.append(float(min(ebar, ton)))
    else:
        state.clean_history.append(float(ebar))

    return float(ton), bool(should_trigger)
