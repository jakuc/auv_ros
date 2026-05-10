"""pid.py — Skalarne PID z anti-windup i presetem całkującym."""

import numpy as np


class PID:
    """
    Skalarne PID.

    preset_integral: wartość startowa członu I — do kompensacji stałych
    zakłóceń (np. nadmiarowy wypór) bez czekania na rozruch całkowania.
    """

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        max_integral: float | None = None,
        max_output:   float | None = None,
        preset_integral: float = 0.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.max_output   = max_output
        self._integral    = preset_integral
        self._prev_error  = None

    def reset(self, preset_integral: float = 0.0) -> None:
        self._integral   = preset_integral
        self._prev_error = None

    def step(self, error: float, dt: float) -> float:
        dt = max(dt, 1e-6)
        self._integral += error * dt
        if self.max_integral is not None:
            self._integral = float(np.clip(self._integral,
                                           -self.max_integral, self.max_integral))
        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        if self.max_output is not None:
            output = float(np.clip(output, -self.max_output, self.max_output))
        return output
