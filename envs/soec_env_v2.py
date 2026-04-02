"""
soec_env_v2.py — SOEC Gymnasium environment v2 for MAPIRL-DT.

State  : [v_cell, dT, util, RE_now, RE_predicted]
          v_cell       : cell voltage          [0.90, 1.40] V
          dT           : temp deviation        [-50,  50]  °C
          util         : fuel utilisation      [0.0,  1.0]
          RE_now       : current RE signal     [0.0,  1.0]  (normalised wind)
          RE_predicted : LSTM t+1 forecast     [0.0,  1.0]

Action : delta_current  [-0.05, 0.05]  (per-step current increment)

Reward : r = w_load * r_load  +  w_safe * r_safe  +  w_stab * r_stab
          r_load = -|RE_now - util|               load matching
          r_safe = 0  if v_cell in [0.9, 1.4]     safety
                   -1 otherwise (hard penalty)
          r_stab = -|action - prev_action|         smoothness

Improvements over v1:
  - dT and utilisation added to state (richer plant info)
  - RE signal split into RE_now + RE_predicted (LSTM lookahead)
  - Composite reward with tunable weights
  - info dict exposes load_error, safety_violation, dT, util for W&B logging

Compatible with stable-baselines3 SAC / RE-SAC.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# SOEC plant constants  (realistic solid-oxide electrolyser parameters)
# ---------------------------------------------------------------------------
V_NOM       = 1.20    # V  nominal cell voltage
V_LOW       = 0.90    # V  lower safety bound
V_HIGH      = 1.40    # V  upper safety bound
DT_MAX      = 50.0    # °C max temperature deviation from setpoint
UTIL_NOM    = 0.70    # nominal fuel utilisation
I_MAX       = 1.0     # A  normalised max current
DT_STEP     = 0.5     # °C temperature drift per step (simplified dynamics)
TAU_V       = 0.85    # voltage lag coefficient  (first-order approximation)
TAU_T       = 0.95    # temperature lag coefficient
EPISODE_LEN = 200     # steps per episode


# ---------------------------------------------------------------------------
# Reward weights (tunable via constructor)
# ---------------------------------------------------------------------------
DEFAULT_W_LOAD = 1.0
DEFAULT_W_SAFE = 5.0   # heavy safety penalty
DEFAULT_W_STAB = 0.5


class SOECEnvV2(gym.Env):
    """
    SOEC load-following environment v2.

    Parameters
    ----------
    re_profile : np.ndarray, optional
        Pre-computed normalised RE signal array of shape (T,).
        If None, a synthetic sinusoidal+noise profile is generated each episode.
    re_forecast : np.ndarray, optional
        LSTM forecast array of same length as re_profile, shape (T,).
        If None, RE_predicted = RE_now (naive persistence forecast).
    w_load, w_safe, w_stab : float
        Reward component weights.
    seed : int
        RNG seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        re_profile:  Optional[np.ndarray] = None,
        re_forecast: Optional[np.ndarray] = None,
        w_load: float = DEFAULT_W_LOAD,
        w_safe: float = DEFAULT_W_SAFE,
        w_stab: float = DEFAULT_W_STAB,
        seed:   int   = 42,
    ):
        super().__init__()

        self.w_load = w_load
        self.w_safe = w_safe
        self.w_stab = w_stab
        self._rng   = np.random.default_rng(seed)

        self._re_profile_fixed  = re_profile
        self._re_forecast_fixed = re_forecast

        # ---- Spaces ----
        # obs: [v_cell, dT, util, RE_now, RE_predicted]
        self.observation_space = spaces.Box(
            low  = np.array([V_LOW,  -DT_MAX, 0.0, 0.0, 0.0], dtype=np.float32),
            high = np.array([V_HIGH,  DT_MAX, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        # action: delta_current (same as v1 for checkpoint compatibility)
        self.action_space = spaces.Box(
            low  = np.array([-0.05], dtype=np.float32),
            high = np.array([ 0.05], dtype=np.float32),
        )

        # Internal state (initialised in reset)
        self._v_cell      = V_NOM
        self._dT          = 0.0
        self._util        = UTIL_NOM
        self._current     = 0.5          # normalised [0,1]
        self._prev_action = 0.0
        self._step_count  = 0
        self._re_profile: np.ndarray  = np.array([])
        self._re_forecast: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_re_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic RE profile + naive persistence forecast."""
        t     = np.linspace(0, 4 * np.pi, EPISODE_LEN)
        noise = self._rng.normal(0, 0.05, EPISODE_LEN)
        re    = np.clip(0.5 + 0.35 * np.sin(t) + noise, 0.0, 1.0).astype(np.float32)
        # Persistence forecast: RE_predicted[t] = RE_now[t] (overridden if LSTM provided)
        forecast = np.roll(re, -1)
        forecast[-1] = forecast[-2]
        return re, forecast

    def _get_obs(self) -> np.ndarray:
        t = min(self._step_count, len(self._re_profile) - 1)
        re_now  = float(self._re_profile[t])
        re_pred = float(self._re_forecast[t])
        return np.array(
            [self._v_cell, self._dT, self._util, re_now, re_pred],
            dtype=np.float32,
        )

    def _plant_dynamics(self, action: float) -> None:
        """
        Simplified first-order SOEC dynamics.

        Voltage:     v[t+1] = τ_v·v[t] + (1-τ_v)·(V_NOM + α·ΔI)
        Temperature: dT[t+1] = τ_T·dT[t] + β·(I - I_nom)²  (ohmic heating)
        Utilisation: util    = I / I_max  (algebraic)
        """
        alpha = 0.30   # voltage sensitivity to current
        beta  = 2.0    # thermal sensitivity

        self._current     = np.clip(self._current + action, 0.0, I_MAX)
        v_ss              = V_NOM + alpha * (self._current - 0.5)
        self._v_cell      = TAU_V * self._v_cell + (1 - TAU_V) * v_ss
        self._v_cell      = float(np.clip(self._v_cell, V_LOW - 0.05, V_HIGH + 0.05))
        dT_drive          = beta * (self._current - 0.5) ** 2
        self._dT          = float(np.clip(
            TAU_T * self._dT + (1 - TAU_T) * dT_drive * DT_MAX,
            -DT_MAX, DT_MAX,
        ))
        self._util = float(np.clip(self._current / I_MAX, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._re_profile_fixed is not None:
            self._re_profile  = self._re_profile_fixed
            self._re_forecast = (
                self._re_forecast_fixed
                if self._re_forecast_fixed is not None
                else np.roll(self._re_profile, -1)
            )
        else:
            self._re_profile, self._re_forecast = self._make_re_profile()

        # Randomise initial conditions slightly
        self._v_cell      = float(self._rng.uniform(1.10, 1.30))
        self._dT          = float(self._rng.uniform(-5.0, 5.0))
        self._util        = UTIL_NOM
        self._current     = 0.5
        self._prev_action = 0.0
        self._step_count  = 0

        return self._get_obs(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action_scalar = float(np.clip(action[0], -0.05, 0.05))

        # Plant dynamics
        self._plant_dynamics(action_scalar)

        t      = min(self._step_count, len(self._re_profile) - 1)
        re_now = float(self._re_profile[t])

        # ---- Reward components ----
        # Load matching: how well utilisation tracks RE signal
        r_load = -abs(re_now - self._util)

        # Safety: hard penalty if voltage outside bounds
        safety_violation = not (V_LOW <= self._v_cell <= V_HIGH)
        r_safe = -1.0 if safety_violation else 0.0

        # Stability: penalise large action changes
        r_stab = -abs(action_scalar - self._prev_action)

        reward = (
            self.w_load * r_load
            + self.w_safe * r_safe
            + self.w_stab * r_stab
        )

        self._prev_action = action_scalar
        self._step_count += 1

        terminated = False
        truncated  = self._step_count >= EPISODE_LEN

        info = {
            "load_error":        abs(re_now - self._util),
            "safety_violation":  safety_violation,
            "v_cell":            self._v_cell,
            "dT":                self._dT,
            "util":              self._util,
            "RE_now":            re_now,
            "r_load":            r_load,
            "r_safe":            r_safe,
            "r_stab":            r_stab,
            "wind_power_norm":   re_now,   # for ForecastAugmentedEnv compatibility
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        print(
            f"step={self._step_count:4d}  "
            f"v={self._v_cell:.3f}V  "
            f"dT={self._dT:+.1f}°C  "
            f"util={self._util:.2f}  "
            f"RE={self._re_profile[min(self._step_count, len(self._re_profile)-1)]:.2f}"
        )
