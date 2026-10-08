"""
PHASE 9 -- Intensity-aware, rule-based severity estimator ("severity v2").
---------------------------------------------------------------------------
Severity here is a SIMULATION/RESEARCH INDEX, not an ML prediction and not a
medically or automotive safety-certified scale. It is computed ONLY from
measured window features. It uses no scenario label, no ground truth, no
Random Forest class or probability, and imports no ML library.

LEVELS      LOW < MEDIUM < HIGH < CRITICAL
INPUT       the unchanged window features (window_features.py), one 1 s window.
            Five of the 36 are used, each for a physical reason:

  channel        feature                 why it indicates severity
  -------------  ----------------------  ----------------------------------------
  acceleration   accel_mag_max (g)       peak specific force = size of the crash
                                         pulse (deceleration/impact intensity)
  angular        gyro_mag_max (deg/s)    peak rotation rate: spin, yaw-out, roll
  speed loss     -speed_kmh_first_to_last  km/h lost within the window: how abrupt
                 (km/h)                  the loss of vehicle momentum was
  vibration      vibration_level_mean    sustained structural shaking/scraping
  (rotation      gyro_mag_mean (deg/s)   MEAN rate over the window: high only if
   persistence)                          rotation is sustained, not a single spike;
                                         used only for the rollover-like rule)

SCORE (0-100)
  1. Each channel is normalised to c in [0, 1]:
         c = clip((value - lo) / (hi - lo), 0, 1)
     lo = "ordinary" ceiling, hi = "strong event" level (see THRESHOLD SOURCES).
     Clipping means one extreme number cannot dominate the score.
  2. score = 100 * (w_accel*c_accel + w_gyro*c_gyro + w_speed*c_speed + w_vib*c_vib)
     with weights 0.35 / 0.30 / 0.20 / 0.15 (sum 1). Because every weight is
     < 0.5, ONE channel alone can never reach more than 35 points (MEDIUM):
     a huge acceleration spike with calm gyro, speed and vibration stays
     MEDIUM at most. Two channels at full scale give 45-65 (HIGH needs most of
     that), and CRITICAL (>= 75) needs strong evidence from at least three.
  3. Level bands on the score:  < 25 LOW,  25-50 MEDIUM,  50-75 HIGH,
     >= 75 CRITICAL  (lower bound inclusive).
  4. Rollover-like escalation: if rotation is SUSTAINED (gyro_mag_mean >
     sustained_rotation_dps) AND score >= escalation_min_score, severity is
     raised to CRITICAL, because sustained rotation with other corroborating
     evidence is the rollover signature a single-peak score under-weights.

"active" channels (c >= active_fraction) are listed as triggered rules, so the
output answers "why this level?" with the evidence that was elevated.

THRESHOLD SOURCES (every value is tagged in SeverityConfig.provenance)
  training-data-derived (calibrate_severity_config, TRAINING recordings only):
    lo  = 99th percentile of the quantity over training windows of the four
          non-accident scenarios (Normal Driving, Pothole, Hard Braking, Sharp
          Turn): "above what ordinary driving and routine events produce 99%
          of the time" (same recipe as the Phase 8 threshold detector).
    hi  = 90th percentile of the quantity over training windows of the four
          accident scenarios: a strong-but-not-record-breaking event for this
          simulator. P90 (not max) so a few extreme windows do not stretch the
          scale; it saturates at 1.0 for the top ~10% of accident windows.
    sustained_rotation_dps = 99th percentile of gyro_mag_mean over training
          non-accident windows.
    (speed loss is NOT training-derived: in the simulated training data the
    1 s speed loss of Hard Braking windows (P99 ~50 km/h) exceeds that of
    accident windows (P90 ~31 km/h), so the percentile recipe gives hi < lo.
    It is therefore an engineering assumption, see below.)
  engineering assumptions (NOT learned; chosen from physical reasoning and
  fixed before looking at any held-out data): the four channel weights, the
  band cut-offs 25/50/75, active_fraction 0.25, escalation_min_score 50, and
  the speed-loss anchors lo = 14 km/h (~0.4 g sustained for 1 s: firm braking)
  and hi = 35 km/h (~1 g for 1 s, about the tyre-friction limit of braking, so
  a larger 1 s loss indicates a collision rather than braking alone).
  The test split is never read. Scenario labels are used ONLY inside
  calibrate_severity_config() to pick which training windows define lo/hi;
  assess() never sees them. There is no ground-truth severity label anywhere in
  the dataset, so NO severity accuracy is computed (see README / report).

LIMITATIONS: synthetic data only; anchors are tied to this simulator's intensity
scale; synthetic peaks exceed the MPU6050 firmware ranges (+/-2 g, +/-250
deg/s) which a real sensor would clip; in STM32 hardware mode vibration and
speed are placeholders, so those two channels carry no information there.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from common import FEATURE_COLUMNS
from window_features import WINDOW_SIZE, extract_window_features

DEFAULT_CONFIG_PATH = "severity_config_v2.json"
SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

# Scenario names are used ONLY by the calibration function to select training windows.
_ACCIDENT_SCENARIOS = frozenset({"Minor Accident", "Severe Accident", "Rollover", "Multi-Impact Collision"})
_NON_ACCIDENT_SCENARIOS = frozenset({"Normal Driving", "Pothole", "Hard Braking", "Sharp Turn"})

# Engineering assumption (see module docstring): 1 g sustained for 1 s = 35.3 km/h.
SPEED_LOSS_LO_KMH = 14.0
SPEED_LOSS_HI_KMH = 35.0

LO_PERCENTILE = 99.0
HI_PERCENTILE = 90.0

SCALE_NOTE = ("Simulation/research severity index (0-100) from sensor evidence only; "
              "not an ML output and not a certified medical or automotive severity scale.")


@dataclass(frozen=True)
class SeverityConfig:
    # --- training-data-derived anchors (see module docstring) ---
    accel_lo_g: float
    accel_hi_g: float
    gyro_lo_dps: float
    gyro_hi_dps: float
    speed_loss_lo_kmh: float
    speed_loss_hi_kmh: float
    vibration_lo: float
    vibration_hi: float
    sustained_rotation_dps: float
    # --- engineering assumptions ---
    w_accel: float = 0.35
    w_gyro: float = 0.30
    w_speed: float = 0.20
    w_vibration: float = 0.15
    band_medium: float = 25.0
    band_high: float = 50.0
    band_critical: float = 75.0
    active_fraction: float = 0.25
    escalation_min_score: float = 50.0
    provenance: Dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for lo, hi, name in [(self.accel_lo_g, self.accel_hi_g, "accel"), (self.gyro_lo_dps, self.gyro_hi_dps, "gyro"),
                             (self.speed_loss_lo_kmh, self.speed_loss_hi_kmh, "speed_loss"),
                             (self.vibration_lo, self.vibration_hi, "vibration")]:
            if not hi > lo:
                raise ValueError(f"{name}: hi ({hi}) must be greater than lo ({lo})")
        if min(self.w_accel, self.w_gyro, self.w_speed, self.w_vibration) < 0 or \
                not math.isclose(self.w_accel + self.w_gyro + self.w_speed + self.w_vibration, 1.0, abs_tol=1e-9):
            raise ValueError("weights must be non-negative and sum to 1")
        if not 0 <= self.band_medium < self.band_high < self.band_critical <= 100:
            raise ValueError("bands must satisfy 0 <= medium < high < critical <= 100")

    def to_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SeverityConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# channel id, human name, feature description, value function, (lo, hi, weight) config fields, rule id
_CHANNELS = [
    ("acceleration", "peak acceleration magnitude", "accel_mag_max", lambda f: f["accel_mag_max"],
     "accel_lo_g", "accel_hi_g", "w_accel", "g", "high_peak_acceleration"),
    ("angular", "peak angular rate", "gyro_mag_max", lambda f: f["gyro_mag_max"],
     "gyro_lo_dps", "gyro_hi_dps", "w_gyro", "deg/s", "elevated_angular_rate"),
    ("speed_loss", "speed lost within the window", "-speed_kmh_first_to_last",
     lambda f: -f["speed_kmh_first_to_last"],
     "speed_loss_lo_kmh", "speed_loss_hi_kmh", "w_speed", "km/h", "significant_speed_loss"),
    ("vibration", "mean vibration level", "vibration_level_mean", lambda f: f["vibration_level_mean"],
     "vibration_lo", "vibration_hi", "w_vibration", "", "elevated_vibration"),
]

REQUIRED_FEATURES = ["accel_mag_max", "gyro_mag_max", "gyro_mag_mean", "vibration_level_mean",
                     "speed_kmh_first_to_last"]


def _validate(features: Mapping[str, float]) -> None:
    missing = [k for k in REQUIRED_FEATURES if k not in features]
    if missing:
        raise ValueError(f"features is missing required keys: {missing}")
    bad = [k for k in REQUIRED_FEATURES
           if isinstance(features[k], bool)
           or not isinstance(features[k], (int, float, np.integer, np.floating))
           or not math.isfinite(float(features[k]))]
    if bad:
        raise ValueError(f"features contain non-numeric or non-finite values for: {bad}")


def level_for_score(score: float, cfg: SeverityConfig) -> str:
    """Score band -> level (lower bound inclusive)."""
    if score >= cfg.band_critical:
        return "CRITICAL"
    if score >= cfg.band_high:
        return "HIGH"
    if score >= cfg.band_medium:
        return "MEDIUM"
    return "LOW"


def assess(features: Mapping[str, float], config: Optional[SeverityConfig] = None) -> Dict[str, Any]:
    """
    Estimate severity for one window's feature dict.

    Returns
    -------
    {
      "severity": "LOW"|"MEDIUM"|"HIGH"|"CRITICAL",
      "score": float 0-100,
      "triggered_rules": [rule ids for active channels, "sustained_rotation", "rollover_escalation"],
      "reasons": [plain-English sentences],
      "evidence": {channel: {feature, value, lo, hi, normalized, weight, points, active}},
      "sustained_rotation": bool, "escalated": bool,
      "scale_note": str,
    }
    """
    _validate(features)
    cfg = config if config is not None else load_default_config()
    f = {k: float(features[k]) for k in REQUIRED_FEATURES}

    evidence: Dict[str, Dict[str, Any]] = {}
    triggered: List[str] = []
    reasons: List[str] = []
    score = 0.0
    for key, name, feature, value_fn, lo_f, hi_f, w_f, unit, rule_id in _CHANNELS:
        value, lo, hi, w = float(value_fn(f)), getattr(cfg, lo_f), getattr(cfg, hi_f), getattr(cfg, w_f)
        c = min(max((value - lo) / (hi - lo), 0.0), 1.0)
        active = c >= cfg.active_fraction and c > 0.0
        evidence[key] = {"feature": feature, "value": value, "lo": lo, "hi": hi, "normalized": c,
                         "weight": w, "points": 100.0 * w * c, "active": bool(active)}
        score += 100.0 * w * c
        if active:
            triggered.append(rule_id)
            reasons.append(f"{name} {value:.3g} {unit} is {c:.0%} of the way from the ordinary level "
                           f"({lo:.3g}) to the strong-event level ({hi:.3g})".replace("  ", " "))

    severity = level_for_score(score, cfg)

    sustained = f["gyro_mag_mean"] > cfg.sustained_rotation_dps
    if sustained:
        triggered.append("sustained_rotation")
        reasons.append(f"rotation is sustained (mean angular rate {f['gyro_mag_mean']:.3g} deg/s > "
                       f"{cfg.sustained_rotation_dps:.3g}), not a single spike")
    escalated = bool(sustained and score >= cfg.escalation_min_score and severity != "CRITICAL")
    if escalated:
        severity = "CRITICAL"
        triggered.append("rollover_escalation")
        reasons.append(f"sustained rotation with corroborating evidence (score {score:.1f} >= "
                       f"{cfg.escalation_min_score:g}) raises severity to CRITICAL")

    n_active = sum(e["active"] for e in evidence.values())
    if n_active <= 1 and not escalated:
        reasons.append("only %s sensor channel shows elevated evidence, so severity is limited to MEDIUM or below"
                       % ("one" if n_active else "no"))
    return {"severity": severity, "score": round(score, 3), "triggered_rules": triggered, "reasons": reasons,
            "evidence": evidence, "sustained_rotation": bool(sustained), "escalated": escalated,
            "scale_note": SCALE_NOTE}


def assess_window(window_readings: Union[pd.DataFrame, Sequence[Mapping[str, float]]],
                  config: Optional[SeverityConfig] = None) -> Dict[str, Any]:
    """assess() on a raw window of exactly WINDOW_SIZE readings (keys: common.FEATURE_COLUMNS)."""
    window_df = (window_readings if isinstance(window_readings, pd.DataFrame)
                 else pd.DataFrame(list(window_readings)))
    if len(window_df) != WINDOW_SIZE:
        raise ValueError(f"Expected exactly {WINDOW_SIZE} readings in a window, got {len(window_df)}.")
    missing = [c for c in FEATURE_COLUMNS if c not in window_df.columns]
    if missing:
        raise ValueError(f"window_readings is missing required keys: {missing}")
    return assess(extract_window_features(window_df), config)


def load_default_config(path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> SeverityConfig:
    try:
        return SeverityConfig.from_json(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No severity config at '{path}'. Run `python severity_v2.py` first "
            "(it calibrates on the TRAINING recordings only)."
        )


# ---------------------------------------------------------------------------
# CALIBRATION (the only place scenario labels are used; training windows only)
# ---------------------------------------------------------------------------

def calibrate_severity_config(train_windows: pd.DataFrame, label_column: str = "scenario") -> SeverityConfig:
    """Derive the anchors from TRAINING windows only (see module docstring for the recipe)."""
    non_acc = train_windows[train_windows[label_column].isin(_NON_ACCIDENT_SCENARIOS)]
    acc = train_windows[train_windows[label_column].isin(_ACCIDENT_SCENARIOS)]
    if non_acc.empty or acc.empty:
        raise ValueError("train_windows must contain both accident and non-accident windows")

    quantities = {
        "accel": ("accel_mag_max", lambda d: d["accel_mag_max"]),
        "gyro": ("gyro_mag_max", lambda d: d["gyro_mag_max"]),
        "vibration": ("vibration_level_mean", lambda d: d["vibration_level_mean"]),
    }
    anchors, table = {}, {}
    for key, (feature, fn) in quantities.items():
        lo = float(np.percentile(fn(non_acc), LO_PERCENTILE))
        hi = float(np.percentile(fn(acc), HI_PERCENTILE))
        table[key] = {"feature": feature, "lo": lo, "hi": hi}
        anchors[key] = (lo, hi)
    sustained = float(np.percentile(non_acc["gyro_mag_mean"], LO_PERCENTILE))

    # Diagnostic only: why speed loss is not training-derived (see module docstring).
    sl_non = float(np.percentile(-non_acc["speed_kmh_first_to_last"], LO_PERCENTILE))
    sl_acc = float(np.percentile(-acc["speed_kmh_first_to_last"], HI_PERCENTILE))
    table["speed_loss"] = {"feature": "-speed_kmh_first_to_last", "lo": SPEED_LOSS_LO_KMH, "hi": SPEED_LOSS_HI_KMH,
                           "source": "engineering assumption",
                           "rejected_percentile_recipe": {"P99_non_accident": sl_non, "P90_accident": sl_acc,
                                                          "why_rejected": "hi < lo: simulated hard braking loses "
                                                                          "more speed per second than accident windows"}}
    anchors["speed_loss"] = (SPEED_LOSS_LO_KMH, SPEED_LOSS_HI_KMH)

    cfg_kwargs = dict(
        accel_lo_g=anchors["accel"][0], accel_hi_g=anchors["accel"][1],
        gyro_lo_dps=anchors["gyro"][0], gyro_hi_dps=anchors["gyro"][1],
        speed_loss_lo_kmh=anchors["speed_loss"][0], speed_loss_hi_kmh=anchors["speed_loss"][1],
        vibration_lo=anchors["vibration"][0], vibration_hi=anchors["vibration"][1],
        sustained_rotation_dps=sustained,
    )
    provenance = {
        "disclaimer": SCALE_NOTE + " Synthetic data only; no real severity data was used.",
        "training_derived": {
            "lo": f"P{LO_PERCENTILE:g} over TRAINING windows of Normal Driving/Pothole/Hard Braking/Sharp Turn "
                  f"(n={len(non_acc)}): the ceiling of ordinary driving and routine events",
            "hi": f"P{HI_PERCENTILE:g} over TRAINING windows of Minor/Severe Accident, Rollover, Multi-Impact "
                  f"(n={len(acc)}): a strong event; P90 not max so outliers do not stretch the scale",
            "sustained_rotation_dps": f"P{LO_PERCENTILE:g} of gyro_mag_mean over the same non-accident training windows",
            "fields": ["accel_lo_g", "accel_hi_g", "gyro_lo_dps", "gyro_hi_dps",
                       "vibration_lo", "vibration_hi", "sustained_rotation_dps"],
            "channel_anchors": table,
        },
        "engineering_assumptions": {
            "fields": ["speed_loss_lo_kmh", "speed_loss_hi_kmh", "w_accel", "w_gyro", "w_speed", "w_vibration",
                       "band_medium", "band_high",
                       "band_critical", "active_fraction", "escalation_min_score"],
            "note": "chosen from physical reasoning before examining held-out data; weights < 0.5 each so no "
                    "single sensor channel can exceed 35 points (MEDIUM)",
        },
        "n_train_windows": int(len(train_windows)),
        "test_set_used": False,
        "severity_accuracy": "not computed: the dataset has no ground-truth severity labels",
    }
    return SeverityConfig(**cfg_kwargs, provenance=provenance)


def _train_windows_from_phase7_split() -> pd.DataFrame:
    """Window ONLY the training recordings of the Phase 7 split (test recordings are never windowed here)."""
    # imported lazily so `import severity_v2` pulls in no ML library
    import train_expanded_model_v2 as t2
    from train_expanded_model import build_windows_from_dataset
    df = pd.read_csv(t2.DATASET_CSV_V2)
    train_df, _test_df = t2.split_by_recording(df)       # test half is discarded unread
    return build_windows_from_dataset(train_df)


def main() -> None:
    train_windows = _train_windows_from_phase7_split()
    cfg = calibrate_severity_config(train_windows)
    cfg.to_json(DEFAULT_CONFIG_PATH)
    print("=" * 72)
    print("PHASE 9: severity_v2 calibration (TRAINING recordings only; SYNTHETIC data)")
    print("=" * 72)
    print(f"Training windows used: {len(train_windows)} "
          f"({train_windows['recording_id'].nunique()} recordings); test set not read.")
    print(f"{'channel':12s}{'feature':28s}{'lo':>10s}{'hi':>10s}   source")
    for key, row in cfg.provenance["training_derived"]["channel_anchors"].items():
        src = row.get("source", "training: lo=P99 non-accident, hi=P90 accident")
        print(f"{key:12s}{row['feature']:28s}{row['lo']:>10.4g}{row['hi']:>10.4g}   {src}")
    print(f"sustained_rotation_dps (P99 non-acc gyro_mag_mean): {cfg.sustained_rotation_dps:.4g}")
    print(f"weights accel/gyro/speed/vib: {cfg.w_accel}/{cfg.w_gyro}/{cfg.w_speed}/{cfg.w_vibration}   "
          f"bands: {cfg.band_medium:g}/{cfg.band_high:g}/{cfg.band_critical:g}")
    print(f"\nSaved: {DEFAULT_CONFIG_PATH}")


if __name__ == "__main__":
    main()
