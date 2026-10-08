# AI-Based Smart Accident Detection & Emergency Response System

A student/research prototype that classifies short windows of motion-sensor data
(accelerometer, gyroscope, vibration, speed) into six driving events, maps the
predicted event to a severity level, and simulates an emergency-alert flow
(countdown, cancel, "sent") in a live Streamlit dashboard. Sensor data can come
from a built-in **simulator** or from an **STM32 + MPU6050** board over serial.

> ## Read this first: the ML results are based on SYNTHETIC data
>
> Every model in this repository was trained and evaluated **only on simulated
> data produced by hand-written generators** (`sensor_simulator.py`,
> `dataset_generator.py`). No real vehicle, crash, or real-world driving
> recording was ever used. Any accuracy/F1 figure produced by the training
> scripts measures how well a model separates *synthetic recordings from the
> same generator* - a much lower bar than real-world accident detection.
> **This is not a validated or certified safety system, and these numbers must
> not be cited as evidence of real-world detection performance.**
> See [Research-integrity limitations](#research-integrity-limitations).

## Contents

- [System architecture](#system-architecture)
- [Project history (Phases 1-6)](#project-history-phases-1-6)
- [Current features](#current-features)
- [Repository layout](#repository-layout)
- [Setup (E: drive virtual environment)](#setup-e-drive-virtual-environment)
- [Running the simulator](#running-the-simulator)
- [Running the Streamlit dashboard](#running-the-streamlit-dashboard)
- [Running the tests](#running-the-tests)
- [STM32 integration](#stm32-integration)
- [Serial format](#serial-format)
- [Research-integrity limitations](#research-integrity-limitations)

## System architecture

```
 SensorSource (sensor_source.py)                     common interface: start() / read() / stop()
  |-- SimulatedSensorSource   <- dataset_generator.generate_recording()   (synthetic, 10 Hz, 50 samples)
  '-- STM32SensorSource       <- serial port <- STM32 + MPU6050 firmware  (real accel/gyro)
            |
            |  one reading (dict with the 8 common.FEATURE_COLUMNS) per read()
            v
 RollingWindowBuffer (realtime_pipeline.py)          last 10 readings = 1 second
            |
            v
 extract_window_features (window_features.py)        36 features per window
            |
            v
 Random Forest (accident_classifier_model_expanded.joblib, via predict_expanded.py)
            |   -> event label + confidence + class probabilities
            v
 severity_for_label()                                fixed, hand-authored lookup (NOT learned)
            |
            v
 next_status()  +  countdown logic in app.py         MONITORING / EVENT DETECTED /
            |                                        ALERT COUNTDOWN / ALERT SENT / ALERT CANCELLED
            v
 Streamlit dashboard (app.py)                        live values, charts, prediction, severity, status,
                                                     simulated GPS. No real SMS/GSM message is ever sent.
```

Reading contract for `SensorSource.read()`:

| Return value  | Meaning                                                                  |
|---------------|--------------------------------------------------------------------------|
| `dict`        | A valid reading containing every key in `common.FEATURE_COLUMNS`.        |
| `None`        | The stream ended or became unavailable. For hardware sources, `last_error` says why (see below). |
| `NO_DATA_YET` | *Live sources only.* The stream is alive but no valid sample arrived during this short call - keep calling `read()`. |

The simulator ends with plain `None` and no `last_error` (normal end of a
recording). `STM32SensorSource` sets `last_error` for every abnormal end (port
cannot be opened, cable unplugged, board silent, garbled data), and the
dashboard shows that message instead of "Recording finished".

## Project history (Phases 1-6)

| Phase | What it added | Key files |
|-------|---------------|-----------|
| 1 | Sensor-data **simulator** for six scenarios (Normal Driving, Pothole, Hard Braking, Sharp Turn, Minor Accident, Severe Accident); one 5 s / 10 Hz recording each, saved as `simulated_*.csv`. | `sensor_simulator.py`, `simulated_*.csv` |
| 2 | **Row-based** Random Forest on individual sensor rows. Reported accuracy 0.75 (`PHASE2_ACCURACY` in `train_expanded_model.py`); mainly confused Normal Driving with Pothole because most rows of a brief event look like normal driving. | `common.py`, `train_model.py`, `predict.py`, `accident_classifier_model.joblib` |
| 3 | **Window-based** features (1 s windows; mean, std, range, first-to-last per channel + accel/gyro magnitude). Reported accuracy 0.833, but on only one recording per class (6 test windows) - too small to trust. Experimental. | `window_features.py`, `train_window_model.py`, `predict_window.py`, `accident_classifier_model_windowed.joblib` |
| 4 | **Expanded synthetic dataset** (600 independent randomized recordings, 100 per scenario) and a **recording-level** train/test split done *before* windowing, so no recording's windows land on both sides (no leakage). Same model type/hyper-parameters as Phases 2-3 for a like-for-like comparison. | `dataset_generator.py`, `expanded_synthetic_dataset.csv`, `train_expanded_model.py`, `accident_classifier_model_expanded.joblib` |
| 5 | **Real-time software simulator**: sensor abstraction, rolling window buffer, live prediction, severity lookup, alert state machine with cancellable countdown, simulated GPS, Streamlit dashboard. Uses the Phase 4 model read-only. | `sensor_source.py`, `realtime_pipeline.py`, `predict_expanded.py`, `app.py` |
| 6 | **STM32 + MPU6050 hardware path**: firmware that streams CSV over UART, `STM32SensorSource` that parses it, dashboard source selector. Not verified on real hardware yet (see below). | `stm32_firmware/`, `STM32SensorSource` in `sensor_source.py` |
| 7 | **8-scenario research dataset and model**: adds simulated *Rollover* and *Multi-Impact Collision*, a new 800-recording dataset, a new Random Forest and predictor, all in NEW files (Phases 2-6 untouched). See [Phase 7](#phase-7-8-scenario-synthetic-dataset-and-model). Not yet wired into the dashboard. | `dataset_generator_v2.py`, `expanded_synthetic_dataset_v2.csv`, `train_expanded_model_v2.py`, `predict_expanded_v2.py`, `accident_classifier_model_expanded_v2.joblib`, `evaluation_report_expanded_v2.json` |
| 8 | **ML vs traditional threshold detector** on the Phase 7 held-out data, with an explicit accident definition, train-only threshold calibration and a full comparison report. See [Phase 8](#phase-8-ml-vs-threshold-detector). Analysis only; not wired into the dashboard. | `threshold_detector.py`, `comparison_report.py`, `threshold_config_v1.json`, `comparison_report.json`, `comparison_report.csv` |
| 9 | **Rule-based severity estimator (`severity_v2`)**: LOW/MEDIUM/HIGH/CRITICAL from sensor evidence only. See [Phase 9](#phase-9-severity-v2). Not yet wired into the dashboard. | `severity_v2.py`, `severity_config_v2.json`, `severity_qualitative_v2.py`, `severity_qualitative_v2.json` |
| 10 | **Recording and replay**: every live run can be saved to `recordings/` as one versioned JSON file and replayed in a new REPLAY tab. See [Phase 10](#phase-10-recording-and-replay). | `recording_store.py`, `replay.py`, `replay_ui.py`, additive edits to `app.py` |
| cleanup | Foundation cleanup (no new behaviour of the model): git repository, `requirements.txt`, pytest suite, STM32 disconnect-reporting fix, non-blocking serial reads, this README. | `tests/`, `requirements*.txt`, `pytest.ini` |

The Phase 4 accuracy is not recorded in this repository; it is printed when
`train_expanded_model.py` runs (and **that script overwrites the committed
Phase 4 model**, so only re-run it deliberately).

## Current features

- Six-class event classifier (window-based Random Forest, 36 features, 1 s windows at 10 Hz).
- Rule-based severity (NORMAL / LOW / MEDIUM / HIGH) from the predicted class - not predicted by the ML model.
- Alert state machine: only HIGH severity (Severe Accident) starts a configurable (3-30 s) countdown; the user can cancel; otherwise it ends as a **simulated** "ALERT SENT".
- Two selectable sensor sources: simulator and STM32 over serial.
- Hardware fault reporting in the UI (port cannot be opened, unplugged mid-stream, board silent, garbled data) and a "waiting for data" state.
- Serial reads that never block the UI for more than ~one 0.1 s serial timeout per refresh.
- Automated tests (parsing, features, state machine, split integrity, model interface, source behaviour, headless app runs).

Placeholders / simulated parts - do not mistake these for real data:
GPS is a simulated random walk; in STM32 mode `vibration_level` (0.3) and
`speed_kmh` (0.0) are fixed placeholder constants, not measurements; the
emergency message is never actually sent.

## Phase 7: 8-scenario synthetic dataset and model

Phase 7 adds two simulated scenarios to the original six (Normal Driving, Pothole, Hard Braking, Sharp Turn, Minor Accident, Severe Accident). **Everything is synthetic.** The dashboard (`app.py`) still uses the Phase 4 six-class model; nothing in Phase 7 is connected to it yet.

| New scenario | Simulated signature (hand-written, simplified physics; details in `dataset_generator_v2.py`) |
|---|---|
| Rollover | Lateral skid/yaw, then a 0.8-1.8 s roll-rate pulse (200-450 deg/s) about the longitudinal axis. Roll angle is the integral of the rate and gravity rotates through the accelerometer's y-z plane, so the vehicle can end on its side or roof. Ground-contact impulses at each 90 deg, rising vibration, speed falling to 0-15 km/h. |
| Multi-Impact Collision | 2-4 distinct narrow impacts (2.5-7 g, random directions) 5-11 samples apart, smaller disturbances between them, speed dropping in a step at each impact. |

Files (all new; the Phase 4 dataset, models and training scripts are not modified):

| File | Role |
|---|---|
| `dataset_generator_v2.py` | Generates 8 scenarios x 100 recordings x 50 samples (10 Hz, 5 s) = 800 recordings / 40,000 rows, seed 200. The six original generators are reused unchanged from `dataset_generator.py`. `n` and `sample_rate_hz` are configurable (the new scenarios integrate with `dt = 1/rate`; `window_features.WINDOW_SIZE` stays 10 samples). |
| `expanded_synthetic_dataset_v2.csv` | Output; same columns as the Phase 4 CSV. |
| `train_expanded_model_v2.py` | Phase 4 methodology: stratified 80/20 split **by `recording_id` before windowing** (asserted disjoint, plus a check that no recording has windows on both sides), the unchanged 36 window features, `RandomForestClassifier(n_estimators=200, random_state=42)`. Prints and saves the evaluation. |
| `accident_classifier_model_expanded_v2.joblib` | New model (8 classes). |
| `evaluation_report_expanded_v2.json` | Machine-readable evaluation (metrics, confusion matrix, feature importances, counts). |
| `predict_expanded_v2.py` | Same `predict_window_event()` interface as `predict_expanded.py`, loads only the v2 model. |

Reproduce (the training script overwrites only the `*_v2` model and report):

```powershell
python dataset_generator_v2.py
python train_expanded_model_v2.py
```

Result of the committed run, on 160 held-out synthetic test recordings (800 windows): accuracy 0.825, macro-F1 0.823. For context, re-running Phase 4's recipe in memory on the Phase 4 dataset gives 0.817 accuracy over 6 classes. **Both are synthetic-vs-synthetic with different class sets and are not like-for-like, and neither says anything about real-world performance.** Per-class metrics, the confusion matrix and feature importances are in the training output and the JSON report. The weakest classes are Pothole and Normal Driving (a short event leaves most of its windows looking like normal driving, because each window inherits its recording's label).

Phase 7 limitations: the two new signatures are the author's assumptions rather than measured crashes; train and test recordings come from the same generator; the synthetic roll rates and impacts exceed the +/-250 deg/s and +/-2 g range the MPU6050 firmware configures; Phase 7 includes no threshold detector, severity mapping for the new classes, replay, or dashboard changes.

## Phase 8: ML vs threshold detector

Research question: *does multi-sensor ML classification discriminate accidents/events better than a conventional threshold detector?* Everything below is **synthetic data**; nothing here says how either approach would behave on real vehicles.

- `threshold_detector.py`: transparent rule-based detector. No ML, no labels at inference. Binary gate fires if ANY of: peak acceleration magnitude, peak gyro magnitude, mean vibration, or speed lost across the 1 s window exceeds its threshold (strictly greater). It returns `event_detected`, `classification` (`ACCIDENT`/`NORMAL`), `triggered_rules`, and `severity_inputs` (the measured value and threshold for each triggered rule). A secondary 7-way cascade (`event_class`) exists but can never output Multi-Impact Collision. No severity is assigned.
- **Accident definition (explicit):** accident = Minor Accident, Severe Accident, Rollover, Multi-Impact Collision. Non-accident = Normal Driving, Pothole, Hard Braking, Sharp Turn. The ML model's 8-class output is mapped to this binary the same way.
- **Threshold selection:** the four gate thresholds are the 99th percentile of each quantity over **training** windows of the four non-accident scenarios; cascade noise floors are the 99th percentile (1st for braking) over training Normal Driving windows; three cascade constants (roll-dominance 2.0, min roll rate 100 deg/s, severe level 6 g) are author-chosen from physical reasoning. The recipe was fixed before the test set was scored and never changed afterwards. The author also wrote the simulator, so the constants are not blind to it.
- **Fairness:** the identical Phase 7 split (640 train / 160 test recordings; 3200 / 800 windows). Both detectors get the same 36 window features, no labels. The ML accuracy is asserted equal to the Phase 7 report.

```powershell
python comparison_report.py      # calibrates on train, scores test once, writes the 3 output files
```

Outputs: `comparison_report.json` (full detail), `comparison_report.csv` (flat tables), `threshold_config_v1.json` (calibrated thresholds with provenance).

Test-set results (800 windows / 160 recordings; accident = positive):

| Level | Detector | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Window | ML | 400 | 400 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Window | Threshold | 206 | 387 | 13 | 194 | 0.741 | 0.941 | 0.515 | 0.666 | 0.033 | 0.485 |
| Recording (any window flagged) | ML | 80 | 80 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| Recording (any window flagged) | Threshold | 80 | 67 | 13 | 0 | 0.919 | 0.860 | 1.000 | 0.925 | 0.163 | 0.000 |

8-class (secondary): ML accuracy 0.825 / macro-F1 0.823; threshold cascade 0.426 / 0.406. Detection time (recording start to the end of the first flagged 1 s window; **not** referenced to true event onset, which the dataset does not store): ML median 1.0 s, threshold median 2.0 s, both detecting 80/80 accident recordings.

**How to read this: the ML result is not credible evidence of better event discrimination.** A diagnostic in the report shows the ML model flags 100% of the *first* window (samples 0-9, before the event) of Minor Accident, Rollover, Severe Accident and Multi-Impact recordings. The accident generators use higher baseline sensor noise and different speed ranges than the non-accident ones, and the model learned that simulator fingerprint rather than only the event. The threshold detector does not have this shortcut (it flags 5% of those Minor/Rollover first windows), which is why it misses the pre-event windows that the inherited labels call positive. The perfect ML binary score therefore overstates what it would do on real data, and the threshold detector's lower window-level recall is partly the label convention. The comparison does support narrower statements: on this simulator the threshold detector raises more false alarms on Hard Braking (45% of test recordings flagged) and Pothole (15%), and it cannot separate single-window accident subtypes. No claim of "AI is X% better" is made.

Phase 8 limitations: synthetic data and one generator for train and test; the synthetic rollover/impact signals exceed the +/-2 g and +/-250 deg/s MPU6050 ranges and are **not** clipped here (a real sensor would saturate; future hardware-realism work); in STM32 hardware mode vibration and speed are placeholders, which would disable two of the four rules; windows within a recording are correlated, so McNemar p-values are rough; 20 test recordings per class gives wide sampling uncertainty.

## Phase 9: severity_v2

`severity_v2.py` estimates how severe a 1-second window is from **measured sensor features only** (no ML, no scenario labels, no model probabilities; it imports no ML library). It is a **simulation/research index (0-100), not a certified medical or automotive severity scale.** The existing `severity_for_label()` lookup, the state machine and the dashboard are unchanged; `severity_v2` is not integrated yet (`from severity_v2 import assess, assess_window`).

Inputs (5 of the 36 window features): `accel_mag_max` (crash-pulse size), `gyro_mag_max` (rotation rate), speed lost in the window (`-speed_kmh_first_to_last`), `vibration_level_mean`, and `gyro_mag_mean` (only to detect *sustained* rotation).

Logic: each of the first four is normalised to 0-1 between an "ordinary" level and a "strong event" level, then `score = 100 x (0.35 accel + 0.30 gyro + 0.20 speed loss + 0.15 vibration)`. No weight reaches 0.5, so **one sensor alone can never exceed 35 points (MEDIUM)**: a huge acceleration spike with calm gyro/speed/vibration stays MEDIUM. Bands: <25 LOW, 25-50 MEDIUM, 50-75 HIGH, >=75 CRITICAL (lower bound inclusive). Rollover-like escalation: sustained rotation (`gyro_mag_mean` above its threshold) plus score >= 50 raises the level to CRITICAL.

`assess()` returns `severity`, `score`, `triggered_rules`, plain-English `reasons`, per-channel `evidence` (value, anchors, normalised value, weight, points, active) and a scale note, so "why HIGH?" is answerable.

Threshold sources (all recorded in `severity_config_v2.json`, `provenance`):
- **Training-derived** (`python severity_v2.py`, 640 training recordings only; the test half is never read): lower anchors = 99th percentile over training windows of Normal/Pothole/Hard Braking/Sharp Turn; upper anchors = 90th percentile over training windows of the four accident scenarios; sustained-rotation threshold = 99th percentile of `gyro_mag_mean` over the non-accident training windows. Scenario labels are used only inside this calibration function.
- **Engineering assumptions**: the weights, band cut-offs 25/50/75, "active channel" fraction 0.25, escalation score 50, and the **speed-loss anchors (14 and 35 km/h)**. Speed loss could not be training-derived: simulated Hard Braking loses more speed per second than accident windows (the percentile recipe gave upper < lower and the calibration refused it), so the anchors come from physics (about 0.4 g and about 1 g sustained for 1 s; beyond the tyre-friction limit a loss implies a collision).

Evaluation: there is **no ground-truth severity label** in the data, so no severity accuracy is computed or implied. What exists is (a) rule/unit tests, (b) the calibration summary above, and (c) `python severity_qualitative_v2.py`, which scores 30 fresh synthetic recordings per scenario (seed 9009, not in the dataset, not the test set) and writes `severity_qualitative_v2.json`. Each recording's highest-severity window: Normal Driving / Pothole / Hard Braking / Sharp Turn are all LOW; Minor Accident mostly MEDIUM (7 HIGH, 2 LOW); Multi-Impact mostly HIGH (19 of 30, 5 CRITICAL); Severe Accident and Rollover almost all CRITICAL (30/30 and 29/30). This is a plausibility check on a synthetic scale the author designed, not validation. A formal severity evaluation would need new, independently labelled severity data.

Limitations: synthetic data only; anchors are tied to this simulator's intensity scale; synthetic peaks exceed the MPU6050 firmware's +/-2 g / +/-250 deg/s range, which a real sensor would clip (so the upper anchors may be unreachable on hardware); in STM32 hardware mode vibration and speed are placeholders, leaving two of four channels uninformative; severity is estimated per window, with no memory across windows.

## Phase 10: recording and replay

**SIMULATION REPLAY. Recorded synthetic data - not real-world accident data.** Every run started in the LIVE SIMULATION tab (checkbox "Record this run", on by default) is saved to `recordings/<recording_id>.json` and can be replayed in the **REPLAY** tab. `recordings/` is git-ignored (generated runtime data); override the location with the `ACCIDENT_RECORDINGS_DIR` environment variable. No database or new dependency.

- `recording_store.py`: `RecordingStore` (`start_recording`, `save`, `load`, `list_recordings`, `delete`) and `Recorder` (`append_sample`, `append_decision`, `append_transition`, `finish`). Saves are atomic (temp file + rename) and never overwrite; ids are strictly validated, so `load`/`delete` cannot escape the folder. Corrupt files are reported as errors, never raised out of `list_recordings()`.
- **Schema (`schema_version: 1`)**, one JSON object: `recording_id` (`<UTC time>-<scenario slug>-<8 random hex>`), `created_utc`/`finished_utc`, `scenario`, `source_type` (`simulated`/`stm32`), `synthetic`, `sample_rate_hz`, `duration_s`, `software` and `models` (classifier / threshold-config / severity-config **file names + md5**, never paths), optional `alert` (`countdown_total_s`), and
  - `samples`: `t` (stream seconds), `wall_time_utc`, the 8 sensor channels, `gps [lat, lon]`;
  - `decisions`: per window, `sample_index` + `t` (sensor values are not copied) with `ai` (label, confidence, class probabilities), `threshold` (event_detected, classification, triggered_rules, event_class), `severity` (severity_v2 level, score, triggered rules), `legacy_severity`, `status`;
  - `state_transitions`: `from`, `to`, `t`, `sample_index`, `reason` (countdown started, cancelled, sent; also those that happen after the stream has ended);
  - `summary`: derived (scenario, duration, AI result and confidence, threshold result, max severity, peak acceleration and angular rate, final emergency status, alert outcome); recomputable from the rest.
  Loading rejects unknown/newer `schema_version`, missing required fields, non-finite values, non-chronological samples and bad references with a clear message; optional parts may be absent.
- `replay.py`: `ReplaySession` (READY/PLAYING/PAUSED/STOPPED/FINISHED; `play`, `pause`, `stop`, `reset`, `step`) emits the saved samples in order and reproduces the **recorded** AI, threshold, severity, GPS and emergency-state values; it predicts nothing and loads no model. `build_event_timeline()` lists AI event detected, threshold event detected, severity changes, GPS updates (one per second), countdown started, alert cancelled/sent.
- **Recorded vs recomputed:** everything in the replay is labelled `RECORDED RESULT`. The optional `Recompute now` panel (`recompute_decisions`, `compare_recorded_vs_recomputed`) re-runs the *current* models over the saved samples, is labelled `LIVE RECOMPUTATION`, never modifies the recording, and reports differences (e.g. after a model changed).
- **UI:** `app.py` gained a LIVE SIMULATION / REPLAY tab pair and recording hooks; live behaviour is unchanged. The REPLAY tab lists recordings (corrupt ones are shown in a collapsed "unreadable" list), shows metadata and summary, Replay / Pause / Step / Stop / Reset, speed 1-10x, current speed/acceleration/gyro/vibration/GPS/AI/threshold/severity/emergency state, sensor charts, the event timeline, and a confirmed Delete.

Limitations: the live run records the AI decision of the Phase 4 six-class model the dashboard uses (the Phase 7 eight-class model is not wired into the dashboard); the threshold detector and severity_v2 are recorded alongside but do not drive the alert state machine; GPS is the simulated random walk; STM32 recordings contain real accelerometer/gyro only (vibration and speed are placeholders); a recording is finalized only after the stream has ended and any alert countdown has resolved, so closing the browser mid-run loses that run; do not run a live simulation and a replay playback at the same time; replay pacing is approximate (UI refresh every 0.25 s).

## Repository layout

```
app.py                       Streamlit dashboard
sensor_source.py             SensorSource interface, SimulatedSensorSource, STM32SensorSource, parse_stm32_line
realtime_pipeline.py         rolling buffer, severity lookup, next_status(), simulated GPS
window_features.py           windowing + feature extraction (feature definitions)
common.py                    FEATURE_COLUMNS, label column, paths
predict_expanded.py          Phase 4 model wrapper used by the dashboard
dataset_generator.py         Phase 4 synthetic dataset / recording generator
train_*.py, predict*.py      Phase 2 / 3 / 4 training and prediction scripts
sensor_simulator.py          Phase 1 interactive simulator
stm32_firmware/              MPU6050 driver + USER CODE snippets for STM32CubeIDE
tests/                       pytest suite
requirements.txt             runtime dependencies (pinned)
requirements-dev.txt         runtime + pytest
```

## Setup (E: drive virtual environment)

The project lives in `E:\AI_Accident_Detection` and uses a virtual environment
in `E:\AI_Accident_Detection\.venv`. Python 3.13 was used for development.

PowerShell:

```powershell
cd E:\AI_Accident_Detection

# create (once)
python -m venv E:\AI_Accident_Detection\.venv

# activate (each new terminal)
E:\AI_Accident_Detection\.venv\Scripts\Activate.ps1
# if script execution is blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# (or use cmd.exe:  E:\AI_Accident_Detection\.venv\Scripts\activate.bat)

# install
python -m pip install --upgrade pip
pip install -r requirements.txt          # app + models
pip install -r requirements-dev.txt      # same, plus pytest (needed to run the tests)
```

`scikit-learn` is pinned exactly because the `.joblib` files are pickles that
can fail or misbehave under a different version. Without activating, you can
always call the interpreter directly: `E:\AI_Accident_Detection\.venv\Scripts\python.exe ...`.

## Running the simulator

Interactive Phase 1 generator (pick a scenario, print readings, save a CSV):

```powershell
python sensor_simulator.py
```

Self-contained STM32 parsing/retry check with a fake serial port (no hardware):

```powershell
python sensor_source.py
```

The live simulator is the dashboard in *Simulated* mode (next section). Each
**Start** generates a fresh randomized 5 s recording and streams it one sample
at a time.

## Running the Streamlit dashboard

```powershell
cd E:\AI_Accident_Detection
streamlit run app.py
```

Open the URL it prints (normally <http://localhost:8501>). Choose a data source
in the sidebar, pick a scenario (simulated mode), set the alert countdown, and
press **Start**. **Stop** halts the stream; **Reset** clears all state.

The dashboard loads `accident_classifier_model_expanded.joblib` read-only; it
never retrains or modifies models.

## Running the tests

```powershell
pip install -r requirements-dev.txt
python -m pytest                    # everything (~80 s)
python -m pytest -m "not slow"      # skip the retraining check and the headless-app tests
```

What is covered: `parse_stm32_line()` (valid, malformed, missing/invalid
values, range limits); `STM32SensorSource` with fake serial ports (bad lines,
unplug, silent board, bounded `read()` time, error messages); feature
extraction against hand-computed values; `next_status()` and the alert
countdown; recording-level train/test separation (including running
`train_expanded_model.main()` with its output redirected to a temp file); the
committed Phase 4 model's interface; and headless Streamlit runs of
`app.py` with a faked serial port.

Notes:
- Tests check interfaces and logic, **not model accuracy**, and never overwrite the committed CSVs or `.joblib` files (one test asserts this by hashing them).
- The `test_app_smoke.py` tests create many short-lived asyncio event loops; on some Windows hosts (and in sandboxes that block loopback sockets) that intermittently raises `PermissionError: [WinError 10013]`. The fixture retries that call, but if it persists it is an environment limit, not an app failure.
- All serial hardware in the tests is faked; **no test touches real STM32 hardware.**

## STM32 integration

Hardware path: **MPU6050** (I2C1) -> **STM32** -> **USART2** (USB virtual COM port or USB-UART adapter) -> PC -> `STM32SensorSource`.

1. In STM32CubeIDE create a project for your board, enable **I2C1** and **USART2**, and add `stm32_firmware/mpu6050.c` / `.h` to `Core/Src` / `Core/Inc`.
2. Paste the blocks in `stm32_firmware/main_user_code.c` into the matching `USER CODE` sections of the generated `main.c`. (That file is a copy-paste reference, not a standalone project.)
3. The firmware configures the MPU6050 for +/-2 g and +/-250 deg/s and sends one line every ~100 ms (about 10 Hz, matching the 10 Hz the model was trained on).
4. Set USART2 to **115200 baud** (the Python default) - or pass a different `baudrate=` to `STM32SensorSource`.
5. Find the COM port in Windows Device Manager -> *Ports (COM & LPT)*. Close any other program using it (serial monitors, the IDE console).
6. In the dashboard choose **STM32 (Serial, real accel/gyro)**, enter the port (e.g. `COM5`), press **Start**.

How faults appear in the dashboard (red "SENSOR SOURCE PROBLEM" banner, taken from `STM32SensorSource.last_error`):

| Situation | Message starts with |
|-----------|---------------------|
| pyserial missing | `pyserial is not installed` |
| port wrong / busy / absent | `Could not open serial port 'COMx'` |
| cable pulled mid-run | `Serial read failed on COMx (disconnected?)` |
| board powered but silent / wrong port | `No valid data from the STM32 on COMx for 5 s` |
| wrong baud rate / noise | `Gave up after 20 consecutive unreadable/malformed lines` |

A single garbled line is skipped silently; a brief pause shows "waiting for data".

**Verification status:** the Python side is tested only against *fake* serial
ports. The firmware has been written and reviewed but **not compiled or run on
real hardware** by this project's tooling. Treat the first real hardware run as
the actual test.

## Serial format

ASCII, one sample per line, comma-separated, terminated by `\r\n`:

```
timestamp_ms,accel_x_g,accel_y_g,accel_z_g,gyro_x_dps,gyro_y_dps,gyro_z_dps
1234,0.02,-0.01,1.01,2.3,-1.7,4.2
```

- Exactly 7 numeric fields. `timestamp_ms` is parsed and then dropped.
- Accelerations in g, angular rates in degrees/second.
- A line is rejected (skipped) if it is empty, has the wrong field count, contains a non-numeric/missing value or `nan`/`inf`, is not ASCII, or has `|accel| > 20 g` or `|gyro| > 3000 deg/s` (corruption checks, not physics limits).
- `vibration_level` and `speed_kmh` are not on the wire; the host fills them with the fixed placeholders 0.3 and 0.0 so the reading has all model input columns.

## Phase 10.5 -- Virtual hardware lab

A third data source, **Virtual Hardware (Phase 10.5)**, plus a **VIRTUAL HARDWARE LAB** tab with a
schematic-style, animated view of the project's hardware architecture:

```
VirtualVehicle -> MPU6050 (registers, 0x68) / vibration sensor / GPS
   -> I2C1 -> virtual STM32 (boot, 10 Hz polling) -> USART2 telemetry line
   -> VirtualHardwareSensorSource -> RollingWindowBuffer -> window features -> Random Forest
      + threshold detector + severity_v2 -> emergency state machine -> recording / replay -> UI
   (STM32 -> USART1 -> ESP32/GSM stub: status text only)
```

Code: `virtual_hardware/` (`vehicle`, `mpu6050`, `vibration_sensor`, `gps`, `i2c`, `uart`, `stm32`, `state`,
`source`, plus `schematic` / `inspector` / `lab_ui` for the UI). Tests: `tests/test_virtual_hardware*.py`.

* It is a **project-specific functional simulation, not a cycle-accurate STM32 emulator**; the C firmware in
  `stm32_firmware/` is not executed. GPS is simulated and **no real SMS/GSM message is ever sent**.
* The scenario selects what the virtual *vehicle* does; the classifier only sees the sensor readings that came
  out of the virtual UART. Eight scenarios: the six Phase 4 ones plus Rollover and Multi-Impact Collision.
* The MPU6050 model is register-oriented (WHO_AM_I, config, sample-rate divider, output registers, sleep bit,
  clear-on-read INT_STATUS) and enforces its +-2 g / +-250 deg/s range: values beyond it are clipped, a
  saturation flag is set (shown as SATURATED) and the pre-clipping value is kept for diagnostics.
* **Known train/live mismatch:** the Phase 7 model was trained on synthetic signals that exceed those limits
  (impacts of several g), so clipped virtual-hardware data is outside its training distribution. The model was
  deliberately **not** retrained; predictions in this mode are exploratory (a later research phase should
  address this). Accident scenarios are often labelled "Hard Braking" by the AI while severity_v2 still rates
  them HIGH/CRITICAL -- shown as-is.
* The simulation clock is deterministic: the same seed and scenario give an identical run (10 Hz, 100 ms/tick).
  Sidebar controls: Start, Pause, Reset, **Step one tick**, seed. Recordings use `source_type = "virtual_hardware"`
  with optional `virtual_hardware` / per-sample `hw` blocks (schema version unchanged; old recordings still load).

## Phase 11 -- Controlled experiment runner

`experiments/` measures the **frozen** ML classifier and the **frozen** Phase 8 threshold detector on the *same*
virtual-hardware recordings. Nothing is trained, tuned or calibrated; the model, `threshold_config_v1.json` and
`severity_config_v2.json` are SHA-256 hashed before/after and the run aborts if any changed.

```
python -m experiments.experiment_runner                      # experiments/experiment_config.json (8 x 50 = 400 recordings, ~12 min)
python -m experiments.experiment_runner --runs-per-scenario 3 --out experiments/results_dev
```

* One recording per (scenario, seed) through vehicle -> MPU6050 -> I2C -> STM32 -> UART; both detectors get the
  identical sliding 10-sample windows. Seeds: scenario *i* uses `11000 + 100*i + k`. Results go to
  `experiments/results/<experiment_id>/` (never overwritten) together with the Phase 10 recordings, so every run can be replayed.
* **Ground truth** = the scenario and the virtual vehicle's own state, never a detector. **Event onset** = first sample
  with vehicle impact envelope `shock > 0.1`; latency = (first positive window ending at/after onset - onset) x 100 ms in
  *simulation* time. A positive before onset is a `PRE_EVENT_FALSE_ALARM`: never a detection, never a (negative) latency.
* Recording-level and window-level metrics are reported separately (definitions in the summary file).
* Rollover and Multi-Impact Collision are **not classes of the ML model**: its actual output is reported, they are not scored
  as classes, and only enter the binary accident / non-accident metrics. No 8-class ML confusion matrix is produced.
* **RESEARCH LAB** tab: shows the result files (no placeholder numbers; runs an experiment only on button press).
* Synthetic data, finite recordings, paired design: results describe the virtual sensor environment, not real-world accuracy.

## Research-integrity limitations

- **All ML results are from synthetic data.** Training and test data come from hand-written parametric generators (noise plus smooth event bumps). They encode the author's assumptions, not measured physics.
- The Phase 4 test set contains held-out *recordings* but from the *same generator*; it shows generalisation across synthetic recordings only, not to real vehicles, sensors, mounting positions, road conditions or real crashes.
- Phase 2 and Phase 3 figures come from tiny samples (Phase 3: 6 test windows) and are not statistically meaningful.
- **Severity is a fixed lookup table**, not learned or validated.
- In STM32 mode the model receives real accelerometer/gyroscope values but **placeholder** vibration and speed values, and it was trained with realistic values for those channels - so its predictions on hardware are exploratory and may be arbitrary.
- GPS is simulated; the emergency alert is simulated; no SMS/GSM/e-call path exists.
- The firmware and the serial link have not been validated on real hardware in this repository's tooling.
- The system is not certified, not safety-rated, and must not be relied on to detect real accidents or summon help.
