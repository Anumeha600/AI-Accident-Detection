"""
PHASE 11 -- fixed definitions shared by the runner, the evaluation and the reports.

GROUND TRUTH comes ONLY from the simulation: the scenario that was set on the virtual vehicle and the
vehicle's own per-tick state (`VehicleState.shock`). Neither the ML model nor the threshold detector is
ever consulted to decide what is "an accident" or "when the event starts".

EVENT ONSET (documented definition, fixed before any result was looked at)
    onset_tick = first sample whose virtual-vehicle impact envelope `shock` exceeds ONSET_SHOCK_THRESHOLD
    event_end_tick = last such sample.
    `shock` is the Gaussian impact/brake/turn/pothole envelope of the vehicle model (0 = nothing happening,
    ~1 = the peak of a typical event), so onset is "the vehicle's event is measurably under way", not the
    event's peak. Ticks are 0-based sample indices; time_ms = tick * 100 (10 Hz), i.e. SIMULATION time.

WINDOW PHASES (a window is the 10 samples ending at tick `end`, sliding with stride 1, as in the app)
    PRE_EVENT   end < onset_tick                       (event has not started yet)
    EVENT       end >= onset_tick and start <= event_end_tick   (window touches the event)
    POST_EVENT  start > event_end_tick                 (event is over; excluded from window-level binary metrics)
    NO_EVENT    the scenario has no event (Normal Driving)
"""

ONSET_SHOCK_THRESHOLD = 0.1
SAMPLE_PERIOD_MS = 100

ACCIDENT_SCENARIOS = ("Minor Accident", "Severe Accident", "Rollover", "Multi-Impact Collision")
NON_ACCIDENT_SCENARIOS = ("Normal Driving", "Pothole", "Hard Braking", "Sharp Turn")
ALL_SCENARIOS = NON_ACCIDENT_SCENARIOS + ACCIDENT_SCENARIOS

# explicit scenario -> binary ground truth (1 = accident)
GROUND_TRUTH = {**{s: 1 for s in ACCIDENT_SCENARIOS}, **{s: 0 for s in NON_ACCIDENT_SCENARIOS}}

PRE_EVENT, EVENT, POST_EVENT, NO_EVENT = "PRE_EVENT", "EVENT", "POST_EVENT", "NO_EVENT"

METHODS = ("ml", "threshold")

RECORDING_COLUMNS = [
    "experiment_id", "recording_id", "scenario", "seed", "ground_truth",
    "event_onset_tick", "event_onset_ms", "event_end_tick", "sample_count", "window_count",
    "saturated_ticks", "saturated_in_event",
    "ml_top_class", "ml_max_confidence", "ml_binary_decision", "ml_detection_tick", "ml_detection_ms",
    "ml_latency_ms", "ml_pre_event_false_alarm", "ml_positive_windows",
    "threshold_binary_decision", "threshold_detection_tick", "threshold_detection_ms", "threshold_latency_ms",
    "threshold_pre_event_false_alarm", "threshold_positive_windows",
]
