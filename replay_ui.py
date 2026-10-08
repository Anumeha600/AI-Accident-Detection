"""
PHASE 10 -- Streamlit UI for the REPLAY tab (rendered by app.py).
------------------------------------------------------------------------------
SIMULATION REPLAY. Recorded synthetic data - not real-world accident data.

Everything shown from a loaded recording is the RECORDED RESULT: nothing is
predicted here. The optional "LIVE RECOMPUTATION" panel is separate and says
so. The pure logic lives in replay.py / recording_store.py; this file only
draws widgets.
"""

import time
from typing import Any, Dict, Optional

import streamlit as st

from recording_store import RecordingError, RecordingStore
from replay import (
    FINISHED, PAUSED, PLAYING, READY, STOPPED, RECORDED_LABEL, RECOMPUTED_LABEL,
    REPLAY_BANNER, SYNTHETIC_NOTICE, ReplaySession, compare_recorded_vs_recomputed,
)

REPLAY_TICK_SECONDS = 0.25      # wall time between UI refreshes while playing
SPEEDS = [1, 2, 5, 10]


def _fmt(value: Optional[float], spec: str = ".2f", unit: str = "") -> str:
    return "-" if value is None else f"{value:{spec}}{unit}"


def _option_label(row: Dict[str, Any]) -> str:
    return (f"{row['created_utc'][:19].replace('T', ' ')} | {row['scenario']} | "
            f"{row['duration_s']:.1f}s | {row['recording_id']}")


def _load_session(store: RecordingStore, recording_id: str) -> None:
    st.session_state.replay_error = None
    st.session_state.replay_compare = None
    try:
        st.session_state.replay_session = ReplaySession(store.load(recording_id))
    except RecordingError as exc:
        st.session_state.replay_session = None
        st.session_state.replay_error = str(exc)
    st.session_state.replay_id = recording_id


def render(store: RecordingStore) -> None:
    for key, default in (("replay_session", None), ("replay_id", None), ("replay_error", None),
                         ("replay_compare", None)):
        if key not in st.session_state:
            st.session_state[key] = default

    st.warning(f"{REPLAY_BANNER} -- {SYNTHETIC_NOTICE} This is a replay of a saved run, not a real accident. "
               "Hardware (STM32) recordings contain real accelerometer/gyro values only; vibration, speed, GPS "
               "and the emergency alert are placeholders/simulated.")

    rows = store.list_recordings()
    good = [r for r in rows if not r.get("error")]
    bad = [r for r in rows if r.get("error")]
    if bad:
        with st.expander(f"{len(bad)} unreadable recording file(s) (skipped)"):
            for r in bad:
                st.write(f"{r['file']}: {r['error']}")
    if not good:
        st.info("No saved recordings yet. Run a simulation in the LIVE SIMULATION tab (with 'Record this run' "
                "ticked) and it will appear here.")
        return

    ids = [r["recording_id"] for r in good]
    labels = {r["recording_id"]: _option_label(r) for r in good}
    current = st.session_state.replay_id if st.session_state.replay_id in ids else ids[0]
    selected = st.selectbox("Saved recording", ids, index=ids.index(current), format_func=labels.get,
                            key="replay_selected_id")
    if selected != st.session_state.replay_id or (st.session_state.replay_session is None
                                                  and st.session_state.replay_error is None):
        _load_session(store, selected)

    session: Optional[ReplaySession] = st.session_state.replay_session
    if session is None:
        st.error(f"Cannot load this recording: {st.session_state.replay_error}")
        return

    rec = session.recording
    summary = rec.summary

    # --- summary ----------------------------------------------------------------
    st.subheader(f"Recording summary ({RECORDED_LABEL})")
    s1 = st.columns(4)
    s1[0].metric("Scenario", str(summary["scenario"]))
    s1[1].metric("Duration", _fmt(summary["duration_s"], ".1f", " s"))
    s1[2].metric("AI result", str(summary["ai_result"] or "-"))
    s1[3].metric("AI confidence", _fmt(summary["ai_confidence"], ".1%") if summary["ai_confidence"] else "-")
    s2 = st.columns(4)
    s2[0].metric("Threshold detector", {True: "Event detected", False: "No event", None: "-"}[
        summary["threshold_event_detected"]])
    s2[1].metric("Max severity", f"{summary['severity_max'] or '-'}"
                 + (f" ({summary['severity_max_score']:.0f})" if summary["severity_max"] else ""))
    s2[2].metric("Peak acceleration", _fmt(summary["peak_accel_g"], ".2f", " g"))
    s2[3].metric("Peak angular rate", _fmt(summary["peak_gyro_dps"], ".0f", " deg/s"))
    st.write(f"**Emergency status:** {summary['emergency_final_status'] or '-'}"
             f"  |  **Alert outcome:** {summary['alert_outcome'] or 'none'}"
             f"  |  **Source:** {summary['source_type']}  |  **Sample rate:** {rec.sample_rate_hz:g} Hz")
    with st.expander("Metadata (schema, software, models)"):
        d = rec.data
        st.json({"recording_id": rec.recording_id, "schema_version": d["schema_version"],
                 "created_utc": d["created_utc"], "finished_utc": d.get("finished_utc"),
                 "synthetic": d.get("synthetic", True), "software": d.get("software"),
                 "models": d.get("models"), "alert": d.get("alert"), "samples": summary["n_samples"],
                 "decisions": summary["n_decisions"], "state_transitions": summary["n_transitions"]})

    # --- controls ------------------------------------------------------------------
    c = st.columns([1, 1, 1, 1, 1, 2])
    speed = c[5].selectbox("Speed", SPEEDS, index=0, format_func=lambda x: f"{x}x", key="replay_speed")
    if c[0].button("Replay", key="replay_btn_play"):
        session.reset()
        session.play()
    if c[1].button("Pause", key="replay_btn_pause", disabled=session.status != PLAYING):
        session.pause()
    if c[2].button("Step", key="replay_btn_step", disabled=session.status == FINISHED):
        if session.status in (READY, STOPPED):
            session.status = PAUSED
        session.step(1)
    if c[3].button("Stop", key="replay_btn_stop", disabled=session.status in (READY, FINISHED, STOPPED)):
        session.stop()
    if c[4].button("Reset", key="replay_btn_reset"):
        session.reset()

    if session.status == PLAYING:
        step = max(1, round(REPLAY_TICK_SECONDS * speed * rec.sample_rate_hz))
        session.step(step)
    elif session.status == PAUSED and session.position == 0:
        pass

    # --- recorded state --------------------------------------------------------------
    state = session.state_at()
    st.progress(session.position / session.n_samples if session.n_samples else 0.0,
                text=f"{REPLAY_BANNER} -- {RECORDED_LABEL} -- {session.status} -- sample "
                     f"{session.position}/{session.n_samples}")
    if session.n_samples == 0:
        st.info("This recording contains no samples.")
        return

    sample = state.get("sample")
    m1 = st.columns(5)
    m1[0].metric("Elapsed", _fmt(state["elapsed_s"], ".1f", " s"))
    m1[1].metric("Speed", _fmt(sample["speed_kmh"], ".1f", " km/h") if sample else "-")
    m1[2].metric("Vibration", _fmt(sample["vibration_level"], ".2f") if sample else "-")
    m1[3].metric("Accel (x,y,z g)", f"{sample['accel_x_g']:.2f}, {sample['accel_y_g']:.2f}, {sample['accel_z_g']:.2f}"
                 if sample else "-")
    m1[4].metric("Gyro (x,y,z deg/s)", f"{sample['gyro_x_dps']:.0f}, {sample['gyro_y_dps']:.0f}, "
                 f"{sample['gyro_z_dps']:.0f}" if sample else "-")
    gps = state["gps"]
    m2 = st.columns(5)
    m2[0].metric("GPS (recorded)", f"{gps[0]:.5f}, {gps[1]:.5f}" if gps else "-")
    ai = state["ai"]
    m2[1].metric("AI classification", ai["label"] if ai else "-")
    m2[2].metric("AI confidence", f"{ai['confidence']:.1%}" if ai else "-")
    th = state["threshold"]
    m2[3].metric("Threshold detector", ("ACCIDENT" if th["event_detected"] else "normal") if th else "-")
    sv = state["severity"]
    m2[4].metric("Severity (v2)", f"{sv['severity']} ({sv['score']:.0f})" if sv else "-")
    if th and th["triggered_rules"]:
        st.caption("Threshold rules triggered: " + ", ".join(th["triggered_rules"]))
    if sv and sv.get("triggered_rules"):
        st.caption("Severity evidence: " + ", ".join(sv["triggered_rules"]))
    status = state["status"]
    {"ALERT COUNTDOWN": st.error, "ALERT SENT": st.error, "ALERT CANCELLED": st.warning}.get(status, st.info)(
        f"Emergency state (recorded): {status}")

    # --- sensor timeline -----------------------------------------------------------------
    table = session.sensor_table()
    if not table.empty:
        st.subheader("Sensor timeline (recorded)")
        t = table.set_index("t")
        st.line_chart(t[["accel_mag_g", "gyro_mag_dps"]])
        st.line_chart(t[["speed_kmh", "vibration_level"]])

    # --- event timeline ----------------------------------------------------------------------
    st.subheader("Event timeline (recorded)")
    shown = state["events"]
    if shown:
        st.dataframe([{"t (s)": round(e["t"], 2), "event": e["text"], "kind": e["kind"]} for e in shown],
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No events yet.")
    with st.expander(f"Full recorded timeline ({len(session.timeline)} events)"):
        st.dataframe([{"t (s)": round(e["t"], 2), "event": e["text"], "kind": e["kind"]} for e in session.timeline],
                     use_container_width=True, hide_index=True)

    # --- separate, clearly labelled recomputation ----------------------------------------------
    with st.expander(f"{RECOMPUTED_LABEL} (optional; separate from the recorded result)"):
        st.caption("Re-runs the CURRENT AI model, threshold detector and severity_v2 on the saved samples and "
                   "compares with what was recorded. It does not change the recording or the replay above.")
        if st.button("Recompute now", key="replay_btn_recompute"):
            st.session_state.replay_compare = compare_recorded_vs_recomputed(rec)
        cmp_ = st.session_state.replay_compare
        if cmp_:
            st.write(f"{cmp_['source']}: compared {cmp_['decisions_compared']} decisions -- "
                     + ("identical." if cmp_["identical"] else f"{len(cmp_['mismatches'])} difference(s)."))
            if cmp_["mismatches"]:
                st.dataframe(cmp_["mismatches"][:200], hide_index=True)

    # --- delete ----------------------------------------------------------------------------------
    with st.expander("Delete this recording"):
        confirm = st.checkbox("I want to permanently delete this recording", key="replay_confirm_delete")
        if st.button("Delete", key="replay_btn_delete", disabled=not confirm):
            try:
                store.delete(rec.recording_id)
            except RecordingError as exc:
                st.error(str(exc))
            else:
                st.session_state.replay_session = None
                st.session_state.replay_id = None
                st.rerun()

    if session.status == PLAYING:
        time.sleep(REPLAY_TICK_SECONDS)
        st.rerun()
