"""
Streamlit glue for the VIRTUAL HARDWARE LAB tab. All drawing lives in schematic.py / panels.py /
inspector.py (pure, testable); this module only lays the pieces out:

    header + badges
    scenario bar
    [ hardware canvas (hero) | AI analysis / emergency state / component inspector ]
    [ oscilloscope (one channel at a time) ]
    [ collapsed: UART telemetry | I²C bus log ]
    [ collapsed: simulation limitations ]

Simulation controls (Start / Pause / Reset / Step, seed, countdown, recording) are in the sidebar.
"""

import streamlit as st

from virtual_hardware.inspector import COMPONENTS, inspector_rows
from virtual_hardware.panels import PANEL_CSS, header_html, ai_html, emergency_html, inspector_html
from virtual_hardware.schematic import build_lab_html, build_scope_html, SCOPE_CHANNELS
from virtual_hardware.vehicle import VIRTUAL_SCENARIOS

CANVAS_HEIGHT = 700          # fallback only; st.iframe(height="content") sizes itself to the schematic
SCOPE_HEIGHT = 200

DISCLAIMER = ("This is a project-specific functional simulation, not a cycle-accurate STM32 emulator. "
              "The C firmware in stm32_firmware/ is not executed; its behaviour (I2C polling, UART telemetry) "
              "is re-implemented in Python.")
SATURATION_NOTE = ("MPU6050 range limits are modelled (±2 g / ±250 °/s). The Phase 7 model was trained on synthetic "
                   "signals that exceed these limits (e.g. impacts of several g), so clipped virtual-hardware data "
                   "creates a train/live distribution mismatch. The model is NOT retrained in this phase; "
                   "predictions here are exploratory and do not show real-world accuracy.")


def _iframe(html, height):
    if hasattr(st, "iframe"):                       # st.components.v1.html is deprecated in recent Streamlit
        st.iframe(html, height="content")
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)


def render_limitations():
    with st.expander("Simulation limitations"):
        st.markdown("**VIRTUAL HARDWARE SIMULATION.** " + DISCLAIMER)
        st.markdown("GPS is **SIMULATED**. The emergency alert is a **SIMULATED EMERGENCY ALERT**: "
                    "**NO REAL SMS/GSM MESSAGE WAS SENT**. All data is synthetic.")
        st.markdown(SATURATION_NOTE)


def render_lab(source, mode_active, scenario, running, paused, pipeline, history, on_scenario, on_cancel):
    """
    source       : a VirtualHardwareSensorSource (the live one, or an idle power-on one)
    mode_active  : True when "Virtual Hardware" is the selected data source
    pipeline     : {"label","confidence","severity","threshold","status","countdown","run_state"} from the real pipeline
    """
    snapshot = source.rig.snapshot()
    st.markdown(PANEL_CSS + header_html(snapshot, pipeline, mode_active), unsafe_allow_html=True)

    if not mode_active:
        st.info("Showing the virtual hardware at power-on. Choose **Virtual Hardware (Phase 10.5)** under "
                "Data source in the sidebar, then press Start (or Step one tick) to run it.")

    main, side = st.columns([3.1, 1.15], gap="medium")
    with side:
        if mode_active:
            # keep the (keyed) dropdown in step with a scenario changed from the sidebar
            if st.session_state.get("lab_scenario") != scenario:
                st.session_state["lab_scenario"] = scenario
            st.selectbox("Scenario", VIRTUAL_SCENARIOS, key="lab_scenario",
                         on_change=lambda: on_scenario(st.session_state["lab_scenario"]))
        selected = st.selectbox("Inspect component", COMPONENTS, index=COMPONENTS.index("MPU6050"), key="lab_inspect")
        st.markdown(ai_html(pipeline) + emergency_html(snapshot, pipeline), unsafe_allow_html=True)
        if mode_active and pipeline.get("status") == "ALERT COUNTDOWN":
            st.button("Cancel simulated alert", key="lab_cancel", on_click=on_cancel, use_container_width=True)
        st.markdown(inspector_html(inspector_rows(snapshot, selected), selected), unsafe_allow_html=True)
    with main:
        _iframe(build_lab_html(snapshot, pipeline, selected, animate=bool(running)), CANVAS_HEIGHT)

    st.markdown('<div class="hw-sub" style="margin:6px 0 2px 0;font-weight:700;letter-spacing:1px">'
                'LIVE SENSOR OSCILLOSCOPE</div>', unsafe_allow_html=True)
    channel = st.radio("Oscilloscope channel", list(SCOPE_CHANNELS), horizontal=True, key="lab_scope_channel",
                       label_visibility="collapsed")
    _iframe(build_scope_html(history, channel), SCOPE_HEIGHT)

    left, right = st.columns(2)
    with left:
        with st.expander("UART TELEMETRY"):
            u = snapshot["uart"]
            st.markdown(f"**{u['name']}** @ {u['baud']} baud · **{u['packets']}** packets · "
                        f"{u['wire_time_ms']:.1f} ms per packet on the wire")
            st.code("\n".join(u["recent"]) or "(no packets yet)", language=None)
            st.caption("timestamp_ms, accel x/y/z (g), gyro x/y/z (°/s), vibration, speed (km/h), lat, lon, gps_fix")
    with right:
        with st.expander("I²C BUS LOG"):
            i = snapshot["i2c"]
            st.markdown(f"**{i['bus']}** · devices: " + ", ".join(f"0x{a:02X}" for a in i["devices"])
                        + f" · **{i['count']}** transactions")
            st.code("\n".join(i["recent"]) or "(no transactions yet)", language=None)
            if i["last_phases"]:
                st.caption("last transaction on the wire: " + " → ".join(i["last_phases"]))
    render_limitations()
