"""
Engineering-simulator style HTML/SVG view of the virtual hardware (original artwork, no third-party assets).

build_lab_html(snapshot, pipeline, selected, animate) -> self-contained HTML document (status bar + schematic)
build_scope_html(history, channel)                   -> one-channel oscilloscope document
Both are pure functions rendered inside an iframe by lab_ui. Everything drawn is derived from the rig
snapshot (vehicle state, sensor values, bus/UART activity, MCU and alert state); nothing is decorative
state. Colours are semantic: green = normal, cyan = active data, amber = warning/event, red = emergency,
gray = inactive.

    pipeline : {"label","confidence","severity","status","countdown"} (display only)
    animate  : False freezes the moving data packets (paused / idle)
"""

from html import escape

VB_W, VB_H = 860, 760

C = {"bg": "#0b0f14", "card": "#10161d", "line": "#26333d", "txt": "#d6e2ea", "dim": "#7b8d99",
     "green": "#2ecc71", "cyan": "#3ec9ff", "amber": "#ffb020", "red": "#ff4d5e", "gray": "#4a5863"}

CSS = f"""
:root{{--bg:{C['bg']};--card:{C['card']};--line:{C['line']};--txt:{C['txt']};--dim:{C['dim']}}}
*{{box-sizing:border-box}}
html,body{{margin:0;background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden}}
.bar{{display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;padding:8px 12px;border-bottom:1px solid var(--line);
font-size:12px;color:var(--dim);flex:0 0 auto}}
.bar b{{color:var(--txt);font-weight:600}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}}
.chip{{margin-left:auto;padding:3px 10px;border:1px solid;border-radius:3px;font-weight:700;letter-spacing:.5px}}
svg{{display:block;width:100%;height:auto}}
text{{font-family:'Segoe UI',system-ui,sans-serif}}
.card{{fill:var(--card);stroke:var(--line);stroke-width:1}}
.sel{{stroke:{C['cyan']};stroke-width:2}}
.ttl{{fill:var(--txt);font-size:14px;font-weight:700;letter-spacing:.6px}}
.sub{{fill:var(--dim);font-size:10.5px;letter-spacing:.8px}}
.lab{{fill:var(--dim);font-size:12px}}
.val{{fill:var(--txt);font-size:13px;font-family:Consolas,'Courier New',monospace}}
.wl{{fill:var(--dim);font-size:11px;letter-spacing:.8px}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
@keyframes vshake{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-2px)}}}}
@keyframes ishake{{0%,100%{{transform:translate(0,0)}}25%{{transform:translate(-3px,2px)}}75%{{transform:translate(3px,-2px)}}}}
.pulse{{animation:pulse .7s infinite}}.vshake{{animation:vshake .2s infinite}}.ishake{{animation:ishake .12s infinite}}
"""


def _t(x, y, text, cls="lab", anchor="start", fill=None, pulse=False):
    f = f' fill="{fill}"' if fill else ""
    return f'<text x="{x}" y="{y}" class="{cls}{" pulse" if pulse else ""}" text-anchor="{anchor}"{f}>{escape(str(text))}</text>'


def _dot(x, y, color, pulse=False, r=4):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"{' class="pulse"' if pulse else ''}/>'


def _card(x, y, w, h, title, sub, key, selected, color):
    cls = "card sel" if key == selected else "card"
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" class="{cls}"/>'
            + _dot(x + 14, y + 18, color) + _t(x + 26, y + 22, title, "ttl") + _t(x + 26, y + 36, sub, "sub"))


def _kv(x, y, label, value, color=None, vx=62):
    return _t(x, y, label, "lab") + _t(x + vx, y, value, "val", fill=color)


def _packets(path, count, dur, tick, color, on):
    if not on:
        return ""
    out = []
    for i in range(count):
        begin = -((tick * 0.1 + i * dur / count) % dur)      # offset by tick so the phase is continuous across redraws
        out.append(f'<circle r="3.5" fill="{color}"><animateMotion dur="{dur}s" begin="{begin:.2f}s" '
                   f'repeatCount="indefinite" path="{path}"/></circle>')
    return "".join(out)


def _wire(path, active, color, label="", lx=0, ly=0, anchor="start", dashed=False, arrow=True):
    col = color if active else C["gray"]
    mk = f' marker-end="url(#ar-{"on" if active else "off"}-{color[1:]})"' if arrow else ""
    dash = ' stroke-dasharray="4 4"' if dashed else ""
    out = f'<path d="{path}" fill="none" stroke="{col}" stroke-width="{2 if active else 1.5}"{dash}{mk}/>'
    if label:
        out += _t(lx, ly, label, "wl", anchor)
    return out


def _markers():
    defs = []
    for col in (C["cyan"], C["red"]):
        for state, fill in (("on", col), ("off", C["gray"])):
            defs.append(f'<marker id="ar-{state}-{col[1:]}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
                        f'markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10z" fill="{fill}"/></marker>')
    return "<defs>" + "".join(defs) + "</defs>"


# --------------------------------------------------------------------------- components
def _vehicle(s, selected):
    v = s["vehicle"] or {}
    state = v.get("vehicle_state", "STOPPED")
    kind, shock = v.get("event_kind", "none"), v.get("shock", 0.0)
    heading, roll, pitch = v.get("heading_deg", 90.0), v.get("roll_deg", 0.0), v.get("pitch_deg", 0.0)
    speed, acc = v.get("speed_kmh", 0.0), v.get("acceleration_long_g", 0.0)
    col = C["red"] if state in ("COLLISION", "ROLLOVER") else C["amber"] if state in (
        "POTHOLE IMPACT", "HARD BRAKING", "TURNING", "STOPPED (CRASHED)", "POST-IMPACT") else C["green"]
    shake = ("ishake" if (kind in ("impact", "rollover") and shock > 0.3) else
             "vshake" if (kind == "pothole" and shock > 0.3) else "")
    braking = acc < -0.15
    x, y, w, h = 280, 20, 300, 140
    cx, cy = x + 54, y + 84
    streaks = "".join(f'<line x1="{cx - 5 + i * 5}" y1="{cy + 34}" x2="{cx - 5 + i * 5}" '
                      f'y2="{cy + 34 + min(speed, 90) * 0.22:.0f}" stroke="{C["gray"]}" stroke-width="1"/>'
                      for i in range(3)) if speed > 3 else ""
    tail = C["red"] if braking else "#5a2a30"
    car = (f'<g class="{shake}"><g transform="rotate({heading:.1f} {cx} {cy})">{streaks}'
           f'<rect x="{cx - 14}" y="{cy - 30}" width="28" height="60" rx="8" fill="#16212b" stroke="{col}" stroke-width="1.6"/>'
           f'<rect x="{cx - 10}" y="{cy - 18}" width="20" height="11" rx="2" fill="#0b0f14" stroke="{C["dim"]}" stroke-width=".8"/>'
           f'<rect x="{cx - 10}" y="{cy + 6}" width="20" height="9" rx="2" fill="#0b0f14" stroke="{C["dim"]}" stroke-width=".8"/>'
           f'<rect x="{cx - 12}" y="{cy + 28}" width="8" height="3" fill="{tail}"/>'
           f'<rect x="{cx + 4}" y="{cy + 28}" width="8" height="3" fill="{tail}"/>'
           f'<rect x="{cx - 12}" y="{cy - 31}" width="8" height="3" fill="#cfd8dc"/>'
           f'<rect x="{cx + 4}" y="{cy - 31}" width="8" height="3" fill="#cfd8dc"/>')
    for wx, wy in ((-17, -22), (13, -22), (-17, 12), (13, 12)):
        car += f'<rect x="{cx + wx}" y="{cy + wy}" width="4" height="10" rx="1" fill="{C["gray"]}"/>'
    if kind in ("impact", "rollover") and shock > 0.3:        # impact flash, only while the vehicle model reports a shock
        car += (f'<polygon points="{cx},{cy - 52} {cx + 5},{cy - 38} {cx + 18},{cy - 44} {cx + 9},{cy - 33} '
                f'{cx + 20},{cy - 24} {cx + 4},{cy - 28} {cx - 2},{cy - 40} {cx - 8},{cy - 28} {cx - 20},{cy - 30} '
                f'{cx - 9},{cy - 36} {cx - 16},{cy - 48} {cx - 5},{cy - 41}" fill="{C["red"]}" opacity=".85"/>')
    car += "</g></g>"
    gx = x + 112                       # roll (rear view) and pitch (side view) gauges: the vehicle's real roll / pitch
    rear = (f'<g transform="rotate({roll:.1f} {gx + 20} {y + 112})"><rect x="{gx + 8}" y="{y + 104}" width="24" '
            f'height="12" rx="3" fill="#16212b" stroke="{col}"/><rect x="{gx + 12}" y="{y + 98}" width="16" '
            f'height="8" rx="2" fill="#16212b" stroke="{col}"/></g>')
    side = (f'<g transform="rotate({-pitch:.1f} {gx + 75} {y + 112})"><rect x="{gx + 57}" y="{y + 105}" width="36" '
            f'height="10" rx="3" fill="#16212b" stroke="{col}"/><rect x="{gx + 65}" y="{y + 99}" width="18" '
            f'height="7" rx="2" fill="#16212b" stroke="{col}"/></g>')
    return "".join([_card(x, y, w, h, "VIRTUAL VEHICLE", s["scenario"].upper(), "Vehicle", selected, col), car,
                    _kv(gx, y + 54, "SPEED", f"{speed:5.1f} km/h", vx=44),
                    _kv(gx, y + 70, "HEADING", f"{heading:5.0f}°", vx=60),
                    _kv(gx, y + 86, "ROLL", f"{roll:+6.1f}°", vx=44), _kv(gx + 92, y + 86, "PITCH", f"{pitch:+5.1f}°", vx=42),
                    rear, side, _t(gx + 20, y + 132, "ROLL", "wl", "middle"), _t(gx + 75, y + 132, "PITCH", "wl", "middle"),
                    _t(x + w - 10, y + 22, state, "val", "end", fill=col)])


def _mpu(s, selected):
    m = s["mpu6050"]
    a, g = m["accel"], m["gyro"]
    online = (not m["sleeping"]) and s["stm32"]["cpu"] == "RUNNING"
    x, y, w, h = 280, 210, 300, 150
    out = [_card(x, y, w, h, "MPU6050", "IMU SENSOR", "MPU6050", selected, C["green"] if online else C["gray"]),
           _t(x + 14, y + 58, "I²C", "lab"), _t(x + 44, y + 58, f"0x{m['address']:02X}", "val"),
           _dot(x + 190, y + 55, C["green"] if online else C["amber"]),
           _t(x + 200, y + 58, "ONLINE" if online else "SLEEP/OFF", "val", fill=C["green"] if online else C["amber"]),
           _t(x + w - 14, y + 22, f"±{m['accel_range_g']:g} g · ±{m['gyro_range_dps']:g} °/s", "sub", "end")]
    for i, (n, v) in enumerate(zip(("AX", "AY", "AZ"), a)):
        out.append(_kv(x + 14, y + 78 + i * 17, n, f"{v:+7.3f} g", vx=28))
    for i, (n, v) in enumerate(zip(("GX", "GY", "GZ"), g)):
        out.append(_kv(x + 150, y + 78 + i * 17, n, f"{v:+8.2f} °/s", vx=28))
    if m["saturated"]:
        axes = ",".join(k.upper() for k, v in m["saturated_axes"].items() if v)
        out.append(f'<rect x="{x + 14}" y="{y + 126}" width="{w - 28}" height="18" rx="3" fill="{C["red"]}" class="pulse"/>'
                   f'<text x="{x + w / 2}" y="{y + 139}" text-anchor="middle" style="fill:#fff;font-size:11px;font-weight:700">'
                   f'SATURATED  ({escape(axes)})</text>')
    else:
        out.append(_t(x + 14, y + 139, "SATURATION", "lab") + _t(x + 90, y + 139, "NO", "val"))
    return "".join(out)


def _stm32(s, selected):
    m = s["stm32"]
    x, y, w, h = 280, 410, 300, 160
    ok = m["cpu"] == "RUNNING"
    led = {"GREEN": C["green"], "YELLOW": C["amber"], "RED": C["red"]}[m["led"]]
    act = lambda on: C["cyan"] if on else C["gray"]                                          # noqa: E731
    ev_col = C["red"] if m["alert_output"] else C["amber"] if m["mcu_event"] else C["dim"]
    return "".join([
        _card(x, y, w, h, "STM32", "MAIN CONTROLLER", "STM32", selected, C["green"] if ok else C["red"]),
        _kv(x + 14, y + 58, "CPU", m["cpu"], C["green"] if ok else C["red"]),
        _kv(x + 14, y + 76, "I²C1", m["i2c1"], act(m["i2c1"] == "ACTIVE")),
        _kv(x + 14, y + 94, "USART2", str(m["baud"]), act(m["usart2"] == "ACTIVE")),
        _kv(x + 14, y + 112, "SAMPLE", f"{m['sample_rate_hz']} Hz"),
        _kv(x + 156, y + 58, "SAMPLES", str(m["samples"])),
        _kv(x + 156, y + 76, "GPIO", m["gpio_status"], C["green"] if ok else C["red"]),
        _t(x + 156, y + 94, "LED", "lab"), _dot(x + 214, y + 91, led, pulse=m["led"] == "RED", r=5),
        _t(x + 226, y + 94, m["led"], "val", fill=led),
        _t(x + 156, y + 112, "BUZZER", "lab"),
        _t(x + 214, y + 112, "ACTIVE" if m["buzzer"] else "OFF", "val", fill=C["red"] if m["buzzer"] else C["dim"],
           pulse=m["buzzer"]),
        _dot(x + 18, y + 138, ev_col, pulse=m["alert_output"]), _t(x + 30, y + 142, m["status_text"], "val", fill=ev_col)])


def _vibration(s, selected):
    lv = s["vibration"]["level"]
    status, col = ("NORMAL", C["green"]) if lv < 1.5 else ("ELEVATED", C["amber"]) if lv < 4 else ("SHOCK", C["red"])
    x, y, w, h = 16, 400, 220, 100
    return "".join([_card(x, y, w, h, "VIBRATION", "ADC1 · ANALOG SENSOR", "Vibration sensor", selected, col),
                    _kv(x + 14, y + 58, "LEVEL", f"{lv:6.2f}"), _kv(x + 14, y + 76, "STATUS", status, col),
                    f'<rect x="{x + 120}" y="{y + 50}" width="{w - 134}" height="8" fill="#0b0f14" stroke="{C["line"]}"/>',
                    f'<rect x="{x + 120}" y="{y + 50}" width="{min(lv / 10, 1) * (w - 134):.0f}" height="8" fill="{col}"/>'])


def _gps(s, selected):
    g = s["gps"]
    x, y, w, h = 16, 520, 220, 135
    fixc = C["green"] if g["fix"] else C["amber"]
    return "".join([_card(x, y, w, h, "GPS", "SIMULATED GPS · USART3", "GPS", selected, fixc),
                    _t(x + 14, y + 58, "FIX", "lab"), _dot(x + 82, y + 55, fixc),
                    _t(x + 92, y + 58, "YES" if g["fix"] else "NO", "val", fill=fixc),
                    _kv(x + 14, y + 76, "LAT", f"{g['latitude']:.6f}"), _kv(x + 14, y + 94, "LON", f"{g['longitude']:.6f}"),
                    _kv(x + 14, y + 112, "SPEED", f"{g['speed_kmh']:.0f} km/h")])


def _host(s, p, selected):
    x, y, w, h = 624, 410, 220, 150
    conf = p.get("confidence")
    col = C["red"] if "ALERT" in str(p.get("status")) else C["cyan"]
    return "".join([_card(x, y, w, h, "HOST / AI", "USART2 RX · PIPELINE", "Host", selected, col),
                    _kv(x + 14, y + 58, "EVENT", str(p.get("label") or "—"), vx=60),
                    _kv(x + 14, y + 76, "CONF", "—" if conf is None else f"{conf:.0%}", vx=60),
                    _kv(x + 14, y + 94, "SEVERITY", str(p.get("severity") or "—"), vx=60),
                    _kv(x + 14, y + 112, "STATE", str(p.get("status") or "MONITORING"), col, vx=60)])


def _esp(s, p, selected):
    u = s["alert_unit"]["state"]
    status = ("STANDBY" if u == "IDLE" else "ARMED" if u.startswith("ARMED") else
              "ALERT SENT (SIM)" if u.startswith("SIMULATED") else "CANCELLED")
    cd = p.get("countdown")
    alert = ("COUNTDOWN" + (f" T-{cd}s" if cd is not None else "") if status == "ARMED" else
             "SIMULATED" if status.startswith("ALERT") else "NONE")
    col = C["red"] if status == "ARMED" or status.startswith("ALERT") else C["amber"] if status == "CANCELLED" else C["green"]
    x, y, w, h = 280, 615, 300, 120
    return "".join([_card(x, y, w, h, "ESP32 / GSM", "EMERGENCY MODULE · USART1", "ESP32/GSM", selected, col),
                    _kv(x + 14, y + 58, "STATUS", status, col), _kv(x + 14, y + 76, "ALERT", alert, col),
                    _t(x + 14, y + 104, "NO REAL SMS/GSM MESSAGE WAS SENT", "sub")])


def _wires(s, tick, live):
    m = s["stm32"]
    i2c_on = live and m["i2c1"] == "ACTIVE"
    uart_on = live and m["usart2"] == "ACTIVE"
    gps_on = live and s["gps"]["fix"]
    alert_on = m["alert_output"]
    busy = m["mcu_event"] or m["alert_output"]
    veh_mpu, i2c = "M430 160 V210", "M430 360 V410"
    vib, gps = "M236 450 H280", "M236 580 H258 V540 H280"
    usart2, esp = "M580 480 H624", "M430 570 V615"
    return "".join([
        _wire("M280 90 H8 V585 H16 M8 450 H16", live, C["cyan"], dashed=True, arrow=False),
        _wire(veh_mpu, live, C["cyan"], "SENSOR DATA", 440, 190),
        _wire(i2c, i2c_on, C["cyan"], "I²C1 · SCL/SDA", 440, 392),
        _wire(vib, live, C["cyan"], "ADC", 258, 442, "middle"), _wire(gps, gps_on, C["cyan"], "UART", 252, 532, "middle"),
        _wire(usart2, uart_on, C["cyan"], "USART2", 602, 472, "middle"),
        _wire(esp, alert_on, C["red"], "UART · USART1", 440, 596),
        _packets("M280 90 H8 V450 H16", 2, 1.2, tick, C["cyan"], live),
        _packets(veh_mpu, 2, 0.7, tick, C["cyan"], live),
        _packets(i2c, 4 if busy else 2, 0.35 if busy else 0.7, tick, C["cyan"], i2c_on),
        _packets(vib, 2, 0.8, tick, C["cyan"], live), _packets(gps, 2, 1.0, tick, C["cyan"], gps_on),
        _packets(usart2, 2, 0.6, tick, C["cyan"], uart_on),
        _packets(esp, 3, 0.6, tick, C["red"], alert_on),
    ])


def emergency_state(s, p):
    """(text, colour) of the system-level state chip; shared with the page panels."""
    status = (p or {}).get("status") or "MONITORING"
    if status == "ALERT COUNTDOWN":
        cd = p.get("countdown")
        return "EMERGENCY COUNTDOWN" + (f"  {cd}" if cd is not None else ""), C["red"]
    if status == "ALERT SENT":
        return "SIMULATED ALERT SENT", C["red"]
    if status == "ALERT CANCELLED":
        return "ALERT CANCELLED", C["amber"]
    if status == "EVENT DETECTED" or s["stm32"]["mcu_event"]:
        return "EVENT DETECTED", C["amber"]
    return "SYSTEM NORMAL", C["green"]


def build_lab_html(snapshot, pipeline, selected="MPU6050", animate=True):
    s, p = snapshot, pipeline or {}
    m = s["stm32"]
    online = m["cpu"] == "RUNNING"
    chip, chip_col = emergency_state(s, p)
    bar = (f'<div class="bar"><span><span class="dot" style="background:{C["green"] if online else C["red"]}"></span>'
           f'<b>{"SYSTEM ONLINE" if online else "SYSTEM FAULT"}</b></span>'
           f'<span>I²C1 <b>{m["i2c1"]}</b></span><span>USART2 <b>{m["baud"]}</b></span>'
           f'<span><b>{m["sample_rate_hz"]} Hz</b></span><span><b>{m["samples"]}</b> samples</span>'
           f'<span><b>{m["i2c_errors"]}</b> I²C errors</span>'
           f'<span class="chip" style="color:{chip_col};border-color:{chip_col}">{escape(chip)}</span></div>')
    svg = "".join([f'<svg viewBox="0 0 {VB_W} {VB_H}" preserveAspectRatio="xMidYMid meet" '
                   f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Virtual hardware schematic">',
                   _markers(), _wires(s, s["tick"], bool(animate)),
                   _vehicle(s, selected), _mpu(s, selected), _stm32(s, selected), _vibration(s, selected),
                   _gps(s, selected), _host(s, p, selected), _esp(s, p, selected),
                   _t(16, 28, "VIRTUAL HARDWARE SIMULATION", "sub"),
                   _t(16, 44, f"t = {s['tick'] / 10:.1f} s", "val", fill=C["dim"]),
                   "</svg>"])
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{bar}{svg}</body></html>"


# --------------------------------------------------------------------------- oscilloscope
SCOPE_CHANNELS = {
    "ACCEL": ("accel_mag", "|a|  (g)", C["green"]), "GYRO": ("gyro_mag", "|ω|  (°/s)", C["cyan"]),
    "SPEED": ("speed", "speed  (km/h)", C["amber"]), "VIBRATION": ("vibration", "vibration  (level)", "#c792ea"),
}


def build_scope_html(history, channel="ACCEL", width=1000, height=190):
    key, unit, color = SCOPE_CHANNELS[channel]
    data = list((history or {}).get(key) or [])[-60:]
    pad_l, pad_r, pad_t, pad_b = 54, 14, 14, 20
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="{C["card"]}"/>']
    for i in range(5):                                           # graticule
        gy = pad_t + ph * i / 4
        parts.append(f'<line x1="{pad_l}" y1="{gy:.0f}" x2="{width - pad_r}" y2="{gy:.0f}" stroke="{C["line"]}" stroke-width=".7"/>')
    for i in range(7):
        gx = pad_l + pw * i / 6
        parts.append(f'<line x1="{gx:.0f}" y1="{pad_t}" x2="{gx:.0f}" y2="{pad_t + ph}" stroke="{C["line"]}" stroke-width=".7"/>')
    cur = "—"
    if len(data) >= 2:
        lo, hi = min(data), max(data)
        if hi - lo < 1e-6:
            lo, hi = lo - 0.5, hi + 0.5
        pts = " ".join(f"{pad_l + pw * j / 59:.1f},{pad_t + ph * (1 - (v - lo) / (hi - lo)):.1f}" for j, v in enumerate(data))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{pad_t + 4}" text-anchor="end" class="sub">{hi:.2f}</text>'
                     f'<text x="{pad_l - 6}" y="{pad_t + ph + 4}" text-anchor="end" class="sub">{lo:.2f}</text>')
        cur = f"{data[-1]:.2f}"
    parts.append(f'<text x="{pad_l + 6}" y="{pad_t + 14}" class="lab">{escape(unit)}</text>'
                 f'<text x="{width - pad_r - 6}" y="{pad_t + 16}" text-anchor="end" class="val" '
                 f'style="font-size:15px;fill:{color}">{cur}</text>'
                 f'<text x="{pad_l}" y="{height - 5}" class="sub">-6 s</text>'
                 f'<text x="{width - pad_r}" y="{height - 5}" text-anchor="end" class="sub">now</text>')
    svg = (f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
           f'{"".join(parts)}</svg>')
    return (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}svg{{height:100%}}</style></head>"
            f"<body>{svg}</body></html>")
