"""
Compact, dark "instrument panel" HTML for the Virtual Hardware Lab page (header, AI analysis,
emergency state, component inspector). Pure functions returning single-line HTML (no blank lines /
indentation, so Markdown never turns them into code blocks). Display only.
"""

from html import escape

from virtual_hardware.schematic import C, emergency_state

PANEL_CSS = f"""<style>
.block-container{{padding-top:2.2rem!important}}
.hw-head{{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;margin:0 0 6px 0}}
.hw-title{{font-size:1.25rem;font-weight:800;letter-spacing:1.5px;line-height:1.1}}
.hw-sub{{font-size:.8rem;opacity:.65;letter-spacing:.3px}}
.hw-badges{{display:flex;flex-wrap:wrap;gap:6px;margin-left:auto}}
.hw-badge{{font-size:.68rem;font-weight:700;letter-spacing:.8px;padding:2px 8px;border:1px solid {C['line']};
border-radius:3px;background:{C['card']};color:{C['dim']}}}
.hw-badge.live{{color:{C['green']};border-color:{C['green']}}}.hw-badge.warn{{color:{C['amber']};border-color:{C['amber']}}}
.hw-badge.cyan{{color:{C['cyan']};border-color:{C['cyan']}}}
.hw-card{{background:{C['bg']};border:1px solid {C['line']};border-radius:4px;padding:10px 12px;margin-bottom:10px;color:{C['txt']};
font-family:'Segoe UI',system-ui,sans-serif}}
.hw-card h4{{margin:0 0 8px 0;font-size:.72rem;letter-spacing:1.2px;color:{C['dim']};font-weight:700}}
.hw-row{{display:flex;justify-content:space-between;gap:10px;font-size:.82rem;padding:2px 0;border-bottom:1px solid #151d25}}
.hw-row:last-child{{border-bottom:0}}.hw-row span:first-child{{color:{C['dim']}}}
.hw-row span:first-child{{white-space:nowrap}}.hw-row span:last-child{{font-size:.76rem;font-family:Consolas,'Courier New',monospace;text-align:right}}
.hw-state{{font-size:1.05rem;font-weight:800;letter-spacing:.8px}}
.hw-count{{font-size:2.6rem;font-weight:800;line-height:1;font-family:Consolas,monospace}}
.hw-note{{font-size:.68rem;color:{C['dim']};margin-top:6px}}
</style>"""


def _badge(text, kind=""):
    return f'<span class="hw-badge {kind}">{escape(text)}</span>'


def header_html(snapshot, pipeline, mode_active):
    run = (pipeline or {}).get("run_state", "IDLE") if mode_active else "STANDBY"
    kind = {"RUNNING": "live", "PAUSED": "warn", "FINISHED": "cyan"}.get(run, "")
    badges = "".join([_badge("VIRTUAL SIMULATION", "cyan"), _badge("10 Hz"), _badge(run, kind),
                      _badge("SIMULATED"), _badge("SYNTHETIC DATA"), _badge("NO REAL GSM", "warn")])
    t = snapshot["tick"] / 10
    return (f'<div class="hw-head"><div><div class="hw-title">VIRTUAL HARDWARE LAB</div>'
            f'<div class="hw-sub">AI-Based Smart Accident Detection &amp; Emergency Response System</div></div>'
            f'<div class="hw-badges">{badges}</div></div>'
            f'<div class="hw-sub">Scenario <b>{escape(snapshot["scenario"])}</b> &nbsp;·&nbsp; Seed <b>{snapshot["seed"]}</b>'
            f' &nbsp;·&nbsp; Simulation time <b>{t:.1f} s</b> &nbsp;·&nbsp; Samples <b>{snapshot["stm32"]["samples"]}</b></div>')


def ai_html(pipeline):
    p = pipeline or {}
    conf = p.get("confidence")
    sev = str(p.get("severity") or "—")
    sev_col = C["red"] if sev.startswith(("HIGH", "CRITICAL")) else C["amber"] if sev.startswith("MEDIUM") else C["green"] \
        if p.get("severity") else C["dim"]
    th = str(p.get("threshold") or "—")
    th_col = C["red"] if th.startswith("ACCIDENT") else C["dim"]
    rows = [("Prediction", escape(str(p.get("label") or "—")), None),
            ("Confidence", "—" if conf is None else f"{conf:.1%}", None),
            ("Severity", escape(sev), sev_col), ("Threshold", escape(th), th_col),
            ("System", escape(str(p.get("status") or "MONITORING")), None)]
    body = "".join(f'<div class="hw-row"><span>{k}</span><span{f" style=color:{c}" if c else ""}>{v}</span></div>'
                   for k, v, c in rows)
    return (f'<div class="hw-card"><h4>AI EVENT ANALYSIS</h4>{body}'
            f'<div class="hw-note">Model trained on synthetic data only; it can misclassify, especially on '
            f'clipped (saturated) accident signals. Shown as-is.</div></div>')


def emergency_html(snapshot, pipeline):
    text, col = emergency_state(snapshot, pipeline)
    status = (pipeline or {}).get("status")
    cd = (pipeline or {}).get("countdown")
    icon = "⚠" if col == C["amber"] else "●" if col == C["green"] else "▲"
    count = f'<div class="hw-count" style="color:{col}">{cd}</div>' if status == "ALERT COUNTDOWN" and cd is not None else ""
    return (f'<div class="hw-card" style="border-color:{col if col != C["green"] else C["line"]}"><h4>EMERGENCY STATE</h4>'
            f'<div class="hw-state" style="color:{col}">{icon} {escape(text.split("  ")[0])}</div>{count}'
            f'<div class="hw-note">SIMULATED EMERGENCY ALERT · NO REAL SMS/GSM MESSAGE WAS SENT</div></div>')


def inspector_html(rows, component):
    body = "".join(f'<div class="hw-row"><span>{escape(k)}</span><span>{escape(str(v))}</span></div>' for k, v in rows)
    return f'<div class="hw-card"><h4>COMPONENT INSPECTOR · {escape(component.upper())}</h4>{body}</div>'
