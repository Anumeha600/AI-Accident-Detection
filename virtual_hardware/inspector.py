"""Component inspector data: (label, value) rows per component, from a rig snapshot. Pure, UI-free."""

COMPONENTS = ["Vehicle", "MPU6050", "STM32", "Vibration sensor", "GPS", "ESP32/GSM"]


def _yn(flag):
    return "YES" if flag else "NO"


def inspector_rows(snapshot, component):
    s = snapshot
    mpu, stm, gps = s["mpu6050"], s["stm32"], s["gps"]
    if component == "MPU6050":
        a, g = mpu["accel"], mpu["gyro"]
        sat = mpu["saturated"]
        return [
            ("DEVICE", "MPU6050"),
            ("I2C ADDRESS", f"0x{mpu['address']:02X}"),
            ("WHO_AM_I", "—" if mpu["who_am_i"] is None else f"0x{mpu['who_am_i']:02X}"),
            ("ACCEL RANGE", f"±{mpu['accel_range_g']:g}g"),
            ("GYRO RANGE", f"±{mpu['gyro_range_dps']:g}°/s"),
            ("SAMPLE RATE", f"{mpu['sample_rate_hz']:g} Hz"),
            ("ACCEL X", f"{a[0]:+.3f} g"), ("ACCEL Y", f"{a[1]:+.3f} g"), ("ACCEL Z", f"{a[2]:+.3f} g"),
            ("GYRO X", f"{g[0]:+.2f} °/s"), ("GYRO Y", f"{g[1]:+.2f} °/s"), ("GYRO Z", f"{g[2]:+.2f} °/s"),
            ("SATURATION", _yn(sat) + (" — SATURATED (" + ", ".join(
                k.upper() for k, v in mpu["saturated_axes"].items() if v) + ")" if sat else "")),
            ("PRE-SATURATION", ", ".join(
                f"{k.upper()}={v:+.2f}" for k, v in mpu["physical_before_saturation"].items())),
        ]
    if component == "STM32":
        return [
            ("MCU", "STM32 (functional model, not a cycle-accurate emulator)"),
            ("CPU", stm["cpu"]), ("I2C1", stm["i2c1"]), ("USART2", stm["usart2"]),
            ("BAUD", str(stm["baud"])), ("SAMPLE RATE", f"{stm['sample_rate_hz']} Hz"),
            ("SAMPLES", str(stm["samples"])), ("GPIO/STATUS", stm["gpio_status"]),
            ("EVENT", stm["status_text"]), ("EVENTS COUNTED", str(stm["event_count"])),
            ("LED", stm["led"]), ("BUZZER", "ACTIVE" if stm["buzzer"] else "OFF"),
            ("I2C ERRORS", str(stm["i2c_errors"])), ("UPTIME", f"{stm['uptime_ms'] / 1000:.1f} s"),
        ]
    if component == "GPS":
        return [("DEVICE", "SIMULATED GPS"), ("FIX", _yn(gps["fix"])),
                ("LAT", f"{gps['latitude']:.6f}"), ("LON", f"{gps['longitude']:.6f}"),
                ("SPEED", f"{gps['speed_kmh']:.1f} km/h"), ("HEADING", f"{gps['heading_deg']:.0f}°")]
    if component == "Vibration sensor":
        return [("DEVICE", "VIBRATION SENSOR (ADC1)"), ("LEVEL", f"{s['vibration']['level']:.3f}")]
    if component == "ESP32/GSM":
        return [("DEVICE", "ESP32/GSM ALERT UNIT (stub)"), ("STATE", s["alert_unit"]["state"]),
                ("LAST COMMANDS", ", ".join(s["alert_unit"]["commands"]) or "—"),
                ("REAL SMS/GSM SENT", "NO — NO REAL SMS/GSM MESSAGE WAS SENT")]
    v = s["vehicle"] or {}
    return [("DEVICE", "VIRTUAL VEHICLE"), ("STATE", v.get("vehicle_state", "—")),
            ("SCENARIO", s["scenario"]), ("SPEED", f"{v.get('speed_kmh', 0.0):.1f} km/h"),
            ("ROLL", f"{v.get('roll_deg', 0.0):.1f}°"), ("PITCH", f"{v.get('pitch_deg', 0.0):.1f}°"),
            ("HEADING", f"{v.get('heading_deg', 0.0):.0f}°"), ("YAW RATE", f"{v.get('yaw_rate_dps', 0.0):.1f} °/s"),
            ("SEED", str(s["seed"]))]
