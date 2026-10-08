"""
Phase 10.5 -- virtual hardware: clock, vehicle, MPU6050 registers, I2C, UART, STM32, sensor source,
and the hand-off to the EXISTING pipeline / recording / replay / emergency state machine.
No Streamlit here (see test_virtual_hardware_app.py).
"""

import math

import pytest

import replay as rp
from common import FEATURE_COLUMNS
from realtime_pipeline import RollingWindowBuffer, next_status, severity_for_label
from recording_store import RecordingStore, validate_recording_dict
from replay import decide_window
from sensor_source import SensorSource, NO_DATA_YET
from virtual_hardware import VirtualClock, VirtualHardwareRig, VirtualHardwareSensorSource, VIRTUAL_SCENARIOS
from virtual_hardware import mpu6050 as m
from virtual_hardware.gps import VirtualGPS
from virtual_hardware.i2c import VirtualI2CBus, I2CNack
from virtual_hardware.inspector import COMPONENTS, inspector_rows
from virtual_hardware.schematic import build_lab_html, build_scope_html, SCOPE_CHANNELS
from virtual_hardware.panels import header_html, ai_html, emergency_html, inspector_html
from virtual_hardware.uart import VirtualUART
from virtual_hardware.vehicle import VirtualVehicle
from virtual_hardware.vibration_sensor import VibrationSensor


def run_rig(scenario, seed=1, ticks=80):
    rig = VirtualHardwareRig(seed, scenario)
    return rig, [rig.tick() for _ in range(ticks)]


def make_mpu(clock=None):
    return m.MPU6050((clock or VirtualClock(0)).rng("mpu6050"))


def awake_mpu():
    dev = make_mpu()
    dev.write_register(m.REG_PWR_MGMT_1, 0x00)
    return dev


# 1 ---------------------------------------------------------------- clock
def test_clock_is_deterministic_and_ticks_are_100ms():
    a, b, c = VirtualClock(5), VirtualClock(5), VirtualClock(6)
    assert list(a.rng("x").normal(size=4)) == list(b.rng("x").normal(size=4))
    assert list(a.rng("x").normal(size=4)) != list(c.rng("x").normal(size=4))
    assert list(a.rng("x").normal(size=2)) != list(a.rng("y").normal(size=2))   # streams are independent
    assert a.time_ms == 0 and a.timestamp() == "10:24:30.000"
    a.advance(); a.advance()
    assert a.time_ms == 200 and a.timestamp() == "10:24:30.200" and a.time_s == pytest.approx(0.2)
    a.reset()
    assert a.tick == 0


def test_same_seed_gives_identical_runs_and_different_seed_does_not():
    def lines(seed):
        rig, _ = run_rig("Severe Accident", seed, 40)
        return [p.text for p in rig.uart.history]
    assert lines(3) == lines(3)
    assert lines(3) != lines(4)


# 2 ---------------------------------------------------------------- vehicle state
def test_vehicle_state_during_normal_driving():
    clock = VirtualClock(1)
    v = VirtualVehicle(clock)
    first = None
    for _ in range(30):
        clock.advance()
        st = v.step()
        first = first or st
    assert st.scenario == "Normal Driving" and st.vehicle_state == "DRIVING" and not st.crashed
    assert 40 <= st.speed_kmh <= 61 and st.accel_g == pytest.approx((0, 0, 1))
    assert st.timestamp_ms == 3000 and st.tick == 30
    assert abs(st.roll_deg) < 1 and abs(st.pitch_deg) < 1
    assert (st.latitude, st.longitude) != (first.latitude, first.longitude)    # it moves
    assert set(st.as_dict()) >= {"speed_kmh", "heading_deg", "roll_deg", "pitch_deg", "yaw_rate_dps",
                                 "latitude", "longitude", "vehicle_state", "timestamp_ms", "acceleration_long_g"}


# 3 ---------------------------------------------------------------- scenario transitions
def test_scenarios_produce_their_characteristic_physics():
    def run(sc):
        _, ticks = run_rig(sc, 2)
        return [t.vehicle for t in ticks]
    pothole, braking, turn = run("Pothole"), run("Hard Braking"), run("Sharp Turn")
    assert min(s.accel_g[2] for s in pothole) < 0.5
    assert min(s.accel_g[0] for s in braking) < -0.4 and braking[-1].speed_kmh < braking[0].speed_kmh - 20
    assert max(abs(s.yaw_rate_dps) for s in turn) > 70
    severe = run("Severe Accident")
    assert any(s.vehicle_state == "COLLISION" for s in severe)
    assert severe[-1].vehicle_state == "STOPPED (CRASHED)" and severe[-1].crashed
    assert severe[-1].speed_kmh < 9
    assert max(abs(s.accel_g[0]) for s in severe) > 5          # the true (pre-sensor) impact exceeds +-2 g
    roll = run("Rollover")
    assert roll[-1].roll_deg == pytest.approx(180, abs=1) and any(s.vehicle_state == "ROLLOVER" for s in roll)
    assert roll[-1].accel_g[2] < -0.9                                      # resting upside down
    multi = run("Multi-Impact Collision")
    peaks = [i for i in range(1, len(multi) - 1)
             if multi[i].shock > 0.3 and multi[i].shock >= multi[i - 1].shock and multi[i].shock > multi[i + 1].shock]
    assert len(peaks) >= 3 and multi[-1].speed_kmh < 1


def test_scenario_can_change_live_and_position_persists():
    clock = VirtualClock(0)
    v = VirtualVehicle(clock)
    for _ in range(10):
        clock.advance(); v.step()
    before = v.state
    v.set_scenario("Severe Accident")
    clock.advance()
    after = v.step()
    assert v.scenario == "Severe Accident" and after.scenario == "Severe Accident"
    assert abs(after.latitude - before.latitude) < 1e-3 and not after.crashed
    with pytest.raises(ValueError):
        v.set_scenario("Teleport")
    assert VIRTUAL_SCENARIOS == ["Normal Driving", "Pothole", "Hard Braking", "Sharp Turn",
                                 "Minor Accident", "Severe Accident", "Rollover", "Multi-Impact Collision"]


# 4-7 -------------------------------------------------------------- MPU6050 registers
def test_mpu6050_who_am_i_and_i2c_address():
    dev = make_mpu()
    assert dev.read_register(m.REG_WHO_AM_I) == 0x68 == m.WHO_AM_I_VALUE
    assert dev.address == 0x68 == m.I2C_ADDRESS
    rig = VirtualHardwareRig(0)
    assert rig.i2c.devices == [0x68] and rig.stm32.who_am_i == 0x68


def test_mpu6050_register_reads_and_burst_decode():
    dev = awake_mpu()
    assert dev.read_register(m.REG_PWR_MGMT_1) == 0x00
    dev.accel_noise_g = dev.gyro_noise_dps = 0.0
    dev.sense((0.5, -0.25, 1.0), (10.0, -20.0, 30.0), dt_ms=100)
    raw = dev.read_registers(m.REG_ACCEL_XOUT_H, 14)
    assert len(raw) == 14
    assert int.from_bytes(raw[0:2], "big", signed=True) == 8192            # 0.5 g * 16384
    assert int.from_bytes(raw[4:6], "big", signed=True) == 16384           # 1 g
    assert int.from_bytes(raw[8:10], "big", signed=True) == 1310           # 10 dps * 131
    s = dev.read_sensor_sample()
    assert (s.accel_x_g, s.accel_y_g, s.accel_z_g) == pytest.approx((0.5, -0.25, 1.0), abs=1e-3)
    assert (s.gyro_x_dps, s.gyro_y_dps, s.gyro_z_dps) == pytest.approx((10.0, -20.0, 30.0), abs=0.5)
    assert dev.read_register(0x30) == 0x00                                   # unmapped
    # INT_STATUS: DATA_RDY is set by a latched sample and cleared by reading it
    assert dev.read_register(m.REG_INT_STATUS) & m.INT_DATA_RDY
    assert dev.read_register(m.REG_INT_STATUS) == 0
    with pytest.raises(ValueError):
        dev.read_registers(m.REG_WHO_AM_I, 0)


def test_mpu6050_powers_up_asleep_and_wakes_on_register_write():
    dev = make_mpu()
    assert dev.sleeping and dev.read_register(m.REG_PWR_MGMT_1) == 0x40
    assert dev.sense((0, 0, 1), (0, 0, 0)) is False and dev.samples_latched == 0
    dev.write_register(m.REG_PWR_MGMT_1, 0x00)
    assert not dev.sleeping and dev.sense((0, 0, 1), (0, 0, 0)) is True


def test_mpu6050_register_writes_configure_range_rate_and_ignore_read_only():
    dev = awake_mpu()
    assert dev.accel_full_scale_g == 2 and dev.gyro_full_scale_dps == 250
    dev.write_register(m.REG_ACCEL_CONFIG, 0x10)           # AFS_SEL = 2
    dev.write_register(m.REG_GYRO_CONFIG, 0x08)            # FS_SEL = 1
    assert dev.accel_full_scale_g == 8 and dev.gyro_full_scale_dps == 500
    dev.write_register(m.REG_CONFIG, 0x03)
    dev.write_register(m.REG_SMPLRT_DIV, 99)
    assert dev.sample_rate_hz == pytest.approx(10.0)
    assert dev.read_register(m.REG_SMPLRT_DIV) == 99
    dev.write_register(m.REG_WHO_AM_I, 0x12)               # read-only: ignored
    dev.write_register(m.REG_ACCEL_XOUT_H, 0xFF)
    assert dev.read_register(m.REG_WHO_AM_I) == 0x68 and dev.ignored_writes == 2
    dev.write_register(m.REG_PWR_MGMT_1, 0x80)             # DEVICE_RESET restores defaults, incl. sleep
    assert dev.sleeping and dev.accel_full_scale_g == 2
    with pytest.raises(ValueError):
        dev.write_register(m.REG_CONFIG, 0x1FF)


def test_mpu6050_sample_rate_gates_latching():
    dev = awake_mpu()
    dev.write_register(m.REG_CONFIG, 0x03)
    dev.write_register(m.REG_SMPLRT_DIV, 99)               # 10 Hz
    assert [dev.sense((0, 0, 1), (0, 0, 0), dt_ms=50) for _ in range(4)] == [False, True, False, True]


# 8-9 -------------------------------------------------------------- saturation
def test_accelerometer_saturation_clips_flags_and_keeps_physical_value():
    dev = awake_mpu()
    dev.accel_noise_g = dev.gyro_noise_dps = 0.0
    dev.sense((-9.0, 1.0, 8.0), (0, 0, 0), dt_ms=100)
    s = dev.read_sensor_sample()
    assert s.accel_x_g == pytest.approx(-2.0, abs=1e-3) and s.accel_z_g == pytest.approx(2.0, abs=1e-3)
    assert s.accel_y_g == pytest.approx(1.0, abs=1e-3)
    assert dev.saturated and dev.saturated_axes["ax"] and dev.saturated_axes["az"] and not dev.saturated_axes["ay"]
    assert dev.last_physical["ax"] == pytest.approx(-9.0) and dev.last_physical["az"] == pytest.approx(8.0)
    assert m.decode_burst(dev.read_registers(m.REG_ACCEL_XOUT_H, 14)).saturated      # visible at the int16 rail
    dev.sense((0.1, 0.1, 1.0), (0, 0, 0), dt_ms=100)
    assert not dev.saturated                                                         # clears with the next in-range sample


def test_gyroscope_saturation_clips_and_flags():
    dev = awake_mpu()
    dev.accel_noise_g = dev.gyro_noise_dps = 0.0
    dev.sense((0, 0, 1), (400.0, -900.0, 100.0), dt_ms=100)
    s = dev.read_sensor_sample()
    assert s.gyro_x_dps == pytest.approx(250.0, abs=0.2) and s.gyro_y_dps == pytest.approx(-250.0, abs=0.2)
    assert s.gyro_z_dps == pytest.approx(100.0, abs=0.5)
    assert dev.saturated_axes["gx"] and dev.saturated_axes["gy"] and not dev.saturated_axes["gz"]
    assert dev.last_physical["gy"] == pytest.approx(-900.0)
    dev.write_register(m.REG_GYRO_CONFIG, 0x18)                                      # +-2000 deg/s: no longer saturated
    dev.sense((0, 0, 1), (400.0, -900.0, 100.0), dt_ms=100)
    assert not dev.saturated and dev.read_sensor_sample().gyro_y_dps == pytest.approx(-900.0, abs=1.0)


def test_end_to_end_severe_accident_saturates_the_telemetry():
    rig, ticks = run_rig("Severe Accident", 1)
    assert any(rig.snapshot() and t.packet for t in ticks)
    assert rig.stm32.saturation_seen and rig.mpu.saturated_axes is not None
    peak = max(abs(float(p.text.split(",")[1])) for p in rig.uart.history)
    assert peak <= 2.0001                                                            # the pipeline never sees more than 2 g


# 10 --------------------------------------------------------------- vibration
def test_vibration_sensor_responds_to_events():
    def peak(scenario):
        sensor = VibrationSensor(VirtualClock(4).rng("v"))
        return max(sensor.sense(t.vehicle) for t in run_rig(scenario, 4)[1])
    quiet, pothole, severe = peak("Normal Driving"), peak("Pothole"), peak("Severe Accident")
    assert peak("Hard Braking") > quiet and peak("Sharp Turn") > quiet and peak("Minor Accident") > quiet
    assert peak("Multi-Impact Collision") > pothole
    assert 0.2 <= quiet < 2.0 and pothole > quiet and severe > pothole > 0
    assert severe > 5
    sensor = VibrationSensor(VirtualClock(0).rng("v"))
    assert all(sensor.sense(t.vehicle) >= 0 for t in run_rig("Hard Braking", 0)[1])


# 11 --------------------------------------------------------------- GPS
def test_gps_follows_vehicle_and_reports_fix_status():
    rig, ticks = run_rig("Normal Driving", 2, 20)
    g0, g1 = ticks[0], ticks[-1]
    rep = rig.gps.report()
    assert rep["fix"] and 12.9 < rep["latitude"] < 13.1 and 77.5 < rep["longitude"] < 77.7
    assert (rep["latitude"], rep["longitude"]) != (12.9716, 77.5946)
    assert rep["speed_kmh"] == pytest.approx(g1.vehicle.speed_kmh, abs=10) and 0 <= rep["heading_deg"] < 360
    # cold start: no fix until the configured number of ticks
    clock = VirtualClock(0)
    cold = VirtualGPS(clock.rng("g"), fix_after_ticks=3)
    veh = VirtualVehicle(clock)
    fixes = []
    for _ in range(4):
        clock.advance(); fixes.append(cold.sense(veh.step())["fix"])
    assert fixes == [False, False, True, True]


def test_gps_keeps_position_after_a_crash():
    rig, ticks = run_rig("Severe Accident", 1, 80)
    last = rig.gps.report()
    for _ in range(5):
        rig.tick()
    now = rig.gps.report()
    assert abs(now["latitude"] - last["latitude"]) < 5e-5 and abs(now["longitude"] - last["longitude"]) < 5e-5
    assert now["speed_kmh"] < 12


# 12 --------------------------------------------------------------- I2C
def test_i2c_transactions_are_logged_with_wire_phases():
    clock = VirtualClock(0)
    bus = VirtualI2CBus(clock, "I2C1", log_size=5)
    dev = make_mpu(clock)
    bus.register_device(0x68, dev)
    with pytest.raises(ValueError):
        bus.register_device(0x68, dev)
    with pytest.raises(ValueError):
        bus.register_device(0x02, dev)
    clock.advance(); clock.advance()
    assert bus.read(0x68, m.REG_WHO_AM_I, 1) == b"\x68"
    tx = bus.last()
    assert (tx.bus, tx.op, tx.address, tx.register_name, tx.data, tx.ack) == ("I2C1", "READ", 0x68, "WHO_AM_I", b"\x68", True)
    assert tx.time == "10:24:30.200" and tx.phases[0] == "START" and tx.phases[-1] == "STOP"
    assert "ADDR 0x68 W" in tx.phases and "RESTART" in tx.phases and "ADDR 0x68 R" in tx.phases
    assert "WHO_AM_I" in tx.summary() and "0x68" in tx.summary()
    bus.write(0x68, m.REG_PWR_MGMT_1, 0)
    assert bus.last().op == "WRITE" and not dev.sleeping
    with pytest.raises(I2CNack):
        bus.read(0x69, 0x00, 1)
    assert bus.nack_count == 1 and bus.last().ack is False
    for _ in range(10):
        bus.read(0x68, m.REG_WHO_AM_I)
    assert len(bus.log) == 5 and bus.transaction_count == 13             # bounded log, full count


def test_stm32_boot_sequence_is_visible_on_the_bus():
    rig = VirtualHardwareRig(0)
    ops = [(t.op, t.register_name) for t in rig.i2c.log]
    assert ops[0] == ("READ", "WHO_AM_I")
    assert ("WRITE", "PWR_MGMT_1") in ops and ("WRITE", "SMPLRT_DIV") in ops and ("WRITE", "ACCEL_CONFIG") in ops
    assert rig.mpu.sample_rate_hz == pytest.approx(10.0) and not rig.mpu.sleeping


# 13 --------------------------------------------------------------- UART
def test_uart_telemetry_packets():
    rig, _ = run_rig("Normal Driving", 1, 10)
    u = rig.uart
    assert u.baud == 115200 and u.name == "USART2" and u.packet_count == 10 and len(u.history) == 10
    fields = u.last_packet.text.split(",")
    assert len(fields) == 12 and fields[0] == "1000" and fields[-1] == "1"
    assert u.last_packet.wire_time_ms == pytest.approx(u.last_packet.n_bytes * 10 * 1000 / 115200)
    assert u.last_packet.wire_time_ms < 100
    assert u.tx_buffer_bytes > 0
    first = u.read_line()
    assert first.startswith(b"100,") and first.endswith(b"\r\n")
    clock = VirtualClock(0)
    bare = VirtualUART(clock, baud=9600)
    pkt = bare.transmit("hello\r\n")
    assert pkt.wire_time_ms == pytest.approx(7 * 10 * 1000 / 9600) and bare.last_packet is pkt
    assert bare.read_line() == b"hello\r\n" and bare.read_line() is None
    with pytest.raises(ValueError):
        VirtualUART(clock, baud=0)


# 14 --------------------------------------------------------------- STM32
def test_stm32_sampling_loop_state():
    rig = VirtualHardwareRig(0)
    assert rig.stm32.cpu_state == "RUNNING" and rig.stm32.samples == 0 and rig.stm32.sample_rate_hz == 10
    for i in range(1, 8):
        res = rig.tick()
        assert res.packet is not None and rig.stm32.samples == i
    snap = rig.snapshot()["stm32"]
    assert (snap["cpu"], snap["i2c1"], snap["usart2"], snap["baud"], snap["gpio_status"]) == \
        ("RUNNING", "ACTIVE", "ACTIVE", 115200, "OK")
    assert snap["uptime_ms"] == 700 and snap["led"] == "GREEN" and snap["status_text"] == "NO EVENT"


def test_stm32_faults_when_the_sensor_is_missing_or_wrong():
    rig = VirtualHardwareRig(0)
    rig.i2c._devices.clear()                       # sensor unplugged
    rig.stm32.reset()
    assert rig.stm32.boot() is False and rig.stm32.cpu_state == "FAULT" and "no device" in rig.stm32.fault_reason
    assert rig.tick().packet is None and rig.stm32.samples == 0
    rig2 = VirtualHardwareRig(0)
    rig2.mpu._regs[m.REG_WHO_AM_I] = 0x70          # wrong chip
    rig2.stm32.reset()
    assert rig2.stm32.boot() is False and "WHO_AM_I" in rig2.stm32.fault_reason


def test_stm32_flags_events_and_drives_the_alert_output():
    rig, _ = run_rig("Severe Accident", 1, 60)
    assert rig.stm32.event_count >= 1
    rig.set_emergency_status("ALERT COUNTDOWN")
    s = rig.snapshot()
    assert s["stm32"]["alert_output"] and s["stm32"]["buzzer"] and s["stm32"]["led"] == "RED"
    assert s["stm32"]["status_text"] == "ALERT OUTPUT ACTIVE" and s["alert_unit"]["state"].startswith("ARMED")
    rig.set_emergency_status("ALERT SENT")
    assert rig.alert_unit.state == "SIMULATED ALERT SENT" and rig.stm32.buzzer
    rig.set_emergency_status("ALERT CANCELLED")
    assert not rig.stm32.alert_output and rig.alert_unit.state == "CANCELLED" and not rig.stm32.buzzer
    assert rig.uart_alert.packet_count == 3 and rig.uart_alert.name == "USART1"


# 15-17 ------------------------------------------------------------ SensorSource + pipeline
def test_virtual_source_implements_the_sensor_source_interface():
    src = VirtualHardwareSensorSource(seed=1, run_ticks=5)
    assert isinstance(src, SensorSource)
    assert src.read() is None                                   # not started
    src.start("Pothole")
    readings = []
    while (r := src.read()) is not None:
        assert r is not NO_DATA_YET
        readings.append(r)
    assert len(readings) == 5 and src.progress == (5, 5) and src.last_error is None
    assert all(set(r) == set(FEATURE_COLUMNS) and all(isinstance(v, float) for v in r.values()) for r in readings)
    src.stop()
    assert src.read() is None
    with pytest.raises(RuntimeError):
        src.start("Not a scenario")


def test_ten_sequential_simulation_ticks():
    src = VirtualHardwareSensorSource(seed=9)
    src.start("Normal Driving")
    stamps = []
    for i in range(10):
        r = src.read()
        stamps.append(src.last_timestamp_ms)
        assert abs(r["accel_z_g"] - 1.0) < 0.3 and 20 < r["speed_kmh"] < 80 and r["vibration_level"] > 0
    assert stamps == [100 * (i + 1) for i in range(10)]
    snap = src.rig.snapshot()
    assert snap["stm32"]["samples"] == 10 and snap["uart"]["packets"] == 10 and snap["tick"] == 10
    assert src.gps_position is not None and src.hardware_info()["tick"] == 10


def test_virtual_readings_feed_the_existing_ai_pipeline_without_the_scenario_label():
    def labels(scenario, seed=7):
        src = VirtualHardwareSensorSource(seed=seed)
        src.start(scenario)
        buf, out = RollingWindowBuffer(), []
        while (r := src.read()) is not None:
            assert "scenario" not in r                                   # the classifier never sees the scenario
            buf.push(r)
            if buf.is_full():
                d = decide_window(buf.as_list())
                out.append(d)
        return out
    braking, severe = labels("Hard Braking"), labels("Severe Accident")
    assert len(braking) == 80 - 9
    for d in braking[:1] + severe[:1]:
        assert d["ai"]["label"] and 0 <= d["ai"]["confidence"] <= 1 and d["ai"]["probabilities"]
        assert d["severity"]["severity"] and d["threshold"]["classification"] in ("ACCIDENT", "NORMAL")
    # the AI's answer comes from the signal: different physics -> different predictions and Phase 9 severity
    assert {d["ai"]["label"] for d in braking} != {d["ai"]["label"] for d in severe}
    assert max(d["severity"]["score"] for d in severe) > max(d["severity"]["score"] for d in braking)
    assert sum(d["ai"]["label"] == "Hard Braking" for d in braking) > len(braking) / 2


def test_hand_over_to_emergency_state_machine():
    src = VirtualHardwareSensorSource(seed=1)
    src.start("Severe Accident")
    buf, status = RollingWindowBuffer(), "MONITORING"
    while (r := src.read()) is not None:
        buf.push(r)
        if buf.is_full():
            status = next_status(status, severity_for_label(decide_window(buf.as_list())["ai"]["label"]))
            src.rig.set_emergency_status(status)
    assert status == "ALERT COUNTDOWN"
    snap = src.rig.snapshot()
    assert snap["stm32"]["alert_output"] and snap["stm32"]["led"] == "RED" and snap["alert_unit"]["state"].startswith("ARMED")


# 18-19 ------------------------------------------------------------ recording / replay
def test_virtual_run_recording_round_trips_and_replays(tmp_path):
    store = RecordingStore(tmp_path)
    src = VirtualHardwareSensorSource(seed=2, run_ticks=30)
    src.start("Severe Accident")
    rec = store.start_recording("Severe Accident", "virtual_hardware", 10.0)
    rec.set_virtual_hardware({"seed": 2, "notice": "VIRTUAL HARDWARE SIMULATION"})
    buf = RollingWindowBuffer()
    while (r := src.read()) is not None:
        buf.push(r)
        rec.append_sample(r, gps=src.gps_position, hardware=src.hardware_info())
        if buf.is_full():
            d = decide_window(buf.as_list())
            rec.append_decision(ai=d["ai"], threshold=d["threshold"], severity=d["severity"],
                                legacy_severity=d["legacy_severity"], status="MONITORING")
    saved = rec.finish(save=True)
    d = saved.data
    assert d["source_type"] == "virtual_hardware" and d["synthetic"] is True and d["scenario"] == "Severe Accident"
    assert [s["t"] for s in d["samples"][:3]] == [0.0, 0.1, 0.2]                       # simulated time, not wall clock
    assert d["virtual_hardware"]["seed"] == 2 and all("gps" in s and "hw" in s for s in d["samples"])
    assert any(s["hw"]["saturated"] for s in d["samples"]) and "mcu_event" in d["samples"][0]["hw"]
    assert validate_recording_dict(d) == []
    loaded = store.load(saved.recording_id)
    assert loaded.summary["n_samples"] == 30 and loaded.summary["severity_max"]
    session = rp.ReplaySession(loaded)
    assert session.n_samples == 30
    state = session.state_at(29)
    assert state["index"] == 29 and state["gps"] == pytest.approx(tuple(d["samples"][29]["gps"]))
    assert state["ai"]["label"] and state["severity"]["severity"] and state["status"] == "MONITORING"
    assert state["sample"]["hw"]["tick"] == 30


def test_recordings_without_the_new_optional_blocks_still_validate(tmp_path):
    store = RecordingStore(tmp_path)
    rec = store.start_recording("Normal Driving", "simulated", 10.0)
    rec.append_sample({k: 1.0 for k in FEATURE_COLUMNS}, gps=(12.9, 77.5))
    d = rec.finish(save=True).data
    assert "virtual_hardware" not in d and "hw" not in d["samples"][0] and d["synthetic"] is True
    bad = dict(d, virtual_hardware="nope")
    assert validate_recording_dict(bad)


# UI data ---------------------------------------------------------------------------
def test_lab_html_shows_the_hardware_and_the_honesty_labels():
    rig = VirtualHardwareRig(1, "Severe Accident")
    while not rig.mpu.saturated:
        rig.tick()
    pipeline = {"label": "Severe Accident", "confidence": 0.9, "severity": "HIGH (80)",
                "status": "ALERT COUNTDOWN", "countdown": 7}
    rig.set_emergency_status("ALERT COUNTDOWN")
    html = build_lab_html(rig.snapshot(), pipeline, "MPU6050", animate=True)
    for needle in ("MPU6050", "STM32", "I²C1", "USART2", "ESP32 / GSM", "SIMULATED GPS", "HOST / AI",
                   "NO REAL SMS/GSM MESSAGE WAS SENT", "VIRTUAL HARDWARE SIMULATION", "ALERT OUTPUT ACTIVE",
                   "animateMotion", "0x68", "ACTIVE", "SYSTEM ONLINE", "EMERGENCY COUNTDOWN", "ARMED",
                   "VIRTUAL VEHICLE", "VIBRATION"):
        assert needle in html, needle
    assert html.count("<svg") == 1 and 'class=pulse' not in html          # well-formed: every element is closed
    calm = VirtualHardwareRig(0)
    calm.tick()
    html2 = build_lab_html(calm.snapshot(), {}, "STM32", animate=False)
    assert "SATURATED" not in html2 and "animateMotion" not in html2 and "SYSTEM NORMAL" in html2
    assert "SATURATED" in html


def test_schematic_animation_is_tied_to_simulation_state():
    rig = VirtualHardwareRig(1, "Normal Driving")
    for _ in range(5):
        rig.tick()
    calm = build_lab_html(rig.snapshot(), {}, animate=True)
    assert 'class="ishake"' not in calm and 'fill="#2ecc71"' in calm and "SATURATED" not in calm
    assert "M430 570 V615" in calm and calm.count('fill="#ff4d5e"><animateMotion') == 0   # no alert packets
    crash = VirtualHardwareRig(1, "Severe Accident")
    while crash.last_vehicle is None or crash.last_vehicle.vehicle_state != "COLLISION":
        crash.tick()
    html = build_lab_html(crash.snapshot(), {}, animate=True)
    assert 'class="ishake"' in html and "COLLISION" in html and "<polygon" in html      # impact flash
    crash.set_emergency_status("ALERT COUNTDOWN")
    assert 'fill="#ff4d5e"><animateMotion' in build_lab_html(crash.snapshot(), {}, animate=True)
    roll = VirtualHardwareRig(1, "Rollover")
    while roll.last_vehicle is None or roll.last_vehicle.roll_deg < 90:
        roll.tick()
    assert f"rotate({roll.last_vehicle.roll_deg:.1f}" in build_lab_html(roll.snapshot(), {}, animate=False)


def test_oscilloscope_shows_one_channel_of_live_data():
    hist = {"accel_mag": [1.0, 1.1, 3.0], "gyro_mag": [0, 5, 9], "speed": [50, 40, 10], "vibration": [0.2, 1, 5]}
    assert list(SCOPE_CHANNELS) == ["ACCEL", "GYRO", "SPEED", "VIBRATION"]
    a, v = build_scope_html(hist, "ACCEL"), build_scope_html(hist, "VIBRATION")
    assert a.count("<polyline") == 1 and "3.00" in a and "5.00" in v and a != v
    assert "<polyline" not in build_scope_html({}, "SPEED")                        # no data yet: empty graticule


def test_page_panels_state_hierarchy_and_honesty():
    rig = VirtualHardwareRig(1, "Severe Accident")
    snap = rig.snapshot()
    normal = emergency_html(snap, {"status": "MONITORING"})
    assert "SYSTEM NORMAL" in normal and "NO REAL SMS/GSM MESSAGE WAS SENT" in normal
    cd = emergency_html(snap, {"status": "ALERT COUNTDOWN", "countdown": 9})
    assert "EMERGENCY COUNTDOWN" in cd and ">9<" in cd
    assert "SIMULATED ALERT SENT" in emergency_html(snap, {"status": "ALERT SENT"})
    assert "EVENT DETECTED" in emergency_html(snap, {"status": "EVENT DETECTED"})
    ai = ai_html({"label": "Hard Braking", "confidence": 0.724, "severity": "CRITICAL (92)",
                  "threshold": "ACCIDENT", "status": "ALERT COUNTDOWN"})
    assert "Hard Braking" in ai and "72.4%" in ai and "CRITICAL" in ai and "can misclassify" in ai
    head = header_html(snap, {"run_state": "RUNNING"}, True)
    for needle in ("VIRTUAL HARDWARE LAB", "VIRTUAL SIMULATION", "10 Hz", "RUNNING", "SIMULATED", "SYNTHETIC DATA", "NO REAL GSM"):
        assert needle in head
    assert "STANDBY" in header_html(snap, {}, False)
    assert "\n\n" not in head + ai + normal      # never parsed as a Markdown code block
    assert "COMPONENT INSPECTOR" in inspector_html([("A", "1")], "MPU6050")


def test_component_inspector_rows():
    rig, _ = run_rig("Severe Accident", 1, 36)
    snap = rig.snapshot()
    rows = {c: dict(inspector_rows(snap, c)) for c in COMPONENTS}
    mp = rows["MPU6050"]
    assert mp["DEVICE"] == "MPU6050" and mp["I2C ADDRESS"] == "0x68" and mp["WHO_AM_I"] == "0x68"
    assert mp["ACCEL RANGE"] == "±2g" and mp["GYRO RANGE"] == "±250°/s"
    assert all(k in mp for k in ("ACCEL X", "ACCEL Y", "ACCEL Z", "GYRO X", "GYRO Y", "GYRO Z", "SATURATION"))
    st32 = rows["STM32"]
    assert st32["CPU"] == "RUNNING" and st32["BAUD"] == "115200" and st32["SAMPLE RATE"] == "10 Hz"
    assert int(st32["SAMPLES"]) == 36 and st32["I2C1"] in ("ACTIVE", "IDLE")
    assert rows["GPS"]["FIX"] == "YES" and "SIMULATED" in rows["GPS"]["DEVICE"]
    assert float(rows["Vibration sensor"]["LEVEL"]) >= 0
    assert "NO REAL SMS/GSM MESSAGE WAS SENT" in rows["ESP32/GSM"]["REAL SMS/GSM SENT"]
    assert math.isfinite(float(rows["Vehicle"]["SPEED"].split()[0]))
