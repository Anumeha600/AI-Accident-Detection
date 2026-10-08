"""sensor_source.parse_stm32_line(): valid, malformed, missing and invalid values."""

import pytest

from sensor_source import parse_stm32_line, STM32_SENSOR_FIELDS

VALID = "1234,0.02,-0.01,1.01,2.3,-1.7,4.2"


def test_valid_line_str():
    r = parse_stm32_line(VALID)
    assert r == {
        "accel_x_g": 0.02, "accel_y_g": -0.01, "accel_z_g": 1.01,
        "gyro_x_dps": 2.3, "gyro_y_dps": -1.7, "gyro_z_dps": 4.2,
    }


def test_valid_line_bytes_with_crlf():
    assert parse_stm32_line(VALID.encode() + b"\r\n") == parse_stm32_line(VALID)


def test_timestamp_is_dropped_and_only_sensor_fields_returned():
    r = parse_stm32_line(VALID)
    assert set(r) == set(STM32_SENSOR_FIELDS)
    assert "timestamp_ms" not in r
    assert all(isinstance(v, float) for v in r.values())


def test_integers_and_scientific_notation_accepted():
    r = parse_stm32_line("0,0,0,1,0,0,0")
    assert r["accel_z_g"] == 1.0
    assert parse_stm32_line("1,1e-2,0,1,0,0,0")["accel_x_g"] == pytest.approx(0.01)


def test_surrounding_whitespace_tolerated():
    assert parse_stm32_line("  " + VALID + " \n") is not None


@pytest.mark.parametrize("line", [
    "",                                     # empty
    "   \r\n",                              # whitespace only
    b"",                                    # empty bytes
    "garbage,not,a,valid,line",             # wrong field count + non-numeric
    "1234,0.02,-0.01,1.01,2.3,-1.7",        # too few fields (6)
    "1234,0.02,-0.01,1.01,2.3,-1.7,4.2,9",  # too many fields (8)
    "hello",                                # no commas at all
    "1234;0.02;-0.01;1.01;2.3;-1.7;4.2",    # wrong delimiter
])
def test_malformed_lines_return_none(line):
    assert parse_stm32_line(line) is None


@pytest.mark.parametrize("line", [
    "1234,,-0.01,1.01,2.3,-1.7,4.2",        # missing value (empty field)
    "1234,0.02,-0.01,1.01,2.3,-1.7,",       # trailing missing value
    "1234,0.02,abc,1.01,2.3,-1.7,4.2",      # non-numeric value
    "1234,0.02,-0.01,1.01,2.3,-1.7,4.2x",   # junk suffix
    "abc,0.02,-0.01,1.01,2.3,-1.7,4.2",     # non-numeric timestamp
])
def test_missing_or_non_numeric_values_return_none(line):
    assert parse_stm32_line(line) is None


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "NaN"])
def test_non_finite_values_rejected(bad):
    assert parse_stm32_line(f"1,{bad},0,1,0,0,0") is None
    assert parse_stm32_line(f"1,0,0,1,0,{bad},0") is None


@pytest.mark.parametrize("line", [
    "1,20.01,0,1,0,0,0",    # accel x just above limit
    "1,0,-20.01,1,0,0,0",   # accel y just below limit
    "1,0,0,9999,0,0,0",     # accel z way out of range
    "1,0,0,1,3000.5,0,0",   # gyro x above limit
    "1,0,0,1,0,-3001,0",    # gyro y below limit
    "1,0,0,1,0,0,1e9",      # gyro z absurd
])
def test_out_of_range_values_rejected(line):
    assert parse_stm32_line(line) is None


def test_values_exactly_on_the_limit_are_accepted():
    assert parse_stm32_line("1,20,-20,20,3000,-3000,3000") is not None


@pytest.mark.parametrize("raw", [b"\xff\xfe,1,2,3,4,5,6", b"12\x80,0,0,1,0,0,0"])
def test_non_ascii_bytes_return_none(raw):
    assert parse_stm32_line(raw) is None
