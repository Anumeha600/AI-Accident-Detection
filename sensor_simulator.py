"""
SIMULATED / SYNTHETIC SENSOR DATA GENERATOR
--------------------------------------------
Project: AI-Based Smart Accident Detection & Emergency Response System (STM32)
Phase 1: Sensor Data Simulator

IMPORTANT:
    All data produced by this script is 100% SIMULATED / SYNTHETIC.
    It does NOT come from a real accelerometer, gyroscope, or any physical
    sensor, and must NOT be treated as real experimental measurements.
    It exists only to develop and test the rest of the pipeline before
    real STM32 sensor hardware is connected.

WHAT THIS SCRIPT DOES:
    1. Lets you pick one of six driving scenarios.
    2. Generates a short time-series of fake sensor readings for that
       scenario (accelerometer, gyroscope, vibration, speed).
    3. Prints the generated readings to the terminal.
    4. Saves the full reading to a CSV file.

DESIGN NOTE (for future phases):
    All simulation logic lives inside generate_sensor_data(). Later, when
    real STM32 hardware is available, this is the ONLY function that needs
    to be swapped out (e.g. for a function that reads from a serial/UART
    port). Everything else -- display, CSV saving, the menu -- can stay
    the same.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SAMPLE_RATE_HZ = 10      # how many sensor readings per second
DURATION_SECONDS = 5     # how long each simulated recording lasts
NUM_SAMPLES = SAMPLE_RATE_HZ * DURATION_SECONDS  # -> 50 samples per run

SCENARIOS = {
    "1": "Normal Driving",
    "2": "Pothole",
    "3": "Hard Braking",
    "4": "Sharp Turn",
    "5": "Minor Accident",
    "6": "Severe Accident",
}

# Keep results the same across runs while still "looking" random.
# Comment out this line if you want different random data every run.
np.random.seed(42)


# ---------------------------------------------------------------------------
# SMALL MATH HELPERS
# ---------------------------------------------------------------------------

def noise(n, std):
    """Random small fluctuations (sensor jitter) centered on 0."""
    return np.random.normal(loc=0.0, scale=std, size=n)


def gaussian_bump(n, center, width, amplitude):
    """
    A smooth "bump" shape that rises and falls, used to simulate a short
    physical event (a bump in the road, an impact, etc.) instead of an
    unrealistic instant spike.
    """
    idx = np.arange(n)
    return amplitude * np.exp(-((idx - center) ** 2) / (2 * width ** 2))


def sigmoid_transition(n, center, steepness, start_value, end_value):
    """
    A smooth S-shaped transition from start_value to end_value, used to
    simulate speed dropping (e.g. during braking or a crash) instead of
    changing instantly.
    """
    idx = np.arange(n)
    s = 1 / (1 + np.exp(-steepness * (idx - center)))
    return start_value + (end_value - start_value) * s


# ---------------------------------------------------------------------------
# SCENARIO GENERATORS
# Each function returns a dictionary of numpy arrays, one per sensor channel.
# Accelerometer values are in "g" (1g = normal gravity, resting Z axis = 1g).
# Gyroscope values are in degrees/second. Vibration is a unitless 0-10 scale.
# Speed is in km/h.
# ---------------------------------------------------------------------------

def simulate_normal_driving(n):
    return {
        "accel_x": noise(n, 0.05),
        "accel_y": noise(n, 0.05),
        "accel_z": 1.0 + noise(n, 0.03),          # resting on gravity
        "gyro_x": noise(n, 1.0),
        "gyro_y": noise(n, 1.0),
        "gyro_z": noise(n, 1.0),
        "vibration": np.abs(noise(n, 0.3)) + 0.2,  # gentle road vibration
        "speed": 50 + noise(n, 2),                 # steady ~50 km/h
    }


def simulate_pothole(n):
    center = n // 2
    dip = gaussian_bump(n, center, width=1.2, amplitude=1.5)  # sharp, brief
    return {
        "accel_x": noise(n, 0.05),
        "accel_y": noise(n, 0.05),
        "accel_z": 1.0 + noise(n, 0.03) - dip,     # wheel drops then rebounds
        "gyro_x": noise(n, 1.0) + dip * 5,
        "gyro_y": noise(n, 1.0),
        "gyro_z": noise(n, 1.0),
        "vibration": np.abs(noise(n, 0.3)) + 0.2 + dip * 3,  # spike in vibration
        "speed": 50 + noise(n, 2) - dip * 3,       # tiny, brief speed dip
    }


def simulate_hard_braking(n):
    center = n // 2
    brake = gaussian_bump(n, center, width=6, amplitude=0.6)  # longer event
    return {
        "accel_x": noise(n, 0.05) - brake,          # strong deceleration
        "accel_y": noise(n, 0.05),
        "accel_z": 1.0 + noise(n, 0.03),
        "gyro_x": noise(n, 1.0),
        "gyro_y": noise(n, 1.0),
        "gyro_z": noise(n, 1.0),
        "vibration": np.abs(noise(n, 0.3)) + 0.2 + brake * 2,
        "speed": sigmoid_transition(n, center, steepness=0.8,
                                     start_value=70, end_value=20) + noise(n, 1.5),
    }


def simulate_sharp_turn(n):
    center = n // 2
    turn = gaussian_bump(n, center, width=4, amplitude=1.0)
    return {
        "accel_x": noise(n, 0.05),
        "accel_y": noise(n, 0.05) + turn * 0.5,     # lateral (sideways) force
        "accel_z": 1.0 + noise(n, 0.03),
        "gyro_x": noise(n, 1.0),
        "gyro_y": noise(n, 1.0),
        "gyro_z": noise(n, 1.0) + turn * 100,       # fast rotation around Z
        "vibration": np.abs(noise(n, 0.3)) + 0.2 + turn * 1.5,
        "speed": 45 + noise(n, 2) - turn * 5,       # slight slow-down in turn
    }


def simulate_minor_accident(n):
    center = n // 2
    impact = gaussian_bump(n, center, width=2, amplitude=1.0)
    return {
        "accel_x": noise(n, 0.1) - impact * 2.5,
        "accel_y": noise(n, 0.1) + impact * 1.5,
        "accel_z": 1.0 + noise(n, 0.1) + impact * 2.0,
        "gyro_x": noise(n, 3) + impact * 40,
        "gyro_y": noise(n, 3) + impact * 30,
        "gyro_z": noise(n, 3) + impact * 40,
        "vibration": np.abs(noise(n, 0.5)) + 0.3 + impact * 5,
        "speed": sigmoid_transition(n, center, steepness=1.0,
                                     start_value=45, end_value=10) + noise(n, 1.5),
    }


def simulate_severe_accident(n):
    center = n // 2
    impact = gaussian_bump(n, center, width=2, amplitude=1.0)
    chaos = noise(n, 1.0) * impact  # extra chaotic shaking during the impact
    return {
        "accel_x": noise(n, 0.15) - impact * 9 + chaos,
        "accel_y": noise(n, 0.15) + impact * 6 + chaos,
        "accel_z": 1.0 + noise(n, 0.15) + impact * 8 + chaos,
        "gyro_x": noise(n, 5) + impact * 150,
        "gyro_y": noise(n, 5) + impact * 120,
        "gyro_z": noise(n, 5) + impact * 150,
        "vibration": np.abs(noise(n, 0.5)) + 0.3 + impact * 9,
        "speed": sigmoid_transition(n, center, steepness=1.2,
                                     start_value=60, end_value=0) + np.abs(noise(n, 1.0)),
    }


SCENARIO_GENERATORS = {
    "Normal Driving": simulate_normal_driving,
    "Pothole": simulate_pothole,
    "Hard Braking": simulate_hard_braking,
    "Sharp Turn": simulate_sharp_turn,
    "Minor Accident": simulate_minor_accident,
    "Severe Accident": simulate_severe_accident,
}


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT FOR GENERATING DATA
# (This is the function to swap out later for real STM32 sensor readings.)
# ---------------------------------------------------------------------------

def generate_sensor_data(scenario_name, num_samples=NUM_SAMPLES, sample_rate_hz=SAMPLE_RATE_HZ):
    """
    Generate a synthetic time-series of sensor data for the given scenario.
    Returns a pandas DataFrame with one row per sample.
    """
    generator = SCENARIO_GENERATORS[scenario_name]
    raw = generator(num_samples)

    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds=i / sample_rate_hz) for i in range(num_samples)]

    # Speed can never physically be negative -- clip it at 0.
    speed = np.clip(raw["speed"], a_min=0, a_max=None)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "accel_x_g": raw["accel_x"],
        "accel_y_g": raw["accel_y"],
        "accel_z_g": raw["accel_z"],
        "gyro_x_dps": raw["gyro_x"],
        "gyro_y_dps": raw["gyro_y"],
        "gyro_z_dps": raw["gyro_z"],
        "vibration_level": np.clip(raw["vibration"], a_min=0, a_max=None),
        "speed_kmh": speed,
        "scenario": scenario_name,
    })

    # Round numeric columns for cleaner display/CSV output.
    numeric_cols = df.columns.drop(["timestamp", "scenario"])
    df[numeric_cols] = df[numeric_cols].round(3)

    return df


def save_to_csv(df, scenario_name):
    filename = f"simulated_{scenario_name.lower().replace(' ', '_')}.csv"
    df.to_csv(filename, index=False)
    return filename


def display_data(df, scenario_name):
    print("\n" + "=" * 70)
    print(f"SIMULATED / SYNTHETIC DATA -- Scenario: {scenario_name}")
    print("(This is NOT real sensor data -- generated for testing only)")
    print("=" * 70)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print(df.head(10).to_string(index=False))
    print(f"... ({len(df)} total samples generated) ...")
    print(df.tail(5).to_string(index=False))
    print("-" * 70)


def run_scenario(scenario_name):
    df = generate_sensor_data(scenario_name)
    display_data(df, scenario_name)
    filename = save_to_csv(df, scenario_name)
    print(f"Saved {len(df)} rows to: {filename}\n")


def main():
    print("SIMULATED SENSOR DATA GENERATOR (Phase 1 prototype)")
    print("All output is SYNTHETIC test data, not real sensor readings.\n")
    print("Select a scenario to simulate:")
    for key, name in SCENARIOS.items():
        print(f"  {key}. {name}")
    print("  7. Run ALL scenarios")

    choice = input("\nEnter your choice (1-7): ").strip()

    if choice == "7":
        for name in SCENARIOS.values():
            run_scenario(name)
    elif choice in SCENARIOS:
        run_scenario(SCENARIOS[choice])
    else:
        print("Invalid choice. Please run the script again and enter 1-7.")


if __name__ == "__main__":
    main()
