# data_simulator.py
# Author: Senior AI Engineer and IoT Specialist
# Description: Generates synthetic time-series accelerometer data for normal
#              yarn winding operations and for yarn break events.

import numpy as np
import pandas as pd
import os

# --- Configuration ---
SAMPLING_RATE_HZ = 4000  # Sampling rate of the MEMS accelerometer
DURATION_S = 10          # Duration of the signal to generate in seconds
NORMAL_DATA_FILE = "normal_data.csv"
BREAK_DATA_FILE = "break_data.csv"
BREAK_TIME_S = 5         # Time at which the yarn break anomaly occurs

def generate_normal_signal(duration_s, sampling_rate_hz):
    """
    Generates a realistic baseline signal simulating normal machine operation.

    The signal is composed of multiple sine waves to mimic machine hum and
    Gaussian noise to represent the yarn's vibration against the sensor.

    Args:
        duration_s (int): The duration of the signal in seconds.
        sampling_rate_hz (int): The sampling rate of the sensor in Hz.

    Returns:
        np.ndarray: A 1D NumPy array representing the normal vibration signal.
    """
    print(f"Generating normal signal ({duration_s}s at {sampling_rate_hz}Hz)...")
    num_samples = int(duration_s * sampling_rate_hz)
    time = np.linspace(0, duration_s, num_samples, endpoint=False)

    # 1. Baseline machine hum (sum of a few sine waves)
    signal = 0.5 * np.sin(2 * np.pi * 50 * time)   # 50Hz component
    signal += 0.2 * np.sin(2 * np.pi * 120 * time)  # 120Hz component
    signal += 0.1 * np.sin(2 * np.pi * 300 * time)  # 300Hz component

    # 2. Add low-amplitude Gaussian noise for normal yarn vibration
    noise = np.random.normal(0, 0.1, num_samples)
    normal_signal = signal + noise

    print("Normal signal generated.")
    return normal_signal

def generate_break_signal(duration_s, sampling_rate_hz, break_time_s):
    """
    Generates a signal that includes a yarn break anomaly.

    It starts with a normal signal and introduces a sharp, high-frequency
    transient at a specified time to simulate the "snap" of a yarn break.

    Args:
        duration_s (int): The total duration of the signal in seconds.
        sampling_rate_hz (int): The sampling rate of the sensor in Hz.
        break_time_s (int): The timestamp (in seconds) where the break occurs.

    Returns:
        np.ndarray: A 1D NumPy array representing the signal with a break event.
    """
    print(f"Generating yarn break signal (break at {break_time_s}s)...")
    # First, generate the normal baseline signal
    normal_signal = generate_normal_signal(duration_s, sampling_rate_hz)

    # 2. Introduce a sharp transient for the yarn break
    break_start_index = int(break_time_s * sampling_rate_hz)
    break_duration_samples = int(0.05 * sampling_rate_hz) # 50ms duration for the snap
    break_end_index = break_start_index + break_duration_samples

    # Create the anomaly: a high-frequency, high-amplitude burst
    break_time = np.linspace(0, 0.05, break_duration_samples, endpoint=False)
    break_anomaly = 2.5 * np.sin(2 * np.pi * 800 * break_time) # High-frequency snap
    break_anomaly += np.random.normal(0, 0.5, break_duration_samples) # Add noise to the snap

    # Inject the anomaly into the normal signal
    break_signal = normal_signal.copy()
    break_signal[break_start_index:break_end_index] += break_anomaly

    print("Yarn break signal generated.")
    return break_signal

def save_to_csv(signal, filename):
    """Saves the generated signal to a CSV file."""
    print(f"Saving signal to {filename}...")
    df = pd.DataFrame({'sensor_value': signal})
    df.to_csv(filename, index=False)
    print(f"Successfully saved to {filename}.")

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Generate and save the normal operation data
    normal_signal = generate_normal_signal(DURATION_S, SAMPLING_RATE_HZ)
    save_to_csv(normal_signal, os.path.join('data', NORMAL_DATA_FILE))

    # Generate and save the yarn break data
    break_signal = generate_break_signal(DURATION_S, SAMPLING_RATE_HZ, BREAK_TIME_S)
    save_to_csv(break_signal, os.path.join('data', BREAK_DATA_FILE))

    print("\nData simulation complete.")
    print(f"Files saved in 'data/' directory: {NORMAL_DATA_FILE}, {BREAK_DATA_FILE}")
