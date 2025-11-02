# run_on_pi.py
# Author: Senior AI Engineer and IoT Specialist
# Description: Real-time anomaly detection script for a Raspberry Pi.
#              It loads the trained FastAI model, reads sensor data in a loop,
#              calculates reconstruction loss, and triggers a GPIO pin if a
#              yarn break is detected.

import time
import numpy as np
import torch
from fastai.learner import load_learner
import json
# Mock GPIO for development on non-Pi systems
try:
    import RPi.GPIO as GPIO
    IS_PI = True
except (ImportError, RuntimeError):
    print("WARNING: RPi.GPIO library not found. Using mock GPIO.")
    IS_PI = False

# --- Configuration ---
MODEL_PATH = 'yarn_break_model.pkl'
THRESHOLD_PATH = 'anomaly_threshold.json'
WINDOW_SIZE = 1024  # Must match the window size used in training
RELAY_PIN = 17      # BCM pin number to control the relay/machine stop
SERIAL_PORT = '/dev/ttyUSB0' # Example serial port for ESP32
BAUD_RATE = 115200

# --- Placeholder for Serial Communication ---
# In a real application, you would use pyserial to read data from the ESP32
# For this script, we simulate it.
# import serial
# try:
#     ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
# except:
#     ser = None
#     print(f"WARNING: Could not open serial port {SERIAL_PORT}")

def setup_gpio():
    """Initializes GPIO pins."""
    if IS_PI:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RELAY_PIN, GPIO.OUT)
        GPIO.output(RELAY_PIN, GPIO.LOW) # Ensure relay is off initially
        print(f"GPIO pin {RELAY_PIN} initialized as OUTPUT.")

def trigger_relay(state):
    """Activates or deactivates the relay."""
    if IS_PI:
        if state:
            GPIO.output(RELAY_PIN, GPIO.HIGH) # Turn relay ON
        else:
            GPIO.output(RELAY_PIN, GPIO.LOW) # Turn relay OFF

def read_data_from_sensor(window_size):
    """
    Placeholder function to read a window of data from the sensor.
    In a real implementation, this function would handle serial communication
    with the ESP32 to receive `window_size` samples.
    """
    # --- REAL IMPLEMENTATION LOGIC ---
    # buffer = []
    # while len(buffer) < window_size:
    #     if ser and ser.in_waiting > 0:
    #         line = ser.readline().decode('utf-8').rstrip()
    #         try:
    #             buffer.append(float(line))
    #         except ValueError:
    #             pass # Ignore non-numeric lines
    # return np.array(buffer)
    # --- SIMULATION LOGIC ---
    # Simulate a small chance of a break for demonstration
    if np.random.rand() > 0.99:
        # Generate a simulated break signal
        anomaly = 2.5 * np.sin(np.linspace(0, 2 * np.pi * 5, 100))
        signal = np.random.normal(0, 0.1, window_size)
        signal[500:600] += anomaly
    else:
        # Generate normal noise
        signal = np.random.normal(0, 0.1, window_size)
    
    time.sleep(0.1) # Simulate data acquisition time
    return signal.astype(np.float32)

def preprocess_data(data, mean, std):
    """Normalizes the raw sensor data using stats from training."""
    return (data - mean) / std

def calculate_loss(input_tensor, reconstructed_tensor):
    """Calculates the Mean Squared Error between two tensors."""
    return torch.nn.functional.mse_loss(input_tensor, reconstructed_tensor).item()

if __name__ == "__main__":
    print("Initializing Yarn Break Detection System on Pi...")

    # 1. Load Model and Threshold
    try:
        learn = load_learner(MODEL_PATH)
        print("FastAI model loaded successfully.")
        with open(THRESHOLD_PATH, 'r') as f:
            config = json.load(f)
        threshold = config['threshold']
        norm_mean = config['normalization_mean']
        norm_std = config['normalization_std']
        print(f"Anomaly threshold and normalization stats loaded: Threshold={threshold:.6f}")
    except FileNotFoundError:
        print(f"ERROR: Model '{MODEL_PATH}' or threshold file '{THRESHOLD_PATH}' not found.")
        print("Please run train_model.py first.")
        exit()

    # 2. Setup GPIO
    setup_gpio()

    print("\n--- Starting Real-Time Inference Loop ---")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            # a. Read a window of data
            raw_data = read_data_from_sensor(WINDOW_SIZE)
            
            # b. Pre-process the data
            normalized_data = preprocess_data(raw_data, norm_mean, norm_std)
            
            # c. Get prediction from the model
            # The input needs to be a tensor with batch and channel dimensions: [1, 1, 1024]
            input_tensor = torch.tensor(normalized_data).unsqueeze(0).unsqueeze(0)
            
            # learn.predict returns a tuple, the reconstructed tensor is the first element
            reconstructed_tensor, _, _ = learn.predict(input_tensor)
            
            # d. Calculate reconstruction loss (MSE)
            loss = calculate_loss(input_tensor.squeeze(0), reconstructed_tensor)

            # e. Decision Logic
            if loss > threshold:
                print(f"🔴 ANOMALY DETECTED! Loss: {loss:.6f} > Threshold: {threshold:.6f}")
                print("   YARN BREAK DETECTED! Triggering relay.")
                trigger_relay(True)
                time.sleep(5) # Wait for 5 seconds before resetting
                trigger_relay(False)
                print("   Relay reset. Resuming monitoring.")
            else:
                print(f"🟢 Normal Operation. Loss: {loss:.6f}")

    except KeyboardInterrupt:
        print("\n--- Shutting down ---")
    finally:
        if IS_PI:
            GPIO.cleanup()
            print("GPIO cleaned up.")
