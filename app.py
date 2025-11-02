# app.py
# Author: Senior AI Engineer and IoT Specialist
# Description: Creates a simple web UI using Gradio to demonstrate the yarn
#              break detection model. Users can upload a CSV file of sensor
#              data to test the model's performance.

import gradio as gr
import pandas as pd
import numpy as np
import torch
from fastai.learner import load_learner
import json
import os

# --- Configuration ---
MODEL_PATH = 'yarn_break_model.pkl'
THRESHOLD_PATH = 'anomaly_threshold.json'
WINDOW_SIZE = 1024
STEP_SIZE = 256 # Use a larger step for faster processing in the demo

# 1. Load the trained model and anomaly threshold ONCE at startup
print("Loading model and configuration...")
try:
    # Load the model onto the CPU, which is sufficient for inference
    learn = load_learner(MODEL_PATH, cpu=True)
    with open(THRESHOLD_PATH, 'r') as f:
        config = json.load(f)
    THRESHOLD = config['threshold']
    NORM_MEAN = config['normalization_mean']
    NORM_STD = config['normalization_std']
    print("Model and config loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Make sure '{MODEL_PATH}' and '{THRESHOLD_PATH}' are in the same directory.")
    print("Run train_model.py to generate these files.")
    learn = None
    THRESHOLD = float('inf')

def create_windows(data, window_size, step_size):
    """Slices time-series data into overlapping windows."""
    windows = []
    for i in range(0, len(data) - window_size, step_size):
        windows.append(data[i:i + window_size])
    return np.array(windows, dtype=np.float32)

def predict_yarn_status(data_file):
    """
    Backend prediction function for the Gradio interface.
    
    Args:
        data_file (File object): The uploaded CSV file from the Gradio interface.

    Returns:
        tuple: A tuple containing the status string and the max reconstruction error.
    """
    if not learn or not data_file:
        return "🔴 MODEL NOT LOADED", "N/A"

    try:
        # Load and process the uploaded CSV
        df = pd.read_csv(data_file.name)
        if 'sensor_value' not in df.columns:
            return "🔴 ERROR: CSV must have a 'sensor_value' column.", "N/A"
            
        data = df['sensor_value'].values
        
        # Normalize the data using the same stats from training
        normalized_data = (data - NORM_MEAN) / NORM_STD
        
        # Create windows from the data
        windows = create_windows(normalized_data, WINDOW_SIZE, STEP_SIZE)
        if len(windows) == 0:
            return "🔴 ERROR: Data file is too short to create a window.", "N/A"
            
        max_loss = 0
        is_anomaly_detected = False

        # Calculate reconstruction loss for each window
        for window in windows:
            # CORRECTED: Pass the 1D NumPy array (the item) directly to predict.
            # FastAI handles the transformations and batching.
            reconstructed_tensor, _, _ = learn.predict(window)
            
            # For calculating loss, the original input window needs to be a tensor
            input_tensor_for_loss = torch.tensor(window).unsqueeze(0)
            loss = torch.nn.functional.mse_loss(input_tensor_for_loss, reconstructed_tensor).item()
            
            if loss > max_loss:
                max_loss = loss
            
            if loss > THRESHOLD:
                is_anomaly_detected = True
        
        # Determine the final status
        if is_anomaly_detected:
            status = "🔴 YARN BREAK DETECTED"
        else:
            status = "🟢 Normal Operation"
            
        return status, f"{max_loss:.6f}"

    except Exception as e:
        return f"🔴 ERROR: {str(e)}", "N/A"


# Create Gradio Interface
print("Launching Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft(), title="Yarn Break Detection System") as demo:
    gr.Markdown(
        """
        # 🤖 Smart Yarn Break Detection Demo
        Upload a CSV file containing sensor vibration data to check for anomalies.
        The CSV file must have a single column named `sensor_value`.
        You can use the `normal_data.csv` or `break_data.csv` files from the `data/` directory.
        """
    )
    # CORRECTED: Prepare the threshold text outside of the f-string to avoid ValueError
    threshold_text = f"{THRESHOLD:.6f}" if learn else "N/A"
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Sensor Data CSV", file_types=[".csv"])
            submit_btn = gr.Button("Analyze Data", variant="primary")
        with gr.Column(scale=2):
            status_output = gr.Label(label="System Status")
            error_output = gr.Textbox(label="Max Reconstruction Error (Loss)", interactive=False)
            gr.Markdown(f"**Current Anomaly Threshold:** `{threshold_text}`")


    submit_btn.click(
        fn=predict_yarn_status,
        inputs=file_input,
        outputs=[status_output, error_output]
    )

if __name__ == "__main__":
    demo.launch()

