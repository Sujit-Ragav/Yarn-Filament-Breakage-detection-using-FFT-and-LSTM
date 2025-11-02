# train_model.py
# Author: Senior AI Engineer and IoT Specialist
# Description: Trains a 1D Convolutional Autoencoder using FastAI to learn
#              the patterns of normal yarn vibration. The model is trained
#              exclusively on normal data to detect anomalies (breaks) based
#              on reconstruction error.

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastai.tabular.all import *
import os
import json

# --- Configuration ---
DATA_FILE_PATH = os.path.join('data', 'normal_data.csv')
WINDOW_SIZE = 1024  # Size of each data window fed to the model
STEP_SIZE = 128     # Overlap between consecutive windows
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
MODEL_EXPORT_NAME = 'yarn_break_model.pkl'
THRESHOLD_FILE_NAME = 'anomaly_threshold.json'

def create_windows(data, window_size, step_size):
    """
    Slices a long time-series signal into smaller, overlapping windows.

    Args:
        data (np.ndarray): The input time-series data.
        window_size (int): The number of samples in each window.
        step_size (int): The number of samples to slide the window forward.

    Returns:
        np.ndarray: A 2D array where each row is a window of the time-series.
    """
    windows = []
    for i in range(0, len(data) - window_size, step_size):
        windows.append(data[i:i + window_size])
    return np.array(windows, dtype=np.float32)

# Custom Item Transform to add a channel dimension for Conv1d
class AddChannel(Transform):
    def encodes(self, x): return torch.as_tensor(x).unsqueeze(0)
    def decodes(self, x): return x.squeeze(0)

# 1D Convolutional Autoencoder Model Definition
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder: Compresses the input
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=2, padding=31), # Output: 16 x 512
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2), # Output: 16 x 256
            nn.Conv1d(16, 8, kernel_size=32, stride=2, padding=15), # Output: 8 x 128
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2), # Output: 8 x 64 (Latent Vector)
        )
        # Decoder: Reconstructs the input from the compressed representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(8, 8, kernel_size=32, stride=4, padding=14), # Output: 8 x 256
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, kernel_size=64, stride=2, padding=31), # Output: 16 x 512
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=128, stride=2, padding=63), # Output: 1 x 1024
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    print("Starting model training process...")

    # 1. Load and Preprocess Data
    print(f"Loading data from {DATA_FILE_PATH}...")
    df = pd.read_csv(DATA_FILE_PATH)
    data = df['sensor_value'].values

    # Normalize data (important for neural networks)
    mean, std = data.mean(), data.std()
    data_normalized = (data - mean) / std
    
    # Save normalization stats for inference
    norm_stats = {'mean': float(mean), 'std': float(std)}
    
    print("Creating time-series windows...")
    windows = create_windows(data_normalized, WINDOW_SIZE, STEP_SIZE)
    # Note: We keep `windows` as a NumPy array and let the transform handle conversion
    
    # 2. Create FastAI DataLoaders
    # The input and target are the same for an autoencoder
    # Create a random 80/20 train/validation split to be able to calculate validation loss
    splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(windows))

    dsets = Datasets(windows, tfms=[[AddChannel], [AddChannel]], splits=splits)
    dls = dsets.dataloaders(bs=BATCH_SIZE, after_batch=ToTensor(), shuffle=True)
    
    print(f"DataLoaders created. Number of windows: {len(windows)}")
    print(f"  - Training windows: {len(dls.train_ds)}")
    print(f"  - Validation windows: {len(dls.valid_ds)}")


    # 3. Model, Loss Function, and Learner
    model = ConvAutoencoder()
    loss_func = nn.MSELoss()
    learn = Learner(dls, model, loss_func=loss_func)

    # 4. Train the Model
    print("Finding optimal learning rate...")
    #lr_find_res = learn.lr_find()
    #print(f"Suggested learning rate: {lr_find_res.valley}")
    
    print(f"Training for {EPOCHS} epochs with learning rate {LEARNING_RATE}...")
    learn.fit_one_cycle(EPOCHS, LEARNING_RATE)
    learn.recorder.plot_loss()

    # 5. Calculate Anomaly Threshold
    print("Calculating anomaly detection threshold...")
    # Get predictions (reconstructions) for the validation set
    reconstructions, targets = learn.get_preds()

    # Calculate reconstruction loss for each sample
    losses = [loss_func(r, t).item() for r, t in zip(reconstructions, targets)]
    losses = np.array(losses)

    # Calculate threshold: mean + 3 * std_dev
    mean_loss = np.mean(losses)
    std_loss = np.std(losses)
    threshold = mean_loss + 3 * std_loss

    print(f"  - Mean Reconstruction Loss: {mean_loss:.6f}")
    print(f"  - Std Dev of Loss: {std_loss:.6f}")
    print(f"  - Calculated Anomaly Threshold: {threshold:.6f}")
    
    # 6. Export Model and Threshold
    print(f"Exporting model to {MODEL_EXPORT_NAME}...")
    learn.export(MODEL_EXPORT_NAME)
    
    threshold_data = {
        'threshold': threshold,
        'normalization_mean': float(mean),
        'normalization_std': float(std)
    }
    
    with open(THRESHOLD_FILE_NAME, 'w') as f:
        json.dump(threshold_data, f)
        
    print(f"Threshold saved to {THRESHOLD_FILE_NAME}.")
    print("\nTraining complete. Model and threshold are ready for inference.")

