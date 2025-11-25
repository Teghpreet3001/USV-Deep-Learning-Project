# compare_models.py
# Compare MLP vs LSTM model outputs and performance
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from models.mlp import ResidualMLP
from models.lstm import ResidualLSTM
from models.physics import Fossen3DOF

# Load data
csv_path = "~/Downloads/processed_new.csv"
df = pd.read_csv(csv_path, parse_dates=["time"])

inputs_cols = ["u_filt", "v_filt", "r_filt", "cmd_thrust.port", "cmd_thrust.starboard"]
target_cols = ["du_dt", "dv_dt", "dr_dt"]

inputs = torch.tensor(df[inputs_cols].values, dtype=torch.float32)
measured_accel = torch.tensor(df[target_cols].values, dtype=torch.float32)

print("=" * 80)
print("MODEL COMPARISON: MLP vs LSTM")
print("=" * 80)

# Hyperparameters
in_dim = 5
out_dim = 3
hidden_dim = 128
seq_len = 30  # Must match the sequence length used for LSTM training

# Load models
print("\n--- Loading Models ---")
mlp_model = ResidualMLP(in_dim=in_dim, hidden=hidden_dim, out_dim=out_dim)
lstm_model = ResidualLSTM(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=2, 
                          out_dim=out_dim, dropout=0.1)

try:
    mlp_model.load_state_dict(torch.load("residual_mlp.pth"))
    print("✓ MLP model loaded")
except FileNotFoundError:
    print("✗ MLP model not found. Please train MLP first (run train/train.py)")

try:
    lstm_model.load_state_dict(torch.load("residual_lstm.pth"))
    print("✓ LSTM model loaded")
except FileNotFoundError:
    print("✗ LSTM model not found. Please train LSTM first (run train/train_lstm.py)")

fossen_model = Fossen3DOF()

# -------------------- MLP EVALUATION --------------------
print("\n--- Evaluating MLP ---")
mlp_model.eval()
with torch.no_grad():
    u, v, r, tL, tR = inputs.T
    physics_accel_mlp = fossen_model.forward(u, v, r, tL, tR)
    mlp_residual_pred = mlp_model(inputs)
    mlp_hybrid_accel = physics_accel_mlp + mlp_residual_pred

# -------------------- LSTM EVALUATION --------------------
print("--- Evaluating LSTM ---")
lstm_model.eval()

# Create sequence dataset for LSTM
class SequenceDataset(Dataset):
    def __init__(self, inputs, targets, seq_len):
        self.inputs = inputs
        self.targets = targets
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.inputs) - self.seq_len + 1
    
    def __getitem__(self, idx):
        x_seq = self.inputs[idx:idx+self.seq_len]
        y = self.targets[idx+self.seq_len-1]
        return x_seq, y

lstm_dataset = SequenceDataset(inputs, measured_accel, seq_len)
lstm_loader = DataLoader(lstm_dataset, batch_size=64, shuffle=False)

with torch.no_grad():
    all_lstm_residuals = []
    all_lstm_physics = []
    all_lstm_measured = []
    
    for x_seq_batch, measured_batch in lstm_loader:
        u, v, r, tL, tR = x_seq_batch[:, -1, :].T
        physics_accel_batch = fossen_model.forward(u, v, r, tL, tR)
        lstm_residual_pred = lstm_model(x_seq_batch)
        
        all_lstm_residuals.append(lstm_residual_pred)
        all_lstm_physics.append(physics_accel_batch)
        all_lstm_measured.append(measured_batch)
    
    lstm_residual_pred = torch.cat(all_lstm_residuals, dim=0)
    lstm_physics_accel = torch.cat(all_lstm_physics, dim=0)
    lstm_measured_accel = torch.cat(all_lstm_measured, dim=0)
    lstm_hybrid_accel = lstm_physics_accel + lstm_residual_pred

# Pad LSTM results for comparison (since we lose seq_len-1 samples)
pad_size = seq_len - 1
lstm_residual_padded = torch.cat([torch.zeros(pad_size, out_dim), lstm_residual_pred], dim=0)
lstm_hybrid_padded = torch.cat([torch.zeros(pad_size, out_dim), lstm_hybrid_accel], dim=0)
lstm_physics_padded = torch.cat([torch.zeros(pad_size, out_dim), lstm_physics_accel], dim=0)

# For fair comparison, use only the samples where both models have predictions
# (i.e., skip first pad_size samples for MLP)
mlp_residual_eval = mlp_residual_pred[pad_size:]
mlp_hybrid_eval = mlp_hybrid_accel[pad_size:]
mlp_physics_eval = physics_accel_mlp[pad_size:]
measured_accel_eval = measured_accel[pad_size:]

# -------------------- METRICS COMPARISON --------------------
print("\n" + "=" * 80)
print("QUANTITATIVE COMPARISON (on overlapping samples)")
print("=" * 80)

acc_labels = ["u_dot (surge)", "v_dot (sway)", "r_dot (yaw rate)"]

# MLP Metrics
mlp_mse_hybrid = torch.mean((mlp_hybrid_eval - measured_accel_eval)**2, dim=0)
mlp_mae_hybrid = torch.mean(torch.abs(mlp_hybrid_eval - measured_accel_eval), dim=0)
mlp_mse_physics = torch.mean((mlp_physics_eval - measured_accel_eval)**2, dim=0)
mlp_mae_physics = torch.mean(torch.abs(mlp_physics_eval - measured_accel_eval), dim=0)

# LSTM Metrics
lstm_mse_hybrid = torch.mean((lstm_hybrid_accel - lstm_measured_accel)**2, dim=0)
lstm_mae_hybrid = torch.mean(torch.abs(lstm_hybrid_accel - lstm_measured_accel), dim=0)
lstm_mse_physics = torch.mean((lstm_physics_accel - lstm_measured_accel)**2, dim=0)
lstm_mae_physics = torch.mean(torch.abs(lstm_physics_accel - lstm_measured_accel), dim=0)

print("\nMLP Results:")
for i in range(3):
    print(f"  {acc_labels[i]}:")
    print(f"    Hybrid - MSE: {mlp_mse_hybrid[i]:.6f}, MAE: {mlp_mae_hybrid[i]:.6f}")
    print(f"    Physics - MSE: {mlp_mse_physics[i]:.6f}, MAE: {mlp_mae_physics[i]:.6f}")

print("\nLSTM Results:")
for i in range(3):
    print(f"  {acc_labels[i]}:")
    print(f"    Hybrid - MSE: {lstm_mse_hybrid[i]:.6f}, MAE: {lstm_mae_hybrid[i]:.6f}")
    print(f"    Physics - MSE: {lstm_mse_physics[i]:.6f}, MAE: {lstm_mae_physics[i]:.6f}")

print("\nImprovement (LSTM vs MLP):")
for i in range(3):
    mse_improvement = ((mlp_mse_hybrid[i] - lstm_mse_hybrid[i]) / mlp_mse_hybrid[i] * 100).item()
    mae_improvement = ((mlp_mae_hybrid[i] - lstm_mae_hybrid[i]) / mlp_mae_hybrid[i] * 100).item()
    better = "✓ LSTM better" if mse_improvement > 0 else "✗ MLP better"
    print(f"  {acc_labels[i]}: MSE {mse_improvement:+.2f}%, MAE {mae_improvement:+.2f}% {better}")

# -------------------- RESIDUAL COMPARISON --------------------
print("\n" + "=" * 80)
print("RESIDUAL PREDICTION COMPARISON")
print("=" * 80)

# Compare how similar the residual predictions are
residual_diff = torch.abs(mlp_residual_eval - lstm_residual_pred)
print("\nMean absolute difference between MLP and LSTM residual predictions:")
for i, label in enumerate(acc_labels):
    mean_diff = torch.mean(residual_diff[:, i]).item()
    max_diff = torch.max(residual_diff[:, i]).item()
    print(f"  {label}: Mean={mean_diff:.6f}, Max={max_diff:.6f}")

# Correlation between MLP and LSTM residuals
print("\nCorrelation between MLP and LSTM residual predictions:")
for i, label in enumerate(acc_labels):
    mlp_res = mlp_residual_eval[:, i].numpy()
    lstm_res = lstm_residual_pred[:, i].numpy()
    corr = np.corrcoef(mlp_res, lstm_res)[0, 1]
    print(f"  {label}: {corr:.4f} (1.0 = identical, 0.0 = uncorrelated)")

# -------------------- PLOTTING --------------------
print("\n--- Generating Comparison Plots ---")

# Plot 1: Residual predictions comparison
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for i in range(3):
    axs[i].plot(mlp_residual_eval[:, i].numpy(), label=f"MLP Residual", 
                alpha=0.7, linewidth=1.5)
    axs[i].plot(lstm_residual_pred[:, i].numpy(), label=f"LSTM Residual", 
                alpha=0.7, linewidth=1.5)
    axs[i].set_ylabel(f"{acc_labels[i]}\nResidual [m/s²]")
    axs[i].legend()
    axs[i].grid(True, alpha=0.3)
axs[2].set_xlabel("Sample index")
plt.suptitle("Residual Predictions: MLP vs LSTM", fontsize=14)
plt.tight_layout()
plt.savefig("residual_comparison.png", dpi=150)
print("✓ Saved residual_comparison.png")

# Plot 2: Hybrid predictions vs measured
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for i in range(3):
    axs[i].plot(measured_accel_eval[:, i].numpy(), label="Measured", 
                color="black", linewidth=2, alpha=0.8)
    axs[i].plot(mlp_hybrid_eval[:, i].numpy(), label="MLP Hybrid", 
                color="blue", linestyle="--", alpha=0.7)
    axs[i].plot(lstm_hybrid_accel[:, i].numpy(), label="LSTM Hybrid", 
                color="red", linestyle="-.", alpha=0.7)
    axs[i].set_ylabel(f"{acc_labels[i]}\nAcceleration [m/s²]")
    axs[i].legend()
    axs[i].grid(True, alpha=0.3)
axs[2].set_xlabel("Sample index")
plt.suptitle("Hybrid Predictions vs Measured: MLP vs LSTM", fontsize=14)
plt.tight_layout()
plt.savefig("hybrid_comparison.png", dpi=150)
print("✓ Saved hybrid_comparison.png")

# Plot 3: Error comparison
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for i in range(3):
    mlp_error = torch.abs(mlp_hybrid_eval[:, i] - measured_accel_eval[:, i]).numpy()
    lstm_error = torch.abs(lstm_hybrid_accel[:, i] - lstm_measured_accel[:, i]).numpy()
    
    axs[i].plot(mlp_error, label=f"MLP Error (MAE={mlp_mae_hybrid[i]:.6f})", 
                color="blue", alpha=0.7)
    axs[i].plot(lstm_error, label=f"LSTM Error (MAE={lstm_mae_hybrid[i]:.6f})", 
                color="red", alpha=0.7)
    axs[i].set_ylabel(f"{acc_labels[i]}\nAbsolute Error [m/s²]")
    axs[i].legend()
    axs[i].grid(True, alpha=0.3)
axs[2].set_xlabel("Sample index")
plt.suptitle("Prediction Error Comparison: MLP vs LSTM", fontsize=14)
plt.tight_layout()
plt.savefig("error_comparison.png", dpi=150)
print("✓ Saved error_comparison.png")

print("\n" + "=" * 80)
print("Comparison complete! Check the generated plots.")
print("=" * 80)

