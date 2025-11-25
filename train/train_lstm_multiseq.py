# train_lstm_multiseq.py
# Train LSTM models with multiple sequence lengths and compare results
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from models.lstm import ResidualLSTM
from models.physics import Fossen3DOF

# Hyperparameters
in_dim = 5      # [u, v, r, thrust_L, thrust_R]
out_dim = 3     # [u_dot, v_dot, r_dot]
hidden_dim = 128
num_layers = 2
lr = 1e-3
epochs = 100
batch_size = 64
dropout = 0.1

# Sequence lengths to test
# At 10 Hz sampling: 10=1s, 20=2s, 30=3s, 50=5s
sequence_lengths = [10, 20, 30, 50]

# Data Loading
csv_path = "~/Downloads/processed_new.csv"
df = pd.read_csv(csv_path, parse_dates=["time"])
print(df.columns)

inputs_cols = ["u_filt", "v_filt", "r_filt", "cmd_thrust.port", "cmd_thrust.starboard"]
target_cols = ["du_dt", "dv_dt", "dr_dt"]

print(df[inputs_cols + target_cols].isna().sum())
print(df[inputs_cols + target_cols].describe())

inputs = torch.tensor(df[inputs_cols].values, dtype=torch.float32)
measured_accel = torch.tensor(df[target_cols].values, dtype=torch.float32)

print("Inputs shape:", inputs.shape)            # (N, 5)
print("Measured accel shape:", measured_accel.shape)  # (N, 3)

# -------------------- AUTOCORRELATION ANALYSIS --------------------
print("\n--- Analyzing Residual Temporal Structure ---")
fossen_model = Fossen3DOF()
u, v, r, tL, tR = inputs.T
with torch.no_grad():
    physics_accel = fossen_model.forward(u, v, r, tL, tR)
    residuals = measured_accel - physics_accel

residuals_np = residuals.numpy()
acc_labels = ["u_dot (surge)", "v_dot (sway)", "r_dot (yaw rate)"]

print("\nResidual Statistics:")
for i, label in enumerate(acc_labels):
    residual_mag = np.mean(np.abs(residuals_np[:, i]))
    measured_mag = np.mean(np.abs(measured_accel[:, i].numpy()))
    ratio = residual_mag / measured_mag if measured_mag > 0 else 0
    print(f"  {label}: Mean |residual|={residual_mag:.6f}, Mean |measured|={measured_mag:.6f}, Ratio={ratio:.4f}")

print("\nAutocorrelation Analysis (lag 1-5):")
print("  (High autocorrelation > 0.3 suggests temporal dependencies)")
for i, label in enumerate(acc_labels):
    residual_series = residuals_np[:, i]
    autocorrs = []
    for lag in range(1, 6):
        if len(residual_series) > lag:
            corr = np.corrcoef(residual_series[:-lag], residual_series[lag:])[0, 1]
            autocorrs.append(corr)
        else:
            autocorrs.append(0.0)
    
    max_autocorr = max(autocorrs) if autocorrs else 0.0
    temporal_indicator = "✓ TEMPORAL" if max_autocorr > 0.3 else "✗ State-dependent"
    print(f"  {label}:")
    print(f"    Lag 1-5 autocorr: {[f'{c:.4f}' for c in autocorrs]}")
    print(f"    Max autocorr: {max_autocorr:.4f} {temporal_indicator}")

# Create sequences for LSTM
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

# Store results for each sequence length
results = {}

print("\n" + "=" * 80)
print("TRAINING LSTM MODELS WITH DIFFERENT SEQUENCE LENGTHS")
print("=" * 80)

for seq_len in sequence_lengths:
    print(f"\n{'='*80}")
    print(f"Training LSTM with sequence length: {seq_len} ({seq_len/10:.1f}s at 10 Hz)")
    print(f"{'='*80}")
    
    dataset = SequenceDataset(inputs, measured_accel, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(loader)}")
    
    # Model + Loss + Optimizer
    model = ResidualLSTM(
        in_dim=in_dim, 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        out_dim=out_dim,
        dropout=dropout
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_seq_batch, measured_accel_batch in loader:
            u, v, r, tL, tR = x_seq_batch[:, -1, :].T
            
            # Physics-based prediction
            model_accel = fossen_model.forward(u, v, r, tL, tR)
            
            # Compute residual target
            residual_target = measured_accel_batch - model_accel
            
            # Neural network predicts residual
            residual_pred = model(x_seq_batch)
            
            # Compute loss
            loss = criterion(residual_pred, residual_target)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")
    
    # Save model
    model_path = f"residual_lstm_seq{seq_len}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved as {model_path}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        eval_dataset = SequenceDataset(inputs, measured_accel, seq_len)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        all_residual_preds = []
        all_measured_accels = []
        all_model_accels = []
        
        for x_seq_batch, measured_accel_batch in eval_loader:
            u, v, r, tL, tR = x_seq_batch[:, -1, :].T
            model_accel = fossen_model.forward(u, v, r, tL, tR)
            residual_pred = model(x_seq_batch)
            
            all_residual_preds.append(residual_pred)
            all_measured_accels.append(measured_accel_batch)
            all_model_accels.append(model_accel)
        
        residual_pred = torch.cat(all_residual_preds, dim=0)
        measured_accel_eval = torch.cat(all_measured_accels, dim=0)
        model_accel = torch.cat(all_model_accels, dim=0)
        hybrid_accel = model_accel + residual_pred
    
    # Compute metrics
    mse_hybrid = torch.mean((hybrid_accel - measured_accel_eval)**2, dim=0)
    mae_hybrid = torch.mean(torch.abs(hybrid_accel - measured_accel_eval), dim=0)
    mse_physics = torch.mean((model_accel - measured_accel_eval)**2, dim=0)
    mae_physics = torch.mean(torch.abs(model_accel - measured_accel_eval), dim=0)
    
    # Store results
    results[seq_len] = {
        'loss_history': loss_history,
        'mse_hybrid': mse_hybrid,
        'mae_hybrid': mae_hybrid,
        'mse_physics': mse_physics,
        'mae_physics': mae_physics,
        'final_loss': loss_history[-1],
        'residual_pred': residual_pred,
        'hybrid_accel': hybrid_accel,
        'measured_accel': measured_accel_eval,
        'model_accel': model_accel,
        'pad_size': seq_len - 1
    }
    
    print(f"\nEvaluation Metrics (seq_len={seq_len}):")
    for i in range(3):
        print(f"  {acc_labels[i]}: MSE hybrid={mse_hybrid[i]:.6f}, MAE hybrid={mae_hybrid[i]:.6f} | "
              f"MSE physics={mse_physics[i]:.6f}, MAE physics={mae_physics[i]:.6f}")

# -------------------- COMPARISON ANALYSIS --------------------
print("\n" + "=" * 80)
print("SEQUENCE LENGTH COMPARISON")
print("=" * 80)

# Find best sequence length for each metric
print("\nBest Sequence Length by Metric:")
for i, label in enumerate(acc_labels):
    best_mse_seq = min(sequence_lengths, key=lambda s: results[s]['mse_hybrid'][i].item())
    best_mae_seq = min(sequence_lengths, key=lambda s: results[s]['mae_hybrid'][i].item())
    best_mse_val = results[best_mse_seq]['mse_hybrid'][i].item()
    best_mae_val = results[best_mae_seq]['mae_hybrid'][i].item()
    
    print(f"  {label}:")
    print(f"    Best MSE: seq_len={best_mse_seq} (MSE={best_mse_val:.6f})")
    print(f"    Best MAE: seq_len={best_mae_seq} (MAE={best_mae_val:.6f})")

# Overall best (average across all axes)
avg_mse = {s: torch.mean(results[s]['mse_hybrid']).item() for s in sequence_lengths}
avg_mae = {s: torch.mean(results[s]['mae_hybrid']).item() for s in sequence_lengths}
best_overall_mse = min(sequence_lengths, key=lambda s: avg_mse[s])
best_overall_mae = min(sequence_lengths, key=lambda s: avg_mae[s])

print(f"\nOverall Best (averaged across all axes):")
print(f"  Best MSE: seq_len={best_overall_mse} (avg MSE={avg_mse[best_overall_mse]:.6f})")
print(f"  Best MAE: seq_len={best_overall_mae} (avg MAE={avg_mae[best_overall_mae]:.6f})")

# Comparison table
print("\n" + "-" * 80)
print("Detailed Comparison Table:")
print("-" * 80)
print(f"{'Seq Len':<10} {'Time (s)':<12} {'Final Loss':<15} {'Avg MSE':<15} {'Avg MAE':<15}")
print("-" * 80)
for seq_len in sequence_lengths:
    time_sec = seq_len / 10.0
    final_loss = results[seq_len]['final_loss']
    avg_mse_val = avg_mse[seq_len]
    avg_mae_val = avg_mae[seq_len]
    print(f"{seq_len:<10} {time_sec:<12.1f} {final_loss:<15.6f} {avg_mse_val:<15.6f} {avg_mae_val:<15.6f}")

# -------------------- PLOTTING --------------------
print("\n--- Generating Comparison Plots ---")

# Plot 1: Training loss curves
plt.figure(figsize=(12, 6))
for seq_len in sequence_lengths:
    plt.plot(results[seq_len]['loss_history'], label=f'seq_len={seq_len} ({seq_len/10:.1f}s)', linewidth=2)
plt.title("LSTM Training Loss: Sequence Length Comparison", fontsize=14)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("lstm_multiseq_training_loss.png", dpi=150)
print("✓ Saved lstm_multiseq_training_loss.png")

# Plot 2: MSE comparison by sequence length
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, label in enumerate(acc_labels):
    mse_values = [results[s]['mse_hybrid'][i].item() for s in sequence_lengths]
    axs[i].plot(sequence_lengths, mse_values, marker='o', linewidth=2, markersize=8)
    axs[i].set_title(f"{label}")
    axs[i].set_xlabel("Sequence Length")
    axs[i].set_ylabel("MSE")
    axs[i].grid(True, alpha=0.3)
    axs[i].set_xticks(sequence_lengths)
plt.suptitle("MSE by Sequence Length", fontsize=14)
plt.tight_layout()
plt.savefig("lstm_multiseq_mse_comparison.png", dpi=150)
print("✓ Saved lstm_multiseq_mse_comparison.png")

# Plot 3: MAE comparison by sequence length
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, label in enumerate(acc_labels):
    mae_values = [results[s]['mae_hybrid'][i].item() for s in sequence_lengths]
    axs[i].plot(sequence_lengths, mae_values, marker='o', linewidth=2, markersize=8, color='orange')
    axs[i].set_title(f"{label}")
    axs[i].set_xlabel("Sequence Length")
    axs[i].set_ylabel("MAE")
    axs[i].grid(True, alpha=0.3)
    axs[i].set_xticks(sequence_lengths)
plt.suptitle("MAE by Sequence Length", fontsize=14)
plt.tight_layout()
plt.savefig("lstm_multiseq_mae_comparison.png", dpi=150)
print("✓ Saved lstm_multiseq_mae_comparison.png")

# Plot 4: Predictions comparison (use longest sequence for reference)
ref_seq_len = max(sequence_lengths)
ref_results = results[ref_seq_len]
ref_pad = ref_results['pad_size']

# Align all predictions to same length (use shortest for comparison)
min_samples = min(len(results[s]['hybrid_accel']) for s in sequence_lengths)

fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for i in range(3):
    # Plot measured (reference)
    measured_ref = results[ref_seq_len]['measured_accel'][:min_samples, i].numpy()
    axs[i].plot(measured_ref, label="Measured", color="black", linewidth=2, alpha=0.8)
    
    # Plot each sequence length
    colors = plt.cm.viridis(np.linspace(0, 1, len(sequence_lengths)))
    for idx, seq_len in enumerate(sequence_lengths):
        hybrid = results[seq_len]['hybrid_accel'][:min_samples, i].numpy()
        axs[i].plot(hybrid, label=f'seq_len={seq_len}', 
                   color=colors[idx], linestyle='--', alpha=0.7, linewidth=1.5)
    
    axs[i].set_ylabel(f"{acc_labels[i]}\nAcceleration [m/s²]")
    axs[i].legend(ncol=3, fontsize=8)
    axs[i].grid(True, alpha=0.3)
axs[2].set_xlabel("Sample index")
plt.suptitle("Hybrid Predictions: Sequence Length Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("lstm_multiseq_predictions.png", dpi=150)
print("✓ Saved lstm_multiseq_predictions.png")

print("\n" + "=" * 80)
print("Multi-sequence length training complete!")
print(f"Best overall sequence length: {best_overall_mse} (by MSE)")
print("=" * 80)

