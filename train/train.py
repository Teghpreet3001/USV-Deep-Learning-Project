# train.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
# Import models
from models.mlp import ResidualMLP
from models.physics import Fossen3DOF  # physics-based model

# Hyperparameters
in_dim = 5      # [u, v, r, thrust_L, thrust_R]
out_dim = 3     # [u_dot, v_dot, r_dot]
hidden = 128
lr = 1e-3
epochs = 100
batch_size = 64

# Replace with Data Loading
# u = torch.randn(1000, 1)
# v = torch.randn(1000, 1)
# r = torch.randn(1000, 1)
# thrust_L = torch.randn(1000, 1)
# thrust_R = torch.randn(1000, 1)
# measured_accel = torch.randn(1000, 3)


csv_path = "~/Downloads/processed_new.csv"
df = pd.read_csv(csv_path,parse_dates=["time"])
print(df.columns)


inputs_cols = ["u_filt", "v_filt", "r_filt", "cmd_thrust.port", "cmd_thrust.starboard"]
target_cols = ["du_dt", "dv_dt", "dr_dt"]  # measured accelerations; SHOULD WE USE IMU OR CALCULATED ACCELS?

print(df[inputs_cols + target_cols].isna().sum())
print(df[inputs_cols + target_cols].describe())

inputs = torch.tensor(df[inputs_cols].values, dtype=torch.float32)
measured_accel = torch.tensor(df[target_cols].values, dtype=torch.float32)


print("Inputs shape:", inputs.shape)            # (N, 5)
print("Measured accel shape:", measured_accel.shape)  # (N, 3)

# -------------------- AUTOCORRELATION ANALYSIS --------------------
# Compute residuals from physics model to check if they're temporal
print("\n--- Analyzing Residual Temporal Structure ---")
fossen_model = Fossen3DOF()
u, v, r, tL, tR = inputs.T
with torch.no_grad():
    physics_accel = fossen_model.forward(u, v, r, tL, tR)
    residuals = measured_accel - physics_accel

# Convert to numpy for autocorrelation analysis
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

dataset = TensorDataset(inputs, measured_accel)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model + Loss + Optimizer
model = ResidualMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# fossen_model already created above for autocorrelation analysis

# --------------------
# print("\n🔍 Running dry run (single batch check)...")

# x_batch, measured_accel_batch = next(iter(loader))
# print("x_batch:", x_batch.shape)
# print("measured_accel_batch:", measured_accel_batch.shape)

# # Unpack
# u, v, r, tL, tR = x_batch.T
# print(tR)
# # Physics-based prediction
# model_accel = fossen_model.forward(u, v, r, tL, tR)
# print("model_accel:", model_accel.shape)

# # Residual target
# residual_target = measured_accel_batch - model_accel
# print("residual_target:", residual_target.shape)

# # NN prediction
# residual_pred = model(x_batch)
# print("residual_pred:", residual_pred.shape)

# # Loss
# loss = criterion(residual_pred, residual_target)
# print("Initial loss:", loss.item())

# print("✅ Dry run passed — proceeding to training.\n")

loss_history= []

# Training Loop
for epoch in range(epochs):
    total_loss = 0
    for x_batch, measured_accel_batch in loader:
        u, v, r, tL, tR = x_batch.T
        # Step 1: Physics-based prediction
        model_accel = fossen_model.forward(u,v,r,tL,tR)

        # Step 2: Compute residual target
        residual_target = measured_accel_batch - model_accel

        # Step 3: Neural network predicts residual
        residual_pred = model(x_batch)

        # Step 4: Compute loss
        loss = criterion(residual_pred, residual_target)

        # Step 5: Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.6f}")

# Plot training loss curve
plt.figure()
plt.plot(loss_history)
plt.title("Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Save the trained model
torch.save(model.state_dict(), "residual_mlp.pth")
print("✅ Model saved as residual_mlp.pth")

# -------------------- EVALUATION --------------------
model.eval()
with torch.no_grad():
    u, v, r, tL, tR = inputs.T
    model_accel = fossen_model.forward(u, v, r, tL, tR)
    residual_pred = model(inputs)
    hybrid_accel = model_accel + residual_pred

# -------------------- QUANTITATIVE EVALUATION --------------------
# Compute per-axis MSE and MAE
mse_hybrid = torch.mean((hybrid_accel - measured_accel)**2, dim=0)
mae_hybrid = torch.mean(torch.abs(hybrid_accel - measured_accel), dim=0)

mse_model = torch.mean((model_accel - measured_accel)**2, dim=0)
mae_model = torch.mean(torch.abs(model_accel - measured_accel), dim=0)

acc_labels = ["u_dot (surge)", "v_dot (sway)", "r_dot (yaw rate)"]
for i in range(3):
    print(f"{acc_labels[i]}: MSE hybrid={mse_hybrid[i]:.6f}, MAE hybrid={mae_hybrid[i]:.6f} | "
          f"MSE physics={mse_model[i]:.6f}, MAE physics={mae_model[i]:.6f}")

# -------------------- PLOTTING: Model, Residual, Hybrid, Measured --------------------
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

acc_labels = ["u_dot (surge)", "v_dot (sway)", "r_dot (yaw rate)"]
colors = ["blue", "green", "red", "black"]
linestyles = ["--", ":", "-.", "-"]  # model, residual, hybrid, measured

for i in range(3):
    axs[i].plot(model_accel[:, i], label=f"Physics {acc_labels[i]}", color=colors[0], linestyle=linestyles[0])
    axs[i].plot(residual_pred[:, i], label=f"Residual {acc_labels[i]}", color=colors[1], linestyle=linestyles[1])
    axs[i].plot(hybrid_accel[:, i], label=f"Hybrid {acc_labels[i]}", color=colors[2], linestyle=linestyles[2])
    axs[i].plot(measured_accel[:, i], label=f"Measured {acc_labels[i]}", color=colors[3], linestyle=linestyles[3])
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_ylabel("Acceleration [m/s²]")

axs[2].set_xlabel("Sample index")
plt.suptitle("Measured vs Physics vs Residual vs Hybrid Accelerations")
plt.show()
