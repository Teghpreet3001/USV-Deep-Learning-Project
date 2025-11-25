# LSTM vs MLP Implementation Comparison

## Overview
Both models implement a **residual learning approach** where a neural network learns to predict the residual (error) between physics-based model predictions and measured accelerations. The key difference is that the LSTM leverages temporal sequences while the MLP processes individual timesteps independently.

---

## Architecture Differences

### MLP (Multi-Layer Perceptron)
- **Architecture**: Simple feedforward network
  - 3 fully connected layers: `in_dim (5) → hidden (128) → hidden (128) → out_dim (3)`
  - ReLU activations between layers
  - Processes single timesteps: `(batch, 5) → (batch, 3)`

### LSTM (Long Short-Term Memory)
- **Architecture**: Recurrent neural network with LSTM cells
  - LSTM layers: `input_size=5, hidden_size=128, num_layers=2`
  - Dropout (0.1) between LSTM layers for regularization
  - Final fully connected layer: `hidden_dim (128) → out_dim (3)`
  - Processes sequences: `(batch, seq_len=10, 5) → (batch, 3)`
  - Uses only the **last timestep** output from the LSTM sequence

---

## Data Handling Differences

### MLP
- **Input**: Single timestep vectors `(batch, 5)`
  - Features: `[u, v, r, thrust_L, thrust_R]`
- **Dataset**: `TensorDataset` - simple pairing of inputs and targets
- **No temporal context**: Each sample is independent

### LSTM
- **Input**: Sequences of timesteps `(batch, seq_len=10, 5)`
  - Same features but organized as sliding windows
- **Dataset**: Custom `SequenceDataset` class
  - Creates overlapping sequences of length 10
  - Target is the acceleration at the **last timestep** of each sequence
- **Temporal context**: Model sees past 10 timesteps to predict current acceleration

**Sequence Creation Example:**
```python
# For sequence length 10:
# Sequence 0: timesteps [0:10] → target at timestep 9
# Sequence 1: timesteps [1:11] → target at timestep 10
# Sequence 2: timesteps [2:12] → target at timestep 11
# ...
```

---

## Training Process

### Common Elements (Both Models)
1. **Hybrid Approach**: Both use the same residual learning framework
   - Physics model (`Fossen3DOF`) predicts base acceleration
   - Neural network predicts residual (error)
   - Final prediction = physics prediction + residual prediction

2. **Loss Function**: MSE between predicted residual and actual residual
   ```python
   residual_target = measured_accel - physics_prediction
   residual_pred = model(input)
   loss = MSE(residual_pred, residual_target)
   ```

3. **Optimizer**: Adam with learning rate 1e-3

### Key Differences

| Aspect | MLP | LSTM |
|--------|-----|------|
| **Input Shape** | `(batch, 5)` | `(batch, 10, 5)` |
| **Physics Input** | Uses entire batch | Uses last timestep `[:, -1, :]` |
| **Temporal Awareness** | None | Learns from 10-step history |
| **Memory** | Stateless | Maintains hidden state across sequence |

---

## Why LSTM is Different

### 1. **Temporal Dependencies**
- **MLP**: Treats each timestep independently. Cannot learn that current acceleration depends on past velocities/inputs.
- **LSTM**: Can learn temporal patterns. For example:
  - If velocity has been increasing, acceleration might be different than if it's been constant
  - Can capture momentum effects and dynamic transitions

### 2. **Sequence Processing**
- **MLP**: `forward(x)` where `x` is `(batch, 5)`
- **LSTM**: `forward(x_seq)` where `x_seq` is `(batch, 10, 5)`
  - LSTM processes all 10 timesteps sequentially
  - Hidden state accumulates information across the sequence
  - Only the final hidden state is used for prediction

### 3. **Information Flow**
- **MLP**: Direct mapping from current state to residual
- **LSTM**: Information flows through time:
  ```
  t-9 → t-8 → ... → t-1 → t (prediction)
  ```
  Each timestep updates the hidden state, allowing the model to "remember" relevant past information.

---

## Implementation Details

### LSTM Architecture (`models/lstm.py`)
```python
class ResidualLSTM(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=128, num_layers=2, out_dim=3, dropout=0.1):
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, 
                           num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        # x: (batch, seq_len, in_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        output = self.fc(lstm_out[:, -1, :])  # Use last timestep
        return output  # (batch, out_dim)
```

### MLP Architecture (`models/mlp.py`)
```python
class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=3):
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)  # x: (batch, in_dim) → (batch, out_dim)
```

---

## Evaluation Differences

### MLP
- Can evaluate on all samples directly
- No padding needed

### LSTM
- Loses `seq_len - 1 = 9` samples at the beginning (can't create sequences)
- Requires padding for visualization to align with original data
- Evaluation uses the same sequence-based approach

---

## When to Use Each

### Use MLP when:
- Temporal dependencies are minimal
- Computational efficiency is critical
- You want a simpler, faster model
- Current state is sufficient for prediction

### Use LSTM when:
- Temporal patterns matter (e.g., momentum, acceleration trends)
- History provides important context
- You want to capture dynamic system behavior
- Sequential dependencies exist in the data

---

## Key Takeaways for Presentation

1. **Both use residual learning**: Neural network learns the error between physics model and reality
2. **LSTM adds temporal awareness**: Processes sequences of 10 timesteps instead of single timesteps
3. **LSTM can learn dynamics**: Can capture how past states influence current acceleration
4. **Trade-off**: LSTM is more complex but potentially more accurate for time-series data
5. **Same hybrid framework**: Both combine physics-based predictions with learned residuals

---

## Hyperparameters Comparison

| Parameter | MLP | LSTM |
|-----------|-----|------|
| Input dimension | 5 | 5 |
| Hidden size | 128 | 128 |
| Output dimension | 3 | 3 |
| Sequence length | N/A | 10 |
| Number of layers | 3 (FC) | 2 (LSTM) + 1 (FC) |
| Dropout | None | 0.1 |
| Learning rate | 1e-3 | 1e-3 |
| Batch size | 64 | 64 |
| Epochs | 100 | 100 |


