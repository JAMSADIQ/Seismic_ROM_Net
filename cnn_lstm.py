import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import svd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


def load_and_reshape_data(file_path, num_timesteps, num_points, num_velocities):
    data = np.loadtxt(file_path)
    expected_size = num_timesteps * num_points * num_velocities
    if data.size != expected_size:
        raise ValueError(f"Data size {data.size} does not match expected size {expected_size}")
    return data.reshape((num_timesteps, num_points, num_velocities))

# POD-based ROM computation
def compute_pod_solutions(data, velocities_to_plot, velocity_values, num_modes):
    """
    Apply POD and reconstruct solutions for selected velocities using reduced basis.
    """
    num_timesteps, num_points, num_velocities = data.shape
    pod_solutions = {} # collect the reconstructed reduced solution at each time step for each velocity
                       # Each key is a velocity, and each value is a list of reduced solutions over time
    pod_bases = {}
    pod_coefficients = {}

    # Map velocity value to index in the data
    velocity_to_index = {v: i for i, v in enumerate(velocity_values)}

    for timestep in range(num_timesteps):
        snapshot_matrix = data[timestep, :, :]  # Shape: (num_points, num_velocities)

        # POD = SVD of the snapshot matrix
        Phi, singular_values, VT = svd(snapshot_matrix, full_matrices=False)

        # Truncate to first `num_modes` POD modes (can also be selected automatically or we give a value)
        pod_basis = Phi[:, :num_modes]                       # Spatial POD modes (basis) left singular matrix
        Sigma_r = np.diag(singular_values[:num_modes])       # Energy (singular values) 
        temporal_modes = VT[:num_modes, :]                   # Temporal/parametric info also right singular matrix

        for velocity in velocities_to_plot:
            if velocity not in velocity_to_index:
                raise ValueError(f"Velocity {velocity} not found in velocity_values.")
            v_idx = velocity_to_index[velocity]

            # Compute modal coefficients α = Σ_r × VT_r[:, v_idx]
            alpha = Sigma_r @ temporal_modes[:, v_idx]  # Shape: (num_modes,)
            reduced_solution = pod_basis @ alpha        # Linear combination: Φ × α, here I want alpha to be trained

            pod_solutions.setdefault(velocity, []).append(reduced_solution) # for time series data

            # Save basis and coefficients from the first timestep only
            if timestep == 0:
                pod_bases[velocity] = pod_basis
                pod_coefficients.setdefault(velocity, []).append(alpha)
            else:
                pod_coefficients[velocity].append(alpha)

    # Convert all lists to arrays
    for v in pod_solutions:
        pod_solutions[v] = np.array(pod_solutions[v])
        pod_coefficients[v] = np.array(pod_coefficients[v])

    return pod_solutions, pod_bases, pod_coefficients

# Save reconstructed solutions
def save_pod_solutions(pod_solutions, output_folder='./'):
    for velocity, solution in pod_solutions.items():
        filepath = os.path.join(output_folder, f"pod_solution_{velocity}.txt")
        np.savetxt(filepath, solution, fmt="%.6e")
    print(f"POD solutions saved to folder: {output_folder}")

# Save solution plots
def save_pod_plots(pod_solutions, time_values, output_folder='./'):
    for velocity, solution in pod_solutions.items():
        plt.figure(figsize=(10, 6))
        plt.plot(time_values, solution[:, 0], label=f'POD @ {velocity} m/s', linewidth=1.5)
        plt.title(f'POD Reduced Solution at {velocity} m/s')
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement [cm]")
        plt.grid(True)
        plt.legend()
        file_path = os.path.join(output_folder, f"pod_solution_{velocity}.png")
        plt.savefig(file_path)
        plt.close()
    print(f"POD plots saved to folder: {output_folder}")

# Save POD basis and coefficients
def save_pod_components(pod_bases, pod_coefficients, output_folder='./'):
    for v, basis in pod_bases.items():
        np.savetxt(os.path.join(output_folder, f"pod_basis_{v}.txt"), basis, fmt='%.6e')
    for v, coeffs in pod_coefficients.items():
        np.savetxt(os.path.join(output_folder, f"pod_coeffs_{v}.txt"), coeffs, fmt='%.6e')
    print("POD basis functions and coefficients saved.")

# ------------------- PARAMETERS -------------------
file_path = '/home/jsadiq/Downloads/Shafqat/merged_rom.txt'#'reduced_order_method.txt'
num_timesteps = 40001
num_points = 9
num_velocities = 201#101
num_modes = 5

#use all data to train not fixed ones 
velocities_to_plot = np.linspace(100, 900, 201)#num_velocities) #[100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500]
velocity_values = np.linspace(100, 900, 201)#num_velocities)

time_interval = 0.002
time_values = np.linspace(0, (num_timesteps - 1) * time_interval, num_timesteps)

# ------------------- EXECUTION -------------------
data = load_and_reshape_data(file_path, num_timesteps, num_points, num_velocities)

pod_solutions, pod_bases, pod_coefficients = compute_pod_solutions(data, velocities_to_plot, velocity_values, num_modes)

# Prepare data for training
X = []  # Input features (velocities)
y = []  # Output targets (alpha coeffients)
#augment same data taking so much memory
#for i in range(2):
for velocity, alphas in pod_coefficients.items():
    #print(alphas.shape)
    X.append(velocity)
    y.append(alphas)

X = np.array(X)
y = np.array(y)


X = np.array(X).reshape(-1, 1)
y = np.array(y)  # Shape: (N, 40001, 5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize X
X_mean, X_std = X_train.mean(), X_train.std()
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Convert to tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# --------------
# 2. Dataset & Dataloader
# --------------

class VelocityTimeSeriesDataset(Dataset):
    def __init__(self, X, y, timesteps):
        self.X = X
        self.y = y
        self.T = timesteps

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        velocity = self.X[idx]
        # Repeat scalar velocity to form a sequence
        seq = velocity.repeat(self.T).unsqueeze(1)  # (T, 1)
        return seq, self.y[idx]  # (T, 1), (T, D)

timesteps = y.shape[1]
n_modes = y.shape[2]

train_dataset = VelocityTimeSeriesDataset(X_train_tensor, y_train_tensor, timesteps)
test_dataset = VelocityTimeSeriesDataset(X_test_tensor, y_test_tensor, timesteps)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# --------------
# 3. CNN-LSTM Model
# --------------

class CNNLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=n_modes):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, T, 1) → (B, 1, T)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))  # (B, 8, T)
        x = x.permute(0, 2, 1)        # (B, T, 8)
        out, _ = self.lstm(x)         # (B, T, H)
        return self.fc(out)           # (B, T, D)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM().to(device)

# --------------
# 4. Training Loop
# --------------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_losses[-1]:.6f} | Val Loss: {val_losses[-1]:.6f}")

# --------------
# 5. Plot Loss Curve
# --------------

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------
# 6. Predict on One Test Sample and Plot
# --------------

model.eval()
with torch.no_grad():
    x_sample = X_test_tensor[0:1].repeat(timesteps, 1).unsqueeze(0).to(device)  # (1, T, 1)
    true_y = y_test_tensor[0].cpu().numpy()  # (T, D)
    pred_y = model(x_sample).cpu().squeeze(0).numpy()  # (T, D)

# Plot for mode 0
plt.figure(figsize=(10, 4))
plt.plot(true_y[:, 0], label='True Mode 0')
plt.plot(pred_y[:, 0], '--', label='Predicted Mode 0')
plt.title("True vs Predicted Coefficient (Mode 0)")
plt.xlabel("Timestep")
plt.ylabel("Coefficient Value")
plt.legend()
plt.tight_layout()
plt.show()

