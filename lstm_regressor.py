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
from sklearn.model_selection import train_test_split


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

# Assuming these are already loaded
# X: shape (n_samples,)
# y: shape (n_samples, 40001, 5)

X = np.array(X)  # scalar velocities
y = np.array(y)  # time series coefficients

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize velocity (input)
X_mean, X_std = X_train.mean(), X_train.std()
X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32).unsqueeze(1)  # (N, 1)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32).unsqueeze(1)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # (N, T, D)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Custom Dataset
class VelocityToCoefficientsDataset(Dataset):
    def __init__(self, X, y, timesteps, input_repeat):
        self.X = X
        self.y = y
        self.timesteps = timesteps
        self.input_repeat = input_repeat

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Repeat the scalar input as a sequence
        input_seq = self.X[idx].repeat(self.input_repeat).unsqueeze(1)  # (T, 1)
        return input_seq, self.y[idx]  # (T, 1), (T, D)

timesteps = y.shape[1]
n_modes = y.shape[2]

train_dataset = VelocityToCoefficientsDataset(X_train_tensor, y_train_tensor, timesteps, input_repeat=timesteps)
test_dataset = VelocityToCoefficientsDataset(X_test_tensor, y_test_tensor, timesteps, input_repeat=timesteps)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# --- 2. Define LSTM Model ---

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=34, output_dim=n_modes, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # x: (B, T, 1)
        out = self.linear(out)  # (B, T, D)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRegressor().to(device)

# --- 3. Train Model ---
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

n_epochs = 10
train_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_loss:.6f}")

# --- 4. Evaluate ---
model.eval()
with torch.no_grad():
    for x_test_batch, y_test_batch in test_loader:
        x_test_batch, y_test_batch = x_test_batch.to(device), y_test_batch.to(device)
        y_pred = model(x_test_batch)
        break  # just take the first batch for plotting

# --- 5. Plot Training Loss ---
plt.plot(train_losses)
plt.title("Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# --- 6. Plot Prediction vs Ground Truth ---
true = y_test_batch[0].cpu().numpy()
pred = y_pred[0].cpu().numpy()

plt.figure(figsize=(10, 4))
for i in range(n_modes):
    plt.plot(true[:, i], label=f"True Mode {i}", alpha=0.7)
    plt.plot(pred[:, i], '--', label=f"Pred Mode {i}", alpha=0.7)
plt.title("ROM Coefficient Time Series (First Test Sample)")
plt.xlabel("Timestep")
plt.ylabel("Coefficient")
plt.legend()
plt.tight_layout()
plt.show()

