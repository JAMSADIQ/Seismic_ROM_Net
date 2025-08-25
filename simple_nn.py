import numpy as np
import seaborn as sns
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


##save_pod_solutions(pod_solutions, output_folder)
#save_pod_plots(pod_solutions, time_values, output_folder)
#save_pod_components(pod_bases, pod_coefficients, output_folder)
################Neural Network to get pod_coefficients

#######################Here we train a NN ###########################
#Step I
# Prepare data for training
vel_arr = []  # Input features (velocities)
tvlmax = []  # Output targets (alpha coeffients)
alpmax= []
#augment same data taking so much memory
#for i in range(2):
idx = 0
for velocity, alphas in pod_coefficients.items():
#for velocity, alphas in pod_solutions.items():
    plt.figure(figsize=(12, 5))
    sns.heatmap(alphas.T, cmap='viridis', cbar=True, xticklabels=4000, yticklabels=[f'Alpha {i+1}' for i in range(9)])
    plt.xlabel('Time Index')
    plt.ylabel('Alpha Index')
    plt.title('Alpha Values Over Time (Heatmap)')
    plt.show()
    stds = np.std(alphas, axis=0)
    plt.figure(figsize=(8, 4))
    plt.bar(range(1, 10), stds)
    plt.xlabel('Alpha Index')
    plt.ylabel('Standard Deviation')
    plt.title('Variability of Each Alpha Component')
    plt.grid(True)
    plt.show()


    fig, axes = plt.subplots(nrows=9, ncols=1, figsize=(12, 18), sharex=True)
    for i in range(9):
        axes[i].plot(time_values, alphas[:, i], label=f'Alpha {i+1}')
        axes[i].set_ylabel(f'Alpha {i+1}')
        axes[i].legend(loc='upper right')
        axes[i].grid(True)

    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    plt.show()

quit()
for k in range(2):
    print(alphas.shape)
    quit()
    LS_alpha = alphas[:, 0]
    max_idx = np.argmax(LS_alpha)
    print(time_values[max_idx])
    tvlmax.append(time_values[max_idx])
    alpmax.append(LS_alpha[max_idx])
    idx += 1
    if idx %10 == 0:
        plt.figure()
        plt.plot(time_values, LS_alpha)
        plt.title(f'maxalp{alphas[max_idx]}')
        plt.show()

    #print(alphas.shape)
    #X.append(velocity)
    #y.append(alphas)



quit()
X = np.array(X)
y = np.array(y)



# Flatten y into (N, T*D)
X = X.reshape(-1, 1)
y = y.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize velocity
X_mean, X_std = X_train.mean(), X_train.std()
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Dataset and loader
class VelocityDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(VelocityDataset(X_train_tensor, y_train_tensor), batch_size=8, shuffle=True)
test_loader = DataLoader(VelocityDataset(X_test_tensor, y_test_tensor), batch_size=8)

# ------------------------
# 2. Define MLP Model
# ------------------------

class MLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=200005):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_dim=1, output_dim=y_train.shape[1]).to(device)

# ------------------------
# 3. Train the Model
# ------------------------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 15
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

# ------------------------
# 4. Plot Loss Curves
# ------------------------

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------
# 5. Accuracy Plot on Test Sample
# ------------------------

model.eval()
with torch.no_grad():
    x_sample = X_test_tensor[0:1].to(device)
    y_true = y_test_tensor[0].cpu().reshape(40001, 5)
    y_pred = model(x_sample).cpu().reshape(40001, 5)

# Plot true vs predicted for mode 0
plt.plot(y_true[:, 0], label='True Mode 0')
plt.plot(y_pred[:, 0], '--', label='Predicted Mode 0')
plt.title("True vs Predicted ROM Coefficient (Mode 0)")
plt.xlabel("Timestep")
plt.ylabel("Coefficient Value")
plt.legend()
plt.tight_layout()
plt.show()
