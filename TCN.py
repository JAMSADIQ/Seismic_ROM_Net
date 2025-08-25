import numpy as np
from numpy.linalg import svd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# ------------------- STEP 1: LOAD AND PREPROCESS -------------------

def load_and_reshape_data(file_path, num_timesteps, num_points, num_velocities):
    data = np.loadtxt(file_path)
    expected_size = num_timesteps * num_points * num_velocities
    if data.size != expected_size:
        raise ValueError(f"Data size {data.size} does not match expected size {expected_size}")
    return data.reshape((num_timesteps, num_points, num_velocities))

def compute_pod_solutions(data, velocities_to_plot, velocity_values, num_modes):
    num_timesteps, num_points, num_velocities = data.shape
    pod_solutions = {}
    pod_bases = {}
    pod_coefficients = {}
    velocity_to_index = {v: i for i, v in enumerate(velocity_values)}

    for timestep in range(num_timesteps):
        snapshot_matrix = data[timestep, :, :]  # Shape: (num_points, num_velocities)
        Phi, singular_values, VT = svd(snapshot_matrix, full_matrices=False)
        pod_basis = Phi[:, :num_modes]
        Sigma_r = np.diag(singular_values[:num_modes])
        temporal_modes = VT[:num_modes, :]

        for velocity in velocities_to_plot:
            if velocity not in velocity_to_index:
                raise ValueError(f"Velocity {velocity} not found in velocity_values.")
            v_idx = velocity_to_index[velocity]
            alpha = Sigma_r @ temporal_modes[:, v_idx]  # (num_modes,)
            reduced_solution = pod_basis @ alpha
            pod_solutions.setdefault(velocity, []).append(reduced_solution)
            if timestep == 0:
                pod_bases[velocity] = pod_basis
                pod_coefficients.setdefault(velocity, []).append(alpha)
            else:
                pod_coefficients[velocity].append(alpha)

    for v in pod_solutions:
        pod_solutions[v] = np.array(pod_solutions[v])
        pod_coefficients[v] = np.array(pod_coefficients[v])

    return pod_solutions, pod_bases, pod_coefficients

# ------------------- STEP 2: DATA LOADING -------------------

file_path = '/home/jsadiq/Downloads/Shafqat/merged_rom.txt'
num_timesteps = 40001
num_points = 9
num_velocities = 201
num_modes = 5

velocities_to_plot = np.linspace(100, 900, 201)
velocity_values = np.linspace(100, 900, 201)

data = load_and_reshape_data(file_path, num_timesteps, num_points, num_velocities)
pod_solutions, pod_bases, pod_coefficients = compute_pod_solutions(data, velocities_to_plot, velocity_values, num_modes)

# Prepare inputs and targets
X = []
y = []

for velocity, alphas in pod_coefficients.items():
    X.append(velocity)
    y.append(alphas)  # shape (num_timesteps, num_modes)

X = np.array(X)  # shape (num_velocities,)
y = np.array(y)  # shape (num_velocities, num_timesteps, num_modes)

# Transpose y to (201, 40001, 5) → one sample per velocity
y = y.transpose(1, 0, 2)

# ------------------ STEP 2: TRAIN/TEST SPLIT ------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ STEP 3: DATASET & DATALOADER ------------------

class VelocityPODDataset(Dataset):
    def __init__(self, velocities, pod_series):
        self.X = (velocities - velocities.min()) / (velocities.max() - velocities.min())
        self.X = torch.tensor(self.X, dtype=torch.float32).unsqueeze(1)  # (N, 1)
        self.y = torch.tensor(pod_series, dtype=torch.float32)  # (N, T, 5)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = VelocityPODDataset(X_train, y_train)
test_dataset  = VelocityPODDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=4)

# ------------------ STEP 4: TCN MODEL ------------------

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)

class TCNFullSequence(nn.Module):
    def __init__(self, input_channels, output_channels, seq_len, num_layers=4, kernel_size=3):
        super().__init__()
        channels = 32
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_c = input_channels if i == 0 else channels
            layers.append(TemporalBlock(in_c, channels, kernel_size, dilation))
        self.tcn = nn.Sequential(*layers)
        self.output_layer = nn.Conv1d(channels, output_channels, 1)
        self.seq_len = seq_len

    def forward(self, x):
        x = x.unsqueeze(2)  # (B, 1, 1)
        x = x.repeat(1, 1, self.seq_len)  # (B, 1, T)
        y = self.tcn(x)  # (B, channels, T)
        return self.output_layer(y).permute(0, 2, 1)  # (B, T, modes)

# ------------------ STEP 5: TRAINING ------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = y.shape[1]  # 40001
num_modes = y.shape[2]  # 5

model = TCNFullSequence(input_channels=1, output_channels=num_modes, seq_len=seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

train_losses = []

for epoch in range(5):  # increase to 50–100 for real training
    model.train()
    total_loss = 0.0
    for vx, target in train_loader:
        vx = vx.to(device)  # (B, 1)
        target = target.to(device)  # (B, T, M)
        pred = model(vx)  # (B, T, M)

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}")

# ------------------ STEP 6: PLOT LOSS ------------------

plt.plot(train_losses, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------ STEP 7: TEST EXAMPLE ------------------

model.eval()
with torch.no_grad():
    for vx, true_y in test_loader:
        vx = vx.to(device)
        pred_y = model(vx)  # (B, T, M)
        pred_y = pred_y.cpu().numpy()
        true_y = true_y.numpy()

        # Plot first prediction
        plt.figure(figsize=(10, 4))
        plt.plot(pred_y[0][:, 0], label='Pred α₀')
        plt.plot(true_y[0][:, 0], label='True α₀', linestyle='--')
        plt.title(f"Alpha Mode 0 over Time (Velocity = {vx[0].item():.1f})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        break

