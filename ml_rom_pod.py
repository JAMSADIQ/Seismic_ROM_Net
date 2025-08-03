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



# Function to load and reshape the data
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
file_path = 'merged_rom.txt'#'reduced_order_method.txt'
num_timesteps = 40001
num_points = 9
num_velocities = 201#101
num_modes = 5

#use all data to train not fixed ones 
velocities_to_plot = np.linspace(100, 500, 50)#num_velocities) #[100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500]
velocity_values = np.linspace(100, 500, 50)#num_velocities)

time_interval = 0.002
time_values = np.linspace(0, (num_timesteps - 1) * time_interval, num_timesteps)

# ------------------- EXECUTION -------------------
data = load_and_reshape_data(file_path, num_timesteps, num_points, num_velocities)

pod_solutions, pod_bases, pod_coefficients = compute_pod_solutions(
    data, velocities_to_plot, velocity_values, num_modes
)


##save_pod_solutions(pod_solutions, output_folder)
#save_pod_plots(pod_solutions, time_values, output_folder)
#save_pod_components(pod_bases, pod_coefficients, output_folder)
################Neural Network to get pod_coefficients

#######################Here we train a NN ###########################
#Step I
# Prepare data for training
X = []  # Input features (velocities)
y = []  # Output targets (alpha coeffients)
#augment same data taking so much memory
for i in range(2):
    for velocity, alphas in pod_coefficients.items():
    #print(velocity, alphas.shape)
        X.append(velocity)
        y.append(alphas)

X = np.array(X)
y = np.array(y)

# Dataset class in torch format
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class VelocityMatrixDataset(Dataset):
    def __init__(self, velocities, matrices, exact_solutions=None):
        """
        Args:
            velocities (list or np.array): List of velocities
            matrices (list of np.arrays): List of 40001x5 matrices
            exact_solutions (list of np.arrays): 
                     List of exact solution matrices (optional)
        """
        self.velocities = torch.FloatTensor(velocities)
        self.matrices = torch.FloatTensor(np.array(matrices))  # Converts list to 3D tensor
        self.exact_solutions = torch.FloatTensor(np.array(exact_solutions)) if exact_solutions is not None else None

    def __len__(self):
        return len(self.velocities)
    
    def __getitem__(self, idx):
        if self.exact_solutions is not None:
            return self.velocities[idx], self.matrices[idx], self.exact_solutions[idx]
        return self.velocities[idx], self.matrices[idx]

# NN  modify it as you like
class ROMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ROMNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU() #change activation fucntion

    #I am simply trying a feed forward neural network
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training parameters
input_size = 1  # Since input is just velocity (scalar)
hidden_size = 32 # modify it and see if it improves
output_size = 40001 * 5  # Flattened output size (40001*5)
batch_size = 2  # Small batch size as we dont have to much data 
num_epochs = 100  # Can be adjusted
learning_rate = 0.001
validation_split = 0.2  # 20% for validation

# Assuming you have X (velocities) and y (matrices) loaded
# X should be a list/array of 8 velocities
# y should be a list of 8 matrices, each 40001x5

# Create dataset 
full_dataset = VelocityMatrixDataset(X, y, exact_solutions=None)  # Add exact_solutions if available

# Split into training and validation
train_size = int((1 - validation_split) * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss and optimizer
model = ROMNet(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# For plotting
train_losses = []
val_losses = []
epochs = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    
    # Training phase
    for batch_velocities, batch_matrices in train_loader:
        # Flatten matrices
        batch_matrices_flat = batch_matrices.view(batch_matrices.size(0), -1)
        
        # Forward pass
        outputs = model(batch_velocities.unsqueeze(1))
        loss = criterion(outputs, batch_matrices_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item() * batch_velocities.size(0)
    
    # Calculate average training loss for the epoch
    epoch_train_loss /= len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # Validation phase
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for batch_velocities, batch_matrices in val_loader:
            batch_matrices_flat = batch_matrices.view(batch_matrices.size(0), -1)
            outputs = model(batch_velocities.unsqueeze(1))
            loss = criterion(outputs, batch_matrices_flat)
            epoch_val_loss += loss.item() * batch_velocities.size(0)
    
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    epochs.append(epoch + 1)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {epoch_train_loss:.6f}, Validation Loss: {epoch_val_loss:.6f}')
        print('-' * 50)

fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot(epochs, train_losses, 'r-', label='Training Loss')
line2, = ax.plot(epochs, val_losses, 'b-', label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training and Validation Loss')
ax.legend()
ax.grid(True)
plt.show()

# Final evaluation function
def evaluate_model(velocity, exact_matrix=None):
    model.eval()
    with torch.no_grad():
        velocity_tensor = torch.FloatTensor([velocity]).unsqueeze(1)
        output_flat = model(velocity_tensor)
        predicted_matrix = output_flat.view(40001, 5).numpy()

        if exact_matrix is not None:
            error = np.mean((predicted_matrix - exact_matrix) ** 2)
            print(f'MSE Error: {error:.6f}')

            # Plot comparison for first column (or any specific column)
            plt.figure(figsize=(12, 6))
            plt.plot(predicted_matrix[:, 0], 'r-', label='Predicted')
            plt.plot(exact_matrix[:, 0], 'b--', label='Exact')
            plt.title(f'Comparison for velocity = {velocity}')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.show()

    return predicted_matrix

# Example usage:
test_velocity = X[0]  # First velocity in your dataset
exact_matrix = y[0]   # Corresponding exact matrix
predicted_matrix = evaluate_model(test_velocity, exact_matrix)
quit()

for epoch in range(num_epochs):
    for batch_velocities, batch_matrices in dataloader:
        # Flatten the matrices for training
        batch_matrices_flat = batch_matrices.view(batch_matrices.size(0), -1)
        
        # Forward pass
        outputs = model(batch_velocities.unsqueeze(1))  # Add dimension for scalar input
        loss = criterion(outputs, batch_matrices_flat)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training finished!')

# Function to predict matrix from velocity
def predict_matrix(velocity):
    model.eval()
    with torch.no_grad():
        velocity_tensor = torch.FloatTensor([velocity]).unsqueeze(1)
        output_flat = model(velocity_tensor)
        # Reshape to 4001x5 matrix
        predicted_matrix = output_flat.view(40001, 5)
    return predicted_matrix.numpy()

# Example usage:
test_velocity = 100 # Your test velocity
predicted_matrix = predict_matrix(test_velocity)
print(predicted_matrix - pod_coefficients[100])  # Should be (4001, 5)
