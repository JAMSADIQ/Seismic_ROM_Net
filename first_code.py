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


#Understand and visualize data
# ------------------- PARAMETERS -------------------
file_path = '/home/jsadiq/Downloads/Shafqat/merged_rom.txt'#'reduced_order_method.txt'
num_timesteps = 40001
num_points = 9
num_velocities = 201
num_modes = 5
data = np.loadtxt(file_path)


expected_size = num_timesteps * num_points * num_velocities
if data.size != expected_size:
    raise ValueError(f"Data size {data.size} does not match expected size {expected_size}")
data = data.reshape((num_timesteps, num_points, num_velocities))
time_interval = 0.002
time_values = np.linspace(0, (num_timesteps - 1) * time_interval, num_timesteps)
#use all data to train not fixed ones 
#velocities_to_plot = np.linspace(0, 201, 50)#num_velocities) #[100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500]
velocity_values = np.linspace(100, 900, 201) #np.linspace(1, 201, 201)#num_velocities)
velocities_to_plot =  np.random.choice(velocity_values, size=8, replace=False)

num_modes = 10

# POD-based ROM computation
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
    print("PHI shape:", Phi.shape, singular_values.shape, VT.shape)

    # Truncate to first `num_modes` POD modes (can also be selected automatically or we give a value)
    pod_basis = Phi[:, :num_modes]                       # Spatial POD modes (basis) left singular matrix
    Sigma_r = np.diag(singular_values[:num_modes])       # Energy (singular values) 
    temporal_modes = VT[:num_modes, :]                   # Temporal/parametric info also right singular matrix

    for velocity in velocity_values:
        if velocity not in velocity_to_index:
            raise ValueError(f"Velocity {velocity} not found in velocity_values.")
        v_idx = velocity_to_index[velocity]

        # Compute modal coefficients α = Σ_r × VT_r[:, v_idx]
        alpha = Sigma_r @ temporal_modes[:, v_idx]  # Shape: (num_modes,)
        #print(alpha.shape)
        reduced_solution = pod_basis @ alpha        # Linear combination: Φ × α, here I want alpha to be trained

        print("podbasis,alpha, reduced sol", pod_basis.shape, alpha.shape, reduced_solution.shape)
        pod_solutions.setdefault(velocity, []).append(reduced_solution) # for time series data
        # Save basis and coefficients from the first timestep only
        if timestep == 0:
            pod_bases[velocity] = pod_basis
            pod_coefficients.setdefault(velocity, []).append(alpha)
        else:
            pod_coefficients[velocity].append(alpha)

        #plt.plot(alpha)
        #plt.show()
# Convert all lists to arrays
for v in pod_solutions:
    pod_solutions[v] = np.array(pod_solutions[v])
    pod_coefficients[v] = np.array(pod_coefficients[v])

for velocity, solution in pod_solutions.items():
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, solution[:, 0], label=f'POD @ {velocity} m/s', linewidth=1.5)
    plt.title(f'POD Reduced Solution at {velocity} m/s')
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement [cm]")
    plt.grid(True)
    plt.legend()
    #file_path = os.path.join(output_folder, f"pod_solution_{velocity}.png")
    #plt.savefig(file_path)
    plt.show()
    #print(f"POD plots saved to folder: {output_folder}")
