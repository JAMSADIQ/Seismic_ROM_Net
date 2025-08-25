import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.linalg import svd

#import ace_tools as tools

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

velarr = np.zeros(201)
tmax = np.zeros(201) 
Amax = np.zeros(201)

for idx,  (velocity, alphas) in enumerate(pod_solutions.items()):
    velarr[idx] = velocity
    LS_alpha = alphas[:, 4] 
    max_idx = np.argmax(LS_alpha)
    tmax[idx] = time_values[max_idx]
    Amax[idx] = LS_alpha[max_idx]



plt.plot(velarr, Amax, 'r+')
plt.show()
X_train = velarr.reshape(-1, 1)
vel_train = X_train
y_train = np.stack([tmax, Amax], axis=1)
# New velocities to test model generalization
vel_test = np.arange(910, 1001, 10).reshape(-1, 1)
model_preds = {}


#k-Nearest Neighbors
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_train_knn = knn.predict(X_train)
y_test_knn = knn.predict(vel_test)
model_preds['k-NN'] = (y_train_knn, y_test_knn)

# 2. Gaussian Process
gpr = GaussianProcessRegressor(kernel=RBF())
gpr.fit(X_train, y_train)
y_train_gpr = gpr.predict(X_train)
y_test_gpr = gpr.predict(vel_test)
model_preds['Gaussian Process'] = (y_train_gpr, y_test_gpr)

# 3. Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_train_rf = rf.predict(X_train)
y_test_rf = rf.predict(vel_test)
model_preds['Random Forest'] = (y_train_rf, y_test_rf)



# Plot predictions for each model
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, (y_train_pred, y_test_pred)) in enumerate(model_preds.items()):
    axs[i].plot(vel_train, y_train_pred[:, 1], 'r+', label="Train (Predicted Amax)")
    axs[i].plot(vel_test.ravel(), y_test_pred[:, 1], 'b+', label="Test (Extrapolated Amax)")
    axs[i].set_title(f"{name} - Amax Prediction")
    axs[i].set_xlabel("Velocity")
    axs[i].set_ylabel("Predicted Amax")
    axs[i].legend()
    axs[i].grid(True)

plt.tight_layout()
plt.show()

# Choose one model (e.g., Polynomial Regression with degree=3)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(vel_test)

model = LinearRegression().fit(X_train_poly, y_train)
y_test_pred = model.predict(X_test_poly)
y_train_pred = model.predict(X_train_poly)




quit()
# Split data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Dictionary to store predictions and metrics
results = {}
metrics_test = {}

# Helper function to evaluate model
def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics_test[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
    results[name] = y_pred

# ---------------------------------------------
# MODEL 1: Polynomial Regression (degree 3)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
model_poly = LinearRegression().fit(X_train_poly, y_train)
pred_poly = model_poly.predict(X_test_poly)
evaluate_model("Polynomial Regression", y_test, pred_poly)

# MODEL 2: Gaussian Process Regression
gpr = GaussianProcessRegressor(kernel=RBF())
gpr.fit(X_train, y_train)
pred_gpr = gpr.predict(X_test)
evaluate_model("Gaussian Process", y_test, pred_gpr)

# MODEL 3: Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
evaluate_model("Random Forest", y_test, pred_rf)

# MODEL 4: k-Nearest Neighbors Regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
evaluate_model("k-NN", y_test, pred_knn)

# ---------------------------------------------
# Print metrics
df_metrics = pd.DataFrame(metrics_test).T
print("\nModel Performance on Test Set:\n")
print(df_metrics)

# ---------------------------------------------
# Plotting predicted vs true values
fig, axs = plt.subplots(2, len(results), figsize=(20, 8))

for i, (name, pred) in enumerate(results.items()):
    # tmax plot
    axs[0, i].scatter(y_test[:, 0], pred[:, 0], alpha=0.7)
    axs[0, i].plot([min(y_test[:, 0]), max(y_test[:, 0])],
                   [min(y_test[:, 0]), max(y_test[:, 0])], 'r--')
    axs[0, i].set_title(f"{name} - tmax")
    axs[0, i].set_xlabel("True tmax")
    axs[0, i].set_ylabel("Predicted tmax")

    # Amax plot
    axs[1, i].scatter(y_test[:, 1], pred[:, 1], alpha=0.7)
    axs[1, i].plot([min(y_test[:, 1]), max(y_test[:, 1])],
                   [min(y_test[:, 1]), max(y_test[:, 1])], 'r--')
    axs[1, i].set_title(f"{name} - Amax")
    axs[1, i].set_xlabel("True Amax")
    axs[1, i].set_ylabel("Predicted Amax")

plt.tight_layout()
plt.show()



#quit()

# Store results
results = {}
# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model_poly = LinearRegression().fit(X_poly, y)
pred_poly = model_poly.predict(X_poly)
results['Polynomial Regression'] = pred_poly

# Gaussian Process Regression
gpr = GaussianProcessRegressor(kernel=RBF())
gpr.fit(X, y)
pred_gpr = gpr.predict(X)
results['Gaussian Process'] = pred_gpr

# Random Forest
rf = RandomForestRegressor()
rf.fit(X, y)
pred_rf = rf.predict(X)
results['Random Forest'] = pred_rf

# k-Nearest Neighbors
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X, y)
pred_knn = knn.predict(X)
results['k-NN'] = pred_knn

# Evaluate
metrics = {}
for name, pred in results.items():
    mse = mean_squared_error(y, pred)
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    metrics[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}


import pandas as pd

# Convert metrics to DataFrame
df_metrics = pd.DataFrame(metrics).T

# Print the DataFrame
print("Model Comparison Metrics:\n")
print(df_metrics)

# Plot prediction vs true
fig, axs = plt.subplots(2, len(results), figsize=(20, 8))
for i, (name, pred) in enumerate(results.items()):
    axs[0, i].scatter(y[:, 0], pred[:, 0], alpha=0.7)
    axs[0, i].plot([min(tmax), max(tmax)], [min(tmax), max(tmax)], 'r--')
    axs[0, i].set_title(f"{name} - tmax")
    axs[0, i].set_xlabel("True")
    axs[0, i].set_ylabel("Predicted")

    axs[1, i].scatter(y[:, 1], pred[:, 1], alpha=0.7)
    axs[1, i].plot([min(Amax), max(Amax)], [min(Amax), max(Amax)], 'r--')
    axs[1, i].set_title(f"{name} - Amax")
    axs[1, i].set_xlabel("True")
    axs[1, i].set_ylabel("Predicted")

plt.tight_layout()
plt.show()
