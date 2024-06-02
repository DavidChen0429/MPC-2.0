from src.model import drone_dynamics
import matplotlib.pyplot as plt
import numpy as np

# define simulation parameters
dt = 0.1
t_final = 20.0 # seconds
time_steps = int(t_final / dt)
time = np.linspace(0, t_final, time_steps)

Q = np.eye(12)
parms = {"Q": Q, "R": np.eye(4), "N": 10, "Qf": Q, "dynamic": True}
drone_model = drone_dynamics.Quadrotor(parms)
drone_model.initiate(feedback="output")

# design the observer
Lobs, Klqr_aug = drone_model.luenberger_observer(parms)

# Test the observer
x_bag, u_bag, xhat_bag = drone_model.get_ss_bag_vectors(time_steps)
x_0 = np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
xhat_0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
u_0 = np.array([0, 0, 0, 0])
x_bag[:, 0] = x_0
u_bag[:, 0] = u_0
xhat_bag[:, 0] = xhat_0

x_x_ref = np.zeros(time_steps)
x_y_ref = np.zeros(time_steps)
x_z_ref = np.ones(time_steps) * 10
d = 1
d_ref = d*np.ones((1, time_steps))
x_ref_series = np.vstack([x_x_ref, x_y_ref, x_z_ref, np.zeros((9, time_steps)), d_ref])


for k in range(time_steps - 1):
    x = x_bag[:, k]
    x_ref = x_ref_series[:, k]
    xhat = xhat_bag[:, k]
    u = Klqr_aug @ (x_ref - x)
    x_next = drone_model.dsys.A @ x + drone_model.dsys.B @ u
    y = drone_model.dsys.C @ x

    y_hat = drone_model.dsys.C @ xhat
    error = y - y_hat
    x_hat = drone_model.dsys.A @ xhat + drone_model.dsys.B @ u + Lobs @ error

    xhat_bag[:, k+1] = x_hat
    x_bag[:, k+1] = x_next
    u_bag[:, k+1] = u

# Visualize the results
plt.subplot(1, 3, 1)
plt.step(time, x_bag[0, :], '#1f77b4', label="x_x")
plt.step(time, x_bag[1, :], '#ff7f0e', label="x_y")
plt.step(time, x_bag[2, :], '#2ca02c', label="x_z")
plt.step(time, x_bag[12, :], '#d62728', label="d")
plt.xlabel("Time [s]")
plt.title("Actual States")
plt.legend()

plt.subplot(1, 3, 2)
plt.step(time, xhat_bag[0, :], '#1f77b4', linestyle='--', label="x_x hat")
plt.step(time, xhat_bag[1, :], '#ff7f0e', linestyle='--', label="x_y hat")
plt.step(time, xhat_bag[2, :], '#2ca02c', linestyle='--', label="x_z hat")
plt.step(time, xhat_bag[12, :], '#d62728', label="d hat")
plt.xlabel("Time [s]")
plt.title("Estimated States")
plt.legend()

plt.subplot(1, 3, 3)
plt.step(time, x_bag[0, :] - xhat_bag[0, :], '#1f77b4', label="error x_x")
plt.step(time, x_bag[1, :] - xhat_bag[1, :], '#ff7f0e', label="error x_y")
plt.step(time, x_bag[2, :] - xhat_bag[2, :], '#2ca02c', label="error x_z")
plt.step(time, x_bag[12, :] - xhat_bag[12, :], '#d62728', label="error d")
plt.xlabel("Time [s]")
plt.title("Error States")
plt.legend()

plt.tight_layout()
plt.show()