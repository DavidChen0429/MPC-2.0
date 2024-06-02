from model import drone_dynamics
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

# Augment the system with a constant disturbance term d
d = [1]
Bd = np.array([1,0,0,0,0,0,0,0,0,0,0,0]).reshape((12, 1)) # disturbance on x
Cd = np.array([0,0,0,0,0,0,0,0,0,0,0,0]).reshape(12, 1) # measure only x
A_aug, B_aug, C_aug, D_aug = drone_model.augment_sys_disturbance(d, Bd, Cd)

# Design the observer
Lobs, Klqr_aug = drone_model.Luenberger_observer(parms)

# Test the observer
x_bag, u_bag = drone_model.get_ss_bag_vectors(time_steps) # np.zeros((self.n_states, N))
x_aug_bag = np.vstack([x_bag, np.zeros((1, time_steps))])
x_aug_hat = np.zeros((13, time_steps))
x0_aug = np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
x0_hat = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
u0 = np.array([0, 0, 0, 0])
x_aug_bag[:, 0] = x0_aug
u_bag[:, 0] = u0
x_aug_hat[:, 0] = x0_hat

x_x_ref = np.zeros(time_steps)
x_y_ref = np.zeros(time_steps)
x_z_ref = np.ones(time_steps) * 10
x_ref = np.vstack((x_x_ref, x_y_ref, x_z_ref, np.zeros((9, time_steps))))
x_ref_aug = np.vstack([x_ref, d*np.ones((1, time_steps))])

for k in range(time_steps-1):
    x_aug_current = x_aug_bag[:, k]
    x_ref_current = x_ref_aug[:, k]
    x_hat_current = x_aug_hat[:, k]
    u = Klqr_aug @ (x_ref_current - x_aug_current)
    x_next = drone_model.dAugsys.A @ x_aug_current + drone_model.dAugsys.B @ u
    y = drone_model.dAugsys.C @ x_aug_current

    y_hat = drone_model.dAugsys.C @ x_hat_current
    error = y - y_hat
    x_hat = drone_model.dAugsys.A @ x_hat_current + drone_model.dAugsys.B @ u + Lobs @ error

    x_aug_hat[:, k+1] = x_hat
    x_aug_bag[:, k+1] = x_next
    u_bag[:, k+1] = u

# Visualize the results
plt.subplot(1, 3, 1)
plt.step(time, x_aug_bag[0, :], '#1f77b4', label="x_x")
plt.step(time, x_aug_bag[1, :], '#ff7f0e', label="x_y")
plt.step(time, x_aug_bag[2, :], '#2ca02c', label="x_z")
plt.step(time, x_aug_bag[12, :], '#d62728', label="d")
plt.xlabel("Time [s]")
plt.title("Actual States")
plt.legend()

plt.subplot(1, 3, 2)
plt.step(time, x_aug_hat[0, :], '#1f77b4', linestyle='--', label="x_x hat")
plt.step(time, x_aug_hat[1, :], '#ff7f0e', linestyle='--', label="x_y hat")
plt.step(time, x_aug_hat[2, :], '#2ca02c', linestyle='--', label="x_z hat")
plt.step(time, x_aug_hat[12, :], '#d62728', label="d hat")
plt.xlabel("Time [s]")
plt.title("Estimated States")
plt.legend()

plt.subplot(1, 3, 3)
plt.step(time, x_aug_bag[0, :] - x_aug_hat[0, :], '#1f77b4', label="error x_x")
plt.step(time, x_aug_bag[1, :] - x_aug_hat[1, :], '#ff7f0e', label="error x_y")
plt.step(time, x_aug_bag[2, :] - x_aug_hat[2, :], '#2ca02c', label="error x_z")
plt.step(time, x_aug_bag[12, :] - x_aug_hat[12, :], '#d62728', label="error d")
plt.xlabel("Time [s]")
plt.title("Error States")
plt.legend()

plt.tight_layout()
plt.show()

# Those figures won't show a nice reference tracking, which is fine becasue the system is uncontrollable
# The purpose is to show that the observer is able to estimate the states and the disturbance term