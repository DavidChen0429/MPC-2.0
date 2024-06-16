from src.model import drone_dynamics
from src.planner import trajectory_generation
import matplotlib.pyplot as plt
import numpy as np


# define simulation parameters
dt = 0.1
t_final = 10.0  # seconds
time_steps = int(t_final / dt)
time = np.linspace(0, t_final, time_steps)

R = np.diag(np.array([1, 1, 1, 1]))
Q = np.eye(12) * 0.5
N = 5
parms1 = {"Q": Q, "R": np.eye(4), "N": 10, "Qf": Q, "dynamic": True}
drone_model1 = drone_dynamics.Quadrotor(parms1)
K, P = drone_model1.make_dlqr_controller(parms1)
parms = {"Q": Q, "R": R, "N": N, "Qf": P, "dynamic": True}
drone_model = drone_dynamics.Quadrotor(parms)

# initialize state and input arrays
x_bag, u_bag = drone_model.get_ss_bag_vectors(time_steps) #vnp.zeros((self.n_states, N))
x0 = np.array([5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
u0 = np.array([0, 0, 0, 0])
x_bag[:, 0] = x0
u_bag[:, 0] = u0
x_ref = trajectory_generation.hover_traj(time_steps)  # reference trajectory (state)

# do optimal target selection
Bd = np.array([1,0,0,0,0,0,0,0,0,0,0,0]).reshape((12, 1)) # disturbance on x
Cd = np.array([0,0,0,0,0,0,0,0,0,0,0,0]).reshape((12, 1))
parms["Bd"] = Bd
parms["Cd"] = Cd
drone_model.augment_sys_disturbance([1], Bd, Cd)
L_obs, K_aug = drone_model.Luenberger_observer(parms)
xhat = x_bag.copy()
xhat0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
xhat[:, 0] = 0
dhat = np.zeros((1, xhat.shape[1]))


def observer_dynamics(k, u, y):
    A_tilde = drone_model.dAugsys.A
    B_tilde = drone_model.dAugsys.B
    C_tilde = drone_model.dAugsys.C

    obs_state = np.hstack([xhat[:, k], dhat[:, k]])
    obst_state_next = A_tilde @ obs_state + B_tilde @ u + L_obs @ (y - C_tilde @ obs_state)
    return obst_state_next[:12], obst_state_next[12] #xhat, d_hat

y_ref = np.array([0., 0., 10.])

for k in range(time_steps-1):
    x_current = x_bag[:, k]

    xhat[:, k + 1], dhat[:, k + 1] = observer_dynamics(k, u_bag[:, k], x_bag[:, k])
    x_ref, u_ref = drone_model.ots_online(y_ref, dhat[:, k], parms)

    x_next, u = drone_model.step(x_current, x_ref, u_ref, cont_type="MPC", sim_system="linear", parms=parms)
    x_bag[:, k+1] = x_next
    u_bag[:, k+1] = u

# plot the results
plt.subplot(3, 2, 1)
plt.step(time, xhat[0, :], label=r"$\hat{x}$")
plt.step(time, x_bag[0, :], label="x")
plt.step(time, [y_ref[0] for _ in time], linestyle='--', label="ref")
# plt.step(time, dhat[:, :].reshape((time_steps,)), label=r"$\hat{d}$")
# plt.step(time, [1 for _ in time], label="d")
plt.xlabel("Time [s]")
plt.ylabel("x [m]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 3)
plt.step(time, xhat[1, :], label=r"$\hat{y}$")
plt.step(time, x_bag[1, :], label="y")
plt.step(time, [y_ref[1] for _ in time], linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("y [m]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 5)
plt.step(time, xhat[2, :], label=r"$\hat{z}$")
plt.step(time, x_bag[2, :], label="z")
plt.step(time, [y_ref[2] for _ in time], linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("z [m]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 2)
plt.step(time, xhat[3, :], label=r"$\hat{\phi}$")
plt.step(time, x_bag[3, :], label=r"$\phi$")
plt.xlabel("Time [s]")
plt.ylabel(r"$\phi$ [rad]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 4)
plt.step(time, xhat[4, :], label=r"$\hat{\theta}$")
plt.step(time, x_bag[4, :], label=r"$\theta$")
plt.xlabel("Time [s]")
plt.ylabel(r"$\theta$ [rad]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 6)
plt.step(time, xhat[5, :], label=r"$\hat{\psi}$")
plt.step(time, x_bag[5, :], label=r"$\psi$")
plt.xlabel("Time [s]")
plt.ylabel(r"$\psi$ [rad]")
plt.legend(loc='lower right')
plt.show()