from src.model import drone_dynamics
from src.planner import trajectory_generation
import matplotlib.pyplot as plt
import numpy as np


# define simulation parameters
dt = 0.1
t_final = 20.0  # seconds
time_steps = int(t_final / dt)
time = np.linspace(0, t_final, time_steps)

Q = np.eye(12)
parms = {"Q": Q, "R": np.eye(4), "N": 10, "Qf": Q, "dynamic": True}
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
xhat[0,0] = 0
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
plt.figure()
plt.step(time, x_bag[2, :], '#1f77b4', label="x_x")
plt.step(time, xhat[2, :], 'orange', label=r"$\hat{x}_x$")
plt.step(time, [x_ref[2] for _ in time], '#1f77b4', linestyle='--', label="x_x ref")
plt.xlabel("Time [s]")
plt.legend()
plt.show()

plt.subplot(3, 2, 1)
plt.step(time, x_bag[0, :], '#1f77b4', label="x")
plt.step(time, xhat[0, :], '#2ca02c', label=r"$\hat{x}_x$")
plt.step(time, [y_ref[0] for _ in time], '#d62728', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("x [m]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 3)
plt.step(time, x_bag[1, :], '#1f77b4', label="y")
plt.step(time, xhat[1, :], '#2ca02c', label=r"$\hat{x}_y$")
plt.step(time, [y_ref[1] for _ in time], '#d62728', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("y [m]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 5)
plt.step(time, x_bag[2, :], '#1f77b4', label="z")
plt.step(time, xhat[2, :], '#2ca02c', label=r"$\hat{x}_z$")
plt.step(time, [y_ref[2] for _ in time], '#d62728', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("z [m]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 2)
plt.step(time, x_bag[3, :], '#1f77b4', label=r"$\phi$")
plt.step(time, xhat[3, :], '#2ca02c', label=r"$\hat{x}_z$")
#plt.step(time, [y_ref[2] for _ in time], '#d62728', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel(r"$\phi$ [rad]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 4)
plt.step(time, x_bag[4, :], '#1f77b4', label=r"$\theta$")
plt.step(time, xhat[4, :], '#2ca02c', label=r"$\hat{x}_\theta$")
#plt.step(time, [y_ref[2] for _ in time], '#d62728', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel(r"$\theta$ [rad]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 6)
plt.step(time, x_bag[5, :], '#1f77b4', label=r"$\psi$")
plt.step(time, xhat[5, :], '#2ca02c', label=r"$\hat{x}_\psi$")
#plt.step(time, [y_ref[2] for _ in time], '#d62728', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel(r"$\psi$ [rad]")
plt.legend(loc='lower right')
#
# plt.subplot(3, 2, 3)
# plt.step(time, resultsX[0][1, :], '#1f77b4', label="N=2")
# plt.step(time, resultsX[1][1, :], '#2ca02c', label="N=5")
# plt.step(time, resultsX[2][1, :], '#d62728', label="N=10")
# plt.step(time, resultsX[3][1, :], '#9467bd', label="N=50")
# plt.step(time, resultsX[4][1, :], '#8c564b', label="N=100")
# plt.step(time, [x_ref[1] for _ in time], '#ff7f0e', linestyle='--', label="ref")
# plt.xlabel("Time [s]")
# plt.ylabel("y [m]")
# plt.legend(loc='lower right')
#
# plt.subplot(3, 2, 5)
# plt.step(time, resultsX[0][2, :], '#1f77b4', label="N=2")
# plt.step(time, resultsX[1][2, :], '#2ca02c', label="N=5")
# plt.step(time, resultsX[2][2, :], '#d62728', label="N=10")
# plt.step(time, resultsX[3][2, :], '#9467bd', label="N=50")
# plt.step(time, resultsX[4][2, :], '#8c564b', label="N=100")
# plt.step(time, [x_ref[2] for _ in time], '#ff7f0e', linestyle='--', label="ref")
# plt.xlabel("Time [s]")
# plt.ylabel("z [m]")
# plt.legend(loc='lower right')
#
# plt.subplot(3, 2, 2)
# plt.step(time, resultsX[0][4, :], '#1f77b4', label="N=2")
# plt.step(time, resultsX[1][4, :], '#2ca02c', label="N=5")
# plt.step(time, resultsX[2][4, :], '#d62728', label="N=10")
# plt.step(time, resultsX[3][4, :], '#9467bd', label="N=50")
# plt.step(time, resultsX[4][4, :], '#8c564b', label="N=100")
# plt.step(time, [x_ref[4] for _ in time], '#ff7f0e', linestyle='--', label="ref")
# plt.xlabel("Time [s]")
# plt.ylabel("phi [rad]")
# plt.legend(loc='lower right')
#
# plt.subplot(3, 2, 4)
# plt.step(time, resultsX[0][5, :], '#1f77b4', label="N=2")
# plt.step(time, resultsX[1][5, :], '#2ca02c', label="N=5")
# plt.step(time, resultsX[2][5, :], '#d62728', label="N=10")
# plt.step(time, resultsX[3][5, :], '#9467bd', label="N=50")
# plt.step(time, resultsX[4][5, :], '#8c564b', label="N=100")
# plt.step(time, [x_ref[5] for _ in time], '#ff7f0e', linestyle='--', label="ref")
# plt.xlabel("Time [s]")
# plt.ylabel("theta [rad]")
# plt.legend(loc='lower right')
#
# plt.subplot(3, 2, 6)
# plt.step(time, resultsX[0][6, :], '#1f77b4', label="N=2")
# plt.step(time, resultsX[1][6, :], '#2ca02c', label="N=5")
# plt.step(time, resultsX[2][6, :], '#d62728', label="N=10")
# plt.step(time, resultsX[3][6, :], '#9467bd', label="N=50")
# plt.step(time, resultsX[4][6, :], '#8c564b', label="N=100")
# plt.step(time, [x_ref[6] for _ in time], '#ff7f0e', linestyle='--', label="ref")
# plt.xlabel("Time [s]")
# plt.ylabel("psi [rad]")
# plt.legend(loc='lower right')

plt.show()