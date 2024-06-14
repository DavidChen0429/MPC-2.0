from model import drone_dynamics
from planner import trajectory_generation
import matplotlib.pyplot as plt
import numpy as np

resultsX = []
resultsU = []
# define simulation parameters
dt = 0.1
t_final = 10.0  # seconds
time_steps = int(t_final / dt)
time = np.linspace(0, t_final, time_steps)
N = 10

Q = np.diag(np.array([10, 10, 100, 10, 10, 10, 1, 1, 1, 1, 1, 1]))
R1 = np.diag(np.array([1, 1, 1, 1]))
R2 = R1 * 10
R3 = R1 * 100
R4 = R1 * 1000
R_list = [R1, R2, R3, R4]

for R in R_list:
    parms1 = {"Q": Q, "R": R, "N": N, "Qf": Q, "dynamic": True}
    drone_model1 = drone_dynamics.Quadrotor(parms1)
    K, P = drone_model1.make_dlqr_controller(parms1)
    parms = {"Q": Q, "R": R, "N": N, "Qf": P, "dynamic": True}
    # parms = {"Q": Q, "R": R, "N": N, "Qf": Q, "dynamic": True}
    drone_model = drone_dynamics.Quadrotor(parms)

    # initialize state and input arrays
    x_bag, u_bag = drone_model.get_ss_bag_vectors(time_steps) #vnp.zeros((self.n_states, N))
    x0 = np.array([1, 0, 1, 0.5, 0, 0.5, 0, 0, 0, 0, 0, 0])
    u0 = np.array([0, 0, 0, 0])
    x_bag[:, 0] = x0
    u_bag[:, 0] = u0
    x_ref = trajectory_generation.hover_traj(time_steps)  # reference trajectory (state)

    # do optimal target selection
    x_ref, u_ref = drone_model.ots(x_ref[:3, 0], parms)

    for k in range(time_steps-1):
        x_current = x_bag[:, k]
        x_next, u = drone_model.step(x_current, x_ref, u_ref, cont_type="MPC", sim_system="linear", parms=parms)
        x_bag[:, k+1] = x_next
        u_bag[:, k+1] = u

    resultsX.append(x_bag)
    resultsU.append(u_bag)

# color: #d62728, #9467bd, #8c564b, #e377c2, #7f7f7f, #bcbd22, #17becf
# plot the results
plt.figure()
plt.subplot(2, 2, 1)
plt.step(time, resultsX[0][2, :], '#1f77b4', label="R1")
plt.step(time, resultsX[1][2, :], '#2ca02c', label="R2")
plt.step(time, resultsX[2][2, :], '#d62728', label="R3")
plt.step(time, resultsX[3][2, :], '#9467bd', label="R4")
plt.step(time, [x_ref[2] for _ in time], '#ff7f0e', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("z [m]")
plt.legend(loc='lower right')

plt.subplot(2, 2, 2)
plt.step(time, resultsU[0][0, :], '#1f77b4', label="R1")
plt.step(time, resultsU[1][0, :], '#2ca02c', label="R2")
plt.step(time, resultsU[2][0, :], '#d62728', label="R3")
plt.step(time, resultsU[3][0, :], '#9467bd', label="R4")
plt.xlabel("Time [s]")
plt.ylabel("Total Force [N]")
plt.legend(loc='lower right')

plt.subplot(2, 2, 3)
plt.step(time, resultsU[0][1, :], '#1f77b4', label="R1")
plt.step(time, resultsU[1][1, :], '#2ca02c', label="R2")
plt.step(time, resultsU[2][1, :], '#d62728', label="R3")
plt.step(time, resultsU[3][1, :], '#9467bd', label="R4")
plt.xlabel("Time [s]")
plt.ylabel("Torque X [Nm]")
plt.legend(loc='lower right')

plt.subplot(2, 2, 4)
plt.step(time, resultsU[0][3, :], '#1f77b4', label="R1")
plt.step(time, resultsU[1][3, :], '#2ca02c', label="R2")
plt.step(time, resultsU[2][3, :], '#d62728', label="R3")
plt.step(time, resultsU[3][3, :], '#9467bd', label="R4")
plt.xlabel("Time [s]")
plt.ylabel("Torque Z [Nm]")
plt.legend(loc='lower right')

plt.show()