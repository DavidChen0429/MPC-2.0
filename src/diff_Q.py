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

R_diag = np.array([1, 1, 1, 1])
R = np.diag(R_diag)
Q1 = np.diag(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
Q2 = Q1 * 10
Q3 = Q1 * 100
Q4 = Q1 * 0.1
Q5 = Q1 * 0.01
Q_list = [Q1, Q2, Q3, Q4, Q5]

for Q in Q_list:
    parms1 = {"Q": Q, "R": R, "N": N, "Qf": Q, "dynamic": True}
    drone_model1 = drone_dynamics.Quadrotor(parms1)
    K, P = drone_model1.make_dlqr_controller(parms1)
    parms = {"Q": Q, "R": R, "N": N, "Qf": P, "dynamic": True}
    # parms = {"Q": Q, "R": R, "N": N, "Qf": Q, "dynamic": True}
    drone_model = drone_dynamics.Quadrotor(parms)

    # initialize state and input arrays
    x_bag, u_bag = drone_model.get_ss_bag_vectors(time_steps) #vnp.zeros((self.n_states, N))
    x0 = np.array([1, 1, 5, 0.5, 0.5, 0.5, 0, 0, 0, 0, 0, 0])
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
plt.subplot(3, 2, 1)
plt.step(time, resultsX[0][0, :], '#1f77b4', label="Q1")
plt.step(time, resultsX[1][0, :], '#2ca02c', label="Q2")
plt.step(time, resultsX[2][0, :], '#d62728', label="Q3")
plt.step(time, resultsX[3][0, :], '#9467bd', label="Q4")
plt.step(time, resultsX[4][0, :], '#8c564b', label="Q5")
plt.step(time, [x_ref[0] for _ in time], '#ff7f0e', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("x [m]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 3)
plt.step(time, resultsX[0][1, :], '#1f77b4', label="Q1")
plt.step(time, resultsX[1][1, :], '#2ca02c', label="Q2")
plt.step(time, resultsX[2][1, :], '#d62728', label="Q3")
plt.step(time, resultsX[3][1, :], '#9467bd', label="Q4")
plt.step(time, resultsX[4][1, :], '#8c564b', label="Q5")
plt.step(time, [x_ref[1] for _ in time], '#ff7f0e', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("y [m]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 5)
plt.step(time, resultsX[0][2, :], '#1f77b4', label="Q1")
plt.step(time, resultsX[1][2, :], '#2ca02c', label="Q2")
plt.step(time, resultsX[2][2, :], '#d62728', label="Q3")
plt.step(time, resultsX[3][2, :], '#9467bd', label="Q4")
plt.step(time, resultsX[4][2, :], '#8c564b', label="Q5")
plt.step(time, [x_ref[2] for _ in time], '#ff7f0e', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("z [m]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 2)
plt.step(time, resultsX[0][4, :], '#1f77b4', label="Q1")
plt.step(time, resultsX[1][4, :], '#2ca02c', label="Q2")
plt.step(time, resultsX[2][4, :], '#d62728', label="Q3")
plt.step(time, resultsX[3][4, :], '#9467bd', label="Q4")
plt.step(time, resultsX[4][4, :], '#8c564b', label="Q5")
plt.step(time, [x_ref[4] for _ in time], '#ff7f0e', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("phi [rad]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 4)
plt.step(time, resultsX[0][5, :], '#1f77b4', label="Q1")
plt.step(time, resultsX[1][5, :], '#2ca02c', label="Q2")
plt.step(time, resultsX[2][5, :], '#d62728', label="Q3")
plt.step(time, resultsX[3][5, :], '#9467bd', label="Q4")
plt.step(time, resultsX[4][5, :], '#8c564b', label="Q5")
plt.step(time, [x_ref[5] for _ in time], '#ff7f0e', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("theta [rad]")
plt.legend(loc='lower right')

plt.subplot(3, 2, 6)
plt.step(time, resultsX[0][6, :], '#1f77b4', label="Q1")
plt.step(time, resultsX[1][6, :], '#2ca02c', label="Q2")
plt.step(time, resultsX[2][6, :], '#d62728', label="Q3")
plt.step(time, resultsX[3][6, :], '#9467bd', label="Q4")
plt.step(time, resultsX[4][6, :], '#8c564b', label="Q5")
plt.step(time, [x_ref[6] for _ in time], '#ff7f0e', linestyle='--', label="ref")
plt.xlabel("Time [s]")
plt.ylabel("psi [rad]")
plt.legend(loc='lower right')

plt.show()