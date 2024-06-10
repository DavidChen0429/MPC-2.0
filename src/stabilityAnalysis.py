from model import drone_dynamics
from planner import trajectory_generation
from estimateXf import max_control_admissable_set
import matplotlib.pyplot as plt
import numpy as np

# define simulation parameters
dt = 0.1
t_final = 20.0  # seconds
time_steps = int(t_final / dt)
time = np.linspace(0, t_final, time_steps)

Q = np.eye(12)
# Q = np.block([[np.eye(3) * 0.5, np.zeros((3, 9))],
#               [np.zeros((9, 3)), np.eye(9)]])
parms1 = {"Q": Q, "R": np.eye(4), "N": 10, "Qf": Q, "dynamic": True}
drone_model1 = drone_dynamics.Quadrotor(parms1)
K, P = drone_model1.make_dlqr_controller(parms1)
parms = {"Q": Q, "R": np.eye(4), "N": 10, "Qf": P, "dynamic": True}
drone_model = drone_dynamics.Quadrotor(parms)

################### Xf generation ###################
x_lim = np.array([1000, 1000, 1000, np.pi/2, np.pi/2, 100, 2, 2, 2, 3*np.pi, 3*np.pi, 3*np.pi])
u_lim = np.array([30, 1.4715, 1.4715, 0.0196])
H, h = max_control_admissable_set(drone_model.dsys.A, drone_model.dsys.B, drone_model.K, x_lim, u_lim)
print('H:', H.shape, type(H))   
print('h:', h.shape, type(h))

x0_tested = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # [-20, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0]
if np.all(np.dot(H, x0_tested) <= h):
    print('x0 is in the set X_f')
else:
    print('x0 is NOT in the set X_f')

x_bagMPC, u_bagMPC = drone_model.get_ss_bag_vectors(time_steps) # np.zeros((self.n_states, N))
x_bagLQR, u_bagLQR = drone_model.get_ss_bag_vectors(time_steps) # np.zeros((self.n_states, N))
x0 = np.array(x0_tested)
u0 = np.array([0, 0, 0, 0])
x_bagMPC[:, 0] = x0
u_bagMPC[:, 0] = u0
x_bagLQR[:, 0] = x0
u_bagLQR[:, 0] = u0
x_ref = trajectory_generation.hover_traj(time_steps)  # reference trajectory (state)
u_ref =  np.zeros((4, time_steps)) # reference trajectory (input)

# MPC vs LQR simulation
for k in range(time_steps-1):
    x_ref_current = x_ref[:, k]
    u_ref_current = u_ref[:, k]
    x_currentMPC = x_bagMPC[:, k]
    x_currentLQR = x_bagLQR[:, k]

    # # MPC
    x_nextMPC, uMPC = drone_model.step(x_currentMPC, x_ref_current, u_ref_current, cont_type="MPC", sim_system="linear", parms=parms)
    x_bagMPC[:, k + 1] = x_nextMPC
    u_bagMPC[:, k + 1] = uMPC

    # Real-MPC
    # uMPC = drone_model.real_mpc(x_currentMPC, x_ref_current, u_ref_current, Q=parms["Q"], R=parms["R"], N=parms["N"], Qf=parms["Qf"], dynamic=True)
    # x_nextMPC = drone_model.dsys.A @ x_currentLQR + drone_model.dsys.B @ uMPC
    # x_bagMPC[:, k + 1] = x_nextMPC
    # u_bagMPC[:, k + 1] = uMPC

    # Finite Horizon Unconstrained LQR
    uLQR, _ = drone_model.Nlqr(x_currentLQR, x_ref_current, u_ref_current, Q=parms["Q"], R=parms["R"], N=parms["N"], Qf=parms["Qf"], dynamic=False)
    x_nextLQR = drone_model.dsys.A @ x_currentLQR + drone_model.dsys.B @ uLQR
    x_bagLQR[:, k + 1] = x_nextLQR
    u_bagLQR[:, k + 1] = uLQR

plt.figure()
plt.subplot(2, 2, 1)
plt.step(time, x_bagMPC[0, :], '#1f77b4', label="x")
plt.step(time, x_bagMPC[1, :], '#ff7f0e', label="y")
plt.step(time, x_bagMPC[2, :], '#2ca02c', label="z")
plt.step(time, x_ref[0, :], '#1f77b4', linestyle='--', label="x_ref")
plt.step(time, x_ref[1, :], '#ff7f0e', linestyle='--', label="y_ref")
plt.step(time, x_ref[2, :], '#2ca02c', linestyle='--', label="z_ref")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("MPC")
plt.legend()

plt.subplot(2, 2, 2)
plt.step(time, x_bagLQR[0, :], '#1f77b4', label="x")
plt.step(time, x_bagLQR[1, :], '#ff7f0e', label="y")
plt.step(time, x_bagLQR[2, :], '#2ca02c', label="z")
plt.step(time, x_ref[0, :], '#1f77b4', linestyle='--', label="x_ref")
plt.step(time, x_ref[1, :], '#ff7f0e', linestyle='--', label="y_ref")
plt.step(time, x_ref[2, :], '#2ca02c', linestyle='--', label="z_ref")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.title("LQR")
plt.legend()

plt.subplot(2, 2, 3)
plt.step(time, x_bagMPC[3, :], '#1f77b4', label="phi")
plt.step(time, x_bagMPC[4, :], '#ff7f0e', label="theta")
plt.step(time, x_bagMPC[5, :], '#2ca02c', label="psi")
plt.step(time, x_ref[3, :], '#1f77b4', linestyle='--', label="phi_ref")
plt.step(time, x_ref[4, :], '#ff7f0e', linestyle='--', label="theta_ref")
plt.step(time, x_ref[5, :], '#2ca02c', linestyle='--', label="psi_ref")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.legend()

plt.subplot(2, 2, 4)
plt.step(time, x_bagLQR[3, :], '#1f77b4', label="phi")
plt.step(time, x_bagLQR[4, :], '#ff7f0e', label="theta")
plt.step(time, x_bagLQR[5, :], '#2ca02c', label="psi")
plt.step(time, x_ref[3, :], '#1f77b4', linestyle='--', label="phi_ref")
plt.step(time, x_ref[4, :], '#ff7f0e', linestyle='--', label="theta_ref")
plt.step(time, x_ref[5, :], '#2ca02c', linestyle='--', label="psi_ref")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.legend()

plt.show()

# do optimal target selection
#x_ref, u_ref = drone_model.ots(x_ref[:3, 0], parms)

################### Lyapunov Decrease ###################
# x_bag, u_bag = drone_model.get_ss_bag_vectors(time_steps) # np.zeros((self.n_states, N))
# x0 = np.array(x0_tested)
# u0 = np.array([0, 0, 0, 0])
# x_bag[:, 0] = x0
# u_bag[:, 0] = u0

# # Cost
# l_xu = np.zeros(time_steps) # stage cost optimal
# Vf_x = np.zeros(time_steps) # terminal cost optimal
# Vf_fxu = np.zeros(time_steps)   # terminal cost optimal next state
# A = drone_model.dsys.A
# B = drone_model.dsys.B

# for k in range(time_steps-1):
#     x_ref_current = x_ref[:, k]
#     u_ref_current = u_ref[:, k]
#     x_current = x_bag[:, k]
#     uMPC, xMPC = drone_model.mpc(x_current, x_ref_current, u_ref_current, Q=parms["Q"], R=parms["R"], N=parms["N"], Qf=parms["Qf"], dynamic=parms["dynamic"])
#     x_next = drone_model.dsys.A @ x_current + drone_model.dsys.B @ uMPC
#     buffer = xMPC[:, -1]
#     l_xu[k] = (buffer-x_ref_current).T @ parms['Q'] @ (buffer-x_ref_current) + (uMPC-u_ref_current).T @ parms['R'] @ (uMPC-u_ref_current)
#     x_bag[:, k+1] = x_next
#     u_bag[:, k+1] = uMPC

# # Calculate cost
# for k in range(time_steps-1):
#     x_next_optimal = A @ x_bag[:, k] + B @ u_bag[:, k]
#     Vf_x[k] = (x_bag[:,k]-x_ref[:,k]).T @ P @ (x_bag[:,k]-x_ref[:,k])
#     Vf_fxu[k] = (x_next_optimal-x_ref[:,k+1]).T @ P @ (x_next_optimal-x_ref[:,k+1])

# # Show Lyanpunov decrease
# plt.figure()
# plt.step(np.linspace(0, dt * len(l_xu[2:-1]), len(l_xu[2:-1])), -1 * l_xu[2:-1], '#1f77b4', label="-l(x,u)")
# plt.step(np.linspace(0, dt * len(l_xu[2:-1]), len(l_xu[2:-1])), Vf_fxu[2:-1] - Vf_x[2:-1], '#d62728', label="Vf(f(x,u))-Vf(x)")
# plt.xlabel("Time [s]")
# plt.title("Lyapunov Decrease")
# plt.legend()
# plt.show()