from model import drone_dynamics
from planner import trajectory_generation
import matplotlib.pyplot as plt
import numpy as np


# define simulation parameters
dt = 0.4
t_final = 20.0  # seconds
time_steps = int(t_final / dt)
time = np.linspace(0, t_final, time_steps)

Q = np.eye(12)
parms = {"Q": Q, "R": np.eye(4), "N": 10, "Qf": Q, "dynamic": True}
drone_model = drone_dynamics.Quadrotor(parms)
K, P = drone_model.make_dlqr_controller(parms)

# initialize state and input arrays
x_bag, u_bag = drone_model.get_ss_bag_vectors(time_steps) # np.zeros((self.n_states, N))
x0 = np.array([5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
u0 = np.array([0, 0, 0, 0])
x_bag[:, 0] = x0
u_bag[:, 0] = u0
x_ref = trajectory_generation.hover_traj(time_steps)  # reference trajectory (state)
u_ref =  np.zeros((4, time_steps)) # reference trajectory (input)

# do optimal target selection
#x_ref, u_ref = drone_model.ots(x_ref[:3, 0], parms)

# Cost
l_xu = np.zeros(time_steps) # stage cost optimal
Vf_x = np.zeros(time_steps) # terminal cost optimal
Vf_fxu = np.zeros(time_steps)   # terminal cost optimal next state
A = drone_model.dsys.A
B = drone_model.dsys.B

for k in range(time_steps-1):
    x_ref_current = x_ref[:, k]
    u_ref_current = u_ref[:, k]
    x_current = x_bag[:, k]
    u, xMPC = drone_model.mpc(x_current, x_ref_current, u_ref_current, Q=parms["Q"], R=parms["R"], N=parms["N"], Qf=parms["Qf"], dynamic=parms["dynamic"])
    x_next = drone_model.dsys.A @ x_current + drone_model.dsys.B @ u
    buffer = xMPC[:, -1]
    l_xu[k] = (buffer-x_ref_current).T @ parms['Q'] @ (buffer-x_ref_current) + (u-u_ref_current).T @ parms['R'] @ (u-u_ref_current)
    x_bag[:, k+1] = x_next
    u_bag[:, k+1] = u

# Calculate cost
for k in range(time_steps-1):
    Vf_x[k] = (x_bag[:,k]-x_ref[:,k]).T @ P @ (x_bag[:,k]-x_ref[:,k])
    x_next_optimal = A @ x_bag[:, k] + B @ u_bag[:, k]
    Vf_fxu[k] = (x_next_optimal-x_ref[:,k]).T @ P @ (x_next_optimal-x_ref[:,k])

# Show Lyanpunov decrease
plt.figure()
plt.step(np.linspace(0, dt * len(l_xu[2:-1]), len(l_xu[2:-1])), -1 * l_xu[2:-1], '#1f77b4', label="-l(x,u)")
plt.step(np.linspace(0, dt * len(l_xu[2:-1]), len(l_xu[2:-1])), Vf_fxu[2:-1] - Vf_x[2:-1], '#d62728', label="V_f(f(x,u))-V_f(x)")
plt.xlabel("Time [s]")
plt.title("Lyapunov Decrease")
plt.legend()
plt.show()