# Simulate the LQR controller for hovering the drone at a single point
from Model import drone_dynamics
from Planner import trajectory_generation
import matplotlib.pyplot as plt
import numpy as np
import control as ctrl

drone_model = drone_dynamics.Quadrotor()

# define simulation parameters
dt = 0.1 
t_final = 20.0 # seconds
time_steps = int(t_final / dt)
time = np.linspace(0, t_final, time_steps)

# LQR gain
x_operating = np.zeros((12, 1))
u_operating = np.array([10, 0, 0, 0]).reshape((-1, 1))
A, B, C, D = drone_model.linearize(x_operating, u_operating)
lqr_K = drone_model.K
Q = np.eye(12)
R = np.eye(4)
sys_continuous = ctrl.ss(A, B, C, D)
sys_discrete = ctrl.c2d(sys_continuous, dt, method='zoh')
lqr_K, _, _ = ctrl.dlqr(sys_discrete.A, sys_discrete.B, Q, R)

# initialize state and input arrays
x_bag, u_bag = drone_model.get_ss_bag_vectors(time_steps) #vnp.zeros((self.n_states, N))
x0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
u0 = np.array([0, 0, 0, 0])
x_bag[:, 0] = x0
u_bag[:, 0] = u0
x_ref = trajectory_generation.hover_traj(time_steps)  # reference trajectory (state)

# simulate the drone dynamics
for k in range(time_steps-1):
    x_current = x_bag[:, k]
    x_ref_current = x_ref[:, k]
    x_next, u = drone_model.step(x_current, x_ref_current, cont_type="LQR")
    x_bag[:, k+1] = x_next
    u_bag[:, k+1] = u


# Visualization
plt.figure()
plt.step(x_bag[0, :], '#1f77b4', label="x_x")
plt.step(x_ref[0,:], '#1f77b4', linestyle='--', label="x_x ref")
plt.step(x_bag[1, :], '#ff7f0e', label="x_y")
plt.step(x_ref[1,:], '#ff7f0e', linestyle='--', label="x_y ref")
plt.step(x_bag[2, :], '#2ca02c', label="x_z")
plt.step(x_ref[2,:], '#2ca02c', linestyle='--', label="x_z ref")
#plt.step(x_bag[5, :], '#d62728', label="x_psi")
plt.title("Constrainted LQR Hovering Simulation")
plt.legend()
plt.show()


plt.figure()
plt.step(u_bag[0, :], '#1f77b4', label="F")
plt.step(u_bag[1, :], '#ff7f0e', label="Tx")
plt.step(u_bag[2, :], '#2ca02c', label="Ty")
plt.step(u_bag[3, :], '#d62728', label="Tz")
plt.title("Constrainted LQR Control Inputs")
plt.legend()
plt.show()