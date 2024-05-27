# Simulate the LQR controller for hovering the drone at a single point
from Model import drone_dynamics
from Planner import trajectory_generation
#from Planner import SafeFlightPolytope
import matplotlib.pyplot as plt
import numpy as np

drone_model = drone_dynamics.Quadrotor()

# define simulation parameters
dt = 0.1 
t_final = 20.0 # seconds
time_steps = int(t_final / dt)
time = np.linspace(0, t_final, time_steps)

# initialize state and input arrays
x_bag, u_bag = drone_model.get_ss_bag_vectors(time_steps) #vnp.zeros((self.n_states, N))
x0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
u0 = np.array([0, 0, 0, 0])
x_bag[:, 0] = x0
u_bag[:, 0] = u0
x_ref = trajectory_generation.hover_traj(time_steps)  # reference trajectory (state)

# # === MPC definition
# # ===== Cost function
# # ====== Stage cost
# Q=np.eye(12)
# R=np.eye(4)
# # ====== Terminal cost
# P = drone_model.P # Solution of the discrete time algebraic Riccati equation
# # ===== Prediction Horizon
# N=10
# # ===== Constraints
# # ======= Input contraints
# u_max = np.array([30, 1.4715, 1.4715, 0.0196])
# u_min = np.array([-10, -1.4715, -1.4715, -0.0196])
# # ======= State constraints
# x_max = np.array([np.inf, np.inf, np.inf, np.pi/2, np.pi/2, np.inf, 2, 2, 2, np.inf, np.inf, np.inf])
# x_min = np.array([-np.inf, -np.inf, -np.inf, -np.pi/2, -np.pi/2, -np.inf, -2, -2, -2, -np.inf, -np.inf, -np.inf])

# Simulation
for k in range(time_steps-1):
    x_current = x_bag[:, k]
    x_ref_current = x_ref[:, k]
    x_next, u = drone_model.step(x_current, x_ref_current, cont_type="MPC")
    x_bag[:, k+1] = x_next
    u_bag[:, k+1] = u


# Visualization
plt.figure()
plt.step(time, x_bag[0, :], '#1f77b4', label="x_x")
plt.step(time, x_ref[0,:], '#1f77b4', linestyle='--', label="x_x ref")
plt.step(time, x_bag[1, :], '#ff7f0e', label="x_y")
plt.step(time, x_ref[1,:], '#ff7f0e', linestyle='--', label="x_y ref")
plt.step(time, x_bag[2, :], '#2ca02c', label="x_z")
plt.step(time, x_ref[2,:], '#2ca02c', linestyle='--', label="x_z ref")
#plt.step(x_bag[5, :], '#d62728', label="x_psi")
plt.xlabel("Time [s]")
plt.title("Constrainted LQR Hovering Simulation")
plt.legend()
plt.show()


plt.figure()
plt.step(time, u_bag[0, :], '#1f77b4', label="F")
plt.step(time, u_bag[1, :], '#ff7f0e', label="Tx")
plt.step(time,u_bag[2, :], '#2ca02c', label="Ty")
plt.step(time, u_bag[3, :], '#d62728', label="Tz")
plt.xlabel("Time [s]")
plt.title("Constrainted LQR Control Inputs")
plt.legend()
plt.show()