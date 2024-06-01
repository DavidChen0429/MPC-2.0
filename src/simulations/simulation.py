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
x_ref, u_ref = drone_model.ots(x_ref[:3, 0], parms)

for k in range(time_steps-1):
    x_current = x_bag[:, k]
    #x_ref_current = x_ref[:, k]
    x_next, u = drone_model.step(x_current, x_ref, u_ref, cont_type="MPC", sim_system="linear", parms=parms)
    x_bag[:, k+1] = x_next
    u_bag[:, k+1] = u

# plot the results
plt.figure()
plt.step(time, x_bag[0, :], '#1f77b4', label="x_x")
plt.step(time, [x_ref[0] for _ in time], '#1f77b4', linestyle='--', label="x_x ref")
plt.xlabel("Time [s]")
plt.legend()
plt.show()