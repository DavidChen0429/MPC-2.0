from src.model import drone_dynamics
from src.planner import trajectory_generation
import matplotlib.pyplot as plt
import numpy as np

# define simulation parameters
dt = 0.1
t_final = 15.0  # seconds
time_steps = int(t_final / dt)
time = np.linspace(0, t_final, time_steps)

Q = np.eye(12)
parms = {"Q": Q, "R": np.eye(4), "N": 10, "Qf": Q, "dynamic": True}
drone_model = drone_dynamics.Quadrotor(parms)

# initialize state and input arrays
x_bag, u_bag = drone_model.get_ss_bag_vectors(time_steps) #vnp.zeros((self.n_states, N))
x_bag_nl, u_bag_nl = drone_model.get_ss_bag_vectors(time_steps)
x0 = np.array([0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0])
u0 = np.array([0, 0, 0, 0])
x_bag[:, 0] = x0
u_bag[:, 0] = u0
x_bag_nl[:, 0] = x0
u_bag_nl[:, 0] = u0
x_ref = trajectory_generation.hover_traj(time_steps)  # reference trajectory (state)

for k in range(time_steps-1):
    x_ref_current = x_ref[:, k]
    # linear
    x = x_bag[:, k]
    x_next, u = drone_model.step(x, x_ref_current, cont_type="MPC", sim_system="linear", parms=parms)
    x_bag[:, k + 1] = x_next
    u_bag[:, k + 1] = u

    # non-linear
    x = x_bag_nl[:, k]
    x_next, u = drone_model.step(x, x_ref_current, cont_type="MPC", sim_system="non-linear", parms=parms)
    x_bag_nl[:, k+1] = x_next
    u_bag_nl[:, k+1] = u

plt.plot(time, x_bag[2, :], label="linear")
plt.plot(time, x_bag_nl[2, :], label="non-linear")
plt.plot(time, x_ref[2, :], label="reference")
plt.legend()
plt.show()