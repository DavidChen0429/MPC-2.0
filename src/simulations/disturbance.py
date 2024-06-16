from src.model.drone_dynamics import Observer, Quadrotor
import matplotlib.pyplot as plt
import numpy as np

# define simulation parameters
dt = 0.1
t_final = 10. # seconds
time_steps = int(t_final / dt)
time = np.linspace(0, t_final, time_steps)

Q = np.eye(12)
parms = {"Q": Q, "R": np.eye(4), "N": 10, "Qf": Q, "dynamic": True}
x_0 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
drone_model = Quadrotor(x_0)
drone_model.initiate(parms)

# design the observer
xhat_0 = np.zeros(13)
Bd = np.array([1,0,0,0,0,0,0,0,0,0,0,0]).reshape((12, 1)) # TODO: disturbance on x
Cd = np.array([0,0,0,0,0,0,0,0,0,0,0,0]).reshape(12, 1) # measure only x
#Cd = np.array([0,0,0]).reshape(3, 1) # measure only x
observer = Observer(drone_model.dsys, xhat_0, Bd, Cd, parms)

# Test the observer
u_0 = np.array([0, 0, 0, 0])
x, u, xhat = drone_model.get_ss_bag_vectors(time_steps, x_0, u_0, xhat_0)

x_x_ref = np.zeros(time_steps)
x_y_ref = np.zeros(time_steps)
x_z_ref = np.ones(time_steps) * 10
x_ref = np.vstack([x_x_ref, x_y_ref, x_z_ref, np.zeros((9, time_steps))])


for k in range(time_steps - 1):
    u[:, k] = drone_model.K @ (x_ref[:, k] - xhat[:12, k])
    y_k = drone_model.dsys.C @ x[:, k]
    xhat[:, k + 1] = observer.step(u[:, k], y_k)
    x[:, k + 1] = drone_model.step(u[:, k], sim_system="linear")


# Visualize the results
plt.step(time, x[0, :], label="x")
plt.step(time, xhat[0, :], label="xhat")
plt.step(time, xhat[12, :], label="dhat")
plt.xlabel("Time [s]")
plt.legend()
plt.show()