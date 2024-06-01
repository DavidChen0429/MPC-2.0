from model import drone_dynamics
from planner import trajectory_generation
import matplotlib.pyplot as plt
import numpy as np

drone_model = drone_dynamics.Quadrotor()

# define simulation parameters
dt = 0.1 
t_final = 20.0 # seconds
time_steps = int(t_final / dt)
time = np.linspace(0, t_final, time_steps)

# Augment the system with a constant disturbance term d
d = [1]
Bd = np.array([1,0,0,0,0,0,0,0,0,0,0,0]).reshape((12, 1)) # disturbance on x
Cd = np.array([1,0,0,0,0,0,0,0,0,0,0,0]).reshape(12, 1) # measure only x
A_aug, B_aug, C_aug, D_aug = drone_model.augment_sys_disturbance(d, Bd, Cd)
drone_model.Luenberger_observer()