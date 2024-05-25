# Compute the Xf: invariant constraint admissible set

from Model import drone_dynamics
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

def max_control_admissable_set(A, B, K, x_lim, u_up, u_lo):
    """
    Extension of algorithm 3.2 in the paper:
    "Linear Systems with State and Control Constraints: The Theory and Application of Maximal Output Admissible Sets"
    by Elmer G. Gilbert, and Kok Tin Tan

    Inputs:
    A: System matrix
    B: Control matrix
    K: Feedback gain for optimal LQR
    u_up: Upper control input constraints
    u_lo: Lower control input constraints
    x_lim: State constraints

    Outputs:
    H: Polyhedron representing the set of admissable states
    h: Polyhedron representing the set of admissable states
    such that Hx <= h represents the control invariant admissable set X_f
    """

    # Initialization
    f = np.concatenate((x_lim, x_lim))  # (2n, 1)    
    s = f.shape[0]                      # Number of constraints
    A_K = A - np.dot(B, K)             # A_K
    H = []
    h = []
    exit_flag = 0
    t = 0

    while not exit_flag:
        print(f"\tX_f generation, k = {t}")

        H.append(np.linalg.matrix_power(A_K, t))
        H.append(-np.linalg.matrix_power(A_K, t))
        h.extend(f)
        J = np.vstack([np.linalg.matrix_power(A_K, t+1), -np.linalg.matrix_power(A_K, t+1)])    
        opt_val = np.zeros(s)

        for i in range(s):
            c = -J[i, :]
            bounds = [(None, None)] * A.shape[1]  # No explicit bounds on state variables
            res = linprog(c, A_ub=np.vstack(H), b_ub=np.array(h), bounds=bounds, method='highs')

            if res.status == 3:  # If problem is unbounded
                fval = np.inf
            else:
                fval = res.fun

            opt_val[i] = -fval - f[i]

        # Check if solution is feasible
        if np.all(opt_val <= 0 - np.finfo(float).eps) and res.status != 3:
            exit_flag = 1
            print(f"\tDone! Needed {t} iterations")
            print(f"\tNumber of constraints: {len(H)}")
        else:
            t += 1

    H = np.array(H)
    h = np.array(h)
    
    return H, h

# Example usage
A = np.array([[1, 1], [0, 0.9]])
B = np.array([[1], [0.5]])
K = np.array([[1, 0.5]])
x_lim = np.array([5, 5])
u_up = 1
u_lo = -1
H, h = max_control_admissable_set(A, B, K, x_lim, u_up, u_lo)
print('H:', H, H.shape)    # (4, 2, 2)
print('h:', h, h.shape)

# x0 = [1, 1]
# in_set = np.all(np.dot(H, x0) <= h)

# if in_set:
#     print('x0 is in the set X_f')
# else:
#     print('x0 is NOT in the set X_f')


# drone_model = drone_dynamics.Quadrotor()

# # define simulation parameters
# dt = 0.1 
# t_final = 20.0 # seconds
# time_steps = int(t_final / dt)
# time = np.linspace(0, t_final, time_steps)

# A = drone_model.dsys.A
# B = drone_model.dsys.B
# C = drone_model.dsys.C
# K_lqr = drone_model.K

# print(A, B, C, K)  # (12, 12) (12, 4) (12, 12) (4, 12)