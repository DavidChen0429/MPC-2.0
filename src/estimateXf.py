# Compute the Xf: invariant constraint admissible set
# The maxiaml control invariant admissable set X_f is represented by Hx <= h

"""
Theory footnotes:

The maximal control invariant admissable set X_f is defined as the set of states x that can be steered to the origin while satisfying 
the constraints on the states and control inputs. The estimation of X_f uses the property that O_{infy} is dinitely determined if and 
only if O_t = O_{t+1} for some t. Based on the definition (2, 3) and Theorem 2.3 on the original paper, X_f can be estimated by solving
the following optimization problem:
"""

from model import drone_dynamics
import numpy as np
from scipy.optimize import linprog

def max_control_admissable_set(A, B, K, x_lim, u_lim):
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
    nx = len(x_lim)                                     # Number of states 12
    nu = len(u_lim)                                     # Number of control inputs 4
    f = np.concatenate((u_lim, u_lim, x_lim, x_lim))    # Constraints    32
    s = f.shape[0]                                      # Number of constraints 32
    A_K = A - B @ K                                     # A_K closed loop system matrix 
    H = np.empty((0, A.shape[1]))                       # (0, 12)
    h = np.empty(0)                                     # (0,)
    exit_flag = 0
    t = 0

    while not exit_flag:
        print(f"\tX_f generation, k = {t}")
        H = np.vstack([K @ np.linalg.matrix_power(A_K, t), 
                       -K @ np.linalg.matrix_power(A_K, t), 
                       np.linalg.matrix_power(A_K, t), 
                       -np.eye(nx) @ np.linalg.matrix_power(A_K, t), H])
        h = np.hstack([f, h])
        J = np.vstack([K @ np.linalg.matrix_power(A_K, t+1),
                       -K @ np.linalg.matrix_power(A_K, t+1),
                       np.linalg.matrix_power(A_K, t+1),
                       -np.eye(nx) @ np.linalg.matrix_power(A_K, t+1)])   
        opt_val = np.zeros(s)

        for i in range(s):
            res = linprog(-J[i, :], A_ub=np.array(H), b_ub=np.array(h), method='highs') # solve the opti problem
            if res.status == 2:     # If problem is infeasible
                fval = 0
            elif res.status == 3:   # If problem is unbounded
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

# Q = np.eye(12)
# parms = {"Q": Q, "R": np.eye(4), "N": 10, "Qf": Q, "dynamic": True}
# drone_model = drone_dynamics.Quadrotor(parms)
# A = drone_model.dsys.A  # (12, 12)
# B = drone_model.dsys.B  # (12, 4)
# C = drone_model.dsys.C  # (12, 12)
# K_lqr = drone_model.K   # (4, 12)

# x_lim = np.array([1000, 1000, 1000, np.pi/2, np.pi/2, 100, 2, 2, 2, 3*np.pi, 3*np.pi, 3*np.pi])
# u_lim = np.array([30, 1.4715, 1.4715, 0.0196])
# H, h = max_control_admissable_set(A, B, K_lqr, x_lim, u_lim)
# print('H:', H.shape, type(H))   
# print('h:', h.shape, type(h))

# x0 = [5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# if np.all(np.dot(H, x0) <= h):
#     print('x0 is in the set X_f')
# else:
#     print('x0 is NOT in the set X_f')