import casadi as ca
import numpy as np
import control as ctrl
import cvxpy as cp

class HIV_model:
    def __init__(self):
        # System info
        self.n_states = 5 # x, y, z, phi, theta, psi, derivatives of before
        self.n_inputs = 1 # F, Tx, Ty, Tz
        self.x = ca.SX.sym('x', self.n_states)  # State variables
        self.u = ca.SX.sym('u', self.n_inputs)  # Input variable
        self.dt = 0.1

        # linearized state space
        self.A = np.zeros((5, 5))
        self.B = np.zeros((5, 1))
        self.C = np.eye(5) # assume full information feedback
        self.D = np.zeros((5, 1))

        x_operating = np.zeros((5, 1))
        u_operating = np.array([0])
        self.A, self.B, self.C, self.D = self.linearize(x_operating, u_operating)
    
    def build_x_dot(self):
        self.x_dot = ca.vertcat(self.x[1], -self.x[0] + self.u)

    def nonlinear_dyn(self):
        self.x = ca.SX.sym('x', self.n_states)  # State variables
        self.u = ca.SX.sym('u', self.n_inputs)  # Input variable

        # System parameters
        self.d = 0.1
        self.beta = 1
        self.a = 0.2
        self.p1 = 1
        self.p2 = 1
        self.c1 = 1
        self.c2 = 1
        self.b1 = 0.1
        self.b2 = 0.01
        self.lamb = 1
        self.q = 0.5
        self.eta = 0.9799
        self.h = 0.1

        # nonlinear dynamic 
        x_x, x_y, x_z1, x_z2, x_w = ca.vertsplit(self.x, 1)
        u = ca.vertsplit(self.u, 1)
        
        dx_dx = self.lamb - self.d * x_x - self.beta * (1 - self.eta * u) * x_x * x_y
        dx_dy = self.beta * (1 - self.eta * u) * x_x * x_y - self.a * x_y - self.p1 * x_z1 * x_y - self.p2 * x_z2 * x_y
        dx_z1 = self.c1 * x_z1 * x_y - self.b1 * x_z1
        dx_w = self.c2 * x_x * x_y * x_w - self.c2 * self.q * x_y * x_w - self.b2 * x_w
        dx_z2 = self.c2 * self.q * x_y * x_w - self.h * x_z2

        x_dot = ca.vertcat(dx_dx, dx_dy, dx_z1, dx_z2, dx_w)
        self.f = x_dot
        self.dynamics = ca.Function("HIV_dyn", [self.x, self.u], [self.x + self.dt * x_dot])
        
        jac_dyn_x = ca.jacobian(self.x + self.dt * x_dot, self.x)
        jac_dyn_u = ca.jacobian(self.x + self.dt * x_dot, self.u)
        self.jac_dyn_x = ca.Function("jac_dyn_x", [self.x, self.u], [jac_dyn_x])
        self.jac_dyn_u = ca.Function("jac_dyn_u", [self.x, self.u], [jac_dyn_u])

        print("The dynamics were built.")

    def linearize(self, x_operating, u_operating):
        A = ca.Function("A", [self.x, self.u], [ca.jacobian(self.f, self.x)])(x_operating, u_operating)
        B = ca.Function("B", [self.x, self.u], [ca.jacobian(self.f, self.u)])(x_operating, u_operating)
        C = np.eye(12)
        D = np.zeros((self.n_states, self.n_inputs))

        #make numerical matrices and cast into numpy array
        A = np.array(ca.DM(A))
        B = np.array(ca.DM(B))
        return A, B, C, D

    def compute_next_state(self, x, u):
        # x_next = x + self.dt * self.dyn_fun(x, u)
        x_next = self.dynamics(x, u)
        return x_next

if __name__ == "__main__":
    model = HIV_model()
    print(model.linearize(np.zeros((5, 1)), np.array([0])))
    """model.nonlinear_dyn()
    x = np.array([1, 1, 1, 1, 1])
    u = np.array([1])
    x_next = model.compute_next_state(x, u)
    print(x_next)"""