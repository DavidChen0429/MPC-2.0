import casadi as ca
import numpy as np
import control as ctrl
import cvxpy as cp
#import pybullet as p



class Quadrotor:
    def __init__(self):
        self.n_states = 12 # x, y, z, phi, theta, psi, derivatives of before
        self.n_inputs = 4 # F, Tx, Ty, Tz
        self.x = ca.SX.sym('x', self.n_states)  # State variables
        self.u = ca.SX.sym('u', self.n_inputs)  # Input variable
        self.dt = 0.1

        self.build_dyn()  # build the dynamical model

        # linearized state space
        self.A = np.zeros((12, 12))
        self.B = np.zeros((12, 4))
        self.C = np.eye(12) # assume full information feedback
        self.D = np.zeros((12, 4))

        #
        self.dsys = 0
        self.csys = 0

        # build dynamics etc
        x_operating = np.zeros((12, 1))
        u_operating = np.array([10, 0, 0, 0]).reshape((-1, 1))  # hovering (mg 0 0 0)
        self.A, self.B, self.C, self.D = self.linearize(x_operating, u_operating)
        self.K, self.P = self.make_dlqr_controller()

    def build_x_dot(self):
        self.x_dot = ca.vertcat(self.x[1], -self.x[0] + self.u)

    def build_dyn(self):
        #self.dyn = ca.Function("bicycle_dynamics_dt_fun", [self.x, self.u], [self.x + self.dt * self.x_dot])
        self.x = ca.SX.sym("x", self.n_states)
        self.u = ca.SX.sym("u", self.n_inputs)

        m = 1  # kg
        g = 9.80665  # m/s^2
        Ix, Iy, Iz = 0.11, 0.11, 0.04  # kg m^2
        l = 0.2  # m (this drops out when controlling via torques)

        # non linear dynamics
        x_x, x_y, x_z, x_phi, x_theta, x_psi, x_dx, x_dy, x_dz, x_dphi, x_dtheta, x_dpsi = ca.vertsplit(self.x, 1)
        u_F, u_Tx, u_Ty, u_Tz = ca.vertsplit(self.u, 1)

        dx_x = x_dx
        dx_y = x_dy
        dx_z = x_dz
        dx_phi = x_dphi
        dx_theta = x_dtheta
        dx_psi = x_dpsi
        dx_dx = u_F/m * (ca.cos(x_phi)*ca.sin(x_theta)*ca.cos(x_psi) + ca.sin(x_phi)*ca.sin(x_psi))
        dx_dy = u_F/m * (ca.cos(x_phi)*ca.sin(x_theta)*ca.sin(x_psi)+ca.sin(x_phi)*ca.cos(x_psi))
        dx_dz = u_F/m * ca.cos(x_phi) * ca.cos(x_theta) - g
        dx_dphi = 1/Ix * (u_Tx + x_dtheta * x_dpsi*(Iy - Iz))
        dx_dtheta = 1/Iy * (u_Ty + x_dpsi*x_dphi*(Iz - Ix))
        dx_dpsi = 1/Iz * (u_Tz + x_dphi*x_dtheta*(Ix-Iy))

        x_dot = ca.vertcat(dx_x, dx_y, dx_z, dx_phi, dx_theta, dx_psi,
                           dx_dx, dx_dy, dx_dz, dx_dphi, dx_dtheta, dx_dpsi)
        self.f = x_dot
        self.dynamics = ca.Function("quadrotor_dyn", [self.x, self.u], [self.x + self.dt * x_dot])

        # this is continuous or discrete time?
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

    def make_dlqr_controller(self):
        x_operating = np.zeros((12, 1))
        u_operating = np.array([10, 0, 0, 0]).reshape((-1, 1))
        A, B, C, D = self.linearize(x_operating, u_operating)
        Q = np.eye(12)
        R = np.eye(4)
        sys_continuous = ctrl.ss(A, B, C, D)
        self.csys = sys_continuous
        sys_discrete = ctrl.c2d(sys_continuous, self.dt, method='zoh')
        self.dsys = sys_discrete
        K, P, _ = ctrl.dlqr(self.dsys.A, self.dsys.B, Q, R)
        return K, P

    def get_ss_bag_vectors(self, N):
        """N is the number of simulation steps, thus number of concatinated x vectors"""
        x_bag = np.zeros((self.n_states, N))
        u_bag = np.zeros((self.n_inputs, N))
        return x_bag, u_bag
    
    def augment_sys_disturbance(self, d, Bd, Cd):
        # Augment the system with a constant disturbance term d
        # State becomes [x, d]
        # A matrix becomes [A Bd; 0 1]
        # B matrix becomes [B; 0]
        # C matrix becomes [C Cd]
        self.nd = len(d)
        A_aug = np.vstack([np.hstack([self.dsys.A, Bd]), np.hstack([np.zeros((1, self.n_states)), np.eye(1)])]) # 13x13
        B_aug = np.vstack([self.dsys.B, np.zeros((1, self.n_inputs))])  # 13x4
        C_aug = np.hstack([self.dsys.C, Cd])    # 12x13
        D_aug = np.zeros((self.n_states, self.n_inputs))    # 12x4
        print(A_aug.shape, B_aug.shape, C_aug.shape, D_aug.shape)
        sysAug_continuous = ctrl.ss(A_aug, B_aug, C_aug, D_aug)
        sysAug_discrete = ctrl.c2d(sysAug_continuous, self.dt, method='zoh')
        self.dAugsys = sysAug_discrete

        # Check conditions (Observability Original System and Augmented System)
        # Construct test matrix [I-A -Bd; C Cd]
        test_matrix = np.vstack([np.hstack([np.eye(self.n_states) - self.dsys.A, -Bd]), np.hstack([self.dsys.C, Cd])])
        if (np.linalg.matrix_rank(ctrl.obsv(model.A, model.C)) == model.n_states) and  (np.linalg.matrix_rank(test_matrix) == self.n_states + self.nd):
            print("Condition Fullfilled: System is observable and Augmented System is observable")
        else:
            print("You'er Fucked")
        return A_aug, B_aug, C_aug, D_aug
    
    def Luenberger_observer(self):
    # Build Luenberger Observer
        print(np.linalg.eigvals(model.dAugsys.A))   # Poles for original systems


    def mpc(self, x0, x_goal, Q=np.eye(12), R=np.eye(4), N=10, Qf=np.eye(12), dynamic=False):
        x = cp.Variable((12, N))
        u = cp.Variable((4, N))
        cost = sum(cp.quad_form(x[:, i] - x_goal.T, Q) + cp.quad_form(u[:, i], R) for i in range(N))  # stage cost
        cost += cp.quad_form(x[:, N-1] - x_goal.T, Qf)  # terminal cost
        obj = cp.Minimize(cost)

        cons = [x[:, 0] == x0]
        for i in range(1, N):
            cons += [x[:, i] == self.dsys.A @ x[:, i-1] + self.dsys.B @ u[:, i-1]]

        u_max = np.array([30, 1.4715, 1.4715, 0.0196])
        u_min = np.array([-10, -1.4715, -1.4715, -0.0196])

        if dynamic==True:
            cons += [u <= np.array([u_max]).T, u >= np.array([u_min]).T]                            # Input limit
            cons += [x[3:5, :] <= 0.5*np.pi*np.ones((2,1)), x[3:5,:] >= -0.5*np.pi*np.ones((2, 1))]   # Pitch and roll limit
            cons += [x[6:9, :] <= 2*np.ones((3,1)), x[6:9,:] >= -2*np.ones((3, 1))]                   # Speed limit
            cons += [x[9:12, :] <= 3*np.pi*np.ones((3,1)), x[9:12,:] >= -3*np.pi*np.ones((3, 1))]     # Angular speed limit

        prob = cp.Problem(obj, cons)
        prob.solve(verbose=False)

        return u.value[:, 0], x.value

    def step(self, x, x_ref, cont_type="LQR", sim_system="linear", parms=None):
        # discrete step
        if cont_type == "LQR":
            u = self.K @ (x_ref - x)
        elif cont_type == "c-LQR":  # constrained LQR
            u = self.K @ (x_ref - x)
            u_max = np.array([30, 1.4715, 1.4715, 0.0196])
            u_min = np.array([-10, -1.4715, -1.4715, -0.0196])
            u = np.clip(u, u_min, u_max)  # saturation function
        elif cont_type == "MPC":
            u, _ = self.mpc(x, x_ref, Q=parms["Q"], R=parms["R"], N=parms["N"], Qf=parms["Qf"], dynamic=parms["dynamic"])
        else:
            raise ValueError("Specified controller type not known.")

        if sim_system == "linear":
            x_next = self.dsys.A @ x + self.dsys.B @ u
        elif sim_system == "non-linear":
            raise NotImplementedError
        else:
            raise ValueError("Sim system argument must be linear or non-linear.")
        return x_next, u


if __name__ == "__main__":
    model = Quadrotor()
    print("A:", model.A, "\nB:", model.B, "\nC:", model.C, "\nD:", model.D)
    print("K:", model.K.shape)
    print("P:", model.P.shape)
    print("Eigenvalue of P:", np.linalg.eigvals(model.P))  # unique positive semi-definite solution of the Riccati equation

    # Check controlability
    if np.linalg.matrix_rank(ctrl.ctrb(model.A, model.B)) == model.n_states:
        print("System is controllable")
    else:
        print("System is not controllable")

    # Check observability
    if np.linalg.matrix_rank(ctrl.obsv(model.A, model.C)) == model.n_states:
        print("System is observable")
    else:
        print("System is not observable")

    # Check discrete system poles
    poles = np.linalg.eigvals(model.dsys.A)
    print("Poles of discrete system:", poles)

    # # Luenberger observer
    # L = ctrl.place(model.A.T, model.C.T, poles).T
    # print("L:", L)
    # print("Eigenvalues of A-LC:", np.linalg.eigvals(model.A - L @ model.C))

    # Augmented system
    d = [1]
    Bd = np.array([1,0,0,0,0,0,0,0,0,0,0,0]).reshape((12, 1)) # disturbance on x
    Cd = np.array([1,0,0,0,0,0,0,0,0,0,0,0]).reshape(12, 1) # measure only x
    A_aug, B_aug, C_aug, D_aug = model.augment_sys_disturbance(d, Bd, Cd)
    model.Luenberger_observer()