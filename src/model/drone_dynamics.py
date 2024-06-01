import casadi as ca
import numpy as np
import control as ctrl
import cvxpy as cp

class Quadrotor:
    def __init__(self, parms):
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
        self.K, self.P = self.make_dlqr_controller(parms=parms)

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

    def make_dlqr_controller(self, parms):
        x_operating = np.zeros((12, 1))
        u_operating = np.array([10, 0, 0, 0]).reshape((-1, 1))
        A, B, C, D = self.linearize(x_operating, u_operating)
        sys_continuous = ctrl.ss(A, B, C, D)
        self.csys = sys_continuous
        sys_discrete = ctrl.c2d(sys_continuous, self.dt, method='zoh')
        self.dsys = sys_discrete
        K, P, _ = ctrl.dlqr(self.dsys.A, self.dsys.B, parms["Q"], parms["R"])
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
        A_aug = np.vstack([np.hstack([self.dsys.A, Bd]), np.hstack([np.zeros((1, self.n_states)), np.eye(1)*0.9])]) # 13x13
        B_aug = np.vstack([self.dsys.B, np.zeros((1, self.n_inputs))])  # 13x4
        C_aug = np.hstack([self.dsys.C, Cd])    # 12x13
        D_aug = np.zeros((self.n_states, self.n_inputs))    # 12x4
        sysAug_discrete = ctrl.ss(A_aug, B_aug, C_aug, D_aug)
        self.dAugsys = sysAug_discrete

        # Check conditions (Observability Original System and Augmented System)
        # Construct test matrix [I-A -Bd; C Cd]
        test_matrix = np.vstack([np.hstack([np.eye(self.n_states) - self.dsys.A, -Bd]), np.hstack([self.dsys.C, Cd])])
        if (np.linalg.matrix_rank(ctrl.obsv(self.dsys.A, self.dsys.C)) == self.n_states) and (np.linalg.matrix_rank(test_matrix) == self.n_states + self.nd):
            print("Condition Fullfilled: System is observable and Augmented System is observable")
        else:
            print("ERROR! Not observable")
        
        # Check controlability of the augmented system
        if np.linalg.matrix_rank(ctrl.ctrb(self.dAugsys.A, self.dAugsys.B)) == self.dAugsys.A.shape[0]:
            print("The augmented system is controllable.")
        else:
            print("The augmented system is not controllable.")

        return A_aug, B_aug, C_aug, D_aug
    
    def Luenberger_observer(self, parms):
    # Build Luenberger Observer
        Q_aug = np.eye(13)
        Q_aug[:12, :12] = parms["Q"]
        R_aug = parms["R"]
        K_aug, P_aug, _ = ctrl.dlqr(self.dAugsys.A, self.dAugsys.B, Q_aug, R_aug) # 4*13
        CL_poles = np.linalg.eigvals(self.dAugsys.A - self.dAugsys.B @ K_aug)
        Observer_poles = CL_poles
        L = ctrl.place(self.dAugsys.A.T, self.dAugsys.C.T, Observer_poles).T # 13*12
        print(CL_poles, '\n' ,np.linalg.eigvals(self.dAugsys.A - L @ self.dAugsys.C), '\n', Observer_poles)

        

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


def is_stablizable(A, B):
    n = A.shape[0]
    eigenvalues = np.linalg.eigvals(A)
    unstable_eigenvalues = []
    for lambda_ in eigenvalues:
        if np.abs(lambda_) >= 1:
            unstable_eigenvalues.append(lambda_)
        controllable = True
        for lambda_ in unstable_eigenvalues:
            matrix = np.hstack((A - lambda_ * np.eye(n), B))
            if np.linalg.matrix_rank(matrix) < n:
                controllable = False
                break

        if controllable:
            print("The pair (A, B) is stabilizable.")
        else:
            print("The pair (A, B) is not stabilizable.")

if __name__ == "__main__":
    Q = np.eye(12)
    parms = {"Q": Q, "R": np.eye(4), "N": 10, "Qf": Q, "dynamic": True}
    model = Quadrotor(parms)

    # Augmented system
    d = [1]
    Bd = np.array([1,0,0,0,0,0,0,0,0,0,0,0]).reshape((12, 1)) # disturbance on x
    Cd = np.array([0,0,0,0,0,0,0,0,0,0,0,0]).reshape(12, 1) # measure only x
    A_aug, B_aug, C_aug, D_aug = model.augment_sys_disturbance(d, Bd, Cd)