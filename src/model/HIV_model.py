import casadi as ca
import numpy as np
import control as ctrl
import cvxpy as cp

class HIV_model:
    def __init__(self):
        self.n_states = 5 # x, y, z1, w, z2 
        self.n_inputs = 1 # dose of drug
        self.x = ca.SX.sym('x', self.n_states)  # State variables
        self.u = ca.SX.sym('u', self.n_inputs)  # Input variable
        self.dt = 0.1

        self.build_dyn()  # build the dynamical model

        # linearized state space
        self.A = np.zeros((5, 5))
        self.B = np.zeros((5, 1))
        self.C = np.eye(5) # assume full information feedback, need to be changed for observer
        self.D = np.zeros((5, 1))

        self.dsys = 0
        self.csys = 0

        # build dynamics etc
        x_operating = np.zeros((5, 1))
        u_operating = np.array([0])
        self.A, self.B, self.C, self.D = self.linearize(x_operating, u_operating)
        self.K = self.make_dlqr_controller()
    
    def build_x_dot(self):
        self.x_dot = ca.vertcat(self.x[1], -self.x[0] + self.u)

    def build_dyn(self):
        self.x = ca.SX.sym('x', self.n_states)  # State variables
        self.u = ca.SX.sym('u', self.n_inputs)  # Input variable

        # System parameters (check overleaf report for more info)
        d = 0.1
        beta = 1.0
        a = 0.2
        p1, p2 = 1.0, 1.0
        c1, c2 = 1.0, 1.0
        b1, b2 = 0.1, 0.01
        lamb = 1.0
        q = 0.5
        eta = 0.9799
        h = 0.1

        # nonlinear dynamic 
        x_x, x_y, x_z1, x_z2, x_w = ca.vertsplit(self.x, 1)
        u_therapy = self.u
        
        dx_x = lamb - d*x_x - beta*(1 - eta*u_therapy)*x_x*x_y
        dx_y = beta * (1 - eta*u_therapy)*x_x*x_y - a*x_y - p1*x_z1*x_y - p2*x_z2*x_y
        dx_z1 = c1*x_z1*x_y - b1*x_z1
        dx_w = c2*x_x*x_y*x_w - c2*q*x_y*x_w - b2*x_w
        dx_z2 = c2*q*x_y*x_w - h*x_z2

        x_dot = ca.vertcat(dx_x, dx_y, dx_z1, dx_z2, dx_w)
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
        C = np.eye(5)
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
        x_operating = np.zeros((5, 1))
        u_operating = np.array([1.0]).reshape((1, 1))
        A, B, C, D = self.linearize(x_operating, u_operating)
        Q = np.eye(5)
        R = np.eye(1)
        sys_continuous = ctrl.ss(A, B, C, D)
        self.csys = sys_continuous
        sys_discrete = ctrl.c2d(sys_continuous, self.dt, method='zoh')
        self.dsys = sys_discrete
        K, _, _ = ctrl.dlqr(self.dsys.A, self.dsys.B, Q, R)
        return K

    def get_ss_bag_vectors(self, N):
        """N is the number of simulation steps, thus number of concatinated x vectors"""
        x_bag = np.zeros((self.n_states, N))
        u_bag = np.zeros((self.n_inputs, N))
        return x_bag, u_bag

    def mpc(self, x0, x_goal, A_ineq, b_ineq, Q=np.eye(5), R=np.eye(1), N=10, render=True, deltaB=None, dynamic=False):
        x = cp.Variable((5, N))
        u = cp.Variable((1, N))
        cost = sum(cp.quad_form(x[:, i] - x_goal.T, Q) + cp.quad_form(u[:, i], R) for i in range(N))
        cost += cp.quad_form(x[:3,N-1] - x_goal[:3], np.eye(3))
        obj = cp.Minimize(cost)

        #cons = [x[:, 0] == self.dsys.A @ x0 + self.dsys.B @ u[:, 0]]
        cons = [x[:, 0] == x0]
        #cons += [x[:3,N-1] == x_goal[:3]]
        for i in range(1, N):
            cons += [x[:, i] == self.dsys.A @ x[:, i - 1] + self.dsys.B @ u[:, i-1]]
            #cons += [x[2, i]==x0[2]]

            if deltaB is not None:
                cons += [(A_ineq @ x[0:3, i])<= (b_ineq + np.array(deltaB) * i).flatten()]

        u_max = np.array([1.0])
        u_min = np.array([0.0])
        if dynamic==True:
            cons += [u <= np.array([u_max]).T, u >= np.array([u_min]).T]
            cons += [x[6:9,:] <= 2*np.ones((3,1)), x[6:9,:] >= -2*np.ones((3,1))]

        if deltaB is None:
            cons += [A_ineq @ x[0:3, :] <= b_ineq]

        prob = cp.Problem(obj, cons)
        prob.solve(verbose=False)

        # print("optimal value", prob.value)
        # print(u.value)
        # print(x.value)

        point_id = []
        line_id = []
        if render == True:
            pred_x = np.hstack((x0[0:3].reshape((3, 1)), x.value[0:3, :]))
            for i in range(N):
                point_id.append(p.addUserDebugPoints([pred_x[:, i + 1].tolist()], [[0, 0, 1]], 5))
                line_id.append(p.addUserDebugLine(pred_x[:, i].tolist(), pred_x[:, i + 1].tolist(), [0, 1, 0], 3))
        #print(u.value[:,0])
        return u.value[:,0], x.value, point_id, line_id

    def step(self, x, x_ref, cont_type="LQR", info_dict=None, dynamic=False):
        # discrete step
        u = 0
        if cont_type == "LQR":
            u = self.K @ (x_ref - x)
        elif cont_type == "MPC":
            A_ineq, b_ineq = info_dict["A"], info_dict["b"]
            u, _, point_id, line_id = self.mpc(x, x_ref, A_ineq, b_ineq, Q=np.diag([1,1,1,0,0,0,0,0,0,0,0,0]), R=np.eye(4), N=10, render=True, deltaB=info_dict["deltaB"],dynamic=dynamic)
        x_next = self.dsys.A @ x + self.dsys.B @ u
        return x_next,point_id, line_id

if __name__ == "__main__":
    model = HIV_model()
    print(model.n_states)
    """model.nonlinear_dyn()
    x = np.array([1, 1, 1, 1, 1])
    u = np.array([1])
    x_next = model.compute_next_state(x, u)
    print(x_next)"""