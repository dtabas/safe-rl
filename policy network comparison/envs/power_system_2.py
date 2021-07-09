import numpy as np
from envs.generate_vertices import vertices
from pytope import Polytope
import torch
import scipy.optimize
class Power_system_2:

    def __init__(self):
        self.obs_dim = 3
        self.action_dim = 1
        self.max_speed = 8
        self.max_action = 15.
        self.dt = .05
        self.g = -1.
        self.m = 1.
        self.l = 1.
        self.safe_th = 1.  # safe region [-1, 1]
        self.env_name = 'power_system_2'
        
        self.d = 0.1 # damping

        self.seed()
        
        # Linearized dynamics:
        A = np.array([[1,self.dt],[0,1-self.dt*self.d]]) # Linear portion of dynamics
        C = 3*self.g/(2*self.l) * np.array([[self.dt**2],[self.dt]])@np.array([[1,0]]) # Linearized nonlinear portion of dynamics
        self.A = A + C
        self.B = 3/(self.m*self.l**2) * np.array([[self.dt**2],[self.dt]]) # Control input
        self.E = 3*self.g/(2*self.l) * np.array([[self.dt**2],[self.dt]]) # Linearization error disturbance input
        
        # State and input bounds:
        self.noise_max = .5
        d_max = self.safe_th - np.sin(self.safe_th) + self.noise_max # Max linearization error inside safe set, plus noise
        
        # Constraint sets:
        self.X = Polytope(lb = (-self.safe_th,-self.max_speed),ub = (self.safe_th,self.max_speed)) # Safe set
        self.U = Polytope(lb = -self.max_action, ub = self.max_action) # Control set
        self.D = Polytope(lb = -d_max, ub = d_max) # Disturbance set
        
        self.Y,self.V,self.Fi,self.gi = vertices(self.A,self.B,self.E,self.X,self.U,self.D,self.dt,self.env_name)
        
        self.num_vertex = np.shape(self.Y)[1]

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self,u,b,k,t,s):
        th, thdot = self.state  # th := theta
        x = torch.tensor([[th],[thdot]]).type(torch.FloatTensor)

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        d = self.d
        Y = self.Y
        P = Y.T@torch.inverse(Y@Y.T)

        u = np.clip(u, -self.max_action, self.max_action)[0]

        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2) + 2*max(0,abs(angle_normalize(th))-self.safe_th)
        
        
        
        # + 1*torch.norm(P@(Y@b.T-x))
        #costs = np.abs(th) + .1*thdot**2 + .001*(u**2) + 10*torch.norm(P@(Y@b.T-x))
        
        #penalty = torch.norm(P@(Y@b.T-x))
        
        np.random.seed(s*10000+k*10+int(t/10))
        #noise = np.random.uniform(low=-self.noise_max, high=self.noise_max)
        noise = np.random.triangular(left=-self.noise_max, mode = self.noise_max,right=self.noise_max) # noise with bias so that there is something to learn
        self.seed()

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*(u-noise) - d*thdot) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}#, penalty

    def reset(self):
        #high = np.array([1, 0])  # initialize in the safe region
        # high = np.array([np.pi, 1])         # initialize
        #self.state = np.random.uniform(low=-high, high=high)
        
        #alpha_0 = np.random.uniform(low=0,high = 1,size=(self.num_vertex))**5
        #alpha_0 = alpha_0/np.sum(alpha_0)
        #self.state = self.Y @ alpha_0
        
        th = np.random.uniform(low = -.9*self.safe_th,high=.9*self.safe_th)
        thdot_max = ((scipy.optimize.linprog(np.array([0,1]),A_ub=self.Fi,b_ub=self.gi)).fun)
        thdot_min = ((scipy.optimize.linprog(np.array([0,-1]),A_ub=self.Fi,b_ub=self.gi)).fun)
        thdot=np.random.uniform(low=.5*thdot_min,high=.5*thdot_max)
        
        self.state = np.array([th,thdot])

        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def get_action_vertex(self, obs):
        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        th = np.arccos(obs[:, 0]) * np.sign(obs[:, 1])
        thdot = obs[:, 2]
        action_u = (+self.safe_th - th - thdot * dt - 3*g/(2*l) * np.sin(th) * (dt**2)) / (3*(dt**2) / (m * (l**2)))
        action_l = (-self.safe_th - th - thdot * dt - 3*g/(2*l) * np.sin(th) * (dt**2)) / (3*(dt**2) / (m * (l**2)))

        action_u = np.clip(action_u, -self.max_action, self.max_action)
        action_l = np.clip(action_l, -self.max_action, self.max_action)

        return np.expand_dims(np.stack(zip(action_l, action_u)), axis=2)
    
    def get_state(self,state):
        x = torch.zeros((state.size()[0],state.size()[1]-1))
        th = torch.arccos(state[:,0]) * torch.sign(state[:,1])
        thdot = state[:,2]
        x[:,0] = th
        x[:,1] = thdot
        return x


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)