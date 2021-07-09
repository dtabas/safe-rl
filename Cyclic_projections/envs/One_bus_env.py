import numpy as np
import torch
import pickle

from pytope import Polytope
from envs.generate_vertices import vertices

class PowerSys_1bus:

    def __init__(self):
        self.obs_dim = 2
        self.action_dim = 1
        self.max_speed = 1
        self.max_action = 8#2.
        self.dt = .05
        self.k = -2#1
        self.m = .3#1.
        self.d = -1.5#.1
        self.safe_th = 1.  # safe region [-1, 1]
        self.d_max = 1
        self.num_vertex = 2
        self.env_name = 'onebus'
        
        Ac = np.array([[0,1],[-self.k/self.m,-self.d/self.m]])
        Bc = np.array([[0],[1]])
        Ec = np.array([[0],[-1]])
        
        self.A = np.eye(2) + self.dt * Ac
        self.B = self.dt * Bc
        self.E = self.dt * Ec

        self.seed()
        
        # Constraint sets:
        self.X = Polytope(lb = (-self.safe_th,-self.max_speed),ub = (self.safe_th,self.max_speed)) # Safe set
        self.U = Polytope(lb = -self.max_action, ub = self.max_action) # Control set
        self.D = Polytope(lb = -self.d_max, ub = self.d_max) # Disturbance set
        
        self.Y,self.V = vertices(self.A,self.B,self.E,self.X,self.U,self.D,self.dt,self.env_name)
        
        self.num_vertex = np.shape(self.Y)[1]

    def seed(self, seed=None):
        np.random.seed(seed)

    def step(self, u,b):
        
        #assert np.shape(self.state) == (2,)
        
        th, thdot = self.state  # th := theta
        x = torch.tensor([[th],[thdot]]).type(torch.FloatTensor)

        u = np.clip(u, -self.max_action, self.max_action)[0]
        
        Y = self.Y
        P = Y.T@torch.inverse(Y@Y.T)

        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2) + 1*torch.norm(P@(Y@b.T-x))
        
        penalty = torch.norm(P@(Y@b.T-x))
        
        d = (np.random.rand()*2-1)*self.d_max
        #d = self.d_max * np.sign(thdot) # TODO this is a test
        
        x = np.reshape(self.state,(2,1))
        
        newx = self.A@x + self.B*u + self.E*d
        
        newx[1] = np.clip(newx[1],-self.max_speed,self.max_speed)

        self.state = np.reshape(newx,(2,))
        #assert np.shape(self.state) == (2,)
        return self._get_obs(), -costs, False, {}, penalty
        

    def reset(self):
        high = np.array([1, 0])  # initialize in the safe region
        # high = np.array([np.pi, 1])         # initialize
        self.state = np.random.uniform(low=-high, high=high)
        
        #self.state = np.array([0,0]) # TODO this is a test
        
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta,thetadot])
    
    def get_action_halfspace(self, obs):
        dt = self.dt
        max_action = self.max_action
        
        x = self.convert(obs)
        batch = np.shape(x)[0]
        
        H = self.B
        def f(x):
            batch = np.shape(x)[0]
            #assert np.shape(x) == (batch,2)
            newx = x@self.A.T
            newx = np.expand_dims(newx,axis=2)
            #assert np.shape(newx) == (batch,2,1)
            return newx
        
        # State space safety constraints for invariance:
        Fx = torch.tensor(self.Fx)
        gx = torch.tensor(self.gx)
        
        # Action space safety constraints (one-step look ahead):
        Fs = (Fx@H).unsqueeze(0).repeat(batch,1,1)
        gs = gx - Fx@f(x);
                
        # Action set at time t:
        Ft = torch.cat((self.Fu.repeat(batch,1,1),Fs),dim = 1)
        gt = torch.cat((self.gu.repeat(batch,1,1),gs),dim = 1)
        
        return Ft,gt
    
    def convert(self,obs):
        if np.ndim(obs) == 3:
            th = obs[:,:,0]
            thdot = obs[:,:, 1]
        elif np.ndim(obs) == 2:
            th = obs[:,0]
            thdot = obs[:,1]
        x = np.column_stack((th,thdot))
        return x
    
    def get_state(self,state):
        return state
        
def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
