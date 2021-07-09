import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
from pytope import Polytope
if __name__ == '__main__':
    from generate_invariant_set import invariant_set
else:
    from envs.generate_invariant_set import invariant_set
import torch
#%%%

def vertices(A,B,E,X,U,D,h,env_name):
    # Generate the matrices Y and V. 
    # The columns of V are the vertices of the action polytope at each vertex of the invariant set. 
    # The columns of Y are the vertices of the invariant set, repeated once for each corresponding vertex of the action polytope.
    # The set {x: Fx @ x <= gx} describes the target set: u must be chosen such that Fx @ (Ax + Bu) <= gx. This set is smaller than the invariant set in order to account for disturbances.
    # The set {x: Fi @ x <= gi} is the actual invariant set.
    
    # Generate invariant and target set:
    Fx,gx,Fi,gi = invariant_set(A,B,E,X,U,D,h,env_name)
    S_targ = Polytope(A = Fx, b = gx)
    S_safe = Polytope(A = Fi, b = gi)
    
    # Get dimensions:
    p = np.shape(S_safe.V)[0]
    n,m = np.shape(B)
    
    # Matrix whose columns are vertices of invariant set:
    Y = (Polytope(A = Fi, b = gi).V).T
    
    if __name__ == '__main__':
        plt.figure(2)
        U.plot(fill=False,edgecolor=(0,0,0))
        plt.autoscale(enable=True)
    
    # Build V matrix and expand Y matrix:
    V = np.zeros((m,p))
    for i in range (p):
        x = np.reshape(Y[:,i],(n,1))
        Ui_H = np.block([[Fx@B,gx - Fx@A@x],[U.A,U.b]])
        Ui = Polytope(A = Ui_H[:,:-1],b = Ui_H[:,-1])
        if __name__ == '__main__':
            plt.figure(2)
            UUi = Polytope(V=np.column_stack((Ui.V,np.ones((np.shape(Ui.V)[0],1))*i)))
            UUi = UUi+Polytope(lb=(-.1,0),ub=(.1,0))
            UUi.plot(fill=False,edgecolor = (0,0,1))
            #print(Ui.V)
        qi = np.shape(Ui.V)[0] # Number of vertices of Ui
        Y_new_i = np.tile(np.reshape(Y[:,i],(n,1)),(1,qi))
        if i == 0:
            V = Ui.V.T
            Y_new = Y_new_i
        else:
            V = np.append(V,Ui.V.T,axis = 1)
            Y_new = np.append(Y_new,Y_new_i,axis = 1)
            
    Y = Y_new
    p = np.shape(Y)[1]
    
    Y = torch.tensor(Y).type(torch.FloatTensor)
    V = torch.tensor(V).type(torch.FloatTensor)
    
    return Y,V,Fi,gi

if __name__ == '__main__':
    
    def parameters_onebus():
        dt = .05
        k = -2
        m = .3
        d = -1.5
        safe_th = 1.  # safe region [-1, 1]
        d_max = 1
        num_vertex = 2
        env_name = 'onebus'
        max_speed = 1
        max_action = 8.
        
        Ac = np.array([[0,1],[-k/m,-d/m]])
        Bc = np.array([[0],[1]])
        Ec = np.array([[0],[-1]])
        
        A = np.eye(2) + dt * Ac
        B = dt * Bc
        E = dt * Ec
        
        # Constraint sets:
        X = Polytope(lb = (-safe_th,-max_speed),ub = (safe_th,max_speed)) # Safe set
        U = Polytope(lb = -max_action, ub = max_action) # Control set
        D = Polytope(lb = -d_max, ub = d_max) # Disturbance set
        
        return A,B,E,X,U,D,dt,env_name
    
    def parameters_pendulum():
        # Parameters:
        h = .05
        g = -10.
        m = 1.
        l = 1.
        env_name = 'pendulum'
        
        # Linearized dynamics:
        A = np.array([[1,h],[0,1]]) # Linear portion of dynamics
        C = 3*g/(2*l) * np.array([[h**2],[h]])@np.array([[1,0]]) # Linearized nonlinear portion of dynamics
        A = A + C # Linearized dynamics
        B = 3/(m*l**2) * np.array([[h**2],[h]]) # Control input
        E = 3*g/(2*l) * np.array([[h**2],[h]]) # Linearization error disturbance input
        
        # State and input bounds:
        theta_max = 1. # Max angle
        omega_max = 8 # Max speed
        u_max = 15 # Max control
        noise_max = 0
        d_max = theta_max - np.sin(theta_max) + noise_max # Max linearization error inside safe set, plus noise
        
        # Constraints sets:
        X = Polytope(lb = (-theta_max,-omega_max),ub = (theta_max,omega_max)) # Safe set
        U = Polytope(lb = -u_max, ub = u_max) # Control set
        D = Polytope(lb = -d_max, ub = d_max) # Disturbance set
        
        return A,B,E,X,U,D,h,env_name
        
    A,B,E,X,U,D,h,env_name = parameters_pendulum()
    
    Y,V = vertices(A,B,E,X,U,D,h,env_name)
    print(np.round(Y,2))
    
    p = Y.size()[1]
    x_t = torch.tensor([[0.5],[0.5]])
    z = torch.ones((p,1))
    
    a_1 = torch.rand((p,1))**5
    a_1 = a_1/torch.norm(a_1,p=1)
    a_1_traj = a_1
    plt.figure(3)
    for i in range(10):
        a_1 = (torch.eye(p) - Y.T@torch.inverse(Y@Y.T)@Y) @ a_1
        a_1 = a_1 + Y.T@torch.inverse(Y@Y.T)@x_t
        a_1 = torch.maximum(a_1,torch.zeros((p,1)))
        #a_1 = a_1/torch.norm(a_1,p=1)
        a_1 = a_1 + z/p*(1-torch.sum(a_1))
        a_1_traj = torch.cat((a_1_traj,a_1),dim = 1)
    plt.plot(a_1_traj.T)
    
    print(np.linalg.eig(A))
    
    '''d_2 = torch.rand((p,1))
    b_2 = torch.maximum(d_2, -torch.tensor(scipy.linalg.pinv(P))@c_4)
    a_2 = torch.tensor(scipy.linalg.inv(U)) @ b_2
    a_1 = c_3 @ a_2 + c_4
    action = V @ a_1
    print(c_3@a_2 + c_4)'''
    
    '''w = torch.ones((p,1))
    c_1 = torch.cat((torch.cat((Y@Y.T,Y@w),dim=1),torch.cat(((Y@w).T,torch.tensor([[1.]])),dim=1)),dim=0)
    c_2 = torch.cat((Y,w.T),dim = 0)
    c_3 = torch.eye(p) - c_2.T @ torch.inverse(c_1) @ c_2
    
    c_4 = c_2.T @ torch.inverse(c_1) @ torch.cat((x_t,torch.tensor([[1.]])),dim = 0)
    U,P = scipy.linalg.polar(c_3,side='left')
    
    a_2 = (torch.rand((p,1)))
    a_2 = a_2/torch.norm(a_2,p=1)
    for i in range(3):
        a_1 = c_3 @ a_2
        a_2 = torch.maximum(a_1,torch.zeros((p,1)))
        print(a_1)'''