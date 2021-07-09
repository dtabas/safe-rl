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

XX = np.array([[-.5,0],[0,7.5],[.6,5],[.95,-7.5]])

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
    Y = (S_safe.V).T
    YY = Y
    
    if __name__ == '__main__':
        plt.figure(3,figsize=(8,4),dpi=500)
        plt.subplot(122)
        plt.plot(U.V,[0,0],'-ok',label='U',linewidth=3)
        plt.autoscale(enable=True)
    
    # Build V matrix and expand Y matrix:
    V = np.zeros((m,p))
    for i,x in enumerate(list(YY.T)):
        x = np.reshape(x,(n,1))
        Ui_H = np.block([[Fx@B,gx - Fx@A@x],[U.A,U.b]])
        Ui = Polytope(A = Ui_H[:,:-1],b = Ui_H[:,-1])
        qi = np.shape(Ui.V)[0] # Number of vertices of Ui
        Y_new_i = np.tile(np.reshape(Y[:,i],(n,1)),(1,qi))
        if i == 0:
            V = Ui.V.T
            Y_new = Y_new_i
        else:
            V = np.append(V,Ui.V.T,axis = 1)
            Y_new = np.append(Y_new,Y_new_i,axis = 1)
        
    if __name__ == '__main__':
        for i,x in enumerate(list(XX)):
            x = np.reshape(x,(n,1))
            Ui_H = np.block([[Fx@B,gx - Fx@A@x],[U.A,U.b]])
            Ui = Polytope(A = Ui_H[:,:-1],b = Ui_H[:,-1])
            plt.figure(3)
            plt.subplot(122)
            if i == 0:
                plt.plot(Ui.V,(i+1)*np.ones(len(Ui.V)),'-bo',label=r'$\Omega(x_i)$',linewidth=3)
            else: 
                plt.plot(Ui.V,(i+1)*np.ones(len(Ui.V)),'-bo',linewidth=3)
            
    Y = Y_new
    p = np.shape(Y)[1]
    
    Y = torch.tensor(Y).type(torch.FloatTensor)
    V = torch.tensor(V).type(torch.FloatTensor)
    
    if __name__ == '__main__':
        return Y,V,YY,S_safe
    else:
        return Y,V,Fx,gx,Fi,gi

if __name__ == '__main__':
    
    def parameters_power_system_2():
        max_speed = 8
        max_action = 15.
        dt = .05
        g = -1.
        m = 1.
        l = 1.
        safe_th = 1.  # safe region [-1, 1]
        env_name = 'power_system_2'
        
        d = 0.1 # damping
        
        # Linearized dynamics:
        A = np.array([[1,dt],[0,1-dt*d]]) # Linear portion of dynamics
        C = 3*g/(2*l) * np.array([[dt**2],[dt]])@np.array([[1,0]]) # Linearized nonlinear portion of dynamics
        A = A + C
        B = 3/(m*l**2) * np.array([[dt**2],[dt]]) # Control input
        E = 3*g/(2*l) * np.array([[dt**2],[dt]]) # Linearization error disturbance input
        
        # State and input bounds:
        noise_max = .5
        d_max = safe_th - np.sin(safe_th) + noise_max # Max linearization error inside safe set, plus noise
        
        # Constraint sets:
        X = Polytope(lb = (-safe_th,-max_speed),ub = (safe_th,max_speed)) # Safe set
        U = Polytope(lb = -max_action, ub = max_action) # Control set
        D = Polytope(lb = -d_max, ub = d_max) # Disturbance set
        return A,B,E,X,U,D,dt,env_name
    
    def parameters_pendulum():
        # Parameters:
        h = .05
        g = 10.
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
        
    A,B,E,X,U,D,h,env_name = parameters_power_system_2()
    
    Y,V,YY,S = vertices(A,B,E,X,U,D,h,env_name)
    print(np.round(Y,2))
    
    p = Y.size()[1]
    
    z = np.ones((1,p)) 
    
    for i,x in enumerate(list(XX)):
        x = np.reshape(x,(2,1))
        vmin = scipy.optimize.linprog(c=V.numpy().flatten(),A_eq = np.block([[Y.numpy()],[z]]),b_eq = np.block([[x],[1]]),bounds=(0,None)).fun
        vmax = -scipy.optimize.linprog(c=-V.numpy().flatten(),A_eq = np.block([[Y.numpy()],[z]]),b_eq = np.block([[x],[1]]),bounds=(0,None)).fun
        
        plt.figure(3)
        plt.subplot(122)
        if i == 0:
            plt.plot([vmin,vmax],(i+1)*np.ones(2),'--ro',label = r'$V(x_i)$',linewidth=3)
        else: 
            plt.plot([vmin,vmax],(i+1)*np.ones(2),'--ro',linewidth=3)
        
    plt.legend(fontsize=15)
    
    plt.figure(3)
    
    
    plt.subplot(121)
    X.plot(alpha = 0.5,color = (0,1,0),label = 'X')
    S.plot(alpha=0.5,color = (0,0,1),label = 'S')
    plt.xlabel('Angle (rad)',fontsize=25)
    plt.ylabel('Frequency (rad/sec)',fontsize=25)
    plt.title('Safe and invariant sets',fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks([-8,-4,0,4,8],fontsize=20)
    
    plt.subplot(121)
    plt.plot(XX[:,0],XX[:,1],'kd',label=r'$x_i$')
    plt.legend(fontsize=15)
    plt.annotate(r'$x_1$',.05+XX[0,:],fontsize=20)
    plt.annotate(r'$x_2$',np.array([0,-1.5])+XX[1,:],fontsize=20)
    plt.annotate(r'$x_3$',np.array([0,-2.])+XX[2,:],fontsize=20)
    plt.annotate(r'$x_4$',np.array([-.3,.3])+XX[3,:],fontsize=20)
    
    
    plt.subplot(122)
    plt.xlabel('Control input',fontsize=25)
    plt.ylabel('Sample point',fontsize=25)
    plt.title('Sample action sets',fontsize=25)
    plt.yticks(ticks = [0,1,2,3,4],labels=['U',r'$x_1$',r'$x_2$',r'$x_3$',r'$x_4$'],fontsize=20)
    plt.xticks([-15,0,15],fontsize=20)
    
    plt.tight_layout()
    
    '''a_1 = torch.rand((p,1))**5
    a_1 = a_1/torch.norm(a_1,p=1)
    a_1_traj = a_1
    plt.figure(3)
    for i in range(10):
        a_1 = (torch.eye(p) - Y.T@torch.inverse(Y@Y.T)@Y) @ a_1
        a_1 = a_1 + Y.T@torch.inverse(Y@Y.T)@x
        a_1 = torch.maximum(a_1,torch.zeros((p,1)))
        #a_1 = a_1/torch.norm(a_1,p=1)
        a_1 = a_1 + z/p*(1-torch.sum(a_1))
        a_1_traj = torch.cat((a_1_traj,a_1),dim = 1)
    plt.plot(a_1_traj.T)'''
    
    def newton_step(Y,a,x,t):
        n,p = np.shape(Y)
        z = np.ones(p)
        P = Y.T@np.linalg.inv(Y@Y.T)
        #g = Y.T@Y@a - Y.T@x + np.ones((p,p))@a - z - 1/t * np.diag(1/a) @ z
        g = P@Y@a - P@x + np.ones((p,p))@a - z - 1/t * np.diag(1/a) @ z
        #Z = np.block([[P@Y],[z.T]])
        Z1 = np.block([P,np.ones((p,1))])
        Z2 = np.block([[Y],[np.ones((1,p))]])
        Dinv = np.diag(a**2)
        Hinv = t*Dinv - t**2*Dinv@Z1@np.linalg.inv(np.eye(n+1) + t*Z2@Dinv@Z1)@Z2@Dinv
        da_nt = -Hinv@g
        return a + .25*da_nt
    
    Y = Y.numpy()
    P = Y.T@np.linalg.inv(Y@Y.T)
    
    x = np.array([.6,5])
    a = np.random.rand(p)
    a = a**10
    a = a/sum(a)
    
    penalty_traj = [np.linalg.norm(P@(Y@a-x))]
    for t in np.logspace(2,7,15):
        for j in range(3):
            a = newton_step(Y,a,x,t)
            penalty_traj.append(np.linalg.norm(P@(Y@a-x)))
    plt.figure()
    plt.semilogy(penalty_traj)