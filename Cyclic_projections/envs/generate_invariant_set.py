import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
from pytope import Polytope
import os
import pickle

#%%%

def invariant_set(A,B,E,X,U,D,h,env_name):
    
    params = {'A':A,'B':B,'E':E,'X':X.V,'U':U.V,'D':D.V,'h':h}
    filename = env_name + '_invariant_set.pickle'
    if os.path.isfile(filename):
        sys = pickle.load(open(filename,'rb'))
        
        same = True
        for param in params:
            check = (params[param] != sys['params'][param])
            if np.any(check):
                same = False
                
        if same:
            
            Fx = sys['Fx']
            gx = sys['gx']
            Fi = sys['Fi']
            gi = sys['gi']
            
            X.plot(fill = False, edgecolor = (0,0,0))
            Polytope(Fi,gi).plot(fill = False, edgecolor = (0,1,0))
            Polytope(Fx,gx).plot(fill = False, edgecolor = (1,0,0))
            plt.autoscale(enable=True)
            
            return Fx,gx,Fi,gi
        
    
    e = .001 # Tolerance for invariant set computations
    c = 1 # Counter
    
    while c < 100:
        
        # Plot
        plt.figure(1)
        if c == 1:
            X.plot(fill = False,edgecolor = (0,1-1/c,0),label = 'Safety set')
        else:
            X.plot(fill = False,edgecolor = (0,1-1/c,0))
        plt.xlabel('Angle (rad)')
        plt.ylabel('Angular speed (rad/sec)')
        plt.autoscale(enable = True)
        
        # 1. Erode 
        #if D == 0:
        #    P = X
        #else:
        P = X - E*D
        
        # 2. Expand
        m = np.shape(U.A)[0]
        n = np.shape(P.A)[1]
        M = Polytope(np.block([[P.A@A, P.A@B],[np.zeros((m,n)),U.A]]),np.vstack((P.b,U.b)))
    
        # 3. Project
        R = M.project((0,1))
        R.determine_H_rep()
        R.determine_V_rep()
        
        # 4. Intersect
        Y = R & X
        
        # 5. Evaluate stopping criterion
        if (Y*(1+e)).contains(X):
            X = Y
            X.plot(fill = False,edgecolor = (0,0,1),label = 'Invariant set')
            break
        else:
            X = Y
        
        print([c,np.shape(Y.V)[0]])
        c += 1
        
    #if D == 0:
    #    S = X
    #else:
    S = X - E*D
    S.plot(fill = False, edgecolor = (1,0,0),label = 'Target set')
    plt.legend()
    
    sys = {'params':params,'Fx':S.A,'gx':S.b,'Fi':X.A,'gi':X.b}
    pickle.dump(sys,open(filename,'wb'))
    
    return S.A,S.b,X.A,X.b