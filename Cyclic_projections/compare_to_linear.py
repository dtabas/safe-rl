import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
import pickle

#Compare to linear/no control:
from envs.power_system_2 import Power_system_2
import torch
import control
from pytope import Polytope

plt.rcParams["font.family"] = "Times New Roman"


def dlqr_calculate(G, H, Q, R, returnPE=False):
  '''
  Discrete-time Linear Quadratic Regulator calculation.
  State-feedback control  u[k] = -K*x[k]

  How to apply the function:    
      K = dlqr_calculate(G,H,Q,R)
      K, P, E = dlqr_calculate(G,H,Q,R, return_solution_eigs=True)

  Inputs:
    G, H, Q, R  -> all numpy arrays  (simple float number not allowed)
    returnPE: define as True to return Ricatti solution and final eigenvalues

  Returns:
    K: state feedback gain
    P: Ricatti equation solution
    E: eigenvalues of (G-HK)  (closed loop z-domain poles)
  '''
  from scipy.linalg import solve_discrete_are, inv, eig
  P = solve_discrete_are(G, H, Q, R)  #Solução Ricatti
  K = inv(H.T@P@H + R)@H.T@P@G    #K = (B^T P B + R)^-1 B^T P A 

  if returnPE == False:   return K

  from numpy.linalg import eigvals
  eigs = np.array([eigvals(G-H@K)]).T
  return K, P, eigs

def compute_Uxt(x,env):
    Fx = env.Fx
    gx = env.gx
    A = env.A
    B = env.B
    H = env.U.A
    ubar = env.max_action
    x = np.reshape(x,(2,1))
    umin = scipy.optimize.linprog(c=np.array([1.]),A_ub  = Fx@B,b_ub = (gx-Fx@A@x),bounds=(-ubar,ubar)).fun
    umax = scipy.optimize.linprog(c=np.array([-1.]),A_ub  = Fx@B,b_ub = (gx-Fx@A@x),bounds=(-ubar,ubar)).fun
    
    
    Ui_H = np.block([[Fx@B,gx - Fx@A@x],[env.U.A,env.U.b]])
    Ui = Polytope(A = Ui_H[:,:-1],b = Ui_H[:,-1])

    umin = min(Ui.V)
    umax = max(Ui.V)
    
    
    return [umin,umax]

def compute_Vxt(x,env):
    x = np.reshape(x,(2,1))
    z = np.ones(env.num_vertex)
    V = env.V
    Y = env.Y
    vmin = scipy.optimize.linprog(c=V.numpy().flatten(),A_eq = np.block([[Y.numpy()],[z]]),b_eq = np.block([[x],[1]]),bounds=(0,None)).fun
    vmax = -scipy.optimize.linprog(c=-V.numpy().flatten(),A_eq = np.block([[Y.numpy()],[z]]),b_eq = np.block([[x],[1]]),bounds=(0,None)).fun
    return [vmin,vmax]

k = 108

yr = []
yp = []

env = Power_system_2()


w_all_traj = []
speed_all_traj = []

for s in range(10,30):

    plt.figure(2,figsize=(8,8),dpi=500)
    initial_state = np.array([.8,4.5])
    T = 25
    
    rpn = pickle.load((open('results/' + env.env_name + str(s) + '/' + 'rpn' + '/last_policy.pickle',mode = 'rb')))
    
    num_steps=T
    env.state = initial_state
    episode_reward = 0
    max_th = 0
    ave_th = 0
    max_penalty = 0
    ave_penalty = 0
    traj = [env.state[0]]
    W = 0
    theta = env.state[0]
    thetadot = env.state[1]
    state = np.array([np.cos(theta), np.sin(theta), thetadot])
    u_traj = []
    w_traj = []
    speed_traj = [thetadot]
    
    for t in range(num_steps):
        action,b,w = rpn.get_action(state)
        u_traj.append(action)
        
        x = np.array([np.arccos(state[0]) * np.sign(state[1]),state[2]])
        
        if s == 10:
            u_t = compute_Uxt(x,env)
            v_t = compute_Vxt(x,env)
            
            plt.figure(2)
            plt.subplot(222)
            if t == 0:
                plt.plot([t,t],u_t,'-bo',label = r'$\Omega(x_t)$',linewidth = 1)
                plt.plot([t,t],v_t,'--ro',label = r'$V(x_t)$',linewidth = 1)
            else:
                plt.plot([t,t],u_t,'-bo',linewidth = 1)
                plt.plot([t,t],v_t,'--ro',linewidth = 1)
        
        next_state, reward, done, _, penalty = env.step(action,b,k,t,s)
    
        state = next_state
        episode_reward += reward
        th = np.arccos(state[0]) * np.sign(state[1])
        max_th = max(max_th, abs(th))
        max_penalty = max(max_penalty, abs(penalty))
        ave_th += abs(th)
        ave_penalty += abs(penalty)
        traj.append(th)
        W += w
        w_traj.append(w)
        speed_traj.append(state[2])
    w_all_traj.append(w_traj)
    speed_all_traj.append(speed_traj)
        
        
    yr.append(traj)
    plt.figure(2)
    plt.subplot(221)
    #plt.plot(traj,label = 'RCI policy network')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    if s ==10:
        plt.figure(2)
        plt.subplot(222)
        plt.plot(u_traj,label = 'RCI-PN')
        plt.xlabel('Time step',fontsize=25)
        plt.ylabel('Control input',fontsize=25)
        plt.title('Controls',fontsize=25)
        plt.legend(loc = 'upper right',fontsize=10)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
    
    
    pn = pickle.load((open('results/' + env.env_name + str(s) + '/' + 'pn' + '/last_policy.pickle',mode = 'rb')))
    
    env.state = initial_state
    episode_reward = 0
    max_th = 0
    ave_th = 0
    max_penalty = 0
    ave_penalty = 0
    traj = [env.state[0]]
    W = 0
    theta = env.state[0]
    thetadot = env.state[1]
    state = np.array([np.cos(theta), np.sin(theta), thetadot])
    u_traj = []
    
    for t in range(num_steps):
        action = pn.get_action(state)
        u_traj.append(action)
        
        x = np.array([np.arccos(state[0]) * np.sign(state[1]),state[2]])
        
        next_state, reward, done, _, penalty = env.step(action,b,k,t,s)
    
        state = next_state
        episode_reward += reward
        th = np.arccos(state[0]) * np.sign(state[1])
        max_th = max(max_th, abs(th))
        max_penalty = max(max_penalty, abs(penalty))
        ave_th += abs(th)
        ave_penalty += abs(penalty)
        traj.append(th)
        W += w
    
    yp.append(traj)
    plt.figure(2)
    plt.subplot(221)
    #plt.plot(traj,label = 'Policy network')
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

plt.subplot(221)
plt.plot(np.mean(yr,axis = 0),label = 'RCI-PN')
plt.fill_between(range(26),np.max(yr,axis=0),np.min(yr,axis=0),alpha = 0.5)

plt.subplot(221)
plt.plot(np.mean(yp,axis = 0),label = 'PN')
plt.fill_between(range(26),np.max(yp,axis=0),np.min(yp,axis=0),alpha = 0.5)



#%% LQR stuff

K = dlqr_calculate(env.A,env.B,np.diag([1,.1]),np.array([.001]))
env.state = initial_state
b = torch.zeros((env.num_vertex))
traj = [initial_state[0]]
for t in range(T):
    #obs = state
    #x = np.array([np.arccos(obs[0]) * np.sign(obs[1]), obs[2]])
    action = -K@env.state
    action = np.clip(action,a_min=-15,a_max=15)
    #print(action)
    next_state, reward, done, _, penalty = env.step(action,b,k,t,s)
    state = next_state
    th = np.arccos(state[0]) * np.sign(state[1])
    traj.append(th)
plt.figure(2)
plt.subplot(221)
#plt.plot(traj,label = 'LQR')
#plt.ylim([-.1,1.5])
plt.title(r'Angle trajectories',fontsize=25)
plt.xlabel('Time step',fontsize=25)
plt.ylabel('Angle (rad)',fontsize=25)
plt.plot([0,T],[1,1],'--k',label=r'$\delta_{lim}$')

plt.legend(fontsize = 10,framealpha = 0.5)

#%% Number of iterations

plt.figure(2)
plt.subplot(223)
plt.plot(np.mean(w_all_traj,axis=0))
plt.fill_between(range(25),np.max(w_all_traj,axis=0),np.min(w_all_traj,axis=0),alpha = 0.5)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Number of projection iterations',fontsize=25)
plt.ylabel('Iterations',fontsize=25)
plt.xlabel('Time step',fontsize=25)




#%% State space plots

plt.figure(2)
plt.subplot(224)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('State space trajectories',fontsize=25)
plt.ylabel('Angular velocity',fontsize=25)
plt.xlabel('Angle',fontsize=25)
for i in range(20):
    plt.plot(yr[i],speed_all_traj[i])
S = Polytope(env.Fi,env.gi)
S.plot(alpha = 0.2)
plt.ylim([-3.5,6])
plt.xlim([-.25,1.2])

plt.tight_layout()