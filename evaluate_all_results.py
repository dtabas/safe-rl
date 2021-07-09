import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.optimize
import matplotlib.pyplot as plt
import pickle
#%%%


ENV = 'power_system_2'
names = ['RCI Network','Policy Network']
folders = ['rpn','pn']
#colors = []#'b'#'r'

plt.figure(1,figsize=(12,4),dpi=500)
plt.rcParams["font.family"] = "Times New Roman"

x = list(range(200))
a = .5
for i,j in zip(names,folders):
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    for seed in range(10,30):
        env = ENV + str(seed)
        
        if j == 'rpn':
            path = 'mirror descent RL first attempt/results/' + env + '/' + j + '/'
        elif j == 'pn':
            path = 'policy network comparison/results/' + env + '/' + j + '/'
        
        y1.append(pickle.load(open(path + 'rewards.pickle',mode = 'rb')))
        y2.append(pickle.load(open(path + 'max_angle.pickle',mode = 'rb')))
        y3.append(pickle.load(open(path + 'ave_angle.pickle',mode = 'rb')))
        
        if j == 'rpn':
            y4.append(pickle.load(open(path +  'projection_steps.pickle',mode = 'rb')))
        
    m1_1 = np.mean(y1,axis=0)
    m3_1 = np.mean(y3,axis=0)
    
    m1_0 = np.min(y1,axis = 0)
    m1_2 = np.max(y1,axis=0)
    
    m2_0 = np.max(y2,axis = 0)
    m2_1 = np.mean(y2,axis=0)
    
    m3_0 = np.min(y3,axis = 0)
    m3_2 = np.max(y3,axis=0)
    
    if j == 'rpn':
        m4_0 = np.min(y4,axis=0)
        m4_1 = np.mean(y4,axis=0)
        m4_2 = np.max(y4,axis=0)


    plt.subplot(1,3,1)
    #plt.plot(m1,'-',label = i,linewidth=3)
    #plt.fill_between(x,m1-s1,m1+s1, alpha=0.5)
    plt.plot(m1_1,'-',linewidth=3,label=i)
    plt.fill_between(x,m1_0,m1_2, alpha=a)
    plt.title('Rewards',fontsize=25)
    plt.ylabel('Reward',fontsize=25)
    plt.legend(fontsize=15)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlabel('Episode',fontsize=25)
    #plt.yscale('symlog')
    plt.ylim([-100,0])
    

    plt.figure(1)
    plt.subplot(1,3,2)
    plt.plot(m2_1,'-',linewidth=3)
    plt.fill_between(x,m2_0,m2_1, alpha=a)
    plt.title('Max angle',fontsize=25)
    plt.ylabel('Max angle (rad)',fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlabel('Episode',fontsize=25)
    plt.ylim([.2,2])
    
    plt.figure(1)
    plt.subplot(1,3,3)
    #plt.plot(m3,'-',label = i,linewidth=3)
    #plt.fill_between(x,m3-s3,m3+s3, alpha=0.5)
    plt.plot(m3_1,'-',linewidth=3)
    plt.fill_between(x,m3_0,m3_2, alpha=a)
    plt.title('Average angle ',fontsize=25)
    plt.ylabel('Avg. angle (rad)',fontsize=25)
    plt.xlabel('Episode',fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylim([0,.8])
    
    a = .3
    
plt.tight_layout()

plt.subplot(1,3,2)
plt.plot([0,len(m2_1)-1],[1,1],'--k',label = r'$\delta_{lim}$')
plt.legend(fontsize=20)

plt.figure(2,figsize=(12,4),dpi=500)
plt.plot(m4_1,'-',linewidth=3)
plt.fill_between(x,m4_0,m4_2,alpha=.5)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.title('Average number of cyclic projetion algorithm iterations',fontsize=25)
plt.ylabel('Iterations',fontsize=25)
plt.xlabel('Episode',fontsize=25)


'''        
    plt.figure(10,figsize=(5,2))
    f = open(env + '/' + j + '/projection_steps.pickle',mode = 'rb')
    y = pickle.load(f)
    plt.plot(y,label = i)
    plt.title('Average number of projection iterations')
    
    f = open(env + '/' + j + '/training_traj.pickle',mode = 'rb')
    plt.figure(2)
    Y = pickle.load(f)
    for k in range(len(Y)):
        ep = k
        if k >190:
            plt.plot(Y[ep],label = f'Episode {ep}')
    #plt.legend()
    plt.title('Trajectories ' + i + ' ' + env)

plt.figure(1)
plt.subplot(3,1,2)
plt.plot([0,len(y)-1],[1,1],'--k',label = 'lim')

plt.subplot(3,1,1)
plt.legend()
plt.title('Rewards ' + env)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.tight_layout()


plt.figure(2)
plt.plot([0,100],[0,0],'--k')'''