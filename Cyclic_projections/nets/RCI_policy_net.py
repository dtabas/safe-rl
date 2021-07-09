import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
global cc
cc = 0

class RCIPolicyNetwork(nn.Module):
    def __init__(self, env, obs_dim, num_vertex, hidden_dim, init_w=3e-3):
        super(RCIPolicyNetwork, self).__init__()

        self.env = env

        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_vertex)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        
        a = F.relu(self.linear1(state))
        a = F.relu(self.linear2(a))
        a = F.softmax(self.linear3(a), dim = 1)
        b = a
        
        x = self.env.get_state(state)
        
        Y = self.env.Y
        P = Y.T@torch.inverse(Y@Y.T)
        #print(torch.norm(P@(Y@b.T-x.T)).detach())
        #print(torch.eig(P@Y))
        
        aa = [torch.norm(P@(Y@a.T-x.T),p=2,dim=0).detach()]
        #for i in range(20):
        i = 0
        
        while np.any(torch.norm((a@Y.T - x) @ P.T, p=2, dim=1).detach().numpy() > 1e-6):
            
            i = i + 1
            
            '''eta = 1 #20/torch.sqrt(torch.tensor(i+1))
            #eta = torch.tensor(bb)/torch.sqrt(torch.tensor(i+1))
            #g = a@(self.env.Y.T@self.env.Y) - x @ self.env.Y
            g = (a@Y.T - x) @ P.T
            eg = torch.exp(-eta * g)
            a = F.normalize(a*eg,p=1,dim=1)'''
            
            mu = 1.5 # under/over-projection parameter, in [0,2]
            
            delta_k = a - a @ (P@Y).T + x @ P.T
            delta_k = mu*delta_k + (1-mu)*a # over/under-project
            gamma_k = torch.clamp(delta_k,min = 0)
            ##gamma_k = F.softmax(delta_k,dim=1)
            gamma_k = mu*gamma_k + (1-mu)*delta_k # over/under-project
            alpha_k_plus = F.normalize(gamma_k,p=1,dim=1) # Relative entropy projection
            alpha_k_plus = mu*alpha_k_plus + (1-mu)*gamma_k # over/under-project
            #a = a + 1/self.env.num_vertex*(1-torch.sum(a,dim=1,keepdim=True)) # Euclidean projection
            
            a = alpha_k_plus
            
            aa.append(torch.norm((a@Y.T - x) @ P.T, p=2, dim=1).detach().numpy())
            
            if i > 50:
                break
            
        global cc
        cc += 1
        
        if cc % 10 == 0:
            if a.size()[0] == 128:
                plt.figure(10)
                plt.semilogy([c[120] for c in aa])
        '''if a.size()[0] == 1:
            plt.figure(11)
            plt.semilogy(aa)'''
        #print(aa[0])
        
        action = a @ self.env.V.T
        
        return action,b,i

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action,b,w = self.forward(state)
        return action.detach().cpu().numpy()[0],b.detach().cpu()[0],w