from __future__ import print_function
import torch
import numpy as np
import sinkhorn
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import train
import timeit
import os
import itertools

one = torch.FloatTensor([1])

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias:
            m.bias.data.fill_(0.01)


class SinkhornNet(nn.Module):

    def __init__(self, k, d, device="cpu"):
        super(SinkhornNet, self).__init__()
        self.fc1 = nn.Linear(1, d*k, bias=False)
        self.fc2 = nn.Linear(1, k, bias=False)
        self.d = d
        self.proj = True
        self.device = device
        
    def forward(self, z):
        x = self.fc1(z)
        #alpha = F.softmax(self.fc2(z), dim=0)
        alpha = self.fc2(z)
        return alpha, x.view(-1, self.d)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def projection(self):
        torch.clamp_(self.fc1.weight.data, min=-1, max=1)
        self.fc2.weight.data = train.simplex_proj(self.fc2.weight.data.flatten(), device=self.device).view(-1, 1)



class DCNet(nn.Module):

    def __init__(self, k, d, y, device="cpu"):
        super(DCNet, self).__init__()
        self.d = d
        self.y = y
        self.K = len(y)
        self.gamma = torch.rand(k, self.K, device=device)
        self.proj = False # use projected gradient step
        self.device = device
        
    def forward(self, z):
        x = -torch.sign(torch.mm(self.gamma, self.y))
        return self.gamma, x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class DescentNet(nn.Module):

    def __init__(self, k, d, K, beta, device="cpu"):
        super(DescentNet, self).__init__()
        self.fc1 = nn.Linear(1, d*k, bias=False)
        self.fc2 = nn.Linear(1, k*K, bias=False)
        self.d = d
        self.K = K
        self.beta = beta
        self.proj = True
        self.device = device

    def forward(self, z):
        x = self.fc1(z).view(-1, self.d)
        #gamma = self.beta*F.softmax(self.fc2(z).view(-1, self.K), dim=0)
        gamma = self.fc2(z).view(-1, self.K)
        return gamma, x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def projection(self):
        torch.clamp_(self.fc1.weight.data, min=-1, max=1)
        gamma = self.fc2.weight.clone().view(-1, self.K)
        #marginale beta
        for k in range(self.K):
            gamma[:,k] = train.simplex_proj(gamma[:,k], self.beta[k], device=self.device)
        self.fc2.weight.data = gamma.view(-1, 1)
    

if __name__ == '__main__':
    
    one = torch.FloatTensor([1])

    expstart = 1
    manual_seed = 137
    np.random.seed(seed=manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    nexp = 2
    cost = sinkhorn._linear_cost
    time_allowed = 10 # time spent per training in s

    #dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dev = torch.device('cpu')
    print('Device {}'.format(dev))

    dimlambK_range = [(20, 0.1, 100), (20, 0.5, 100), (40, 0.1, 100)]
    
    # Sinkhorn parameters
    sinkiterrange = [5]
    sinklrrange = [1., 10.]

    # Descent parameters
    descentlrrange = [1e-5, 10.]

    # DC parameters
    dclrrange = [1e-5, 1e-4]
    dcdualiterrange = [5]
    t0 = timeit.default_timer()

    for dim, lamb, K in dimlambK_range:
        os.system('mkdir experiments/type_data_K{0}_dim{1}'.format(K, dim))

        # Simulate Sinkhorn
        for s in itertools.product(sinkiterrange, sinklrrange):
            for exp in range(expstart, expstart+nexp):

                ## generate prior
                try:
                    y = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/y_{2}.npy'.format(K, dim, exp)))
                except:
                    y = 2*torch.rand(K, dim)-1
                    y = (y.t() / torch.sum(torch.abs(y), dim=1)).t()
                    np.save('experiments/type_data_K{0}_dim{1}/y_{2}.npy'.format(K, dim, exp), y)
                try:
                    beta = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/beta_{2}.npy'.format(K, dim, exp)))
                except:
                    beta = F.softmax(torch.rand(K))
                    np.save('experiments/type_data_K{0}_dim{1}/beta_{2}.npy'.format(K, dim, exp), beta)

                sinkiter = s[0]
                sinklr = s[1]

                p = os.path.isfile('experiments/sinkhorn/{0}_lamb{1}_k{2}_dim{3}_sinkiter{4}_lr{5}_sinkhorn_{6}/losses.npy'.format(exp, lamb, K, dim, sinkiter, sinklr, dev))

                if not(p):

                    if dev!="cpu":
                        y = y.to(dev)
                        beta = beta.to(dev)

                    # train if not already done for these parameters
                    print('Train Sinkhorn expe {}: sinkiter={}, sinklr={}, lamb={}, dim={}, K={}.'.format(exp, sinkiter, sinklr, lamb, dim, K))
                    net = SinkhornNet(K+2, dim, device=dev)

                    if dev!="cpu":
                        net.to(dev)
                        
                    net.apply(init_weights)
                    if net.proj:
                        net.projection()
                    train.train_sinkhorn(net, y, beta, lamb=lamb, niter_sink=sinkiter, learning_rate=sinklr, cost=cost, max_time=time_allowed, experiment=exp,
                                        verbose=False, err_threshold=1e-3, device=dev)



        for descentlr in descentlrrange:
            for exp in range(expstart, expstart+nexp):
                y = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/y_{2}.npy'.format(K, dim, exp)))
                beta = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/beta_{2}.npy'.format(K, dim, exp)))

                p = os.path.isfile('experiments/descent/{0}_lamb{1}_k{2}_dim{3}_lr{4}_descent_{5}/losses.npy'.format(exp,lamb,K, dim, descentlr, dev))
                if not(p):

                    if dev!="cpu":
                        y = y.to(dev)
                        beta = beta.to(dev)

                    ## Descent experiment
                    print('Train Descent expe {}: descentlr={}, lamb={}, dim={}, K={}.'.format(exp, descentlr, lamb, dim, K))
                    net = DescentNet(K+2, dim, K, beta, device=dev)

                    if dev!="cpu":
                        net.to(dev)

                    net.apply(init_weights)
                    if net.proj:
                        net.projection()
                    train.train_descent(net, y, beta, lamb=lamb, learning_rate=descentlr, cost=cost, max_time=time_allowed, verbose=False, experiment=exp, device=dev)


        for s in itertools.product(dcdualiterrange, dclrrange):
            for exp in range(expstart, expstart+nexp):
                y = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/y_{2}.npy'.format(K, dim, exp)))
                beta = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/beta_{2}.npy'.format(K, dim, exp)))


                dcdualiter = s[0]
                dclr = s[1]

                p = os.path.isfile('experiments/dc/{0}_lamb{1}_k{2}_dim{3}_dualiter{4}_lr{5}_dc_{6}/losses.npy'.format(exp, lamb, K, dim, dcdualiter, dclr, dev))

                if not(p):

                    if dev!="cpu":
                        y = y.to(dev)
                        beta = beta.to(dev)

                    ## DC experiment
                    print('Train DC expe {}: dcdualiter={}, dclr={}, lamb={}, dim={}, K={}.'.format(exp, dcdualiter, dclr, lamb, dim, K))
                    net = DCNet(K+2, dim, y, device=dev)

                    if dev!="cpu":
                        net.to(dev)

                    net.apply(init_weights)
                    if net.proj:
                        net.projection()
                    train.train_dc(net, y, beta, lamb=lamb, learning_rate=dclr, cost=cost, max_time=time_allowed, dual_iter=dcdualiter, err_threshold=1e-4, 
                                verbose=False, experiment=exp, device=dev)

    print('Total time of simulation: {} s'.format(timeit.default_timer()-t0))


