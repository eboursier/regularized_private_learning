from __future__ import print_function
import torch
import numpy as np
import sinkhorn
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import train
import argparse
import timeit
import util

one = torch.FloatTensor([1])

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias:
            m.bias.data.fill_(0.01)


class SinkhornNet(nn.Module):

    def __init__(self, k, d):
        super(SinkhornNet, self).__init__()
        self.fc1 = nn.Linear(1, d*k, bias=False)
        self.fc2 = nn.Linear(1, k, bias=False)
        self.d = d
        self.proj = True
        
    def forward(self, z):
        x = self.fc1(z)
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
        self.fc2.weight.data = train.simplex_proj(self.fc2.weight.data.flatten()).view(-1, 1)



class DCNet(nn.Module):

    def __init__(self, k, d, y):
        super(DCNet, self).__init__()
        self.d = d
        self.y = y
        self.K = len(y)
        self.gamma = torch.rand(k, self.K)
        self.proj = False # use projected gradient step
        
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

    def __init__(self, k, d, K, beta):
        super(DescentNet, self).__init__()
        self.fc1 = nn.Linear(1, d*k, bias=False)
        self.fc2 = nn.Linear(1, k*K, bias=False)
        self.d = d
        self.K = K
        self.beta = beta
        self.proj = True

    def forward(self, z):
        x = self.fc1(z).view(-1, self.d)
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
        # marginale beta
        for k in range(self.K):
            gamma[:,k] = train.simplex_proj(gamma[:,k], self.beta[k])
        self.fc2.weight.data = gamma.view(-1, 1)
    

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser = util.get_args(parser)

	args = parser.parse_args()
	manual_seed = 137
	np.random.seed(seed=manual_seed)
	torch.manual_seed(manual_seed)
	    
	cost = sinkhorn._linear_cost


	for exp in range(args.expstart, args.expstart+args.nexp):
		## generate prior
		y = 2*torch.rand(args.K, args.dim)-1
		beta = F.softmax(torch.rand(args.K))

		## Sinkhorn experiment
		net = SinkhornNet(args.K+2, args.dim)
		net.apply(init_weights)
		if net.proj:
			net.projection()
		train.train_sinkhorn(net, y, beta, lamb=args.lamb, niter_sink=args.sinkiter, learning_rate=args.sinklr, cost=cost, max_iter=args.sinkmaxiter, experiment=exp,
							 verbose=args.verbose, verbose_freq=args.verbose_freq, err_threshold=1e-3)

		## Descent experiment
		net = DescentNet(args.K+2, args.dim, args.K, beta)
		net.apply(init_weights)
		if net.proj:
			net.projection()
		train.train_descent(net, y, beta, lamb=args.lamb, learning_rate=args.descentlr, cost=cost, max_iter=args.descentmaxiter, verbose=args.verbose, experiment=exp,
							verbose_freq=args.verbose_freq)

		## DC experiment
		net = DCNet(args.K+2, args.dim, y)
		net.apply(init_weights)
		if net.proj:
			net.projection()
		train.train_dc(net, y, beta, lamb=args.lamb, learning_rate=args.dclr, cost=cost, max_iter=args.dcmaxiter, dual_iter=args.dcdualiter, err_threshold=1e-4, 
						verbose=args.verbose, experiment=exp, verbose_freq=args.verbose_freq)

	print('Experiments ended')