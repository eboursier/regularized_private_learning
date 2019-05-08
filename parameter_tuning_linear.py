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
from linear_run import *
import itertools
import os

one = torch.FloatTensor([1])

K = 100
dim = 20
lamb = 0.1
expstart = 1
manual_seed = 137
np.random.seed(seed=manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
nexp = 200
cost = sinkhorn._linear_cost

#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dev = torch.device('cpu')
print('Device {}'.format(dev))

os.system('mkdir experiments/type_data_K{0}_dim{1}'.format(K, dim))

# Sinkhorn tuning

sinkiterrange = [5]
sinklrrange = [1]
sinkmaxiter = 3000

for s in itertools.product(sinkiterrange, sinklrrange):
	for exp in range(nexp):

		## generate prior
		try:
			y = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/y_{2}.npy'.format(K, dim, exp+1)))
		except:
			y = 2*torch.rand(K, dim)-1
			y = (y.t() / torch.sum(torch.abs(y), dim=1)).t()
			np.save('experiments/type_data_K{0}_dim{1}/y_{2}.npy'.format(K, dim, exp+1), y)

		try:
			beta = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/beta_{2}.npy'.format(K, dim, exp+1)))
		except:
			beta = F.softmax(torch.rand(K))
			np.save('experiments/type_data_K{0}_dim{1}/beta_{2}.npy'.format(K, dim, exp+1), beta)

		sinkiter = s[0]
		sinklr = s[1]

		p = os.path.isfile('experiments/sinkhorn/{0}_lamb{1}_k{2}_dim{3}_sinkiter{4}_lr{5}_sinkhorn_{6}/losses.npy'.format(exp+1, lamb, K, dim, sinkiter, sinklr, dev))

		if not(p):

			if dev!="cpu":
				y = y.to(dev)
				beta = beta.to(dev)

			# train if not already done for these parameters
			net = SinkhornNet(K+2, dim, device=dev)

			if dev!="cpu":
				net.to(dev)
				
			net.apply(init_weights)
			if net.proj:
				net.projection()
			train.train_sinkhorn(net, y, beta, lamb=lamb, niter_sink=sinkiter, learning_rate=sinklr, cost=cost, max_iter=sinkmaxiter, experiment=exp+1,
								verbose=False, err_threshold=1e-3, device=dev)


# Descent training
descentlrrange = [1e-5, 10.]
descentmaxiter = 1000

for descentlr in descentlrrange:
	for exp in range(nexp):
		y = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/y_{2}.npy'.format(K, dim, exp+1)))
		beta = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/beta_{2}.npy'.format(K, dim, exp+1)))

		p = os.path.isfile('experiments/descent/{0}_lamb{1}_k{2}_dim{3}_lr{4}_descent_{5}/losses.npy'.format(exp+1,lamb,K, dim, descentlr, dev))
		if not(p):

			if dev!="cpu":
				y = y.to(dev)
				beta = beta.to(dev)

			## Descent experiment
			net = DescentNet(K+2, dim, K, beta, device=dev)

			if dev!="cpu":
				net.to(dev)

			net.apply(init_weights)
			if net.proj:
				net.projection()
			train.train_descent(net, y, beta, lamb=lamb, learning_rate=descentlr, cost=cost, max_iter=descentmaxiter, verbose=False, experiment=exp+1, device=dev)


# DC training
dclrrange = [1e-4]
dcdualiterrange = [5]
dcmaxiter = 250

for s in itertools.product(dcdualiterrange, dclrrange):
	for exp in range(nexp):
		y = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/y_{2}.npy'.format(K, dim, exp+1)))
		beta = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/beta_{2}.npy'.format(K, dim, exp+1)))


		dcdualiter = s[0]
		dclr = s[1]

		p = os.path.isfile('experiments/dc/{0}_lamb{1}_k{2}_dim{3}_dualiter{4}_lr{5}_dc_{6}/losses.npy'.format(exp+1, lamb, K, dim, dcdualiter, dclr, dev))

		if not(p):

			if dev!="cpu":
				y = y.to(dev)
				beta = beta.to(dev)

			## DC experiment
			net = DCNet(K+2, dim, y, device=dev)

			if dev!="cpu":
				net.to(dev)

			net.apply(init_weights)
			if net.proj:
				net.projection()
			train.train_dc(net, y, beta, lamb=lamb, learning_rate=dclr, cost=cost, max_iter=dcmaxiter, dual_iter=dcdualiter, err_threshold=1e-4, 
						verbose=False, experiment=exp+1, device=dev)



