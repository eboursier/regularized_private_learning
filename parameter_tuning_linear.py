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
lamb = 1.
expstart = 1
manual_seed = 137
np.random.seed(seed=manual_seed)
torch.manual_seed(manual_seed)
nexp = 10
cost = sinkhorn._linear_cost

dev = "cpu"
if dev=="gpu":
	device = torch.device("cuda:0")
	print('Device {}'.format(device))
else:
	print('Device {}'.format(dev))

os.system('mkdir experiments/type_data_K{0}_dim{1}'.format(K, dim))

# Sinkhorn tuning

sinkiterrange = [5]
sinklrrange = np.geomspace(1e-2, 1, 3)
sinkmaxiter = 5000

for s in itertools.product(sinkiterrange, sinklrrange):
	for exp in range(nexp):

		## generate prior
		try:
			y = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/y_{2}.npy'.format(K, dim, exp+1)))
		except:
			y = 2*torch.rand(K, dim)-1
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
			# train if not already done for these parameters
			net = SinkhornNet(K+2, dim)

			if gpu:
				net.to(device)
				y.to(device)
				beta.to(device)

			net.apply(init_weights)
			if net.proj:
				net.projection()
			train.train_sinkhorn(net, y, beta, lamb=lamb, niter_sink=sinkiter, learning_rate=sinklr, cost=cost, max_iter=sinkmaxiter, experiment=exp+1,
								verbose=False, err_threshold=1e-3, device=dev)


# Descent training
descentlrrange = np.geomspace(1e-4, 100, 10)
descentmaxiter = 50000

for descentlr in descentlrrange:
	for exp in range(nexp):
		y = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/y_{2}.npy'.format(K, dim, exp+1)))
		beta = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/beta_{2}.npy'.format(K, dim, exp+1)))

		p = os.path.isfile('experiments/descent/{0}_lamb{1}_k{2}_dim{3}_lr{4}_descent_{6}/losses.npy'.format(exp+1,lamb,K, dim, descentlr, dev))
		if not(p):
			## Descent experiment
			net = DescentNet(K+2, dim, K, beta)

			if gpu:
				net.to(device)
				y.to(device)
				beta.to(device)

			net.apply(init_weights)
			if net.proj:
				net.projection()
			train.train_descent(net, y, beta, lamb=lamb, learning_rate=descentlr, cost=cost, max_iter=descentmaxiter, verbose=False, experiment=exp+1, device=dev)


# DC training
dclrrange = [0.00001]
dcdualiterrange = [5]
dcmaxiter = 1000

for s in itertools.product(dcdualiterrange, dclrrange):
	for exp in range(nexp):
		y = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/y_{2}.npy'.format(K, dim, exp+1)))
		beta = torch.from_numpy(np.load('experiments/type_data_K{0}_dim{1}/beta_{2}.npy'.format(K, dim, exp+1)))


		dcdualiter = s[0]
		dclr = s[1]

		p = os.path.isfile('experiments/dc/{0}_lamb{1}_k{2}_dim{3}_dualiter{4}_lr{5}_dc_{6}/losses.npy'.format(exp+1, lamb, K, dim, dcdualiter, dclr, dev))

		if not(p):
			## DC experiment
			net = DCNet(K+2, dim, y)

			if gpu:
				net.to(device)
				y.to(device)
				beta.to(device)

			net.apply(init_weights)
			if net.proj:
				net.projection()
			train.train_dc(net, y, beta, lamb=lamb, learning_rate=dclr, cost=cost, max_iter=dcmaxiter, dual_iter=dcdualiter, err_threshold=1e-4, 
						verbose=False, experiment=exp+1)



