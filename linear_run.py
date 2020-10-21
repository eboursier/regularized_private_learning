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
from utils import *

one = torch.FloatTensor([1])


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias:
            m.bias.data.fill_(0.01)

# we define pytorch networks so we can use automatic differentiation,
# but they only have a single layer. So they are not really networks.


class SinkhornNet(nn.Module):
    """
    Network for optimizing through Sinkhorn structures
    """

    def __init__(self, k, d, device="cpu"):
        super(SinkhornNet, self).__init__()
        self.fc1 = nn.Linear(1, d*k, bias=False)
        self.fc2 = nn.Linear(1, k, bias=False)
        self.d = d
        self.nactions = k
        self.proj = True
        self.device = device

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
        """
        projection on the simplex (projected gradient)
        """
        torch.clamp_(self.fc1.weight.data, min=-1, max=1)
        self.fc2.weight.data = train.simplex_proj(self.fc2.weight.data.flatten(
        ), device=self.device).view(-1, 1)


class DCNet(nn.Module):
    """
    Network for optimizing with DC scheme
    """

    def __init__(self, k, d, y, device="cpu"):
        super(DCNet, self).__init__()
        self.d = d
        self.y = y
        self.nactions = k
        self.K = len(y)
        self.gamma = torch.rand(k, self.K, device=device)
        self.proj = False  # use projected gradient step
        self.device = device

    def forward(self, z):
        # best actions given a joint distribution gamma
        x = -torch.sign(torch.mm(self.gamma, self.y))
        return self.gamma, x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class DescentNet(nn.Module):
    """
    Network for optimizing with gradient descent scheme
    """

    def __init__(self, k, d, K, beta, device="cpu"):
        super(DescentNet, self).__init__()
        self.fc1 = nn.Linear(1, d*k, bias=False)
        self.fc2 = nn.Linear(1, k*K, bias=False)
        self.d = d
        self.K = K
        self.nactions = k
        self.beta = beta
        self.proj = True
        self.device = device

    def forward(self, z):
        x = self.fc1(z).view(-1, self.d)
        # gamma = self.beta*F.softmax(self.fc2(z).view(-1, self.K), dim=0)
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
        # projection to guarantee marginale beta
        for k in range(self.K):
            gamma[:, k] = train.simplex_proj(
                gamma[:, k], self.beta[k], device=self.device)
        self.fc2.weight.data = gamma.view(-1, 1)


if __name__ == '__main__':

    one = torch.FloatTensor([1])

    expstart = 1
    manual_seed = 137
    np.random.seed(seed=manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    nexp = 200
    cost = sinkhorn._linear_cost
    time_allowed = 8  # time spent per training in s

    # dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dev = torch.device('cpu')
    print('Device {}'.format(dev))

    # different values of dim, lamb and K to simulate
    dimlambK_range = [(20, 0.1, 100), (20, 0.5, 100), (40, 0.1, 100)]

    # Sinkhorn parameters to try
    # (sinkiter, learningrate, optimizer, differentiation, warm_restart)
    sinkparams = [
        #(5, 1e-1, "SGD", "analytic", True, 1),
        #(5, 1e-2, "adam", "analytic", False, 1),
        #(5, 1e-2, "rms", "analytic", False, 1),
        #(5, 1e-2, "adam", "automatic", False, 1),
        #(5, 1e-2, "rms", "automatic", False, 1),
        (5, 1e-2, "adam", "analytic", True, 1),
        (5, 1e-2, "rms", "analytic", True, 1),
        #(5, 1e-2, "rms", "analytic", True, 1.5),
        #(5, 1e-2, "rms", "analytic", True, 2),
        #(5, 1e-2, "rms", "analytic", True, 3),
        #(5, 1e-2, "rms", "analytic", True, 5)
    ]

    # (sinkiter, learningrate, optimizer, differentiation, warm_restart)
    sinkmomentums = [0.95]

    # Descent parameters to try
    # (lamb, learningrate, optimizer)
    descentparams = [(0.5, 1e-4, "adam"), (0.1, 1e-2, "adam"),
                     (0.5, 1e-4, "rms"), (0.1, 0.1, "rms")]

    # DC parameters to try
    # (dualiter, learningrate)
    dcparams = [(5, 1e-5), (5, 1e-4)]
    t0 = timeit.default_timer()

    for dim, lamb, K in dimlambK_range:
        os.system('mkdir experiments/type_data_K{0}_dim{1}'.format(K, dim))

        # Simulate Sinkhorn
        for sinkiter, sinklr, optim, diff, warm_restart, actmult in sinkparams:
            nactions = np.int((K+2)*actmult)
            moms = sinkmomentums if optim == "SGD" else [0]
            restart_string = "_warm" if warm_restart else ""
            for exp in range(expstart, expstart+nexp):
                for mom in moms:
                    momstring = "_{}".format(mom) if mom != 0 else ""
                    # generate prior which will be the same for all different
                    # optimization schemes (but will change in different runs)
                    yfile = 'type_data_K{}_dim{}/y_{}.npy'.format(K, dim, exp)
                    betafile = 'type_data_K{}_dim{}/beta_{}.npy'.format(
                        K, dim, exp)
                    try:
                        y = torch.from_numpy(np.load('experiments/'+yfile))
                    except:
                        y = 2*torch.rand(K, dim)-1
                        y = (y.t() / torch.sum(torch.abs(y), dim=1)).t()
                        np.save('experiments/'+yfile, y)
                    try:
                        beta = torch.from_numpy(
                            np.load('experiments/'+betafile))
                    except:
                        beta = F.softmax(torch.rand(K))
                        np.save('experiments/'+betafile, beta)

                    # warm restart
                    sink_fold = 'experiments/sinkhorn/{}_lamb{}_k{}'.format(
                        exp, lamb, K)

                    if nactions != K+2:
                        sink_fold += '_actions{}'.format(nactions)

                    sink_fold += '_dim{}_sinkiter{}_lr{}_'.format(
                        dim, sinkiter, sinklr)

                    sink_fold += 'sinkhorn_{}_{}_{}{}{}/'.format(
                        dev, optim, diff, momstring, restart_string)

                    p = os.path.isfile(sink_fold+'losses.npy')
                    if not(p):  # train if not already done
                        if dev != "cpu":
                            y = y.to(dev)
                            beta = beta.to(dev)

                        print_params(algo='Sinkhorn', exp=exp,
                                     sinkiter=sinkiter, sinklr=sinklr,
                                     lamb=lamb, dim=dim, K=K,
                                     nactions=nactions,
                                     optim=optim, diff=diff,
                                     warm_restart=warm_restart,
                                     momentum=mom)
                        net = SinkhornNet(nactions, dim, device=dev)

                        if dev != "cpu":
                            net.to(dev)

                        net.apply(init_weights)
                        if net.proj:
                            net.projection()
                        train.train_sinkhorn(net, y, beta, lamb=lamb,
                                             niter_sink=sinkiter,
                                             experiment=exp,
                                             cost=cost, learning_rate=sinklr,
                                             verbose=False,
                                             max_time=time_allowed,
                                             err_threshold=1e-3, device=dev,
                                             optim=optim, differentiation=diff,
                                             warm_restart=warm_restart,
                                             momentum=mom)

        # Simulate gradient descent
        for lambd, descentlr, optim in descentparams:
            moms = descentmomentums if optim == "SGD" else [0]
            nactions = K+2
            if lamb == lambd:  # different learning rates for different lamb
                for mom in moms:
                    momstring = "_{}".format(mom) if mom != 0 else ""
                    for exp in range(expstart, expstart+nexp):
                        yfile = 'type_data_K{}_dim{}/y_{}.npy'.format(
                            K, dim, exp)
                        betafile = 'type_data_K{}_dim{}/beta_{}.npy'.format(
                            K, dim, exp)
                        y = torch.from_numpy(np.load('experiments/'+yfile))
                        beta = torch.from_numpy(
                            np.load('experiments/'+betafile))

                        desc_fold = 'experiments/descent/{}_lamb{}'.format(
                            exp, lamb)

                        if nactions != K+2:
                            desc_fold += '_actions{}'.format(nactions)

                        desc_fold += '_k{}_dim{}_lr{}'.format(
                            K, dim, descentlr)
                        desc_fold += '_descent_{}_{}{}/'.format(
                            dev, optim, momstring)
                        p = os.path.isfile(desc_fold+'losses.npy')
                        if not(p):
                            if dev != "cpu":
                                y = y.to(dev)
                                beta = beta.to(dev)

                            # Descent experiment
                            print_params(algo='Descent', exp=exp,
                                         descentlr=descentlr, lamb=lamb,
                                         dim=dim, K=K,
                                         nactions=nactions, optim=optim,
                                         momentum=mom)
                            net = DescentNet(nactions, dim, K,
                                             beta, device=dev)

                            if dev != "cpu":
                                net.to(dev)

                            net.apply(init_weights)
                            if net.proj:
                                net.projection()
                            train.train_descent(net, y, beta, lamb=lamb,
                                                learning_rate=descentlr,
                                                max_time=time_allowed,
                                                verbose=False, experiment=exp,
                                                device=dev, optim=optim,
                                                cost=cost, momentum=mom)

        # Simulate DCA
        for dcdualiter, dclr in dcparams:
            nactions = K+2
            for exp in range(expstart, expstart+nexp):
                yfile = 'type_data_K{}_dim{}/y_{}.npy'.format(K, dim, exp)
                betafile = 'type_data_K{}_dim{}/beta_{}.npy'.format(
                    K, dim, exp)
                y = torch.from_numpy(np.load('experiments/'+yfile))
                beta = torch.from_numpy(np.load('experiments/'+betafile))

                dc_fold = 'experiments/dc/{}_lamb{}'.format(exp, lamb)

                if nactions != K+2:
                    dc_fold += '_actions{}'.format(nactions)

                dc_fold += '_k{}_dim{}_dualiter{}'.format(K, dim, dcdualiter)
                dc_fold += '_lr{}_dc_{}/'.format(dclr, dev)
                p = os.path.isfile(dc_fold+'losses.npy')
                if not(p):
                    if dev != "cpu":
                        y = y.to(dev)
                        beta = beta.to(dev)

                    # DC experiment
                    print_params(algo='DC', exp=exp, dcdualiter=dcdualiter,
                                 dclr=dclr, lamb=lamb, dim=dim, K=K,
                                 nactions=nactions)
                    net = DCNet(nactions, dim, y, device=dev)

                    if dev != "cpu":
                        net.to(dev)

                    net.apply(init_weights)
                    if net.proj:
                        net.projection()
                    train.train_dc(net, y, beta, lamb=lamb, learning_rate=dclr,
                                   cost=cost, max_time=time_allowed,
                                   dual_iter=dcdualiter, err_threshold=1e-4,
                                   verbose=False, experiment=exp, device=dev)
    print('Total time of simulation: {} seconds'.format(
        timeit.default_timer()-t0))
