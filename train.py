import torch
from torch.autograd import Variable
import timeit
import torch.nn as nn
import torch.nn.functional as F
import os
import sinkhorn

import numpy as np

#import base_module


# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class Net(nn.Module):
    def __init__(self, decoder):
        super(Net, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output

def simplex_proj(x, p=1, device="cpu"):
    """
    compute the projection of x onto the simplex defined by x_i>=0 and sum x_i <= p
    See https://arxiv.org/abs/1101.6081
    """
    u,_ = torch.sort(x, descending=True)
    cs = u.cumsum(0) - p
    ind = torch.arange(len(x), dtype=torch.float, device=device) + 1
    cond = u - cs / ind > 0
    rho = ind[cond][-1]
    theta = cs[cond][-1] / rho
    return F.relu(x - theta)

def train_sinkhorn(net, y, beta, lamb = 1, niter_sink = 1, max_iter=1000, cost=sinkhorn._squared_distances,
            learning_rate=0.1, err_threshold=1e-4, experiment=0, verbose=False, verbose_freq=100, device="cpu", **kwargs):
    """
    learn a discrete distribution (alpha, x) with a prior (beta, y)
    """

    if experiment!=0:
        os.system('mkdir experiments/sinkhorn/{0}_lamb{1}_k{2}_dim{3}_sinkiter{4}_lr{5}_sinkhorn_{6}'.format(experiment,lamb,y.size(0), y.size(1), niter_sink, learning_rate, device))

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
    one = torch.FloatTensor([1]).to(device)

    iterations = 0
    loss_profile = []
    time_profile = []
    start_time = timeit.default_timer()
    running_time = 0
    while iterations<max_iter:

        # ---------------------------
        #        Optimize over net
        # ---------------------------
        time = timeit.default_timer()
        optimizer.zero_grad()

        alpha, x = net(one)


        ###### Sinkhorn loss #########

        loss, _ = sinkhorn.sinkhorn_loss_primal(alpha, x, beta, y, lamb, niter=niter_sink, cost=cost, err_threshold=err_threshold, verbose=False, **kwargs)
        loss.backward(one)
        optimizer.step()

        # projected gradient
        if net.proj:
            net.projection()

        running_time += (timeit.default_timer()-time) # for the sinkhorn method, it takes some time to compute the accurate loss from the estimated loss in the training
        time_profile.append(running_time)    # (as niter_sink can be small). In this case, we compute the true loss for the plot but it does not count in the running time

        # compute the true loss for plots and does not count in the running time

        _, gamma = sinkhorn.sinkhorn_loss_primal(alpha, x, beta, y, lamb, niter=100, cost=cost, err_threshold=1e-4)
        loss_p = torch.sum(gamma*cost(x,y)) + lamb*sinkhorn._KL(alpha, beta, gamma, epsilon=0)
        loss_profile.append(loss_p.cpu().detach().numpy())


        iterations += 1
        
        if verbose:
            if iterations%verbose_freq == 0:
               print('iterations='+str(iterations))

    if verbose:
        t_expe = (timeit.default_timer()-start_time)
        print('done in {0} s'.format(t_expe))
        print('total running time: {0} s'.format(running_time))
    if experiment!=0:
        # save data
        torch.save(net, 'experiments/sinkhorn/{0}_lamb{1}_k{2}_dim{3}_sinkiter{4}_lr{5}_sinkhorn_{6}/network'.format(experiment,lamb,y.size(0), y.size(1), 
                                                                                                                        niter_sink, learning_rate, device))
        np.save('experiments/sinkhorn/{0}_lamb{1}_k{2}_dim{3}_sinkiter{4}_lr{5}_sinkhorn_{6}/losses.npy'.format(experiment,lamb,y.size(0), y.size(1), niter_sink, 
                                                                                                                    learning_rate, device), loss_profile)
        np.save('experiments/sinkhorn/{0}_lamb{1}_k{2}_dim{3}_sinkiter{4}_lr{5}_sinkhorn_{6}/time.npy'.format(experiment,lamb,y.size(0), y.size(1), niter_sink, 
                                                                                                                learning_rate, device), time_profile)
    return loss_profile

def train_descent(net, y, beta, lamb = 1, max_iter=1000, cost=sinkhorn._squared_distances,
                learning_rate=0.1, experiment=0, verbose=False, verbose_freq=100, device="cpu",  **kwargs):
    """
    learn a discrete distribution (alpha, x) with a prior (beta, y)
    """

    if experiment!=0:
        os.system('mkdir experiments/descent/{0}_lamb{1}_k{2}_dim{3}_lr{4}_descent_{5}'.format(experiment,lamb,y.size(0), y.size(1), learning_rate, device))

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
    one = torch.FloatTensor([1]).to(device)

    iterations = 0
    loss_profile = []
    time_profile = []
    start_time = timeit.default_timer()
    running_time = 0
    while iterations<max_iter:

        # ---------------------------
        #        Optimize over net
        # ---------------------------

        time = timeit.default_timer()
        optimizer.zero_grad()
        gamma, x = net(one)
        alpha = torch.sum(gamma, dim=1)

        ###### Total loss #########
        C = cost(x, y, **kwargs)

        loss = torch.sum( gamma * C )  + lamb*sinkhorn._KL(alpha, beta, gamma)
        loss.backward(one)
        optimizer.step()

        if net.proj: # for projected gradient
            net.projection()

        running_time += (timeit.default_timer()-time) # reasons explained in train_sinkhorn
        time_profile.append(running_time)

        loss_profile.append(loss.cpu().detach().numpy())

        iterations += 1
        
        if verbose:
            if iterations%verbose_freq == 0:
               print('iterations='+str(iterations))

    if verbose:
        t_expe = (timeit.default_timer()-start_time)
        print('done in {0} s'.format(t_expe))
        print('total running time: {0} s'.format(running_time))
    if experiment!=0:
        # save data
        torch.save(net, 'experiments/descent/{0}_lamb{1}_k{2}_dim{3}_lr{4}_descent_{5}/network'.format(experiment,lamb,y.size(0), y.size(1), learning_rate, device))
        np.save('experiments/descent/{0}_lamb{1}_k{2}_dim{3}_lr{4}_descent_{5}/losses.npy'.format(experiment,lamb,y.size(0), y.size(1), learning_rate, device), loss_profile)
        np.save('experiments/descent/{0}_lamb{1}_k{2}_dim{3}_lr{4}_descent_{5}/time.npy'.format(experiment,lamb,y.size(0), y.size(1), learning_rate, device), time_profile)
    return loss_profile

def train_dc(net, y, beta, lamb = 1, max_iter=1000, cost=sinkhorn._squared_distances, err_threshold=1e-4, dual_iter=100, debug=False,
            learning_rate=0.01, experiment=0, verbose=False, verbose_freq=100, device="cpu", **kwargs):
    """
    learn a discrete distribution (alpha, x) with a prior (beta, y)
    """

    if experiment!=0:
        os.system('mkdir experiments/dc/{0}_lamb{1}_k{2}_dim{3}_dualiter{4}_lr{5}_dc_{6}'.format(experiment,lamb,y.size(0), y.size(1), dual_iter, learning_rate, device))

    one = torch.FloatTensor([1]).to(device)

    iterations = 0
    loss_profile = []
    time_profile = []
    start_time = timeit.default_timer()
    running_time = 0
    while iterations<max_iter:

        # ---------------------------
        #        Optimize over net
        # ---------------------------
        time = timeit.default_timer()
        gamma, x = net(one)
        dual_var = -torch.mm(x, y.t()) # dual iteration of DCA
        gamma_it = solve_relaxed_primal(dual_var, beta, gamma, lamb=lamb, max_iter=dual_iter, learning_rate=learning_rate, err_threshold=err_threshold, debug=debug, device=device) # primal iteration of DCA

        running_time += (timeit.default_timer()-time) # reasons explained in train_sinkhorn
        time_profile.append(running_time)

        alpha = torch.sum(gamma_it, 1)
        #C = cost(x, y, **kwargs)
        #loss = torch.sum( gamma_it * C )  + lamb*sinkhorn._KL(alpha, beta, gamma_it)
        loss = lamb*sinkhorn._KL(alpha, beta, gamma_it) - torch.sum(gamma_it*dual_var)
        loss_profile.append(loss.cpu().detach().numpy())    
        net.gamma = gamma_it
        iterations += 1
        
        if verbose:
            if iterations%verbose_freq == 0:
               print('iterations='+str(iterations))
    if verbose:
        t_expe = (timeit.default_timer()-start_time)
        print('done in {0} s'.format(t_expe))
        print('total running time: {0} s'.format(running_time))
    if experiment!=0:
        # save data
        torch.save(net, 'experiments/dc/{0}_lamb{1}_k{2}_dim{3}_dualiter{4}_lr{5}_dc_{6}/network'.format(experiment,lamb,y.size(0), y.size(1), dual_iter, learning_rate, device))
        np.save('experiments/dc/{0}_lamb{1}_k{2}_dim{3}_dualiter{4}_lr{5}_dc_{6}/losses.npy'.format(experiment,lamb,y.size(0), y.size(1), dual_iter, learning_rate, device), loss_profile)
        np.save('experiments/dc/{0}_lamb{1}_k{2}_dim{3}_dualiter{4}_lr{5}_dc_{6}/time.npy'.format(experiment,lamb,y.size(0), y.size(1), dual_iter, learning_rate, device), time_profile)
    return loss_profile

def relax_primal_loss(primal_var, dual_var, beta, lamb):
    alpha = torch.sum(primal_var, 1)
    g = lamb*sinkhorn._KL(alpha, beta, primal_var)
    h_relax = torch.sum(primal_var*dual_var)
    return g-h_relax

def solve_relaxed_primal(dual_var, beta, gamma, lamb=1, max_iter=100, learning_rate=0.01, err_threshold=1e-4, return_losses=False, debug=False, device="cpu"):
    """
    Find an approximated solution of inf(lambda KL(gamma, gamma_1 \times beta) - <gamma, dual_var>) in the feasible set of gamma using projected gradient
    """
    
    prim_var = gamma.clone()
    prim_var.requires_grad_(True)
    optimizer = torch.optim.SGD([prim_var], lr=learning_rate, momentum=0)
    loss = torch.sum(torch.Tensor([float('inf')])) # initialize inf with tensor type
    old_loss = 0

    if return_losses:  
        loss_profile = []

    iterations = 0
    while (iterations<max_iter) and (torch.abs(old_loss-loss)>err_threshold):

        # ---------------------------
        #        Optimize over net
        # ---------------------------
        old_loss = loss.clone()
        optimizer.zero_grad()

        ###### Total loss #########
        loss = relax_primal_loss(prim_var, dual_var, beta, lamb)
        loss.backward()
        optimizer.step()
        if return_losses:
            loss_profile.append(loss.detach().numpy())

        #projection
        for k in range(prim_var.size(1)):
            prim_var.data[:, k] = simplex_proj(prim_var.data[:,k], beta[k], device)
        iterations += 1
        

    if debug:
        print('{} dual iterations'.format(iterations))
    if return_losses:
        return prim_var.detach(), loss_profile[1:]
    else:
        return prim_var.detach()