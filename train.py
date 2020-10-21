import torch
from torch.autograd import Variable
import timeit
import torch.nn as nn
import torch.nn.functional as F
import os
import sinkhorn

import numpy as np


class Net(nn.Module):
    def __init__(self, decoder):
        super(Net, self).__init__()
        self.decoder = decoder

    def forward(self, input):
        output = self.decoder(input)
        return output


def simplex_proj(x, p=1, device="cpu"):
    """
    compute the projection of x onto the simplex defined by
    x_i>=0 and sum x_i <= p.
    See https://arxiv.org/abs/1101.6081
    """
    u, _ = torch.sort(x, descending=True)
    cs = u.cumsum(0) - p
    ind = torch.arange(len(x), dtype=torch.float, device=device) + 1
    cond = u - cs / ind > 0
    rho = ind[cond][-1]
    theta = cs[cond][-1] / rho
    return F.relu(x - theta)

# train parameters of net with the Sinkhorn scheme


def train_sinkhorn(net, y, beta, lamb=1, niter_sink=1, max_time=10,
                   cost=sinkhorn._squared_distances, experiment=0,
                   differentiation="analytic", learning_rate=0.1,
                   err_threshold=1e-4, verbose=False, verbose_freq=100,
                   device="cpu", optim="descent", warm_restart=False,
                   **kwargs):
    """
    learn a discrete distribution (alpha, x) with a prior (beta, y)
    """
    momentum = kwargs.get('momentum', 0)
    momstring = "_{}".format(momentum) if momentum != 0 else ""
    restart_string = "_warm" if warm_restart else ""
    actionstr = "_actions{}".format(net.nactions) if (
        net.nactions != y.size(0)+2) else ""

    # if experiment !+0, save the simulation
    fold = '{}_lamb{}_k{}{}_dim{}_sinkiter{}_lr{}'.format(experiment, lamb,
                                                          y.size(0), actionstr,
                                                          y.size(1),
                                                          niter_sink,
                                                          learning_rate)
    fold += '_sinkhorn_{}_{}_{}{}{}'.format(device, optim,
                                            differentiation,
                                            momstring,
                                            restart_string)
    if experiment != 0:
        os.system('mkdir experiments/sinkhorn/' + fold)

    # optimizer choice
    if optim == "SGD":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=learning_rate, momentum=momentum)
    elif optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optim == "rms":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    else:
        print('Invalid choice of optimizer.')
        return None
    one = torch.FloatTensor([1]).to(device)

    iterations = 0
    loss_profile = []  # evolution of the loss
    time_profile = []  # evolution of the training time
    if verbose:
        u_profile = []
        v_profile = []  # evolution of u and v (sinkhorn dual variables)
        iteration_profile = []  # number of iterations per sinkhorn call
        alpha_profile = []
        x_profile = []
    start_time = timeit.default_timer()
    running_time = 0
    # convergence of Sinkhorn is assumed if we use analytic differentiation
    conv = (differentiation == "analytic")

    while running_time < max_time:  # remains time to train

        # ---------------------------
        #        Optimize over net
        # ---------------------------
        time = timeit.default_timer()
        optimizer.zero_grad()

        # if decreasing_rate:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = learning_rate/(iterations+1)

        alpha, x = net(one)  # output of the net (parameters to optimize)

        ###### Sinkhorn loss #########
        if iterations == 0:
            # compute the loss
            z = sinkhorn.sinkhorn_loss_primal(alpha, x, beta, y, lamb,
                                              niter=niter_sink,
                                              verbose=verbose,
                                              cost=cost, convergence=conv,
                                              err_threshold=err_threshold,
                                              warm_restart=warm_restart)

        else:
            # compute the loss
            z = sinkhorn.sinkhorn_loss_primal(alpha, x, beta, y, lamb,
                                              niter=niter_sink,
                                              verbose=verbose,
                                              cost=cost, convergence=conv,
                                              err_threshold=err_threshold,
                                              warm_restart=warm_restart,
                                              u=u, v=v)

        # if verbose:
        #     loss, _, u, v, sink_iter = z
        # else:
        loss, _, u, v = z

        loss.backward(one)  # automatic differentiation
        optimizer.step()  # gradient descent step

        # if projected gradient
        if net.proj:
            net.projection()

        # for the sinkhorn method, it takes some time to compute the accurate
        # loss from the estimated loss in the training
        running_time += (timeit.default_timer()-time)
        # because we need to do sinkhorn algorithm with more iterations
        # (as niter_sink is small). In this case, we compute the true loss
        # for the plot but it does not count in the running time
        time_profile.append(running_time)

        # compute the true loss for plots and does not count in running time

        # 100 is enough with the chosen parameters in the experiments
        _, gamma, _, _ = sinkhorn.sinkhorn_loss_primal(
            alpha, x, beta, y, lamb, niter=100, cost=cost, err_threshold=1e-4)
        loss_p = torch.sum(gamma*cost(x, y)) + lamb * \
            sinkhorn._KL(alpha, beta, gamma, epsilon=0)
        # print(gamma.shape)
        # print(torch.sum(gamma))
        loss_profile.append(loss_p.cpu().detach().numpy())

        iterations += 1

        if verbose:
            u_profile.append(u.detach().numpy())
            v_profile.append(v.detach().numpy())
            iteration_profile.append(sink_iter)
            alpha_profile.append(alpha.detach().numpy())
            x_profile.append(x.detach().numpy())
            if iterations % verbose_freq == 0:
                print('iterations='+str(iterations))

    if verbose:
        t_expe = (timeit.default_timer()-start_time)
        print('done in {0} s'.format(t_expe))
        print('total running time: {0} s'.format(running_time))
    if experiment != 0:
        # save data
        torch.save(net, 'experiments/sinkhorn/'+fold+'/network')
        np.save('experiments/sinkhorn/'+fold+'/losses.npy', loss_profile)
        np.save('experiments/sinkhorn/'+fold+'/time.npy', time_profile)
    if verbose:
        u_profile = np.array(u_profile)
        v_profile = np.array(v_profile)
        iteration_profile = np.array(iteration_profile)
        alpha_profile = np.array(alpha_profile)
        x_profile = np.array(x_profile)
        return (loss_profile, time_profile, u_profile,
                v_profile, iteration_profile,
                x_profile, alpha_profile)
    else:
        return loss_profile


def train_descent(net, y, beta, lamb=1, max_time=10,
                  cost=sinkhorn._squared_distances, learning_rate=0.1,
                  experiment=0, verbose=False, verbose_freq=100,
                  device="cpu", optim="descent",  **kwargs):
    """
    learn a discrete distribution (gamma, x) with a prior (beta, y) using gradient descent on (gamma, x). Similar structure than train_sinkhorn
    """
    momentum = kwargs.get('momentum', 0)
    momstring = "_{}".format(momentum) if momentum != 0 else ""
    actionstr = "_actions{}".format(net.nactions) if (
        net.nactions != y.size(0)+2) else ""

    fold = '{}_lamb{}{}_k{}_dim{}_lr{}_descent_{}_{}{}'.format(
        experiment, lamb, actionstr, y.size(0), y.size(1), learning_rate,
        device, optim, momstring)
    if experiment != 0:
        os.system('mkdir experiments/descent/'+fold)

    # optimizer choice
    if optim == "SGD":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=learning_rate,
            momentum=momentum)
    elif optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optim == "rms":
        optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    else:
        print('Invalid choice of optimizer.')
        return None
    one = torch.FloatTensor([1]).to(device)

    iterations = 0
    loss_profile = []
    time_profile = []
    start_time = timeit.default_timer()
    running_time = 0
    while running_time < max_time:

        # ---------------------------
        #        Optimize over net
        # ---------------------------

        time = timeit.default_timer()
        optimizer.zero_grad()
        gamma, x = net(one)  # output of the network. Parameters to optimize
        alpha = torch.sum(gamma, dim=1)

        ###### Total loss #########
        C = cost(x, y)

        loss = torch.sum(gamma * C) + lamb * \
            sinkhorn._KL(alpha, beta, gamma)  # loss
        loss.backward(one)  # autodiff
        optimizer.step()  # gradient descent step

        if net.proj:  # for projected gradient
            net.projection()

        # reasons explained in train_sinkhorn
        running_time += (timeit.default_timer()-time)
        time_profile.append(running_time)

        # here the training loss is accurate
        loss_profile.append(loss.cpu().detach().numpy())

        iterations += 1

        if verbose:
            if iterations % verbose_freq == 0:
                print('iterations='+str(iterations))

    if verbose:
        t_expe = (timeit.default_timer()-start_time)
        print('done in {0} s'.format(t_expe))
        print('total running time: {0} s'.format(running_time))
    if experiment != 0:
        # save data
        torch.save(net, 'experiments/descent/'+fold+'/network')
        np.save('experiments/descent/'+fold+'/losses.npy', loss_profile)
        np.save('experiments/descent/'+fold+'/time.npy', time_profile)
    return loss_profile


def train_dc(net, y, beta, lamb=1, max_time=10, err_threshold=1e-4,
             cost=sinkhorn._squared_distances,  dual_iter=100, debug=False,
             learning_rate=0.01, experiment=0, verbose=False,
             verbose_freq=100, device="cpu", **kwargs):
    """
    learn a discrete distribution (gamma) with a prior (beta, y) using DCA algorithm.
    """
    actionstr = "_actions{}".format(net.nactions) if (
        net.nactions != y.size(0)+2) else ""

    fold = '{}_lamb{}{}_k{}_dim{}_dualiter{}_lr{}_dc_{}'.format(
        experiment, lamb, actionstr, y.size(0), y.size(1),
        dual_iter, learning_rate, device)
    if experiment != 0:
        os.system('mkdir experiments/dc/'+fold)

    one = torch.FloatTensor([1]).to(device)

    iterations = 0
    loss_profile = []
    time_profile = []
    start_time = timeit.default_timer()
    running_time = 0
    while running_time < max_time:

        # ---------------------------
        #        Optimize over net
        # ---------------------------
        time = timeit.default_timer()
        # gamma is the parameter to optimize (
        # the best x for a given gamma is automatically computed here)
        gamma, x = net(one)
        dual_var = -torch.mm(x, y.t())  # dual iteration of DCA
        # primal iteration of DCA
        gamma_it = solve_relaxed_primal(dual_var, beta, gamma,
                                        lamb=lamb, max_iter=dual_iter,
                                        learning_rate=learning_rate,
                                        err_threshold=err_threshold,
                                        debug=debug, device=device)

        # reasons explained in train_sinkhorn
        running_time += (timeit.default_timer()-time)
        time_profile.append(running_time)

        alpha = torch.sum(gamma_it, 1)
        # C = cost(x, y, **kwargs)
        # loss = torch.sum( gamma_it * C )  + lamb*sinkhorn._KL(alpha, beta, gamma_it)
        loss = lamb*sinkhorn._KL(alpha, beta, gamma_it) - \
            torch.sum(gamma_it*dual_var)  # acurate loss
        loss_profile.append(loss.cpu().detach().numpy())
        net.gamma = gamma_it
        iterations += 1

        if verbose:
            if iterations % verbose_freq == 0:
                print('iterations='+str(iterations))
    if verbose:
        t_expe = (timeit.default_timer()-start_time)
        print('done in {0} s'.format(t_expe))
        print('total running time: {0} s'.format(running_time))
    if experiment != 0:
        # save data
        torch.save(net, 'experiments/dc/'+fold+'/network')
        np.save('experiments/dc/'+fold+'/losses.npy', loss_profile)
        np.save('experiments/dc/'+fold+'/time.npy', time_profile)
    return loss_profile


def relax_primal_loss(primal_var, dual_var, beta, lamb):
    """
    Loss to minimize in the primal iteration of the DCA algorithm
    """
    alpha = torch.sum(primal_var, 1)
    g = lamb*sinkhorn._KL(alpha, beta, primal_var)
    h_relax = torch.sum(primal_var*dual_var)
    return g-h_relax


def solve_relaxed_primal(dual_var, beta, gamma, lamb=1, max_iter=100,
                         learning_rate=0.01, err_threshold=1e-4,
                         return_losses=False, debug=False, device="cpu"):
    """
    Find an approximated solution of inf(lambda KL(gamma, gamma_1 \times beta) - <gamma, dual_var>) in the feasible set of gamma using projected gradient.
    Primal iteration of the DCA algorithm.
    """

    prim_var = gamma.clone()
    prim_var.requires_grad_(True)
    optimizer = torch.optim.SGD([prim_var], lr=learning_rate, momentum=0)
    # initialize inf with tensor type
    loss = torch.sum(torch.Tensor([float('inf')]))
    old_loss = 0

    if return_losses:
        loss_profile = []

    iterations = 0
    while (iterations < max_iter) and \
            (torch.abs(old_loss-loss) > err_threshold):

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

        # projection
        for k in range(prim_var.size(1)):
            prim_var.data[:, k] = simplex_proj(
                prim_var.data[:, k], beta[k], device)
        iterations += 1

    if debug:
        print('{} dual iterations'.format(iterations))
    if return_losses:
        return prim_var.detach(), loss_profile[1:]
    else:
        return prim_var.detach()
