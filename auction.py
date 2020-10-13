import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import sinkhorn
import os
import numpy as np
from train import simplex_proj


def LazySecondPriceLoss(net, input, y, size_batch, nb_opponents=1,
                        distribution="exponential", rp=0, eta=1000,
                        device="cpu", adv_type="exponential", adv_param=0.5):
    """
    Estimate the loss for training. When precisely evaluating the loss, set rp to 0 and eta to 10000.
    """
    _, output = net(input)
    output_grad = []
    true_val = input@y.view(1, -1)  # accurate value
    # we need the loss to be positive for stability with small lambda
    loss = torch.ones(net.nactions, y.size(0), device=device)
    for i, out in enumerate(output):
        grad = torch.autograd.grad(
            torch.sum(out), input, retain_graph=True,
            create_graph=True)[0].flatten()
        virtual = out - grad  # virtual value
        indicator = torch.sigmoid(eta*(virtual-rp))
        if adv_type == "uniform":
            # uniform on [0,1] adversary
            winning = torch.min(out, torch.ones(out.size()))
        elif adv_type == "exponential":
            winning = 1 - torch.exp(-out/adv_param)  # exp 1 adversary
        if nb_opponents > 1:
            # in case of several opponents;
            # If loop to optimize autograd for one opponent
            winning = winning**nb_opponents
        # utility given by Nedelec et al.
        l = (true_val - virtual[:, None])*winning[:, None]*indicator[:, None]
        loss[i, :] -= 1/size_batch*torch.sum(l, dim=0)

    return loss


class BidderStrategy(nn.Module):
    """
    Network simulating the strategy of a bidder
    """

    def __init__(self, size_layer=200, nactions=12, device="cpu"):
        super().__init__()
        self.size_layer = size_layer
        self.nactions = nactions

        self.fc1 = []
        # fc1[i] -> fc2[i] represents an action beta_i
        for i in range(self.nactions):
            self.fc1.append(nn.Linear(1, self.size_layer))
            torch.nn.init.uniform_(self.fc1[i].bias, a=-1.0, b=1.0)
            torch.nn.init.uniform_(self.fc1[i].weight, a=-1.0, b=2.0)
        self.fc1 = nn.ModuleList(self.fc1)

        # fc3 is just the parameters alpha (without hidden layer)
        self.fc3 = nn.Linear(1, self.nactions, bias=False)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

        self.fc2 = []
        for i in range(self.nactions):
            self.fc2.append(nn.Linear(self.size_layer, 1))
            torch.nn.init.normal_(self.fc2[i].weight, mean=2*(i+1)/(
                self.size_layer*self.nactions), std=0.01*(i+1)/self.nactions)
            torch.nn.init.normal_(
                self.fc2[i].bias, mean=0.0, std=0.001*(i+1)/self.nactions)
        self.fc2 = nn.ModuleList(self.fc2)
        self.device = device
        self.one = torch.FloatTensor([1]).to(device)
        self.proj = False

    def forward(self, inp):
        out = []
        for i in range(self.nactions):
            # compute beta_i(x) for every x
            out.append(self.fc2[i](F.relu(self.fc1[i](inp))).flatten())
        alpha = F.softmax(self.fc3(self.one), dim=0)
        return alpha, out

    def alpha(self):
        # use softmax for regularity/stability compared to projected gradient
        return F.softmax(self.fc3(self.one), dim=0)


def train_sinkhorn_auction(net, y, beta, lamb=1, niter_sink=5, max_iter=1000,
                           learning_rate=0.1, err_threshold=1e-4, experiment=0,
                           verbose=False, verbose_freq=100, device="cpu",
                           distribution="exponential", size_batch=2500,
                           differentation="automatic"):
    """
    learn a discrete distribution (alpha, x) with a prior (beta, y)
    differentation = "automatic" for AutoDiff method
                   = "analytic"  using dual solution in the differentation
    """
    start_time = timeit.default_timer()
    if experiment != 0:
        folder = 'experiments/sinkhorn_auction/'
        folder += '{}_lamb{}_k{}_'.format(experiment, lamb, y.size(0))
        folder += '_sinkiter{}_lr{}'.format(niter_sink, learning_rate)
        folder += '_sinkhorn_{}_batch{}'.format(device, size_batch)
        os.system('mkdir ' + folder)

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
    one = torch.FloatTensor([1]).to(device)

    distrib = torch.distributions.Exponential(
        torch.tensor(1.))  # exponential(1) distribution

    iterations = 0
    loss_profile = []

    # convergence of Sinkhorn is assumed if we use analytic differentation
    conv = (differentation == "analytic")

    while iterations < max_iter:
        # slowly decrease the learning rate
        optimizer.param_groups[-1]['lr'] *= 0.99

        # ---------------------------
        #        Optimize over net
        # ---------------------------

        input = torch.zeros((size_batch, 1), requires_grad=True)
        samples = distrib.sample((size_batch, 1))
        input.data = samples.clone()
        input = input.to(device)

        ###### Compute loss matrix ###
        C = LazySecondPriceLoss(net, input, y, size_batch,
                                distribution="exponential", device=device)

        ###### Sinkhorn loss #########
        alpha = net.alpha()
        # loss function
        loss, _ = sinkhorn.sinkhorn_loss_primal(alpha, C, beta, y, lamb,
                                                niter=niter_sink,
                                                cost="matrix",
                                                err_threshold=err_threshold,
                                                verbose=False,
                                                convergence=conv)
        loss.backward(one)  # autodiff
        optimizer.step()  # gradient step

        loss_profile.append(loss.cpu().detach().numpy())

        # projected gradient (if needed)
        if net.proj:
            net.projection()

        optimizer.zero_grad()
        iterations += 1

        if verbose:
            if iterations % verbose_freq == 0:
                print('Loss at iterations {}: {}'.format(
                    iterations, loss.cpu().detach().numpy()))

    if verbose:
        t_expe = (timeit.default_timer()-start_time)
        print('done in {0} s'.format(t_expe))
    if experiment != 0:
        # save data
        torch.save(net, folder+'/network')
        np.save(folder+'/losses.npy', loss_profile)
        l, _, _, pl, ul = eval(net, y, beta, lamb=lamb,
                               niter_sink=(int)(1e5), size_batch=(int)(1e6))
        np.save(folder + '/eval_loss.npy', l.detach().numpy())
        np.save(folder + '/eval_privacy_loss.npy', pl.detach().numpy())
        np.save(folder + '/eval_utility_loss.npy', ul.detach().numpy())
    return loss_profile


def eval(net, y, beta, lamb=1, niter_sink=5, err_threshold=1e-4,
         size_batch=100000, rp=0, eta=10000):
    """
    Evaluate the true loss by sampling with a larger batch size and removing the smoothing parameters.
    """
    distrib = torch.distributions.Exponential(torch.tensor(1.))
    input = torch.zeros((size_batch, 1), requires_grad=True)
    samples = distrib.sample((size_batch, 1))
    input.data = samples.clone()
    ###### Compute loss matrix ###
    C = LazySecondPriceLoss(net, input, y, size_batch,
                            distribution="exponential", rp=rp, eta=eta)

    ###### Sinkhorn loss #########
    alpha = net.alpha()
    _, gamma = sinkhorn.sinkhorn_loss_primal(
        alpha, C, beta, y, lamb, niter=niter_sink,
        cost="matrix", err_threshold=err_threshold, verbose=False)
    true_alpha = torch.sum(gamma, dim=1)
    p_loss = sinkhorn._KL(true_alpha, beta, gamma, epsilon=0)
    u_loss = torch.sum(gamma*C)
    loss = u_loss + lamb*p_loss
    return loss, gamma, C, p_loss, u_loss


if __name__ == '__main__':
    lambrange = np.concatenate(
        (np.linspace(0.0005, 1, 20), np.geomspace(0.0005, 1, 20), [0.01, 0.1]))
    lambrange = np.sort(lambrange[1:-1])
    lambrange = [0.1]
    niter_sink = 1000  # require a large niter_sink for small values of lamb
    niter = 1000
    lr = 0.01
    batch = 1000
    size_layer = 100
    K = 10
    nactions = 2+K
    y = torch.linspace(1./K, 1., K)  # discretization of uniform distribution
    beta = torch.ones(K)/K
    nexp = 1
    err_threshold = 1e-3
    device = "cpu"
    startexp = 11
    differentation = "automatic"
    # print all parameters in console
    print('Begin simulations on {}. Parameters:'.format(device))
    print('lambrange: {} to {} with {} points'.format(
        lambrange[0], lambrange[-1], len(lambrange)))
    print('Sinkhorn iterations: {}'.format(niter_sink))
    print('Descent iterations: {}'.format(niter))
    print('Learning_rate: {}'.format(lr))
    print('Batch size: {}'.format(batch))
    print('Layer size: {}'.format(size_layer))
    print('K: {}'.format(K))
    print('Number of experiments: {}'.format(nexp))
    print('Differentation: {}'.format(differentation))
    for lamb in lambrange:
        for exp in range(startexp, nexp+startexp):
            p = os.path.isfile(folder+'/eval_utility_loss.npy')
            if not(p):
                printf('Simulate for lambda={}'.format(lamb))
                print('experiment {}'.format(exp))
                net = BidderStrategy(size_layer=100, nactions=nactions)
                if net.proj:
                    net.projection()
                train_sinkhorn_auction(net, y, beta, lamb=lamb,
                                       niter_sink=niter_sink,
                                       max_iter=niter, device=device,
                                       learning_rate=lr, verbose=False,
                                       err_threshold=err_threshold,
                                       verbose_freq=100,
                                       size_batch=batch, experiment=exp,
                                       differentation=differentation)
