import torch
from torch.autograd import Variable
import timeit

# inspired by the code from Interpolating between Optimal Transport
# and MMD using Sinkhorn Divergences by Feydy et al
# and Learning Generative Models with Sinkhorn Divergences from Genevay et al


def _squared_distances(x, y, **kwargs):
    """
    Returns the matrix of $|x_i-y_j|^2$.
    """
    d = kwargs.get('dim', 2)
    x_col = x.unsqueeze(1)  # x.dimshuffle(0, 'x', 1)
    y_lin = y.unsqueeze(0)  # y.dimshuffle('x', 0, 1)
    c = torch.mul(x_col - y_lin, x_col-y_lin)
    c = c.sum(dim=d)
    return c

# linear cost used for the first example


def _linear_cost(x, y):
    return torch.matmul(x, y.t())


def _KL(alpha, beta, Gamma, epsilon=1e-3):
    """
    return the KL privacy cost with the joint distribution Gamma, with marginales (alpha, x) and (beta, y)
    """
    P = torch.mm((alpha+epsilon/len(alpha)).unsqueeze(1),
                 beta.unsqueeze(0))  # add epsilon for stability
    Z = Gamma * torch.log((Gamma+epsilon/(Gamma.size(0)*Gamma.size(1)))/P)
    if epsilon == 0:
        Z[torch.isnan(Z)] = 0  # 0 log(0) = 0
    return torch.sum(Z)


def _scal(alpha, f):
    return torch.dot(alpha.view(-1), f.view(-1))


def _lse(v_ij, epsilon=1e-6):
    """[lse(v_ij)]_i = log sum_j exp(v_ij), with numerical accuracy."""
    V_i = torch.max(v_ij, 1)[0].view(-1, 1)
    # add epsilon for numerical stability
    return V_i + ((v_ij - V_i).exp().sum(1)+epsilon).log().view(-1, 1)


def _Sinkhorn_ops(lamb, x, y, cost=_squared_distances, epsilon=1e-6, **kwargs):
    """
    Given:
    - an exponent p = 1 or 2
    - a regularization strength ε > 0
    - point clouds x_i and y_j, encoded as N-by-D and M-by-D torch arrays,

    Returns a pair of routines S_x, S_y such that
      [S_x(f_i)]_j = -log sum_i exp( f_i - |x_i-y_j|^p / ε )
      [S_y(f_j)]_i = -log sum_j exp( f_j - |x_i-y_j|^p / ε )
    """

    # We precompute the |x_i-y_j|^p matrix once and for all...
    if cost == "matrix":
        # directly sent the cost matrix instead of x and y. Used for auctions.
        C = x/lamb
    else:
        C = cost(x, y, **kwargs)/lamb  # cost matrix
    CT = C.t()

    # Before wrapping it up in a simple pair
    # of operators - don't forget the minus!
    def S_x(f_i): return -_lse(f_i.view(1, -1) - CT, epsilon=epsilon)

    def S_y(f_j): return -_lse(f_j.view(1, -1) - C, epsilon=epsilon)
    return S_x, S_y


def sinkhorn_loss_primal(alpha, x, beta, y, lamb, niter=1000,
                         cost=_squared_distances, err_threshold=1e-5,
                         verbose=False, epsilon=1e-6, convergence=True,
                         warm_restart=False, **kwargs):
    """
    Given two discrete measures (alpha,x) and (beta,y) 
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    def M(u, v, C):
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \lambda$"
        return (-C + u.unsqueeze(0) + v.unsqueeze(1))/lamb

    # Precompute the logs of the measures' weights
    alpha_log = (alpha+epsilon).log().view(-1, 1)
    beta_log = (beta+epsilon).log().view(-1, 1)

    # warm restart trick to speed up Sinkhorn convergence
    if warm_restart and convergence:
        v = kwargs.get('v', torch.zeros_like(alpha_log))
        u = kwargs.get('u', torch.zeros_like(beta_log))
    else:
        v, u = torch.zeros_like(alpha_log), torch.zeros_like(
            beta_log)  # Sampled influence fields

    # do not use autodiff for this step if convergence is set to True
    torch.set_grad_enabled(not convergence)

    # Softmin operators (divided by ε, as it's slightly cheaper...)
    S_x, S_y = _Sinkhorn_ops(lamb, x, y, cost, epsilon=epsilon)

    iterations = 0
    err = err_threshold + 1.

    # stop if niter or convergence
    while (iterations < niter-1) and (err > err_threshold):
        u1 = u  # check the update

        u = S_x(v + alpha_log)
        v = S_y(u + beta_log)
        err = (u - u1).abs().sum()

        iterations += 1

    torch.set_grad_enabled(True)
    # One last step, which allows us to bypass PyTorch's backprop engine
    # if required (as explained in the paper by Feydy et al)
    if not convergence:
        u = S_x(v + alpha_log)
        v = S_y(u + beta_log)
    # Assume that we have converged, and can thus use
    # the "exact" (and cheap!) gradient's formula
    else:
        S_x, _ = _Sinkhorn_ops(lamb, x.detach(), y, cost)
        _, S_y = _Sinkhorn_ops(lamb, x, y.detach(), cost)
        u = S_x((v + alpha_log).detach())
        v = S_y((u + beta_log).detach())

    iterations += 1

    a, b = lamb*u.view(-1), lamb*v.view(-1)  # corresponds to g, f

    if cost == "matrix":
        # directly sent the cost matrix instead of x and y. Used for auctions.
        C = x
    else:
        C = cost(x, y)  # cost matrix

    # optimal transport plan g = diag(a)*K*diag(b)
    Gamma = torch.exp(M(a, b, C))*alpha.unsqueeze(1)*beta.unsqueeze(0)
    loss = _scal(alpha, b) + _scal(beta, a)
    # if verbose:
    #     return loss, Gamma, u, v, iterations
    # else:
    return loss, Gamma, u, v
