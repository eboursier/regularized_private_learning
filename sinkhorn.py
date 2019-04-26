import torch
from torch.autograd import Variable
import pdb

def _squared_distances(x,y, **kwargs) :
	"Returns the matrix of $|x_i-y_j|^2$."
	d = kwargs.get('dim', 2)
	x_col = x.unsqueeze(1) #x.dimshuffle(0, 'x', 1)
	y_lin = y.unsqueeze(0) #y.dimshuffle('x', 0, 1)
	c = torch.mul(x_col - y_lin, x_col-y_lin)
	c = c.sum(dim=d)
	return c 

def _linear_cost(x,y) :
	return torch.matmul(x,y.t())


def _pseudo_inv_dist(x, y, **kwargs):
	data = kwargs.get('data', None)
	if data is not None:
		x_d = torch.matmul(x, data)
		y_d = torch.matmul(y, data)
	else:
		x_d = x
		y_d = y

	x_d = torch.inverse(torch.matmul(torch.transpose(x_d, 1, 2), x_d))
	y_d = torch.inverse(torch.matmul(torch.transpose(y_d, 1, 2), y_d))
	return _squared_distances(x_d, y_d, **{'dim':(2,3)})

def _operator_dist(x, y):
	"""
	Compute the squared operator norm between two matrices (tensor version)
	"""
	C = torch.zeros(len(x), len(y))
	for i in range(len(x)):
		for j in range(len(y)):
			_, S, _ = torch.svd(x[i]-y[j], compute_uv=False)
			C[i,j] = S.max()
	return C*C


def _KL(alpha, beta, Gamma, epsilon=1e-3) :
	"return the KL privacy cost with the joint distribution Gamma, with marginales (alpha, x) and (beta, y)"
	P = torch.mm((alpha+epsilon/len(alpha)).unsqueeze(1), beta.unsqueeze(0)) # add epsilon for stability
	Z = Gamma * torch.log((Gamma+epsilon/(Gamma.size(0)*Gamma.size(1)))/P)
	if epsilon==0:
		Z[torch.isnan(Z)] = 0 # correction for 0 log(0)
	return torch.sum(Z)


def sinkhorn_loss_primal(alpha,x,beta,y,lamb,niter=1000, cost=_squared_distances, err_threshold=1e-5, verbose=False, epsilon=1e-6, **kwargs) :
	
	"""
	Given two discrete measures (alpha,x) and (beta,y) 
	outputs an approximation of the OT cost with regularization parameter epsilon
	niter is the max. number of steps in sinkhorn loop
	"""
	C = cost(x, y, **kwargs) # cost matrix

	def M(u,v)  : 
		"$M_{ij} = (-c_{ij} + u_i + v_j) / \lambda$"
		return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / lamb

	lse = lambda A    : torch.log(torch.exp(A).sum( 1, keepdim = True ) + 1e-7) # slight modif to prevent NaN
	
	# Actual Sinkhorn loop ......................................................................
	u,v,err = 0.*(alpha+epsilon/len(alpha)), 0.*beta, torch.sum(torch.Tensor([float('inf')])) # add epsilon to alpha for stability
	iterations = 0

	while (iterations<niter) and (torch.abs(err)>err_threshold):
		u1 = u # check the update
		
		u =  lamb * ( torch.log(alpha+epsilon/len(alpha)) - lse(M(u,v)).squeeze() ) + u
		v =  lamb * ( torch.log(beta) - lse(M(u,v).t()).squeeze()) + v 
		err = (u - u1).abs().sum()

		iterations += 1


	Gamma = torch.exp( M(u,v) )            # Eventual transport plan g = diag(a)*K*diag(b)
	cost  = torch.sum( Gamma * C )  + lamb*_KL(alpha, beta, Gamma)  # total cost
	if verbose:
		print('Iteration until convergence:', iterations)
	return cost, Gamma