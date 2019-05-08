import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
import sinkhorn
from train import simplex_proj

def LazySecondPriceLoss(net, input, y, size_batch,nb_opponents=1, distribution="exponential", rp=0, eta=1000, device="cpu"):
  """
  Estimate the loss for training. When precisely evaluating the loss, set rp to 0 and eta to 100.
  """
  _, output = net(input)
  output_grad = []
  true_val = input@y.view(1,-1)
  loss = torch.ones(net.nactions, y.size(0), device=device)  # we need the loss to be positive for stability with small lambda
  for i, out in enumerate(output):
      grad = torch.autograd.grad(torch.sum(out),input,retain_graph=True, create_graph=True)[0].flatten()
      virtual = out - grad
      indicator = torch.sigmoid(eta*(virtual-rp))
      winning = torch.ones(out.size(), device=device)
      if nb_opponents>1:
        winning = winning**nb_opponents # in case of several opponents; If loop to optimize autograd for one opponent
      l = (true_val - virtual[:, None])*winning[:, None]*indicator[:, None]
      loss[i,:] -= 1/size_batch*torch.sum(l, dim=0)

  return loss

class BidderStrategy(nn.Module):
    def __init__(self, size_layer = 200, nactions=12, device="cpu"):
        super().__init__()
        self.size_layer = size_layer
        self.nactions = nactions

        self.fc1 = []
        for i in range(self.nactions):
          self.fc1.append(nn.Linear(1, self.size_layer))
          torch.nn.init.uniform_(self.fc1[i].bias,a=-1.0, b=1.0)
          torch.nn.init.uniform_(self.fc1[i].weight,a=-1.0, b=2.0)
        self.fc1 = nn.ModuleList(self.fc1)
        
        self.fc3 = nn.Linear(1, self.nactions, bias=False)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

        self.fc2 = []
        for i in range(self.nactions):
          self.fc2.append(nn.Linear(self.size_layer, 1))
          torch.nn.init.normal_(self.fc2[i].weight,mean=2/self.size_layer, std = 0.01)
          torch.nn.init.normal_(self.fc2[i].bias,mean=0.0, std = 0.001)
        self.fc2 = nn.ModuleList(self.fc2)
        self.device = device
        self.one = torch.FloatTensor([1]).to(device)
        self.proj = True

    def forward(self, inp):
      out = []
      for i in range(self.nactions):
        out.append(self.fc2[i](F.relu(self.fc1[i](inp))).flatten())
      alpha = self.fc3(self.one)
      return alpha, out

    def alpha(self):
      return F.softmax(self.fc3(self.one), dim=0)

    def projection(self):
      self.fc3.weight.data = simplex_proj(self.fc3.weight.data.flatten(), device=self.device).view(-1, 1)



def train_sinkhorn_auction(net, y, beta, lamb=1, niter_sink = 5, max_iter=1000, learning_rate=0.1, err_threshold=1e-4, experiment=0, verbose=False, 
                            verbose_freq=100, device="cpu", distribution="exponential", size_batch=2500):
    """
    learn a discrete distribution (alpha, x) with a prior (beta, y)
    """
    start_time = timeit.default_timer()
    if experiment!=0:
        os.system('mkdir experiments/sinkhorn_auction/{0}_lamb{1}_k{2}_sinkiter{4}_lr{5}_sinkhorn_{6}_batch{7}'.format(experiment,lamb,y.size(0), 
                    y.size(1), niter_sink, learning_rate, device, size_batch))

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0)
    one = torch.FloatTensor([1]).to(device)

    distrib = torch.distributions.Exponential(torch.tensor(1.))

    iterations = 0
    loss_profile = []
    while iterations<max_iter:
      optimizer.param_groups[-1]['lr']*=0.99

      # ---------------------------
      #        Optimize over net
      # ---------------------------

      input = torch.zeros((size_batch, 1), requires_grad=True)
      samples = distrib.sample((size_batch, 1))
      input.data = samples.clone()
      input = input.to(device)

      ###### Compute loss matrix ###
      C = LazySecondPriceLoss(net, input, y, size_batch, distribution="exponential", device=device)

      ###### Sinkhorn loss #########
      alpha = net.alpha()
      loss, _ = sinkhorn.sinkhorn_loss_primal(alpha, C, beta, y, lamb, niter=niter_sink, cost="matrix", err_threshold=err_threshold, verbose=False)
      loss.backward(one)
      optimizer.step()

      loss_profile.append(loss.cpu().detach().numpy())

      # projected gradient
      if net.proj:
          net.projection()

      optimizer.zero_grad()
      iterations += 1
      
      if verbose:
          if iterations%verbose_freq == 0:
             print('Loss at iterations {}: {}'.format(iterations, loss.cpu().detach().numpy()))

    if verbose:
        t_expe = (timeit.default_timer()-start_time)
        print('done in {0} s'.format(t_expe))
    if experiment!=0:
        # save data
        torch.save(net, 'experiments/sinkhorn_auction/{0}_lamb{1}_k{2}_sinkiter{4}_lr{5}_sinkhorn_{6}_batch{7}/network'.format(experiment,lamb,y.size(0), 
                    y.size(1), niter_sink, learning_rate, device, size_batch))
        np.save('experiments/sinkhorn_auction/{0}_lamb{1}_k{2}_sinkiter{4}_lr{5}_sinkhorn_{6}_batch{7}/losses.npy'.format(experiment,lamb,y.size(0), 
                    y.size(1), niter_sink, learning_rate, device, size_batch), loss_profile)
        l, _, _ = eval(net, y, beta, lamb=lamb, niter_sink=niter_sink, size_batch=(int)(10**6))
        np.save('experiments/sinkhorn_auction/{0}_lamb{1}_k{2}_sinkiter{4}_lr{5}_sinkhorn_{6}_batch{7}/eval.npy'.format(experiment,lamb,y.size(0), 
                    y.size(1), niter_sink, learning_rate, device, size_batch), l)
    return loss_profile

def eval(net, y, beta, lamb=1, niter_sink = 5, err_threshold=1e-4, size_batch=2500, rp=0, eta=10000):
  distrib = torch.distributions.Exponential(torch.tensor(1.))
  input = torch.zeros((size_batch, 1), requires_grad=True)
  samples = distrib.sample((size_batch, 1))
  input.data = samples.clone()
  ###### Compute loss matrix ###
  C = LazySecondPriceLoss(net, input, y, size_batch, distribution="exponential", rp=rp, eta=eta)

  ###### Sinkhorn loss #########
  alpha = net.alpha()
  _, gamma = sinkhorn.sinkhorn_loss_primal(alpha, C, beta, y, lamb, niter=niter_sink, cost="matrix", err_threshold=err_threshold, verbose=False)
  loss = torch.sum(gamma*C) + lamb*sinkhorn._KL(alpha, beta, gamma, epsilon=0)
  return loss, gamma, C
