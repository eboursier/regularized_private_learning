# regularized_private_learning

Code used for the simulations of the paper "Utility/Privacy Trade-off through the lens of Optimal Transport" accepted at AISTATS2020.

Parts of the code are adapted from https://github.com/audeg/Sinkhorn-GAN and https://github.com/jeanfeydy/geomloss and https://github.com/toma5692/learning_to_bid_ICML. Credits to them.

To make it work, first create the following repositories: experiments/; figures/; experiments/dc; experiments/descent; experiments/sinkhorn; experiments/sinkhorn_auction

For the first example, first run linear_run.py and then plot the results using linear_toy_plot.ipynb (after modifications, the run parameters have to be changed to fit those used in the paper, with automatic differentiation)
For the second, first run auction.py and then plot the results using auction_plot.ipynb (after modifications, precise to use automatic differentiation to fit the paper experiments)

Required libraries:
numpy; 
pytorch; 
timeit; 
matplotlib; 
