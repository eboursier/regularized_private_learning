# regularized_private_learning

Code used for the simulations of the Paper "Utility/Privacy Trade-off through the lens of Optimal Transport".

Parts of the code are adapted from https://github.com/audeg/Sinkhorn-GAN and https://github.com/toma5692/learning_to_bid_ICML. Credits to them.

To make it work, first create the following repositories: experiments/; figures/; experiments/dc; experiments/descent; experiments/sinkhorn; experiments/sinkhorn_auction


For the first example, first run linear_run.py and then plot the results using linear_toy_plot.ipynb.
For the second, first run auction.py and then plot the results using auction_plot.ipynb

Required libraries:
numpy; 
pytorch; 
timeit; 
matplotlib; 
