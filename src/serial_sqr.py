import matplotlib.pyplot as plt

from data_utils import get_matrices, get_topics
from nmf import NMF, plot_loss

eps = 1e-16 #a small error to prevent 0/0
iteration = 100 #number of iterations
K = 25 #number of categories (matrix rank)

X, W, H = get_matrices(K)

W, H, squared_out, execution_time = NMF(X, W, H, iterations=10, eps=eps, return_time=True)

print("Trained in {} ms".format(execution_time))

#plot the objective function, should monotonically decrease to a local minimum
plot_loss(squared_out, "Euclidean Loss of NMF")

get_topics(W)