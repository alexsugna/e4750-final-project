from nmf import NMF, plot_loss
from data_utils import get_matrices, get_topics

err = 1e-16 #a small error to prevent 0/0
iteration = 10 #number of iterations
K = 25 #number of categories (matrix rank)

X, W, H = get_matrices(K)

W, H, loss, time = NMF(X, W, H, iterations=iteration, loss='divergence', eps=err, return_time=True)
print("Trained in {} ms".format(time))

#plot the objective function, should monotonically decrease to a local minimum
plot_loss(loss, "Divergence Loss of NMF", loss_type="Divergence")

get_topics(W)
