import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.decomposition import NMF

from data_utils import get_matrices, get_topics


err = 1e-16 #a small error to prevent 0/0
iteration = 100 #number of iterations
K = 25 #number of categories (matrix rank)

X, W, H = get_matrices(K)

start = time.time()

model = NMF(n_components=K, init='random', max_iter=iteration)
W = model.fit_transform(X, W=W, H=H)
H = model.components_

end = time.time()

print("Trained in {} s".format(end - start))

get_topics(W)
