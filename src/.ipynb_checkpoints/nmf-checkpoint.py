"""
Our serial implementation of NMF
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from loss_functions import euclidean_loss_numpy, divergence_loss_numpy


def NMF(X, W, H, iterations=100, loss='euclidean', eps=1e-16, return_time=True):
    """
    Performs NumPy (serial) NMF.
    
    params:
        X (N, M): the original data matrix
        
        W (N, K): the W matrix (of articles by topic)
        
        H (K, M): the H matrix (of topics by word)
        
        iterations=100 (int): the number of matrix factorization updates
        
        loss='euclidean' (string): one of ['euclidean', 'divergence'] to specify the loss function
        
        eps=1e-16 (float): small value epsilon added for numerical stability
        
        return_time=True (bool): return the execution time in ms
        
    returns:
        W (N, K): the factored matrix W
        
        H (K, M): the factored matrix H
        
        squared_out (list): List of loss at each iteration
    """
    if return_time:
        start = time.time()
        
    losses = [] #keep track of objective function for each iteration
    
    for i in range(1, iterations+1):
        if i % 10 == 0:
            print('iteration %d' % i)

        #objective function
        if loss == 'euclidean':
            
            Wt = W.T #w transpose
            H = H * Wt.dot(X) / (Wt.dot(W).dot(H) + eps) #update H

            Ht = H.T #H transpose
            W = W * X.dot(Ht) / (W.dot(H).dot(Ht) + eps) #update W
            
            iter_loss = euclidean_loss_numpy(X, W, H)
            
        elif loss == 'divergence': 
            
            P = X / (W.dot(H)+eps) #intermediate step (purple matrix in notes)
            Wt = W.T
            Wt = Wt / Wt.sum(axis=1).reshape(-1, 1) #normalize rows
            H = H * (Wt.dot(P))  #update H
            P = X / (W.dot(H)+eps)
            Ht = H.T
            Ht = Ht / Ht.sum(axis=0).reshape(1, -1) #normalize columns
            W = W * (P.dot(Ht))  #update W
            
            iter_loss = divergence_loss_numpy(X, W, H, eps)

        else:
            raise Exception('Loss function "{}" not supported.'.format(loss))
            
        losses.append(iter_loss)
        
    if return_time:
        end = time.time()
        return W, H, losses, (end-start)*1e3
    
    return W, H, losses


def plot_loss(loss, title, xlab="Iterations", ylab="Loss", loss_type='Euclidean Distance'):
    """
    Makes a plot of the given loss history.
    """
    fig = plt.figure()
    plt.plot(np.arange(1, len(loss) + 1), loss, label=loss_type)
    plt.grid()
    plt.legend()
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(title)
    plt.savefig('{}.png'.format(title))
        
    