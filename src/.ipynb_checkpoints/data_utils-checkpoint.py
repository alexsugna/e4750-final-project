"""
Reads txt dataset and returns as NumPy Arrays
"""

import pandas as pd
import numpy as np

DATA_PATH = 'data/nyt_data.txt'
VOCAB_PATH = 'data/nyt_vocab.dat'

DATA = pd.read_csv(DATA_PATH, sep='\n', header=None)
VOCAB = pd.read_csv(VOCAB_PATH, sep='\n', header=None, names=['words'])


def get_matrices(K, random_seed=0):
    """
    Return the matrices X, W, and H for NMF where X is generated from 
    the NYT vocab dataset and W and H are generated with a random seed.
    
    params:
        K (int): number of factored categories (components)
        
        random_seed=0 (int): the seed for random matrix generation
        
        TODO: in the future when we add another dataset, include argument to 
              specify the dataset
        
    returns:
        X (N, M): the original data matrix
        
        W (N, K): the W matrix (of articles by topic)
        
        H (K, M): the H matrix (of topics by word)
    """
    M = DATA.shape[0] #number of words
    N = VOCAB.shape[0] #number of articles
    X = np.zeros((N, M)) #define the original data matrix
    
    #load data, every row is a single document
    #format “idx:cnt” with commas separating each unique wXord in the document
    for column in range(M):
        row = DATA.iloc[column].values[0].split(',')
        for item in row:
            index, count = map(int, item.split(':'))
            X[index-1][column] = count
    
    np.random.seed(random_seed)

    W = np.random.uniform(1, 2, (N, K)) #initialize W and H to random values between 1 and 2
    H = np.random.uniform(1, 2, (K, M))
    
    return X.astype(np.float32), W.astype(np.float32), H.astype(np.float32)


def get_topics(W, top_n=10, print_results=True):
    """
    Visualize the matrix factorization results.
    
    params:
        W (N, K): the factored W matrix (of articles by topic)
        top_n=10 (int): the number of relevant elements to show in each category
        print_results=True (bool): if true, DataFrame is printed, else it is returned
        
    returns:
        data (pd.DataFrame): the top_n most likely elements for each category K_i
    """
    W = W / W.sum(axis=0).reshape(1,-1) #normalize the categories
    
    K = W.shape[1]
    
    data = pd.DataFrame(index=range(top_n), columns=['Topic_%d' % i for i in range(1, K+1)])
    
    for i in range(K):
        column = 'Topic_' + str(i+1)
        Wi = W[:, i]
        dt = pd.DataFrame(Wi, columns=['weight'])
        dt['words'] = VOCAB
        dt = dt.sort_values(by='weight', ascending=False)[:top_n].reset_index(drop=True)
        data[column] = dt['weight'].map(lambda x: ('%.4f')%x) + ' ' + dt['words']

    if print_results:
        print(data)
        
    return data


def get_n_matrices(K, X_shape, random_seed=0):
    """
    Return the matrices X, W, and H for NMF where X of size X_shape
    is generated randomly.
    
    params:
        K (int): number of factored categories (components)
        
        X_shape (tuple of ints): dimensions of X (N, M)
        
        random_seed=0 (int): the seed for random matrix generation
        
    returns:
        X (N, M): randomly generated matrix
        
        W (N, K): the W matrix 
        
        H (K, M): the H matrix 
    """
    np.random.seed(random_seed) # initialize random seed
    
    X = np.random.randint(0, high=10, size=X_shape) # generate random X

    W = np.random.uniform(1, 2, (X_shape[0], K)) #initialize W and H to random values between 1 and 2
    H = np.random.uniform(1, 2, (K, X_shape[1]))
    
    return X.astype(np.float32), W.astype(np.float32), H.astype(np.float32) # return matrices as float32
    
    
    

    
    
    
    
    
    
    
    
    