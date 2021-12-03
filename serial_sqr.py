import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = 'nyt_data.txt'
vocab_path = 'nyt_vocab.dat'
data = pd.read_csv(data_path, sep='\n', header=None)
vocab = pd.read_csv(vocab_path, sep='\n', header=None, names=['words'])

N = 3012 #number of words
M = 8447 #number of articles
X = np.zeros((N, M)) #define the original data matrix

#load data, every row is a single document
#format “idx:cnt” with commas separating each unique word in the document
for column in range(M):
    row = data.iloc[column].values[0].split(',')
    for item in row:
        index, count = map(int, item.split(':'))
        X[index-1][column] = count

err = 1e-16 #a small error to prevent 0/0
iteration = 100 #number of iterations
K = 25 #number of categories (matrix rank)

W = np.random.uniform(1, 2, (N, K)) #initialize W and H to random values between 1 and 2
H = np.random.uniform(1, 2, (K, M))

squared_out = [] #keep track of objective function for each iteration
for i in range(1, iteration+1):
    if i % 10 == 0:
        print('iteration %d' % i)
        
    Wt = W.T #w transpose
    H = H * Wt.dot(X) / (Wt.dot(W).dot(H) + err) #update H

    Ht = H.T #H transpose
    W = W * X.dot(Ht) / (W.dot(H).dot(Ht) + err) #update W

    #squared error objective function
    squared_out.append(np.matrix(np.square(X - (W.dot(H)))).sum())

#plot the objective function, should monotonically decrease to a local minimum
fig = plt.figure()
plt.plot(np.arange(1, 101), squared_out, label='squared error objective function')
plt.grid()
plt.legend()
plt.ylabel('value')
plt.xlabel('iteration')
plt.title('Squared error NMF')
plt.savefig('test_squared_error.png')

W = W / W.sum(axis=0).reshape(1,-1) #normalize the 25 categories

#find the 10 most used words for each category and print
data = pd.DataFrame(index=range(10), columns=['Topic_%d' % i for i in range(1, 26)])
for i in range(25):
    column = 'Topic_' + str(i+1)
    Wi = W[:, i]
    dt = pd.DataFrame(Wi, columns=['weight'])
    dt['words'] = vocab
    dt = dt.sort_values(by='weight', ascending=False)[:10].reset_index(drop=True)
    data[column] = dt['weight'].map(lambda x: ('%.4f')%x) + ' ' + dt['words']
print(data)
