"""
  author: Sierkinhane
  since: 2018-9-23 10:14:38
  description: the implement of logistic regression(done)
"""

from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from math import exp, log

## Hyper-paramters definition
LR = 0.075
EPOCH = 4
THETA = np.random.normal(0, 0.1, 3).reshape(3, 1) # learnable parameters

## First step: Generating dummy two categories of data.
# 1. data(300)
X, Y = make_blobs(n_samples=300, centers=2, n_features=2, random_state=3)
# show data distribution
# plt.scatter(X[:,0], X[:, 1], c=Y, edgecolors='white', marker='s')
# plt.show()

# 2. split into train/val set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
plt.scatter(X_train[:,0], X_train[:, 1], c=Y_train, edgecolors='white', marker='s')
plt.show()
# for theta0, x0 = 1
o_train = np.ones([X_train.shape[0], 1], dtype=X_train.dtype)
o_test = np.ones([X_test.shape[0], 1], dtype=X_test.dtype)
X_train = np.concatenate((o_train, X_train), axis=1)
X_test = np.concatenate((o_test, X_test), axis=1)
print("X_train shape : {0}".format(X_train.shape))
print("Y_train shape : {0}".format(Y_train.shape))
print("X_test shape : {0}".format(X_test.shape))
print("y_test shape : {0}".format(Y_test.shape))
'''
X_train shape : (210, 2)
y_train shape : (90, 2)
X_test shape : (210,)
y_test shape : (90,)
'''
H_train = np.zeros([Y_train.shape[0], 1], dtype=Y_train.dtype)
H_test = np.zeros([Y_test.shape[0], 1], dtype=Y_test.dtype)

## Second step: Making hypothesis
def sigmoid_function(z):
	
	g = 1 / (1 + exp(-z))

	return g

def hypothesis(x, THETA):
	
	hypothesis = np.matmul(THETA.T, x)
	hypothesis = sigmoid_function(hypothesis[0])

	return hypothesis

## Third step: Define a loss function
def compute_loss(X, Y, THETA):
	
	loss = 0
	for x, y in zip(X, Y):
		h_x = hypothesis(x, THETA)
		# if h_x == 1 --> log(1-1) --> error
		if h_x == 1:
			h_x = 1-0.0000000000001
		loss += (-y) *(log(h_x) - (1-y) *log(1-h_x))

	return loss/(X.shape[0])

## Fourth step: Updating parameters
def update_parameters(THETA, LR, y, h_x, x):
	
	x = np.reshape(x, THETA.shape)
	THETA = THETA + LR *(y - h_x) * x

	return THETA

if __name__ == '__main__':
	
	plt.figure(0)
	plt.ion()
	for epoch in range(EPOCH):
		i = 0 # retrieve H_x
		for x, y in zip(X_train, Y_train):
			loss = compute_loss(X_train, Y_train, THETA)
			print('[{0}/{1}] loss is: {2}'.format(epoch+1, EPOCH, loss))
			H_train[i] = hypothesis(x, THETA)
			THETA = update_parameters(THETA, LR, y, H_train[i], x)

			plt.cla()
			plt.scatter(X_train[:, 1], X_train[:, 2], c=H_train[:, 0], edgecolors='white', marker='s')
			plt.pause(0.001)
			i+=1
	plt.ioff()
	plt.show()

	## TEST 
	i = 0
	for x, y in zip(X_test, Y_test):

		H_test[i] = hypothesis(x, THETA)
		i+=1

	plt.figure(1)
	x = np.linspace(-7, 4, 50)
	plt.scatter(X_test[:, 1], X_test[:, 2], c=H_test[:, 0], edgecolors='white', marker='s')
	plt.show()