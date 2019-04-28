"""
  author: Sierkinhane
  since: 2018-9-22 10:37:17
  description: linear regression(done)
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
import time

# make dummy regression data
X, y = datasets.make_regression(n_samples=250, n_features=1, noise=20, random_state=0, bias=50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# for theata0, x0 = 1
o_train = np.ones([X_train.shape[0], 1], dtype=X_train.dtype)
o_test = np.ones([X_test.shape[0], 1], dtype=X_test.dtype)
X_train = np.concatenate((o_train, X_train), axis=1)
X_test = np.concatenate((o_test, X_test), axis=1)

# show data
plt.scatter(X_test[:, 1], y_test, c='red', edgecolors='white')
plt.scatter(X_train[:, 1], y_train, c='orange', edgecolors='white')
plt.ylim((-25, 125))
plt.xlim((-3, 3))
plt.show()

# learnable parameter theta
THETA = np.zeros([2, 1], dtype=np.float32)
# learning rate
lr_sto = 0.01
lr_bat = 0.0001
# Epoch
Epoch = 2

def hypothesis(x, θ):

	# hypothesis
	h_x = np.matmul(θ.T, x)

	return h_x

# single training example
def loss_function(h_x, y):
	# loss function
	J = 1 /2 *(h_x - y) *(h_x - y)
	return J 

def update_parameters(THETA, h_x, x, y, lr):
	
	x = np.reshape(x, THETA.shape)
	THETA = THETA - lr *(h_x - y) *x

	return THETA

def batch_update_parameters(THETA, gradient, lr):
	
	THETA = THETA - lr *gradient

	return THETA

# for plotting
p_x = np.linspace(-2.5, 2.5, 50)

# stochastic gradient descent
plt.ion()
for epoch in range(Epoch):
	for x, y in zip(X_train, y_train):
		# make hypothesis
		h_x = hypothesis(x, THETA)
		# compute loss
		loss = loss_function(h_x, y)
		# Excecute gradient descent to update parameters
		THETA = update_parameters(THETA, h_x, x, y, lr_sto)

		print("[{0}/{1}], loss: {2}".format(epoch+1, Epoch, loss[0]))
		plt.cla()
		plt.scatter(X_train[:, 1], y_train, c='purple', marker='o', edgecolors='white')
		plt.plot(p_x, THETA[1, 0] * p_x + THETA[0, 0], c='orange')
		plt.ylim((-25, 125))
		plt.xlim((-3, 3))
		plt.pause(0.01)
plt.ioff()
plt.show()

# batch gradient descent
# loss = 0
# plt.ion()
# for epoch in range(200):
# 	gradient = np.zeros([2, 1], dtype=np.float32)
# 	for x, y in zip(X_train, y_train):
# 		h_x = hypothesis(x, THETA)
# 		loss += loss_function(h_x, y)
# 		x = np.reshape(x, THETA.shape)
# 		gradient += (h_x - y) *x
# 	THETA = batch_update_parameters(THETA, gradient, lr_bat)
# 	print("[{0}/{1}], loss: {2}".format(epoch, 100, loss[0]/(y_train.shape[0])))

# 	plt.cla()
# 	plt.scatter(X_train[:, 1], y_train, c='red', marker='o', edgecolors='white')
# 	plt.plot(p_x, THETA[1, 0] * p_x + THETA[0, 0], c='blue')
# 	plt.ylim((-25, 125))
# 	plt.xlim((-3, 3))
# 	plt.pause(0.01)

# plt.ioff()
# plt.show()

# test
p_y = THETA[1, 0] * X_test + THETA[0, 0]
plt.scatter(X_test[:, 1], y_test, c='red', marker='o', edgecolors='white')
plt.plot(X_test, p_y, c='orange')
plt.show()
