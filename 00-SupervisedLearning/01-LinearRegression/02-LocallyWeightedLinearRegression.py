"""
  author: Sierkinhane
  since: 2018-9-22 10:37:49
  description: implement of LWR(done)
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
import time
from math import exp

# τ for w(i)
t = 0.1
# local weights
x_point = 4.5
# Learnable parameter theta
theta = np.zeros([2, 1], dtype=np.float32)
# Learning rate
lr_sto = 0.01
# Epoch
Epoch = 2

# make dummy regression data
noise = np.random.normal(0, 2, size=250)
X = np.linspace(-5, 5, 250)
y = X**2 + 2 + noise
# y = X**2 + 2 + noise
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# for theata0, x0 = 1
o_train = np.ones([X_train.shape[0], 1], dtype=X_train.dtype)
o_test = np.ones([X_test.shape[0], 1], dtype=X_test.dtype)
X_train = np.concatenate((o_train, np.reshape(X_train, (o_train.shape))), axis=1)
X_test = np.concatenate((o_test, np.reshape(X_test, (o_test.shape))), axis=1)

# show data
plt.scatter(X_test[:, 1], y_test, c='red', edgecolors='white')
plt.scatter(X_train[:, 1], y_train, c='orange', edgecolors='white')
plt.ylim((-8, 42))
plt.xlim((-7.5, 7.5))
plt.show()

def hypothesis(x, θ):

	# hypothesis
	h_x = np.matmul(θ.T, x)

	return h_x


# single training example
def loss_function(h_x, x, x_point, y):

	# w(i): non-negative valued weights
	w = exp(-(x[1] - x_point)**2 / (2*(t**2)))
	# loss function
	J = 1 /2 *w *(h_x - y) *(h_x - y)
	return J 

def parameter_update(theta, h_x, x, x_point, y, lr):
	
	# w(i): non-negative valued weights
	w = exp(-(x[1] - x_point)**2 / (2*(t**2)))
	# update
	x = np.reshape(x, theta.shape)
	theta = theta - lr *w *(h_x - y) *x

	return theta

# for plotting
p_x = np.linspace(-x_point-3, x_point+3, 100)

# stochastic gradient descent
plt.ion()
for epoch in range(Epoch):
	for x, y in zip(X_train, y_train):
		# make hypothesis
		h_x = hypothesis(x, theta)
		# compute loss
		loss = loss_function(h_x, x, x_point, y)
		# Excecute gradient descent to update parameters
		theta = parameter_update(theta, h_x, x, x_point, y, lr_sto)

		print("[{0}/{1}], loss: {2}".format(epoch+1, Epoch, loss[0]))
		plt.cla()
		plt.scatter(X_train[:, 1], y_train, c='purple', marker='o', edgecolors='white')
		plt.ylim((-8, 42))
		plt.xlim((-7.5, 7.5))
		plt.plot(p_x, theta[1, 0] * p_x + theta[0, 0], c='orange')
		plt.pause(0.01)
plt.ioff()
plt.show()

# test
p_y = theta[1, 0] * p_x + theta[0, 0]
plt.scatter(X_test[:, 1], y_test, c='red', marker='o', edgecolors='white')
plt.ylim((-8, 42))
plt.xlim((-7.5, 7.5))
plt.plot(p_x, p_y, c='orange')
plt.show()