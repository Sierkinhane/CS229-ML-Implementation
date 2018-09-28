"""
  author: Sierkinhane
  since: 2018-9-22 10:37:41
  description: (done)
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
from numpy.linalg import inv
import time

# make dummy regression data
X, y = datasets.make_regression(n_samples=250, n_features=1, noise=20, random_state=0, bias=50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# show data
plt.scatter(X_test, y_test, c='red', edgecolors='white')
plt.scatter(X_train, y_train, c='orange', edgecolors='white')
plt.show()

# learnable parameter theta
theta = np.zeros([2, 1], dtype=np.float32)
# add the first column value of 1 for theta0(bias) entry.
X_train = np.concatenate((np.ones([175, 1], dtype=np.float32), X_train), axis=1)

# the final solution for the linear regression 
theta = np.matmul(inv(np.matmul(X_train.T, X_train)), np.matmul(X_train.T, y_train))
theta = np.reshape(theta, [2, 1])

# training set
p_x = np.linspace(-2.5, 2.5, 50)
plt.figure(0)
plt.scatter(X_train[:, 1], y_train, c='purple', marker='o', edgecolors='white')
plt.plot(p_x, theta[1, 0] * p_x + theta[0, 0], c='orange')
plt.show()

# test
plt.figure(1)
p_y = theta[1, 0] * X_test + theta[0, 0]
plt.scatter(X_test, y_test, c='red', marker='o', edgecolors='white')
plt.plot(X_test, p_y, c='orange')
plt.show()