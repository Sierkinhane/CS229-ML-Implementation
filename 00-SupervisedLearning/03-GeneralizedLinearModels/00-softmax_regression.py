"""
  author: Sierkinhane
  since: 2018-9-26 12:56:09
  description: softmax regression(done)
"""

from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt 
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from math import exp
import time
np.random.seed(1)

# Hyper-prameters
LR = 0.05
DECAY_RATE = 0.005
THETA = np.random.normal(0, 0.5, 3*3).reshape(3, 3)
EPOCH = 20

## 1. Gernerate four categories data
X, y = make_blobs(n_samples=150, n_features=2, centers=3, random_state=3)  # state 12
# transform y to onehot vector
encoder = OneHotEncoder(categorical_features='all')
y = encoder.fit_transform(np.reshape(y, (150,1))).toarray()
# print(y, y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)
plt.scatter(X_train[:, 0], X_train[:, 1], c=np.argmax(Y_train, axis=1), edgecolors='white')
plt.show()

o_train = np.ones([X_train.shape[0], 1], dtype=X_train.dtype)
o_test = np.ones([X_test.shape[0], 1], dtype=X_test.dtype)
X_train = np.concatenate((o_train, X_train), axis=1)
X_test = np.concatenate((o_test, X_test), axis=1)
print("X_train shape : {0}".format(X_train.shape))
print("Y_train shape : {0}".format(Y_train.shape))
print("X_test shape : {0}".format(X_test.shape))
print("y_test shape : {0}".format(Y_test.shape))

## 2. Make a hypothesis
def hypothesis(x, THETA):
	x = np.reshape(x, (3, 1))
	temp = np.matmul(THETA.T, x) / 100
	temp = np.exp(temp)
	denominator = np.sum(temp)
	hypothesis = temp / denominator # normalize into 1

	return hypothesis

## 3. Loss definition
def compute_loss(X, Y,THETA):

	loss = 0
	for x, y in zip(X, Y):
		x = np.reshape(x, (3, 1))
		y = np.reshape(y, (3, 1))     # 
		h_x = hypothesis(x, THETA)    # hypothesis (3, 1)
		label = np.argmax(y, axis=0)  # the category of prediction
		loss += (-np.log(h_x[label][0] + 0.0000001))  # loss = - y * log(y')

	return loss

## 4. Parameters updating
def update_parameters(THETA, x, y):

	x = np.reshape(x, (3, 1))
	y = np.reshape(y, (3, 1))

	h_x = hypothesis(x, THETA)

	label = np.argmax(y, axis=0)
    # θk := θk - （-yk * (1/y'k) * x)  k --> the class, yk and y'k are real number, x is a vector 
	THETA[:, label] = THETA[:, label] - LR *(-y[label][0] * (1 / h_x[label][0] * x))

	return THETA

# for plotting
H_train = np.zeros((Y_train.shape[0], Y_train.shape[1]))
H_test = np.zeros((Y_test.shape[0], Y_test.shape[1]))

if __name__ == '__main__':
	
	plt.figure(0)
	plt.ion()
	for epoch in range(EPOCH):
		# learning rate decay
		LR = LR * (1 / (1 + DECAY_RATE * epoch))
		i = 0 # retrieve H_x
		for x, y in zip(X_train, Y_train):
			# shape of x = (1, 3)
			# shape of y = (1, 3)
			# hence we should transpose it.
			loss = compute_loss(X_train, Y_train, THETA)
			print('[{0}/{1}] loss is: {2}'.format(epoch+1, EPOCH, loss[0]))
			H_train[i] = hypothesis(x, THETA).T
			THETA = update_parameters(THETA, x, y)

			plt.cla()
			plt.scatter(X_train[:, 1], X_train[:, 2], c=np.argmax(H_train, axis=1), edgecolors='white', marker='s')
			plt.pause(0.001)
			i+=1

	plt.ioff()
	plt.show()

	# test
	i = 0
	for x, y in zip(X_test, Y_test):
		H_test[i] = hypothesis(x, THETA).T
		i+=1
	plt.scatter(X_test[:, 1], X_test[:, 2], c=np.argmax(H_test, axis=1), edgecolors='white', marker='s')
	plt.show()
