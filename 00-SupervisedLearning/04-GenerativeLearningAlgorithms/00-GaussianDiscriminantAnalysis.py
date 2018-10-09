"""
  author: Sierkinhane
  since: 2018-10-9 13:55:56
  description: Guassian Discriminant Analysis(GDA)
"""

from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import multivariate_normal

from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy.ma as ma
from numpy.random import uniform, seed
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)

# Hyper-parameters
fi = 0                                             # probablity of y = 1
miu_0 = np.zeros((2, 1))  # mean of y = 0
miu_1 = np.zeros((2, 1))  # mean of y = 1
COV = np.zeros((2, 2))    # convariance of two Gaussians

## 1. Generate two categories data
X, y = make_blobs(n_samples=300, n_features=2, centers=2, random_state=12)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, edgecolors='white')
# plt.show()

def update_parameters(X_train, Y_train, fi, miu_0, miu_1, COV):
    
    # update μ0, μ1, φ
    l_1 = 0
    for x, y in zip(X_train, Y_train):
        x = x.reshape(2, 1)
        l_1 += 1 * y
        if y == 0:

            miu_0 += x
        else:
            miu_1 += x

    fi = l_1 / (X_train.shape[0])
    miu_0 /= (X_train.shape[0] - l_1)
    miu_1 /= l_1

    # update Covariance of two variables
    for x, y in zip(X_train, Y_train):
        x = x.reshape(2, 1)
        if y == 0:
            COV += np.matmul(x - miu_0, (x - miu_0).T)
        else:
            COV += np.matmul(x - miu_1, (x - miu_1).T)
    COV /= X_train.shape[0]

    return fi, miu_0, miu_1, COV

def tow_d_gaussian(x, mu, COV):

    n = mu.shape[0]
    COV_det = np.linalg.det(COV)
    COV_inv = np.linalg.inv(COV)
    N = np.sqrt((2*np.pi)**n*COV_det)

    fac = np.einsum('...k,kl,...l->...',x-mu,COV_inv,x-mu)

    return np.exp(-fac/2)/N

if __name__ == '__main__':

    # obtain rule of paramters exploiting log-likelihood
    fi, miu_0, miu_1, COV = update_parameters(X_train, Y_train, fi, miu_0, miu_1, COV)

    # plotting
    fig =plt.figure()
    # ax = fig.gca(projection='3d') # 3d plotting
    ax = fig.gca()
    ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, edgecolors='white')

    N = 60
    X = np.linspace(-10,-2,N)
    Y = np.linspace(-2,8,N)
    X,Y = np.meshgrid(X,Y)

    pos = np.empty(X.shape+(2,))
    pos[:,:,0]= X
    pos[:,:,1] = Y

    miu_0 = np.reshape(miu_0, (1, 2))[0]
    miu_1 = np.reshape(miu_1, (1, 2))[0]

    Z1 = tow_d_gaussian(pos, miu_0, COV)
    Z2 = tow_d_gaussian(pos, miu_1, COV)
    
    cset = ax.contour(X,Y,Z1,zdir='z',offset=-0.15)
    cset = ax.contour(X,Y,Z2,zdir='z',offset=-0.15)

   
    # 3d plotting
    # ax.plot_surface(X,Y,Z1,rstride=3,cstride=3,linewidth=1,antialiased =True)
    # ax.plot_surface(X,Y,Z2,rstride=3,cstride=3,linewidth=1,antialiased =True)
    
    # ax.set_zlim(-0.15,0.2)
    # ax.set_zticks(np.linspace(0,0.2,5))
    # ax.view_init(12,-12)


    plt.show()
