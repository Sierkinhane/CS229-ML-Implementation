import numpy as np 
import matplotlib.pyplot as plt 

mu, sigma = 0, 0.1
gaussian_dis = np.random.normal(mu, sigma, 10000)
count, bins, ignored = plt.hist(gaussian_dis, 50, density=True, facecolor='g', alpha=.75)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')

plt.xlabel('x')
plt.ylabel('Probability')
plt.title('Histogram of Gaussian')
plt.text(60, .025, r'$\mu=0,\ \sigma=0.1$')
# plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()