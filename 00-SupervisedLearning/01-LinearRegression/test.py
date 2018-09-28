import numpy as np
import matplotlib.pyplot as plt 

noise = np.random.normal(0, 1, size=100)
x = np.linspace(-3, 3, 100)
y = x**2 + 2 + noise
plt.scatter(x, y)
plt.show()