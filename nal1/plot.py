# adapted from 
# https://pythonnumericalmethods.berkeley.edu/notebooks/chapter12.02-3D-Plotting.html
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')




Z = np.loadtxt("outmat.txt")

x = np.linspace(-1, 1, Z.shape[1])
y = np.linspace(-1, 1, Z.shape[1])
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

fig.colorbar(surf, shrink=0.5, aspect=8)

plt.show()