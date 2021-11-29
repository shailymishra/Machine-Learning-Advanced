import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from numpy import linalg as LA


fig = plt.figure()
ax = fig.add_subplot(321, aspect='equal')
# Parallelogram
xy = np.array([[0,0], [7,2], [12,11],[5,9]])
mean = np.mean(xy, axis=0)
print('Mean', mean)
cov = np.cov(xy.T)
print('Cov::')
print(cov)
eigenvalues, eigenvectors = LA.eig(cov)
print('Eigen values', eigenvalues)
print('Eigen vectors', eigenvectors)
ax.add_patch(patches.Polygon(xy, fill=False))
plt.quiver(mean[0],mean[1],eigenvectors[0,0],eigenvectors[0,1], color=['r'],scale=5, label="eigen vector[0]")
plt.quiver(mean[0],mean[1],eigenvectors[1,0],eigenvectors[1,1], color=['g'],scale=5, label="eigen vector[1]")

ax1 = fig.add_subplot(322, aspect='equal')
# Parallelogram : Square
xy1 = np.array([[0,0], [0,3], [3,3],[3,0]])
mean1 = np.mean(xy1, axis=0)
print('Mean', mean1)
cov1 = np.cov(xy1.T)
print('Cov::')
print(cov1)
eigenvalues1, eigenvectors1 = LA.eig(cov1)
print('Eigen values', eigenvalues1)
print('Eigen vectors', eigenvectors1)
ax1.add_patch(patches.Polygon(xy1, fill=False))
ax1.quiver(mean1[0],mean1[1],eigenvectors1[0,0],eigenvectors1[0,1], color=['r'],scale=5, label="eigen vector[0]")
ax1.quiver(mean1[0],mean1[1],eigenvectors1[1,0],eigenvectors1[1,1], color=['g'],scale=5, label="eigen vector[1]")


ax2 = fig.add_subplot(323, aspect='equal')
# Parallelogram : Rectangle
xy2 = np.array([[0,0], [0,2], [4,2],[4,0]])
mean2 = np.mean(xy2, axis=0)
print('Mean', mean2)
cov2 = np.cov(xy2.T)
print('Cov::')
print(cov2)
eigenvalues2, eigenvectors2 = LA.eig(cov2)
print('Eigen values', eigenvalues2)
print('Eigen vectors', eigenvectors2)
ax2.add_patch(patches.Polygon(xy2, fill=False))
ax2.quiver(mean2[0],mean2[1],eigenvectors2[0,0],eigenvectors2[0,1], color=['r'],scale=5, label="eigen vector[0]")
ax2.quiver(mean2[0],mean2[1],eigenvectors2[1,0],eigenvectors2[1,1], color=['g'],scale=5, label="eigen vector[1]")


ax3 = fig.add_subplot(324, aspect='equal')
# Parallelogram : Rectangle
xy3 = np.array([[0,0], [-1,3], [4,3],[5,0]])
mean3 = np.mean(xy3, axis=0)
print('Mean', mean3)
cov3 = np.cov(xy3.T)
print('Cov::')
print(cov3)
eigenvalues3, eigenvectors3 = LA.eig(cov3)
print('Eigen values', eigenvalues3)
print('Eigen vectors', eigenvectors3)
ax3.add_patch(patches.Polygon(xy3, fill=False))
ax3.quiver(mean3[0],mean3[1],eigenvectors3[0,0],eigenvectors3[0,1], color=['r'],scale=5, label="eigen vector[0]")
ax3.quiver(mean3[0],mean3[1],eigenvectors3[1,0],eigenvectors3[1,1], color=['g'],scale=5, label="eigen vector[1]")


ax4 = fig.add_subplot(325, aspect='equal')
# Parallelogram 
xy4 = np.array([[0,0], [1,3], [5,3],[4,0]])
mean4 = np.mean(xy4, axis=0)
print('Mean', mean4)
cov4 = np.cov(xy4.T)
print('Cov::')
print(cov4)
eigenvalues4, eigenvectors4 = LA.eig(cov4)
print('Eigen values', eigenvalues4)
print('Eigen vectors', eigenvectors4)
ax4.add_patch(patches.Polygon(xy4, fill=False))
ax4.quiver(mean4[0],mean4[1],eigenvectors4[0,0],eigenvectors4[0,1], color=['r'],scale=5, label="eigen vector[0]")
ax4.quiver(mean4[0],mean4[1],eigenvectors4[1,0],eigenvectors4[1,1], color=['g'],scale=5, label="eigen vector[1]")
plt.show() 
