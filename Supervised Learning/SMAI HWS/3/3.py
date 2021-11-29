import numpy
import matplotlib.pyplot as plt 
import random
from numpy import linalg as LA
m, b = 5, 3
lower, upper = -50, 50
xstart = 0
xstop = 1000
x = numpy.linspace(-10,xstop,10)

samplepoints = 1000
x1 = [numpy.random.randint(xstart, xstop) for i in range(samplepoints)]
y1 = [numpy.random.randint(m*x+b+lower, m*x+b+upper) for x in x1]
meanx = numpy.mean(x1)
meany = numpy.mean(y1)
print('mean:', [meanx, meany])
X = numpy.stack((x1, y1), axis=0)
Covar = numpy.cov(X)
print('cov', Covar)
X = X.T
mean_vec = numpy.mean(X, axis=0)

covmatrix = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0]-1)
print('covmatrix', covmatrix)
# print('is cov same',(newmatrix - new_mean_vec).T.dot((newmatrix - new_mean_vec)) / (newmatrix.shape[0]-1))

eigenvalue, eigenvector = LA.eig(Covar)
print('Eigen Values ', eigenvalue)
print('Eigen Vectors ', eigenvector)
plt.plot(x,m*x+b, label= "Line y =mx+c",linestyle='solid',color = "k")
plt.scatter(x1, y1, label="random points near line" , c='c')
plt.quiver(meanx,meany,eigenvector[0,0],eigenvector[0,1], color=['r'],scale=5, label="eigen vector[0]")
plt.quiver(meanx,meany,eigenvector[1,0],eigenvector[1,1], color=['g'],scale=5, label="eigen vector[1]")
plt.legend()
plt.show()
