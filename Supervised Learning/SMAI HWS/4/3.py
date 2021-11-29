import numpy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random

print('----------Question 1--------------------')
meanclassA = numpy.random.randn(3)
print('mean of classA', meanclassA)
meanclassB = numpy.random.randn(3)
print('mean of classB', meanclassB)
cov = [[100, 0, 0], [0, 50, 0], [0, 0, 1]]  # diagonal covariance
print('Covariance matrix', cov)

xclassA, yclassA, zclassA = numpy.random.multivariate_normal(
    meanclassA, cov, 1000).T

xclassB, yclassB, zclassB = numpy.random.multivariate_normal(
    meanclassB, cov, 1000).T

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.scatter(xclassA, yclassA, zclassA,label= "classA", color='red', marker='x')
ax.scatter(xclassB, yclassB, zclassB, label= "classB", color='blue', marker='o')
ax.set_xlabel('x axis')     
ax.set_ylabel('y axis')     
ax.set_zlabel('z axis')   
plt.legend()  

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.scatter(xclassA, yclassA, zclassA, label= "classA", color='red', marker='x')
ax.plot(xclassA, zclassA, 'y+', label= "xz projection", zdir='y', zs=-50)
ax.plot(yclassA, zclassA, 'g+', label= "yz projection", zdir='x', zs=-50)
ax.plot(xclassA, yclassA, 'k+', label= "xy projection", zdir='z', zs=-4)
plt.legend()  

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.scatter(xclassB, yclassB, zclassB,label= "classB", color='blue', marker='o')
ax.plot(xclassB, zclassB, 'y+',  label= "xz projection", zdir='y', zs=-50)
ax.plot(yclassB, zclassB, 'g+',  label= "yz projection", zdir='x', zs=-50)
ax.plot(xclassB, yclassB, 'k+', label= "xy projection", zdir='z', zs=-4)
plt.legend()
plt.title('Question1, different mean, and diagonal covariance matrix')
plt.show()

print('----------Question 2--------------------')
matrixSize = 3
A = numpy.random.rand(matrixSize,matrixSize)
cov = numpy.dot(A,A.transpose()) #PSD Cov matrix
print('Cov matrix PSD', cov)
meanclassA = numpy.random.randn(3)
print('mean of classA', meanclassA)
meanclassB = numpy.random.randn(3)
print('mean of classB', meanclassB)

xclassA, yclassA, zclassA = numpy.random.multivariate_normal(
    meanclassA, cov, 1000).T

xclassB, yclassB, zclassB = numpy.random.multivariate_normal(
    meanclassB, cov, 1000).T

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.scatter(xclassA, yclassA, zclassA,label= "classA", color='red', marker='x')
ax.scatter(xclassB, yclassB, zclassB, label= "classB", color='blue', marker='o')
ax.set_xlabel('x axis')     
ax.set_ylabel('y axis')     
ax.set_zlabel('z axis')   
plt.legend()  

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.scatter(xclassA, yclassA, zclassA, label= "classA", color='red', marker='x')
ax.plot(xclassA, zclassA, 'y+', label= "xz projection", zdir='y', zs=-5)
ax.plot(yclassA, zclassA, 'g+', label= "yz projection", zdir='x', zs=-5)
ax.plot(xclassA, yclassA, 'k+', label= "xy projection", zdir='z', zs=-4)
plt.legend()  

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.scatter(xclassB, yclassB, zclassB,label= "classB", color='blue', marker='o')
ax.plot(xclassB, zclassB, 'y+',  label= "xz projection", zdir='y', zs=-5)
ax.plot(yclassB, zclassB, 'g+',  label= "yz projection", zdir='x', zs=-5)
ax.plot(xclassB, yclassB, 'k+', label= "xy projection", zdir='z', zs=-4)
plt.legend()
plt.title('Question2, different mean, and PSD matrix')
plt.show()

print('----------Question 3--------------------')
matrixSize = 3
A = numpy.random.rand(matrixSize,matrixSize)
covClassA = numpy.dot(A,A.transpose()) #PSD Cov matrix
print('Covariance matrix for class A', covClassA)
A = numpy.random.rand(matrixSize,matrixSize)
covClassB = numpy.dot(A,A.transpose()) #PSD Cov matrix
print('Covariance matrix for class B', covClassB)
mean = numpy.random.randn(3)
print('mean', mean)

xclassA, yclassA, zclassA = numpy.random.multivariate_normal(
    mean, covClassA, 1000).T

xclassB, yclassB, zclassB = numpy.random.multivariate_normal(
    mean, covClassB, 1000).T

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.scatter(xclassA, yclassA, zclassA,label= "classA", color='red', marker='x')
ax.scatter(xclassB, yclassB, zclassB, label= "classB", color='blue', marker='o')
ax.set_xlabel('x axis')     
ax.set_ylabel('y axis')     
ax.set_zlabel('z axis')   
plt.legend()  

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.scatter(xclassA, yclassA, zclassA, label= "classA", color='red', marker='x')
ax.plot(xclassA, zclassA, 'y+', label= "xz projection", zdir='y', zs=-5)
ax.plot(yclassA, zclassA, 'g+', label= "yz projection", zdir='x', zs=-5)
ax.plot(xclassA, yclassA, 'k+', label= "xy projection", zdir='z', zs=-4)
plt.legend()  

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.scatter(xclassB, yclassB, zclassB,label= "classB", color='blue', marker='o')
ax.plot(xclassB, zclassB, 'y+',  label= "xz projection", zdir='y', zs=-5)
ax.plot(yclassB, zclassB, 'g+',  label= "yz projection", zdir='x', zs=-5)
ax.plot(xclassB, yclassB, 'k+', label= "xy projection", zdir='z', zs=-4)
plt.legend()
plt.title('Question3, same mean, and different cov matrix')
plt.show()