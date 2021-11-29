from __future__ import print_function

from sklearn import svm
import numpy as np
from scipy.linalg import qr
import matplotlib.pyplot as plt
from numpy import linalg as LA


def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)

    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [int(x) for x in num[1:]]

    return data, labels


# gradient Descent

train_data, train_labels = read_data("sample_train.csv")
train_data = train_data / 255
test_data, test_labels = read_data("sample_test.csv")
test_data = test_data / 255

class01trainsample = train_data[(train_labels[:] == 2) | (train_labels[:]==1)]
class01trainlabels = train_labels[(train_labels[:] == 2) | (train_labels[:]==1)]
class01testsample = test_data[(test_labels[:] == 2) | (test_labels[:]==1)]
class01testlabels = test_labels[(test_labels[:] == 2) | (test_labels[:]==1)]

## 1
clf = svm.SVC(C = 0.0000000001,kernel="linear")
clf.fit(class01trainsample, class01trainlabels)
confidence = clf.score(class01testsample, class01testlabels)
print('Accuracy for C = 0.00000001 is ',confidence)
clf = svm.SVC(C = 0.000001,kernel="linear")
clf.fit(class01trainsample, class01trainlabels)
confidence = clf.score(class01testsample, class01testlabels)
print('Accuracy for C = 0.000001 is ',confidence)
clf = svm.SVC(C = 1,kernel="linear")
clf.fit(class01trainsample, class01trainlabels)
confidence = clf.score(class01testsample, class01testlabels)
print('Accuracy for C = 1 is ',confidence)

## 2 
clf = svm.SVC(C = 1,kernel="linear")
clf.fit(class01trainsample, class01trainlabels)
class0trainsample = train_data[(train_labels[:] == 2)]
class1trainsample = train_data[(train_labels[:] == 1)]

classtrainsample = np.vstack((class0trainsample,class1trainsample))

cov = np.cov(classtrainsample.T) # need to do transpose to use np.cov
[eigenvalues, eigenvectors] = LA.eigh(cov)
eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)
matrix_w = np.hstack((eig_pairs[0][1].reshape(784,1), eig_pairs[1][1].reshape(784,1)))
transformed = classtrainsample.dot(matrix_w)
plt.plot(transformed[0:600,0], transformed[0:600,1], 'o', color='blue',  label='Class 2')
plt.plot(transformed[600:1200,0], transformed[600:1200,1], 'x', color='yellow',  label='Class 1')


w=clf.coef_[0].dot(matrix_w)
I=clf.intercept_
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]
plt.plot(xx,yy, label= "Decision Boundary",linestyle='solid',color = "k")
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend()
plt.title('Transformed samples for C = 1')
plt.show()

### 3

supportvector = clf.support_vectors_.dot(matrix_w)
plt.plot(transformed[0:600,0], transformed[0:600,1], 'o', color='blue',  label='Class 2')
plt.plot(transformed[600:1200,0], transformed[600:1200,1], 'x', color='yellow',  label='Class 1') 
plt.scatter(supportvector[:, 0], supportvector[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k', label="Support Vectors")
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.legend()
plt.show()

