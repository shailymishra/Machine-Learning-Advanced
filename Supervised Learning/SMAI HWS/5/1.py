import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA


def generateXYInRange0to10( mu, cov, samplesize  ):
    x, y = np.random.multivariate_normal(mu,cov, samplesize).T
    issueindex = (np.where((x<0) | (y<0) | (y>10) | (x>10)))[0]
    if(len(issueindex)>0):
        append_x,append_y = generateXYInRange0to10(mu, cov, len(issueindex))
        x = np.delete(x, issueindex)
        y = np.delete(y, issueindex)
        x = np.append(x,append_x)
        y = np.append(y,append_y)
    return (x,y)
    
def getSlopeAndInterceptfor2DLine(w,x0):
    intercept = (w[0]*x0[0] + w[1]*x0[1]) // w[1]
    slope = -(w[0])// w[1]
    return (intercept,slope)


mu1 = np.array([3, 3])
mu2 = np.array([7, 7])
commoncov = np.array([[3, 0], [0, 3]])

cov1 = np.array([[3, 1], [2, 3]])
cov2 = np.array([[7, 2], [1, 7]]),
testSamplesx = np.random.uniform(low=0, high=10, size=100)
testSamplesy = np.random.uniform(low=0, high=10, size=100)

## For Question 1
# WHen covariance is same
x1,y1 = generateXYInRange0to10( mu1,commoncov,1000)
x2,y2 = generateXYInRange0to10( mu2,commoncov,1000)
classAClassificationx1 = []
classAClassificationy1 = []
classBClassificationx1 = []
classBClassificationy1 = []
w=mu1 - mu2
x0 = 0.5*(mu1+mu2)
c = w.T.dot(x0)
m = w.T
for i in range(len(testSamplesx)):
    x_test = [testSamplesx[i], testSamplesy[i]]
    line = m.dot(x_test) - c
    if( line > 0):
        classAClassificationx1.append(testSamplesx[i])
        classAClassificationy1.append(testSamplesy[i])
    else:
        classBClassificationx1.append(testSamplesx[i])
        classBClassificationy1.append(testSamplesy[i])

x = np.linspace(-15,15,1000)
plt.subplot(2, 1, 1)
intercept,slope = getSlopeAndInterceptfor2DLine(w,x0)
print(slope)
print(intercept)
plt.plot(x,slope*x+intercept, label= "Line y =mx+c",linestyle='solid',color = "k")
plt.scatter(classAClassificationx1, classAClassificationy1, label= "classA", color= "green", s=10) 
plt.scatter(classBClassificationx1, classBClassificationy1, label= "classB", color= "red", s=10) 
plt.legend()
plt.title("When Covariance is same and diagonal")
#####New Next Question
inverse1 = np.linalg.inv(cov1)
inverse2 = np.linalg.inv(cov2)

W1i = -0.5 * (inverse1)
w1i = inverse1.dot(mu1)
(sign, logdet1) = np.linalg.slogdet(cov1)
w1i0 = -0.5*( mu1.T.dot(inverse1).dot(mu1) + logdet1 )

W2i = -0.5 * (inverse2)
w2i = inverse2.dot(mu2)
(sign, logdet2) = np.linalg.slogdet(cov2)
w2i0 = -0.5*( mu1.T.dot(inverse1).dot(mu1) + logdet2 )

classAClassificationx2 = []
classAClassificationy2 = []
classBClassificationx2 = []
classBClassificationy2 = []

for i in range(len(testSamplesx)):
    x_test = [testSamplesx[i], testSamplesy[i]]
    x_test = np.array(x_test)
    log_prob1 = x_test.dot(W1i).dot(x_test) + w1i.dot(x_test) + w1i0
    log_prob2 = x_test.dot(W2i).dot(x_test) + w2i.dot(x_test) + w2i0
    
    if( log_prob1 > log_prob2 ):
        classAClassificationx2.append(testSamplesx[i])
        classAClassificationy2.append(testSamplesy[i])
    else:
        classBClassificationx2.append(testSamplesx[i])
        classBClassificationy2.append(testSamplesy[i])

x = np.linspace(-15,15,1000)
plt.subplot(2, 1, 2)
plt.scatter(classAClassificationx2, classAClassificationy2, label= "classA", color= "green", s=10) 
plt.scatter(classBClassificationx2, classBClassificationy2, label= "classB", color= "red", s=10) 
plt.title("When Covariance is different")
plt.legend()
plt.show()



