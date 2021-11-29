# Take 100 samples
# chose a k = 10 for now
# divide sample
import numpy as np
import random
import matplotlib.pyplot as plt

def linearregression(A,b):
    ATA = A.T.dot(A)
    ATb = A.T.dot(b)
    parameters = np.linalg.lstsq(ATA, ATb)[0]
    # print(parameters)
    # print(  np.linalg.solve(ATA, ATb))
    return parameters

def kfold(k,sample_size,x,y):
    ksets = []
    i = 0
    while i < sample_size:
        ksets.append([i,i+k-1])
        i = i+k
    # print(ksets)
    ones = np.ones(sample_size-k)
    errors = np.array([])
    for i in range(sample_size//k):
        todeleteindex = ksets[i]
        index = range(todeleteindex[0],todeleteindex[1]+1)
        a = np.delete(x, index)
        A = np.column_stack((a,ones))
        b = np.delete(y,index)
        parameters = linearregression(A,b)
        error = (np.abs(b - (parameters[0] * (a) + parameters[1])))
        errors = np.append(errors,error)
    # errors = np.array(errors)
    return errors



mu = 0
sigma = 1
sample_size = 100
noise = np.random.normal(mu, sigma, sample_size)

x = np.linspace(-10, 10, num=100)
y = [  (np.sin(x[i]) + noise[i] ) for i in range(len(x))]
y = np.array(y)

k = range(2,100)
meansCorrespondingToK = np.array([])
varianceCorrespondingToK = np.array([])

for i in k:
    print('i in k',i)
    errors = kfold(i,sample_size,x,y)
    mean = np.mean(errors)
    var = np.var(errors)
    meansCorrespondingToK = np.append(meansCorrespondingToK,mean)
    varianceCorrespondingToK = np.append(varianceCorrespondingToK,var)

plt.subplot(2,1,1)
plt.plot(k,meansCorrespondingToK, label= "Mean of Error",linestyle='solid')
plt.plot(k,varianceCorrespondingToK, label= "Variance of Error",linestyle='solid')
plt.legend()
plt.title("Plot for 100 points")
plt.legend()



#### 100000

sample_size1 = 10000
noise1 = np.random.normal(mu, sigma, sample_size1)

x1 = np.linspace(-1000, 1000, num=10000)
print(len(x1))
y1 = [  (np.sin(x1[i]) + noise1[i] ) for i in range(len(x1))]
y1 = np.array(y1)

k1 = range(2,1000)
meansCorrespondingToK1 = np.array([])
varianceCorrespondingToK1 = np.array([])

for i in k1:
    print('k',i)
    errors1 = kfold(i,sample_size1,x1,y1)
    mean1 = np.mean(errors1)
    var1 = np.var(errors1)
    meansCorrespondingToK1 = np.append(meansCorrespondingToK1,mean1)
    varianceCorrespondingToK1 = np.append(varianceCorrespondingToK1,var1)

plt.subplot(2,1,2)
plt.plot(k1,meansCorrespondingToK1, label= "Mean of Error",linestyle='solid')
plt.plot(k1,varianceCorrespondingToK1, label= "Variance of Error",linestyle='solid')
plt.legend()
plt.title("Plot for 100,000 points")
plt.legend()
plt.show()