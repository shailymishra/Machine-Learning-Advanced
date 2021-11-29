# Class A is wTx >0
# Class B is wTx <=0

import numpy
import matplotlib.pyplot as plt 

samplesize = 50

# Generating All random data for class A and Class B
xclassA = numpy.random.uniform(-1,1, size=samplesize)
yclassA = numpy.random.uniform(-1,1, size=samplesize)
xclassB = numpy.random.uniform(-1,1, size=samplesize)
yclassB = numpy.random.uniform(-1,1, size=samplesize)
ones = numpy.ones(samplesize)
xyclassA = numpy.c_[xclassA, yclassA,ones]
xyclassB = numpy.c_[xclassB, yclassB,ones]

# Creating w list
wList = [[1, 1 ,0], [-1,-1,0],[0,0.5,0],[1,-1,5],[1.0,1.0,0.3]]

# Calculating the accuracy
overallAccuracy = []
for w in wList :
    correct = [0 ,0]
    for cord in xyclassA: 
        if(numpy.dot(w,cord)>0):
            correct[0] = correct[0] + 1
    correct[0]= correct[0]*100/samplesize
    for cord in xyclassB: 
        if(numpy.dot(w,cord)<=0):
            correct[1] = correct[1] + 1
    correct[1]= correct[1]*100/samplesize
    overallAccuracy.append(correct)

print(overallAccuracy)
x = numpy.linspace(-2,2,1000)
plt.subplot(2, 2, 1)
plt.scatter(xclassA, yclassA, label= "classA", color= "green", s=10) 
plt.scatter(xclassB, yclassB, label= "classB", color= "blue", s=10) 
plt.plot(x,-x,linestyle='solid',color = "black")
plt.title('for W = [1, 1, 0]')
plt.legend() 

plt.subplot(2, 2, 2)
plt.scatter(xclassA, yclassA, label= "classA", color= "green", s=10) 
plt.scatter(xclassB, yclassB, label= "classB", color= "blue", s=10) 
plt.plot(x,x+5,linestyle='solid',color = "black")
plt.title('for W = [1, -1, 5]')
plt.legend() 

plt.subplot(2, 2, 3)
plt.scatter(xclassA, yclassA, label= "classA", color= "green", s=10) 
plt.scatter(xclassB, yclassB, label= "classB", color= "blue", s=10) 
plt.plot(x,-x-0.3,linestyle='solid',color = "black")
plt.title('for W = [1.0, 1.0, 0.3]')

plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
plt.show() 

