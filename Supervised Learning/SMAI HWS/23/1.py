import numpy as np
import math
import matplotlib.pyplot as plt
import random

X=np.array([[2,-1],[2,1],[0,-1],[0,1],[1,0.5],[1,-0.5],[1,1],[1,-1]])
plt.scatter(X[:,0],X[:,1])
cent1=np.array([[0.5,0]])
cent2=np.array([[1.5,0]])
plt.scatter(cent1[0][0],cent1[0][1], label="Centroid - class1", marker='x')
plt.scatter(cent2[0][0],cent2[0][1], label="Centroid - class2", marker='x')


#calculating distances and labelling
arry1 = []
arry2 = []
for row in X:
    d1=dist = np.linalg.norm(row-cent1)
    d2=dist = np.linalg.norm(row-cent2)
    if d1 > d2:
        arry1.append(row)
    elif(d1<d2):
        arry2.append(row)
    elif(d1==d2):
        #Randomly assignmed class
        choice = random.randint(0, 1)
        if(choice == 0):
            arry1.append(row)
        else:
            arry2.append(row)
    continue

arry1 = np.array(arry1)
arry2 = np.array(arry2)
plt.scatter(arry1[:,0],arry1[:,1],c='r',label="Class1")
plt.scatter(arry2[:,0],arry2[:,1],c='b', label="Class2")
plt.legend()
plt.show()
    
