import numpy as np

wk = np.array([1,0,-1])
wknext = wk

learning_rate = 1
points_labels = np.array([[1,1,1,-1],
[2,2,1,1],
[-1,1,1,1],
[1,-1,1,1],
[-1,-1,1,-1],
[-2,-2,1,-1]]
)

def calculateMisclassified():
    misclassified = []
    for point in points_labels:
        tempoint = point[:3]
        wtx = np.dot(tempoint,wk)
        if(point[3]==-1 and wtx>=0):
            misclassified.append(point)
        if(point[3]==1  and wtx <=0 ):
            misclassified.append(point)
    return misclassified


iterations = 0
max_iterations = 20
print('Wknext -------------------')
while(iterations < max_iterations):
    wk = wknext
    addterms = np.array([0,0,0])
    misclassifed = calculateMisclassified()
    for point in misclassifed:
        term = (point[3] * point[:3])
        addterms +=  term    
    wknext = wk + addterms
    iterations += 1
    print(wknext)




