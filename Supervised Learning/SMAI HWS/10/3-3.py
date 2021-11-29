import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle 

# Y' = 0.2989 R + 0.5870 G + 0.1140 B 
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def readData():
    image_data = np.array([])
    image_labels = np.array([])
    fileindex = 1
    while(fileindex<6):
        filename = "cifar-10-batches-py/data_batch_{}".format(fileindex)
        print(filename)
        f = open(filename, 'rb')
        datadict = cPickle.load(f,encoding='latin1')
        f.close()
        X = datadict["data"] 
        Y = datadict['labels']

        if(fileindex==1):
            image_data = np.array(X)
            image_labels = np.array(Y)
        else:
            image_data = np.vstack((image_data, X))
            image_labels = np.append(image_labels,Y)
        fileindex += 1
    print(image_data.shape)
    image_data =  image_data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")   
    return image_data,image_labels 


def convertAllIntoGrayScale(image_data):
    print(len(image_data))
    length = len(image_data)
    grayscaleImageData = image_data
    grayscaleImageData = []
    for i in range(length):
        grayscaleImageData.append(rgb2gray(image_data[i]))
    grayscaleImageData = np.array(grayscaleImageData)
    return grayscaleImageData

def calculateSimiliary(classA, classB):
    meanA = np.mean(classA)

    covB = np.cov(classB.T)
    print(covB.shape)

    eigen_vals, eigen_vecs = np.linalg.eigh(covB)
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eigen_pairs.sort(reverse=True)

    W = eigen_pairs[0][1]
    for i in range(1,20):
        W  = np.column_stack((W, eigen_pairs[i][1]))
    print(W.shape)
    transformed = np.dot(classA,W)
    limit = 1024-20
    zerocolumn = np.zeros(limit)

    print(transformed.shape)

    error = 0
    for i in range(len(transformed)):
        vector = np.append(transformed[i], zerocolumn)
        error += np.linalg.norm(vector-meanA)
    error = error/5000
    print(error)
    return error

def getCategoryData(data,labels,category):
    categorydata = []
    indexes = np.where(labels==category)[0]
    for i in indexes:
        categorydata.append(data[i])
    return np.array(categorydata)


image_data,image_labels = readData()
print('Done reading')
gray_scale_image_data = convertAllIntoGrayScale(image_data)
print('Done Converting')

classes = {}
for i in range(10):
    classes[i] = getCategoryData(gray_scale_image_data,image_labels,i).reshape(5000,1024)


similaryMatrix = np.zeros([10,10])

print('similaryMatrix Matrix ')
print( )
print()

for i in range(10):
    for j in range(10):
        similaryMatrix[i][j] =  calculateSimiliary(classes[i], classes[j])

for i in range(10):
    for j in range(i+1):
        similaryMatrix[i][j] = 0.5 * (similaryMatrix[i][j] + similaryMatrix[j][i])
        similaryMatrix[j][i] = similaryMatrix[i][j]
    
    print('---------------------------------------------------')
    print('For class ', i)
    print(similaryMatrix[i])
    print('----')
    topvalues = np.sort(similaryMatrix[i])
    print(topvalues)
    print(np.where(similaryMatrix[i] == topvalues[0]))
    print(np.where(similaryMatrix[i] == topvalues[1]))
    print(np.where(similaryMatrix[i] == topvalues[2]))
    print(np.where(similaryMatrix[i] == topvalues[3]))
    print('---------------------------------------------------')


