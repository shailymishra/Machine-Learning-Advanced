# %matplotlib inline
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

def calculateMean(data,labels, isGrayScale = False):
    imagecount = 0
    if(isGrayScale):
        mean = np.zeros([10,32,32])
    else:
        mean = np.zeros([10,32,32,3])

    while(imagecount<10):
        indexes = np.where(labels==imagecount)[0]
        for i in indexes:
            mean[imagecount] += data[i]
            length = len(indexes)
        mean[imagecount] = (mean[imagecount] / length)
        imagecount += 1
    return mean

image_data,image_labels = readData()
print('Done reading')
gray_scale_image_data = convertAllIntoGrayScale(image_data)
print('Done Converting')

gray_scale_mean = calculateMean(gray_scale_image_data,image_labels,True)

gray_scale_mean = gray_scale_mean.reshape(10,1024)

meandifferencematrix = np.zeros([10,10])

print('Mean Differemce Matrix ')
print( )
print()

for i in range(10):
    for j in range(10):
        meandifferencematrix[i][j] =  np.linalg.norm(gray_scale_mean[i]-gray_scale_mean[j])

print(meandifferencematrix)