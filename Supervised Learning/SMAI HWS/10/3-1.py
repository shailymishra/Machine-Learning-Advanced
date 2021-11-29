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
        print(indexes)
        for i in indexes:
            mean[imagecount] += data[i]
            length = len(indexes)
        mean[imagecount] = (mean[imagecount] / length)
        imagecount += 1
    return mean


def plotMean(data,labels, isGrayScale=False):
    mean = calculateMean(data,labels,isGrayScale)
    mean = mean.astype(np.uint8)
    fig, axes1 = plt.subplots(5,2,figsize=(3,3))
    image_count = 0
    for j in range(5):
        for k in range(2):
            print(image_count)
            axes1[j][k].set_axis_off()
            if(isGrayScale):
                axes1[j][k].imshow(mean[image_count], cmap=plt.get_cmap('gray'), interpolation='none')
            else:
                axes1[j][k].imshow(mean[image_count])
            image_count += 1
    plt.title('Mean of Images for each category')
    plt.show()

image_data,image_labels = readData()
print('Done reading')
gray_scale_image_data = convertAllIntoGrayScale(image_data)
print('Done Converting')
print('Ploting Mean for RGB')
plotMean(image_data,image_labels)
print('Ploting Mean for Gray')
plotMean(gray_scale_image_data,image_labels,True)


matrix = gray_scale_image_data.reshape(50000, 1024)

cov = np.cov(matrix.T)
eigen_vals, eigen_vecs = np.linalg.eigh(cov)
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(reverse=True)
W = np.matrix([])

W = eigen_pairs[0][1]
for i in range(1,20):
    W  = np.column_stack((W, eigen_pairs[i][1]))
transformed = np.dot(matrix,W)
limit = 1024-20
zerocolumn = np.zeros(limit)

## you multiple the scaled data


errors = []
for i in range(len(transformed)):
    vector = np.append(transformed[i], zerocolumn)
    error = np.linalg.norm(vector-matrix[i])
    errors.append(error)


categorizederror = []
for category in range(10):
    indexes = np.where(image_labels==category)
    indexes = indexes[0]
    error = 0
    for i in indexes:
        error += errors[i]
    categorizederror.append(error)

for i in range(len(categorizederror)):
    categorizederror[i] = categorizederror[i]/5000

# categorizederror = categorizederror /5000
print('Errors', categorizederror)

plt.bar(range(10), categorizederror, alpha=0.5, align='center',
        label='Individual')
plt.ylabel('Error')
plt.xlabel('Category')
plt.tight_layout()
plt.show()


