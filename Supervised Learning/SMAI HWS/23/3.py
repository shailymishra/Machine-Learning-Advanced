from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import OrderedDict
from sklearn.cluster import KMeans 


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
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("sample_train.csv")

pca = PCA(n_components=2)
X_k = pca.fit_transform(train_data)
print(X_k.shape)
# then plot

kmeans = KMeans(n_clusters = 10, max_iter = 500) 
kmeans.fit(X_k) 
print(kmeans.labels_.shape)

colors = ['yellow', 'red', 'yellowgreen', 'magenta', 'blue', 'darkgreen', 'purple',
 'brown', 'black', 'orange', 'cyan']


for data, label in zip(X_k,train_labels):
    label = int(label)
    plt.scatter(data[0], data[1], label=label, color=colors[label])
plt.xlabel('x axis')
plt.ylabel('y axis')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title('Ground Truth')
plt.show()

for data, label in zip(X_k,kmeans.labels_):
    label = int(label)
    plt.scatter(data[0], data[1], label=label, color=colors[label])
plt.xlabel('x axis')
plt.ylabel('y axis')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title('K means Clustering')
plt.show()


# For each cluster find out original what is where
# 
metrix = np.zeros((10, 11))
for index in range(len(kmeans.labels_)):
    klabel = kmeans.labels_[index]
    orglabl = int(train_labels[index])
    metrix[klabel][orglabl] += 1
    metrix[klabel][10] +=1
print(metrix)

kmeans_labels = []
for ki in range(10):
    maxpoints = np.max(metrix.T[ki])
    findindex = np.where(metrix.T[ki]==maxpoints)
    kmeans_labels.append(findindex[0][0])
# print(kmeans_labels)

for ki in range(10):
    metrix[ki] = metrix[ki]/ metrix[ki][10]

kmeans_labels = []
for oi in range(10):
    maxpoints = np.max(metrix.T[oi])
    findindex = np.where(metrix.T[oi]==maxpoints)
    kmeans_labels.append(findindex[0][0])
print(kmeans_labels)

