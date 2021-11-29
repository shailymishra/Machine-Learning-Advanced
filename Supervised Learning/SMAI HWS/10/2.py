import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from mpl_toolkits import mplot3d


def read_data(filename,dim):
    with open(filename, 'r') as f:
        lines = f.readlines()
    num_points = len(lines)
    dim_points = dim
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = num[0]
        data[ind] = num[1:]
        
    return (data, labels)

dim = 13
wine_data, wine_labels = read_data("wine_data.csv",dim)


rescaling_data = wine_data.T
for i in range(dim):
    column = rescaling_data[i]
    print(column.shape)
    mean = np.mean(column)
    var = np.std(column)
    print('Before doing ::', np.mean(rescaling_data[i]), '... var is :' , np.std(rescaling_data[i]))
    rescaling_data[i] = (rescaling_data[i]-mean) / var
    print('After doing ::', np.mean(rescaling_data[i]), '... var is :' , np.std(rescaling_data[i]))

rescaling_data = rescaling_data.T
P = np.dot(rescaling_data.T, rescaling_data) / wine_data.shape[0]
print(P.shape)
eigen_vals, eigen_vecs = np.linalg.eigh(P)
print('\nEigenvalues: \n%s' % eigen_vals)


tot = sum(np.abs(eigen_vals))
var_exp = [(i / tot) for i in sorted(np.abs(eigen_vals), reverse=True)]

plt.bar(range(1, eigen_vals.size + 1), var_exp, alpha=0.5, align='center',
        label='Individual')
plt.ylabel('Eigen Spectrum')
plt.xlabel('Principal components')
plt.tight_layout()
plt.show()

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(reverse=True)

W = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))


## you multiple the scaled data
transformed = rescaling_data.dot(W)
plt.scatter(transformed[0:59:,0], transformed[0:59:,1], color='blue' , label='Class 1')
plt.scatter(transformed[59:130:,0], transformed[59:130:,1], color='green',  label='Class 2')
plt.scatter(transformed[130:178:,0], transformed[130:178:,1], color='orange',  label='Class 3')
plt.title('Z_pca')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
