from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# dataset and targets
X = np.c_[(0, 0), (1, 1), (0, 1), (1, 0)].T
Y = [0] * 2 + [1] * 2
Y = np.array(Y)
X = np.array(X)

# create xor dataset
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# SVM


def plot_decision_regions(X, y, classifier, axs, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('green', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    axs.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    axs.set_xlim(xx1.min(), xx1.max())
    axs.set_ylim(xx2.min(), xx2.max())
    axs.set_xticks(())
    axs.set_yticks(())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        axs.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        axs.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidth=1, marker='o',
                    s=55, label='test set')


fig, axs = plt.subplots(3, 3, figsize=(15, 6), facecolor='w', edgecolor='k')
C = [0.1, 1, 100]
gamma = 2

j = 0
for kernel in ('sigmoid', 'poly', 'rbf'):
    i = 0
    for c in C:
        if(kernel == "poly"):
            clf = SVC(kernel=kernel, gamma=gamma, C=c, degree=2)
        else:
            clf = SVC(kernel=kernel, gamma=gamma, C=c)
        clf.fit(X_xor, y_xor)
        plot_decision_regions(X_xor, y_xor, clf, axs[i, j])
        axs[i, j].set_title(
            'Kernel : {} , C : {}, Gamma : {}'.format(kernel, c, gamma))
        i += 1
    j += 1
plt.legend(loc='upper left')
plt.show()

# axs[i,j].scatter(X_xor[:, 0], X_xor[:, 1], c=Y, zorder=10, cmap="winter",
#             edgecolors='k')
# # plt.axis('tight')
# x_min = -3
# x_max = 3
# y_min = -3
# y_max = 3
# XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
# Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
# # Put the result into a color plot
# Z = Z.reshape(XX.shape)
# axs[i,j].pcolormesh(XX, YY, Z > 0, cmap="winter")
# axs[i,j].contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
#             levels=[-.5, 0, .5])
