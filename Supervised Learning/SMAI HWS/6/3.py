import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA


def parametersForGx(mu,covmatrix):
    inverse = np.linalg.inv(covmatrix)
    (sign, logdet) = np.linalg.slogdet(covmatrix)
    Wi = -0.5 * (inverse)
    wi = inverse.dot(mu)
    print('--------------')
    print(logdet)
    print(mu.T.dot(inverse).dot(mu))
    wi0 = -0.5*( mu.T.dot(inverse).dot(mu) + logdet )
    print('--------------')
    return Wi,wi,wi0


w1traindata = np.array([
    [0,0],
    [0,1],
    [2,0],
    [3,2],
    [3,3],
    [2,2],
    [2,0]
])

w2traindata = np.array([
    [7,7],
    [8,6],
    [9,7],
    [8,10],
    [7,10],
    [8,9],
    [7,11]
])

w1mean = np.mean(w1traindata, axis=0)
w1cov = (w1traindata - w1mean).T.dot((w1traindata - w1mean)) / (w1traindata.shape[0]-1)
W1,w1,w10 = parametersForGx(w1mean,w1cov)
print(W1)
print(w1)
print(w10)
print('----------')


w2mean = np.mean(w2traindata, axis=0)
w2cov = (w2traindata - w2mean).T.dot((w2traindata - w2mean)) / (w2traindata.shape[0]-1)
W2,w2,w20 = parametersForGx(w2mean,w2cov)
print(W2)
print(w2)
print(w20)

plt.scatter(w1traindata.T[0], w1traindata.T[1], label= "classA", color= "green", s=10) 
plt.scatter(w2traindata.T[0], w2traindata.T[1], label= "classB", color= "red", s=10) 
plt.legend()
plt.title("Plot with both the decision boundary, one in cyan and one in pink. almost similiar just shifted by log2")
# plt.show()

delta = 0.025
xrange = np.arange(-10.0, 70.0, delta)
yrange = np.arange(-10.0, 70.0, delta)
X, Y = np.meshgrid(xrange,yrange)

F = 16637*X**2 + 25932*X*Y - 9093 * Y**2 - 549468 * X - 156042 * Y + 2844200


import matplotlib.pyplot
matplotlib.pyplot.contour(X, Y, (F + np.log(2)), [0], colors=['pink'])
matplotlib.pyplot.contour(X, Y, (F), [0], colors=['cyan'],)
matplotlib.pyplot.show()



# def discr_func(x, cov_mat, mu_vec):
#     print('Entering')
#     x_vec = np.array([[x],[y]])
#     W_i = (-1/2) * np.linalg.inv(cov_mat)
#     w_i = np.linalg.inv(cov_mat).dot(mu_vec)
        
#     omega_i_p1 = (((-1/2) * (mu_vec).T).dot(np.linalg.inv(cov_mat))).dot(mu_vec)
#     omega_i_p2 = (-1/2) * np.log(np.linalg.det(cov_mat))
#     omega_i = omega_i_p1 - omega_i_p2

#     g = ((x_vec.T).dot(W_i)).dot(x_vec) + (w_i.T).dot(x_vec) + omega_i
#     # print('g is', g)
#     return g


# def decision_boundary(x_vec, mu_vec, mu_vec2):    
#     g1 = ()
#     g2 = 2*( (x_vec-mu_vec2).T.dot((x_vec-mu_vec2)) )
#     return g1 - g2






