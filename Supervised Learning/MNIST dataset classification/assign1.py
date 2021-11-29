# pragma solidity ^0.4.24;


# contract MyContract{
#     string myStringVar;

#     constructor() public{
#         myStringVar = "My Default Value";
#     }

#     // Public - you can run the method from outside the contract

#     // view means don't changing anything in code
#     // so no money will be cost, no matter how many time we returns
#     // but it is slow, so we do cache in database
#     // need to tell what you are returning, it helps in compling. (otherwise it cost moneym becasue space is
#     // is expensive. ) - code, time, space all = money

#     function get() public view returns(string){
#         return myStringVar;
#     }

#     function set(string newValue) public{
#         myStringVar = newValue;
#     }
# }


# #experiment
# # a = np.random.rand(5,5)
# a = np.array([
#     [-1, -3.5],
#     [1, 3.5]
# ])
# meanx = np.mean(a,axis=0)
# print(meanx)
# # print('a',a)
# cov_a =  np.cov(a.T)
# # print('cov_a',cov_a)
# eigenvalues_a, eigenvectors_a = LA.eigh(cov_a)
# eigenvalues_a.sort()
# # print('before', eigenvalues_a)
# eigenvalues_a = eigenvalues_a[::-1]
# total_a = sum(eigenvalues_a)
# var_exp_a = [(i / total_a)*100 for i in eigenvalues_a]

# # print(cov.shape)
# # print('rank',matrix_rank(cov))
# fig = go.Figure(data=go.Bar(y=var_exp_a))
# # fig.show()

# amatrix = np.array([
#     [1, 2 , 3],
#     [2, 4 ,5],
#     [3, 5 , 6],
# ])
# # amatrix = np.array([
# #     [3, 7 , 5],
# #     [-2, -2 , -4],
# #     [-1, -5 , -1],
# # ])
# print('rank of this matrix',matrix_rank(amatrix) )
# covmatrix = np.cov(amatrix.T)
# mean_vec = np.mean(amatrix, axis=0)
# # print('is cov same',(amatrix - mean_vec).T.dot((amatrix - mean_vec)) / (amatrix.shape[0]-1))
# # print('covariance of matrix', covmatrix )
# # print('rank of cov matrix', matrix_rank(covmatrix))
# values,vectors = LA.eigh(amatrix)
# evofcovmatrix,evsofcovmatrix = LA.eigh(covmatrix)
# print('evofcov', evofcovmatrix)
# # print('eigen value',x)
# from numpy import diag
# Q = vectors
# R = vectors.T
# L = diag(values)
# B = Q.dot(L).dot(R)
# # print('is B as cov/', B)
# print('------------')
# newmatrix = R.dot(amatrix)
# # print('L.dot(R)', L.dot(R))
# # print('R.dot(amtarix)', newmatrix)
# # print('R.Q.dot(L).dot(R)', R.dot(Q).dot(L).dot(R))
# # print('--------------')

# new_mean_vec = np.mean(newmatrix, axis=0)
# # print('new mean', new_mean_vec)
# # print('is cov same',(newmatrix - new_mean_vec).T.dot((newmatrix - new_mean_vec)) / (newmatrix.shape[0]-1))
# newcovmatrix = np.cov(newmatrix.T)
# # print('new cov', newcovmatrix)
# testcovmatrix = (Q.T).dot(covmatrix).dot(Q)
# # print('is this cov matrix sae', testcovmatrix)
# testvalues, testvectors = LA.eigh(testcovmatrix)
# evalues, evectors = LA.eigh(newcovmatrix)
# print('new eigen values', evalues)
# # print('testvalues eigen values', testvalues)

# print('new cov matrix',newcovmatrix )
# print('test', R.dot(covmatrix).dot(R.T))

# ortho = np.array([
#     [0,0,1,0,0],
#     [0,0,0,0,1],
#     [1,0,0,0,0],
#     [0,1,0,0,0],
#     [0,0,0,1,0],
# ])
# import scipy
# import math
# a = np.random.random(size=(2, 2))
# # ortho = scipy.linalg.orth(a)
# ortho = np.array([
#     [-1/math.sqrt(2), 1/math.sqrt(2)],
#     [1/math.sqrt(2) ,1/math.sqrt(2)]
# ])

# # print('lets see', np.dot(ortho.T,ortho))
# newa = np.dot(ortho,a)
# new_cov_a =  np.cov(newa.T)
# new_eigenvalues_a, new_eigenvectors_a = LA.eigh(new_cov_a)
# new_eigenvalues_a.sort()
# # print('before', eigenvalues_a)
# # print('after', new_eigenvalues_a)
# new_eigenvalues_a = new_eigenvalues_a[::-1]
# new_total_a = sum(new_eigenvalues_a)
# new_var_exp_a = [(i / new_total_a)*100 for i in new_eigenvalues_a]
# fig = go.Figure(data=go.Bar(y=new_var_exp_a))
# fig.show()


# experiment
# a = np.array([[0,0,0,0],
#               [0,0,0,1],
#               [0,0,1,0],
#              [0,0,1,1],
#              [0,1,0,0],
#              [0,1,0,1],
#              [0,1,1,0],
#              [0,1,1,1],

#               [1,0,0,0],
#               [1,0,0,1],
#               [1,0,1,0],
#              [1,0,1,1],
#              [1,1,0,0],
#              [1,1,0,1],
#              [1,1,1,0],
#              [1,1,1,1]
#              ])
# cov_a =  np.cov(a.T)
# # print(a)
# # print('rank',matrix_rank(a))
# # print(cov_a)
# eigenvalues_a, eigenvectors_a = LA.eigh(cov_a)
# eigenvalues_a.sort()
# eigenvalues_a = eigenvalues_a[::-1]
# # print(eigenvalues_a)
# # print(eigenvectors_a)
# total_a = sum(eigenvalues_a)
# var_exp_a = [(i / total_a)*100 for i in eigenvalues_a]

# print(cov.shape)
# print('rank',matrix_rank(cov))
# fig = go.Figure(data=go.Bar(y=var_exp_a[:99]))
# fig.show()


# 1. take a A
# 2. calculate its covariance matrix
# 3. CovA = Q eigen QT
# 4. multiply A by Q.
# 5. calculate covariance matrix now,
# 6. what is the cov matrix now, because technically it should be QT Q eigen QT Q

from numpy import linalg as LA
import plotly.graph_objects as go
from numpy.linalg import matrix_rank
import numpy as np
from numpy import diag
import matplotlib.pyplot as plt

column = np.array([[1, 1, 1]]).T
row = np.array([[1, 2, 3]])
rank1matrix = column.dot(row)

X = np.array([
    [2, 0, 3],
    [1, 5, 0],
    [3, -5, -3]
])

covmatrix = np.cov(X.T)
print(covmatrix)

eigenvalues, eigenvectors = LA.eigh(covmatrix)

print(eigenvalues)
newMatrix = X.dot(rank1matrix)
newcovmatrix = np.cov(newMatrix)
values,vectors = LA.eigh(newcovmatrix)

values.sort()
values = values[::-1]
total = sum(values)
newmatrixcov_eigenvalues = [(i / total)*100 for i in values]

eigenvalues.sort()
eigenvalues = eigenvalues[::-1]
total = sum(eigenvalues)
matrixcov_eigenvalues = [(i / total)*100 for i in eigenvalues]
fig1 = go.Figure(data=go.Bar(y=matrixcov_eigenvalues))
fig2 = go.Figure(data=go.Bar(y=newmatrixcov_eigenvalues))
fig1.show()
fig2.show()



# Q = np.array([
#     [1/3, -2/3, 2/3],
#     [2/3, -1/3, -2/3],
#     [2/3, 2/3, 1/3],
# ])

# X = np.array([
#     [-1, 0, 3],
#     [1, 5, 0],
#     [0, -5, -3]
# ])
# covmatrix = X.T.dot(X) / (X.shape[0]-1)

# eigenvalues, eigenvectors = LA.eigh(covmatrix)
# Q = eigenvectors
# R = eigenvectors.T
# L = diag(eigenvalues)
# eigenvalues.sort()
# eigenvalues = eigenvalues[::-1]
# total = sum(eigenvalues)
# x_eigenvalues = [(i / total)*100 for i in eigenvalues]


# QX = Q.dot(X)
# mean = np.mean(QX, axis=0)
# meanrow = np.mean(Q,axis=1)

# print('QX',QX)

# print('mean row',meanrow)
# print('mean', mean)
# covmatrixQX = np.cov(QX.T)
# print(covmatrixQX)
# values, vectors = LA.eigh(covmatrixQX)
# newQ = vectors
# newR = vectors.T
# newL = diag(values)
# values.sort()
# values = values[::-1]
# total = sum(values)
# qx_eigenvalues = [(i / total)*100 for i in values]
# fig1 = go.Figure(data=go.Bar(y=x_eigenvalues))
# fig2 = go.Figure(data=go.Bar(y=qx_eigenvalues))
# fig1.show()
# fig2.show()

# ------------------------------------FOR XQ------------------------------------------
# XQ = X.dot(Q)
# mean = np.mean(XQ, axis=0)
# covmatrixXQ = np.cov(XQ.T)
# print(covmatrixXQ)
# print(L)
# values, vectors = LA.eigh(covmatrixXQ)
# newQ = vectors
# newR = vectors.T
# newL = diag(values)
# values.sort()
# values = values[::-1]
# total = sum(values)
# xq_eigenvalues = [(i / total)*100 for i in values]
# plt.subplot(2, 2, 1)
# fig1 = go.Figure(data=go.Bar(y=x_eigenvalues))
# plt.subplot(2, 2, 2)
# fig2 = go.Figure(data=go.Bar(y=xq_eigenvalues))
# fig1.show()
# fig2.show()


# mean_vec = np.mean(X, axis=0)
# cov = (X - mean_vec).T.dot((X - mean_vec) / (X.shape[0]-1))
# print('mean', mean_vec)




# Print accuracy using Bayesian pairwise majority voting method
  
import itertools
all_numbers_data = {};
# Calculate means for each class from data
# now for pair (0,1) calculate cov matrix. and all inv, and constant term.
# 
pairs =(list(itertools.combinations(range(10), 2)))
def train_pairwise (train_data, train_labels):
    mus = [];
    n_features = len(train_labels)
    for i in range(10):
        print('i',i)
        train_sample = train_data[train_labels[:]==i]
        mu = train_sample.mean ( axis=0 )
        cov = np.cov(train_sample.T)
        all_numbers_data[i]={
            "mu":mu,
            "cov":cov
        }
    for pair in pairs:
        print(pair)
        cov = 0.5* (all_numbers_data[pair[0]]["cov"] +  all_numbers_data[pair[1]]["cov"] )
        inverse, constant_term = getParametersFromCov(cov,n_features)
        dict_alldata[pair]={
            "inverse":inverse,
            "constant_term" : constant_term,
        }
    return dict_alldata


def getParametersFromCov(cov,n_features):
    eigenvalues, eigenvectors = LA.eigh(cov)
    eigen = np.column_stack((eigenvalues,eigenvectors))
    eigen = eigen[(-eigen[:,0]).argsort()]
    sortedEigenValues = np.array(eigen[:,0])
    sortedEigenVectors = np.array(eigen[:,1:])
    approx_rank_k = len((np.where(sortedEigenValues>0))[0])
    first_k_eigen_values = sortedEigenValues[:approx_rank_k]
    first_k_eigen_values_inv = np.reciprocal(first_k_eigen_values)
    first_k_eigen_vectors = sortedEigenVectors[:approx_rank_k,:]
    inverse = first_k_eigen_vectors.T.dot(diag(first_k_eigen_values_inv)).dot(first_k_eigen_vectors)
    (sign, logdet) = np.linalg.slogdet(diag(first_k_eigen_values))
    constant_term = -0.5*(logdet) -  n_features*0.5*np.log(2.*np.pi)
    return (inverse, constant_term)
    
dict_alldata = train_pairwise(train_data,train_labels)

def predict_classwise(x_test):
    votes={}
    for i in range(10):
        votes[i]=0
    
    for pair in pairs:
            mu_0 = all_numbers_data[pair[0]]["mu"]
            mu_1 = all_numbers_data[pair[1]]["mu"]
            current_inv = dict_alldata[pair]["inverse"]
            current_constant_term = dict_alldata[pair]["constant_term"]
            
            s_0 = x_test - mu_0
            exp_term_0 =  -0.5*( s_0.T.dot(current_inv).dot(s_0)) 
            log_prob_0 = exp_term_0 * current_constant_term
            
            s_1 = x_test - mu_1
            exp_term_1 =  -0.5*( s_1.T.dot(current_inv).dot(s_1)) 
            log_prob_1 = exp_term_1 * current_constant_term
            if(log_prob_0 > log_prob_1):
                votes[pair[0]] = votes[pair[0]] + 1
            else:
                votes[pair[1]] = votes[pair[1]] + 1
    print(votes)
    maximum_votes_number = max(votes, key=votes.get)
    return maximum_votes_number

        
accuracy = 0;
for index in range(len(test_data)):
    identified_number = (predict_classwise(test_data[index]))
    print('Identified ',identified_number, 'actually is', test_labels[index])
    if(identified_number == test_labels[index]):
        accuracy = accuracy + 1
print('Accuracy',accuracy)
     

