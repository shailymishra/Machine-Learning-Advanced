# Function x^2
import matplotlib.pyplot as plt

import math
import numpy as np

def calculateDerivative(x):
    return 2*x

def isUnderLimit(limit, value):
    return (value > (-limit) and value < limit)


def gradientDescent(learning_rate,currentvalue,limit, maxiteration = 100):
    xcurrent = currentvalue
    xnext = xcurrent
    derivative = calculateDerivative(xcurrent)
    iteration = 0
    errors = [np.abs(xcurrent)]
    iterations = [iteration]
    while(iteration < maxiteration):
        iteration = iteration + 1
        xcurrent = xnext
        xnext = xcurrent - learning_rate * derivative
        errors.append(np.abs(xnext))
        iterations.append(iteration)
        derivative = calculateDerivative(xnext)
        if(isUnderLimit(limit,derivative)):
            print('w:::', xnext)
            break
    print('iteration:::', iteration)
    return errors, iterations


errors, iterations = gradientDescent(0.1,-2, 1e-8)
plt.plot(iterations,errors, label= "Covergence Learning Rate = 0.1",linestyle='solid')
plt.title("Error Vs Iteration")
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()


errors1, iterations1 = gradientDescent(0.2,-2, 1e-8)
errors2, iterations2 = gradientDescent(1,-2, 1e-8)
errors3, iterations3 = gradientDescent(1.01,-2, 1e-8)
#
plt.plot(iterations1,errors1, label= "learning rate => 0.2 = Convergence",linestyle='solid')
plt.plot(iterations2,errors2, label= "learning rate => 1 = Oscillation",linestyle='solid')
plt.plot(iterations3,errors3, label= "learning rate => 1.01 = Divergence",linestyle='solid')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title("Error Vs Iteration for starting x = -2")
plt.legend()
plt.show()

# def ourfunction(x):
#     return x**2
# yvalues = []
# for x in range(len(xvalues)):
#     yvalues.append(ourfunction(x))
# plt.scatter(xvalues, yvalues , label= "classA", color= "green", s=10) 
# x = np.linspace(-100,100,1000)
# y = x**2
# plt.plot(x,y)

