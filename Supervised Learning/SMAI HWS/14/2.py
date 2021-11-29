import numpy as np

def lr(z):
    a = 1 / (1 + np.exp(-z))
    return a

### Random values Generation
p = np.random.normal(0, 1, 500) / 10000  # y = 1
q = np.random.normal(0, 4, 500) / 10000  # y = -1
sample = [p, q]
sample_min = np.min(sample)
sample = np.array(sample)
w = 2
eta = 0.1
update_sum = 0

updateValueList = [w]
### Gradient descent implementation of Linear Regression
for i in range(2):
    for j in range(500):
        if i == 0:
            update_sum += (1 - sample[i, j] * w) * (-sample[i, j])
        if i == 1:
            update_sum += (-1 - sample[i, j] * w) * (-sample[i, j])
w = w - 2 * eta * update_sum
updateValueList.append(w)
while np.abs(update_sum) > 0.001:
    for i in range(2):
        for j in range(500):
            if i == 0:
                update_sum += (1 - sample[i, j] * w) * (-sample[i, j])
            if i == 1:
                update_sum += (-1 - sample[i, j] * w) * (-sample[i, j])
    w = w - 2 * eta * update_sum
    updateValueList.append(w)

# print(updateValueList)
import matplotlib.pyplot as plt
# lenofupdatevalues = len(updateValueList)
# print(lenofupdatevalues)
# xx = np.linspace(0,1,lenofupdatevalues)
# print(len(xx))
# plt.scatter(xx, updateValueList, label= "Decision Boundary",linestyle='solid',color = "k")
print('Global Minima by Linear Regression we got', update_sum)
# plt.show()

### Gradient Descent implementation of Logistic Regression
update_sum = 0
w = 2
updateValueList = [w]
for i in range(2):
    for j in range(500):
        if i == 0:
            update_sum += (1 - lr(sample[i, j] * w)) * (-sample[i, j]) * (lr(sample[i, j] * w)) ** 2
        if i == 1:
            update_sum += (-1 - lr(sample[i, j] * w)) * (-sample[i, j]) * (lr(sample[i, j] * w)) ** 2
w = w - 2 * eta * update_sum
updateValueList.append(w)

while update_sum > 0.001:
    for i in range(2):
        for j in range(500):
            if i == 0:
                update_sum += (1 - lr(sample[i, j] * w)) * (-sample[i, j]) * (lr(sample[i, j] * w)) ** 2
            if i == 1:
                update_sum += (-1 - lr(sample[i, j] * w)) * (-sample[i, j]) * (lr(sample[i, j] * w)) ** 2
    w = w - 2 * eta * update_sum
    updateValueList.append(w)

# lenofupdatevalues = len(updateValueList)
# print(lenofupdatevalues)
# xx = np.linspace(0,1,lenofupdatevalues)
# print(len(xx))
# plt.scatter(xx, updateValueList, label= "Decision Boundary",linestyle='solid',color = "k")
print('Global Minima by Logistic Regression we got', update_sum)
# plt.show()