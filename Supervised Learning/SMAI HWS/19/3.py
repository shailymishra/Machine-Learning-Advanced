# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # from sklearn.model_selection import train_test_split
# # # from numpy import exp, array, random, dot, tanh


# # # # Create Random Samples
# # # class1mean = [2, 3]
# # # class1cov = [[6, 4], [4, 5]]  #Positive SemiDefininte Matrix
# # # class2mean = [7,9]
# # # class2cov = [[3,2],[2,5]]  #Positive SemiDefininte Matrix

# # # samplesize = 100
# # # # class1x, class1y = np.random.multivariate_normal(class1mean,class1cov, samplesize).T
# # # class1samples = np.random.multivariate_normal(class1mean,class1cov, samplesize)
# # # class2samples = np.random.multivariate_normal(class2mean,class2cov, samplesize)
# # # plt.scatter(class1samples.T[0], class1samples.T[1], c="orange", label="Class1")
# # # plt.scatter(class2samples.T[0], class2samples.T[1], c="blue", label="Class2")
# # # plt.legend()
# # # plt.axis('equal')
# # # # plt.show()
# # # class1y = np.zeros(samplesize)
# # # class2y = np.ones(samplesize)

# # # ## Divide into train and test
# # # class1sampleTrain, class1sampleTest, class1yTrain, class1yTest  = train_test_split(class1samples,class1y, test_size = 0.2, random_state = 0)
# # # class2sampleTrain, class2sampleTest, class2yTrain, class2yTest , = train_test_split(class2samples,class2y, test_size = 0.2, random_state = 0)

# # # trainsamples = np.concatenate((class1sampleTrain, class2sampleTrain))
# # # trainy = np.concatenate((class1yTrain,class2yTrain))
# # # testsamples = np.concatenate((class1sampleTest, class2sampleTest))
# # # testy = np.concatenate((class1yTest,class2yTest))


# # # class NeuralNetwork():

# # #     def __init__(self):

# # #         # Using seed to make sure it'll
# # #         # generate same weights in every run
# # #         random.seed(1)

# # #         # 3x1 Weight matrix
# # # #         self.weight_matrix = 2 * random.random((3, 1)) - 1
# # #         # 2x2 Level 1 Weight Matrix
# # #         self.level1weight_matrix = 2 * random.random((2, 2)) - 1
# # #         # 2x1 Level 2 Weight Matrix
# # #         self.level2weight_matrix = 2 * random.random((2, 1)) - 1

# # #     # tanh as activation fucntion

# # #     def tanh(self, x):
# # #         return tanh(x)

# # #     # derivative of tanh function.
# # #     # Needed to calculate the gradients.
# # #     def tanh_derivative(self, x):
# # #         return 1.0 - tanh(x) ** 2

# # #     # Sigmoid as activation function
# # #     def sigmoid(self,x):
# # #         return 1/(1+np.exp(-x))

# # #     # Derivative of sigmoid function
# # #     # Needed to calculate the gradients
# # #     def sigmoid_derivative(self,x):
# # #         return self.sigmoid((x))*(1-self.sigmoid(x))

# # #     # forward propagation
# # #     def forward_propagation(self, inputs):
# # #         level1output = self.tanh(dot(inputs, self.level1weight_matrix))
# # #         level2output = self.sigmoid(dot(level1output, self.level2weight_matrix))
# # #         return level1output,level2output

# # #     # training the neural network.
# # #     def train(self, train_inputs, train_outputs,
# # #               num_train_iterations):

# # #         # Number of iterations we want to
# # #         # perform for this set of input.
# # #         for iteration in range(num_train_iterations):
# # #             # output = self.forward_propagation(train_inputs)
# # #             level1output, output = self.forward_propagation(train_inputs)
# # #             # Calculate the error in the output.
# # #             error = train_outputs - output
# # #             # multiply the error by input and then
# # #             # by gradient of tanh funtion to calculate
# # #             # the adjustment needs to be made in weights
# # #             print('Level1output.shape',level1output.shape)
# # #             print('Level1output.T.shape',level1output.T.shape)
# # #             level2adjustments = dot(level1output.T, error  * self.sigmoid_derivative(output))
# # #             print('level2adjustments.shape', level2adjustments.shape)
# # # #             level2adjustments =
# # #             adjustment = dot(train_inputs.T, error *
# # #                              self.tanh_derivative(output))
# # #             # print('adjustment...', adjustment.shape)
# # #             # Adjust the weight matrix
# # # #             self.weight_matrix += adjustment


# # # # Driver Code
# # # if __name__ == "__main__":

# # #     neural_network = NeuralNetwork()
# # #     print('Random weights at the start of training')
# # # #     print(neural_network.weight_matrix)

# # #     print('trainsamples...',trainsamples.shape)
# # #     print('trainy...',trainy.shape)

# # #     neural_network.train(trainsamples, trainy, 1)

# # #     print('New weights after training')
# # # #     print(neural_network.weight_matrix)

# # #     # Test the neural network with a new situation.
# # #     print("Testing network on new examples ->")
# # # #     print(neural_network.forward_propagation(array([1, 0, 0])))


# # importing the library
# import numpy as np
# import matplotlib.pyplot as plt

# def create_data(obs):
#     h=obs//2
#     # creating the input array
#     class_zeros = np.random.multivariate_normal([6,7], [[1.,.95],[.95,1.]], h)
#     class_ones = np.random.multivariate_normal([1,44], [[1.,.85],[.85,1.]], h)
#     x = class_zeros
#     x = np.append(x,class_ones).reshape(obs,2)
#     plt.plot(class_zeros,'o',color='r')
#     plt.plot(class_ones,'x',color='b')
#     plt.show()

#     # creating the output array

#     y=np.zeros((h))
#     y=np.append(y,np.ones((h))).reshape(obs,1)

#     #print(y)
#     return x,y

# # defining the Sigmoid Function
# def sigmoid (x):
#     return 1/(1 + np.exp(-x))

# # derivative of Sigmoid Function
# def derivatives_sigmoid(x):
#     return x * (1 - x)

# # defining the Sigmoid Function
# def tanh (x):
#     return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

# # derivative of Sigmoid Function
# def derivatives_tanh(x):
#     return (1 - x**2)

# def accuracy(p,t):
#     c=0
#     for i in range(t.shape[0]):
#         if(p[i]==t[i]):
#             c += 1
#     return c

# def cross_entropy(output, y_target):
#     return - np.sum(np.log(output) * (y_target), axis=1)

# def cost(y_target, output):
#     summ=np.sum((output-y_target)**2)
#     return np.mean(summ)

# def NeuralNetwork(x,y): # 2
#     # initializing the variables
#     epoch=500 # number of training iterations
#     lr=0.1 # learning rate
#     inputlayer_neurons = x.shape[1] # number of features in data set
#     hiddenlayer_neurons =2 # number of hidden layers neurons
#     output_neurons = 1 # number of neurons at output layer

#     # initializing weight and bias
#     wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
#     #bh=np.random.uniform(size=(1,hiddenlayer_neurons))
#     wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
#     #bout=np.random.uniform(size=(1,output_neurons))

#     # training the model
#     for i in range(epoch):

#         #Forward Propogation
#         hidden_layer_input1=np.dot(x,wh)
#         hidden_layer_input=hidden_layer_input1 #+ bh
#         hiddenlayer_activations = tanh(hidden_layer_input)
#         output_layer_input1=np.dot(hiddenlayer_activations,wout)
#         output_layer_input= output_layer_input1 #+ bout
#         output = sigmoid(output_layer_input)
#         #print(output.shape)

#         #Backpropagation
#         E = y-output # error (t-a)
#         slope_output_layer = derivatives_sigmoid(output) # derivation of sigmoid function = a(1-a)
#         slope_hidden_layer = derivatives_tanh(hiddenlayer_activations) # derivation of tanh function = (1-a**2)
#         d_output = E * slope_output_layer # (t-a) * a(1-a)
#         Error_at_hidden_layer = d_output.dot(wout.T) # (t-a)(a(1-a))(wout)
#         d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer # (t-a)(a(1-a))(wout)((1-a*2))
#         wout += hiddenlayer_activations.T.dot(d_output) *lr
#         #bout += np.sum(d_output, axis=0,keepdims=True) *lr
#         wh += x.T.dot(d_hiddenlayer) *lr
#         #bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

#         acc = accuracy(np.round(output),y)
#         acc = acc/y.shape[0]
#         if(i%10 == 0):
#             print('Epoch no.=',i,' ,Accuracy=',acc)
#         cost_p=cost(y,output)
#         plt.plot(i,cost_p,'o',color='b')
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     plt.show()
#     return wout,bout

# train_data,train_t = create_data(80)
# wout,bout=NeuralNetwork(train_data,train_t)
# test,t = create_data(20)
# hidden_layer_input1=np.dot(test,wh)
# hidden_layer_input=hidden_layer_input1
# hiddenlayer_activations = tanh(hidden_layer_input)
# output_layer_input1=np.dot(hiddenlayer_activations,wout)
# output_layer_input= output_layer_input1
# output = sigmoid(output_layer_input)
# acc = accuracy(np.round(output),t)
# acc = acc / t.shape[0]
# print('Accuracy=',acc)


# # import matplotlib.pyplot as plt
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from numpy import exp, array, random, dot, tanh


# # # Create Random Samples
# # class1mean = [2, 3]
# # class1cov = [[6, 4], [4, 5]]  # Positive SemiDefininte Matrix
# # class2mean = [7, 9]
# # class2cov = [[3, 2], [2, 5]]  # Positive SemiDefininte Matrix

# # samplesize = 100
# # # class1x, class1y = np.random.multivariate_normal(class1mean,class1cov, samplesize).T
# # class1samples = np.random.multivariate_normal(
# #     class1mean, class1cov, samplesize)
# # class2samples = np.random.multivariate_normal(
# #     class2mean, class2cov, samplesize)
# # plt.scatter(class1samples.T[0], class1samples.T[1], c="orange", label="Class1")
# # plt.scatter(class2samples.T[0], class2samples.T[1], c="blue", label="Class2")
# # plt.legend()
# # plt.axis('equal')
# # # plt.show()
# # class1y = np.zeros(samplesize)
# # class2y = np.ones(samplesize)

# # # Divide into train and test
# # class1sampleTrain, class1sampleTest, class1yTrain, class1yTest = train_test_split(
# #     class1samples, class1y, test_size=0.2, random_state=0)
# # class2sampleTrain, class2sampleTest, class2yTrain, class2yTest, = train_test_split(
# #     class2samples, class2y, test_size=0.2, random_state=0)

# # trainsamples = np.concatenate((class1sampleTrain, class2sampleTrain))
# # trainy = np.concatenate((class1yTrain, class2yTrain))
# # testsamples = np.concatenate((class1sampleTest, class2sampleTest))
# # testy = np.concatenate((class1yTest, class2yTest))


# # class NeuralNetwork():

# #     def __init__(self):

# #         # Using seed to make sure it'll
# #         # generate same weights in every run
# #         random.seed(1)

# #         # 3x1 Weight matrix
# # #         self.weight_matrix = 2 * random.random((3, 1)) - 1
# #         # 2x2 Level 1 Weight Matrix
# #         self.level1weight_matrix = 2 * random.random((2, 2)) - 1
# #         # 2x1 Level 2 Weight Matrix
# #         self.level2weight_matrix = 2 * random.random((2, 1)) - 1

# #     # tanh as activation fucntion

# #     def tanh(self, x):
# #         return tanh(x)

# #     # derivative of tanh function.
# #     # Needed to calculate the gradients.
# #     def tanh_derivative(self, x):
# #         return 1.0 - tanh(x) ** 2

# #     # Sigmoid as activation function
# #     def sigmoid(self, x):
# #         return 1/(1+np.exp(-x))

# #     # Derivative of sigmoid function
# #     # Needed to calculate the gradients
# #     def sigmoid_derivative(self, x):
# #         return self.sigmoid((x))*(1-self.sigmoid(x))

# #     # forward propagation
# #     def forward_propagation(self, inputs):
# #         level1output = self.tanh(dot(inputs, self.level1weight_matrix))
# #         level2output = self.sigmoid(
# #             dot(level1output, self.level2weight_matrix))
# #         return level1output, level2output

# #     # training the neural network.
# #     def train(self, train_inputs, train_outputs,
# #               num_train_iterations, test_inputs, test_outputs):
# #         # Number of iterations we want to
# #         # perform for this set of input.
# #         for iteration in range(num_train_iterations):
# #             # output = self.forward_propagation(train_inputs)
# #             level1output, output = self.forward_propagation(train_inputs)
# #             # Calculate the error in the output.
# #             error = train_outputs - output
# #             # multiply the error by input and then
# #             # by gradient of tanh funtion to calculate
# #             # the adjustment needs to be made in weights
# #             delta2 = error * self.sigmoid_derivative(output)
# #             level2adjustments = dot(level1output.T, delta2)
# #             delta1 = delta2 * self.tanh_derivative(level1output)
# #             delta1.T[0] = delta1.T[0] * self.level2weight_matrix[0]
# #             delta1.T[1] = delta1.T[1] * self.level2weight_matrix[1]
# #             level1adjustment = dot(train_inputs.T, delta1)

# #             # Adjust the weight matrix
# #             self.level2weight_matrix += level2adjustments
# #             self.level1weight_matrix += level1adjustment

# #             print('Epoch::', iteration, ' Accu : ', self.calculateAccuracy(test_inputs, test_outputs))

# #     def accuracy(self, pred, true):
# #         c=0
# #         for i in range(true.shape[0]):
# #             if(pred[i] == true[i]):
# #                 c += 1
# #         return c

# #     def calculateAccuracy(self, testdata, testlabels):
# #         level1output, output=self.forward_propagation(testdata)
# #         acc=self.accuracy(np.round(output), testlabels)
# #         acc=acc/testlabels.shape[0]
# #         return acc


# # # Driver Code
# # if __name__ == "__main__":

# #     neural_network=NeuralNetwork()
# #     print('Random weights at the start of training')
# # #     print(neural_network.weight_matrix)
# #     trainy=np.c_[trainy]
# #     testy=np.c_[testy]

# #     neural_network.train(trainsamples, trainy, 5000, testsamples, testy)

# #     print('New weights after training')
# #     print(neural_network.level2weight_matrix)
# #     print(neural_network.level1weight_matrix)

# #     # Test the neural network with a new situation.
# #     print("Testing network on new examples ->")
# #     print('testsamples[0]', testsamples[0].shape)
# #     print(neural_network.forward_propagation(testsamples[0]))
# #     print('Acutal Output', testy[0])


# importing the library
import numpy as np
import matplotlib.pyplot as plt


def create_data(obs):
    h = obs//2
    # creating the input array
    class_zeros = np.random.multivariate_normal(
        [4, 6], [[1., .95], [.95, 1.]], h)
    class_ones = np.random.multivariate_normal(
        [0, 0], [[1., .85], [.85, 1.]], h)
    x = class_zeros
    x = np.append(x, class_ones).reshape(obs, 2)
    plt.scatter(class_zeros.T[0], class_zeros.T[1] , color='green', label="Class 0")
    plt.scatter(class_ones.T[0], class_ones.T[1], color='blue', label="Class 1")
    plt.legend()
    plt.show()

    # creating the output array

    y = np.zeros((h))
    y = np.append(y, np.ones((h))).reshape(obs, 1)

    # print(y)
    return x, y

# defining the Sigmoid Function


def sigmoid(x):
    return 1/(1 + np.exp(-x))

# derivative of Sigmoid Function


def derivatives_sigmoid(x):
    return x * (1 - x)

# defining the Sigmoid Function


def tanh(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

# derivative of Sigmoid Function


def derivatives_tanh(x):
    return (1 - x**2)


def accuracy(p, t):
    c = 0
    for i in range(t.shape[0]):
        if(p[i] == t[i]):
            c += 1
    return c


def cross_entropy(output, y_target):
    return - np.sum(np.log(output) * (y_target), axis=1)


def cost(y_target, output):
    summ = np.sum((output-y_target)**2)
    return np.mean(summ)


def NeuralNetwork(x, y, typeOfWInitialization="random"):  # 2
    # initializing the variables
    epoch = 500  # number of training iterations
    lr = 0.1  # learning rate
    inputlayer_neurons = x.shape[1]  # number of features in data set
    hiddenlayer_neurons = 2  # number of hidden layers neurons
    output_neurons = 1  # number of neurons at output layer

    if(typeOfWInitialization == 'random'):
        # initializing weight and bias
        wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
        bh = np.random.uniform(size=(1, hiddenlayer_neurons))
        wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
        bout = np.random.uniform(size=(1, output_neurons))
    elif (typeOfWInitialization == 'ones'):
        wh = np.ones((inputlayer_neurons, hiddenlayer_neurons))
        bh = np.ones((1, hiddenlayer_neurons))
        wout = np.ones((hiddenlayer_neurons, output_neurons))
        bout = np.ones((1, output_neurons))
    elif (typeOfWInitialization == 'zeros'):
        wh = np.zeros((inputlayer_neurons, hiddenlayer_neurons))
        bh = np.zeros((1, hiddenlayer_neurons))
        wout = np.zeros((hiddenlayer_neurons, output_neurons))
        bout = np.zeros((1, output_neurons))

    # training the model
    for i in range(epoch):

        # Forward Propogation
        hidden_layer_input1 = np.dot(x, wh)
        hidden_layer_input = hidden_layer_input1 + bh
        hiddenlayer_activations = tanh(hidden_layer_input)
        output_layer_input1 = np.dot(hiddenlayer_activations, wout)
        output_layer_input = output_layer_input1 + bout
        output = sigmoid(output_layer_input)
        # print(output.shape)

        # Backpropagation
        E = y-output  # error (t-a)
        # derivation of sigmoid function = a(1-a)
        slope_output_layer = derivatives_sigmoid(output)
        # derivation of tanh function = (1-a**2)
        slope_hidden_layer = derivatives_tanh(hiddenlayer_activations)
        d_output = E * slope_output_layer  # (t-a) * a(1-a)
        Error_at_hidden_layer = d_output.dot(wout.T)  # (t-a)*(a(1-a))*(wout)
        d_hiddenlayer = Error_at_hidden_layer * \
            slope_hidden_layer  # (t-a)*(a(1-a))*(wout)*((1-a**2))
        wout += hiddenlayer_activations.T.dot(d_output) * lr
        bout += np.sum(d_output, axis=0, keepdims=True) * lr
        wh += x.T.dot(d_hiddenlayer) * lr
        bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr

        acc = accuracy(np.round(output), y)
        acc = acc/y.shape[0]
        if(i % 10 == 0):
            print('Epoch no.=', i, ' ,Accuracy=', acc)
        cost_p = cost(y, output)
        plt.plot(i, cost_p, 'o', color='orange')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Type of Initialzation is {} '.format(typeOfWInitialization) )
    plt.show()
    return wh,bh,wout, bout


train_data, train_t = create_data(80)

##### Weights = 0
wh,bh,wout, bout = NeuralNetwork(train_data, train_t, 'zeros')
test, t = create_data(20)
hidden_layer_input1 = np.dot(test, wh)
hidden_layer_input = hidden_layer_input1
hiddenlayer_activations = tanh(hidden_layer_input)
output_layer_input1 = np.dot(hiddenlayer_activations, wout)
output_layer_input = output_layer_input1
output = sigmoid(output_layer_input)
acc = accuracy(np.round(output), t)
acc = acc / t.shape[0]
print('Accuracy=', acc)


##### Weights = 1
wh,bh,wout, bout = NeuralNetwork(train_data, train_t, 'ones')
test, t = create_data(20)
hidden_layer_input1 = np.dot(test, wh)
hidden_layer_input = hidden_layer_input1
hiddenlayer_activations = tanh(hidden_layer_input)
output_layer_input1 = np.dot(hiddenlayer_activations, wout)
output_layer_input = output_layer_input1
output = sigmoid(output_layer_input)
acc = accuracy(np.round(output), t)
acc = acc / t.shape[0]
print('Accuracy=', acc)


# Random Initializaation
wh,bh,wout, bout = NeuralNetwork(train_data, train_t)
test, t = create_data(20)
hidden_layer_input1 = np.dot(test, wh)
hidden_layer_input = hidden_layer_input1
hiddenlayer_activations = tanh(hidden_layer_input)
output_layer_input1 = np.dot(hiddenlayer_activations, wout)
output_layer_input = output_layer_input1
output = sigmoid(output_layer_input)
acc = accuracy(np.round(output), t)
acc = acc / t.shape[0]
print('Accuracy=', acc)



