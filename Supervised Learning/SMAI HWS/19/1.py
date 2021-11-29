from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y, epoch, typeofweightinitialise = "random"):
        self.input = x
        if(typeofweightinitialise=='random'):
            self.weights1 = np.random.rand(self.input.shape[1], 4)
            self.weights2 = np.random.rand(4, 1)
        elif(typeofweightinitialise=='ones'):
            self.weights1 = np.ones((trainsamples.shape[1], 4))
            self.weights2 = np.ones((4, 1))
        elif(typeofweightinitialise=='zeros'):
            self.weights1 = np.zeros((trainsamples.shape[1], 4))
            self.weights2 = np.zeros((4, 1))
        self.typeofweightinitialise = typeofweightinitialise
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.epoch=epoch
        self.losslist = []
    def feedforward(self, inputvalue):
        self.layer1 = sigmoid(np.dot(inputvalue, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(
            self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(
            self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self,showgraph=True):
        for i in range(self.epoch):
            nn.feedforward(self.input)
            nn.backprop()
            self.losslist.append(self.calculateLoss())
        
        if(showgraph):
            plt.plot(range(self.epoch), self.losslist  , c="red", label="Loss")
            plt.legend()
            plt.title("Initialisation of weights is : {}".format(self.typeofweightinitialise))
            plt.show()


    def calculateLoss(self):
        loss = 0
        for i in range(len(self.input)):
            nn.feedforward(self.input[i])
            loss += (nn.output - self.y[i])**2
        print('Loss is ', loss)
        return loss

    def accuracy(self, pred, true):
        c = 0
        for i in range(true.shape[0]):
            if(pred[i] == true[i]):
                c += 1
        return c

    def calculateAccuracy(self, predictedout, actualout):
        acc = self.accuracy(np.round(predictedout), actualout)
        acc = acc/actualout.shape[0]
        return acc

    def test(self, testx, testy):
        self.feedforward(testx)
        print(self.calculateAccuracy(self.output, testy))


if __name__ == "__main__":
    def create_data(samplesize, mu1, cov1, mu2, cov2):
        class_zeros = np.random.multivariate_normal(mu1, cov1, samplesize)
        class_ones = np.random.multivariate_normal(mu2, cov2, samplesize)
        return class_zeros, class_ones

    # Creating Data...........................
    samplesize = 50
    mu1 = np.array([2, 3])
    mu2 = np.array([0, 0])
    cov1 = np.array([[10, 0.01], [0.01, 10]])
    cov2 = np.array([[1, 0], [0, 3]])
    
    class0y = np.zeros(samplesize)
    class1y = np.ones(samplesize)
    class_zeros, class_ones = create_data(samplesize, mu1, cov1, mu2, cov2)
    plt.scatter(class_zeros.T[0], class_zeros.T[1], c="orange", label="Class0")
    plt.scatter(class_ones.T[0], class_ones.T[1], c="blue", label="Class1")
    plt.legend()
    plt.show()

    # Splitting data into train and test............................
    # and randomizing
    class0trainsample, class0testsample, class0ytrain, class0ytest = train_test_split(
        class_zeros, class0y, test_size=0.2, random_state=0)
    class1trainsample, class1testsample, class1ytrain, class1ytest, = train_test_split(
        class_ones, class1y, test_size=0.2, random_state=0)

    trainsamples = np.concatenate((class0trainsample, class1trainsample))
    trainyclassification = np.concatenate((class0ytrain, class1ytrain))
    testsamples = np.concatenate((class0testsample, class1testsample))
    testyclassification = np.concatenate((class0ytest, class1ytest))
    train_data = list(zip(trainsamples, trainyclassification))
    np.random.shuffle(train_data)
    trainsamples, trainyclassification = zip(*train_data)
    test_data = list(zip(testsamples, testyclassification))
    np.random.shuffle(test_data)
    testsamples, testyclassification = zip(*test_data)

    trainsamples = np.array(trainsamples)
    trainyclassification = np.c_[trainyclassification].T
    testsamples = np.array(testsamples)
    testyclassification = np.c_[testyclassification].T
    
    ones = np.ones((trainsamples.shape[0],trainsamples.shape[1]+1))
    ones[:,:-1] = trainsamples
    trainsamples = ones

    ones = np.ones((testsamples.shape[0],testsamples.shape[1]+1))
    ones[:,:-1] = testsamples
    testsamples = ones
    
    epoch = 500

    ### All weights are zero
    nn = NeuralNetwork(trainsamples, trainyclassification, epoch,'zeros')
    nn.train()
    nn.test( testsamples, testyclassification)
    nn.test( trainsamples,trainyclassification)

    
    ### All weights are ones
    nn = NeuralNetwork(trainsamples, trainyclassification, epoch,'ones')
    nn.train()
    nn.test( testsamples, testyclassification)
    nn.test( trainsamples,trainyclassification)

    ### Random Weights
    nn = NeuralNetwork(trainsamples, trainyclassification, epoch)
    nn.train()
    nn.test( testsamples, testyclassification)
    nn.test( trainsamples,trainyclassification)


